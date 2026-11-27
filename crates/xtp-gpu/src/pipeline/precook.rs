//! `xtp precook`'s side in this crate: a shader container of the title's
//! pack, and a graphics or mesh pipeline an earlier session made, made
//! ready the way the title's first use of it makes it, through the same
//! calls, so that a session after a translator change starts with its
//! translations and its host pipelines already made.
//!
//! - Every container is translated for the host's wave sizes
//!   ([`crate::device::host_target`]) by [`crate::shader_cache`]'s
//!   `host_shader`, the function the title's pipelines call: same
//!   translation, same key, same entry. The title finds it on first use.
//! - A compute program's host pipeline is made from its translation alone
//!   (its root signature is the translation's bindings:
//!   `host::HostPipeline::new`), so it is made here too, by that function,
//!   which stores it into the host pipeline library as a first dispatch
//!   does; [`finish`] saves the library.
//! - Graphics and mesh pipelines need the title's graphics state as well,
//!   which only a session sees: sessions keep the description of each one
//!   the title creates ([`super::record`]), and [`replay`] makes it again as
//!   the session did, by the same function with the same shaders and state,
//!   into the same library entry. A description whose shaders do not
//!   translate any more is refused at once, as in a session. The file of
//!   descriptions is rewritten here once it grows past its bound
//!   (`record::compact`).
//! - The pack's effects ([`super::effects`]) name the shaders of each pass,
//!   not its state: their shader sets ([`ShaderSet`]) say how much of the
//!   title the kept descriptions cover ([`Recorded::shader_sets`]).
//! - What a whole pass depends on besides the pack and the translator, for
//!   the precook's note: the kept descriptions and the library file
//!   ([`cache`]).
//! - The precook reuses the pipeline library across translator changes.
//!   Each translation key has its own generation of the library file,
//!   seeded from the last one opened ([`generation`]); the precook can
//!   compact one past [`LIBRARY_REBUILD_BYTES`] ([`rebuild_library`]) into
//!   the pipelines used again, and a translator back to one whose
//!   generation is still kept finds its pipelines there.
//! - A pipeline the library holds is not loaded here ([`super::library`]'s
//!   `without_loading`): it is made through the same function, which names
//!   it from its translated shaders, root signature and state, and its name
//!   found in the library's names is the answer ([`Counts::known`]). After
//!   a translator change, the pipelines whose translations did not change
//!   cost their name alone; a session loads each when it needs it.
//! - After a translator change, each translation is compared with the one
//!   the translator before kept ([`compare_with`], [`compared`]): what the
//!   change altered, counted. The answer is always this build's.

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use xtp_shader::{Container, FourCC, HwStage};

use super::Shader;
use super::host::{GraphicsState, HostPipeline};

/// What became of one container.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    /// Its translation is in the cache (made now or found there: [`counts`]
    /// tells how many of each), and for a compute program with pipelines
    /// asked, its host pipeline in the library.
    Ready,
    /// The translator refuses it: the reason, kept in the cache as the
    /// title's first use would keep it.
    Refused(String),
    /// A compute program that gets no host pipeline (the reasons:
    /// [`refusals`]).
    NoPipeline,
}

/// What the translations are made for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ready {
    /// The wave sizes translated for, as (min, max) lanes.
    pub lanes: (u32, u32),
    /// The key the translations are kept under (this build's translator
    /// and settings, and the target), in hexadecimal.
    pub key: String,
}

/// Makes the host device (the title's adapter: `XTP_ADAPTER` as in a run)
/// and says what translations are made for; why not, without a device.
pub fn start() -> Result<Ready, String> {
    crate::host::created().as_ref().map_err(Clone::clone)?;
    let target = crate::device::host_target();
    let key = crate::shader_cache::host_key();
    Ok(Ready {
        lanes: (target.lanes_min, target.lanes_max),
        key: key.iter().map(|b| format!("{b:02x}")).collect(),
    })
}

/// The stage of the program in `container` that a translation takes (its
/// first vertex, pixel, mesh or compute program, as `translate_for` takes
/// it); `None` if it holds none.
pub fn stage(container: &[u8]) -> Option<HwStage> {
    let c = Container::parse(container).ok()?;
    let part = c.part(FourCC::DXIL)?;
    xtp_shader::native_programs(part.data)
        .ok()?
        .into_iter()
        .map(|p| p.stage)
        .find(|s| matches!(s, HwStage::Vs | HwStage::Ps | HwStage::Ms | HwStage::Cs))
}

/// Precooks `container`, whose program is of `stage` ([`stage`]): its
/// translation into the shader cache and, with `pipelines`, a compute
/// program's host pipeline into the library.
pub fn container(container: &[u8], stage: Option<HwStage>, pipelines: bool) -> Outcome {
    match prepare(container, stage, pipelines) {
        Ok(Some(compute)) => compute.pipeline(),
        Ok(None) => Outcome::Ready,
        Err(outcome) => outcome,
    }
}

/// A translated compute program, ready for the pipeline phase.
pub struct Compute(Arc<Shader>);

impl Compute {
    /// Makes the pipeline from the same description as a dispatch, unless
    /// the library holds it (then it is not loaded either).
    pub fn pipeline(&self) -> Outcome {
        let shaders = std::slice::from_ref(&self.0);
        let _ = super::library::take_last();
        let made = super::library::without_loading(|| HostPipeline::new(shaders, None).is_some());
        if made && super::library::take_last().cached {
            super::library::note_cached(&super::make::key(shaders, None));
        }
        if made {
            Outcome::Ready
        } else {
            Outcome::NoPipeline
        }
    }
}

/// Translates one container and keeps its compute shader alive for the
/// pipeline phase, so building the pipeline does not read the cache again.
pub fn prepare(
    container: &[u8],
    stage: Option<HwStage>,
    pipelines: bool,
) -> Result<Option<Compute>, Outcome> {
    let target = crate::device::host_target();
    if !(pipelines && stage == Some(HwStage::Cs)) {
        return crate::shader_cache::host_shader(container, target)
            .map(|_| None)
            .map_err(Outcome::Refused);
    }
    let shader = super::shader("compute", container);
    if shader.translated().is_some() {
        return Ok(Some(Compute(shader)));
    }
    // The shared shader keeps only the verdict. Recover the cached reason.
    Err(match crate::shader_cache::host_shader(container, target) {
        Err(reason) => Outcome::Refused(reason),
        Ok(_) => Outcome::NoPipeline,
    })
}

/// What the precook made and found since the start.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Counts {
    /// Shaders translated, and found in the cache (refusals are neither).
    pub translated: u64,
    pub cached: u64,
    /// Host pipelines compiled, and loaded from the library.
    pub made: u64,
    pub loaded: u64,
    /// Host pipelines the library holds by their names, not loaded.
    pub known: u64,
}

/// The counts so far.
pub fn counts() -> Counts {
    let (cached, translated) = crate::shader_cache::counts();
    Counts {
        translated,
        cached,
        made: super::library::MADE.load(Ordering::Relaxed),
        loaded: super::library::LOADED.load(Ordering::Relaxed),
        known: super::library::KNOWN.load(Ordering::Relaxed),
    }
}

/// From now on, compares each translation made with the entry the
/// translator of `key` (another build's key, as [`Ready::key`] gives it in
/// hexadecimal) kept for its container. False if `key` is not one, or a
/// comparison is set already.
pub fn compare_with(key: &str) -> bool {
    if key.len() != 64 || !key.is_ascii() {
        return false;
    }
    let mut bytes = [0u8; 32];
    for (i, b) in bytes.iter_mut().enumerate() {
        match u8::from_str_radix(&key[2 * i..2 * i + 2], 16) {
            Ok(v) => *b = v,
            Err(_) => return false,
        }
    }
    crate::shader_cache::compare_with(bytes)
}

/// How the translations made compare with the translator before's
/// ([`compare_with`]).
pub fn compared() -> crate::shader_cache::Compared {
    crate::shader_cache::compared_counts()
}

/// The host pipelines refused so far, by reason, most first.
pub fn refusals() -> Option<String> {
    super::refusals()
}

/// A library file past this size is made a new generation by the next
/// precook (the cache's had grown to 2.3 GB on 2026-11-25).
pub const LIBRARY_REBUILD_BYTES: u64 = 2 << 30;

/// What a precook's pipelines depend on besides the pack and the
/// translator: the descriptions sessions kept, the library file, and the
/// settings that change a pipeline without changing a translation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cache {
    /// The BLAKE3 of the file of kept descriptions, in hexadecimal; `none`
    /// without one.
    pub recorded: String,
    /// The library file's size (0: none).
    pub library: u64,
    /// [`PipelineSettings::text`] of this run's.
    pub settings: String,
}

/// The cache's state now ([`Cache`]); an empty one without a pipeline
/// cache folder.
pub fn cache() -> Result<Cache, String> {
    let settings = PipelineSettings::now().text();
    let Some(dir) = super::library::dir() else {
        return Ok(Cache {
            recorded: "none".into(),
            library: 0,
            settings,
        });
    };
    Ok(Cache {
        recorded: digest_in(dir)?,
        library: super::library::file_bytes().unwrap_or(0),
        settings,
    })
}

/// What changes the host pipelines besides the translations (the key
/// their cache entries are kept under) and the title's descriptions: each
/// setting and host capability that changes a pipeline's description (its
/// root signature, the state the host makes) or whether it is made at all,
/// by its effective value. A precook's note says it, and a pass for other
/// settings is a whole pass (Codex review of d286ffd, point 1: with
/// `XTP_ROOT_SRV_DIRECT=1` set after a pass, the note still said up to
/// date, its translation key the same, and no pipeline of the new root
/// signatures was made: a session made each at its first use).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PipelineSettings {
    /// Root SRVs bound as one-view tables, not root descriptors
    /// (`XTP_ROOT_SRV_DIRECT` unset): the root signatures.
    pub root_srv_tables: bool,
    /// Wave64 mesh programs refused on hosts without 64-lane waves
    /// (`XTP_WAVE64_MESH=refuse`): which mesh pipelines are made.
    pub wave64_mesh_refused: bool,
    /// The host's wave sizes and shader model: which programs it runs.
    pub lanes: Option<(u32, u32)>,
    pub model: Option<(u32, u32)>,
    /// The host has a depth bounds test (in a pipeline's state) and mesh
    /// shaders (whether mesh pipelines are made).
    pub depth_bounds: bool,
    pub mesh_shaders: bool,
}

impl PipelineSettings {
    /// This run's (the host must be made for its capabilities).
    pub fn now() -> PipelineSettings {
        let host = crate::host::host();
        PipelineSettings {
            root_srv_tables: super::host::bounded_root_srvs(),
            wave64_mesh_refused: super::host::wave64_mesh_refused(),
            lanes: crate::device::host_wave_lanes(),
            model: crate::device::host_shader_model(),
            depth_bounds: host.is_some_and(|h| h.depth_bounds),
            mesh_shaders: host.is_some_and(|h| h.mesh_shaders()),
        }
    }

    /// One line that says them all, as the note keeps it.
    pub fn text(&self) -> String {
        let pair = |p: Option<(u32, u32)>, sep: char| {
            p.map_or_else(|| "unknown".to_owned(), |(a, b)| format!("{a}{sep}{b}"))
        };
        let yes = |b: bool| if b { "yes" } else { "no" };
        format!(
            "root-srv {} wave64-mesh {} lanes {} model {} depth-bounds {} mesh {}",
            if self.root_srv_tables {
                "tables"
            } else {
                "direct"
            },
            if self.wave64_mesh_refused {
                "refused"
            } else {
                "translated"
            },
            pair(self.lanes, '-'),
            pair(self.model, '.'),
            yes(self.depth_bounds),
            yes(self.mesh_shaders),
        )
    }
}

/// The BLAKE3 of the file of kept descriptions in `dir` now, in
/// hexadecimal; `none` without one.
fn digest_in(dir: &std::path::Path) -> Result<String, String> {
    let path = dir.join(super::record::FILE);
    match std::fs::File::open(&path) {
        Ok(file) => {
            let mut hasher = blake3::Hasher::new();
            hasher
                .update_reader(file)
                .map_err(|e| format!("reading {}: {e}", path.display()))?;
            Ok(hasher.finalize().to_hex().to_string())
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok("none".into()),
        Err(e) => Err(format!("reading {}: {e}", path.display())),
    }
}

/// Makes the library a new generation (see the module): to be asked
/// before any pipeline is made; false if that is too late.
pub fn rebuild_library() -> bool {
    super::library::rebuild()
}

/// This translation key's generation of the pipeline library: what its
/// file holds besides this translator's pipelines.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Generation {
    /// The translation key's first 16 hexadecimal digits: its files are
    /// `host-pipelines-<id>.bin` and `.names`.
    pub id: String,
    /// The generation its file was seeded from (another translator's, or
    /// `legacy`: the one file of before generations), and not compacted
    /// since: it holds that translator's pipelines too. `None` for one that
    /// started empty or was compacted.
    pub from: Option<String>,
    /// Its file's size just after its last compaction, if that was past
    /// [`LIBRARY_REBUILD_BYTES`].
    pub compacted: Option<u64>,
    /// It had no file before this process (started empty or seeded now):
    /// its file is not the one an earlier pass of this key left, whatever
    /// its size, so a note of that pass does not hold for it.
    pub new: bool,
}

/// This process's generation of the library ([`Generation`]), the oldest
/// generations let go past their count or bytes; `None` without a pipeline
/// cache folder. The host must be made, and this asked before any pipeline
/// is (it settles which file the library opens).
pub fn generation() -> Option<Generation> {
    super::library::generation().map(|(g, new)| Generation {
        id: g.id,
        from: g.from,
        compacted: g.compacted,
        new,
    })
}

/// The graphics and mesh pipelines sessions kept ([`super::record`]), each
/// once, ready to make again ([`replay`]).
pub struct Recorded {
    pipelines: Vec<Description>,
    /// The shader sets of the descriptions that read back, each once.
    pub shader_sets: HashSet<ShaderSet>,
    /// Kept descriptions that are not made again: the entries that do not
    /// read back, those whose state does not decode, those whose
    /// containers are not kept, and repeats (descriptions this build reads
    /// as the same pipeline).
    pub damaged: usize,
    pub missing: usize,
    pub repeated: usize,
    /// Of those made again, the pipelines a session drew with (the early
    /// list's): they come first.
    pub drawn: usize,
    /// After the kept ones, those [`derive`] added: they come last.
    pub derived: usize,
    /// The early keys of those made again, each once.
    keys: HashSet<String>,
    /// The file's size, and where it is (none without a pipeline cache).
    pub bytes: u64,
    pub path: Option<std::path::PathBuf>,
    /// The BLAKE3 of the file as read, in hexadecimal (`none` without
    /// one): what [`replay`] makes is this file's, whatever a session
    /// appends to it meanwhile, so that a note says this one.
    pub digest: String,
    /// The file was rewritten first, past its bound: its size before, and
    /// the descriptions that went.
    pub compacted: Option<(u64, usize)>,
    /// The file was of an earlier version, started again: its size.
    pub earlier: Option<u64>,
    /// The file is past its bound but was not rewritten: another process
    /// (a session) has it open.
    pub not_compacted: bool,
}

/// A kept pipeline as the session made it: its shaders, and its graphics
/// state read again from the title's stream.
struct Description {
    shaders: Vec<Arc<Shader>>,
    state: GraphicsState,
    /// The stream it was read from, and its shaders' containers' BLAKE3.
    stream: super::record::Stream,
    hashes: Vec<[u8; 32]>,
}

/// A pipeline's shader set: its containers' BLAKE3, in order, whatever
/// their stages (the pack's effects put a mesh program in the vertex
/// list).
pub type ShaderSet = Vec<[u8; 32]>;

/// The shader set of `containers`.
pub fn shader_set<'a>(containers: impl IntoIterator<Item = &'a [u8; 32]>) -> ShaderSet {
    let mut set: ShaderSet = containers.into_iter().copied().collect();
    set.sort_unstable();
    set
}

// SAFETY: the graphics state's input elements point into its own names,
// and it is only read once made (as `make::Pending`'s).
unsafe impl Sync for Description {}

impl Recorded {
    /// How many descriptions there are to make again.
    pub fn len(&self) -> usize {
        self.pipelines.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pipelines.is_empty()
    }
}

/// The pipelines sessions kept in the pipeline cache's folder, each once
/// under this build's key; none without a cache or a file. Its containers
/// join the shaders a session would share ([`super::shader`]).
pub fn recorded() -> Result<Recorded, String> {
    match super::library::dir() {
        Some(dir) => recorded_in(dir, super::record::COMPACT_AT, super::record::COMPACT_TO),
        None => Ok(Recorded::none()),
    }
}

impl Recorded {
    /// No descriptions, from no file.
    fn none() -> Recorded {
        Recorded {
            pipelines: Vec::new(),
            shader_sets: HashSet::new(),
            damaged: 0,
            missing: 0,
            repeated: 0,
            drawn: 0,
            derived: 0,
            keys: HashSet::new(),
            bytes: 0,
            path: None,
            digest: "none".into(),
            compacted: None,
            earlier: None,
            not_compacted: false,
        }
    }
}

/// [`recorded`] of the file in `dir`, rewritten first down to `to` bytes
/// if it is larger than `at` (or started again if it is of an earlier
/// version), unless another process has it open. The pipelines a session
/// drew with come first: a session keeps every pipeline the title creates
/// (some 80,000), and draws with a few thousand of them.
fn recorded_in(dir: &std::path::Path, at: u64, to: u64) -> Result<Recorded, String> {
    let mut out = Recorded::none();
    let mut file = super::record::read(dir)?;
    if file.bytes > at || file.earlier {
        match super::record::compact(dir, to)? {
            Some(c) => {
                tracing::info!(
                    before = c.before,
                    after = c.after,
                    kept = c.kept,
                    dropped = c.dropped,
                    earlier = c.earlier,
                    "recorded pipelines: rewritten"
                );
                if c.earlier {
                    out.earlier = Some(c.before);
                } else {
                    out.compacted = Some((c.before, c.dropped));
                }
            }
            None => {
                tracing::info!(
                    bytes = file.bytes,
                    "recorded pipelines: not rewritten, another process has the file open"
                );
                out.not_compacted = true;
            }
        }
        file = super::record::read(dir)?;
    }
    out.path = Some(dir.join(super::record::FILE));
    out.bytes = file.bytes;
    out.digest = file.digest.clone();
    out.damaged = file.damaged;
    let mut others = Vec::new();
    for kept in &file.pipelines {
        out.shader_sets
            .insert(shader_set(kept.shaders.iter().map(|(_, hash)| hash)));
        let Some(state) = kept.stream.state() else {
            out.damaged += 1;
            continue;
        };
        let Some(shaders) = kept
            .shaders
            .iter()
            .map(|(stage, hash)| Some(super::shader(stage, file.container(hash)?)))
            .collect::<Option<Vec<_>>>()
        else {
            out.missing += 1;
            continue;
        };
        let key = super::make::key(&shaders, Some(&state));
        if !out.keys.insert(key.clone()) {
            out.repeated += 1;
            continue;
        }
        let d = Description {
            shaders,
            state,
            stream: kept.stream.clone(),
            hashes: kept.shaders.iter().map(|&(_, hash)| hash).collect(),
        };
        if super::library::wanted_early(&key) {
            out.pipelines.push(d);
        } else {
            others.push(d);
        }
    }
    out.drawn = out.pipelines.len();
    out.pipelines.extend(others);
    Ok(out)
}

/// What [`derive`] found and added.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Derived {
    /// The shader sets of the pack's passes that draw, and of them those the
    /// kept descriptions have.
    pub sets: usize,
    pub covered: usize,
    /// The kept descriptions' shader sets that no pass names.
    pub other: usize,
    /// The sets no kept description has that got derived descriptions, and
    /// how many.
    pub derived_sets: usize,
    pub descriptions: usize,
}

/// How many of a mesh pass's state bytes [`derive`] compares, tier after
/// tier: ten, six, then none (its technique's name alone).
const TIERS: [usize; 3] = [10, 6, 0];

/// Kept descriptions (their indices) by a pass's technique name and state
/// bytes.
type ByPass<'a> = std::collections::HashMap<(&'a str, &'a [u8]), Vec<usize>>;

/// A derived pipeline's shaders, and their containers' BLAKE3.
type Shaders = (Vec<Arc<Shader>>, Vec<[u8; 32]>);

/// Adds to `recorded`, after the kept descriptions, descriptions derived for
/// the passes of the pack's `effects` whose shader set no kept description
/// has (an effect the session did not load: another place's materials, the
/// interface's): the pass's shaders, each with a graphics state kept for
/// passes like it. The pack gives a pass's shaders but not its state, which
/// the title overrides at creation; the kept descriptions give the states
/// the title makes, its targets, depth and vertex formats:
///
/// - a mesh pass takes the states of the kept passes of the same technique
///   (by name, in any effect) with the same state bytes: the first ten,
///   else the first six, else any. Tried on the 9,830 mesh shader sets a
///   Series S session to the world kept (2026-11-25), each predicted from
///   the others, this gave 96 % of their 24,644 descriptions, among 34,404
///   tries (69 % of them right);
/// - a vertex pass takes the states its vertex program was kept with (its
///   input layout is its own): 84 % of the 55,875 descriptions of the 2,323
///   vertex shader sets, among 47,556 tries (98 % right);
/// - a pass with a program of another list (geometry: the title made none)
///   gets none.
///
/// `container` gives the container of a record of the pack. The same
/// shaders with the same state are added once, and not again if kept.
/// Without `add`, nothing is added: the sets are counted alone.
pub fn derive(
    recorded: &mut Recorded,
    effects: &super::effects::Effects,
    add: bool,
    container: impl Fn(usize) -> Option<Vec<u8>>,
) -> Derived {
    derive_with(recorded, effects, add, container, |code| {
        stage(code) == Some(HwStage::Ms)
    })
}

/// [`derive`], whose vertex list's programs are mesh ones as `mesh` says.
fn derive_with(
    recorded: &mut Recorded,
    effects: &super::effects::Effects,
    add: bool,
    container: impl Fn(usize) -> Option<Vec<u8>>,
    mesh: impl Fn(&[u8]) -> bool,
) -> Derived {
    use super::effects::{COMPUTE, PIXEL, VERTEX};
    use std::collections::{BTreeMap, HashMap};
    let mut technique = vec![""; effects.passes.len()];
    for e in &effects.effects {
        for t in &e.techniques {
            for i in t.passes.clone() {
                technique[i] = &t.name;
            }
        }
    }
    // Each program's container BLAKE3, by record; each drawing pass's set.
    let mut hashes: HashMap<usize, Option<[u8; 32]>> = HashMap::new();
    let mut hash = |i: usize| {
        *hashes
            .entry(i)
            .or_insert_with(|| container(i).map(|c| *blake3::hash(&c).as_bytes()))
    };
    let mut set_passes: BTreeMap<ShaderSet, Vec<usize>> = BTreeMap::new();
    for (i, p) in effects.passes.iter().enumerate().filter(|(_, p)| p.draws()) {
        let programs = p
            .programs
            .iter()
            .enumerate()
            .filter(|&(list, _)| list != COMPUTE)
            .filter_map(|(_, r)| *r)
            .map(&mut hash)
            .collect::<Option<Vec<_>>>();
        if let Some(programs) = programs {
            set_passes.entry(shader_set(&programs)).or_default().push(i);
        }
    }
    let mut out = Derived {
        sets: set_passes.len(),
        covered: set_passes
            .keys()
            .filter(|s| recorded.shader_sets.contains(*s))
            .count(),
        other: recorded
            .shader_sets
            .iter()
            .filter(|s| !set_passes.contains_key(*s))
            .count(),
        ..Derived::default()
    };
    // The kept descriptions by vertex program, and the mesh ones by their
    // passes' technique and state bytes, tier by tier.
    // Each by the first kept description of its stream: thousands share a
    // few states, and a state is tried once.
    let mut by_vertex: HashMap<[u8; 32], Vec<usize>> = HashMap::new();
    let mut by_pass: [ByPass; TIERS.len()] = Default::default();
    let mut first: HashMap<&[u8], usize> = HashMap::new();
    for (d, desc) in recorded.pipelines.iter().enumerate() {
        let d = *first.entry(desc.stream.bytes()).or_insert(d);
        match desc.shaders.first().map(|s| s.stage) {
            Some("vertex") => by_vertex.entry(desc.hashes[0]).or_default().push(d),
            Some("mesh") => {
                for &p in set_passes
                    .get(&shader_set(&desc.hashes))
                    .into_iter()
                    .flatten()
                {
                    for (&n, pool) in TIERS.iter().zip(&mut by_pass) {
                        let key = (technique[p], &effects.passes[p].state[..n]);
                        pool.entry(key).or_default().push(d);
                    }
                }
            }
            _ => {}
        }
    }
    for list in by_vertex
        .values_mut()
        .chain(by_pass.iter_mut().flat_map(HashMap::values_mut))
    {
        list.sort_unstable();
        list.dedup();
    }
    let mut derived = Vec::new();
    for (set, passes) in &set_passes {
        if !add || recorded.shader_sets.contains(set) {
            continue;
        }
        let mut from: Vec<usize> = Vec::new();
        let mut shaders: Option<Shaders> = None;
        for &p in passes {
            let pass = &effects.passes[p];
            let vertex = pass.programs[VERTEX].expect("a pass that draws");
            if pass
                .programs
                .iter()
                .enumerate()
                .any(|(list, r)| r.is_some() && list != VERTEX && list != PIXEL)
            {
                continue;
            }
            let Some(code) = container(vertex) else {
                continue;
            };
            let mesh = mesh(&code);
            let found = if mesh {
                TIERS
                    .iter()
                    .zip(&by_pass)
                    .find_map(|(&n, pool)| pool.get(&(technique[p], &pass.state[..n])))
            } else {
                hash(vertex).and_then(|h| by_vertex.get(&h))
            };
            let Some(found) = found else {
                continue;
            };
            from.extend(found);
            if shaders.is_none() {
                let mut v = vec![super::shader(if mesh { "mesh" } else { "vertex" }, &code)];
                let mut h = vec![*blake3::hash(&code).as_bytes()];
                if let Some(pixel) = pass.programs[PIXEL] {
                    let Some(code) = container(pixel) else {
                        continue;
                    };
                    v.push(super::shader("pixel", &code));
                    h.push(*blake3::hash(&code).as_bytes());
                }
                shaders = Some((v, h));
            }
        }
        let Some((shaders, hashes)) = shaders else {
            continue;
        };
        from.sort_unstable();
        from.dedup();
        let before = derived.len();
        for d in from {
            let stream = &recorded.pipelines[d].stream;
            let Some(state) = stream.state() else {
                continue;
            };
            if recorded
                .keys
                .insert(super::make::key(&shaders, Some(&state)))
            {
                derived.push(Description {
                    shaders: shaders.clone(),
                    state,
                    stream: stream.clone(),
                    hashes: hashes.clone(),
                });
            }
        }
        out.derived_sets += usize::from(derived.len() > before);
    }
    out.descriptions = derived.len();
    recorded.derived = derived.len();
    recorded.pipelines.extend(derived);
    out
}

/// What became of one kept pipeline made again.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Replayed {
    /// Its host pipeline is in the library: made now or found there
    /// ([`counts`] tells how many of each).
    Ready,
    /// One of its shaders does not translate (the translator's refusal,
    /// kept in the shader cache).
    Untranslated,
    /// The host refused it, or a console program is refused before any
    /// translation (the reasons: [`refusals`]).
    Refused,
}

/// Makes the host pipeline of `recorded`'s pipeline `i` into the library,
/// as the session's first draw with it would: the same function, shaders
/// and state. One the library holds is not loaded.
pub fn replay(recorded: &Recorded, i: usize) -> Replayed {
    let d = &recorded.pipelines[i];
    let _ = super::library::take_last();
    if super::library::without_loading(|| HostPipeline::new(&d.shaders, Some(&d.state)).is_some()) {
        if super::library::take_last().cached {
            super::library::note_cached(&super::make::key(&d.shaders, Some(&d.state)));
        }
        return Replayed::Ready;
    }
    if d.shaders
        .iter()
        .any(|s| matches!(s.translated.get(), Some(None)))
    {
        Replayed::Untranslated
    } else {
        Replayed::Refused
    }
}

/// Saves the pipeline library if pipelines were added; why not, if the
/// save fails (what it was to write is written by the next save). The list
/// of title pipelines to start early is read first: no title pipeline was
/// created here, so nothing else read it, and the save writes back what it
/// holds.
pub fn finish() -> Result<(), String> {
    let _ = super::library::wanted_early("");
    match super::library::save(true) {
        super::library::Saved::Failed(e) => Err(e),
        super::library::Saved::Nothing | super::library::Saved::Written => Ok(()),
    }
}

#[cfg(test)]
pub(in crate::pipeline) mod tests {
    /// Each pipeline setting changes the note's line ([`PipelineSettings`]).
    #[test]
    fn each_pipeline_setting_changes_its_line() {
        let base = super::PipelineSettings {
            root_srv_tables: true,
            wave64_mesh_refused: false,
            lanes: Some((32, 32)),
            model: Some((6, 7)),
            depth_bounds: true,
            mesh_shaders: true,
        };
        assert_eq!(
            base.text(),
            "root-srv tables wave64-mesh translated lanes 32-32 model 6.7 depth-bounds yes mesh yes"
        );
        let changed: [fn(&mut super::PipelineSettings); 6] = [
            |s| s.root_srv_tables = false,
            |s| s.wave64_mesh_refused = true,
            |s| s.lanes = Some((32, 64)),
            |s| s.model = None,
            |s| s.depth_bounds = false,
            |s| s.mesh_shaders = false,
        ];
        for (i, change) in changed.iter().enumerate() {
            let mut other = base;
            change(&mut other);
            assert_ne!(other.text(), base.text(), "setting {i}");
        }
    }

    use super::*;

    /// A container's stage is its translated program's; none without one.
    /// (`apps/xtp/tests/precook.rs` precooks a real compute program.)
    #[test]
    fn stages_are_read_from_the_container() {
        assert_eq!(stage(b"not a container"), None);
        assert_eq!(stage(&dxbc(&[])), None, "no DXIL part");
        assert_eq!(stage(&dxbc(&[(b"DXIL", &[0; 8])])), None, "no program");
    }

    /// A DXBC container holding `parts`.
    fn dxbc(parts: &[(&[u8; 4], &[u8])]) -> Vec<u8> {
        let header = 0x20 + 4 * parts.len();
        let mut body = Vec::new();
        let mut offsets = Vec::new();
        for (fourcc, data) in parts {
            offsets.push((header + body.len()) as u32);
            body.extend(*fourcc);
            body.extend((data.len() as u32).to_le_bytes());
            body.extend(*data);
        }
        let mut out = b"DXBC".to_vec();
        out.extend([0; 16]);
        out.extend(1u16.to_le_bytes());
        out.extend(0u16.to_le_bytes());
        out.extend(((header + body.len()) as u32).to_le_bytes());
        out.extend((parts.len() as u32).to_le_bytes());
        for o in offsets {
            out.extend(o.to_le_bytes());
        }
        out.extend(body);
        out
    }

    /// A shader of `container`'s bytes, shared as a session shares it
    /// ([`super::super::shader`]), whose translation is already made:
    /// `source` compiled by FXC for `profile`, or, with no source, the
    /// translator's refusal. The translator is not this test's business
    /// (`apps/xtp/tests/precook.rs` translates a real program).
    pub(in crate::pipeline) fn translated(
        stage: &'static str,
        kind: HwStage,
        source: Option<&str>,
        profile: windows::core::PCSTR,
        container: &[u8],
    ) -> Arc<Shader> {
        translated_with(stage, kind, source, profile, container, |_| {})
    }

    /// [`translated`], its translation's bindings changed by `edit`.
    fn translated_with(
        stage: &'static str,
        kind: HwStage,
        source: Option<&str>,
        profile: windows::core::PCSTR,
        container: &[u8],
        edit: impl FnOnce(&mut xtp_shader::emit::HostShader),
    ) -> Arc<Shader> {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        let translated = source.map(|source| {
            let mut blob = None;
            // SAFETY: the source lives through compilation and the blob
            // owns its bytes.
            let dxil = unsafe {
                D3DCompile(
                    source.as_ptr().cast(),
                    source.len(),
                    windows::core::PCSTR::null(),
                    None,
                    None,
                    windows::core::s!("main"),
                    profile,
                    0,
                    0,
                    &mut blob,
                    None,
                )
                .unwrap();
                let blob = blob.unwrap();
                std::slice::from_raw_parts(
                    blob.GetBufferPointer().cast::<u8>(),
                    blob.GetBufferSize(),
                )
                .to_vec()
            };
            let mut host = xtp_shader::emit::HostShader {
                stage: kind,
                dxil,
                group_base: None,
                user_data: vec![],
                samplers: vec![],
                static_samplers: vec![],
                root_cbvs: vec![],
                root_srvs: vec![],
                root_uavs: vec![],
                stride_table: false,
                run_time_buffers: 0,
            };
            edit(&mut host);
            super::super::Translated {
                host,
                used: vec![],
                wave64: false,
            }
        });
        let shader = super::super::shader(stage, container);
        assert!(
            shader.translated.set(translated).is_ok(),
            "a container of this test's own"
        );
        shader
    }

    fn words(words: &[u32]) -> Vec<u8> {
        words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    /// The file of kept descriptions is rewritten once past its bound,
    /// before it is read: within room, nothing goes; without, all.
    #[test]
    fn the_kept_file_is_rewritten_past_its_bound() {
        let dir = tempfile::tempdir().unwrap();
        let stream = super::super::record::stream_of(&[(14, words(&[3]))]);
        let shaders = vec![
            super::super::shader("vertex", b"precook bound: vertex"),
            super::super::shader("pixel", b"precook bound: pixel"),
        ];
        assert!(super::super::record::keep_in(dir.path(), &shaders, &stream));
        let before = std::fs::metadata(dir.path().join(super::super::record::FILE))
            .unwrap()
            .len();
        let r = recorded_in(dir.path(), u64::MAX, u64::MAX).unwrap();
        assert_eq!((r.compacted, r.len()), (None, 1));
        let r = recorded_in(dir.path(), 0, u64::MAX).unwrap();
        assert_eq!((r.compacted, r.len()), (Some((before, 0)), 1));
        let r = recorded_in(dir.path(), 0, 0).unwrap();
        assert_eq!((r.compacted, r.len()), (Some((before, 1)), 0));
    }

    /// A pipeline a session drew with (the early list's) is made first,
    /// whatever its place in the file.
    #[test]
    fn pipelines_a_session_drew_with_come_first() {
        let dir = tempfile::tempdir().unwrap();
        let keep = |pixel: &[u8]| {
            let shaders = vec![
                super::super::shader("vertex", b"precook drawn: vertex"),
                super::super::shader("pixel", pixel),
            ];
            let stream = super::super::record::stream_of(&[(14, words(&[3]))]);
            assert!(super::super::record::keep_in(dir.path(), &shaders, &stream));
            super::super::make::key(&shaders, stream.state().as_ref())
        };
        let first = keep(b"precook drawn: never drawn");
        let drawn = keep(b"precook drawn: drawn");
        super::super::library::note_used(&drawn);
        let r = recorded_in(dir.path(), u64::MAX, u64::MAX).unwrap();
        let keys: Vec<String> = r
            .pipelines
            .iter()
            .map(|d| super::super::make::key(&d.shaders, Some(&d.state)))
            .collect();
        assert_eq!((r.drawn, keys), (1, vec![drawn, first]));
    }

    /// Derived descriptions: a mesh pass takes the states kept for its
    /// technique, the closest state bytes first; a vertex pass those its
    /// vertex program was kept with; a pass like no kept one gets none. The
    /// sets are counted, and a derived description goes after the kept ones.
    #[test]
    fn descriptions_are_derived_for_the_sets_no_session_kept() {
        use super::super::effects::{Effect, Effects, LISTS, Pass, Technique};
        let dir = tempfile::tempdir().unwrap();
        let code: Vec<Vec<u8>> = [
            "mesh a", "pixel a", "vs b", "pixel b", "mesh c", "pixel c", "mesh d", "pixel d",
            "mesh e", "pixel e",
        ]
        .iter()
        .map(|n| format!("derive {n}").into_bytes())
        .collect();
        let stream = |topology: u32| super::super::record::stream_of(&[(14, words(&[topology]))]);
        let keep = |stages: [&'static str; 2], records: [usize; 2], topology| {
            let shaders: Vec<_> = stages
                .iter()
                .zip(records)
                .map(|(stage, r)| super::super::shader(stage, &code[r]))
                .collect();
            assert!(super::super::record::keep_in(
                dir.path(),
                &shaders,
                &stream(topology)
            ));
        };
        keep(["mesh", "pixel"], [0, 1], 3);
        keep(["mesh", "pixel"], [0, 1], 4);
        keep(["vertex", "pixel"], [2, 3], 2);
        let pass = |effect, vertex: usize, pixel: usize, state: u8| {
            let mut programs = [None; LISTS];
            programs[0] = Some(vertex);
            programs[1] = Some(pixel);
            let mut bytes = [0u8; 18];
            bytes[8] = state;
            Pass {
                effect,
                programs,
                state: bytes,
                outside: false,
            }
        };
        let technique = |name: &str, passes| Technique {
            name: name.into(),
            passes,
        };
        let effects = Effects {
            effects: vec![
                Effect {
                    name: "kept".into(),
                    techniques: vec![technique("draw", 0..1), technique("depth", 1..2)],
                },
                Effect {
                    name: "not loaded".into(),
                    techniques: vec![
                        technique("draw", 2..4),
                        technique("depth", 4..5),
                        technique("other", 5..6),
                    ],
                },
            ],
            passes: vec![
                pass(0, 0, 1, 1),
                pass(0, 2, 3, 0),
                // The same technique and state bytes.
                pass(1, 4, 5, 1),
                // The same technique, other state bytes past the sixth.
                pass(1, 6, 5, 2),
                // The vertex program kept with another pixel one.
                pass(1, 2, 7, 0),
                // A technique no session kept.
                pass(1, 8, 9, 1),
            ],
        };
        let mut r = recorded_in(dir.path(), u64::MAX, u64::MAX).unwrap();
        assert_eq!(r.len(), 3);
        let d = derive_with(
            &mut r,
            &effects,
            true,
            |i| code.get(i).cloned(),
            |c| c.starts_with(b"derive mesh"),
        );
        assert_eq!(
            d,
            Derived {
                sets: 6,
                covered: 2,
                other: 0,
                derived_sets: 3,
                descriptions: 5,
            }
        );
        assert_eq!((r.len(), r.derived), (8, 5));
        let stages: Vec<Vec<&str>> = r.pipelines[3..]
            .iter()
            .map(|d| d.shaders.iter().map(|s| s.stage).collect())
            .collect();
        assert_eq!(
            stages.iter().filter(|s| s[..] == ["mesh", "pixel"]).count(),
            4
        );
        assert_eq!(
            stages
                .iter()
                .filter(|s| s[..] == ["vertex", "pixel"])
                .count(),
            1
        );
        // Derived once: not again, and nothing without `add`.
        let again = derive_with(
            &mut r,
            &effects,
            true,
            |i| code.get(i).cloned(),
            |c| c.starts_with(b"derive mesh"),
        );
        assert_eq!((again.descriptions, r.len()), (0, 8));
        let mut fresh = recorded_in(dir.path(), u64::MAX, u64::MAX).unwrap();
        let counted = derive_with(
            &mut fresh,
            &effects,
            false,
            |i| code.get(i).cloned(),
            |c| c.starts_with(b"derive mesh"),
        );
        assert_eq!(
            (counted.sets, counted.covered, counted.descriptions),
            (6, 2, 0)
        );
    }

    /// The descriptions made again are those of the file as read: its
    /// digest is that file's, not the one a session appends to meanwhile
    /// (the note says what was made, and the next precook makes the rest).
    #[test]
    fn the_digest_is_the_files_as_read() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(super::super::record::FILE);
        let none = recorded_in(dir.path(), u64::MAX, u64::MAX).unwrap();
        assert_eq!((none.len(), none.digest.as_str()), (0, "none"));
        let stream = super::super::record::stream_of(&[(14, words(&[3]))]);
        let keep = |pixel: &[u8]| {
            let shaders = vec![
                super::super::shader("vertex", b"precook digest: vertex"),
                super::super::shader("pixel", pixel),
            ];
            assert!(super::super::record::keep_in(dir.path(), &shaders, &stream));
        };
        keep(b"precook digest: pixel a");
        let read = recorded_in(dir.path(), u64::MAX, u64::MAX).unwrap();
        let digest = blake3::hash(&std::fs::read(&path).unwrap())
            .to_hex()
            .to_string();
        assert_eq!((read.len(), read.digest.as_str()), (1, digest.as_str()));
        // A session appends after the read.
        keep(b"precook digest: pixel b");
        assert_eq!(read.digest, digest, "the file as read");
        assert_ne!(digest_in(dir.path()).unwrap(), digest, "the file now");
    }

    /// A graphics pipeline a session kept is made again into the library
    /// by the same function, with the state read again from its stream:
    /// compiled the first time, the next time held by the library's names
    /// and not loaded (a session loads it). Two
    /// descriptions this build reads as one pipeline are made once; one
    /// whose shader no longer translates is refused at once.
    #[test]
    fn kept_graphics_pipelines_are_precooked_into_the_library() {
        let test =
            "pipeline::precook::tests::kept_graphics_pipelines_are_precooked_into_the_library";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        let vs = translated(
            "vertex",
            HwStage::Vs,
            Some(
                "float4 main(uint id : SV_VertexID) : SV_Position { return float4(id, 0, 0, 1); }",
            ),
            windows::core::s!("vs_5_1"),
            b"precook test: vertex",
        );
        let ps = translated(
            "pixel",
            HwStage::Ps,
            Some("float4 main() : SV_Target { return float4(0.25, 0.5, 0.75, 1); }"),
            windows::core::s!("ps_5_1"),
            b"precook test: pixel",
        );
        let refused = translated(
            "pixel",
            HwStage::Ps,
            None,
            windows::core::PCSTR::null(),
            b"precook test: refused pixel",
        );
        // Triangles into one R8G8B8A8_UNORM target, no depth, no culling.
        let stream = super::super::record::stream_of(&[
            (10, words(&[3, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0])),
            (11, words(&[0, 1, 2, 0, 0xffff, 1, 1, 1, 8, 1, 1, 1, 8])),
            (14, words(&[3])),
            (15, words(&[28, 0, 0, 0, 0, 0, 0, 0, 1])),
        ]);
        // The same with a node mask, which this build does not read.
        let masked = super::super::record::stream_of(&[
            (10, words(&[3, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0])),
            (11, words(&[0, 1, 2, 0, 0xffff, 1, 1, 1, 8, 1, 1, 1, 8])),
            (14, words(&[3])),
            (15, words(&[28, 0, 0, 0, 0, 0, 0, 0, 1])),
            (18, words(&[1])),
        ]);
        let shaders = vec![vs.clone(), ps.clone()];
        let dir = tempfile::tempdir().unwrap();
        let keep = super::super::record::keep_in;
        assert!(keep(dir.path(), &shaders, &stream));
        assert!(!keep(dir.path(), &shaders, &stream), "kept once");
        assert!(keep(dir.path(), &shaders, &masked));
        let other = vec![vs.clone(), refused.clone()];
        assert!(keep(dir.path(), &other, &stream));

        let recorded = recorded_in(dir.path(), u64::MAX, u64::MAX).unwrap();
        assert_eq!(
            (
                recorded.len(),
                recorded.repeated,
                recorded.missing,
                recorded.damaged
            ),
            (2, 1, 0, 0)
        );
        assert!(
            Arc::ptr_eq(&recorded.pipelines[0].shaders[0], &vs),
            "shared"
        );
        let made = counts().made;
        assert_eq!(replay(&recorded, 0), Replayed::Ready);
        assert!(counts().made > made, "compiled");
        // Made again by the precook: its name is held, so it is not loaded.
        // (Other tests of this process load and make pipelines meanwhile:
        // the counts only say that this one was answered by its name.)
        let known = counts().known;
        assert_eq!(replay(&recorded, 0), Replayed::Ready);
        assert!(counts().known > known, "held by the library");
        assert_eq!(replay(&recorded, 1), Replayed::Untranslated);

        // Keep the same translated shaders and change only the topology.
        // The new state needs its own library entry.
        let mut state = stream.state().unwrap();
        state.topology = windows::Win32::Graphics::Direct3D12::D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE;
        let made = counts().made;
        assert!(HostPipeline::new(&shaders, Some(&state)).is_some());
        assert!(counts().made > made, "compiled for the changed topology");
        let loaded = counts().loaded;
        assert!(HostPipeline::new(&shaders, Some(&state)).is_some());
        assert!(counts().loaded > loaded, "loaded for the same topology");
    }

    /// A pipeline is named by its root signature as well as by its
    /// translated code: two compute programs whose DXIL is byte for byte
    /// the same but whose bindings differ (a root CBV more) are two
    /// pipelines, and the precook does not answer the second by the
    /// first's name. (Whether a pipeline is answered by its name is told
    /// by the stand-in it gets: counts are the process's.)
    #[test]
    fn the_same_code_with_another_root_signature_is_another_pipeline() {
        let test = "pipeline::precook::tests::the_same_code_with_another_root_signature_is_another_pipeline";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        if !super::super::library::has_library() {
            return;
        }
        let source = "[numthreads(11, 3, 1)] void main() {}";
        let plain = translated(
            "compute",
            HwStage::Cs,
            Some(source),
            windows::core::s!("cs_5_1"),
            b"precook root test: plain",
        );
        let rooted = translated_with(
            "compute",
            HwStage::Cs,
            Some(source),
            windows::core::s!("cs_5_1"),
            b"precook root test: a root CBV more",
            |host| host.root_cbvs = vec![0],
        );
        let dxil = |s: &Shader| s.translated().map(|t| t.host.dxil.clone());
        assert_eq!(dxil(&plain), dxil(&rooted), "the same code");
        let precooked = |shader: &Arc<Shader>| {
            super::super::library::without_loading(|| {
                HostPipeline::new(std::slice::from_ref(shader), None).expect("a pipeline")
            })
        };
        let is_stand_in = |p: &HostPipeline| super::super::library::is_stand_in(&p.pso);
        // Made or answered by its name, then by its name.
        let _ = precooked(&plain);
        assert!(is_stand_in(&precooked(&plain)), "held by its name");
        assert!(
            !is_stand_in(&precooked(&rooted)),
            "another root signature: not answered by the plain one's name"
        );
        assert!(is_stand_in(&precooked(&rooted)), "then by its own");
    }
}
