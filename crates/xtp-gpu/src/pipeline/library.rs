//! Compiled host pipelines kept on disk, and the title pipelines worth
//! starting early (decision 0007, "Pipeline cache").
//!
//! 1. Every host pipeline goes into the host's pipeline library, named by
//!    a hash of all that its host description holds (the translated
//!    shaders' code, which changes when a translation does, our root
//!    signature, the graphics state). The library is saved under the cache
//!    folder; the host refuses a library of another adapter or driver, and
//!    an entry whose description differs, so neither is ever used.
//! 2. The title pipelines a run drew or dispatched with are listed by a key
//!    known when the title creates them (their shader containers and
//!    graphics state); the next run hands those to the workers at
//!    creation, during loading.
//! 3. Beside them, the whole descriptions of the graphics and mesh
//!    pipelines the title creates ([`super::record`]), from which `xtp
//!    precook` makes them into the library: those of the whole game after
//!    one session, and again after a translator change.
//!
//! Loads and stores of different pipelines run side by side: the library
//! is thread-safe and synchronises itself, but for two loads of the same
//! pipeline at once, which "may modify the state of that pipeline within
//! the library in a non-thread-safe manner" (D3D12's remarks on
//! `CreatePipelineLibrary`): one pipeline's name is used by one thread at a
//! time ([`NAMES`]). Serialising takes the library alone, so that its size
//! cannot change between asking for it and writing. With every call
//! serialised, the workers made some 35 pipelines a second, most of them
//! loads (a Series S session of 2026-11-25, `20261125-022031`: 1,465
//! loaded and 607 made, some 400 ms each on 14 workers), and its world
//! dropped some 900 draws a frame "being made" for a minute.
//!
//! - A save counts as done once its file is written: the pipelines it
//!   holds stop counting as added then, not before, so that a save that
//!   fails is made again by the next one.
//! - Generations: a library cannot drop an entry, and every translator
//!   change adds a name for each pipeline whose code changed, beside the
//!   old one. `xtp precook` compacts oversized files ([`rebuild`]): the
//!   library starts empty, what it asks for is taken from the generation
//!   before when that has it (loaded there, stored here) and made
//!   otherwise, and the save replaces the file with the new generation
//!   alone. What only sessions asked for is made again when one asks.
//! - One library file per translator ([`current`]): sessions of branches
//!   whose translators differ alternate on the PC, and one file compacted
//!   to one translator made the other's pipelines again at every switch
//!   (some 90,000, 13 to 23 minutes). Each translation key has its own
//!   generation (`host-pipelines-<id>.bin` and its names file), and a
//!   process opens only its own: the file it reads into memory is no
//!   larger than before. A key with no generation yet starts from the
//!   most recently opened one, its file linked (or copied), so that a
//!   change that alters few programs still finds every other pipeline by
//!   name. [`GENERATIONS_FILE`] lists them, most recently opened first;
//!   past [`GENERATIONS_KEPT`] of them or [`GENERATIONS_BYTES`] of files,
//!   the least recently opened go (never the current one). A file linked
//!   from another counts its bytes again (the budget errs on the safe
//!   side). The one file of before generations (`host-pipelines.bin`)
//!   becomes the first generation made. The list also says what a
//!   generation holds besides its translator's pipelines (seeded from
//!   another, or compacted, and its size then: [`Lineage`]), from which
//!   `xtp precook` decides its next compaction. Every write of a library
//!   file replaces it by a rename, so a generation linked from another
//!   never changes the other's file. The list is changed by one thread at a
//!   time in a process; two processes of different translators on one
//!   cache at once could each let the other's generation go: sessions and
//!   precooks on a machine run one at a time.
//! - The names the library holds: D3D12 cannot list them, so every save
//!   writes them beside the file (`host-pipelines-<id>.names`, [`Place`]),
//!   with the file's size and BLAKE3, which the next process checks before
//!   it trusts them. A library read without them (an older file, or one saved by a process that died
//!   before writing them) learns them as they load.
//! - The precook does not load a pipeline the library is known to hold
//!   ([`without_loading`]): its work is to have each pipeline in the file,
//!   and a session loads it when it needs it. That round trip was the whole
//!   cost of an unchanged pipeline after a translator change.
//! - A name the library holds but whose entry does not load for its
//!   description (StorePipeline refuses a name it holds, so before this the
//!   pipeline was compiled again in every session) gets the pipeline under
//!   an alternate name ([`SLOTS`]), which the next session loads.
//! - Such a name is remembered as broken in the names file: the precook
//!   answers a pipeline by its names only when one of them is held and not
//!   known to fail, and otherwise makes it (under the next name), so that
//!   a pipeline a load refused is not counted ready again. What no process
//!   has loaded yet cannot be known to fail without a load: the session
//!   that first finds such an entry makes the pipeline beside it and says
//!   so in the names file, and the precooks after it count it no more.
//! - Beside `early-pipelines.txt`, `slow-pipelines.txt` remembers title
//!   pipelines whose creation (not a library hit) took at least [`SLOW_MS`],
//!   so the scheduler can throttle early builds of slow pipelines without
//!   throttling library hits. Beside each generation's library file,
//!   `host-pipelines-<id>.keys` lists the early keys stored in that file
//!   (validated against its [`FileId`]) so library hits are recognised
//!   before translation.

use std::cell::Cell;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{HSTRING, Interface, PCWSTR};

static DIR: OnceLock<PathBuf> = OnceLock::new();

/// Keeps the pipeline library and the early list in `dir` from now on.
pub fn use_pipeline_cache(dir: PathBuf) {
    let _ = DIR.set(dir);
}

/// The folder the pipeline cache is kept in, once set.
pub(crate) fn dir() -> Option<&'static std::path::Path> {
    DIR.get().map(PathBuf::as_path)
}

/// Host pipelines created, loaded from the library, and title pipelines
/// started at creation, since the start.
pub(crate) static MADE: AtomicU64 = AtomicU64::new(0);
pub(crate) static LOADED: AtomicU64 = AtomicU64::new(0);
pub(crate) static EARLY: AtomicU64 = AtomicU64::new(0);
/// Host pipelines the precook found in the library's names and did not
/// load ([`without_loading`]), since the start.
pub(crate) static KNOWN: AtomicU64 = AtomicU64::new(0);
/// Microseconds pipelines waited for the library (their name, or a save),
/// and spent loading from it, since the start (the library's line).
static LOCK_US: AtomicU64 = AtomicU64::new(0);
static LOAD_US: AtomicU64 = AtomicU64::new(0);

struct Library {
    library: ID3D12PipelineLibrary1,
    /// The blob the library was made from: it must outlive the library.
    _blob: Vec<u8>,
    /// Changes since the last save, including replacing a generation.
    added: AtomicU64,
    /// While a new generation is made ([`rebuild`]): the one before, read
    /// only.
    previous: Option<Generation>,
    /// The names this library holds, as far as this process knows.
    names: RwLock<Names>,
    /// Names learned since the names file was last written.
    names_changed: AtomicBool,
    /// Early keys ([`super::make::key`]) of title pipelines known to be
    /// held by this library file, so library hits are never throttled.
    /// Shared with [`OPEN_KEYS`] so [`in_library`] never blocks on a save.
    keys: Arc<RwLock<HashSet<String>>>,
    /// Early keys learned since the keys file was last written.
    keys_changed: AtomicBool,
    /// The library file this library holds the same pipelines as: the one
    /// it was read from (`_blob`, accepted: hashed when first needed) or
    /// the one last saved. `None` with `_blob` not the file's (empty, or
    /// rejected by the host).
    file: Mutex<Option<FileId>>,
    blob_is_file: bool,
}

/// A generation of the library read only, the blob it was made from, and
/// the names it holds if its names file was its own.
struct Generation {
    library: ID3D12PipelineLibrary1,
    _blob: Vec<u8>,
    names: Option<Names>,
}

impl Library {
    /// A library of `blob` (read from the file when `blob_is_file`), with
    /// the names and early keys known to be in it.
    fn new(
        library: ID3D12PipelineLibrary1,
        blob: Vec<u8>,
        names: Names,
        keys: HashSet<String>,
        file: Option<FileId>,
        blob_is_file: bool,
        previous: Option<Generation>,
    ) -> Library {
        Library {
            library,
            _blob: blob,
            added: AtomicU64::new(0),
            previous,
            names: RwLock::new(names),
            names_changed: AtomicBool::new(false),
            keys: Arc::new(RwLock::new(keys)),
            keys_changed: AtomicBool::new(false),
            file: Mutex::new(file),
            blob_is_file,
        }
    }

    fn names(&self) -> std::sync::RwLockReadGuard<'_, Names> {
        self.names.read().unwrap_or_else(|e| e.into_inner())
    }

    /// `name` is held: stored by this process, refused by StorePipeline as
    /// one the library holds, or loaded.
    fn hold(&self, name: &str) {
        let mut names = self.names.write().unwrap_or_else(|e| e.into_inner());
        if names.held.insert(name.to_owned()) {
            self.names_changed.store(true, Ordering::Relaxed);
        }
    }

    /// A load of `name` for the description of that name loaded (`ok`), or
    /// failed: the name is held and mended, or marked broken if it is known
    /// to be held (a load also fails for a name the library does not hold,
    /// which a library read without its names may not know of).
    fn loaded(&self, name: &str, ok: bool) {
        let mut names = self.names.write().unwrap_or_else(|e| e.into_inner());
        let changed = if ok {
            let held = names.held.insert(name.to_owned());
            names.broken.remove(name) | held
        } else {
            names.held.contains(name) && names.broken.insert(name.to_owned())
        };
        if changed {
            self.names_changed.store(true, Ordering::Relaxed);
        }
    }

    /// Stores `pso` under the first of `slots` (its names, [`slot`]) that
    /// does not hold an entry this pipeline's description fails to load
    /// from: `failed` says the slots whose load failed here. A slot the
    /// library holds that was not tried holds this pipeline already (stored
    /// meanwhile by another thread: a name is its description's hash), and
    /// nothing more is stored; one that was tried and failed holds another
    /// entry, and the next slot is tried. A name StorePipeline refuses is
    /// held, by the same rules.
    /// Returns whether `pso` is now held in the library under a valid slot.
    fn store(
        &self,
        slots: &[(String, HSTRING)],
        failed: &[bool; SLOTS],
        pso: &ID3D12PipelineState,
    ) -> bool {
        for ((name, wide), &failed) in slots.iter().zip(failed) {
            if self.names().held.contains(name) {
                if failed {
                    continue;
                }
                return true;
            }
            // SAFETY: a live pipeline; a name already stored is refused,
            // which leaves the library as it was.
            match unsafe { self.library.StorePipeline(PCWSTR(wide.as_ptr()), pso) } {
                Ok(()) => {
                    self.added.fetch_add(1, Ordering::Relaxed);
                    self.hold(name);
                    return true;
                }
                Err(e) if e.code() == windows::Win32::Foundation::E_INVALIDARG => {
                    self.hold(name);
                    if !failed {
                        return true;
                    }
                    // Held, and its load failed here: another entry.
                    self.loaded(name, false);
                }
                Err(e) => {
                    tracing::debug!(name, error = %e, "host pipeline not stored");
                    return false;
                }
            }
        }
        tracing::debug!(
            name = slots[0].0,
            "host pipeline not stored: each of its names is held"
        );
        false
    }
}

/// A library file: its size and BLAKE3, which say whether a names file is
/// that file's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FileId {
    bytes: u64,
    blake3: [u8; 32],
}

impl FileId {
    fn of(bytes: &[u8]) -> FileId {
        FileId {
            bytes: bytes.len() as u64,
            blake3: *blake3::hash(bytes).as_bytes(),
        }
    }
}

/// The names a library holds, as far as a process knows them.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct Names {
    held: HashSet<String>,
    /// Of `held`, the names whose entry did not load for the description
    /// of that name when a process last loaded it (another description's
    /// entry, or one the driver refuses).
    broken: HashSet<String>,
    /// `held` is every name the library holds: it started empty, or its
    /// names file said so. Otherwise a name not in it may still be there.
    complete: bool,
}

/// The names file beside the one library file of before generations
/// ([`LIBRARY_FILE`]; a generation's is [`Place::names`]), and the first
/// line of a names file. Version 2 marks broken names ([`Names::broken`]);
/// version 1, which did not, is read too.
const NAMES_FILE: &str = "host-pipelines.names";
const NAMES_FORMAT: &str = "xtp pipeline names 2";
const NAMES_FORMAT_1: &str = "xtp pipeline names 1";
/// After a broken name in the names file.
const BROKEN: &str = " broken";

impl Names {
    /// The names of a library that holds none.
    fn empty() -> Names {
        Names {
            complete: true,
            ..Names::default()
        }
    }

    /// The names of a library read without its names file: none known.
    fn unknown() -> Names {
        Names::default()
    }

    /// Whether the library holds some name of the pipeline named `name`
    /// that is not known to fail to load ([`Names::broken`]): what the
    /// precook answers without loading ([`without_loading`]).
    fn ready_any(&self, name: &str) -> bool {
        (0..SLOTS)
            .map(|k| slot(name, k))
            .any(|n| self.held.contains(&n) && !self.broken.contains(&n))
    }

    /// The names file of a library file `file` holding these names.
    fn text(&self, file: &FileId) -> String {
        let mut held: Vec<&str> = self.held.iter().map(String::as_str).collect();
        held.sort_unstable();
        let mut text = format!(
            "{NAMES_FORMAT}\nlibrary {} {} {}\n",
            file.bytes,
            blake3::Hash::from(file.blake3).to_hex(),
            if self.complete { "complete" } else { "partial" }
        );
        for name in held {
            text.push_str(name);
            if self.broken.contains(name) {
                text.push_str(BROKEN);
            }
            text.push('\n');
        }
        text
    }

    /// The names a names file reading `text` gives the library file of
    /// `blob`; `None` if it is another file's (or not a names file).
    fn read(text: &str, blob: &[u8]) -> Option<(Names, FileId)> {
        let mut lines = text.lines();
        let marks = match lines.next()? {
            NAMES_FORMAT => true,
            NAMES_FORMAT_1 => false,
            _ => return None,
        };
        let mut head = lines.next()?.strip_prefix("library ")?.split(' ');
        let bytes: u64 = head.next()?.parse().ok()?;
        let hash = head.next()?;
        let complete = match head.next()? {
            "complete" => true,
            "partial" => false,
            _ => return None,
        };
        if head.next().is_some() || bytes != blob.len() as u64 {
            return None;
        }
        // Hashed only once the size agrees: some 0.5 s per GiB.
        let file = FileId::of(blob);
        (blake3::Hash::from(file.blake3).to_hex().as_str() == hash).then_some(())?;
        let mut names = Names {
            complete,
            ..Names::default()
        };
        for line in lines {
            let (name, broken) = match line.strip_suffix(BROKEN) {
                Some(name) if marks => (name, true),
                _ => (line, false),
            };
            is_name(name).then_some(())?;
            if broken {
                names.broken.insert(name.to_owned());
            }
            names.held.insert(name.to_owned());
        }
        Some((names, file))
    }
}

/// Whether `text` is a pipeline's name in the library ([`name`]) or one of
/// its alternates ([`slot`]).
fn is_name(text: &str) -> bool {
    let (name, k) = match text.split_once('~') {
        Some((name, k)) => (
            name,
            k.parse::<usize>().ok().filter(|k| (1..SLOTS).contains(k)),
        ),
        None => (text, Some(0)),
    };
    k.is_some() && name.len() == 32 && name.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// The names a pipeline may be stored under: its own, then alternates for
/// when the entry of its own name does not load for its description.
const SLOTS: usize = 4;

/// The `k`th name of the pipeline named `name` (`0`: its own).
fn slot(name: &str, k: usize) -> String {
    if k == 0 {
        name.to_owned()
    } else {
        format!("{name}~{k}")
    }
}

/// The early-key manifest beside a generation's library file
/// ([`Place::keys`]): lists title-pipeline keys ([`super::make::key`])
/// known to be stored in that library file, validated against its
/// [`FileId`].
const KEYS_FILE: &str = "host-pipelines.keys";
const KEYS_FORMAT: &str = "xtp pipeline keys 1";

/// Whether `text` is a 32-hex-digit early key ([`super::make::key`]).
fn is_early_key(text: &str) -> bool {
    text.len() == 32 && text.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

fn keys_text(keys: &HashSet<String>, file: &FileId) -> String {
    let mut sorted: Vec<&str> = keys.iter().map(String::as_str).collect();
    sorted.sort_unstable();
    let mut text = format!(
        "{KEYS_FORMAT}\nlibrary {} {}\n",
        file.bytes,
        blake3::Hash::from(file.blake3).to_hex()
    );
    for key in sorted {
        text.push_str(key);
        text.push('\n');
    }
    text
}

/// Reads a keys manifest if it matches `file` (or `blob`, hashing `blob`
/// only when its size matches and `known` is `None`).
fn read_keys(text: &str, blob: &[u8], known: Option<FileId>) -> Option<(HashSet<String>, FileId)> {
    let mut lines = text.lines();
    if lines.next()? != KEYS_FORMAT {
        return None;
    }
    let mut head = lines.next()?.strip_prefix("library ")?.split(' ');
    let bytes: u64 = head.next()?.parse().ok()?;
    let hash = head.next()?;
    if head.next().is_some() || bytes != blob.len() as u64 {
        return None;
    }
    let file = known.unwrap_or_else(|| FileId::of(blob));
    (blake3::Hash::from(file.blake3).to_hex().as_str() == hash).then_some(())?;
    let mut keys = HashSet::new();
    for line in lines {
        is_early_key(line).then_some(())?;
        keys.insert(line.to_owned());
    }
    Some((keys, file))
}

/// Outcome of the most recent [`pipeline`] call on this thread.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct LastPipeline {
    /// Loaded from the library (or answered via [`without_loading`])
    /// rather than compiled.
    pub(crate) hit: bool,
    /// Held by this process's library (loaded, known, or stored without
    /// failing).
    pub(crate) cached: bool,
}

thread_local! {
    /// Set by [`without_loading`] on the precook's threads.
    static SKIP: Cell<bool> = const { Cell::new(false) };
    /// Outcome of the most recent [`pipeline`] call on this thread.
    static LAST: Cell<LastPipeline> = const { Cell::new(LastPipeline { hit: false, cached: false }) };
}

/// Clears and returns the outcome of the most recent [`pipeline`] call on
/// this thread.
pub(crate) fn take_last() -> LastPipeline {
    LAST.replace(LastPipeline::default())
}

/// A pipeline of this process, answered for a pipeline the library holds
/// that [`without_loading`] does not load ([`found_in`] keeps the first it
/// makes or loads).
struct StandIn(ID3D12PipelineState);
// SAFETY: D3D12 objects are free-threaded, and this one is only held.
unsafe impl Send for StandIn {}
// SAFETY: as above.
unsafe impl Sync for StandIn {}
static STAND_IN: OnceLock<StandIn> = OnceLock::new();

/// Runs `make` (a host pipeline's creation, `HostPipeline::new`) with every
/// pipeline the library's names hold answered without loading it: the
/// pipeline `make` gets is then another one of this process (the first it
/// made or loaded), so only the precook calls this, as it keeps nothing it
/// makes. A pipeline the names do not hold is loaded or made as always.
/// The precook's work is to have each pipeline in the library file; a
/// session loads it, by its name, when it needs it.
pub(crate) fn without_loading<R>(make: impl FnOnce() -> R) -> R {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            SKIP.set(self.0);
        }
    }
    let _restore = Restore(SKIP.replace(true));
    make()
}

// SAFETY: D3D12 objects are free-threaded. The library synchronises its
// calls itself but for loads of one pipeline, which the name locks
// ([`NAMES`]) keep apart, and serialising is made with the library's write
// lock held, alone.
unsafe impl Send for Library {}
// SAFETY: as above.
unsafe impl Sync for Library {}

/// Locks by a pipeline's name: one thread at a time loads or stores a
/// pipeline of a name (one of these locks: a few names share one).
static NAMES: [Mutex<()>; 64] = [const { Mutex::new(()) }; 64];

/// The name lock of `name`.
fn name_lock(name: &str) -> &'static Mutex<()> {
    let h = name.bytes().fold(0usize, |h, b| {
        h.wrapping_mul(31).wrapping_add(usize::from(b))
    });
    &NAMES[h % NAMES.len()]
}

/// The one library file of before generations per translator, and its
/// names file: taken over by the first generation made ([`seed`]).
const LIBRARY_FILE: &str = "host-pipelines.bin";
const EARLY_FILE: &str = "early-pipelines.txt";

/// Where a generation's library file, its names file and its early-key
/// manifest are.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Place {
    library: PathBuf,
    names: PathBuf,
    keys: PathBuf,
}

impl Place {
    /// The files of generation `id` in `dir`.
    fn of(dir: &Path, id: &str) -> Place {
        Place {
            library: dir.join(format!("host-pipelines-{id}.bin")),
            names: dir.join(format!("host-pipelines-{id}.names")),
            keys: dir.join(format!("host-pipelines-{id}.keys")),
        }
    }

    /// The one library file of before generations ([`LIBRARY_FILE`]).
    fn legacy(dir: &Path) -> Place {
        Place {
            library: dir.join(LIBRARY_FILE),
            names: dir.join(NAMES_FILE),
            keys: dir.join(KEYS_FILE),
        }
    }
}

/// The generations of the library, one per translation key, most recently
/// opened first: one line each, `<id> own`, `<id> compacted <bytes>` or
/// `<id> from <id>` ([`Lineage`]).
const GENERATIONS_FILE: &str = "generations.txt";
const GENERATIONS_FORMAT: &str = "xtp pipeline generations 1";
/// Generations kept at most, the current one included, and the bytes their
/// files take at most (the current one is kept whatever its size). A
/// generation linked from another counts that file's bytes again.
const GENERATIONS_KEPT: usize = 4;
const GENERATIONS_BYTES: u64 = 8 << 30;
/// What a generation is seeded from when it takes over the library file of
/// before generations ([`LIBRARY_FILE`]).
pub(crate) const LEGACY: &str = "legacy";

/// A generation of the library: its translation key's, and what it holds
/// besides that translator's pipelines.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Lineage {
    /// The translation key's first 16 hexadecimal digits ([`generation_id`]).
    pub(crate) id: String,
    /// The generation its file started as (linked or copied: it holds that
    /// translator's pipelines too), [`LEGACY`] for the file of before, and
    /// not compacted since; `None` for one that started empty or was
    /// compacted.
    pub(crate) from: Option<String>,
    /// Its file's size just after its last compaction, while past the
    /// precook's bound (`precook::LIBRARY_REBUILD_BYTES`); `None` without
    /// one, or below the bound then.
    pub(crate) compacted: Option<u64>,
}

impl Lineage {
    /// A generation that starts empty.
    fn own(id: &str) -> Lineage {
        Lineage {
            id: id.to_owned(),
            from: None,
            compacted: None,
        }
    }

    fn line(&self) -> String {
        match (&self.from, self.compacted) {
            (Some(from), _) => format!("{} from {from}", self.id),
            (None, Some(bytes)) => format!("{} compacted {bytes}", self.id),
            (None, None) => format!("{} own", self.id),
        }
    }

    fn parse(line: &str) -> Option<Lineage> {
        let mut words = line.split(' ');
        let id = words.next().filter(|id| is_id(id))?;
        let mut lineage = Lineage::own(id);
        match (words.next()?, words.next()) {
            ("own", None) => {}
            ("compacted", Some(bytes)) => lineage.compacted = Some(bytes.parse().ok()?),
            ("from", Some(from)) if from == LEGACY || is_id(from) => {
                lineage.from = Some(from.to_owned());
            }
            _ => return None,
        }
        words.next().is_none().then_some(lineage)
    }
}

/// Whether `text` is a generation's id: 16 lower-case hexadecimal digits.
fn is_id(text: &str) -> bool {
    text.len() == 16 && text.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// The generation of this process's translation key: its translator, its
/// settings and the host's wave sizes (`shader_cache::host_key`). The host
/// must be made.
fn generation_id() -> String {
    crate::shader_cache::host_key()[..8]
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// The generations listed in `dir`, most recently opened first; none if
/// the list is missing or damaged (the files are then found by
/// [`settle`]).
fn read_generations(dir: &Path) -> Vec<Lineage> {
    let Ok(text) = std::fs::read_to_string(dir.join(GENERATIONS_FILE)) else {
        return Vec::new();
    };
    let mut lines = text.lines();
    if lines.next() != Some(GENERATIONS_FORMAT) {
        return Vec::new();
    }
    let mut list: Vec<Lineage> = Vec::new();
    for line in lines {
        match Lineage::parse(line) {
            Some(g) if !list.iter().any(|h| h.id == g.id) => list.push(g),
            Some(_) => {}
            None => return Vec::new(),
        }
    }
    list
}

fn write_generations(dir: &Path, list: &[Lineage]) -> std::io::Result<()> {
    let mut text = format!("{GENERATIONS_FORMAT}\n");
    for g in list {
        text.push_str(&g.line());
        text.push('\n');
    }
    write_atomically(&dir.join(GENERATIONS_FILE), text.as_bytes())
}

/// One process at a time changes the list of generations ([`settle`],
/// [`record_compaction`]); a precook and a session do not run together.
static GENERATIONS: Mutex<()> = Mutex::new(());

/// What [`settle`] did.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Settled {
    place: Place,
    lineage: Lineage,
    /// The generations that went, most recently opened first.
    evicted: Vec<String>,
    /// It had no file: started now (seeded, or empty). Its file is not the
    /// one an earlier process of its key left, whatever its size.
    new: bool,
}

/// Makes generation `id` in `dir` the most recently opened, its file
/// seeded from the most recently opened other one if it has none
/// ([`seed`]), and lets the oldest go past `kept` generations or `budget`
/// bytes of files. Library files of generations the list does not hold
/// (written by a process whose list was not) count as the oldest.
fn settle(dir: &Path, id: &str, kept: usize, budget: u64) -> std::io::Result<Settled> {
    let _one = GENERATIONS.lock().unwrap_or_else(|e| e.into_inner());
    std::fs::create_dir_all(dir)?;
    let mut list = read_generations(dir);
    let mut found: Vec<String> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok()?.file_name().into_string().ok())
        .filter_map(|n| {
            let id = n.strip_prefix("host-pipelines-")?.strip_suffix(".bin")?;
            is_id(id).then(|| id.to_owned())
        })
        .filter(|id| !list.iter().any(|g| g.id == *id))
        .collect();
    found.sort_unstable();
    list.extend(found.iter().map(|id| Lineage::own(id)));
    let place = Place::of(dir, id);
    let listed = list.iter().position(|g| g.id == id).map(|i| list.remove(i));
    let (lineage, new) = match listed {
        Some(lineage) if place.library.exists() => (lineage, false),
        // Listed, but its file is gone: seeded again.
        _ => (
            Lineage {
                from: seed(dir, &place, &list),
                ..Lineage::own(id)
            },
            true,
        ),
    };
    list.insert(0, lineage.clone());
    let evicted = evict(dir, &mut list, kept, budget);
    write_generations(dir, &list)?;
    Ok(Settled {
        place,
        lineage,
        evicted,
        new,
    })
}

/// Starts generation `place` as the most recently opened one of `list`
/// whose file is there (linked, else copied, with its names file), or as
/// the library file of before ([`LIBRARY_FILE`], moved). What it started
/// from: `None` if it starts empty.
fn seed(dir: &Path, place: &Place, list: &[Lineage]) -> Option<String> {
    let _ = std::fs::remove_file(&place.names);
    let _ = std::fs::remove_file(&place.keys);
    for g in list {
        let from = Place::of(dir, &g.id);
        if !from.library.exists() {
            continue;
        }
        let linked = std::fs::hard_link(&from.library, &place.library)
            .or_else(|_| std::fs::copy(&from.library, &place.library).map(|_| ()));
        match linked {
            Ok(()) => {
                // The names file says the file's size and BLAKE3: the same
                // bytes, the same names.
                let _ = std::fs::copy(&from.names, &place.names);
                tracing::info!(
                    from = g.id,
                    "host pipeline library: a new generation for this translator, from the last one opened"
                );
                return Some(g.id.clone());
            }
            Err(e) => {
                tracing::warn!(from = g.id, error = %e, "host pipeline library: a generation not seeded");
                let _ = std::fs::remove_file(&place.library);
            }
        }
    }
    let legacy = Place::legacy(dir);
    if legacy.library.exists() {
        match std::fs::rename(&legacy.library, &place.library) {
            Ok(()) => {
                let _ = std::fs::rename(&legacy.names, &place.names);
                tracing::info!(
                    "host pipeline library: the one file of before is this translator's generation now"
                );
                return Some(LEGACY.to_owned());
            }
            Err(e) => {
                tracing::warn!(error = %e, "host pipeline library: the file of before not taken over");
            }
        }
    }
    None
}

/// Keeps the first of `list` (the current generation) and those after it
/// while they are at most `kept` and their files at most `budget` bytes;
/// the others' files go, and their lines but for a file that could not be
/// removed. A listed generation without a file goes from the list. The
/// generations that went.
fn evict(dir: &Path, list: &mut Vec<Lineage>, kept: usize, budget: u64) -> Vec<String> {
    let mut evicted = Vec::new();
    let (mut count, mut total) = (0usize, 0u64);
    list.retain(|g| {
        let place = Place::of(dir, &g.id);
        let bytes = std::fs::metadata(&place.library).map(|m| m.len());
        let first = count == 0;
        if first {
            count += 1;
            total = bytes.unwrap_or(0);
            return true;
        }
        let Ok(bytes) = bytes else {
            return false;
        };
        if count < kept && total.saturating_add(bytes) <= budget {
            count += 1;
            total += bytes;
            return true;
        }
        match std::fs::remove_file(&place.library) {
            Ok(()) => {
                let _ = std::fs::remove_file(&place.names);
                let _ = std::fs::remove_file(&place.keys);
                tracing::info!(
                    generation = g.id,
                    bytes,
                    "host pipeline library: the least recently opened generation goes"
                );
                evicted.push(g.id.clone());
                false
            }
            Err(e) => {
                tracing::warn!(generation = g.id, error = %e, "host pipeline library: a generation could not go");
                true
            }
        }
    });
    evicted
}

/// Notes that generation `id` in `dir` was compacted into a file of `bytes`
/// ([`rebuild`]): it holds its translator's pipelines alone, and `bytes` is
/// its baseline if past `bound`.
fn record_compaction(dir: &Path, id: &str, bytes: u64, bound: u64) -> std::io::Result<()> {
    let _one = GENERATIONS.lock().unwrap_or_else(|e| e.into_inner());
    let mut list = read_generations(dir);
    let compacted = Lineage {
        compacted: (bytes > bound).then_some(bytes),
        ..Lineage::own(id)
    };
    match list.iter_mut().find(|g| g.id == id) {
        Some(g) => *g = compacted,
        None => list.insert(0, compacted),
    }
    write_generations(dir, &list)
}

/// This process's generation of the library: where its files are, and what
/// it held when settled ([`settle`], on first use: the host must be made).
/// `None` without a cache folder.
fn current() -> Option<&'static Settled> {
    static CURRENT: OnceLock<Option<Settled>> = OnceLock::new();
    CURRENT
        .get_or_init(|| {
            let dir = DIR.get()?;
            let id = generation_id();
            match settle(dir, &id, GENERATIONS_KEPT, GENERATIONS_BYTES) {
                Ok(s) => {
                    tracing::info!(
                        generation = s.lineage.id,
                        from = s.lineage.from.as_deref(),
                        new = s.new,
                        evicted = ?s.evicted,
                        "host pipeline library generation"
                    );
                    Some(s)
                }
                Err(e) => {
                    tracing::warn!(error = %e, "host pipeline library generations not listed");
                    // Its file, as it is: not known to be the one before.
                    Some(Settled {
                        place: Place::of(dir, &id),
                        lineage: Lineage::own(&id),
                        evicted: Vec::new(),
                        new: true,
                    })
                }
            }
        })
        .as_ref()
}

/// This process's generation of the library, as settled ([`current`]), and
/// whether it started in this process (`Settled::new`); `None` without a
/// cache folder.
pub(crate) fn generation() -> Option<(Lineage, bool)> {
    current().map(|s| (s.lineage.clone(), s.new))
}

/// Title pipelines listed for an early start, at most.
const EARLY_MAX: usize = 8192;

/// The library is opened on first use: once opened, [`rebuild`] comes too
/// late.
static OPENED: AtomicBool = AtomicBool::new(false);
static REBUILD: AtomicBool = AtomicBool::new(false);
/// The library was opened as a new generation ([`new_generation`]): what
/// it saves holds this translator's pipelines alone ([`record_compaction`]),
/// even when the file before was rejected or empty.
static REBUILT: AtomicBool = AtomicBool::new(false);

static LIBRARY: OnceLock<Option<RwLock<Library>>> = OnceLock::new();
/// Shared handle to the opened library's early-key set, readable by
/// [`in_library`] without taking the outer `Library` lock (which saves hold
/// exclusively while serialising).
static OPEN_KEYS: OnceLock<Arc<RwLock<HashSet<String>>>> = OnceLock::new();

fn library() -> Option<&'static RwLock<Library>> {
    let mut wake = false;
    let lib = LIBRARY
        .get_or_init(|| {
            OPENED.store(true, Ordering::SeqCst);
            let device = crate::host::host()?.device.cast::<ID3D12Device1>().ok()?;
            // This translator's generation alone.
            let place = current().map(|s| &s.place);
            let blob = place
                .and_then(|p| std::fs::read(&p.library).ok())
                .unwrap_or_default();
            let names = place.and_then(|p| std::fs::read_to_string(&p.names).ok());
            let keys = place.and_then(|p| std::fs::read_to_string(&p.keys).ok());
            let library = if REBUILD.load(Ordering::SeqCst) {
                let library = new_generation(&device, blob, names.as_deref());
                REBUILT.store(library.is_some(), Ordering::SeqCst);
                library
            } else {
                open(&device, blob, names.as_deref(), keys.as_deref())
            };
            if let Some(lib) = &library {
                let _ = OPEN_KEYS.set(Arc::clone(&lib.keys));
                wake = !lib
                    .keys
                    .read()
                    .unwrap_or_else(|e| e.into_inner())
                    .is_empty();
            } else {
                tracing::warn!("no host pipeline library: pipelines are not kept");
            }
            library.map(RwLock::new)
        })
        .as_ref();
    if wake {
        // Workers sleeping behind `slow_limit` while the library was still
        // opening can now pick up library hits from `Queues::slow`.
        super::make::wake_workers();
    }
    lib
}

/// Whether the title pipeline with early key `key` is known to be held by
/// the already-open pipeline library (never blocks on opening or saving).
pub(crate) fn in_library(key: &str) -> bool {
    let Some(keys) = OPEN_KEYS.get() else {
        return false;
    };
    keys.read().unwrap_or_else(|e| e.into_inner()).contains(key)
}

/// Records that the title pipeline with early key `key` is held by this
/// generation's pipeline library (loaded from it or stored into it).
pub(crate) fn note_cached(key: &str) {
    if !is_early_key(key) {
        return;
    }
    let Some(open_keys) = OPEN_KEYS.get() else {
        return;
    };
    if open_keys
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .contains(key)
    {
        return;
    }
    let Some(Some(library)) = LIBRARY.get() else {
        return;
    };
    let l = library.read().unwrap_or_else(|e| e.into_inner());
    let mut keys = l.keys.write().unwrap_or_else(|e| e.into_inner());
    if keys.insert(key.to_owned()) {
        l.keys_changed.store(true, Ordering::Release);
    }
}

/// Makes the library a new generation, the file's library its generation
/// before (see the module): asked before the library's first use (`xtp
/// precook`'s start), else too late, which it says.
pub(crate) fn rebuild() -> bool {
    REBUILD.store(true, Ordering::SeqCst);
    !OPENED.load(Ordering::SeqCst)
}

/// The size of this process's generation's library file, if there is one
/// (settled first: [`current`]).
pub(crate) fn file_bytes() -> Option<u64> {
    std::fs::metadata(&current()?.place.library)
        .ok()
        .map(|m| m.len())
}

/// The host's library of `bytes`, which must outlive it.
fn make(device: &ID3D12Device1, bytes: &[u8]) -> windows::core::Result<ID3D12PipelineLibrary1> {
    // SAFETY: the callers keep the bytes beside the library.
    unsafe {
        device
            .CreatePipelineLibrary::<ID3D12PipelineLibrary>(bytes)?
            .cast()
    }
}

/// The library in `blob`, or an empty one if the host rejects the blob (an
/// adapter or driver change, damage), with the names `names` (the names
/// file's text) and early keys `keys` (the keys manifest's text) it holds
/// if they are that blob's.
fn open(
    device: &ID3D12Device1,
    blob: Vec<u8>,
    names: Option<&str>,
    keys: Option<&str>,
) -> Option<Library> {
    if !blob.is_empty() {
        match make(device, &blob) {
            Ok(library) => {
                let read = names.and_then(|text| Names::read(text, &blob));
                tracing::info!(
                    bytes = blob.len(),
                    names = read.as_ref().map(|(n, _)| n.held.len()),
                    "host pipeline library loaded"
                );
                let (names, mut file) = match read {
                    Some((names, file)) => (names, Some(file)),
                    None => (Names::unknown(), None),
                };
                let keys = match keys.and_then(|text| read_keys(text, &blob, file)) {
                    Some((keys, id)) => {
                        file = Some(id);
                        keys
                    }
                    None => HashSet::new(),
                };
                return Some(Library::new(library, blob, names, keys, file, true, None));
            }
            Err(e) => {
                tracing::info!(error = %e, "host pipeline library rejected; starting an empty one")
            }
        }
    } else {
        tracing::info!("no host pipeline library yet; starting an empty one");
    }
    let library = make(device, &[]).ok()?;
    Some(Library::new(
        library,
        Vec::new(),
        Names::empty(),
        HashSet::new(),
        None,
        false,
        None,
    ))
}

/// An empty library, the one in `blob` its generation before (none if the
/// host rejects it), with the names `names` says it holds.
fn new_generation(device: &ID3D12Device1, blob: Vec<u8>, names: Option<&str>) -> Option<Library> {
    let previous = if blob.is_empty() {
        None
    } else {
        match make(device, &blob) {
            Ok(library) => Some(Generation {
                names: names
                    .and_then(|text| Names::read(text, &blob))
                    .map(|(n, _)| n),
                library,
                _blob: blob,
            }),
            Err(e) => {
                tracing::info!(error = %e, "host pipeline library rejected: the new generation starts from nothing");
                None
            }
        }
    };
    tracing::info!(
        before = previous.as_ref().map_or(0, |g| g._blob.len()),
        "host pipeline library: a new generation, from the one before"
    );
    let library = make(device, &[]).ok()?;
    let library = Library::new(
        library,
        Vec::new(),
        Names::empty(),
        HashSet::new(),
        None,
        false,
        previous,
    );
    // Even a generation with no surviving pipelines replaces its file.
    library.added.store(1, Ordering::Relaxed);
    Some(library)
}

/// A pipeline's name in the library: the hash of what its host
/// description holds.
pub(crate) fn name(kind: &str, root: &[u8], code: &[&[u8]], state: &str) -> String {
    let mut h = blake3::Hasher::new();
    for part in [kind.as_bytes(), root]
        .into_iter()
        .chain(code.iter().copied())
        .chain([state.as_bytes()])
    {
        h.update(&(part.len() as u64).to_le_bytes());
        h.update(part);
    }
    h.finalize().to_hex()[..32].to_owned()
}

/// The pipeline named `name`: loaded from the library by `load` if it is
/// there with the same description, else made by `create` and stored.
pub(crate) fn pipeline(
    name: &str,
    load: impl Fn(&ID3D12PipelineLibrary1, PCWSTR) -> windows::core::Result<ID3D12PipelineState>,
    create: impl FnOnce() -> windows::core::Result<ID3D12PipelineState>,
) -> windows::core::Result<ID3D12PipelineState> {
    let Some(library) = library() else {
        LAST.set(LastPipeline::default());
        MADE.fetch_add(1, Ordering::Relaxed);
        return create();
    };
    match found_in(library, name, load, create) {
        Ok((pso, source, cached)) => {
            LAST.set(LastPipeline {
                hit: !matches!(source, Source::Made),
                cached,
            });
            Ok(pso)
        }
        Err(e) => {
            LAST.set(LastPipeline::default());
            Err(e)
        }
    }
}

/// Where a pipeline came from ([`found_in`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Source {
    Library,
    /// The generation before (stored into the library).
    Previous,
    Made,
    /// Held by the library and not loaded ([`without_loading`]): the
    /// pipeline answered is a stand-in.
    Known,
}

/// [`pipeline`] in `library`, and where it came from.
///
/// A pipeline is looked for under each of its names ([`slot`]) the library
/// is known to hold, and under its own name when the library's names are
/// not all known. Made, it is stored under the first name the library does
/// not hold: an entry of its name that does not load for this description
/// keeps its name, and the pipeline goes beside it, where the next session
/// finds it.
fn found_in(
    library: &RwLock<Library>,
    name: &str,
    load: impl Fn(&ID3D12PipelineLibrary1, PCWSTR) -> windows::core::Result<ID3D12PipelineState>,
    create: impl FnOnce() -> windows::core::Result<ID3D12PipelineState>,
) -> windows::core::Result<(ID3D12PipelineState, Source, bool)> {
    if SKIP.get()
        && let Some(stand_in) = STAND_IN.get()
        && library
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .names()
            .ready_any(name)
    {
        KNOWN.fetch_add(1, Ordering::Relaxed);
        return Ok((stand_in.0.clone(), Source::Known, true));
    }
    let slots: Vec<(String, HSTRING)> = (0..SLOTS)
        .map(|k| {
            let name = slot(name, k);
            let wide = HSTRING::from(&name);
            (name, wide)
        })
        .collect();
    let since = |t: std::time::Instant| t.elapsed().as_micros() as u64;
    // The name's lock, then the library's (a save takes it alone).
    let take = || {
        let asked = std::time::Instant::now();
        let held = name_lock(name).lock().unwrap_or_else(|e| e.into_inner());
        let l = library.read().unwrap_or_else(|e| e.into_inner());
        LOCK_US.fetch_add(since(asked), Ordering::Relaxed);
        (held, l)
    };
    // The slots worth a load in a library of `names`: those it holds, or
    // all of them when `names` may miss some (a library read without its
    // names file: a load that finds no entry costs little).
    let tried = |names: Option<&Names>| -> Vec<usize> {
        (0..SLOTS)
            .filter(|&k| names.is_none_or(|n| !n.complete || n.held.contains(&slots[k].0)))
            .collect()
    };
    // The slots of this library whose load failed: their entries are other
    // descriptions', and a store goes past them ([`Library::store`]).
    let mut failed = [false; SLOTS];
    let loaded = {
        let (_name, l) = take();
        let started = std::time::Instant::now();
        let mut loaded = None;
        let ours = tried(Some(&l.names()));
        for k in ours {
            match load(&l.library, PCWSTR(slots[k].1.as_ptr())) {
                Ok(pso) => {
                    // Learned, when the names were not all known, and
                    // mended if a load failed before.
                    l.loaded(&slots[k].0, true);
                    loaded = Some((pso, Source::Library, true));
                    break;
                }
                Err(_) => {
                    failed[k] = true;
                    l.loaded(&slots[k].0, false);
                }
            }
        }
        if loaded.is_none()
            && let Some(previous) = &l.previous
        {
            for k in tried(previous.names.as_ref()) {
                if let Ok(pso) = load(&previous.library, PCWSTR(slots[k].1.as_ptr())) {
                    let stored = l.store(&slots, &failed, &pso);
                    loaded = Some((pso, Source::Previous, stored));
                    break;
                }
            }
        }
        LOAD_US.fetch_add(since(started), Ordering::Relaxed);
        loaded
    };
    if let Some(found) = loaded {
        LOADED.fetch_add(1, Ordering::Relaxed);
        keep_stand_in(&found.0);
        return Ok(found);
    }
    // Compiled outside the locks: compiles take up to seconds.
    let pso = create()?;
    MADE.fetch_add(1, Ordering::Relaxed);
    let (_name, l) = take();
    let stored = l.store(&slots, &failed, &pso);
    keep_stand_in(&pso);
    Ok((pso, Source::Made, stored))
}

/// For tests: whether `pso` is the stand-in [`without_loading`] answers
/// for a pipeline the library's names hold.
#[cfg(test)]
pub(crate) fn is_stand_in(pso: &ID3D12PipelineState) -> bool {
    STAND_IN.get().is_some_and(|s| s.0 == *pso)
}

/// For tests: whether this process has a pipeline library (none: every
/// pipeline is made, and [`without_loading`] answers none by its name).
#[cfg(test)]
pub(crate) fn has_library() -> bool {
    library().is_some()
}

/// Keeps `pso` as the stand-in [`without_loading`] answers, if none is.
fn keep_stand_in(pso: &ID3D12PipelineState) {
    if STAND_IN.get().is_none() {
        let _ = STAND_IN.set(StandIn(pso.clone()));
    }
}

/// What a save did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Saved {
    /// Nothing to write (nothing added, the early list as it was), or not
    /// yet time to, or no library or cache folder.
    Nothing,
    Written,
    /// The reason; what was to be written still is.
    Failed(String),
}

/// Writes the library if pipelines were added since the last save, and the
/// early list with it or if it changed, once `SAVE_EVERY` has passed (or
/// `now`). The library is serialised only when it holds something new: it
/// takes the lock every load waits for, and the cache's library had grown
/// to 2.3 GB (2026-11-25), which a session wrote again every 20 s while it
/// drew new pipelines that the library already held.
///
/// One save at a time ([`write_library`]): a worker's gives way to one
/// under way, one asked `now` (at the end of a run or of a precook) waits
/// for it.
pub(crate) fn save(now: bool) -> Saved {
    static LAST: Mutex<Option<std::time::Instant>> = Mutex::new(None);
    const SAVE_EVERY: std::time::Duration = std::time::Duration::from_secs(20);
    let (Some(dir), Some(Settled { place, lineage, .. })) = (DIR.get(), current()) else {
        return Saved::Nothing;
    };
    let Some(library) = library() else {
        return if REBUILD.load(Ordering::SeqCst) {
            Saved::Failed("the host could not create the replacement pipeline library".into())
        } else {
            Saved::Nothing
        };
    };
    {
        let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
        if !now && last.is_some_and(|t| t.elapsed() < SAVE_EVERY) {
            return Saved::Nothing;
        }
        *last = Some(std::time::Instant::now());
    }
    let mut library_bytes = None;
    let saved = write_library(library, place, now, |bytes| {
        library_bytes = bytes;
        // Under the save's lock: whether the early or slow list changed
        // since the last save that wrote it.
        let used_changed = USED_CHANGED.swap(false, Ordering::AcqRel);
        let slow_changed = SLOW_CHANGED.swap(false, Ordering::AcqRel);
        if bytes.is_none() && !used_changed && !slow_changed {
            return Ok(false);
        }
        let early_count = if used_changed {
            let used: Vec<String> = used().iter().flatten().cloned().collect();
            if let Err(e) = write_atomically(&dir.join(EARLY_FILE), used.join("\n").as_bytes()) {
                USED_CHANGED.store(true, Ordering::Release);
                if slow_changed {
                    SLOW_CHANGED.store(true, Ordering::Release);
                }
                return Err(e);
            }
            used.len()
        } else {
            USED.lock()
                .unwrap_or_else(|e| e.into_inner())
                .as_ref()
                .map_or(0, HashSet::len)
        };
        let slow_count = if slow_changed {
            let snapshot = slow_ones().clone().unwrap_or_default();
            if let Err(e) = write_slow(dir, &snapshot) {
                SLOW_CHANGED.store(true, Ordering::Release);
                return Err(e);
            }
            snapshot.len()
        } else {
            SLOW.lock()
                .unwrap_or_else(|e| e.into_inner())
                .as_ref()
                .map_or(0, HashMap::len)
        };
        let (created, unkept, distinct) = super::record::created_counts();
        tracing::info!(
            library = bytes.is_some(),
            bytes = bytes.unwrap_or(0),
            early = early_count,
            slow = slow_count,
            made = MADE.load(Ordering::Relaxed),
            loaded = LOADED.load(Ordering::Relaxed),
            known = KNOWN.load(Ordering::Relaxed),
            started_early = EARLY.load(Ordering::Relaxed),
            created,
            created_unkept = unkept,
            created_distinct = distinct,
            recorded = super::record::kept(),
            lock_wait_ms = LOCK_US.load(Ordering::Relaxed) / 1000,
            load_ms = LOAD_US.load(Ordering::Relaxed) / 1000,
            "host pipeline library saved"
        );
        Ok(true)
    });
    if let Saved::Failed(e) = &saved {
        tracing::warn!(error = %e, "host pipeline library not saved");
    }
    // A new generation written: the list says it holds its translator's
    // pipelines alone now, and its size (the last save's, a precook's
    // final one, is the one that stays).
    if let (Saved::Written, Some(bytes)) = (&saved, library_bytes)
        && REBUILT.load(Ordering::SeqCst)
        && let Err(e) = record_compaction(
            dir,
            &lineage.id,
            bytes as u64,
            super::precook::LIBRARY_REBUILD_BYTES,
        )
    {
        tracing::warn!(error = %e, "host pipeline library generations not listed");
    }
    saved
}

/// Writes `library` into `place` if pipelines were added to it, then runs
/// `then` with its size (`None`: none written), which says whether it
/// wrote something too. The pipelines the library holds stop counting as
/// added once both are done, not before.
///
/// One at a time, from what it serialises to that count: two at once would
/// both write what was added before them, then both stop counting it (the
/// count would wrap), and share the temporary files. With `wait`, it waits
/// for one under way; without, it gives way to it ([`Saved::Nothing`]).
fn write_library(
    library: &RwLock<Library>,
    place: &Place,
    wait: bool,
    then: impl FnOnce(Option<usize>) -> std::io::Result<bool>,
) -> Saved {
    static SAVING: Mutex<()> = Mutex::new(());
    let _saving = if wait {
        SAVING.lock().unwrap_or_else(|e| e.into_inner())
    } else {
        match SAVING.try_lock() {
            Ok(saving) => saving,
            Err(std::sync::TryLockError::Poisoned(e)) => e.into_inner(),
            Err(std::sync::TryLockError::WouldBlock) => return Saved::Nothing,
        }
    };
    let path = &place.library;
    let tmp = path.with_extension(format!("{}.tmp", std::process::id()));
    let serialised = (|| -> std::io::Result<_> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let (mapped, added, names, keys) = {
            // Alone: no store between the size and the bytes, nor between
            // the bytes and the names/keys they hold.
            let l = library.write().unwrap_or_else(|e| e.into_inner());
            let added = l.added.load(Ordering::Relaxed);
            if added == 0 {
                // The file holds what the library holds: the names and
                // early keys learned since are its.
                let names = l
                    .names_changed
                    .swap(false, Ordering::Relaxed)
                    .then(|| l.names().clone());
                let keys = l
                    .keys_changed
                    .swap(false, Ordering::AcqRel)
                    .then(|| l.keys.read().unwrap_or_else(|e| e.into_inner()).clone());
                drop(l);
                if let Some(names) = names {
                    write_names(library, place, &names, None);
                }
                if let Some(keys) = keys {
                    write_keys(library, place, &keys, None);
                }
                return Ok(None);
            }
            l.names_changed.store(false, Ordering::Relaxed);
            l.keys_changed.store(false, Ordering::AcqRel);
            let names = l.names().clone();
            let keys = l.keys.read().unwrap_or_else(|e| e.into_inner()).clone();
            (
                LibraryFile::serialize(&tmp, &l.library)?,
                added,
                names,
                keys,
            )
        };
        let size = mapped.size;
        mapped.flush()?;
        // SAFETY: the view holds `size` bytes the library wrote, and lives
        // until `mapped` is dropped below.
        let file =
            FileId::of(unsafe { std::slice::from_raw_parts(mapped.view.Value.cast(), size) });
        drop(mapped);
        std::fs::rename(&tmp, path)?;
        write_names(library, place, &names, Some(file));
        if keys.is_empty() {
            let _ = std::fs::remove_file(&place.keys);
        } else {
            write_keys(library, place, &keys, Some(file));
        }
        Ok(Some((size, added)))
    })();
    let serialised = match serialised {
        Ok(result) => result,
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            return Saved::Failed(e.to_string());
        }
    };
    let written = then(serialised.map(|(size, _)| size));
    match (written, serialised) {
        (Err(e), _) => Saved::Failed(e.to_string()),
        (Ok(_), Some((_, added))) => {
            // Those stored meanwhile still count, for the next save.
            let l = library.read().unwrap_or_else(|e| e.into_inner());
            l.added.fetch_sub(added, Ordering::Relaxed);
            Saved::Written
        }
        (Ok(true), None) => Saved::Written,
        (Ok(false), None) => Saved::Nothing,
    }
}

/// Writes the names file of the library file in `place`: `names`, held by the
/// file just written (`file`, its identity), or by the file the library was
/// read from or last saved to (`None`). The names stay to write again if it
/// fails, or if that file is not known (a library that did not read back),
/// and a file whose names are not written is read without them: what the
/// names file says is always its library file's.
fn write_names(library: &RwLock<Library>, place: &Place, names: &Names, file: Option<FileId>) {
    let l = library.read().unwrap_or_else(|e| e.into_inner());
    let file = {
        let mut known = l.file.lock().unwrap_or_else(|e| e.into_inner());
        if file.is_some() {
            *known = file;
        } else if known.is_none() && l.blob_is_file {
            *known = Some(FileId::of(&l._blob));
        }
        *known
    };
    let written = file.map(|file| write_atomically(&place.names, names.text(&file).as_bytes()));
    match written {
        Some(Ok(())) => {}
        Some(Err(e)) => {
            tracing::warn!(error = %e, "host pipeline names not written");
            l.names_changed.store(true, Ordering::Relaxed);
        }
        None => l.names_changed.store(true, Ordering::Relaxed),
    }
}

/// Writes the early-key manifest of the library file in `place`, matching
/// [`write_names`].
fn write_keys(
    library: &RwLock<Library>,
    place: &Place,
    keys: &HashSet<String>,
    file: Option<FileId>,
) {
    let l = library.read().unwrap_or_else(|e| e.into_inner());
    let file = {
        let mut known = l.file.lock().unwrap_or_else(|e| e.into_inner());
        if file.is_some() {
            *known = file;
        } else if known.is_none() && l.blob_is_file {
            *known = Some(FileId::of(&l._blob));
        }
        *known
    };
    let written = file.map(|file| write_atomically(&place.keys, keys_text(keys, &file).as_bytes()));
    match written {
        Some(Ok(())) => {}
        Some(Err(e)) => {
            tracing::warn!(error = %e, "host pipeline keys not written");
            l.keys_changed.store(true, Ordering::Relaxed);
        }
        None => l.keys_changed.store(true, Ordering::Relaxed),
    }
}

/// D3D12 needs one contiguous output buffer. A writable view of the
/// temporary file avoids a second library-sized allocation in the heap.
struct LibraryFile {
    file: std::fs::File,
    view: windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS,
    size: usize,
}

impl LibraryFile {
    fn serialize(
        path: &std::path::Path,
        library: &ID3D12PipelineLibrary1,
    ) -> std::io::Result<Self> {
        use std::os::windows::{
            fs::OpenOptionsExt,
            io::{AsRawHandle, FromRawHandle, OwnedHandle},
        };
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, PAGE_READWRITE,
        };

        // SAFETY: a live library under its exclusive lock.
        let size = unsafe { library.GetSerializedSize() };
        if size > isize::MAX as usize {
            return Err(std::io::Error::other(
                "pipeline library is too large to map",
            ));
        }
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .share_mode(0)
            .open(path)?;
        file.set_len(size as u64)?;
        // SAFETY: an unnamed mapping of this file, extended to exactly size.
        let section = unsafe {
            CreateFileMappingW(
                file.as_raw_handle(),
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                0,
                std::ptr::null(),
            )
        };
        if section.is_null() {
            return Err(std::io::Error::last_os_error());
        }
        // SAFETY: the newly created handle is ours; the view retains its
        // section independently once this handle closes.
        let section = unsafe { OwnedHandle::from_raw_handle(section) };
        // SAFETY: the section covers the entire file and permits writes.
        let view = unsafe { MapViewOfFile(section.as_raw_handle(), FILE_MAP_WRITE, 0, 0, size) };
        if view.Value.is_null() {
            return Err(std::io::Error::last_os_error());
        }
        let mapped = Self { file, view, size };
        // SAFETY: the exclusively opened file has one writable view of size bytes.
        unsafe {
            let bytes = std::slice::from_raw_parts_mut(view.Value.cast(), size);
            library.Serialize(bytes).map_err(std::io::Error::other)?;
        }
        Ok(mapped)
    }

    fn flush(&self) -> std::io::Result<()> {
        use windows_sys::Win32::System::Memory::FlushViewOfFile;
        // SAFETY: this view and its file stay open through the flush.
        if unsafe { FlushViewOfFile(self.view.Value, self.size) } == 0 {
            return Err(std::io::Error::last_os_error());
        }
        self.file.sync_all()
    }
}

impl Drop for LibraryFile {
    fn drop(&mut self) {
        // SAFETY: this object owns the sole view and unmaps it exactly once.
        unsafe { windows_sys::Win32::System::Memory::UnmapViewOfFile(self.view) };
    }
}

/// Writes `bytes` to a temporary file beside `path`, then renames it over
/// `path`: a reader never sees a part. One save at a time uses it
/// ([`save`]).
fn write_atomically(path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)
}

/// How long each title pipeline that took long to make took, in ms (next to
/// [`EARLY_FILE`]; a key may appear more than once and the maximum stays).
const SLOW_FILE: &str = "slow-pipelines.txt";
/// A pipeline that takes this long (in ms) to make is remembered as slow
/// ([`slow`]).
pub(crate) const SLOW_MS: u64 = 1000;

static SLOW: Mutex<Option<HashMap<String, u64>>> = Mutex::new(None);
static SLOW_CHANGED: AtomicBool = AtomicBool::new(false);

fn read_slow(dir: &Path) -> HashMap<String, u64> {
    let Ok(text) = std::fs::read_to_string(dir.join(SLOW_FILE)) else {
        return HashMap::new();
    };
    let mut map = HashMap::new();
    for line in text.lines() {
        if let Some((key, ms)) = line.split_once(' ')
            && !key.is_empty()
            && !key.chars().any(char::is_whitespace)
            && let Ok(ms) = ms.trim().parse::<u64>()
            && ms >= SLOW_MS
        {
            map.entry(key.to_owned())
                .and_modify(|prev: &mut u64| *prev = (*prev).max(ms))
                .or_insert(ms);
        }
    }
    map
}

fn write_slow(dir: &Path, slow: &HashMap<String, u64>) -> std::io::Result<()> {
    use std::fmt::Write;
    let mut entries: Vec<(&str, u64)> = slow.iter().map(|(k, &ms)| (k.as_str(), ms)).collect();
    entries.sort_unstable_by(|a, b| a.0.cmp(b.0));
    let mut text = String::new();
    for (key, ms) in entries {
        let _ = writeln!(text, "{key} {ms}");
    }
    write_atomically(&dir.join(SLOW_FILE), text.as_bytes())
}

fn slow_ones() -> std::sync::MutexGuard<'static, Option<HashMap<String, u64>>> {
    let mut slow = SLOW.lock().unwrap_or_else(|e| e.into_inner());
    if slow.is_none() {
        *slow = Some(DIR.get().map_or_else(HashMap::new, |d| read_slow(d)));
    }
    slow
}

/// Whether making the title pipeline with `key` took at least [`SLOW_MS`]
/// in this or an earlier run.
pub(crate) fn slow(key: &str) -> bool {
    slow_ones().as_ref().is_some_and(|s| s.contains_key(key))
}

/// Records that making the title pipeline with `key` (not a library hit)
/// took `ms` milliseconds.
pub(crate) fn note_time(key: &str, ms: u64) {
    if ms < SLOW_MS || key.is_empty() || key.chars().any(char::is_whitespace) {
        return;
    }
    let mut slow = slow_ones();
    let known = slow
        .get_or_insert_default()
        .entry(key.to_owned())
        .or_insert(0);
    if ms > *known {
        *known = ms;
        SLOW_CHANGED.store(true, Ordering::Release);
    }
}

/// Title pipeline keys drawn or dispatched with (this run and the last).
static USED: Mutex<Option<HashSet<String>>> = Mutex::new(None);
static USED_CHANGED: AtomicBool = AtomicBool::new(false);

fn used() -> std::sync::MutexGuard<'static, Option<HashSet<String>>> {
    let mut used = USED.lock().unwrap_or_else(|e| e.into_inner());
    if used.is_none() {
        let listed = DIR
            .get()
            .and_then(|d| std::fs::read_to_string(d.join(EARLY_FILE)).ok())
            .unwrap_or_default();
        let set: HashSet<String> = listed
            .lines()
            .filter(|l| !l.is_empty())
            .map(str::to_owned)
            .collect();
        tracing::info!(pipelines = set.len(), "title pipelines to start early");
        *used = Some(set);
    }
    used
}

/// Whether the title pipeline with `key` was drawn with in an earlier run.
pub(crate) fn wanted_early(key: &str) -> bool {
    used().as_ref().is_some_and(|s| s.contains(key))
}

/// The title pipeline with `key` was drawn or dispatched with.
pub(crate) fn note_used(key: &str) {
    let mut used = used();
    let set = used.get_or_insert_with(HashSet::new);
    if set.len() < EARLY_MAX && set.insert(key.to_owned()) {
        USED_CHANGED.store(true, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A generation's id, for the tests that save into a folder.
    const ID: &str = "0123456789abcdef";

    /// Generation [`ID`]'s files in `dir`.
    fn place(dir: &Path) -> Place {
        Place::of(dir, ID)
    }

    #[test]
    fn names_change_with_what_goes_into_the_pipeline() {
        let a = name("graphics", b"root", &[b"vs", b"ps"], "state");
        assert_eq!(a, name("graphics", b"root", &[b"vs", b"ps"], "state"));
        assert_ne!(a, name("graphics", b"root", &[b"vs", b"ps2"], "state"));
        assert_ne!(a, name("graphics", b"root", &[b"vs", b"ps"], "state2"));
        assert_ne!(a, name("graphics", b"root2", &[b"vs", b"ps"], "state"));
        assert_ne!(a, name("mesh", b"root", &[b"vs", b"ps"], "state"));
        // Parts do not run into each other.
        assert_ne!(a, name("graphics", b"root", &[b"vsp", b"s"], "state"));
    }

    #[test]
    fn a_rejected_blob_gives_an_empty_library() {
        let test = "pipeline::library::tests::a_rejected_blob_gives_an_empty_library";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let device = gpu.device.cast::<ID3D12Device1>().expect("ID3D12Device1");
        // The layer reports the blob it cannot read; the library falls back.
        let library = crate::testgpu::expect_layer_errors(|| {
            open(&device, b"not a pipeline library".repeat(16), None, None)
                .expect("an empty library")
        });
        // SAFETY: a plain query on a live library.
        assert!(unsafe { library.library.GetSerializedSize() } > 0);
        assert!(library._blob.is_empty());
    }

    /// Host pipelines of a few names, each asked for by several threads at
    /// once: every ask gets its pipeline (loads of one name kept apart,
    /// stores beside loads), most from the library, which still serialises
    /// after.
    #[test]
    fn pipelines_are_loaded_and_stored_side_by_side() {
        let test = "pipeline::library::tests::pipelines_are_loaded_and_stored_side_by_side";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        if library().is_none() {
            return;
        }
        let shaders: Vec<_> = [1, 2, 4, 8]
            .map(|n| {
                super::super::precook::tests::translated(
                    "compute",
                    xtp_shader::HwStage::Cs,
                    Some(&format!("[numthreads({n}, 1, 1)] void main() {{}}")),
                    windows::core::s!("cs_5_1"),
                    format!("library test: compute {n}").as_bytes(),
                )
            })
            .into();
        let loaded = LOADED.load(Ordering::Relaxed);
        std::thread::scope(|s| {
            for t in 0..8 {
                let shaders = &shaders;
                s.spawn(move || {
                    for k in 0..12 {
                        let shader = &shaders[(t + k) % shaders.len()];
                        let made = super::super::host::HostPipeline::new(
                            std::slice::from_ref(shader),
                            None,
                        );
                        assert!(made.is_some(), "a pipeline for {shader:?}");
                    }
                });
            }
        });
        // 96 asks of 4 pipelines: each compiled by at most the 8 threads.
        assert!(LOADED.load(Ordering::Relaxed) - loaded >= 96 - 4 * 8);
        let l = library()
            .unwrap()
            .write()
            .unwrap_or_else(|e| e.into_inner());
        // SAFETY: a plain query on a live library, alone.
        assert!(unsafe { l.library.GetSerializedSize() } > 0);
    }

    /// A compute kernel of `n` threads whose code holds its root signature.
    struct Kernel {
        code: Vec<u8>,
        root: ID3D12RootSignature,
    }

    fn kernel(device: &ID3D12Device, n: u32) -> Kernel {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        let source = format!("[RootSignature(\"\")] [numthreads({n}, 1, 1)] void main() {{}}");
        let mut blob = None;
        // SAFETY: the source lives through compilation and the blob owns
        // its bytes.
        let code = unsafe {
            D3DCompile(
                source.as_ptr().cast(),
                source.len(),
                windows::core::PCSTR::null(),
                None,
                None,
                windows::core::s!("main"),
                windows::core::s!("cs_5_1"),
                0,
                0,
                &mut blob,
                None,
            )
            .unwrap();
            let blob = blob.unwrap();
            std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize())
                .to_vec()
        };
        // SAFETY: the code holds the root signature.
        let root = unsafe { device.CreateRootSignature(0, &code) }.unwrap();
        Kernel { code, root }
    }

    /// `k`'s pipeline named `name` in `library`, and where it came from.
    fn ask(library: &RwLock<Library>, device: &ID3D12Device, k: &Kernel, name: &str) -> Source {
        let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            // SAFETY: a copy of a live root signature, not released by the
            // description.
            pRootSignature: unsafe { std::mem::transmute_copy(&k.root) },
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: k.code.as_ptr().cast(),
                BytecodeLength: k.code.len(),
            },
            ..Default::default()
        };
        found_in(
            library,
            name,
            // SAFETY: the description points at live data.
            |l, name| unsafe { l.LoadComputePipeline(name, &desc) },
            // SAFETY: as above.
            || unsafe { device.CreateComputePipelineState(&desc) },
        )
        .unwrap()
        .1
    }

    /// A save that fails leaves the library's pipelines to save: the next
    /// save writes them, and only then do they stop counting as added.
    #[test]
    fn a_failed_save_is_made_again() {
        let test = "pipeline::library::tests::a_failed_save_is_made_again";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let device = host.device.cast::<ID3D12Device1>().unwrap();
        let library = RwLock::new(open(&device, Vec::new(), None, None).unwrap());
        library.read().unwrap().added.store(2, Ordering::Relaxed);
        let dir = tempfile::tempdir().unwrap();
        // A file where the folder should be.
        let not_a_folder = dir.path().join("pipelines");
        std::fs::write(&not_a_folder, b"in the way").unwrap();
        assert!(matches!(
            write_library(&library, &place(&not_a_folder), true, |_| Ok(true)),
            Saved::Failed(_)
        ));
        assert_eq!(library.read().unwrap().added.load(Ordering::Relaxed), 2);
        // The early list not written: neither is the save.
        let good = dir.path().join("cache");
        assert!(matches!(
            write_library(&library, &place(&good), true, |_| Err(
                std::io::Error::other("full")
            )),
            Saved::Failed(_)
        ));
        assert_eq!(library.read().unwrap().added.load(Ordering::Relaxed), 2);
        let mut size = None;
        assert_eq!(
            write_library(&library, &place(&good), true, |bytes| {
                size = bytes;
                Ok(true)
            }),
            Saved::Written
        );
        assert_eq!(library.read().unwrap().added.load(Ordering::Relaxed), 0);
        let written = std::fs::metadata(place(&good).library).unwrap().len();
        assert_eq!(Some(written as usize), size);
        assert_eq!(
            write_library(&library, &place(&good), true, |_| Ok(false)),
            Saved::Nothing,
            "nothing new"
        );
    }

    /// Two saves at once: one asked to wait writes after the one under way,
    /// which counts what it wrote first, so that what was added stops
    /// counting once and the count does not wrap; one that does not wait
    /// gives way.
    #[test]
    fn saves_at_once_count_what_they_wrote_once() {
        let test = "pipeline::library::tests::saves_at_once_count_what_they_wrote_once";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let device = host.device.cast::<ID3D12Device1>().unwrap();
        let library = RwLock::new(open(&device, Vec::new(), None, None).unwrap());
        library.read().unwrap().added.store(2, Ordering::Relaxed);
        let dir = tempfile::tempdir().unwrap();
        let (inside, go) = (std::sync::Barrier::new(2), std::sync::Barrier::new(2));
        std::thread::scope(|s| {
            let first = s.spawn(|| {
                write_library(&library, &place(dir.path()), true, |_| {
                    inside.wait();
                    go.wait();
                    Ok(true)
                })
            });
            inside.wait();
            // The first is between its write and its count.
            assert_eq!(
                write_library(&library, &place(dir.path()), false, |_| Ok(true)),
                Saved::Nothing,
                "gives way"
            );
            let second =
                s.spawn(|| write_library(&library, &place(dir.path()), true, |_| Ok(false)));
            go.wait();
            assert_eq!(first.join().unwrap(), Saved::Written);
            assert_eq!(
                second.join().unwrap(),
                Saved::Nothing,
                "nothing new by then"
            );
        });
        assert_eq!(library.read().unwrap().added.load(Ordering::Relaxed), 0);
    }

    /// A new generation takes what is asked for from the one before, and
    /// keeps only that: what was not asked for is made again later.
    #[test]
    fn a_new_generation_keeps_what_is_asked_for() {
        let test = "pipeline::library::tests::a_new_generation_keeps_what_is_asked_for";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let device = host.device.cast::<ID3D12Device1>().unwrap();
        let (a, b) = (kernel(&host.device, 3), kernel(&host.device, 5));
        let dir = tempfile::tempdir().unwrap();
        let saved = |library: &RwLock<Library>| {
            assert_eq!(
                write_library(library, &place(dir.path()), true, |_| Ok(true)),
                Saved::Written
            );
            std::fs::read(place(dir.path()).library).unwrap()
        };
        let old = RwLock::new(open(&device, Vec::new(), None, None).unwrap());
        assert_eq!(ask(&old, &host.device, &a, "a"), Source::Made);
        assert_eq!(ask(&old, &host.device, &b, "b"), Source::Made);
        // Even a matching name must not return a pipeline whose
        // description has different bytecode.
        let mismatch = crate::testgpu::expect_layer_errors(|| ask(&old, &host.device, &b, "a"));
        assert_eq!(mismatch, Source::Made);
        assert_eq!(ask(&old, &host.device, &a, "a"), Source::Library);
        let new = RwLock::new(new_generation(&device, saved(&old), None).unwrap());
        assert_eq!(ask(&new, &host.device, &a, "a"), Source::Previous);
        assert_eq!(ask(&new, &host.device, &a, "a"), Source::Library);
        let next = RwLock::new(open(&device, saved(&new), None, None).unwrap());
        assert_eq!(ask(&next, &host.device, &a, "a"), Source::Library);
        assert_eq!(
            ask(&next, &host.device, &b, "b"),
            Source::Made,
            "not asked for"
        );
        let empty = RwLock::new(new_generation(&device, saved(&next), None).unwrap());
        let reopened = RwLock::new(open(&device, saved(&empty), None, None).unwrap());
        assert_eq!(
            ask(&reopened, &host.device, &a, "a"),
            Source::Made,
            "nothing survived"
        );
    }

    /// A names file is its library file's: read back with the names it was
    /// written with, and refused for another file (another size, or other
    /// bytes of the same size), another format or a line that is no name.
    #[test]
    fn names_are_read_back_for_their_own_file_only() {
        let blob = b"a serialised pipeline library".repeat(8);
        let a = "0123456789abcdef0123456789abcdef";
        let names = Names {
            held: [a.to_owned(), format!("{a}~1")].into(),
            broken: [a.to_owned()].into(),
            complete: true,
        };
        let file = FileId::of(&blob);
        let text = names.text(&file);
        assert_eq!(Names::read(&text, &blob), Some((names.clone(), file)));
        // A file of the first version marks none broken, and its marks are
        // no names.
        let first = text.replace(NAMES_FORMAT, NAMES_FORMAT_1);
        assert_eq!(Names::read(&first, &blob), None, "a mark in version 1");
        let first = first.replace(BROKEN, "");
        let unmarked = Names {
            broken: HashSet::new(),
            ..names.clone()
        };
        assert_eq!(Names::read(&first, &blob), Some((unmarked, file)));
        let partial = Names {
            complete: false,
            ..names.clone()
        };
        assert_eq!(
            Names::read(&partial.text(&file), &blob).map(|(n, _)| n),
            Some(partial)
        );
        let mut other = blob.clone();
        other[3] ^= 1;
        assert_eq!(Names::read(&text, &other), None, "other bytes");
        assert_eq!(Names::read(&text, &blob[1..]), None, "another size");
        assert_eq!(Names::read(&format!("{text}{a}~4\n"), &blob), None);
        assert_eq!(Names::read(&format!("{text}not a name\n"), &blob), None);
        assert_eq!(
            Names::read(&text.replace(NAMES_FORMAT, "xtp pipeline names 0"), &blob),
            None
        );
        assert_eq!(Names::read("", &blob), None);
        assert!(is_name(a) && is_name(&format!("{a}~3")));
        assert!(!is_name(&format!("{a}~0")) && !is_name(&a.to_uppercase()) && !is_name("a"));
        assert!(names.ready_any(a) && !names.ready_any(&a.replace('0', "f")));
        // Ready by a name held and not broken only.
        let only_broken = Names {
            held: [a.to_owned()].into(),
            ..names.clone()
        };
        assert!(!only_broken.ready_any(a), "its one name is broken");
        assert_eq!(
            Names::read(&format!("{text}{a}~2{BROKEN}{BROKEN}\n"), &blob),
            None
        );
    }

    /// An entry of a pipeline's name that does not load for its
    /// description (here: another kernel stored under that name) used to
    /// make the pipeline again each time, since StorePipeline refuses a
    /// name the library holds. The pipeline is stored under the next name
    /// instead, and found there from then on, after a save too; the names
    /// file says which names the saved library holds.
    #[test]
    fn a_name_that_does_not_load_gets_the_next_one() {
        let test = "pipeline::library::tests::a_name_that_does_not_load_gets_the_next_one";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let device = host.device.cast::<ID3D12Device1>().unwrap();
        let (a, b) = (kernel(&host.device, 3), kernel(&host.device, 5));
        let name = "00112233445566778899aabbccddeeff";
        let library = RwLock::new(open(&device, Vec::new(), None, None).unwrap());
        assert_eq!(ask(&library, &host.device, &a, name), Source::Made);
        let other =
            || crate::testgpu::expect_layer_errors(|| ask(&library, &host.device, &b, name));
        assert_eq!(other(), Source::Made, "the entry of its name is another's");
        assert_eq!(other(), Source::Library, "found under the next name");
        assert_eq!(ask(&library, &host.device, &a, name), Source::Library);
        let held = library.read().unwrap().names().clone();
        assert_eq!(held.held, [name.to_owned(), slot(name, 1)].into());
        assert!(held.complete, "started empty");

        let dir = tempfile::tempdir().unwrap();
        assert_eq!(
            write_library(&library, &place(dir.path()), true, |_| Ok(true)),
            Saved::Written
        );
        let blob = std::fs::read(place(dir.path()).library).unwrap();
        let text = std::fs::read_to_string(place(dir.path()).names).unwrap();
        assert_eq!(Names::read(&text, &blob), Some((held, FileId::of(&blob))));
        let reopened = RwLock::new(open(&device, blob, Some(&text), None).unwrap());
        let other =
            || crate::testgpu::expect_layer_errors(|| ask(&reopened, &host.device, &b, name));
        assert_eq!(other(), Source::Library, "repaired for the next process");
        assert_eq!(
            reopened.read().unwrap().added.load(Ordering::Relaxed),
            0,
            "nothing made"
        );
    }

    /// The precook does not load a pipeline the library's names hold
    /// ([`without_loading`]); one they do not hold is made and stored as
    /// always. A library read without its names learns those that load,
    /// and a save writes the names learned.
    #[test]
    fn the_precook_does_not_load_what_the_names_hold() {
        let test = "pipeline::library::tests::the_precook_does_not_load_what_the_names_hold";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let device = host.device.cast::<ID3D12Device1>().unwrap();
        let (a, b) = (kernel(&host.device, 3), kernel(&host.device, 5));
        let (name_a, name_b) = (
            "a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0",
            "b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0",
        );
        let dir = tempfile::tempdir().unwrap();
        let library = RwLock::new(open(&device, Vec::new(), None, None).unwrap());
        assert_eq!(ask(&library, &host.device, &a, name_a), Source::Made);
        assert_eq!(
            write_library(&library, &place(dir.path()), true, |_| Ok(true)),
            Saved::Written
        );
        let blob = std::fs::read(place(dir.path()).library).unwrap();
        let text = std::fs::read_to_string(place(dir.path()).names).unwrap();

        // Read with its names: held, so not loaded; another is made. (The
        // count is the process's: other tests' precooks, the replay's,
        // may add to it meanwhile. `Source::Known` says this ask's.)
        let known = KNOWN.load(Ordering::Relaxed);
        let named = RwLock::new(open(&device, blob.clone(), Some(&text), None).unwrap());
        assert_eq!(
            without_loading(|| ask(&named, &host.device, &a, name_a)),
            Source::Known
        );
        assert!(KNOWN.load(Ordering::Relaxed) > known, "counted");
        assert_eq!(
            without_loading(|| ask(&named, &host.device, &b, name_b)),
            Source::Made
        );
        assert_eq!(
            ask(&named, &host.device, &a, name_a),
            Source::Library,
            "a session loads it"
        );

        // Read without its names (an older file): loaded once, then known.
        let unnamed = RwLock::new(open(&device, blob, None, None).unwrap());
        assert_eq!(
            without_loading(|| ask(&unnamed, &host.device, &a, name_a)),
            Source::Library
        );
        assert_eq!(
            without_loading(|| ask(&unnamed, &host.device, &a, name_a)),
            Source::Known
        );
        // Nothing added: the save writes the names learned alone, for the
        // file as it is.
        std::fs::remove_file(place(dir.path()).names).unwrap();
        assert_eq!(
            write_library(&unnamed, &place(dir.path()), true, |_| Ok(false)),
            Saved::Nothing
        );
        let blob = std::fs::read(place(dir.path()).library).unwrap();
        let text = std::fs::read_to_string(place(dir.path()).names).unwrap();
        let (learned, _) = Names::read(&text, &blob).expect("the file's names");
        assert_eq!(
            (learned.held, learned.complete),
            ([name_a.to_owned()].into(), false)
        );
    }

    /// The Codex review of 12556d6: the precook answered a pipeline by any
    /// name the library held, before the driver, even one whose entry does
    /// not load for its description. A load that fails now marks the name
    /// broken, in the names file too, and the precook does not answer by a
    /// broken name: it makes the pipeline under the next one, by which it
    /// answers from then on.
    #[test]
    fn the_precook_does_not_answer_by_a_broken_name() {
        let test = "pipeline::library::tests::the_precook_does_not_answer_by_a_broken_name";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let device = host.device.cast::<ID3D12Device1>().unwrap();
        let (a, b) = (kernel(&host.device, 3), kernel(&host.device, 7));
        let name = "c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0";
        let dir = tempfile::tempdir().unwrap();
        let saved = |library: &RwLock<Library>| {
            let wrote = write_library(library, &place(dir.path()), true, |_| Ok(true));
            assert!(!matches!(wrote, Saved::Failed(_)), "{wrote:?}");
            let blob = std::fs::read(place(dir.path()).library).unwrap();
            let text = std::fs::read_to_string(place(dir.path()).names).unwrap();
            (blob, text)
        };
        // The library's entry of the name is another description's.
        let library = RwLock::new(open(&device, Vec::new(), None, None).unwrap());
        assert_eq!(ask(&library, &host.device, &a, name), Source::Made);
        let (blob, text) = saved(&library);

        // A session's load of `b` by that name fails, and its compile too
        // (nothing goes under the next name): the names file says broken.
        let session = RwLock::new(open(&device, blob.clone(), Some(&text), None).unwrap());
        let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            // SAFETY: a copy of a live root signature, not released by the
            // description.
            pRootSignature: unsafe { std::mem::transmute_copy(&b.root) },
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: b.code.as_ptr().cast(),
                BytecodeLength: b.code.len(),
            },
            ..Default::default()
        };
        let refused = crate::testgpu::expect_layer_errors(|| {
            found_in(
                &session,
                name,
                // SAFETY: the description points at live data.
                |l, name| unsafe { l.LoadComputePipeline(name, &desc) },
                || Err(windows::Win32::Foundation::E_FAIL.into()),
            )
        });
        assert!(refused.is_err());
        let (blob, text) = saved(&session);
        let (names, _) = Names::read(&text, &blob).expect("its names");
        assert_eq!(names.broken, [name.to_owned()].into(), "{text}");
        assert!(!names.ready_any(name));

        // The precook does not answer by it: it loads, then makes `b`,
        // under the next name.
        let precook = RwLock::new(open(&device, blob, Some(&text), None).unwrap());
        let made = crate::testgpu::expect_layer_errors(|| {
            without_loading(|| ask(&precook, &host.device, &b, name))
        });
        assert_eq!(made, Source::Made, "not answered by a broken name");
        assert_eq!(
            without_loading(|| ask(&precook, &host.device, &b, name)),
            Source::Known,
            "by the next name"
        );
        let (blob, text) = saved(&precook);
        let (names, _) = Names::read(&text, &blob).expect("its names");
        assert_eq!(
            (names.held, names.broken),
            (
                [name.to_owned(), slot(name, 1)].into(),
                [name.to_owned()].into()
            )
        );

        // A session after it loads `b` by the next name; `a` still loads
        // by its own, which mends it.
        let next = RwLock::new(open(&device, blob, Some(&text), None).unwrap());
        let found = crate::testgpu::expect_layer_errors(|| ask(&next, &host.device, &b, name));
        assert_eq!(found, Source::Library);
        assert_eq!(ask(&next, &host.device, &a, name), Source::Library);
        assert!(next.read().unwrap().names().broken.is_empty(), "mended");
    }

    const A: &str = "aaaaaaaaaaaaaaaa";
    const B: &str = "bbbbbbbbbbbbbbbb";
    const C: &str = "cccccccccccccccc";
    const D: &str = "dddddddddddddddd";
    const E: &str = "eeeeeeeeeeeeeeee";

    /// Saves `bytes` as `place`'s library file and `names` as its names,
    /// the way a save does: by a rename over the file (never through a
    /// link).
    fn save_as(place: &Place, bytes: &[u8], names: &str) {
        write_atomically(&place.library, bytes).unwrap();
        write_atomically(&place.names, names.as_bytes()).unwrap();
    }

    fn ids(list: &[Lineage]) -> Vec<&str> {
        list.iter().map(|g| g.id.as_str()).collect()
    }

    /// A generation's line in the list reads back as it was written; a line
    /// that is none is refused, and so is the whole list then.
    #[test]
    fn generation_lines_read_back() {
        let own = Lineage::own(ID);
        let from = Lineage {
            from: Some(B.into()),
            ..Lineage::own(ID)
        };
        let legacy = Lineage {
            from: Some(LEGACY.into()),
            ..Lineage::own(ID)
        };
        let compacted = Lineage {
            compacted: Some(3 << 30),
            ..Lineage::own(ID)
        };
        for g in [&own, &from, &legacy, &compacted] {
            assert_eq!(Lineage::parse(&g.line()).as_ref(), Some(g), "{}", g.line());
        }
        assert_eq!(own.line(), format!("{ID} own"));
        assert_eq!(from.line(), format!("{ID} from {B}"));
        assert_eq!(compacted.line(), format!("{ID} compacted {}", 3u64 << 30));
        for bad in [
            String::new(),
            ID.to_owned(),
            format!("{ID} own more"),
            format!("{} own", ID.to_uppercase()),
            format!("{ID}0 own"),
            format!("{ID} from elsewhere"),
            format!("{ID} from"),
            format!("{ID} compacted many"),
            format!("{ID} compacted"),
            format!("{ID} lost"),
        ] {
            assert_eq!(Lineage::parse(&bad), None, "{bad}");
        }
        let dir = tempfile::tempdir().unwrap();
        write_generations(dir.path(), &[compacted.clone(), Lineage::own(A)]).unwrap();
        assert_eq!(
            read_generations(dir.path()),
            [compacted.clone(), Lineage::own(A)]
        );
        let text = std::fs::read_to_string(dir.path().join(GENERATIONS_FILE)).unwrap();
        assert!(
            text.starts_with(&format!("{GENERATIONS_FORMAT}\n")),
            "{text}"
        );
        std::fs::write(
            dir.path().join(GENERATIONS_FILE),
            format!("{text}not a generation\n"),
        )
        .unwrap();
        assert_eq!(read_generations(dir.path()), [], "damaged");
        std::fs::write(
            dir.path().join(GENERATIONS_FILE),
            text.replace(GENERATIONS_FORMAT, "xtp pipeline generations 0"),
        )
        .unwrap();
        assert_eq!(read_generations(dir.path()), [], "another version");
    }

    /// A translation key with no generation starts from the last one opened:
    /// its file linked (or copied) and its names file copied, so that it
    /// finds every pipeline whose translations did not change by its name.
    /// Saving the new generation replaces its own file only.
    #[test]
    fn a_new_generation_starts_from_the_last_one_opened() {
        let dir = tempfile::tempdir().unwrap();
        let d = dir.path();
        let a = settle(d, A, GENERATIONS_KEPT, u64::MAX).unwrap();
        assert_eq!(a.lineage, Lineage::own(A), "nothing to start from");
        assert!(a.new);
        assert_eq!(a.place, Place::of(d, A));
        assert!(!a.place.library.exists());
        save_as(&a.place, b"A's pipelines", "A's names");

        let b = settle(d, B, GENERATIONS_KEPT, u64::MAX).unwrap();
        assert_eq!(b.lineage.from.as_deref(), Some(A));
        assert!(b.new);
        assert_eq!(std::fs::read(&b.place.library).unwrap(), b"A's pipelines");
        assert_eq!(
            std::fs::read_to_string(&b.place.names).unwrap(),
            "A's names"
        );
        assert_eq!(read_generations(d), [b.lineage.clone(), Lineage::own(A)]);

        save_as(&b.place, b"A's and B's pipelines", "B's names");
        assert_eq!(std::fs::read(&a.place.library).unwrap(), b"A's pipelines");
        assert_eq!(
            std::fs::read_to_string(&a.place.names).unwrap(),
            "A's names"
        );

        // A third starts from B, the last opened, not A.
        let c = settle(d, C, GENERATIONS_KEPT, u64::MAX).unwrap();
        assert_eq!(c.lineage.from.as_deref(), Some(B));
        assert_eq!(
            std::fs::read(&c.place.library).unwrap(),
            b"A's and B's pipelines"
        );
        assert_eq!(ids(&read_generations(d)), [C, B, A]);
        // Opened again, a generation keeps what it was.
        let again = settle(d, C, GENERATIONS_KEPT, u64::MAX).unwrap();
        assert_eq!(again.lineage, c.lineage);
        assert!(!again.new, "its file is the one it left");
        // Its file gone (let go by another process's list): seeded again,
        // and new, whatever its size (the precook's note does not hold it).
        std::fs::remove_file(&again.place.library).unwrap();
        let reseeded = settle(d, C, GENERATIONS_KEPT, u64::MAX).unwrap();
        assert_eq!(reseeded.lineage.from.as_deref(), Some(B));
        assert!(reseeded.new);
    }

    /// A, then B (whose translator differs: compacted to its own pipelines),
    /// then A again: A opens its own file as it left it, with its names, and
    /// makes nothing. Before generations, the one file held B's pipelines
    /// alone by then, and A's were all made again.
    #[test]
    fn a_translator_back_finds_its_own_generation() {
        let dir = tempfile::tempdir().unwrap();
        let d = dir.path();
        let a = settle(d, A, GENERATIONS_KEPT, u64::MAX).unwrap();
        save_as(&a.place, b"A's pipelines", "A's names");
        record_compaction(d, A, 13, 5).unwrap();
        let a_then = read_generations(d)[0].clone();
        assert_eq!(a_then.compacted, Some(13));

        let b = settle(d, B, GENERATIONS_KEPT, u64::MAX).unwrap();
        assert_eq!(b.lineage.from.as_deref(), Some(A));
        // B's precook compacts it: B's pipelines alone, below the bound.
        save_as(&b.place, b"B's", "B's names");
        record_compaction(d, B, 3, 5).unwrap();
        assert_eq!(read_generations(d), [Lineage::own(B), a_then.clone()]);

        let back = settle(d, A, GENERATIONS_KEPT, u64::MAX).unwrap();
        assert_eq!(back.lineage, a_then, "its own, as it was");
        assert!(!back.new);
        assert_eq!(
            std::fs::read(&back.place.library).unwrap(),
            b"A's pipelines"
        );
        assert_eq!(
            std::fs::read_to_string(&back.place.names).unwrap(),
            "A's names"
        );
        assert_eq!(back.evicted, Vec::<String>::new());
        assert_eq!(read_generations(d), [a_then, Lineage::own(B)]);
        assert_eq!(std::fs::read(&b.place.library).unwrap(), b"B's");
    }

    /// The one library file of before generations becomes the first
    /// generation made (moved, with its names); once there are generations,
    /// a new one starts from them, and the file of before is left alone.
    #[test]
    fn the_file_of_before_becomes_the_first_generation() {
        let dir = tempfile::tempdir().unwrap();
        let d = dir.path();
        let legacy = Place::legacy(d);
        save_as(&legacy, b"the one file", "its names");
        let a = settle(d, A, GENERATIONS_KEPT, u64::MAX).unwrap();
        assert_eq!(a.lineage.from.as_deref(), Some(LEGACY));
        assert_eq!(std::fs::read(&a.place.library).unwrap(), b"the one file");
        assert_eq!(
            std::fs::read_to_string(&a.place.names).unwrap(),
            "its names"
        );
        assert!(!legacy.library.exists() && !legacy.names.exists(), "moved");

        // An older build wrote the file of before again meanwhile.
        save_as(&legacy, b"an older build's", "its names");
        let b = settle(d, B, GENERATIONS_KEPT, u64::MAX).unwrap();
        assert_eq!(b.lineage.from.as_deref(), Some(A));
        assert_eq!(std::fs::read(&legacy.library).unwrap(), b"an older build's");
    }

    /// Past `kept` generations or `budget` bytes of files, the least
    /// recently opened go, files and names; the current one stays whatever
    /// its size. A listed generation whose file is gone leaves the list, and
    /// a file the list does not hold counts as the oldest.
    #[test]
    fn the_least_recently_opened_generations_go() {
        let dir = tempfile::tempdir().unwrap();
        let d = dir.path();
        let hundred = [7u8; 100];
        for id in [A, B, C, D] {
            let s = settle(d, id, 10, u64::MAX).unwrap();
            save_as(&s.place, &hundred, "names");
        }
        assert_eq!(ids(&read_generations(d)), [D, C, B, A]);

        // By count: E (from D) and two more.
        let e = settle(d, E, 3, u64::MAX).unwrap();
        assert_eq!(e.lineage.from.as_deref(), Some(D));
        assert_eq!(e.evicted, [B, A]);
        assert_eq!(ids(&read_generations(d)), [E, D, C]);
        for id in [A, B] {
            let gone = Place::of(d, id);
            assert!(!gone.library.exists() && !gone.names.exists(), "{id}");
        }
        assert!(Place::of(d, C).library.exists());

        // By bytes: 250 hold the current one and one more.
        let c = settle(d, C, 10, 250).unwrap();
        assert_eq!(c.evicted, [D]);
        assert_eq!(ids(&read_generations(d)), [C, E]);

        // The current one stays, past the budget alone.
        let e = settle(d, E, 10, 50).unwrap();
        assert_eq!(e.evicted, [C]);
        assert_eq!(ids(&read_generations(d)), [E]);
        assert!(Place::of(d, E).library.exists());

        // A file the list does not hold (written while the list was lost):
        // listed, as the oldest. A listed generation whose file is gone:
        // dropped.
        save_as(&Place::of(d, A), &hundred, "names");
        write_generations(d, &[Lineage::own(E), Lineage::own(B)]).unwrap();
        let e = settle(d, E, 10, u64::MAX).unwrap();
        assert_eq!(e.evicted, Vec::<String>::new());
        assert_eq!(ids(&read_generations(d)), [E, A]);
        // A damaged list: the files are listed again.
        std::fs::write(d.join(GENERATIONS_FILE), "damaged").unwrap();
        let a = settle(d, A, 10, u64::MAX).unwrap();
        assert_eq!(a.lineage, Lineage::own(A));
        assert_eq!(ids(&read_generations(d)), [A, E]);
    }

    /// The slow-pipelines file round-trips cleanly: entries are written
    /// sorted, read back identically, duplicate keys keep their longest
    /// time, and malformed lines are ignored.
    #[test]
    fn slow_pipelines_file_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        assert!(read_slow(dir.path()).is_empty(), "missing file");

        let mut slow = HashMap::new();
        slow.insert("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_owned(), 1_500);
        slow.insert("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_owned(), 4_200);
        write_slow(dir.path(), &slow).unwrap();
        assert_eq!(read_slow(dir.path()), slow);

        // Duplicate keys keep the longest measured duration; entries below
        // `SLOW_MS` and malformed lines are skipped without losing valid ones.
        let path = dir.path().join(SLOW_FILE);
        std::fs::write(
            &path,
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa 1200\r\n\
             not-a-valid-line\r\n\
              9999\r\n\
             aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa 3800\r\n\
             cccccccccccccccccccccccccccccccc nope\r\n\
             dddddddddddddddddddddddddddddddd 999\r\n\
             bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb 2000\r\n",
        )
        .unwrap();
        let read = read_slow(dir.path());
        assert_eq!(read.len(), 2);
        assert_eq!(read["aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"], 3_800);
        assert_eq!(read["bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"], 2_000);

        // `note_time` records only durations at or above `SLOW_MS` and
        // rejects keys containing whitespace.
        let fast = "11111111111111111111111111111111";
        let lagged = "22222222222222222222222222222222";
        note_time(fast, SLOW_MS - 1);
        assert!(!super::slow(fast));
        note_time("bad key", SLOW_MS);
        assert!(!super::slow("bad key"));
        note_time("", SLOW_MS);
        assert!(!super::slow(""));
        note_time(lagged, SLOW_MS);
        assert!(super::slow(lagged));
    }

    /// An early-key manifest reads back only for its own library file, and
    /// is not copied across translator generations when seeding.
    #[test]
    fn keys_manifest_matches_its_own_file_only() {
        let blob = b"a serialised pipeline library".repeat(8);
        let file = FileId::of(&blob);
        let a = "0123456789abcdef0123456789abcdef".to_owned();
        let b = "fedcba9876543210fedcba9876543210".to_owned();
        let keys: HashSet<String> = [a.clone(), b].into();
        let text = keys_text(&keys, &file);
        assert_eq!(
            read_keys(&text, &blob, Some(file)),
            Some((keys.clone(), file))
        );
        assert_eq!(read_keys(&text, &blob, None), Some((keys, file)));
        let mut other = blob.clone();
        other[0] ^= 1;
        assert_eq!(read_keys(&text, &other, None), None, "other bytes");
        assert_eq!(read_keys(&text, &blob[1..], None), None, "other size");
        assert_eq!(
            read_keys(&format!("{text}not-a-key\n"), &blob, Some(file)),
            None
        );

        // Seeding a new generation does not copy the old generation's keys
        // manifest, and eviction removes it.
        let dir = tempfile::tempdir().unwrap();
        let d = dir.path();
        let g1 = settle(d, A, 2, u64::MAX).unwrap();
        save_as(&g1.place, &blob, "names");
        write_atomically(&g1.place.keys, text.as_bytes()).unwrap();
        let g2 = settle(d, B, 2, u64::MAX).unwrap();
        assert!(!g2.place.keys.exists(), "not copied to new translator");
        save_as(&g2.place, &blob, "names");
        let _g3 = settle(d, C, 2, u64::MAX).unwrap();
        assert!(!g1.place.keys.exists(), "removed on eviction");
    }

    /// A pipeline that could not be stored (all alternate slots occupied by
    /// other descriptions) is not reported as cached, and a library's shared
    /// key set stays readable while the outer library lock is held for saving.
    #[test]
    fn unstored_pipeline_is_not_reported_cached_and_keys_stay_readable_during_save() {
        let test = "pipeline::library::tests::unstored_pipeline_is_not_reported_cached_and_keys_stay_readable_during_save";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let device = host.device.cast::<ID3D12Device1>().unwrap();
        let library = RwLock::new(open(&device, Vec::new(), None, None).unwrap());
        let name = "1234567890abcdef1234567890abcdef";
        let ask_cached = |k: &Kernel, n: &str| {
            let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                // SAFETY: a copy of a live root signature, not released by the
                // description.
                pRootSignature: unsafe { std::mem::transmute_copy(&k.root) },
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: k.code.as_ptr().cast(),
                    BytecodeLength: k.code.len(),
                },
                ..Default::default()
            };
            let (_, source, cached) = found_in(
                &library,
                n,
                // SAFETY: the description points at live data.
                |l, name| unsafe { l.LoadComputePipeline(name, &desc) },
                // SAFETY: as above.
                || unsafe { host.device.CreateComputePipelineState(&desc) },
            )
            .unwrap();
            (source, cached)
        };

        // Fill all `SLOTS` with distinct descriptions under the same name.
        for i in 0..SLOTS {
            let k = kernel(&host.device, (i as u32) + 1);
            let (source, cached) = crate::testgpu::expect_layer_errors(|| ask_cached(&k, name));
            assert_eq!(source, Source::Made);
            assert!(cached, "slot {i} was stored");
        }
        // One more incompatible description under the same name has no slot
        // left: it compiles, but must not be reported as cached in the library.
        let overflow = kernel(&host.device, (SLOTS as u32) + 1);
        let (source, cached) = crate::testgpu::expect_layer_errors(|| ask_cached(&overflow, name));
        assert_eq!(source, Source::Made);
        assert!(!cached, "no slot left: not stored in library");

        // The shared keys set stays readable even while `library.write()` is
        // held exclusively (as during `LibraryFile::serialize`).
        let shared_keys = {
            let l = library.read().unwrap();
            l.keys
                .write()
                .unwrap()
                .insert("abcdef0123456789abcdef0123456789".to_owned());
            Arc::clone(&l.keys)
        };
        let _saving = library.write().unwrap();
        assert!(
            shared_keys
                .read()
                .unwrap()
                .contains("abcdef0123456789abcdef0123456789")
        );
        drop(_saving);

        // Saving a library whose keys set is empty removes any stale `.keys`
        // file beside it.
        let dir = tempfile::tempdir().unwrap();
        let p = place(dir.path());
        std::fs::write(&p.keys, "stale").unwrap();
        library.read().unwrap().keys.write().unwrap().clear();
        assert_eq!(
            write_library(&library, &p, true, |_| Ok(true)),
            Saved::Written
        );
        assert!(
            !p.keys.exists(),
            "stale keys file removed on empty keys save"
        );
    }
}
