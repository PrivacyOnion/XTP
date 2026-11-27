//! Host pipelines are made by workers, not by the title thread that first
//! records a draw with them. Translating a shader and the host driver's
//! compile take from milliseconds to seconds each (structurisation of the
//! biggest compute programs ~0.4 s, some driver compiles 3.4 s), and BTP 6's
//! first world frame wants over 1,200 new pipelines: made
//! on its recording threads, that frame took more than 30 s and the
//! title's GPU watchdog ended the run (decision 0007, "Pipelines made off
//! the title's threads").
//!
//! A first use queues the pipeline ahead of those started early and waits
//! for it within a budget per title frame ([`Budget`]): a few
//! milliseconds for graphics pipelines, more for compute ones, whose
//! results later passes read (docs/analysis/gpu.md, G12). The cheap ones
//! (most, and every one the pipeline library holds) are there at once;
//! past the budget, a draw or dispatch whose pipeline is not ready is
//! dropped, as an untranslated one is, until a worker has made it: a few
//! frames, instead of up to half a second of waiting per frame (the budget
//! before).
//!
//! Heavy pipelines ([`heavy`]: a large console program, whose translation
//! or driver compile takes seconds, up to 38 s) have a pool of workers of
//! their own, so a cheap pipeline never waits behind them: with one pool,
//! the Series S world dropped 566 to 1,122 draws a frame "being made" while
//! slow programs held most of the workers (docs/analysis/drops.md, §5.3).
//! The idea is OpenScarlett's (a lane for heavy pipelines), written again
//! here. Each pool makes its pipelines first use first, then those started
//! early, and no title thread waits for a heavy one.
//!
//! Early pipelines remembered from an earlier run as taking at least 1 s to
//! make (`slow-pipelines.txt`) are limited to at most a quarter of a pool's
//! workers at once when not known to be in the library, so cheap early
//! pipelines keep flowing. Passes over the whole screen
//! ([`GraphicsState::screen_pass`]) get two workers of their own and a
//! separate per-frame wait budget (`XTP_SCREEN_WAIT_MS`), so one-shot screen
//! passes are not dropped behind thousands of others. Per-frame wait budgets
//! are reserved up front and refunded on wake so concurrent title threads
//! share one budget instead of multiplying it.
//!
//! shortcut: a pass the title runs only once and whose pipeline takes
//! longer than the budget is lost; `XTP_PIPELINE_WAIT_MS` or
//! `XTP_SCREEN_WAIT_MS` raises it.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use super::Shader;
use super::host::{GraphicsState, HostPipeline};

/// What a host pipeline is made from, and the host pipeline once made.
pub(crate) struct Pending {
    shaders: Vec<Arc<Shader>>,
    pub(crate) graphics: Option<Box<GraphicsState>>,
    /// [`IDLE`], [`EARLY`], [`URGENT`] or [`MAKING`]: the queue it was
    /// handed to last, or taken out of it. Changed under the queues' lock.
    queued: AtomicU8,
    /// [`CHEAP`] or [`HEAVY`]: the pool it was handed to last, which makes
    /// it. Changed under the queues' lock.
    pool: AtomicU8,
    /// Taken by a worker from [`Queues::slow`] (counts against `slow_now`
    /// until finished).
    was_slow: AtomicBool,
    made: OnceLock<Option<HostPipeline>>,
    /// Set by the call that made it, once what that call counts is in
    /// place ([`Pending::make`]).
    finished: AtomicBool,
    /// What identifies it across runs (shader containers and state:
    /// [`key`]), and whether a draw has used it yet.
    pub(crate) key: String,
    used: AtomicBool,
    /// A pass over the whole screen ([`GraphicsState::screen_pass`], not a
    /// mesh pipeline), which gets dedicated workers and its own wait budget.
    screen: bool,
    /// The graphics state as the title's stream gave it, kept for the
    /// precook once a worker has made the pipeline ([`super::record`]).
    pub(crate) stream: Option<super::record::Stream>,
    /// Why the host refused it, once it has (diagnostics).
    #[cfg(feature = "diagnostics")]
    refusal: OnceLock<String>,
}

// SAFETY: D3D12 objects are free-threaded; the graphics state (whose input
// elements point into its own names) is only read after creation.
unsafe impl Send for Pending {}
// SAFETY: as above.
unsafe impl Sync for Pending {}

/// Not queued yet; queued as started early; queued for a first use; taken
/// by the worker that makes it (VIctory #546: a pipeline is never handed to
/// a second worker, which would wait on the first's compile).
const IDLE: u8 = 0;
const EARLY: u8 = 1;
const URGENT: u8 = 2;
const MAKING: u8 = 3;

/// The pools: cheap pipelines, heavy ones ([`heavy`]).
const CHEAP: usize = 0;
const HEAVY: usize = 1;
const POOL_NAMES: [&str; 2] = ["cheap", "heavy"];

/// What identifies a title pipeline of `shaders` and `graphics` across
/// runs: its early key (its description is kept under a key of its own,
/// `super::record`'s).
pub(crate) fn key(shaders: &[Arc<Shader>], graphics: Option<&GraphicsState>) -> String {
    let mut key = blake3::Hasher::new();
    for s in shaders {
        key.update(s.stage.as_bytes());
        key.update(s.hash.as_bytes());
    }
    key.update(
        graphics
            .map(|g| g.describe())
            .unwrap_or_default()
            .as_bytes(),
    );
    key.finalize().to_hex()[..32].to_owned()
}

impl Pending {
    /// A title pipeline of `shaders`, with the graphics state `graphics`
    /// read from the title's `stream` (none for a compute pipeline or a
    /// plain description).
    pub(crate) fn new(
        shaders: Vec<Arc<Shader>>,
        graphics: Option<Box<GraphicsState>>,
        stream: Option<super::record::Stream>,
    ) -> Pending {
        let screen = graphics.as_deref().is_some_and(GraphicsState::screen_pass)
            && shaders.iter().all(|s| s.stage != "mesh");
        Pending {
            key: key(&shaders, graphics.as_deref()),
            shaders,
            graphics,
            queued: AtomicU8::new(IDLE),
            pool: AtomicU8::new(CHEAP as u8),
            was_slow: AtomicBool::new(false),
            made: OnceLock::new(),
            finished: AtomicBool::new(false),
            used: AtomicBool::new(false),
            screen,
            stream,
            #[cfg(feature = "diagnostics")]
            refusal: OnceLock::new(),
        }
    }

    /// Makes the host pipeline (once); true for the call that made it,
    /// which runs `counted` before signalling it finished: whoever sees
    /// `finished` sees what `counted` counted.
    pub(super) fn make(&self, counted: impl FnOnce()) -> bool {
        let mut first = false;
        self.made.get_or_init(|| {
            first = true;
            #[cfg(feature = "diagnostics")]
            let _ = super::host::take_refusal();
            let _ = super::library::take_last();
            let started = Instant::now();
            let made = HostPipeline::new(&self.shaders, self.graphics.as_deref());
            let ms = started.elapsed().as_millis() as u64;
            let last = super::library::take_last();
            if !last.hit {
                super::library::note_time(&self.key, ms);
            }
            if made.is_some() && last.cached {
                super::library::note_cached(&self.key);
            }
            if ms >= SLOW_MS {
                tracing::info!(
                    ms,
                    shaders = ?self.shaders,
                    program_bytes = self.shaders.iter().map(|s| s.container.len()).max(),
                    pool = self.pool_name(),
                    made = made.is_some(),
                    "host pipeline slow to make (its translations and the driver's compile)"
                );
            }
            #[cfg(feature = "diagnostics")]
            if made.is_none()
                && let Some(why) = super::host::take_refusal()
            {
                let _ = self.refusal.set(why);
            }
            made
        });
        if first {
            counted();
            self.finished.store(true, Ordering::Release);
            let (lock, done) = &DONE;
            let held = lock.lock().unwrap_or_else(|e| e.into_inner());
            done.notify_all();
            drop(held);
            // Once the title can draw with it: kept for the precook, made
            // or not (a translator that refuses its shaders today may take
            // them tomorrow). Compute pipelines are the pack's.
            if let (Some(stream), false) = (&self.stream, self.compute()) {
                super::record::made(&self.shaders, stream);
            }
        }
        first
    }

    fn compute(&self) -> bool {
        matches!(&*self.shaders, [cs] if cs.stage == "compute")
    }

    /// The pool it was handed to last.
    fn pool(&self) -> usize {
        usize::from(self.pool.load(Ordering::Relaxed))
    }

    fn pool_name(&self) -> &'static str {
        POOL_NAMES[self.pool()]
    }
}

/// A console program of this many bytes or more (its shader container) is
/// heavy to make. In three Series S sessions of 2026-11-24 (runs
/// `20261124-180617`, `-210555` and `-212205`, cold shader caches), the
/// 171 to 204 containers of 48 KiB or more took 93 to 97 % of all the
/// translation time: 635 to 794 s, up to 38 s each (most of it on programs
/// the translator refuses in the end), and the largest driver compiles
/// (13 s). The 1,116 to 2,016 smaller ones took 24 to 47 s together, the
/// slowest 0.7 to 3.8 s; from 32 to 48 KiB the median was under 0.1 s.
const HEAVY_BYTES: usize = 48 << 10;

/// Whether making `p` may hold a worker for seconds: one of its shaders is
/// a large console program ([`HEAVY_BYTES`]) not known to be refused (a
/// refused translation fails its pipelines at once). It is known before
/// anything is made; the translated size is not: it comes with the
/// translation, the slow part, and a refused program has none.
fn heavy(p: &Pending) -> bool {
    p.shaders
        .iter()
        .any(|s| s.container.len() >= HEAVY_BYTES && !matches!(s.translated.get(), Some(None)))
}

/// Why a pipeline has no host pipeline, and which shaders that concerns
/// (diagnostics): "being made" and the pool making it ("being made
/// (heavy)"); "does not translate", with its shaders that do not;
/// "refused" and the host's reason, with all its shaders. `None` if it has
/// one.
#[cfg(feature = "diagnostics")]
pub(crate) fn missing(p: &Pending) -> Option<(String, Option<String>)> {
    match p.made.get() {
        Some(Some(_)) => None,
        None => Some((format!("being made ({})", p.pool_name()), None)),
        Some(None) => {
            let failed: Vec<String> = p
                .shaders
                .iter()
                .filter(|s| matches!(s.translated.get(), Some(None)))
                .map(|s| format!("{} {}", s.stage, s.hash))
                .collect();
            Some(if failed.is_empty() {
                let why = p
                    .refusal
                    .get()
                    .map_or("refused".into(), |r| format!("refused, {r}"));
                (why, Some(format!("{:?}", p.shaders)))
            } else {
                ("does not translate".into(), Some(failed.join(" + ")))
            })
        }
    }
}

/// Signalled whenever a pipeline has been made.
static DONE: (Mutex<()>, Condvar) = (Mutex::new(()), Condvar::new());

/// A pipeline that takes a worker this long or more is logged.
const SLOW_MS: u64 = 100;

/// Waiting allowed per title frame, over all threads: for graphics
/// pipelines, then for compute ones.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Budget([Duration; 2]);

impl Budget {
    const DEFAULT: Budget = Budget([Duration::from_millis(2), Duration::from_millis(8)]);

    /// `XTP_PIPELINE_WAIT_MS`: `<graphics>,<compute>` milliseconds, or one
    /// number for both.
    fn parse(text: &str) -> Option<Budget> {
        let ms: Vec<u64> = text
            .split(',')
            .map(|n| n.trim().parse().ok())
            .collect::<Option<_>>()?;
        let [graphics, compute] = match ms[..] {
            [both] => [both, both],
            [graphics, compute] => [graphics, compute],
            _ => return None,
        };
        Some(Budget([graphics, compute].map(Duration::from_millis)))
    }

    /// The run's, read once.
    fn get() -> Budget {
        static BUDGET: OnceLock<Budget> = OnceLock::new();
        *BUDGET.get_or_init(|| {
            let Ok(text) = std::env::var("XTP_PIPELINE_WAIT_MS") else {
                return Budget::DEFAULT;
            };
            match Budget::parse(&text) {
                Some(budget) => {
                    tracing::info!(?budget, "XTP_PIPELINE_WAIT_MS: pipeline waits per frame");
                    budget
                }
                None => {
                    tracing::warn!(
                        text,
                        "XTP_PIPELINE_WAIT_MS expects <graphics>,<compute> milliseconds"
                    );
                    Budget::DEFAULT
                }
            }
        })
    }
}

/// Default per-frame wait budget for passes over the whole screen (in ms):
/// long enough for a cold fullscreen pass on its dedicated worker without
/// risking multi-second stalls.
const DEFAULT_SCREEN_WAIT_MS: u64 = 500;

/// Per-frame wait budget for passes over the screen (`XTP_SCREEN_WAIT_MS`,
/// default 500 ms; `0` disables the separate screen-pass workers and wait
/// budget).
fn screen_budget() -> Duration {
    static BUDGET: OnceLock<Duration> = OnceLock::new();
    *BUDGET.get_or_init(|| {
        let Ok(text) = std::env::var("XTP_SCREEN_WAIT_MS") else {
            return Duration::from_millis(DEFAULT_SCREEN_WAIT_MS);
        };
        match text.trim().parse::<u64>() {
            Ok(ms) => {
                tracing::info!(
                    ms,
                    "XTP_SCREEN_WAIT_MS: screen-pass pipeline wait per frame"
                );
                Duration::from_millis(ms)
            }
            Err(_) => {
                tracing::warn!(text, "XTP_SCREEN_WAIT_MS expects a number of milliseconds");
                Duration::from_millis(DEFAULT_SCREEN_WAIT_MS)
            }
        }
    })
}

/// Whether passes over the screen get dedicated workers and their own wait
/// budget.
fn screens_first() -> bool {
    !screen_budget().is_zero()
}

/// Whether the frame's wait budget is shared across all threads by reserving
/// what is left up front and refunding what was not used. Counted only after
/// each wait ends, N concurrent recording threads each see an empty spent
/// counter and wait N times the budget in one frame.
/// On by default; `XTP_PIPELINE_WAIT_SHARED=0` counts after waiting as before.
fn shared_budget() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("XTP_PIPELINE_WAIT_SHARED").is_none_or(|v| v != "0"))
}

/// Whether early pipelines remembered as taking >= 1 s (`slow-pipelines.txt`)
/// are throttled to a quarter of the pool's workers.
/// On by default; `XTP_PIPELINE_SLOW=0` schedules all early pipelines in
/// order without throttling.
fn slow_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("XTP_PIPELINE_SLOW").is_none_or(|v| v != "0"))
}

/// How many workers of a pool of `workers` may make early slow pipelines at
/// once: a quarter of them (at least 1), or unlimited when
/// `XTP_PIPELINE_SLOW=0`.
fn slow_limit_for(workers: usize) -> usize {
    if !slow_enabled() {
        usize::MAX
    } else {
        (workers / 4).max(1)
    }
}

const KIND_GRAPHICS: usize = 0;
const KIND_COMPUTE: usize = 1;
const KIND_SCREEN: usize = 2;

/// The waiting done (or reserved by threads waiting now) in one title frame,
/// per kind (graphics, compute, screen pass).
struct Frame {
    spent: [Mutex<(Option<u64>, Duration)>; 3],
}

impl Frame {
    const fn new() -> Frame {
        Frame {
            spent: [
                Mutex::new((None, Duration::ZERO)),
                Mutex::new((None, Duration::ZERO)),
                Mutex::new((None, Duration::ZERO)),
            ],
        }
    }

    /// What is left of `budget` for `kind` in title frame `frame`. When
    /// `shared` is true, reserves that whole remainder up front so other
    /// threads in the same frame do not wait on top of it; the caller must
    /// pass the returned duration back to [`Self::finish`]. A thread arriving
    /// late from an older frame gets zero without resetting the current frame.
    fn reserve(&self, frame: u64, kind: usize, budget: Duration, shared: bool) -> Duration {
        let mut spent = self.spent[kind].lock().unwrap_or_else(|e| e.into_inner());
        match spent.0 {
            Some(cur) if frame < cur => return Duration::ZERO,
            Some(cur) if cur == frame => {}
            _ => *spent = (Some(frame), Duration::ZERO),
        }
        let left = budget.saturating_sub(spent.1);
        if shared {
            spent.1 = spent.1.saturating_add(left);
        }
        left
    }

    /// Accounts for `waited` out of a previously reserved `left` in `frame`
    /// for `kind`, refunding the unused part when `shared` is true.
    fn finish(&self, frame: u64, kind: usize, left: Duration, waited: Duration, shared: bool) {
        let clamped = waited.min(left);
        {
            let mut spent = self.spent[kind].lock().unwrap_or_else(|e| e.into_inner());
            if spent.0 == Some(frame) {
                if shared {
                    spent.1 = spent.1.saturating_sub(left - clamped);
                } else {
                    spent.1 = spent.1.saturating_add(clamped);
                }
            }
        }
        let stat_kind = usize::from(kind == KIND_COMPUTE);
        WAITED_US[stat_kind].fetch_add(
            u64::try_from(waited.as_micros()).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
    }

    #[cfg(test)]
    fn left(&self, frame: u64, kind: usize, budget: Duration) -> Duration {
        let spent = self.spent[kind].lock().unwrap_or_else(|e| e.into_inner());
        match spent.0 {
            Some(cur) if frame < cur => Duration::ZERO,
            Some(cur) if cur == frame => budget.saturating_sub(spent.1),
            _ => budget,
        }
    }
}

/// Waiting since the last [`take_wait_us`], in microseconds (graphics,
/// compute).
static WAITED_US: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

/// Microseconds title threads waited for pipelines (graphics, compute)
/// since the last call (for the frame line).
pub(crate) fn take_wait_us() -> [u64; 2] {
    [0, 1].map(|k| WAITED_US[k].swap(0, Ordering::Relaxed))
}

/// Pipelines queued and not made yet, in both pools.
static WAITING: AtomicU64 = AtomicU64::new(0);

/// Pipelines queued and not made yet (for the frame line).
pub(crate) fn waiting() -> u64 {
    WAITING.load(Ordering::Relaxed)
}

/// The host pipeline if it is made (or, for a cheap or screen-pass one, is
/// made within the frame's budget); queues it ahead of the early ones on
/// first use.
pub(crate) fn ready(p: &Arc<Pending>) -> Option<&HostPipeline> {
    static FRAME: Frame = Frame::new();
    let frame = crate::frame::current();
    report(frame);
    POOLS.ready(p, &FRAME, frame, Budget::get(), screen_budget())
}

/// At the title's creation of a pipeline: starts it at once if an earlier
/// run drew with it (during loading, so the world's first frames find it).
pub(crate) fn early(p: &Arc<Pending>) {
    if !p.shaders.is_empty() && super::library::wanted_early(&p.key) {
        super::library::EARLY.fetch_add(1, Ordering::Relaxed);
        POOLS.start(p, EARLY);
    }
}

pub(super) fn start_builder(p: &Arc<Pending>) {
    POOLS.start(p, EARLY);
}

/// Wakes regular pool workers (e.g. after the pipeline library finishes
/// opening so workers sleeping behind `slow_limit` can pick up library hits).
pub(super) fn wake_workers() {
    POOLS.wake_workers();
}

/// Un seul verdict pour toute la session. Un refus suffit, même si d'autres
/// compilations sont en cours. Le succès exige le dernier pipeline hôte.
fn builder_verdict(states: impl Iterator<Item = Option<bool>>) -> Option<bool> {
    let mut ready = Some(true);
    for state in states {
        match state {
            Some(false) => return Some(false),
            None => ready = None,
            Some(true) => {}
        }
    }
    ready
}

pub(super) fn admit_builder<'a>(
    members: impl Iterator<Item = &'a Arc<Pending>> + Clone,
) -> Option<bool> {
    let state = |p: &Arc<Pending>| p.made.get().map(Option::is_some);
    let admitted = builder_verdict(members.clone().map(state))?;
    let total = members.clone().count();
    if admitted {
        tracing::info!(members = total, "BVH builder admitted");
    } else if let Some(p) = members.clone().find(|p| state(p) == Some(false)) {
        tracing::warn!(members = total, refused = %p.shaders[0].hash, "BVH builder held");
    }
    Some(admitted)
}

#[test]
fn builder_requires_every_host_pipeline() {
    let verdict = |states: &[Option<bool>]| builder_verdict(states.iter().copied());
    assert_eq!(verdict(&[Some(true), None]), None);
    assert_eq!(verdict(&[Some(true), None, Some(false)]), Some(false));
    assert_eq!(verdict(&[Some(false), Some(true)]), Some(false));
    assert_eq!(verdict(&[Some(true), Some(true)]), Some(true));
}

/// Waits for `p` while title frame `frame` has budget left for its kind.
fn wait_with(
    p: &Pending,
    frames: &Frame,
    frame: u64,
    budget: Budget,
    screen_budget: Duration,
    shared: bool,
) {
    let screen = p.screen && !screen_budget.is_zero();
    let (kind, limit) = if screen {
        (KIND_SCREEN, screen_budget)
    } else if p.compute() {
        (KIND_COMPUTE, budget.0[KIND_COMPUTE])
    } else {
        (KIND_GRAPHICS, budget.0[KIND_GRAPHICS])
    };
    if limit.is_zero() {
        return;
    }
    let left = frames.reserve(frame, kind, limit, shared);
    if left.is_zero() {
        return;
    }
    let started = Instant::now();
    let (lock, done) = &DONE;
    let mut held = lock.lock().unwrap_or_else(|e| e.into_inner());
    while p.made.get().is_none() {
        let rest = left.saturating_sub(started.elapsed());
        if rest.is_zero() {
            break;
        }
        held = done
            .wait_timeout(held, rest)
            .unwrap_or_else(|e| e.into_inner())
            .0;
    }
    drop(held);
    let waited = started.elapsed();
    if screen && waited > Duration::from_millis(500) {
        tracing::info!(
            ms = waited.as_millis() as u64,
            made = p.made.get().is_some(),
            shaders = ?p.shaders,
            "a pass over the screen was waited for (XTP_SCREEN_WAIT_MS=0 disables the separate budget)"
        );
    }
    frames.finish(frame, kind, left, waited, shared);
}

#[cfg(test)]
fn wait(p: &Pending, frames: &Frame, frame: u64, budget: Budget) {
    wait_with(p, frames, frame, budget, Duration::ZERO, true);
}

/// One pool's pipelines: screen passes first, then first uses, then early
/// ones, then early ones remembered as slow (capped by `slow_limit`). A
/// pipeline started early and then used is in two queues (of this pool or
/// of the other); the early entry is passed over once the pipeline is no
/// longer [`EARLY`].
struct Queues {
    /// Requested passes over the screen: before all others, and the only
    /// ones [`SCREEN_WORKERS`] take.
    screen: VecDeque<Arc<Pending>>,
    urgent: VecDeque<Arc<Pending>>,
    early: VecDeque<Arc<Pending>>,
    /// Early pipelines known to take >= 1 s to make, and how many of them
    /// are currently being made by this pool's workers.
    slow: VecDeque<Arc<Pending>>,
    slow_now: usize,
    /// Being made by the pool's workers.
    making: Vec<Arc<Pending>>,
    /// Whether a pipeline was passed over since a worker last finished,
    /// for one of `making` ([`Queues::pop_with`]).
    passed: bool,
}

impl Queues {
    const fn new() -> Queues {
        Queues {
            screen: VecDeque::new(),
            urgent: VecDeque::new(),
            early: VecDeque::new(),
            slow: VecDeque::new(),
            slow_now: 0,
            making: Vec::new(),
            passed: false,
        }
    }

    /// Queues `p` as `how` ([`EARLY`] or [`URGENT`]) unless it is queued
    /// that way already, or taken: whether it was not queued before, or
    /// `None` if nothing changed.
    #[cfg(test)]
    fn push(&mut self, p: &Arc<Pending>, how: u8) -> Option<bool> {
        self.push_as(p, how, false, false)
    }

    fn push_as(
        &mut self,
        p: &Arc<Pending>,
        how: u8,
        slow: bool,
        screens_first: bool,
    ) -> Option<bool> {
        let was = p.queued.load(Ordering::Relaxed);
        if was >= how {
            return None;
        }
        p.queued.store(how, Ordering::Relaxed);
        if how == URGENT && p.screen && screens_first {
            self.screen.push_back(p.clone());
        } else if how == URGENT {
            self.urgent.push_back(p.clone());
        } else if slow {
            self.slow.push_back(p.clone());
        } else {
            self.early.push_back(p.clone());
        }
        Some(was == IDLE)
    }

    /// The next pipeline to make, taken ([`MAKING`]) and listed as being
    /// made: screen passes first, then (for non-screen workers) first uses,
    /// early ones, and early slow ones up to `slow_limit` at once. Library
    /// hits (`is_cached`) are never throttled by `slow_limit`. Pipelines
    /// sharing an untranslated heavy shader with one in `making` are passed
    /// over ([`waits_for`]) and keep their place.
    fn pop_with(
        &mut self,
        screens_only: bool,
        slow_limit: usize,
        is_cached: impl Fn(&Pending) -> bool,
    ) -> Option<Arc<Pending>> {
        let Queues {
            screen,
            urgent,
            early,
            slow,
            slow_now,
            making,
            passed,
        } = self;
        let mut p = take(screen, making, None, passed);
        if p.is_none() && !screens_only {
            p = take(urgent, making, None, passed)
                .or_else(|| take(early, making, Some(EARLY), passed));
            if p.is_none() {
                if *slow_now < slow_limit {
                    p = take(slow, making, Some(EARLY), passed);
                    if let Some(p) = &p
                        && !is_cached(p)
                    {
                        p.was_slow.store(true, Ordering::Relaxed);
                        *slow_now += 1;
                    }
                } else {
                    // Library hits queued before the library finished
                    // opening are never throttled.
                    p = take_matching(slow, making, Some(EARLY), passed, &is_cached);
                }
            }
        }
        let p = p?;
        p.queued.store(MAKING, Ordering::Relaxed);
        making.push(p.clone());
        Some(p)
    }

    #[cfg(test)]
    fn pop(&mut self) -> Option<Arc<Pending>> {
        self.pop_with(false, usize::MAX, |_| false)
    }

    /// `p` is made: no longer listed as being made. Returns true if a
    /// waiting worker should be woken (something was passed over for one
    /// being made, or a throttled slow pipeline slot just freed up).
    fn made(&mut self, p: &Arc<Pending>) -> bool {
        self.making.retain(|m| !Arc::ptr_eq(m, p));
        let freed_slow = if p.was_slow.swap(false, Ordering::Relaxed) {
            self.slow_now = self.slow_now.saturating_sub(1);
            true
        } else {
            false
        };
        std::mem::take(&mut self.passed) || freed_slow
    }
}

/// The first entry of `queue` that does not wait for one of `making`,
/// removed; an entry no longer queued as `live` (an early entry since
/// used, or taken) is dropped on the way. `passed` is set if one was
/// passed over.
fn take(
    queue: &mut VecDeque<Arc<Pending>>,
    making: &[Arc<Pending>],
    live: Option<u8>,
    passed: &mut bool,
) -> Option<Arc<Pending>> {
    take_matching(queue, making, live, passed, |_| true)
}

/// Like [`take`], taking only an entry for which `matches` returns true.
fn take_matching(
    queue: &mut VecDeque<Arc<Pending>>,
    making: &[Arc<Pending>],
    live: Option<u8>,
    passed: &mut bool,
    matches: impl Fn(&Pending) -> bool,
) -> Option<Arc<Pending>> {
    let mut i = 0;
    while i < queue.len() {
        let p = &queue[i];
        if live.is_some_and(|how| p.queued.load(Ordering::Relaxed) != how) {
            queue.remove(i);
        } else if !matches(p) {
            i += 1;
        } else if waits_for(p, making) {
            *passed = true;
            i += 1;
        } else {
            return queue.remove(i);
        }
    }
    None
}

/// Whether making `p` now would wait for a translation another worker is
/// doing: one of its heavy programs ([`HEAVY_BYTES`]), not translated yet,
/// is a shader of a pipeline in `making`. A small program's translation
/// takes milliseconds: waiting for it is cheaper than passing over.
fn waits_for(p: &Pending, making: &[Arc<Pending>]) -> bool {
    p.shaders.iter().any(|s| {
        s.container.len() >= HEAVY_BYTES
            && s.translated.get().is_none()
            && making
                .iter()
                .any(|m| m.shaders.iter().any(|t| Arc::ptr_eq(s, t)))
    })
}

/// What a pool did since the pools' last line ([`report`]).
struct Counts {
    /// Queued and not made yet (now, not since the last line).
    waiting: AtomicU64,
    /// Pipelines made, the microseconds that took their workers, and the
    /// longest.
    made: AtomicU64,
    make_us: AtomicU64,
    longest_us: AtomicU64,
    /// Asks for a host pipeline (draws and dispatches recorded, in
    /// diagnostics builds also their trace) that found it being made: what
    /// a frame drops "being made".
    unready: AtomicU64,
}

impl Counts {
    const fn new() -> Counts {
        Counts {
            waiting: AtomicU64::new(0),
            made: AtomicU64::new(0),
            make_us: AtomicU64::new(0),
            longest_us: AtomicU64::new(0),
            unready: AtomicU64::new(0),
        }
    }

    /// What it counted since the last call.
    fn take(&self) -> Taken {
        Taken {
            waiting: self.waiting.load(Ordering::Relaxed),
            made: self.made.swap(0, Ordering::Relaxed),
            make_us: self.make_us.swap(0, Ordering::Relaxed),
            longest_us: self.longest_us.swap(0, Ordering::Relaxed),
            unready: self.unready.swap(0, Ordering::Relaxed),
        }
    }
}

/// [`Counts`] taken.
#[derive(Debug, Clone, Copy, Default)]
struct Taken {
    waiting: u64,
    made: u64,
    make_us: u64,
    longest_us: u64,
    unready: u64,
}

/// Workers started per pool and dedicated to passes over the screen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Started {
    pools: [usize; 2],
    screen: usize,
}

/// Dedicated workers that make passes over the screen alone, besides the
/// regular pools ([`screen_budget`], `XTP_SCREEN_WAIT_MS`).
const SCREEN_WORKERS: usize = 2;

/// The workers, their pools' queues, and what they did.
struct Pools {
    /// Both pools' queues ([`CHEAP`], [`HEAVY`]), under one lock: a
    /// pipeline started early in one pool can be used, and queued again,
    /// in the other.
    queues: Mutex<[Queues; 2]>,
    /// Signalled when a pool has a pipeline to make.
    ready: [Condvar; 2],
    /// Signalled when [`Queues::screen`] in [`CHEAP`] has a screen pass to
    /// make.
    screen_ready: Condvar,
    /// Workers per pool (and screen-pass workers), started on first use.
    workers: OnceLock<Started>,
    counts: [Counts; 2],
}

/// The run's.
static POOLS: Pools = Pools::new();

impl Pools {
    const fn new() -> Pools {
        Pools {
            queues: Mutex::new([Queues::new(), Queues::new()]),
            ready: [Condvar::new(), Condvar::new()],
            screen_ready: Condvar::new(),
            workers: OnceLock::new(),
            counts: [Counts::new(), Counts::new()],
        }
    }

    /// The workers per pool, started now if they are not yet.
    fn workers(&'static self) -> Started {
        *self.workers.get_or_init(|| {
            let processors = std::thread::available_parallelism().map_or(2, |n| n.get());
            let asked = match std::env::var("XTP_PIPELINE_WORKERS") {
                Ok(text) => match text.trim().parse::<usize>() {
                    Ok(n) => Some(n),
                    Err(_) => {
                        tracing::warn!(text, "XTP_PIPELINE_WORKERS expects a number of workers");
                        None
                    }
                },
                Err(_) => None,
            };
            let ram_cap = std::env::var_os("XTP_PIPELINE_RAM_CAP").is_some_and(|v| v == "1");
            let limited_ram = ram_cap && limited_host_ram();
            let screens = if screens_first() { SCREEN_WORKERS } else { 0 };
            self.spawn(wanted_for(processors, asked, ram_cap, limited_ram), screens)
        })
    }

    /// Wakes sleeping workers in both regular pools.
    fn wake_workers(&self) {
        let _held = self.queues.lock().unwrap_or_else(|e| e.into_inner());
        self.ready[CHEAP].notify_all();
        self.ready[HEAVY].notify_all();
    }

    /// Starts `wanted` workers per pool and `wanted_screen` screen-pass
    /// workers: how many started.
    fn spawn(&'static self, wanted: [usize; 2], wanted_screen: usize) -> Started {
        let mut started = [0; 2];
        for pool in [CHEAP, HEAVY] {
            let slow_limit = slow_limit_for(wanted[pool]);
            started[pool] = (0..wanted[pool])
                .filter(|&i| {
                    let name = match pool {
                        CHEAP => format!("xtp-pipelines-{i}"),
                        _ => format!("xtp-pipelines-heavy-{i}"),
                    };
                    let spawned = worker(name).spawn(move || self.work(pool, false, slow_limit));
                    if let Err(e) = &spawned {
                        tracing::warn!(error = %e, pool = POOL_NAMES[pool], "no pipeline worker");
                    }
                    spawned.is_ok()
                })
                .count();
        }
        let screen = (0..wanted_screen)
            .filter(|&i| {
                let name = format!("xtp-pipelines-screen-{i}");
                let spawned = worker(name).spawn(move || self.work(CHEAP, true, 0));
                if let Err(e) = &spawned {
                    tracing::warn!(error = %e, pool = "screen", "no pipeline worker");
                }
                spawned.is_ok()
            })
            .count();
        tracing::info!(
            cheap = started[CHEAP],
            heavy = started[HEAVY],
            screen,
            heavy_from_bytes = HEAVY_BYTES,
            "pipeline workers"
        );
        Started {
            pools: started,
            screen,
        }
    }

    /// The host pipeline if it is made, or (for a cheap, library-cached, or
    /// screen-pass one) is made within the frame's budget; queues it ahead
    /// of the early ones on first use. An uncached heavy non-screen pipeline
    /// is not waited for: it takes seconds, and the frame's budget is the
    /// cheap ones'.
    fn ready<'a>(
        &'static self,
        p: &'a Arc<Pending>,
        frames: &Frame,
        frame: u64,
        budget: Budget,
        screen_budget: Duration,
    ) -> Option<&'a HostPipeline> {
        if p.made.get().is_none() {
            let screens = !screen_budget.is_zero();
            self.start_as(p, URGENT, screens);
            let screen = p.screen && screens;
            if screen || !heavy(p) || super::library::in_library(&p.key) {
                wait_with(p, frames, frame, budget, screen_budget, shared_budget());
            }
            if p.made.get().is_none() {
                self.counts[p.pool()]
                    .unready
                    .fetch_add(1, Ordering::Relaxed);
            }
        }
        let made = p.made.get()?.as_ref();
        if made.is_some()
            && !p.used.load(Ordering::Relaxed)
            && !p.used.swap(true, Ordering::Relaxed)
        {
            super::library::note_used(&p.key);
        }
        made
    }

    /// Hands `p` to its pool's workers ([`pool_for`]): once, and once more
    /// ahead of the early ones on a first use, unless a worker has taken
    /// it already.
    fn start(&'static self, p: &Arc<Pending>, how: u8) {
        self.start_as(p, how, screens_first());
    }

    fn start_as(&'static self, p: &Arc<Pending>, how: u8, screens_first: bool) {
        if p.queued.load(Ordering::Relaxed) >= how {
            return;
        }
        let started = self.workers();
        let screen_lane = how == URGENT && p.screen && screens_first;
        if started.pools == [0, 0] && !(screen_lane && started.screen > 0) {
            // No worker can take `p`: make it here instead.
            let was = {
                let _held = self.queues.lock().unwrap_or_else(|e| e.into_inner());
                p.queued.swap(MAKING, Ordering::Relaxed)
            };
            if was == IDLE {
                WAITING.fetch_add(1, Ordering::Relaxed);
                self.counts[p.pool()]
                    .waiting
                    .fetch_add(1, Ordering::Relaxed);
                self.finish(p);
            }
            return;
        }
        let pool = if screen_lane && (started.pools[CHEAP] > 0 || started.screen > 0) {
            CHEAP
        } else {
            pool_for(p, started.pools)
        };
        let slow = how == EARLY
            && slow_enabled()
            && !super::library::in_library(&p.key)
            && super::library::slow(&p.key);
        let mut queues = self.queues.lock().unwrap_or_else(|e| e.into_inner());
        let was = p.pool();
        match queues[pool].push_as(p, how, slow, screens_first) {
            Some(true) => {
                PEAK.fetch_max(
                    WAITING.fetch_add(1, Ordering::Relaxed) + 1,
                    Ordering::Relaxed,
                );
                self.counts[pool].waiting.fetch_add(1, Ordering::Relaxed);
            }
            Some(false) if was != pool => {
                self.counts[was].waiting.fetch_sub(1, Ordering::Relaxed);
                self.counts[pool].waiting.fetch_add(1, Ordering::Relaxed);
            }
            Some(false) => {}
            None => return,
        }
        p.pool.store(pool as u8, Ordering::Relaxed);
        if screen_lane && pool == CHEAP {
            self.screen_ready.notify_one();
        }
        self.ready[pool].notify_one();
    }

    /// A worker of `pool`: makes its pipelines, first uses first, at the
    /// title's own priority. Below it, the world's busy threads starved the
    /// workers (VIctory #1505: 236 pipelines a minute at the start screen,
    /// 7 a minute in the world), and XTP's Series S world dropped 571 to
    /// 747 draws a frame waiting for a pipeline "being made".
    fn work(&self, pool: usize, screens_only: bool, slow_limit: usize) {
        let mut done: Option<Arc<Pending>> = None;
        loop {
            let next = {
                let mut queues = self.queues.lock().unwrap_or_else(|e| e.into_inner());
                if let Some(done) = done.take()
                    && queues[pool].made(&done)
                {
                    // What it passed over (or a throttled slow pipeline) may
                    // be free to run now.
                    self.ready[pool].notify_all();
                    if pool == CHEAP && !queues[CHEAP].screen.is_empty() {
                        self.screen_ready.notify_all();
                    }
                }
                loop {
                    if let Some(p) = queues[pool].pop_with(screens_only, slow_limit, |p| {
                        super::library::in_library(&p.key)
                    }) {
                        break p;
                    }
                    let cond = if screens_only {
                        &self.screen_ready
                    } else {
                        &self.ready[pool]
                    };
                    queues = cond.wait(queues).unwrap_or_else(|e| e.into_inner());
                }
            };
            self.finish(&next);
            done = Some(next);
        }
    }

    /// Makes `p` and keeps the counts, in place before `p` is signalled
    /// finished.
    fn finish(&self, p: &Pending) {
        let started = Instant::now();
        let made = p.make(|| {
            let us = started.elapsed().as_micros() as u64;
            let counts = &self.counts[p.pool()];
            counts.made.fetch_add(1, Ordering::Relaxed);
            counts.make_us.fetch_add(us, Ordering::Relaxed);
            counts.longest_us.fetch_max(us, Ordering::Relaxed);
            counts.waiting.fetch_sub(1, Ordering::Relaxed);
        });
        if !made {
            return;
        }
        if WAITING.fetch_sub(1, Ordering::Relaxed) == 1 {
            drained();
        } else {
            super::library::save(false);
        }
    }
}

/// The pool that makes `p`: the heavy one for an uncached heavy pipeline
/// ([`heavy`]), else the cheap one; the other one if that has no worker.
/// Library hits are never sent to the heavy pool.
fn pool_for(p: &Pending, workers: [usize; 2]) -> usize {
    let pool = if heavy(p) && !super::library::in_library(&p.key) {
        HEAVY
    } else {
        CHEAP
    };
    if workers[pool] > 0 { pool } else { 1 - pool }
}

/// Stack of each pipeline worker: room for the driver's shader compiler,
/// which runs on it (reserved address space; committed only as used).
/// Every title pipeline is compiled by [`Pending::make`], from
/// [`Pools::finish`]: on a worker, or, when none could start, on the
/// title's thread with the title's stack.
const PIPELINE_STACK: usize = 64 << 20;

/// A pipeline worker's thread, before it starts: its name and its stack.
fn worker(name: String) -> std::thread::Builder {
    std::thread::Builder::new()
        .name(name)
        // The driver compiles pipelines on this thread, recursively:
        // NVIDIA's compiler (nvgpucomp64) overflowed the default 2 MB
        // on a world compute program and refused it as out of memory.
        .stack_size(PIPELINE_STACK)
}

/// Pipeline workers at most, both pools.
///
/// One translation can hold a worker for seconds: in the Series S world
/// 16 compute programs took 20 to 28 s each before the translator refused
/// them (lane masks read as data; docs/analysis/drops.md), and with eight
/// workers a few of them held most of the pool while cheap pipelines
/// waited, with 8 to 10 of the process's threads running on 32 logical
/// processors. The workers run at the title's own priority.
const MAX_WORKERS: usize = 16;

/// Workers of the heavy pool, of 8 workers or more (one below): a few
/// heavy programs at a time, while all the others make cheap pipelines.
const HEAVY_WORKERS: usize = 2;

/// Workers per pool ([`CHEAP`], [`HEAVY`]) for `processors` logical
/// processors: half of them, 2 to [`MAX_WORKERS`], of which
/// [`HEAVY_WORKERS`] (one, of fewer than 8) are the heavy pool's.
#[cfg(test)]
fn wanted(processors: usize) -> [usize; 2] {
    wanted_for(processors, None, false, false)
}

/// Workers per pool ([`CHEAP`], [`HEAVY`]) for `processors` logical
/// processors: `asked` (`XTP_PIPELINE_WORKERS`, 1 to 64) overrides the
/// default; `ram_cap` (`XTP_PIPELINE_RAM_CAP=1`, off by default) caps the
/// default at 8 on hosts with at most 32 GiB of physical RAM (`limited_ram`).
fn wanted_for(
    processors: usize,
    asked: Option<usize>,
    ram_cap: bool,
    limited_ram: bool,
) -> [usize; 2] {
    let all = match asked {
        Some(n) => n.clamp(1, 64),
        None if ram_cap && limited_ram => (processors / 2).clamp(2, 8),
        None => (processors / 2).clamp(2, MAX_WORKERS),
    };
    let heavy = if all >= 8 {
        HEAVY_WORKERS
    } else if all >= 2 {
        1
    } else {
        0
    };
    [all - heavy, heavy]
}

/// Whether the host has at most 32 GiB of physical RAM (used only when
/// `XTP_PIPELINE_RAM_CAP=1`).
fn limited_host_ram() -> bool {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    let mut status = MEMORYSTATUSEX {
        dwLength: size_of::<MEMORYSTATUSEX>() as u32,
        ..Default::default()
    };
    // SAFETY: a sized structure of ours.
    unsafe { GlobalMemoryStatusEx(&mut status) != 0 && status.ullTotalPhys <= 32u64 << 30 }
}

/// The most pipelines waiting at once since the workers last ran out.
static PEAK: AtomicU64 = AtomicU64::new(0);

/// The workers ran out of work: logged after a batch of 10 or more, and
/// the cache saved (at most every 20 s).
fn drained() {
    use super::library::{EARLY, LOADED, MADE};
    if PEAK.swap(0, Ordering::Relaxed) >= 10 {
        tracing::info!(
            made = MADE.load(Ordering::Relaxed),
            loaded = LOADED.load(Ordering::Relaxed),
            started_early = EARLY.load(Ordering::Relaxed),
            "host pipelines all made"
        );
    }
    super::library::save(false);
}

/// The pools' line comes at most this often.
const REPORT_EVERY: Duration = Duration::from_secs(10);

/// The pools' line ([`describe`]), at the first ask for a pipeline of a
/// title frame, [`REPORT_EVERY`] at most; none while the pools have
/// nothing to say.
fn report(frame: u64) {
    static SEEN: AtomicU64 = AtomicU64::new(u64::MAX);
    static LAST: Mutex<Option<(Instant, u64)>> = Mutex::new(None);
    if SEEN.load(Ordering::Relaxed) == frame || SEEN.swap(frame, Ordering::Relaxed) == frame {
        return;
    }
    let Ok(mut last) = LAST.try_lock() else {
        return;
    };
    let now = Instant::now();
    let &mut (at, since) = last.get_or_insert((now, frame));
    if now.duration_since(at) < REPORT_EVERY {
        return;
    }
    *last = Some((now, frame));
    let taken = [CHEAP, HEAVY].map(|pool| POOLS.counts[pool].take());
    if let Some(line) = describe(taken, frame.saturating_sub(since)) {
        tracing::info!("{line}");
    }
}

/// The pools' line for what they did over `frames` title frames: what a
/// frame found being made, what was made and how long that took, and what
/// waits. `None` if nothing was made, found being made, or waits.
fn describe(taken: [Taken; 2], frames: u64) -> Option<String> {
    if taken
        .iter()
        .all(|t| t.made == 0 && t.unready == 0 && t.waiting == 0)
    {
        return None;
    }
    let per_frame = |n: u64| n as f64 / frames.max(1) as f64;
    let time = |us: u64| {
        if us >= 1_000_000 {
            format!("{:.1} s", us as f64 / 1e6)
        } else {
            format!("{} ms", us / 1000)
        }
    };
    let made = |t: &Taken, pool: usize| {
        let name = POOL_NAMES[pool];
        match t.made {
            0 => format!("0 {name}"),
            n => format!(
                "{n} {name} (mean {}, longest {})",
                time(t.make_us / n),
                time(t.longest_us)
            ),
        }
    };
    let [cheap, heavy] = &taken;
    Some(format!(
        "pipeline pools over {frames} title frames: being made when drawn {:.1}/frame heavy, \
         {:.1} cheap; made {}, {}; waiting {} heavy, {} cheap",
        per_frame(heavy.unready),
        per_frame(cheap.unready),
        made(heavy, HEAVY),
        made(cheap, CHEAP),
        heavy.waiting,
        cheap.waiting,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    fn until_finished(p: &Pending) {
        let started = Instant::now();
        while !p.finished.load(Ordering::Acquire) && started.elapsed() < Duration::from_secs(60) {
            std::thread::yield_now();
        }
    }

    /// A shader whose console program is `bytes` long, not translated.
    fn program(stage: &'static str, bytes: usize) -> Arc<Shader> {
        Arc::new(Shader {
            stage,
            hash: format!("{bytes:016x}"),
            container: vec![0; bytes],
            native: OnceLock::new(),
            translated: OnceLock::new(),
        })
    }

    /// Pools of their own with `workers` per pool, for a test.
    fn pools(workers: [usize; 2]) -> &'static Pools {
        let pools: &'static Pools = Box::leak(Box::new(Pools::new()));
        let started = pools.spawn(workers, 0);
        assert_eq!(started.pools, workers, "workers started");
        pools.workers.set(started).unwrap();
        pools
    }

    /// A first use queues the pipeline and, within the budget, gets the
    /// worker's answer (here: no host pipeline for no shaders).
    #[test]
    fn first_use_is_made_by_a_worker() {
        let p = Arc::new(Pending::new(Vec::new(), None, None));
        assert!(ready(&p).is_none());
        // Queued for a first use, or taken by a worker already.
        assert!(p.queued.load(Ordering::Relaxed) >= URGENT);
        until_finished(&p);
        assert!(matches!(p.made.get(), Some(None)));
    }

    /// A pipeline that is not made in time costs a frame its budget once:
    /// later uses in that frame drop at once, the next frame waits again.
    #[test]
    fn waits_stop_at_the_frames_budget() {
        let never = Pending::new(Vec::new(), None, None);
        let frames = Frame::new();
        let budget = Budget([Duration::from_millis(30), Duration::from_millis(60)]);
        let timed = |frame| {
            let started = Instant::now();
            wait(&never, &frames, frame, budget);
            started.elapsed()
        };
        let first = timed(7);
        assert!(first >= Duration::from_millis(30), "{first:?}");
        assert!(timed(7) < Duration::from_millis(10), "the budget is spent");
        assert!(
            timed(8) >= Duration::from_millis(30),
            "a new frame waits again"
        );
        // Compute pipelines have their own.
        assert_eq!(frames.left(8, 1, budget.0[1]), Duration::from_millis(60));
        assert!(take_wait_us()[0] >= 60_000);
    }

    /// A worker's stack, as the thread itself sees it, is the 64 MiB kept
    /// for the driver's compiler, not Rust's default 2 MiB (VIctory #1129),
    /// and only reserved: what is committed at its start is less than the
    /// whole default stack.
    #[test]
    fn workers_have_room_for_the_drivers_compiler() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEMORY_BASIC_INFORMATION, VirtualQuery,
        };
        let (reserved, committed) = worker("xtp-pipelines-test".into())
            .spawn(|| {
                let (mut low, mut high) = (0, 0);
                // SAFETY: two out-pointers to this thread's own locals.
                unsafe {
                    windows::Win32::System::Threading::GetCurrentThreadStackLimits(
                        &mut low, &mut high,
                    );
                }
                let (mut at, mut committed) = (low, 0);
                while at < high {
                    // SAFETY: a query of our own stack's pages into a local.
                    let info = unsafe {
                        let mut info = std::mem::zeroed::<MEMORY_BASIC_INFORMATION>();
                        let n = VirtualQuery(at as *const _, &mut info, size_of_val(&info));
                        assert_ne!(n, 0, "VirtualQuery");
                        info
                    };
                    let end = (info.BaseAddress as usize + info.RegionSize).min(high);
                    if info.State == MEM_COMMIT {
                        committed += end - at;
                    }
                    at = end;
                }
                (high - low, committed)
            })
            .unwrap()
            .join()
            .unwrap();
        assert!(reserved >= PIPELINE_STACK, "{reserved} bytes reserved");
        assert!(
            committed < 2 << 20,
            "{committed} bytes committed at the start"
        );
    }

    #[test]
    fn budgets_read_one_or_two_numbers() {
        let ms = |g, c| Some(Budget([Duration::from_millis(g), Duration::from_millis(c)]));
        assert_eq!(Budget::parse("5"), ms(5, 5));
        assert_eq!(Budget::parse(" 1, 20"), ms(1, 20));
        assert_eq!(Budget::parse("0,0"), ms(0, 0));
        assert_eq!(Budget::parse("1,2,3"), None);
        assert_eq!(Budget::parse("fast"), None);
        assert_eq!(Budget::parse(""), None);
    }

    /// A pipeline started early and then used is queued again, ahead, and
    /// made (and counted) once.
    #[test]
    fn early_then_used_is_made_once() {
        let p = Arc::new(Pending::new(Vec::new(), None, None));
        POOLS.start(&p, EARLY);
        POOLS.start(&p, URGENT);
        POOLS.start(&p, EARLY);
        assert!(p.queued.load(Ordering::Relaxed) >= URGENT);
        until_finished(&p);
        assert_eq!(p.queued.load(Ordering::Relaxed), MAKING);
        assert!(matches!(p.made.get(), Some(None)));
        assert!(!p.make(|| {}), "already finished");
    }

    /// First uses come out first; a pipeline started early and then used
    /// comes out once, at its first use's place, and is not handed out
    /// again once taken (VIctory #546's test).
    #[test]
    fn first_uses_come_first_and_each_pipeline_once() {
        let pending = || Arc::new(Pending::new(Vec::new(), None, None));
        let (earlier, promoted, used) = (pending(), pending(), pending());
        let mut queues = Queues::new();
        assert_eq!(queues.push(&earlier, EARLY), Some(true));
        assert_eq!(queues.push(&promoted, EARLY), Some(true));
        assert_eq!(queues.push(&used, URGENT), Some(true));
        assert_eq!(queues.push(&promoted, URGENT), Some(false));
        assert_eq!(queues.push(&promoted, URGENT), None);
        assert_eq!(queues.push(&promoted, EARLY), None);

        assert!(Arc::ptr_eq(&queues.pop().unwrap(), &used));
        assert!(Arc::ptr_eq(&queues.pop().unwrap(), &promoted));
        assert_eq!(promoted.queued.load(Ordering::Relaxed), MAKING);
        assert_eq!(queues.push(&promoted, URGENT), None, "taken already");
        assert!(Arc::ptr_eq(&queues.pop().unwrap(), &earlier));
        assert!(queues.pop().is_none(), "promoted's early entry passed over");
        assert_eq!(queues.making.len(), 3, "all three being made");
        assert!(!queues.made(&used), "nothing passed over");
        assert_eq!(queues.making.len(), 2);
    }

    /// A pipeline is heavy when one of its console programs is 48 KiB or
    /// more, unless that program is known to be refused.
    #[test]
    fn large_programs_make_heavy_pipelines() {
        let pipeline = |shaders: Vec<Arc<Shader>>| Pending::new(shaders, None, None);
        let small = program("vertex", HEAVY_BYTES - 1);
        let large = program("pixel", HEAVY_BYTES);
        assert!(!heavy(&pipeline(vec![small.clone()])));
        assert!(heavy(&pipeline(vec![large.clone()])));
        assert!(heavy(&pipeline(vec![small.clone(), large.clone()])));
        assert!(!heavy(&pipeline(Vec::new())));
        assert!(large.translated.set(None).is_ok());
        assert!(
            !heavy(&pipeline(vec![small, large])),
            "a refused translation fails at once"
        );
        assert_eq!(
            pool_for(&pipeline(vec![program("compute", 1 << 20)]), [3, 0]),
            CHEAP,
            "no heavy worker"
        );
        assert_eq!(
            pool_for(&pipeline(Vec::new()), [0, 1]),
            HEAVY,
            "no cheap worker"
        );
    }

    /// Half the logical processors, 2 to 16, of which 2 (1 of fewer than
    /// 8 workers) for the heavy pipelines.
    #[test]
    fn workers_split_between_the_pools() {
        assert_eq!(wanted(32), [14, 2]);
        assert_eq!(wanted(64), [14, 2]);
        assert_eq!(wanted(16), [6, 2]);
        assert_eq!(wanted(12), [5, 1]);
        assert_eq!(wanted(4), [1, 1]);
        assert_eq!(wanted(1), [1, 1]);
    }

    /// A pipeline whose heavy program another worker is translating is
    /// passed over, keeps its place, and is taken once that one is made.
    /// A small program shared with one being made does not hold it back.
    #[test]
    fn a_pipeline_waiting_for_a_translation_is_passed_over() {
        let large = program("compute", HEAVY_BYTES);
        let small = program("vertex", 64);
        let pending = |shaders: Vec<Arc<Shader>>| Arc::new(Pending::new(shaders, None, None));
        let (first, twin, other) = (
            pending(vec![large.clone()]),
            pending(vec![large.clone()]),
            pending(vec![small.clone(), program("pixel", HEAVY_BYTES + 4)]),
        );
        let mut queues = Queues::new();
        for p in [&first, &twin, &other] {
            assert_eq!(queues.push(p, URGENT), Some(true));
        }
        assert!(Arc::ptr_eq(&queues.pop().unwrap(), &first));
        assert!(
            Arc::ptr_eq(&queues.pop().unwrap(), &other),
            "the twin waits for the first's translation"
        );
        let sharing = pending(vec![small.clone()]);
        assert_eq!(queues.push(&sharing, URGENT), Some(true));
        assert!(
            Arc::ptr_eq(&queues.pop().unwrap(), &sharing),
            "a small program is not waited for"
        );
        assert!(queues.pop().is_none(), "the twin still waits");
        // The first is made (its program refused).
        assert!(large.translated.set(None).is_ok());
        assert!(queues.made(&first), "the twin was passed over");
        assert!(Arc::ptr_eq(&queues.pop().unwrap(), &twin));
        assert!(!queues.made(&twin));
    }

    #[test]
    fn the_pools_line_says_what_each_pool_did() {
        let quiet = [Taken::default(); 2];
        assert_eq!(describe(quiet, 10), None);
        let cheap = Taken {
            waiting: 3,
            made: 120,
            make_us: 120 * 35_000,
            longest_us: 900_000,
            unready: 3_272,
        };
        let heavy = Taken {
            waiting: 10,
            made: 4,
            make_us: 4 * 12_100_000,
            longest_us: 27_000_000,
            unready: 851,
        };
        assert_eq!(
            describe([cheap, heavy], 10).unwrap(),
            "pipeline pools over 10 title frames: being made when drawn 85.1/frame heavy, \
             327.2 cheap; made 4 heavy (mean 12.1 s, longest 27.0 s), 120 cheap (mean 35 ms, \
             longest 900 ms); waiting 10 heavy, 3 cheap"
        );
        let waiting = Taken {
            waiting: 1,
            ..Taken::default()
        };
        assert!(
            describe([Taken::default(), waiting], 0)
                .unwrap()
                .contains("made 0 heavy, 0 cheap")
        );
    }

    /// Words a big console compute program inserts before its store, to be
    /// heavy ([`HEAVY_BYTES`]).
    const HEAVY_WORDS: usize = 16 << 10;

    /// A compute shader of a console program,
    /// `crate::shader_cache::wave64_program(false)` with `extra` ALU
    /// instructions on the stored value before its store (`salt` makes
    /// another program of the same size); its container is the program,
    /// which tells a heavy one. Its translation, by the translator for
    /// this host, is made at once, or, `held`, on a thread of its own that
    /// holds it until the sender returned sends or is dropped: a slow
    /// translation, as long as the test needs, which a worker making the
    /// pipeline waits for as for another worker's.
    fn compute(extra: usize, salt: u8, held: bool) -> (Arc<Shader>, Option<mpsc::Sender<()>>) {
        let (c, mut native) = crate::shader_cache::wave64_program(false);
        let mut words = vec![
            // v_add_nc_u32 v2, <salt>, v2
            0x4A04_0400 | (0x80 + u32::from(salt % 64)),
        ];
        words.extend((1..extra).map(|i| {
            if i % 2 == 0 {
                0x4A04_0408 // v_add_nc_u32 v2, s8, v2
            } else {
                0x3404_0481 // v_lshlrev_b32 v2, 1, v2
            }
        }));
        // After the prologue (7 words) and the value's read (1 word).
        let at = 8 * 4;
        native
            .code
            .splice(at..at, words.iter().flat_map(|w| w.to_le_bytes()));
        let shader = Arc::new(Shader {
            stage: "compute",
            hash: format!("{:08x}{salt:08x}", native.code.len()),
            container: native.code.clone(),
            native: OnceLock::new(),
            translated: OnceLock::new(),
        });
        let translate = move || {
            let host =
                xtp_shader::emit::translate_program_for(&c, &native, crate::device::host_target())
                    .unwrap_or_else(|(step, e)| panic!("{}: {e}", step.name()));
            Some(super::super::Translated {
                host,
                used: Vec::new(),
                wave64: native.wave64,
            })
        };
        if !held {
            assert!(shader.translated.set(translate()).is_ok());
            return (shader, None);
        }
        let (entered, translating) = mpsc::channel();
        let (release, released) = mpsc::channel::<()>();
        let s = shader.clone();
        std::thread::spawn(move || {
            s.translated.get_or_init(|| {
                entered.send(()).unwrap();
                let _ = released.recv();
                translate()
            });
        });
        translating.recv().unwrap();
        (shader, Some(release))
    }

    /// A heavy and a cheap pipeline asked for together, the heavy one
    /// first, made on the host (WARP in the check): the cheap one is ready
    /// first, within the frame's budget, while the heavy one, not waited
    /// for, is still being made (its translation held); each pool counts
    /// what it made and what was found being made. With one queue for
    /// both, the cheap one waits until the heavy one is made.
    #[test]
    fn a_cheap_pipeline_does_not_wait_behind_a_heavy_one() {
        let test = "pipeline::make::tests::a_cheap_pipeline_does_not_wait_behind_a_heavy_one";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        let pipeline = |extra, salt, held| {
            let (shader, release) = compute(extra, salt, held);
            (Arc::new(Pending::new(vec![shader], None, None)), release)
        };
        let frames = Frame::new();

        let split = pools([1, 1]);
        let ((heavy, release), (cheap, _)) =
            (pipeline(HEAVY_WORDS, 1, true), pipeline(0, 2, false));
        assert!(super::heavy(&heavy) && !super::heavy(&cheap));
        let budget = Budget([Duration::from_secs(30); 2]);
        assert!(
            split
                .ready(&heavy, &frames, 1, budget, Duration::ZERO)
                .is_none(),
            "a heavy pipeline is not waited for"
        );
        assert!(
            split
                .ready(&cheap, &frames, 1, budget, Duration::ZERO)
                .is_some(),
            "the cheap one is made within the budget"
        );
        assert!(
            !heavy.finished.load(Ordering::Relaxed),
            "the cheap one is ready first"
        );
        drop(release);
        until_finished(&heavy);
        assert!(
            matches!(heavy.made.get(), Some(Some(_))),
            "the heavy one is made too"
        );
        // Each pool's counts are in place once its pipeline is finished
        // (`ready` returns the cheap one as soon as it is made).
        until_finished(&cheap);
        let [cheap_did, heavy_did] = split.counts.each_ref().map(Counts::take);
        assert_eq!(
            (heavy_did.made, heavy_did.unready, heavy_did.waiting),
            (1, 1, 0)
        );
        assert_eq!(
            (cheap_did.made, cheap_did.unready, cheap_did.waiting),
            (1, 0, 0)
        );
        // No comparison of the two pools' longest times: on a loaded check
        // machine the cheap pipeline's creation can outlast the heavy one's
        // (its hold ends as soon as `release` drops), and the order is
        // already pinned by "the cheap one is ready first" above.

        // One worker, one queue: the cheap one waits behind.
        let one = pools([1, 0]);
        let ((heavy, release), (cheap, _)) =
            (pipeline(HEAVY_WORDS, 3, true), pipeline(0, 4, false));
        let short = Budget([Duration::from_millis(300); 2]);
        assert!(
            one.ready(&heavy, &frames, 2, short, Duration::ZERO)
                .is_none()
        );
        assert!(
            one.ready(&cheap, &frames, 2, short, Duration::ZERO)
                .is_none(),
            "with one queue, the cheap one is not made while the heavy one is"
        );
        assert!(!cheap.finished.load(Ordering::Relaxed));
        drop(release);
        until_finished(&cheap);
        assert!(
            heavy.finished.load(Ordering::Relaxed),
            "with one queue, the cheap one is made after the heavy one"
        );
        assert!(matches!(cheap.made.get(), Some(Some(_))));
    }

    /// Early pipelines known to take long leave workers for the others: at
    /// most `slow_limit` run at once, a draw asking for one runs at once
    /// without counting against the cap, and a library hit is never
    /// throttled.
    #[test]
    fn pipelines_known_to_take_long_leave_workers_for_the_others() {
        let new = || Arc::new(Pending::new(Vec::new(), None, None));
        let (long, longer, asked, quick, cached) = (new(), new(), new(), new(), new());
        let mut queues = Queues::new();
        for p in [&long, &longer, &asked, &cached] {
            assert_eq!(queues.push_as(p, EARLY, true, true), Some(true));
        }
        assert_eq!(queues.push_as(&quick, EARLY, false, true), Some(true));
        // The quick one first, then one long one at a time; a draw asking
        // for one waits for nobody.
        assert!(Arc::ptr_eq(
            &queues.pop_with(false, 1, |_| false).unwrap(),
            &quick
        ));
        assert!(Arc::ptr_eq(
            &queues.pop_with(false, 1, |_| false).unwrap(),
            &long
        ));
        assert!(queues.pop_with(false, 1, |_| false).is_none());
        // A library hit is never throttled, even when the slow limit is full.
        assert!(Arc::ptr_eq(
            &queues
                .pop_with(false, 1, |p| std::ptr::eq(p, Arc::as_ptr(&cached)))
                .unwrap(),
            &cached
        ));
        assert!(!cached.was_slow.load(Ordering::Relaxed));
        assert_eq!(queues.slow_now, 1);

        assert_eq!(queues.push_as(&asked, URGENT, true, true), Some(false));
        assert!(Arc::ptr_eq(
            &queues.pop_with(false, 1, |_| false).unwrap(),
            &asked
        ));
        assert!(!asked.was_slow.load(Ordering::Relaxed));
        assert!(queues.pop_with(false, 1, |_| false).is_none());
        assert!(queues.made(&long), "freed a slow slot");
        assert!(Arc::ptr_eq(
            &queues.pop_with(false, 1, |_| false).unwrap(),
            &longer
        ));
        assert!(queues.pop_with(false, 1, |_| false).is_none());
    }

    /// Passes over the screen go before all others, and screen-pass workers
    /// take nothing else.
    #[test]
    fn passes_over_the_screen_go_first() {
        use windows::Win32::Graphics::Direct3D12::{
            D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE, D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        };
        use windows::Win32::Graphics::Dxgi::Common::{
            DXGI_FORMAT_D32_FLOAT, DXGI_FORMAT_R8G8B8A8_UNORM,
        };

        let mut screen_state = GraphicsState::default();
        screen_state.targets.NumRenderTargets = 1;
        screen_state.targets.RTFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        screen_state.topology = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        assert!(screen_state.screen_pass());

        let mut no_targets = screen_state.clone();
        no_targets.targets.NumRenderTargets = 0;
        assert!(!no_targets.screen_pass());

        let mut line_state = screen_state.clone();
        line_state.topology = D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE;
        assert!(!line_state.screen_pass());

        // Depth format set with default DepthEnable=true is not a screen pass;
        // disabling both depth and stencil makes it one again.
        let mut with_depth = screen_state.clone();
        // SAFETY: subobject type 16 is a 4-byte DXGI_FORMAT at +4.
        unsafe {
            let mut sub = [0u8; 8];
            sub[0..4].copy_from_slice(&16u32.to_le_bytes());
            sub[4..8].copy_from_slice(&(DXGI_FORMAT_D32_FLOAT.0 as u32).to_le_bytes());
            with_depth.read(16, sub.as_ptr());
        }
        assert!(!with_depth.screen_pass());
        // SAFETY: subobject type 11 is a D3D12_DEPTH_STENCIL_DESC at +4.
        unsafe {
            let ds = windows::Win32::Graphics::Direct3D12::D3D12_DEPTH_STENCIL_DESC::default();
            let mut sub = vec![0u8; 4 + size_of_val(&ds)];
            sub[0..4].copy_from_slice(&11u32.to_le_bytes());
            std::ptr::copy_nonoverlapping(
                std::ptr::from_ref(&ds).cast::<u8>(),
                sub.as_mut_ptr().add(4),
                size_of_val(&ds),
            );
            with_depth.read(11, sub.as_ptr());
        }
        assert!(with_depth.screen_pass());

        let screen = Arc::new(Pending::new(
            vec![program("vertex", 64), program("pixel", 64)],
            Some(Box::new(screen_state.clone())),
            None,
        ));
        assert!(screen.screen);
        let mesh = Arc::new(Pending::new(
            vec![program("mesh", 64), program("pixel", 64)],
            Some(Box::new(screen_state.clone())),
            None,
        ));
        assert!(!mesh.screen, "mesh pipelines are not screen passes");
        let world = Arc::new(Pending::new(Vec::new(), None, None));
        assert!(!world.screen);

        let mut queues = Queues::new();
        assert_eq!(queues.push_as(&world, URGENT, false, true), Some(true));
        assert_eq!(queues.push_as(&screen, URGENT, false, true), Some(true));
        assert!(Arc::ptr_eq(
            &queues.pop_with(true, 1, |_| false).unwrap(),
            &screen
        ));
        assert!(queues.pop_with(true, 1, |_| false).is_none());
        assert!(Arc::ptr_eq(
            &queues.pop_with(false, 1, |_| false).unwrap(),
            &world
        ));

        // With `screens_first = false` (`XTP_SCREEN_WAIT_MS=0`), a screen pass
        // goes into the regular `urgent` queue rather than `screen`.
        let screen_off = Arc::new(Pending::new(
            vec![program("vertex", 64), program("pixel", 64)],
            Some(Box::new(screen_state)),
            None,
        ));
        assert_eq!(
            queues.push_as(&screen_off, URGENT, false, false),
            Some(true)
        );
        assert!(queues.pop_with(true, 1, |_| false).is_none());
        assert!(Arc::ptr_eq(
            &queues.pop_with(false, 1, |_| false).unwrap(),
            &screen_off
        ));
    }

    /// Dedicated screen workers make a screen pass while the regular worker is
    /// blocked on a heavy pipeline, and a pool with only screen workers still
    /// makes non-screen pipelines inline instead of hanging.
    #[test]
    fn screen_workers_make_screen_passes_and_non_screen_falls_back_inline() {
        use windows::Win32::Graphics::Direct3D12::D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM;

        let mut screen_state = GraphicsState::default();
        screen_state.targets.NumRenderTargets = 1;
        screen_state.targets.RTFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        screen_state.topology = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;

        // Pool with 0 regular workers and 1 screen worker: non-screen work
        // (and screen passes when `screens_first = false`) must fall back
        // inline instead of queueing onto an empty pool forever.
        let screen_only: &'static Pools = Box::leak(Box::new(Pools::new()));
        let started = screen_only.spawn([0, 0], 1);
        assert_eq!(
            started,
            Started {
                pools: [0, 0],
                screen: 1
            }
        );
        screen_only.workers.set(started).unwrap();

        let regular = Arc::new(Pending::new(Vec::new(), None, None));
        screen_only.start_as(&regular, URGENT, true);
        assert!(regular.finished.load(Ordering::Acquire), "made inline");

        let screen_disabled = Arc::new(Pending::new(
            Vec::new(),
            Some(Box::new(screen_state.clone())),
            None,
        ));
        screen_only.start_as(&screen_disabled, URGENT, false);
        assert!(
            screen_disabled.finished.load(Ordering::Acquire),
            "made inline when screen lane disabled"
        );

        let screen = Arc::new(Pending::new(
            Vec::new(),
            Some(Box::new(screen_state.clone())),
            None,
        ));
        screen_only.start_as(&screen, URGENT, true);
        until_finished(&screen);
        assert!(
            screen.finished.load(Ordering::Acquire),
            "made by screen worker"
        );

        // End-to-end with WARP: block the sole regular worker on a held heavy
        // compute shader; a screen pass still completes immediately on its
        // dedicated screen worker.
        let test = "pipeline::make::tests::screen_workers_make_screen_passes_and_non_screen_falls_back_inline";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        let with_screen: &'static Pools = Box::leak(Box::new(Pools::new()));
        let started = with_screen.spawn([1, 0], 1);
        assert_eq!(
            started,
            Started {
                pools: [1, 0],
                screen: 1
            }
        );
        with_screen.workers.set(started).unwrap();

        let (heavy_shader, release) = compute(HEAVY_WORDS, 11, true);
        let heavy_p = Arc::new(Pending::new(vec![heavy_shader], None, None));
        let frames = Frame::new();
        let budget = Budget([Duration::from_millis(2), Duration::from_millis(8)]);
        assert!(
            with_screen
                .ready(&heavy_p, &frames, 1, budget, Duration::from_millis(500))
                .is_none()
        );
        // Wait until the regular worker has claimed `heavy_p`.
        let claimed = Instant::now();
        while heavy_p.queued.load(Ordering::Relaxed) != MAKING
            && claimed.elapsed() < Duration::from_secs(5)
        {
            std::thread::yield_now();
        }
        assert_eq!(heavy_p.queued.load(Ordering::Relaxed), MAKING);

        let screen_p = Arc::new(Pending::new(Vec::new(), Some(Box::new(screen_state)), None));
        let _ = with_screen.ready(&screen_p, &frames, 1, budget, Duration::from_secs(5));
        until_finished(&screen_p);
        assert!(
            !heavy_p.finished.load(Ordering::Acquire),
            "screen pass completed while regular worker was still busy"
        );
        drop(release);
        until_finished(&heavy_p);
        assert!(matches!(heavy_p.made.get(), Some(Some(_))));
    }

    /// Four threads sharing a 40 ms budget in one frame return within about
    /// 40 ms in total (not 160 ms), and unused reserved budget is refunded
    /// when a pipeline finishes early.
    #[test]
    fn four_threads_sharing_a_frame_budget_wait_at_most_the_budget() {
        let never = Pending::new(Vec::new(), None, None);
        let frames = Frame::new();
        let budget = Budget([Duration::from_millis(40), Duration::from_millis(40)]);
        let ready_gate = std::sync::Barrier::new(4);
        let wall_start = Instant::now();
        let per_thread: Vec<Duration> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..4)
                .map(|_| {
                    s.spawn(|| {
                        ready_gate.wait();
                        let started = Instant::now();
                        wait_with(&never, &frames, 1, budget, Duration::ZERO, true);
                        started.elapsed()
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let wall = wall_start.elapsed();
        let total: Duration = per_thread.iter().copied().sum();
        assert!(wall >= Duration::from_millis(30), "waited once: {wall:?}");
        assert!(
            total < Duration::from_millis(120),
            "shared 40 ms budget across 4 threads took {total:?} total ({per_thread:?})"
        );
        // Three threads saw 0 ms left and returned promptly.
        let fast = per_thread
            .iter()
            .filter(|&&d| d < Duration::from_millis(20))
            .count();
        assert_eq!(
            fast, 3,
            "three threads returned immediately: {per_thread:?}"
        );

        // Refund: a thread whose pipeline finishes early gives the unused
        // budget back to the same frame.
        let quick = Arc::new(Pending::new(Vec::new(), None, None));
        let quick_budget = Budget([Duration::from_millis(200), Duration::from_millis(200)]);
        let q = quick.clone();
        std::thread::scope(|s| {
            let waiter = s.spawn(|| {
                wait_with(&quick, &frames, 2, quick_budget, Duration::ZERO, true);
            });
            std::thread::sleep(Duration::from_millis(15));
            q.make(|| {});
            waiter.join().unwrap();
        });
        let remaining = frames.left(2, KIND_GRAPHICS, Duration::from_millis(200));
        assert!(
            remaining > Duration::from_millis(50),
            "unused budget was refunded: {remaining:?}"
        );
    }

    /// A late thread from an older frame gets zero budget without resetting
    /// the current frame; unshared mode (`XTP_PIPELINE_WAIT_SHARED=0`) lets
    /// concurrent threads wait independently; huge budgets and `u64::MAX`
    /// frame numbers do not overflow or collide with a sentinel.
    #[test]
    fn frame_budget_edge_cases_do_not_reset_or_overflow() {
        let frames = Frame::new();
        let budget = Duration::from_millis(40);
        assert_eq!(frames.reserve(5, KIND_GRAPHICS, budget, true), budget);
        // A late thread from frame 4 must not reset frame 5's reservation.
        assert_eq!(
            frames.reserve(4, KIND_GRAPHICS, budget, true),
            Duration::ZERO
        );
        assert_eq!(frames.left(5, KIND_GRAPHICS, budget), Duration::ZERO);
        frames.finish(5, KIND_GRAPHICS, budget, Duration::from_millis(10), true);
        assert_eq!(
            frames.left(5, KIND_GRAPHICS, budget),
            Duration::from_millis(30)
        );

        // Unshared mode: two threads both reserve the full budget, and huge
        // budgets saturate instead of panicking on overflow.
        let unshared = Frame::new();
        let huge = Duration::from_millis(u64::MAX);
        assert_eq!(unshared.reserve(u64::MAX, KIND_SCREEN, huge, false), huge);
        assert_eq!(unshared.reserve(u64::MAX, KIND_SCREEN, huge, false), huge);
        unshared.finish(u64::MAX, KIND_SCREEN, huge, huge, false);
        unshared.finish(u64::MAX, KIND_SCREEN, huge, huge, false);
        assert_eq!(unshared.left(u64::MAX, KIND_SCREEN, huge), Duration::ZERO);
    }

    /// The default worker count is not lowered on <= 32 GiB RAM; the RAM
    /// cap applies only when requested (`XTP_PIPELINE_RAM_CAP=1`), and an
    /// explicit worker count (`XTP_PIPELINE_WORKERS`) overrides both.
    #[test]
    fn worker_count_keeps_default_and_honors_switches() {
        assert_eq!(wanted_for(32, None, false, true), [14, 2]);
        assert_eq!(wanted_for(32, None, true, false), [14, 2]);
        assert_eq!(wanted_for(32, None, true, true), [6, 2]);
        assert_eq!(wanted_for(12, None, true, true), [5, 1]);
        assert_eq!(wanted_for(32, Some(0), true, true), [1, 0]);
        assert_eq!(wanted_for(32, Some(12), true, true), [10, 2]);
        assert_eq!(wanted_for(32, Some(100), false, false), [62, 2]);
    }
}
