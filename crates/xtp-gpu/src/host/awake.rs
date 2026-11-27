//! Keeps the host GPU's clocks up between submissions (`XTP_CARD_AWAKE=1`, off
//! by default).
//!
//! Between submissions in a frame the GPU can sit idle for a millisecond or two
//! many times while the CPU prepares the next submission. The driver can read
//! those gaps as a light load and lower GPU clocks, making the following
//! submission start slowly.
//!
//! When enabled, a worker thread runs one 64-thread compute group in a loop on
//! its own compute queue, taking a small slice of the GPU's wave slots so the
//! card stays active. The submit path only records a monotonic timestamp in an
//! atomic (no locks, blocking or allocation). Once no submission has gone out
//! for 250 ms (`DEFAULT_QUIET_MS`), the worker stops dispatching and sleeps
//! until submissions resume. When `XTP_CARD_AWAKE` is unset or not `1`, nothing
//! is created (no thread, no queue, no GPU resources).

use std::ffi::OsStr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use windows::Win32::Foundation::CloseHandle;
use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
use windows::core::{Interface, s};

/// One 64-thread compute group looping to keep the card active between
/// submissions (~2 ms per dispatch at [`DEFAULT_ITERATIONS`] on full clocks).
const SHADER: &str = r#"
#define RS "RootConstants(num32BitConstants=1, b0), UAV(u0)"
cbuffer Params : register(b0) { uint iterations; };
RWByteAddressBuffer sum : register(u0);
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint x = id.x;
    [loop] for (uint i = 0; i < iterations; i++) x = x * 1664525u + 1013904223u;
    sum.Store(id.x * 4u, x);
}
"#;

const DEFAULT_ITERATIONS: u32 = 400_000;
const DEFAULT_QUIET_MS: u64 = 250;
const IDLE_SLEEP: Duration = Duration::from_millis(2);

#[derive(Clone, Copy)]
struct Config {
    quiet_ms: u64,
    iterations: u32,
}

impl Config {
    const DEFAULT: Self = Self {
        quiet_ms: DEFAULT_QUIET_MS,
        iterations: DEFAULT_ITERATIONS,
    };
}

/// Whether `XTP_CARD_AWAKE` enables the keep-awake loop (`1` only; off by default).
fn enabled_value(value: Option<&OsStr>) -> bool {
    value
        .and_then(OsStr::to_str)
        .is_some_and(|v| v.trim() == "1")
}

/// Reads `XTP_CARD_AWAKE` once (off by default; `1` enables it).
fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = enabled_value(std::env::var_os("XTP_CARD_AWAKE").as_deref());
        if on {
            tracing::info!("XTP_CARD_AWAKE: keeping the card awake between submissions");
        }
        on
    })
}

/// Lock-free state shared between the submit path and the keep-awake worker.
struct Shared {
    origin: Instant,
    /// Milliseconds since `origin` plus one when the last submission went out (0: none).
    last_ms: AtomicU64,
    stop: AtomicBool,
    #[cfg(test)]
    dispatches: AtomicU64,
    #[cfg(test)]
    quiet: AtomicBool,
}

impl Shared {
    fn new() -> Self {
        Self {
            origin: Instant::now(),
            last_ms: AtomicU64::new(0),
            stop: AtomicBool::new(false),
            #[cfg(test)]
            dispatches: AtomicU64::new(0),
            #[cfg(test)]
            quiet: AtomicBool::new(true),
        }
    }

    fn now_ms(&self) -> u64 {
        self.origin.elapsed().as_millis() as u64
    }

    fn active(&self, quiet_ms: u64) -> bool {
        let last = self.last_ms.load(Ordering::Relaxed);
        last != 0 && self.now_ms().saturating_sub(last - 1) < quiet_ms
    }
}

/// `ID3D12Device` moved onto the keep-awake worker thread.
struct SendDevice(ID3D12Device);

// SAFETY: `ID3D12Device` methods are free-threaded.
unsafe impl Send for SendDevice {}

/// Handle to the keep-awake worker thread. Dropping it stops the worker, waits
/// for any in-flight compute dispatch to complete, and joins the thread.
pub(super) struct Awake {
    shared: Arc<Shared>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Awake {
    /// Starts the keep-awake worker for `device` if `XTP_CARD_AWAKE=1` is set.
    /// Returns `None` without creating a thread or GPU objects otherwise.
    pub(super) fn start_if_enabled(device: &ID3D12Device) -> Option<Self> {
        Self::start_if(enabled(), device, Config::DEFAULT)
    }

    fn start_if(on: bool, device: &ID3D12Device, config: Config) -> Option<Self> {
        if !on {
            return None;
        }
        let shared = Arc::new(Shared::new());
        let worker_shared = Arc::clone(&shared);
        let device = SendDevice(device.clone());
        let thread = match std::thread::Builder::new()
            .name("xtp-card-awake".into())
            .spawn(move || {
                let device = device;
                if let Err(e) = run(&device.0, &worker_shared, config) {
                    tracing::warn!(error = %e, "the card is not kept awake");
                }
            }) {
            Ok(thread) => thread,
            Err(e) => {
                tracing::warn!(error = %e, "could not start card keep-awake thread");
                return None;
            }
        };
        Some(Self {
            shared,
            thread: Some(thread),
        })
    }

    /// Notes that a submission was sent to the host queue.
    /// Lock-free and non-allocating: stores a monotonic timestamp only.
    pub(super) fn note_submission(&self) {
        let ms = self.shared.now_ms();
        self.shared
            .last_ms
            .store(ms.saturating_add(1), Ordering::Relaxed);
    }

    #[cfg(test)]
    fn dispatches(&self) -> u64 {
        self.shared.dispatches.load(Ordering::Relaxed)
    }

    #[cfg(test)]
    fn is_quiet(&self) -> bool {
        self.shared.quiet.load(Ordering::Acquire)
    }
}

impl Drop for Awake {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            thread.thread().unpark();
            let _ = thread.join();
        }
    }
}

fn compile() -> windows::core::Result<ID3DBlob> {
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: `SHADER` and string literals live through the call; out pointers are ours.
    let compiled = unsafe {
        D3DCompile(
            SHADER.as_ptr().cast(),
            SHADER.len(),
            s!("xtp card awake"),
            None,
            None,
            s!("main"),
            s!("cs_5_1"),
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = compiled {
        if let Some(errors) = errors {
            // SAFETY: the error blob's own readable byte range.
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    errors.GetBufferPointer().cast::<u8>(),
                    errors.GetBufferSize(),
                )
            };
            tracing::warn!("card awake shader: {}", String::from_utf8_lossy(bytes));
        }
        return Err(e);
    }
    code.ok_or_else(windows::core::Error::empty)
}

fn run(device: &ID3D12Device, shared: &Shared, config: Config) -> windows::core::Result<()> {
    let code = compile()?;
    // SAFETY: plain D3D12 object creation; `code` lives through root signature
    // and PSO creation and carries the embedded root signature. Two closed
    // lists are submitted in turn with at most two dispatches in flight so a
    // list is never reused while running. All objects outlive the final wait.
    unsafe {
        let bytes =
            std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
        let pso: ID3D12PipelineState =
            device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: bytes.as_ptr().cast(),
                    BytecodeLength: bytes.len(),
                },
                ..Default::default()
            })?;
        let queue: ID3D12CommandQueue = device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_COMPUTE,
            Priority: D3D12_COMMAND_QUEUE_PRIORITY_NORMAL.0,
            ..Default::default()
        })?;
        super::name(&queue, || "xtp card awake compute queue".into());
        let mut sum: Option<ID3D12Resource> = None;
        device.CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            },
            D3D12_HEAP_FLAG_NONE,
            &D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 256,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                ..Default::default()
            },
            D3D12_RESOURCE_STATE_COMMON,
            None,
            &mut sum,
        )?;
        let sum = sum.ok_or_else(windows::core::Error::empty)?;
        super::name(&sum, || "xtp card awake buffer".into());
        let allocator: ID3D12CommandAllocator =
            device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE)?;
        let mut lists = Vec::with_capacity(2);
        for i in 0..2 {
            let list: ID3D12GraphicsCommandList =
                device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, &allocator, &pso)?;
            super::name(&list, || format!("xtp card awake list {i}"));
            list.SetComputeRootSignature(&root);
            list.SetComputeRoot32BitConstant(0, config.iterations, 0);
            list.SetComputeRootUnorderedAccessView(1, sum.GetGPUVirtualAddress());
            list.Dispatch(1, 1, 1);
            list.Close()?;
            lists.push(Some(list.cast::<ID3D12CommandList>()?));
        }
        let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
        let event = CreateEventW(None, false, false, None)?;
        let mut signalled = 0u64;
        let mut result = Ok(());
        while !shared.stop.load(Ordering::Acquire) && result.is_ok() {
            if !shared.active(config.quiet_ms) {
                #[cfg(test)]
                shared.quiet.store(true, Ordering::Release);
                std::thread::park_timeout(IDLE_SLEEP);
                continue;
            }
            #[cfg(test)]
            shared.quiet.store(false, Ordering::Release);
            let completed = fence.GetCompletedValue();
            if completed == u64::MAX {
                result = device.GetDeviceRemovedReason();
                break;
            }
            // Keep at most two dispatches in flight (one per closed list).
            if signalled >= 2 && completed < signalled - 1 {
                result = fence.SetEventOnCompletion(signalled - 1, event);
                if result.is_ok() {
                    WaitForSingleObject(event, 1000);
                    result = device.GetDeviceRemovedReason();
                }
                continue;
            }
            queue.ExecuteCommandLists(&lists[(signalled % 2) as usize..][..1]);
            let next = signalled + 1;
            result = queue.Signal(&fence, next);
            if result.is_ok() {
                signalled = next;
                #[cfg(test)]
                shared.dispatches.fetch_add(1, Ordering::Relaxed);
            }
        }
        #[cfg(test)]
        shared.quiet.store(true, Ordering::Release);
        let completed = fence.GetCompletedValue();
        if completed < signalled
            && completed != u64::MAX
            && fence.SetEventOnCompletion(signalled, event).is_ok()
        {
            WaitForSingleObject(event, INFINITE);
        }
        let _ = CloseHandle(event);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wait_until(timeout: Duration, mut step: impl FnMut(), check: impl Fn() -> bool) -> bool {
        let deadline = Instant::now() + timeout;
        while !check() && Instant::now() < deadline {
            step();
            std::thread::sleep(Duration::from_millis(20));
        }
        check()
    }

    #[test]
    fn env_switch_is_off_unless_set_to_one() {
        for off in [
            None,
            Some(""),
            Some("0"),
            Some("false"),
            Some("true"),
            Some("2"),
        ] {
            assert!(!enabled_value(off.map(OsStr::new)), "{off:?}");
        }
        for on in ["1", " 1 "] {
            assert!(enabled_value(Some(OsStr::new(on))), "{on:?}");
        }
    }

    #[test]
    fn does_nothing_when_off() {
        let Some(gpu) = crate::testgpu::device("host::awake::tests::does_nothing_when_off") else {
            return;
        };
        assert!(Awake::start_if(false, &gpu.device, Config::DEFAULT).is_none());
        if std::env::var_os("XTP_CARD_AWAKE").is_none() {
            assert!(gpu.host().awake.is_none());
        }
    }

    #[test]
    fn dispatches_while_submissions_arrive_then_goes_quiet() {
        let Some(gpu) = crate::testgpu::device(
            "host::awake::tests::dispatches_while_submissions_arrive_then_goes_quiet",
        ) else {
            return;
        };
        let config = Config {
            quiet_ms: 150,
            iterations: 2_000,
        };
        let awake = Awake::start_if(true, &gpu.device, config).expect("keep-awake thread starts");

        std::thread::sleep(Duration::from_millis(25));
        assert_eq!(awake.dispatches(), 0);
        assert!(awake.is_quiet());

        assert!(
            wait_until(
                Duration::from_secs(10),
                || awake.note_submission(),
                || awake.dispatches() >= 1
            ),
            "expected at least one keep-awake dispatch while submissions arrived"
        );
        for _ in 0..3 {
            awake.note_submission();
            std::thread::sleep(Duration::from_millis(20));
        }
        assert!(
            wait_until(Duration::from_secs(10), || {}, || awake.is_quiet()),
            "expected keep-awake worker to go quiet after quiet window"
        );

        let settled = awake.dispatches();
        std::thread::sleep(Duration::from_millis(100));
        assert_eq!(
            awake.dispatches(),
            settled,
            "expected no dispatches while quiet"
        );

        assert!(
            wait_until(
                Duration::from_secs(10),
                || awake.note_submission(),
                || awake.dispatches() > settled,
            ),
            "expected keep-awake worker to resume dispatching after a new submission"
        );
        drop(awake);
    }
}
