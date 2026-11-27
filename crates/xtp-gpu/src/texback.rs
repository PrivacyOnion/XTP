//! Textures shaders write that a title's CPU code reads back.
//!
//! On the console the CPU and GPU share memory, so what a compute pass writes
//! into a texture through an unordered-access view is immediately visible in
//! title memory. On Windows it stays in the host texture copy unless copied
//! back.
//!
//! For textures matching a configured kind (`(width, height, dxgi_format)`),
//! configured by `XTP_TEXTURE_BACK` (comma-separated `<width>x<height>f<format>`,
//! `1` for the build's defaults, `0`/`off`/`false` to disable; unset uses the
//! build's defaults supplied via [`configure_texture_readback`], or off if
//! none), a list whose shaders bind one as a UAV copies it into a readback
//! buffer when the list closes. Once the submission completes, the linear
//! texels are tiled back into the texture's title memory in the console's
//! layout using [`xtp_formats::texture::Layout::tile`].
//!
//! Because UAV-writable textures carry `ALLOW_UNORDERED_ACCESS`,
//! [`crate::texture::title_written`] is false for them and their aliases, so
//! landing bytes into title memory never triggers an echo re-upload as a CPU
//! write.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use windows::Win32::Graphics::Direct3D12::*;

use crate::object::This;

/// Per-build default texture kinds `(width, height, dxgi_format)`, supplied
/// before GPU work starts (for example by `xtp-game` through the launcher).
static DEFAULT_KINDS: OnceLock<Vec<(u64, u32, u32)>> = OnceLock::new();

/// Sets the default texture kinds read back when `XTP_TEXTURE_BACK` is unset
/// or `1`. Must be called before the first UAV texture bind; later calls are
/// ignored.
pub fn configure_texture_readback(defaults: &[(u64, u32, u32)]) {
    let _ = DEFAULT_KINDS.set(defaults.to_vec());
}

fn parse_kinds(spec: &str, defaults: &[(u64, u32, u32)]) -> Vec<(u64, u32, u32)> {
    let trimmed = spec.trim();
    if trimmed.is_empty()
        || trimmed == "0"
        || trimmed.eq_ignore_ascii_case("off")
        || trimmed.eq_ignore_ascii_case("false")
    {
        return Vec::new();
    }
    if trimmed == "1" {
        return defaults.to_vec();
    }
    trimmed
        .split(',')
        .filter_map(|entry| {
            let (size, format) = entry.trim().split_once('f')?;
            let (w, h) = size.split_once('x')?;
            Some((w.parse().ok()?, h.parse().ok()?, format.parse().ok()?))
        })
        .collect()
}

/// The kinds copied back: `(width, height, dxgi_format)`.
///
/// Reads `XTP_TEXTURE_BACK`: `0`/`off`/`false` disables readback, `1` or unset
/// uses the defaults from [`configure_texture_readback`] (empty if unset), and
/// a comma-separated `<width>x<height>f<format>` list overrides them.
fn kinds() -> &'static [(u64, u32, u32)] {
    static KINDS: OnceLock<Vec<(u64, u32, u32)>> = OnceLock::new();
    KINDS.get_or_init(|| {
        let defaults = DEFAULT_KINDS.get().map_or(&[][..], Vec::as_slice);
        let kinds = match std::env::var("XTP_TEXTURE_BACK") {
            Ok(spec) => parse_kinds(&spec, defaults),
            Err(_) => defaults.to_vec(),
        };
        if !kinds.is_empty() {
            tracing::info!(
                ?kinds,
                "textures of these kinds that shaders write are copied back into title memory (XTP_TEXTURE_BACK)"
            );
        }
        kinds
    })
}

#[cfg(test)]
thread_local! {
    static TEST_KINDS: std::cell::RefCell<Option<Vec<(u64, u32, u32)>>> =
        const { std::cell::RefCell::new(None) };
}

/// Whether title texture `resource` is of a kind copied back.
pub(crate) fn wanted(resource: This) -> bool {
    #[cfg(test)]
    if let Some(hit) = TEST_KINDS.with_borrow(|over| {
        over.as_ref().map(|kinds| {
            !kinds.is_empty()
                && crate::resource::texture_kind(resource).is_some_and(|k| kinds.contains(&k))
        })
    }) {
        return hit;
    }
    let kinds = kinds();
    !kinds.is_empty() && crate::resource::texture_kind(resource).is_some_and(|k| kinds.contains(&k))
}

/// An outstanding readback to a specific title texture allocation. Mapping and
/// resource lifetime changes invalidate only the readbacks that could touch
/// replaced or released memory.
struct Ticket {
    owner: usize,
    at: usize,
    end: usize,
    valid: AtomicBool,
}

static TICKETS: Mutex<Vec<Weak<Ticket>>> = Mutex::new(Vec::new());
static TRACKING: AtomicBool = AtomicBool::new(false);

fn track(owner: usize, at: usize, end: usize) -> Arc<Ticket> {
    let ticket = Arc::new(Ticket {
        owner,
        at,
        end,
        valid: AtomicBool::new(true),
    });
    let mut tickets = TICKETS.lock().unwrap_or_else(|e| e.into_inner());
    tickets.retain(|weak| weak.strong_count() != 0);
    tickets.push(Arc::downgrade(&ticket));
    TRACKING.store(true, Ordering::Relaxed);
    ticket
}

fn invalidate(matches: impl Fn(&Ticket) -> bool) {
    if !TRACKING.load(Ordering::Relaxed) {
        return;
    }
    TICKETS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .retain(|weak| {
            let Some(ticket) = weak.upgrade() else {
                return false;
            };
            if matches(&ticket) {
                ticket.valid.store(false, Ordering::Release);
            }
            true
        });
}

pub(crate) fn released(owner: usize) {
    invalidate(|ticket| ticket.owner == owner);
}

pub(crate) fn replaced(at: usize) {
    invalidate(|ticket| ticket.at == at);
}

pub(crate) fn remapped(at: usize, len: usize) {
    let end = at.saturating_add(len);
    invalidate(|ticket| at < ticket.end && ticket.at < end);
}

pub(crate) struct Target {
    pub(crate) copy: Arc<crate::resource::HostTexture>,
    layout: xtp_formats::texture::Layout,
    ticket: Arc<Ticket>,
}

impl Target {
    pub(crate) fn new(resource: This) -> Option<Self> {
        // Keep the host copy alive independently of the title resource lifetime.
        let copy = crate::resource::host_copy(resource)?;
        let (layout, at, size) = crate::resource::title_texture(resource)?;
        let end = at.checked_add(size).filter(|&end| at != 0 && end > at)?;
        let ticket = track(resource as usize, at, end);
        Some(Self {
            copy,
            layout,
            ticket,
        })
    }

    pub(crate) fn owner(&self) -> usize {
        self.ticket.owner
    }
}

/// The readback buffer and console layout for a texture copied at list close.
pub(crate) struct Back {
    buffer: ID3D12Resource,
    footprints: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
    rows: Vec<u32>,
    mips: u32,
    layout: xtp_formats::texture::Layout,
    ticket: Arc<Ticket>,
}

impl Back {
    pub(crate) fn range(&self) -> (usize, usize) {
        (self.ticket.at, self.ticket.end)
    }
}

/// Records copies of every subresource of `target` (already in `COPY_SOURCE`)
/// into a readback buffer.
pub(crate) fn record(list: &ID3D12GraphicsCommandList, target: Target) -> Option<Back> {
    if !target.ticket.valid.load(Ordering::Acquire) {
        return None;
    }
    let host = crate::host::host()?;
    let texture = &target.copy.texture;
    // SAFETY: a live host texture held by `target.copy`.
    let desc = unsafe { texture.GetDesc() };
    let mips = u32::from(desc.MipLevels.max(1));
    let subresources = mips * u32::from(desc.DepthOrArraySize.max(1));
    let n = subresources as usize;
    let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); n];
    let mut rows = vec![0u32; n];
    let mut total = 0u64;
    // SAFETY: out arrays hold `n == subresources` entries.
    unsafe {
        host.device.GetCopyableFootprints(
            &desc,
            0,
            subresources,
            0,
            Some(footprints.as_mut_ptr()),
            Some(rows.as_mut_ptr()),
            None,
            Some(&mut total),
        );
    }
    // The last footprint's required size can omit row-end padding, while
    // `Layout::tile` reads pitched rows; make sure the buffer covers them.
    for (footprint, &rows) in footprints.iter().zip(&rows) {
        let format = footprint.Footprint;
        let pitched = u64::from(format.RowPitch)
            .checked_mul(u64::from(rows))?
            .checked_mul(u64::from(format.Depth))?;
        total = total.max(footprint.Offset.checked_add(pitched)?);
    }
    let buffer = host.read_back_buffer(total)?;
    for (i, footprint) in footprints.iter().enumerate() {
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(&buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: *footprint,
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: i as u32,
            },
        };
        // SAFETY: an open list; the texture is in `COPY_SOURCE` and `buffer` in `COPY_DEST`.
        unsafe { list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
    }
    Some(Back {
        buffer,
        footprints,
        rows,
        mips,
        layout: target.layout,
        ticket: target.ticket,
    })
}

/// Once the submission has run, tiles the copied texels back into the same
/// title allocation. Returns the number of bytes written.
pub(crate) fn land(back: &Back) -> usize {
    static COPIED: AtomicU64 = AtomicU64::new(0);
    static STALE: AtomicU64 = AtomicU64::new(0);
    let _ticket_guard = TICKETS.lock().unwrap_or_else(|e| e.into_inner());
    if !back.ticket.valid.load(Ordering::Acquire) {
        let skipped = STALE.fetch_add(1, Ordering::Relaxed) + 1;
        if skipped.is_power_of_two() {
            tracing::info!(
                skipped,
                "skipping texture readbacks whose title allocation changed"
            );
        }
        return 0;
    }
    let (at, end) = back.range();
    let size = end - at;
    let mut tiled = vec![0u8; size];
    // Preserve title bytes that the subresources do not cover (padding/tail).
    if !crate::host::read_into(at as *const u8, &mut tiled) {
        return 0;
    }
    let mut data = std::ptr::null_mut();
    // SAFETY: our readback buffer, written by the completed submission.
    unsafe {
        if back.buffer.Map(0, None, Some(&mut data)).is_err() || data.is_null() {
            return 0;
        }
    }
    for (i, (footprint, &rows)) in back.footprints.iter().zip(&back.rows).enumerate() {
        let (mip, slice) = (i as u32 % back.mips, i as u32 / back.mips);
        let f = footprint.Footprint;
        let pitch = f.RowPitch as usize;
        let len = pitch * rows as usize * f.Depth as usize;
        // SAFETY: subresource slice within the mapped readback buffer bounds
        // checked when computing `total` in `record`.
        let linear = unsafe {
            std::slice::from_raw_parts(data.cast::<u8>().add(footprint.Offset as usize), len)
        };
        if let Err(e) =
            back.layout
                .tile(&mut tiled, mip, slice, linear, pitch, pitch * rows as usize)
        {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                tracing::warn!(?e, mip, slice, "a texture copied back could not be tiled")
            });
            // SAFETY: mapped above.
            unsafe { back.buffer.Unmap(0, None) };
            return 0;
        }
    }
    // SAFETY: mapped above.
    unsafe { back.buffer.Unmap(0, None) };
    crate::host::copy_mapped(at as *mut u8, tiled.as_ptr(), size, true);
    let copied = COPIED.fetch_add(1, Ordering::Relaxed) + 1;
    if copied == 1 || copied.is_multiple_of(128) {
        tracing::info!(copied, bytes = size, "texture readback landed");
    }
    size
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_readbacks_for_changed_allocations_are_invalidated() {
        let first = track(91, 0x1000, 0x2000);
        let next = track(92, 0x2000, 0x3000);
        let far = track(93, 0x4000, 0x5000);

        remapped(0x0f00, 0x200);
        assert!(!first.valid.load(Ordering::Acquire));
        assert!(next.valid.load(Ordering::Acquire));
        replaced(0x2000);
        assert!(!next.valid.load(Ordering::Acquire));
        assert!(far.valid.load(Ordering::Acquire));

        released(93);
        assert!(!far.valid.load(Ordering::Acquire));
        assert!(track(94, 0x2000, 0x3000).valid.load(Ordering::Acquire));
    }

    #[test]
    fn parses_texture_back_kinds() {
        let defaults = [(256, 256, 41), (64, 64, 16)];
        assert!(parse_kinds("0", &defaults).is_empty());
        assert!(parse_kinds("off", &defaults).is_empty());
        assert!(parse_kinds("  ", &defaults).is_empty());
        assert_eq!(parse_kinds("1", &defaults), defaults);
        assert_eq!(
            parse_kinds("16x16f41, 128x128f36, bad", &defaults),
            vec![(16, 16, 41), (128, 128, 36)]
        );
    }

    #[test]
    fn compute_uav_texture_lands_tiled_in_title_memory() {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_FLOAT;
        use windows::core::s;

        use crate::object::{HRESULT, S_OK, guid, slot};
        use crate::testgpu::TitleList;

        let test = "texback::tests::compute_uav_texture_lands_tiled_in_title_memory";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };

        struct ResetKinds;
        impl Drop for ResetKinds {
            fn drop(&mut self) {
                TEST_KINDS.with_borrow_mut(|over| *over = None);
            }
        }
        const W: u32 = 16;
        const H: u32 = 16;
        const R32_FLOAT: u32 = 41;
        TEST_KINDS.with_borrow_mut(|over| *over = Some(vec![(u64::from(W), H, R32_FLOAT)]));
        let _reset = ResetKinds;

        #[repr(C)]
        struct TexDesc {
            dimension: u32,
            alignment: u64,
            width: u64,
            height: u32,
            depth_or_array_size: u16,
            mip_levels: u16,
            format: u32,
            sample_count: u32,
            sample_quality: u32,
            layout: u32,
            flags: u32,
        }
        const DEFAULT_HEAP: [u32; 5] = [1, 0, 0, 1, 1];
        const TEXTURE_2D: u32 = 3;
        const ALLOW_UAV: u32 = 0x4;
        const UNORDERED_ACCESS: u32 = 0x8;
        let desc = TexDesc {
            dimension: TEXTURE_2D,
            alignment: 0,
            width: u64::from(W),
            height: H,
            depth_or_array_size: 1,
            mip_levels: 1,
            format: R32_FLOAT,
            sample_count: 1,
            sample_quality: 0,
            layout: 0,
            flags: ALLOW_UAV,
        };
        let resource_iid = guid(
            0x6964_42be,
            0xa72e,
            0x4059,
            [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
        );
        let mut resource: This = std::ptr::null();
        let hr: HRESULT = crate::resource::create_committed(
            std::ptr::null(),
            DEFAULT_HEAP.as_ptr().cast(),
            0,
            (&raw const desc).cast(),
            0,
            std::ptr::null(),
            &resource_iid,
            &mut resource,
        );
        assert_eq!(hr, S_OK);
        assert!(!resource.is_null());

        struct Release(This);
        impl Drop for Release {
            fn drop(&mut self) {
                slot::<extern "system" fn(This) -> u32>(self.0, 0x10)(self.0);
            }
        }
        let _release = Release(resource);

        let host_tex = crate::resource::host_resource(resource).expect("host texture");
        assert!(
            !crate::texture::title_written(host_tex),
            "UAV textures are not re-uploaded from title memory"
        );

        let source = r#"
#define RS "DescriptorTable(UAV(u0))"
RWTexture2D<float> dst : register(u0);
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    dst[id.xy] = float(id.y * 16u + id.x + 1u);
}
"#;
        let mut code: Option<ID3DBlob> = None;
        // SAFETY: source string and entry names live across D3DCompile.
        unsafe {
            D3DCompile(
                source.as_ptr().cast(),
                source.len(),
                s!("texback_test"),
                None,
                None,
                s!("main"),
                s!("cs_5_1"),
                0,
                0,
                &mut code,
                None,
            )
        }
        .expect("compute shader compiles");
        let code = code.expect("compiled bytecode");
        // SAFETY: compiled blob owns its buffer until dropped.
        let bytes = unsafe {
            std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize())
        };
        // SAFETY: live device and valid shader bytecode containing embedded root signature.
        let (root, pso, desc_heap) = unsafe {
            let root: ID3D12RootSignature = gpu.device.CreateRootSignature(0, bytes).expect("root");
            let pso: ID3D12PipelineState = gpu
                .device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: bytes.as_ptr().cast(),
                        BytecodeLength: bytes.len(),
                    },
                    ..Default::default()
                })
                .expect("pso");
            let desc_heap: ID3D12DescriptorHeap = gpu
                .device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: 1,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                })
                .expect("uav heap");
            gpu.device.CreateUnorderedAccessView(
                host_tex,
                None,
                Some(&D3D12_UNORDERED_ACCESS_VIEW_DESC {
                    Format: DXGI_FORMAT_R32_FLOAT,
                    ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
                    Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                        Texture2D: D3D12_TEX2D_UAV {
                            MipSlice: 0,
                            PlaneSlice: 0,
                        },
                    },
                }),
                desc_heap.GetCPUDescriptorHandleForHeapStart(),
            );
            (root, pso, desc_heap)
        };

        let list = TitleList::new();
        let hl = list.host();
        hl.texture_written(resource);
        crate::list::ensure_state(hl, resource, UNORDERED_ACCESS);
        // SAFETY: open host command list with live pipeline, root signature, and descriptor heap.
        unsafe {
            hl.list.SetDescriptorHeaps(&[Some(desc_heap.clone())]);
            hl.list.SetComputeRootSignature(&root);
            hl.list.SetPipelineState(&pso);
            hl.list
                .SetComputeRootDescriptorTable(0, desc_heap.GetGPUDescriptorHandleForHeapStart());
            hl.list.Dispatch(W / 8, H / 8, 1);
        }
        list.run();

        let (layout, at, size) =
            crate::resource::title_texture(resource).expect("title texture layout");
        // SAFETY: title resource is still alive and committed for `size` bytes.
        let tiled = unsafe { std::slice::from_raw_parts(at as *const u8, size) };
        let pitch = (W * 4) as usize;
        let depth_pitch = pitch * H as usize;
        let mut linear = vec![0u8; depth_pitch];
        layout
            .detile(tiled, 0, 0, &mut linear, pitch, depth_pitch)
            .expect("detile landed texture");

        let mut expected = Vec::with_capacity((W * H) as usize);
        for y in 0..H {
            for x in 0..W {
                expected.push((y * W + x + 1) as f32);
            }
        }
        let got: Vec<f32> = linear
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&c| f32::from_le_bytes(c))
            .collect();
        assert_eq!(got, expected);
        // The default 2D swizzle (`SW_4KB_S`) is tiled, not row-major linear.
        assert_ne!(&tiled[..linear.len()], linear.as_slice());
    }
}
