//! End-to-end tests of typed-format normalized buffer loads
//! (`tbuffer_load_format_*`) lowered to DXIL and executed on WARP / host GPU.
//!
//! Verifies that UNORM and SNORM loads produce the exact nearest `f32` to
//! `code / max` (with SNORM clamped at `-1.0` and endpoints pinned to exact
//! `1.0` / `-1.0`).

use super::host::{Dispatch, Host, Root, Stride, View};
use super::valu_tests::ONE_AT_A_TIME;
use super::{HostShader, Target, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};

const THREADS: u32 = 256;
const FMT_8_UNORM: u8 = 1;
const FMT_8_SNORM: u8 = 2;
const FMT_16_UNORM: u8 = 7;
const FMT_16_SNORM: u8 = 8;
const FMT_10_11_11_UNORM: u8 = 30;
const FMT_10_10_10_2_SNORM: u8 = 45;

/// VOP2 `v_lshlrev_b32 vdst, shift, vsrc1` (`shift` in 0..=64).
const fn v_lshlrev_b32(vdst: u8, shift: u32, vsrc1: u8) -> u32 {
    0x1A << 25 | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | (128 + shift)
}

/// MTBUF `tbuffer_load_format_{x,xy,xyz,xyzw} vdata, vaddr, s[8:11], 0 format:fmt offen`.
const fn tbuffer_load(channels: u32, fmt: u8, vdata: u8, vaddr: u8) -> [u32; 2] {
    [
        0x3A << 26 | (fmt as u32) << 19 | ((channels - 1) & 7) << 16 | 1 << 12,
        0x80 << 24 | 2 << 16 | (vdata as u32) << 8 | vaddr as u32,
    ]
}

/// MUBUF `buffer_store_dword{,x2,x3,x4} vdata, vaddr, s[12:15], 0 offen`.
const fn buffer_store(channels: u32, vdata: u8, vaddr: u8) -> [u32; 2] {
    let op = [0x1C, 0x1D, 0x1F, 0x1E][channels as usize - 1];
    [
        0x38 << 26 | op << 18 | 1 << 12,
        0x80 << 24 | 3 << 16 | (vdata as u32) << 8 | vaddr as u32,
    ]
}

/// Builds, lowers and validates a 256-thread compute shader that loads each
/// thread's element (`1 << elem_shift` bytes) with `tbuffer_load_format_*`
/// from slot 0 and writes `channels` dwords at `1 << out_shift` stride to slot 1.
fn build_cs(fmt: u8, channels: u32, elem_shift: u32, out_shift: u32) -> HostShader {
    let mut words = vec![
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], null
        0xF408_0302,
        0xFA00_0020, // s_load_dwordx4 s[12:15], s[4:5], 0x20
    ];
    let addr = if elem_shift == 0 {
        0
    } else {
        words.push(v_lshlrev_b32(1, elem_shift, 0));
        1
    };
    words.extend_from_slice(&tbuffer_load(channels, fmt, 2, addr));
    words.push(v_lshlrev_b32(6, out_shift, 0));
    words.extend_from_slice(&buffer_store(channels, 2, 6));
    words.push(0xBF81_0000); // s_endpgm

    let entry = |kind, table_offset_dwords| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords,
        lower_bound: table_offset_dwords / 8,
        stride: 0,
        dwords: 4,
        user_data: 4,
        slot: 1,
    };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        float_mode: 0,
        vgprs: 8,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 4,
            slot: 1,
        }],
        used: vec![entry(ResourceType::Srv, 0), entry(ResourceType::Uav, 8)],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1,
            threads: [THREADS, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let mut shader = super::tests::emit_for(&native, Target::ANY).expect("emit_for");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil).expect("DXIL validation");
    }
    shader
}

fn run(shader: &HostShader, input: &[u32], out_dwords: u32, check: impl Fn(&str, &[u32])) {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for (label, host) in Host::test_hosts() {
        let (root, pso) = host.compute_pipeline(shader).expect("compute_pipeline");
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let views = [
            View::Buffer {
                data: input,
                stride: Stride::Raw,
            },
            View::RawUav(out_dwords),
        ];
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &views,
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .expect("dispatch");
        check(label, &out[0]);
    }
}

fn unorm_bits(code: u32, bits: u32) -> u32 {
    ((code as f64 / ((1u64 << bits) - 1) as f64) as f32).to_bits()
}

fn snorm_bits(code: i32, bits: u32) -> u32 {
    let max = ((1u64 << (bits - 1)) - 1) as f64;
    ((code as f64 / max).max(-1.0) as f32).to_bits()
}

fn all_bytes_packed() -> Vec<u32> {
    (0..64u32)
        .map(|i| {
            let b = 4 * i;
            b | ((b + 1) << 8) | ((b + 2) << 16) | ((b + 3) << 24)
        })
        .collect()
}

/// 256 16-bit codes: endpoints, +/-1 boundaries, powers of two, and a spread.
fn sample_codes_16() -> Vec<u16> {
    let mut codes: Vec<u16> = vec![
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 511, 512,
        1023, 1024, 2047, 2048, 4095, 4096, 8191, 8192, 16383, 16384, 32764, 32765, 32766, 32767,
        32768, 32769, 32770, 32771, 49151, 49152, 65530, 65531, 65532, 65533, 65534, 65535,
    ];
    let mut step = 137u16;
    while codes.len() < THREADS as usize {
        if !codes.contains(&step) {
            codes.push(step);
        }
        step = step.wrapping_add(257);
    }
    codes
}

#[test]
fn unorm8_and_snorm8_load_all_codes_to_nearest_float() {
    let input = all_bytes_packed();
    run(
        &build_cs(FMT_8_UNORM, 1, 0, 2),
        &input,
        THREADS,
        |l, out| {
            for c in 0..256u32 {
                let got = out[c as usize];
                assert_eq!(got, unorm_bits(c, 8), "{l}: 8_UNORM code {c}");
                assert_eq!((f32::from_bits(got) * 255.0) as u32, c, "{l}: trunc {c}");
            }
        },
    );
    run(
        &build_cs(FMT_8_SNORM, 1, 0, 2),
        &input,
        THREADS,
        |l, out| {
            for raw in 0..256u32 {
                let c = i32::from(raw as i8);
                assert_eq!(out[raw as usize], snorm_bits(c, 8), "{l}: 8_SNORM {c}");
            }
        },
    );
}

#[test]
fn unorm16_and_snorm16_load_endpoints_and_spread_to_nearest_float() {
    let codes = sample_codes_16();
    let input: Vec<u32> = codes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&[lo, hi]| u32::from(lo) | (u32::from(hi) << 16))
        .collect();
    let cs_u16 = build_cs(FMT_16_UNORM, 1, 1, 2);
    run(&cs_u16, &input, THREADS, |l, out| {
        for (i, &c) in codes.iter().enumerate() {
            assert_eq!(out[i], unorm_bits(u32::from(c), 16), "{l}: 16_UNORM {c}");
        }
    });
    let cs_s16 = build_cs(FMT_16_SNORM, 1, 1, 2);
    run(&cs_s16, &input, THREADS, |l, out| {
        for (i, &raw) in codes.iter().enumerate() {
            let c = i32::from(raw as i16);
            assert_eq!(out[i], snorm_bits(c, 16), "{l}: 16_SNORM {c}");
        }
    });
}

/// Covers the integer-only `exact_norm` path (11-bit width in `10_11_11_UNORM`)
/// and 2-bit / 10-bit SNORM channels (`10_10_10_2_SNORM`).
#[test]
fn packed_unorm_and_snorm_formats_load_exact_nearest_floats() {
    let input11: Vec<u32> = (0..THREADS)
        .map(|i| {
            let x = [0, 1, 2, 7, 8, 1023, 1024, 2046, 2047]
                .get(i as usize)
                .copied()
                .unwrap_or((i * 19) & 0x7FF);
            let y = (2047 - x) & 0x7FF;
            let z = [0, 1, 1022, 1023]
                .get(i as usize)
                .copied()
                .unwrap_or((i * 13) & 0x3FF);
            x | (y << 11) | (z << 22)
        })
        .collect();
    let cs11 = build_cs(FMT_10_11_11_UNORM, 3, 2, 4);
    run(&cs11, &input11, THREADS * 4, |l, out| {
        for (i, &w) in input11.iter().enumerate() {
            let b = i * 4;
            assert_eq!(out[b], unorm_bits(w & 0x7FF, 11), "{l}: 11b ch0 #{i}");
            assert_eq!(
                out[b + 1],
                unorm_bits((w >> 11) & 0x7FF, 11),
                "{l}: 11b ch1 #{i}"
            );
            assert_eq!(
                out[b + 2],
                unorm_bits((w >> 22) & 0x3FF, 10),
                "{l}: 10b ch2 #{i}"
            );
        }
    });

    let sext = |v: u32, bits: u32| ((v << (32 - bits)) as i32) >> (32 - bits);
    let input2: Vec<u32> = (0..THREADS)
        .map(|i| {
            let c0 = i & 3;
            let c1 = [0, 1, 511, 512, 513, 1023]
                .get(i as usize)
                .copied()
                .unwrap_or((i * 17) & 0x3FF);
            c0 | (c1 << 2) | (((1023 - c1) & 0x3FF) << 12) | (((i * 29) & 0x3FF) << 22)
        })
        .collect();
    let cs2 = build_cs(FMT_10_10_10_2_SNORM, 4, 2, 4);
    run(&cs2, &input2, THREADS * 4, |l, out| {
        for (i, &w) in input2.iter().enumerate() {
            let b = i * 4;
            assert_eq!(out[b], snorm_bits(sext(w & 3, 2), 2), "{l}: 2b ch0 #{i}");
            assert_eq!(
                out[b + 1],
                snorm_bits(sext((w >> 2) & 0x3FF, 10), 10),
                "{l}: 10b ch1 #{i}"
            );
            assert_eq!(
                out[b + 2],
                snorm_bits(sext((w >> 12) & 0x3FF, 10), 10),
                "{l}: 10b ch2 #{i}"
            );
            assert_eq!(
                out[b + 3],
                snorm_bits(sext((w >> 22) & 0x3FF, 10), 10),
                "{l}: 10b ch3 #{i}"
            );
        }
    });
}
