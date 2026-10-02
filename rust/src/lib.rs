// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Garry Boyer
//! Optional accelerator for fluxcode's encoder: `compress_unit` builds a unit's bytes from its rows
//! (layout choice, body packing, flush points and the zstd frame) in one call without the GIL.
//!
//! python-zstandard holds the GIL inside `flush(FLUSH_BLOCK)`, which is where the compression of the
//! buffered data happens, so flushed frames don't scale with threads there. This is a port of
//! `_unit.compress` and what it calls (`_bitpacking.write_unit`, `wide_share`, `_format.flush_points`,
//! `_unit.compress_body`) for units without a time axis; the unit bytes are the same as the Python path's
//! while both link the same libzstd (see `zstd_version`).

use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use zstd_safe::zstd_sys::ZSTD_EndDirective::{ZSTD_e_end, ZSTD_e_flush};
use zstd_safe::{CCtx, CParameter, InBuffer, OutBuffer};

const BLOCK_FLAG_NONFINITE: u8 = 0x08;
// BYTES_PER_FLAGS + BYTES_PER_SIZE + BYTES_PER_PARAM + BYTES_PER_ANCHOR
const METADATA_BYTES_PER_BLOCK: usize = 1 + 2 + 2 + 8;
const HEADER_BYTES: usize = 8;
const FORMAT_VERSION: u8 = 1;
const UNIT_FLAG_BYTE_PLANES: u8 = 1;
const FLUSH_MIN_DENSITY: usize = 16;
const BYTE_PLANES_BIT: u32 = 7;
const BYTE_PLANES_MAX_SHARE: f64 = 0.01;

/// A unit's rows and the offsets of each block in the samples, residual groups and code groups.
struct Rows<'a> {
    n: usize,
    ng: usize,
    flags: &'a [u8],
    sizes: &'a [i64],
    params: &'a [i64],
    anchors: &'a [i64],
    residuals: &'a [i16],
    codes: &'a [u8],
    so: Vec<usize>,
    go: Vec<usize>,
    co: Vec<usize>,
}

impl<'a> Rows<'a> {
    /// The rows with their offsets (`_format._fill_layout`), or an error if the fields disagree.
    fn new(
        flags: &'a [u8], sizes: &'a [i64], params: &'a [i64], anchors: &'a [i64], residuals: &'a [i16],
        codes: &'a [u8],
    ) -> Result<Self, String> {
        let n = flags.len();
        let mismatch = || "rows don't match the layout".to_string();
        if n > u16::MAX as usize || sizes.len() != n || params.len() != n || anchors.len() != n {
            return Err(mismatch());
        }
        let (mut so, mut go, mut co) = (vec![0usize; n + 1], vec![0usize; n + 1], vec![0usize; n + 1]);
        for b in 0..n {
            if !(0..=u16::MAX as i64).contains(&sizes[b]) {
                return Err(mismatch());
            }
            let size = sizes[b] as usize;
            let groups = size.div_ceil(8);
            so[b + 1] = so[b] + size;
            go[b + 1] = go[b] + groups;
            co[b + 1] = co[b] + if flags[b] & BLOCK_FLAG_NONFINITE != 0 { groups } else { 0 };
        }
        if residuals.len() != so[n] || codes.len() != so[n] {
            return Err(mismatch());
        }
        Ok(Rows { n, ng: go[n], flags, sizes, params, anchors, residuals, codes, so, go, co })
    }
}

#[inline(always)]
fn zigzag(r: i16) -> u16 {
    ((r << 1) ^ (r >> 15)) as u16
}

#[inline(always)]
fn transpose8(mut x: u64) -> u64 {
    let mut d = (x ^ (x >> 7)) & 0x00AA00AA00AA00AA;
    x = x ^ d ^ (d << 7);
    d = (x ^ (x >> 14)) & 0x0000CCCC0000CCCC;
    x = x ^ d ^ (d << 14);
    d = (x ^ (x >> 28)) & 0x00000000F0F0F0F0;
    x ^ d ^ (d << 28)
}

/// The uncompressed body of a unit without a time axis, in the byte-plane layout: the metadata columns,
/// the zigzagged residuals as a low and a high byte plane (each block padded with zeros to whole groups
/// of 8), then the non-finite code planes.
fn write_body(r: &Rows) -> Vec<u8> {
    let (n, ng, ncg) = (r.n, r.ng, r.co[r.n]);
    let res_start = METADATA_BYTES_PER_BLOCK * n;
    let mut raw = vec![0u8; res_start + 16 * ng + 2 * ncg];
    let (params, anchors) = ((1 + 2) * n, (1 + 2 + 2) * n);
    for b in 0..n {
        raw[b] = r.flags[b];
        raw[n + b] = r.sizes[b] as u8;
        raw[2 * n + b] = (r.sizes[b] >> 8) as u8;
        raw[params + b] = r.params[b] as u8;
        raw[params + n + b] = (r.params[b] >> 8) as u8;
        for k in 0..8 {
            raw[anchors + k * n + b] = (r.anchors[b] >> (8 * k)) as u8;
        }
    }
    let (res, codes_raw) = raw[res_start..].split_at_mut(16 * ng);
    let (low, high) = res.split_at_mut(8 * ng);
    let (p0, p1) = codes_raw.split_at_mut(ncg);
    for b in 0..n {
        let (s0, len) = (r.so[b], r.sizes[b] as usize);
        let s = 8 * r.go[b];
        let (low, high) = (&mut low[s..s + len], &mut high[s..s + len]);
        for ((&v, l), h) in r.residuals[s0..s0 + len].iter().zip(low).zip(high) {
            let z = zigzag(v);
            *l = z as u8;
            *h = (z >> 8) as u8;
        }
        if r.flags[b] & BLOCK_FLAG_NONFINITE != 0 {
            let c0 = r.co[b];
            let block = &r.codes[s0..s0 + len];
            let full = block.chunks_exact(8);
            let tail = full.remainder();
            let mut g = 0;
            for chunk in full {
                // byte 0 of the transpose holds every code's low bit, byte 1 the high bit
                let t = transpose8(u64::from_le_bytes(chunk.try_into().unwrap()));
                p0[c0 + g] = t as u8;
                p1[c0 + g] = (t >> 8) as u8;
                g += 1;
            }
            if !tail.is_empty() {
                let mut word = [0u8; 8];
                word[..tail.len()].copy_from_slice(tail);
                let t = transpose8(u64::from_le_bytes(word));
                p0[c0 + g] = t as u8;
                p1[c0 + g] = (t >> 8) as u8;
            }
        }
    }
    raw
}

/// The bit-plane layout of a byte-plane body: its residual region with every group of 8 bytes transposed
/// into one byte of each of the 16 bit planes. The rest of the body is the same.
fn to_bit_planes(byte_body: &[u8], n: usize, ng: usize) -> Vec<u8> {
    let start = METADATA_BYTES_PER_BLOCK * n;
    let end = start + 16 * ng;
    let mut raw = Vec::with_capacity(byte_body.len());
    raw.extend_from_slice(&byte_body[..start]);
    raw.resize(end, 0);
    raw.extend_from_slice(&byte_body[end..]);
    let (low, high) = byte_body[start..end].split_at(8 * ng);
    let planes = &mut raw[start..end];
    let word = |bytes: &[u8]| u64::from_le_bytes(bytes.try_into().unwrap());
    // 8 groups at a time: bit-transpose each group's word, then transpose the 8x8 bytes of the 8 results
    // so that each plane takes the 8 bytes of its 8 groups in one store
    let batches = ng / 8;
    for (i, (lw, hw)) in low.chunks_exact(64).zip(high.chunks_exact(64)).enumerate() {
        let g = 8 * i;
        let mut lo: [u64; 8] = std::array::from_fn(|j| transpose8(word(&lw[8 * j..8 * j + 8])));
        let mut hi: [u64; 8] = std::array::from_fn(|j| transpose8(word(&hw[8 * j..8 * j + 8])));
        transpose_bytes(&mut lo);
        transpose_bytes(&mut hi);
        for k in 0..8 {
            planes[k * ng + g..k * ng + g + 8].copy_from_slice(&lo[k].to_le_bytes());
            planes[(8 + k) * ng + g..(8 + k) * ng + g + 8].copy_from_slice(&hi[k].to_le_bytes());
        }
    }
    for g in 8 * batches..ng {
        let lo = transpose8(word(&low[8 * g..8 * g + 8])).to_le_bytes();
        let hi = transpose8(word(&high[8 * g..8 * g + 8])).to_le_bytes();
        for k in 0..8 {
            planes[k * ng + g] = lo[k];
            planes[(8 + k) * ng + g] = hi[k];
        }
    }
    raw
}

/// Transposes the 8x8 matrix of bytes whose rows are the words (byte j of word k becomes byte k of word j).
#[inline(always)]
fn transpose_bytes(w: &mut [u64; 8]) {
    #[inline(always)]
    fn swap(w: &mut [u64; 8], i: usize, j: usize, shift: u32, mask: u64) {
        let t = ((w[i] >> shift) ^ w[j]) & mask;
        w[j] ^= t;
        w[i] ^= t << shift;
    }
    const M8: u64 = 0x00FF00FF00FF00FF;
    const M16: u64 = 0x0000FFFF0000FFFF;
    const M32: u64 = 0x00000000FFFFFFFF;
    swap(w, 0, 1, 8, M8);
    swap(w, 2, 3, 8, M8);
    swap(w, 4, 5, 8, M8);
    swap(w, 6, 7, 8, M8);
    swap(w, 0, 2, 16, M16);
    swap(w, 1, 3, 16, M16);
    swap(w, 4, 6, 16, M16);
    swap(w, 5, 7, 16, M16);
    swap(w, 0, 4, 32, M32);
    swap(w, 1, 5, 32, M32);
    swap(w, 2, 6, 32, M32);
    swap(w, 3, 7, 32, M32);
}

/// Share of residual slots whose zigzagged residual reaches 2^BYTE_PLANES_BIT, from a byte-plane body.
fn wide_share_byte_planes(raw: &[u8], n: usize, ng: usize) -> f64 {
    if ng == 0 {
        return 0.0;
    }
    let start = METADATA_BYTES_PER_BLOCK * n;
    let (low, high) = raw[start..start + 16 * ng].split_at(8 * ng);
    let low_mask = ((0xFFu32 << BYTE_PLANES_BIT) & 0xFF) as u8;
    let count: usize = low.iter().zip(high).filter(|&(&l, &h)| (l & low_mask) | h != 0).count();
    count as f64 / (8 * ng) as f64
}

/// Where zstd blocks end: after the metadata and after each residual plane that is dense enough.
fn flush_points(raw: &[u8], n: usize, ng: usize, byte_planes: bool) -> Vec<usize> {
    if ng == 0 {
        return vec![];
    }
    let start = METADATA_BYTES_PER_BLOCK * n;
    let (plane_bytes, planes) = if byte_planes { (8 * ng, 2) } else { (ng, 16) };
    let mut points = vec![start];
    for p in 0..planes {
        let ps = start + p * plane_bytes;
        // counted in chunks of 4096 so that each inner sum fits u32 lanes, which the compiler vectorizes
        let nz = raw[ps..ps + plane_bytes]
            .chunks(4096)
            .map(|c| c.iter().map(|&v| (v != 0) as u32).sum::<u32>() as usize)
            .sum::<usize>();
        if nz * FLUSH_MIN_DENSITY > plane_bytes {
            points.push(ps + plane_bytes);
        }
    }
    points
}

thread_local! {
    /// One zstd context per thread, reused across frames like python-zstandard's compressor
    /// (creating one allocates its workspace).
    static CCTX: std::cell::RefCell<CCtx<'static>> = std::cell::RefCell::new(CCtx::create());
}

/// `header` followed by the zstd frame of `body`, ended after each of the `cuts` (with none, a single
/// block-ending call, which is what python-zstandard's `compress` does). With a `limit`, gives up (None)
/// as soon as the blocks written so far are longer than it: the frame can only grow.
fn compress_with_header(
    header: &[u8], body: &[u8], cuts: &[usize], level: i32, limit: Option<usize>,
) -> Result<Option<Vec<u8>>, String> {
    CCTX.with(|c| {
        let cctx = &mut *c.borrow_mut();
        let e = |c: usize| zstd_safe::get_error_name(c).to_string();
        cctx.reset(zstd_safe::ResetDirective::SessionOnly).map_err(e)?;
        cctx.set_parameter(CParameter::CompressionLevel(level)).map_err(e)?;
        cctx.set_parameter(CParameter::ChecksumFlag(false)).map_err(e)?;
        cctx.set_parameter(CParameter::ContentSizeFlag(true)).map_err(e)?;
        cctx.set_pledged_src_size(Some(body.len() as u64)).map_err(e)?;
        let mut out: Vec<u8> = Vec::with_capacity(header.len() + zstd_safe::compress_bound(body.len()) + 64 * (cuts.len() + 2));
        out.extend_from_slice(header);
        let mut run = |input: &[u8], op: zstd_safe::zstd_sys::ZSTD_EndDirective| -> Result<usize, String> {
            let mut inb = InBuffer::around(input);
            loop {
                // the capacity above covers the worst case, so this only guards against a stalled loop
                if out.len() == out.capacity() {
                    out.reserve(1 << 16);
                }
                let pos = out.len();
                let remaining = cctx.compress_stream2(&mut OutBuffer::around_pos(&mut out, pos), &mut inb, op).map_err(e)?;
                if inb.pos == input.len() && remaining == 0 {
                    return Ok(out.len());
                }
            }
        };
        let mut prev = 0;
        for &cut in cuts {
            if prev < cut && cut < body.len() {
                let len = run(&body[prev..cut], ZSTD_e_flush)?;
                prev = cut;
                if limit.is_some_and(|l| len > l) {
                    return Ok(None);
                }
            }
        }
        run(&body[prev..], ZSTD_e_end)?;
        Ok(Some(out))
    })
}

/// A unit as its header and the smallest frame of the candidate bodies (byte planes or not) over the zstd
/// levels. Ties go to byte planes, then to the earlier level, whatever the order the candidates are tried
/// in; a candidate that can't beat the best so far is dropped as soon as its flushed blocks show it.
fn build(r: &Rows, candidates: &[(bool, &[u8])], flush: bool, levels: &[i32]) -> Result<Vec<u8>, String> {
    let (n, ng) = (r.n, r.ng);
    let mut best: Option<(Vec<u8>, usize)> = None;
    for &(byte_planes, body) in candidates {
        let mut header = [0u8; HEADER_BYTES];
        header[0] = FORMAT_VERSION;
        header[1] = if byte_planes { UNIT_FLAG_BYTE_PLANES } else { 0 };
        header[2..4].copy_from_slice(&(n as u16).to_le_bytes());
        header[4..8].copy_from_slice(&(r.so[n] as u32).to_le_bytes());
        let cuts = if flush { flush_points(body, n, ng, byte_planes) } else { vec![] };
        for (i, &level) in levels.iter().enumerate() {
            let rank = if byte_planes { i } else { levels.len() + i };
            // a frame of this length or less wins (a tie only against a worse rank)
            let limit = best.as_ref().map(|(b, best_rank)| if rank < *best_rank { b.len() } else { b.len() - 1 });
            if let Some(unit) = compress_with_header(&header, body, &cuts, level, limit)? {
                if limit.map_or(true, |l| unit.len() <= l) {
                    best = Some((unit, rank));
                }
            }
        }
    }
    best.map(|(unit, _)| unit).ok_or_else(|| "no zstd levels".to_string())
}

fn compress_unit_impl(r: &Rows, layout: &str, flush: bool, levels: &[i32]) -> Result<Vec<u8>, String> {
    let (n, ng) = (r.n, r.ng);
    // the byte-plane body is the cheaper one to write; the bit-plane body is derived from it
    let byte_body = write_body(r);
    let byte_planes_first = wide_share_byte_planes(&byte_body, n, ng) < BYTE_PLANES_MAX_SHARE;
    match layout {
        "byte" => build(r, &[(true, &byte_body)], flush, levels),
        "bit" => build(r, &[(false, &to_bit_planes(&byte_body, n, ng))], flush, levels),
        "heuristic" if byte_planes_first => build(r, &[(true, &byte_body)], flush, levels),
        "heuristic" => build(r, &[(false, &to_bit_planes(&byte_body, n, ng))], flush, levels),
        "best" => {
            let bit_body = to_bit_planes(&byte_body, n, ng);
            // the heuristic's pick first: the other one is often dropped partway
            let (byte, bit) = ((true, byte_body.as_slice()), (false, bit_body.as_slice()));
            build(r, &if byte_planes_first { [byte, bit] } else { [bit, byte] }, flush, levels)
        }
        _ => Err(format!("unknown layout {layout:?}")),
    }
}

/// compress_unit(block_flags, block_sizes, grid_params, value_anchors, residuals, codes,
///               layout, flush, zstd_levels) -> bytes
/// A unit without a time axis: header plus zstd frame, as _unit.compress builds it.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn compress_unit<'py>(
    py: Python<'py>,
    block_flags: PyReadonlyArray1<u8>,
    block_sizes: PyReadonlyArray1<i64>,
    grid_params: PyReadonlyArray1<i64>,
    value_anchors: PyReadonlyArray1<i64>,
    residuals: PyReadonlyArray1<i16>,
    codes: PyReadonlyArray1<u8>,
    layout: &str,
    flush: bool,
    zstd_levels: Vec<i32>,
) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
    let rows = Rows::new(
        block_flags.as_slice()?, block_sizes.as_slice()?, grid_params.as_slice()?, value_anchors.as_slice()?,
        residuals.as_slice()?, codes.as_slice()?,
    )
    .map_err(PyValueError::new_err)?;
    let unit = py.detach(|| compress_unit_impl(&rows, layout, flush, &zstd_levels)).map_err(PyValueError::new_err)?;
    Ok(pyo3::types::PyBytes::new(py, &unit))
}

/// The version of the libzstd this extension links, as (major, minor, release): the unit bytes match
/// python-zstandard's only if its `ZSTD_VERSION` is the same.
#[pyfunction]
fn zstd_version() -> (u32, u32, u32) {
    let v = zstd_safe::version_number();
    (v / 10000, v / 100 % 100, v % 100)
}

#[pymodule]
fn fluxcode_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compress_unit, m)?)?;
    m.add_function(wrap_pyfunction!(zstd_version, m)?)?;
    Ok(())
}
