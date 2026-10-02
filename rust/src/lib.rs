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

const BLOCK_FLAG_NONFINITE: u8 = 0x08;
// BYTES_PER_FLAGS + BYTES_PER_SIZE + BYTES_PER_PARAM + BYTES_PER_ANCHOR
const METADATA_BYTES_PER_BLOCK: usize = 1 + 2 + 2 + 8;
const HEADER_BYTES: usize = 8;

#[inline(always)]
fn transpose8(mut x: u64) -> u64 {
    let mut d = (x ^ (x >> 7)) & 0x00AA00AA00AA00AA;
    x = x ^ d ^ (d << 7);
    d = (x ^ (x >> 14)) & 0x0000CCCC0000CCCC;
    x = x ^ d ^ (d << 14);
    d = (x ^ (x >> 28)) & 0x00000000F0F0F0F0;
    x ^ d ^ (d << 28)
}

// ---- compress: rows -> unit bytes (layout choice, body packing, flushed zstd frame) ----

use zstd_safe::{CCtx, CParameter, InBuffer, OutBuffer};

const FLUSH_MIN_DENSITY: usize = 16;
const BYTE_PLANES_BIT: u32 = 7;
const BYTE_PLANES_MAX_SHARE: f64 = 0.01;
const FORMAT_VERSION: u8 = 1;
const UNIT_FLAG_BYTE_PLANES: u8 = 1;

struct Rows<'a> {
    flags: &'a [u8],
    sizes: &'a [i64],
    params: &'a [i64],
    anchors: &'a [i64],
    residuals: &'a [i16],
    codes: &'a [u8],
    so: Vec<i64>,
    go: Vec<i64>,
    co: Vec<i64>,
}

#[inline(always)]
fn zigzag(r: i16) -> u16 {
    ((r << 1) ^ (r >> 15)) as u16
}

/// The uncompressed body of a unit without a time axis (bit_planes: 16 bit planes or 2 byte planes).
fn write_body(r: &Rows, byte_planes: bool) -> Vec<u8> {
    let n = r.flags.len();
    let ng = r.go[n] as usize;
    let ncg = r.co[n] as usize;
    let res_start = METADATA_BYTES_PER_BLOCK * n;
    let code_start = res_start + 16 * ng;
    let mut raw = vec![0u8; code_start + 2 * ncg];
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
    let max_groups = (0..n).map(|b| (r.go[b + 1] - r.go[b]) as usize).max().unwrap_or(0);
    let (mut lowb, mut highb) = (vec![0u8; 8 * max_groups], vec![0u8; 8 * max_groups]);
    for b in 0..n {
        let (s0, len) = (r.so[b] as usize, r.sizes[b] as usize);
        let block = &r.residuals[s0..s0 + len];
        let (g0, g1) = (r.go[b] as usize, r.go[b + 1] as usize);
        if byte_planes {
            let (low, high) = res.split_at_mut(8 * ng);
            let s = 8 * g0;
            let (low, high) = (&mut low[s..s + len], &mut high[s..s + len]);
            for ((&v, l), h) in block.iter().zip(low).zip(high) {
                let z = zigzag(v);
                *l = z as u8;
                *h = (z >> 8) as u8;
            }
        } else {
            // zigzag into whole 8-byte groups (the padding stays zero), then transpose each group
            let ngb = g1 - g0;
            lowb[..8 * ngb].fill(0);
            highb[..8 * ngb].fill(0);
            for ((&v, l), h) in block.iter().zip(lowb.iter_mut()).zip(highb.iter_mut()) {
                let z = zigzag(v);
                *l = z as u8;
                *h = (z >> 8) as u8;
            }
            let mut planes: Vec<&mut [u8]> = Vec::with_capacity(16);
            let mut rest: &mut [u8] = res;
            let mut consumed = 0;
            for k in 0..16 {
                let skip = k * ng + g0 - consumed;
                let (_, tail) = rest.split_at_mut(skip);
                let (mine, tail) = tail.split_at_mut(ngb);
                planes.push(mine);
                rest = tail;
                consumed = k * ng + g0 + ngb;
            }
            for (g, (lw, hw)) in lowb[..8 * ngb].chunks_exact(8).zip(highb[..8 * ngb].chunks_exact(8)).enumerate() {
                let lo = transpose8(u64::from_le_bytes(lw.try_into().unwrap())).to_le_bytes();
                let hi = transpose8(u64::from_le_bytes(hw.try_into().unwrap())).to_le_bytes();
                for k in 0..8 {
                    planes[k][g] = lo[k];
                    planes[8 + k][g] = hi[k];
                }
            }
        }
        if r.flags[b] & BLOCK_FLAG_NONFINITE != 0 {
            let c0 = r.co[b] as usize;
            let (p0, p1) = codes_raw.split_at_mut(ncg);
            for (g, chunk) in r.codes[s0..s0 + len].chunks(8).enumerate() {
                let (mut b0, mut b1) = (0u8, 0u8);
                for (k, &c) in chunk.iter().enumerate() {
                    b0 |= (c & 1) << k;
                    b1 |= ((c >> 1) & 1) << k;
                }
                p0[c0 + g] = b0;
                p1[c0 + g] = b1;
            }
        }
    }
    raw
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

fn flush_points(raw: &[u8], n: usize, ng: usize, byte_planes: bool) -> Vec<usize> {
    if ng == 0 {
        return vec![];
    }
    let start = METADATA_BYTES_PER_BLOCK * n;
    let (plane_bytes, planes) = if byte_planes { (8 * ng, 2) } else { (ng, 16) };
    let mut points = vec![start];
    for p in 0..planes {
        let ps = start + p * plane_bytes;
        let nz = raw[ps..ps + plane_bytes].chunks(4096).map(|c| c.iter().map(|&v| (v != 0) as u32).sum::<u32>() as usize).sum::<usize>();
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

fn compress_flushed(body: &[u8], cuts: &[usize], level: i32) -> Result<Vec<u8>, String> {
    CCTX.with(|c| compress_flushed_with(&mut c.borrow_mut(), body, cuts, level))
}

fn compress_flushed_with(cctx: &mut CCtx<'static>, body: &[u8], cuts: &[usize], level: i32) -> Result<Vec<u8>, String> {
    let e = |c: usize| zstd_safe::get_error_name(c).to_string();
    cctx.reset(zstd_safe::ResetDirective::SessionOnly).map_err(e)?;
    cctx.set_parameter(CParameter::CompressionLevel(level)).map_err(e)?;
    cctx.set_parameter(CParameter::ChecksumFlag(false)).map_err(e)?;
    cctx.set_parameter(CParameter::ContentSizeFlag(true)).map_err(e)?;
    cctx.set_pledged_src_size(Some(body.len() as u64)).map_err(e)?;
    let mut out: Vec<u8> = Vec::with_capacity(zstd_safe::compress_bound(body.len()) + 64 * (cuts.len() + 2));
    let mut run = |input: &[u8], op: zstd_safe::zstd_sys::ZSTD_EndDirective| -> Result<(), String> {
        let mut inb = InBuffer::around(input);
        loop {
            if out.len() == out.capacity() {
                out.reserve(1 << 16);
            }
            let pos = out.len();
            let remaining = cctx.compress_stream2(&mut OutBuffer::around_pos(&mut out, pos), &mut inb, op).map_err(e)?;
            if inb.pos == input.len() && remaining == 0 {
                return Ok(());
            }
        }
    };
    use zstd_safe::zstd_sys::ZSTD_EndDirective::*;
    let mut prev = 0;
    for &cut in cuts {
        if prev < cut && cut < body.len() {
            run(&body[prev..cut], ZSTD_e_continue)?;
            run(&[], ZSTD_e_flush)?;
            prev = cut;
        }
    }
    run(&body[prev..], ZSTD_e_end)?;
    Ok(out)
}

/// The smallest frame of a body over the zstd levels (ties keep the earlier level).
fn frame(body: &[u8], n: usize, ng: usize, byte_planes: bool, flush: bool, levels: &[i32]) -> Result<Vec<u8>, String> {
    let cuts = if flush { flush_points(body, n, ng, byte_planes) } else { vec![] };
    let mut best: Option<Vec<u8>> = None;
    for &level in levels {
        let f = if flush {
            compress_flushed(body, &cuts, level)?
        } else {
            CCTX.with(|c| {
                let cctx = &mut *c.borrow_mut();
                let e = |c: usize| zstd_safe::get_error_name(c).to_string();
                cctx.reset(zstd_safe::ResetDirective::SessionOnly).map_err(e)?;
                cctx.set_parameter(CParameter::CompressionLevel(level)).map_err(e)?;
                cctx.set_parameter(CParameter::ChecksumFlag(false)).map_err(e)?;
                cctx.set_parameter(CParameter::ContentSizeFlag(true)).map_err(e)?;
                let mut out: Vec<u8> = Vec::with_capacity(zstd_safe::compress_bound(body.len()));
                cctx.compress2(&mut out, body).map_err(e)?;
                Ok::<_, String>(out)
            })?
        };
        if best.as_ref().map_or(true, |b| f.len() < b.len()) {
            best = Some(f);
        }
    }
    best.ok_or_else(|| "no zstd levels".to_string())
}

/// A unit as its 8 header bytes and its zstd frame.
type Unit = ([u8; HEADER_BYTES], Vec<u8>);

fn compress_unit_impl(r: &Rows, layout: &str, flush: bool, levels: &[i32]) -> Result<Unit, String> {
    let n = r.flags.len();
    let num_samples = r.so[n] as usize;
    let ng = r.go[n] as usize;
    let build = |body: &Vec<u8>, byte_planes: bool| -> Result<Unit, String> {
        let mut header = [0u8; HEADER_BYTES];
        header[0] = FORMAT_VERSION;
        header[1] = if byte_planes { UNIT_FLAG_BYTE_PLANES } else { 0 };
        header[2..4].copy_from_slice(&(n as u16).to_le_bytes());
        header[4..8].copy_from_slice(&(num_samples as u32).to_le_bytes());
        Ok((header, frame(body, n, ng, byte_planes, flush, levels)?))
    };
    match layout {
        "bit" | "byte" => {
            let bp = layout == "byte";
            build(&write_body(r, bp), bp)
        }
        "heuristic" => {
            // the statistic is read from the byte-plane body, the cheaper one to write
            let first = write_body(r, true);
            if wide_share_byte_planes(&first, n, ng) < BYTE_PLANES_MAX_SHARE {
                build(&first, true)
            } else {
                build(&write_body(r, false), false)
            }
        }
        _ => {
            let (byte, bit) = (build(&write_body(r, true), true)?, build(&write_body(r, false), false)?);
            Ok(if byte.1.len() <= bit.1.len() { byte } else { bit })
        }
    }
}

/// compress_unit(block_flags, block_sizes, grid_params, value_anchors, residuals, codes,
///               layout, flush, zstd_levels, sample_offsets, group_offsets, code_offsets) -> bytes
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
    sample_offsets: PyReadonlyArray1<i64>,
    group_offsets: PyReadonlyArray1<i64>,
    code_offsets: PyReadonlyArray1<i64>,
) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
    let rows = Rows {
        flags: block_flags.as_slice()?,
        sizes: block_sizes.as_slice()?,
        params: grid_params.as_slice()?,
        anchors: value_anchors.as_slice()?,
        residuals: residuals.as_slice()?,
        codes: codes.as_slice()?,
        so: sample_offsets.as_slice()?.to_vec(),
        go: group_offsets.as_slice()?.to_vec(),
        co: code_offsets.as_slice()?.to_vec(),
    };
    let n = rows.flags.len();
    if rows.sizes.len() != n || rows.params.len() != n || rows.anchors.len() != n || rows.so.len() != n + 1
        || rows.go.len() != n + 1 || rows.co.len() != n + 1 || rows.residuals.len() != rows.so[n] as usize
        || (rows.co[n] > 0 && rows.codes.len() != rows.so[n] as usize)
    {
        return Err(PyValueError::new_err("rows don't match the layout"));
    }
    let unit = py.detach(|| compress_unit_impl(&rows, layout, flush, &zstd_levels)).map_err(PyValueError::new_err)?;
    pyo3::types::PyBytes::new_with(py, HEADER_BYTES + unit.1.len(), |buf| {
        buf[..HEADER_BYTES].copy_from_slice(&unit.0);
        buf[HEADER_BYTES..].copy_from_slice(&unit.1);
        Ok(())
    })
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
