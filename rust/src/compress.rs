// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Garry Boyer
//! Compression policy (`fluxcode/_compress.py`): which residual layout to use, where zstd blocks end,
//! and which candidate frame to keep.

use crate::bitpacking::{to_bit_planes, wide_share_byte_planes, write_body};
use crate::format::{residual_start, Rows, Shape};
use zstd_safe::zstd_sys::ZSTD_EndDirective::{ZSTD_e_end, ZSTD_e_flush};
use zstd_safe::{CCtx, CParameter, InBuffer, OutBuffer};

pub const FLUSH_MIN_DENSITY: usize = 16;
pub const BYTE_PLANES_BIT: u32 = 7;
pub const BYTE_PLANES_MAX_SHARE: f64 = 0.01;

/// Where zstd blocks end: after the metadata and after each residual plane that is dense enough.
fn flush_points(raw: &[u8], n: usize, ng: usize, has_time: bool, byte_planes: bool) -> Vec<usize> {
    if ng == 0 {
        return vec![];
    }
    let start = residual_start(n, has_time);
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
/// All candidates are tried at a level before the next level, so the cheap level's frames set the limit
/// for the expensive ones.
fn build(s: &Shape, candidates: &[(bool, &[u8])], flush: bool, levels: &[i32]) -> Result<Vec<u8>, String> {
    let prepared: Vec<_> = candidates
        .iter()
        .map(|&(byte_planes, body)| {
            let cuts = if flush { flush_points(body, s.n, s.ng, s.has_time(), byte_planes) } else { vec![] };
            (byte_planes, body, s.header(byte_planes), cuts)
        })
        .collect();
    let mut best: Option<(Vec<u8>, usize)> = None;
    for (i, &level) in levels.iter().enumerate() {
        for (byte_planes, body, header, cuts) in &prepared {
            let rank = if *byte_planes { i } else { levels.len() + i };
            // a frame of this length or less wins (a tie only against a worse rank)
            let limit = best.as_ref().map(|(b, best_rank)| if rank < *best_rank { b.len() } else { b.len() - 1 });
            if let Some(unit) = compress_with_header(header, body, cuts, level, limit)? {
                if limit.map_or(true, |l| unit.len() <= l) {
                    best = Some((unit, rank));
                }
            }
        }
    }
    best.map(|(unit, _)| unit).ok_or_else(|| "no zstd levels".to_string())
}

/// The unit of a byte-plane body (`_compress.pack`): the layout chosen and the frame built as `layout`,
/// `flush` and `levels` (an `Effort`) say. The bit-plane body is derived from the byte-plane one.
pub fn pack(s: &Shape, byte_body: &[u8], layout: &str, flush: bool, levels: &[i32]) -> Result<Vec<u8>, String> {
    let (n, ng, has_time) = (s.n, s.ng, s.has_time());
    if byte_body.len() < s.residual_start() + 16 * ng {
        return Err("body too short for its residual planes".to_string());
    }
    let byte_planes_predicted = || wide_share_byte_planes(byte_body, n, ng, has_time, BYTE_PLANES_BIT) < BYTE_PLANES_MAX_SHARE;
    match layout {
        "byte" => build(s, &[(true, byte_body)], flush, levels),
        "bit" => build(s, &[(false, &to_bit_planes(byte_body, n, ng, has_time))], flush, levels),
        "heuristic" if byte_planes_predicted() => build(s, &[(true, byte_body)], flush, levels),
        "heuristic" => build(s, &[(false, &to_bit_planes(byte_body, n, ng, has_time))], flush, levels),
        "best" => {
            let bit_body = to_bit_planes(byte_body, n, ng, has_time);
            let (byte, bit) = ((true, byte_body), (false, bit_body.as_slice()));
            // with flushes the heuristic's pick goes first, since the other is often dropped partway; without
            // them nothing can be dropped and the order is free (so the statistic isn't computed)
            build(s, &if flush && !byte_planes_predicted() { [bit, byte] } else { [byte, bit] }, flush, levels)
        }
        _ => Err(format!("unknown layout {layout:?}")),
    }
}

/// A unit without a time axis from its rows (`_compress.compress`).
pub fn compress_unit(r: &Rows, layout: &str, flush: bool, levels: &[i32]) -> Result<Vec<u8>, String> {
    pack(&r.shape, &write_body(r), layout, flush, levels)
}
