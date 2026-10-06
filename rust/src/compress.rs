// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Garry Boyer
//! Compression policy (`fluxcode/_compress.py`): which residual layout to use, where zstd blocks
//! end, and which candidate frame to keep.

use crate::bitpacking::{to_bit_planes, wide_share_byte_planes, write_body};
use crate::format::{Rows, Shape, BIT_PLANES, OCTET_SAMPLES};
use zstd_safe::zstd_sys::ZSTD_EndDirective::{self, ZSTD_e_end, ZSTD_e_flush};
use zstd_safe::{CCtx, CParameter, InBuffer, OutBuffer};

pub const FLUSH_MIN_DENSITY: usize = 16;
pub const BYTE_PLANES_BIT: u32 = 7;
pub const BYTE_PLANES_MAX_SHARE: f64 = 0.01;

/// Bytes counted per inner sum when measuring a plane's density: small enough that each sum fits
/// u32 lanes, which the compiler vectorizes.
const DENSITY_CHUNK_BYTES: usize = 4096;

/// Which residual layout(s) to compress (`Effort.layout` in `_compress.py`).
#[derive(Clone, Copy)]
pub enum Layout {
    /// Byte planes only (tests).
    Byte,
    /// Bit planes only (tests).
    Bit,
    /// Byte planes if few residuals reach 2^`BYTE_PLANES_BIT`, else bit planes: one compression.
    Heuristic,
    /// Both, the smaller kept.
    Best,
}

impl Layout {
    pub fn parse(name: &str) -> Result<Self, String> {
        match name {
            "byte" => Ok(Layout::Byte),
            "bit" => Ok(Layout::Bit),
            "heuristic" => Ok(Layout::Heuristic),
            "best" => Ok(Layout::Best),
            _ => Err(format!("unknown layout {name:?}")),
        }
    }
}

/// Where zstd blocks end: after the metadata and after each residual plane that is dense enough.
fn flush_points(body: &[u8], shape: &Shape, byte_planes: bool) -> Vec<usize> {
    let num_octets = shape.num_octets;
    if num_octets == 0 {
        return vec![];
    }
    let start = shape.residual_start();
    let (plane_bytes, num_planes) = if byte_planes {
        (OCTET_SAMPLES * num_octets, 2)
    } else {
        (num_octets, BIT_PLANES)
    };
    let mut points = vec![start];
    for plane_idx in 0..num_planes {
        let plane_start = start + plane_idx * plane_bytes;
        let nonzero_bytes = body[plane_start..plane_start + plane_bytes]
            .chunks(DENSITY_CHUNK_BYTES)
            .map(|chunk| chunk.iter().map(|&byte| (byte != 0) as u32).sum::<u32>() as usize)
            .sum::<usize>();
        if nonzero_bytes * FLUSH_MIN_DENSITY > plane_bytes {
            points.push(plane_start + plane_bytes);
        }
    }
    points
}

thread_local! {
    /// One zstd context per thread, reused across frames like python-zstandard's compressor
    /// (creating one allocates its workspace).
    static CCTX: std::cell::RefCell<CCtx<'static>> = std::cell::RefCell::new(CCtx::create());
}

fn zstd_error(code: usize) -> String {
    zstd_safe::get_error_name(code).to_string()
}

/// A zstd frame being written with a header in front of it.
struct FrameWriter<'a> {
    cctx: &'a mut CCtx<'static>,
    out: Vec<u8>,
}

impl<'a> FrameWriter<'a> {
    /// Starts a frame of `body_len` bytes at `level`, to be ended by `num_cuts` + 1 calls to
    /// `write` at most.
    fn new(
        cctx: &'a mut CCtx<'static>,
        header: &[u8],
        body_len: usize,
        level: i32,
        num_cuts: usize,
    ) -> Result<Self, String> {
        cctx.reset(zstd_safe::ResetDirective::SessionOnly)
            .map_err(zstd_error)?;
        cctx.set_parameter(CParameter::CompressionLevel(level))
            .map_err(zstd_error)?;
        cctx.set_parameter(CParameter::ChecksumFlag(false))
            .map_err(zstd_error)?;
        cctx.set_parameter(CParameter::ContentSizeFlag(true))
            .map_err(zstd_error)?;
        cctx.set_pledged_src_size(Some(body_len as u64))
            .map_err(zstd_error)?;
        let capacity = header.len() + zstd_safe::compress_bound(body_len) + 64 * (num_cuts + 2);
        let mut out = Vec::with_capacity(capacity);
        out.extend_from_slice(header);
        Ok(FrameWriter { cctx, out })
    }

    /// Compresses `input`, ending a block (`ZSTD_e_flush`) or the frame (`ZSTD_e_end`). Returns the
    /// length of the output so far, header included.
    fn write(&mut self, input: &[u8], directive: ZSTD_EndDirective) -> Result<usize, String> {
        let mut input_buffer = InBuffer::around(input);
        loop {
            // the capacity from `new` covers the worst case, so this only guards against a stalled
            // loop
            if self.out.len() == self.out.capacity() {
                self.out.reserve(1 << 16);
            }
            let position = self.out.len();
            let remaining = self
                .cctx
                .compress_stream2(
                    &mut OutBuffer::around_pos(&mut self.out, position),
                    &mut input_buffer,
                    directive,
                )
                .map_err(zstd_error)?;
            if input_buffer.pos == input.len() && remaining == 0 {
                return Ok(self.out.len());
            }
        }
    }
}

/// `header` followed by the zstd frame of `body`, ended after each of the `cuts` (with none, a
/// single block-ending call, which is what python-zstandard's `compress` does). With a `max_len`,
/// gives up (None) as soon as the blocks written so far are longer than it: the frame can only
/// grow.
fn compress_with_header(
    header: &[u8],
    body: &[u8],
    cuts: &[usize],
    level: i32,
    max_len: Option<usize>,
) -> Result<Option<Vec<u8>>, String> {
    CCTX.with(|cell| {
        let mut cctx = cell.borrow_mut();
        let mut writer = FrameWriter::new(&mut cctx, header, body.len(), level, cuts.len())?;
        let mut previous_cut = 0;
        for &cut in cuts {
            if previous_cut < cut && cut < body.len() {
                let written = writer.write(&body[previous_cut..cut], ZSTD_e_flush)?;
                previous_cut = cut;
                if max_len.is_some_and(|max_len| written > max_len) {
                    return Ok(None);
                }
            }
        }
        writer.write(&body[previous_cut..], ZSTD_e_end)?;
        Ok(Some(writer.out))
    })
}

/// A unit as its header and the smallest frame of the candidate bodies (byte planes or not) over
/// the zstd levels. Ties go to byte planes, then to the earlier level, whatever the order the
/// candidates are tried in; a candidate that can't beat the best so far is dropped as soon as its
/// flushed blocks show it. All candidates are tried at a level before the next level, so the cheap
/// level's frames set the limit for the expensive ones.
fn smallest_unit(
    shape: &Shape,
    candidates: &[(bool, &[u8])],
    flush: bool,
    levels: &[i32],
) -> Result<Vec<u8>, String> {
    // each candidate with its header and cuts
    let prepared: Vec<_> = candidates
        .iter()
        .map(|&(byte_planes, body)| {
            let cuts = if flush {
                flush_points(body, shape, byte_planes)
            } else {
                vec![]
            };
            (byte_planes, body, shape.header(byte_planes), cuts)
        })
        .collect();
    // the best unit so far and its rank (byte planes first, each by level), which decides ties
    let mut best: Option<(Vec<u8>, usize)> = None;
    for (level_idx, &level) in levels.iter().enumerate() {
        for (byte_planes, body, header, cuts) in &prepared {
            let rank = if *byte_planes {
                level_idx
            } else {
                levels.len() + level_idx
            };
            // a frame of this length or less wins: a tie only against a worse rank
            let max_len = best.as_ref().map(|(unit, best_rank)| {
                if rank < *best_rank {
                    unit.len()
                } else {
                    unit.len() - 1
                }
            });
            if let Some(unit) = compress_with_header(header, body, cuts, level, max_len)? {
                if max_len.is_none_or(|max_len| unit.len() <= max_len) {
                    best = Some((unit, rank));
                }
            }
        }
    }
    best.map(|(unit, _)| unit)
        .ok_or_else(|| "no zstd levels".to_string())
}

/// The unit of a byte-plane body (`_compress.pack`): the layout chosen and the frame built as
/// `layout`, `flush` and `levels` (an `Effort`) say. The bit-plane body is derived from the
/// byte-plane one.
pub fn pack(
    shape: &Shape,
    byte_body: &[u8],
    layout: Layout,
    flush: bool,
    levels: &[i32],
) -> Result<Vec<u8>, String> {
    let (num_blocks, num_octets, has_time) = (shape.num_blocks, shape.num_octets, shape.has_time());
    if byte_body.len() < shape.residual_start() + BIT_PLANES * num_octets {
        return Err("body too short for its residual planes".to_string());
    }
    let byte_planes_predicted = || {
        wide_share_byte_planes(byte_body, num_blocks, num_octets, has_time, BYTE_PLANES_BIT)
            < BYTE_PLANES_MAX_SHARE
    };
    let bit_body = || to_bit_planes(byte_body, num_blocks, num_octets, has_time);
    match layout {
        Layout::Byte => smallest_unit(shape, &[(true, byte_body)], flush, levels),
        Layout::Bit => smallest_unit(shape, &[(false, &bit_body())], flush, levels),
        Layout::Heuristic if byte_planes_predicted() => {
            smallest_unit(shape, &[(true, byte_body)], flush, levels)
        }
        Layout::Heuristic => smallest_unit(shape, &[(false, &bit_body())], flush, levels),
        Layout::Best => {
            let bit_body = bit_body();
            let (byte, bit) = ((true, byte_body), (false, bit_body.as_slice()));
            // with flushes the heuristic's pick goes first, since the other is often dropped
            // partway; without them nothing can be dropped and the order is free (so the statistic
            // isn't computed)
            let candidates = if flush && !byte_planes_predicted() {
                [bit, byte]
            } else {
                [byte, bit]
            };
            smallest_unit(shape, &candidates, flush, levels)
        }
    }
}

/// A unit without a time axis from its rows (`_compress.compress`).
pub fn compress_unit(
    rows: &Rows,
    layout: Layout,
    flush: bool,
    levels: &[i32],
) -> Result<Vec<u8>, String> {
    pack(&rows.shape, &write_body(rows), layout, flush, levels)
}
