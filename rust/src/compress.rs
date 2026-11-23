// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Garry Boyer
//! Compression policy: residual layout selection, density-based zstd block flushing, and
//! early-aborted candidate framing.

use crate::bitpacking::{to_bit_planes, wide_residual_fraction, write_body};
use crate::format::{BlockColumns, GroupShape, BIT_PLANES, SAMPLES_PER_PLANE_BYTE};
use zstd_safe::zstd_sys::ZSTD_EndDirective::{self, ZSTD_e_end, ZSTD_e_flush};
use zstd_safe::{CCtx, CParameter, InBuffer, OutBuffer};

/// Inverse density threshold for flushing a zstd block after a residual plane: a plane is flushed
/// when strictly more than `1 / FLUSH_MIN_DENSITY` (1/16) of its bytes are non-zero.
pub const FLUSH_MIN_DENSITY: usize = 16;
/// Bit threshold (`2^BYTE_PLANES_BIT`) used by the layout heuristic to classify wide residuals.
pub const BYTE_PLANES_BIT: u32 = 7;
/// Maximum fraction of wide residuals for which the heuristic chooses byte planes over bit planes.
pub const BYTE_PLANES_MAX_SHARE: f64 = 0.01;

/// Chunk size (in bytes) for inner non-zero-byte summation when measuring plane density.
/// Keeping each inner sum bounded by 4096 allows auto-vectorization in `u32` lanes without overflow.
const DENSITY_CHUNK_BYTES: usize = 4096;

/// Policy controlling which residual layout(s) to compress.
#[derive(Clone, Copy)]
pub enum ResidualLayout {
    /// Choose byte planes when fewer than `BYTE_PLANES_MAX_SHARE` of residuals reach
    /// `2^BYTE_PLANES_BIT`, otherwise choose bit planes (one layout compressed).
    Heuristic,
    /// Compress both byte-plane and bit-plane layouts and keep the smaller frame.
    Best,
}

impl ResidualLayout {
    /// Parses a layout policy name (`"heuristic"` or `"best"`), returning `Err` for unknown names.
    pub fn parse(name: &str) -> Result<Self, String> {
        match name {
            "heuristic" => Ok(ResidualLayout::Heuristic),
            "best" => Ok(ResidualLayout::Best),
            _ => Err(format!("unknown layout {name:?}")),
        }
    }
}

/// Computes byte offsets in `body` where zstd blocks should be flushed.
///
/// Returns an empty vector when `shape.num_plane_bytes == 0`. Otherwise places a flush right after
/// the metadata/time prefix, plus a flush after every residual plane whose non-zero byte density
/// exceeds `1 / FLUSH_MIN_DENSITY`.
fn flush_points(body: &[u8], shape: &GroupShape, byte_planes: bool) -> Vec<usize> {
    // Flushing after metadata and after each dense residual plane lets zstd build separate
    // Huffman/FSE tables for planes with distinct symbol distributions while coalescing sparse
    // high-order planes into a single block.
    let num_plane_bytes = shape.num_plane_bytes;
    if num_plane_bytes == 0 {
        return vec![];
    }
    let start = shape.residual_start();
    let (plane_bytes, num_planes) = if byte_planes {
        (SAMPLES_PER_PLANE_BYTE * num_plane_bytes, 2)
    } else {
        (num_plane_bytes, BIT_PLANES)
    };
    let mut points = vec![start];
    for plane_idx in 0..num_planes {
        let plane_start = start + plane_idx * plane_bytes;
        // Sum non-zero bytes in 4096-byte chunks so LLVM auto-vectorizes the inner loop in u32
        // lanes without widening every byte to usize.
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
    /// Reusable per-thread zstd compression context, avoiding repeated workspace allocations
    /// across frames (matching `python-zstandard`'s compressor reuse).
    static CCTX: std::cell::RefCell<CCtx<'static>> = std::cell::RefCell::new(CCtx::create());
}

/// Converts a libzstd error code into its human-readable error message.
fn zstd_error(code: usize) -> String {
    zstd_safe::get_error_name(code).to_string()
}

/// Streaming zstd frame builder that prepends an uncompressed block-group header.
struct FrameWriter<'a> {
    /// Borrowed thread-local zstd compression context.
    cctx: &'a mut CCtx<'static>,
    /// Output buffer holding the header followed by compressed zstd bytes.
    out: Vec<u8>,
}

impl<'a> FrameWriter<'a> {
    /// Initializes a zstd frame for a `body_len`-byte payload at `level` with `header` prepended.
    ///
    /// Pre-allocates output capacity for the worst-case compressed size plus up to `num_cuts + 1`
    /// flushed zstd blocks. Returns `Err` if libzstd rejects any session parameter.
    fn new(
        cctx: &'a mut CCtx<'static>,
        header: &[u8],
        body_len: usize,
        level: i32,
        num_cuts: usize,
    ) -> Result<Self, String> {
        // Reset session state (retaining allocated tables), configure deterministic framing flags
        // (no checksum, explicit content size), and pre-size the output buffer.
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
        // `compress_bound` covers a single unsplit frame; add 64 bytes per flush boundary for
        // extra block headers.
        let capacity = header.len() + zstd_safe::compress_bound(body_len) + 64 * (num_cuts + 2);
        let mut out = Vec::with_capacity(capacity);
        out.extend_from_slice(header);
        Ok(FrameWriter { cctx, out })
    }

    /// Feeds `input` into the zstd stream and finishes the block (`ZSTD_e_flush`) or frame
    /// (`ZSTD_e_end`).
    ///
    /// Returns the total output length written so far, including the leading header.
    fn write(&mut self, input: &[u8], directive: ZSTD_EndDirective) -> Result<usize, String> {
        // Loop until libzstd has consumed all input bytes and flushed all internal buffered output.
        let mut input_buffer = InBuffer::around(input);
        loop {
            // Pre-allocated capacity handles the worst case; reserve more only as a safety guard
            // against a stalled loop.
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

/// Compresses `body` at zstd `level` with `header` prepended, flushing zstd blocks at `cuts`.
///
/// When `max_len` is `Some(limit)`, aborts early and returns `Ok(None)` as soon as the bytes
/// flushed so far exceed `limit` (since subsequent blocks can only grow the output). With no
/// `cuts`, emits a single `ZSTD_e_end` call matching `python-zstandard`'s one-shot `compress`.
fn compress_with_header(
    header: &[u8],
    body: &[u8],
    cuts: &[usize],
    level: i32,
    max_len: Option<usize>,
) -> Result<Option<Vec<u8>>, String> {
    // Borrow the thread-local CCtx, flush each non-empty interior slice between consecutive cuts
    // (checking `max_len` after every flush), and finalize the tail with `ZSTD_e_end`.
    CCTX.with(|cell| {
        let mut cctx = cell.borrow_mut();
        let mut writer = FrameWriter::new(&mut cctx, header, body.len(), level, cuts.len())?;
        let mut previous_cut = 0;
        for &cut in cuts {
            if previous_cut < cut && cut < body.len() {
                let written = writer.write(&body[previous_cut..cut], ZSTD_e_flush)?;
                previous_cut = cut;
                if max_len.is_some_and(|limit| written > limit) {
                    return Ok(None);
                }
            }
        }
        writer.write(&body[previous_cut..], ZSTD_e_end)?;
        Ok(Some(writer.out))
    })
}

/// Compresses each `(byte_planes, body)` candidate across all zstd `levels` and returns the
/// smallest resulting block group (header + zstd frame).
///
/// Ties are broken deterministically regardless of candidate order: byte planes beat bit planes,
/// and among equal layouts the earlier level in `levels` wins. Returns `Err` if `levels` is empty
/// or libzstd fails.
fn select_smallest_group(
    shape: &GroupShape,
    candidates: &[(bool, &[u8])],
    flush: bool,
    levels: &[i32],
) -> Result<Vec<u8>, String> {
    // 1. Precompute each candidate's 8-byte header and optional density-based flush cuts.
    // 2. Sweep levels in the outer loop and layouts in the inner loop so fast, low-level frames
    //    establish tight `max_len` early-abort bounds before slower levels run.
    // 3. Assign each `(layout, level)` pair a deterministic tie-break rank (byte planes first,
    //    then level index) so equal-size frames choose the same winner regardless of evaluation order.
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

    // Tracks `(best_group_bytes, best_rank)`.
    let mut best: Option<(Vec<u8>, usize)> = None;
    for (level_idx, &level) in levels.iter().enumerate() {
        for (byte_planes, body, header, cuts) in &prepared {
            let rank = if *byte_planes {
                level_idx
            } else {
                levels.len() + level_idx
            };
            // Allow a tie only if this candidate has a strictly better (smaller) tie-break rank.
            let max_len = best.as_ref().map(|(group, best_rank)| {
                if rank < *best_rank {
                    group.len()
                } else {
                    group.len() - 1
                }
            });
            if let Some(group) = compress_with_header(header, body, cuts, level, max_len)? {
                if max_len.is_none_or(|limit| group.len() <= limit) {
                    best = Some((group, rank));
                }
            }
        }
    }
    best.map(|(group, _)| group)
        .ok_or_else(|| "no zstd levels".to_string())
}

/// Packs and compresses an uncompressed byte-plane body into a complete block group (header +
/// zstd frame) according to `layout`, `flush`, and `levels`.
///
/// Derives the bit-plane body on demand when needed. Returns `Err` if `byte_body` is shorter than
/// the residual section implied by `shape`, if `levels` is empty, or if libzstd fails.
pub fn pack(
    shape: &GroupShape,
    byte_body: &[u8],
    layout: ResidualLayout,
    flush: bool,
    levels: &[i32],
) -> Result<Vec<u8>, String> {
    // Validate that `byte_body` covers the residual planes, then dispatch to either the heuristic
    // (compress one layout) or exhaustive `Best` mode (compress both, ordering the heuristic's
    // favorite first when flushing so the runner-up can abort early).
    let (num_blocks, num_plane_bytes, has_time) =
        (shape.num_blocks, shape.num_plane_bytes, shape.has_time());
    if byte_body.len() < shape.residual_start() + BIT_PLANES * num_plane_bytes {
        return Err("body too short for its residual planes".to_string());
    }
    let predict_byte_planes = || {
        wide_residual_fraction(
            byte_body,
            num_blocks,
            num_plane_bytes,
            has_time,
            BYTE_PLANES_BIT,
        ) < BYTE_PLANES_MAX_SHARE
    };
    let build_bit_body = || to_bit_planes(byte_body, num_blocks, num_plane_bytes, has_time);

    match layout {
        ResidualLayout::Heuristic if predict_byte_planes() => {
            select_smallest_group(shape, &[(true, byte_body)], flush, levels)
        }
        ResidualLayout::Heuristic => {
            select_smallest_group(shape, &[(false, &build_bit_body())], flush, levels)
        }
        ResidualLayout::Best => {
            let bit_body = build_bit_body();
            let (byte_cand, bit_cand) = ((true, byte_body), (false, bit_body.as_slice()));
            // With flushes enabled, try the heuristic's predicted winner first so it sets a tight
            // early-abort bound; without flushes no early abort is possible, so skip the scan.
            let candidates = if flush && !predict_byte_planes() {
                [bit_cand, byte_cand]
            } else {
                [byte_cand, bit_cand]
            };
            select_smallest_group(shape, &candidates, flush, levels)
        }
    }
}

/// Serializes validated columnar block data (without a time axis) into a compressed block group.
pub fn compress_group(
    columns: &BlockColumns,
    layout: ResidualLayout,
    flush: bool,
    levels: &[i32],
) -> Result<Vec<u8>, String> {
    pack(&columns.shape, &write_body(columns), layout, flush, levels)
}
