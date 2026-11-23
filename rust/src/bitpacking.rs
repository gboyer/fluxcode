// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Garry Boyer
//! Packing columnar block-group data into uncompressed bodies, converting between byte-plane and
//! bit-plane residual layouts, and evaluating the residual-width layout heuristic.

use crate::format::{
    block_anchors_start, block_sizes_start, grid_exponents_start, grid_scales_start,
    residual_start, BlockColumns, BIT_PLANES, BLOCK_FLAG_SAMPLE_CODES, BYTES_PER_ANCHOR,
    BYTES_PER_ESCAPED_VALUE, BYTES_PER_EXPONENT, BYTES_PER_SCALE, BYTES_PER_SIZE,
    CODE_BYTES_PER_PLANE_BYTE, METADATA_BYTES_PER_BLOCK, RESIDUAL_BYTES_PER_PLANE_BYTE,
    SAMPLES_PER_PLANE_BYTE,
};
use crate::planes::{self, transpose_8x8_bits};

/// Maps a signed 16-bit residual to an unsigned 16-bit integer ordered by magnitude
/// (`0 -> 0, -1 -> 1, 1 -> 2, -2 -> 3, ...`) so small-magnitude residuals have leading zero bits.
#[inline(always)]
fn zigzag(residual: i16) -> u16 {
    ((residual << 1) ^ (residual >> 15)) as u16
}

/// Writes the low `num_bytes` little-endian bytes of `value` for `block_idx` into a byte-planed
/// metadata column starting at `field_start`.
///
/// Byte `k` of `value` is placed at `field_start + k * num_blocks + block_idx` so all blocks'
/// byte `k` values are contiguous.
fn put_byte_planed(
    body: &mut [u8],
    field_start: usize,
    num_blocks: usize,
    block_idx: usize,
    value: i64,
    num_bytes: usize,
) {
    for byte_idx in 0..num_bytes {
        body[field_start + byte_idx * num_blocks + block_idx] = (value >> (8 * byte_idx)) as u8;
    }
}

/// Builds the uncompressed byte-plane body for a block group without a time axis.
///
/// The returned buffer contains, in order: per-block metadata columns, zigzagged residuals split
/// into a low and a high byte plane (each block zero-padded to whole 8-sample groups), 2-bit
/// sample-code planes for flagged blocks, and byte-planed escaped float64 values.
pub fn write_body(columns: &BlockColumns) -> Vec<u8> {
    // Compute exact section byte lengths, allocate a zeroed buffer (so partial 8-sample group
    // padding is already zero), then populate each section through disjoint mutable slices.
    let (num_blocks, num_plane_bytes) = (columns.shape.num_blocks, columns.shape.num_plane_bytes);
    let num_code_plane_bytes = columns.code_offsets[num_blocks];
    let residuals_start = METADATA_BYTES_PER_BLOCK * num_blocks;
    let residual_bytes = RESIDUAL_BYTES_PER_PLANE_BYTE * num_plane_bytes;
    let code_bytes = CODE_BYTES_PER_PLANE_BYTE * num_code_plane_bytes;
    let escaped_bytes = BYTES_PER_ESCAPED_VALUE * columns.escaped_values.len();

    let mut body = vec![0u8; residuals_start + residual_bytes + code_bytes + escaped_bytes];
    write_metadata_columns(columns, &mut body);
    let (residual_planes, tail) = body[residuals_start..].split_at_mut(residual_bytes);
    let (code_planes, escape_planes) = tail.split_at_mut(code_bytes);
    write_residual_planes(columns, residual_planes);
    write_sample_code_planes(columns, code_planes, num_code_plane_bytes);
    write_escaped_values(columns.escaped_values, escape_planes);
    body
}

/// Writes escaped IEEE 754 float64 values in byte-planed order (all byte 0s, ..., all byte 7s)
/// into `escape_planes`.
///
/// Precondition: `escape_planes.len() == BYTES_PER_ESCAPED_VALUE * escaped_values.len()`.
fn write_escaped_values(escaped_values: &[f64], escape_planes: &mut [u8]) {
    // Scatter each float64's 8 little-endian bytes across the 8 contiguous byte planes so zstd
    // sees correlated sign/exponent bytes grouped together.
    let num_escaped = escaped_values.len();
    if num_escaped == 0 {
        return;
    }
    for (idx, &val) in escaped_values.iter().enumerate() {
        let bits = val.to_bits();
        for byte_idx in 0..BYTES_PER_ESCAPED_VALUE {
            escape_planes[byte_idx * num_escaped + idx] = (bits >> (8 * byte_idx)) as u8;
        }
    }
}

/// Writes the five per-block metadata columns (flags, block sizes, grid exponents, grid scales,
/// and block anchors) into the leading section of `body`.
fn write_metadata_columns(columns: &BlockColumns, body: &mut [u8]) {
    // Compute the starting offset of each multi-byte column once, then transpose each block's
    // metadata fields into their respective byte planes.
    let num_blocks = columns.shape.num_blocks;
    let (sizes_start, exponents_start, scales_start, anchors_start) = (
        block_sizes_start(num_blocks),
        grid_exponents_start(num_blocks),
        grid_scales_start(num_blocks),
        block_anchors_start(num_blocks),
    );
    for block_idx in 0..num_blocks {
        body[block_idx] = columns.block_flags[block_idx];
        put_byte_planed(
            body,
            sizes_start,
            num_blocks,
            block_idx,
            columns.block_sizes[block_idx],
            BYTES_PER_SIZE,
        );
        put_byte_planed(
            body,
            exponents_start,
            num_blocks,
            block_idx,
            columns.grid_exponents[block_idx],
            BYTES_PER_EXPONENT,
        );
        put_byte_planed(
            body,
            scales_start,
            num_blocks,
            block_idx,
            columns.grid_scales[block_idx],
            BYTES_PER_SCALE,
        );
        put_byte_planed(
            body,
            anchors_start,
            num_blocks,
            block_idx,
            columns.block_anchors[block_idx],
            BYTES_PER_ANCHOR,
        );
    }
}

/// Writes zigzag-encoded 16-bit residuals into `residual_planes` as a low byte plane followed by
/// a high byte plane.
///
/// Precondition: `residual_planes` is zero-initialized and has length
/// `RESIDUAL_BYTES_PER_PLANE_BYTE * columns.shape.num_plane_bytes`. Trailing padding slots in each
/// block's final 8-sample group remain zero.
fn write_residual_planes(columns: &BlockColumns, residual_planes: &mut [u8]) {
    // Split the residual buffer into its two equal-length byte planes, then write each block's
    // zigzagged low and high bytes at its 8-sample-aligned plane offset.
    let (low_plane, high_plane) =
        residual_planes.split_at_mut(SAMPLES_PER_PLANE_BYTE * columns.shape.num_plane_bytes);
    for block_idx in 0..columns.shape.num_blocks {
        let first_sample = columns.sample_offsets[block_idx];
        let block_len = columns.block_sizes[block_idx] as usize;
        // Align each block to a whole 8-sample boundary so bit-plane transposition never mixes
        // samples from different blocks in the same plane byte.
        let plane_start = SAMPLES_PER_PLANE_BYTE * columns.plane_byte_offsets[block_idx];
        let low_bytes = &mut low_plane[plane_start..plane_start + block_len];
        let high_bytes = &mut high_plane[plane_start..plane_start + block_len];
        let block_residuals = &columns.residuals[first_sample..first_sample + block_len];
        for ((&residual, low), high) in block_residuals.iter().zip(low_bytes).zip(high_bytes) {
            let zigzagged = zigzag(residual);
            *low = zigzagged as u8;
            *high = (zigzagged >> 8) as u8;
        }
    }
}

/// Writes the 2-bit sample codes of blocks with `BLOCK_FLAG_SAMPLE_CODES` set into `code_planes`
/// as two bit planes (plane 0 = low bit, plane 1 = high bit).
///
/// Precondition: `code_planes.len() == CODE_BYTES_PER_PLANE_BYTE * num_code_plane_bytes`.
fn write_sample_code_planes(
    columns: &BlockColumns,
    code_planes: &mut [u8],
    num_code_plane_bytes: usize,
) {
    // Iterate over flagged blocks, chunking each block's per-sample codes into 8-sample groups
    // and packing each group into one byte of plane 0 and one byte of plane 1.
    let (plane0, plane1) = code_planes.split_at_mut(num_code_plane_bytes);
    for block_idx in 0..columns.shape.num_blocks {
        if columns.block_flags[block_idx] & BLOCK_FLAG_SAMPLE_CODES == 0 {
            continue;
        }
        let first_sample = columns.sample_offsets[block_idx];
        let block_codes = &columns.sample_codes
            [first_sample..first_sample + columns.block_sizes[block_idx] as usize];
        for (group_idx, group_codes) in block_codes.chunks(SAMPLES_PER_PLANE_BYTE).enumerate() {
            let (byte0, byte1) = pack_eight_sample_codes(group_codes);
            plane0[columns.code_offsets[block_idx] + group_idx] = byte0;
            plane1[columns.code_offsets[block_idx] + group_idx] = byte1;
        }
    }
}

/// Packs up to 8 two-bit sample codes into `(plane0_byte, plane1_byte)`.
///
/// Bit `j` of `plane0_byte` is bit 0 of `group_codes[j]`, and bit `j` of `plane1_byte` is bit 1.
/// Missing entries in a partial final group (< 8 samples) are zero-padded.
fn pack_eight_sample_codes(group_codes: &[u8]) -> (u8, u8) {
    // Load the 1–8 code bytes into a zero-padded 8-byte word and reuse the 8×8 bit transpose:
    // byte 0 of the transpose collects every code's bit 0, and byte 1 collects every code's bit 1.
    let mut word = [0u8; SAMPLES_PER_PLANE_BYTE];
    word[..group_codes.len()].copy_from_slice(group_codes);
    let transposed = transpose_8x8_bits(u64::from_le_bytes(word));
    (transposed as u8, (transposed >> 8) as u8)
}

/// Converts a byte-plane uncompressed body into the equivalent bit-plane body.
///
/// Copies the metadata/time prefix and code/escaped-value suffix verbatim, and transposes the
/// residual region from 2 byte planes into 16 bit planes.
pub fn to_bit_planes(
    byte_body: &[u8],
    num_blocks: usize,
    num_plane_bytes: usize,
    has_time: bool,
) -> Vec<u8> {
    // Copy non-residual sections unchanged, then transpose the residual region's low and high
    // byte planes into 16 bit planes via the SIMD-accelerated transposer.
    let start = residual_start(num_blocks, has_time);
    let end = start + RESIDUAL_BYTES_PER_PLANE_BYTE * num_plane_bytes;
    let mut bit_body = vec![0u8; byte_body.len()];
    bit_body[..start].copy_from_slice(&byte_body[..start]);
    bit_body[end..].copy_from_slice(&byte_body[end..]);
    let (low_plane, high_plane) =
        byte_body[start..end].split_at(SAMPLES_PER_PLANE_BYTE * num_plane_bytes);
    planes::pack_bit_planes(
        low_plane,
        high_plane,
        &mut bit_body[start..end],
        num_plane_bytes,
        true,
    );
    bit_body
}

/// Returns the fraction of padded residual slots in a byte-plane body whose zigzagged residual is
/// at least `2^bit` (returns `0.0` when `num_plane_bytes == 0`).
///
/// Precondition: `bit <= 8` (so any non-zero high byte automatically satisfies the threshold).
pub fn wide_residual_fraction(
    body: &[u8],
    num_blocks: usize,
    num_plane_bytes: usize,
    has_time: bool,
    bit: u32,
) -> f64 {
    // Build a mask for bits `bit..8` of the low byte; a residual reaches `2^bit` iff either those
    // low-byte bits or any bit in the high byte is non-zero.
    debug_assert!(bit <= 8);
    if num_plane_bytes == 0 {
        return 0.0;
    }
    let start = residual_start(num_blocks, has_time);
    let planes = &body[start..start + BIT_PLANES * num_plane_bytes];
    let (low_plane, high_plane) = planes.split_at(SAMPLES_PER_PLANE_BYTE * num_plane_bytes);
    let low_mask = ((0xFFu32 << bit) & 0xFF) as u8;
    let wide_slots = low_plane
        .iter()
        .zip(high_plane)
        .filter(|&(&low, &high)| (low & low_mask) | high != 0)
        .count();
    wide_slots as f64 / (SAMPLES_PER_PLANE_BYTE * num_plane_bytes) as f64
}
