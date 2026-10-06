// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Garry Boyer
//! Packing rows into the body's bytes, and the residual layouts (`fluxcode/_bitpacking.py`).

use crate::format::{
    block_sizes_start, grid_params_start, residual_start, value_anchor_start, Rows, BIT_PLANES,
    BLOCK_FLAG_NONFINITE, BYTES_PER_ANCHOR, BYTES_PER_PARAM, BYTES_PER_SIZE, CODE_BYTES_PER_OCTET,
    METADATA_BYTES_PER_BLOCK, OCTET_SAMPLES, RESIDUAL_BYTES_PER_OCTET,
};
use crate::planes::{self, transpose8};

#[inline(always)]
fn zigzag(residual: i16) -> u16 {
    ((residual << 1) ^ (residual >> 15)) as u16
}

/// Writes the low `num_bytes` bytes of a value into a byte-planed field: byte `i` goes to the
/// `i`-th run of `num_blocks` bytes (`_format.put_int64`, `put_int16`).
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

/// The uncompressed body of a block group without a time axis, in the byte-plane layout: the metadata
/// columns, the zigzagged residuals as a low and a high byte plane (each block padded with zeros to
/// whole octets of 8 samples), then the non-finite code planes.
pub fn write_body(rows: &Rows) -> Vec<u8> {
    let (num_blocks, num_octets) = (rows.shape.num_blocks, rows.shape.num_octets);
    let num_code_octets = rows.code_offsets[num_blocks];
    let residuals_start = METADATA_BYTES_PER_BLOCK * num_blocks;
    let residual_bytes = RESIDUAL_BYTES_PER_OCTET * num_octets;
    let mut body =
        vec![0u8; residuals_start + residual_bytes + CODE_BYTES_PER_OCTET * num_code_octets];
    write_columns(rows, &mut body);
    let (residual_planes, code_planes) = body[residuals_start..].split_at_mut(residual_bytes);
    write_residual_planes(rows, residual_planes);
    write_code_planes(rows, code_planes, num_code_octets);
    body
}

/// Writes the flags, block_sizes, grid_params and value_anchor fields.
fn write_columns(rows: &Rows, body: &mut [u8]) {
    let num_blocks = rows.shape.num_blocks;
    let (sizes_start, params_start, anchors_start) = (
        block_sizes_start(num_blocks),
        grid_params_start(num_blocks),
        value_anchor_start(num_blocks),
    );
    for block_idx in 0..num_blocks {
        body[block_idx] = rows.block_flags[block_idx];
        put_byte_planed(
            body,
            sizes_start,
            num_blocks,
            block_idx,
            rows.block_sizes[block_idx],
            BYTES_PER_SIZE,
        );
        put_byte_planed(
            body,
            params_start,
            num_blocks,
            block_idx,
            rows.grid_params[block_idx],
            BYTES_PER_PARAM,
        );
        put_byte_planed(
            body,
            anchors_start,
            num_blocks,
            block_idx,
            rows.value_anchors[block_idx],
            BYTES_PER_ANCHOR,
        );
    }
}

/// Writes the zigzagged residuals as a low and a high byte plane, the padding past each block's
/// samples zero.
fn write_residual_planes(rows: &Rows, residual_planes: &mut [u8]) {
    let (low_plane, high_plane) =
        residual_planes.split_at_mut(OCTET_SAMPLES * rows.shape.num_octets);
    for block_idx in 0..rows.shape.num_blocks {
        let first_sample = rows.sample_offsets[block_idx];
        let block_len = rows.block_sizes[block_idx] as usize;
        let plane_start = OCTET_SAMPLES * rows.octet_offsets[block_idx];
        let low_bytes = &mut low_plane[plane_start..plane_start + block_len];
        let high_bytes = &mut high_plane[plane_start..plane_start + block_len];
        let block_residuals = &rows.residuals[first_sample..first_sample + block_len];
        for ((&residual, low), high) in block_residuals.iter().zip(low_bytes).zip(high_bytes) {
            let zigzagged = zigzag(residual);
            *low = zigzagged as u8;
            *high = (zigzagged >> 8) as u8;
        }
    }
}

/// Writes the non-finite codes of the flagged blocks as two code bit planes.
fn write_code_planes(rows: &Rows, code_planes: &mut [u8], num_code_octets: usize) {
    let (plane0, plane1) = code_planes.split_at_mut(num_code_octets);
    for block_idx in 0..rows.shape.num_blocks {
        if rows.block_flags[block_idx] & BLOCK_FLAG_NONFINITE == 0 {
            continue;
        }
        let first_sample = rows.sample_offsets[block_idx];
        let block_codes =
            &rows.codes[first_sample..first_sample + rows.block_sizes[block_idx] as usize];
        for (octet_idx, octet_codes) in block_codes.chunks(OCTET_SAMPLES).enumerate() {
            let (byte0, byte1) = pack_code_octet(octet_codes);
            plane0[rows.code_offsets[block_idx] + octet_idx] = byte0;
            plane1[rows.code_offsets[block_idx] + octet_idx] = byte1;
        }
    }
}

/// The two code plane bytes of up to 8 codes: bit `j` of the first is the low bit of code `j`, of
/// the second the high bit. Missing codes of a partial last octet count as 0.
fn pack_code_octet(octet_codes: &[u8]) -> (u8, u8) {
    let mut word = [0u8; OCTET_SAMPLES];
    word[..octet_codes.len()].copy_from_slice(octet_codes);
    // byte 0 of the transpose holds every code's low bit, byte 1 the high bit
    let transposed = transpose8(u64::from_le_bytes(word));
    (transposed as u8, (transposed >> 8) as u8)
}

/// The bit-plane layout of a byte-plane body: its residual region with the 8 bytes of every octet
/// transposed into one byte of each of the 16 bit planes. The rest of the body is the same
/// (`_bitpacking.to_bit_planes`).
pub fn to_bit_planes(
    byte_body: &[u8],
    num_blocks: usize,
    num_octets: usize,
    has_time: bool,
) -> Vec<u8> {
    let start = residual_start(num_blocks, has_time);
    let end = start + RESIDUAL_BYTES_PER_OCTET * num_octets;
    let mut bit_body = vec![0u8; byte_body.len()];
    bit_body[..start].copy_from_slice(&byte_body[..start]);
    bit_body[end..].copy_from_slice(&byte_body[end..]);
    let (low_plane, high_plane) = byte_body[start..end].split_at(OCTET_SAMPLES * num_octets);
    planes::pack_bit_planes(
        low_plane,
        high_plane,
        &mut bit_body[start..end],
        num_octets,
        true,
    );
    bit_body
}

/// Share of residual slots whose zigzagged residual reaches 2^bit, from a byte-plane body
/// (`_bitpacking.wide_share`). Only for `bit` of at most 8, where the high byte needs no mask.
pub fn wide_share_byte_planes(
    body: &[u8],
    num_blocks: usize,
    num_octets: usize,
    has_time: bool,
    bit: u32,
) -> f64 {
    debug_assert!(bit <= 8);
    if num_octets == 0 {
        return 0.0;
    }
    let start = residual_start(num_blocks, has_time);
    let planes = &body[start..start + BIT_PLANES * num_octets];
    let (low_plane, high_plane) = planes.split_at(OCTET_SAMPLES * num_octets);
    let low_mask = ((0xFFu32 << bit) & 0xFF) as u8;
    let wide_slots = low_plane
        .iter()
        .zip(high_plane)
        .filter(|&(&low, &high)| (low & low_mask) | high != 0)
        .count();
    wide_slots as f64 / (OCTET_SAMPLES * num_octets) as f64
}
