// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Garry Boyer
//! Packing rows into the body's bytes, and the residual layouts (`fluxcode/_bitpacking.py`).

use crate::format::{residual_start, Rows, BLOCK_FLAG_NONFINITE, METADATA_BYTES_PER_BLOCK};
use crate::planes::{self, transpose8};

#[inline(always)]
fn zigzag(r: i16) -> u16 {
    ((r << 1) ^ (r >> 15)) as u16
}

/// The uncompressed body of a unit without a time axis, in the byte-plane layout: the metadata columns,
/// the zigzagged residuals as a low and a high byte plane (each block padded with zeros to whole groups
/// of 8), then the non-finite code planes.
pub fn write_body(r: &Rows) -> Vec<u8> {
    let (n, ng, ncg) = (r.shape.n, r.shape.ng, r.co[r.shape.n]);
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
/// into one byte of each of the 16 bit planes. The rest of the body is the same
/// (`_bitpacking.to_bit_planes`).
pub fn to_bit_planes(byte_body: &[u8], n: usize, ng: usize, has_time: bool) -> Vec<u8> {
    let start = residual_start(n, has_time);
    let end = start + 16 * ng;
    let mut raw = vec![0u8; byte_body.len()];
    raw[..start].copy_from_slice(&byte_body[..start]);
    raw[end..].copy_from_slice(&byte_body[end..]);
    let (low, high) = byte_body[start..end].split_at(8 * ng);
    planes::transpose(low, high, &mut raw[start..end], ng, true);
    raw
}

/// Share of residual slots whose zigzagged residual reaches 2^bit, from a byte-plane body
/// (`_bitpacking.wide_share`).
pub fn wide_share_byte_planes(raw: &[u8], n: usize, ng: usize, has_time: bool, bit: u32) -> f64 {
    if ng == 0 {
        return 0.0;
    }
    let start = residual_start(n, has_time);
    let (low, high) = raw[start..start + 16 * ng].split_at(8 * ng);
    let low_mask = ((0xFFu32 << bit) & 0xFF) as u8;
    let count: usize = low.iter().zip(high).filter(|&(&l, &h)| (l & low_mask) | h != 0).count();
    count as f64 / (8 * ng) as f64
}
