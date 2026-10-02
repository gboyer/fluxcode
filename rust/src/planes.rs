// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Garry Boyer
//! The bit-plane transposition: the residual region of a byte-plane body (a low and a high byte plane of
//! zigzagged residuals, in groups of 8) as the 16 bit planes of the other layout.
//!
//! Group g of the low plane holds 8 residual bytes; byte g of bit plane k holds bit k of those 8 bytes (bit j
//! for sample j). The scalar code transposes each group as a 64-bit word. NEON (aarch64) and SSE2 (x86_64,
//! part of its baseline) do 16 groups at a time: the same bit transpose on two words per register, then the
//! 8x8 bytes of the registers' words so that each plane takes 16 consecutive bytes in one store. Other targets
//! use the scalar code. This module uses only std, so `rustc --test src/planes.rs` runs its tests.

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Which transposition `transpose` uses.
pub const PATH: &str = if cfg!(target_arch = "aarch64") {
    "neon"
} else if cfg!(all(target_arch = "x86_64", target_feature = "sse2")) {
    "sse2"
} else {
    "scalar"
};

/// Transposes the bits of each group of 8 bytes held as a word (bit j of byte i becomes bit i of byte j).
#[inline(always)]
pub fn transpose8(mut x: u64) -> u64 {
    let mut d = (x ^ (x >> 7)) & 0x00AA00AA00AA00AA;
    x = x ^ d ^ (d << 7);
    d = (x ^ (x >> 14)) & 0x0000CCCC0000CCCC;
    x = x ^ d ^ (d << 14);
    d = (x ^ (x >> 28)) & 0x00000000F0F0F0F0;
    x ^ d ^ (d << 28)
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

/// The groups from `first` on, 8 at a time and then one at a time.
fn scalar(low: &[u8], high: &[u8], planes: &mut [u8], ng: usize, first: usize) {
    let word = |bytes: &[u8]| u64::from_le_bytes(bytes.try_into().unwrap());
    let (low, high) = (&low[8 * first..], &high[8 * first..]);
    let (low_batches, high_batches) = (low.chunks_exact(64), high.chunks_exact(64));
    let (low_rest, high_rest) = (low_batches.remainder(), high_batches.remainder());
    let rest_start = ng - low_rest.len() / 8;
    for (i, (lw, hw)) in low_batches.zip(high_batches).enumerate() {
        let g = first + 8 * i;
        let mut lo: [u64; 8] = std::array::from_fn(|j| transpose8(word(&lw[8 * j..8 * j + 8])));
        let mut hi: [u64; 8] = std::array::from_fn(|j| transpose8(word(&hw[8 * j..8 * j + 8])));
        transpose_bytes(&mut lo);
        transpose_bytes(&mut hi);
        for k in 0..8 {
            planes[k * ng + g..k * ng + g + 8].copy_from_slice(&lo[k].to_le_bytes());
            planes[(8 + k) * ng + g..(8 + k) * ng + g + 8].copy_from_slice(&hi[k].to_le_bytes());
        }
    }
    for (i, (lw, hw)) in low_rest.chunks_exact(8).zip(high_rest.chunks_exact(8)).enumerate() {
        let (g, lo, hi) = (rest_start + i, transpose8(word(lw)).to_le_bytes(), transpose8(word(hw)).to_le_bytes());
        for k in 0..8 {
            planes[k * ng + g] = lo[k];
            planes[(8 + k) * ng + g] = hi[k];
        }
    }
}

/// The 8 planes of 16 groups (128 bytes): plane k's bytes for the 16 groups.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn batch16(src: &[u8; 128]) -> [[u8; 16]; 8] {
    // each register holds two groups, one per 64-bit lane
    #[inline(always)]
    unsafe fn bits(x: uint8x16_t) -> uint8x16_t {
        let x = vreinterpretq_u64_u8(x);
        let d = vandq_u64(veorq_u64(x, vshrq_n_u64::<7>(x)), vdupq_n_u64(0x00AA00AA00AA00AA));
        let x = veorq_u64(veorq_u64(x, d), vshlq_n_u64::<7>(d));
        let d = vandq_u64(veorq_u64(x, vshrq_n_u64::<14>(x)), vdupq_n_u64(0x0000CCCC0000CCCC));
        let x = veorq_u64(veorq_u64(x, d), vshlq_n_u64::<14>(d));
        let d = vandq_u64(veorq_u64(x, vshrq_n_u64::<28>(x)), vdupq_n_u64(0x00000000F0F0F0F0));
        vreinterpretq_u8_u64(veorq_u64(veorq_u64(x, d), vshlq_n_u64::<28>(d)))
    }
    unsafe {
        let mut r: [uint8x16_t; 8] = std::array::from_fn(|j| bits(vld1q_u8(src.as_ptr().add(16 * j))));
        // 8x8 bytes of each lane, across the registers (trn at 8, 16 and 32 bits)
        for (i, j) in [(0, 1), (2, 3), (4, 5), (6, 7)] {
            (r[i], r[j]) = (vtrn1q_u8(r[i], r[j]), vtrn2q_u8(r[i], r[j]));
        }
        for (i, j) in [(0, 2), (1, 3), (4, 6), (5, 7)] {
            let (a, b) = (vreinterpretq_u16_u8(r[i]), vreinterpretq_u16_u8(r[j]));
            (r[i], r[j]) = (vreinterpretq_u8_u16(vtrn1q_u16(a, b)), vreinterpretq_u8_u16(vtrn2q_u16(a, b)));
        }
        for (i, j) in [(0, 4), (1, 5), (2, 6), (3, 7)] {
            let (a, b) = (vreinterpretq_u32_u8(r[i]), vreinterpretq_u32_u8(r[j]));
            (r[i], r[j]) = (vreinterpretq_u8_u32(vtrn1q_u32(a, b)), vreinterpretq_u8_u32(vtrn2q_u32(a, b)));
        }
        // register k now holds byte k of the even groups (low half) and of the odd groups (high half)
        let mut out = [[0u8; 16]; 8];
        for k in 0..8 {
            vst1q_u8(out[k].as_mut_ptr(), vzip1q_u8(r[k], vextq_u8::<8>(r[k], r[k])));
        }
        out
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn batch16(src: &[u8; 128]) -> [[u8; 16]; 8] {
    #[inline(always)]
    unsafe fn bits(x: __m128i) -> __m128i {
        let d = _mm_and_si128(_mm_xor_si128(x, _mm_srli_epi64::<7>(x)), _mm_set1_epi64x(0x00AA00AA00AA00AA));
        let x = _mm_xor_si128(_mm_xor_si128(x, d), _mm_slli_epi64::<7>(d));
        let d = _mm_and_si128(_mm_xor_si128(x, _mm_srli_epi64::<14>(x)), _mm_set1_epi64x(0x0000CCCC0000CCCC));
        let x = _mm_xor_si128(_mm_xor_si128(x, d), _mm_slli_epi64::<14>(d));
        let d = _mm_and_si128(_mm_xor_si128(x, _mm_srli_epi64::<28>(x)), _mm_set1_epi64x(0x00000000F0F0F0F0));
        _mm_xor_si128(_mm_xor_si128(x, d), _mm_slli_epi64::<28>(d))
    }
    unsafe {
        // register j holds groups 2j (low half) and 2j+1 (high half), their bits transposed
        let r: [__m128i; 8] = std::array::from_fn(|j| bits(_mm_loadu_si128(src.as_ptr().add(16 * j) as *const __m128i)));
        // bytes of the groups as 8-byte rows; unpack at 8, 16 and 32 bits, even and odd groups apart
        let a: [__m128i; 4] = std::array::from_fn(|j| _mm_unpacklo_epi8(r[2 * j], r[2 * j + 1])); // groups 4j, 4j+2
        let b: [__m128i; 4] = std::array::from_fn(|j| _mm_unpackhi_epi8(r[2 * j], r[2 * j + 1])); // groups 4j+1, 4j+3
        let mut out = [[0u8; 16]; 8];
        // [bytes 0-3 of groups 0,2,4,6 | .. 4-7], then of groups 8..14
        let (c, d) = ([_mm_unpacklo_epi16(a[0], a[1]), _mm_unpackhi_epi16(a[0], a[1])], [_mm_unpacklo_epi16(a[2], a[3]), _mm_unpackhi_epi16(a[2], a[3])]);
        let (e, f) = ([_mm_unpacklo_epi16(b[0], b[1]), _mm_unpackhi_epi16(b[0], b[1])], [_mm_unpacklo_epi16(b[2], b[3]), _mm_unpackhi_epi16(b[2], b[3])]);
        for h in 0..2 {
            // byte pairs k = 4h..4h+4 of the even groups (ev) and the odd groups (od), 8 bytes each
            let (ev_01, ev_23) = (_mm_unpacklo_epi32(c[h], d[h]), _mm_unpackhi_epi32(c[h], d[h]));
            let (od_01, od_23) = (_mm_unpacklo_epi32(e[h], f[h]), _mm_unpackhi_epi32(e[h], f[h]));
            let planes = [
                _mm_unpacklo_epi8(ev_01, od_01),
                _mm_unpackhi_epi8(ev_01, od_01),
                _mm_unpacklo_epi8(ev_23, od_23),
                _mm_unpackhi_epi8(ev_23, od_23),
            ];
            for (i, p) in planes.into_iter().enumerate() {
                _mm_storeu_si128(out[4 * h + i].as_mut_ptr() as *mut __m128i, p);
            }
        }
        out
    }
}

/// Fills the 16 bit planes (`planes`, 16 * ng bytes) from the low and high byte planes (8 * ng bytes each).
/// With `vector` false the scalar code does it all (for tests).
pub fn transpose(low: &[u8], high: &[u8], planes: &mut [u8], ng: usize, vector: bool) {
    let mut g = 0;
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    if vector {
        for (lw, hw) in low.chunks_exact(128).zip(high.chunks_exact(128)) {
            let (lo, hi) = (batch16(lw.try_into().unwrap()), batch16(hw.try_into().unwrap()));
            for k in 0..8 {
                planes[k * ng + g..k * ng + g + 16].copy_from_slice(&lo[k]);
                planes[(8 + k) * ng + g..(8 + k) * ng + g + 16].copy_from_slice(&hi[k]);
            }
            g += 16;
        }
    }
    let _ = vector;
    scalar(low, high, planes, ng, g);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(ng: usize, seed: u64) -> Vec<u8> {
        let mut s = seed;
        (0..16 * ng)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (s >> 56) as u8
            })
            .collect()
    }

    /// Plane k, byte g, bit j is bit k of sample j of group g.
    fn naive(low: &[u8], high: &[u8], ng: usize) -> Vec<u8> {
        let mut planes = vec![0u8; 16 * ng];
        for g in 0..ng {
            for j in 0..8 {
                for k in 0..8 {
                    planes[k * ng + g] |= ((low[8 * g + j] >> k) & 1) << j;
                    planes[(8 + k) * ng + g] |= ((high[8 * g + j] >> k) & 1) << j;
                }
            }
        }
        planes
    }

    #[test]
    fn every_path_matches_the_definition() {
        println!("path: {PATH}");
        for ng in (0..70).chain([127, 128, 129, 1000, 7501]) {
            let d = data(ng, ng as u64 + 1);
            let (low, high) = d.split_at(8 * ng);
            let want = naive(low, high, ng);
            for vector in [false, true] {
                let mut planes = vec![0u8; 16 * ng];
                transpose(low, high, &mut planes, ng, vector);
                assert_eq!(planes, want, "ng {ng} vector {vector}");
            }
        }
    }
}
