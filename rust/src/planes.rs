// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Garry Boyer
//! The bit-plane transposition: the residual region of a byte-plane body (a low and a high byte
//! plane of zigzagged residuals, in octets of 8 samples) as the 16 bit planes of the other layout.
//!
//! ```text
//! low plane:   octet 0 = [b0 b1 b2 b3 b4 b5 b6 b7]  octet 1 = [...]  ...  (8 bytes per octet)
//! bit plane k: byte g  = bit k of b0..b7 of octet g, with bit j taken from byte j
//! ```
//!
//! Bit planes 0-7 come from the low plane's bytes, 8-15 from the high plane's. The scalar code
//! transposes each octet as a 64-bit word. NEON (aarch64) and SSE2 (x86_64, part of its baseline)
//! do 16 octets at a time: the same bit transpose on two words per register, then the 8x8 bytes of
//! the registers' words so that each plane takes 16 consecutive bytes in one store. Other targets
//! use the scalar code. This module uses only std, so `rustc --test src/planes.rs` runs its tests.

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Which transposition `pack_bit_planes` uses.
pub const PATH: &str = if cfg!(target_arch = "aarch64") {
    "neon"
} else if cfg!(all(target_arch = "x86_64", target_feature = "sse2")) {
    "sse2"
} else {
    "scalar"
};

/// Transposes the bits of the 8 bytes of each octet, held as a word (bit j of byte i becomes bit i
/// of byte j).
#[inline(always)]
pub fn transpose8(mut word: u64) -> u64 {
    // swap the 1-bit, 2-bit and 4-bit sub-matrices (spaced 7, 14 and 28 bits apart)
    let mut delta = (word ^ (word >> 7)) & 0x00AA00AA00AA00AA;
    word = word ^ delta ^ (delta << 7);
    delta = (word ^ (word >> 14)) & 0x0000CCCC0000CCCC;
    word = word ^ delta ^ (delta << 14);
    delta = (word ^ (word >> 28)) & 0x00000000F0F0F0F0;
    word ^ delta ^ (delta << 28)
}

/// Transposes the 8x8 matrix of bytes whose rows are the words (byte j of word i becomes byte i of
/// word j).
#[inline(always)]
fn transpose_bytes(words: &mut [u64; 8]) {
    /// Swaps the `shift`-bit blocks selected by `mask` between word `i` (high part) and word `j`
    /// (low part).
    #[inline(always)]
    fn swap(words: &mut [u64; 8], i: usize, j: usize, shift: u32, mask: u64) {
        let diff = ((words[i] >> shift) ^ words[j]) & mask;
        words[j] ^= diff;
        words[i] ^= diff << shift;
    }
    const MASK_8: u64 = 0x00FF00FF00FF00FF;
    const MASK_16: u64 = 0x0000FFFF0000FFFF;
    const MASK_32: u64 = 0x00000000FFFFFFFF;
    swap(words, 0, 1, 8, MASK_8);
    swap(words, 2, 3, 8, MASK_8);
    swap(words, 4, 5, 8, MASK_8);
    swap(words, 6, 7, 8, MASK_8);
    swap(words, 0, 2, 16, MASK_16);
    swap(words, 1, 3, 16, MASK_16);
    swap(words, 4, 6, 16, MASK_16);
    swap(words, 5, 7, 16, MASK_16);
    swap(words, 0, 4, 32, MASK_32);
    swap(words, 1, 5, 32, MASK_32);
    swap(words, 2, 6, 32, MASK_32);
    swap(words, 3, 7, 32, MASK_32);
}

/// The octets from `first_octet` on, 8 at a time and then one at a time.
fn scalar(
    low_plane: &[u8],
    high_plane: &[u8],
    bit_planes: &mut [u8],
    num_octets: usize,
    first_octet: usize,
) {
    let word = |bytes: &[u8]| u64::from_le_bytes(bytes.try_into().unwrap());
    let (low_plane, high_plane) = (
        &low_plane[8 * first_octet..],
        &high_plane[8 * first_octet..],
    );
    let (low_batches, high_batches) = (low_plane.chunks_exact(64), high_plane.chunks_exact(64));
    let (low_rest, high_rest) = (low_batches.remainder(), high_batches.remainder());
    let rest_first_octet = num_octets - low_rest.len() / 8;
    // batches of 8 octets: transpose each octet's bits, then the 8x8 bytes of the batch, so that
    // each bit plane takes 8 consecutive bytes
    for (batch_idx, (low_batch, high_batch)) in low_batches.zip(high_batches).enumerate() {
        let octet = first_octet + 8 * batch_idx;
        let mut low_words: [u64; 8] =
            std::array::from_fn(|j| transpose8(word(&low_batch[8 * j..8 * j + 8])));
        let mut high_words: [u64; 8] =
            std::array::from_fn(|j| transpose8(word(&high_batch[8 * j..8 * j + 8])));
        transpose_bytes(&mut low_words);
        transpose_bytes(&mut high_words);
        for plane in 0..8 {
            bit_planes[plane * num_octets + octet..plane * num_octets + octet + 8]
                .copy_from_slice(&low_words[plane].to_le_bytes());
            bit_planes[(8 + plane) * num_octets + octet..(8 + plane) * num_octets + octet + 8]
                .copy_from_slice(&high_words[plane].to_le_bytes());
        }
    }
    // the remaining octets one at a time
    for (rest_idx, (low_octet, high_octet)) in low_rest
        .chunks_exact(8)
        .zip(high_rest.chunks_exact(8))
        .enumerate()
    {
        let octet = rest_first_octet + rest_idx;
        let low_bytes = transpose8(word(low_octet)).to_le_bytes();
        let high_bytes = transpose8(word(high_octet)).to_le_bytes();
        for plane in 0..8 {
            bit_planes[plane * num_octets + octet] = low_bytes[plane];
            bit_planes[(8 + plane) * num_octets + octet] = high_bytes[plane];
        }
    }
}

/// The 8 bit planes of 16 octets (128 bytes of one byte plane): each plane's bytes for the 16
/// octets.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn batch16(src: &[u8; 128]) -> [[u8; 16]; 8] {
    // each register holds two octets, one per 64-bit lane
    #[inline(always)]
    unsafe fn transpose_bits(regs: uint8x16_t) -> uint8x16_t {
        let words = vreinterpretq_u64_u8(regs);
        let delta = vandq_u64(
            veorq_u64(words, vshrq_n_u64::<7>(words)),
            vdupq_n_u64(0x00AA00AA00AA00AA),
        );
        let words = veorq_u64(veorq_u64(words, delta), vshlq_n_u64::<7>(delta));
        let delta = vandq_u64(
            veorq_u64(words, vshrq_n_u64::<14>(words)),
            vdupq_n_u64(0x0000CCCC0000CCCC),
        );
        let words = veorq_u64(veorq_u64(words, delta), vshlq_n_u64::<14>(delta));
        let delta = vandq_u64(
            veorq_u64(words, vshrq_n_u64::<28>(words)),
            vdupq_n_u64(0x00000000F0F0F0F0),
        );
        vreinterpretq_u8_u64(veorq_u64(veorq_u64(words, delta), vshlq_n_u64::<28>(delta)))
    }
    unsafe {
        let mut regs: [uint8x16_t; 8] =
            std::array::from_fn(|j| transpose_bits(vld1q_u8(src.as_ptr().add(16 * j))));
        // transpose the 8x8 bytes of each lane across the registers (trn at 8, 16 and 32 bits)
        for (i, j) in [(0, 1), (2, 3), (4, 5), (6, 7)] {
            (regs[i], regs[j]) = (vtrn1q_u8(regs[i], regs[j]), vtrn2q_u8(regs[i], regs[j]));
        }
        for (i, j) in [(0, 2), (1, 3), (4, 6), (5, 7)] {
            let (a, b) = (vreinterpretq_u16_u8(regs[i]), vreinterpretq_u16_u8(regs[j]));
            (regs[i], regs[j]) = (
                vreinterpretq_u8_u16(vtrn1q_u16(a, b)),
                vreinterpretq_u8_u16(vtrn2q_u16(a, b)),
            );
        }
        for (i, j) in [(0, 4), (1, 5), (2, 6), (3, 7)] {
            let (a, b) = (vreinterpretq_u32_u8(regs[i]), vreinterpretq_u32_u8(regs[j]));
            (regs[i], regs[j]) = (
                vreinterpretq_u8_u32(vtrn1q_u32(a, b)),
                vreinterpretq_u8_u32(vtrn2q_u32(a, b)),
            );
        }
        // register k now holds byte k of the even octets (low half) and of the odd octets (high
        // half)
        let mut planes = [[0u8; 16]; 8];
        for plane in 0..8 {
            vst1q_u8(
                planes[plane].as_mut_ptr(),
                vzip1q_u8(regs[plane], vextq_u8::<8>(regs[plane], regs[plane])),
            );
        }
        planes
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn batch16(src: &[u8; 128]) -> [[u8; 16]; 8] {
    #[inline(always)]
    unsafe fn transpose_bits(regs: __m128i) -> __m128i {
        let delta = _mm_and_si128(
            _mm_xor_si128(regs, _mm_srli_epi64::<7>(regs)),
            _mm_set1_epi64x(0x00AA00AA00AA00AA),
        );
        let regs = _mm_xor_si128(_mm_xor_si128(regs, delta), _mm_slli_epi64::<7>(delta));
        let delta = _mm_and_si128(
            _mm_xor_si128(regs, _mm_srli_epi64::<14>(regs)),
            _mm_set1_epi64x(0x0000CCCC0000CCCC),
        );
        let regs = _mm_xor_si128(_mm_xor_si128(regs, delta), _mm_slli_epi64::<14>(delta));
        let delta = _mm_and_si128(
            _mm_xor_si128(regs, _mm_srli_epi64::<28>(regs)),
            _mm_set1_epi64x(0x00000000F0F0F0F0),
        );
        _mm_xor_si128(_mm_xor_si128(regs, delta), _mm_slli_epi64::<28>(delta))
    }
    unsafe {
        // register j holds octets 2j (low half) and 2j+1 (high half), their bits transposed
        let regs: [__m128i; 8] = std::array::from_fn(|j| {
            transpose_bits(_mm_loadu_si128(src.as_ptr().add(16 * j) as *const __m128i))
        });
        // The bytes of the octets form an 8x8 byte matrix per octet; transposing it takes three
        // rounds of unpacking, at 8, 16 and 32 bits, with the even and odd octets kept apart.
        //
        // Round 1 (8 bits): interleave the bytes of register pairs. `even_bytes[j]` holds octets
        // 4j and 4j+2, `odd_bytes[j]` octets 4j+1 and 4j+3.
        let even_bytes: [__m128i; 4] =
            std::array::from_fn(|j| _mm_unpacklo_epi8(regs[2 * j], regs[2 * j + 1]));
        let odd_bytes: [__m128i; 4] =
            std::array::from_fn(|j| _mm_unpackhi_epi8(regs[2 * j], regs[2 * j + 1]));
        // Round 2 (16 bits): [bytes 0-3 | bytes 4-7] of octets 0,2,4,6 (`even_first`) and of
        // octets 8,10,12,14 (`even_second`), and likewise for the odd octets.
        let even_first = unpack_pair_16(even_bytes[0], even_bytes[1]);
        let even_second = unpack_pair_16(even_bytes[2], even_bytes[3]);
        let odd_first = unpack_pair_16(odd_bytes[0], odd_bytes[1]);
        let odd_second = unpack_pair_16(odd_bytes[2], odd_bytes[3]);
        let mut planes = [[0u8; 16]; 8];
        for half in 0..2 {
            // Round 3 (32 bits): of the even octets (and likewise the odd ones), `_01` holds bytes
            // 4 * half and 4 * half + 1 of all 8 octets, `_23` bytes 4 * half + 2 and + 3
            let (even_01, even_23) = unpack_pair_32(even_first[half], even_second[half]);
            let (odd_01, odd_23) = unpack_pair_32(odd_first[half], odd_second[half]);
            // Final round (8 bits): interleave even and odd octets, giving 4 planes of 16 bytes
            // each
            let four_planes = [
                _mm_unpacklo_epi8(even_01, odd_01),
                _mm_unpackhi_epi8(even_01, odd_01),
                _mm_unpacklo_epi8(even_23, odd_23),
                _mm_unpackhi_epi8(even_23, odd_23),
            ];
            for (plane_idx, plane) in four_planes.into_iter().enumerate() {
                _mm_storeu_si128(
                    planes[4 * half + plane_idx].as_mut_ptr() as *mut __m128i,
                    plane,
                );
            }
        }
        planes
    }
}

/// Both 16-bit unpacks of two registers, as `[low, high]`.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn unpack_pair_16(a: __m128i, b: __m128i) -> [__m128i; 2] {
    [_mm_unpacklo_epi16(a, b), _mm_unpackhi_epi16(a, b)]
}

/// Both 32-bit unpacks of two registers, as `(low, high)`.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn unpack_pair_32(a: __m128i, b: __m128i) -> (__m128i, __m128i) {
    (_mm_unpacklo_epi32(a, b), _mm_unpackhi_epi32(a, b))
}

/// Fills the 16 bit planes (`bit_planes`, 16 * num_octets bytes) from the low and high byte planes
/// (8 * num_octets bytes each). With `use_vector` false the scalar code does it all (for tests).
pub fn pack_bit_planes(
    low_plane: &[u8],
    high_plane: &[u8],
    bit_planes: &mut [u8],
    num_octets: usize,
    use_vector: bool,
) {
    let mut vectorized_octets = 0;
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    if use_vector {
        for (low_batch, high_batch) in low_plane
            .chunks_exact(128)
            .zip(high_plane.chunks_exact(128))
        {
            let (low_planes, high_planes) = (
                batch16(low_batch.try_into().unwrap()),
                batch16(high_batch.try_into().unwrap()),
            );
            for plane in 0..8 {
                let first = plane * num_octets + vectorized_octets;
                bit_planes[first..first + 16].copy_from_slice(&low_planes[plane]);
                let first = (8 + plane) * num_octets + vectorized_octets;
                bit_planes[first..first + 16].copy_from_slice(&high_planes[plane]);
            }
            vectorized_octets += 16;
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let _ = use_vector;
    scalar(
        low_plane,
        high_plane,
        bit_planes,
        num_octets,
        vectorized_octets,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(ng: usize, seed: u64) -> Vec<u8> {
        let mut s = seed;
        (0..16 * ng)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (s >> 56) as u8
            })
            .collect()
    }

    /// Plane k, byte g, bit j is bit k of sample j of octet g.
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
                pack_bit_planes(low, high, &mut planes, ng, vector);
                assert_eq!(planes, want, "ng {ng} vector {vector}");
            }
        }
    }
}
