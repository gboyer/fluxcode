// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Garry Boyer
//! Bit-plane transposition for 16-bit zigzag-encoded residuals.
//!
//! Converts the residual section from a 2-byte-plane layout (low bytes followed by high bytes,
//! grouped into 8-sample blocks) into 16 bit planes where each 8-sample group becomes 1 byte per
//! bit plane:
//!
//! ```text
//! low byte plane:   group 0 = [b0 b1 b2 b3 b4 b5 b6 b7], group 1 = [...], ...
//! bit plane k:      byte g  = [bit k of b0 .. bit k of b7] of group g
//!                   (bit j of byte g comes from sample j of group g)
//! ```
//!
//! Bit planes 0–7 are transposed from `low_plane` and bit planes 8–15 from `high_plane`.
//! On `aarch64` (NEON) and `x86_64` (SSE2), 16 groups (128 bytes per byte plane) are transposed
//! per SIMD iteration; remaining groups (and other targets) use a batched 8-group scalar path.
//! This module depends only on `std`, so `rustc --test src/planes.rs` runs its unit tests directly.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Name of the active SIMD transposition backend compiled into this target:
/// `"neon"`, `"sse2"`, or `"scalar"`.
pub const SIMD_BACKEND: &str = if cfg!(target_arch = "aarch64") {
    "neon"
} else if cfg!(all(target_arch = "x86_64", target_feature = "sse2")) {
    "sse2"
} else {
    "scalar"
};

/// Transposes an 8×8 bit matrix packed in little-endian byte order inside a `u64`.
///
/// Bit `j` of input byte `i` moves to bit `i` of output byte `j`, turning 8 sample bytes into
/// 8 single-byte bit-plane slices (one byte per bit position `0..8`).
#[inline(always)]
pub fn transpose_8x8_bits(mut word: u64) -> u64 {
    // Perform a 3-stage butterfly (delta-swap) transpose on the 8×8 bit matrix:
    // 1. Swap off-diagonal 1×1 bits inside each 2×2 block (distance 7 bits).
    // 2. Swap off-diagonal 2×2 blocks inside each 4×4 block (distance 14 bits).
    // 3. Swap the two off-diagonal 4×4 quadrants of the 8×8 matrix (distance 28 bits).

    // Stage 1: transpose each 2×2 bit sub-matrix.
    let mut delta = (word ^ (word >> 7)) & 0x00AA00AA00AA00AA;
    word = word ^ delta ^ (delta << 7);
    // Stage 2: transpose each 2×2 block of 2×2 bit sub-matrices.
    delta = (word ^ (word >> 14)) & 0x0000CCCC0000CCCC;
    word = word ^ delta ^ (delta << 14);
    // Stage 3: transpose the 2×2 grid of 4×4 bit quadrants.
    delta = (word ^ (word >> 28)) & 0x00000000F0F0F0F0;
    word ^ delta ^ (delta << 28)
}

/// Transposes an 8×8 matrix of bytes in place, where each `u64` holds one row of 8 bytes.
///
/// Byte `j` of `words[i]` moves to byte `i` of `words[j]`. After transposing 8 groups' bits,
/// this gathers the 8 output bytes belonging to each bit plane into a single contiguous `u64`.
#[inline(always)]
fn transpose_8x8_bytes(words: &mut [u64; 8]) {
    // Apply the same 3-stage butterfly structure at byte granularity: swap 1-byte sub-blocks
    // between adjacent rows, 2-byte sub-blocks across stride-2 pairs, then 4-byte halves across
    // stride-4 pairs.

    /// Swaps the `shift`-bit sub-blocks selected by `mask` between the high half of `words[i]`
    /// and the low half of `words[j]`.
    #[inline(always)]
    fn swap_subblocks(words: &mut [u64; 8], i: usize, j: usize, shift: u32, mask: u64) {
        let diff = ((words[i] >> shift) ^ words[j]) & mask;
        words[j] ^= diff;
        words[i] ^= diff << shift;
    }

    const MASK_8: u64 = 0x00FF00FF00FF00FF;
    const MASK_16: u64 = 0x0000FFFF0000FFFF;
    const MASK_32: u64 = 0x00000000FFFFFFFF;

    // Stage 1: transpose 2×2 byte blocks between adjacent rows.
    swap_subblocks(words, 0, 1, 8, MASK_8);
    swap_subblocks(words, 2, 3, 8, MASK_8);
    swap_subblocks(words, 4, 5, 8, MASK_8);
    swap_subblocks(words, 6, 7, 8, MASK_8);
    // Stage 2: transpose 2×2 blocks of 2-byte lanes across stride-2 row pairs.
    swap_subblocks(words, 0, 2, 16, MASK_16);
    swap_subblocks(words, 1, 3, 16, MASK_16);
    swap_subblocks(words, 4, 6, 16, MASK_16);
    swap_subblocks(words, 5, 7, 16, MASK_16);
    // Stage 3: transpose 4-byte quadrants across stride-4 row pairs.
    swap_subblocks(words, 0, 4, 32, MASK_32);
    swap_subblocks(words, 1, 5, 32, MASK_32);
    swap_subblocks(words, 2, 6, 32, MASK_32);
    swap_subblocks(words, 3, 7, 32, MASK_32);
}

/// Scalar fallback that transposes 8-sample groups starting at `first_group` into `bit_planes`.
///
/// Preconditions: `low_plane` and `high_plane` each have length `8 * num_plane_bytes`,
/// `bit_planes` has length `16 * num_plane_bytes`, and `first_group <= num_plane_bytes`.
fn pack_bit_planes_scalar(
    low_plane: &[u8],
    high_plane: &[u8],
    bit_planes: &mut [u8],
    num_plane_bytes: usize,
    first_group: usize,
) {
    // Process full batches of 8 groups (64 bytes per byte plane) by bit-transposing each group
    // and then byte-transposing the 8 resulting words so every bit plane gets an 8-byte contiguous
    // write. Finish any remaining tail (< 8 groups) one group at a time.
    let (low_tail, high_tail) = (
        &low_plane[8 * first_group..],
        &high_plane[8 * first_group..],
    );
    let (low_batches, low_rem) = low_tail.as_chunks::<64>();
    let (high_batches, high_rem) = high_tail.as_chunks::<64>();
    let rem_first_group = num_plane_bytes - low_rem.len() / 8;

    // Fast scalar loop: 8 groups at a time → 8 contiguous bytes written to each of the 16 planes.
    for (batch_idx, (low_batch, high_batch)) in low_batches.iter().zip(high_batches).enumerate() {
        let group = first_group + 8 * batch_idx;
        let (low_groups, _) = low_batch.as_chunks::<8>();
        let (high_groups, _) = high_batch.as_chunks::<8>();
        let mut low_words: [u64; 8] =
            std::array::from_fn(|j| transpose_8x8_bits(u64::from_le_bytes(low_groups[j])));
        let mut high_words: [u64; 8] =
            std::array::from_fn(|j| transpose_8x8_bits(u64::from_le_bytes(high_groups[j])));
        transpose_8x8_bytes(&mut low_words);
        transpose_8x8_bytes(&mut high_words);
        for plane in 0..8 {
            bit_planes[plane * num_plane_bytes + group..plane * num_plane_bytes + group + 8]
                .copy_from_slice(&low_words[plane].to_le_bytes());
            bit_planes
                [(8 + plane) * num_plane_bytes + group..(8 + plane) * num_plane_bytes + group + 8]
                .copy_from_slice(&high_words[plane].to_le_bytes());
        }
    }

    // Tail loop: scatter the remaining 0..7 groups one byte per plane.
    let (low_rem_groups, _) = low_rem.as_chunks::<8>();
    let (high_rem_groups, _) = high_rem.as_chunks::<8>();
    for (rem_idx, (&low_group, &high_group)) in
        low_rem_groups.iter().zip(high_rem_groups).enumerate()
    {
        let group = rem_first_group + rem_idx;
        let low_bytes = transpose_8x8_bits(u64::from_le_bytes(low_group)).to_le_bytes();
        let high_bytes = transpose_8x8_bits(u64::from_le_bytes(high_group)).to_le_bytes();
        for plane in 0..8 {
            bit_planes[plane * num_plane_bytes + group] = low_bytes[plane];
            bit_planes[(8 + plane) * num_plane_bytes + group] = high_bytes[plane];
        }
    }
}

/// Transposes 16 consecutive 8-sample groups (128 bytes of one byte plane) into 8 bit planes of
/// 16 bytes each using AArch64 NEON intrinsics.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn transpose_16_groups_simd(src: &[u8; 128]) -> [[u8; 16]; 8] {
    // Load 16 groups into 8 128-bit registers (even group in low 64-bit lane, odd group in high
    // lane). Run the 8×8 bit transpose independently in both lanes, then run 3 rounds of `vtrn`
    // (8-, 16-, and 32-bit) to transpose the 8×8 byte matrices of even and odd groups in parallel.
    // Finally, zip the low (even) and high (odd) 8-byte halves of each register into 16 contiguous
    // bytes per bit plane.

    /// Runs the 3-stage 8×8 bit transpose independently on both 64-bit lanes of a NEON register.
    #[inline(always)]
    unsafe fn transpose_bits_neon(regs: uint8x16_t) -> uint8x16_t {
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
        // Load two 8-sample groups per register and bit-transpose each group in place.
        let mut regs: [uint8x16_t; 8] =
            std::array::from_fn(|j| transpose_bits_neon(vld1q_u8(src.as_ptr().add(16 * j))));

        // Round 1 (8-bit `vtrn`): pair corresponding plane bytes from adjacent registers within
        // each 64-bit half (even groups stay in low 64 bits, odd groups in high 64 bits).
        for (i, j) in [(0, 1), (2, 3), (4, 5), (6, 7)] {
            (regs[i], regs[j]) = (vtrn1q_u8(regs[i], regs[j]), vtrn2q_u8(regs[i], regs[j]));
        }
        // Round 2 (16-bit `vtrn`): group 4 consecutive even (and odd) plane bytes per 32-bit lane.
        for (i, j) in [(0, 2), (1, 3), (4, 6), (5, 7)] {
            let (a, b) = (vreinterpretq_u16_u8(regs[i]), vreinterpretq_u16_u8(regs[j]));
            (regs[i], regs[j]) = (
                vreinterpretq_u8_u16(vtrn1q_u16(a, b)),
                vreinterpretq_u8_u16(vtrn2q_u16(a, b)),
            );
        }
        // Round 3 (32-bit `vtrn`): complete the 8×8 byte transpose inside each 64-bit half.
        for (i, j) in [(0, 4), (1, 5), (2, 6), (3, 7)] {
            let (a, b) = (vreinterpretq_u32_u8(regs[i]), vreinterpretq_u32_u8(regs[j]));
            (regs[i], regs[j]) = (
                vreinterpretq_u8_u32(vtrn1q_u32(a, b)),
                vreinterpretq_u8_u32(vtrn2q_u32(a, b)),
            );
        }

        // `regs[plane]` now holds `[even groups 0,2,..,14 | odd groups 1,3,..,15]` for `plane`.
        // Rotate the high half down with `vextq_u8::<8>` and interleave via `vzip1q_u8` to restore
        // sequential group order `0..16`.
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

/// Transposes 16 consecutive 8-sample groups (128 bytes of one byte plane) into 8 bit planes of
/// 16 bytes each using x86_64 SSE2 intrinsics.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn transpose_16_groups_simd(src: &[u8; 128]) -> [[u8; 16]; 8] {
    // Load 16 groups into 8 `__m128i` registers (even group in low 64-bit lane, odd group in high
    // lane) and bit-transpose each 64-bit lane. Because SSE2 lacks NEON's in-lane `vtrn`, run 3
    // rounds of low/high unpacks (at 8, 16, and 32 bits) keeping even and odd groups in separate
    // register sets, then interleave even and odd groups in a final 8-bit unpack to emit 4 planes
    // per half.

    /// Runs the 3-stage 8×8 bit transpose independently on both 64-bit lanes of an SSE2 register.
    #[inline(always)]
    unsafe fn transpose_bits_sse2(regs: __m128i) -> __m128i {
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
        // Register `j` holds groups `2j` (low 64 bits) and `2j + 1` (high 64 bits), bit-transposed.
        let regs: [__m128i; 8] = std::array::from_fn(|j| {
            transpose_bits_sse2(_mm_loadu_si128(src.as_ptr().add(16 * j) as *const __m128i))
        });

        // Round 1 (8-bit unpack): separate even groups (`unpacklo`) and odd groups (`unpackhi`),
        // interleaving plane bytes of adjacent register pairs (`4j` with `4j+2`, `4j+1` with `4j+3`).
        let even_bytes: [__m128i; 4] =
            std::array::from_fn(|j| _mm_unpacklo_epi8(regs[2 * j], regs[2 * j + 1]));
        let odd_bytes: [__m128i; 4] =
            std::array::from_fn(|j| _mm_unpackhi_epi8(regs[2 * j], regs[2 * j + 1]));

        // Round 2 (16-bit unpack): gather 4 groups per plane into `[planes 0-3, planes 4-7]`
        // for the first 4 and last 4 even (and odd) groups.
        let even_first = unpack_pair_16(even_bytes[0], even_bytes[1]);
        let even_second = unpack_pair_16(even_bytes[2], even_bytes[3]);
        let odd_first = unpack_pair_16(odd_bytes[0], odd_bytes[1]);
        let odd_second = unpack_pair_16(odd_bytes[2], odd_bytes[3]);

        let mut planes = [[0u8; 16]; 8];
        for half in 0..2 {
            // Round 3 (32-bit unpack): combine all 8 even (or odd) groups per plane. `*_01` holds
            // planes `4*half` and `4*half + 1`; `*_23` holds planes `4*half + 2` and `4*half + 3`.
            let (even_01, even_23) = unpack_pair_32(even_first[half], even_second[half]);
            let (odd_01, odd_23) = unpack_pair_32(odd_first[half], odd_second[half]);

            // Final round (8-bit unpack): interleave even and odd groups into sequential order
            // `0..16`, yielding 4 complete 16-byte bit planes.
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

/// Returns both 16-bit unpacks of two SSE2 registers as `[low, high]`.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn unpack_pair_16(a: __m128i, b: __m128i) -> [__m128i; 2] {
    [_mm_unpacklo_epi16(a, b), _mm_unpackhi_epi16(a, b)]
}

/// Returns both 32-bit unpacks of two SSE2 registers as `(low, high)`.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn unpack_pair_32(a: __m128i, b: __m128i) -> (__m128i, __m128i) {
    (_mm_unpacklo_epi32(a, b), _mm_unpackhi_epi32(a, b))
}

/// Transposes `low_plane` and `high_plane` into 16 bit planes written into `bit_planes`.
///
/// Preconditions:
/// - `low_plane.len() == 8 * num_plane_bytes`
/// - `high_plane.len() == 8 * num_plane_bytes`
/// - `bit_planes.len() == 16 * num_plane_bytes`
///
/// When `use_vector` is `false`, bypasses SIMD and runs only the scalar path (used by tests to
/// cross-check both implementations).
pub fn pack_bit_planes(
    low_plane: &[u8],
    high_plane: &[u8],
    bit_planes: &mut [u8],
    num_plane_bytes: usize,
    use_vector: bool,
) {
    // Process full 16-group (128-byte) chunks with SIMD when enabled on aarch64/x86_64, then
    // hand off any remaining groups (< 16) to the batched scalar implementation.
    let mut vectorized_groups = 0;

    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    if use_vector {
        let (low_batches, _) = low_plane.as_chunks::<128>();
        let (high_batches, _) = high_plane.as_chunks::<128>();
        for (low_batch, high_batch) in low_batches.iter().zip(high_batches) {
            let (low_planes, high_planes) = (
                transpose_16_groups_simd(low_batch),
                transpose_16_groups_simd(high_batch),
            );
            // Copy each plane's 16 contiguous bytes into its slice of `bit_planes`.
            for plane in 0..8 {
                let first = plane * num_plane_bytes + vectorized_groups;
                bit_planes[first..first + 16].copy_from_slice(&low_planes[plane]);
                let first = (8 + plane) * num_plane_bytes + vectorized_groups;
                bit_planes[first..first + 16].copy_from_slice(&high_planes[plane]);
            }
            vectorized_groups += 16;
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let _ = use_vector;

    pack_bit_planes_scalar(
        low_plane,
        high_plane,
        bit_planes,
        num_plane_bytes,
        vectorized_groups,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Generates `16 * num_groups` deterministic pseudo-random bytes using a 64-bit LCG.
    fn generate_test_data(num_groups: usize, seed: u64) -> Vec<u8> {
        let mut rng_state = seed;
        (0..16 * num_groups)
            .map(|_| {
                rng_state = rng_state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (rng_state >> 56) as u8
            })
            .collect()
    }

    /// Reference bit-by-bit transposition: bit `sample_idx` of byte `group_idx` in plane `bit_idx`
    /// is bit `bit_idx` of sample `sample_idx` in group `group_idx`.
    fn pack_bit_planes_reference(low: &[u8], high: &[u8], num_groups: usize) -> Vec<u8> {
        let mut planes = vec![0u8; 16 * num_groups];
        for group_idx in 0..num_groups {
            for sample_idx in 0..8 {
                for bit_idx in 0..8 {
                    planes[bit_idx * num_groups + group_idx] |=
                        ((low[8 * group_idx + sample_idx] >> bit_idx) & 1) << sample_idx;
                    planes[(8 + bit_idx) * num_groups + group_idx] |=
                        ((high[8 * group_idx + sample_idx] >> bit_idx) & 1) << sample_idx;
                }
            }
        }
        planes
    }

    #[test]
    fn every_path_matches_the_definition() {
        println!("simd_backend: {SIMD_BACKEND}");
        for num_groups in (0..70).chain([127, 128, 129, 1000, 7501]) {
            let data = generate_test_data(num_groups, num_groups as u64 + 1);
            let (low, high) = data.split_at(8 * num_groups);
            let expected = pack_bit_planes_reference(low, high, num_groups);
            for use_vector in [false, true] {
                let mut actual = vec![0u8; 16 * num_groups];
                pack_bit_planes(low, high, &mut actual, num_groups, use_vector);
                assert_eq!(
                    actual, expected,
                    "num_groups {num_groups} use_vector {use_vector}"
                );
            }
        }
    }
}
