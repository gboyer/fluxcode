# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Low-level bit-plane and byte-plane codecs for residuals, codes, and time residuals.

Each block's plane data is aligned to 8-sample plane bytes
(``SAMPLES_PER_PLANE_BYTE = 8``), so 8x8 bit-matrix transposes operate on whole 64-bit
words and each block can be packed or unpacked independently.

Higher-level body serialization, time-column validation, and binary body splicing live
in ``_body``.
"""

import numpy as np
from numba import njit

from ._format import (
    BITS_PER_SAMPLE_CODE,
    BYTES_PER_RESIDUAL_SAMPLE,
    SAMPLES_PER_PLANE_BYTE,
    TIME_LOW_PLANES,
    plane_byte_count,
    residual_planes_start,
    sample_code_planes_start,
)

# ======================================================================================
# Core 8x8 bit-plane transpose primitives
# ======================================================================================


@njit(inline="always")
def _transpose_8x8_bits(matrix_bits: np.uint64) -> np.uint64:
    """Transposes an 8x8 bit matrix stored in a uint64 word (self-inverse).

    Maps bit ``(byte k, bit j)`` to ``(byte j, bit k)``.
    """
    # Classic Hacker's Delight 3-stage delta-swap transpose (1x1 -> 2x2 -> 4x4).
    delta = (matrix_bits ^ (matrix_bits >> np.uint64(7))) & np.uint64(
        0x00AA00AA00AA00AA
    )
    matrix_bits = matrix_bits ^ delta ^ (delta << np.uint64(7))
    delta = (matrix_bits ^ (matrix_bits >> np.uint64(14))) & np.uint64(
        0x0000CCCC0000CCCC
    )
    matrix_bits = matrix_bits ^ delta ^ (delta << np.uint64(14))
    delta = (matrix_bits ^ (matrix_bits >> np.uint64(28))) & np.uint64(
        0x00000000F0F0F0F0
    )
    return matrix_bits ^ delta ^ (delta << np.uint64(28))


@njit(inline="always")
def _scatter_8_bit_planes(
    word: np.uint64, bit_planes: np.ndarray, row_start: int, col_idx: int
) -> None:
    """Transposes an 8-byte word and writes its 8 bytes into rows ``row_start..+8``."""
    transposed = _transpose_8x8_bits(word)
    for bit_idx in range(8):
        bit_planes[row_start + bit_idx, col_idx] = np.uint8(
            (transposed >> np.uint64(8 * bit_idx)) & np.uint64(0xFF)
        )


@njit(inline="always")
def _gather_8_bit_planes(
    bit_planes: np.ndarray, row_start: int, col_idx: int
) -> np.uint64:
    """Gathers 8 bytes from rows ``row_start..+8`` at ``col_idx`` and transposes."""
    gathered = np.uint64(0)
    for bit_idx in range(8):
        gathered |= np.uint64(bit_planes[row_start + bit_idx, col_idx]) << np.uint64(
            8 * bit_idx
        )
    return _transpose_8x8_bits(gathered)


@njit(inline="always")
def _shuffle_byte_plane(
    byte_plane: np.ndarray,
    bit_planes: np.ndarray,
    row_start: int,
    plane_byte_offset: int,
    num_plane_bytes: int,
) -> None:
    """Transposes ``num_plane_bytes`` 8-byte chunks into 8 consecutive bit planes."""
    words = byte_plane[: SAMPLES_PER_PLANE_BYTE * num_plane_bytes].view(np.uint64)
    for group_idx in range(num_plane_bytes):
        _scatter_8_bit_planes(
            words[group_idx], bit_planes, row_start, plane_byte_offset + group_idx
        )


@njit(inline="always")
def _unshuffle_byte_plane(
    bit_planes: np.ndarray,
    row_start: int,
    plane_byte_offset: int,
    num_plane_bytes: int,
    out_byte_plane: np.ndarray,
) -> None:
    """Transposes 8 consecutive bit planes back into contiguous 8-byte chunks."""
    words = out_byte_plane.view(np.uint64)
    for group_idx in range(num_plane_bytes):
        words[group_idx] = _gather_8_bit_planes(
            bit_planes, row_start, plane_byte_offset + group_idx
        )


# ======================================================================================
# 16-bit value residual planes & zigzag encoding
# ======================================================================================


@njit(inline="always")
def residual_bit_planes_view(
    body: np.ndarray, num_blocks: int, num_plane_bytes: int, has_time: bool
) -> np.ndarray:
    """Returns a ``(16, num_plane_bytes)`` view of the 16 residual bit planes."""
    start = residual_planes_start(num_blocks, has_time)
    size = BYTES_PER_RESIDUAL_SAMPLE * SAMPLES_PER_PLANE_BYTE * num_plane_bytes
    return body[start : start + size].reshape(16, num_plane_bytes)


@njit(inline="always")
def residual_byte_planes_view(
    body: np.ndarray, num_blocks: int, num_plane_bytes: int, has_time: bool
) -> np.ndarray:
    """Returns a ``(2, 8 * num_plane_bytes)`` view of low/high residual byte planes."""
    start = residual_planes_start(num_blocks, has_time)
    size = BYTES_PER_RESIDUAL_SAMPLE * SAMPLES_PER_PLANE_BYTE * num_plane_bytes
    return body[start : start + size].reshape(
        2, SAMPLES_PER_PLANE_BYTE * num_plane_bytes
    )


@njit(nogil=True, cache=True)
def wide_residual_fraction(
    body: np.ndarray,
    num_blocks: int,
    num_plane_bytes: int,
    has_time: bool,
    bit_idx: int,
) -> float:
    """Computes the fraction of residual slots whose zigzagged value reaches
    ``2^bit_idx``.

    Slots include trailing zero padding (8 slots per plane byte). Used by the
    compression heuristic to choose between byte planes and bit planes.
    """
    # SWAR (SIMD Within A Register) on uint64 words: mask bits >= bit_idx in each byte
    # lane, OR-fold each lane into its bit 0, then sum the 8 lane flags in one multiply.
    if not num_plane_bytes:
        return 0.0
    byte_planes = residual_byte_planes_view(body, num_blocks, num_plane_bytes, has_time)
    low_words = byte_planes[0].view(np.uint64)
    high_words = byte_planes[1].view(np.uint64)
    ones = np.uint64(0x0101010101010101)
    low_mask = np.uint64(
        ((0xFF << bit_idx) & 0xFF if bit_idx < 8 else 0) * 0x0101010101010101
    )
    high_mask = np.uint64(((0xFF << max(bit_idx - 8, 0)) & 0xFF) * 0x0101010101010101)
    wide_count = 0
    for word_idx in range(num_plane_bytes):
        word = (low_words[word_idx] & low_mask) | (high_words[word_idx] & high_mask)
        word |= word >> np.uint64(4)
        word |= word >> np.uint64(2)
        word |= word >> np.uint64(1)
        wide_count += int(((word & ones) * ones) >> np.uint64(56))
    return wide_count / (SAMPLES_PER_PLANE_BYTE * num_plane_bytes)


@njit(inline="always")
def zigzag_encode_block(
    residuals: np.ndarray, out_low_bytes: np.ndarray, out_high_bytes: np.ndarray
) -> None:
    """Zigzag-encodes signed int16 residuals and splits them into low/high bytes."""
    for sample_idx in range(residuals.shape[0]):
        signed_val = residuals[sample_idx]
        zigzag = np.uint16((signed_val << np.int16(1)) ^ (signed_val >> np.int16(15)))
        out_low_bytes[sample_idx] = np.uint8(zigzag)
        out_high_bytes[sample_idx] = np.uint8(zigzag >> np.uint16(8))


@njit(inline="always")
def unzigzag_sample(
    low_bytes: np.ndarray, high_bytes: np.ndarray, sample_idx: int
) -> np.int16:
    """Decodes one signed int16 residual from low and high zigzag byte arrays."""
    zigzag = np.uint16(low_bytes[sample_idx]) | (
        np.uint16(high_bytes[sample_idx]) << np.uint16(8)
    )
    return np.int16(zigzag >> np.uint16(1)) ^ -np.int16(zigzag & np.uint16(1))


@njit(nogil=True, cache=True)
def shuffle_residual_bytes(
    low_bytes: np.ndarray,
    high_bytes: np.ndarray,
    bit_planes: np.ndarray,
    plane_byte_offset: int,
    num_plane_bytes: int,
) -> None:
    """Transposes zigzagged low/high byte arrays into 16 residual bit planes."""
    _shuffle_byte_plane(low_bytes, bit_planes, 0, plane_byte_offset, num_plane_bytes)
    _shuffle_byte_plane(high_bytes, bit_planes, 8, plane_byte_offset, num_plane_bytes)


@njit(nogil=True, cache=True)
def unshuffle_residual_block(
    bit_planes: np.ndarray,
    plane_byte_offset: int,
    num_plane_bytes: int,
    out_low_bytes: np.ndarray,
    out_high_bytes: np.ndarray,
) -> None:
    """Reconstructs a block's zigzagged low/high byte arrays from 16 bit planes."""
    _unshuffle_byte_plane(
        bit_planes, 0, plane_byte_offset, num_plane_bytes, out_low_bytes
    )
    _unshuffle_byte_plane(
        bit_planes, 8, plane_byte_offset, num_plane_bytes, out_high_bytes
    )


@njit(nogil=True, cache=True)
def _convert_residual_planes(
    src_body: np.ndarray,
    num_blocks: int,
    num_plane_bytes: int,
    has_time: bool,
    to_bits: bool,
    dst_body: np.ndarray,
) -> None:
    """Converts the residual region between 2 byte planes and 16 bit planes."""
    if not num_plane_bytes:
        return
    if to_bits:
        byte_planes = residual_byte_planes_view(
            src_body, num_blocks, num_plane_bytes, has_time
        )
        bit_planes = residual_bit_planes_view(
            dst_body, num_blocks, num_plane_bytes, has_time
        )
        shuffle_residual_bytes(
            byte_planes[0], byte_planes[1], bit_planes, 0, num_plane_bytes
        )
    else:
        bit_planes = residual_bit_planes_view(
            src_body, num_blocks, num_plane_bytes, has_time
        )
        byte_planes = residual_byte_planes_view(
            dst_body, num_blocks, num_plane_bytes, has_time
        )
        unshuffle_residual_block(
            bit_planes, 0, num_plane_bytes, byte_planes[0], byte_planes[1]
        )


def to_bit_planes(
    byte_body: np.ndarray, num_blocks: int, num_plane_bytes: int, has_time: bool
) -> np.ndarray:
    """Converts a byte-plane uncompressed body into a bit-plane uncompressed body."""
    bit_body = byte_body.copy()
    _convert_residual_planes(
        byte_body, num_blocks, num_plane_bytes, has_time, True, bit_body
    )
    return bit_body


def to_byte_planes(
    bit_body: np.ndarray, num_blocks: int, num_plane_bytes: int, has_time: bool
) -> np.ndarray:
    """Converts a bit-plane uncompressed body into a byte-plane uncompressed body."""
    byte_body = bit_body.copy()
    _convert_residual_planes(
        bit_body, num_blocks, num_plane_bytes, has_time, False, byte_body
    )
    return byte_body


# ======================================================================================
# 2-bit sample exception code planes
# ======================================================================================


@njit(inline="always")
def sample_code_planes_view(
    body: np.ndarray,
    num_blocks: int,
    num_plane_bytes: int,
    num_code_plane_bytes: int,
    has_time: bool,
) -> np.ndarray:
    """Returns a ``(2, num_code_plane_bytes)`` view of the 2-bit sample code planes."""
    start = sample_code_planes_start(num_blocks, num_plane_bytes, has_time)
    total_bytes = num_code_plane_bytes * BITS_PER_SAMPLE_CODE
    return body[start : start + total_bytes].reshape(2, num_code_plane_bytes)


@njit(inline="always")
def _pack_sample_code_byte(
    sample_codes: np.ndarray,
    group_idx: int,
    num_valid: int,
    code_planes: np.ndarray,
    plane_byte_offset: int,
) -> None:
    """Packs up to 8 two-bit sample codes into one byte of each code plane."""
    plane0_byte = 0
    plane1_byte = 0
    for bit_idx in range(num_valid):
        code_val = sample_codes[SAMPLES_PER_PLANE_BYTE * group_idx + bit_idx]
        plane0_byte |= (code_val & 1) << bit_idx
        plane1_byte |= ((code_val >> 1) & 1) << bit_idx
    code_planes[0, plane_byte_offset + group_idx] = plane0_byte
    code_planes[1, plane_byte_offset + group_idx] = plane1_byte


@njit(nogil=True, cache=True)
def pack_sample_codes(
    sample_codes: np.ndarray, code_planes: np.ndarray, plane_byte_offset: int
) -> None:
    """Packs a block's 2-bit sample exception codes into two bit planes."""
    # Split full 8-sample groups from the optional partial tail so the inner loop over
    # 8 bits has a constant trip count and vectorizes cleanly under Numba/LLVM.
    num_samples = sample_codes.shape[0]
    num_full_groups = num_samples // SAMPLES_PER_PLANE_BYTE
    for group_idx in range(num_full_groups):
        _pack_sample_code_byte(
            sample_codes,
            group_idx,
            SAMPLES_PER_PLANE_BYTE,
            code_planes,
            plane_byte_offset,
        )
    remainder = num_samples % SAMPLES_PER_PLANE_BYTE
    if remainder:
        _pack_sample_code_byte(
            sample_codes, num_full_groups, remainder, code_planes, plane_byte_offset
        )


@njit(inline="always")
def unpack_sample_code(plane0_byte: int, plane1_byte: int, bit_idx: int) -> int:
    """Extracts one sample's 2-bit exception code from a pair of code-plane bytes."""
    return ((plane0_byte >> bit_idx) & 1) | (((plane1_byte >> bit_idx) & 1) << 1)


# ======================================================================================
# 64-bit time residual bit planes
# ======================================================================================


@njit(inline="always")
def time_residual_planes_views(
    body: np.ndarray,
    num_blocks: int,
    num_plane_bytes: int,
    num_code_plane_bytes: int,
    num_time_low_bytes: int,
    num_time_high_bytes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns 2D views of low (0-31) and high (32-63) time residual bit planes."""
    low_start = (
        sample_code_planes_start(num_blocks, num_plane_bytes, True)
        + num_code_plane_bytes * BITS_PER_SAMPLE_CODE
    )
    low_bytes = TIME_LOW_PLANES * num_time_low_bytes
    high_bytes = TIME_LOW_PLANES * num_time_high_bytes
    time_low_planes = body[low_start : low_start + low_bytes].reshape(
        TIME_LOW_PLANES, num_time_low_bytes
    )
    high_start = low_start + low_bytes
    time_high_planes = body[high_start : high_start + high_bytes].reshape(
        TIME_LOW_PLANES, num_time_high_bytes
    )
    return time_low_planes, time_high_planes


@njit(inline="always")
def _gather_byte_lane(values: np.ndarray, base: int, shift: np.uint64) -> np.uint64:
    """Packs byte ``shift // 8`` from ``values[base..base+8]`` into one uint64 word."""
    mask = np.uint64(0xFF)
    return np.uint64(
        ((values[base] >> shift) & mask)
        | (((values[base + 1] >> shift) & mask) << np.uint64(8))
        | (((values[base + 2] >> shift) & mask) << np.uint64(16))
        | (((values[base + 3] >> shift) & mask) << np.uint64(24))
        | (((values[base + 4] >> shift) & mask) << np.uint64(32))
        | (((values[base + 5] >> shift) & mask) << np.uint64(40))
        | (((values[base + 6] >> shift) & mask) << np.uint64(48))
        | (((values[base + 7] >> shift) & mask) << np.uint64(56))
    )


@njit(inline="always")
def _time_byte_level_planes(
    time_low_planes: np.ndarray,
    time_high_planes: np.ndarray,
    time_low_offset: int,
    time_high_offset: int,
    byte_idx: int,
    first_group: int,
    end_group: int,
) -> np.ndarray:
    """Returns an ``(8, end_group - first_group)`` view of 8 bit planes for one byte."""
    if byte_idx < 4:
        return time_low_planes[
            8 * byte_idx : 8 * byte_idx + 8,
            time_low_offset + first_group : time_low_offset + end_group,
        ]
    high_level = byte_idx - 4
    return time_high_planes[
        8 * high_level : 8 * high_level + 8,
        time_high_offset + first_group : time_high_offset + end_group,
    ]


@njit(inline="always")
def _unpack_time_plane_byte(
    level_planes: np.ndarray, group_idx: int, byte_idx: int, out_bytes: np.ndarray
) -> None:
    """Unpacks column ``group_idx`` of 8 bit planes into byte ``byte_idx`` of 8 u64s."""
    transposed = _gather_8_bit_planes(level_planes, 0, group_idx)
    for sample_offset in range(SAMPLES_PER_PLANE_BYTE):
        out_bytes[byte_idx + 8 * sample_offset] = np.uint8(
            (transposed >> np.uint64(8 * sample_offset)) & np.uint64(0xFF)
        )


@njit(nogil=True, cache=True)
def shuffle_time_residuals(
    time_residuals: np.ndarray,
    time_low_planes: np.ndarray,
    time_high_planes: np.ndarray,
    time_low_offset: int,
    time_high_offset: int,
    scratch_tail: np.ndarray,
) -> None:
    """Packs a block's zigzagged uint64 time residuals into time residual bit planes.

    Only byte levels up to the highest set bit of any residual are written; callers
    must zero the target planes beforehand so unwritten upper planes remain zero.
    """
    num_samples = time_residuals.shape[0]
    num_full_groups = num_samples // SAMPLES_PER_PLANE_BYTE
    num_plane_bytes = plane_byte_count(num_samples)
    # Upper bytes that are zero in every sample can be skipped (planes are pre-zeroed).
    all_bits = np.uint64(0)
    for sample_idx in range(num_samples):
        all_bits |= time_residuals[sample_idx]
    num_active_bytes = 0
    while num_active_bytes < 8 and (all_bits >> np.uint64(8 * num_active_bytes)) != 0:
        num_active_bytes += 1
    if num_plane_bytes > num_full_groups:
        scratch_tail[:] = 0
        tail_start = SAMPLES_PER_PLANE_BYTE * num_full_groups
        scratch_tail[: num_samples - tail_start] = time_residuals[tail_start:]
    for byte_idx in range(num_active_bytes):
        level_planes = _time_byte_level_planes(
            time_low_planes,
            time_high_planes,
            time_low_offset,
            time_high_offset,
            byte_idx,
            0,
            num_plane_bytes,
        )
        shift = np.uint64(8 * byte_idx)
        for group_idx in range(num_full_groups):
            word = _gather_byte_lane(
                time_residuals, SAMPLES_PER_PLANE_BYTE * group_idx, shift
            )
            _scatter_8_bit_planes(word, level_planes, 0, group_idx)
        if num_plane_bytes > num_full_groups:
            word = _gather_byte_lane(scratch_tail, 0, shift)
            _scatter_8_bit_planes(word, level_planes, 0, num_full_groups)


@njit(nogil=True, cache=True)
def _count_active_tail_bytes(
    time_low_planes: np.ndarray,
    time_high_planes: np.ndarray,
    time_low_offset: int,
    time_high_offset: int,
    num_plane_bytes: int,
    num_valid: int,
) -> int:
    """Finds the number of active byte levels in a block's partial trailing byte."""
    # Mask off padding bits so stray bits in corrupt inputs do not inflate levels.
    valid_mask = (1 << num_valid) - 1
    num_levels = 8 if time_high_offset >= 0 else 4
    while num_levels > 0:
        tail_planes = _time_byte_level_planes(
            time_low_planes,
            time_high_planes,
            time_low_offset,
            time_high_offset,
            num_levels - 1,
            num_plane_bytes - 1,
            num_plane_bytes,
        )
        level_bits = 0
        for bit_idx in range(8):
            level_bits |= tail_planes[bit_idx, 0]
        if level_bits & valid_mask:
            break
        num_levels -= 1
    return num_levels


@njit(nogil=True, cache=True)
def unshuffle_time_residuals(
    time_low_planes: np.ndarray,
    time_high_planes: np.ndarray,
    time_low_offset: int,
    time_high_offset: int,
    out_time_residuals: np.ndarray,
    scratch_tail: np.ndarray,
) -> None:
    """Reconstructs a block's zigzagged uint64 time residuals from bit planes."""
    # Scan downward from the top byte level to skip all-zero upper planes (most time
    # residuals are small), zero once if any upper levels are skipped, then transpose
    # only the active byte levels for full groups followed by any partial tail group.
    num_samples = out_time_residuals.shape[0]
    num_full_groups = num_samples // SAMPLES_PER_PLANE_BYTE
    num_plane_bytes = plane_byte_count(num_samples)
    remainder = num_samples % SAMPLES_PER_PLANE_BYTE
    num_tail_bytes = 0
    if remainder:
        num_tail_bytes = _count_active_tail_bytes(
            time_low_planes,
            time_high_planes,
            time_low_offset,
            time_high_offset,
            num_plane_bytes,
            remainder,
        )
    num_active_bytes = 8 if time_high_offset >= 0 else 4
    while num_active_bytes > num_tail_bytes:
        level_planes = _time_byte_level_planes(
            time_low_planes,
            time_high_planes,
            time_low_offset,
            time_high_offset,
            num_active_bytes - 1,
            0,
            num_full_groups,
        )
        level_nonzero = False
        for bit_idx in range(8):
            for group_idx in range(num_full_groups):
                level_nonzero |= level_planes[bit_idx, group_idx] != 0
        if level_nonzero:
            break
        num_active_bytes -= 1
    if num_active_bytes < 8:
        out_time_residuals[:] = 0
    out_bytes = out_time_residuals.view(np.uint8)
    for byte_idx in range(num_active_bytes):
        level_planes = _time_byte_level_planes(
            time_low_planes,
            time_high_planes,
            time_low_offset,
            time_high_offset,
            byte_idx,
            0,
            num_full_groups,
        )
        for group_idx in range(num_full_groups):
            _unpack_time_plane_byte(
                level_planes, group_idx, byte_idx, out_bytes[64 * group_idx :]
            )
    if remainder:
        scratch_tail[:] = 0
        tail_bytes = scratch_tail.view(np.uint8)
        for byte_idx in range(num_active_bytes):
            level_planes = _time_byte_level_planes(
                time_low_planes,
                time_high_planes,
                time_low_offset,
                time_high_offset,
                byte_idx,
                0,
                num_plane_bytes,
            )
            _unpack_time_plane_byte(
                level_planes, num_plane_bytes - 1, byte_idx, tail_bytes
            )
        num_full_samples = SAMPLES_PER_PLANE_BYTE * (num_plane_bytes - 1)
        out_time_residuals[num_full_samples:] = scratch_tail[
            : num_samples - num_full_samples
        ]
