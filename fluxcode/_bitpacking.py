# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Packing a unit's rows into body bytes and back: the residual bit planes and byte planes,
the non-finite code planes and the time residual planes, plus the body writers and readers.

The body layout itself (field order, offsets, the Layout) is described in _format. Every
block's bytes in each plane field start on a byte boundary, so a block packs and unpacks on
its own.
"""

import numpy as np
from numba import njit

from ._format import (
    BLOCK_FLAG_IRREGULAR_TIME,
    BLOCK_FLAG_LONG_TIME,
    BLOCK_FLAG_NONFINITE,
    BYTES_PER_ANCHOR,
    BYTES_PER_FLAGS,
    BYTES_PER_PARAM,
    BYTES_PER_RESIDUAL_SAMPLE,
    BYTES_PER_SIZE,
    NONFINITE_BITS_PER_SAMPLE,
    TIME_BYTES_PER_BLOCK,
    TIME_SHORT_PLANES,
    Layout,
    TimeRows,
    UnitRows,
    allocate_time_rows,
    block_sizes_start,
    code_planes_start,
    get_int16,
    get_int64,
    grid_params_start,
    layout,
    plane_groups,
    put_int16,
    put_int64,
    read_layout,
    residual_start,
    time_ref_start,
    time_start_start,
    time_step_start,
    unit_size,
    value_anchor_start,
)


@njit(inline="always")
def _transpose8(matrix_bits: np.uint64) -> np.uint64:
    """Performs an 8x8 bit-matrix transpose on an unsigned 64-bit integer.

    Interchanges row and column bit indices such that bit (byte k, bit j) maps
    to bit (byte j, bit k). The operation is an involution (its own inverse).

    Args:
        matrix_bits: 64-bit integer viewed as eight 8-bit bytes.

    Returns:
        Transposed 64-bit integer.
    """
    # Swap 1-bit sub-matrices spaced by 7 bits
    delta_bits = (matrix_bits ^ (matrix_bits >> np.uint64(7))) & np.uint64(0x00AA00AA00AA00AA)
    matrix_bits = matrix_bits ^ delta_bits ^ (delta_bits << np.uint64(7))
    # Swap 2-bit sub-matrices spaced by 14 bits
    delta_bits = (matrix_bits ^ (matrix_bits >> np.uint64(14))) & np.uint64(0x0000CCCC0000CCCC)
    matrix_bits = matrix_bits ^ delta_bits ^ (delta_bits << np.uint64(14))
    # Swap 4-bit sub-matrices spaced by 28 bits
    delta_bits = (matrix_bits ^ (matrix_bits >> np.uint64(28))) & np.uint64(0x00000000F0F0F0F0)
    return matrix_bits ^ delta_bits ^ (delta_bits << np.uint64(28))


@njit(inline="always")
def planes_view(raw_unit: np.ndarray, num_blocks: int, num_groups: int, has_time: bool) -> np.ndarray:
    """Provides a 2D view into the 16 residual bit planes of an uncompressed unit.

    Args:
        raw_unit: 1D uint8 array containing uncompressed unit data.
        num_blocks: Total number of blocks.
        num_groups: Bytes per plane: the groups of all blocks.
        has_time: Whether the unit has a time axis (time columns before the residuals).

    Returns:
        2D uint8 array of shape (16, num_groups) representing bit planes.
    """
    start_offset = residual_start(num_blocks, has_time)
    end_offset = start_offset + BYTES_PER_RESIDUAL_SAMPLE * 8 * num_groups
    return raw_unit[start_offset:end_offset].reshape(16, num_groups)


@njit(inline="always")
def byte_planes_view(raw_unit: np.ndarray, num_blocks: int, num_groups: int, has_time: bool) -> np.ndarray:
    """Provides a 2D view into the residual field as 2 byte planes (flags bit 0).

    Args:
        raw_unit: 1D uint8 array containing uncompressed unit data.
        num_blocks: Total number of blocks.
        num_groups: Groups of all blocks.
        has_time: Whether the unit has a time axis (time columns before the residuals).

    Returns:
        2D uint8 array of shape (2, 8 * num_groups): the low bytes, then the high bytes, each
        block padded with zeros to a multiple of 8 like the bit planes.
    """
    start_offset = residual_start(num_blocks, has_time)
    end_offset = start_offset + BYTES_PER_RESIDUAL_SAMPLE * 8 * num_groups
    return raw_unit[start_offset:end_offset].reshape(2, 8 * num_groups)


@njit(inline="always")
def zigzag_block(residuals: np.ndarray, out_low_bytes: np.ndarray, out_high_bytes: np.ndarray) -> None:
    """Zigzag-encodes int16 residuals and splits them into low and high bytes.

    Args:
        residuals: 1D int16 array of residual differences for the block.
        out_low_bytes: Output uint8 array of length n receiving the lower 8 bits.
        out_high_bytes: Output uint8 array of length n receiving the upper 8 bits.
    """
    for sample_idx in range(residuals.shape[0]):
        signed_residual = residuals[sample_idx]
        # Zigzag transform: maps 0->0, -1->1, 1->2, -2->3 to keep small magnitudes small
        zigzag_val = np.uint16((signed_residual << np.int16(1)) ^ (signed_residual >> np.int16(15)))
        out_low_bytes[sample_idx] = np.uint8(zigzag_val)
        out_high_bytes[sample_idx] = np.uint8(zigzag_val >> np.uint16(8))


@njit(nogil=True, cache=True)
def shuffle_bytes(
    low_bytes: np.ndarray,
    high_bytes: np.ndarray,
    bit_planes: np.ndarray,
    group_offset: int,
    num_groups: int,
) -> None:
    """Packs a block's zigzag low and high bytes into unit bit planes.

    Transposes each group of 8 bytes as an 8x8 bit matrix and scatters its rows across the
    16 bit planes (0-7 from the low bytes, 8-15 from the high).

    Args:
        low_bytes: uint8 array of at least 8 * num_groups low zigzag bytes (padding included).
        high_bytes: uint8 array of at least 8 * num_groups high zigzag bytes.
        bit_planes: 2D uint8 array of shape (16, total groups) holding bit planes.
        group_offset: The block's first byte in each plane.
        num_groups: The block's bytes in each plane.
    """
    # Reinterpret byte buffers as 64-bit words for 8-byte group transposition
    low_words64 = low_bytes[:8 * num_groups].view(np.uint64)
    high_words64 = high_bytes[:8 * num_groups].view(np.uint64)
    for group_idx in range(num_groups):
        # Transpose each 8x8 bit matrix
        transposed_low = _transpose8(low_words64[group_idx])
        transposed_high = _transpose8(high_words64[group_idx])
        for bit_idx in range(8):
            # Extract transposed bytes into corresponding bit planes (0-7 low, 8-15 high)
            bit_planes[bit_idx, group_offset + group_idx] = np.uint8(
                (transposed_low >> np.uint64(8 * bit_idx)) & np.uint64(0xFF)
            )
            bit_planes[8 + bit_idx, group_offset + group_idx] = np.uint8(
                (transposed_high >> np.uint64(8 * bit_idx)) & np.uint64(0xFF)
            )


@njit(nogil=True, cache=True)
def shuffle_block(
    residuals: np.ndarray,
    bit_planes: np.ndarray,
    group_offset: int,
    scratch_low_bytes: np.ndarray,
    scratch_high_bytes: np.ndarray,
) -> None:
    """Zigzag-encodes int16 residuals and packs them into unit bit planes.

    Converts signed differences to unsigned integers via zigzag mapping, splits
    them into low and high byte arrays, and packs those with shuffle_bytes.

    Args:
        residuals: 1D int16 array of n residual differences for the block.
        bit_planes: 2D uint8 array of shape (16, total groups) holding bit planes.
        group_offset: The block's first byte in each plane.
        scratch_low_bytes: uint8 scratch array of at least 8 * plane_groups(n).
        scratch_high_bytes: uint8 scratch array of at least 8 * plane_groups(n).
    """
    num_samples = residuals.shape[0]
    num_8byte_groups = plane_groups(num_samples)
    zigzag_block(residuals, scratch_low_bytes, scratch_high_bytes)
    # The padding past n is zero
    scratch_low_bytes[num_samples:8 * num_8byte_groups] = 0
    scratch_high_bytes[num_samples:8 * num_8byte_groups] = 0
    shuffle_bytes(scratch_low_bytes, scratch_high_bytes, bit_planes, group_offset, num_8byte_groups)


@njit(nogil=True, cache=True)
def unshuffle_block(
    bit_planes: np.ndarray,
    group_offset: int,
    num_groups: int,
    out_low_bytes: np.ndarray,
    out_high_bytes: np.ndarray,
) -> None:
    """Reconstructs zigzag-encoded bytes for a block from the 16 bit planes.

    Gathers bits from planes 0-7 into low bytes and planes 8-15 into high bytes,
    reversing the 8x8 bit transpose.

    Args:
        bit_planes: 2D uint8 array of shape (16, total groups) holding bit planes.
        group_offset: The block's first byte in each plane.
        num_groups: The block's bytes in each plane.
        out_low_bytes: Output uint8 array of at least 8 * num_groups receiving lower zigzag
            bytes (the padding past n included).
        out_high_bytes: Output uint8 array of at least 8 * num_groups receiving upper zigzag
            bytes.
    """
    low_words64 = out_low_bytes.view(np.uint64)
    high_words64 = out_high_bytes.view(np.uint64)
    for group_idx in range(num_groups):
        gathered_low = np.uint64(0)
        gathered_high = np.uint64(0)
        for bit_idx in range(8):
            # Gather one byte from each bit plane into a 64-bit word
            gathered_low |= np.uint64(bit_planes[bit_idx, group_offset + group_idx]) << np.uint64(8 * bit_idx)
            gathered_high |= np.uint64(bit_planes[8 + bit_idx, group_offset + group_idx]) << np.uint64(8 * bit_idx)
        # Transpose back: transposition is self-inverse
        low_words64[group_idx] = _transpose8(gathered_low)
        high_words64[group_idx] = _transpose8(gathered_high)


@njit(inline="always")
def unzigzag16(low_bytes: np.ndarray, high_bytes: np.ndarray, sample_idx: int) -> np.int16:
    """Decodes a single signed int16 value from low and high zigzag bytes.

    Reverses the zigzag transformation (u >> 1) ^ -(u & 1).

    Args:
        low_bytes: 1D uint8 array containing lower zigzag bytes.
        high_bytes: 1D uint8 array containing upper zigzag bytes.
        sample_idx: Sample index within the byte arrays.

    Returns:
        Decoded signed 16-bit integer.
    """
    # Assemble uint16 from little-endian bytes
    zigzag_val = np.uint16(low_bytes[sample_idx]) | (np.uint16(high_bytes[sample_idx]) << np.uint16(8))
    # Unzigzag: shift out the sign bit and XOR with sign mask
    return np.int16(zigzag_val >> np.uint16(1)) ^ -np.int16(zigzag_val & np.uint16(1))


@njit(inline="always")
def code_planes_view(
    raw_unit: np.ndarray, num_blocks: int, num_groups: int, num_code_groups: int, has_time: bool
) -> np.ndarray:
    """Provides a 2D view into the non-finite code planes of an uncompressed unit.

    Args:
        raw_unit: 1D uint8 array containing uncompressed unit data.
        num_blocks: Total number of blocks.
        num_groups: Groups of all blocks.
        num_code_groups: Groups of the blocks carrying non-finite codes.
        has_time: Whether the unit has a time axis (time columns before the residuals).

    Returns:
        2D uint8 array of shape (2, num_code_groups).
    """
    start_offset = code_planes_start(num_blocks, num_groups, has_time)
    total_code_bytes = num_code_groups * NONFINITE_BITS_PER_SAMPLE
    return raw_unit[start_offset:start_offset + total_code_bytes].reshape(2, num_code_groups)


@njit(inline="always")
def time_planes_views(
    raw_unit: np.ndarray, num_blocks: int, num_groups: int, num_code_groups: int, num_short_groups: int,
    num_long_groups: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Provides 2D views into the time residual planes of an uncompressed unit with a time axis.

    Args:
        raw_unit: 1D uint8 array containing uncompressed unit data.
        num_blocks: Total number of blocks.
        num_groups: Groups of all blocks.
        num_code_groups: Groups of the blocks carrying non-finite codes.
        num_short_groups: Groups of the irregular blocks.
        num_long_groups: Groups of the long blocks.

    Returns:
        A tuple of (short_planes, long_planes): 2D uint8 arrays of shape
        (32, num_short_groups) holding planes 0-31 of every irregular block, and
        (32, num_long_groups) holding planes 32-63 of the long ones.
    """
    short_start = code_planes_start(num_blocks, num_groups, True) + num_code_groups * NONFINITE_BITS_PER_SAMPLE
    short_bytes = TIME_SHORT_PLANES * num_short_groups
    long_bytes = TIME_SHORT_PLANES * num_long_groups
    short_planes = raw_unit[short_start:short_start + short_bytes].reshape(TIME_SHORT_PLANES, num_short_groups)
    long_start = short_start + short_bytes
    long_planes = raw_unit[long_start:long_start + long_bytes].reshape(TIME_SHORT_PLANES, num_long_groups)
    return short_planes, long_planes


@njit(inline="always")
def _put_code_group(
    sample_codes: np.ndarray, group_idx: int, num_valid: int, code_planes: np.ndarray, plane_byte_offset: int
) -> None:
    """Packs the codes of samples 8 * group_idx + (0..num_valid - 1) into one byte of each code
    plane; bits past num_valid stay zero."""
    plane0_byte = 0
    plane1_byte = 0
    for bit_idx in range(num_valid):
        code_val = sample_codes[8 * group_idx + bit_idx]
        # Bit 0 of code goes to plane 0; bit 1 goes to plane 1
        plane0_byte |= (code_val & 1) << bit_idx
        plane1_byte |= ((code_val >> 1) & 1) << bit_idx
    code_planes[0, plane_byte_offset + group_idx] = plane0_byte
    code_planes[1, plane_byte_offset + group_idx] = plane1_byte


@njit(nogil=True, cache=True)
def put_codes(sample_codes: np.ndarray, code_planes: np.ndarray, plane_byte_offset: int) -> None:
    """Packs 2-bit non-finite sample codes for a block into two code bit planes.

    Args:
        sample_codes: 1D uint8 array of n codes (values in 0..3).
        code_planes: 2D uint8 array of shape (2, code groups) holding code planes.
        plane_byte_offset: The block's first byte in each code plane.
    """
    num_samples = sample_codes.shape[0]
    num_full_groups = num_samples // 8
    # Full groups with a constant bit count (it vectorizes), then a partial last group
    for group_idx in range(num_full_groups):
        _put_code_group(sample_codes, group_idx, 8, code_planes, plane_byte_offset)
    if num_samples % 8:
        _put_code_group(sample_codes, num_full_groups, num_samples % 8, code_planes, plane_byte_offset)


@njit(inline="always")
def get_code(plane0_byte: int, plane1_byte: int, bit_idx: int) -> int:
    """Extracts a 2-bit non-finite code for sample k from plane bytes.

    Args:
        plane0_byte: Byte from code plane 0.
        plane1_byte: Byte from code plane 1.
        bit_idx: Bit position within the bytes (0..7).

    Returns:
        Integer code in 0..3.
    """
    # Combine bit k from plane 0 and plane 1
    return ((plane0_byte >> bit_idx) & 1) | (((plane1_byte >> bit_idx) & 1) << 1)


@njit(inline="always")
def _get_code_group(
    code_planes: np.ndarray, plane_byte_offset: int, group_idx: int, num_valid: int, out_codes: np.ndarray
) -> None:
    """Unpacks the codes of samples 8 * group_idx + (0..num_valid - 1) from one byte of each code plane."""
    plane0_byte = code_planes[0, plane_byte_offset + group_idx]
    plane1_byte = code_planes[1, plane_byte_offset + group_idx]
    for bit_idx in range(num_valid):
        # Extract 2-bit code for each sample in the 8-sample group
        out_codes[8 * group_idx + bit_idx] = get_code(plane0_byte, plane1_byte, bit_idx)


@njit(nogil=True, cache=True)
def get_codes(code_planes: np.ndarray, plane_byte_offset: int, out_codes: np.ndarray) -> None:
    """Unpacks 2-bit non-finite sample codes for a flagged block.

    Args:
        code_planes: 2D uint8 array of shape (2, code groups) holding code planes.
        plane_byte_offset: The block's first byte in each code plane.
        out_codes: Output 1D uint8 array of length n receiving unpacked codes.
    """
    num_samples = out_codes.shape[0]
    num_full_groups = num_samples // 8
    # Full groups with a constant bit count (it vectorizes), then a partial last group
    for group_idx in range(num_full_groups):
        _get_code_group(code_planes, plane_byte_offset, group_idx, 8, out_codes)
    if num_samples % 8:
        _get_code_group(code_planes, plane_byte_offset, num_full_groups, num_samples % 8, out_codes)


@njit(inline="always")
def _byte_word(v0: np.uint64, v1: np.uint64, v2: np.uint64, v3: np.uint64, v4: np.uint64, v5: np.uint64,
               v6: np.uint64, v7: np.uint64, shift: np.uint64) -> np.uint64:
    """Packs byte (shift / 8) of 8 values into one word: byte k of the word from value k."""
    mask = np.uint64(0xFF)
    return (
        ((v0 >> shift) & mask)
        | (((v1 >> shift) & mask) << np.uint64(8))
        | (((v2 >> shift) & mask) << np.uint64(16))
        | (((v3 >> shift) & mask) << np.uint64(24))
        | (((v4 >> shift) & mask) << np.uint64(32))
        | (((v5 >> shift) & mask) << np.uint64(40))
        | (((v6 >> shift) & mask) << np.uint64(48))
        | (((v7 >> shift) & mask) << np.uint64(56))
    )


@njit(inline="always")
def _level_planes(
    short_planes: np.ndarray,
    long_planes: np.ndarray,
    short_offset: int,
    long_offset: int,
    byte_idx: int,
    first_group: int,
    end_group: int,
) -> np.ndarray:
    """The 8 bit planes of a block's byte level byte_idx: levels 0-3 are in the short planes
    (from the block's offset among irregular blocks), 4-7 in the long planes (offset among
    long blocks). The view holds the block's groups first_group to end_group."""
    if byte_idx < 4:
        return short_planes[8 * byte_idx:8 * byte_idx + 8, short_offset + first_group:short_offset + end_group]
    level = byte_idx - 4
    return long_planes[8 * level:8 * level + 8, long_offset + first_group:long_offset + end_group]


@njit(inline="always")
def _put_time_group(
    values: np.ndarray, base: int, shift: np.uint64, level_planes: np.ndarray, group_idx: int
) -> None:
    """Packs byte (shift / 8) of values[base:base + 8] into byte group_idx of 8 bit planes."""
    transposed_word = _transpose8(_byte_word(
        values[base], values[base + 1], values[base + 2], values[base + 3],
        values[base + 4], values[base + 5], values[base + 6], values[base + 7],
        shift,
    ))
    for bit_idx in range(8):
        level_planes[bit_idx, group_idx] = np.uint8((transposed_word >> np.uint64(8 * bit_idx)) & np.uint64(0xFF))


@njit(inline="always")
def _get_time_group(level_planes: np.ndarray, group_idx: int, byte_idx: int, out_bytes: np.ndarray) -> None:
    """Unpacks byte group_idx of 8 bit planes into byte byte_idx of the 8 uint64 samples viewed
    as bytes by out_bytes."""
    gathered_word = np.uint64(0)
    for bit_idx in range(8):
        gathered_word |= np.uint64(level_planes[bit_idx, group_idx]) << np.uint64(8 * bit_idx)
    # Transpose back: byte k of the result is byte byte_idx of sample 8 * group_idx + k
    transposed_word = _transpose8(gathered_word)
    for sample_offset in range(8):
        out_bytes[byte_idx + 8 * sample_offset] = np.uint8(
            (transposed_word >> np.uint64(8 * sample_offset)) & np.uint64(0xFF)
        )


@njit(nogil=True, cache=True)
def shuffle_time_residuals(
    time_residuals: np.ndarray,
    short_planes: np.ndarray,
    long_planes: np.ndarray,
    short_offset: int,
    long_offset: int,
    scratch_tail: np.ndarray,
) -> None:
    """Packs a block's zigzagged uint64 time residuals into its time residual bit planes.

    Bit k of byte i of plane j is bit j of time_residuals[8 * i + k], as for the residual
    planes. Only the bytes up to the highest set bit of any residual are written: the planes
    above it must already be zero (write_time_rows zeroes the field first).

    Args:
        time_residuals: 1D uint64 array of the block's n zigzagged time residuals, below 2^32
            unless the block is long.
        short_planes: 2D uint8 array of planes 0-31 of the irregular blocks.
        long_planes: 2D uint8 array of planes 32-63 of the long blocks.
        short_offset: The block's first byte in the short planes.
        long_offset: The block's first byte in the long planes (unused for a short block).
        scratch_tail: uint64 scratch array of 8 for a partial last group.
    """
    num_samples = time_residuals.shape[0]
    num_full_groups = num_samples // 8
    num_groups = plane_groups(num_samples)
    # Bytes above the highest set bit of every residual are zero in all samples
    all_bits = np.uint64(0)
    for sample_idx in range(num_samples):
        all_bits |= time_residuals[sample_idx]
    num_active_bytes = 0
    while num_active_bytes < 8 and (all_bits >> np.uint64(8 * num_active_bytes)) != 0:
        num_active_bytes += 1
    if num_groups > num_full_groups:
        # A partial last group packs from a zero-padded copy
        scratch_tail[:] = 0
        scratch_tail[:num_samples - 8 * num_full_groups] = time_residuals[8 * num_full_groups:]
    for byte_idx in range(num_active_bytes):
        level_planes = _level_planes(short_planes, long_planes, short_offset, long_offset, byte_idx, 0, num_groups)
        shift = np.uint64(8 * byte_idx)
        for group_idx in range(num_full_groups):
            _put_time_group(time_residuals, 8 * group_idx, shift, level_planes, group_idx)
        if num_groups > num_full_groups:
            _put_time_group(scratch_tail, 0, shift, level_planes, num_full_groups)


@njit(nogil=True, cache=True)
def _time_tail_levels(
    short_planes: np.ndarray, long_planes: np.ndarray, short_offset: int, long_offset: int, stride: int,
    num_valid: int,
) -> int:
    """The number of byte levels up to the highest with a nonzero bit among the first num_valid
    samples of a block's partial last group (its padding bits are ignored)."""
    valid_mask = (1 << num_valid) - 1
    num_levels = 8 if long_offset >= 0 else 4
    while num_levels > 0:
        tail_planes = _level_planes(short_planes, long_planes, short_offset, long_offset, num_levels - 1, stride - 1,
                                    stride)
        level_bits = 0
        for bit_idx in range(8):
            level_bits |= tail_planes[bit_idx, 0]
        if level_bits & valid_mask:
            break
        num_levels -= 1
    return num_levels


@njit(nogil=True, cache=True)
def _unshuffle_time_tail(
    short_planes: np.ndarray,
    long_planes: np.ndarray,
    short_offset: int,
    long_offset: int,
    stride: int,
    num_active_bytes: int,
    out_time_residuals: np.ndarray,
    scratch_tail: np.ndarray,
) -> None:
    """Unpacks the partial last group of a block's time residuals: its samples past n are padding."""
    scratch_tail[:] = 0
    tail_bytes = scratch_tail.view(np.uint8)
    for byte_idx in range(num_active_bytes):
        level_planes = _level_planes(short_planes, long_planes, short_offset, long_offset, byte_idx, 0, stride)
        _get_time_group(level_planes, stride - 1, byte_idx, tail_bytes)
    num_full_samples = 8 * (stride - 1)
    out_time_residuals[num_full_samples:] = scratch_tail[:out_time_residuals.shape[0] - num_full_samples]


@njit(nogil=True, cache=True)
def unshuffle_time_residuals(
    short_planes: np.ndarray,
    long_planes: np.ndarray,
    short_offset: int,
    long_offset: int,
    out_time_residuals: np.ndarray,
    scratch_tail: np.ndarray,
) -> int:
    """Reconstructs a block's zigzagged uint64 time residuals from its time residual bit planes.

    Args:
        short_planes: 2D uint8 array of planes 0-31 of the irregular blocks.
        long_planes: 2D uint8 array of planes 32-63 of the long blocks.
        short_offset: The block's first byte in the short planes.
        long_offset: The block's first byte in the long planes, or -1 for a short block.
        out_time_residuals: Output 1D uint64 array of the block's n residuals.
        scratch_tail: uint64 scratch array of 8 for a partial last group.

    Returns:
        The number of byte levels with a nonzero plane byte (at most 4 for a short block).
    """
    num_samples = out_time_residuals.shape[0]
    # Full groups here; a partial last group (its plane bytes after them) is unpacked separately
    num_groups = num_samples // 8
    stride = plane_groups(num_samples)
    num_tail_bytes = 0
    if num_samples % 8:
        num_tail_bytes = _time_tail_levels(short_planes, long_planes, short_offset, long_offset, stride,
                                           num_samples % 8)
    # Byte levels above the highest nonzero plane byte of this block are zero in every
    # sample: find them with a contiguous scan and skip them (residuals are mostly small)
    num_active_bytes = 8 if long_offset >= 0 else 4
    while num_active_bytes > num_tail_bytes:
        level_planes = _level_planes(
            short_planes, long_planes, short_offset, long_offset, num_active_bytes - 1, 0, num_groups
        )
        level_nonzero = False
        for bit_idx in range(8):
            for group_idx in range(num_groups):
                level_nonzero |= level_planes[bit_idx, group_idx] != 0
        if level_nonzero:
            break
        num_active_bytes -= 1
    if num_active_bytes < 8:
        out_time_residuals[:] = 0
    # Little-endian bytes of the residuals: byte byte_idx of sample s is at 8 * s + byte_idx
    out_bytes = out_time_residuals.view(np.uint8)
    for byte_idx in range(num_active_bytes):
        level_planes = _level_planes(short_planes, long_planes, short_offset, long_offset, byte_idx, 0, num_groups)
        for group_idx in range(num_groups):
            gathered_word = np.uint64(0)
            for bit_idx in range(8):
                gathered_word |= np.uint64(level_planes[bit_idx, group_idx]) << np.uint64(8 * bit_idx)
            # Transpose back: byte k of the result is byte byte_idx of sample 8 * group_idx + k
            transposed_word = _transpose8(gathered_word)
            out_byte_idx = 64 * group_idx + byte_idx
            for sample_offset in range(8):
                out_bytes[out_byte_idx + 8 * sample_offset] = np.uint8(
                    (transposed_word >> np.uint64(8 * sample_offset)) & np.uint64(0xFF)
                )
    if num_samples % 8:
        _unshuffle_time_tail(
            short_planes, long_planes, short_offset, long_offset, stride, num_active_bytes, out_time_residuals,
            scratch_tail,
        )
    return num_active_bytes


@njit(nogil=True, cache=True)
def write_time_rows(
    block_flags: np.ndarray,
    sample_offsets: np.ndarray,
    group_offsets: np.ndarray,
    code_offsets: np.ndarray,
    short_offsets: np.ndarray,
    long_offsets: np.ndarray,
    time_starts: np.ndarray,
    time_steps: np.ndarray,
    time_refs: np.ndarray,
    time_residuals: np.ndarray,
    out_raw_unit: np.ndarray,
) -> None:
    """Serializes the time_start, time_step, time_ref and time_residual_planes fields.

    Args:
        block_flags: 1D uint8 array of block flags (BLOCK_FLAG_IRREGULAR_TIME selects the residual planes).
        sample_offsets, group_offsets, code_offsets, short_offsets, long_offsets: The Layout.
        time_starts: 1D int64 array of block start times (empty blocks' are ignored).
        time_steps: 1D int64 array of block time steps.
        time_refs: 1D uint64 array of block reference quotients.
        time_residuals: 1D uint64 array of every sample's zigzagged time residual (only
            irregular blocks' are read).
        out_raw_unit: Uncompressed body whose value fields are written by write_rows.
    """
    num_blocks = block_flags.shape[0]
    short_planes, long_planes = time_planes_views(
        out_raw_unit, num_blocks, int(group_offsets[num_blocks]), int(code_offsets[num_blocks]), int(short_offsets[num_blocks]),
        int(long_offsets[num_blocks]),
    )
    scratch_tail = np.empty(8, np.uint64)
    # One contiguous fill: each block then writes only the planes its residuals reach
    short_planes[:, :] = 0
    long_planes[:, :] = 0
    previous_start = np.int64(0)
    seen_samples = False
    for block_idx in range(num_blocks):
        if sample_offsets[block_idx + 1] == sample_offsets[block_idx]:
            # An empty block has no times: its columns are 0
            put_int64(out_raw_unit, time_start_start(num_blocks), num_blocks, block_idx, 0)
            put_int64(out_raw_unit, time_step_start(num_blocks), num_blocks, block_idx, 0)
            put_int64(out_raw_unit, time_ref_start(num_blocks), num_blocks, block_idx, 0)
            continue
        # The first start is stored as is; later starts as the uint64 increase over the previous start
        start_increase = time_starts[block_idx]
        if seen_samples:
            start_increase = np.int64(np.uint64(time_starts[block_idx]) - np.uint64(previous_start))
        previous_start = time_starts[block_idx]
        seen_samples = True
        put_int64(out_raw_unit, time_start_start(num_blocks), num_blocks, block_idx, start_increase)
        put_int64(out_raw_unit, time_step_start(num_blocks), num_blocks, block_idx, time_steps[block_idx])
        put_int64(out_raw_unit, time_ref_start(num_blocks), num_blocks, block_idx, np.int64(time_refs[block_idx]))
        if block_flags[block_idx] & BLOCK_FLAG_IRREGULAR_TIME:
            shuffle_time_residuals(
                time_residuals[sample_offsets[block_idx]:sample_offsets[block_idx + 1]],
                short_planes,
                long_planes,
                short_offsets[block_idx],
                long_offsets[block_idx],
                scratch_tail,
            )


TIME_ROWS_OK: int = 0
"""read_time_rows status: the start times are valid."""

TIME_ROWS_BAD_START: int = 1
"""read_time_rows status: a start time is int64 minimum (NaT) or overflows int64."""

TIME_ROWS_BAD_LONG: int = 2
"""read_time_rows status: a long block's planes 32-63 are all zero (it should be short)."""

TIME_ROWS_BAD_EMPTY: int = 3
"""read_time_rows status: an empty block's time_start, time_step or time_ref isn't 0."""

TIME_ROWS_BAD_SINGLE: int = 4
"""read_time_rows status: a block of one sample has a nonzero time step or reference, or
irregular times (the encoder's only form is regular with step and reference 0)."""


@njit(nogil=True, cache=True)
def _long_planes_zero(long_planes: np.ndarray, long_offset: int, num_samples: int) -> bool:
    """Whether a long block's planes 32-63 are zero in its samples (padding bits ignored)."""
    num_groups = plane_groups(num_samples)
    tail_mask = np.uint8((1 << (num_samples % 8)) - 1) if num_samples % 8 else np.uint8(0xFF)
    for plane_idx in range(TIME_SHORT_PLANES):
        for group_idx in range(num_groups - 1):
            if long_planes[plane_idx, long_offset + group_idx]:
                return False
        if long_planes[plane_idx, long_offset + num_groups - 1] & tail_mask:
            return False
    return True


@njit(nogil=True, cache=True)
def read_time_columns(
    raw_unit: np.ndarray,
    sample_offsets: np.ndarray,
    group_offsets: np.ndarray,
    code_offsets: np.ndarray,
    short_offsets: np.ndarray,
    long_offsets: np.ndarray,
    out_time_starts: np.ndarray,
    out_time_steps: np.ndarray,
    out_time_refs: np.ndarray,
) -> tuple[int, int]:
    """Deserializes the time_start, time_step and time_ref fields and validates the time axis,
    without unpacking any time residual planes.

    Args:
        raw_unit: 1D uint8 array of uncompressed body bytes of a unit with a time axis.
        sample_offsets, group_offsets, code_offsets, short_offsets, long_offsets: The Layout.
        out_time_starts: Output 1D int64 array receiving the block start times (0 for an
            empty block).
        out_time_steps: Output 1D int64 array receiving the block time steps.
        out_time_refs: Output 1D uint64 array receiving the block reference quotients.

    Returns:
        A tuple of (status, block_idx): TIME_ROWS_OK, or TIME_ROWS_BAD_START and the
        first block whose start is int64 minimum or overflows int64, TIME_ROWS_BAD_LONG
        and the first long block whose planes 32-63 are all zero, TIME_ROWS_BAD_EMPTY and
        the first empty block with a nonzero time column, or TIME_ROWS_BAD_SINGLE and the
        first one-sample block that isn't regular with step and reference 0.
    """
    num_blocks = out_time_starts.shape[0]
    _, long_planes = time_planes_views(
        raw_unit, num_blocks, int(group_offsets[num_blocks]), int(code_offsets[num_blocks]), int(short_offsets[num_blocks]),
        int(long_offsets[num_blocks]),
    )
    int64_max = np.uint64(0x7FFFFFFFFFFFFFFF)
    seen_samples = False
    previous_start = np.int64(0)
    for block_idx in range(num_blocks):
        stored_start = get_int64(raw_unit, time_start_start(num_blocks), num_blocks, block_idx)
        out_time_steps[block_idx] = get_int64(raw_unit, time_step_start(num_blocks), num_blocks, block_idx)
        out_time_refs[block_idx] = np.uint64(get_int64(raw_unit, time_ref_start(num_blocks), num_blocks, block_idx))
        block_len = sample_offsets[block_idx + 1] - sample_offsets[block_idx]
        if block_len == 0:
            if stored_start != 0 or out_time_steps[block_idx] != 0 or out_time_refs[block_idx] != 0:
                return TIME_ROWS_BAD_EMPTY, block_idx
            out_time_starts[block_idx] = 0
            continue
        if not seen_samples:
            if stored_start == np.int64(-0x8000000000000000):
                return TIME_ROWS_BAD_START, block_idx
            out_time_starts[block_idx] = stored_start
        else:
            # The increase is unsigned: the start must stay at most int64 maximum
            headroom = int64_max - np.uint64(previous_start)
            if np.uint64(stored_start) > headroom:
                return TIME_ROWS_BAD_START, block_idx
            out_time_starts[block_idx] = np.int64(np.uint64(previous_start) + np.uint64(stored_start))
        previous_start = out_time_starts[block_idx]
        seen_samples = True
        if block_len == 1 and (
            out_time_steps[block_idx] != 0 or out_time_refs[block_idx] != 0 or raw_unit[block_idx] & BLOCK_FLAG_IRREGULAR_TIME
        ):
            return TIME_ROWS_BAD_SINGLE, block_idx
        if raw_unit[block_idx] & BLOCK_FLAG_LONG_TIME and _long_planes_zero(long_planes, long_offsets[block_idx], block_len):
            return TIME_ROWS_BAD_LONG, block_idx
    return TIME_ROWS_OK, 0


@njit(nogil=True, cache=True)
def read_time_residuals(
    raw_unit: np.ndarray,
    sample_offsets: np.ndarray,
    group_offsets: np.ndarray,
    code_offsets: np.ndarray,
    short_offsets: np.ndarray,
    long_offsets: np.ndarray,
    block_ids: np.ndarray,
    out_time_residuals: np.ndarray,
) -> None:
    """Unpacks the zigzagged time residuals of the given blocks (those that are irregular).

    Args:
        raw_unit: 1D uint8 array of uncompressed body bytes of a unit with a time axis.
        sample_offsets, group_offsets, code_offsets, short_offsets, long_offsets: The Layout.
        block_ids: 1D int64 array of the blocks to unpack.
        out_time_residuals: Output 1D uint64 array of every sample receiving the irregular
            given blocks' residuals (other samples are not written).
    """
    num_blocks = sample_offsets.shape[0] - 1
    short_planes, long_planes = time_planes_views(
        raw_unit, num_blocks, int(group_offsets[num_blocks]), int(code_offsets[num_blocks]), int(short_offsets[num_blocks]),
        int(long_offsets[num_blocks]),
    )
    scratch_tail = np.empty(8, np.uint64)
    for block_idx in block_ids:
        if raw_unit[block_idx] & BLOCK_FLAG_IRREGULAR_TIME:
            unshuffle_time_residuals(
                short_planes,
                long_planes,
                short_offsets[block_idx],
                long_offsets[block_idx] if raw_unit[block_idx] & BLOCK_FLAG_LONG_TIME else -1,
                out_time_residuals[sample_offsets[block_idx]:sample_offsets[block_idx + 1]],
                scratch_tail,
            )


def read_time_rows(
    raw_unit: np.ndarray,
    sample_offsets: np.ndarray,
    group_offsets: np.ndarray,
    code_offsets: np.ndarray,
    short_offsets: np.ndarray,
    long_offsets: np.ndarray,
    out_time_starts: np.ndarray,
    out_time_steps: np.ndarray,
    out_time_refs: np.ndarray,
    out_time_residuals: np.ndarray,
    block_ids: np.ndarray | None = None,
) -> tuple[int, int]:
    """Deserializes the time fields: read_time_columns, then the time residuals of block_ids
    (every block if None) unless the columns are invalid.

    Returns:
        read_time_columns' (status, block_idx).
    """
    offsets = (sample_offsets, group_offsets, code_offsets, short_offsets, long_offsets)
    status = read_time_columns(raw_unit, *offsets, out_time_starts, out_time_steps, out_time_refs)
    if status[0] == TIME_ROWS_OK:
        if block_ids is None:
            block_ids = np.arange(out_time_starts.shape[0], dtype=np.int64)
        read_time_residuals(raw_unit, *offsets, block_ids, out_time_residuals)
    return status


@njit(nogil=True, cache=True)
def write_rows(
    block_flags: np.ndarray,
    block_sizes: np.ndarray,
    grid_params: np.ndarray,
    value_anchors: np.ndarray,
    residuals: np.ndarray,
    codes: np.ndarray,
    sample_offsets: np.ndarray,
    group_offsets: np.ndarray,
    code_offsets: np.ndarray,
    byte_planes: bool,
    has_time: bool,
    out_raw_unit: np.ndarray,
) -> None:
    """Serializes the value fields of a unit into a preallocated uncompressed body buffer.

    The time fields (if has_time) are written separately by write_time_rows.

    Args:
        block_flags: 1D uint8 array of block flags.
        block_sizes: 1D int64 array of block sizes.
        grid_params: 1D int64 array of block grid parameters.
        value_anchors: 1D int64 array of block value anchors (float64 bits or decimal grid index).
        residuals: 1D int16 array of every sample's difference residual.
        codes: 1D uint8 array of every sample's code (only flagged blocks' are read; may be
            empty if no block is flagged).
        sample_offsets, group_offsets, code_offsets: The Layout's offsets.
        byte_planes: Store the residuals as 2 byte planes instead of 16 bit planes.
        has_time: Whether the unit has a time axis (time columns before the residuals).
        out_raw_unit: Output 1D uint8 array of length unit_size(...).
    """
    num_blocks = block_flags.shape[0]
    max_len = 0
    for block_idx in range(num_blocks):
        max_len = max(max_len, block_sizes[block_idx])
    scratch_low_bytes = np.empty(8 * plane_groups(max_len), np.uint8)
    scratch_high_bytes = np.empty(8 * plane_groups(max_len), np.uint8)
    # Obtain views into residual and non-finite plane regions
    num_groups = int(group_offsets[num_blocks])
    planes = planes_view(out_raw_unit, num_blocks, num_groups, has_time)
    bplanes = byte_planes_view(out_raw_unit, num_blocks, num_groups, has_time)
    cplanes = code_planes_view(out_raw_unit, num_blocks, num_groups, int(code_offsets[num_blocks]), has_time)
    for block_idx in range(num_blocks):
        out_raw_unit[block_idx] = block_flags[block_idx]
        block_len = block_sizes[block_idx]
        out_raw_unit[block_sizes_start(num_blocks) + block_idx] = np.uint8(block_len & 0xFF)
        out_raw_unit[block_sizes_start(num_blocks) + num_blocks + block_idx] = np.uint8(block_len >> 8)
        # Store the grid parameter (int16) and value anchor (int64) byte-planed
        put_int16(out_raw_unit, grid_params_start(num_blocks), num_blocks, block_idx, grid_params[block_idx])
        put_int64(out_raw_unit, value_anchor_start(num_blocks), num_blocks, block_idx, value_anchors[block_idx])
        block_residuals = residuals[sample_offsets[block_idx]:sample_offsets[block_idx + 1]]
        if byte_planes:
            # Zigzag and store the low and high bytes in place, then zero the padding
            sample_start = 8 * group_offsets[block_idx]
            zigzag_block(
                block_residuals,
                bplanes[0, sample_start:sample_start + block_len],
                bplanes[1, sample_start:sample_start + block_len],
            )
            bplanes[:, sample_start + block_len:8 * group_offsets[block_idx + 1]] = 0
        else:
            # Zigzag, transpose, and store residual bit planes
            shuffle_block(block_residuals, planes, group_offsets[block_idx], scratch_low_bytes, scratch_high_bytes)
        if block_flags[block_idx] & BLOCK_FLAG_NONFINITE:
            # Store 2-bit code planes for flagged blocks
            put_codes(codes[sample_offsets[block_idx]:sample_offsets[block_idx + 1]], cplanes, code_offsets[block_idx])


@njit(nogil=True, cache=True)
def read_rows(
    raw_unit: np.ndarray,
    byte_planes: bool,
    has_time: bool,
    sample_offsets: np.ndarray,
    group_offsets: np.ndarray,
    code_offsets: np.ndarray,
    out_grid_params: np.ndarray,
    out_value_anchors: np.ndarray,
    out_residuals: np.ndarray,
    out_codes: np.ndarray,
) -> None:
    """Deserializes the value fields of a unit from an uncompressed body buffer.

    The block flags and sizes are read directly (they give the Layout).

    Args:
        raw_unit: 1D uint8 array containing uncompressed body bytes.
        byte_planes: Whether the residuals are stored as byte planes (flags bit 0).
        has_time: Whether the unit has a time axis (time columns before the residuals).
        sample_offsets, group_offsets, code_offsets: The Layout's offsets.
        out_grid_params: Output 1D int64 array of length num_blocks receiving grid parameters.
        out_value_anchors: Output 1D int64 array of length num_blocks receiving value anchors.
        out_residuals: Output 1D int16 array of every sample receiving the residuals.
        out_codes: Output 1D uint8 array of every sample receiving codes for flagged blocks
            (unflagged blocks' samples are left unmodified).
    """
    num_blocks = out_grid_params.shape[0]
    max_groups = 0
    for block_idx in range(num_blocks):
        max_groups = max(max_groups, group_offsets[block_idx + 1] - group_offsets[block_idx])
    scratch_low_bytes = np.empty(8 * max_groups, np.uint8)
    scratch_high_bytes = np.empty(8 * max_groups, np.uint8)
    # Access views into bit plane sections
    num_groups = int(group_offsets[num_blocks])
    planes = planes_view(raw_unit, num_blocks, num_groups, has_time)
    bplanes = byte_planes_view(raw_unit, num_blocks, num_groups, has_time)
    cplanes = code_planes_view(raw_unit, num_blocks, num_groups, int(code_offsets[num_blocks]), has_time)
    for block_idx in range(num_blocks):
        # Read the byte-planed grid parameter and value anchor
        out_grid_params[block_idx] = get_int16(raw_unit, grid_params_start(num_blocks), num_blocks, block_idx)
        out_value_anchors[block_idx] = get_int64(raw_unit, value_anchor_start(num_blocks), num_blocks, block_idx)
        first_sample = sample_offsets[block_idx]
        block_len = sample_offsets[block_idx + 1] - first_sample
        if byte_planes:
            sample_start = 8 * group_offsets[block_idx]
            scratch_low_bytes[:block_len] = bplanes[0, sample_start:sample_start + block_len]
            scratch_high_bytes[:block_len] = bplanes[1, sample_start:sample_start + block_len]
        else:
            # Unshuffle bit planes into zigzag low/high bytes
            unshuffle_block(
                planes, group_offsets[block_idx], group_offsets[block_idx + 1] - group_offsets[block_idx],
                scratch_low_bytes, scratch_high_bytes,
            )
        for sample_idx in range(block_len):
            # Reverse zigzag mapping to recover signed residual differences
            out_residuals[first_sample + sample_idx] = unzigzag16(scratch_low_bytes, scratch_high_bytes, sample_idx)
        if raw_unit[block_idx] & BLOCK_FLAG_NONFINITE:
            # Unpack non-finite codes for flagged blocks
            get_codes(cplanes, code_offsets[block_idx], out_codes[first_sample:first_sample + block_len])


def write_unit(
    block_flags: np.ndarray,
    block_sizes: np.ndarray,
    grid_params: np.ndarray,
    value_anchors: np.ndarray,
    residuals: np.ndarray,
    codes: np.ndarray | None = None,
    byte_planes: bool = False,
    time_rows: TimeRows | None = None,
) -> np.ndarray:
    """Serializes unit components into a newly allocated uncompressed body.

    Args:
        block_flags: 1D uint8 array of block flags.
        block_sizes: 1D integer array of block sizes.
        grid_params: 1D int64 array of block grid parameters.
        value_anchors: 1D int64 array of block value anchors.
        residuals: 1D int16 array of every sample's difference residual.
        codes: Optional 1D uint8 array of every sample's code. Required if any block has the
            BLOCK_FLAG_NONFINITE flag set.
        byte_planes: Store the residuals as 2 byte planes instead of 16 bit planes.
        time_rows: The time axis rows, or None for a unit without a time axis.

    Returns:
        1D uint8 array containing the serialized uncompressed body.

    Raises:
        ValueError: If any block is flagged non-finite but codes are omitted, or flagged
            irregular in a unit without a time axis.
    """
    num_blocks = block_flags.shape[0]
    block_sizes = np.asarray(block_sizes, np.int64)
    offsets = layout(block_flags, block_sizes)
    if codes is None:
        if offsets.code_offsets[-1]:
            raise ValueError("flagged blocks need codes")
        codes = np.zeros(0, np.uint8)
    has_time = time_rows is not None
    if offsets.short_offsets[-1] and not has_time:
        raise ValueError("irregular time blocks need time rows")
    raw_unit = np.empty(unit_size(num_blocks, offsets, has_time), np.uint8)
    write_rows(
        block_flags, block_sizes, grid_params, value_anchors, residuals, codes, offsets.sample_offsets,
        offsets.group_offsets, offsets.code_offsets, byte_planes, has_time, raw_unit,
    )
    if time_rows is not None:
        write_time_rows(block_flags, *offsets, *time_rows, raw_unit)
    return raw_unit


def read_unit(
    raw_unit: bytes | np.ndarray,
    num_blocks: int,
    byte_planes: bool = False,
    has_time: bool = False,
) -> UnitRows:
    """Deserializes an uncompressed body into constituent arrays.

    Args:
        raw_unit: Byte buffer or uint8 array containing the uncompressed body.
        num_blocks: Number of blocks in the unit.
        byte_planes: Whether the residuals are stored as byte planes (flags bit 0).
        has_time: Whether the unit has a time axis.

    Returns:
        The unit's rows.

    Raises:
        ValueError: If the buffer size does not match its block flags and sizes, or a time
            column is invalid.
    """
    raw_arr = np.frombuffer(raw_unit, np.uint8) if not isinstance(raw_unit, np.ndarray) else raw_unit
    # The flags and sizes give the layout, so check they exist first
    if raw_arr.shape[0] < (BYTES_PER_FLAGS + BYTES_PER_SIZE) * num_blocks:
        raise ValueError(f"a body of {raw_arr.shape[0]} bytes doesn't hold {num_blocks} blocks")
    block_flags = raw_arr[:num_blocks].copy()
    block_sizes, offsets = read_layout(raw_arr, num_blocks)
    if raw_arr.shape[0] != unit_size(num_blocks, offsets, has_time):
        raise ValueError(f"a body of {raw_arr.shape[0]} bytes doesn't hold its {num_blocks} blocks")
    num_samples = int(offsets.sample_offsets[-1])
    grid_params = np.empty(num_blocks, np.int64)
    value_anchors = np.empty(num_blocks, np.int64)
    residuals = np.empty(num_samples, np.int16)
    codes = np.zeros(num_samples, np.uint8)
    read_rows(
        raw_arr, byte_planes, has_time, offsets.sample_offsets, offsets.group_offsets, offsets.code_offsets,
        grid_params, value_anchors, residuals, codes,
    )
    time_rows = None
    if has_time:
        # Zeroed: read_time_rows leaves regular blocks' residuals untouched
        time_rows = allocate_time_rows(num_blocks, num_samples, zero_residuals=True)
        check_time_rows_status(*read_time_rows(raw_arr, *offsets, *time_rows))
    return UnitRows(block_flags, block_sizes, grid_params, value_anchors, residuals, codes, time_rows)


@njit(inline="always")
def _copy_bytes(dst: np.ndarray, dst_offset: int, src: np.ndarray, src_offset: int, num_bytes: int) -> None:
    """Copies bytes between 1D arrays (sliced first, as in _copy_columns)."""
    dst_bytes = dst[dst_offset:dst_offset + num_bytes]
    src_bytes = src[src_offset:src_offset + num_bytes]
    for byte_idx in range(num_bytes):
        dst_bytes[byte_idx] = src_bytes[byte_idx]


@njit(inline="always")
def _copy_columns(dst: np.ndarray, dst_offset: int, src: np.ndarray, src_offset: int, num_columns: int) -> None:
    """Copies num_columns columns of every row of a 2D plane field.

    Each row is sliced first and indexed from 0: numba's slice assignment between arrays,
    and indexing with a runtime signed offset (its negative-index wraparound), both keep the
    loop from vectorizing, at about 10x the cost.
    """
    for row_idx in range(src.shape[0]):
        dst_row = dst[row_idx, dst_offset:dst_offset + num_columns]
        src_row = src[row_idx, src_offset:src_offset + num_columns]
        for column_idx in range(num_columns):
            dst_row[column_idx] = src_row[column_idx]


@njit(nogil=True, cache=True)
def _splice_body(
    old_body: np.ndarray,
    old_sample_offsets: np.ndarray,
    old_group_offsets: np.ndarray,
    old_code_offsets: np.ndarray,
    old_short_offsets: np.ndarray,
    old_long_offsets: np.ndarray,
    old_byte_planes: bool,
    has_time: bool,
    old_time_starts: np.ndarray,
    new_ranks: np.ndarray,
    new_params: np.ndarray,
    new_anchors: np.ndarray,
    new_residuals: np.ndarray,
    new_codes: np.ndarray,
    new_sample_offsets: np.ndarray,
    new_time_starts: np.ndarray,
    new_time_steps: np.ndarray,
    new_time_refs: np.ndarray,
    new_time_residuals: np.ndarray,
    block_flags: np.ndarray,
    block_sizes: np.ndarray,
    sample_offsets: np.ndarray,
    group_offsets: np.ndarray,
    code_offsets: np.ndarray,
    short_offsets: np.ndarray,
    long_offsets: np.ndarray,
    byte_planes: bool,
    out_body: np.ndarray,
) -> None:
    """Writes a body whose blocks come from the new rows (new_ranks[b] >= 0) or are old block
    b, whose bytes are copied (see splice_body)."""
    num_blocks = block_flags.shape[0]
    old_num_blocks = old_sample_offsets.shape[0] - 1
    old_groups, groups = int(old_group_offsets[old_num_blocks]), int(group_offsets[num_blocks])
    old_planes = planes_view(old_body, old_num_blocks, old_groups, has_time)
    old_bplanes = byte_planes_view(old_body, old_num_blocks, old_groups, has_time)
    old_cplanes = code_planes_view(old_body, old_num_blocks, old_groups, int(old_code_offsets[old_num_blocks]), has_time)
    planes = planes_view(out_body, num_blocks, groups, has_time)
    bplanes = byte_planes_view(out_body, num_blocks, groups, has_time)
    cplanes = code_planes_view(out_body, num_blocks, groups, int(code_offsets[num_blocks]), has_time)
    if has_time:
        old_short, old_long = time_planes_views(
            old_body, old_num_blocks, old_groups, int(old_code_offsets[old_num_blocks]),
            int(old_short_offsets[old_num_blocks]), int(old_long_offsets[old_num_blocks]),
        )
        short_planes, long_planes = time_planes_views(
            out_body, num_blocks, groups, int(code_offsets[num_blocks]), int(short_offsets[num_blocks]),
            int(long_offsets[num_blocks]),
        )
        # New blocks write only the levels their residuals reach
        short_planes[:, :] = 0
        long_planes[:, :] = 0
    else:
        old_short = old_long = short_planes = long_planes = np.zeros((TIME_SHORT_PLANES, 0), np.uint8)
    max_len = 0
    for block_idx in range(num_blocks):
        max_len = max(max_len, block_sizes[block_idx])
    scratch_low_bytes = np.empty(8 * plane_groups(max_len), np.uint8)
    scratch_high_bytes = np.empty(8 * plane_groups(max_len), np.uint8)
    scratch_tail = np.empty(8, np.uint64)
    # The column fields as (start in the new body, start in the old body, byte planes)
    column_fields = [
        (0, 0, 1),
        (block_sizes_start(num_blocks), block_sizes_start(old_num_blocks), BYTES_PER_SIZE),
        (grid_params_start(num_blocks), grid_params_start(old_num_blocks), BYTES_PER_PARAM),
        (value_anchor_start(num_blocks), value_anchor_start(old_num_blocks), BYTES_PER_ANCHOR),
    ]
    if has_time:
        column_fields.append((time_start_start(num_blocks), time_start_start(old_num_blocks), TIME_BYTES_PER_BLOCK))
    previous_start = np.int64(0)
    seen_samples = False
    block_idx = 0
    while block_idx < num_blocks:
        rank = new_ranks[block_idx]
        if rank < 0:
            # A run of carried blocks: consecutive in every field of both bodies, so each row
            # of each field is one copy
            run_end = block_idx + 1
            while run_end < num_blocks and new_ranks[run_end] < 0:
                run_end += 1
            for field_start, old_field_start, num_planes in column_fields:
                # time_start, time_step and time_ref are consecutive fields: one pass
                for plane_idx in range(num_planes):
                    _copy_bytes(
                        out_body, field_start + plane_idx * num_blocks + block_idx,
                        old_body, old_field_start + plane_idx * old_num_blocks + block_idx, run_end - block_idx,
                    )
            if has_time:
                # Starts are stored as the increase over the previous non-empty block's: only
                # the run's first non-empty block follows a block that may have changed
                last_filled = -1
                for run_idx in range(block_idx, run_end):
                    if block_sizes[run_idx]:
                        if last_filled < 0:
                            start = old_time_starts[run_idx]
                            put_int64(
                                out_body, time_start_start(num_blocks), num_blocks, run_idx,
                                np.int64(np.uint64(start) - np.uint64(previous_start)) if seen_samples else start,
                            )
                        last_filled = run_idx
                if last_filled >= 0:
                    previous_start = old_time_starts[last_filled]
                    seen_samples = True
            _copy_planes(
                old_planes, old_bplanes, old_cplanes, old_short, old_long, old_group_offsets, old_code_offsets,
                old_short_offsets, old_long_offsets, old_byte_planes, planes, bplanes, cplanes, short_planes,
                long_planes, group_offsets, code_offsets, short_offsets, long_offsets, byte_planes, block_flags,
                block_sizes, block_idx, run_end, scratch_low_bytes, scratch_high_bytes,
            )
            block_idx = run_end
            continue
        block_len = block_sizes[block_idx]
        flags = block_flags[block_idx]
        out_body[block_idx] = flags
        out_body[block_sizes_start(num_blocks) + block_idx] = np.uint8(block_len & 0xFF)
        out_body[block_sizes_start(num_blocks) + num_blocks + block_idx] = np.uint8(block_len >> 8)
        put_int16(out_body, grid_params_start(num_blocks), num_blocks, block_idx, new_params[rank])
        put_int64(out_body, value_anchor_start(num_blocks), num_blocks, block_idx, new_anchors[rank])
        if has_time:
            start_increase, time_step, time_ref = np.int64(0), np.int64(0), np.int64(0)
            if block_len:
                start = new_time_starts[rank]
                time_step, time_ref = new_time_steps[rank], np.int64(new_time_refs[rank])
                start_increase = np.int64(np.uint64(start) - np.uint64(previous_start)) if seen_samples else start
                previous_start = start
                seen_samples = True
            put_int64(out_body, time_start_start(num_blocks), num_blocks, block_idx, start_increase)
            put_int64(out_body, time_step_start(num_blocks), num_blocks, block_idx, time_step)
            put_int64(out_body, time_ref_start(num_blocks), num_blocks, block_idx, time_ref)
        block_idx += 1
        if block_len == 0:
            continue
        group_offset = group_offsets[block_idx - 1]
        first_sample = new_sample_offsets[rank]
        block_residuals = new_residuals[first_sample:first_sample + block_len]
        if byte_planes:
            sample_start = 8 * group_offset
            zigzag_block(
                block_residuals,
                bplanes[0, sample_start:sample_start + block_len],
                bplanes[1, sample_start:sample_start + block_len],
            )
            bplanes[:, sample_start + block_len:8 * group_offsets[block_idx]] = 0
        else:
            shuffle_block(block_residuals, planes, group_offset, scratch_low_bytes, scratch_high_bytes)
        if flags & BLOCK_FLAG_NONFINITE:
            put_codes(new_codes[first_sample:first_sample + block_len], cplanes, code_offsets[block_idx - 1])
        if has_time and flags & BLOCK_FLAG_IRREGULAR_TIME:
            shuffle_time_residuals(
                new_time_residuals[first_sample:first_sample + block_len],
                short_planes,
                long_planes,
                short_offsets[block_idx - 1],
                long_offsets[block_idx - 1],
                scratch_tail,
            )


@njit(nogil=True, cache=True)
def _copy_planes(
    old_planes: np.ndarray,
    old_bplanes: np.ndarray,
    old_cplanes: np.ndarray,
    old_short: np.ndarray,
    old_long: np.ndarray,
    old_group_offsets: np.ndarray,
    old_code_offsets: np.ndarray,
    old_short_offsets: np.ndarray,
    old_long_offsets: np.ndarray,
    old_byte_planes: bool,
    planes: np.ndarray,
    bplanes: np.ndarray,
    cplanes: np.ndarray,
    short_planes: np.ndarray,
    long_planes: np.ndarray,
    group_offsets: np.ndarray,
    code_offsets: np.ndarray,
    short_offsets: np.ndarray,
    long_offsets: np.ndarray,
    byte_planes: bool,
    block_flags: np.ndarray,
    block_sizes: np.ndarray,
    first_block: int,
    end_block: int,
    scratch_low_bytes: np.ndarray,
    scratch_high_bytes: np.ndarray,
) -> None:
    """Copies the plane bytes of the carried blocks first_block to end_block: each field's
    bytes of a run of blocks are consecutive in both bodies. Residuals in the other plane
    mode are converted block by block."""
    num_groups = group_offsets[end_block] - group_offsets[first_block]
    old_group_offset, group_offset = old_group_offsets[first_block], group_offsets[first_block]
    if byte_planes == old_byte_planes:
        if byte_planes:
            _copy_columns(bplanes, 8 * group_offset, old_bplanes, 8 * old_group_offset, 8 * num_groups)
        else:
            _copy_columns(planes, group_offset, old_planes, old_group_offset, num_groups)
    else:
        for block_idx in range(first_block, end_block):
            block_len = block_sizes[block_idx]
            block_groups = group_offsets[block_idx + 1] - group_offsets[block_idx]
            old_start, start = 8 * old_group_offsets[block_idx], 8 * group_offsets[block_idx]
            # The copies go through _copy_bytes: numba's slice assignment costs several times more
            if byte_planes:
                # Bit planes to byte planes: the zigzag bytes, then zero padding
                unshuffle_block(old_planes, old_group_offsets[block_idx], block_groups, scratch_low_bytes,
                                scratch_high_bytes)
                scratch_low_bytes[block_len:8 * block_groups] = 0
                scratch_high_bytes[block_len:8 * block_groups] = 0
                _copy_bytes(bplanes[0], start, scratch_low_bytes, 0, 8 * block_groups)
                _copy_bytes(bplanes[1], start, scratch_high_bytes, 0, 8 * block_groups)
            else:
                # Byte planes to bit planes
                _copy_bytes(scratch_low_bytes, 0, old_bplanes[0], old_start, block_len)
                _copy_bytes(scratch_high_bytes, 0, old_bplanes[1], old_start, block_len)
                scratch_low_bytes[block_len:8 * block_groups] = 0
                scratch_high_bytes[block_len:8 * block_groups] = 0
                shuffle_bytes(scratch_low_bytes, scratch_high_bytes, planes, group_offsets[block_idx], block_groups)
    _copy_columns(cplanes, code_offsets[first_block], old_cplanes, old_code_offsets[first_block],
                  code_offsets[end_block] - code_offsets[first_block])
    _copy_columns(short_planes, short_offsets[first_block], old_short, old_short_offsets[first_block],
                  short_offsets[end_block] - short_offsets[first_block])
    _copy_columns(long_planes, long_offsets[first_block], old_long, old_long_offsets[first_block],
                  long_offsets[end_block] - long_offsets[first_block])


def splice_body(
    old_body: np.ndarray,
    old_layout: Layout,
    old_byte_planes: bool,
    old_time_starts: np.ndarray | None,
    new_block_ids: np.ndarray,
    new_rows: UnitRows,
    byte_planes: bool,
) -> np.ndarray:
    """Builds the body of a unit with some blocks replaced or appended, copying the others' bytes.

    In every plane field a block's bytes start on a byte boundary, so a block carried over
    is copied from the old body as it is: no unpacking, and its padding bits stay as they
    were. Only when the plane modes differ is a carried block's residual field converted
    (between bit and byte planes, on its own). The new blocks are packed from their rows.
    Without padding bits set in the old body, the result is write_unit of the merged rows.

    Args:
        old_body: 1D uint8 array of the old unit's uncompressed body.
        old_layout: Its Layout.
        old_byte_planes: Whether its residuals are byte planes.
        old_time_starts: 1D int64 array of its absolute block start times
            (read_time_columns), or None for a unit without a time axis.
        new_block_ids: 1D int64 array of the new blocks' indices, increasing. Every block
            past the old unit's end must be among them.
        new_rows: The new blocks' rows, in new_block_ids order (time_rows set exactly when
            the unit has a time axis).
        byte_planes: Store the residuals as byte planes instead of bit planes.

    Returns:
        1D uint8 array of the new body.
    """
    old_num_blocks = old_layout.sample_offsets.shape[0] - 1
    num_blocks = max(old_num_blocks, int(new_block_ids[-1]) + 1) if new_block_ids.shape[0] else old_num_blocks
    has_time = old_time_starts is not None
    new_ranks = np.full(num_blocks, -1, np.int64)
    new_ranks[new_block_ids] = np.arange(new_block_ids.shape[0])
    if (new_ranks[old_num_blocks:] < 0).any():
        raise ValueError("every block past the old unit's end must be new")
    block_flags = np.empty(num_blocks, np.uint8)
    block_flags[:old_num_blocks] = old_body[:old_num_blocks]
    block_flags[new_block_ids] = new_rows.block_flags
    block_sizes = np.empty(num_blocks, np.int64)
    block_sizes[:old_num_blocks] = np.diff(old_layout.sample_offsets)
    block_sizes[new_block_ids] = new_rows.block_sizes
    offsets = layout(block_flags, block_sizes)
    out_body = np.empty(unit_size(num_blocks, offsets, has_time), np.uint8)
    new_time = new_rows.time_rows
    if new_time is None:
        new_time = allocate_time_rows(0, 0)
    _splice_body(
        old_body, *old_layout, old_byte_planes, has_time,
        old_time_starts if old_time_starts is not None else np.zeros(0, np.int64),
        new_ranks, new_rows.grid_params, new_rows.value_anchors, new_rows.residuals, new_rows.codes,
        np.concatenate([[0], np.cumsum(new_rows.block_sizes)]).astype(np.int64), *new_time,
        block_flags, block_sizes, *offsets, byte_planes, out_body,
    )
    return out_body


def check_time_rows_status(status: int, block_idx: int) -> None:
    """Raises the error for a read_time_rows status other than TIME_ROWS_OK.

    Raises:
        ValueError: If the status reports a corrupt time column.
    """
    if status == TIME_ROWS_BAD_START:
        raise ValueError(f"block {block_idx}: start time out of range (corrupt unit)")
    if status == TIME_ROWS_BAD_LONG:
        raise ValueError(f"block {block_idx}: long time residuals fit in 32 bits (corrupt unit)")
    if status == TIME_ROWS_BAD_EMPTY:
        raise ValueError(f"block {block_idx}: empty block with nonzero time columns (corrupt unit)")
    if status == TIME_ROWS_BAD_SINGLE:
        raise ValueError(
            f"block {block_idx}: a single sample with a nonzero time step or reference, or irregular (corrupt unit)"
        )
