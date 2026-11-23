# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Low-level bit-plane and byte-plane codecs for residuals, sample codes, and time residuals.

Each block's plane data is aligned to 8-sample plane bytes (``SAMPLES_PER_PLANE_BYTE = 8``),
so 8x8 bit-matrix transposes operate on whole 64-bit words and each block can be packed or
unpacked independently.

Higher-level body serialization, time-column validation, and binary body splicing live in
``_body``.
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


@njit(inline="always")
def _transpose_8x8_bits(matrix_bits: np.uint64) -> np.uint64:
    """Transposes an 8x8 bit matrix stored in a uint64 word.

    Maps bit ``(byte k, bit j)`` to ``(byte j, bit k)``. The transform is self-inverse.

    Args:
        matrix_bits: 64-bit word viewed as eight 8-bit rows.

    Returns:
        Transposed 64-bit word.
    """
    # Classic Hacker's Delight 3-stage delta-swap transpose (1x1 -> 2x2 -> 4x4 sub-blocks).
    # Stage 1: swap off-diagonal 1x1 bits separated by 7 bit positions.
    delta_bits = (matrix_bits ^ (matrix_bits >> np.uint64(7))) & np.uint64(0x00AA00AA00AA00AA)
    matrix_bits = matrix_bits ^ delta_bits ^ (delta_bits << np.uint64(7))
    # Stage 2: swap off-diagonal 2x2 bit blocks separated by 14 bit positions.
    delta_bits = (matrix_bits ^ (matrix_bits >> np.uint64(14))) & np.uint64(0x0000CCCC0000CCCC)
    matrix_bits = matrix_bits ^ delta_bits ^ (delta_bits << np.uint64(14))
    # Stage 3: swap off-diagonal 4x4 bit blocks separated by 28 bit positions.
    delta_bits = (matrix_bits ^ (matrix_bits >> np.uint64(28))) & np.uint64(0x00000000F0F0F0F0)
    return matrix_bits ^ delta_bits ^ (delta_bits << np.uint64(28))


@njit(inline="always")
def residual_bit_planes_view(body: np.ndarray, num_blocks: int, num_plane_bytes: int, has_time: bool) -> np.ndarray:
    """Returns a 2D view of the 16 residual bit planes in an uncompressed body.

    Args:
        body: 1D uint8 array of uncompressed body bytes.
        num_blocks: Number of blocks in the group.
        num_plane_bytes: Total 8-sample plane bytes across all blocks.
        has_time: Whether the body includes time metadata columns before the residuals.

    Returns:
        2D uint8 array of shape ``(16, num_plane_bytes)``.
    """
    start_offset = residual_planes_start(num_blocks, has_time)
    end_offset = start_offset + BYTES_PER_RESIDUAL_SAMPLE * SAMPLES_PER_PLANE_BYTE * num_plane_bytes
    return body[start_offset:end_offset].reshape(16, num_plane_bytes)


@njit(inline="always")
def residual_byte_planes_view(body: np.ndarray, num_blocks: int, num_plane_bytes: int, has_time: bool) -> np.ndarray:
    """Returns a 2D view of the 2 residual byte planes in an uncompressed body.

    Args:
        body: 1D uint8 array of uncompressed body bytes.
        num_blocks: Number of blocks in the group.
        num_plane_bytes: Total 8-sample plane bytes across all blocks.
        has_time: Whether the body includes time metadata columns before the residuals.

    Returns:
        2D uint8 array of shape ``(2, 8 * num_plane_bytes)``: row 0 holds all low bytes,
        row 1 holds all high bytes, with each block zero-padded to a multiple of 8 samples.
    """
    start_offset = residual_planes_start(num_blocks, has_time)
    end_offset = start_offset + BYTES_PER_RESIDUAL_SAMPLE * SAMPLES_PER_PLANE_BYTE * num_plane_bytes
    return body[start_offset:end_offset].reshape(2, SAMPLES_PER_PLANE_BYTE * num_plane_bytes)


@njit(nogil=True, cache=True)
def wide_residual_fraction(
    body: np.ndarray, num_blocks: int, num_plane_bytes: int, has_time: bool, bit_idx: int
) -> float:
    """Computes the fraction of residual slots whose zigzagged value reaches ``2^bit_idx``.

    Slots include trailing zero padding (8 slots per plane byte). Used by the compression
    heuristic to choose between byte planes and bit planes.

    Args:
        body: 1D uint8 array of a byte-plane uncompressed body.
        num_blocks: Number of blocks in the group.
        num_plane_bytes: Total 8-sample plane bytes across all blocks.
        has_time: Whether the body has a time axis.
        bit_idx: Bit threshold in ``0..15``.

    Returns:
        Fraction in ``[0.0, 1.0]`` of slots with any bit in ``bit_idx..15`` set (0.0 if empty).
    """
    # Process 8 residual slots at a time using SWAR (SIMD Within A Register) on uint64 words:
    # mask bits >= bit_idx in each byte lane, OR-fold each lane into its bit 0, then sum the
    # 8 lane flags via a single multiply-and-shift.
    if not num_plane_bytes:
        return 0.0
    byte_planes = residual_byte_planes_view(body, num_blocks, num_plane_bytes, has_time)
    low_words = byte_planes[0].view(np.uint64)
    high_words = byte_planes[1].view(np.uint64)
    ones = np.uint64(0x0101010101010101)
    # Replicate the per-byte threshold mask across all 8 byte lanes of a 64-bit word.
    low_mask = np.uint64(((0xFF << bit_idx) & 0xFF if bit_idx < 8 else 0) * 0x0101010101010101)
    high_mask = np.uint64(((0xFF << max(bit_idx - 8, 0)) & 0xFF) * 0x0101010101010101)
    wide_count = 0
    for word_idx in range(num_plane_bytes):
        word = (low_words[word_idx] & low_mask) | (high_words[word_idx] & high_mask)
        # Fold any set bit in each byte down to bit 0 of that byte.
        word |= word >> np.uint64(4)
        word |= word >> np.uint64(2)
        word |= word >> np.uint64(1)
        # Multiplying by 0x0101... accumulates the 8 lane bits into the top byte (bits 56..63).
        wide_count += int(((word & ones) * ones) >> np.uint64(56))
    return wide_count / (SAMPLES_PER_PLANE_BYTE * num_plane_bytes)


@njit(inline="always")
def zigzag_encode_block(residuals: np.ndarray, out_low_bytes: np.ndarray, out_high_bytes: np.ndarray) -> None:
    """Zigzag-encodes signed int16 residuals and splits them into low and high byte arrays.

    Maps signed integers ``0, -1, 1, -2, 2, ...`` to unsigned ``0, 1, 2, 3, 4, ...`` so small
    magnitudes have leading zero bits regardless of sign.

    Args:
        residuals: 1D int16 array of difference residuals for one block.
        out_low_bytes: Output 1D uint8 array of length ``len(residuals)`` for low bytes.
        out_high_bytes: Output 1D uint8 array of length ``len(residuals)`` for high bytes.
    """
    for sample_idx in range(residuals.shape[0]):
        signed_residual = residuals[sample_idx]
        zigzag_val = np.uint16((signed_residual << np.int16(1)) ^ (signed_residual >> np.int16(15)))
        out_low_bytes[sample_idx] = np.uint8(zigzag_val)
        out_high_bytes[sample_idx] = np.uint8(zigzag_val >> np.uint16(8))


@njit(inline="always")
def unzigzag_sample(low_bytes: np.ndarray, high_bytes: np.ndarray, sample_idx: int) -> np.int16:
    """Decodes one signed int16 residual from low and high zigzag byte arrays.

    Args:
        low_bytes: 1D uint8 array of lower zigzag bytes.
        high_bytes: 1D uint8 array of upper zigzag bytes.
        sample_idx: Index of the sample to decode.

    Returns:
        Signed 16-bit residual.
    """
    zigzag_val = np.uint16(low_bytes[sample_idx]) | (np.uint16(high_bytes[sample_idx]) << np.uint16(8))
    # Inverse zigzag: right-shift magnitude and XOR with all-ones sign mask when LSB is 1.
    return np.int16(zigzag_val >> np.uint16(1)) ^ -np.int16(zigzag_val & np.uint16(1))


@njit(nogil=True, cache=True)
def shuffle_residual_bytes(
    low_bytes: np.ndarray,
    high_bytes: np.ndarray,
    bit_planes: np.ndarray,
    plane_byte_offset: int,
    num_plane_bytes: int,
) -> None:
    """Transposes zigzagged low/high byte arrays into 16 residual bit planes.

    Args:
        low_bytes: 1D uint8 array of at least ``8 * num_plane_bytes`` low zigzag bytes.
        high_bytes: 1D uint8 array of at least ``8 * num_plane_bytes`` high zigzag bytes.
        bit_planes: Destination 2D uint8 array of shape ``(16, total_plane_bytes)``.
        plane_byte_offset: Starting plane-byte column for this block.
        num_plane_bytes: Number of 8-sample plane bytes to transpose.
    """
    # View each 8-byte group as a uint64 word, run an 8x8 bit transpose, and scatter the
    # resulting 8 bytes into bit planes 0-7 (low) and 8-15 (high).
    low_words = low_bytes[: SAMPLES_PER_PLANE_BYTE * num_plane_bytes].view(np.uint64)
    high_words = high_bytes[: SAMPLES_PER_PLANE_BYTE * num_plane_bytes].view(np.uint64)
    for group_idx in range(num_plane_bytes):
        transposed_low = _transpose_8x8_bits(low_words[group_idx])
        transposed_high = _transpose_8x8_bits(high_words[group_idx])
        for bit_idx in range(8):
            bit_planes[bit_idx, plane_byte_offset + group_idx] = np.uint8(
                (transposed_low >> np.uint64(8 * bit_idx)) & np.uint64(0xFF)
            )
            bit_planes[8 + bit_idx, plane_byte_offset + group_idx] = np.uint8(
                (transposed_high >> np.uint64(8 * bit_idx)) & np.uint64(0xFF)
            )


@njit(nogil=True, cache=True)
def unshuffle_residual_block(
    bit_planes: np.ndarray,
    plane_byte_offset: int,
    num_plane_bytes: int,
    out_low_bytes: np.ndarray,
    out_high_bytes: np.ndarray,
) -> None:
    """Reconstructs a block's zigzagged low and high byte arrays from 16 residual bit planes.

    Args:
        bit_planes: Source 2D uint8 array of shape ``(16, total_plane_bytes)``.
        plane_byte_offset: Starting plane-byte column for this block.
        num_plane_bytes: Number of 8-sample plane bytes to unpack.
        out_low_bytes: Destination 1D uint8 array of length at least ``8 * num_plane_bytes``.
        out_high_bytes: Destination 1D uint8 array of length at least ``8 * num_plane_bytes``.
    """
    # Gather 8 bytes from planes 0-7 and 8-15 into uint64 words and apply the self-inverse
    # 8x8 bit transpose to recover contiguous low and high bytes for 8 samples at a time.
    low_words = out_low_bytes.view(np.uint64)
    high_words = out_high_bytes.view(np.uint64)
    for group_idx in range(num_plane_bytes):
        gathered_low = np.uint64(0)
        gathered_high = np.uint64(0)
        for bit_idx in range(8):
            gathered_low |= np.uint64(bit_planes[bit_idx, plane_byte_offset + group_idx]) << np.uint64(8 * bit_idx)
            gathered_high |= np.uint64(bit_planes[8 + bit_idx, plane_byte_offset + group_idx]) << np.uint64(8 * bit_idx)
        low_words[group_idx] = _transpose_8x8_bits(gathered_low)
        high_words[group_idx] = _transpose_8x8_bits(gathered_high)


@njit(nogil=True, cache=True)
def _bit_planes_into(
    byte_body: np.ndarray, num_blocks: int, num_plane_bytes: int, has_time: bool, out_body: np.ndarray
) -> None:
    """Transposes the residual region of a byte-plane body into ``out_body`` as bit planes.

    Args:
        byte_body: Source 1D uint8 uncompressed body with byte-planed residuals.
        num_blocks: Number of blocks in the group.
        num_plane_bytes: Total 8-sample plane bytes across all blocks.
        has_time: Whether the body has a time axis.
        out_body: Destination 1D uint8 body array (non-residual bytes must already be copied).
    """
    # Because every block is padded to a multiple of 8 samples, all blocks' plane bytes are
    # contiguous and can be transposed in one sweep.
    if num_plane_bytes:
        byte_planes = residual_byte_planes_view(byte_body, num_blocks, num_plane_bytes, has_time)
        bit_planes = residual_bit_planes_view(out_body, num_blocks, num_plane_bytes, has_time)
        shuffle_residual_bytes(byte_planes[0], byte_planes[1], bit_planes, 0, num_plane_bytes)


@njit(nogil=True, cache=True)
def _byte_planes_into(
    bit_body: np.ndarray, num_blocks: int, num_plane_bytes: int, has_time: bool, out_body: np.ndarray
) -> None:
    """Transposes the residual region of a bit-plane body into ``out_body`` as byte planes.

    Args:
        bit_body: Source 1D uint8 uncompressed body with bit-planed residuals.
        num_blocks: Number of blocks in the group.
        num_plane_bytes: Total 8-sample plane bytes across all blocks.
        has_time: Whether the body has a time axis.
        out_body: Destination 1D uint8 body array (non-residual bytes must already be copied).
    """
    if num_plane_bytes:
        byte_planes = residual_byte_planes_view(out_body, num_blocks, num_plane_bytes, has_time)
        bit_planes = residual_bit_planes_view(bit_body, num_blocks, num_plane_bytes, has_time)
        unshuffle_residual_block(bit_planes, 0, num_plane_bytes, byte_planes[0], byte_planes[1])


def to_bit_planes(byte_body: np.ndarray, num_blocks: int, num_plane_bytes: int, has_time: bool) -> np.ndarray:
    """Converts a byte-plane uncompressed body into a bit-plane uncompressed body.

    Args:
        byte_body: 1D uint8 array of a body whose residuals are stored as 2 byte planes.
        num_blocks: Number of blocks in the group.
        num_plane_bytes: Total 8-sample plane bytes across all blocks.
        has_time: Whether the body has a time axis.

    Returns:
        New 1D uint8 body array with the residual region converted to 16 bit planes.
    """
    bit_body = byte_body.copy()
    _bit_planes_into(byte_body, num_blocks, num_plane_bytes, has_time, bit_body)
    return bit_body


def to_byte_planes(bit_body: np.ndarray, num_blocks: int, num_plane_bytes: int, has_time: bool) -> np.ndarray:
    """Converts a bit-plane uncompressed body into a byte-plane uncompressed body.

    Args:
        bit_body: 1D uint8 array of a body whose residuals are stored as 16 bit planes.
        num_blocks: Number of blocks in the group.
        num_plane_bytes: Total 8-sample plane bytes across all blocks.
        has_time: Whether the body has a time axis.

    Returns:
        New 1D uint8 body array with the residual region converted to 2 byte planes.
    """
    byte_body = bit_body.copy()
    _byte_planes_into(bit_body, num_blocks, num_plane_bytes, has_time, byte_body)
    return byte_body


@njit(inline="always")
def sample_code_planes_view(
    body: np.ndarray, num_blocks: int, num_plane_bytes: int, num_code_plane_bytes: int, has_time: bool
) -> np.ndarray:
    """Returns a 2D view of the 2-bit sample code planes in an uncompressed body.

    Args:
        body: 1D uint8 array of uncompressed body bytes.
        num_blocks: Number of blocks in the group.
        num_plane_bytes: Total residual plane bytes across all blocks.
        num_code_plane_bytes: Total plane bytes across blocks with ``BLOCK_FLAG_SAMPLE_CODES``.
        has_time: Whether the body has a time axis.

    Returns:
        2D uint8 array of shape ``(2, num_code_plane_bytes)``.
    """
    start_offset = sample_code_planes_start(num_blocks, num_plane_bytes, has_time)
    total_code_bytes = num_code_plane_bytes * BITS_PER_SAMPLE_CODE
    return body[start_offset : start_offset + total_code_bytes].reshape(2, num_code_plane_bytes)


@njit(inline="always")
def _pack_sample_code_byte(
    sample_codes: np.ndarray, group_idx: int, num_valid: int, code_planes: np.ndarray, plane_byte_offset: int
) -> None:
    """Packs up to 8 two-bit sample codes into one byte of each code plane.

    Args:
        sample_codes: 1D uint8 array of per-sample codes (values 0..3).
        group_idx: Zero-based 8-sample group index within the block.
        num_valid: Number of valid samples in this group (1..8); higher bits stay zero.
        code_planes: Destination 2D uint8 array of shape ``(2, total_code_plane_bytes)``.
        plane_byte_offset: Starting plane-byte column for this block.
    """
    plane0_byte = 0
    plane1_byte = 0
    for bit_idx in range(num_valid):
        code_val = sample_codes[SAMPLES_PER_PLANE_BYTE * group_idx + bit_idx]
        # Bit 0 of the 2-bit code goes into plane 0; bit 1 goes into plane 1.
        plane0_byte |= (code_val & 1) << bit_idx
        plane1_byte |= ((code_val >> 1) & 1) << bit_idx
    code_planes[0, plane_byte_offset + group_idx] = plane0_byte
    code_planes[1, plane_byte_offset + group_idx] = plane1_byte


@njit(nogil=True, cache=True)
def pack_sample_codes(sample_codes: np.ndarray, code_planes: np.ndarray, plane_byte_offset: int) -> None:
    """Packs a block's 2-bit sample exception codes into two bit planes.

    Args:
        sample_codes: 1D uint8 array of per-sample codes (values in 0..3).
        code_planes: Destination 2D uint8 array of shape ``(2, total_code_plane_bytes)``.
        plane_byte_offset: Starting plane-byte column for this block.
    """
    # Split full 8-sample groups from the optional partial trailing group so the inner
    # loop over 8 bits has a constant trip count and vectorizes cleanly under Numba/LLVM.
    num_samples = sample_codes.shape[0]
    num_full_groups = num_samples // SAMPLES_PER_PLANE_BYTE
    for group_idx in range(num_full_groups):
        _pack_sample_code_byte(sample_codes, group_idx, SAMPLES_PER_PLANE_BYTE, code_planes, plane_byte_offset)
    remainder = num_samples % SAMPLES_PER_PLANE_BYTE
    if remainder:
        _pack_sample_code_byte(sample_codes, num_full_groups, remainder, code_planes, plane_byte_offset)


@njit(inline="always")
def unpack_sample_code(plane0_byte: int, plane1_byte: int, bit_idx: int) -> int:
    """Extracts one sample's 2-bit exception code from a pair of code-plane bytes.

    Args:
        plane0_byte: Byte from code plane 0 for the sample's 8-sample group.
        plane1_byte: Byte from code plane 1 for the sample's 8-sample group.
        bit_idx: Sample position within the 8-sample group (0..7).

    Returns:
        Two-bit exception code in ``0..3``.
    """
    return ((plane0_byte >> bit_idx) & 1) | (((plane1_byte >> bit_idx) & 1) << 1)


@njit(inline="always")
def time_residual_planes_views(
    body: np.ndarray,
    num_blocks: int,
    num_plane_bytes: int,
    num_code_plane_bytes: int,
    num_time_low_bytes: int,
    num_time_high_bytes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns 2D views of the low (0-31) and high (32-63) time residual bit planes.

    Args:
        body: 1D uint8 array of uncompressed body bytes (must have a time axis).
        num_blocks: Number of blocks in the group.
        num_plane_bytes: Total residual plane bytes across all blocks.
        num_code_plane_bytes: Total sample code plane bytes across flagged blocks.
        num_time_low_bytes: Total plane bytes across irregular-time blocks.
        num_time_high_bytes: Total plane bytes across wide-time blocks.

    Returns:
        Tuple ``(time_low_planes, time_high_planes)`` of shapes
        ``(32, num_time_low_bytes)`` and ``(32, num_time_high_bytes)``.
    """
    low_start = (
        sample_code_planes_start(num_blocks, num_plane_bytes, True) + num_code_plane_bytes * BITS_PER_SAMPLE_CODE
    )
    low_bytes = TIME_LOW_PLANES * num_time_low_bytes
    high_bytes = TIME_LOW_PLANES * num_time_high_bytes
    time_low_planes = body[low_start : low_start + low_bytes].reshape(TIME_LOW_PLANES, num_time_low_bytes)
    high_start = low_start + low_bytes
    time_high_planes = body[high_start : high_start + high_bytes].reshape(TIME_LOW_PLANES, num_time_high_bytes)
    return time_low_planes, time_high_planes


@njit(inline="always")
def _gather_byte_lane(
    val0: np.uint64,
    val1: np.uint64,
    val2: np.uint64,
    val3: np.uint64,
    val4: np.uint64,
    val5: np.uint64,
    val6: np.uint64,
    val7: np.uint64,
    shift: np.uint64,
) -> np.uint64:
    """Packs the byte at bit offset ``shift`` from each of eight uint64 values into one word.

    Args:
        val0..val7: Eight uint64 values (``valk`` supplies byte ``k`` of the output word).
        shift: Bit offset of the selected byte (``8 * byte_idx``).

    Returns:
        64-bit word holding byte ``shift // 8`` of ``val0`` through ``val7``.
    """
    mask = np.uint64(0xFF)
    return (
        ((val0 >> shift) & mask)
        | (((val1 >> shift) & mask) << np.uint64(8))
        | (((val2 >> shift) & mask) << np.uint64(16))
        | (((val3 >> shift) & mask) << np.uint64(24))
        | (((val4 >> shift) & mask) << np.uint64(32))
        | (((val5 >> shift) & mask) << np.uint64(40))
        | (((val6 >> shift) & mask) << np.uint64(48))
        | (((val7 >> shift) & mask) << np.uint64(56))
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
    """Returns a ``(8, end_group - first_group)`` view of the 8 bit planes for one byte level.

    Byte levels 0-3 live in ``time_low_planes`` (bit planes 0-31); byte levels 4-7 live in
    ``time_high_planes`` (bit planes 32-63).

    Args:
        time_low_planes: 2D uint8 array of bit planes 0-31.
        time_high_planes: 2D uint8 array of bit planes 32-63.
        time_low_offset: Block's starting column in ``time_low_planes``.
        time_high_offset: Block's starting column in ``time_high_planes``.
        byte_idx: Byte level index in ``0..7``.
        first_group: Starting 8-sample group index within the block.
        end_group: Exclusive ending 8-sample group index within the block.

    Returns:
        2D uint8 array slice of shape ``(8, end_group - first_group)``.
    """
    if byte_idx < 4:
        return time_low_planes[
            8 * byte_idx : 8 * byte_idx + 8, time_low_offset + first_group : time_low_offset + end_group
        ]
    high_level = byte_idx - 4
    return time_high_planes[
        8 * high_level : 8 * high_level + 8, time_high_offset + first_group : time_high_offset + end_group
    ]


@njit(inline="always")
def _pack_time_plane_byte(
    values: np.ndarray, base: int, shift: np.uint64, level_planes: np.ndarray, group_idx: int
) -> None:
    """Packs one byte level of 8 uint64 time residuals into column ``group_idx`` of 8 bit planes.

    Args:
        values: 1D uint64 array of zigzagged time residuals.
        base: Starting sample index of the 8-sample group in ``values``.
        shift: Bit shift ``8 * byte_idx`` selecting the byte level.
        level_planes: 2D uint8 view of shape ``(8, num_groups)`` for this byte level.
        group_idx: Target column index in ``level_planes``.
    """
    word = _gather_byte_lane(
        values[base],
        values[base + 1],
        values[base + 2],
        values[base + 3],
        values[base + 4],
        values[base + 5],
        values[base + 6],
        values[base + 7],
        shift,
    )
    transposed = _transpose_8x8_bits(word)
    for bit_idx in range(8):
        level_planes[bit_idx, group_idx] = np.uint8((transposed >> np.uint64(8 * bit_idx)) & np.uint64(0xFF))


@njit(inline="always")
def _unpack_time_plane_byte(level_planes: np.ndarray, group_idx: int, byte_idx: int, out_bytes: np.ndarray) -> None:
    """Unpacks column ``group_idx`` of 8 bit planes into byte ``byte_idx`` of 8 uint64 samples.

    Args:
        level_planes: 2D uint8 view of shape ``(8, num_groups)`` for this byte level.
        group_idx: Source column index in ``level_planes``.
        byte_idx: Destination byte offset (0..7) within each little-endian uint64 sample.
        out_bytes: 1D uint8 view of the 8 destination uint64 residuals (64 bytes).
    """
    gathered = np.uint64(0)
    for bit_idx in range(8):
        gathered |= np.uint64(level_planes[bit_idx, group_idx]) << np.uint64(8 * bit_idx)
    # After transposing back, byte k of `transposed` is byte `byte_idx` of sample k.
    transposed = _transpose_8x8_bits(gathered)
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

    Only byte levels up to the highest set bit of any residual are written; callers must
    zero the target planes beforehand so unwritten upper planes remain zero.

    Args:
        time_residuals: 1D uint64 array of zigzagged time residuals for one irregular block
            (all values must fit in uint32 unless the block is wide-time).
        time_low_planes: 2D uint8 array of bit planes 0-31.
        time_high_planes: 2D uint8 array of bit planes 32-63.
        time_low_offset: Block's starting column in ``time_low_planes``.
        time_high_offset: Block's starting column in ``time_high_planes`` (unused if narrow).
        scratch_tail: Preallocated 1D uint64 scratch array of length 8 for a partial tail group.
    """
    # OR all residuals to find how many byte levels are non-zero, zero-pad any partial trailing
    # group into scratch_tail, then transpose only the active byte levels.
    num_samples = time_residuals.shape[0]
    num_full_groups = num_samples // SAMPLES_PER_PLANE_BYTE
    num_plane_bytes = plane_byte_count(num_samples)
    # Upper bytes that are zero in every sample can be skipped because the planes are pre-zeroed.
    all_bits = np.uint64(0)
    for sample_idx in range(num_samples):
        all_bits |= time_residuals[sample_idx]
    num_active_bytes = 0
    while num_active_bytes < 8 and (all_bits >> np.uint64(8 * num_active_bytes)) != 0:
        num_active_bytes += 1
    if num_plane_bytes > num_full_groups:
        # Copy the trailing 1..7 samples into a zero-padded 8-element buffer.
        scratch_tail[:] = 0
        scratch_tail[: num_samples - SAMPLES_PER_PLANE_BYTE * num_full_groups] = time_residuals[
            SAMPLES_PER_PLANE_BYTE * num_full_groups :
        ]
    for byte_idx in range(num_active_bytes):
        level_planes = _time_byte_level_planes(
            time_low_planes, time_high_planes, time_low_offset, time_high_offset, byte_idx, 0, num_plane_bytes
        )
        shift = np.uint64(8 * byte_idx)
        for group_idx in range(num_full_groups):
            _pack_time_plane_byte(time_residuals, SAMPLES_PER_PLANE_BYTE * group_idx, shift, level_planes, group_idx)
        if num_plane_bytes > num_full_groups:
            _pack_time_plane_byte(scratch_tail, 0, shift, level_planes, num_full_groups)


@njit(nogil=True, cache=True)
def _count_active_tail_bytes(
    time_low_planes: np.ndarray,
    time_high_planes: np.ndarray,
    time_low_offset: int,
    time_high_offset: int,
    num_plane_bytes: int,
    num_valid: int,
) -> int:
    """Finds the number of active byte levels in a block's partial trailing plane byte.

    Args:
        time_low_planes: 2D uint8 array of bit planes 0-31.
        time_high_planes: 2D uint8 array of bit planes 32-63.
        time_low_offset: Block's starting column in ``time_low_planes``.
        time_high_offset: Block's starting column in ``time_high_planes``, or -1 if narrow.
        num_plane_bytes: Total plane bytes allocated for this block.
        num_valid: Number of valid samples in the partial last group (1..7).

    Returns:
        Number of byte levels (0..8) up to the highest level with a non-zero valid bit.
    """
    # Mask off padding bits so stray bits in corrupt inputs do not inflate active levels.
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
def _unshuffle_time_tail(
    time_low_planes: np.ndarray,
    time_high_planes: np.ndarray,
    time_low_offset: int,
    time_high_offset: int,
    num_plane_bytes: int,
    num_active_bytes: int,
    out_time_residuals: np.ndarray,
    scratch_tail: np.ndarray,
) -> None:
    """Unpacks the partial trailing 8-sample group of a block's time residuals.

    Args:
        time_low_planes: 2D uint8 array of bit planes 0-31.
        time_high_planes: 2D uint8 array of bit planes 32-63.
        time_low_offset: Block's starting column in ``time_low_planes``.
        time_high_offset: Block's starting column in ``time_high_planes``, or -1 if narrow.
        num_plane_bytes: Total plane bytes allocated for this block.
        num_active_bytes: Number of non-zero byte levels to unpack.
        out_time_residuals: Destination 1D uint64 array of length ``block_len``.
        scratch_tail: Preallocated 1D uint64 scratch array of length 8.
    """
    scratch_tail[:] = 0
    tail_bytes = scratch_tail.view(np.uint8)
    for byte_idx in range(num_active_bytes):
        level_planes = _time_byte_level_planes(
            time_low_planes, time_high_planes, time_low_offset, time_high_offset, byte_idx, 0, num_plane_bytes
        )
        _unpack_time_plane_byte(level_planes, num_plane_bytes - 1, byte_idx, tail_bytes)
    num_full_samples = SAMPLES_PER_PLANE_BYTE * (num_plane_bytes - 1)
    out_time_residuals[num_full_samples:] = scratch_tail[: out_time_residuals.shape[0] - num_full_samples]


@njit(nogil=True, cache=True)
def unshuffle_time_residuals(
    time_low_planes: np.ndarray,
    time_high_planes: np.ndarray,
    time_low_offset: int,
    time_high_offset: int,
    out_time_residuals: np.ndarray,
    scratch_tail: np.ndarray,
) -> None:
    """Reconstructs a block's zigzagged uint64 time residuals from time residual bit planes.

    Args:
        time_low_planes: 2D uint8 array of bit planes 0-31.
        time_high_planes: 2D uint8 array of bit planes 32-63.
        time_low_offset: Block's starting column in ``time_low_planes``.
        time_high_offset: Block's starting column in ``time_high_planes``, or -1 if narrow.
        out_time_residuals: Destination 1D uint64 array of length ``block_len``.
        scratch_tail: Preallocated 1D uint64 scratch array of length 8 for a partial tail group.
    """
    # Scan downward from the top byte level to skip all-zero upper planes (most time residuals
    # are small), zero the output once if any upper levels are skipped, then transpose only the
    # active byte levels for full 8-sample groups followed by any partial trailing group.
    num_samples = out_time_residuals.shape[0]
    num_full_groups = num_samples // SAMPLES_PER_PLANE_BYTE
    num_plane_bytes = plane_byte_count(num_samples)
    remainder = num_samples % SAMPLES_PER_PLANE_BYTE
    num_tail_bytes = 0
    if remainder:
        num_tail_bytes = _count_active_tail_bytes(
            time_low_planes, time_high_planes, time_low_offset, time_high_offset, num_plane_bytes, remainder
        )
    # Find the highest byte level with any non-zero plane byte across the full groups.
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
    # Zero the destination when upper bytes will not be overwritten.
    if num_active_bytes < 8:
        out_time_residuals[:] = 0
    # View uint64 output as little-endian bytes so byte `byte_idx` of sample s sits at 8 * s + byte_idx.
    out_bytes = out_time_residuals.view(np.uint8)
    for byte_idx in range(num_active_bytes):
        level_planes = _time_byte_level_planes(
            time_low_planes, time_high_planes, time_low_offset, time_high_offset, byte_idx, 0, num_full_groups
        )
        for group_idx in range(num_full_groups):
            _unpack_time_plane_byte(level_planes, group_idx, byte_idx, out_bytes[64 * group_idx :])
    if remainder:
        _unshuffle_time_tail(
            time_low_planes,
            time_high_planes,
            time_low_offset,
            time_high_offset,
            num_plane_bytes,
            num_active_bytes,
            out_time_residuals,
            scratch_tail,
        )
