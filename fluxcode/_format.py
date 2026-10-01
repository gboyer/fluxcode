# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Binary layout and bit-shuffling routines for units.

A unit is a 16-byte header followed by one zstd frame holding the body. The
header (little-endian) is: format version (uint8, 1), flags (uint8; bit 0 selects
byte planes for the residual planes, bits 1-3 hold the time unit, bits 4-7 are 0),
reserved (uint16, 0), block length (uint32), and a uint64 whose low 40 bits are the
sample count and whose top 3 bytes are reserved (0). The unit holds
num_blocks = ceil(num_samples / block_len) blocks; the last one is padded when
num_samples isn't a multiple of block_len.

Each block takes num_groups = ceil(block_len / 8) bytes of every bit plane: when block_len
isn't a multiple of 8, the high bits of its last byte are padding, and byte planes pad each
block to 8 * num_groups bytes. Writers zero the padding; decoders ignore it.

The body, for num_blocks blocks of block_len samples, of which num_nonfinite_blocks
hold non-finite values (NaN, +inf, -inf), num_irregular_blocks have irregular times and
num_long_blocks of those have long time residuals, consists of these consecutive fields (the time fields only when the
header's time unit is not 0):

    1. block_flags (num_blocks bytes):
       Bits 0-1 encode the predictor difference order (0-3); bit 2 indicates
       decimal quantization; bit 3 indicates non-finite code planes; bit 4
       indicates irregular times (time residual planes); bit 5 (with bit 4) indicates
       long time residuals (64 planes instead of 32); bits 6-7 are reserved and must be zero.
    2. grid_params (8 * num_blocks bytes):
       Per-block int64 grid parameter (power-of-two exponent or decimal power),
       stored little-endian in byte-planed order (all byte 0s, all byte 1s, ...).
    3. value_anchor (8 * num_blocks bytes, byte-planed like grid_params):
       Per-block reconstruction base: the bits of the float64 block minimum on the
       power-of-two grid, or the int64 decimal grid index of the minimum.
    4. time_start (8 * num_blocks bytes, byte-planed; time only):
       The first block's start time as int64, then each block's start minus the
       previous block's start as uint64. Starts are non-decreasing by construction.
    5. time_step (8 * num_blocks bytes, byte-planed; time only):
       Per-block int64 step: the GCD of the block's time deltas (0 if all are 0).
    6. time_ref (8 * num_blocks bytes, byte-planed; time only):
       Per-block uint64 reference quotient: sample i > 0's quotient
       (time[i] - time[i-1]) / time_step is time_ref plus its residual (0 in a regular
       block).
    7. residual_planes (16 * num_blocks * num_groups bytes):
       Zigzag-encoded int16 differences mod 2^16, as 16 bit planes across all
       blocks, or (flags bit 0) as 2 byte planes: every low byte, then every high byte.
    8. nonfinite_code_planes (2 * num_nonfinite_blocks * num_groups bytes):
       2 code bit planes for the flagged blocks, encoding sample categories
       (00: finite, 01: NaN, 10: +inf, 11: -inf).
    9. time_residual_planes (32 * (num_irregular_blocks + num_long_blocks) * num_groups
       bytes; time only): per sample of an irregular block, its quotient minus time_ref,
       mod 2^64 and zigzagged (0 for sample 0), as bit planes like the residual planes:
       planes 0-31 of every irregular block, then planes 32-63 of the long blocks.
"""

import enum
import struct
from typing import NamedTuple

import numpy as np
from numba import njit

HEAD_ORDER: int = 0x03
"""Bitmask for the difference predictor order (bits 0-1 of the header byte)."""

HEAD_DECIMAL: int = 0x04
"""Bit flag indicating decimal quantization mode (bit 2 of the header byte)."""

HEAD_NONFINITE: int = 0x08
"""Bit flag indicating the presence of non-finite code planes (bit 3)."""

HEAD_IRREGULAR_TIME: int = 0x10
"""Bit flag indicating irregular times, with time residual planes (bit 4); only valid in
units with a time axis."""

HEAD_LONG_TIME: int = 0x20
"""Bit flag indicating long time residuals, stored in 64 bit planes instead of 32 (bit 5);
only valid with HEAD_IRREGULAR_TIME."""

HEAD_RESERVED: int = 0xC0
"""Reserved bits in the header byte (bits 6-7); decoders must reject these."""

FLAG_BYTE_PLANES: int = 0x01
"""Unit header flag: the residual field holds 2 byte planes instead of 16 bit planes."""

FLAG_TIME_UNIT_SHIFT: int = 1
"""Position of the 3-bit time unit in the unit header flags (bits 1-3)."""

FLAG_TIME_UNIT_MASK: int = 0x0E
"""Unit header flag bits holding the time unit."""

FLAG_RESERVED: int = 0xF0
"""Reserved unit header flag bits (4-7); decoders must reject these."""

MAX_BLOCK_LEN: int = 65536
"""Maximum plausible block length used as a sanity bound for unit validation."""

MIN_BLOCK_LEN: int = 4
"""Minimum block length: order-3 prediction needs a fourth sample to score its differences."""


class NonFiniteCode(enum.IntEnum):
    """Two-bit sample classification codes for non-finite values. Kernels use the
    plain-int CODE_* constants below: numba can't lower IntEnum members in all expressions."""

    FINITE = 0
    NAN = 1
    POS_INF = 2
    NEG_INF = 3


CODE_FINITE: int = int(NonFiniteCode.FINITE)
"""Two-bit code representing a finite sample (00b)."""

CODE_NAN: int = int(NonFiniteCode.NAN)
"""Two-bit code representing a NaN sample (01b)."""

CODE_POS_INF: int = int(NonFiniteCode.POS_INF)
"""Two-bit code representing positive infinity (+inf, 10b)."""

CODE_NEG_INF: int = int(NonFiniteCode.NEG_INF)
"""Two-bit code representing negative infinity (-inf, 11b)."""


class TimeUnit(enum.IntEnum):
    """Time unit codes stored in unit header flags bits 1-3 (0: no time axis)."""

    NONE = 0
    SECONDS = 1
    MILLISECONDS = 2
    MICROSECONDS = 3
    NANOSECONDS = 4


TIME_UNIT_NAMES: dict[int, str] = {
    TimeUnit.SECONDS: "s",
    TimeUnit.MILLISECONDS: "ms",
    TimeUnit.MICROSECONDS: "us",
    TimeUnit.NANOSECONDS: "ns",
}
"""numpy datetime64 unit name of each time unit code."""

TIME_UNIT_CODES: dict[str, int] = {name: code for code, name in TIME_UNIT_NAMES.items()}
"""Time unit code of each numpy datetime64 unit name."""


E_MIN: int = -1074
"""Minimum valid power-of-two exponent (step of smallest subnormal double)."""

E_MAX: int = 1023
"""Maximum valid power-of-two exponent."""

P_MIN: int = -22
"""Minimum valid decimal exponent (10^-22 is an exact double)."""

P_MAX: int = 22
"""Maximum valid decimal exponent (10^22 is an exact double)."""

BYTES_PER_HEADER: int = 1
"""Number of header bytes per block."""

BYTES_PER_PARAM: int = 8
"""Number of parameter bytes per block (int64 little-endian)."""

BYTES_PER_ANCHOR: int = 8
"""Number of anchor bytes per block (float64 bits or int64 grid index)."""

METADATA_BYTES_PER_BLOCK: int = BYTES_PER_HEADER + BYTES_PER_PARAM + BYTES_PER_ANCHOR
"""Total metadata bytes per block (head byte + int64 parameter + anchor = 17 bytes)."""

FORMAT_VERSION: int = 1
"""Unit format version recorded in the first header byte."""

UNIT_HEADER = struct.Struct("<BBHIQ")
"""Unit header: version, flags, reserved, block length, and a uint64 holding the sample count
in its low 40 bits (its top 3 bytes are reserved)."""

SAMPLE_COUNT_BITS: int = 40
"""Width of the sample count in the header: the top 3 bytes of its uint64 are reserved."""

HEADER_BYTES: int = UNIT_HEADER.size
"""Size of the unit header in bytes (16)."""

MAX_UNIT_SAMPLES: int = 1 << 26
"""Largest sample count a unit may hold: a sanity bound (decoders reject larger headers before
decompressing), not a format limit."""

BYTES_PER_RESIDUAL_SAMPLE: int = 2
"""Number of residual difference bytes per sample (int16 mod 2^16)."""

NONFINITE_BITS_PER_SAMPLE: int = 2
"""Number of code bits per sample for flagged blocks (2-bit plane layout)."""

TIME_BYTES_PER_BLOCK: int = 24
"""Bytes of time_start, time_step and time_ref per block in units with a time axis."""

TIME_SHORT_PLANES: int = 32
"""Time residual bit planes stored for every irregular block; a long block stores as many again."""


@njit(inline="always")
def plane_groups(block_len: int) -> int:
    """Bytes per bit plane per block: ceil(block_len / 8), the last group padded with zero bits."""
    return (block_len + 7) // 8


def unit_size(
    num_blocks: int,
    block_len: int,
    num_flagged_blocks: int = 0,
    has_time: bool = False,
    num_irregular_blocks: int = 0,
    num_long_blocks: int = 0,
) -> int:
    """Calculates the size in bytes of a unit's uncompressed body.

    Args:
        num_blocks: Number of blocks in the unit.
        block_len: Number of samples per block.
        num_flagged_blocks: Number of blocks containing non-finite samples.
        has_time: Whether the unit has a time axis (time_start, time_step and time_ref fields).
        num_irregular_blocks: Number of blocks with time residual planes.
        num_long_blocks: Number of those blocks with long time residuals.

    Returns:
        Size of the uncompressed body in bytes.
    """
    metadata_bytes = num_blocks * METADATA_BYTES_PER_BLOCK
    num_groups = plane_groups(block_len)
    residual_bytes = num_blocks * num_groups * 8 * BYTES_PER_RESIDUAL_SAMPLE
    nonfinite_bytes = num_flagged_blocks * num_groups * NONFINITE_BITS_PER_SAMPLE
    time_bytes = 0
    if has_time:
        time_bytes = num_blocks * TIME_BYTES_PER_BLOCK + (num_irregular_blocks + num_long_blocks) * (
            num_groups * TIME_SHORT_PLANES
        )
    return metadata_bytes + residual_bytes + nonfinite_bytes + time_bytes


@njit(nogil=True, cache=True)
def count_flagged(raw_unit: np.ndarray, num_blocks: int) -> int:
    """Counts blocks in an uncompressed unit that contain non-finite samples.

    Args:
        raw_unit: 1D uint8 array of uncompressed unit bytes.
        num_blocks: Number of blocks whose headers are inspected.

    Returns:
        Count of blocks with the non-finite header flag set.
    """
    flagged_count = 0
    for block_idx in range(num_blocks):
        # Test bit 3 (HEAD_NONFINITE) in block's header byte
        flagged_count += (raw_unit[block_idx] & HEAD_NONFINITE) != 0
    return flagged_count


@njit(nogil=True, cache=True)
def count_irregular(raw_unit: np.ndarray, num_blocks: int) -> int:
    """Counts blocks in an uncompressed unit whose time axis is irregular.

    Args:
        raw_unit: 1D uint8 array of uncompressed unit bytes.
        num_blocks: Number of blocks whose headers are inspected.

    Returns:
        Count of blocks with the irregular time flag set.
    """
    irregular_count = 0
    for block_idx in range(num_blocks):
        irregular_count += (raw_unit[block_idx] & HEAD_IRREGULAR_TIME) != 0
    return irregular_count


@njit(nogil=True, cache=True)
def count_long(raw_unit: np.ndarray, num_blocks: int) -> int:
    """Counts blocks in an uncompressed unit flagged with long time residuals.

    Args:
        raw_unit: 1D uint8 array of uncompressed unit bytes.
        num_blocks: Number of blocks whose headers are inspected.

    Returns:
        Count of blocks with the long time flag set.
    """
    long_count = 0
    for block_idx in range(num_blocks):
        long_count += (raw_unit[block_idx] & HEAD_LONG_TIME) != 0
    return long_count


class UnitHeader(NamedTuple):
    """The fields of a parsed unit header.

    Attributes:
        block_len: Samples per block.
        num_samples: Real sample count of the unit.
        num_blocks: ceil(num_samples / block_len).
        byte_planes: Whether the residual planes are byte planes (flags bit 0).
        time_unit: Time unit code (TimeUnit; 0 for a unit without a time axis).
    """

    block_len: int
    num_samples: int
    num_blocks: int
    byte_planes: bool
    time_unit: int


def pack_header(block_len: int, num_samples: int, byte_planes: bool = False, time_unit: int = 0) -> bytes:
    """Builds the 16-byte unit header.

    Args:
        block_len: Samples per block.
        num_samples: Real sample count of the unit.
        byte_planes: Whether the residual planes are byte planes (flags bit 0).
        time_unit: Time unit code (TimeUnit; 0 for no time axis).

    Returns:
        The header bytes.
    """
    flags = (FLAG_BYTE_PLANES if byte_planes else 0) | (time_unit << FLAG_TIME_UNIT_SHIFT)
    return UNIT_HEADER.pack(FORMAT_VERSION, flags, 0, block_len, num_samples)


def unpack_header(unit: bytes) -> UnitHeader:
    """Parses and validates a unit header.

    Args:
        unit: Unit bytes (header followed by the zstd frame).

    Returns:
        The parsed header.

    Raises:
        ValueError: If the header is truncated, has an unsupported version or
            reserved bits or bytes set, an unknown time unit, or describes an
            implausible block length or sample count.
    """
    if len(unit) < HEADER_BYTES:
        raise ValueError(f"unit of {len(unit)} bytes is shorter than its {HEADER_BYTES}-byte header")
    version, flags, reserved, block_len, sample_count_field = UNIT_HEADER.unpack_from(unit)
    if version != FORMAT_VERSION:
        raise ValueError(f"unit format version {version} is not supported (expected {FORMAT_VERSION})")
    if flags & FLAG_RESERVED or reserved or sample_count_field >> SAMPLE_COUNT_BITS:
        raise ValueError("unit header sets reserved bits (not supported by this version)")
    time_unit = (flags & FLAG_TIME_UNIT_MASK) >> FLAG_TIME_UNIT_SHIFT
    if time_unit > TimeUnit.NANOSECONDS:
        raise ValueError(f"unit header time unit {time_unit} is not supported (expected 0 to {int(TimeUnit.NANOSECONDS)})")
    if not MIN_BLOCK_LEN <= block_len <= MAX_BLOCK_LEN:
        raise ValueError(f"unit block length {block_len} is not from {MIN_BLOCK_LEN} to {MAX_BLOCK_LEN}")
    num_samples = sample_count_field
    if not 0 < num_samples <= MAX_UNIT_SAMPLES:
        raise ValueError(f"unit sample count {num_samples} is outside 1..{MAX_UNIT_SAMPLES}")
    return UnitHeader(block_len, num_samples, -(-num_samples // block_len), bool(flags & FLAG_BYTE_PLANES), time_unit)


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
def residual_start(num_blocks: int, has_time: bool) -> int:
    """Offset of the residual planes: after the per-block columns (and the time columns)."""
    return (METADATA_BYTES_PER_BLOCK + (TIME_BYTES_PER_BLOCK if has_time else 0)) * num_blocks


@njit(inline="always")
def planes_view(raw_unit: np.ndarray, num_blocks: int, block_len: int, has_time: bool) -> np.ndarray:
    """Provides a 2D view into the 16 residual bit planes of an uncompressed unit.

    Args:
        raw_unit: 1D uint8 array containing uncompressed unit data.
        num_blocks: Total number of blocks.
        block_len: Block length.
        has_time: Whether the unit has a time axis (time columns before the residuals).

    Returns:
        2D uint8 array of shape (16, num_blocks * plane_groups(block_len)) representing bit planes.
    """
    num_groups = plane_groups(block_len)
    start_offset = residual_start(num_blocks, has_time)
    end_offset = start_offset + num_blocks * BYTES_PER_RESIDUAL_SAMPLE * 8 * num_groups
    return raw_unit[start_offset:end_offset].reshape(16, num_blocks * num_groups)


@njit(inline="always")
def byte_planes_view(raw_unit: np.ndarray, num_blocks: int, block_len: int, has_time: bool) -> np.ndarray:
    """Provides a 2D view into the residual field as 2 byte planes (flags bit 0).

    Args:
        raw_unit: 1D uint8 array containing uncompressed unit data.
        num_blocks: Total number of blocks.
        block_len: Block length.
        has_time: Whether the unit has a time axis (time columns before the residuals).

    Returns:
        2D uint8 array of shape (2, num_blocks * 8 * plane_groups(block_len)): the low bytes, then
        the high bytes, each block padded with zeros to a multiple of 8 like the bit planes.
    """
    padded_len = 8 * plane_groups(block_len)
    start_offset = residual_start(num_blocks, has_time)
    end_offset = start_offset + num_blocks * BYTES_PER_RESIDUAL_SAMPLE * padded_len
    return raw_unit[start_offset:end_offset].reshape(2, num_blocks * padded_len)


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
def shuffle_block(
    residuals: np.ndarray,
    bit_planes: np.ndarray,
    block_idx: int,
    scratch_low_bytes: np.ndarray,
    scratch_high_bytes: np.ndarray,
) -> None:
    """Zigzag-encodes int16 residuals and packs them into unit bit planes.

    Converts signed differences to unsigned integers via zigzag mapping, splits
    them into low and high byte arrays, transposes 8x8 bit groups, and scatters
    bits across the 16 bit planes.

    Args:
        residuals: 1D int16 array of n residual differences for the block.
        bit_planes: 2D uint8 array of shape (16, N * plane_groups(n)) holding bit planes.
        block_idx: Zero-based block index within the unit.
        scratch_low_bytes: uint8 scratch array of length 8 * plane_groups(n), zero past n.
        scratch_high_bytes: uint8 scratch array of length 8 * plane_groups(n), zero past n.
    """
    zigzag_block(residuals, scratch_low_bytes, scratch_high_bytes)
    # Reinterpret byte scratch buffers as 64-bit words for 8-byte group transposition
    low_words64 = scratch_low_bytes.view(np.uint64)
    high_words64 = scratch_high_bytes.view(np.uint64)
    num_8byte_groups = plane_groups(residuals.shape[0])
    plane_byte_offset = block_idx * num_8byte_groups
    for group_idx in range(num_8byte_groups):
        # Transpose each 8x8 bit matrix
        transposed_low = _transpose8(low_words64[group_idx])
        transposed_high = _transpose8(high_words64[group_idx])
        for bit_idx in range(8):
            # Extract transposed bytes into corresponding bit planes (0-7 low, 8-15 high)
            bit_planes[bit_idx, plane_byte_offset + group_idx] = np.uint8(
                (transposed_low >> np.uint64(8 * bit_idx)) & np.uint64(0xFF)
            )
            bit_planes[8 + bit_idx, plane_byte_offset + group_idx] = np.uint8(
                (transposed_high >> np.uint64(8 * bit_idx)) & np.uint64(0xFF)
            )


@njit(nogil=True, cache=True)
def unshuffle_block(
    bit_planes: np.ndarray,
    block_idx: int,
    out_low_bytes: np.ndarray,
    out_high_bytes: np.ndarray,
) -> None:
    """Reconstructs zigzag-encoded bytes for a block from the 16 bit planes.

    Gathers bits from planes 0-7 into low bytes and planes 8-15 into high bytes,
    reversing the 8x8 bit transpose.

    Args:
        bit_planes: 2D uint8 array of shape (16, N * plane_groups(n)) holding bit planes.
        block_idx: Zero-based block index within the unit.
        out_low_bytes: Output uint8 array of length 8 * plane_groups(n) receiving lower zigzag
            bytes (the padding past n included).
        out_high_bytes: Output uint8 array of length 8 * plane_groups(n) receiving upper zigzag
            bytes.
    """
    low_words64 = out_low_bytes.view(np.uint64)
    high_words64 = out_high_bytes.view(np.uint64)
    num_8byte_groups = out_low_bytes.shape[0] // 8
    plane_byte_offset = block_idx * num_8byte_groups
    for group_idx in range(num_8byte_groups):
        gathered_low = np.uint64(0)
        gathered_high = np.uint64(0)
        for bit_idx in range(8):
            # Gather one byte from each bit plane into a 64-bit word
            gathered_low |= np.uint64(bit_planes[bit_idx, plane_byte_offset + group_idx]) << np.uint64(8 * bit_idx)
            gathered_high |= np.uint64(bit_planes[8 + bit_idx, plane_byte_offset + group_idx]) << np.uint64(8 * bit_idx)
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
def grid_params_start(num_blocks: int) -> int:
    """Offset of the param field: after the N head bytes."""
    return num_blocks


@njit(inline="always")
def value_anchor_start(num_blocks: int) -> int:
    """Offset of the value_anchor field: after the block_flags and grid_params fields."""
    return (BYTES_PER_HEADER + BYTES_PER_PARAM) * num_blocks


@njit(inline="always")
def time_start_start(num_blocks: int) -> int:
    """Offset of the time_start field: after the value_anchor field."""
    return METADATA_BYTES_PER_BLOCK * num_blocks


@njit(inline="always")
def time_step_start(num_blocks: int) -> int:
    """Offset of the time_step field: after the time_start field."""
    return (METADATA_BYTES_PER_BLOCK + BYTES_PER_ANCHOR) * num_blocks


@njit(inline="always")
def time_ref_start(num_blocks: int) -> int:
    """Offset of the time_ref field: after the time_step field."""
    return (METADATA_BYTES_PER_BLOCK + 2 * BYTES_PER_ANCHOR) * num_blocks


@njit(inline="always")
def put_int64(raw_unit: np.ndarray, field_start: int, num_blocks: int, block_idx: int, value: int | np.integer) -> None:
    """Writes block b's int64 value into a byte-planed field.

    Distributes the 8 bytes of the value across 8 strides of length num_blocks.

    Args:
        raw_unit: 1D uint8 array of uncompressed body bytes.
        field_start: Offset of the field (grid_params_start or value_anchor_start).
        num_blocks: Total number of blocks N.
        block_idx: Zero-based block index.
        value: Signed 64-bit integer to write.
    """
    value_bits = np.uint64(value)
    for byte_idx in range(8):
        # Place byte k of block b at offset: field_start + k*N + b
        raw_unit[field_start + byte_idx * num_blocks + block_idx] = np.uint8(
            (value_bits >> np.uint64(8 * byte_idx)) & np.uint64(0xFF)
        )


@njit(inline="always")
def get_int64(raw_unit: np.ndarray, field_start: int, num_blocks: int, block_idx: int) -> np.int64:
    """Reads block b's int64 value from a byte-planed field.

    Args:
        raw_unit: 1D uint8 array of uncompressed body bytes.
        field_start: Offset of the field (grid_params_start or value_anchor_start).
        num_blocks: Total number of blocks N.
        block_idx: Zero-based block index.

    Returns:
        Reconstructed signed 64-bit integer.
    """
    value_bits = np.uint64(0)
    for byte_idx in range(8):
        # Reassemble byte k from offset: field_start + k*N + b
        value_bits |= np.uint64(raw_unit[field_start + byte_idx * num_blocks + block_idx]) << np.uint64(8 * byte_idx)
    return np.int64(value_bits)


@njit(inline="always")
def code_planes_start(num_blocks: int, block_len: int, has_time: bool) -> int:
    """Offset of the nonfinite code planes: after the residual planes."""
    return residual_start(num_blocks, has_time) + num_blocks * BYTES_PER_RESIDUAL_SAMPLE * 8 * plane_groups(block_len)


@njit(inline="always")
def code_planes_view(
    raw_unit: np.ndarray, num_blocks: int, block_len: int, num_flagged_blocks: int, has_time: bool
) -> np.ndarray:
    """Provides a 2D view into the non-finite code planes of an uncompressed unit.

    Args:
        raw_unit: 1D uint8 array containing uncompressed unit data.
        num_blocks: Total number of blocks.
        block_len: Block length.
        num_flagged_blocks: Number of blocks carrying non-finite codes.
        has_time: Whether the unit has a time axis (time columns before the residuals).

    Returns:
        2D uint8 array of shape (2, num_flagged_blocks * plane_groups(block_len)).
    """
    num_groups = plane_groups(block_len)
    start_offset = code_planes_start(num_blocks, block_len, has_time)
    total_code_bytes = num_flagged_blocks * num_groups * NONFINITE_BITS_PER_SAMPLE
    return raw_unit[start_offset:start_offset + total_code_bytes].reshape(2, num_flagged_blocks * num_groups)


@njit(inline="always")
def time_planes_views(
    raw_unit: np.ndarray, num_blocks: int, block_len: int, num_flagged_blocks: int
) -> tuple[np.ndarray, np.ndarray]:
    """Provides 2D views into the time residual planes of an uncompressed unit with a time axis.

    Args:
        raw_unit: 1D uint8 array containing uncompressed unit data.
        num_blocks: Total number of blocks.
        block_len: Block length.
        num_flagged_blocks: Number of blocks carrying non-finite codes.

    Returns:
        A tuple of (short_planes, long_planes): 2D uint8 arrays of shape
        (32, num_irregular_blocks * plane_groups(block_len)) holding planes 0-31 of every
        irregular block, and (32, num_long_blocks * plane_groups(block_len)) holding planes 32-63
        of the long ones.
    """
    num_groups = plane_groups(block_len)
    short_start = code_planes_start(num_blocks, block_len, True) + num_flagged_blocks * (
        num_groups * NONFINITE_BITS_PER_SAMPLE
    )
    short_bytes = TIME_SHORT_PLANES * count_irregular(raw_unit, num_blocks) * num_groups
    long_bytes = TIME_SHORT_PLANES * count_long(raw_unit, num_blocks) * num_groups
    short_planes = raw_unit[short_start:short_start + short_bytes].reshape(TIME_SHORT_PLANES, short_bytes // TIME_SHORT_PLANES)
    long_start = short_start + short_bytes
    long_planes = raw_unit[long_start:long_start + long_bytes].reshape(TIME_SHORT_PLANES, long_bytes // TIME_SHORT_PLANES)
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
def put_codes(sample_codes: np.ndarray, code_planes: np.ndarray, flagged_block_idx: int) -> None:
    """Packs 2-bit non-finite sample codes for a block into two code bit planes.

    Args:
        sample_codes: 1D uint8 array of n codes (values in 0..3).
        code_planes: 2D uint8 array of shape (2, F * plane_groups(n)) holding code planes.
        flagged_block_idx: Rank of the flagged block among all flagged blocks (0 <= f < F).
    """
    num_samples = sample_codes.shape[0]
    num_full_groups = num_samples // 8
    plane_byte_offset = flagged_block_idx * plane_groups(num_samples)
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
def get_codes(code_planes: np.ndarray, flagged_block_idx: int, out_codes: np.ndarray) -> None:
    """Unpacks 2-bit non-finite sample codes for a flagged block.

    Args:
        code_planes: 2D uint8 array of shape (2, F * plane_groups(n)) holding code planes.
        flagged_block_idx: Rank of the flagged block among all flagged blocks (0 <= f < F).
        out_codes: Output 1D uint8 array of length n receiving unpacked codes.
    """
    num_samples = out_codes.shape[0]
    num_full_groups = num_samples // 8
    plane_byte_offset = flagged_block_idx * plane_groups(num_samples)
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
    short_rank: int,
    long_rank: int,
    byte_idx: int,
    num_groups: int,
    first_group: int = 0,
    end_group: int = -1,
) -> np.ndarray:
    """The 8 bit planes of a block's byte level byte_idx: levels 0-3 are in the short planes
    (at the block's rank among irregular blocks), 4-7 in the long planes (rank among long blocks).
    num_groups is the block's plane bytes; the view holds groups first_group to end_group
    (all if -1)."""
    end_group = num_groups if end_group < 0 else end_group
    if byte_idx < 4:
        return short_planes[8 * byte_idx:8 * byte_idx + 8, short_rank * num_groups + first_group:short_rank * num_groups + end_group]
    level = byte_idx - 4
    return long_planes[8 * level:8 * level + 8, long_rank * num_groups + first_group:long_rank * num_groups + end_group]


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
    short_rank: int,
    long_rank: int,
    scratch_tail: np.ndarray,
) -> None:
    """Packs a block's zigzagged uint64 time residuals into its time residual bit planes.

    Bit k of byte i of plane j is bit j of time_residuals[8 * i + k], as for the residual
    planes. Only the bytes up to the highest set bit of any residual are written: the planes
    above it must already be zero (write_time_rows zeroes the field first).

    Args:
        time_residuals: 1D uint64 array of block_len zigzagged time residuals, below 2^32
            unless the block is long.
        short_planes: 2D uint8 array of planes 0-31 of the irregular blocks.
        long_planes: 2D uint8 array of planes 32-63 of the long blocks.
        short_rank: Rank of the block among the irregular blocks.
        long_rank: Rank of the block among the long blocks (unused for a short block).
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
        level_planes = _level_planes(short_planes, long_planes, short_rank, long_rank, byte_idx, num_groups)
        shift = np.uint64(8 * byte_idx)
        for group_idx in range(num_full_groups):
            _put_time_group(time_residuals, 8 * group_idx, shift, level_planes, group_idx)
        if num_groups > num_full_groups:
            _put_time_group(scratch_tail, 0, shift, level_planes, num_full_groups)


@njit(nogil=True, cache=True)
def _time_tail_levels(
    short_planes: np.ndarray, long_planes: np.ndarray, short_rank: int, long_rank: int, stride: int, num_valid: int
) -> int:
    """The number of byte levels up to the highest with a nonzero bit among the first num_valid
    samples of a block's partial last group (its padding bits are ignored)."""
    valid_mask = (1 << num_valid) - 1
    num_levels = 8 if long_rank >= 0 else 4
    while num_levels > 0:
        tail_planes = _level_planes(short_planes, long_planes, short_rank, long_rank, num_levels - 1, stride, stride - 1)
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
    short_rank: int,
    long_rank: int,
    stride: int,
    num_active_bytes: int,
    out_time_residuals: np.ndarray,
    scratch_tail: np.ndarray,
) -> None:
    """Unpacks the partial last group of a block's time residuals: its samples past n are padding."""
    scratch_tail[:] = 0
    tail_bytes = scratch_tail.view(np.uint8)
    for byte_idx in range(num_active_bytes):
        level_planes = _level_planes(short_planes, long_planes, short_rank, long_rank, byte_idx, stride)
        _get_time_group(level_planes, stride - 1, byte_idx, tail_bytes)
    num_full_samples = 8 * (stride - 1)
    out_time_residuals[num_full_samples:] = scratch_tail[:out_time_residuals.shape[0] - num_full_samples]


@njit(nogil=True, cache=True)
def unshuffle_time_residuals(
    short_planes: np.ndarray,
    long_planes: np.ndarray,
    short_rank: int,
    long_rank: int,
    out_time_residuals: np.ndarray,
    scratch_tail: np.ndarray,
) -> int:
    """Reconstructs a block's zigzagged uint64 time residuals from its time residual bit planes.

    Args:
        short_planes: 2D uint8 array of planes 0-31 of the irregular blocks.
        long_planes: 2D uint8 array of planes 32-63 of the long blocks.
        short_rank: Rank of the block among the irregular blocks.
        long_rank: Rank of the block among the long blocks, or -1 for a short block.
        out_time_residuals: Output 1D uint64 array of block_len receiving the residuals.
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
        num_tail_bytes = _time_tail_levels(short_planes, long_planes, short_rank, long_rank, stride, num_samples % 8)
    # Byte levels above the highest nonzero plane byte of this block are zero in every
    # sample: find them with a contiguous scan and skip them (residuals are mostly small)
    num_active_bytes = 8 if long_rank >= 0 else 4
    while num_active_bytes > num_tail_bytes:
        level_planes = _level_planes(
            short_planes, long_planes, short_rank, long_rank, num_active_bytes - 1, stride, 0, num_groups
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
        level_planes = _level_planes(short_planes, long_planes, short_rank, long_rank, byte_idx, stride, 0, num_groups)
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
            short_planes, long_planes, short_rank, long_rank, stride, num_active_bytes, out_time_residuals, scratch_tail
        )
    return num_active_bytes


@njit(nogil=True, cache=True)
def write_time_rows(
    block_flags: np.ndarray,
    time_starts: np.ndarray,
    time_steps: np.ndarray,
    time_refs: np.ndarray,
    time_residuals: np.ndarray,
    num_flagged_blocks: int,
    out_raw_unit: np.ndarray,
) -> None:
    """Serializes the time_start, time_step, time_ref and time_residual_planes fields.

    Args:
        block_flags: 1D uint8 array of block flags (HEAD_IRREGULAR_TIME selects the residual planes).
        time_starts: 1D int64 array of block start times.
        time_steps: 1D int64 array of block time steps.
        time_refs: 1D uint64 array of block reference quotients.
        time_residuals: 2D uint64 array of shape (num_blocks, block_len) holding the
            zigzagged time residuals (only rows of irregular blocks are read).
        num_flagged_blocks: Number of blocks carrying non-finite codes.
        out_raw_unit: Uncompressed body whose value fields are written by write_rows.
    """
    num_blocks, block_len = time_residuals.shape
    short_planes, long_planes = time_planes_views(out_raw_unit, num_blocks, block_len, num_flagged_blocks)
    scratch_tail = np.empty(8, np.uint64)
    # One contiguous fill: each block then writes only the planes its residuals reach
    short_planes[:, :] = 0
    long_planes[:, :] = 0
    irregular_counter = 0
    long_counter = 0
    for block_idx in range(num_blocks):
        # The first start is stored as is; later starts as the uint64 increase over the previous start
        start_increase = time_starts[block_idx]
        if block_idx > 0:
            start_increase = np.int64(np.uint64(time_starts[block_idx]) - np.uint64(time_starts[block_idx - 1]))
        put_int64(out_raw_unit, time_start_start(num_blocks), num_blocks, block_idx, start_increase)
        put_int64(out_raw_unit, time_step_start(num_blocks), num_blocks, block_idx, time_steps[block_idx])
        put_int64(out_raw_unit, time_ref_start(num_blocks), num_blocks, block_idx, np.int64(time_refs[block_idx]))
        if block_flags[block_idx] & HEAD_IRREGULAR_TIME:
            shuffle_time_residuals(
                time_residuals[block_idx], short_planes, long_planes, irregular_counter, long_counter, scratch_tail
            )
            irregular_counter += 1
            long_counter += (block_flags[block_idx] & HEAD_LONG_TIME) != 0


TIME_ROWS_OK: int = 0
"""read_time_rows status: the start times are valid."""

TIME_ROWS_BAD_START: int = 1
"""read_time_rows status: a start time is int64 minimum (NaT) or overflows int64."""

TIME_ROWS_BAD_LONG: int = 2
"""read_time_rows status: a long block's planes 32-63 are all zero (it should be short)."""


@njit(nogil=True, cache=True)
def read_time_rows(
    raw_unit: np.ndarray,
    num_flagged_blocks: int,
    out_time_starts: np.ndarray,
    out_time_steps: np.ndarray,
    out_time_refs: np.ndarray,
    out_time_residuals: np.ndarray,
) -> tuple[int, int]:
    """Deserializes the time_start, time_step, time_ref and time_residual_planes fields.

    Args:
        raw_unit: 1D uint8 array of uncompressed body bytes of a unit with a time axis.
        num_flagged_blocks: Number of blocks carrying non-finite codes.
        out_time_starts: Output 1D int64 array receiving the block start times.
        out_time_steps: Output 1D int64 array receiving the block time steps.
        out_time_refs: Output 1D uint64 array receiving the block reference quotients.
        out_time_residuals: Output 2D uint64 array of shape (num_blocks, block_len) receiving
            the zigzagged time residuals of irregular blocks (rows of regular blocks are not
            written).

    Returns:
        A tuple of (status, block_idx): TIME_ROWS_OK, or TIME_ROWS_BAD_START and the
        first block whose start is int64 minimum or overflows int64, or TIME_ROWS_BAD_LONG
        and the first long block whose planes 32-63 are all zero.
    """
    num_blocks, block_len = out_time_residuals.shape
    short_planes, long_planes = time_planes_views(raw_unit, num_blocks, block_len, num_flagged_blocks)
    scratch_tail = np.empty(8, np.uint64)
    int64_max = np.uint64(0x7FFFFFFFFFFFFFFF)
    irregular_counter = 0
    long_counter = 0
    for block_idx in range(num_blocks):
        stored_start = get_int64(raw_unit, time_start_start(num_blocks), num_blocks, block_idx)
        if block_idx == 0:
            if stored_start == np.int64(-0x8000000000000000):
                return TIME_ROWS_BAD_START, 0
            out_time_starts[0] = stored_start
        else:
            # The increase is unsigned: the start must stay at most int64 maximum
            previous_start = out_time_starts[block_idx - 1]
            headroom = int64_max - np.uint64(previous_start)
            if np.uint64(stored_start) > headroom:
                return TIME_ROWS_BAD_START, block_idx
            out_time_starts[block_idx] = np.int64(np.uint64(previous_start) + np.uint64(stored_start))
        out_time_steps[block_idx] = get_int64(raw_unit, time_step_start(num_blocks), num_blocks, block_idx)
        out_time_refs[block_idx] = np.uint64(get_int64(raw_unit, time_ref_start(num_blocks), num_blocks, block_idx))
        if raw_unit[block_idx] & HEAD_IRREGULAR_TIME:
            is_long = raw_unit[block_idx] & HEAD_LONG_TIME
            num_active_bytes = unshuffle_time_residuals(
                short_planes,
                long_planes,
                irregular_counter,
                long_counter if is_long else -1,
                out_time_residuals[block_idx],
                scratch_tail,
            )
            if is_long and num_active_bytes <= 4:
                return TIME_ROWS_BAD_LONG, block_idx
            irregular_counter += 1
            long_counter += is_long != 0
    return TIME_ROWS_OK, 0


@njit(nogil=True, cache=True)
def write_rows(
    block_flags: np.ndarray,
    grid_params: np.ndarray,
    value_anchors: np.ndarray,
    residuals: np.ndarray,
    codes: np.ndarray,
    byte_planes: bool,
    has_time: bool,
    out_raw_unit: np.ndarray,
) -> None:
    """Serializes the value fields of a unit into a preallocated uncompressed body buffer.

    The time fields (if has_time) are written separately by write_time_rows.

    Args:
        block_flags: 1D uint8 array of block flags.
        grid_params: 1D int64 array of block grid parameters.
        value_anchors: 1D int64 array of block value anchors (float64 bits or decimal grid index).
        residuals: 2D int16 array of shape (num_blocks, block_len) holding difference residuals.
        codes: 2D uint8 array of shape (num_blocks, block_len) holding sample codes (only rows
            of flagged blocks are read; may be empty if no block is flagged).
        byte_planes: Store the residuals as 2 byte planes instead of 16 bit planes.
        has_time: Whether the unit has a time axis (time columns before the residuals).
        out_raw_unit: Output 1D uint8 array of length unit_size(...).
    """
    num_blocks, block_len = residuals.shape
    padded_len = 8 * plane_groups(block_len)
    # Zeroed: the padding past block_len stays zero
    scratch_low_bytes = np.zeros(padded_len, np.uint8)
    scratch_high_bytes = np.zeros(padded_len, np.uint8)
    # Obtain views into residual and non-finite plane regions
    planes = planes_view(out_raw_unit, num_blocks, block_len, has_time)
    bplanes = byte_planes_view(out_raw_unit, num_blocks, block_len, has_time)
    cplanes = code_planes_view(out_raw_unit, num_blocks, block_len, count_flagged(block_flags, num_blocks), has_time)
    flagged_counter = 0
    for block_idx in range(num_blocks):
        out_raw_unit[block_idx] = block_flags[block_idx]
        # Store the 8-byte grid parameter and value anchor in byte-planed layout
        put_int64(out_raw_unit, grid_params_start(num_blocks), num_blocks, block_idx, grid_params[block_idx])
        put_int64(out_raw_unit, value_anchor_start(num_blocks), num_blocks, block_idx, value_anchors[block_idx])
        if byte_planes:
            # Zigzag and store the low and high bytes in place, then zero the padding
            sample_start = block_idx * padded_len
            zigzag_block(
                residuals[block_idx],
                bplanes[0, sample_start:sample_start + block_len],
                bplanes[1, sample_start:sample_start + block_len],
            )
            bplanes[:, sample_start + block_len:sample_start + padded_len] = 0
        else:
            # Zigzag, transpose, and store residual bit planes
            shuffle_block(residuals[block_idx], planes, block_idx, scratch_low_bytes, scratch_high_bytes)
        if block_flags[block_idx] & HEAD_NONFINITE:
            # Store 2-bit code planes for flagged blocks
            put_codes(codes[block_idx], cplanes, flagged_counter)
            flagged_counter += 1


@njit(nogil=True, cache=True)
def read_rows(
    raw_unit: np.ndarray,
    byte_planes: bool,
    has_time: bool,
    out_block_flags: np.ndarray,
    out_grid_params: np.ndarray,
    out_value_anchors: np.ndarray,
    out_residuals: np.ndarray,
    out_codes: np.ndarray,
) -> None:
    """Deserializes the value fields of a unit from an uncompressed body buffer.

    Args:
        raw_unit: 1D uint8 array containing uncompressed body bytes.
        byte_planes: Whether the residuals are stored as byte planes (flags bit 0).
        has_time: Whether the unit has a time axis (time columns before the residuals).
        out_block_flags: Output 1D uint8 array of length num_blocks receiving block flags.
        out_grid_params: Output 1D int64 array of length num_blocks receiving grid parameters.
        out_value_anchors: Output 1D int64 array of length num_blocks receiving value anchors.
        out_residuals: Output 2D int16 array of shape (num_blocks, block_len) receiving residuals.
        out_codes: Output 2D uint8 array of shape (num_blocks, block_len) receiving codes for
            flagged blocks (unflagged block rows are left unmodified).
    """
    num_blocks, block_len = out_residuals.shape
    padded_len = 8 * plane_groups(block_len)
    scratch_low_bytes = np.empty(padded_len, np.uint8)
    scratch_high_bytes = np.empty(padded_len, np.uint8)
    # Access views into bit plane sections
    planes = planes_view(raw_unit, num_blocks, block_len, has_time)
    bplanes = byte_planes_view(raw_unit, num_blocks, block_len, has_time)
    cplanes = code_planes_view(raw_unit, num_blocks, block_len, count_flagged(raw_unit, num_blocks), has_time)
    flagged_counter = 0
    for block_idx in range(num_blocks):
        out_block_flags[block_idx] = raw_unit[block_idx]
        # Read the byte-planed grid parameter and value anchor
        out_grid_params[block_idx] = get_int64(raw_unit, grid_params_start(num_blocks), num_blocks, block_idx)
        out_value_anchors[block_idx] = get_int64(raw_unit, value_anchor_start(num_blocks), num_blocks, block_idx)
        if byte_planes:
            sample_start = block_idx * padded_len
            scratch_low_bytes[:block_len] = bplanes[0, sample_start:sample_start + block_len]
            scratch_high_bytes[:block_len] = bplanes[1, sample_start:sample_start + block_len]
        else:
            # Unshuffle bit planes into zigzag low/high bytes
            unshuffle_block(planes, block_idx, scratch_low_bytes, scratch_high_bytes)
        for sample_idx in range(block_len):
            # Reverse zigzag mapping to recover signed residual differences
            out_residuals[block_idx, sample_idx] = unzigzag16(scratch_low_bytes, scratch_high_bytes, sample_idx)
        if out_block_flags[block_idx] & HEAD_NONFINITE:
            # Unpack non-finite codes for flagged blocks
            get_codes(cplanes, flagged_counter, out_codes[block_idx])
            flagged_counter += 1


class TimeRows(NamedTuple):
    """The time axis of a unit as per-block rows.

    Attributes:
        starts: 1D int64 array of block start times (absolute, in the unit's time unit).
        steps: 1D int64 array of block time steps: the GCD of each block's time deltas.
        refs: 1D uint64 array of block reference quotients: sample i > 0's quotient
            (time[i] - time[i-1]) / step is the reference plus its residual.
        residuals: 2D uint64 array of shape (num_blocks, block_len): per sample of an
            irregular block, its quotient minus the reference, mod 2^64 and zigzagged (0 for
            sample 0). Rows of regular blocks (all residuals 0) aren't read or written.
    """

    starts: np.ndarray
    steps: np.ndarray
    refs: np.ndarray
    residuals: np.ndarray


def allocate_time_rows(num_blocks: int, block_len: int, zero_residuals: bool = False) -> TimeRows:
    """Uninitialized time rows; zero_residuals zeroes the residual rows (regular blocks' rows
    are otherwise left as allocated)."""
    residuals = (np.zeros if zero_residuals else np.empty)((num_blocks, block_len), np.uint64)
    return TimeRows(np.empty(num_blocks, np.int64), np.empty(num_blocks, np.int64), np.empty(num_blocks, np.uint64), residuals)


class UnitRows(NamedTuple):
    """The deserialized per-block rows of a unit.

    Attributes:
        block_flags: 1D uint8 array of block flags.
        grid_params: 1D int64 array of block grid parameters.
        value_anchors: 1D int64 array of block value anchors.
        residuals: 2D int16 array of shape (num_blocks, block_len) holding residuals.
        codes: 2D uint8 array of shape (num_blocks, block_len) holding sample codes (all
            zeros for unflagged blocks).
        time_rows: The time axis rows, or None for a unit without a time axis.
    """

    block_flags: np.ndarray
    grid_params: np.ndarray
    value_anchors: np.ndarray
    residuals: np.ndarray
    codes: np.ndarray
    time_rows: TimeRows | None


def write_unit(
    block_flags: np.ndarray,
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
        grid_params: 1D int64 array of block grid parameters.
        value_anchors: 1D int64 array of block value anchors.
        residuals: 2D int16 array of shape (num_blocks, block_len) holding difference residuals.
        codes: Optional 2D uint8 array of shape (num_blocks, block_len) holding sample codes.
            Required if any block has the HEAD_NONFINITE flag set.
        byte_planes: Store the residuals as 2 byte planes instead of 16 bit planes.
        time_rows: The time axis rows, or None for a unit without a time axis.

    Returns:
        1D uint8 array containing the serialized uncompressed body.

    Raises:
        ValueError: If any block is flagged non-finite but codes are omitted, or flagged
            irregular in a unit without a time axis.
    """
    num_blocks, block_len = residuals.shape
    # Count flagged blocks to size the non-finite code planes and the time residual planes
    num_flagged_blocks = int(np.count_nonzero(block_flags & HEAD_NONFINITE))
    num_irregular_blocks = int(np.count_nonzero(block_flags & HEAD_IRREGULAR_TIME))
    num_long_blocks = int(np.count_nonzero(block_flags & HEAD_LONG_TIME))
    if codes is None:
        if num_flagged_blocks:
            raise ValueError("flagged blocks need codes")
        codes = np.zeros((0, block_len), np.uint8)
    has_time = time_rows is not None
    if num_irregular_blocks and not has_time:
        raise ValueError("irregular time blocks need time rows")
    raw_unit = np.empty(
        unit_size(num_blocks, block_len, num_flagged_blocks, has_time, num_irregular_blocks, num_long_blocks), np.uint8
    )
    write_rows(block_flags, grid_params, value_anchors, residuals, codes, byte_planes, has_time, raw_unit)
    if time_rows is not None:
        write_time_rows(block_flags, *time_rows, num_flagged_blocks, raw_unit)
    return raw_unit


def read_unit(
    raw_unit: bytes | np.ndarray,
    num_blocks: int,
    block_len: int,
    byte_planes: bool = False,
    has_time: bool = False,
) -> UnitRows:
    """Deserializes an uncompressed body into constituent arrays.

    Args:
        raw_unit: Byte buffer or uint8 array containing the uncompressed body.
        num_blocks: Number of blocks in the unit.
        block_len: Block length.
        byte_planes: Whether the residuals are stored as byte planes (flags bit 0).
        has_time: Whether the unit has a time axis.

    Returns:
        The unit's rows.

    Raises:
        ValueError: If the buffer size does not match num_blocks blocks of block_len, or a
            block start time is int64 minimum or overflows int64.
    """
    raw_arr = np.frombuffer(raw_unit, np.uint8) if not isinstance(raw_unit, np.ndarray) else raw_unit
    # count_flagged reads the block flags, so check they exist first
    if raw_arr.shape[0] < num_blocks or raw_arr.shape[0] != unit_size(
        num_blocks,
        block_len,
        count_flagged(raw_arr, num_blocks),
        has_time,
        count_irregular(raw_arr, num_blocks),
        count_long(raw_arr, num_blocks),
    ):
        raise ValueError(f"a body of {raw_arr.shape[0]} bytes doesn't hold {num_blocks} blocks of {block_len}")
    block_flags = np.empty(num_blocks, np.uint8)
    grid_params = np.empty(num_blocks, np.int64)
    value_anchors = np.empty(num_blocks, np.int64)
    residuals = np.empty((num_blocks, block_len), np.int16)
    codes = np.zeros((num_blocks, block_len), np.uint8)
    read_rows(raw_arr, byte_planes, has_time, block_flags, grid_params, value_anchors, residuals, codes)
    time_rows = None
    if has_time:
        # Zeroed: read_time_rows leaves regular blocks' residual rows untouched
        time_rows = allocate_time_rows(num_blocks, block_len, zero_residuals=True)
        status, block_idx = read_time_rows(raw_arr, count_flagged(raw_arr, num_blocks), *time_rows)
        if status == TIME_ROWS_BAD_START:
            raise ValueError(f"block {block_idx}: start time out of range (corrupt unit)")
        if status == TIME_ROWS_BAD_LONG:
            raise ValueError(f"block {block_idx}: long time residuals fit in 32 bits (corrupt unit)")
    return UnitRows(block_flags, grid_params, value_anchors, residuals, codes, time_rows)
