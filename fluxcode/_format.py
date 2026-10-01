# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Binary layout and bit-shuffling routines for units.

A unit is an 8-byte header followed by one zstd frame holding the body. The header
(little-endian) is: format version (uint8, 1), flags (uint8; bit 0 selects byte planes for
the residual planes, bits 1-3 hold the time unit, bits 4-7 are 0), the block count (uint16)
and the sample count (uint32). Every block records its own size, 0 to 65,535 samples, and
the block sizes add up to the sample count.

Each block of n samples takes num_groups = ceil(n / 8) bytes of every bit plane (none for an
empty block): when n isn't a multiple of 8, the high bits of its last byte are padding, and
byte planes pad each block to 8 * num_groups bytes. Writers zero the padding; decoders
ignore it. A field "per group" below holds that many bytes per block, the blocks' bytes
consecutive in block order.

The body, for num_blocks blocks, consists of these consecutive fields (the time fields
only when the header's time unit is not 0):

    1. block_flags (num_blocks bytes):
       Bits 0-1 encode the predictor difference order (0-3); bit 2 indicates
       decimal quantization; bit 3 indicates non-finite code planes; bit 4
       indicates irregular times (time residual planes); bit 5 (with bit 4) indicates
       long time residuals (64 planes instead of 32); bits 6-7 are reserved and must be zero.
       An empty block's flags are 0.
    2. block_sizes (2 * num_blocks bytes):
       Per-block uint16 sample count, byte-planed (all low bytes, then all high bytes).
    3. grid_params (8 * num_blocks bytes):
       Per-block int64 grid parameter (power-of-two exponent or decimal power),
       stored little-endian in byte-planed order (all byte 0s, all byte 1s, ...).
    4. value_anchor (8 * num_blocks bytes, byte-planed like grid_params):
       Per-block reconstruction base: the bits of the float64 block minimum on the
       power-of-two grid, or the int64 decimal grid index of the minimum.
    5. time_start (8 * num_blocks bytes, byte-planed; time only):
       The first non-empty block's start time as int64, then each non-empty block's start
       minus the previous non-empty block's start as uint64 (0 for an empty block). Starts
       are non-decreasing by construction.
    6. time_step (8 * num_blocks bytes, byte-planed; time only):
       Per-block int64 step: the GCD of the block's time deltas (0 if all are 0).
    7. time_ref (8 * num_blocks bytes, byte-planed; time only):
       Per-block uint64 reference quotient: sample i > 0's quotient
       (time[i] - time[i-1]) / time_step is time_ref plus its residual (0 in a regular
       block).
    8. residual_planes (16 bytes per group of every block):
       Zigzag-encoded int16 differences mod 2^16, as 16 bit planes across all
       blocks, or (flags bit 0) as 2 byte planes: every low byte, then every high byte.
    9. nonfinite_code_planes (2 bytes per group of the blocks with non-finite values):
       2 code bit planes for the flagged blocks, encoding sample categories
       (00: finite, 01: NaN, 10: +inf, 11: -inf).
    10. time_residual_planes (32 bytes per group of the irregular blocks, then 32 per group
       of the long blocks; time only): per sample of an irregular block, its quotient minus
       time_ref, mod 2^64 and zigzagged (0 for sample 0), as bit planes like the residual
       planes: planes 0-31 of every irregular block, then planes 32-63 of the long blocks.
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

MAX_BLOCK_LEN: int = 0xFFFF
"""Largest block size: the block_sizes field is uint16."""

MAX_BLOCKS: int = 0xFFFF
"""Largest block count: the header field is uint16."""

SHORT_BLOCK_LEN: int = 8
"""Blocks of at most this many samples skip the analysis: they take the finest step
(max_quantize_bits) on the power-of-two grid and order 0."""


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

BYTES_PER_SIZE: int = 2
"""Number of block size bytes per block (uint16)."""

BYTES_PER_PARAM: int = 8
"""Number of parameter bytes per block (int64 little-endian)."""

BYTES_PER_ANCHOR: int = 8
"""Number of anchor bytes per block (float64 bits or int64 grid index)."""

METADATA_BYTES_PER_BLOCK: int = BYTES_PER_HEADER + BYTES_PER_SIZE + BYTES_PER_PARAM + BYTES_PER_ANCHOR
"""Total metadata bytes per block (head byte + uint16 size + int64 parameter + anchor = 19 bytes)."""

FORMAT_VERSION: int = 1
"""Unit format version recorded in the first header byte."""

UNIT_HEADER = struct.Struct("<BBHI")
"""Unit header: version, flags, block count and sample count."""

HEADER_BYTES: int = UNIT_HEADER.size
"""Size of the unit header in bytes (8)."""

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


class Layout(NamedTuple):
    """Where each block's data sits in a unit's flat arrays and plane fields.

    Every array has num_blocks + 1 entries: entry b is block b's offset, and the last entry
    the total.

    Attributes:
        sample_offsets: Offsets of the blocks' samples (cumulative block sizes).
        group_offsets: Offsets of the blocks' bytes in each residual bit plane.
        code_offsets: Offsets in each non-finite code plane (only flagged blocks take bytes).
        short_offsets: Offsets in time residual planes 0-31 (only irregular blocks take bytes).
        long_offsets: Offsets in time residual planes 32-63 (only long blocks take bytes).
    """

    sample_offsets: np.ndarray
    group_offsets: np.ndarray
    code_offsets: np.ndarray
    short_offsets: np.ndarray
    long_offsets: np.ndarray


@njit(nogil=True, cache=True)
def _fill_layout(
    block_flags: np.ndarray,
    block_sizes: np.ndarray,
    out_sample_offsets: np.ndarray,
    out_group_offsets: np.ndarray,
    out_code_offsets: np.ndarray,
    out_short_offsets: np.ndarray,
    out_long_offsets: np.ndarray,
) -> None:
    """Accumulates the offsets of a Layout from the block flags and sizes."""
    out_sample_offsets[0] = out_group_offsets[0] = out_code_offsets[0] = 0
    out_short_offsets[0] = out_long_offsets[0] = 0
    for block_idx in range(block_sizes.shape[0]):
        num_samples = block_sizes[block_idx]
        num_groups = plane_groups(num_samples)
        flags = block_flags[block_idx]
        out_sample_offsets[block_idx + 1] = out_sample_offsets[block_idx] + num_samples
        out_group_offsets[block_idx + 1] = out_group_offsets[block_idx] + num_groups
        out_code_offsets[block_idx + 1] = out_code_offsets[block_idx] + (num_groups if flags & HEAD_NONFINITE else 0)
        out_short_offsets[block_idx + 1] = out_short_offsets[block_idx] + (
            num_groups if flags & HEAD_IRREGULAR_TIME else 0
        )
        out_long_offsets[block_idx + 1] = out_long_offsets[block_idx] + (num_groups if flags & HEAD_LONG_TIME else 0)


def layout(block_flags: np.ndarray, block_sizes: np.ndarray) -> Layout:
    """Computes the Layout of blocks with the given flags and sizes.

    Args:
        block_flags: 1D uint8 array of block flags.
        block_sizes: 1D integer array of block sizes.

    Returns:
        The blocks' Layout.
    """
    num_blocks = block_sizes.shape[0]
    offsets = Layout(*(np.empty(num_blocks + 1, np.int64) for _ in Layout._fields))
    _fill_layout(block_flags, block_sizes.astype(np.int64, copy=False), *offsets)
    return offsets


@njit(nogil=True, cache=True)
def _read_sizes(raw_unit: np.ndarray, num_blocks: int, out_block_sizes: np.ndarray) -> None:
    """Reads the byte-planed uint16 block_sizes field of an uncompressed body."""
    sizes_start = block_sizes_start(num_blocks)
    for block_idx in range(num_blocks):
        out_block_sizes[block_idx] = np.int64(raw_unit[sizes_start + block_idx]) | (
            np.int64(raw_unit[sizes_start + num_blocks + block_idx]) << 8
        )


def read_layout(raw_unit: np.ndarray, num_blocks: int) -> tuple[np.ndarray, Layout]:
    """Reads the block sizes of an uncompressed body (at least 3 * num_blocks bytes) and
    computes their Layout from them and the block flags.

    Returns:
        A tuple of (block_sizes, layout): a 1D int64 array and the Layout.
    """
    block_sizes = np.empty(num_blocks, np.int64)
    _read_sizes(raw_unit, num_blocks, block_sizes)
    offsets = Layout(*(np.empty(num_blocks + 1, np.int64) for _ in Layout._fields))
    _fill_layout(raw_unit[:num_blocks], block_sizes, *offsets)
    return block_sizes, offsets


def unit_size(num_blocks: int, offsets: Layout, has_time: bool = False) -> int:
    """Calculates the size in bytes of a unit's uncompressed body.

    Args:
        num_blocks: Number of blocks in the unit.
        offsets: The blocks' Layout.
        has_time: Whether the unit has a time axis (time_start, time_step and time_ref fields).

    Returns:
        Size of the uncompressed body in bytes.
    """
    size = num_blocks * METADATA_BYTES_PER_BLOCK
    size += int(offsets.group_offsets[-1]) * 8 * BYTES_PER_RESIDUAL_SAMPLE
    size += int(offsets.code_offsets[-1]) * NONFINITE_BITS_PER_SAMPLE
    if has_time:
        size += num_blocks * TIME_BYTES_PER_BLOCK
        size += (int(offsets.short_offsets[-1]) + int(offsets.long_offsets[-1])) * TIME_SHORT_PLANES
    return size


def unit_size_bounds(num_blocks: int, num_samples: int, has_time: bool) -> tuple[int, int]:
    """The smallest and largest body sizes a header's block and sample counts allow.

    The fewest plane bytes are ceil(num_samples / 8) per plane (every block a multiple of 8,
    and no block flagged); the most add up to 7 padding samples per block (but no more than
    one byte per sample), with every block flagged and long.

    Returns:
        A tuple of (smallest, largest) body sizes in bytes.
    """
    columns = num_blocks * (METADATA_BYTES_PER_BLOCK + (TIME_BYTES_PER_BLOCK if has_time else 0))
    fewest_groups = plane_groups(num_samples)
    most_groups = min(num_samples, (num_samples + 7 * num_blocks) // 8)
    plane_bytes_per_group = 8 * BYTES_PER_RESIDUAL_SAMPLE + NONFINITE_BITS_PER_SAMPLE
    if has_time:
        plane_bytes_per_group += 2 * TIME_SHORT_PLANES
    smallest = columns + fewest_groups * 8 * BYTES_PER_RESIDUAL_SAMPLE
    return smallest, columns + most_groups * plane_bytes_per_group


class UnitHeader(NamedTuple):
    """The fields of a parsed unit header.

    Attributes:
        num_blocks: Number of blocks.
        num_samples: Sample count of the unit (the sum of its block sizes).
        byte_planes: Whether the residual planes are byte planes (flags bit 0).
        time_unit: Time unit code (TimeUnit; 0 for a unit without a time axis).
    """

    num_blocks: int
    num_samples: int
    byte_planes: bool
    time_unit: int


def pack_header(num_blocks: int, num_samples: int, byte_planes: bool = False, time_unit: int = 0) -> bytes:
    """Builds the 8-byte unit header.

    Args:
        num_blocks: Number of blocks.
        num_samples: Sample count of the unit.
        byte_planes: Whether the residual planes are byte planes (flags bit 0).
        time_unit: Time unit code (TimeUnit; 0 for no time axis).

    Returns:
        The header bytes.
    """
    flags = (FLAG_BYTE_PLANES if byte_planes else 0) | (time_unit << FLAG_TIME_UNIT_SHIFT)
    return UNIT_HEADER.pack(FORMAT_VERSION, flags, num_blocks, num_samples)


def unpack_header(unit: bytes) -> UnitHeader:
    """Parses and validates a unit header.

    Args:
        unit: Unit bytes (header followed by the zstd frame).

    Returns:
        The parsed header.

    Raises:
        ValueError: If the header is truncated, has an unsupported version or
            reserved bits set, an unknown time unit, or describes an implausible sample
            count.
    """
    if len(unit) < HEADER_BYTES:
        raise ValueError(f"unit of {len(unit)} bytes is shorter than its {HEADER_BYTES}-byte header")
    version, flags, num_blocks, num_samples = UNIT_HEADER.unpack_from(unit)
    if version != FORMAT_VERSION:
        raise ValueError(f"unit format version {version} is not supported (expected {FORMAT_VERSION})")
    if flags & FLAG_RESERVED:
        raise ValueError("unit header sets reserved bits (not supported by this version)")
    time_unit = (flags & FLAG_TIME_UNIT_MASK) >> FLAG_TIME_UNIT_SHIFT
    if time_unit > TimeUnit.NANOSECONDS:
        raise ValueError(f"unit header time unit {time_unit} is not supported (expected 0 to {int(TimeUnit.NANOSECONDS)})")
    if num_samples > min(MAX_UNIT_SAMPLES, num_blocks * MAX_BLOCK_LEN):
        raise ValueError(
            f"unit sample count {num_samples} is over {MAX_UNIT_SAMPLES} or what {num_blocks} blocks can hold"
        )
    return UnitHeader(num_blocks, num_samples, bool(flags & FLAG_BYTE_PLANES), time_unit)


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
def shuffle_block(
    residuals: np.ndarray,
    bit_planes: np.ndarray,
    group_offset: int,
    scratch_low_bytes: np.ndarray,
    scratch_high_bytes: np.ndarray,
) -> None:
    """Zigzag-encodes int16 residuals and packs them into unit bit planes.

    Converts signed differences to unsigned integers via zigzag mapping, splits
    them into low and high byte arrays, transposes 8x8 bit groups, and scatters
    bits across the 16 bit planes.

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
    # Reinterpret byte scratch buffers as 64-bit words for 8-byte group transposition
    low_words64 = scratch_low_bytes.view(np.uint64)
    high_words64 = scratch_high_bytes.view(np.uint64)
    for group_idx in range(num_8byte_groups):
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
def block_sizes_start(num_blocks: int) -> int:
    """Offset of the block_sizes field: after the N head bytes."""
    return BYTES_PER_HEADER * num_blocks


@njit(inline="always")
def grid_params_start(num_blocks: int) -> int:
    """Offset of the grid_params field: after the block_flags and block_sizes fields."""
    return (BYTES_PER_HEADER + BYTES_PER_SIZE) * num_blocks


@njit(inline="always")
def value_anchor_start(num_blocks: int) -> int:
    """Offset of the value_anchor field: after the grid_params field."""
    return (BYTES_PER_HEADER + BYTES_PER_SIZE + BYTES_PER_PARAM) * num_blocks


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
def code_planes_start(num_blocks: int, num_groups: int, has_time: bool) -> int:
    """Offset of the nonfinite code planes: after the residual planes."""
    return residual_start(num_blocks, has_time) + BYTES_PER_RESIDUAL_SAMPLE * 8 * num_groups


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
        block_flags: 1D uint8 array of block flags (HEAD_IRREGULAR_TIME selects the residual planes).
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
        if block_flags[block_idx] & HEAD_IRREGULAR_TIME:
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


@njit(nogil=True, cache=True)
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
) -> tuple[int, int]:
    """Deserializes the time_start, time_step, time_ref and time_residual_planes fields.

    Args:
        raw_unit: 1D uint8 array of uncompressed body bytes of a unit with a time axis.
        sample_offsets, group_offsets, code_offsets, short_offsets, long_offsets: The Layout.
        out_time_starts: Output 1D int64 array receiving the block start times (0 for an
            empty block).
        out_time_steps: Output 1D int64 array receiving the block time steps.
        out_time_refs: Output 1D uint64 array receiving the block reference quotients.
        out_time_residuals: Output 1D uint64 array of every sample receiving the zigzagged
            time residuals of irregular blocks (regular blocks' samples are not written).

    Returns:
        A tuple of (status, block_idx): TIME_ROWS_OK, or TIME_ROWS_BAD_START and the
        first block whose start is int64 minimum or overflows int64, TIME_ROWS_BAD_LONG
        and the first long block whose planes 32-63 are all zero, or TIME_ROWS_BAD_EMPTY and
        the first empty block with a nonzero time column.
    """
    num_blocks = out_time_starts.shape[0]
    short_planes, long_planes = time_planes_views(
        raw_unit, num_blocks, int(group_offsets[num_blocks]), int(code_offsets[num_blocks]), int(short_offsets[num_blocks]),
        int(long_offsets[num_blocks]),
    )
    scratch_tail = np.empty(8, np.uint64)
    int64_max = np.uint64(0x7FFFFFFFFFFFFFFF)
    seen_samples = False
    previous_start = np.int64(0)
    for block_idx in range(num_blocks):
        stored_start = get_int64(raw_unit, time_start_start(num_blocks), num_blocks, block_idx)
        out_time_steps[block_idx] = get_int64(raw_unit, time_step_start(num_blocks), num_blocks, block_idx)
        out_time_refs[block_idx] = np.uint64(get_int64(raw_unit, time_ref_start(num_blocks), num_blocks, block_idx))
        if sample_offsets[block_idx + 1] == sample_offsets[block_idx]:
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
        if raw_unit[block_idx] & HEAD_IRREGULAR_TIME:
            is_long = raw_unit[block_idx] & HEAD_LONG_TIME
            num_active_bytes = unshuffle_time_residuals(
                short_planes,
                long_planes,
                short_offsets[block_idx],
                long_offsets[block_idx] if is_long else -1,
                out_time_residuals[sample_offsets[block_idx]:sample_offsets[block_idx + 1]],
                scratch_tail,
            )
            if is_long and num_active_bytes <= 4:
                return TIME_ROWS_BAD_LONG, block_idx
    return TIME_ROWS_OK, 0


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
        # Store the 8-byte grid parameter and value anchor in byte-planed layout
        put_int64(out_raw_unit, grid_params_start(num_blocks), num_blocks, block_idx, grid_params[block_idx])
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
        if block_flags[block_idx] & HEAD_NONFINITE:
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
        out_grid_params[block_idx] = get_int64(raw_unit, grid_params_start(num_blocks), num_blocks, block_idx)
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
        if raw_unit[block_idx] & HEAD_NONFINITE:
            # Unpack non-finite codes for flagged blocks
            get_codes(cplanes, code_offsets[block_idx], out_codes[first_sample:first_sample + block_len])


class TimeRows(NamedTuple):
    """The time axis of a unit as per-block rows.

    Attributes:
        starts: 1D int64 array of block start times (absolute, in the unit's time unit; 0
            for an empty block).
        steps: 1D int64 array of block time steps: the GCD of each block's time deltas.
        refs: 1D uint64 array of block reference quotients: sample i > 0's quotient
            (time[i] - time[i-1]) / step is the reference plus its residual.
        residuals: 1D uint64 array of every sample: per sample of an irregular block, its
            quotient minus the reference, mod 2^64 and zigzagged (0 for sample 0). Regular
            blocks' samples (all residuals 0) aren't read or written.
    """

    starts: np.ndarray
    steps: np.ndarray
    refs: np.ndarray
    residuals: np.ndarray


def allocate_time_rows(num_blocks: int, num_samples: int, zero_residuals: bool = False) -> TimeRows:
    """Uninitialized time rows; zero_residuals zeroes the residuals (regular blocks' samples
    are otherwise left as allocated)."""
    residuals = (np.zeros if zero_residuals else np.empty)(num_samples, np.uint64)
    return TimeRows(np.empty(num_blocks, np.int64), np.empty(num_blocks, np.int64), np.empty(num_blocks, np.uint64), residuals)


class UnitRows(NamedTuple):
    """The deserialized per-block rows of a unit.

    Attributes:
        block_flags: 1D uint8 array of block flags.
        block_sizes: 1D int64 array of block sizes.
        grid_params: 1D int64 array of block grid parameters.
        value_anchors: 1D int64 array of block value anchors.
        residuals: 1D int16 array of every sample's residual.
        codes: 1D uint8 array of every sample's code (zeros in unflagged blocks).
        time_rows: The time axis rows, or None for a unit without a time axis.
    """

    block_flags: np.ndarray
    block_sizes: np.ndarray
    grid_params: np.ndarray
    value_anchors: np.ndarray
    residuals: np.ndarray
    codes: np.ndarray
    time_rows: TimeRows | None


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
            HEAD_NONFINITE flag set.
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
    if raw_arr.shape[0] < (BYTES_PER_HEADER + BYTES_PER_SIZE) * num_blocks:
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
