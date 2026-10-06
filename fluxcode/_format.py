# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Binary layout of units: the header, the field offsets, the Layout and the row types.

The code that packs rows into body bytes and back (bit planes, byte planes, code planes, time
residual planes) is in _bitpacking.

A unit is an 8-byte header followed by one zstd frame holding the body. The header
(little-endian) is: format version (uint8, 1), flags (uint8; bit 0 selects byte planes for
the residual planes, bits 1-3 hold the time unit, bits 4-7 are 0), the block count (uint16)
and the sample count (uint32). Every block records its own size, 0 to 65,535 samples, and
the block sizes add up to the sample count.

Each block of n samples takes num_octets = ceil(n / 8) bytes of every bit plane (none for an
empty block): when n isn't a multiple of 8, the high bits of its last byte are padding, and
byte planes pad each block to 8 * num_octets bytes. Writers zero the padding; decoders
ignore it. A field "per octet" below holds that many bytes per block, the blocks' bytes
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
    3. grid_params (2 * num_blocks bytes):
       Per-block int16 grid parameter (power-of-two exponent or decimal power),
       byte-planed (all low bytes, then all high bytes).
    4. value_anchor (8 * num_blocks bytes, byte-planed: all byte 0s, all byte 1s, ...):
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
    8. residual_planes (16 bytes per octet of every block):
       Zigzag-encoded int16 differences mod 2^16, as 16 bit planes across all
       blocks, or (flags bit 0) as 2 byte planes: every low byte, then every high byte.
    9. nonfinite_code_planes (2 bytes per octet of the blocks with non-finite values):
       2 code bit planes for the flagged blocks, encoding sample categories
       (00: finite, 01: NaN, 10: +inf, 11: -inf).
    10. time_residual_planes (32 bytes per octet of the irregular blocks, then 32 per octet
       of the long blocks; time only): per sample of an irregular block, its quotient minus
       time_ref, mod 2^64 and zigzagged (0 for sample 0), as bit planes like the residual
       planes: planes 0-31 of every irregular block, then planes 32-63 of the long blocks.
"""

import enum
import struct
from typing import NamedTuple

import numpy as np
from numba import njit

BLOCK_FLAG_ORDER: int = 0x03
"""Block flags: the difference predictor order (bits 0-1)."""

BLOCK_FLAG_DECIMAL: int = 0x04
"""Block flags: decimal quantization (bit 2)."""

BLOCK_FLAG_NONFINITE: int = 0x08
"""Block flags: non-finite code planes present (bit 3)."""

BLOCK_FLAG_IRREGULAR_TIME: int = 0x10
"""Block flags: irregular times, with time residual planes (bit 4); only valid in units with
a time axis."""

BLOCK_FLAG_LONG_TIME: int = 0x20
"""Block flags: long time residuals, stored in 64 bit planes instead of 32 (bit 5); only valid
with BLOCK_FLAG_IRREGULAR_TIME."""

BLOCK_FLAG_RESERVED: int = 0xC0
"""Block flags: reserved bits (6-7); decoders reject them."""

UNIT_FLAG_BYTE_PLANES: int = 0x01
"""Unit header flag: the residual field holds 2 byte planes instead of 16 bit planes."""

UNIT_FLAG_TIME_UNIT_SHIFT: int = 1
"""Position of the 3-bit time unit in the unit header flags (bits 1-3)."""

UNIT_FLAG_TIME_UNIT_MASK: int = 0x0E
"""Unit header flag bits holding the time unit."""

UNIT_FLAG_RESERVED: int = 0xF0
"""Reserved unit header flag bits (4-7); decoders must reject these."""

MAX_BLOCK_LEN: int = 0xFFFF
"""Largest block size: the block_sizes field is uint16."""

MAX_BLOCKS: int = 0xFFFF
"""Largest block count: the header field is uint16."""

SHORT_BLOCK_LEN: int = 8
"""Blocks of at most this many samples skip the analysis: they take the finest step
(max_quantize_bits), on a decimal grid if one is detected, and order 0."""


class NonFiniteCode(enum.IntEnum):
    """Two-bit sample classification codes for non-finite values.

    Kernels use the plain-int CODE_* constants below: numba can't lower IntEnum members in
    all expressions.
    """

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


class TimeUnitCode(enum.IntEnum):
    """Time unit codes stored in unit header flags bits 1-3 (0: no time axis)."""

    NONE = 0
    SECONDS = 1
    MILLISECONDS = 2
    MICROSECONDS = 3
    NANOSECONDS = 4


TIME_UNIT_NAMES: dict[int, str] = {
    TimeUnitCode.SECONDS: "s",
    TimeUnitCode.MILLISECONDS: "ms",
    TimeUnitCode.MICROSECONDS: "us",
    TimeUnitCode.NANOSECONDS: "ns",
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

BYTES_PER_FLAGS: int = 1
"""Bytes of block_flags per block."""

BYTES_PER_SIZE: int = 2
"""Bytes of block_sizes per block (uint16)."""

BYTES_PER_PARAM: int = 2
"""Bytes of grid_params per block (int16)."""

BYTES_PER_ANCHOR: int = 8
"""Bytes of value_anchor per block (float64 bits or int64 grid index)."""

METADATA_BYTES_PER_BLOCK: int = BYTES_PER_FLAGS + BYTES_PER_SIZE + BYTES_PER_PARAM + BYTES_PER_ANCHOR
"""Bytes of the value columns per block (flags, size, grid parameter, anchor: 13)."""

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

BYTES_PER_TIME_COLUMN: int = 8
"""Bytes of each time column (time_start, time_step, time_ref) per block."""

TIME_BYTES_PER_BLOCK: int = 3 * BYTES_PER_TIME_COLUMN
"""Bytes of the time columns per block in units with a time axis (24)."""

TIME_SHORT_PLANES: int = 32
"""Time residual bit planes stored for every irregular block; a long block stores as many again."""


@njit(inline="always")
def plane_octets(block_len: int) -> int:
    """Calculates the byte width required per bit plane for a given block length.

    Each octet holds 8 samples. Blocks whose length is not a multiple of 8 are
    padded with zero bits in their trailing byte.

    Args:
        block_len: Number of samples in the block.

    Returns:
        Number of bytes per bit plane (ceil(block_len / 8)).
    """
    return (block_len + 7) // 8


class Layout(NamedTuple):
    """Where each block's data sits in a unit's flat arrays and plane fields.

    Every array has num_blocks + 1 entries: entry b is block b's offset, and the last entry
    the total.

    Attributes:
        sample_offsets: Offsets of the blocks' samples (cumulative block sizes).
        octet_offsets: Offsets of the blocks' bytes in each residual bit plane.
        code_offsets: Offsets in each non-finite code plane (only flagged blocks take bytes).
        short_offsets: Offsets in time residual planes 0-31 (only irregular blocks take bytes).
        long_offsets: Offsets in time residual planes 32-63 (only long blocks take bytes).
    """

    sample_offsets: np.ndarray
    octet_offsets: np.ndarray
    code_offsets: np.ndarray
    short_offsets: np.ndarray
    long_offsets: np.ndarray


@njit(nogil=True, cache=True)
def _fill_layout(
    block_flags: np.ndarray,
    block_sizes: np.ndarray,
    out_sample_offsets: np.ndarray,
    out_octet_offsets: np.ndarray,
    out_code_offsets: np.ndarray,
    out_short_offsets: np.ndarray,
    out_long_offsets: np.ndarray,
) -> None:
    """Accumulates cumulative layout offsets from block flags and sizes.

    Args:
        block_flags: 1D uint8 array of per-block flags.
        block_sizes: 1D int64 array of sample counts per block.
        out_sample_offsets: Preallocated 1D int64 array for sample offsets.
        out_octet_offsets: Preallocated 1D int64 array for byte-plane octet offsets.
        out_code_offsets: Preallocated 1D int64 array for non-finite code plane offsets.
        out_short_offsets: Preallocated 1D int64 array for short time residual offsets.
        out_long_offsets: Preallocated 1D int64 array for long time residual offsets.
    """
    out_sample_offsets[0] = out_octet_offsets[0] = out_code_offsets[0] = 0
    out_short_offsets[0] = out_long_offsets[0] = 0
    for block_idx in range(block_sizes.shape[0]):
        num_samples = block_sizes[block_idx]
        num_octets = plane_octets(num_samples)
        flags = block_flags[block_idx]
        out_sample_offsets[block_idx + 1] = out_sample_offsets[block_idx] + num_samples
        out_octet_offsets[block_idx + 1] = out_octet_offsets[block_idx] + num_octets
        out_code_offsets[block_idx + 1] = out_code_offsets[block_idx] + (num_octets if flags & BLOCK_FLAG_NONFINITE else 0)
        out_short_offsets[block_idx + 1] = out_short_offsets[block_idx] + (
            num_octets if flags & BLOCK_FLAG_IRREGULAR_TIME else 0
        )
        out_long_offsets[block_idx + 1] = out_long_offsets[block_idx] + (num_octets if flags & BLOCK_FLAG_LONG_TIME else 0)


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
    """Reads the byte-planed uint16 block_sizes field of an uncompressed body.

    Args:
        raw_unit: 1D uint8 array of uncompressed unit body bytes.
        num_blocks: Total number of blocks in the unit.
        out_block_sizes: Preallocated 1D int64 array to receive block sizes.
    """
    sizes_start = block_sizes_start(num_blocks)
    for block_idx in range(num_blocks):
        out_block_sizes[block_idx] = np.int64(raw_unit[sizes_start + block_idx]) | (
            np.int64(raw_unit[sizes_start + num_blocks + block_idx]) << 8
        )


def read_layout(raw_unit: np.ndarray, num_blocks: int) -> tuple[np.ndarray, Layout]:
    """Reads block sizes and computes the layout of an uncompressed body.

    Args:
        raw_unit: 1D uint8 array of uncompressed body bytes.
        num_blocks: Total number of blocks in the unit.

    Returns:
        A tuple of (block_sizes, layout): 1D int64 array of block sample counts
        and the computed Layout offsets tuple.
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
    size += int(offsets.octet_offsets[-1]) * 8 * BYTES_PER_RESIDUAL_SAMPLE
    size += int(offsets.code_offsets[-1]) * NONFINITE_BITS_PER_SAMPLE
    if has_time:
        size += num_blocks * TIME_BYTES_PER_BLOCK
        size += (int(offsets.short_offsets[-1]) + int(offsets.long_offsets[-1])) * TIME_SHORT_PLANES
    return size


def unit_size_bounds(num_blocks: int, num_samples: int, has_time: bool) -> tuple[int, int]:
    """Calculates lower and upper bounds on uncompressed body size.

    The fewest plane bytes occur when every block size is a multiple of 8 and no
    block is flagged for non-finite values or irregular times. The maximum occurs
    when every block has 7 padding samples and all blocks carry non-finite code
    planes and 64-plane long time residuals.

    Args:
        num_blocks: Total number of blocks.
        num_samples: Total number of samples across all blocks.
        has_time: Whether the unit includes time axis columns and planes.

    Returns:
        A tuple of (smallest, largest) body sizes in bytes.
    """
    columns = num_blocks * (METADATA_BYTES_PER_BLOCK + (TIME_BYTES_PER_BLOCK if has_time else 0))
    fewest_octets = plane_octets(num_samples)
    most_octets = min(num_samples, (num_samples + 7 * num_blocks) // 8)
    plane_bytes_per_octet = 8 * BYTES_PER_RESIDUAL_SAMPLE + NONFINITE_BITS_PER_SAMPLE
    if has_time:
        plane_bytes_per_octet += 2 * TIME_SHORT_PLANES
    smallest = columns + fewest_octets * 8 * BYTES_PER_RESIDUAL_SAMPLE
    return smallest, columns + most_octets * plane_bytes_per_octet


class UnitHeader(NamedTuple):
    """The fields of a parsed unit header.

    Attributes:
        num_blocks: Number of blocks.
        num_samples: Sample count of the unit (the sum of its block sizes).
        byte_planes: Whether the residual planes are byte planes (flags bit 0).
        time_unit: Time unit code (TimeUnitCode; 0 for a unit without a time axis).
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
        time_unit: Time unit code (TimeUnitCode; 0 for no time axis).

    Returns:
        The header bytes.
    """
    flags = (UNIT_FLAG_BYTE_PLANES if byte_planes else 0) | (time_unit << UNIT_FLAG_TIME_UNIT_SHIFT)
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
    if flags & UNIT_FLAG_RESERVED:
        raise ValueError("unit header sets reserved bits (not supported by this version)")
    time_unit = (flags & UNIT_FLAG_TIME_UNIT_MASK) >> UNIT_FLAG_TIME_UNIT_SHIFT
    if time_unit > TimeUnitCode.NANOSECONDS:
        raise ValueError(f"unit header time unit {time_unit} is not supported (expected 0 to {int(TimeUnitCode.NANOSECONDS)})")
    if num_samples > min(MAX_UNIT_SAMPLES, num_blocks * MAX_BLOCK_LEN):
        raise ValueError(
            f"unit sample count {num_samples} is over {MAX_UNIT_SAMPLES} or what {num_blocks} blocks can hold"
        )
    return UnitHeader(num_blocks, num_samples, bool(flags & UNIT_FLAG_BYTE_PLANES), time_unit)


@njit(inline="always")
def residual_start(num_blocks: int, has_time: bool) -> int:
    """Calculates the byte offset of residual planes in an uncompressed body.

    Residual planes begin immediately after the per-block value metadata columns
    (and optional time columns).

    Args:
        num_blocks: Total number of blocks in the unit.
        has_time: Whether the unit includes time axis columns.

    Returns:
        Offset in bytes where residual planes begin.
    """
    return (METADATA_BYTES_PER_BLOCK + (TIME_BYTES_PER_BLOCK if has_time else 0)) * num_blocks


@njit(inline="always")
def block_sizes_start(num_blocks: int) -> int:
    """Calculates the byte offset of the block_sizes field.

    Args:
        num_blocks: Total number of blocks in the unit.

    Returns:
        Offset in bytes where the block_sizes field begins.
    """
    return BYTES_PER_FLAGS * num_blocks


@njit(inline="always")
def grid_params_start(num_blocks: int) -> int:
    """Calculates the byte offset of the grid_params field.

    Args:
        num_blocks: Total number of blocks in the unit.

    Returns:
        Offset in bytes where the grid_params field begins.
    """
    return (BYTES_PER_FLAGS + BYTES_PER_SIZE) * num_blocks


@njit(inline="always")
def value_anchor_start(num_blocks: int) -> int:
    """Calculates the byte offset of the value_anchor field.

    Args:
        num_blocks: Total number of blocks in the unit.

    Returns:
        Offset in bytes where the value_anchor field begins.
    """
    return (BYTES_PER_FLAGS + BYTES_PER_SIZE + BYTES_PER_PARAM) * num_blocks


@njit(inline="always")
def time_start_start(num_blocks: int) -> int:
    """Calculates the byte offset of the time_start field.

    Args:
        num_blocks: Total number of blocks in the unit.

    Returns:
        Offset in bytes where the time_start field begins.
    """
    return METADATA_BYTES_PER_BLOCK * num_blocks


@njit(inline="always")
def time_step_start(num_blocks: int) -> int:
    """Calculates the byte offset of the time_step field.

    Args:
        num_blocks: Total number of blocks in the unit.

    Returns:
        Offset in bytes where the time_step field begins.
    """
    return (METADATA_BYTES_PER_BLOCK + BYTES_PER_TIME_COLUMN) * num_blocks


@njit(inline="always")
def time_ref_start(num_blocks: int) -> int:
    """Calculates the byte offset of the time_ref field.

    Args:
        num_blocks: Total number of blocks in the unit.

    Returns:
        Offset in bytes where the time_ref field begins.
    """
    return (METADATA_BYTES_PER_BLOCK + 2 * BYTES_PER_TIME_COLUMN) * num_blocks


@njit(inline="always")
def put_int64(raw_unit: np.ndarray, field_start: int, num_blocks: int, block_idx: int, value: int | np.integer) -> None:
    """Writes a block's int64 value into a byte-planed field.

    Byte i of the value goes to the i-th run of num_blocks bytes of the field.

    Args:
        raw_unit: 1D uint8 array of uncompressed body bytes.
        field_start: Offset of the field (value_anchor_start or a time column's).
        num_blocks: Total number of blocks.
        block_idx: Zero-based block index.
        value: Signed 64-bit integer to write.
    """
    value_bits = np.uint64(value)
    for byte_idx in range(8):
        # Byte byte_idx goes to field_start + byte_idx * num_blocks + block_idx
        raw_unit[field_start + byte_idx * num_blocks + block_idx] = np.uint8(
            (value_bits >> np.uint64(8 * byte_idx)) & np.uint64(0xFF)
        )


@njit(inline="always")
def get_int64(raw_unit: np.ndarray, field_start: int, num_blocks: int, block_idx: int) -> np.int64:
    """Reads a block's int64 value from a byte-planed field.

    Args:
        raw_unit: 1D uint8 array of uncompressed body bytes.
        field_start: Offset of the field (value_anchor_start or a time column's).
        num_blocks: Total number of blocks.
        block_idx: Zero-based block index.

    Returns:
        The block's signed 64-bit integer.
    """
    value_bits = np.uint64(0)
    for byte_idx in range(8):
        # Byte byte_idx comes from field_start + byte_idx * num_blocks + block_idx
        value_bits |= np.uint64(raw_unit[field_start + byte_idx * num_blocks + block_idx]) << np.uint64(8 * byte_idx)
    return np.int64(value_bits)


@njit(inline="always")
def put_int16(raw_unit: np.ndarray, field_start: int, num_blocks: int, block_idx: int, value: int | np.integer) -> None:
    """Writes a block's int16 value into a byte-planed field.

    The low byte goes to the first run of num_blocks bytes of the field, the high byte to
    the second.

    Args:
        raw_unit: 1D uint8 array of uncompressed body bytes.
        field_start: Offset where the target field starts.
        num_blocks: Total number of blocks in the unit.
        block_idx: Zero-based block index.
        value: Signed 16-bit integer value to write.
    """
    raw_unit[field_start + block_idx] = np.uint8(value & 0xFF)
    raw_unit[field_start + num_blocks + block_idx] = np.uint8((value >> 8) & 0xFF)


@njit(inline="always")
def get_int16(raw_unit: np.ndarray, field_start: int, num_blocks: int, block_idx: int) -> np.int64:
    """Reads a block's int16 value from a byte-planed field, sign-extended to int64.

    Args:
        raw_unit: 1D uint8 array of uncompressed body bytes.
        field_start: Offset where the target field starts.
        num_blocks: Total number of blocks in the unit.
        block_idx: Zero-based block index.

    Returns:
        Sign-extended 64-bit integer value.
    """
    value_bits = np.int64(raw_unit[field_start + block_idx]) | (np.int64(raw_unit[field_start + num_blocks + block_idx]) << 8)
    return value_bits - ((value_bits & 0x8000) << 1)


@njit(inline="always")
def code_planes_start(num_blocks: int, num_octets: int, has_time: bool) -> int:
    """Calculates the byte offset of non-finite code planes.

    Args:
        num_blocks: Total number of blocks in the unit.
        num_octets: Total octet count across all blocks.
        has_time: Whether the unit includes time axis columns.

    Returns:
        Offset in bytes where non-finite code planes begin.
    """
    return residual_start(num_blocks, has_time) + BYTES_PER_RESIDUAL_SAMPLE * 8 * num_octets


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
    """Allocates pre-sized arrays for time-axis rows.

    Args:
        num_blocks: Total number of blocks.
        num_samples: Total number of samples across all blocks.
        zero_residuals: If True, residual array is zero-initialized; otherwise left uninitialized.

    Returns:
        A TimeRows tuple containing allocated arrays.
    """
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
