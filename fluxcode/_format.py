# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Binary wire format for block groups: header, field offsets, BodyLayout, and column
structs.

Low-level bit-plane and byte-plane codecs live in ``_bitpacking``; full body
serialization, deserialization, and splicing live in ``_body``.

Wire layout of a block group ---------------------------- A block group is an 8-byte
little-endian header followed by a single zstd frame containing the uncompressed
``body``:

    Header (8 bytes, ``<BBHI``):
        - ``version`` (uint8): format version (1).
        - ``flags`` (uint8):
            - bit 0: residual planes stored as 2 byte planes instead of 16 bit planes.
            - bits 1-3: time unit code (``TimeUnitCode``; 0 means no time axis).
            - bit 4: ``group_time_step`` uint64 trailer present at the end of the body.
            - bits 5-7: reserved (must be 0).
        - ``num_blocks`` (uint16): number of blocks in the group (0..65535).
        - ``num_samples`` (uint32): total sample count (sum of all block sizes).

Plane-byte alignment -------------------- Each block of ``n`` samples occupies
``num_plane_bytes = ceil(n / 8)`` bytes in every bit plane (0 bytes for an empty block),
where each plane byte holds 8 samples (``SAMPLES_PER_PLANE_BYTE = 8``). When ``n`` is
not a multiple of 8, trailing bits in the last plane byte are zero-padded; in byte-plane
mode each block is likewise zero-padded to ``8 * num_plane_bytes`` bytes. Because every
block starts on a byte boundary in every plane field, individual blocks can be decoded
or copied independently.

Uncompressed body fields (for ``num_blocks`` blocks)
---------------------------------------------------- Fields 6-8 and 11 are present only
when the header's time unit is non-zero:

    1. ``block_flags`` (``num_blocks`` bytes):
       Bits 0-1: predictor difference order (0-3).
       Bit 2: decimal quantization (base 10 instead of base 2).
       Bit 3: 2-bit sample code planes present (NaN and/or escaped values).
       Bit 4: irregular timestamps (time residual planes present).
       Bit 5: wide time residuals (64 bit planes instead of 32; requires bit 4).
       Bit 6: scaled grid (step ``N / D * B^e`` with ``N, D`` in ``grid_scales``).
       Bit 7: reserved (must be 0). Empty blocks have flags == 0.
    2. ``block_sizes`` (``2 * num_blocks`` bytes, byte-planed uint16):
       Per-block sample count (low bytes for all blocks, then high bytes).
    3. ``grid_exponents`` (``2 * num_blocks`` bytes, byte-planed int16):
       Per-block quantization exponent (power-of-two ``e`` or decimal ``p``).
    4. ``grid_scales`` (``4 * num_blocks`` bytes, byte-planed uint32):
       Per-block ``N | (D << 16)`` for scaled blocks; 0 for unscaled blocks.
    5. ``block_anchors`` (``8 * num_blocks`` bytes, byte-planed):
       Per-block reconstruction anchor for signed int16 grid indices ``q``:
       finite IEEE-754 float64 bits (power-of-two grid) or signed int64 grid index
       ``K0`` (decimal or scaled grid).
    6. ``time_starts`` (``8 * num_blocks`` bytes, byte-planed; time axis only):
       First non-empty block's start time as int64; each subsequent non-empty block's
       unsigned increase over the previous non-empty block's start (0 for empty blocks).
       Divided by ``group_time_step`` when header flag bit 4 is set.
    7. ``time_steps`` (``8 * num_blocks`` bytes, byte-planed; time axis only):
       Per-block int64 GCD of within-block timestamp deltas (0 if all deltas are 0).
       Divided by ``group_time_step`` when header flag bit 4 is set.
    8. ``time_ref_quotients`` (``8 * num_blocks`` bytes, byte-planed; time axis only):
       Per-block uint64 reference quotient: sample ``i > 0`` has quotient
       ``(time[i] - time[i - 1]) / time_step = time_ref_quotient + residual``
       (0 if regular).
    9. ``residual_planes`` (16 bytes per plane byte of every block):
       Zigzag-encoded int16 differences mod 2^16, stored either as 16 bit planes across
       all blocks or (header flag bit 0) as 2 byte planes (all low bytes, then all
       high bytes).
    10. ``sample_code_planes`` (2 bytes per plane byte of blocks with bit 3 set):
        2 bit planes encoding per-sample exception codes (00: on-grid, 01: NaN,
        10: escaped float64, 11: repeat previous escaped float64).
    11. ``time_residual_planes`` (32 bytes per plane byte of irregular blocks, then
        32 bytes per plane byte of wide-time blocks; time axis only):
        Per-sample zigzagged ``(quotient - time_ref_quotient) mod 2^64`` (0 for
        sample 0), stored as bit planes 0-31 for all irregular blocks followed by
        planes 32-63 for wide-time blocks.
    12. ``escaped_values`` (``8 * num_escapes`` bytes, byte-planed; optional):
        Escaped float64 values (±inf and outliers) in block/sample order, one per
        code 10.
    13. ``group_time_step`` (8 bytes, little-endian uint64; iff header flag bit 4):
        Common divisor >= 2 applied to all stored ``time_starts`` increases and
        ``time_steps``.
"""

import enum
import struct
from typing import NamedTuple

import numpy as np
from numba import njit

# Per-block flag bits (stored in the block_flags column).
BLOCK_FLAG_ORDER: int = 0x03
"""Block flag mask: predictor difference order (bits 0-1, values 0..3)."""

BLOCK_FLAG_DECIMAL: int = 0x04
"""Block flag bit 2: decimal quantization (base 10 instead of base 2)."""

BLOCK_FLAG_SAMPLE_CODES: int = 0x08
"""Block flag bit 3: 2-bit sample code planes present for NaN and/or escaped samples."""

BLOCK_FLAG_IRREGULAR_TIME: int = 0x10
"""Block flag bit 4: irregular timestamps with time residual planes (time axis only)."""

BLOCK_FLAG_WIDE_TIME: int = 0x20
"""Block flag bit 5: wide time residuals in 64 bit planes instead of 32 (requires bit
4).
"""

BLOCK_FLAG_SCALED: int = 0x40
"""Block flag bit 6: scaled grid with step ``(N / D) * 2^e`` (or ``10^p`` with bit 2).
"""

BLOCK_FLAG_RESERVED: int = 0x80
"""Block flag bit 7: reserved; decoders reject any block with this bit set."""

# Group header flag bits (stored in byte 1 of the 8-byte group header).
GROUP_FLAG_BYTE_PLANES: int = 0x01
"""Header flag bit 0: residual field is stored as 2 byte planes instead of 16 bit
planes.
"""

GROUP_FLAG_TIME_UNIT_SHIFT: int = 1
"""Bit position of the 3-bit time unit code in header flags (bits 1-3)."""

GROUP_FLAG_TIME_UNIT_MASK: int = 0x0E
"""Bit mask for the time unit code in header flags (bits 1-3)."""

GROUP_FLAG_GROUP_TIME_STEP: int = 0x10
"""Header flag bit 4: time_starts increases and time_steps are divided by a uint64
``group_time_step >= 2`` trailer appended in the last 8 bytes of the body."""

GROUP_FLAG_RESERVED: int = 0xE0
"""Header flag bits 5-7: reserved; decoders reject any group with these bits set."""

GROUP_TIME_STEP_BYTES: int = 8
"""Byte length of the optional uint64 ``group_time_step`` trailer."""

MAX_BLOCK_LEN: int = 0xFFFF
"""Maximum samples per block (``block_sizes`` is uint16)."""

MAX_BLOCKS: int = 0xFFFF
"""Maximum blocks per group (header ``num_blocks`` is uint16)."""

SHORT_BLOCK_LEN: int = 8
"""Blocks with at most this many samples skip statistical analysis and use order 0 at
``max_quantize_bits`` (on a decimal grid if detected)."""

SAMPLES_PER_PLANE_BYTE: int = 8
"""Number of samples packed into each byte of a bit plane."""

# Two-bit sample exception codes (plain ints so Numba kernels can inline them directly).
CODE_ON_GRID: int = 0
"""Two-bit sample code (00b): sample is quantized on the block's grid."""

CODE_NAN: int = 1
"""Two-bit sample code (01b): canonical quiet NaN."""

CODE_ESCAPE: int = 2
"""Two-bit sample code (10b): escaped float64 value (±inf or outlier) from the trailer.
"""

CODE_REPEAT_ESCAPE: int = 3
"""Two-bit sample code (11b): repeat of the block's previous escaped float64 value."""

BYTES_PER_ESCAPED_VALUE: int = 8
"""Bytes per escaped sample (float64, stored byte-planed in the body trailer)."""


class TimeUnitCode(enum.IntEnum):
    """Time unit codes stored in group header flags bits 1-3 (0 means no time axis)."""

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
"""Mapping from ``TimeUnitCode`` to NumPy ``datetime64`` unit string."""

TIME_UNIT_CODES: dict[str, int] = {name: code for code, name in TIME_UNIT_NAMES.items()}
"""Mapping from NumPy ``datetime64`` unit string to ``TimeUnitCode``."""

# Valid exponent and scale bounds for quantization grids.
E_MIN: int = -1074
"""Minimum valid power-of-two exponent (step of the smallest IEEE-754 subnormal double).
"""

E_MAX: int = 1023
"""Maximum valid power-of-two exponent."""

P_MIN: int = -22
"""Minimum valid decimal exponent (10^-22 is an exact IEEE-754 double)."""

P_MAX: int = 22
"""Maximum valid decimal exponent (10^22 is an exact IEEE-754 double)."""

SCALED_E_MIN: int = -1022
"""Minimum power-of-two exponent for a scaled grid so ``(K0 + N * q) * 2^e`` is a normal
double.
"""

SCALED_E_MAX: int = 970
"""Maximum power-of-two exponent for a scaled grid so ``(K0 + N * q) * 2^e < 2^53 *
2^970`` stays finite.
"""

SCALED_P_MAX: int = 0
"""Maximum decimal exponent for a scaled grid (reconstruction divides by ``D * 10^-p``
in one rounding).
"""

SCALE_SHIFT: int = 16
"""Bit shift of denominator ``D`` inside packed uint32 ``grid_scales`` (``N | (D <<
16)``).
"""

SCALE_MASK: int = 0xFFFF
"""16-bit mask for numerator ``N`` and denominator ``D`` in ``grid_scales`` (1..65535).
"""

INDEX_LIMIT: int = 1 << 52
"""Exclusive magnitude bound on decimal/scaled anchor index ``K0`` so ``K0 + N * q``
stays below 2^53.
"""

EXACT_INT_LIMIT: int = 1 << 53
"""Upper bound on integers representable exactly in IEEE-754 float64."""


@njit(inline="always")
def scale_valid(
    decimal: bool, exponent: int, num: int | np.integer, den: int | np.integer
) -> bool:
    """Checks whether a scaled grid's exponent, numerator ``N``, and denominator ``D``
    are valid.

    Args:
        decimal: True for a base-10 scaled grid, False for a base-2 scaled grid.
        exponent: Decimal power ``p`` or power-of-two exponent ``e``.
        num: Numerator ``N`` (step in fine grid points).
        den: Denominator ``D``.

    Returns:
        True if ``N`` and ``D`` are in ``[1, 65535]`` and the exponent satisfies the
        grid's exact-arithmetic bounds (``D * 5^-p < 2^53`` for decimal; ``[-1022,
        970]`` for binary).
    """
    # Validate 16-bit positive rational components first, then check base-specific
    # exponent bounds.
    if not (1 <= num <= SCALE_MASK and 1 <= den <= SCALE_MASK):
        return False
    if decimal:
        # D * 10^-p = (D * 5^-p) * 2^-p is an exact float64 iff D * 5^-p < 2^53.
        return P_MIN <= exponent <= SCALED_P_MAX and bool(
            np.int64(5) ** -exponent < EXACT_INT_LIMIT // den
        )
    return SCALED_E_MIN <= exponent <= SCALED_E_MAX


# Per-block column byte widths.
BYTES_PER_FLAGS: int = 1
"""Bytes per block in ``block_flags`` (uint8)."""

BYTES_PER_SIZE: int = 2
"""Bytes per block in ``block_sizes`` (uint16)."""

BYTES_PER_EXPONENT: int = 2
"""Bytes per block in ``grid_exponents`` (int16)."""

BYTES_PER_SCALE: int = 4
"""Bytes per block in ``grid_scales`` (uint32: ``N | (D << 16)``)."""

BYTES_PER_ANCHOR: int = 8
"""Bytes per block in ``block_anchors`` (float64 bits or int64 grid index)."""

METADATA_BYTES_PER_BLOCK: int = (
    BYTES_PER_FLAGS
    + BYTES_PER_SIZE
    + BYTES_PER_EXPONENT
    + BYTES_PER_SCALE
    + BYTES_PER_ANCHOR
)
"""Total bytes per block across the 5 value metadata columns (17 bytes)."""

FORMAT_VERSION: int = 1
"""Wire format version stored in byte 0 of the group header."""

GROUP_HEADER = struct.Struct("<BBHI")
"""Struct layout for the 8-byte little-endian group header (version, flags, num_blocks,
num_samples).
"""

HEADER_BYTES: int = GROUP_HEADER.size
"""Size of the group header in bytes (8)."""

MAX_GROUP_SAMPLES: int = 1 << 26
"""Maximum sample count accepted by decoders before decompressing a zstd frame."""

BYTES_PER_RESIDUAL_SAMPLE: int = 2
"""Bytes per sample in the residual field (int16 differences mod 2^16)."""

BITS_PER_SAMPLE_CODE: int = 2
"""Bits per sample in the sample code planes (2 bit planes)."""

BYTES_PER_TIME_COLUMN: int = 8
"""Bytes per block in each time metadata column (``time_starts``, ``time_steps``,
``time_ref_quotients``).
"""

TIME_BYTES_PER_BLOCK: int = 3 * BYTES_PER_TIME_COLUMN
"""Total bytes per block across the 3 time metadata columns (24 bytes)."""

TIME_LOW_PLANES: int = 32
"""Number of bit planes in the low half of time residuals (planes 0-31); wide blocks add
32 more.
"""


@njit(inline="always")
def plane_byte_count(block_len: int) -> int:
    """Computes the number of bytes per bit plane needed for a block of ``block_len``
    samples.

    Args:
        block_len: Number of samples in the block (non-negative).

    Returns:
        ``ceil(block_len / 8)``: number of 8-sample plane bytes per bit plane.
    """
    return (block_len + (SAMPLES_PER_PLANE_BYTE - 1)) // SAMPLES_PER_PLANE_BYTE


class BodyLayout(NamedTuple):
    """Cumulative per-block offsets into a block group's flat arrays and plane fields.

    Every offset array has length ``num_blocks + 1``: entry ``b`` is block ``b``'s start
    offset, and entry ``num_blocks`` is the total across all blocks.

    Attributes:
        sample_offsets: Cumulative sample counts (start index of each block's samples).
        plane_byte_offsets: Cumulative 8-sample plane-byte counts in each residual bit
            plane.
        code_plane_offsets: Cumulative plane-byte counts in each sample code plane
            (non-zero only for blocks with ``BLOCK_FLAG_SAMPLE_CODES``).
        time_low_offsets: Cumulative plane-byte counts in time residual planes 0-31
            (non-zero only for blocks with ``BLOCK_FLAG_IRREGULAR_TIME``).
        time_high_offsets: Cumulative plane-byte counts in time residual planes 32-63
            (non-zero only for blocks with ``BLOCK_FLAG_WIDE_TIME``).
    """

    sample_offsets: np.ndarray
    plane_byte_offsets: np.ndarray
    code_plane_offsets: np.ndarray
    time_low_offsets: np.ndarray
    time_high_offsets: np.ndarray


@njit(nogil=True, cache=True)
def _fill_body_layout(
    block_flags: np.ndarray,
    block_sizes: np.ndarray,
    out_sample_offsets: np.ndarray,
    out_plane_byte_offsets: np.ndarray,
    out_code_plane_offsets: np.ndarray,
    out_time_low_offsets: np.ndarray,
    out_time_high_offsets: np.ndarray,
) -> None:
    """Populates the five cumulative offset arrays of a ``BodyLayout`` in a single pass.

    Args:
        block_flags: 1D uint8 array of per-block flags.
        block_sizes: 1D int64 array of per-block sample counts.
        out_sample_offsets: Preallocated 1D int64 array of length ``num_blocks + 1``.
        out_plane_byte_offsets: Preallocated 1D int64 array of length ``num_blocks +
            1``.
        out_code_plane_offsets: Preallocated 1D int64 array of length ``num_blocks +
            1``.
        out_time_low_offsets: Preallocated 1D int64 array of length ``num_blocks + 1``.
        out_time_high_offsets: Preallocated 1D int64 array of length ``num_blocks + 1``.
    """
    # Walk all blocks once, accumulating sample counts and conditional plane-byte
    # counts.
    out_sample_offsets[0] = out_plane_byte_offsets[0] = out_code_plane_offsets[0] = 0
    out_time_low_offsets[0] = out_time_high_offsets[0] = 0
    for block_idx in range(block_sizes.shape[0]):
        num_samples = block_sizes[block_idx]
        num_plane_bytes = plane_byte_count(num_samples)
        flags = block_flags[block_idx]
        # Every block contributes its samples and residual plane bytes.
        out_sample_offsets[block_idx + 1] = out_sample_offsets[block_idx] + num_samples
        out_plane_byte_offsets[block_idx + 1] = (
            out_plane_byte_offsets[block_idx] + num_plane_bytes
        )
        # Optional plane fields advance only when the corresponding block flag is set.
        out_code_plane_offsets[block_idx + 1] = out_code_plane_offsets[block_idx] + (
            num_plane_bytes if flags & BLOCK_FLAG_SAMPLE_CODES else 0
        )
        out_time_low_offsets[block_idx + 1] = out_time_low_offsets[block_idx] + (
            num_plane_bytes if flags & BLOCK_FLAG_IRREGULAR_TIME else 0
        )
        out_time_high_offsets[block_idx + 1] = out_time_high_offsets[block_idx] + (
            num_plane_bytes if flags & BLOCK_FLAG_WIDE_TIME else 0
        )


def compute_body_layout(block_flags: np.ndarray, block_sizes: np.ndarray) -> BodyLayout:
    """Computes the ``BodyLayout`` for blocks with the given flags and sample counts.

    Args:
        block_flags: 1D uint8 array of per-block flags.
        block_sizes: 1D integer array of per-block sample counts.

    Returns:
        Populated ``BodyLayout`` with all five cumulative offset arrays.
    """
    num_blocks = block_sizes.shape[0]
    offsets = BodyLayout(
        *(np.empty(num_blocks + 1, np.int64) for _ in BodyLayout._fields)
    )
    _fill_body_layout(block_flags, block_sizes.astype(np.int64, copy=False), *offsets)
    return offsets


@njit(nogil=True, cache=True)
def _read_block_sizes(
    body: np.ndarray, num_blocks: int, out_block_sizes: np.ndarray
) -> None:
    """Reads the byte-planed uint16 ``block_sizes`` column from an uncompressed body.

    Args:
        body: 1D uint8 array of uncompressed body bytes.
        num_blocks: Number of blocks in the group.
        out_block_sizes: Preallocated 1D int64 array of length ``num_blocks``.
    """
    sizes_start = block_sizes_start(num_blocks)
    for block_idx in range(num_blocks):
        # Reassemble uint16 from the low-byte plane and high-byte plane.
        out_block_sizes[block_idx] = np.int64(body[sizes_start + block_idx]) | (
            np.int64(body[sizes_start + num_blocks + block_idx]) << 8
        )


def read_body_layout(
    body: np.ndarray, num_blocks: int
) -> tuple[np.ndarray, BodyLayout]:
    """Reads ``block_sizes`` and computes the ``BodyLayout`` of an uncompressed body.

    Args:
        body: 1D uint8 array of uncompressed body bytes.
        num_blocks: Number of blocks in the group.

    Returns:
        Tuple ``(block_sizes, layout)`` where ``block_sizes`` is a 1D int64 array of
        per-block sample counts and ``layout`` is the corresponding ``BodyLayout``.
    """
    block_sizes = np.empty(num_blocks, np.int64)
    _read_block_sizes(body, num_blocks, block_sizes)
    offsets = BodyLayout(
        *(np.empty(num_blocks + 1, np.int64) for _ in BodyLayout._fields)
    )
    _fill_body_layout(body[:num_blocks], block_sizes, *offsets)
    return block_sizes, offsets


def body_byte_size(num_blocks: int, offsets: BodyLayout, has_time: bool) -> int:
    """Computes the byte length of an uncompressed body excluding optional trailers.

    Does not include the ``escaped_values`` trailer or the ``group_time_step`` trailer.

    Args:
        num_blocks: Number of blocks in the group.
        offsets: The group's ``BodyLayout``.
        has_time: Whether the group includes time columns and time residual planes.

    Returns:
        Size of the core uncompressed body in bytes.
    """
    # Sum value metadata, residual planes, sample code planes, and optional time fields.
    size = num_blocks * METADATA_BYTES_PER_BLOCK
    size += (
        int(offsets.plane_byte_offsets[-1])
        * SAMPLES_PER_PLANE_BYTE
        * BYTES_PER_RESIDUAL_SAMPLE
    )
    size += int(offsets.code_plane_offsets[-1]) * BITS_PER_SAMPLE_CODE
    if has_time:
        size += num_blocks * TIME_BYTES_PER_BLOCK
        size += (
            int(offsets.time_low_offsets[-1]) + int(offsets.time_high_offsets[-1])
        ) * TIME_LOW_PLANES
    return size


def body_size_bounds(
    num_blocks: int, num_samples: int, has_time: bool
) -> tuple[int, int]:
    """Computes tight lower and upper bounds on the total uncompressed body size.

    Used by decoders to validate the zstd content size before allocating or
    decompressing.

    Args:
        num_blocks: Number of blocks in the group.
        num_samples: Total number of samples across all blocks.
        has_time: Whether the group has a time axis.

    Returns:
        Tuple ``(min_bytes, max_bytes)`` bounding the valid uncompressed body length,
        including any escaped values and optional ``group_time_step`` trailer.
    """
    # Minimum: all block lengths are multiples of 8 with no sample codes, irregular
    # times, or escapes. Maximum: every block incurs 7 padding samples, has sample codes
    # + 64-plane times, and every sample is escaped, plus an 8-byte group_time_step
    # trailer when timed.
    columns = num_blocks * (
        METADATA_BYTES_PER_BLOCK + (TIME_BYTES_PER_BLOCK if has_time else 0)
    )
    min_plane_bytes = plane_byte_count(num_samples)
    max_plane_bytes = min(
        num_samples,
        (num_samples + (SAMPLES_PER_PLANE_BYTE - 1) * num_blocks)
        // SAMPLES_PER_PLANE_BYTE,
    )
    max_bytes_per_plane_byte = (
        SAMPLES_PER_PLANE_BYTE * BYTES_PER_RESIDUAL_SAMPLE + BITS_PER_SAMPLE_CODE
    )
    if has_time:
        max_bytes_per_plane_byte += 2 * TIME_LOW_PLANES
    min_size = (
        columns + min_plane_bytes * SAMPLES_PER_PLANE_BYTE * BYTES_PER_RESIDUAL_SAMPLE
    )
    max_size = (
        columns
        + max_plane_bytes * max_bytes_per_plane_byte
        + BYTES_PER_ESCAPED_VALUE * num_samples
        + GROUP_TIME_STEP_BYTES * has_time
    )
    return min_size, max_size


class GroupHeader(NamedTuple):
    """Parsed fields of an 8-byte block group header.

    Attributes:
        num_blocks: Number of blocks in the group.
        num_samples: Total sample count across all blocks.
        byte_planes: True if residual planes are stored as 2 byte planes; False for 16
            bit planes.
        time_unit: ``TimeUnitCode`` integer (0 when the group has no time axis).
        group_time_step: True if time_starts increases and time_steps are divided by a
            ``group_time_step`` trailer in the last 8 bytes of the body.
    """

    num_blocks: int
    num_samples: int
    byte_planes: bool
    time_unit: int
    group_time_step: bool = False


def pack_header(
    num_blocks: int,
    num_samples: int,
    byte_planes: bool,
    time_unit: int,
    group_time_step: bool = False,
) -> bytes:
    """Encodes the 8-byte block group header.

    Args:
        num_blocks: Number of blocks in the group.
        num_samples: Total sample count across all blocks.
        byte_planes: Whether residuals are stored as 2 byte planes (flag bit 0).
        time_unit: ``TimeUnitCode`` integer (0 for no time axis).
        group_time_step: Whether a ``group_time_step`` trailer is present (flag bit 4).

    Returns:
        8-byte little-endian header.
    """
    flags = (
        (GROUP_FLAG_BYTE_PLANES if byte_planes else 0)
        | (time_unit << GROUP_FLAG_TIME_UNIT_SHIFT)
        | (GROUP_FLAG_GROUP_TIME_STEP if group_time_step else 0)
    )
    return GROUP_HEADER.pack(FORMAT_VERSION, flags, num_blocks, num_samples)


def unpack_header(group: bytes) -> GroupHeader:
    """Parses and validates the 8-byte header of a compressed block group.

    Args:
        group: Compressed block group bytes (8-byte header followed by a zstd frame).

    Returns:
        Parsed ``GroupHeader``.

    Raises:
        ValueError: If the header is truncated, uses an unsupported version, sets
            reserved flag bits, specifies an unknown time unit, sets ``group_time_step``
            without a time axis, or declares an impossible sample count.
    """
    # Check length, unpack the fixed header struct, then validate version, flags, and
    # counts.
    if len(group) < HEADER_BYTES:
        raise ValueError(
            f"block group of {len(group)} bytes is shorter than its "
            f"{HEADER_BYTES}-byte header"
        )
    version, flags, num_blocks, num_samples = GROUP_HEADER.unpack_from(group)
    if version != FORMAT_VERSION:
        raise ValueError(
            f"block group format version {version} is not supported "
            f"(expected {FORMAT_VERSION})"
        )
    if flags & GROUP_FLAG_RESERVED:
        raise ValueError(
            "block group header sets reserved bits (not supported by this version)"
        )
    time_unit = (flags & GROUP_FLAG_TIME_UNIT_MASK) >> GROUP_FLAG_TIME_UNIT_SHIFT
    if time_unit > TimeUnitCode.NANOSECONDS:
        raise ValueError(
            f"block group header time unit {time_unit} is not supported "
            f"(expected 0 to {int(TimeUnitCode.NANOSECONDS)})"
        )
    if num_samples > min(MAX_GROUP_SAMPLES, num_blocks * MAX_BLOCK_LEN):
        raise ValueError(
            f"block group sample count {num_samples} is over {MAX_GROUP_SAMPLES} "
            f"or what {num_blocks} blocks can hold"
        )
    has_group_time_step = bool(flags & GROUP_FLAG_GROUP_TIME_STEP)
    if has_group_time_step and not time_unit:
        raise ValueError("block group header sets group_time_step without a time axis")
    return GroupHeader(
        num_blocks,
        num_samples,
        bool(flags & GROUP_FLAG_BYTE_PLANES),
        time_unit,
        has_group_time_step,
    )


# Column and plane field byte-offset helpers within an uncompressed body.


@njit(inline="always")
def block_sizes_start(num_blocks: int) -> int:
    """Returns the starting byte offset of ``block_sizes`` in an uncompressed body."""
    return BYTES_PER_FLAGS * num_blocks


@njit(inline="always")
def grid_exponents_start(num_blocks: int) -> int:
    """Returns the starting byte offset of ``grid_exponents`` in an unpacked body."""
    return (BYTES_PER_FLAGS + BYTES_PER_SIZE) * num_blocks


@njit(inline="always")
def grid_scales_start(num_blocks: int) -> int:
    """Returns the starting byte offset of ``grid_scales`` in an uncompressed body."""
    return (BYTES_PER_FLAGS + BYTES_PER_SIZE + BYTES_PER_EXPONENT) * num_blocks


@njit(inline="always")
def block_anchors_start(num_blocks: int) -> int:
    """Returns the starting byte offset of ``block_anchors`` in an uncompressed body."""
    return (
        BYTES_PER_FLAGS + BYTES_PER_SIZE + BYTES_PER_EXPONENT + BYTES_PER_SCALE
    ) * num_blocks


@njit(inline="always")
def time_starts_start(num_blocks: int) -> int:
    """Returns the starting byte offset of ``time_starts`` in an uncompressed body."""
    return METADATA_BYTES_PER_BLOCK * num_blocks


@njit(inline="always")
def time_steps_start(num_blocks: int) -> int:
    """Returns the starting byte offset of ``time_steps`` in an uncompressed body."""
    return (METADATA_BYTES_PER_BLOCK + BYTES_PER_TIME_COLUMN) * num_blocks


@njit(inline="always")
def time_ref_quotients_start(num_blocks: int) -> int:
    """Returns the starting byte offset of ``time_ref_quotients`` in an uncompressed
    body.
    """
    return (METADATA_BYTES_PER_BLOCK + 2 * BYTES_PER_TIME_COLUMN) * num_blocks


@njit(inline="always")
def residual_planes_start(num_blocks: int, has_time: bool) -> int:
    """Returns the starting byte offset of ``residual_planes`` in an uncompressed body.

    Args:
        num_blocks: Number of blocks in the group.
        has_time: Whether the group includes the three time metadata columns.

    Returns:
        Byte offset immediately following the per-block metadata columns.
    """
    return (
        METADATA_BYTES_PER_BLOCK + (TIME_BYTES_PER_BLOCK if has_time else 0)
    ) * num_blocks


@njit(inline="always")
def sample_code_planes_start(
    num_blocks: int, num_plane_bytes: int, has_time: bool
) -> int:
    """Returns the starting byte offset of ``sample_code_planes`` in an uncompressed
    body.

    Args:
        num_blocks: Number of blocks in the group.
        num_plane_bytes: Total 8-sample plane bytes across all blocks.
        has_time: Whether the group includes the three time metadata columns.

    Returns:
        Byte offset immediately following the residual planes.
    """
    return (
        residual_planes_start(num_blocks, has_time)
        + BYTES_PER_RESIDUAL_SAMPLE * SAMPLES_PER_PLANE_BYTE * num_plane_bytes
    )


# Byte-planed metadata column readers and writers.


@njit(inline="always")
def put_int64(
    body: np.ndarray,
    field_start: int,
    num_blocks: int,
    block_idx: int,
    value: int | np.integer,
) -> None:
    """Writes a block's int64 value into a byte-planed metadata column.

    Args:
        body: 1D uint8 array of uncompressed body bytes.
        field_start: Byte offset where the column begins.
        num_blocks: Number of blocks in the group (stride between byte planes).
        block_idx: Zero-based block index.
        value: Signed 64-bit integer to write.
    """
    value_bits = np.uint64(value)
    for byte_idx in range(8):
        body[field_start + byte_idx * num_blocks + block_idx] = np.uint8(
            (value_bits >> np.uint64(8 * byte_idx)) & np.uint64(0xFF)
        )


@njit(inline="always")
def get_int64(
    body: np.ndarray, field_start: int, num_blocks: int, block_idx: int
) -> np.int64:
    """Reads a block's int64 value from a byte-planed metadata column.

    Args:
        body: 1D uint8 array of uncompressed body bytes.
        field_start: Byte offset where the column begins.
        num_blocks: Number of blocks in the group (stride between byte planes).
        block_idx: Zero-based block index.

    Returns:
        Signed 64-bit integer value.
    """
    value_bits = np.uint64(0)
    for byte_idx in range(8):
        value_bits |= np.uint64(
            body[field_start + byte_idx * num_blocks + block_idx]
        ) << np.uint64(8 * byte_idx)
    return np.int64(value_bits)


@njit(inline="always")
def put_uint32(
    body: np.ndarray,
    field_start: int,
    num_blocks: int,
    block_idx: int,
    value: int | np.integer,
) -> None:
    """Writes a block's uint32 value into a byte-planed metadata column.

    Args:
        body: 1D uint8 array of uncompressed body bytes.
        field_start: Byte offset where the column begins.
        num_blocks: Number of blocks in the group (stride between byte planes).
        block_idx: Zero-based block index.
        value: Unsigned 32-bit integer to write.
    """
    for byte_idx in range(4):
        body[field_start + byte_idx * num_blocks + block_idx] = np.uint8(
            (value >> (8 * byte_idx)) & 0xFF
        )


@njit(inline="always")
def get_uint32(
    body: np.ndarray, field_start: int, num_blocks: int, block_idx: int
) -> np.int64:
    """Reads a block's uint32 value from a byte-planed metadata column.

    Args:
        body: 1D uint8 array of uncompressed body bytes.
        field_start: Byte offset where the column begins.
        num_blocks: Number of blocks in the group (stride between byte planes).
        block_idx: Zero-based block index.

    Returns:
        Unsigned 32-bit value widened to int64.
    """
    value = np.int64(0)
    for byte_idx in range(4):
        value |= np.int64(body[field_start + byte_idx * num_blocks + block_idx]) << (
            8 * byte_idx
        )
    return value


@njit(inline="always")
def put_int16(
    body: np.ndarray,
    field_start: int,
    num_blocks: int,
    block_idx: int,
    value: int | np.integer,
) -> None:
    """Writes a block's int16 value into a byte-planed metadata column.

    Args:
        body: 1D uint8 array of uncompressed body bytes.
        field_start: Byte offset where the column begins.
        num_blocks: Number of blocks in the group (stride between byte planes).
        block_idx: Zero-based block index.
        value: Signed 16-bit integer to write.
    """
    body[field_start + block_idx] = np.uint8(value & 0xFF)
    body[field_start + num_blocks + block_idx] = np.uint8((value >> 8) & 0xFF)


@njit(inline="always")
def get_int16(
    body: np.ndarray, field_start: int, num_blocks: int, block_idx: int
) -> np.int64:
    """Reads a block's int16 value from a byte-planed metadata column, sign-extended to
    int64.

    Args:
        body: 1D uint8 array of uncompressed body bytes.
        field_start: Byte offset where the column begins.
        num_blocks: Number of blocks in the group (stride between byte planes).
        block_idx: Zero-based block index.

    Returns:
        Sign-extended 64-bit integer value.
    """
    value_bits = np.int64(body[field_start + block_idx]) | (
        np.int64(body[field_start + num_blocks + block_idx]) << 8
    )
    # Sign-extend bit 15 without branching: subtracting 2^16 when bit 15 is set maps
    # [32768..65535] to [-32768..-1].
    return value_bits - ((value_bits & 0x8000) << 1)


# In-memory columnar representations of an encoded block group.


class EncodedTimeColumns(NamedTuple):
    """Per-block and per-sample encoded time columns for a block group.

    Attributes:
        starts: 1D int64 array of absolute block start timestamps (0 for empty blocks).
        steps: 1D int64 array of per-block time step GCDs.
        refs: 1D uint64 array of per-block reference quotients.
        residuals: 1D uint64 array of per-sample zigzagged quotient residuals (only
            populated for blocks flagged ``BLOCK_FLAG_IRREGULAR_TIME``).
    """

    starts: np.ndarray
    steps: np.ndarray
    refs: np.ndarray
    residuals: np.ndarray


def allocate_time_columns(num_blocks: int, num_samples: int) -> EncodedTimeColumns:
    """Allocates uninitialized arrays for ``EncodedTimeColumns``.

    Args:
        num_blocks: Number of blocks.
        num_samples: Total number of samples across all blocks.

    Returns:
        ``EncodedTimeColumns`` with pre-sized arrays.
    """
    return EncodedTimeColumns(
        np.empty(num_blocks, np.int64),
        np.empty(num_blocks, np.int64),
        np.empty(num_blocks, np.uint64),
        np.empty(num_samples, np.uint64),
    )


class EncodedBlockColumns(NamedTuple):
    """In-memory columnar representation of all encoded blocks in a block group.

    Attributes:
        block_flags: 1D uint8 array of per-block flags.
        block_sizes: 1D int64 array of per-block sample counts.
        grid_exponents: 1D int64 array of per-block quantization exponents.
        grid_scales: 1D int64 array of packed ``N | (D << 16)`` scales (0 if unscaled).
        block_anchors: 1D int64 array of per-block anchors (float64 bits or int64 grid
            index).
        residuals: 1D int16 array of per-sample difference residuals.
        sample_codes: 1D uint8 array of per-sample 2-bit exception codes (read only for
            blocks with ``BLOCK_FLAG_SAMPLE_CODES``).
        time_columns: ``EncodedTimeColumns``, or None for a group without a time axis.
        escaped_values: 1D float64 array of escaped sample values in block/sample order
            (one entry per ``CODE_ESCAPE``).
    """

    block_flags: np.ndarray
    block_sizes: np.ndarray
    grid_exponents: np.ndarray
    grid_scales: np.ndarray
    block_anchors: np.ndarray
    residuals: np.ndarray
    sample_codes: np.ndarray
    time_columns: EncodedTimeColumns | None
    escaped_values: np.ndarray = np.zeros(0)


@njit(inline="always")
def _gcd64(a: np.uint64, b: np.uint64) -> np.uint64:
    """Computes the greatest common divisor of two uint64 values (Euclid)."""
    while b:
        a, b = b, a % b
    return a


@njit(nogil=True, cache=True)
def group_time_step(body: np.ndarray, sample_offsets: np.ndarray) -> np.uint64:
    """Computes the group-wide GCD of stored time_starts increases and per-block
    time_steps.

    Args:
        body: 1D uint8 array of uncompressed body bytes (must have a time axis).
        sample_offsets: 1D int64 array of cumulative sample offsets (length ``num_blocks
            + 1``).

    Returns:
        Unsigned 64-bit GCD across all non-empty blocks' ``time_starts`` increases
        (excluding the first non-empty block's absolute start) and ``time_steps``, or 0
        if all are 0.
    """
    # Fold every non-empty block's start increase (after the first) and time step into a
    # running GCD. Exit early once the GCD reaches 1 since no common factor > 1 can
    # exist.
    num_blocks = sample_offsets.shape[0] - 1
    step = np.uint64(0)
    seen_nonempty = False
    for block_idx in range(num_blocks):
        if sample_offsets[block_idx + 1] == sample_offsets[block_idx]:
            continue
        if seen_nonempty:
            step = _gcd64(
                step,
                np.uint64(
                    get_int64(
                        body, time_starts_start(num_blocks), num_blocks, block_idx
                    )
                ),
            )
        seen_nonempty = True
        step = _gcd64(
            step,
            np.uint64(
                get_int64(body, time_steps_start(num_blocks), num_blocks, block_idx)
            ),
        )
        if step == 1:
            break
    return step


@njit(nogil=True, cache=True)
def scale_time_columns(
    body: np.ndarray, sample_offsets: np.ndarray, step: np.uint64, divide: bool
) -> bool:
    """Divides or multiplies a body's stored time_starts increases and time_steps by
    ``step`` in place.

    Args:
        body: 1D uint8 array of uncompressed body bytes (modified in place).
        sample_offsets: 1D int64 array of cumulative sample offsets (length ``num_blocks
            + 1``).
        step: Positive uint64 common factor (``group_time_step``).
        divide: True to divide (encoding), False to multiply back (decoding).

    Returns:
        True on success; False if multiplying back would overflow uint64 (corrupt
        input).
    """
    # Scale every non-empty block's time_step and (for blocks after the first)
    # time_start increase. The first non-empty block stores an absolute int64 timestamp
    # and is never scaled.
    num_blocks = sample_offsets.shape[0] - 1
    max_multiplicand = np.uint64(0xFFFFFFFFFFFFFFFF) // step
    seen_nonempty = False
    for block_idx in range(num_blocks):
        if sample_offsets[block_idx + 1] == sample_offsets[block_idx]:
            continue
        for field_start, is_scaled in (
            (time_starts_start(num_blocks), seen_nonempty),
            (time_steps_start(num_blocks), True),
        ):
            if not is_scaled:
                continue
            value = np.uint64(get_int64(body, field_start, num_blocks, block_idx))
            if divide:
                value = value // step
            else:
                if value > max_multiplicand:
                    return False
                value = value * step
            put_int64(body, field_start, num_blocks, block_idx, np.int64(value))
        seen_nonempty = True
    return True
