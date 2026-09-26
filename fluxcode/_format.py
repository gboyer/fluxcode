# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Binary layout and bit-shuffling routines for units.

A unit is a 16-byte header followed by one zstd frame holding the body. The
header (little-endian) is: format version (uint8, 1), flags (uint8, 0),
reserved (uint16, 0), block length n (uint32), sample count S (uint64). The unit
holds N = ceil(S / n) blocks; the last one is padded when S isn't a multiple of n.

The body, for N blocks of n samples with F of the blocks containing non-finite
values (NaN, +inf, -inf), consists of five consecutive fields:

    1. head (N bytes):
       Per-block flags: bits 0-1 encode the predictor difference order (0-3);
       bit 2 indicates decimal quantization; bit 3 indicates non-finite code
       planes; bits 4-7 are reserved and must be zero.
    2. param (8 * N bytes):
       Per-block int64 parameters (power-of-two exponent e or decimal power p),
       stored little-endian in byte-planed order (all byte 0s, all byte 1s, ...).
    3. anchor (8 * N bytes, byte-planed like param):
       Per-block reconstruction base: the bits of the float64 minimum lo on the
       power-of-two grid, or the int64 grid index K0 of the minimum on a decimal grid.
    4. residual (16 * N * n / 8 bytes):
       16 bit planes across all N blocks, storing zigzag-encoded int16
       differences mod 2^16.
    5. nonfinite (2 * F * n / 8 bytes, omitted if F == 0):
       2 code bit planes for the F flagged blocks, encoding sample categories
       (00: finite, 01: NaN, 10: +inf, 11: -inf).
"""

import enum
import struct

import numpy as np
from numba import njit

HEAD_ORDER: int = 0x03
"""Bitmask for the difference predictor order (bits 0-1 of the header byte)."""

HEAD_DECIMAL: int = 0x04
"""Bit flag indicating decimal quantization mode (bit 2 of the header byte)."""

HEAD_NONFINITE: int = 0x08
"""Bit flag indicating the presence of non-finite code planes (bit 3)."""

HEAD_RESERVED: int = 0xF0
"""Reserved bits in the header byte (bits 4-7); decoders must reject these."""

MAX_BLOCK_LEN: int = 65536
"""Maximum plausible block length used as a sanity bound for unit validation."""


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
"""Unit header: version, flags, reserved, block length n, sample count S."""

HEADER_BYTES: int = UNIT_HEADER.size
"""Size of the unit header in bytes (16)."""

MAX_UNIT_SAMPLES: int = 1 << 26
"""Largest sample count a unit may hold: a sanity bound (decoders reject larger headers before
decompressing), not a format limit."""

BYTES_PER_RESIDUAL_SAMPLE: int = 2
"""Number of residual difference bytes per sample (int16 mod 2^16)."""

NONFINITE_BITS_PER_SAMPLE: int = 2
"""Number of code bits per sample for flagged blocks (2-bit plane layout)."""


def unit_size(num_blocks: int, block_len: int, num_flagged_blocks: int = 0) -> int:
    """Calculates the size in bytes of a unit's uncompressed body.

    Args:
        num_blocks: Number of blocks in the unit.
        block_len: Number of samples per block (must be a multiple of 8).
        num_flagged_blocks: Number of blocks containing non-finite samples.

    Returns:
        Size of the uncompressed body in bytes.
    """
    metadata_bytes = num_blocks * METADATA_BYTES_PER_BLOCK
    residual_bytes = num_blocks * (block_len * BYTES_PER_RESIDUAL_SAMPLE)
    nonfinite_bytes = num_flagged_blocks * (block_len * NONFINITE_BITS_PER_SAMPLE // 8)
    return metadata_bytes + residual_bytes + nonfinite_bytes


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


def pack_header(block_len: int, num_samples: int) -> bytes:
    """Builds the 16-byte unit header.

    Args:
        block_len: Samples per block n.
        num_samples: Real sample count S of the unit.

    Returns:
        The header bytes.
    """
    return UNIT_HEADER.pack(FORMAT_VERSION, 0, 0, block_len, num_samples)


def unpack_header(unit: bytes) -> tuple[int, int, int]:
    """Parses and validates a unit header.

    Args:
        unit: Unit bytes (header followed by the zstd frame).

    Returns:
        A tuple of (block_len, num_samples, num_blocks).

    Raises:
        ValueError: If the header is truncated, has an unsupported version or
            nonzero flags, or describes an implausible block length or sample count.
    """
    if len(unit) < HEADER_BYTES:
        raise ValueError(f"unit of {len(unit)} bytes is shorter than its {HEADER_BYTES}-byte header")
    version, flags, reserved, block_len, num_samples = UNIT_HEADER.unpack_from(unit)
    if version != FORMAT_VERSION:
        raise ValueError(f"unit format version {version} is not supported (expected {FORMAT_VERSION})")
    if flags or reserved:
        raise ValueError("unit header sets reserved bits (not supported by this version)")
    if not (0 < block_len <= MAX_BLOCK_LEN and block_len % 8 == 0):
        raise ValueError(f"unit block length {block_len} is not a multiple of 8 from 8 to {MAX_BLOCK_LEN}")
    if not 0 < num_samples <= MAX_UNIT_SAMPLES:
        raise ValueError(f"unit sample count {num_samples} is outside 1..{MAX_UNIT_SAMPLES}")
    return block_len, num_samples, -(-num_samples // block_len)


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
def planes_view(raw_unit: np.ndarray, num_blocks: int, block_len: int) -> np.ndarray:
    """Provides a 2D view into the 16 residual bit planes of an uncompressed unit.

    Args:
        raw_unit: 1D uint8 array containing uncompressed unit data.
        num_blocks: Total number of blocks N.
        block_len: Block length n.

    Returns:
        2D uint8 array of shape (16, N * (n // 8)) representing bit planes.
    """
    # Offset starts after N head bytes, 8*N parameter bytes and 8*N anchor bytes
    start_offset = METADATA_BYTES_PER_BLOCK * num_blocks
    end_offset = num_blocks * (METADATA_BYTES_PER_BLOCK + BYTES_PER_RESIDUAL_SAMPLE * block_len)
    return raw_unit[start_offset:end_offset].reshape(16, num_blocks * (block_len // 8))


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
        residuals: 1D int16 array of residual differences for the block.
        bit_planes: 2D uint8 array of shape (16, N * (n // 8)) holding bit planes.
        block_idx: Zero-based block index within the unit.
        scratch_low_bytes: Preallocated uint8 scratch array for lower bytes.
        scratch_high_bytes: Preallocated uint8 scratch array for upper bytes.
    """
    num_samples = residuals.shape[0]
    for sample_idx in range(num_samples):
        signed_residual = residuals[sample_idx]
        # Zigzag transform: maps 0->0, -1->1, 1->2, -2->3 to keep small magnitudes small
        zigzag_val = np.uint16((signed_residual << np.int16(1)) ^ (signed_residual >> np.int16(15)))
        # Split into lower 8 bits and upper 8 bits
        scratch_low_bytes[sample_idx] = np.uint8(zigzag_val)
        scratch_high_bytes[sample_idx] = np.uint8(zigzag_val >> np.uint16(8))
    # Reinterpret byte scratch buffers as 64-bit words for 8-byte group transposition
    low_words64 = scratch_low_bytes.view(np.uint64)
    high_words64 = scratch_high_bytes.view(np.uint64)
    num_8byte_groups = num_samples // 8
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
        bit_planes: 2D uint8 array of shape (16, N * (n // 8)) holding bit planes.
        block_idx: Zero-based block index within the unit.
        out_low_bytes: Output uint8 array of length n receiving lower zigzag bytes.
        out_high_bytes: Output uint8 array of length n receiving upper zigzag bytes.
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
def param_start(num_blocks: int) -> int:
    """Offset of the param field: after the N head bytes."""
    return num_blocks


@njit(inline="always")
def anchor_start(num_blocks: int) -> int:
    """Offset of the anchor field: after the head and param fields."""
    return (BYTES_PER_HEADER + BYTES_PER_PARAM) * num_blocks


@njit(inline="always")
def put_int64(raw_unit: np.ndarray, field_start: int, num_blocks: int, block_idx: int, value: int) -> None:
    """Writes block b's int64 value into a byte-planed field.

    Distributes the 8 bytes of the value across 8 strides of length num_blocks.

    Args:
        raw_unit: 1D uint8 array of uncompressed body bytes.
        field_start: Offset of the field (param_start or anchor_start).
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
        field_start: Offset of the field (param_start or anchor_start).
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
def code_planes_view(
    raw_unit: np.ndarray, num_blocks: int, block_len: int, num_flagged_blocks: int
) -> np.ndarray:
    """Provides a 2D view into the non-finite code planes of an uncompressed unit.

    Args:
        raw_unit: 1D uint8 array containing uncompressed unit data.
        num_blocks: Total number of blocks N.
        block_len: Block length n.
        num_flagged_blocks: Number of blocks carrying non-finite codes.

    Returns:
        2D uint8 array of shape (2, num_flagged_blocks * (block_len // 8)).
    """
    # Starts after head, param, anchor, and 16 residual bit planes
    start_offset = num_blocks * (METADATA_BYTES_PER_BLOCK + BYTES_PER_RESIDUAL_SAMPLE * block_len)
    total_code_bytes = num_flagged_blocks * (block_len * NONFINITE_BITS_PER_SAMPLE // 8)
    return raw_unit[start_offset:start_offset + total_code_bytes].reshape(2, num_flagged_blocks * (block_len // 8))


@njit(nogil=True, cache=True)
def put_codes(sample_codes: np.ndarray, code_planes: np.ndarray, flagged_block_idx: int) -> None:
    """Packs 2-bit non-finite sample codes for a block into two code bit planes.

    Args:
        sample_codes: 1D uint8 array of n codes (values in 0..3).
        code_planes: 2D uint8 array of shape (2, F * (n // 8)) holding code planes.
        flagged_block_idx: Rank of the flagged block among all flagged blocks (0 <= f < F).
    """
    num_8byte_groups = sample_codes.shape[0] // 8
    plane_byte_offset = flagged_block_idx * num_8byte_groups
    for group_idx in range(num_8byte_groups):
        plane0_byte = 0
        plane1_byte = 0
        for bit_idx in range(8):
            code_val = sample_codes[8 * group_idx + bit_idx]
            # Bit 0 of code goes to plane 0; bit 1 goes to plane 1
            plane0_byte |= (code_val & 1) << bit_idx
            plane1_byte |= ((code_val >> 1) & 1) << bit_idx
        code_planes[0, plane_byte_offset + group_idx] = plane0_byte
        code_planes[1, plane_byte_offset + group_idx] = plane1_byte


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


@njit(nogil=True, cache=True)
def get_codes(code_planes: np.ndarray, flagged_block_idx: int, out_codes: np.ndarray) -> None:
    """Unpacks 2-bit non-finite sample codes for a flagged block.

    Args:
        code_planes: 2D uint8 array of shape (2, F * (n // 8)) holding code planes.
        flagged_block_idx: Rank of the flagged block among all flagged blocks (0 <= f < F).
        out_codes: Output 1D uint8 array of length n receiving unpacked codes.
    """
    num_8byte_groups = out_codes.shape[0] // 8
    plane_byte_offset = flagged_block_idx * num_8byte_groups
    for group_idx in range(num_8byte_groups):
        plane0_byte = code_planes[0, plane_byte_offset + group_idx]
        plane1_byte = code_planes[1, plane_byte_offset + group_idx]
        for bit_idx in range(8):
            # Extract 2-bit code for each sample in the 8-sample group
            out_codes[8 * group_idx + bit_idx] = get_code(plane0_byte, plane1_byte, bit_idx)


@njit(nogil=True, cache=True)
def write_rows(
    headers: np.ndarray,
    parameters: np.ndarray,
    anchors: np.ndarray,
    residuals: np.ndarray,
    codes: np.ndarray,
    out_raw_unit: np.ndarray,
) -> None:
    """Serializes unit components into a preallocated uncompressed body buffer.

    Args:
        headers: 1D uint8 array of N block header bytes.
        parameters: 1D int64 array of N block parameters.
        anchors: 1D int64 array of N block anchors (float64 bits or decimal grid index).
        residuals: 2D int16 array of shape (N, n) holding difference residuals.
        codes: 2D uint8 array of shape (N, n) holding sample codes (only rows of
            flagged blocks are read; may be empty if no block is flagged).
        out_raw_unit: Output 1D uint8 array of length unit_size(N, n, F).
    """
    num_blocks, block_len = residuals.shape
    scratch_low_bytes = np.empty(block_len, np.uint8)
    scratch_high_bytes = np.empty(block_len, np.uint8)
    # Obtain views into residual and non-finite plane regions
    planes = planes_view(out_raw_unit, num_blocks, block_len)
    cplanes = code_planes_view(out_raw_unit, num_blocks, block_len, count_flagged(headers, num_blocks))
    flagged_counter = 0
    for block_idx in range(num_blocks):
        # Store 1-byte header
        out_raw_unit[block_idx] = headers[block_idx]
        # Store 8-byte parameter and anchor in byte-planed layout
        put_int64(out_raw_unit, param_start(num_blocks), num_blocks, block_idx, parameters[block_idx])
        put_int64(out_raw_unit, anchor_start(num_blocks), num_blocks, block_idx, anchors[block_idx])
        # Zigzag, transpose, and store residual bit planes
        shuffle_block(residuals[block_idx], planes, block_idx, scratch_low_bytes, scratch_high_bytes)
        if headers[block_idx] & HEAD_NONFINITE:
            # Store 2-bit code planes for flagged blocks
            put_codes(codes[block_idx], cplanes, flagged_counter)
            flagged_counter += 1


@njit(nogil=True, cache=True)
def read_rows(
    raw_unit: np.ndarray,
    out_headers: np.ndarray,
    out_parameters: np.ndarray,
    out_anchors: np.ndarray,
    out_residuals: np.ndarray,
    out_codes: np.ndarray,
) -> None:
    """Deserializes unit components from an uncompressed body buffer.

    Args:
        raw_unit: 1D uint8 array containing uncompressed body bytes.
        out_headers: Output 1D uint8 array of length N receiving header bytes.
        out_parameters: Output 1D int64 array of length N receiving parameters.
        out_anchors: Output 1D int64 array of length N receiving anchors.
        out_residuals: Output 2D int16 array of shape (N, n) receiving residuals.
        out_codes: Output 2D uint8 array of shape (N, n) receiving codes for flagged
            blocks (unflagged block rows are left unmodified).
    """
    num_blocks, block_len = out_residuals.shape
    scratch_low_bytes = np.empty(block_len, np.uint8)
    scratch_high_bytes = np.empty(block_len, np.uint8)
    # Access views into bit plane sections
    planes = planes_view(raw_unit, num_blocks, block_len)
    cplanes = code_planes_view(raw_unit, num_blocks, block_len, count_flagged(raw_unit, num_blocks))
    flagged_counter = 0
    for block_idx in range(num_blocks):
        # Read header byte
        out_headers[block_idx] = raw_unit[block_idx]
        # Read byte-planed parameter and anchor
        out_parameters[block_idx] = get_int64(raw_unit, param_start(num_blocks), num_blocks, block_idx)
        out_anchors[block_idx] = get_int64(raw_unit, anchor_start(num_blocks), num_blocks, block_idx)
        # Unshuffle bit planes into zigzag low/high bytes
        unshuffle_block(planes, block_idx, scratch_low_bytes, scratch_high_bytes)
        for sample_idx in range(block_len):
            # Reverse zigzag mapping to recover signed residual differences
            out_residuals[block_idx, sample_idx] = unzigzag16(scratch_low_bytes, scratch_high_bytes, sample_idx)
        if out_headers[block_idx] & HEAD_NONFINITE:
            # Unpack non-finite codes for flagged blocks
            get_codes(cplanes, flagged_counter, out_codes[block_idx])
            flagged_counter += 1


def write_unit(
    headers: np.ndarray,
    parameters: np.ndarray,
    anchors: np.ndarray,
    residuals: np.ndarray,
    codes: np.ndarray | None = None,
) -> np.ndarray:
    """Serializes unit components into a newly allocated uncompressed body.

    Args:
        headers: 1D uint8 array of N block header bytes.
        parameters: 1D int64 array of N block parameters.
        anchors: 1D int64 array of N block anchors.
        residuals: 2D int16 array of shape (N, n) holding difference residuals.
        codes: Optional 2D uint8 array of shape (N, n) holding sample codes.
            Required if any block has the HEAD_NONFINITE flag set.

    Returns:
        1D uint8 array containing the serialized uncompressed body.

    Raises:
        ValueError: If any block is flagged non-finite but codes are omitted.
    """
    num_blocks, block_len = residuals.shape
    # Count flagged blocks to size non-finite code planes
    num_flagged_blocks = int(np.count_nonzero(headers & HEAD_NONFINITE))
    if codes is None:
        if num_flagged_blocks:
            raise ValueError("flagged blocks need codes")
        codes = np.zeros((0, block_len), np.uint8)
    raw_unit = np.empty(unit_size(num_blocks, block_len, num_flagged_blocks), np.uint8)
    write_rows(headers, parameters, anchors, residuals, codes, raw_unit)
    return raw_unit


def read_unit(
    raw_unit: bytes | np.ndarray, num_blocks: int, block_len: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deserializes an uncompressed body into constituent arrays.

    Args:
        raw_unit: Byte buffer or uint8 array containing the uncompressed body.
        num_blocks: Number of blocks N in the unit.
        block_len: Block length n.

    Returns:
        A tuple of (headers, parameters, anchors, residuals, codes):
            headers: 1D uint8 array of N block header bytes.
            parameters: 1D int64 array of N block parameters.
            anchors: 1D int64 array of N block anchors.
            residuals: 2D int16 array of shape (N, n) holding residuals.
            codes: 2D uint8 array of shape (N, n) holding sample codes (all zeros
                for unflagged blocks).

    Raises:
        ValueError: If the buffer size does not match N blocks of n samples.
    """
    raw_arr = np.frombuffer(raw_unit, np.uint8) if not isinstance(raw_unit, np.ndarray) else raw_unit
    # count_flagged reads the N head bytes, so check they exist first
    if raw_arr.shape[0] < num_blocks or raw_arr.shape[0] != unit_size(
        num_blocks, block_len, count_flagged(raw_arr, num_blocks)
    ):
        raise ValueError(f"a body of {raw_arr.shape[0]} bytes doesn't hold {num_blocks} blocks of {block_len}")
    headers = np.empty(num_blocks, np.uint8)
    parameters = np.empty(num_blocks, np.int64)
    anchors = np.empty(num_blocks, np.int64)
    residuals = np.empty((num_blocks, block_len), np.int16)
    codes = np.zeros((num_blocks, block_len), np.uint8)
    read_rows(raw_arr, headers, parameters, anchors, residuals, codes)
    return headers, parameters, anchors, residuals, codes
