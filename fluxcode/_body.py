# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Uncompressed body serialization, time-field reading/validation, and binary body splicing.

Builds uncompressed byte-plane bodies from in-memory ``EncodedBlockColumns``, deserializes
and validates time columns and time residual planes, and splices updated or appended blocks
into an existing body by copying untouched blocks verbatim at the byte level.
"""

import numpy as np
from numba import njit

from ._bitpacking import (
    pack_sample_codes,
    residual_byte_planes_view,
    sample_code_planes_view,
    shuffle_time_residuals,
    time_residual_planes_views,
    unshuffle_time_residuals,
    zigzag_encode_block,
)
from ._format import (
    BLOCK_FLAG_IRREGULAR_TIME,
    BLOCK_FLAG_SAMPLE_CODES,
    BLOCK_FLAG_WIDE_TIME,
    BYTES_PER_ANCHOR,
    BYTES_PER_EXPONENT,
    BYTES_PER_SCALE,
    BYTES_PER_SIZE,
    SAMPLES_PER_PLANE_BYTE,
    TIME_BYTES_PER_BLOCK,
    TIME_LOW_PLANES,
    BodyLayout,
    EncodedBlockColumns,
    EncodedTimeColumns,
    allocate_time_columns,
    block_anchors_start,
    block_sizes_start,
    body_byte_size,
    compute_body_layout,
    get_int64,
    grid_exponents_start,
    grid_scales_start,
    plane_byte_count,
    put_int16,
    put_int64,
    put_uint32,
    time_ref_quotients_start,
    time_starts_start,
    time_steps_start,
)

# Validation status codes returned by read_time_columns / read_time_fields (plain ints for Numba).
TIME_FIELDS_OK: int = 0
"""Time columns and wide-time plane flags are valid."""

TIME_FIELDS_BAD_START: int = 1
"""A block start time is int64 minimum (NaT) or overflows int64 when accumulated."""

TIME_FIELDS_BAD_WIDE: int = 2
"""A wide-time block has all-zero upper planes 32-63 (it should not set BLOCK_FLAG_WIDE_TIME)."""

TIME_FIELDS_BAD_EMPTY: int = 3
"""An empty block has a non-zero time_start, time_step, or time_ref_quotient."""

TIME_FIELDS_BAD_SINGLE: int = 4
"""A single-sample block has a non-zero time_step or time_ref_quotient, or is flagged irregular."""


@njit(nogil=True, cache=True)
def write_value_fields(
    block_flags: np.ndarray,
    block_sizes: np.ndarray,
    grid_exponents: np.ndarray,
    grid_scales: np.ndarray,
    block_anchors: np.ndarray,
    residuals: np.ndarray,
    sample_codes: np.ndarray,
    sample_offsets: np.ndarray,
    plane_byte_offsets: np.ndarray,
    code_plane_offsets: np.ndarray,
    has_time: bool,
    out_body: np.ndarray,
) -> None:
    """Serializes all value metadata columns, residual byte planes, and sample code planes.

    Time fields (when ``has_time`` is True) are written separately by ``write_time_fields``.

    Args:
        block_flags: 1D uint8 array of per-block flags.
        block_sizes: 1D int64 array of per-block sample counts.
        grid_exponents: 1D int64 array of per-block quantization exponents.
        grid_scales: 1D int64 array of per-block packed scales (``N | (D << 16)``).
        block_anchors: 1D int64 array of per-block anchors (float64 bits or int64 grid index).
        residuals: 1D int16 array of all samples' difference residuals.
        sample_codes: 1D uint8 array of all samples' 2-bit codes (read only for flagged blocks).
        sample_offsets: ``BodyLayout.sample_offsets``.
        plane_byte_offsets: ``BodyLayout.plane_byte_offsets``.
        code_plane_offsets: ``BodyLayout.code_plane_offsets``.
        has_time: Whether the body reserves space for time columns before the residuals.
        out_body: Preallocated 1D uint8 array of length ``body_byte_size(...)``.
    """
    # Iterate block by block: write the five byte-planed metadata columns, zigzag-encode
    # residuals directly into the 2 byte planes with trailing zero padding, and pack 2-bit
    # sample codes whenever BLOCK_FLAG_SAMPLE_CODES is set.
    num_blocks = block_flags.shape[0]
    num_plane_bytes = int(plane_byte_offsets[num_blocks])
    byte_planes = residual_byte_planes_view(out_body, num_blocks, num_plane_bytes, has_time)
    code_planes = sample_code_planes_view(
        out_body, num_blocks, num_plane_bytes, int(code_plane_offsets[num_blocks]), has_time
    )
    for block_idx in range(num_blocks):
        out_body[block_idx] = block_flags[block_idx]
        block_len = block_sizes[block_idx]
        # Write uint16 block size into its two byte planes.
        out_body[block_sizes_start(num_blocks) + block_idx] = np.uint8(block_len & 0xFF)
        out_body[block_sizes_start(num_blocks) + num_blocks + block_idx] = np.uint8(block_len >> 8)
        # Write byte-planed exponent (int16), scale (uint32), and anchor (int64).
        put_int16(out_body, grid_exponents_start(num_blocks), num_blocks, block_idx, grid_exponents[block_idx])
        put_uint32(out_body, grid_scales_start(num_blocks), num_blocks, block_idx, grid_scales[block_idx])
        put_int64(out_body, block_anchors_start(num_blocks), num_blocks, block_idx, block_anchors[block_idx])
        # Zigzag residuals directly into the low and high byte planes and zero-pad to a plane byte boundary.
        block_residuals = residuals[sample_offsets[block_idx] : sample_offsets[block_idx + 1]]
        sample_start = SAMPLES_PER_PLANE_BYTE * plane_byte_offsets[block_idx]
        zigzag_encode_block(
            block_residuals,
            byte_planes[0, sample_start : sample_start + block_len],
            byte_planes[1, sample_start : sample_start + block_len],
        )
        byte_planes[:, sample_start + block_len : SAMPLES_PER_PLANE_BYTE * plane_byte_offsets[block_idx + 1]] = 0
        if block_flags[block_idx] & BLOCK_FLAG_SAMPLE_CODES:
            pack_sample_codes(
                sample_codes[sample_offsets[block_idx] : sample_offsets[block_idx + 1]],
                code_planes,
                code_plane_offsets[block_idx],
            )


@njit(nogil=True, cache=True)
def write_time_fields(
    block_flags: np.ndarray,
    sample_offsets: np.ndarray,
    plane_byte_offsets: np.ndarray,
    code_plane_offsets: np.ndarray,
    time_low_offsets: np.ndarray,
    time_high_offsets: np.ndarray,
    time_starts: np.ndarray,
    time_steps: np.ndarray,
    time_refs: np.ndarray,
    time_residuals: np.ndarray,
    out_body: np.ndarray,
) -> None:
    """Serializes ``time_starts``, ``time_steps``, ``time_ref_quotients``, and time residual planes.

    Args:
        block_flags: 1D uint8 array of per-block flags.
        sample_offsets: ``BodyLayout.sample_offsets``.
        plane_byte_offsets: ``BodyLayout.plane_byte_offsets``.
        code_plane_offsets: ``BodyLayout.code_plane_offsets``.
        time_low_offsets: ``BodyLayout.time_low_offsets``.
        time_high_offsets: ``BodyLayout.time_high_offsets``.
        time_starts: 1D int64 array of absolute block start times (ignored for empty blocks).
        time_steps: 1D int64 array of per-block time step GCDs.
        time_refs: 1D uint64 array of per-block reference quotients.
        time_residuals: 1D uint64 array of per-sample zigzagged time residuals (read only for
            irregular blocks).
        out_body: Preallocated 1D uint8 body array whose value fields are written by ``write_value_fields``.
    """
    # Zero all time residual planes in one contiguous pass so shuffle_time_residuals can skip
    # all-zero upper byte levels. Then delta-encode non-empty block starts and pack irregular blocks.
    num_blocks = block_flags.shape[0]
    time_low_planes, time_high_planes = time_residual_planes_views(
        out_body,
        num_blocks,
        int(plane_byte_offsets[num_blocks]),
        int(code_plane_offsets[num_blocks]),
        int(time_low_offsets[num_blocks]),
        int(time_high_offsets[num_blocks]),
    )
    scratch_tail = np.empty(SAMPLES_PER_PLANE_BYTE, np.uint64)
    time_low_planes[:, :] = 0
    time_high_planes[:, :] = 0
    previous_start = np.int64(0)
    seen_nonempty = False
    for block_idx in range(num_blocks):
        if sample_offsets[block_idx + 1] == sample_offsets[block_idx]:
            # Empty blocks carry no timestamps; all three time columns must be zero.
            put_int64(out_body, time_starts_start(num_blocks), num_blocks, block_idx, 0)
            put_int64(out_body, time_steps_start(num_blocks), num_blocks, block_idx, 0)
            put_int64(out_body, time_ref_quotients_start(num_blocks), num_blocks, block_idx, 0)
            continue
        # Store the first non-empty block's start verbatim; subsequent starts as uint64 increases.
        stored_start = time_starts[block_idx]
        if seen_nonempty:
            stored_start = np.int64(np.uint64(time_starts[block_idx]) - np.uint64(previous_start))
        previous_start = time_starts[block_idx]
        seen_nonempty = True
        put_int64(out_body, time_starts_start(num_blocks), num_blocks, block_idx, stored_start)
        put_int64(out_body, time_steps_start(num_blocks), num_blocks, block_idx, time_steps[block_idx])
        put_int64(out_body, time_ref_quotients_start(num_blocks), num_blocks, block_idx, np.int64(time_refs[block_idx]))
        if block_flags[block_idx] & BLOCK_FLAG_IRREGULAR_TIME:
            shuffle_time_residuals(
                time_residuals[sample_offsets[block_idx] : sample_offsets[block_idx + 1]],
                time_low_planes,
                time_high_planes,
                time_low_offsets[block_idx],
                time_high_offsets[block_idx],
                scratch_tail,
            )


def write_body(
    block_flags: np.ndarray,
    block_sizes: np.ndarray,
    grid_exponents: np.ndarray,
    grid_scales: np.ndarray,
    block_anchors: np.ndarray,
    residuals: np.ndarray,
    sample_codes: np.ndarray,
    time_columns: EncodedTimeColumns | None,
    offsets: BodyLayout,
) -> np.ndarray:
    """Allocates and serializes an uncompressed byte-plane body from encoded block columns.

    Args:
        block_flags: 1D uint8 array of per-block flags.
        block_sizes: 1D integer array of per-block sample counts.
        grid_exponents: 1D int64 array of per-block quantization exponents.
        grid_scales: 1D int64 array of per-block packed scales (``N | (D << 16)``).
        block_anchors: 1D int64 array of per-block anchors.
        residuals: 1D int16 array of all samples' difference residuals.
        sample_codes: 1D uint8 array of all samples' 2-bit codes (read only for flagged blocks).
        time_columns: ``EncodedTimeColumns``, or None for a group without a time axis.
        offsets: Precomputed ``BodyLayout`` for these blocks.

    Returns:
        Newly allocated 1D uint8 array holding the serialized uncompressed body (byte-plane
        residuals; excludes escaped-value and ``group_time_step`` trailers).

    Raises:
        ValueError: If any block is flagged irregular when ``time_columns`` is None.
    """
    num_blocks = block_flags.shape[0]
    block_sizes = np.asarray(block_sizes, np.int64)
    has_time = time_columns is not None
    if offsets.time_low_offsets[-1] and not has_time:
        raise ValueError("irregular time blocks require time columns")
    body = np.empty(body_byte_size(num_blocks, offsets, has_time), np.uint8)
    write_value_fields(
        block_flags,
        block_sizes,
        grid_exponents,
        grid_scales,
        block_anchors,
        residuals,
        sample_codes,
        offsets.sample_offsets,
        offsets.plane_byte_offsets,
        offsets.code_plane_offsets,
        has_time,
        body,
    )
    if time_columns is not None:
        write_time_fields(block_flags, *offsets, *time_columns, body)
    return body


@njit(nogil=True, cache=True)
def _high_time_planes_zero(time_high_planes: np.ndarray, time_high_offset: int, num_samples: int) -> bool:
    """Checks whether a wide-time block's upper bit planes (32-63) are all zero on valid samples.

    Args:
        time_high_planes: 2D uint8 array of bit planes 32-63.
        time_high_offset: Block's starting column in ``time_high_planes``.
        num_samples: Number of samples in the block (padding bits past this are ignored).

    Returns:
        True if every bit in planes 32-63 across the block's valid samples is zero.
    """
    num_plane_bytes = plane_byte_count(num_samples)
    remainder = num_samples % SAMPLES_PER_PLANE_BYTE
    tail_mask = np.uint8((1 << remainder) - 1) if remainder else np.uint8(0xFF)
    for plane_idx in range(TIME_LOW_PLANES):
        for group_idx in range(num_plane_bytes - 1):
            if time_high_planes[plane_idx, time_high_offset + group_idx]:
                return False
        if time_high_planes[plane_idx, time_high_offset + num_plane_bytes - 1] & tail_mask:
            return False
    return True


@njit(nogil=True, cache=True)
def read_time_columns(
    body: np.ndarray,
    sample_offsets: np.ndarray,
    plane_byte_offsets: np.ndarray,
    code_plane_offsets: np.ndarray,
    time_low_offsets: np.ndarray,
    time_high_offsets: np.ndarray,
    out_time_starts: np.ndarray,
    out_time_steps: np.ndarray,
    out_time_refs: np.ndarray,
) -> tuple[int, int]:
    """Deserializes and validates ``time_starts``, ``time_steps``, and ``time_ref_quotients``.

    Does not unpack the time residual planes (use ``read_time_residuals`` or ``read_time_fields``).

    Args:
        body: 1D uint8 array of uncompressed body bytes (must have a time axis).
        sample_offsets: ``BodyLayout.sample_offsets``.
        plane_byte_offsets: ``BodyLayout.plane_byte_offsets``.
        code_plane_offsets: ``BodyLayout.code_plane_offsets``.
        time_low_offsets: ``BodyLayout.time_low_offsets``.
        time_high_offsets: ``BodyLayout.time_high_offsets``.
        out_time_starts: Destination 1D int64 array for absolute block start times (0 if empty).
        out_time_steps: Destination 1D int64 array for per-block time step GCDs.
        out_time_refs: Destination 1D uint64 array for per-block reference quotients.

    Returns:
        Tuple ``(status, failing_block_idx)`` where ``status`` is ``TIME_FIELDS_OK`` (with index 0)
        or one of ``TIME_FIELDS_BAD_*`` with the first failing block index.
    """
    # Reconstruct absolute block start times by accumulating uint64 start increases, verifying
    # no int64 overflow, zero columns on empty blocks, canonical single-sample blocks, and
    # non-empty upper planes on wide-time blocks.
    num_blocks = out_time_starts.shape[0]
    _, time_high_planes = time_residual_planes_views(
        body,
        num_blocks,
        int(plane_byte_offsets[num_blocks]),
        int(code_plane_offsets[num_blocks]),
        int(time_low_offsets[num_blocks]),
        int(time_high_offsets[num_blocks]),
    )
    int64_max = np.uint64(0x7FFFFFFFFFFFFFFF)
    seen_nonempty = False
    previous_start = np.int64(0)
    for block_idx in range(num_blocks):
        stored_start = get_int64(body, time_starts_start(num_blocks), num_blocks, block_idx)
        out_time_steps[block_idx] = get_int64(body, time_steps_start(num_blocks), num_blocks, block_idx)
        out_time_refs[block_idx] = np.uint64(get_int64(body, time_ref_quotients_start(num_blocks), num_blocks, block_idx))
        block_len = sample_offsets[block_idx + 1] - sample_offsets[block_idx]
        if block_len == 0:
            if stored_start != 0 or out_time_steps[block_idx] != 0 or out_time_refs[block_idx] != 0:
                return TIME_FIELDS_BAD_EMPTY, block_idx
            out_time_starts[block_idx] = 0
            continue
        if not seen_nonempty:
            # Reject int64 minimum (NumPy NaT sentinel) as a start timestamp.
            if stored_start == np.int64(-0x8000000000000000):
                return TIME_FIELDS_BAD_START, block_idx
            out_time_starts[block_idx] = stored_start
        else:
            # Verify unsigned addition does not wrap past INT64_MAX.
            headroom = int64_max - np.uint64(previous_start)
            if np.uint64(stored_start) > headroom:
                return TIME_FIELDS_BAD_START, block_idx
            out_time_starts[block_idx] = np.int64(np.uint64(previous_start) + np.uint64(stored_start))
        previous_start = out_time_starts[block_idx]
        seen_nonempty = True
        if block_len == 1 and (
            out_time_steps[block_idx] != 0
            or out_time_refs[block_idx] != 0
            or body[block_idx] & BLOCK_FLAG_IRREGULAR_TIME
        ):
            return TIME_FIELDS_BAD_SINGLE, block_idx
        if body[block_idx] & BLOCK_FLAG_WIDE_TIME and _high_time_planes_zero(
            time_high_planes, time_high_offsets[block_idx], block_len
        ):
            return TIME_FIELDS_BAD_WIDE, block_idx
    return TIME_FIELDS_OK, 0


@njit(nogil=True, cache=True)
def read_time_residuals(
    body: np.ndarray,
    sample_offsets: np.ndarray,
    plane_byte_offsets: np.ndarray,
    code_plane_offsets: np.ndarray,
    time_low_offsets: np.ndarray,
    time_high_offsets: np.ndarray,
    block_ids: np.ndarray,
    out_time_residuals: np.ndarray,
) -> None:
    """Unpacks the zigzagged uint64 time residuals of the specified blocks.

    Only blocks with ``BLOCK_FLAG_IRREGULAR_TIME`` are unpacked; regular blocks are skipped.

    Args:
        body: 1D uint8 array of uncompressed body bytes (must have a time axis).
        sample_offsets: ``BodyLayout.sample_offsets``.
        plane_byte_offsets: ``BodyLayout.plane_byte_offsets``.
        code_plane_offsets: ``BodyLayout.code_plane_offsets``.
        time_low_offsets: ``BodyLayout.time_low_offsets``.
        time_high_offsets: ``BodyLayout.time_high_offsets``.
        block_ids: 1D int64 array of block indices to unpack.
        out_time_residuals: Destination 1D uint64 array sized for all samples in the group.
    """
    num_blocks = sample_offsets.shape[0] - 1
    time_low_planes, time_high_planes = time_residual_planes_views(
        body,
        num_blocks,
        int(plane_byte_offsets[num_blocks]),
        int(code_plane_offsets[num_blocks]),
        int(time_low_offsets[num_blocks]),
        int(time_high_offsets[num_blocks]),
    )
    scratch_tail = np.empty(SAMPLES_PER_PLANE_BYTE, np.uint64)
    for block_idx in block_ids:
        if body[block_idx] & BLOCK_FLAG_IRREGULAR_TIME:
            unshuffle_time_residuals(
                time_low_planes,
                time_high_planes,
                time_low_offsets[block_idx],
                time_high_offsets[block_idx] if body[block_idx] & BLOCK_FLAG_WIDE_TIME else -1,
                out_time_residuals[sample_offsets[block_idx] : sample_offsets[block_idx + 1]],
                scratch_tail,
            )


def read_time_fields(
    body: np.ndarray,
    sample_offsets: np.ndarray,
    plane_byte_offsets: np.ndarray,
    code_plane_offsets: np.ndarray,
    time_low_offsets: np.ndarray,
    time_high_offsets: np.ndarray,
    out_time_starts: np.ndarray,
    out_time_steps: np.ndarray,
    out_time_refs: np.ndarray,
    out_time_residuals: np.ndarray,
    block_ids: np.ndarray,
) -> tuple[int, int]:
    """Deserializes and validates all time columns, then unpacks time residuals for ``block_ids``.

    Args:
        body: 1D uint8 array of uncompressed body bytes.
        sample_offsets: ``BodyLayout.sample_offsets``.
        plane_byte_offsets: ``BodyLayout.plane_byte_offsets``.
        code_plane_offsets: ``BodyLayout.code_plane_offsets``.
        time_low_offsets: ``BodyLayout.time_low_offsets``.
        time_high_offsets: ``BodyLayout.time_high_offsets``.
        out_time_starts: Destination 1D int64 array for block start times.
        out_time_steps: Destination 1D int64 array for block time steps.
        out_time_refs: Destination 1D uint64 array for block reference quotients.
        out_time_residuals: Destination 1D uint64 array for zigzagged time residuals.
        block_ids: 1D int64 array of block indices whose irregular residuals should be unpacked.

    Returns:
        Tuple ``(status, failing_block_idx)`` as returned by ``read_time_columns``.
    """
    offsets = (sample_offsets, plane_byte_offsets, code_plane_offsets, time_low_offsets, time_high_offsets)
    status = read_time_columns(body, *offsets, out_time_starts, out_time_steps, out_time_refs)
    if status[0] == TIME_FIELDS_OK:
        read_time_residuals(body, *offsets, block_ids, out_time_residuals)
    return status


def check_time_fields_status(status: int, block_idx: int) -> None:
    """Raises ``ValueError`` if a ``read_time_columns`` / ``read_time_fields`` status is not OK.

    Args:
        status: ``TIME_FIELDS_*`` status code.
        block_idx: Zero-based index of the failing block.

    Raises:
        ValueError: When ``status != TIME_FIELDS_OK``.
    """
    if status == TIME_FIELDS_BAD_START:
        raise ValueError(f"block {block_idx}: start time out of range (corrupt block group)")
    if status == TIME_FIELDS_BAD_WIDE:
        raise ValueError(f"block {block_idx}: wide time residuals fit in 32 bits (corrupt block group)")
    if status == TIME_FIELDS_BAD_EMPTY:
        raise ValueError(f"block {block_idx}: empty block with nonzero time columns (corrupt block group)")
    if status == TIME_FIELDS_BAD_SINGLE:
        raise ValueError(
            f"block {block_idx}: a single sample with a nonzero time step or reference, or irregular (corrupt block group)"
        )


@njit(inline="always")
def _copy_bytes(dst: np.ndarray, dst_offset: int, src: np.ndarray, src_offset: int, num_bytes: int) -> None:
    """Copies ``num_bytes`` contiguous bytes from ``src[src_offset:]`` to ``dst[dst_offset:]``.

    Args:
        dst: Destination 1D uint8 array.
        dst_offset: Starting write offset in ``dst``.
        src: Source 1D uint8 array.
        src_offset: Starting read offset in ``src``.
        num_bytes: Number of bytes to copy.
    """
    # Slice first and index from 0 so Numba/LLVM vectorizes the copy without negative-index checks.
    dst_bytes = dst[dst_offset : dst_offset + num_bytes]
    src_bytes = src[src_offset : src_offset + num_bytes]
    for byte_idx in range(num_bytes):
        dst_bytes[byte_idx] = src_bytes[byte_idx]


@njit(inline="always")
def _copy_columns(dst: np.ndarray, dst_offset: int, src: np.ndarray, src_offset: int, num_columns: int) -> None:
    """Copies ``num_columns`` consecutive columns across every row of a 2D plane array.

    Args:
        dst: Destination 2D uint8 array.
        dst_offset: Starting column offset in ``dst``.
        src: Source 2D uint8 array.
        src_offset: Starting column offset in ``src``.
        num_columns: Number of columns to copy per row.
    """
    for row_idx in range(src.shape[0]):
        # Slice each row and index from 0 so LLVM vectorizes the inner loop (avoids ~10x slowdown
        # from Numba slice-to-slice assignment or signed-offset negative-index wraparound checks).
        dst_row = dst[row_idx, dst_offset : dst_offset + num_columns]
        src_row = src[row_idx, src_offset : src_offset + num_columns]
        for col_idx in range(num_columns):
            dst_row[col_idx] = src_row[col_idx]


@njit(nogil=True, cache=True)
def _copy_carried_planes(
    old_byte_planes: np.ndarray,
    old_code_planes: np.ndarray,
    old_time_low: np.ndarray,
    old_time_high: np.ndarray,
    old_plane_byte_offsets: np.ndarray,
    old_code_plane_offsets: np.ndarray,
    old_time_low_offsets: np.ndarray,
    old_time_high_offsets: np.ndarray,
    byte_planes: np.ndarray,
    code_planes: np.ndarray,
    time_low_planes: np.ndarray,
    time_high_planes: np.ndarray,
    plane_byte_offsets: np.ndarray,
    code_plane_offsets: np.ndarray,
    time_low_offsets: np.ndarray,
    time_high_offsets: np.ndarray,
    first_block: int,
    end_block: int,
) -> None:
    """Copies all plane regions for a contiguous run of carried-over blocks ``[first_block, end_block)``.

    Args:
        old_*: Plane views and cumulative offset arrays of the source body.
        byte_planes..time_high_offsets: Plane views and cumulative offset arrays of the output body.
        first_block: First block index in the carried run (inclusive).
        end_block: Last block index in the carried run (exclusive).
    """
    # Every plane field stores a run of blocks contiguously, so each field copies in one call.
    num_plane_bytes = plane_byte_offsets[end_block] - plane_byte_offsets[first_block]
    _copy_columns(
        byte_planes,
        SAMPLES_PER_PLANE_BYTE * plane_byte_offsets[first_block],
        old_byte_planes,
        SAMPLES_PER_PLANE_BYTE * old_plane_byte_offsets[first_block],
        SAMPLES_PER_PLANE_BYTE * num_plane_bytes,
    )
    _copy_columns(
        code_planes,
        code_plane_offsets[first_block],
        old_code_planes,
        old_code_plane_offsets[first_block],
        code_plane_offsets[end_block] - code_plane_offsets[first_block],
    )
    _copy_columns(
        time_low_planes,
        time_low_offsets[first_block],
        old_time_low,
        old_time_low_offsets[first_block],
        time_low_offsets[end_block] - time_low_offsets[first_block],
    )
    _copy_columns(
        time_high_planes,
        time_high_offsets[first_block],
        old_time_high,
        old_time_high_offsets[first_block],
        time_high_offsets[end_block] - time_high_offsets[first_block],
    )


@njit(nogil=True, cache=True)
def _splice_body(
    old_body: np.ndarray,
    old_sample_offsets: np.ndarray,
    old_plane_byte_offsets: np.ndarray,
    old_code_plane_offsets: np.ndarray,
    old_time_low_offsets: np.ndarray,
    old_time_high_offsets: np.ndarray,
    has_time: bool,
    old_time_starts: np.ndarray,
    new_ranks: np.ndarray,
    new_exponents: np.ndarray,
    new_scales: np.ndarray,
    new_anchors: np.ndarray,
    new_residuals: np.ndarray,
    new_sample_codes: np.ndarray,
    new_sample_offsets: np.ndarray,
    new_time_starts: np.ndarray,
    new_time_steps: np.ndarray,
    new_time_refs: np.ndarray,
    new_time_residuals: np.ndarray,
    block_flags: np.ndarray,
    block_sizes: np.ndarray,
    sample_offsets: np.ndarray,
    plane_byte_offsets: np.ndarray,
    code_plane_offsets: np.ndarray,
    time_low_offsets: np.ndarray,
    time_high_offsets: np.ndarray,
    out_body: np.ndarray,
) -> None:
    """Populates ``out_body`` by copying carried-over blocks from ``old_body`` and packing new blocks.

    Block ``b`` is packed from the ``new_*`` arrays at index ``new_ranks[b]`` when
    ``new_ranks[b] >= 0``, and copied verbatim from ``old_body`` otherwise.
    """
    # Walk the merged block sequence. Coalesce consecutive carried-over blocks into runs so
    # their metadata columns and plane slices copy in bulk, updating only the first non-empty
    # carried block's delta-encoded time_start. Pack each replaced/appended block individually.
    num_blocks = block_flags.shape[0]
    old_num_blocks = old_sample_offsets.shape[0] - 1
    old_plane_bytes = int(old_plane_byte_offsets[old_num_blocks])
    num_plane_bytes = int(plane_byte_offsets[num_blocks])
    old_byte_planes = residual_byte_planes_view(old_body, old_num_blocks, old_plane_bytes, has_time)
    old_code_planes = sample_code_planes_view(
        old_body, old_num_blocks, old_plane_bytes, int(old_code_plane_offsets[old_num_blocks]), has_time
    )
    byte_planes = residual_byte_planes_view(out_body, num_blocks, num_plane_bytes, has_time)
    code_planes = sample_code_planes_view(
        out_body, num_blocks, num_plane_bytes, int(code_plane_offsets[num_blocks]), has_time
    )
    if has_time:
        old_time_low, old_time_high = time_residual_planes_views(
            old_body,
            old_num_blocks,
            old_plane_bytes,
            int(old_code_plane_offsets[old_num_blocks]),
            int(old_time_low_offsets[old_num_blocks]),
            int(old_time_high_offsets[old_num_blocks]),
        )
        time_low_planes, time_high_planes = time_residual_planes_views(
            out_body,
            num_blocks,
            num_plane_bytes,
            int(code_plane_offsets[num_blocks]),
            int(time_low_offsets[num_blocks]),
            int(time_high_offsets[num_blocks]),
        )
        # Zero time residual planes once so new irregular blocks only write their active byte levels.
        time_low_planes[:, :] = 0
        time_high_planes[:, :] = 0
    else:
        old_time_low = old_time_high = time_low_planes = time_high_planes = np.zeros((TIME_LOW_PLANES, 0), np.uint8)
    scratch_tail = np.empty(SAMPLES_PER_PLANE_BYTE, np.uint64)
    # Descriptor list of metadata columns: (new_field_start, old_field_start, num_byte_planes).
    column_fields = [
        (0, 0, 1),
        (block_sizes_start(num_blocks), block_sizes_start(old_num_blocks), BYTES_PER_SIZE),
        (grid_exponents_start(num_blocks), grid_exponents_start(old_num_blocks), BYTES_PER_EXPONENT),
        (grid_scales_start(num_blocks), grid_scales_start(old_num_blocks), BYTES_PER_SCALE),
        (block_anchors_start(num_blocks), block_anchors_start(old_num_blocks), BYTES_PER_ANCHOR),
    ]
    if has_time:
        # time_starts, time_steps, and time_ref_quotients are contiguous byte-planed columns (24 planes).
        column_fields.append((time_starts_start(num_blocks), time_starts_start(old_num_blocks), TIME_BYTES_PER_BLOCK))
    previous_start = np.int64(0)
    seen_nonempty = False
    block_idx = 0
    while block_idx < num_blocks:
        rank = new_ranks[block_idx]
        if rank < 0:
            # Find the full contiguous run of carried-over blocks [block_idx, run_end).
            run_end = block_idx + 1
            while run_end < num_blocks and new_ranks[run_end] < 0:
                run_end += 1
            for field_start, old_field_start, num_planes in column_fields:
                for plane_idx in range(num_planes):
                    _copy_bytes(
                        out_body,
                        field_start + plane_idx * num_blocks + block_idx,
                        old_body,
                        old_field_start + plane_idx * old_num_blocks + block_idx,
                        run_end - block_idx,
                    )
            if has_time:
                # Re-anchor only the first non-empty block in the run against the preceding block's start.
                last_nonempty = -1
                for run_idx in range(block_idx, run_end):
                    if block_sizes[run_idx]:
                        if last_nonempty < 0:
                            start = old_time_starts[run_idx]
                            put_int64(
                                out_body,
                                time_starts_start(num_blocks),
                                num_blocks,
                                run_idx,
                                np.int64(np.uint64(start) - np.uint64(previous_start)) if seen_nonempty else start,
                            )
                        last_nonempty = run_idx
                if last_nonempty >= 0:
                    previous_start = old_time_starts[last_nonempty]
                    seen_nonempty = True
            _copy_carried_planes(
                old_byte_planes,
                old_code_planes,
                old_time_low,
                old_time_high,
                old_plane_byte_offsets,
                old_code_plane_offsets,
                old_time_low_offsets,
                old_time_high_offsets,
                byte_planes,
                code_planes,
                time_low_planes,
                time_high_planes,
                plane_byte_offsets,
                code_plane_offsets,
                time_low_offsets,
                time_high_offsets,
                block_idx,
                run_end,
            )
            block_idx = run_end
            continue
        # Pack a single replaced or appended block from the new_* arrays.
        block_len = block_sizes[block_idx]
        flags = block_flags[block_idx]
        out_body[block_idx] = flags
        out_body[block_sizes_start(num_blocks) + block_idx] = np.uint8(block_len & 0xFF)
        out_body[block_sizes_start(num_blocks) + num_blocks + block_idx] = np.uint8(block_len >> 8)
        put_int16(out_body, grid_exponents_start(num_blocks), num_blocks, block_idx, new_exponents[rank])
        put_uint32(out_body, grid_scales_start(num_blocks), num_blocks, block_idx, new_scales[rank])
        put_int64(out_body, block_anchors_start(num_blocks), num_blocks, block_idx, new_anchors[rank])
        if has_time:
            stored_start, time_step, time_ref = np.int64(0), np.int64(0), np.int64(0)
            if block_len:
                start = new_time_starts[rank]
                time_step = new_time_steps[rank]
                time_ref = np.int64(new_time_refs[rank])
                stored_start = np.int64(np.uint64(start) - np.uint64(previous_start)) if seen_nonempty else start
                previous_start = start
                seen_nonempty = True
            put_int64(out_body, time_starts_start(num_blocks), num_blocks, block_idx, stored_start)
            put_int64(out_body, time_steps_start(num_blocks), num_blocks, block_idx, time_step)
            put_int64(out_body, time_ref_quotients_start(num_blocks), num_blocks, block_idx, time_ref)
        block_idx += 1
        if block_len == 0:
            continue
        plane_byte_offset = plane_byte_offsets[block_idx - 1]
        first_sample = new_sample_offsets[rank]
        sample_start = SAMPLES_PER_PLANE_BYTE * plane_byte_offset
        zigzag_encode_block(
            new_residuals[first_sample : first_sample + block_len],
            byte_planes[0, sample_start : sample_start + block_len],
            byte_planes[1, sample_start : sample_start + block_len],
        )
        byte_planes[:, sample_start + block_len : SAMPLES_PER_PLANE_BYTE * plane_byte_offsets[block_idx]] = 0
        if flags & BLOCK_FLAG_SAMPLE_CODES:
            pack_sample_codes(
                new_sample_codes[first_sample : first_sample + block_len],
                code_planes,
                code_plane_offsets[block_idx - 1],
            )
        if has_time and flags & BLOCK_FLAG_IRREGULAR_TIME:
            shuffle_time_residuals(
                new_time_residuals[first_sample : first_sample + block_len],
                time_low_planes,
                time_high_planes,
                time_low_offsets[block_idx - 1],
                time_high_offsets[block_idx - 1],
                scratch_tail,
            )


def splice_body(
    old_body: np.ndarray,
    old_layout: BodyLayout,
    old_time_starts: np.ndarray | None,
    new_block_ids: np.ndarray,
    new_columns: EncodedBlockColumns,
    new_sample_offsets: np.ndarray,
) -> tuple[np.ndarray, BodyLayout]:
    """Builds a new uncompressed byte-plane body by replacing or appending selected blocks.

    Untouched blocks are copied verbatim from ``old_body`` without unpacking or re-encoding;
    new and replaced blocks are serialized from ``new_columns``.

    Args:
        old_body: 1D uint8 array of the existing uncompressed body with byte-plane residuals.
        old_layout: ``BodyLayout`` of ``old_body``.
        old_time_starts: 1D int64 array of absolute block start times (from ``read_time_columns``),
            or None for a group without a time axis.
        new_block_ids: Strictly increasing 1D int64 array of replaced/appended block indices.
            Every index past the old group's end must be included.
        new_columns: ``EncodedBlockColumns`` for ``new_block_ids`` in matching order.
        new_sample_offsets: 1D int64 array of cumulative sample offsets into ``new_columns``.

    Returns:
        Tuple ``(new_body, new_layout)`` containing the merged byte-plane body and its ``BodyLayout``.

    Raises:
        ValueError: If any block index past ``old_num_blocks`` is omitted from ``new_block_ids``.
    """
    # Build the merged flags and sizes arrays, compute the new BodyLayout, and delegate to _splice_body.
    old_num_blocks = old_layout.sample_offsets.shape[0] - 1
    num_blocks = max(old_num_blocks, int(new_block_ids[-1]) + 1) if new_block_ids.shape[0] else old_num_blocks
    has_time = old_time_starts is not None
    # Map each merged block index to its position in new_columns (-1 for carried-over blocks).
    new_ranks = np.full(num_blocks, -1, np.int64)
    new_ranks[new_block_ids] = np.arange(new_block_ids.shape[0])
    if (new_ranks[old_num_blocks:] < 0).any():
        raise ValueError("every block past the old block group's end must be new")
    block_flags = np.empty(num_blocks, np.uint8)
    block_flags[:old_num_blocks] = old_body[:old_num_blocks]
    block_flags[new_block_ids] = new_columns.block_flags
    block_sizes = np.empty(num_blocks, np.int64)
    block_sizes[:old_num_blocks] = np.diff(old_layout.sample_offsets)
    block_sizes[new_block_ids] = new_columns.block_sizes
    offsets = compute_body_layout(block_flags, block_sizes)
    out_body = np.empty(body_byte_size(num_blocks, offsets, has_time), np.uint8)
    new_time = new_columns.time_columns
    if new_time is None:
        new_time = allocate_time_columns(0, 0)
    _splice_body(
        old_body,
        *old_layout,
        has_time,
        old_time_starts if old_time_starts is not None else np.zeros(0, np.int64),
        new_ranks,
        new_columns.grid_exponents,
        new_columns.grid_scales,
        new_columns.block_anchors,
        new_columns.residuals,
        new_columns.sample_codes,
        new_sample_offsets,
        *new_time,
        block_flags,
        block_sizes,
        *offsets,
        out_body,
    )
    return out_body, offsets
