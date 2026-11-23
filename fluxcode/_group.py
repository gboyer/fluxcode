# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Block-group orchestration: encoding, decompressing, decoding, and binary splicing.

Bridges the public API in ``_api`` and the low-level Numba/zstd kernels. Blocks in a group
are represented as flat sample arrays accompanied by per-block sizes so fixed-size, explicit,
and time-partitioned blocks share one code path.
"""

from collections.abc import Mapping
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import zstandard

from . import (
    _args,
    _bitpacking,
    _body,
    _compress,
    _decoder,
    _encoder,
    _escapes,
    _format,
    _grids,
    _time,
    _time_rounding,
)
from ._types import DecodedGroup, EncodedGroup, EncodedSeries, Params, UpdatedGroup


def sample_offsets(block_sizes: np.ndarray) -> np.ndarray:
    """Computes cumulative sample offsets from per-block sample counts.

    Args:
        block_sizes: 1D int64 array of sample counts per block.

    Returns:
        1D int64 array of length ``len(block_sizes) + 1`` starting at 0.
    """
    offsets = np.zeros(block_sizes.shape[0] + 1, np.int64)
    np.cumsum(block_sizes, out=offsets[1:])
    return offsets


class BlockStats(NamedTuple):
    """Per-block finite summary statistics computed during encoding.

    Attributes:
        block_min: 1D float64 array of minimum finite sample per block (NaN if none).
        block_max: 1D float64 array of maximum finite sample per block (NaN if none).
        block_mean: 1D float64 array of mean finite sample per block (NaN if none).
    """

    block_min: np.ndarray
    block_max: np.ndarray
    block_mean: np.ndarray


def kernel_args(params: Params) -> tuple[int, int, int, float, float, int, int, float]:
    """Converts ``Params`` into the positional scalar arguments expected by ``_encoder.encode_group``.

    Args:
        params: Encoder configuration parameters.

    Returns:
        Tuple ``(min_bits, max_bits, orders_mask, noise_factor, target_bits, grids_level,
        tolerance_exp, tolerance_step)``.
    """
    return (
        params.min_quantize_bits,
        params.max_quantize_bits,
        sum(1 << order for order in params.diff_orders),
        float(params.noise_floor_sigma),
        float(params.target_bits_per_sample or 0.0),
        _grids.GRIDS_LEVELS[params.grids],
        _grids.tolerance_exponent(params.absolute_tolerance),
        _grids.tolerance_step(params.absolute_tolerance),
    )


def encode_columns(
    samples: np.ndarray,
    block_sizes: np.ndarray,
    params: Params,
    ticks: np.ndarray | None = None,
) -> tuple[_format.EncodedBlockColumns, BlockStats]:
    """Encodes flat block samples (and optional ticks) into in-memory columns and block statistics.

    Args:
        samples: 1D float64 array of all samples in block order.
        block_sizes: 1D int64 array of per-block sample counts summing to ``len(samples)``.
        params: Encoder configuration parameters.
        ticks: Optional 1D int64 array of timestamps in the group's time unit.

    Returns:
        Tuple ``(columns, stats)`` containing ``EncodedBlockColumns`` and ``BlockStats``.

    Raises:
        ValueError: If timestamps contain NaT or decrease anywhere.
    """
    # Encode timestamps first so irregular-time flags are available to guide value noise-floor
    # estimation, then run the value encoder kernel, merge time flags, and collect escaped values.
    num_blocks = block_sizes.shape[0]
    num_samples = samples.shape[0]
    time_flags = np.zeros(num_blocks, np.uint8)
    time_columns = None if ticks is None else encode_time_columns(ticks, block_sizes, time_flags)
    columns = _format.EncodedBlockColumns(
        np.empty(num_blocks, np.uint8),
        block_sizes,
        np.empty(num_blocks, np.int64),
        np.empty(num_blocks, np.int64),
        np.empty(num_blocks, np.int64),
        np.empty(num_samples, np.int16),
        np.empty(num_samples, np.uint8),
        time_columns,
    )
    stats = BlockStats(np.empty(num_blocks), np.empty(num_blocks), np.empty(num_blocks))
    _encoder.encode_group(
        samples,
        sample_offsets(block_sizes),
        np.zeros(0, np.int64) if ticks is None else ticks,
        time_flags[:0] if ticks is None else time_flags,
        *kernel_args(params),
        columns.block_flags,
        columns.grid_exponents,
        columns.grid_scales,
        columns.block_anchors,
        columns.residuals,
        columns.sample_codes,
        *stats,
    )
    # Combine value flags written by the encoder kernel with time flags written by encode_times.
    columns.block_flags[:] |= time_flags
    columns = columns._replace(
        escaped_values=_escapes.collect_escaped_values(
            samples,
            block_sizes,
            columns.block_flags,
            columns.sample_codes,
            stats.block_min,
            stats.block_max,
            params.max_quantize_bits,
        )
    )
    return columns, stats


def encode_time_columns(
    ticks: np.ndarray, block_sizes: np.ndarray, in_out_block_flags: np.ndarray
) -> _format.EncodedTimeColumns:
    """Encodes timestamps into ``EncodedTimeColumns`` and validates non-decreasing order.

    Args:
        ticks: 1D int64 array of all samples' timestamps.
        block_sizes: 1D int64 array of per-block sample counts.
        in_out_block_flags: In-out 1D uint8 array whose time flag bits are updated in place.

    Returns:
        Populated ``EncodedTimeColumns``.

    Raises:
        ValueError: If timestamps contain NaT or decrease anywhere.
    """
    # NaT is INT64_MIN; checking index 0 catches leading NaT, while any later NaT is a decrease.
    if ticks.shape[0] and ticks[0] == _time.INT64_MIN:
        raise ValueError("times contain NaT")
    time_columns = _format.allocate_time_columns(block_sizes.shape[0], ticks.shape[0])
    status, sample_idx = _time.encode_times(ticks, sample_offsets(block_sizes), in_out_block_flags, *time_columns)
    if status != _time.OK:
        raise _args.decrease_error(ticks, sample_idx)
    return time_columns


def stored_tails(
    parsed: "ParsedGroup", time_columns: _format.EncodedTimeColumns, blocks: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Returns the last timestamp and uniform interval (0 if irregular) of non-empty stored blocks.

    Args:
        parsed: Parsed existing block group.
        time_columns: Its ``EncodedTimeColumns`` (metadata columns loaded; irregular residuals
            for ``blocks`` are unpacked on demand).
        blocks: 1D int64 array of non-empty block indices.

    Returns:
        Tuple ``(last_ticks, steps)`` where ``steps[i]`` is the block's positive uniform interval
        if regular with ``>= 2`` samples, else 0.
    """
    # Regular blocks compute their last tick directly from start + (size - 1) * interval; only
    # irregular blocks unpack and expand their time residuals.
    regular = ~(parsed.block_flags[blocks] & _format.BLOCK_FLAG_IRREGULAR_TIME).astype(bool)
    interval = time_columns.steps[blocks] * time_columns.refs[blocks].astype(np.int64)
    sizes = parsed.block_sizes[blocks]
    last = time_columns.starts[blocks] + (sizes - 1) * interval
    irregular = np.flatnonzero(~regular)
    last[irregular] = _last_ticks(parsed, time_columns, blocks[irregular])
    steps = np.where(regular & (sizes >= 2) & (interval > 0), interval, 0)
    return last, steps


def snap_ticks(
    ticks: np.ndarray,
    block_sizes: np.ndarray,
    time_error: float,
    seed_ticks: np.ndarray | None = None,
    seed_steps: np.ndarray | None = None,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Applies lossy ``time_error`` rounding to each block's timestamps and reconciles boundaries.

    Args:
        ticks: 1D int64 array of all samples' timestamps.
        block_sizes: 1D int64 array of per-block sample counts summing to ``len(ticks)``.
        time_error: Relative time error tolerance (``<= 0`` returns ``ticks`` unchanged).
        seed_ticks: Optional 1D int64 array of preceding stored blocks' last ticks (``INT64_MIN``
            where a block follows the previous block in ``ticks``).
        seed_steps: Optional 1D int64 array of those stored blocks' uniform intervals (0 if irregular).
        bounds: Optional ``(lows, highs)`` per-block inclusive rounding bounds.

    Returns:
        Tuple ``(rounded_ticks, max_moves)`` where ``rounded_ticks`` is the original ``ticks``
        array if no tick moved, and ``max_moves`` holds each block's maximum tick displacement.

    Raises:
        ValueError: If timestamps contain NaT or decrease anywhere when ``time_error > 0``.
    """
    # Delegate per-block rounding to _time_rounding.round_blocks, then reconcile any cross-block
    # overlaps with clamp_block_boundaries if at least one tick moved.
    num_blocks = block_sizes.shape[0]
    moves = np.zeros(num_blocks, np.int64)
    if time_error <= 0 or not ticks.shape[0]:
        return ticks, moves
    if ticks[0] == _time.INT64_MIN:
        raise ValueError("times contain NaT")
    offsets = sample_offsets(block_sizes)
    if seed_ticks is None:
        seed_ticks = np.full(num_blocks, _time.INT64_MIN, np.int64)
    if seed_steps is None:
        seed_steps = np.zeros(num_blocks, np.int64)
    if bounds is None:
        bounds = (np.full(num_blocks, _time.INT64_MIN, np.int64), np.full(num_blocks, _time.INT64_MAX, np.int64))
    rounded = np.empty_like(ticks)
    status, sample_idx = _time_rounding.round_blocks(
        ticks, offsets, float(time_error), *bounds, seed_ticks, seed_steps, rounded, moves
    )
    if status != _time.OK:
        raise _args.decrease_error(ticks, sample_idx)
    if not moves.any():
        return ticks, moves
    _time_rounding.clamp_block_boundaries(ticks, offsets, moves, rounded)
    return rounded, moves


def encode(
    samples: np.ndarray,
    block_sizes: np.ndarray,
    params: Params,
    ticks: np.ndarray | None = None,
    time_unit: int = 0,
    *,
    snap: bool = True,
) -> EncodedGroup:
    """Encodes blocks of samples (and optional ticks) into a compressed block group.

    Args:
        samples: 1D float64 array of all samples in block order.
        block_sizes: 1D int64 array of block sizes (0..65535) summing to ``len(samples)``.
        params: Encoder configuration parameters.
        ticks: Optional 1D int64 array of sample timestamps.
        time_unit: ``TimeUnitCode`` integer (0 when untimed).
        snap: Whether to apply ``params.time_error`` rounding (False if already rounded).

    Returns:
        ``EncodedGroup`` holding the compressed bytes and per-block min, max, and mean.

    Raises:
        ValueError: If block/sample counts exceed format limits, or timestamps contain NaT or decrease.
    """
    _args.check_group_counts(block_sizes.shape[0], samples.shape[0], block_sizes)
    if ticks is not None and snap:
        ticks, _ = snap_ticks(ticks, block_sizes, params.time_error)
    columns, stats = encode_columns(samples, block_sizes, params, ticks)
    return EncodedGroup(
        _compress.compress_columns(columns, samples.shape[0], _compress.EFFORTS[params.effort], time_unit), *stats
    )


def encode_series(
    samples: np.ndarray, block_len: int, blocks_per_group: int, ticks: np.ndarray | None, time_unit: int, params: Params
) -> EncodedSeries:
    """Encodes an arbitrary-length series as a sequence of fixed-size compressed block groups.

    Args:
        samples: 1D float64 array of samples.
        block_len: Target number of samples per block.
        blocks_per_group: Target number of blocks per block group.
        ticks: Optional 1D int64 array of timestamps in ``time_unit`` ticks.
        time_unit: ``TimeUnitCode`` integer (0 when untimed).
        params: Encoder configuration parameters.

    Returns:
        ``EncodedSeries`` containing lists of compressed group bytes and per-group block statistics.

    Raises:
        ValueError: If group dimensions exceed limits or timestamps decrease anywhere.
    """
    # Validate cross-group monotonicity and round the entire series' timestamps up front so
    # adjacent groups stay ordered across group boundaries, then encode each group independently.
    samples_per_group = blocks_per_group * block_len
    num_blocks = -(-min(samples_per_group, samples.shape[0]) // block_len)
    _args.check_group_counts(num_blocks, min(samples_per_group, samples.shape[0]))
    if ticks is not None:
        group_starts = np.arange(samples_per_group, ticks.shape[0], samples_per_group)
        decreases = group_starts[ticks[group_starts] < ticks[group_starts - 1]]
        if decreases.shape[0]:
            raise _args.decrease_error(ticks, int(decreases[0]))
        ticks, _ = snap_ticks(ticks, _args.fixed_sizes(ticks.shape[0], block_len), params.time_error)
    parts = []
    for idx in range(0, samples.shape[0], samples_per_group):
        chunk = samples[idx : idx + samples_per_group]
        parts.append(
            encode(
                chunk,
                _args.fixed_sizes(chunk.shape[0], block_len),
                params,
                None if ticks is None else ticks[idx : idx + samples_per_group],
                time_unit,
                snap=False,
            )
        )
    groups, mins, maxs, means = (list(col) for col in zip(*parts, strict=True))
    return EncodedSeries(groups, mins, maxs, means)


class ParsedGroup(NamedTuple):
    """A decompressed and structurally validated block group ready for decoding or splicing.

    Attributes:
        body: 1D uint8 array of the uncompressed body (with any ``group_time_step`` unscaled).
        header: Parsed ``GroupHeader``.
        block_flags: 1D uint8 view of per-block flags inside ``body``.
        block_sizes: 1D int64 array of per-block sample counts.
        layout: Cumulative ``BodyLayout`` offsets into ``body``.
        escaped_values: 1D float64 array of escaped values unpacked from the body trailer.
        escape_offsets: 1D int64 cumulative offset array into ``escaped_values`` per block.
    """

    body: np.ndarray
    header: _format.GroupHeader
    block_flags: np.ndarray
    block_sizes: np.ndarray
    layout: _format.BodyLayout
    escaped_values: np.ndarray
    escape_offsets: np.ndarray

    @property
    def has_time(self) -> bool:
        """Whether the block group includes a time axis."""
        return self.header.time_unit != 0


def decompress(group: bytes) -> ParsedGroup:
    """Parses the header, decompresses the zstd frame, and validates the body layout and metadata.

    Args:
        group: Compressed block group bytes (8-byte header followed by a zstd frame).

    Returns:
        Validated ``ParsedGroup``.

    Raises:
        ValueError: If the header, zstd content size, body length, ``group_time_step`` trailer,
            escape codes, or per-block metadata columns are invalid.
        zstandard.ZstdError: If the zstd frame is corrupt.
    """
    # Validate header and zstd content size before allocating, decompress the body, undo any
    # group_time_step scaling, verify exact byte length including escaped values, and validate
    # all per-block flags, exponents, scales, and anchors.
    header = _format.unpack_header(group)
    num_blocks, num_samples = header.num_blocks, header.num_samples
    has_time = header.time_unit != 0
    frame = memoryview(group)[_format.HEADER_BYTES :]
    content_size = zstandard.frame_content_size(frame)
    if content_size < 0:
        raise ValueError("block group's zstd frame doesn't record its content size")
    smallest, largest = _format.body_size_bounds(num_blocks, num_samples, has_time)
    if not smallest <= content_size <= largest:
        raise ValueError(
            f"block group body of {content_size} bytes doesn't fit {num_blocks} blocks of {num_samples} samples"
        )
    body = np.frombuffer(_compress.get_zstd_codecs()[1].decompress(frame), np.uint8)
    block_flags = body[:num_blocks]
    block_sizes, offsets = _format.read_body_layout(body, num_blocks)
    if header.group_time_step:
        # Unscale time_starts increases and time_steps using the trailing uint64 group_time_step.
        min_size = _format.body_byte_size(num_blocks, offsets, has_time) + _format.GROUP_TIME_STEP_BYTES
        step = (
            body[-_format.GROUP_TIME_STEP_BYTES :].copy().view(np.uint64)[0]
            if body.shape[0] >= min_size
            else np.uint64(0)
        )
        body = body[: -_format.GROUP_TIME_STEP_BYTES].copy()
        if step < 2 or not _format.scale_time_columns(body, offsets.sample_offsets, step, False):
            raise ValueError(
                "block group's group_time_step is under 2 or overflows its time columns (corrupt block group)"
            )
        block_flags = body[:num_blocks]
        header = header._replace(group_time_step=False)
    if int(offsets.sample_offsets[-1]) != num_samples:
        raise ValueError(
            f"block group's block sizes add up to {int(offsets.sample_offsets[-1])}, not its {num_samples} samples"
        )
    # Verify exact uncompressed byte length: core fields + 8 bytes per CODE_ESCAPE.
    fields_size = _format.body_byte_size(num_blocks, offsets, has_time)
    escape_offsets = np.zeros(num_blocks + 1, np.int64)
    escapes_ok = False
    if body.shape[0] >= fields_size:
        escapes_ok = _escapes.count_block_escapes(
            body,
            offsets.sample_offsets,
            offsets.plane_byte_offsets,
            offsets.code_plane_offsets,
            has_time,
            escape_offsets,
        )
    if (
        body.shape[0] < fields_size
        or not escapes_ok
        or body.shape[0] - fields_size != _format.BYTES_PER_ESCAPED_VALUE * int(escape_offsets[-1])
    ):
        raise ValueError(f"block group body of {body.shape[0]} bytes doesn't match its blocks' sizes, flags and codes")
    escaped_values = _escapes.unpack_escaped_values(body, fields_size, int(escape_offsets[-1]))
    validation_status, failing_block_idx = _decoder.validate_body_columns(body, offsets.sample_offsets, has_time)
    if validation_status == _decoder.INVALID_FLAGS:
        raise ValueError(
            f"block {failing_block_idx}: block flags {body[failing_block_idx]:#04x} sets reserved bits "
            "(not supported by this version)"
        )
    if validation_status == _decoder.INVALID_EXPONENT:
        raise ValueError(f"block {failing_block_idx}: parameter out of range (corrupt block group)")
    if validation_status == _decoder.INVALID_ANCHOR:
        raise ValueError(f"block {failing_block_idx}: anchor out of range (corrupt block group)")
    if validation_status == _decoder.INVALID_SCALE:
        raise ValueError(f"block {failing_block_idx}: grid scale invalid for its grid (corrupt block group)")
    return ParsedGroup(body, header, block_flags, block_sizes, offsets, escaped_values, escape_offsets)


def read_time_columns(parsed: ParsedGroup, block_ids: np.ndarray) -> _format.EncodedTimeColumns:
    """Deserializes and validates time metadata columns, unpacking residuals for ``block_ids``.

    Args:
        parsed: Parsed block group with a time axis.
        block_ids: 1D int64 array of block indices whose irregular residuals should be unpacked
            (pass an empty array to read metadata columns only).

    Returns:
        Populated ``EncodedTimeColumns``.

    Raises:
        ValueError: If any block's time metadata columns or wide-time planes are corrupt.
    """
    time_columns = _format.allocate_time_columns(parsed.header.num_blocks, parsed.header.num_samples)
    _body.check_time_fields_status(
        *_body.read_time_fields(parsed.body, *parsed.layout, *time_columns, block_ids=block_ids)
    )
    return time_columns


def read_time_residuals(parsed: ParsedGroup, time_columns: _format.EncodedTimeColumns, block_ids: np.ndarray) -> None:
    """Unpacks irregular time residuals for ``block_ids`` into an existing ``EncodedTimeColumns``.

    Args:
        parsed: Parsed block group with a time axis.
        time_columns: ``EncodedTimeColumns`` whose ``residuals`` array is updated in place.
        block_ids: 1D int64 array of block indices to unpack.
    """
    _body.read_time_residuals(parsed.body, *parsed.layout, block_ids, time_columns.residuals)


def expand_times(
    block_flags: np.ndarray,
    offsets: np.ndarray,
    time_columns: _format.EncodedTimeColumns,
    block_ids: np.ndarray,
    out_ticks: np.ndarray,
) -> None:
    """Expands encoded time columns for ``block_ids`` into ``out_ticks``, raising on corruption.

    Args:
        block_flags: 1D uint8 array of per-block flags.
        offsets: 1D int64 array of cumulative sample offsets.
        time_columns: ``EncodedTimeColumns`` with residuals unpacked for ``block_ids``.
        block_ids: 1D int64 array of block indices to expand.
        out_ticks: Preallocated 1D int64 array sized for all group samples.

    Raises:
        ValueError: If a block's time step, first residual, or reconstructed timestamp is invalid.
    """
    status, block_idx = _time.expand_times(block_flags, offsets, *time_columns, block_ids, out_ticks)
    if status == _time.BAD_STEP:
        raise ValueError(f"block {block_idx}: time step out of range (corrupt block group)")
    if status == _time.BAD_FIRST_RESIDUAL:
        raise ValueError(f"block {block_idx}: first time residual is not 0 (corrupt block group)")
    if status == _time.OVERFLOW:
        raise ValueError(f"block {block_idx}: times overflow int64 (corrupt block group)")


def decode_blocks(
    parsed: ParsedGroup, block_ids: np.ndarray, time_columns: _format.EncodedTimeColumns | None = None
) -> tuple[np.ndarray, np.ndarray | None]:
    """Decodes values (and timestamps if timed) for the requested ``block_ids``.

    Args:
        parsed: Parsed block group.
        block_ids: 1D int64 array of block indices to decode.
        time_columns: Optional pre-read ``EncodedTimeColumns`` (with residuals unpacked for
            ``block_ids``).

    Returns:
        Tuple ``(values, ticks)`` sized for the entire group where only slices belonging to
        ``block_ids`` are populated (``ticks`` is None when untimed).
    """
    offsets = parsed.layout
    values = np.empty(parsed.header.num_samples)
    _decoder.decode_blocks(
        parsed.body,
        offsets.sample_offsets,
        offsets.plane_byte_offsets,
        offsets.code_plane_offsets,
        block_ids,
        values,
        parsed.header.byte_planes,
        parsed.has_time,
        parsed.escaped_values,
        parsed.escape_offsets,
    )
    ticks = None
    if parsed.has_time:
        ticks = np.empty(parsed.header.num_samples, np.int64)
        if time_columns is None:
            time_columns = read_time_columns(parsed, block_ids)
        expand_times(parsed.block_flags, offsets.sample_offsets, time_columns, block_ids, ticks)
    return values, ticks


def time_dtype(time_unit: int) -> np.dtype:
    """Returns the NumPy ``datetime64`` dtype corresponding to a ``TimeUnitCode`` integer.

    Args:
        time_unit: Non-zero ``TimeUnitCode`` integer.

    Returns:
        ``datetime64`` dtype (e.g. ``datetime64[ns]``).
    """
    return np.dtype(f"datetime64[{_format.TIME_UNIT_NAMES[time_unit]}]")


def decode(group: bytes) -> DecodedGroup:
    """Decompresses and decodes all blocks of a block group.

    Args:
        group: Compressed block group bytes.

    Returns:
        ``DecodedGroup`` containing values, optional ``datetime64`` times, and block sizes.
    """
    parsed = decompress(group)
    values, ticks = decode_blocks(parsed, np.arange(parsed.header.num_blocks, dtype=np.int64))
    times = None if ticks is None else ticks.view(time_dtype(parsed.header.time_unit))
    return DecodedGroup(values, times, parsed.block_sizes)


def _block_ticks(parsed: ParsedGroup, time_columns: _format.EncodedTimeColumns, block_ids: np.ndarray) -> np.ndarray:
    """Unpacks time residuals and expands timestamps for ``block_ids`` into a group-sized array."""
    read_time_residuals(parsed, time_columns, block_ids)
    out_ticks = np.empty(parsed.header.num_samples, np.int64)
    expand_times(parsed.block_flags, parsed.layout.sample_offsets, time_columns, block_ids, out_ticks)
    return out_ticks


def _last_ticks(parsed: ParsedGroup, time_columns: _format.EncodedTimeColumns, block_ids: np.ndarray) -> np.ndarray:
    """Returns the last timestamp of each non-empty block in ``block_ids``."""
    if not block_ids.shape[0]:
        return np.zeros(0, np.int64)
    out_ticks = _block_ticks(parsed, time_columns, block_ids)
    return np.asarray(out_ticks[parsed.layout.sample_offsets[block_ids + 1] - 1])


def _check_spliced_order(
    parsed: ParsedGroup,
    time_columns: _format.EncodedTimeColumns,
    block_sizes: np.ndarray,
    block_starts: np.ndarray,
    is_new: np.ndarray,
    new_last_ticks: np.ndarray,
) -> None:
    """Verifies non-decreasing timestamps across every boundary touching a newly spliced block.

    Args:
        parsed: Existing parsed block group.
        time_columns: Existing group's ``EncodedTimeColumns`` (metadata columns loaded).
        block_sizes: 1D int64 array of spliced block sizes.
        block_starts: 1D int64 array of spliced block start timestamps.
        is_new: 1D bool mask marking newly encoded blocks.
        new_last_ticks: 1D int64 array of each non-empty new block's last tick (indexed by block).

    Raises:
        ValueError: If any non-empty block starts before the preceding non-empty block's last tick.
    """
    # Check only consecutive non-empty pairs where at least one block is new; carried-over pairs
    # were already validated when the original group was encoded.
    filled = np.flatnonzero(block_sizes)
    pairs = np.flatnonzero(is_new[filled[:-1]] | is_new[filled[1:]])
    previous, following = filled[pairs], filled[pairs + 1]
    last = new_last_ticks[previous]
    carried = ~is_new[previous]
    last[carried] = _last_ticks(parsed, time_columns, previous[carried])
    decreasing = np.flatnonzero(block_starts[following] < last)
    if decreasing.shape[0]:
        pair_idx = int(decreasing[0])
        raise ValueError(
            f"times must be non-decreasing: block {int(following[pair_idx])} starts at "
            f"{int(block_starts[following[pair_idx]])}, below block {int(previous[pair_idx])}'s last time "
            f"{int(last[pair_idx])}"
        )


def _stored_allowance(
    parsed: ParsedGroup, time_columns: _format.EncodedTimeColumns, block_idx: int, time_error: float
) -> float:
    """Computes the maximum tick displacement ``time_error * slack * mean_interval`` of a stored block."""
    size = int(parsed.block_sizes[block_idx])
    if size < 2:
        return 0.0
    if not parsed.block_flags[block_idx] & _format.BLOCK_FLAG_IRREGULAR_TIME:
        mean = float(time_columns.steps[block_idx]) * float(time_columns.refs[block_idx])
    else:
        last = int(_last_ticks(parsed, time_columns, np.array([block_idx], np.int64))[0])
        mean = (last - int(time_columns.starts[block_idx])) / (size - 1)
    return time_error * _time_rounding.TIME_ERROR_SLACK * mean


class StoredNeighbours(NamedTuple):
    """Immediate non-empty stored neighbours flanking each non-empty new block in a splice.

    Attributes:
        rows: 1D int64 positions of non-empty new blocks inside ``indices``.
        previous: 1D int64 block index of each row's preceding non-empty block (-1 if none).
        following: 1D int64 block index of each row's following non-empty block (-1 if none).
        stored_previous: 1D bool mask indicating ``previous`` is a carried-over stored block.
        stored_following: 1D bool mask indicating ``following`` is a carried-over stored block.
        last_ticks: 1D int64 last ticks of stored preceding blocks (0 elsewhere).
        steps: 1D int64 uniform intervals of stored preceding blocks (0 if irregular or none).
    """

    rows: np.ndarray
    previous: np.ndarray
    following: np.ndarray
    stored_previous: np.ndarray
    stored_following: np.ndarray
    last_ticks: np.ndarray
    steps: np.ndarray


def _stored_neighbours(
    parsed: ParsedGroup, time_columns: _format.EncodedTimeColumns, sizes: np.ndarray, indices: np.ndarray
) -> StoredNeighbours:
    """Finds the carried-over non-empty blocks immediately before and after each new non-empty block."""
    num_old_blocks = parsed.header.num_blocks
    is_new = np.zeros(sizes.shape[0], bool)
    is_new[indices] = True
    filled = np.flatnonzero(sizes)
    rows = np.flatnonzero(sizes[indices] > 0)
    positions = np.searchsorted(filled, indices[rows])
    previous = np.where(positions > 0, filled[np.maximum(positions - 1, 0)], -1)
    following = np.where(positions + 1 < filled.shape[0], filled[np.minimum(positions + 1, filled.shape[0] - 1)], -1)
    stored_previous = (previous >= 0) & ~is_new[np.maximum(previous, 0)]
    stored_following = (following >= 0) & (following < num_old_blocks) & ~is_new[np.maximum(following, 0)]
    last_ticks = np.zeros(rows.shape[0], np.int64)
    steps = np.zeros(rows.shape[0], np.int64)
    last_ticks[stored_previous], steps[stored_previous] = stored_tails(parsed, time_columns, previous[stored_previous])
    return StoredNeighbours(rows, previous, following, stored_previous, stored_following, last_ticks, steps)


def stored_seeds(
    parsed: ParsedGroup, time_columns: _format.EncodedTimeColumns, sizes: np.ndarray, indices: np.ndarray
) -> tuple[StoredNeighbours, np.ndarray, np.ndarray]:
    """Builds ``snap_ticks`` clock-continuation seeds from stored blocks preceding each new block.

    Args:
        parsed: Existing parsed block group.
        time_columns: Existing group's ``EncodedTimeColumns`` (metadata columns loaded).
        sizes: 1D int64 array of spliced block sizes.
        indices: Strictly increasing 1D int64 array of replaced/appended block indices.

    Returns:
        Tuple ``(neighbours, seed_ticks, seed_steps)`` for passing to ``snap_ticks``.
    """
    neighbours = _stored_neighbours(parsed, time_columns, sizes, indices)
    seed_ticks = np.full(indices.shape[0], _time.INT64_MIN, np.int64)
    seeded = neighbours.rows[neighbours.stored_previous]
    seed_ticks[seeded] = neighbours.last_ticks[neighbours.stored_previous]
    seed_steps = np.zeros(indices.shape[0], np.int64)
    seed_steps[seeded] = neighbours.steps[neighbours.stored_previous]
    return neighbours, seed_ticks, seed_steps


def _clamp_to_neighbours(
    parsed: ParsedGroup,
    time_columns: _format.EncodedTimeColumns,
    neighbours: StoredNeighbours,
    new_offsets: np.ndarray,
    moves: np.ndarray,
    time_error: float,
    raw_ticks: np.ndarray,
    in_out_ticks: np.ndarray,
) -> None:
    """Clamps rounded new blocks against adjacent immutable stored blocks when overlap is within rounding slack.

    Args:
        parsed: Existing parsed block group.
        time_columns: Existing group's ``EncodedTimeColumns`` (metadata columns loaded).
        neighbours: Stored neighbour metadata from ``_stored_neighbours``.
        new_offsets: Cumulative sample offsets of the new blocks.
        moves: Per-block maximum displacements returned by ``snap_ticks``.
        time_error: Positive relative time error tolerance.
        raw_ticks: Unrounded input timestamps of the new blocks.
        in_out_ticks: Rounded timestamps of the new blocks, clamped in place.
    """
    # Stored blocks cannot be modified during a splice. If a new block rounded slightly past a
    # stored neighbour (within the larger of the two blocks' rounding allowances), clamp the new
    # block to the stored boundary; genuine out-of-order overlaps are left for _check_spliced_order.
    for row, new_idx in enumerate(neighbours.rows.tolist()):
        first, end = int(new_offsets[new_idx]), int(new_offsets[new_idx + 1])
        block = in_out_ticks[first:end]
        if neighbours.stored_previous[row] and block[0] < neighbours.last_ticks[row]:
            floor = int(neighbours.last_ticks[row])
            stored = _stored_allowance(parsed, time_columns, int(neighbours.previous[row]), time_error)
            if raw_ticks[first] >= floor - max(float(moves[new_idx]), stored):
                np.maximum(block, floor, out=block)
        following = int(neighbours.following[row])
        if neighbours.stored_following[row] and block[-1] > time_columns.starts[following]:
            ceiling = int(time_columns.starts[following])
            stored = _stored_allowance(parsed, time_columns, following, time_error)
            if raw_ticks[end - 1] <= ceiling + max(float(moves[new_idx]), stored):
                np.minimum(block, ceiling, out=block)


def splice(
    parsed: ParsedGroup,
    indices: np.ndarray,
    samples: np.ndarray,
    block_sizes: np.ndarray,
    ticks: np.ndarray | None,
    params: Params,
    old_time_columns: _format.EncodedTimeColumns | None = None,
    *,
    snap: bool = True,
) -> UpdatedGroup:
    """Replaces or appends blocks in ``parsed``, copying all untouched blocks verbatim at the byte level.

    Args:
        parsed: Existing parsed block group.
        indices: Strictly increasing 1D int64 array of replaced/appended block indices.
        samples: 1D float64 array of new blocks' samples in ``indices`` order.
        block_sizes: 1D int64 array of new blocks' sample counts.
        ticks: 1D int64 array of new blocks' timestamps, or None iff ``parsed`` is untimed.
        params: Encoder configuration parameters.
        old_time_columns: Optional pre-read ``EncodedTimeColumns`` for ``parsed``.
        snap: Whether to apply ``params.time_error`` rounding and neighbour clamping (False if
            the caller already rounded).

    Returns:
        ``UpdatedGroup`` with the new compressed bytes, re-encoded block indices (including any
        empty gap blocks appended), and their summary statistics.

    Raises:
        ValueError: If group counts exceed limits or timestamps decrease across any boundary.
    """
    # Fill any skipped indices past num_old_blocks with empty blocks, optionally snap and clamp
    # new timestamps against stored neighbours, encode only the new blocks, verify cross-block
    # monotonicity, convert old body to byte planes if needed, splice body and escapes, and repack.
    num_old_blocks = parsed.header.num_blocks
    num_blocks = max(num_old_blocks, int(indices[-1]) + 1) if indices.shape[0] else num_old_blocks
    _args.check_group_counts(num_blocks, 0)
    gaps = np.setdiff1d(np.arange(num_old_blocks, num_blocks), indices) if num_blocks > num_old_blocks else indices[:0]
    if gaps.shape[0]:
        merged = np.union1d(indices, gaps)
        merged_sizes = np.zeros(merged.shape[0], np.int64)
        merged_sizes[np.searchsorted(merged, indices)] = block_sizes
        indices, block_sizes = merged, merged_sizes

    sizes = np.zeros(num_blocks, np.int64)
    sizes[:num_old_blocks] = parsed.block_sizes
    sizes[indices] = block_sizes
    num_samples = int(sizes.sum())
    _args.check_group_counts(num_blocks, num_samples)
    new_offsets = sample_offsets(block_sizes)
    if ticks is not None and snap and params.time_error > 0:
        if old_time_columns is None:
            old_time_columns = read_time_columns(parsed, np.zeros(0, np.int64))
        neighbours, seed_ticks, seed_steps = stored_seeds(parsed, old_time_columns, sizes, indices)
        raw_ticks = ticks
        ticks, moves = snap_ticks(ticks, block_sizes, params.time_error, seed_ticks, seed_steps)
        if ticks is raw_ticks:
            ticks = ticks.copy()
        _clamp_to_neighbours(
            parsed, old_time_columns, neighbours, new_offsets, moves, params.time_error, raw_ticks, ticks
        )

    new_columns, stats = encode_columns(samples, block_sizes, params, ticks)
    old_starts = None
    if ticks is not None:
        new_time_columns = new_columns.time_columns
        assert new_time_columns is not None
        if old_time_columns is None:
            old_time_columns = read_time_columns(parsed, np.zeros(0, np.int64))
        old_starts = old_time_columns.starts
        is_new = np.zeros(num_blocks, bool)
        is_new[indices] = True
        starts = np.zeros(num_blocks, np.int64)
        starts[:num_old_blocks] = old_starts
        starts[indices] = new_time_columns.starts
        new_last_ticks = np.zeros(num_blocks, np.int64)
        filled = block_sizes > 0
        new_last_ticks[indices[filled]] = ticks[new_offsets[1:][filled] - 1]
        _check_spliced_order(parsed, old_time_columns, sizes, starts, is_new, new_last_ticks)

    has_time = parsed.has_time
    old_body = parsed.body
    # Body splicing operates on byte-planed residuals; transpose first if stored in bit planes.
    if not parsed.header.byte_planes:
        old_body = _bitpacking.to_byte_planes(
            old_body, num_old_blocks, int(parsed.layout.plane_byte_offsets[-1]), has_time
        )
    body, offsets = _body.splice_body(old_body, parsed.layout, old_starts, indices, new_columns, new_offsets)
    escaped_values = _escapes.splice_escaped_values(
        parsed.escaped_values,
        parsed.escape_offsets,
        num_old_blocks,
        num_blocks,
        indices,
        new_columns.block_flags,
        new_columns.sample_codes,
        new_columns.escaped_values,
        new_offsets,
    )
    if escaped_values.shape[0]:
        body = np.concatenate([body, _escapes.pack_escaped_values(escaped_values)])
    group = _compress.pack_body(
        body, offsets, num_blocks, num_samples, _compress.EFFORTS[params.effort], parsed.header.time_unit
    )
    return UpdatedGroup(group, indices, *stats)


def update(
    group: bytes,
    blocks: Mapping[int, npt.ArrayLike],
    params: Params,
    times: Mapping[int, npt.ArrayLike] | None,
    time_unit: str | None = None,
) -> UpdatedGroup:
    """Validates block-update arguments and splices replacement/appended blocks into ``group``.

    Args:
        group: Existing compressed block group bytes.
        blocks: Mapping from block index to 1D sample array-like.
        params: Encoder configuration parameters.
        times: Optional mapping from block index to 1D timestamp array-like (required iff timed).
        time_unit: Optional time unit string checked against the stored group's unit.

    Returns:
        ``UpdatedGroup`` with the updated bytes, modified block indices, and block statistics.

    Raises:
        ValueError: If ``group`` is corrupt, arguments are invalid, or timestamps are out of order.
        zstandard.ZstdError: If the zstd frame is corrupt.
    """
    parsed = decompress(group)
    _args.check_stored_time_unit(time_unit, parsed.header.time_unit)
    indices, samples, sizes, ticks = _args.update_blocks(blocks, times, parsed.header.time_unit)
    if not indices.shape[0]:
        return UpdatedGroup(group, np.zeros(0, np.int64), np.zeros(0), np.zeros(0), np.zeros(0))
    return splice(parsed, indices, samples, sizes, ticks, params)
