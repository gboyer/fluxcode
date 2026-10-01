# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Public API for fluxcode encoding, decoding, and updating time series.

Provides entry points for compressing float64 arrays, optionally with their
timestamps, into independent, self-describing units (storage rows) and
decompressing them back. Encoding also returns per-block summary statistics
(min, max, and mean), which decoding doesn't need.

Blocks can have any size from 0 to 65,535 samples. Three ways to divide a series into them:

- encode_unit / encode: fixed-size blocks of dense, regularly sampled data (the last block
  may be short).
- encode_blocks: explicit block sizes.
- encode_time_blocks / update_time_blocks: blocks of a fixed duration of time, for data
  that arrives incomplete or late.

The implementation is in _unit (units and blocks) and _time_blocks (time division).
"""

from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt

from . import _format, _time_blocks, _unit
from ._time_blocks import DurationLike, TimeLike
from ._types import (
    DEFAULT_PARAMS,
    DecodedUnit,
    EncodedSeries,
    EncodedUnit,
    Params,
    TimeUnit,
    UpdatedUnit,
    is_int,
)

DEFAULT_BLOCK_LEN: int = 1000
"""Default samples per block of encode_unit and encode."""

DEFAULT_BLOCKS_PER_UNIT: int = 60
"""Default blocks per unit of encode: 60 blocks of 1000 make a unit one minute at 1 kHz."""


def _fixed_sizes(num_samples: int, block_len: int) -> np.ndarray:
    """Block sizes of num_samples in blocks of block_len, the last one short if need be.

    Raises:
        ValueError: If block_len is not from 1 to 65,535.
    """
    if not (is_int(block_len) and 1 <= block_len <= _format.MAX_BLOCK_LEN):
        raise ValueError(f"block_len must be from 1 to {_format.MAX_BLOCK_LEN}, got {block_len!r}")
    num_blocks = -(-num_samples // block_len)
    sizes = np.full(num_blocks, block_len, np.int64)
    if num_blocks:
        sizes[-1] = num_samples - (num_blocks - 1) * block_len
    return sizes


def encode_unit(
    x: npt.ArrayLike,
    params: Params = DEFAULT_PARAMS,
    *,
    block_len: int = DEFAULT_BLOCK_LEN,
    times: npt.ArrayLike | None = None,
    time_unit: TimeUnit | None = None,
) -> EncodedUnit:
    """Encodes a time series into a single compressed unit (one storage row).

    Divides x into blocks of block_len samples; the last block holds the rest and may be
    short. The unit records every block's size and its sample count, so decode_unit needs
    nothing else. Non-finite values (NaN, +inf, -inf) are encoded exactly (canonical quiet
    NaN).

    With times, the unit also stores the samples' timestamps exactly: naive (no time
    zone; UTC is recommended, since local time can repeat or skip), non-decreasing, and
    without NaT. Equal consecutive timestamps are allowed.

    Args:
        x: 1D array-like of float64 samples.
        params: Encoder configuration parameters.
        block_len: Samples per block, 1 to 65,535.
        times: Optional timestamps, one per sample: a datetime64[s|ms|us|ns] array (the
            unit is taken from the dtype), or integer ticks with time_unit.
        time_unit: Unit of integer times ('s', 'ms', 'us' or 'ns'); must be None for
            datetime64 times.

    Returns:
        EncodedUnit tuple (unit, block_min, block_max, block_mean) containing:
            unit: Self-describing unit bytes.
            block_min: 1D float64 array of minimum finite values per block.
            block_max: 1D float64 array of maximum finite values per block.
            block_mean: 1D float64 array of finite means per block.
            Blocks containing only non-finite samples receive NaN for min, max,
            and mean. These summary statistics come free with encoding; decoding
            doesn't use them.

    Raises:
        ValueError: If x is empty or needs more than 65,535 blocks or 2^26 samples, or the
            times are invalid (see above) or don't match x in length.
    """
    series_arr = _unit.as_series(x)
    sizes = _fixed_sizes(series_arr.shape[0], block_len)
    ticks, time_unit_code = _unit.series_ticks(times, time_unit, series_arr.shape[0])
    return _unit.encode(series_arr, sizes, params, ticks, time_unit_code)


def encode(
    x: npt.ArrayLike,
    params: Params = DEFAULT_PARAMS,
    *,
    block_len: int = DEFAULT_BLOCK_LEN,
    blocks_per_unit: int = DEFAULT_BLOCKS_PER_UNIT,
    times: npt.ArrayLike | None = None,
    time_unit: TimeUnit | None = None,
) -> EncodedSeries:
    """Encodes an arbitrary-length series into a sequence of compressed units.

    Partitions the input series (and its times, if given) into chunks of
    blocks_per_unit blocks of block_len samples and encodes each chunk as an independent
    unit via encode_unit.

    Args:
        x: 1D array-like of float64 samples.
        params: Encoder configuration parameters.
        block_len: Samples per block, 1 to 65,535.
        blocks_per_unit: Blocks per unit (the last unit may hold fewer), at least 1.
        times: Optional timestamps, one per sample (see encode_unit).
        time_unit: Unit of integer times (see encode_unit).

    Returns:
        EncodedSeries tuple (units, block_mins, block_maxs, block_means):
            units: List of compressed unit byte strings.
            block_mins: List of 1D float64 arrays of block minima.
            block_maxs: List of 1D float64 arrays of block maxima.
            block_means: List of 1D float64 arrays of block means.

    Raises:
        ValueError: If x is empty, a unit would be too large, or the times are invalid or
            don't match x in length.
    """
    series_arr = _unit.as_series(x)
    _fixed_sizes(0, block_len)
    if not (is_int(blocks_per_unit) and blocks_per_unit >= 1):
        raise ValueError(f"blocks_per_unit must be >= 1, got {blocks_per_unit!r}")
    ticks, time_unit_code = _unit.series_ticks(times, time_unit, series_arr.shape[0])
    # Samples per full unit
    samples_per_unit = blocks_per_unit * block_len
    _unit.check_unit_counts(blocks_per_unit, min(samples_per_unit, series_arr.shape[0]))
    if ticks is not None:
        # Each unit checks its own times: check the unit boundaries here
        unit_starts = np.arange(samples_per_unit, ticks.shape[0], samples_per_unit)
        decreases = unit_starts[ticks[unit_starts] < ticks[unit_starts - 1]]
        if decreases.shape[0]:
            raise _unit.decrease_error(ticks, int(decreases[0]))
    parts = []
    for idx in range(0, series_arr.shape[0], samples_per_unit):
        chunk = series_arr[idx:idx + samples_per_unit]
        parts.append(_unit.encode(
            chunk,
            _fixed_sizes(chunk.shape[0], block_len),
            params,
            None if ticks is None else ticks[idx:idx + samples_per_unit],
            time_unit_code,
        ))
    units, mins, maxs, means = (list(col) for col in zip(*parts))
    return EncodedSeries(units, mins, maxs, means)


def encode_blocks(
    x: npt.ArrayLike,
    block_sizes: npt.ArrayLike,
    params: Params = DEFAULT_PARAMS,
    *,
    times: npt.ArrayLike | None = None,
    time_unit: TimeUnit | None = None,
) -> EncodedUnit:
    """Encodes a series divided into blocks of the given sizes into a single unit.

    The blocks are given as one flat array of samples and the size of each block (as in
    Arrow list arrays): block b holds x[offsets[b]:offsets[b + 1]], with offsets the
    cumulative sum of block_sizes from 0. Blocks of a list of arrays are
    encode_blocks(np.concatenate(blocks), [len(b) for b in blocks]).

    Each block can hold 0 to 65,535 samples. An empty block stores nothing and gets NaN
    statistics. A block of at most 8 samples skips the analysis: it is stored at the finest
    step (max_quantize_bits), without differences, on a decimal grid if its samples sit on one
    (so decimal data stays exact) and otherwise on the power-of-two grid.

    Args:
        x: 1D array-like of float64 samples, block after block.
        block_sizes: 1D integer array-like of each block's sample count; they must add up to
            len(x).
        params: Encoder configuration parameters.
        times: Optional timestamps, one per sample (see encode_unit).
        time_unit: Unit of integer times (see encode_unit).

    Returns:
        EncodedUnit tuple (unit, block_min, block_max, block_mean), one statistic per block
        (see encode_unit).

    Raises:
        ValueError: If a block size is out of range, the sizes don't add up to len(x), the
            unit would hold more than 65,535 blocks or 2^26 samples, or the times are
            invalid or don't match x in length.
    """
    series_arr = _unit.as_series(x, allow_empty=True)
    sizes = _unit.as_block_sizes(block_sizes)
    if int(sizes.sum()) != series_arr.shape[0]:
        raise ValueError(f"block_sizes add up to {int(sizes.sum())}, not the {series_arr.shape[0]} samples of x")
    ticks, time_unit_code = _unit.series_ticks(times, time_unit, series_arr.shape[0])
    return _unit.encode(series_arr, sizes, params, ticks, time_unit_code)


def encode_time_blocks(
    x: npt.ArrayLike,
    times: npt.ArrayLike,
    params: Params = DEFAULT_PARAMS,
    *,
    start_time: TimeLike,
    block_duration: DurationLike,
    time_unit: TimeUnit | None = None,
) -> EncodedUnit:
    """Encodes a timed series into a single unit of blocks of a fixed duration.

    Block b holds the samples timed in [start_time + b * block_duration,
    start_time + (b + 1) * block_duration): for example a unit of one hour from start_time
    in blocks of one minute. Blocks without samples are empty, and the unit ends with the
    last block that holds a sample. start_time and block_duration aren't stored in the unit:
    keep them (typically start_time is part of the unit's storage key) and pass the same
    ones to update_time_blocks.

    Args:
        x: 1D array-like of float64 samples (may be empty).
        times: Their timestamps (see encode_unit): non-decreasing, at or after start_time.
        params: Encoder configuration parameters.
        start_time: Start of block 0: a datetime64, a naive datetime (or pandas Timestamp),
            or integer ticks in the times' unit. It must be a whole number of ticks.
        block_duration: Length of every block: a timedelta64, a timedelta (or pandas
            Timedelta), or integer ticks. It must be a positive whole number of ticks.
        time_unit: Unit of integer times (see encode_unit).

    Returns:
        EncodedUnit tuple (unit, block_min, block_max, block_mean), one statistic per block
        (NaN for an empty block).

    Raises:
        ValueError: If a time is before start_time or 65,535 blocks or more after it, a
            block would hold more than 65,535 samples, the unit more than 2^26, or the
            times or time arguments are invalid.
    """
    return _time_blocks.encode_time_blocks(x, times, params, start_time, block_duration, time_unit)


def update_time_blocks(
    unit: bytes,
    x: npt.ArrayLike,
    times: npt.ArrayLike,
    params: Params = DEFAULT_PARAMS,
    *,
    start_time: TimeLike,
    block_duration: DurationLike,
    update_ranges: object,
    time_unit: TimeUnit | None = None,
) -> UpdatedUnit:
    """Replaces the samples in time ranges of a unit made by encode_time_blocks.

    Every existing sample timed within update_ranges is discarded, and the samples of x
    (which must all be timed within them) take their place. start_time and block_duration
    must be the ones the unit was encoded with: they aren't stored in it, so they can't be
    checked.

    Blocks whose time span doesn't meet a range are carried over untouched, without being
    decoded or re-encoded. Blocks wholly inside the ranges are encoded from the new samples
    alone. Blocks that straddle a range boundary are decoded, their samples outside the
    ranges kept and merged with the new ones, and re-encoded: those kept samples move to
    the merged block's grid, within its error bound. Blocks are appended (empty ones to
    fill a gap) when new samples are timed past the unit's end; blocks never are removed.

    Args:
        unit: Existing unit bytes, with a time axis.
        x: 1D array-like of float64 samples (may be empty: then the update only discards).
        times: Their timestamps, non-decreasing: datetime64 in the unit's time unit, or
            integer ticks in it.
        params: Encoder configuration parameters.
        start_time: Start of block 0 (see encode_time_blocks).
        block_duration: Length of every block (see encode_time_blocks).
        update_ranges: The time ranges to replace, each [start, end) with start and end
            like start_time: one (start, end) pair, an iterable of pairs, or an array of
            shape (k, 2) (datetime64 or integer ticks). Ranges may overlap.
        time_unit: Optional: the unit's time unit, checked if given.

    Returns:
        UpdatedUnit tuple (unit, indices, block_min, block_max, block_mean): the new unit,
        and the blocks re-encoded (including emptied and appended ones) with their
        statistics. If nothing changes, the unit is returned as is and indices is empty.

    Raises:
        ValueError: If the unit is corrupt or has no time axis, a sample is timed outside
            update_ranges or before start_time, the times decrease, a block or the unit would
            hold too many samples, or the time arguments are invalid.
        zstandard.ZstdError: If the zstd frame is corrupt.
    """
    return _time_blocks.update_time_blocks(
        unit, x, times, params, start_time, block_duration, update_ranges, time_unit
    )


def decode_unit(unit: bytes) -> DecodedUnit:
    """Decodes a single self-describing unit into its samples, timestamps and block sizes.

    Args:
        unit: Unit bytes, as returned by any encode or update function.

    Returns:
        DecodedUnit tuple (values, times, block_sizes):
            values: Reconstructed 1D float64 array of the unit's samples.
            times: 1D datetime64 array of the same length, in the unit the times were
                encoded with; None if the unit was encoded without times.
            block_sizes: 1D int64 array of each block's sample count.

    Raises:
        ValueError: If the header or body is invalid or out of range.
        zstandard.ZstdError: If the zstd frame is corrupt.
    """
    return _unit.decode(unit)


def decode(units: Sequence[bytes]) -> list[DecodedUnit]:
    """Decodes multiple independent units into their samples and timestamps.

    Args:
        units: Sequence of unit bytes.

    Returns:
        List of DecodedUnit tuples (values, times, block_sizes), one per unit.

    Raises:
        ValueError: If any unit's header or body is invalid (see decode_unit).
        zstandard.ZstdError: If any unit's zstd frame is corrupt.
    """
    return [_unit.decode(unit) for unit in units]


def update(
    unit: bytes,
    blocks: Mapping[int, npt.ArrayLike],
    params: Params = DEFAULT_PARAMS,
    *,
    times: Mapping[int, npt.ArrayLike] | None = None,
) -> UpdatedUnit:
    """Replaces or appends whole blocks within an existing unit.

    Untouched blocks retain their exact residuals, grid parameters, value anchors,
    non-finite codes and times: they are carried over without being decoded or
    re-quantized. A new block can have any size (0 to 65,535), whatever the size of the
    block it replaces. Blocks past the unit's end are appended; any it skips are appended
    empty.

    Args:
        unit: Existing unit bytes.
        blocks: The new blocks by index: replacing a block (0 to num_blocks - 1) or
            appending one (num_blocks or more), each a 1D array-like of float64 samples.
        params: Encoder configuration parameters.
        times: The new blocks' timestamps by the same indices, each with one per sample:
            required if and only if the unit has a time axis. datetime64 in the unit's time
            unit, or integer ticks in it. The updated unit's times must be non-decreasing
            throughout.

    Returns:
        UpdatedUnit tuple (unit, indices, block_min, block_max, block_mean): the new unit,
        and the blocks re-encoded (the keys of blocks, and any empty blocks appended to fill
        a gap) in increasing order with their statistics.

    Raises:
        ValueError: If unit is corrupt, an index is negative, a block is not 1-D or too
            large, the unit would be too large, or times are missing, unexpected, mis-keyed,
            mis-shaped or out of order.
        zstandard.ZstdError: If the zstd frame is corrupt.
    """
    parsed = _unit.decompress(unit)
    if not isinstance(blocks, Mapping):
        raise ValueError(f"blocks must map block indices to samples, got {type(blocks).__name__}")  # noqa: TRY004
    if not all(is_int(idx) and idx >= 0 for idx in blocks):
        raise ValueError(f"block indices must be integers >= 0, got {sorted(map(repr, blocks))}")
    indices = sorted(int(idx) for idx in blocks)
    if parsed.has_time:
        if times is None:
            raise ValueError("the unit has a time axis: times are required")
        if sorted(int(idx) for idx in times) != indices:
            raise ValueError("times must have the same block indices as blocks")
    elif times is not None:
        raise ValueError("the unit has no time axis: times must be None")
    by_index = {int(idx): block for idx, block in blocks.items()}
    samples = [_unit.as_series(by_index[idx], allow_empty=True) for idx in indices]
    sizes = _unit.as_block_sizes([block.shape[0] for block in samples]) if samples else np.zeros(0, np.int64)
    ticks = None
    if times is not None:
        times_by_index = {int(idx): block_times for idx, block_times in times.items()}
        block_ticks = [_unit.as_ticks(times_by_index[idx], None, parsed.header.time_unit)[0] for idx in indices]
        for idx, block, tick_block in zip(indices, samples, block_ticks):
            if tick_block.shape != block.shape:
                raise ValueError(f"block {idx}: times must have the shape of its samples {block.shape}, got {tick_block.shape}")
        ticks = np.concatenate(block_ticks) if block_ticks else np.zeros(0, np.int64)
    if not indices:
        return UpdatedUnit(unit, np.zeros(0, np.int64), np.zeros(0), np.zeros(0), np.zeros(0))
    return _unit.splice(
        parsed, np.array(indices, np.int64), np.concatenate(samples), sizes, ticks, params
    )
