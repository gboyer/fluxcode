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

The functions come in three groups: encoding, decoding and updating. They validate their
arguments and call _unit (units and blocks) or _time_blocks (time division).
"""

from collections.abc import Mapping, Sequence

import numpy.typing as npt

from . import _args, _time_blocks, _unit
from ._args import DurationLike, RangesLike, TimeLike
from ._types import (
    DEFAULT_PARAMS,
    DecodedUnit,
    EncodedSeries,
    EncodedUnit,
    Params,
    TimeUnit,
    UpdatedUnit,
)

DEFAULT_BLOCK_LEN: int = 1000
"""Default samples per block of encode_unit and encode."""

DEFAULT_BLOCKS_PER_UNIT: int = 60
"""Default blocks per unit of encode: 60 blocks of 1000 make a unit one minute at 1 kHz."""


# Encoding

def encode_unit(
    samples: npt.ArrayLike,
    params: Params = DEFAULT_PARAMS,
    *,
    block_len: int = DEFAULT_BLOCK_LEN,
    times: npt.ArrayLike | None = None,
    time_unit: TimeUnit | None = None,
) -> EncodedUnit:
    """Encodes a time series into a single compressed unit (one storage row).

    Divides samples into blocks of block_len samples; the last block holds the rest and may be
    short. The unit records every block's size and its sample count, so decode_unit needs
    nothing else. Non-finite values (NaN, +inf, -inf) are encoded exactly (canonical quiet
    NaN).

    With times, the unit also stores the samples' timestamps exactly: naive (no time
    zone; UTC is recommended, since local time can repeat or skip), non-decreasing, and
    without NaT. Equal consecutive timestamps are allowed.

    Args:
        samples: 1D array-like of float64 samples.
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
            doesn't use them. They are of the given samples: decoded values can be up
            to half a step away, the minimum included.

    Raises:
        ValueError: If samples is empty or needs more than 65,535 blocks or 2^26 samples, or the
            times are invalid (see above) or don't match samples in length.
    """
    series_arr = _args.as_series(samples)
    sizes = _args.fixed_sizes(series_arr.shape[0], block_len)
    ticks, time_unit_code = _args.series_ticks(times, time_unit, series_arr.shape[0])
    return _unit.encode(series_arr, sizes, params, ticks, time_unit_code)


def encode_blocks(
    samples: npt.ArrayLike,
    block_sizes: npt.ArrayLike,
    params: Params = DEFAULT_PARAMS,
    *,
    times: npt.ArrayLike | None = None,
    time_unit: TimeUnit | None = None,
) -> EncodedUnit:
    """Encodes a series divided into blocks of the given sizes into a single unit.

    The blocks are given as one flat array of samples and the size of each block (as in
    Arrow list arrays): block b holds samples[offsets[b]:offsets[b + 1]], with offsets the
    cumulative sum of block_sizes from 0. Blocks of a list of arrays are
    encode_blocks(np.concatenate(blocks), [len(b) for b in blocks]).

    Each block can hold 0 to 65,535 samples. An empty block stores nothing and gets NaN
    statistics. A block of at most 8 samples is stored at the finest step
    (max_quantize_bits), so decimal data in it stays exact.

    Args:
        samples: 1D array-like of float64 samples, block after block.
        block_sizes: 1D integer array-like of each block's sample count; they must add up to
            len(samples).
        params: Encoder configuration parameters.
        times: Optional timestamps, one per sample (see encode_unit).
        time_unit: Unit of integer times (see encode_unit).

    Returns:
        EncodedUnit tuple (unit, block_min, block_max, block_mean), one statistic per block
        (see encode_unit).

    Raises:
        ValueError: If a block size is out of range, the sizes don't add up to len(samples), the
            unit would hold more than 65,535 blocks or 2^26 samples, or the times are
            invalid or don't match samples in length.
    """
    series_arr = _args.as_series(samples, allow_empty=True)
    sizes = _args.as_block_sizes(block_sizes)
    if int(sizes.sum()) != series_arr.shape[0]:
        raise ValueError(f"block_sizes add up to {int(sizes.sum())}, not the {series_arr.shape[0]} samples of input")
    ticks, time_unit_code = _args.series_ticks(times, time_unit, series_arr.shape[0])
    return _unit.encode(series_arr, sizes, params, ticks, time_unit_code)


def encode_time_blocks(
    samples: npt.ArrayLike,
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
        samples: 1D array-like of float64 samples (may be empty).
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
    return _time_blocks.encode_time_blocks(samples, times, params, start_time, block_duration, time_unit)


def encode(
    samples: npt.ArrayLike,
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
        samples: 1D array-like of float64 samples.
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
        ValueError: If samples is empty, a unit would be too large, or the times are invalid or
            don't match samples in length.
    """
    series_arr = _args.as_series(samples)
    _args.check_chunking(block_len, blocks_per_unit)
    ticks, time_unit_code = _args.series_ticks(times, time_unit, series_arr.shape[0])
    return _unit.encode_series(series_arr, block_len, blocks_per_unit, ticks, time_unit_code, params)


# Decoding

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


# Updating

def update(
    unit: bytes,
    blocks: Mapping[int, npt.ArrayLike],
    params: Params = DEFAULT_PARAMS,
    *,
    times: Mapping[int, npt.ArrayLike] | None = None,
    time_unit: TimeUnit | None = None,
) -> UpdatedUnit:
    """Replaces or appends whole blocks within an existing unit.

    Untouched blocks are carried over exactly (values, non-finite values and times),
    without being decoded or re-quantized. A new block can have any size (0 to 65,535),
    whatever the size of the block it replaces. Blocks past the unit's end are appended;
    any it skips are appended empty.

    Args:
        unit: Existing unit bytes.
        blocks: The new blocks by index: replacing a block (0 to num_blocks - 1) or
            appending one (num_blocks or more), each a 1D array-like of float64 samples.
        params: Encoder configuration parameters.
        times: The new blocks' timestamps by the same indices, each with one per sample:
            required if and only if the unit has a time axis. datetime64 in the unit's time
            unit, or integer ticks in it. The updated unit's times must be non-decreasing
            throughout.
        time_unit: Optional: the unit's time unit, checked if given.

    Returns:
        UpdatedUnit tuple (unit, indices, block_min, block_max, block_mean): the new unit,
        and the blocks re-encoded (the keys of blocks, and any empty blocks appended to fill
        a gap) in increasing order with their statistics.

    Raises:
        ValueError: If unit is corrupt, an index is negative, a block is not 1-D or too
            large, the unit would be too large, time_unit isn't the unit's, or times are missing,
            unexpected, mis-keyed, mis-shaped or out of order.
        zstandard.ZstdError: If the zstd frame is corrupt.
    """
    return _unit.update(unit, blocks, params, times, time_unit)


def update_time_blocks(
    unit: bytes,
    samples: npt.ArrayLike,
    times: npt.ArrayLike,
    params: Params = DEFAULT_PARAMS,
    *,
    start_time: TimeLike,
    block_duration: DurationLike,
    delete_ranges: RangesLike | None = None,
    time_unit: TimeUnit | None = None,
) -> UpdatedUnit:
    """Upserts samples into, and deletes time ranges from, a unit made by encode_time_blocks.

    First every existing sample timed within delete_ranges is discarded. Then the new
    samples are added: an existing sample with the same timestamp as a new one is replaced,
    and new samples sharing a timestamp are all kept (in their order). So with only samples
    it is an upsert, with only delete_ranges a deletion, and with both a range replacement
    (new samples may be timed anywhere; those in a deleted range are kept, as they're added
    after the deletion). start_time and block_duration must be the ones the unit was
    encoded with: they aren't stored in it, so they can't be checked.

    Blocks that no range meets and no new sample falls in are carried over untouched,
    without being decoded or re-encoded. Other affected blocks are re-encoded from their
    surviving samples and the new ones. A surviving sample comes back unchanged unless the
    block's new quantization step is coarser than the one it was stored with. However often
    a block is updated, a sample's error stays under one step of the coarsest quantization
    the block has used (decimal data on a decimal grid stays exact). Blocks are appended
    (empty ones to fill a gap) when new samples are timed past the unit's end; blocks are
    never removed.

    Args:
        unit: Existing unit bytes, with a time axis.
        samples: 1D array-like of float64 samples (may be empty: then the update only deletes).
        times: Their timestamps, non-decreasing: datetime64 in the unit's time unit, or
            integer ticks in it.
        params: Encoder configuration parameters.
        start_time: Start of block 0 (see encode_time_blocks).
        block_duration: Length of every block (see encode_time_blocks).
        delete_ranges: Optional time ranges to delete, each [start, end) with start and end
            like start_time: one (start, end) pair, an iterable of pairs, or an array of
            shape (k, 2) (datetime64 or integer ticks). Ranges may overlap.
        time_unit: Optional: the unit's time unit, checked if given.

    Returns:
        UpdatedUnit tuple (unit, indices, block_min, block_max, block_mean): the new unit,
        and the blocks re-encoded (including emptied and appended ones) with their
        statistics. If nothing is deleted and no samples are given, the unit is returned as is and indices
        is empty.

    Raises:
        ValueError: If the unit is corrupt or has no time axis, a sample is timed before
            start_time, the times decrease, a block or the unit would
            hold too many samples, or the time arguments are invalid.
        zstandard.ZstdError: If the zstd frame is corrupt.
    """
    return _time_blocks.update_time_blocks(
        unit, samples, times, params, start_time, block_duration, delete_ranges, time_unit
    )
