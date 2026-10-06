# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Validates and converts the public functions' arguments.

This is the layer between the public API (_api) and the code that encodes, decodes and
updates (_group, _time_blocks). Samples become contiguous float64 arrays, times int64 ticks
with a time unit code, and block sizes int64 arrays.

The argument errors of the public functions come from here, apart from the checks that
need the data itself: non-decreasing times, and the block bounds of timed samples
(_time_blocks.block_ids).
"""

import datetime
from collections.abc import Iterable, Mapping
from fractions import Fraction

import numpy as np
import numpy.typing as npt

from . import _format, _time
from ._types import is_int

TimeLike = np.datetime64 | datetime.datetime | int
"""A point in time: datetime64 or datetime (naive, UTC recommended), or an integer tick."""

DurationLike = np.timedelta64 | datetime.timedelta | int
"""A length of time: timedelta64 or timedelta, or an integer number of ticks."""

RangesLike = tuple[TimeLike, TimeLike] | Iterable[tuple[TimeLike, TimeLike]] | np.ndarray
"""Time ranges, each [start, end): one (start, end) pair, an iterable of pairs, or an array of
shape (k, 2) of datetime64 or integer ticks."""

_UNIT_NANOSECONDS: dict[str, Fraction] = {
    "W": Fraction(604_800 * 10**9),
    "D": Fraction(86_400 * 10**9),
    "h": Fraction(3_600 * 10**9),
    "m": Fraction(60 * 10**9),
    "s": Fraction(10**9),
    "ms": Fraction(10**6),
    "us": Fraction(10**3),
    "ns": Fraction(1),
    "ps": Fraction(1, 10**3),
    "fs": Fraction(1, 10**6),
    "as": Fraction(1, 10**9),
}
"""Length of each fixed-length numpy datetime unit in nanoseconds (months and years vary)."""


def as_series(series_input: npt.ArrayLike, allow_empty: bool = False) -> np.ndarray:
    """Validates and converts input into a contiguous 1D float64 array.

    Args:
        series_input: Array-like input.
        allow_empty: Whether an empty array is accepted.

    Returns:
        Contiguous 1D float64 numpy array.

    Raises:
        ValueError: If series_input is not 1D, or is empty and allow_empty is False.
    """
    series_array = np.ascontiguousarray(series_input, dtype=np.float64)
    # A scalar becomes a 1-element array here, so ndim catches only higher dimensions
    if series_array.ndim != 1 or (series_array.shape[0] == 0 and not allow_empty):
        raise ValueError(f"samples must be a {'' if allow_empty else 'non-empty '}1-D array, got shape {series_array.shape}")
    return series_array


def as_ticks(times: npt.ArrayLike, time_unit: str | None, stored_unit: int = 0) -> tuple[np.ndarray, int]:
    """Converts timestamps into contiguous int64 ticks and their time unit code.

    Args:
        times: datetime64[s|ms|us|ns] array (the unit is taken from the dtype), or an
            integer array of ticks in time_unit.
        time_unit: Block group of integer ticks ('s', 'ms', 'us' or 'ns'); None for datetime64.
        stored_unit: For update: the block group's time unit code, which datetime64 times must
            match and integer ticks are in (time_unit, if given, must match it too). 0 when
            encoding.

    Returns:
        A tuple of (ticks, time_unit_code): a contiguous int64 array with the input's shape
        and the time unit code.

    Raises:
        ValueError: If the dtype isn't datetime64[s|ms|us|ns] or integer, time_unit is
            missing for integer ticks or given for datetime64, the block group doesn't match
            stored_unit, or an unsigned tick exceeds int64.
    """
    times_array = np.asarray(times)
    if times_array.dtype.kind == "M":
        if time_unit is not None:
            raise ValueError("time_unit applies to integer times only: datetime64 times carry their own block group")
        dtype_unit, unit_count = np.datetime_data(times_array.dtype)
        if unit_count != 1 or dtype_unit not in _format.TIME_UNIT_CODES:
            raise ValueError(f"times must be datetime64 in s, ms, us or ns, got {times_array.dtype}")
        unit_code = _format.TIME_UNIT_CODES[dtype_unit]
        ticks = times_array.view(np.int64)
    elif times_array.dtype.kind in "iu" or (times_array.size == 0 and times_array.dtype.kind == "f"):
        # An empty list converts to float64: take it as integer ticks
        unit_code = time_unit_code(time_unit, stored_unit)
        if times_array.dtype.kind == "u" and times_array.size and int(times_array.max()) > _time.INT64_MAX:
            raise ValueError("integer times must fit in int64")
        ticks = times_array.astype(np.int64, copy=False)
    else:
        raise ValueError(
            f"times must be datetime64 (s, ms, us or ns) or integer ticks with time_unit, got dtype {times_array.dtype}"
            + (" (time zone aware times aren't supported: convert to naive UTC)" if times_array.dtype == object else "")
        )
    if stored_unit and unit_code != stored_unit:
        raise ValueError(
            f"times are in {_format.TIME_UNIT_NAMES[unit_code]} but the block group stores "
            f"{_format.TIME_UNIT_NAMES[stored_unit]}"
        )
    return np.ascontiguousarray(ticks), unit_code


def time_unit_code(time_unit: str | None, stored_unit: int = 0) -> int:
    """Resolves the time unit code of integer timestamps.

    Args:
        time_unit: Name of the time unit ('s', 'ms', 'us', or 'ns'), or None to
            use stored_unit.
        stored_unit: Time unit code stored in existing block group, used when time_unit is
            None.

    Returns:
        Integer time unit code.

    Raises:
        ValueError: If time_unit is invalid, or None when no stored_unit is provided.
    """
    unit_names = "', '".join(_format.TIME_UNIT_CODES)
    if time_unit is None:
        if not stored_unit:
            raise ValueError(f"integer times need time_unit ('{unit_names}')")
        return stored_unit
    if time_unit not in _format.TIME_UNIT_CODES:
        raise ValueError(f"time_unit must be one of '{unit_names}', got {time_unit!r}")
    return _format.TIME_UNIT_CODES[time_unit]


def series_ticks(
    times: npt.ArrayLike | None, time_unit: str | None, num_samples: int
) -> tuple[np.ndarray | None, int]:
    """Converts timestamp input into a 1D int64 array of ticks and a block group code.

    Args:
        times: Array-like timestamps, or None if no timestamps are present.
        time_unit: Block group string for integer timestamps; None for datetime64.
        num_samples: Expected sample count matching the series length.

    Returns:
        A tuple of (ticks, unit_code): 1D int64 array of timestamp ticks (or None)
        and the integer time unit code.

    Raises:
        ValueError: If times length does not match num_samples or arguments are
            invalid.
    """
    if times is None:
        if time_unit is not None:
            raise ValueError("time_unit needs times")
        return None, 0
    ticks, unit_code = as_ticks(times, time_unit)
    if ticks.shape != (num_samples,):
        raise ValueError(f"times must be 1-D with one entry per sample ({num_samples}), got shape {ticks.shape}")
    return ticks, unit_code


def as_block_sizes(block_sizes: npt.ArrayLike) -> np.ndarray:
    """Validates and converts input block sizes into a contiguous int64 array.

    Args:
        block_sizes: 1D integer array-like specifying each block's sample count.

    Returns:
        Contiguous 1D int64 array of block sizes.

    Raises:
        ValueError: If block_sizes is not 1D, not integer, or contains values
            outside [0, 65535].
    """
    sizes = np.asarray(block_sizes)
    if sizes.ndim != 1 or not (np.issubdtype(sizes.dtype, np.integer) or sizes.shape[0] == 0):
        raise ValueError(f"block_sizes must be a 1-D integer array, got {sizes.dtype} of shape {sizes.shape}")
    sizes = np.ascontiguousarray(sizes, np.int64)
    if sizes.shape[0] and not (0 <= sizes.min() and sizes.max() <= _format.MAX_BLOCK_LEN):
        raise ValueError(f"block sizes must be from 0 to {_format.MAX_BLOCK_LEN}")
    return sizes


def check_group_counts(num_blocks: int, num_samples: int, block_sizes: np.ndarray | None = None) -> None:
    """Verifies that block and sample counts satisfy format limits.

    Args:
        num_blocks: Total number of blocks in the block group.
        num_samples: Total number of samples in the block group.
        block_sizes: Optional 1D array of block sizes to check individual sizes.

    Raises:
        ValueError: If block count exceeds 65535, sample count exceeds 2^26, or any
            individual block exceeds 65535 samples.
    """
    if block_sizes is not None and block_sizes.shape[0] and int(block_sizes.max()) > _format.MAX_BLOCK_LEN:
        raise ValueError(f"a block holds at most {_format.MAX_BLOCK_LEN} samples, got {int(block_sizes.max())}")
    if num_blocks > _format.MAX_BLOCKS:
        raise ValueError(f"a block group holds at most {_format.MAX_BLOCKS} blocks, got {num_blocks}")
    if num_samples > _format.MAX_GROUP_SAMPLES:
        raise ValueError(f"a block group holds at most {_format.MAX_GROUP_SAMPLES} samples, got {num_samples}")


def decrease_error(ticks: np.ndarray, sample_idx: int) -> ValueError:
    """Constructs a ValueError for timestamps that decrease or contain NaT.

    Args:
        ticks: 1D int64 array of timestamp ticks.
        sample_idx: Index of the sample at which the order violation was found.

    Returns:
        ValueError with a descriptive explanation of the error.
    """
    if (ticks == _time.INT64_MIN).any():
        return ValueError("times contain NaT")
    return ValueError(
        f"times must be non-decreasing: sample {sample_idx} is {int(ticks[sample_idx])}, "
        f"below sample {sample_idx - 1} at {int(ticks[sample_idx - 1])}"
    )


def fixed_sizes(num_samples: int, block_len: int) -> np.ndarray:
    """Partitions total sample count into fixed-length block sizes.

    Args:
        num_samples: Total number of samples to partition.
        block_len: Target sample count per block (1 to 65535).

    Returns:
        1D int64 array of block sizes, where the trailing block holds the remainder.

    Raises:
        ValueError: If block_len is outside [1, 65535].
    """
    if not (is_int(block_len) and 1 <= block_len <= _format.MAX_BLOCK_LEN):
        raise ValueError(f"block_len must be from 1 to {_format.MAX_BLOCK_LEN}, got {block_len!r}")
    num_blocks = -(-num_samples // block_len)
    sizes = np.full(num_blocks, block_len, np.int64)
    if num_blocks:
        sizes[-1] = num_samples - (num_blocks - 1) * block_len
    return sizes


def to_ticks(value: object, time_unit: int, name: str, duration: bool = False) -> int:
    """Converts a time (or a duration) into an exact integer number of ticks.

    Args:
        value: An integer tick count, a datetime64 / timedelta64 scalar, a naive
            datetime.datetime / datetime.timedelta, or anything with to_datetime64 /
            to_timedelta64 (pandas).
        time_unit: The ticks' time unit code.
        name: The argument's name, for errors.
        duration: Whether value is a duration rather than a point in time.

    Returns:
        The value in ticks.

    Raises:
        ValueError: If value has the wrong type, is NaT or time zone aware, isn't a whole
            number of ticks, or doesn't fit int64.
    """
    # timedelta64 subclasses numpy's signed integer: it isn't a tick count
    if isinstance(value, (int, np.integer)) and is_int(value) and not isinstance(value, np.timedelta64):
        ticks = int(value)
    else:
        kind = "timedelta64" if duration else "datetime64"
        converter = getattr(value, f"to_{kind}", None)
        if converter is not None:
            if getattr(value, "tzinfo", None) is not None:
                raise ValueError(f"{name} is time zone aware: convert it to naive UTC")
            value = converter()
        if isinstance(value, datetime.datetime) and not duration:
            if value.tzinfo is not None:
                raise ValueError(f"{name} is time zone aware: convert it to naive UTC")
            value = np.datetime64(value, "us")
        elif isinstance(value, datetime.timedelta) and duration:
            value = np.timedelta64(value, "us")
        if not isinstance(value, np.timedelta64 if duration else np.datetime64):
            raise ValueError(f"{name} must be a {kind}, a {'timedelta' if duration else 'datetime'} or integer ticks, "
                             f"got {value!r}")
        if np.isnat(value):
            raise ValueError(f"{name} is NaT")
        source_unit, unit_count = np.datetime_data(value.dtype)
        if source_unit not in _UNIT_NANOSECONDS:
            raise ValueError(f"{name} is in {source_unit}, which isn't a fixed length of time")
        exact = int(value.view(np.int64)) * unit_count * _UNIT_NANOSECONDS[source_unit] / _UNIT_NANOSECONDS[
            _format.TIME_UNIT_NAMES[time_unit]]
        if exact.denominator != 1:
            raise ValueError(f"{name} {value} isn't a whole number of {_format.TIME_UNIT_NAMES[time_unit]}")
        ticks = int(exact)
    if not _time.INT64_MIN < ticks <= _time.INT64_MAX:
        raise ValueError(f"{name} doesn't fit int64 ticks")
    return ticks


def to_ranges(delete_ranges: RangesLike, time_unit: int) -> np.ndarray:
    """Converts delete ranges into sorted, disjoint [start, end) ranges of ticks.

    Args:
        delete_ranges: One (start, end) pair, or an iterable of them, or an array of shape
            (k, 2), of times like start_time. Ranges may overlap or touch; empty ones are
            ignored.
        time_unit: The ticks' time unit code.

    Returns:
        2D int64 array of shape (k, 2) of non-empty ranges in increasing order, each ending
        before the next starts.

    Raises:
        ValueError: If the ranges aren't pairs of times, or a range ends before it starts.
    """
    if isinstance(delete_ranges, np.ndarray) and delete_ranges.ndim == 0:
        raise ValueError(f"delete_ranges must be (start, end) pairs, got {delete_ranges!r}")
    if isinstance(delete_ranges, np.ndarray):
        items = list(delete_ranges.reshape(1, 2) if delete_ranges.shape == (2,) else delete_ranges)
    elif isinstance(delete_ranges, Iterable):
        try:
            items = list(delete_ranges)
        except TypeError:
            raise ValueError(f"delete_ranges must be (start, end) pairs, got {delete_ranges!r}") from None
        # One pair of times rather than a sequence of pairs
        if len(items) == 2 and not any(isinstance(item, (tuple, list, np.ndarray)) for item in items):
            items = [items]
    else:
        raise ValueError(f"delete_ranges must be (start, end) pairs, got {delete_ranges!r}")  # noqa: TRY004
    pairs = []
    for item in items:
        pair = list(item) if isinstance(item, (tuple, list)) or (isinstance(item, np.ndarray) and item.ndim) else None
        if pair is None or len(pair) != 2:
            raise ValueError(f"delete_ranges must be (start, end) pairs, got {item!r}")
        pairs.append([to_ticks(_scalar(point), time_unit, "delete_ranges") for point in pair])
    ticks = np.array(pairs, np.int64).reshape(-1, 2)
    backwards = np.flatnonzero(ticks[:, 1] < ticks[:, 0])
    if backwards.shape[0]:
        raise ValueError(f"delete range {int(backwards[0])} ends before it starts")
    # Merge into disjoint ranges: sorted by start, each absorbing those that start inside it
    merged: list[list[int]] = []
    for range_start, range_end in sorted(ticks[ticks[:, 1] > ticks[:, 0]].tolist()):
        if merged and range_start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], range_end)
        else:
            merged.append([range_start, range_end])
    return np.array(merged, np.int64).reshape(-1, 2)


def _scalar(value: object) -> object:
    """Extracts a zero-dimensional numpy array element as a native scalar.

    Args:
        value: Scalar value or 0-D numpy array.

    Returns:
        Extracted scalar value (e.g. datetime64 preserves its datetime64 type).
    """
    return value[()] if isinstance(value, np.ndarray) else value


def check_chunking(block_len: int, blocks_per_group: int) -> None:
    """Validates block and block group chunking parameters for multi-group encoding.

    Args:
        block_len: Target sample count per block.
        blocks_per_group: Target block count per block group.

    Raises:
        ValueError: If block_len is not from 1 to 65,535, or blocks_per_group is below 1.
    """
    fixed_sizes(0, block_len)
    if not (is_int(blocks_per_group) and blocks_per_group >= 1):
        raise ValueError(f"blocks_per_group must be >= 1, got {blocks_per_group!r}")


def check_stored_time_unit(time_unit: str | None, stored_unit: int) -> None:
    """Checks an optional user-provided time unit against the block group's time unit code.

    Args:
        time_unit: Optional user-supplied time unit string (or None).
        stored_unit: Stored time unit code from the existing block group header.

    Raises:
        ValueError: If time_unit is invalid, the block group has no time axis, or
            time_unit does not match stored_unit.
    """
    if time_unit is not None and not stored_unit:
        raise ValueError("the block group has no time axis: time_unit must be None")
    if time_unit is not None and time_unit_code(time_unit) != stored_unit:
        raise ValueError(f"time_unit is {time_unit} but the block group stores {_format.TIME_UNIT_NAMES[stored_unit]}")


def update_blocks(
    blocks: Mapping[int, npt.ArrayLike],
    times: Mapping[int, npt.ArrayLike] | None,
    stored_unit: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Validates update's blocks and times against the block group's time unit code.

    Args:
        blocks: Mapping of block indices to sample arrays for update.
        times: Optional mapping of block indices to timestamp arrays.
        stored_unit: Stored time unit code in the block group (0 for no time axis).

    Returns:
        A tuple of (indices, samples, block_sizes, ticks):
            indices: 1D int64 array of block indices in increasing order.
            samples: Flattened 1D float64 array of all replacement samples.
            block_sizes: 1D int64 array of each block's sample count.
            ticks: Flattened 1D int64 array of all replacement timestamp ticks
                (or None if the block group has no time axis).

    Raises:
        ValueError: If an index is invalid, a block isn't 1-D or too large, or
            the times don't match the block group's time axis or the blocks.
    """
    if not isinstance(blocks, Mapping):
        raise ValueError(f"blocks must map block indices to samples, got {type(blocks).__name__}")  # noqa: TRY004
    if not all(is_int(idx) and idx >= 0 for idx in blocks):
        raise ValueError(f"block indices must be integers >= 0, got {sorted(map(repr, blocks))}")
    indices = sorted(int(idx) for idx in blocks)
    # Check upper limit before allocating arrays sized by the indices
    if indices and indices[-1] >= _format.MAX_BLOCKS:
        raise ValueError(f"a block group holds at most {_format.MAX_BLOCKS} blocks, got block index {indices[-1]}")
    # Verify timestamp consistency with block group time axis
    if stored_unit:
        if times is None:
            raise ValueError("the block group has a time axis: times are required")
        if sorted(int(idx) for idx in times) != indices:
            raise ValueError("times must have the same block indices as blocks")
    elif times is not None:
        raise ValueError("the block group has no time axis: times must be None")
    # Convert and validate samples for each block
    by_index = {int(idx): block for idx, block in blocks.items()}
    samples = [as_series(by_index[idx], allow_empty=True) for idx in indices]
    sizes = as_block_sizes([block.shape[0] for block in samples]) if samples else np.zeros(0, np.int64)
    ticks = None
    # Convert and validate timestamps for each block if present
    if times is not None:
        times_by_index = {int(idx): block_times for idx, block_times in times.items()}
        block_ticks = [as_ticks(times_by_index[idx], None, stored_unit)[0] for idx in indices]
        for idx, block, tick_block in zip(indices, samples, block_ticks):
            if tick_block.shape != block.shape:
                raise ValueError(f"block {idx}: times must have the shape of its samples {block.shape}, got {tick_block.shape}")
        ticks = np.concatenate(block_ticks) if block_ticks else np.zeros(0, np.int64)
    return np.array(indices, np.int64), np.concatenate(samples) if samples else np.zeros(0), sizes, ticks


def time_blocks_update(
    samples: npt.ArrayLike,
    times: npt.ArrayLike,
    start_time: TimeLike,
    block_duration: DurationLike,
    delete_ranges: RangesLike | None,
    time_unit: str | None,
    stored_unit: int,
) -> tuple[np.ndarray, np.ndarray, int, int, np.ndarray]:
    """Validates and converts arguments for time-based block update.

    Args:
        samples: 1D array-like of series samples (may be empty).
        times: Array-like of timestamps matching samples in length.
        start_time: Reference starting time for block 0.
        block_duration: Fixed duration spanning each block.
        delete_ranges: Optional time intervals to delete before adding samples.
        time_unit: Optional time unit string, verified against stored_unit.
        stored_unit: Stored time unit code from the existing block group header.

    Returns:
        A tuple of (samples_arr, ticks, start_tick, duration_ticks, ranges):
            samples_arr: 1D float64 array of input samples.
            ticks: 1D int64 array of timestamp ticks.
            start_tick: Starting time as an integer tick count.
            duration_ticks: Block duration as an integer tick count.
            ranges: 2D int64 array of sorted, disjoint [start, end) tick intervals.

    Raises:
        ValueError: If time_unit isn't the block group's, or any argument is invalid.
    """
    check_stored_time_unit(time_unit, stored_unit)
    samples_arr = as_series(samples, allow_empty=True)
    ticks, _ = as_ticks(times, None, stored_unit)
    if ticks.shape != samples_arr.shape:
        raise ValueError(f"times must be 1-D with one entry per sample ({samples_arr.shape[0]}), got shape {ticks.shape}")
    start = to_ticks(start_time, stored_unit, "start_time")
    duration = to_ticks(block_duration, stored_unit, "block_duration", duration=True)
    ranges = to_ranges([] if delete_ranges is None else delete_ranges, stored_unit)
    return samples_arr, ticks, start, duration, ranges
