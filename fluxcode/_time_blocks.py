# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Time-divided blocks: block b of a unit holds the samples timed in
[start_time + b * block_duration, start_time + (b + 1) * block_duration).

The start time and duration aren't stored: the caller keeps them (typically the start is part
of the unit's storage key) and passes the same ones to update_time_blocks. Times, the start,
the duration and delete ranges are all converted to int64 ticks in the times' unit.
"""

import bisect
import datetime
from collections.abc import Iterable
from fractions import Fraction

import numpy as np
import numpy.typing as npt

from . import _format, _time, _unit
from ._types import EncodedUnit, Params, UpdatedUnit, is_int

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


def to_ticks(value: object, time_unit: int, name: str, duration: bool = False) -> int:
    """Converts a time (or a duration) into an exact integer number of ticks.

    Args:
        value: An integer tick count, a datetime64 / timedelta64 scalar, a naive
            datetime.datetime / datetime.timedelta, or anything with to_datetime64 /
            to_timedelta64 (pandas).
        time_unit: The ticks' TimeUnit code.
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
        time_unit: The ticks' TimeUnit code.

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
    """A numpy array element as a scalar of its own type (datetime64 stays datetime64)."""
    return value[()] if isinstance(value, np.ndarray) else value


def block_ids(ticks: np.ndarray, start: int, duration: int) -> np.ndarray:
    """Each tick's block: floor((tick - start) / duration).

    Raises:
        ValueError: If a tick is before the start or past the last possible block.
    """
    if duration <= 0:
        raise ValueError(f"block_duration must be positive, got {duration} ticks")
    if ticks.shape[0] and ticks.min() < start:
        raise ValueError(f"times must be at or after start_time: got {int(ticks.min())} before {start}")
    # As uint64: the difference of two int64 ticks with tick >= start fits
    ids = (ticks.astype(np.uint64) - np.uint64(start % 2**64)) // np.uint64(duration)
    if ids.shape[0] and ids.max() >= _format.MAX_BLOCKS:
        raise ValueError(f"times reach block {int(ids.max())}, past the last block a unit holds ({_format.MAX_BLOCKS - 1})")
    return ids.astype(np.int64)


def chunk(ticks: np.ndarray, start: int, duration: int) -> np.ndarray:
    """The block sizes of sorted ticks divided into blocks of duration from start.

    Unsorted ticks get sizes that the encoder then rejects (they decrease somewhere).
    """
    ids = block_ids(ticks, start, duration)
    return np.bincount(ids, minlength=int(ids[-1]) + 1 if ids.shape[0] else 0).astype(np.int64)


def encode_time_blocks(
    x: npt.ArrayLike,
    times: npt.ArrayLike,
    params: Params,
    start_time: TimeLike,
    block_duration: DurationLike,
    time_unit: str | None,
) -> EncodedUnit:
    """Implements fluxcode.encode_time_blocks."""
    samples = _unit.as_series(x, allow_empty=True)
    ticks, unit_code = _unit.series_ticks(times, time_unit, samples.shape[0])
    assert ticks is not None
    start = to_ticks(start_time, unit_code, "start_time")
    duration = to_ticks(block_duration, unit_code, "block_duration", duration=True)
    return _unit.encode(samples, chunk(ticks, start, duration), params, ticks, unit_code)


def _in_ranges(ticks: np.ndarray, ranges: np.ndarray) -> np.ndarray:
    """Whether each tick is in one of the sorted, disjoint [start, end) ranges."""
    if not ranges.shape[0]:
        return np.zeros(ticks.shape[0], np.bool_)
    range_idx = np.searchsorted(ranges[:, 0], ticks, "right") - 1
    return (range_idx >= 0) & (ticks < ranges[np.maximum(range_idx, 0), 1])


def update_time_blocks(
    unit: bytes,
    x: npt.ArrayLike,
    times: npt.ArrayLike,
    params: Params,
    start_time: TimeLike,
    block_duration: DurationLike,
    delete_ranges: RangesLike | None,
    time_unit: str | None,
) -> UpdatedUnit:
    """Implements fluxcode.update_time_blocks."""
    parsed = _unit.decompress(unit)
    if not parsed.has_time:
        raise ValueError("the unit has no time axis: use update")
    unit_code = parsed.header.time_unit
    if time_unit is not None and _unit.time_unit_code(time_unit) != unit_code:
        raise ValueError(f"time_unit is {time_unit} but the unit stores {_format.TIME_UNIT_NAMES[unit_code]}")
    samples = _unit.as_series(x, allow_empty=True)
    ticks, _ = _unit.as_ticks(times, None, unit_code)
    if ticks.shape != samples.shape:
        raise ValueError(f"times must be 1-D with one entry per sample ({samples.shape[0]}), got shape {ticks.shape}")
    start = to_ticks(start_time, unit_code, "start_time")
    duration = to_ticks(block_duration, unit_code, "block_duration", duration=True)
    ranges = to_ranges([] if delete_ranges is None else delete_ranges, unit_code)
    new_ids = block_ids(ticks, start, duration)
    if new_ids.shape[0] and (np.diff(ticks) < 0).any():
        raise _unit.decrease_error(ticks, int(np.flatnonzero(np.diff(ticks) < 0)[0]) + 1)
    num_old_blocks = parsed.header.num_blocks
    # The existing blocks whose time span meets a range (block bounds as Python ints: past int64
    # if need be, so both ends are clamped to the old blocks before numpy sees them)
    spans = [
        np.arange(min(max(range_start - start, 0) // duration, num_old_blocks),
                  min(-(-(range_end - start) // duration), num_old_blocks))
        for range_start, range_end in ranges.tolist()
        if range_end > start
    ]
    touched_ids = np.unique(np.concatenate(spans)) if spans else np.zeros(0, np.int64)
    # A block whose whole span is inside one (merged) range loses all its samples without being
    # decoded; exact Python ints, as a span can end past int64
    range_starts, range_ends = ranges[:, 0].tolist(), ranges[:, 1].tolist()
    touched = set(touched_ids.tolist())
    covered = set()
    for block_idx in touched:
        block_start = start + block_idx * duration
        range_idx = bisect.bisect_right(range_starts, block_start) - 1
        if range_idx >= 0 and range_ends[range_idx] >= block_start + duration:
            covered.add(block_idx)
    # The existing blocks to decode: those a range meets or a new sample lands in, except the
    # covered ones, which lose everything
    new_blocks = set(np.unique(new_ids).tolist())
    decoded = np.array(sorted(b for b in touched | new_blocks if b < num_old_blocks and b not in covered), np.int64)
    time_rows = _unit.read_time_rows(parsed, decoded)
    old_sizes = parsed.block_sizes
    gathered_values, gathered_ticks = np.zeros(0), np.zeros(0, np.int64)
    gathered_blocks = np.zeros(0, np.int64)
    if decoded.shape[0]:
        all_values, all_ticks = _unit.decode_blocks(parsed, decoded, time_rows)
        assert all_ticks is not None
        # The decoded blocks' samples in one run, in time order (blocks ascend)
        run_sizes = old_sizes[decoded]
        run_starts = parsed.layout.sample_offsets[decoded]
        positions = np.arange(int(run_sizes.sum())) + np.repeat(run_starts - (np.cumsum(run_sizes) - run_sizes),
                                                                 run_sizes)
        gathered_values, gathered_ticks = all_values[positions], all_ticks[positions]
        gathered_blocks = np.repeat(decoded, run_sizes)
    # A decoded sample goes if a range covers it or a new sample has its time. Each new tick
    # matches a run [lo, hi) of the (sorted) decoded ticks: the runs are marked by a running sum
    # of +1 at each lo and -1 at each hi, which is faster than looking every decoded tick up.
    match_lo = np.searchsorted(gathered_ticks, ticks, "left")
    match_hi = np.searchsorted(gathered_ticks, ticks, "right")
    num_gathered = gathered_ticks.shape[0]
    marks = np.bincount(match_lo, minlength=num_gathered + 1) - np.bincount(match_hi, minlength=num_gathered + 1)
    replaced = np.cumsum(marks)[:num_gathered] > 0
    keep = ~(_in_ranges(gathered_ticks, ranges) | replaced)
    # Kept samples share no time with new ones, so a stable sort puts them in order with the new
    # ones, which keep theirs. A tick fixes its block, so each block's samples end up together.
    combined_ticks = np.concatenate([gathered_ticks[keep], ticks])
    order = np.argsort(combined_ticks, kind="stable")
    merged_ticks = combined_ticks[order]
    merged_values = np.concatenate([gathered_values[keep], samples])[order]
    merged_blocks = np.concatenate([gathered_blocks[keep], new_ids])[order]
    # Blocks to re-encode: those that gain a sample or lose one (a block that loses nothing and
    # gains nothing, empty ones included, is carried over)
    num_blocks = max(num_old_blocks, int(new_ids[-1]) + 1 if new_ids.shape[0] else 0)
    changed = np.bincount(new_ids, minlength=num_blocks) > 0
    changed[:num_old_blocks] |= np.bincount(gathered_blocks[~keep], minlength=num_old_blocks) > 0
    covered_ids = np.fromiter(covered, np.int64, len(covered))
    changed[covered_ids] |= old_sizes[covered_ids] > 0
    indices = np.flatnonzero(changed)
    if not indices.shape[0]:
        return UpdatedUnit(unit, np.zeros(0, np.int64), np.zeros(0), np.zeros(0), np.zeros(0))
    sizes = np.bincount(merged_blocks, minlength=num_blocks)[indices]
    if sizes.max() > _format.MAX_BLOCK_LEN:
        raise ValueError(f"a block would hold over {_format.MAX_BLOCK_LEN} samples")
    in_changed = changed[merged_blocks]
    return _unit.splice(
        parsed,
        indices,
        merged_values[in_changed],
        sizes,
        merged_ticks[in_changed],
        params,
        time_rows,
    )
