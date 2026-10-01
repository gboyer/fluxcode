# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Time-divided blocks: block b of a unit holds the samples timed in
[start_time + b * block_duration, start_time + (b + 1) * block_duration).

The start time and duration aren't stored: the caller keeps them (typically the start is part
of the unit's storage key) and passes the same ones to update_time_blocks. Times, the start,
the duration and update ranges are all converted to int64 ticks in the times' unit.
"""

import datetime
from fractions import Fraction

import numpy as np
import numpy.typing as npt

from . import _format, _time, _unit
from ._types import EncodedUnit, Params, UpdatedUnit, is_int

TimeLike = np.datetime64 | datetime.datetime | int
"""A point in time: datetime64 or datetime (naive, UTC recommended), or an integer tick."""

DurationLike = np.timedelta64 | datetime.timedelta | int
"""A length of time: timedelta64 or timedelta, or an integer number of ticks."""

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


def to_ranges(update_ranges: object, time_unit: int) -> np.ndarray:
    """Converts update ranges into a 2D int64 array of [start, end) ticks.

    Args:
        update_ranges: One (start, end) pair, or an iterable of them, or an array of shape
            (k, 2), of times like start_time.
        time_unit: The ticks' TimeUnit code.

    Returns:
        2D int64 array of shape (k, 2).

    Raises:
        ValueError: If the ranges aren't pairs of times, or a range ends before it starts.
    """
    ranges = np.asarray(update_ranges, dtype=None if not isinstance(update_ranges, (list, tuple)) else object)
    if ranges.ndim == 1 and ranges.shape[0] == 2 and not isinstance(ranges[0], (tuple, list, np.ndarray)):
        ranges = ranges.reshape(1, 2)
    elif ranges.ndim == 1:
        # An iterable of pairs that numpy kept as objects
        ranges = np.array([tuple(pair) for pair in ranges], dtype=object).reshape(-1, 2) if ranges.shape[0] else \
            np.zeros((0, 2), object)
    if ranges.ndim != 2 or ranges.shape[1] != 2:
        raise ValueError(f"update_ranges must be (start, end) pairs, got shape {ranges.shape}")
    ticks = np.array(
        [[to_ticks(_scalar(point), time_unit, "update_ranges") for point in pair] for pair in ranges], np.int64
    ).reshape(-1, 2)
    backwards = np.flatnonzero(ticks[:, 1] < ticks[:, 0])
    if backwards.shape[0]:
        raise ValueError(f"update range {int(backwards[0])} ends before it starts")
    return ticks


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
    """Whether each tick is in any of the [start, end) ranges."""
    inside = np.zeros(ticks.shape[0], bool)
    for range_start, range_end in ranges.tolist():
        inside |= (ticks >= range_start) & (ticks < range_end)
    return inside


def update_time_blocks(
    unit: bytes,
    x: npt.ArrayLike,
    times: npt.ArrayLike,
    params: Params,
    start_time: TimeLike,
    block_duration: DurationLike,
    update_ranges: object,
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
    ranges = to_ranges(update_ranges, unit_code)
    if not _in_ranges(ticks, ranges).all():
        outside = int(np.flatnonzero(~_in_ranges(ticks, ranges))[0])
        raise ValueError(f"times must be within update_ranges: sample {outside} at {int(ticks[outside])} isn't")
    new_ids = block_ids(ticks, start, duration)
    if new_ids.shape[0] and (np.diff(ticks) < 0).any():
        raise _unit.decrease_error(ticks, int(np.flatnonzero(np.diff(ticks) < 0)[0]) + 1)
    num_old_blocks = parsed.header.num_blocks
    # The existing blocks whose time span meets a range: as Python ints, past int64 if need be
    touched: set[int] = set()
    covered: set[int] = set()
    for range_start, range_end in ranges.tolist():
        if range_end <= start or range_end == range_start:
            continue
        first = max(range_start - start, 0) // duration
        last = min(-(-(range_end - start) // duration), num_old_blocks)
        for block_idx in range(first, last):
            touched.add(block_idx)
    for block_idx in touched:
        # A block whose whole span is inside the ranges loses all its samples without being decoded
        block_start = start + block_idx * duration
        if _covers(ranges, block_start, block_start + duration):
            covered.add(block_idx)
    straddling = np.array(sorted(touched - covered), np.int64)
    old_values, old_ticks = _unit.decode_blocks(parsed, straddling)
    assert old_ticks is not None
    old_offsets = parsed.layout.sample_offsets
    new_offsets = _unit.sample_offsets(np.bincount(new_ids, minlength=max(num_old_blocks, int(new_ids[-1]) + 1)
                                                   if new_ids.shape[0] else num_old_blocks))
    targets = sorted(touched | set(np.unique(new_ids).tolist()))
    pieces_values, pieces_ticks, sizes, indices = [], [], [], []
    for block_idx in targets:
        new_values = samples[new_offsets[block_idx]:new_offsets[block_idx + 1]]
        new_ticks = ticks[new_offsets[block_idx]:new_offsets[block_idx + 1]]
        if block_idx < num_old_blocks and block_idx not in covered:
            kept_values = old_values[old_offsets[block_idx]:old_offsets[block_idx + 1]]
            kept_ticks = old_ticks[old_offsets[block_idx]:old_offsets[block_idx + 1]]
            keep = ~_in_ranges(kept_ticks, ranges)
            if keep.all() and not new_values.shape[0]:
                continue  # nothing in this block is in a range
            # Kept samples are outside the ranges and new ones inside: no tick is shared
            merged_ticks = np.concatenate([kept_ticks[keep], new_ticks])
            order = np.argsort(merged_ticks, kind="stable")
            new_values = np.concatenate([kept_values[keep], new_values])[order]
            new_ticks = merged_ticks[order]
        elif block_idx < num_old_blocks and not parsed.block_sizes[block_idx] and not new_values.shape[0]:
            continue  # empty and stays empty
        pieces_values.append(new_values)
        pieces_ticks.append(new_ticks)
        sizes.append(new_values.shape[0])
        indices.append(block_idx)
    if any(size > _format.MAX_BLOCK_LEN for size in sizes):
        raise ValueError(f"a block would hold over {_format.MAX_BLOCK_LEN} samples")
    if not indices:
        return UpdatedUnit(unit, np.zeros(0, np.int64), np.zeros(0), np.zeros(0), np.zeros(0))
    return _unit.splice(
        parsed,
        np.array(indices, np.int64),
        np.concatenate(pieces_values),
        np.array(sizes, np.int64),
        np.concatenate(pieces_ticks).astype(np.int64, copy=False),
        params,
    )


def _covers(ranges: np.ndarray, span_start: int, span_end: int) -> bool:
    """Whether the union of the [start, end) ranges covers [span_start, span_end)."""
    position = span_start
    for range_start, range_end in sorted(ranges.tolist()):
        if range_start > position:
            break
        position = max(position, range_end)
        if position >= span_end:
            return True
    return position >= span_end
