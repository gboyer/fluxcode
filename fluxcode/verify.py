# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Checks that an encode or update kept the guarantees of ENCODER.md §6, from its inputs and output.

Each function takes a fluxcode call's result followed by that call's own arguments, and returns
the violations found (an empty list when the call kept its guarantees):

    result = fluxcode.encode_group(x, params, times=t)
    for violation in fluxcode.verify.encode_group(result, x, params, times=t):
        log.error("fluxcode: %s", violation)

It decodes the output and compares it with the input. It uses the decoder and the format's
columns, never the encoder (except fixed_point, whose subject is re-encoding), so an encoder bug
doesn't hide itself. Nothing here runs unless called: the encode path is unchanged, and importing
fluxcode doesn't import this module.

A check costs a decode and two passes over the samples: about as much as the encode itself on
a block group of 60 blocks of 1000 (+86% of the encode for values, +70% with times, +120% with a
time error; fixed_point, two more encodes, +300%). In production, check a sample of block groups,
chosen by a hash of their storage key so that a finding can be reproduced.

Checked, per block (the `check` of each Violation):
    "decode": the output doesn't decode.
    "layout": the time unit, block count, block sizes or re-encoded indices aren't the call's.
    "stats": the returned block_min and block_max aren't the finite samples' own, or the mean
        is off by more than its summation error; NaN for blocks without finite samples.
    "nonfinite": a NaN, +inf or -inf didn't decode as itself, or a finite sample didn't decode
        finite.
    "range_bound": a sample decoded further than range / (2^min_quantize_bits - 1/2).
    "fine_bound": a block neither the noise floor nor the target can coarsen (no target, and
        the noise floor off or fewer than 256 samples) is off by more than
        range / (2^max_quantize_bits - 1/2).
    "step_bound": a sample decoded further than its block's stored grid allows: half the step
        2^e, a quarter of a decimal step 10^p (the detection tolerance), half a scaled step.
    "untouched": a block an update didn't re-encode decodes differently. (The samples
        update_time_blocks keeps in a block it re-encodes are checked by the bounds: on a grid
        holding their points, any other decoded value is a step away, past the step bound.)
    "times": a time that is stored exactly (time_error 0, a regular block, a block of 1 or 2,
        update_time_blocks) decoded differently, or the series isn't non-decreasing.
    "time_error": a rounded time moved more than 1.02 time_error mean intervals of its block
        (in update, the larger of its and its stored neighbours'), or more than one in 20 moved
        more than time_error times their own interval.
    "time_block": a time decoded outside its time block, end included.
    "fixed_point": re-encoding the decoded data changed it, or the bytes from the second encode.

Not checked: the noise floor's f sigma / 2 (it needs the encoder's estimate of sigma), the
bit-exactness of decimal data (which data are decimals is the caller's knowledge), and bounds
spanning several updates (they need the original samples).
"""

from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
from numba import njit

from . import _api, _args, _format, _group
from ._args import DurationLike, RangesLike, TimeLike
from ._format import BLOCK_FLAG_DECIMAL, BLOCK_FLAG_SCALED
from ._types import (
    DEFAULT_PARAMS,
    EncodedGroup,
    EncodedSeries,
    Params,
    TimeUnit,
    UpdatedGroup,
)

TIME_SLACK: float = 1.02
"""How far past time_error mean intervals a rounded time may move (ENCODER.md §6: 2% more)."""

NOISE_FLOOR_MIN_SAMPLES: int = 256
"""Blocks shorter than this never get the noise floor (ENCODER.md §5.2)."""


class Violation(NamedTuple):
    """A guarantee a call didn't keep.

    Attributes:
        check: Which guarantee (see the module documentation).
        block: The block index, or -1 for the whole block group.
        message: What was found, with the bound it broke.
    """

    check: str
    block: int
    message: str

    def __str__(self) -> str:
        where = "block group" if self.block < 0 else f"block {self.block}"
        return f"{self.check}: {where}: {self.message}"


# Encoding

def encode_group(
    result: EncodedGroup,
    samples: npt.ArrayLike,
    params: Params = DEFAULT_PARAMS,
    *,
    block_len: int = _api.DEFAULT_BLOCK_LEN,
    times: npt.ArrayLike | None = None,
    time_unit: TimeUnit | None = None,
) -> list[Violation]:
    """Checks the result of fluxcode.encode_group(samples, params, ...), given the same arguments.

    Returns:
        The violations found; empty if the result keeps the guarantees.
    """
    series = _args.as_series(samples)
    ticks, unit = _args.series_ticks(times, time_unit, series.shape[0])
    return _check_encode(result, series, _args.fixed_sizes(series.shape[0], block_len), ticks, unit, params)


def encode_blocks(
    result: EncodedGroup,
    samples: npt.ArrayLike,
    block_sizes: npt.ArrayLike,
    params: Params = DEFAULT_PARAMS,
    *,
    times: npt.ArrayLike | None = None,
    time_unit: TimeUnit | None = None,
) -> list[Violation]:
    """Checks the result of fluxcode.encode_blocks(samples, block_sizes, params, ...).

    Returns:
        The violations found; empty if the result keeps the guarantees.
    """
    series = _args.as_series(samples, allow_empty=True)
    ticks, unit = _args.series_ticks(times, time_unit, series.shape[0])
    return _check_encode(result, series, _args.as_block_sizes(block_sizes), ticks, unit, params)


def encode_time_blocks(
    result: EncodedGroup,
    samples: npt.ArrayLike,
    times: npt.ArrayLike,
    params: Params = DEFAULT_PARAMS,
    *,
    start_time: TimeLike,
    block_duration: DurationLike,
    time_unit: TimeUnit | None = None,
) -> list[Violation]:
    """Checks the result of fluxcode.encode_time_blocks(samples, times, params, ...).

    Returns:
        The violations found; empty if the result keeps the guarantees.
    """
    series = _args.as_series(samples, allow_empty=True)
    ticks, unit = _args.series_ticks(times, time_unit, series.shape[0])
    assert ticks is not None
    start = _args.to_ticks(start_time, unit, "start_time")
    duration = _args.to_ticks(block_duration, unit, "block_duration", duration=True)
    ids = _time_block_ids(ticks, start, duration)
    sizes = np.bincount(ids, minlength=0).astype(np.int64) if ids.shape[0] else np.zeros(0, np.int64)
    return _check_encode(result, series, sizes, ticks, unit, params, time_blocks=(start, duration))


def encode(
    result: EncodedSeries,
    samples: npt.ArrayLike,
    params: Params = DEFAULT_PARAMS,
    *,
    block_len: int = _api.DEFAULT_BLOCK_LEN,
    blocks_per_group: int = _api.DEFAULT_BLOCKS_PER_GROUP,
    times: npt.ArrayLike | None = None,
    time_unit: TimeUnit | None = None,
) -> list[Violation]:
    """Checks the result of fluxcode.encode(samples, params, ...). Each message names its block group.

    Returns:
        The violations found; empty if the result keeps the guarantees.
    """
    series = _args.as_series(samples)
    ticks, unit = _args.series_ticks(times, time_unit, series.shape[0])
    per_group = block_len * blocks_per_group
    num_groups = -(-series.shape[0] // per_group)
    found = []
    if len(result.groups) != num_groups:
        return [Violation("layout", -1, f"{len(result.groups)} block groups, not {num_groups}")]
    for group_idx in range(num_groups):
        part = slice(group_idx * per_group, (group_idx + 1) * per_group)
        encoded = EncodedGroup(result.groups[group_idx], result.block_mins[group_idx], result.block_maxs[group_idx],
                               result.block_means[group_idx])
        chunk = series[part]
        for violation in _check_encode(encoded, chunk, _args.fixed_sizes(chunk.shape[0], block_len),
                                       None if ticks is None else ticks[part], unit, params):
            found.append(violation._replace(message=f"block group {group_idx}: {violation.message}"))
    return found


# Updating

def update(
    result: UpdatedGroup,
    group: bytes,
    blocks: Mapping[int, npt.ArrayLike],
    params: Params = DEFAULT_PARAMS,
    *,
    times: Mapping[int, npt.ArrayLike] | None = None,
    time_unit: TimeUnit | None = None,
) -> list[Violation]:
    """Checks the result of fluxcode.update(group, blocks, params, ...), given the same arguments.

    Returns:
        The violations found; empty if the result keeps the guarantees.
    """
    old = _decode(group)
    if time_unit is not None:
        _args.check_stored_time_unit(time_unit, old.unit)
    indices, new_values, new_sizes, new_ticks = _args.update_blocks(blocks, times, old.unit)
    num_blocks = max(old.num_blocks, int(indices[-1]) + 1 if indices.shape[0] else 0)
    sizes = np.zeros(num_blocks, np.int64)
    sizes[:old.num_blocks] = old.sizes
    sizes[indices] = new_sizes
    encoded = np.zeros(num_blocks, np.bool_)
    encoded[indices] = True
    encoded[old.num_blocks:] = True
    # The expected series: the stored blocks, with the new ones in their places
    sources = np.where(encoded, 1, 0)
    values, ticks = _assemble(sizes, sources, [old.values, new_values],
                              None if old.ticks is None or new_ticks is None else [old.ticks, new_ticks],
                              [_block_positions(old.offsets, ~encoded[:old.num_blocks], old.sizes),
                               np.arange(new_values.shape[0])])
    allowance = None
    if old.ticks is not None and params.time_error > 0:
        allowance = _neighbour_intervals(old, encoded, sizes)
    return _check_result(result, values, ticks, sizes, encoded, old.unit, params, params.time_error,
                         allowance=allowance)


def update_time_blocks(
    result: UpdatedGroup,
    group: bytes,
    samples: npt.ArrayLike,
    times: npt.ArrayLike,
    params: Params = DEFAULT_PARAMS,
    *,
    start_time: TimeLike,
    block_duration: DurationLike,
    delete_ranges: RangesLike | None = None,
    time_unit: TimeUnit | None = None,
) -> list[Violation]:
    """Checks the result of fluxcode.update_time_blocks(group, samples, times, params, ...).

    The expected block group is the stored one with every sample timed in delete_ranges and every
    sample sharing a new sample's time removed, and the new samples added in time order.

    Returns:
        The violations found; empty if the result keeps the guarantees.
    """
    old = _decode(group)
    if old.ticks is None:
        raise ValueError("the block group has no time axis")
    new_values, new_ticks, start, duration, ranges = _args.time_blocks_update(
        samples, times, start_time, block_duration, delete_ranges, time_unit, old.unit
    )
    new_ids = _time_block_ids(new_ticks, start, duration)
    old_ids = np.repeat(np.arange(old.num_blocks), old.sizes)
    # Survivors: not in a delete range, and not replaced by a new sample at the same time
    deleted = _in_ranges(old.ticks, ranges) | np.isin(old.ticks, new_ticks)
    kept = ~deleted
    num_blocks = max(old.num_blocks, int(new_ids.max()) + 1 if new_ids.shape[0] else 0)
    changed = np.zeros(num_blocks, np.bool_)
    changed[old_ids[deleted]] = True
    changed[new_ids] = True
    changed[old.num_blocks:] = True
    # Each block's survivors and new samples, in time order (stable: equal times keep their order)
    merged_ids = np.concatenate([old_ids[kept], new_ids])
    merged_ticks = np.concatenate([old.ticks[kept], new_ticks])
    order = np.lexsort((merged_ticks, merged_ids))
    merged_values = np.concatenate([old.values[kept], new_values])[order]
    sizes = np.bincount(merged_ids, minlength=num_blocks).astype(np.int64)
    met = _met_blocks(ranges, start, duration, old.num_blocks, num_blocks)
    # Re-encoded: every block that changed, and perhaps blocks a range meets without deleting anything
    reported = np.asarray(result.indices, np.int64)
    if reported.shape[0] and (reported.min() < 0 or reported.max() >= num_blocks):
        return [Violation("layout", -1, f"re-encoded indices {_summary(reported)} outside the {num_blocks} blocks")]
    encoded = np.zeros(num_blocks, np.bool_)
    encoded[reported] = True
    if not (changed <= encoded).all() or not (encoded <= changed | met).all():
        return [Violation("layout", -1, f"re-encoded blocks {_summary(reported)}: must include "
                          f"{_summary(np.flatnonzero(changed))} and lie within {_summary(np.flatnonzero(changed | met))}")]
    return _check_result(result, merged_values, merged_ticks[order], sizes, encoded, old.unit, params, 0.0,
                         time_blocks=(start, duration))


def replace_time_blocks(
    result: UpdatedGroup,
    group: bytes,
    blocks: Iterable[int],
    samples: npt.ArrayLike,
    times: npt.ArrayLike,
    params: Params = DEFAULT_PARAMS,
    *,
    start_time: TimeLike,
    block_duration: DurationLike,
    time_unit: TimeUnit | None = None,
) -> list[Violation]:
    """Checks the result of fluxcode.replace_time_blocks(group, blocks, samples, times, params, ...).

    Returns:
        The violations found; empty if the result keeps the guarantees.
    """
    old = _decode(group)
    if old.ticks is None:
        raise ValueError("the block group has no time axis")
    declared = _args.block_list(blocks)
    new_values, new_ticks, start, duration, _ = _args.time_blocks_update(
        samples, times, start_time, block_duration, None, time_unit, old.unit
    )
    new_ids = _time_block_ids(new_ticks, start, duration)
    if not declared.shape[0]:
        if result.group != group or np.asarray(result.indices).shape[0]:
            return [Violation("layout", -1, "no blocks declared, but the block group changed")]
        return []
    num_blocks = max(old.num_blocks, int(declared[-1]) + 1)
    encoded = np.zeros(num_blocks, np.bool_)
    encoded[declared] = True
    encoded[old.num_blocks:] = True
    sizes = np.zeros(num_blocks, np.int64)
    sizes[:old.num_blocks] = old.sizes
    sizes[encoded] = 0
    sizes += np.bincount(new_ids, minlength=num_blocks)[:num_blocks]
    sources = np.where(encoded, 1, 0)
    values, ticks = _assemble(sizes, sources, [old.values, new_values], [old.ticks, new_ticks],
                              [_block_positions(old.offsets, ~encoded[:old.num_blocks], old.sizes),
                               np.arange(new_values.shape[0])])
    return _check_result(result, values, ticks, sizes, encoded, old.unit, params, params.time_error,
                         time_blocks=(start, duration))


def fixed_point(group: bytes, params: Params = DEFAULT_PARAMS) -> list[Violation]:
    """Checks that a block group's decoded data re-encode to themselves (ENCODER.md §6).

    Re-encodes the decoded values in the same blocks, with the decoded times stored exactly: the
    values decode unchanged, and a third encode gives the second's bytes. With a target it holds
    only nearly always (ENCODER.md §5.8), so nothing is checked. This costs two encodes.

    Returns:
        The violations found; empty if the block group is a fixed point.
    """
    if params.target_bits_per_sample is not None:
        return []
    exact = replace(params, time_error=0.0)
    first = _api.decode_group(group)
    second = _api.encode_blocks(first.values, first.block_sizes, exact, times=first.times).group
    again = _api.decode_group(second).values
    if not _same_doubles(again, first.values):
        moved = np.flatnonzero(again.view(np.int64) != first.values.view(np.int64))
        block = int(np.searchsorted(np.cumsum(first.block_sizes), moved[0], side="right"))
        return [Violation("fixed_point", block, f"{moved.shape[0]} decoded samples changed on re-encode, "
                          f"first sample {int(moved[0])}: {first.values[moved[0]]:.17g} -> {again[moved[0]]:.17g}")]
    third = _api.encode_blocks(again, first.block_sizes, exact, times=first.times).group
    if third != second:
        return [Violation("fixed_point", -1, "the third encode's bytes differ from the second's")]
    return []


# Decoding and the expected series

class _Decoded(NamedTuple):
    """A decoded block group with its grid columns."""

    unit: int
    sizes: np.ndarray
    offsets: np.ndarray
    flags: np.ndarray
    params: np.ndarray
    scales: np.ndarray
    values: np.ndarray
    ticks: np.ndarray | None

    @property
    def num_blocks(self) -> int:
        return self.sizes.shape[0]


def _decode(group: bytes) -> _Decoded:
    """Decodes a block group and reads its flags, grid parameters and grid scales."""
    parsed = _group.decompress(group)
    num_blocks = parsed.header.num_blocks
    values, ticks = _group.decode_blocks(parsed, np.arange(num_blocks, dtype=np.int64))
    raw = parsed.raw_body.astype(np.int64)
    start = _format.grid_params_start(num_blocks)
    grid_params = raw[start:start + num_blocks] | (raw[start + num_blocks:start + 2 * num_blocks] << 8)
    grid_params -= (grid_params & 0x8000) << 1
    start = _format.grid_scale_start(num_blocks)
    scales = np.zeros(num_blocks, np.int64)
    for byte_idx in range(4):
        scales |= raw[start + byte_idx * num_blocks:start + (byte_idx + 1) * num_blocks] << (8 * byte_idx)
    sizes = parsed.block_sizes.astype(np.int64)
    return _Decoded(parsed.header.time_unit, sizes, np.concatenate([[0], np.cumsum(sizes)]),
                    parsed.block_flags.astype(np.int64), grid_params, scales, values, ticks)


def _decode_or_report(group: bytes, found: list[Violation]) -> _Decoded | None:
    """Decodes an output block group, reporting a failure as a violation."""
    try:
        return _decode(group)
    except Exception as exc:  # noqa: BLE001 (any failure to decode is the finding)
        found.append(Violation("decode", -1, f"{type(exc).__name__}: {exc}"))
        return None


def _block_positions(offsets: np.ndarray, which: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """The sample positions of the chosen blocks, block after block."""
    lengths = np.where(which, sizes, 0)
    return np.arange(int(lengths.sum())) + np.repeat(offsets[:-1] - (np.cumsum(lengths) - lengths), lengths)


def _assemble(
    sizes: np.ndarray, sources: np.ndarray, values: list[np.ndarray], ticks: list[np.ndarray] | None,
    positions: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray | None]:
    """The expected series: block b's samples come from source sources[b], in block order.

    positions[s] lists, in block order, the samples of values[s] (and ticks[s]) that go to the
    blocks taken from source s.
    """
    out_values = np.empty(int(sizes.sum()))
    out_ticks = None if ticks is None else np.empty(out_values.shape[0], np.int64)
    sample_sources = np.repeat(sources, sizes)
    for source_idx, source_positions in enumerate(positions):
        target = sample_sources == source_idx
        out_values[target] = values[source_idx][source_positions]
        if out_ticks is not None and ticks is not None:
            out_ticks[target] = ticks[source_idx][source_positions]
    return out_values, out_ticks


def _time_block_ids(ticks: np.ndarray, start: int, duration: int) -> np.ndarray:
    """The time block of each tick: floor((tick - start) / duration), in exact unsigned arithmetic."""
    if ticks.shape[0] and int(ticks.min()) < start:
        raise ValueError("times must be at or after start_time")
    return ((ticks.view(np.uint64) - np.uint64(start % 2**64)) // np.uint64(duration)).astype(np.int64)


def _in_ranges(ticks: np.ndarray, ranges: np.ndarray) -> np.ndarray:
    """Which ticks lie in one of the sorted, disjoint [start, end) ranges."""
    if not ranges.shape[0]:
        return np.zeros(ticks.shape[0], np.bool_)
    idx = np.searchsorted(ranges[:, 0], ticks, side="right") - 1
    return (idx >= 0) & (ticks < ranges[np.maximum(idx, 0), 1])


def _met_blocks(ranges: np.ndarray, start: int, duration: int, num_old: int, num_blocks: int) -> np.ndarray:
    """The stored blocks a delete range overlaps (they may be re-encoded although nothing changed)."""
    met = np.zeros(num_blocks, np.bool_)
    for range_start, range_end in ranges.tolist():
        if range_end <= start:
            continue
        first = max(0, (range_start - start) // duration)
        last = min(num_old - 1, (range_end - 1 - start) // duration)
        met[first:last + 1] = True
    return met


def _neighbour_intervals(old: _Decoded, encoded: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """Per block: the largest mean interval of the stored, non-empty blocks either side of it.

    update clamps a new block to a stored neighbour within the neighbour's allowance.
    """
    assert old.ticks is not None
    num_blocks = sizes.shape[0]
    means = np.zeros(num_blocks)
    for block_idx in range(old.num_blocks):
        if not encoded[block_idx] and old.sizes[block_idx] > 1:
            first, last = int(old.ticks[old.offsets[block_idx]]), int(old.ticks[old.offsets[block_idx + 1] - 1])
            means[block_idx] = (last - first) / (int(old.sizes[block_idx]) - 1)
    allowance = np.zeros(num_blocks)
    stored = np.flatnonzero(~encoded & (sizes > 0))
    for block_idx in np.flatnonzero(encoded).tolist():
        before = stored[stored < block_idx]
        after = stored[stored > block_idx]
        for neighbour in ([before[-1]] if before.shape[0] else []) + ([after[0]] if after.shape[0] else []):
            allowance[block_idx] = max(allowance[block_idx], means[neighbour])
    return allowance


# The checks

def _check_encode(
    result: EncodedGroup, series: np.ndarray, sizes: np.ndarray, ticks: np.ndarray | None, unit: int, params: Params,
    time_blocks: tuple[int, int] | None = None,
) -> list[Violation]:
    """Checks a fresh encode: every block encoded from the given samples."""
    return _check_result(result, series, ticks, sizes, np.ones(sizes.shape[0], np.bool_), unit, params,
                         params.time_error, time_blocks=time_blocks)


def _check_result(
    result: EncodedGroup | UpdatedGroup,
    values: np.ndarray,
    ticks: np.ndarray | None,
    sizes: np.ndarray,
    encoded: np.ndarray,
    unit: int,
    params: Params,
    time_error: float,
    *,
    allowance: np.ndarray | None = None,
    time_blocks: tuple[int, int] | None = None,
) -> list[Violation]:
    """Compares a block group with the expected series.

    Blocks marked encoded are checked against the encoder's bounds for their samples (with the
    statistics in result, in block order); the others must decode bit for bit as given (the
    stored blocks of an update). Times are checked against time_error in encoded blocks and
    exactly elsewhere.
    """
    found: list[Violation] = []
    new = _decode_or_report(result.group, found)
    if new is None:
        return found
    if new.unit != unit:
        return [Violation("layout", -1, f"time unit code {new.unit}, not {unit}")]
    if new.num_blocks != sizes.shape[0] or not np.array_equal(new.sizes, sizes):
        return [Violation("layout", -1, f"block sizes {_summary(new.sizes)}, not {_summary(sizes)}")]
    if isinstance(result, UpdatedGroup) and not np.array_equal(result.indices, np.flatnonzero(encoded)):
        return [Violation("layout", -1, f"re-encoded indices {_summary(result.indices)}, "
                          f"not {_summary(np.flatnonzero(encoded))}")]
    if not encoded.all():  # the blocks an update carried over: bit for bit
        changed = ~np.repeat(encoded, sizes) & ~_same_doubles_mask(new.values, values)
        _report_samples(found, "untouched", changed, new.offsets, "samples of a block not re-encoded changed")
    _check_values(found, values, new, sizes, encoded, params, result)
    if ticks is not None and new.ticks is not None:
        _check_times(found, ticks, new.ticks, sizes, new.offsets, encoded, time_error, allowance, time_blocks)
    return found


def _check_values(
    found: list[Violation], given: np.ndarray, new: _Decoded, sizes: np.ndarray, encoded: np.ndarray,
    params: Params, result: EncodedGroup | UpdatedGroup,
) -> None:
    """The value bounds and statistics of the encoded blocks."""
    half_error, lo, hi, magnitude, count, mean, nonfinite_bad, nonfinite_first = _value_stats(
        given, new.values, new.offsets)
    _report_blocks(found, "nonfinite", encoded, nonfinite_bad, nonfinite_first,
                   "non-finite samples didn't decode as themselves")
    with np.errstate(all="ignore"):
        # At half scale, so that neither the errors nor the ranges of blocks spanning +-DBL_MAX overflow
        half_range = np.where(count > 0, hi / 2 - lo / 2, 0.0)
        half_ulps = 2 * np.spacing(magnitude)  # x - lo and lo + q step each round once (4 ulps)
        bounds = [
            ("range_bound", half_range / (2.0 ** params.min_quantize_bits - 0.5), "range / (2^min_quantize_bits - 1/2)"),
            ("step_bound", _grid_bounds(new) / 2, "the stored grid's"),
        ]
        if params.target_bits_per_sample is None:
            fine = (params.noise_floor_sigma == 0) | (sizes < NOISE_FLOOR_MIN_SAMPLES)
            fine_bound = np.where(fine, half_range / (2.0 ** params.max_quantize_bits - 0.5), np.inf)
            bounds.append(("fine_bound", fine_bound, "range / (2^max_quantize_bits - 1/2)"))
        for check, half_bound, what in bounds:
            over = encoded & (count > 0) & ~(half_error <= half_bound + half_ulps)
            for block_idx in np.flatnonzero(over)[:_MAX_REPORTED].tolist():
                found.append(Violation(check, block_idx, f"max error {2 * half_error[block_idx]:.17g} over "
                                       f"{what} {2 * half_bound[block_idx]:.17g}"))
    _check_stats(found, result, encoded, count, lo, hi, mean, magnitude)


@njit(nogil=True, cache=True)
def _value_stats(given: np.ndarray, got: np.ndarray, offsets: np.ndarray) -> tuple:
    """Per block: the largest error at half scale (finite samples), the finite samples' min, max,
    largest magnitude, count and mean, and the count and first index of the samples whose
    non-finite value (or finiteness) didn't survive.

    A branch-free pass first; blocks with a non-finite value (or a sum that overflows) take a
    second, careful one.
    """
    num_blocks = offsets.shape[0] - 1
    half_error = np.zeros(num_blocks)
    lo = np.full(num_blocks, np.inf)
    hi = np.full(num_blocks, -np.inf)
    magnitude = np.zeros(num_blocks)
    count = np.zeros(num_blocks, np.int64)
    mean = np.zeros(num_blocks)
    bad = np.zeros(num_blocks, np.int64)
    first_bad = np.full(num_blocks, -1, np.int64)
    for block_idx in range(num_blocks):
        first, end = offsets[block_idx], offsets[block_idx + 1]
        if first == end:
            continue
        _all_finite(given, got, first, end, block_idx, half_error, lo, hi, mean)
        if np.isfinite(half_error[block_idx]) and np.isfinite(mean[block_idx]):
            count[block_idx] = end - first
        else:
            _careful(given, got, first, end, block_idx, half_error, lo, hi, count, mean, bad, first_bad)
        if count[block_idx]:
            magnitude[block_idx] = max(-lo[block_idx], hi[block_idx])
    return half_error, lo, hi, magnitude, count, mean, bad, first_bad


@njit(nogil=True, cache=True, fastmath={"reassoc", "nsz"})
def _all_finite(given, got, first, end, block_idx, half_error, lo, hi, mean):
    """The statistics of a block taken as all finite: a non-finite sample makes the error NaN."""
    error, low, high, total, poison = 0.0, np.inf, -np.inf, 0.0, 0.0
    for sample_idx in range(first, end):
        x, y = given[sample_idx], got[sample_idx]
        error = max(error, abs(0.5 * y - 0.5 * x))
        low = min(low, x)
        high = max(high, x)
        total += x
        poison += (x - x) + (y - y)  # NaN if either is non-finite
    half_error[block_idx] = error + poison
    lo[block_idx], hi[block_idx] = low, high
    mean[block_idx] = total / (end - first)


@njit(nogil=True, cache=True)
def _careful(given, got, first, end, block_idx, half_error, lo, hi, count, mean, bad, first_bad):
    """The statistics of a block with non-finite samples, or whose sum overflows."""
    half_error[block_idx], lo[block_idx], hi[block_idx], count[block_idx] = 0.0, np.inf, -np.inf, 0
    total = 0.0
    for sample_idx in range(first, end):
        x, y = given[sample_idx], got[sample_idx]
        if np.isfinite(x):
            if np.isfinite(y):
                half_error[block_idx] = max(half_error[block_idx], abs(0.5 * y - 0.5 * x))
            else:
                bad[block_idx] += 1
                if first_bad[block_idx] < 0:
                    first_bad[block_idx] = sample_idx
            lo[block_idx] = min(lo[block_idx], x)
            hi[block_idx] = max(hi[block_idx], x)
            count[block_idx] += 1
            total += x
        elif not (x == y or (np.isnan(x) and np.isnan(y))):
            bad[block_idx] += 1
            if first_bad[block_idx] < 0:
                first_bad[block_idx] = sample_idx
    mean[block_idx] = total / max(count[block_idx], 1)
    if not np.isfinite(mean[block_idx]):  # the sum overflowed: divide first
        mean[block_idx] = 0.0
        for sample_idx in range(first, end):
            if np.isfinite(given[sample_idx]):
                mean[block_idx] += given[sample_idx] / count[block_idx]


def _check_stats(
    found: list[Violation], result: EncodedGroup | UpdatedGroup, encoded: np.ndarray, count: np.ndarray,
    lo: np.ndarray, hi: np.ndarray, mean: np.ndarray, magnitude: np.ndarray,
) -> None:
    """The returned per-block statistics of the encoded blocks."""
    blocks = np.flatnonzero(encoded)
    stats = [np.asarray(result.block_min), np.asarray(result.block_max), np.asarray(result.block_mean)]
    if any(stat.shape != blocks.shape for stat in stats):
        found.append(Violation("stats", -1, f"{[stat.shape for stat in stats]} statistics for {blocks.shape[0]} blocks"))
        return
    some = count[blocks] > 0
    expected = [np.where(some, lo[blocks], np.nan), np.where(some, hi[blocks], np.nan)]
    with np.errstate(invalid="ignore"):
        # The mean's summation (fastmath, in any order) errs by up to n ulps of the largest sample
        mean_tolerance = 2.0 * count[blocks] * np.spacing(magnitude[blocks])
        mean_ok = np.where(some, np.abs(stats[2] - mean[blocks]) <= mean_tolerance, np.isnan(stats[2]))
    for name, stat, want in zip(("block_min", "block_max"), stats, expected):
        for pos in np.flatnonzero(~((stat == want) | (np.isnan(stat) & np.isnan(want))))[:_MAX_REPORTED].tolist():
            found.append(Violation("stats", int(blocks[pos]), f"{name} {stat[pos]:.17g}, not {want[pos]:.17g}"))
    for pos in np.flatnonzero(~mean_ok)[:_MAX_REPORTED].tolist():
        want_mean = mean[blocks[pos]] if some[pos] else np.nan
        found.append(Violation("stats", int(blocks[pos]), f"block_mean {stats[2][pos]:.17g}, not {want_mean:.17g}"))


def _grid_bounds(new: _Decoded) -> np.ndarray:
    """Per block, the bound its stored grid puts on the error.

    Power-of-two grid: half the step 2^e. Decimal grid: a quarter of 10^p, as detection only takes
    samples within a quarter of the planned power-of-two step, which is finer than 10^p. Scaled
    grid: half its step N / D B^p (the index gcd is lossless on its grid; the rational search
    within a quarter of the planned step, finer than the scaled one).
    """
    decimal = (new.flags & BLOCK_FLAG_DECIMAL) != 0
    scaled = (new.flags & BLOCK_FLAG_SCALED) != 0
    with np.errstate(over="ignore"):
        base = np.where(decimal, 10.0 ** np.clip(new.params, -22, 22), np.ldexp(1.0, np.clip(new.params, -1074, 1023)))
    numerators, denominators = new.scales & 0xFFFF, new.scales >> 16
    step = np.where(scaled, base * numerators / np.maximum(denominators, 1), base)
    return np.where(decimal & ~scaled, step / 4, step / 2)


def _check_times(
    found: list[Violation], given: np.ndarray, got: np.ndarray, sizes: np.ndarray, offsets: np.ndarray,
    encoded: np.ndarray, time_error: float, allowance: np.ndarray | None, time_blocks: tuple[int, int] | None,
) -> None:
    """Order, exact times, the time error of rounded blocks, and time block ranges."""
    steps = np.diff(got)
    if (steps < 0).any():
        first = int(np.flatnonzero(steps < 0)[0]) + 1
        found.append(Violation("times", _block_of(offsets, first), f"time of sample {first} decreases"))
    rounded, wrong, first_wrong, largest, far, mean_interval = _time_stats(given, got, offsets, encoded, time_error)
    _report_blocks(found, "times", ~rounded, wrong, first_wrong, "times stored exactly decoded differently")
    if allowance is not None:
        mean_interval = np.maximum(mean_interval, allowance)
    cap = time_error * TIME_SLACK * mean_interval * (1 + 1e-12) + 1e-6
    for block_idx in np.flatnonzero(rounded & (largest > cap))[:_MAX_REPORTED].tolist():
        found.append(Violation("time_error", block_idx, f"a time moved {largest[block_idx]:g} ticks, over "
                               f"{TIME_SLACK} x {time_error} x the mean interval {mean_interval[block_idx]:g}"))
    for block_idx in np.flatnonzero(rounded & (far > sizes // 20))[:_MAX_REPORTED].tolist():
        found.append(Violation("time_error", block_idx, f"{far[block_idx]} of {sizes[block_idx]} times moved more "
                               f"than {time_error} x their own interval (at most 1 in 20)"))
    if time_blocks is not None:
        start, duration = time_blocks
        block = np.repeat(np.arange(sizes.shape[0]), sizes)
        offset = got.view(np.uint64) - np.uint64(start % 2**64)
        ids = (offset // np.uint64(duration)).astype(np.int64)
        at_end = (ids == block + 1) & (offset % np.uint64(duration) == 0)
        _report_samples(found, "time_block", (got < start) | ~((ids == block) | at_end), offsets,
                        "times decoded outside their time block")


@njit(nogil=True, cache=True)
def _time_stats(given: np.ndarray, got: np.ndarray, offsets: np.ndarray, encoded: np.ndarray,
                time_error: float) -> tuple:
    """Per block, in one pass: whether it was rounded (encoded with a time error, at least 3 times,
    not regular as given); if not, how many times differ (and the first); if so, the largest
    move, how many moved more than time_error times their own interval (the smaller of the two
    either side, as given), and its mean interval. Differences are taken modulo 2^64: given
    times are non-decreasing, so their gaps are exact as unsigned."""
    num_blocks = offsets.shape[0] - 1
    rounded = np.zeros(num_blocks, np.bool_)
    wrong = np.zeros(num_blocks, np.int64)
    first_wrong = np.full(num_blocks, -1, np.int64)
    largest = np.zeros(num_blocks)
    far = np.zeros(num_blocks, np.int64)
    mean_interval = np.zeros(num_blocks)
    for block_idx in range(num_blocks):
        first, end = offsets[block_idx], offsets[block_idx + 1]
        if encoded[block_idx] and time_error > 0 and end - first >= 3:
            gap = np.uint64(given[first + 1] - given[first])
            for sample_idx in range(first + 2, end):
                if np.uint64(given[sample_idx] - given[sample_idx - 1]) != gap:
                    rounded[block_idx] = True
                    break
        if not rounded[block_idx]:
            for sample_idx in range(first, end):
                if got[sample_idx] != given[sample_idx]:
                    wrong[block_idx] += 1
                    if first_wrong[block_idx] < 0:
                        first_wrong[block_idx] = sample_idx
            continue
        mean_interval[block_idx] = float(np.uint64(given[end - 1] - given[first])) / (end - first - 1)
        for sample_idx in range(first, end):
            move = abs(float(got[sample_idx] - given[sample_idx]))
            largest[block_idx] = max(largest[block_idx], move)
            own = np.inf
            if sample_idx > first:
                own = float(np.uint64(given[sample_idx] - given[sample_idx - 1]))
            if sample_idx + 1 < end:
                own = min(own, float(np.uint64(given[sample_idx + 1] - given[sample_idx])))
            if move > time_error * own:
                far[block_idx] += 1
    return rounded, wrong, first_wrong, largest, far, mean_interval


# Helpers

_MAX_REPORTED = 5
"""Violations reported per check (the first blocks): one is enough to act on."""


def _same_doubles_mask(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per sample: the same double, NaN equal to NaN and -0.0 to 0.0 (as decoding gives them)."""
    with np.errstate(invalid="ignore"):
        return (a == b) | (np.isnan(a) & np.isnan(b))


def _same_doubles(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and bool(_same_doubles_mask(a, b).all())


def _block_of(offsets: np.ndarray, sample_idx: int) -> int:
    return int(np.searchsorted(offsets, sample_idx, side="right")) - 1


def _report_blocks(
    found: list[Violation], check: str, which: np.ndarray, counts: np.ndarray, firsts: np.ndarray, what: str,
) -> None:
    """One violation per chosen block with a bad sample (the first few blocks), from per-block counts."""
    for block_idx in np.flatnonzero(which & (counts > 0))[:_MAX_REPORTED].tolist():
        found.append(Violation(check, block_idx, f"{what}: {counts[block_idx]}, first sample {firsts[block_idx]}"))


def _report_samples(found: list[Violation], check: str, bad: np.ndarray, offsets: np.ndarray, what: str) -> None:
    """One violation per block with a bad sample (the first few blocks), naming the first sample."""
    if not bad.any():
        return
    samples = np.flatnonzero(bad)
    blocks = np.searchsorted(offsets, samples, side="right") - 1
    unique_blocks, first = np.unique(blocks, return_index=True)
    counts = np.bincount(blocks - blocks.min())[unique_blocks - blocks.min()]
    for block_idx, sample_idx, count in list(zip(unique_blocks.tolist(), samples[first].tolist(),
                                                 counts.tolist()))[:_MAX_REPORTED]:
        found.append(Violation(check, block_idx, f"{what}: {count}, first sample {sample_idx}"))


def _summary(array: np.ndarray, limit: int = 8) -> str:
    items = np.asarray(array).tolist()
    return str(items) if len(items) <= limit else f"{items[:limit]}... ({len(items)})"
