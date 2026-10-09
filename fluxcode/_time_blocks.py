# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Encodes and updates block groups whose blocks are divided by time.

Block b of a block group holds the samples timed in
[start_time + b * block_duration, start_time + (b + 1) * block_duration).

The start time and duration aren't stored: the caller keeps them (typically the start is
part of the block group's storage key) and passes the same ones to update_time_blocks. Times, the
start, the duration and delete ranges are all converted to int64 ticks in the times' unit.

With a time error (Params.time_error), times are rounded with the quanta of the blocks they
fall in, then assigned to blocks: a time that rounds across a boundary belongs to the block it
lands in, and one that would round below the start time is raised to it (still within half a
quantum of where it was). Every block's times stay within its range.
"""

import numpy as np
import numpy.typing as npt
from numba import njit

from . import _args, _format, _group, _time
from ._args import DurationLike, RangesLike, TimeLike
from ._types import EncodedGroup, Params, UpdatedGroup


def block_ids(ticks: np.ndarray, start: int, duration: int) -> np.ndarray:
    """Calculates the block index for each timestamp tick.

    Args:
        ticks: 1D int64 array of sample timestamp ticks.
        start: Starting timestamp tick of block 0.
        duration: Positive integer duration in ticks per block.

    Returns:
        1D int64 array of block indices corresponding to each tick.

    Raises:
        ValueError: If duration is non-positive, any tick precedes start, or
            any tick maps to a block index at or beyond MAX_BLOCKS.
    """
    if duration <= 0:
        raise ValueError(f"block_duration must be positive, got {duration} ticks")
    if ticks.shape[0] and ticks.min() < start:
        raise ValueError(f"times must be at or after start_time: got {int(ticks.min())} before {start}")
    # Compute block indices as unsigned 64-bit integers to avoid overflow
    ids = (ticks.astype(np.uint64) - np.uint64(start % 2**64)) // np.uint64(duration)
    if ids.shape[0] and ids.max() >= _format.MAX_BLOCKS:
        raise ValueError(f"times reach block {int(ids.max())}, past the last block a block group holds ({_format.MAX_BLOCKS - 1})")
    return ids.astype(np.int64)


def chunk(ticks: np.ndarray, start: int, duration: int) -> np.ndarray:
    """Calculates block sample counts for sorted timestamps.

    Args:
        ticks: 1D int64 array of sorted timestamp ticks.
        start: Starting timestamp tick of block 0.
        duration: Duration of each block in ticks.

    Returns:
        1D int64 array of block sizes from block 0 through the last non-empty block.
    """
    ids = block_ids(ticks, start, duration)
    return np.bincount(ids, minlength=int(ids[-1]) + 1 if ids.shape[0] else 0).astype(np.int64)


def encode_time_blocks(
    samples: npt.ArrayLike,
    times: npt.ArrayLike,
    params: Params,
    start_time: TimeLike,
    block_duration: DurationLike,
    time_unit: str | None,
) -> EncodedGroup:
    """Encodes a timed series into fixed-duration blocks in a single block group.

    Args:
        samples: 1D array-like of series samples (may be empty).
        times: Timestamps matching samples in length and non-decreasing in order.
        params: Encoder configuration parameters.
        start_time: Starting time anchor for block 0.
        block_duration: Duration of each block.
        time_unit: Optional time unit string for integer timestamps.

    Returns:
        EncodedGroup tuple (group, block_min, block_max, block_mean).

    Raises:
        ValueError: If timestamps or time parameters are invalid.
    """
    series_arr = _args.as_series(samples, allow_empty=True)
    ticks, unit_code = _args.series_ticks(times, time_unit, series_arr.shape[0])
    assert ticks is not None
    start = _args.to_ticks(start_time, unit_code, "start_time")
    duration = _args.to_ticks(block_duration, unit_code, "block_duration", duration=True)
    sizes = chunk(ticks, start, duration)
    if params.time_error > 0:
        # Quanta from the blocks the times are in; a time rounded across a boundary moves on
        snapped, _ = _group.snap_ticks(ticks, sizes, params.time_error)
        if snapped is not ticks:
            ticks = np.maximum(snapped, start, out=snapped)
            sizes = chunk(ticks, start, duration)
    return _group.encode(series_arr, sizes, params, ticks, unit_code, snap=False)


@njit(nogil=True, cache=True)
def _merge_samples(
    old_ticks: np.ndarray,
    old_values: np.ndarray,
    old_blocks: np.ndarray,
    new_ticks: np.ndarray,
    new_values: np.ndarray,
    new_blocks: np.ndarray,
    ranges: np.ndarray,
    changed: np.ndarray,
    out_ticks: np.ndarray,
    out_values: np.ndarray,
    block_sizes: np.ndarray,
) -> int:
    """Merges new samples into existing samples in chronological order.

    An existing sample is deleted if any deletion range covers it or if a new sample
    shares its timestamp. New samples at identical timestamps preserve their input order.
    Only blocks that gain or lose samples have their contents written.

    Args:
        old_ticks: 1D int64 array of existing sample timestamps.
        old_values: 1D float64 array of existing sample values.
        old_blocks: 1D int64 array mapping each existing sample to its block index.
        new_ticks: 1D int64 array of new sample timestamps.
        new_values: 1D float64 array of new sample values.
        new_blocks: 1D int64 array mapping each new sample to its block index.
        ranges: 2D int64 array of disjoint [start, end) deletion ranges.
        changed: Output boolean array of every block (initially False), set to True for
            blocks that gain or lose samples.
        out_ticks: Output 1D int64 array receiving merged timestamps.
        out_values: Output 1D float64 array receiving merged sample values.
        block_sizes: Output 1D int64 array of every block (initially 0) receiving the sample
            count of each changed block.

    Returns:
        The total number of samples written to out_ticks and out_values.
    """
    num_old, num_new, num_ranges = old_ticks.shape[0], new_ticks.shape[0], ranges.shape[0]
    keep = np.empty(num_old, np.bool_)
    range_idx = 0
    new_idx = 0
    # Determine which old samples survive deletions and replacements
    for old_idx in range(num_old):
        tick = old_ticks[old_idx]
        while new_idx < num_new and new_ticks[new_idx] < tick:
            new_idx += 1
        while range_idx < num_ranges and ranges[range_idx, 1] <= tick:
            range_idx += 1
        gone = (new_idx < num_new and new_ticks[new_idx] == tick) or (
            range_idx < num_ranges and ranges[range_idx, 0] <= tick
        )
        keep[old_idx] = not gone
        if gone:
            changed[old_blocks[old_idx]] = True
    # Mark blocks with incoming new samples as changed
    for new_idx in range(num_new):
        changed[new_blocks[new_idx]] = True
    write_idx = 0
    new_idx = 0
    # Interleave surviving old samples and new samples into output arrays
    for old_idx in range(num_old + 1):
        while new_idx < num_new and (old_idx == num_old or new_ticks[new_idx] < old_ticks[old_idx]):
            out_ticks[write_idx] = new_ticks[new_idx]
            out_values[write_idx] = new_values[new_idx]
            block_sizes[new_blocks[new_idx]] += 1
            write_idx += 1
            new_idx += 1
        if old_idx < num_old and keep[old_idx] and changed[old_blocks[old_idx]]:
            out_ticks[write_idx] = old_ticks[old_idx]
            out_values[write_idx] = old_values[old_idx]
            block_sizes[old_blocks[old_idx]] += 1
            write_idx += 1
    return write_idx


def _block_masks(ranges: np.ndarray, start: int, duration: int, num_blocks: int) -> tuple[np.ndarray, np.ndarray]:
    """Finds the existing blocks that deletion ranges touch, and those they cover.

    A block is touched if any range overlaps it, and covered if ranges contain it whole.

    Args:
        ranges: 2D int64 array of sorted, disjoint [start, end) deletion ranges.
        start: Starting timestamp tick of block 0.
        duration: Positive integer duration in ticks per block.
        num_blocks: Number of existing blocks.

    Returns:
        A tuple of (touched, covered) boolean arrays of length num_blocks.
    """
    touched = np.zeros(num_blocks, np.bool_)
    covered = np.zeros(num_blocks, np.bool_)
    for range_start, range_end in ranges.tolist():
        if range_end <= start:
            continue
        touched[min(max(range_start - start, 0) // duration, num_blocks):min(-(-(range_end - start) // duration), num_blocks)] = True
        covered[min(max(-(-(range_start - start) // duration), 0), num_blocks):min((range_end - start) // duration, num_blocks)] = True
    return touched, covered


def update_time_blocks(
    group: bytes,
    samples: npt.ArrayLike,
    times: npt.ArrayLike,
    params: Params,
    start_time: TimeLike,
    block_duration: DurationLike,
    delete_ranges: RangesLike | None,
    time_unit: str | None,
) -> UpdatedGroup:
    """Applies point updates and range deletions to time-partitioned block groups.

    Args:
        group: Existing compressed block group bytes.
        samples: Array-like of new samples (may be empty).
        times: Array-like of timestamps for the new samples.
        params: Encoder configuration parameters.
        start_time: Starting time anchor for block 0.
        block_duration: Duration of each block.
        delete_ranges: Optional time intervals to delete before adding samples.
        time_unit: Optional time unit string, checked against block group header.

    Returns:
        UpdatedGroup tuple containing the updated block group and modified block statistics.

    Raises:
        ValueError: If block group lacks a time axis, arguments are invalid, or sizes overflow.
    """
    parsed = _group.decompress(group)
    if not parsed.has_time:
        raise ValueError("the block group has no time axis: use update")
    # Validate series samples, timestamps, and deletion intervals
    series_samples, ticks, start, duration, ranges = _args.time_blocks_update(
        samples, times, start_time, block_duration, delete_ranges, time_unit, parsed.header.time_unit
    )
    new_ids = block_ids(ticks, start, duration)
    steps = np.diff(ticks)
    if new_ids.shape[0] and (steps < 0).any():
        raise _args.decrease_error(ticks, int(np.flatnonzero(steps < 0)[0]) + 1)
    num_old_blocks = parsed.header.num_blocks
    touched, covered = _block_masks(ranges, start, duration, num_old_blocks)
    touched &= ~covered
    # Decode existing blocks touched by deletions or new samples (excluding fully covered ones)
    decoded = np.flatnonzero(touched | _new_blocks_mask(new_ids, num_old_blocks, covered))
    time_rows = _group.read_time_rows(parsed, decoded)
    gathered = _gather(parsed, decoded, time_rows)
    if params.time_error > 0 and ticks.shape[0]:
        ticks = np.maximum(_snap_new_ticks(gathered, ticks, series_samples, new_ids, ranges, params.time_error), start)
        new_ids = block_ids(ticks, start, duration)
        # A time rounded across a boundary belongs to the next block: decode it too, if stored
        extra = np.setdiff1d(np.flatnonzero(_new_blocks_mask(new_ids, num_old_blocks, covered)), decoded)
        if extra.shape[0]:
            decoded = np.union1d(decoded, extra)
            time_rows = _group.read_time_rows(parsed, decoded)
            gathered = _gather(parsed, decoded, time_rows)
    gathered_ticks, gathered_values, gathered_blocks = gathered
    # Determine total block count including newly appended blocks
    num_blocks = max(num_old_blocks, int(new_ids[-1]) + 1 if new_ids.shape[0] else 0)
    changed = np.zeros(num_blocks, np.bool_)
    sizes = np.zeros(num_blocks, np.int64)
    num_gathered = gathered_ticks.shape[0]
    merged_ticks = np.empty(num_gathered + ticks.shape[0], np.int64)
    merged_values = np.empty(num_gathered + ticks.shape[0])
    # Merge existing and new samples chronologically
    num_merged = _merge_samples(
        gathered_ticks, gathered_values, gathered_blocks, ticks, series_samples, new_ids, ranges, changed, merged_ticks,
        merged_values, sizes,
    )
    # Account for covered blocks that lost all existing samples
    covered_ids = np.flatnonzero(covered)
    changed[covered_ids] |= parsed.block_sizes[covered_ids] > 0
    indices = np.flatnonzero(changed)
    if not indices.shape[0]:
        return UpdatedGroup(group, np.zeros(0, np.int64), np.zeros(0), np.zeros(0), np.zeros(0))
    sizes = sizes[indices]
    if sizes.max() > _format.MAX_BLOCK_LEN:
        raise ValueError(f"a block would hold over {_format.MAX_BLOCK_LEN} samples")
    # Splice updated blocks into the block group while carrying unchanged blocks verbatim (the
    # new samples are rounded; stored ones keep their times)
    return _group.splice(
        parsed,
        indices,
        merged_values[:num_merged],
        sizes,
        merged_ticks[:num_merged],
        params,
        time_rows,
        snap=False,
    )


def _new_blocks_mask(new_ids: np.ndarray, num_old_blocks: int, covered: np.ndarray) -> np.ndarray:
    """Marks the stored blocks new samples fall in, except those deletions cover whole.

    Args:
        new_ids: 1D int64 array of the new samples' blocks.
        num_old_blocks: Number of stored blocks.
        covered: 1D bool array of the stored blocks deletions cover.

    Returns:
        1D bool array of num_old_blocks.
    """
    mask = np.zeros(num_old_blocks, np.bool_)
    mask[new_ids[new_ids < num_old_blocks]] = True
    return mask & ~covered


def _gather(
    parsed: _group.ParsedGroup, decoded: np.ndarray, time_rows: _format.TimeRows
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decodes stored blocks into one chronological run.

    Args:
        parsed: The block group.
        decoded: 1D int64 array of the blocks to decode, increasing.
        time_rows: Its time rows, with those blocks' residuals.

    Returns:
        A tuple of (ticks, values, blocks): the blocks' samples, block after block, and the
        block of each.
    """
    if not decoded.shape[0]:
        return np.zeros(0, np.int64), np.zeros(0), np.zeros(0, np.int64)
    all_values, all_ticks = _group.decode_blocks(parsed, decoded, time_rows)
    assert all_ticks is not None
    run_sizes = parsed.block_sizes[decoded]
    run_starts = parsed.layout.sample_offsets[decoded]
    positions = np.arange(int(run_sizes.sum())) + np.repeat(run_starts - (np.cumsum(run_sizes) - run_sizes), run_sizes)
    return all_ticks[positions], all_values[positions], np.repeat(decoded, run_sizes)


def _snap_new_ticks(
    gathered: tuple[np.ndarray, np.ndarray, np.ndarray],
    ticks: np.ndarray,
    values: np.ndarray,
    new_ids: np.ndarray,
    ranges: np.ndarray,
    time_error: float,
) -> np.ndarray:
    """Rounds new samples' ticks to the quanta of the blocks they fall in.

    A block's quantum and phase come from its samples after the update as given (stored and
    new, before rounding), so a few new samples round to the grid of the stored ones.

    Args:
        gathered: (ticks, values, blocks) of the decoded stored blocks (_gather).
        ticks: 1D int64 array of the new samples' ticks, non-decreasing.
        values: 1D float64 array of the new samples.
        new_ids: 1D int64 array of the new samples' blocks.
        ranges: 2D int64 array of the deletion ranges.
        time_error: The time error (> 0).

    Returns:
        1D int64 array of the new samples' rounded ticks (in order).
    """
    old_ticks, old_values, old_blocks = gathered
    num_blocks = max(int(old_blocks[-1]) + 1 if old_blocks.shape[0] else 0, int(new_ids[-1]) + 1)
    changed = np.zeros(num_blocks, np.bool_)
    sizes = np.zeros(num_blocks, np.int64)
    num_total = old_ticks.shape[0] + ticks.shape[0]
    merged_ticks, merged_values = np.empty(num_total, np.int64), np.empty(num_total)
    num_merged = _merge_samples(
        old_ticks, old_values, old_blocks, ticks, values, new_ids, ranges, changed, merged_ticks, merged_values, sizes
    )
    quanta = np.zeros(num_blocks, np.int64)
    regular = np.zeros(num_blocks, np.bool_)
    merged_offsets = _group.sample_offsets(sizes)
    _time.time_quanta(
        merged_ticks[:num_merged], merged_offsets, time_error, np.empty(_time.CADENCE_SAMPLES, np.int64), quanta,
        regular,
    )
    # A regular block is left as it is, as in snap_ticks
    quanta[regular] = 0
    if not quanta.any():
        return ticks
    # Phases from the stored and new samples too: stored ones on a grid hold it
    phases = np.zeros(num_blocks, np.int64)
    _time.time_phases(
        merged_ticks[:num_merged], merged_offsets, quanta, np.full(num_blocks, _time.INT64_MIN, np.int64), phases
    )
    snapped = np.empty_like(ticks)
    new_sizes = np.bincount(new_ids, minlength=num_blocks).astype(np.int64)
    status, _ = _time.snap_times(ticks, _group.sample_offsets(new_sizes), quanta, phases, snapped)
    assert status == _time.OK
    return snapped
