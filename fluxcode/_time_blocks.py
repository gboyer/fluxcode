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

In an update_time_blocks a time moves at most twice: onto a block's own regular lattice (within
half the time error), and onto a 1-2-5 grid, on which it then stays (_round_updates has the rules).
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
        ticks, edits = _round_updates(
            parsed, time_rows, gathered, ticks, new_ids, ranges, params.time_error, start, duration, covered
        )
        new_ids = block_ids(ticks, start, duration)
        # A time rounded across a boundary belongs to the next block: decode it too, if stored
        extra = np.setdiff1d(np.flatnonzero(_new_blocks_mask(new_ids, num_old_blocks, covered)), decoded)
        if extra.shape[0] or edits:
            decoded = np.union1d(decoded, extra)
            time_rows = _group.read_time_rows(parsed, decoded)
            gathered = _gather(parsed, decoded, time_rows)
            for block_idx, block_ticks in edits.items():
                gathered[0][gathered[2] == block_idx] = block_ticks
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
    # ticks are rounded already)
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


def _round_to_grid(ticks: np.ndarray, quantum: int | np.ndarray, phase: int | np.ndarray) -> np.ndarray:
    """Rounds ticks to the nearest phase + k quantum, ties up (as _time.snap_times).

    Args:
        ticks: 1D int64 array of ticks.
        quantum: The grid's step in ticks (> 1), or one per tick.
        phase: The grid's phase, 0 <= phase < quantum, or one per tick.

    Returns:
        1D int64 array of the rounded ticks.
    """
    remainder = (ticks - phase) % quantum
    down = ticks - remainder
    return np.where(remainder < quantum - remainder, down, down + quantum)


def _merged_ticks(old_ticks: np.ndarray, new_ticks: np.ndarray, ranges: np.ndarray) -> np.ndarray:
    """The ticks of one block after an update: its stored samples that survive, and the new ones.

    Args:
        old_ticks: 1D int64 array of the block's stored ticks.
        new_ticks: 1D int64 array of its new ticks, non-decreasing.
        ranges: 2D int64 array of the deletion ranges.

    Returns:
        1D int64 array of the merged ticks.
    """
    num_old, num_new = old_ticks.shape[0], new_ticks.shape[0]
    out = np.empty(num_old + num_new, np.int64)
    num_merged = _merge_samples(
        old_ticks, np.zeros(num_old), np.zeros(num_old, np.int64), new_ticks, np.zeros(num_new),
        np.zeros(num_new, np.int64), ranges, np.zeros(1, np.bool_), out, np.empty(num_old + num_new),
        np.zeros(1, np.int64),
    )
    return out[:num_merged]


def _merged_quantum(merged: np.ndarray, time_error: float) -> int:
    """The quantum the time error gives a block's ticks (_time.time_quanta): 0 if none."""
    quantum = np.zeros(1, np.int64)
    _time.time_quanta(
        merged, np.array([0, merged.shape[0]], np.int64), time_error, np.empty(_time.CADENCE_SAMPLES, np.int64),
        quantum, np.zeros(1, np.bool_),
    )
    return int(quantum[0])


def _merged_phase(merged: np.ndarray, quantum: int) -> int:
    """The phase of a block's grid (_time.time_phases), chosen from the block alone."""
    phase = np.zeros(1, np.int64)
    _time.time_phases(
        merged, np.array([0, merged.shape[0]], np.int64), np.array([quantum], np.int64),
        np.full(1, _time.INT64_MIN, np.int64), phase,
    )
    return int(phase[0])


def _nested_quantum(limit: int, step: int) -> int:
    """The largest 1-2-5 x 10^k value of at most limit that nests with step (divides it or is
    divided by it); any if step is 0. 1 if there is none."""
    decade = 1
    while decade * 10 <= limit:
        decade *= 10
    while decade >= 1:
        for mantissa in (5, 2, 1):
            value = mantissa * decade
            if value <= limit and (step <= 0 or step % value == 0 or value % step == 0):
                return value
        decade //= 10
    return 1


def _stored_grid(ticks: np.ndarray) -> tuple[int, int]:
    """Finds the 1-2-5 grid a stored block's ticks lie on.

    The coarsest step of at most the block's median interval that CADENCE_SHARE of (up to 512
    evenly spaced) ticks lie on, at the phase of the middle tick; the few ticks that order fixes
    moved at the block's ends may be off it.

    Args:
        ticks: 1D int64 array of the block's stored ticks.

    Returns:
        A tuple of (step, tick): the step (0 if none >= 2) and a tick on the grid.
    """
    if ticks.shape[0] < 2:
        return 0, 0
    cap = int(np.median(np.diff(ticks)))
    sample = ticks if ticks.shape[0] <= 512 else ticks[np.linspace(0, ticks.shape[0] - 1, 512).astype(np.int64)]
    middle = int(ticks[ticks.shape[0] // 2])
    relative = sample - middle
    decade = 1
    while decade * 10 <= cap:
        decade *= 10
    while decade >= 1:
        for mantissa in (5, 2, 1):
            step = mantissa * decade
            if 2 <= step <= cap and np.count_nonzero(relative % step == 0) >= _time.CADENCE_SHARE * sample.shape[0]:
                return step, middle
        decade //= 10
    return 0, 0


def _round_updates(
    parsed: _group.ParsedGroup,
    time_rows: _format.TimeRows,
    gathered: tuple[np.ndarray, np.ndarray, np.ndarray],
    ticks: np.ndarray,
    new_ids: np.ndarray,
    ranges: np.ndarray,
    time_error: float,
    start: int,
    duration: int,
    covered: np.ndarray,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Rounds an update's new ticks so that no tick moves further than the time error in all.

    A tick moves at most twice: onto a block's own regular lattice by up to half the time error,
    and onto a 1-2-5 grid, after which it never moves again (later grids nest in it). For each
    block that has stored samples left after the update:

    0. Fewer than _time.MIN_ROUND_SAMPLES samples, or exactly regular: nothing is rounded.
    1. A stored regular block: the new ticks go to its lattice (first tick + k interval) if
       all are within half the time error of it. Stored ticks stay.
    2. Stored ticks that lie on a 1-2-5 grid (_stored_grid): they stay, and the new ticks go to
       the coarsest 1-2-5 grid the merged block's quantum allows that nests with it.
    3. Otherwise the whole block, stored ticks too, is rounded to the grid of its ticks after the
       update (_time.time_quanta, time_phases). If the stored ticks lie on a lattice coarser than
       that quantum, they may be on a lattice by rule 1, and the quantum is at most that of
       half the time error.

    Blocks without stored samples are rounded as in an encode, following the block before them
    (_group.snap_ticks).

    Args:
        parsed: The block group.
        time_rows: Its time rows (the columns at least).
        gathered: (ticks, values, blocks) of the decoded stored blocks (_gather).
        ticks: 1D int64 array of the new samples' ticks, non-decreasing.
        new_ids: 1D int64 array of the new samples' blocks.
        ranges: 2D int64 array of the deletion ranges.
        time_error: The time error (> 0).
        start: Starting tick of block 0.
        duration: Block duration in ticks.
        covered: 1D bool array of the stored blocks the deletion ranges cover whole.

    Returns:
        A tuple of (ticks, edits): the new samples' rounded ticks (non-decreasing, none below
        start), and the rounded stored ticks of each stored block that rule 3 changed.
    """
    old_ticks, _, old_blocks = gathered
    out = ticks.copy()
    edits: dict[int, np.ndarray] = {}
    blocks, first, counts = np.unique(new_ids, return_index=True, return_counts=True)
    old_first = np.searchsorted(old_blocks, blocks, "left")
    old_end = np.searchsorted(old_blocks, blocks, "right")
    flags = parsed.block_flags
    # Every block's ticks after the update, as given: one merge, sliced per block
    num_blocks = int(blocks[-1]) + 1
    if old_blocks.shape[0]:
        num_blocks = max(num_blocks, int(old_blocks[-1]) + 1)
    sizes = np.zeros(num_blocks, np.int64)
    all_merged = np.empty(old_ticks.shape[0] + ticks.shape[0], np.int64)
    _merge_samples(
        old_ticks, np.zeros(old_ticks.shape[0]), old_blocks, ticks, np.zeros(ticks.shape[0]), new_ids, ranges,
        np.zeros(num_blocks, np.bool_), all_merged, np.empty(all_merged.shape[0]), sizes,
    )
    merged_offsets = _group.sample_offsets(sizes)
    sizes = sizes[blocks]
    fresh = sizes == counts  # no stored sample survives
    # Rule 0: nothing is rounded below the minimum size or if the block is exactly regular
    active = np.flatnonzero(~fresh & (sizes >= _time.MIN_ROUND_SAMPLES))
    regular = np.zeros(active.shape[0], np.bool_)
    _time.regular_blocks(all_merged, merged_offsets[blocks[active]], merged_offsets[blocks[active] + 1], regular)
    active = active[~regular]
    # Rule 1, for all blocks at once: a stored regular block's lattice takes the new ticks if they are all near it
    stored_ids = np.minimum(blocks, parsed.header.num_blocks - 1)
    interval = time_rows.steps[stored_ids] * time_rows.refs[stored_ids].astype(np.int64)
    lattice = (
        (blocks < parsed.header.num_blocks) & ((flags[stored_ids] & _format.BLOCK_FLAG_IRREGULAR_TIME) == 0)
        & (parsed.block_sizes[stored_ids] >= 2) & (interval > 0)
    )
    lattice[np.setdiff1d(np.arange(blocks.shape[0]), active)] = False
    which = np.repeat(np.arange(blocks.shape[0]), counts)
    step_t = np.where(lattice, interval, 1)[which]
    snapped = _round_to_grid(ticks, step_t, (time_rows.starts[stored_ids][which] % step_t))
    near = np.abs(snapped - ticks) <= np.floor(0.5 * time_error * step_t).astype(np.int64)
    snapped_blocks = lattice & np.logical_and.reduceat(near, first)
    out[snapped_blocks[which]] = snapped[snapped_blocks[which]]
    for idx in np.setdiff1d(active, np.flatnonzero(snapped_blocks)).tolist():
        block = int(blocks[idx])
        new_slice = slice(first[idx], first[idx] + counts[idx])
        new_t, old_t = ticks[new_slice], old_ticks[old_first[idx]:old_end[idx]]
        merged = all_merged[merged_offsets[block]:merged_offsets[block + 1]]
        irregular = bool(flags[block] & _format.BLOCK_FLAG_IRREGULAR_TIME)
        step = int(time_rows.steps[block])
        interval = step * int(time_rows.refs[block])
        quantum = _merged_quantum(merged, time_error)
        if quantum <= 1:
            continue
        grid, on_grid = _stored_grid(old_t)
        if grid:
            quantum = _nested_quantum(quantum, grid)
            if quantum > 1:
                out[new_slice] = _round_to_grid(new_t, quantum, on_grid % quantum)
            continue
        if (interval if not irregular else step) > quantum:
            quantum = min(quantum, _merged_quantum(merged, 0.5 * time_error))
            if quantum <= 1:
                continue
        phase = _merged_phase(merged, quantum)
        out[new_slice] = _round_to_grid(new_t, quantum, phase)
        low = start + block * duration
        stored = np.clip(_round_to_grid(old_t, quantum, phase), low, low + duration - 1)
        if (stored != old_t).any():
            edits[block] = stored
    fresh_idx = np.flatnonzero(fresh)
    if fresh_idx.shape[0]:
        fresh_counts = counts[fresh_idx]
        positions = np.arange(int(fresh_counts.sum())) + np.repeat(
            first[fresh_idx] - (np.cumsum(fresh_counts) - fresh_counts), fresh_counts
        )
        seed_ticks, seed_steps = _clock_before(
            parsed, time_rows, blocks, fresh, fresh_idx, old_ticks, old_blocks, edits, out, first, counts, ranges,
            covered,
        )
        out[positions], _ = _group.snap_ticks(ticks[positions], fresh_counts, time_error, seed_ticks, seed_steps)
    return np.maximum.accumulate(np.maximum(out, start)), edits


def _clock_before(
    parsed: _group.ParsedGroup,
    time_rows: _format.TimeRows,
    blocks: np.ndarray,
    fresh: np.ndarray,
    fresh_idx: np.ndarray,
    old_ticks: np.ndarray,
    old_blocks: np.ndarray,
    edits: dict[int, np.ndarray],
    new_ticks: np.ndarray,
    new_first: np.ndarray,
    new_counts: np.ndarray,
    ranges: np.ndarray,
    covered: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """The clock each block without stored samples follows: the last tick and regular interval of the
    non-empty block before it, unless that is another such block (they are rounded in sequence).

    A stored block that the update doesn't touch is read from its columns
    (_group.stored_tails); one it does is read from its ticks as rounded.

    Args:
        parsed: The block group.
        time_rows: Its time rows (the columns at least).
        blocks: 1D int64 array of the blocks with new samples.
        fresh: 1D bool array over blocks: no stored sample survives.
        fresh_idx: 1D int64 array of the positions of the fresh blocks.
        old_ticks: 1D int64 array of the decoded stored ticks.
        old_blocks: 1D int64 array of their blocks.
        edits: Rounded stored ticks by block (rule 3).
        new_ticks: 1D int64 array of the new ticks as rounded so far.
        new_first: 1D int64 array: each block's first new tick.
        new_counts: 1D int64 array: each block's number of new ticks.
        ranges: 2D int64 array of the deletion ranges.
        covered: 1D bool array of the stored blocks the deletion ranges cover whole.

    Returns:
        A tuple of (seed_ticks, seed_steps) per fresh block (_group.snap_ticks).
    """
    num_old = parsed.header.num_blocks
    # The blocks that are non-empty after the update: decoded ones by their surviving samples
    nonempty = (parsed.block_sizes > 0) & ~covered
    if old_blocks.shape[0]:
        gone = np.zeros(old_ticks.shape[0], np.bool_)
        if ranges.shape[0]:
            position = np.minimum(np.searchsorted(ranges[:, 1], old_ticks, "right"), ranges.shape[0] - 1)
            gone = (ranges[position, 1] > old_ticks) & (ranges[position, 0] <= old_ticks)
        decoded = np.unique(old_blocks)
        nonempty[decoded] = np.bincount(old_blocks[~gone], minlength=num_old)[decoded] > 0
    nonempty = np.union1d(np.flatnonzero(nonempty), blocks)
    seed_ticks = np.full(fresh_idx.shape[0], _time.INT64_MIN, np.int64)
    seed_steps = np.zeros(fresh_idx.shape[0], np.int64)
    before = np.searchsorted(nonempty, blocks[fresh_idx], "left") - 1
    previous = np.where(before >= 0, nonempty[np.maximum(before, 0)], -1)
    position = np.searchsorted(blocks, previous)
    is_new = (previous >= 0) & (position < blocks.shape[0]) & (blocks[np.minimum(position, blocks.shape[0] - 1)] == previous)
    follows = np.zeros(fresh_idx.shape[0], np.bool_)
    follows[is_new] = fresh[position[is_new]]  # follows another such block: the sequence carries on
    old_first = np.searchsorted(old_blocks, previous, "left")
    old_end = np.searchsorted(old_blocks, previous, "right")
    touched = (previous >= 0) & ((old_end > old_first) | is_new)
    stored = np.flatnonzero((previous >= 0) & ~touched)
    if stored.shape[0]:
        seed_ticks[stored], seed_steps[stored] = _group.stored_tails(parsed, time_rows, previous[stored])
    for row in np.flatnonzero(touched & ~follows).tolist():
        block = int(previous[row])
        old_t = edits.get(block, old_ticks[old_first[row]:old_end[row]])
        idx = int(position[row])
        new_t = new_ticks[new_first[idx]:new_first[idx] + new_counts[idx]] if is_new[row] else old_t[:0]
        merged = _merged_ticks(old_t, new_t, ranges)
        seed_ticks[row] = merged[-1]
        seed_steps[row] = _time.regular_step(merged) if merged.shape[0] >= _time.MIN_ROUND_SAMPLES else 0
    return seed_ticks, seed_steps
