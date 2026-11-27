# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Fixed-duration time-partitioned block groups: encoding, point updates, and block
replacement.

Block ``b`` holds samples timed in ``[start_time + b * block_duration, start_time + (b +
1) * block_duration)``. ``start_time`` and ``block_duration`` are supplied by the caller
(not stored in the group header).

When ``Params.time_error > 0``, each block's timestamps are rounded within
``[block_start, block_end]`` (inclusive of ``block_end`` so a tick near the upper edge
can snap onto the next block's start without leaving its block). Because
``update_time_blocks`` operates by timestamp, it also inspects any preceding block that
may hold a tick rounded up to its upper boundary.
"""

from collections.abc import Iterable

import numpy as np
import numpy.typing as npt
from numba import njit

from . import _args, _format, _group, _time
from ._args import DurationLike, RangesLike, TimeLike
from ._types import EncodedGroup, Params, UpdatedGroup


def assign_time_blocks(ticks: np.ndarray, start: int, duration: int) -> np.ndarray:
    """Maps each timestamp tick to its fixed-duration block index ``(tick - start) //
    duration``.

    Args:
        ticks: 1D int64 array of timestamp ticks.
        start: Anchor tick of block 0.
        duration: Positive block duration in ticks.

    Returns:
        1D int64 array of block indices matching ``ticks``.

    Raises:
        ValueError: If ``duration <= 0``, any tick precedes ``start``, or any block
            index reaches ``MAX_BLOCKS``.
    """
    if duration <= 0:
        raise ValueError(f"block_duration must be positive, got {duration} ticks")
    if ticks.shape[0] and ticks.min() < start:
        raise ValueError(
            f"times must be at or after start_time: got {int(ticks.min())} "
            f"before {start}"
        )
    # Subtract in uint64 to avoid signed int64 overflow when start is negative and ticks
    # are large.
    ids = (ticks.astype(np.uint64) - np.uint64(start % 2**64)) // np.uint64(duration)
    if ids.shape[0] and ids.max() >= _format.MAX_BLOCKS:
        raise ValueError(
            f"times reach block {int(ids.max())}, past the last block a block group "
            f"holds ({_format.MAX_BLOCKS - 1})"
        )
    return ids.astype(np.int64)


def time_block_sizes(ticks: np.ndarray, start: int, duration: int) -> np.ndarray:
    """Computes per-block sample counts from block 0 through the last non-empty time
    block.

    Args:
        ticks: 1D int64 array of non-decreasing timestamp ticks.
        start: Anchor tick of block 0.
        duration: Positive block duration in ticks.

    Returns:
        1D int64 array of block sample counts (empty array when ``ticks`` is empty).
    """
    ids = assign_time_blocks(ticks, start, duration)
    return np.bincount(ids, minlength=int(ids[-1]) + 1 if ids.shape[0] else 0).astype(
        np.int64
    )


def block_bounds(
    indices: np.ndarray, start: int, duration: int
) -> tuple[np.ndarray, np.ndarray]:
    """Computes inclusive ``[low, high]`` timestamp rounding bounds for each block in
    ``indices``.

    Args:
        indices: 1D int64 array of block indices.
        start: Anchor tick of block 0.
        duration: Positive block duration in ticks.

    Returns:
        Tuple ``(lows, highs)`` of 1D int64 arrays, with ``highs`` capped at
        ``INT64_MAX``.
    """
    if not indices.shape[0]:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    if start + (int(indices.max()) + 1) * duration <= _time.INT64_MAX:
        lows = start + indices * duration
        return lows, lows + duration
    # Fall back to Python arbitrary-precision ints when the upper bound approaches
    # INT64_MAX.
    lows = np.array([start + int(idx) * duration for idx in indices.tolist()], np.int64)
    highs = np.array(
        [
            min(start + (int(idx) + 1) * duration, _time.INT64_MAX)
            for idx in indices.tolist()
        ],
        np.int64,
    )
    return lows, highs


def encode_time_blocks(
    samples: npt.ArrayLike,
    times: npt.ArrayLike,
    params: Params,
    start_time: TimeLike,
    block_duration: DurationLike,
    time_unit: str | None,
) -> EncodedGroup:
    """Encodes a timed series into fixed-duration blocks within a single block group.

    Args:
        samples: 1D array-like of float64 samples (may be empty).
        times: Non-decreasing timestamps matching ``samples`` in length.
        params: Encoder configuration parameters.
        start_time: Anchor time for block 0.
        block_duration: Positive duration of each block.
        time_unit: Optional time unit string for integer timestamps.

    Returns:
        ``EncodedGroup`` holding the compressed bytes and per-block min, max, and mean.

    Raises:
        ValueError: If timestamps, time arguments, or block/sample counts are invalid.
    """
    series_arr = _args.as_series(samples, allow_empty=True)
    ticks, unit_code = _args.series_ticks(times, time_unit, series_arr.shape[0])
    assert ticks is not None
    start = _args.to_ticks(start_time, unit_code, "start_time")
    duration = _args.to_ticks(
        block_duration, unit_code, "block_duration", duration=True
    )
    sizes = time_block_sizes(ticks, start, duration)
    if params.time_error > 0:
        indices = np.arange(sizes.shape[0], dtype=np.int64)
        ticks, _ = _group.snap_ticks(
            ticks,
            sizes,
            params.time_error,
            bounds=block_bounds(indices, start, duration),
        )
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
    """Merges surviving existing samples and incoming new samples in chronological
    order.

    Existing samples covered by ``ranges`` or sharing a timestamp with any incoming
    sample are discarded. Only blocks that gain or lose samples are written to
    ``out_ticks``/``out_values``.

    Args:
        old_ticks: 1D int64 array of decoded existing sample timestamps.
        old_values: 1D float64 array of decoded existing sample values.
        old_blocks: 1D int64 array mapping each existing sample to its block index.
        new_ticks: 1D int64 array of incoming sample timestamps.
        new_values: 1D float64 array of incoming sample values.
        new_blocks: 1D int64 array mapping each incoming sample to its block index.
        ranges: 2D int64 array of sorted, disjoint ``[start, end)`` deletion ranges.
        changed: Output 1D bool array (initially False) set True for blocks that
            changed.
        out_ticks: Output 1D int64 array receiving merged timestamps of changed blocks.
        out_values: Output 1D float64 array receiving merged values of changed blocks.
        block_sizes: Output 1D int64 array (initially 0) accumulating sample counts of
            changed blocks.

    Returns:
        Total number of samples written into ``out_ticks`` and ``out_values``.
    """
    # Pass 1: sweep old_ticks against sorted new_ticks and deletion ranges to mark
    # survivors and changed blocks. Pass 2: merge-join surviving samples of changed
    # blocks with new samples.
    num_old, num_new, num_ranges = (
        old_ticks.shape[0],
        new_ticks.shape[0],
        ranges.shape[0],
    )
    keep = np.empty(num_old, np.bool_)
    range_idx = 0
    new_idx = 0
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
    for new_idx in range(num_new):
        changed[new_blocks[new_idx]] = True
    write_idx = 0
    new_idx = 0
    for old_idx in range(num_old + 1):
        while new_idx < num_new and (
            old_idx == num_old or new_ticks[new_idx] < old_ticks[old_idx]
        ):
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


def _block_masks(
    ranges: np.ndarray, start: int, duration: int, num_blocks: int
) -> tuple[np.ndarray, np.ndarray]:
    """Computes boolean masks of existing blocks touched (overlapped) and fully covered
    by ``ranges``.

    Args:
        ranges: 2D int64 array of sorted, disjoint ``[start, end)`` deletion ranges.
        start: Anchor tick of block 0.
        duration: Positive block duration in ticks.
        num_blocks: Number of existing blocks.

    Returns:
        Tuple ``(touched, covered)`` of 1D bool arrays of length ``num_blocks``.
    """
    touched = np.zeros(num_blocks, np.bool_)
    covered = np.zeros(num_blocks, np.bool_)
    for range_start, range_end in ranges.tolist():
        if range_end <= start:
            continue
        touched[
            min(max(range_start - start, 0) // duration, num_blocks) : min(
                -(-(range_end - start) // duration), num_blocks
            )
        ] = True
        covered[
            min(max(-(-(range_start - start) // duration), 0), num_blocks) : min(
                (range_end - start) // duration, num_blocks
            )
        ] = True
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
    """Applies timestamp-keyed upserts and range deletions to a time-partitioned block
    group.

    Args:
        group: Existing compressed block group bytes (must have a time axis).
        samples: 1D array-like of new samples (may be empty).
        times: Non-decreasing timestamps matching ``samples``.
        params: Encoder configuration parameters (``time_error`` must be 0 when adding
            samples).
        start_time: Anchor time for block 0.
        block_duration: Positive duration of each block.
        delete_ranges: Optional ``[start, end)`` time ranges to delete before inserting
            samples.
        time_unit: Optional time unit string checked against the group header.

    Returns:
        ``UpdatedGroup`` with the updated bytes, modified block indices, and block
        statistics.

    Raises:
        ValueError: If the group has no time axis, ``time_error > 0`` with new samples,
            inputs are invalid, or any block exceeds ``MAX_BLOCK_LEN``.
    """
    # Identify blocks overlapped by deletions, boundary-rounded ticks, or incoming
    # samples; decode only those partial blocks (skipping fully covered blocks), merge
    # in chronological order, and splice changed blocks back while carrying untouched
    # blocks verbatim.
    parsed = _group.decompress(group)
    if not parsed.has_time:
        raise ValueError("the block group has no time axis: use update")
    series_samples, ticks, start, duration, ranges = _args.time_blocks_update(
        samples,
        times,
        start_time,
        block_duration,
        delete_ranges,
        time_unit,
        parsed.header.time_unit,
    )
    if params.time_error > 0 and ticks.shape[0]:
        raise ValueError(
            "update_time_blocks stores new times exactly, so it takes no time_error: "
            "replace_time_blocks re-encodes whole blocks with one"
        )
    new_ids = assign_time_blocks(ticks, start, duration)
    steps = np.diff(ticks)
    if new_ids.shape[0] and (steps < 0).any():
        raise _args.decrease_error(ticks, int(np.flatnonzero(steps < 0)[0]) + 1)
    num_old_blocks = parsed.header.num_blocks
    touched, covered = _block_masks(ranges, start, duration, num_old_blocks)
    time_columns = _group.read_time_columns(parsed, np.zeros(0, np.int64))
    # Blocks holding a tick rounded up to their upper boundary must also be decoded when
    # touched.
    ends = _blocks_with_end_tick(
        parsed, time_columns, ticks, new_ids, ranges, start, duration, covered
    )
    touched[ends] = True
    covered[ends] = False
    touched &= ~covered
    decoded = np.flatnonzero(
        touched | _new_blocks_mask(new_ids, num_old_blocks, covered)
    )
    _group.read_time_residuals(parsed, time_columns, decoded)
    gathered_ticks, gathered_values, gathered_blocks = _gather(
        parsed, decoded, time_columns
    )
    num_blocks = max(num_old_blocks, int(new_ids[-1]) + 1 if new_ids.shape[0] else 0)
    changed = np.zeros(num_blocks, np.bool_)
    sizes = np.zeros(num_blocks, np.int64)
    num_gathered = gathered_ticks.shape[0]
    merged_ticks = np.empty(num_gathered + ticks.shape[0], np.int64)
    merged_values = np.empty(num_gathered + ticks.shape[0])
    num_merged = _merge_samples(
        gathered_ticks,
        gathered_values,
        gathered_blocks,
        ticks,
        series_samples,
        new_ids,
        ranges,
        changed,
        merged_ticks,
        merged_values,
        sizes,
    )
    # Mark previously non-empty blocks that were completely wiped by deletion ranges.
    covered_ids = np.flatnonzero(covered)
    changed[covered_ids] |= parsed.block_sizes[covered_ids] > 0
    indices = np.flatnonzero(changed)
    if not indices.shape[0]:
        return _group.empty_update(group, parsed)
    sizes = sizes[indices]
    if sizes.max() > _format.MAX_BLOCK_LEN:
        raise ValueError(f"a block would hold over {_format.MAX_BLOCK_LEN} samples")
    return _group.splice(
        parsed,
        indices,
        merged_values[:num_merged],
        sizes,
        merged_ticks[:num_merged],
        params,
        time_columns,
        snap=False,
    )


def replace_time_blocks(
    group: bytes,
    blocks: Iterable[int],
    samples: npt.ArrayLike,
    times: npt.ArrayLike,
    params: Params,
    start_time: TimeLike,
    block_duration: DurationLike,
    time_unit: str | None,
) -> UpdatedGroup:
    """Replaces whole declared blocks of a time-partitioned block group from source
    samples.

    Args:
        group: Existing compressed block group bytes (must have a time axis).
        blocks: Iterable of distinct block indices to replace.
        samples: 1D array-like of replacement samples (may be empty).
        times: Non-decreasing timestamps matching ``samples``; every tick must fall
            inside ``blocks``.
        params: Encoder configuration parameters (may include ``time_error > 0``).
        start_time: Anchor time for block 0.
        block_duration: Positive duration of each block.
        time_unit: Optional time unit string checked against the group header.

    Returns:
        ``UpdatedGroup`` with the updated bytes, re-encoded block indices, and block
        statistics.

    Raises:
        ValueError: If the group has no time axis, any timestamp falls outside
            ``blocks`` or decreases, any block exceeds ``MAX_BLOCK_LEN``, or arguments
            are invalid.
    """
    # Validate that all incoming samples map into declared blocks, compute per-block
    # sizes, optionally round ticks within each block's [start, end] seeded by preceding
    # stored blocks, and splice the replacement blocks into the group.
    parsed = _group.decompress(group)
    if not parsed.has_time:
        raise ValueError("the block group has no time axis: use update")
    indices = _args.block_list(blocks)
    series_samples, ticks, start, duration, _ = _args.time_blocks_update(
        samples,
        times,
        start_time,
        block_duration,
        None,
        time_unit,
        parsed.header.time_unit,
    )
    ids = assign_time_blocks(ticks, start, duration)
    if ticks.shape[0] > 1 and (np.diff(ticks) < 0).any():
        raise _args.decrease_error(
            ticks, int(np.flatnonzero(np.diff(ticks) < 0)[0]) + 1
        )
    positions = np.searchsorted(indices, ids)
    declared = positions < indices.shape[0]
    declared[declared] = indices[positions[declared]] == ids[declared]
    if not declared.all():
        sample_idx = int(np.flatnonzero(~declared)[0])
        raise ValueError(
            f"sample {sample_idx} is timed in block {int(ids[sample_idx])}, "
            "which blocks doesn't declare"
        )
    if not indices.shape[0]:
        return _group.empty_update(group, parsed)
    sizes = np.bincount(positions, minlength=indices.shape[0]).astype(np.int64)
    if sizes.max() > _format.MAX_BLOCK_LEN:
        raise ValueError(f"a block would hold over {_format.MAX_BLOCK_LEN} samples")
    time_columns = _group.read_time_columns(parsed, np.zeros(0, np.int64))
    if params.time_error > 0 and ticks.shape[0]:
        num_blocks = max(parsed.header.num_blocks, int(indices[-1]) + 1)
        spliced_sizes = np.zeros(num_blocks, np.int64)
        spliced_sizes[: parsed.header.num_blocks] = parsed.block_sizes
        spliced_sizes[indices] = sizes
        _, seed_ticks, seed_steps = _group.stored_seeds(
            parsed, time_columns, spliced_sizes, indices
        )
        ticks, _ = _group.snap_ticks(
            ticks,
            sizes,
            params.time_error,
            seed_ticks,
            seed_steps,
            block_bounds(indices, start, duration),
        )
    return _group.splice(
        parsed, indices, series_samples, sizes, ticks, params, time_columns, snap=False
    )


def _blocks_with_end_tick(
    parsed: _group.ParsedGroup,
    time_columns: _format.EncodedTimeColumns,
    ticks: np.ndarray,
    new_ids: np.ndarray,
    ranges: np.ndarray,
    start: int,
    duration: int,
    covered: np.ndarray,
) -> np.ndarray:
    """Finds stored blocks whose last tick sits at their exact upper boundary and is
    affected by the update.

    Args:
        parsed: Existing parsed block group.
        time_columns: Its ``EncodedTimeColumns`` (metadata columns loaded; candidate
            irregular blocks' residuals are unpacked on demand).
        ticks: 1D int64 array of incoming sample timestamps.
        new_ids: 1D int64 array of block indices for ``ticks``.
        ranges: 2D int64 array of sorted, disjoint ``[start, end)`` deletion ranges.
        start: Anchor tick of block 0.
        duration: Positive block duration in ticks.
        covered: 1D bool mask of stored blocks covered by ``ranges`` on ``[block_start,
            block_end)``.

    Returns:
        Strictly increasing 1D int64 array of stored block indices holding a tick at
        their upper boundary that is either deleted/replaced (if not covered) or
        survives (if covered).
    """
    # Only blocks whose upper boundary coincides with an incoming sample timestamp or a
    # deletion range edge can be affected; inspect only those candidates' stored last
    # ticks.
    num_blocks = parsed.header.num_blocks
    lows, _ = block_bounds(new_ids, start, duration)
    candidates = [new_ids[ticks == lows] - 1]
    edge_ids = [
        (edge - start) // duration - 1
        for edge in ranges.ravel().tolist()
        if edge > start and (edge - start) % duration == 0
    ]
    candidates.append(np.array([idx for idx in edge_ids if idx < num_blocks], np.int64))
    blocks: np.ndarray = np.unique(np.concatenate(candidates))
    blocks = blocks[(blocks >= 0) & (blocks < num_blocks)]
    blocks = blocks[parsed.block_sizes[blocks] > 0]
    if not blocks.shape[0]:
        return blocks
    _, ends = block_bounds(blocks, start, duration)
    deleted = _in_ranges(ends, ranges)
    reached = np.where(covered[blocks], ~deleted, deleted | np.isin(ends, ticks))
    blocks, ends = blocks[reached], ends[reached]
    last_ticks, _ = _group.stored_tails(parsed, time_columns, blocks)
    return np.asarray(blocks[last_ticks == ends])


def _in_ranges(ticks: np.ndarray, ranges: np.ndarray) -> np.ndarray:
    """Returns a boolean mask of which ticks fall inside any sorted, disjoint ``[start,
    end)`` range.
    """
    if not ranges.shape[0]:
        return np.zeros(ticks.shape[0], np.bool_)
    idx = np.searchsorted(ranges[:, 0], ticks, side="right") - 1
    return np.asarray((idx >= 0) & (ticks < ranges[np.maximum(idx, 0), 1]))


def _new_blocks_mask(
    new_ids: np.ndarray, num_old_blocks: int, covered: np.ndarray
) -> np.ndarray:
    """Returns a boolean mask of existing blocks receiving new samples, excluding fully
    covered ones.
    """
    mask = np.zeros(num_old_blocks, np.bool_)
    mask[new_ids[new_ids < num_old_blocks]] = True
    return np.asarray(mask & ~covered)


def _gather(
    parsed: _group.ParsedGroup,
    decoded: np.ndarray,
    time_columns: _format.EncodedTimeColumns,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decodes the specified stored blocks into flat chronological ``(ticks, values,
    block_ids)`` arrays.
    """
    if not decoded.shape[0]:
        return np.zeros(0, np.int64), np.zeros(0), np.zeros(0, np.int64)
    all_values, all_ticks = _group.decode_blocks(parsed, decoded, time_columns)
    assert all_ticks is not None
    run_sizes = parsed.block_sizes[decoded]
    run_starts = parsed.layout.sample_offsets[decoded]
    positions = np.arange(int(run_sizes.sum())) + np.repeat(
        run_starts - (np.cumsum(run_sizes) - run_sizes), run_sizes
    )
    return all_ticks[positions], all_values[positions], np.repeat(decoded, run_sizes)
