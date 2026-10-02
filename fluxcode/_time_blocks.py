# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Time-divided blocks: block b of a unit holds the samples timed in
[start_time + b * block_duration, start_time + (b + 1) * block_duration).

The start time and duration aren't stored: the caller keeps them (typically the start is part
of the unit's storage key) and passes the same ones to update_time_blocks. Times, the start,
the duration and delete ranges are all converted to int64 ticks in the times' unit.
"""

import numpy as np
import numpy.typing as npt
from numba import njit

from . import _args, _format, _unit
from ._args import DurationLike, RangesLike, TimeLike
from ._types import EncodedUnit, Params, UpdatedUnit


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
    samples = _args.as_series(x, allow_empty=True)
    ticks, unit_code = _args.series_ticks(times, time_unit, samples.shape[0])
    assert ticks is not None
    start = _args.to_ticks(start_time, unit_code, "start_time")
    duration = _args.to_ticks(block_duration, unit_code, "block_duration", duration=True)
    return _unit.encode(samples, chunk(ticks, start, duration), params, ticks, unit_code)


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
    """Merges the new samples into the old ones in time order, for the blocks that change.

    An old sample goes if a range covers it or a new sample has its time; each new sample goes
    before the old samples that follow it in time (those at its own time have just gone), and
    new samples at one time keep their order. Both sides are sorted, so this is one pass of
    two cursors each. Only the samples of blocks that change are written: a block changes if
    it gains a sample or loses one.

    Args:
        old_ticks, old_values, old_blocks: The old samples in time order and their blocks.
        new_ticks, new_values, new_blocks: The new samples in time order and their blocks.
        ranges: 2D int64 array of sorted, disjoint [start, end) delete ranges.
        changed: Output bool array per block, set for the blocks that change (False on entry).
        out_ticks, out_values: Output arrays of at least old + new samples receiving the
            samples of the changed blocks.
        block_sizes: Output int64 array per block (zero on entry) receiving each changed
            block's new size.

    Returns:
        The number of samples written.
    """
    num_old, num_new, num_ranges = old_ticks.shape[0], new_ticks.shape[0], ranges.shape[0]
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
    out = 0
    new_idx = 0
    for old_idx in range(num_old + 1):
        while new_idx < num_new and (old_idx == num_old or new_ticks[new_idx] < old_ticks[old_idx]):
            out_ticks[out] = new_ticks[new_idx]
            out_values[out] = new_values[new_idx]
            block_sizes[new_blocks[new_idx]] += 1
            out += 1
            new_idx += 1
        if old_idx < num_old and keep[old_idx] and changed[old_blocks[old_idx]]:
            out_ticks[out] = old_ticks[old_idx]
            out_values[out] = old_values[old_idx]
            block_sizes[old_blocks[old_idx]] += 1
            out += 1
    return out


def _block_masks(ranges: np.ndarray, start: int, duration: int, num_blocks: int) -> tuple[np.ndarray, np.ndarray]:
    """The existing blocks whose time span meets a range, and those inside one (merged) range.

    Block bounds are Python ints: a span can end past int64, so both ends are clamped to the
    blocks before numpy sees them.

    Returns:
        Two bool arrays per existing block: touched, covered. A covered block loses all its samples.
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
    samples, ticks, start, duration, ranges = _args.time_blocks_update(
        x, times, start_time, block_duration, delete_ranges, time_unit, parsed.header.time_unit
    )
    new_ids = block_ids(ticks, start, duration)
    if new_ids.shape[0] and (np.diff(ticks) < 0).any():
        raise _args.decrease_error(ticks, int(np.flatnonzero(np.diff(ticks) < 0)[0]) + 1)
    num_old_blocks = parsed.header.num_blocks
    touched, covered = _block_masks(ranges, start, duration, num_old_blocks)
    # The existing blocks to decode: those a range meets or a new sample lands in, except the
    # covered ones, which lose everything without being decoded
    touched[new_ids[new_ids < num_old_blocks]] = True
    touched &= ~covered
    decoded = np.flatnonzero(touched)
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
    # Blocks to re-encode: those that gain a sample or lose one (a block that loses nothing and
    # gains nothing, empty ones included, is carried over)
    num_blocks = max(num_old_blocks, int(new_ids[-1]) + 1 if new_ids.shape[0] else 0)
    changed = np.zeros(num_blocks, np.bool_)
    sizes = np.zeros(num_blocks, np.int64)
    num_gathered = gathered_ticks.shape[0]
    merged_ticks = np.empty(num_gathered + ticks.shape[0], np.int64)
    merged_values = np.empty(num_gathered + ticks.shape[0])
    num_merged = _merge_samples(
        gathered_ticks, gathered_values, gathered_blocks, ticks, samples, new_ids, ranges, changed, merged_ticks,
        merged_values, sizes,
    )
    covered_ids = np.flatnonzero(covered)
    changed[covered_ids] |= old_sizes[covered_ids] > 0
    indices = np.flatnonzero(changed)
    if not indices.shape[0]:
        return UpdatedUnit(unit, np.zeros(0, np.int64), np.zeros(0), np.zeros(0), np.zeros(0))
    sizes = sizes[indices]
    if sizes.max() > _format.MAX_BLOCK_LEN:
        raise ValueError(f"a block would hold over {_format.MAX_BLOCK_LEN} samples")
    return _unit.splice(
        parsed,
        indices,
        merged_values[:num_merged],
        sizes,
        merged_ticks[:num_merged],
        params,
        time_rows,
    )
