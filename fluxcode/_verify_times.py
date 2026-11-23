# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Time-axis and time-block verification checks for ``fluxcode.verify``."""

import numpy as np
from numba import njit

from ._verify_common import (
    MAX_REPORTED,
    TIME_SLACK,
    Violation,
    block_of,
    report_blocks,
    report_samples,
)
from ._verify_decode import Decoded


def time_block_ids(ticks: np.ndarray, start: int, duration: int) -> np.ndarray:
    """Computes ``floor((tick - start) / duration)`` in unsigned 64-bit arithmetic."""
    if ticks.shape[0] and int(ticks.min()) < start:
        raise ValueError("times must be at or after start_time")
    return ((ticks.view(np.uint64) - np.uint64(start % 2**64)) // np.uint64(duration)).astype(np.int64)


def in_ranges(ticks: np.ndarray, ranges: np.ndarray) -> np.ndarray:
    """Returns a boolean mask of ticks contained in any sorted, disjoint ``[start, end)`` range."""
    if not ranges.shape[0]:
        return np.zeros(ticks.shape[0], np.bool_)
    idx = np.searchsorted(ranges[:, 0], ticks, side="right") - 1
    return np.asarray((idx >= 0) & (ticks < ranges[np.maximum(idx, 0), 1]), dtype=np.bool_)


def met_blocks(ranges: np.ndarray, start: int, duration: int, num_old: int, num_blocks: int) -> np.ndarray:
    """Returns a boolean mask of existing blocks overlapped by any deletion range."""
    met = np.zeros(num_blocks, np.bool_)
    for range_start, range_end in ranges.tolist():
        if range_end <= start:
            continue
        first = max(0, (range_start - start) // duration)
        last = min(num_old - 1, (range_end - 1 - start) // duration)
        met[first : last + 1] = True
    return met


def neighbour_intervals(old: Decoded, encoded: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """Computes per-block maximum mean interval of adjacent carried-over stored blocks."""
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


def check_times(
    found: list[Violation],
    given: np.ndarray,
    got: np.ndarray,
    sizes: np.ndarray,
    offsets: np.ndarray,
    encoded: np.ndarray,
    time_error: float,
    allowance: np.ndarray | None,
    time_blocks: tuple[int, int] | None,
) -> None:
    """Checks monotonicity, exact-time fidelity, ``time_error`` bounds, and time-block membership."""
    steps = np.diff(got)
    if (steps < 0).any():
        first = int(np.flatnonzero(steps < 0)[0]) + 1
        found.append(Violation("times", block_of(offsets, first), f"time of sample {first} decreases"))
    rounded, wrong, first_wrong, largest, far, mean_interval = _time_stats(given, got, offsets, encoded, time_error)
    report_blocks(found, "times", ~rounded, wrong, first_wrong, "times stored exactly decoded differently")
    if allowance is not None:
        mean_interval = np.maximum(mean_interval, allowance)
    cap = time_error * TIME_SLACK * mean_interval * (1 + 1e-12) + 1e-6
    for block_idx in np.flatnonzero(rounded & (largest > cap))[:MAX_REPORTED].tolist():
        found.append(
            Violation(
                "time_error",
                block_idx,
                f"a time moved {largest[block_idx]:g} ticks, over "
                f"{TIME_SLACK} x {time_error} x the mean interval {mean_interval[block_idx]:g}",
            )
        )
    for block_idx in np.flatnonzero(rounded & (far > sizes // 20))[:MAX_REPORTED].tolist():
        found.append(
            Violation(
                "time_error",
                block_idx,
                f"{far[block_idx]} of {sizes[block_idx]} times moved more "
                f"than {time_error} x their own interval (at most 1 in 20)",
            )
        )
    if time_blocks is not None:
        start, duration = time_blocks
        block = np.repeat(np.arange(sizes.shape[0]), sizes)
        offset = got.view(np.uint64) - np.uint64(start % 2**64)
        ids = (offset // np.uint64(duration)).astype(np.int64)
        at_end = (ids == block + 1) & (offset % np.uint64(duration) == 0)
        report_samples(
            found,
            "time_block",
            (got < start) | ~((ids == block) | at_end),
            offsets,
            "times decoded outside their time block",
        )


@njit(nogil=True, cache=True)
def _time_stats(
    given: np.ndarray, got: np.ndarray, offsets: np.ndarray, encoded: np.ndarray, time_error: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Computes per-block timestamp rounding status, exact mismatches, max displacement, and outlier counts."""
    # A block is eligible for rounding iff it was encoded with time_error > 0, has >= 3 ticks,
    # and was not already uniform. Unsigned subtraction keeps all non-decreasing gaps exact.
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
