# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Time axis kernels: per-block analysis of regularity, step and reference, and reconstruction.

Timestamps are int64 ticks in the block group's time unit (seconds to nanoseconds), naive
(no time zone). Within a block they must be non-decreasing. Every block stores a start,
a step (the GCD of its time deltas) and a reference quotient; sample i > 0 has the
quotient (time[i] - time[i-1]) / step, stored as the zigzagged residual
quotient - reference (mod 2^64; 0 for sample 0). A block whose deltas are all equal is
regular: every quotient equals the reference (1, or 0 when all times are equal; a single
delta beyond the int64 maximum is the reference over a step of 1), so it stores no
residuals. Any other block is irregular and stores them.

With a time error e > 0 (Params.time_error) the encoder first rounds each block's ticks
(`round_blocks`) to the coarsest 1-2-5 step of at most 2 e times the block's mean interval for
which at least 95% of its ticks move by at most e times their own interval: a jittered clock
lands back on its own grid and stores as a regular block. A clock whose period no such step
divides (1024 Hz, 60 Hz) goes onto its own lattice instead, on the same condition. The format
doesn't change.
"""

from collections.abc import Callable
from typing import cast

import numpy as np
from numba import njit
from numba.cpython.unsafe.numbers import trailing_zeros as _trailing_zeros_intrinsic

from ._format import BLOCK_FLAG_IRREGULAR_TIME, BLOCK_FLAG_LONG_TIME

_trailing_zeros = cast("Callable[[int | np.integer], int]", _trailing_zeros_intrinsic)
"""Numba intrinsic counting trailing zero bits, typed as its jitted call signature."""

INT64_MAX: int = 0x7FFFFFFFFFFFFFFF
"""Largest int64 tick."""

INT64_MIN: int = -0x8000000000000000
"""Smallest int64, which datetime64 reads as NaT: never a valid tick."""


CADENCE_SAMPLES: int = 15
"""Evenly spaced intervals whose median is a block's scan interval. Where nearly all intervals
are one scan, so is the median of any 15 of them."""

CADENCE_SHARE: float = 0.9
"""Share of a block's intervals that must be one scan for a cadence: a noise floor on irregular
times, and the mean one-scan interval as the period of a lattice of the block's own."""

TIME_ERROR_SLACK: float = 1.02
"""A tick may move up to 2% more than e times the block's mean interval: that mean lands a
little either side of a round period (999.98 us), and a nice time error times a nice period is a
1-2-5 step (2 x 0.1 x 1 ms), so without slack the step would flip between blocks."""


# Outcomes of time axis analysis (encoding) or expansion (decoding), as plain ints for numba
OK: int = 0
"""The times are valid."""

DECREASING: int = 1
"""Encoding: a time is below the previous time in its block."""

BAD_STEP: int = 2
"""Decoding: a time step is negative, or zero on an irregular block."""

BAD_FIRST_RESIDUAL: int = 3
"""Decoding: an irregular block's first time residual is not 0."""

OVERFLOW: int = 4
"""Decoding: a time exceeds int64 maximum."""


@njit(inline="always")
def _gcd(first: np.uint64, second: np.uint64) -> np.uint64:
    """Computes the greatest common divisor of two uint64 values.

    Args:
        first: First uint64 operand (gcd(0, second) = second).
        second: Second uint64 operand.

    Returns:
        Greatest common divisor of first and second.
    """
    while second != 0:
        first, second = second, first % second
    return first


@njit(inline="always")
def _odd_inverse(odd: np.uint64) -> np.uint64:
    """Computes the multiplicative inverse of an odd number mod 2^64 by Newton's iteration.

    Args:
        odd: An odd uint64 integer.

    Returns:
        The uint64 multiplicative inverse satisfying (odd * inverse) mod 2^64 == 1.
    """
    # odd * odd = 1 mod 8, so odd is its own inverse to 3 bits; each step doubles the correct bits
    inverse = odd
    for _ in range(5):
        inverse *= np.uint64(2) - odd * inverse
    return inverse


@njit(inline="always")
def _deltas_gcd(times: np.ndarray, num_deltas: int) -> np.uint64:
    """Computes the GCD of the first num_deltas consecutive time deltas.

    Args:
        times: 1D int64 array of sample timestamps.
        num_deltas: Number of consecutive deltas to evaluate.

    Returns:
        GCD of the examined deltas, or 0 if all examined deltas are 0.
    """
    step_gcd = np.uint64(0)
    for sample_idx in range(1, num_deltas + 1):
        delta = np.uint64(times[sample_idx]) - np.uint64(times[sample_idx - 1])
        # Skip Euclidean division if delta is already a multiple of current GCD
        if step_gcd != 0 and delta % step_gcd == 0:
            continue
        step_gcd = _gcd(step_gcd, delta)
        if step_gcd == 1:
            break
    return step_gcd


@njit(inline="always")
def _divide_deltas(
    times: np.ndarray, step: np.uint64, out_quotients: np.ndarray
) -> tuple[bool, np.uint64, np.uint64]:
    """Divides every time delta by step (> 0), if step divides them all.

    The minimum and total are meaningful only if the division is exact.

    Args:
        times: 1D int64 array of sample timestamps.
        step: Non-zero positive step divisor candidate.
        out_quotients: Output 1D uint64 array receiving quotient values starting at index 1.

    Returns:
        A tuple of (divisible, minimum, total):
            divisible: True if step divides every delta exactly.
            minimum: Minimum quotient observed across the block.
            total: Sum of all quotients across the block.
    """
    minimum = np.uint64(0xFFFFFFFFFFFFFFFF)
    total = np.uint64(0)
    if step == 1:
        # Common nanosecond case: step is 1, deltas are directly the quotients
        for sample_idx in range(1, times.shape[0]):
            quotient = np.uint64(times[sample_idx]) - np.uint64(times[sample_idx - 1])
            out_quotients[sample_idx] = quotient
            minimum = min(minimum, quotient)
            total += quotient
        return True, minimum, total
    # Fast exact division by odd inverse and bit shift
    shift = np.uint64(_trailing_zeros(step))
    odd = step >> shift
    inverse = _odd_inverse(odd)
    multiple_limit = np.uint64(0xFFFFFFFFFFFFFFFF) // odd
    low_mask = (np.uint64(1) << shift) - np.uint64(1)
    low_bits = np.uint64(0)
    largest = np.uint64(0)
    for sample_idx in range(1, times.shape[0]):
        delta = np.uint64(times[sample_idx]) - np.uint64(times[sample_idx - 1])
        quotient = (delta >> shift) * inverse
        low_bits |= delta & low_mask
        largest = max(largest, quotient)
        out_quotients[sample_idx] = quotient
        minimum = min(minimum, quotient)
        # Accumulate quotient total to compute block average
        total += quotient
    return bool(low_bits == 0 and largest <= multiple_limit), minimum, total


@njit(inline="always")
def _reference_quotient(quotients: np.ndarray, minimum: np.uint64, total: np.uint64) -> np.uint64:
    """Computes the reference quotient to center or anchor block residuals against.

    Args:
        quotients: 1D uint64 array of quotient values (with valid data at index 1 and above).
        minimum: Minimum quotient in the block.
        total: Sum of all quotients in the block.

    Returns:
        The selected uint64 reference quotient (either rounded mean or block minimum).
    """
    num_quotients = quotients.shape[0] - 1
    first = np.float64(quotients[1])
    # Float sums in four fixed-order lanes: float addition doesn't reassociate, so one running
    # sum is a serial chain, while fastmath would make the result (and so the block group bytes)
    # depend on the machine's vector width
    sum0 = sum1 = sum2 = sum3 = 0.0
    squares0 = squares1 = squares2 = squares3 = 0.0
    sample_idx = 1
    while sample_idx + 4 <= quotients.shape[0]:
        shifted0 = np.float64(quotients[sample_idx]) - first
        shifted1 = np.float64(quotients[sample_idx + 1]) - first
        shifted2 = np.float64(quotients[sample_idx + 2]) - first
        shifted3 = np.float64(quotients[sample_idx + 3]) - first
        sum0 += shifted0
        sum1 += shifted1
        sum2 += shifted2
        sum3 += shifted3
        squares0 += shifted0 * shifted0
        squares1 += shifted1 * shifted1
        squares2 += shifted2 * shifted2
        squares3 += shifted3 * shifted3
        sample_idx += 4
    while sample_idx < quotients.shape[0]:
        shifted0 = np.float64(quotients[sample_idx]) - first
        sum0 += shifted0
        squares0 += shifted0 * shifted0
        sample_idx += 1
    shifted_sum = (sum0 + sum1) + (sum2 + sum3)
    shifted_squares = (squares0 + squares1) + (squares2 + squares3)
    shifted_mean = shifted_sum / num_quotients
    variance = shifted_squares / num_quotients - shifted_mean * shifted_mean
    mean_above_minimum = shifted_mean - (np.float64(minimum) - first)
    if 3.0 * variance < mean_above_minimum * mean_above_minimum:
        # Rounded mean, without the overflow of total + num_quotients // 2
        count = np.uint64(num_quotients)
        return total // count + np.uint64(2 * (total % count) >= count)
    return minimum


@njit(nogil=True, cache=True)
def encode_times(
    ticks: np.ndarray,
    sample_offsets: np.ndarray,
    out_block_flags: np.ndarray,
    out_time_starts: np.ndarray,
    out_time_steps: np.ndarray,
    out_time_refs: np.ndarray,
    out_time_residuals: np.ndarray,
) -> tuple[int, int]:
    """Analyzes each block's times into a start, a step, a reference and (if irregular) residuals.

    Sets BLOCK_FLAG_IRREGULAR_TIME in out_block_flags for irregular blocks, and BLOCK_FLAG_LONG_TIME for
    those with a residual of 2^32 or more; other bits are kept. An empty block gets 0 for its
    start, step and reference; a block of one sample a step and reference of 0.
    Checks that the times never decrease, within blocks and from each non-empty block to the
    next.

    Args:
        ticks: 1D int64 array of every sample's tick.
        sample_offsets: 1D int64 array of the blocks' sample offsets (num_blocks + 1).
        out_block_flags: In-out 1D uint8 array of block flags.
        out_time_starts: Output 1D int64 array receiving block start times.
        out_time_steps: Output 1D int64 array receiving block time steps.
        out_time_refs: Output 1D uint64 array receiving block reference quotients.
        out_time_residuals: Output 1D uint64 array of every sample receiving the zigzagged
            residuals of irregular blocks (regular blocks' samples are not written).

    Returns:
        A tuple of (status, sample_idx): OK, or DECREASING and the flat index of the first
        time below its predecessor.
    """
    num_blocks = sample_offsets.shape[0] - 1
    int64_max = np.uint64(INT64_MAX)
    previous_last = -1
    for block_idx in range(num_blocks):
        first_sample = sample_offsets[block_idx]
        block_len = sample_offsets[block_idx + 1] - first_sample
        out_block_flags[block_idx] &= ~(BLOCK_FLAG_IRREGULAR_TIME | BLOCK_FLAG_LONG_TIME)
        if block_len:
            # Across blocks: each non-empty block starts at or after the previous one's last time
            if previous_last >= 0 and ticks[first_sample] < ticks[previous_last]:
                return DECREASING, first_sample
            previous_last = first_sample + block_len - 1
        if block_len < 2:
            # No deltas: a step and reference of 0 (and nothing at all for an empty block)
            out_time_starts[block_idx] = ticks[first_sample] if block_len else 0
            out_time_steps[block_idx] = 0
            out_time_refs[block_idx] = 0
            continue
        times = ticks[first_sample:first_sample + block_len]
        out_time_starts[block_idx] = times[0]
        # Read-only regularity and order check, branch-free so that it vectorizes: a decrease
        # is located afterwards. Deltas as uint64: a non-decreasing pair's difference fits.
        first_delta = np.uint64(times[1]) - np.uint64(times[0])
        differing_bits = np.uint64(0)
        decreased = False
        for sample_idx in range(1, block_len):
            differing_bits |= (np.uint64(times[sample_idx]) - np.uint64(times[sample_idx - 1])) ^ first_delta
            decreased |= times[sample_idx] < times[sample_idx - 1]
        if decreased:
            for sample_idx in range(1, block_len):
                if times[sample_idx] < times[sample_idx - 1]:
                    return DECREASING, first_sample + sample_idx
        if differing_bits == 0:
            # Every quotient is 1 (0 when all times are equal). Two or more equal deltas span
            # less than 2^64, so the step fits int64; a single delta beyond int64 maximum is
            # stored as the reference over a step of 1.
            if first_delta > int64_max:
                out_time_steps[block_idx] = 1
                out_time_refs[block_idx] = first_delta
            else:
                out_time_steps[block_idx] = np.int64(first_delta)
                out_time_refs[block_idx] = np.uint64(first_delta != 0)
            continue
        quotients = out_time_residuals[first_sample:first_sample + block_len]
        quotients[0] = 0
        # The GCD of the first few deltas is almost always the block's: try it, dividing every
        # delta in the same pass, and only compute the exact GCD if it doesn't divide them all
        step_gcd = _deltas_gcd(times, min(8, block_len - 1))
        divisible = False
        if 0 < step_gcd <= int64_max:
            divisible, minimum, total = _divide_deltas(times, step_gcd, quotients)
        if not divisible:
            step_gcd = _deltas_gcd(times, block_len - 1)
            if step_gcd > int64_max:
                # Only when one delta spans more than half the int64 range: store raw deltas
                step_gcd = np.uint64(1)
            _, minimum, total = _divide_deltas(times, step_gcd, quotients)
        reference = _reference_quotient(quotients, minimum, total)
        # Zigzagged residuals, in place: wrapping mod 2^64 keeps them exact for any quotient
        all_bits = np.uint64(0)
        for sample_idx in range(1, block_len):
            residual = np.int64(quotients[sample_idx] - reference)
            quotients[sample_idx] = np.uint64((residual << 1) ^ (residual >> 63))
            all_bits |= quotients[sample_idx]
        out_time_steps[block_idx] = np.int64(step_gcd)
        out_time_refs[block_idx] = reference
        out_block_flags[block_idx] |= BLOCK_FLAG_IRREGULAR_TIME
        # Long: a residual needs more than the 32 planes every irregular block stores
        if all_bits >> np.uint64(32):
            out_block_flags[block_idx] |= BLOCK_FLAG_LONG_TIME
    return OK, 0


@njit(nogil=True, cache=True)
def expand_times(
    block_flags: np.ndarray,
    sample_offsets: np.ndarray,
    time_starts: np.ndarray,
    time_steps: np.ndarray,
    time_refs: np.ndarray,
    time_residuals: np.ndarray,
    block_ids: np.ndarray,
    out_times: np.ndarray,
) -> tuple[int, int]:
    """Reconstructs the given blocks' ticks from their start, step, reference and residuals.

    Args:
        block_flags: 1D uint8 array of block flags (BLOCK_FLAG_IRREGULAR_TIME selects the residuals).
        sample_offsets: 1D int64 array of the blocks' sample offsets (num_blocks + 1).
        time_starts: 1D int64 array of block start times.
        time_steps: 1D int64 array of block time steps.
        time_refs: 1D uint64 array of block reference quotients.
        time_residuals: 1D uint64 array of every sample's zigzagged residual (only irregular
            blocks' are read).
        block_ids: 1D int64 array of the blocks to expand.
        out_times: Output 1D int64 array of every sample: each expanded block's ticks are
            written at its sample offsets.

    Returns:
        A tuple of (status, block_idx): OK, or BAD_STEP, BAD_FIRST_RESIDUAL or OVERFLOW
        and the first failing block. One-sample blocks aren't checked here: read_time_rows
        checks them.
    """
    int64_max = np.uint64(INT64_MAX)
    for block_idx in block_ids:
        first_sample = sample_offsets[block_idx]
        block_len = sample_offsets[block_idx + 1] - first_sample
        if block_len == 0:
            continue
        start = np.uint64(time_starts[block_idx])
        step = time_steps[block_idx]
        reference = time_refs[block_idx]
        irregular = block_flags[block_idx] & BLOCK_FLAG_IRREGULAR_TIME
        if step < 0 or (irregular and step == 0):
            return BAD_STEP, block_idx
        times = out_times[first_sample:first_sample + block_len]
        if step == 0:
            times[:] = time_starts[block_idx]
            continue
        step_unsigned = np.uint64(step)
        # time[i] = start + step * (sum of quotients up to i) stays at most int64 maximum iff
        # that sum stays at most (int64 maximum - start) // step: one division per block
        quotient_limit = (int64_max - start) // step_unsigned
        times[0] = time_starts[block_idx]
        if not irregular:
            # Every quotient is the reference
            if reference > quotient_limit // np.uint64(block_len - 1):
                return OVERFLOW, block_idx
            increment = reference * step_unsigned
            for sample_idx in range(1, block_len):
                times[sample_idx] = np.int64(start + np.uint64(sample_idx) * increment)
            continue
        residuals = time_residuals[first_sample:first_sample + block_len]
        if residuals[0] != 0:
            return BAD_FIRST_RESIDUAL, block_idx
        quotient_sum = np.uint64(0)
        for sample_idx in range(1, block_len):
            zigzagged = residuals[sample_idx]
            quotient = reference + ((zigzagged >> np.uint64(1)) ^ (np.uint64(0) - (zigzagged & np.uint64(1))))
            if quotient > quotient_limit - quotient_sum:
                return OVERFLOW, block_idx
            quotient_sum += quotient
            times[sample_idx] = np.int64(start + quotient_sum * step_unsigned)
    return OK, 0


@njit(inline="always")
def median_interval(ticks: np.ndarray, scratch_pivots: np.ndarray) -> int:
    """The median of CADENCE_SAMPLES evenly spaced intervals of a block's ticks.

    Args:
        ticks: 1D int64 array of the block's ticks (at least 2). A decrease gives a negative
            interval.
        scratch_pivots: 1D int64 scratch array of at least CADENCE_SAMPLES.

    Returns:
        The median interval.
    """
    num_intervals = ticks.shape[0] - 1
    # Insertion sort of the evenly spaced intervals (a library sort would allocate)
    pivots = scratch_pivots[:CADENCE_SAMPLES]
    for pivot_idx in range(CADENCE_SAMPLES):
        interval_idx = pivot_idx * (num_intervals - 1) // (CADENCE_SAMPLES - 1)
        pivot = ticks[interval_idx + 1] - ticks[interval_idx]
        insert_idx = pivot_idx
        while insert_idx > 0 and pivots[insert_idx - 1] > pivot:
            pivots[insert_idx] = pivots[insert_idx - 1]
            insert_idx -= 1
        pivots[insert_idx] = pivot
    return pivots[CADENCE_SAMPLES // 2]


@njit(inline="always")
def one_scan(ticks: np.ndarray, scratch_pivots: np.ndarray) -> tuple[int, int, int, int]:
    """A block's scan interval and its number of one-scan intervals.

    The scan interval c is the median of CADENCE_SAMPLES evenly spaced intervals
    (median_interval); an interval strictly between c/2 and 3c/2 is one scan. That separates
    one scan from two (a skipped scan) and covers jittered times. A decrease in the times gives
    a negative interval, never one scan.

    Args:
        ticks: 1D int64 array of the block's ticks (at least 2).
        scratch_pivots: 1D int64 scratch array of at least CADENCE_SAMPLES.

    Returns:
        A tuple of (scan, lower, upper, num_one_scan): an interval d is one scan if
        lower < d < upper. All 0 if the scan isn't in (0, 2^62).
    """
    num_intervals = ticks.shape[0] - 1
    scan = median_interval(ticks, scratch_pivots)
    # Strictly between scan/2 and 3 scan/2 in integers: no rounding, and no overflow below 2^62
    if not 0 < scan < (1 << 62):
        return 0, 0, 0, 0
    lower, upper = scan // 2, scan + (scan + 1) // 2
    # Offset slices from index 0 allow SIMD vectorization without negative index checks
    ticks_next, ticks_prev = ticks[1:], ticks[:num_intervals]
    num_one_scan = 0
    for interval_idx in range(num_intervals):
        interval = ticks_next[interval_idx] - ticks_prev[interval_idx]
        num_one_scan += (interval > lower) & (interval < upper)
    return scan, lower, upper, num_one_scan


@njit(inline="always")
def time_quantum(interval: float, time_error: float) -> int:
    """The largest 1-2-5 x 10^k step of at most 2 time_error interval TIME_ERROR_SLACK ticks.

    Rounding to it moves a tick by at most half of it: time_error times the interval (2% more
    at most). 1-2-5 steps divide the round periods of clocks (any step up to a fifth of a
    1-2-5 period divides it), so a jittered clock on a round phase rounds back onto its own
    grid.

    Args:
        interval: The block's interval in ticks.
        time_error: The time error e (> 0).

    Returns:
        The step, or 1 if no step above 1 fits.
    """
    limit = 2.0 * time_error * TIME_ERROR_SLACK * interval
    best = 1
    decade = 1
    while decade <= limit:
        for mantissa in (1, 2, 5):
            step = mantissa * decade
            if step <= limit:
                best = step
        if decade > INT64_MAX // 50:
            break
        decade *= 10
    return best


@njit(inline="always")
def clock_interval(ticks: np.ndarray, scratch_pivots: np.ndarray, tolerance: float) -> float:
    """A block's clock period, for a lattice of its own: the mean of its intervals within tolerance
    of its scan interval (one_scan), if at least CADENCE_SHARE of its intervals are one scan. Gaps
    don't stretch it, and a late or early tick doesn't pull it (on a clean clock with one late
    tick, the mean of all one-scan intervals would put the whole block on a slightly wrong
    period); trimming a jittered clock's intervals on both sides leaves their mean where it was.

    Args:
        ticks: 1D int64 array of the block's ticks (at least 2).
        scratch_pivots: 1D int64 scratch array of at least CADENCE_SAMPLES.
        tolerance: Largest distance from the scan interval of an interval counted.

    Returns:
        The period, or 0.0 if the block isn't a clock.
    """
    scan, _, _, num_one_scan = one_scan(ticks, scratch_pivots)
    if scan == 0 or num_one_scan < CADENCE_SHARE * (ticks.shape[0] - 1):
        return 0.0
    # Offsets from the scan interval, at most tolerance each: an integer sum, in fixed order
    num_intervals = ticks.shape[0] - 1
    ticks_next, ticks_prev = ticks[1:], ticks[:num_intervals]
    total, count = 0, 0
    for interval_idx in range(num_intervals):
        offset = ticks_next[interval_idx] - ticks_prev[interval_idx] - scan
        near = abs(offset) <= tolerance
        total += offset * near
        count += near
    return scan + total / count if count else 0.0


PHASE_STEPS: int = 10
"""A block's grid is the multiples of its quantum q offset by a phase, a multiple of q /
PHASE_STEPS: ten candidates, so the rounded times stay on a round grid (q/10, a 1-2-5 step too)
and series on the same clock pick the same one."""

PHASE_SAMPLES: int = 64
"""Evenly spaced ticks of a block that choose its phase."""

PHASE_QUANTUM_LIMIT: int = 1 << 56
"""Quanta below which phases are chosen: PHASE_SAMPLES distances of up to half of one add up
within int64 (integer sums vectorize and don't depend on the machine). Coarser blocks (over two
years in ns) keep the epoch's grid."""


@njit(inline="always")
def _round_tick(tick: np.int64, quantum: np.int64, phase: np.int64) -> np.int64:
    """Rounds a tick to the nearest phase + k quantum (quantum > 1, 0 <= phase < quantum), ties
    up; a tick that can't be rounded within the int64 range is returned as it is."""
    int64_min, int64_max = np.int64(INT64_MIN), np.int64(INT64_MAX)
    if tick < int64_min + quantum:
        return tick
    shifted = tick - phase
    # Floored remainder (numba follows Python): 0 <= remainder < quantum
    remainder = shifted % quantum
    down = tick - remainder
    if remainder < quantum - remainder:
        return down
    if down > int64_max - quantum:
        return tick
    return down + quantum


@njit(inline="always")
def block_phase(times: np.ndarray, quantum: np.int64, previous: np.int64, scratch: np.ndarray) -> np.int64:
    """Chooses a rounded block's phase: its grid is phase + k quantum.

    Each of the PHASE_STEPS candidates (multiples of quantum / PHASE_STEPS) is scored by the
    ticks' mean distance to its grid, over PHASE_SAMPLES evenly spaced ticks. A clock on round
    times scores best at 0, the epoch's grid; a free-running one at its own phase. The block
    keeps the previous block's phase (that of its last rounded tick, if a candidate; the epoch's
    grid for the first) unless another scores better by more than half a step per tick: two
    candidates either side of a clock's phase both round it cleanly, and noise would pick
    between them block by block, leaving the block starts uneven; on a noisy clock, it would
    pick a phase a step off. A drifting clock moves on by a step when it has drifted half of
    one.

    Args:
        times: 1D int64 array of the block's ticks (at least 1).
        quantum: The block's quantum (> 1).
        previous: The previous block's last tick as rounded, or INT64_MIN if there is none.
        scratch: 1D int64 scratch array of at least PHASE_SAMPLES.

    Returns:
        The phase, 0 <= phase < quantum (0 for quanta of PHASE_QUANTUM_LIMIT or more).
    """
    if quantum >= PHASE_QUANTUM_LIMIT:
        return np.int64(0)
    block_len = times.shape[0]
    step = max(quantum // PHASE_STEPS, np.int64(1))
    num_candidates = quantum // step
    num_sampled = min(PHASE_SAMPLES, block_len)
    # The sampled ticks' offsets into the grid of phase 0, once
    sampled = scratch[:num_sampled]
    for sampled_idx in range(num_sampled):
        sampled[sampled_idx] = times[sampled_idx * (block_len - 1) // max(num_sampled - 1, 1)] % quantum
    # The previous block's phase if it is a candidate; with none, the epoch's grid (phase 0)
    previous_phase = np.int64(0) if previous == INT64_MIN else np.int64(-1)
    if previous != INT64_MIN and (previous % quantum) % step == 0:
        previous_phase = previous % quantum
    best_cost, best_phase, previous_cost = np.int64(INT64_MAX), np.int64(0), np.int64(INT64_MAX)
    for candidate_idx in range(num_candidates):
        phase = candidate_idx * step
        cost = np.int64(0)
        for sampled_idx in range(num_sampled):
            # |residue - phase| on the circle of the quantum: branch-free, so that it vectorizes
            distance = abs(sampled[sampled_idx] - phase)
            cost += min(distance, quantum - distance)
        if cost < best_cost:
            best_cost, best_phase = cost, phase
        if phase == previous_phase:
            previous_cost = cost
    if previous_cost <= best_cost + num_sampled * step // 2:
        return previous_phase
    return best_phase


FLOAT_ROUNDING_SPAN: int = 1 << 48
"""Largest span of a block's ticks (from its first, rounded) that rounds in float64. The float64
quotient offset/quantum is off by at most about offset x 2^-52 quanta, so its rounded multiple is
off by at most offset x 2^-52 ticks: under 1/16 tick below 2^48. A tick that isn't a tie is at
least half a tick from the rounding point, so it rounds as exact arithmetic would; a tie may go
either way, both within half a quantum."""


@njit(inline="always")
def snap_block(times: np.ndarray, quantum: np.int64, phase: np.int64, out: np.ndarray) -> None:
    """Rounds a block's ticks to the nearest point of its grid: phase + k quantum from tick 0
    (the epoch), so series on the same clock share a grid; ties round up. A tick that can't be
    rounded without leaving the int64 range stays as it is. Rounding keeps the ticks in order.

    Args:
        times: 1D int64 array of the block's ticks (at least 1, no NaT).
        quantum: The block's quantum (> 1).
        phase: Its phase (block_phase), 0 <= phase < quantum.
        out: Output 1D int64 array of the rounded ticks.
    """
    block_len = times.shape[0]
    first, last = times[0], times[block_len - 1]
    base = _round_tick(first, quantum, phase)
    if (np.int64(INT64_MIN) + quantum <= first and last <= np.int64(INT64_MAX) - quantum
            and last - base < FLOAT_ROUNDING_SPAN):
        # Offsets from the rounded first tick, in float64: exact here, and it vectorizes
        # (NEON has no 64-bit integer vector multiply; the rounded offset is an integer
        # below 2^48, so its float64 product is exact). Every offset is at least
        # -quantum/2, so truncating offset/quantum + 0.5 floors.
        quantum_float = np.float64(quantum)
        inverse = 1.0 / quantum_float
        for sample_idx in range(block_len):
            multiple = np.trunc(np.float64(times[sample_idx] - base) * inverse + 0.5)
            out[sample_idx] = base + np.int64(multiple * quantum_float)
    else:
        for sample_idx in range(block_len):
            out[sample_idx] = _round_tick(times[sample_idx], quantum, phase)


@njit(inline="always")
def snap_lattice(times: np.ndarray, origin: np.int64, step: np.int64, min_multiple: np.int64, out: np.ndarray) -> bool:
    """Puts every tick on the nearest point of the lattice origin + k step (k >= min_multiple,
    ties up).

    Args:
        times: 1D int64 array of ticks.
        origin: A point of the lattice (near the ticks).
        step: Its step.
        min_multiple: Smallest k (INT64_MIN for any).
        out: Output 1D int64 array of the ticks on the lattice.

    Returns:
        Whether the ticks were put on it: False (and out not written) for a step outside
        (0, 2^61), which could overflow.
    """
    if not 0 < step < (1 << 61):
        return False
    for sample_idx in range(times.shape[0]):
        offset = times[sample_idx] - origin
        multiple = offset // step
        if 2 * (offset - multiple * step) >= step:
            multiple += 1
        out[sample_idx] = origin + max(multiple, min_multiple) * step
    return True


@njit(inline="always")
def lattice_origin(times: np.ndarray, period: np.int64, quantum: np.int64) -> np.int64:
    """A point of a clock's lattice of the given period: the mean offset from the lattice through
    the first tick of PHASE_SAMPLES evenly spaced ticks within a quarter period of it (a late or
    early tick doesn't pull it, and the first tick's own jitter cancels out), rounded to a
    multiple of quantum / PHASE_STEPS from the epoch, so the times stay round. Float sums: an
    estimate, checked by within_error."""
    first = times[0]
    period_float = np.float64(period)
    block_len = times.shape[0]
    num_sampled = min(PHASE_SAMPLES, block_len)
    total, count = 0.0, 0
    for sampled_idx in range(num_sampled):
        sample_idx = sampled_idx * (block_len - 1) // max(num_sampled - 1, 1)
        offset = np.float64(times[sample_idx] - first)
        residual = offset - np.round(offset / period_float) * period_float
        near = abs(residual) < 0.25 * period_float
        total += residual * near
        count += near
    origin = first + np.int64(np.round(total / max(count, 1)))
    step = max(quantum // PHASE_STEPS, np.int64(1))
    return _round_tick(origin, step, np.int64(0)) if step > 1 else origin


@njit(inline="always")
def fit_lattice(times: np.ndarray, interval: float, quantum: np.int64, max_move: float, out: np.ndarray) -> bool:
    """Puts a clock's ticks on its own lattice, origin + k period: a clock whose period no 1-2-5
    grid divides (1024 Hz, 60 Hz) can then store as regular. round_blocks tries it only where
    the grid leaves the one-scan intervals unequal (scans_equal), and keeps it only if it passes
    within_error.

    The period is the interval rounded to a multiple of the coarsest 1-2-5 step of at most the
    quantum that drifts by at most max_move / 4 over the block (leaving the rest to jitter): a
    1-2-5 period, or 3 ms, is then exact although the mean of a jittered clock's intervals is
    off by a few ns, while 1001 us doesn't become 1000. The origin is lattice_origin.

    Args:
        times: 1D int64 array of the block's ticks (at least 2).
        interval: Its clock period (clock_interval).
        quantum: Its largest quantum (time_quantum of its mean interval, >= 1).
        max_move: Largest distance a tick may move.
        out: Output 1D int64 array of the ticks on the lattice.

    Returns:
        Whether the ticks were put on a lattice.
    """
    num_intervals = times.shape[0] - 1
    step = quantum
    period = np.int64(max(np.round(interval / step), 1.0) * step)
    while step > 1 and abs(interval - period) * num_intervals > 0.25 * max_move:
        step = _finer_step(step)
        period = np.int64(max(np.round(interval / step), 1.0) * step)
    return snap_lattice(times, lattice_origin(times, period, quantum), period, np.int64(INT64_MIN), out)


@njit(inline="always")
def scans_equal(times: np.ndarray, interval: float) -> bool:
    """Whether all of a block's one-scan intervals (strictly between interval / 2 and 3 interval /
    2) are equal: its ticks are a regular clock with gaps, which no lattice makes more regular."""
    lower, upper = 0.5 * interval, 1.5 * interval
    num_intervals = times.shape[0] - 1
    ticks_next, ticks_prev = times[1:], times[:num_intervals]
    scan = np.int64(-1)
    for interval_idx in range(num_intervals):
        delta = ticks_next[interval_idx] - ticks_prev[interval_idx]
        if lower < delta < upper:
            scan = delta
            break
    differing_bits = np.int64(0)
    for interval_idx in range(num_intervals):
        delta = ticks_next[interval_idx] - ticks_prev[interval_idx]
        is_scan = (delta > lower) & (delta < upper)
        differing_bits |= (delta ^ scan) * is_scan
    return differing_bits == 0


@njit(inline="always")
def _finer_step(step: np.int64) -> np.int64:
    """The next 1-2-5 x 10^k step below step (a 1-2-5 step above 1)."""
    decade = np.int64(1)
    while step >= 10 * decade:
        decade *= 10
    mantissa = step // decade
    if mantissa == 1:
        return decade // 2 if decade >= 10 else np.int64(1)
    return decade if mantissa == 2 else 2 * decade


CHECK_CHUNK: int = 64
"""Ticks the branch-free checks (regular_step, within_error, grid_error) take between looks at
whether they can stop."""


@njit(inline="always")
def regular_step(times: np.ndarray) -> int:
    """The common interval of a block of at least 2 ticks if all its intervals are equal and
    positive, else 0."""
    first = times[1] - times[0]
    if first <= 0:
        return 0
    # Branch-free chunks over offset slices, so that it vectorizes and stops soon on irregular ticks
    num_intervals = times.shape[0] - 1
    ticks_next, ticks_prev = times[1:], times[:num_intervals]
    differing_bits = 0
    for chunk_start in range(0, num_intervals, CHECK_CHUNK):
        chunk_end = min(chunk_start + CHECK_CHUNK, num_intervals)
        chunk_next, chunk_prev = ticks_next[chunk_start:chunk_end], ticks_prev[chunk_start:chunk_end]
        for chunk_idx in range(chunk_end - chunk_start):
            differing_bits |= (chunk_next[chunk_idx] - chunk_prev[chunk_idx]) ^ first
        if differing_bits != 0:
            return 0
    return first


TIME_ERROR_OUTLIERS: int = 20
"""One time in this many (5%) may move further than the time error times its own interval."""


@njit(inline="always")
def tick_allowances(times: np.ndarray, time_error: float, out_allowed: np.ndarray) -> None:
    """Each tick's allowed move before it counts as an outlier: time_error times its own interval
    (the smaller of the two either side of it, as given).

    Args:
        times: 1D int64 array of a block's ticks (at least 2).
        time_error: The time error e (> 0).
        out_allowed: Output 1D float64 array, one per tick.
    """
    block_len = times.shape[0]
    out_allowed[0] = time_error * np.float64(times[1] - times[0])
    out_allowed[block_len - 1] = time_error * np.float64(times[block_len - 1] - times[block_len - 2])
    # Offset slices from index 0, so that it vectorizes
    num_inner = block_len - 2
    ticks_prev, ticks_mid, ticks_next = times[:num_inner], times[1:num_inner + 1], times[2:]
    inner = out_allowed[1:num_inner + 1]
    for inner_idx in range(num_inner):
        own = min(ticks_mid[inner_idx] - ticks_prev[inner_idx], ticks_next[inner_idx] - ticks_mid[inner_idx])
        inner[inner_idx] = time_error * np.float64(own)


@njit(inline="always")
def within_allowances(times: np.ndarray, out: np.ndarray, allowed: np.ndarray, max_move: float) -> int:
    """within_error with each tick's allowed move given (tick_allowances)."""
    block_len = times.shape[0]
    max_outliers = block_len // TIME_ERROR_OUTLIERS
    outliers, largest, too_far = 0, 0, 0
    # Branch-free chunks, each over slices of its own (LLVM vectorizes them; it doesn't vectorize
    # an inner loop over a range of the whole arrays); a failing block stops at the chunk it fails in
    for chunk_start in range(0, block_len, CHECK_CHUNK):
        chunk_end = min(chunk_start + CHECK_CHUNK, block_len)
        given, rounded, chunk_allowed = times[chunk_start:chunk_end], out[chunk_start:chunk_end], allowed[chunk_start:chunk_end]
        for chunk_idx in range(chunk_end - chunk_start):
            move = abs(rounded[chunk_idx] - given[chunk_idx])
            move_float = np.float64(move)
            too_far += move_float > max_move
            outliers += move_float > chunk_allowed[chunk_idx]
            largest = max(largest, move)
        if too_far or outliers > max_outliers:
            return -1
    return largest


@njit(inline="always")
def within_error(times: np.ndarray, out: np.ndarray, time_error: float, max_move: float) -> int:
    """Whether rounded ticks keep to the time error: none moved more than max_move, and at most one
    in TIME_ERROR_OUTLIERS more than time_error times its own interval (the smaller of the two
    either side of it, as given).

    Args:
        times: 1D int64 array of a block's ticks as given (at least 2).
        out: 1D int64 array of the same ticks rounded.
        time_error: The time error e (> 0).
        max_move: Largest distance any tick may move.

    Returns:
        The largest move if they keep to it, else -1.
    """
    allowed = np.empty(times.shape[0], np.float64)
    tick_allowances(times, time_error, allowed)
    return within_allowances(times, out, allowed, max_move)


@njit(inline="always")
def _float_offset(bound: np.int64, base: np.int64) -> float:
    """bound - base as a float64, without overflow; beyond +-2^60 (far outside any block) it is
    +-2^60, else exact below 2^53."""
    limit = np.float64(1 << 60)
    if bound >= base:
        return min(np.float64(np.uint64(bound) - np.uint64(base)), limit)
    return -min(np.float64(np.uint64(base) - np.uint64(bound)), limit)


@njit(inline="always")
def grid_error(times: np.ndarray, quantum: np.int64, phase: np.int64, low: np.int64, high: np.int64,
               allowed: np.ndarray, max_move: float, out: np.ndarray) -> int:
    """Rounds the ticks as snap_block and _clamp would, checking them as it goes
    (within_allowances): a failing grid stops at the chunk it fails in, and a passing one is
    read once.

    Returns:
        The largest move if they keep to it (out holds the ticks rounded), else -1 (out partly
        written); -2 if snap_block wouldn't round in float64 here (out not written: the caller
        rounds and checks instead).
    """
    block_len = times.shape[0]
    first, last = times[0], times[block_len - 1]
    base = _round_tick(first, quantum, phase)
    if not (np.int64(INT64_MIN) + quantum <= first and last <= np.int64(INT64_MAX) - quantum
            and last - base < FLOAT_ROUNDING_SPAN):
        return -2
    # As snap_block, in offsets from base: every value below 2^53, so each move is exact
    quantum_float = np.float64(quantum)
    inverse = 1.0 / quantum_float
    low_offset, high_offset = _float_offset(low, base), _float_offset(high, base)
    max_outliers = block_len // TIME_ERROR_OUTLIERS
    outliers, largest, too_far = 0, 0, 0
    # Chunks over slices of their own, as in within_allowances
    for chunk_start in range(0, block_len, CHECK_CHUNK):
        chunk_end = min(chunk_start + CHECK_CHUNK, block_len)
        given, chunk_allowed = times[chunk_start:chunk_end], allowed[chunk_start:chunk_end]
        rounded_ticks = out[chunk_start:chunk_end]
        for chunk_idx in range(chunk_end - chunk_start):
            offset = np.float64(given[chunk_idx] - base)
            rounded = min(max(np.trunc(offset * inverse + 0.5) * quantum_float, low_offset), high_offset)
            rounded_ticks[chunk_idx] = base + np.int64(rounded)
            move = abs(offset - rounded)
            too_far += move > max_move
            outliers += move > chunk_allowed[chunk_idx]
            # An integer maximum: a float one doesn't vectorize (NaN)
            largest = max(largest, int(move))
        if too_far or outliers > max_outliers:
            return -1
    return largest


@njit(inline="always")
def _clamp(out: np.ndarray, low: np.int64, high: np.int64) -> None:
    """Raises ticks below low to it and lowers those above high to it. The ticks are in order
    (rounding keeps them so), so only a run at either end can be outside."""
    block_len = out.shape[0]
    sample_idx = 0
    while sample_idx < block_len and out[sample_idx] < low:
        out[sample_idx] = low
        sample_idx += 1
    sample_idx = block_len - 1
    while sample_idx >= 0 and out[sample_idx] > high:
        out[sample_idx] = high
        sample_idx -= 1


GRID_TEST_CHUNK: int = 64
"""Intervals grid_fails counts between looks at its count."""

GRID_TEST_GIVE_UP: int = 256
"""Intervals after which grid_fails gives up on a grid whose forced count is behind the pace that
could rule it out."""

FLOAT_EXACT_INTERVAL: float = 2.0**52
"""Intervals below which float64 holds them, and their offsets from a grid, exactly."""


@njit(inline="always")
def grid_fails(times: np.ndarray, block_len: int, quantum: np.int64, time_error: float) -> bool:
    """Whether no phase of the grid of quantum can keep a block's ticks to the time error, from
    their intervals alone: a cheap test that lets round_blocks skip a step without choosing its
    phase, rounding and checking.

    Two ticks I apart round to grid points a multiple of quantum apart, so at least one of them
    moves by at least half of I's distance d to the nearest multiple of quantum, and that tick's
    own interval is at most I: if d > 2 time_error I, it moves by more than time_error times its
    own interval. Each tick has two intervals, so more than 2 (n // TIME_ERROR_OUTLIERS) such
    intervals mean more than n // TIME_ERROR_OUTLIERS such ticks, and within_error fails. The
    test can only rule a grid out, never in.

    The caller passes only ticks that round onto the grid whatever the phase: none within a
    quantum of the block's bounds (where a tick may be clamped instead) or of the int64 range.
    The count stops once it rules the grid out, or after GRID_TEST_GIVE_UP intervals if it is
    behind the pace that could.

    Args:
        times: 1D int64 array of a block's ticks (at least 2), less any near its bounds.
        block_len: The whole block's tick count n.
        quantum: The grid's step (> 1).
        time_error: The time error e (> 0).

    Returns:
        True if the grid can't keep to the time error; False if it might.
    """
    num_intervals = times.shape[0] - 1
    limit = 2 * (block_len // TIME_ERROR_OUTLIERS)
    quantum_float = np.float64(quantum)
    inverse = 1.0 / quantum_float
    # A hair under 2 e, so that float rounding never counts an interval that isn't forced
    threshold = 2.0 * time_error * (1.0 - 1e-9)
    ticks_next, ticks_prev = times[1:], times[:num_intervals]
    forced = 0
    for chunk_start in range(0, num_intervals, GRID_TEST_CHUNK):
        chunk_end = min(chunk_start + GRID_TEST_CHUNK, num_intervals)
        # Slices of the chunk's own, so that it vectorizes (as in within_allowances)
        chunk_next, chunk_prev = ticks_next[chunk_start:chunk_end], ticks_prev[chunk_start:chunk_end]
        for chunk_idx in range(chunk_end - chunk_start):
            interval = np.float64(chunk_next[chunk_idx] - chunk_prev[chunk_idx])
            # Distance to the nearest multiple: min(d, quantum - d) also mends a quotient rounded the wrong way at .5
            distance = abs(interval - np.round(interval * inverse) * quantum_float)
            distance = min(distance, quantum_float - distance)
            forced += (distance > threshold * interval) & (interval < FLOAT_EXACT_INTERVAL)
        if forced > limit:
            return True
        if chunk_end >= GRID_TEST_GIVE_UP and forced * num_intervals < limit * chunk_end:
            return False
    return False


@njit(nogil=True, cache=True)
def round_blocks(
    ticks: np.ndarray,
    sample_offsets: np.ndarray,
    time_error: float,
    lows: np.ndarray,
    highs: np.ndarray,
    seed_ticks: np.ndarray,
    seed_steps: np.ndarray,
    out_ticks: np.ndarray,
    out_moves: np.ndarray,
) -> tuple[int, int]:
    """Rounds the blocks' ticks in order, each onto the first candidate that keeps to the time error.

    A block of at least 3 ticks that isn't exactly regular is rounded; its mean interval m is
    the span of its ticks over their intervals. The candidates, each clamped to the block's
    bounds and kept only if within_error (no tick moves more than time_error m, 2% more at
    most; at most 5% more than time_error times their own interval):

    1. The lattice continuing the previous non-empty block, if that is regular with interval d
       (as rounded here) and m rounds to d (the same clock): its last tick + k d, k >= 1.
    2. Its own grid (block_phase, snap_block), of the coarsest 1-2-5 step of at most 2 time_error
       m (time_quantum) that keeps to the time error.
    3. A lattice of its own (fit_lattice), for a clock the grid leaves irregular (one whose
       period no 1-2-5 step divides, like 1024 Hz).

    Otherwise the block is kept. Neighbouring blocks are not reconciled here (clamp_boundaries
    does that), unless bounds keep them apart.

    Args:
        ticks: 1D int64 array of every sample's tick (no NaT).
        sample_offsets: 1D int64 array of the blocks' sample offsets (num_blocks + 1).
        time_error: The time error e (> 0).
        lows: 1D int64 array: the smallest tick each block may round to (INT64_MIN for none).
        highs: 1D int64 array: the largest (INT64_MAX for none).
        seed_ticks: 1D int64 array: the last tick of a stored block before a block, or INT64_MIN
            to carry on from the block before it here.
        seed_steps: 1D int64 array: that stored block's interval if it is regular, else 0 (read
            where seed_ticks is set).
        out_ticks: Output 1D int64 array of every sample's tick, kept or rounded; written only
            if some block moved (else the ticks are as given).
        out_moves: Output 1D int64 array: each block's largest move.

    Returns:
        A tuple of (status, sample_idx): OK, or DECREASING and the index of the first tick below
        its predecessor.
    """
    num_samples = ticks.shape[0]
    # Branch-free order check, so that it vectorizes: a decrease is located afterwards
    decreased = False
    for sample_idx in range(1, num_samples):
        decreased |= ticks[sample_idx] < ticks[sample_idx - 1]
    if decreased:
        for sample_idx in range(1, num_samples):
            if ticks[sample_idx] < ticks[sample_idx - 1]:
                return DECREASING, sample_idx
    pivots = np.empty(CADENCE_SAMPLES, np.int64)
    phase_scratch = np.empty(PHASE_SAMPLES, np.int64)
    trial_ticks = np.empty(num_samples, np.int64)
    allowances = np.empty(num_samples, np.float64)
    # Blocks left as they are aren't copied to out_ticks unless another block moves
    kept = np.zeros(sample_offsets.shape[0] - 1, np.bool_)
    any_moved = False
    previous_last = np.int64(INT64_MIN)
    previous_step = np.int64(0)
    for block_idx in range(sample_offsets.shape[0] - 1):
        first_sample = sample_offsets[block_idx]
        block_len = sample_offsets[block_idx + 1] - first_sample
        out_moves[block_idx] = 0
        if seed_ticks[block_idx] != INT64_MIN:
            previous_last = seed_ticks[block_idx]
            previous_step = seed_steps[block_idx]
        if block_len == 0:
            continue
        times = ticks[first_sample:first_sample + block_len]
        step = np.int64(regular_step(times)) if block_len >= 2 else np.int64(0)
        mean = (times[block_len - 1] - times[0]) / (block_len - 1) if block_len >= 3 else 0.0
        # Exactly regular blocks (and those of 1 or 2 ticks) stay as they are
        if step != 0 or mean <= 0.0:
            kept[block_idx] = True
            previous_last = times[block_len - 1]
            previous_step = step
            continue
        out = out_ticks[first_sample:first_sample + block_len]
        trial = trial_ticks[first_sample:first_sample + block_len]
        allowed = allowances[first_sample:first_sample + block_len]
        tick_allowances(times, time_error, allowed)
        low, high = lows[block_idx], highs[block_idx]
        max_move = time_error * TIME_ERROR_SLACK * mean
        # Each candidate is written to out; the largest move is -1 until one keeps to the error
        largest_move = -1
        # 1. The lattice of the block before, if it is the same clock
        same_clock = previous_step > 0 and times[0] >= previous_last and np.round(mean / previous_step) == 1.0
        if same_clock and snap_lattice(times, previous_last, previous_step, np.int64(1), out):
            _clamp(out, low, high)
            largest_move = within_allowances(times, out, allowed, max_move)
        # 2. Its own grid, the coarsest that keeps to the time error. Once a grid has failed, the
        # steps below it are first tested on the intervals of the ticks no bound or int64 limit
        # can clamp (grid_fails), which rules most failing steps out without a phase
        largest = np.int64(time_quantum(mean, time_error))
        quantum = largest
        safe_first, safe_last = 0, -1
        if largest < (1 << 60):
            safe_low = max(low, np.int64(INT64_MIN) + 2 * largest) + largest
            safe_high = min(high, np.int64(INT64_MAX) - 2 * largest) - largest
            safe_first, safe_last = 0, block_len - 1
            while safe_first < block_len and times[safe_first] < safe_low:
                safe_first += 1
            while safe_last >= 0 and times[safe_last] > safe_high:
                safe_last -= 1
        grid_failed = False
        while largest_move < 0 and quantum > 1:
            if not (grid_failed and safe_last - safe_first >= 1
                    and grid_fails(times[safe_first:safe_last + 1], block_len, quantum, time_error)):
                phase = block_phase(times, quantum, previous_last, phase_scratch)
                largest_move = grid_error(times, quantum, phase, low, high, allowed, max_move, out)
                if largest_move == -2:
                    snap_block(times, quantum, phase, out)
                    _clamp(out, low, high)
                    largest_move = within_allowances(times, out, allowed, max_move)
                grid_failed = True
            quantum = _finer_step(quantum)
        if largest_move < 0:
            out[:] = times
            largest_move = 0
        # 3. A clock still irregular: a lattice of its own
        out_step = np.int64(regular_step(out))
        if out_step == 0:
            interval = clock_interval(times, pivots, max_move)
            clock = interval > 0.0 and not scans_equal(out, interval)
            if clock and fit_lattice(times, interval, largest, max_move, trial):
                _clamp(trial, low, high)
                lattice_move = within_allowances(times, trial, allowed, max_move)
                if lattice_move >= 0:
                    out[:] = trial
                    largest_move = lattice_move
                    out_step = np.int64(regular_step(out))
        out_moves[block_idx] = largest_move
        any_moved |= largest_move > 0
        previous_last = out[block_len - 1]
        previous_step = out_step
    if any_moved:
        for block_idx in range(sample_offsets.shape[0] - 1):
            if kept[block_idx]:
                first_sample, end_sample = sample_offsets[block_idx], sample_offsets[block_idx + 1]
                out_ticks[first_sample:end_sample] = ticks[first_sample:end_sample]
    return OK, 0


@njit(nogil=True, cache=True)
def clamp_boundaries(ticks: np.ndarray, sample_offsets: np.ndarray, moves: np.ndarray, out_ticks: np.ndarray) -> None:
    """Puts rounded blocks back in order where a block's first tick rounded below the ticks before it.

    Each such boundary gets a point m between the two ticks next to it as given (the last before
    it, the block's first); every earlier tick above m is lowered to it and the block's ticks
    below m are raised to it. Every earlier tick was given at or below m and every later one at
    or above it, so a tick is moved only back towards where it was: no tick moves further than
    its rounding did. m is the block's first rounded tick if the block before it moved more (it
    gives way), else that block's last, each limited to the two ticks as given. Boundaries are
    taken in order, so the ticks before each are in order already.

    Args:
        ticks: 1D int64 array of every sample's tick as given (non-decreasing).
        sample_offsets: 1D int64 array of the blocks' sample offsets (num_blocks + 1).
        moves: 1D int64 array of each block's largest move (round_blocks).
        out_ticks: In-out 1D int64 array of every sample's rounded tick (each block in order).
    """
    previous_block, previous_end = -1, 0
    for block_idx in range(sample_offsets.shape[0] - 1):
        first_sample, end = sample_offsets[block_idx], sample_offsets[block_idx + 1]
        if end == first_sample:
            continue
        if previous_block >= 0 and out_ticks[previous_end - 1] > out_ticks[first_sample]:
            low, high = ticks[previous_end - 1], ticks[first_sample]
            if moves[previous_block] >= moves[block_idx]:
                boundary = min(max(out_ticks[first_sample], low), high)
            else:
                boundary = min(max(out_ticks[previous_end - 1], low), high)
            sample_idx = previous_end - 1
            while sample_idx >= 0 and out_ticks[sample_idx] > boundary:
                out_ticks[sample_idx] = boundary
                sample_idx -= 1
            sample_idx = first_sample
            while sample_idx < end and out_ticks[sample_idx] < boundary:
                out_ticks[sample_idx] = boundary
                sample_idx += 1
        previous_block, previous_end = block_idx, end
