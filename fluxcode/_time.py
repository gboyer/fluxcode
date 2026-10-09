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

With a time error e > 0 (Params.time_error) the encoder first rounds each block's ticks to a
1-2-5 step of at most 2 e times the block's interval (`time_quanta`, `snap_times`): a jittered
clock lands back on its own grid and stores as a regular block. The format doesn't change.
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
times, and the mean one-scan interval as the time error's reference."""

TIME_ERROR_SLACK: float = 1.02
"""The time quantum may be up to 2% above 2 e times the interval: a clock's mean interval lands a
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


ONE_SCAN_SUM_LIMIT: int = 1 << 46
"""Scan intervals below which one_scan's sum of one-scan intervals fits int64: 65,535 of them
below 1.5 x 2^46 add up within it. Integer sums vectorize; a float sum is a serial chain."""


@njit(inline="always")
def one_scan(ticks: np.ndarray, scratch_pivots: np.ndarray) -> tuple[int, int, int, int, int, bool]:
    """A block's scan interval and its one-scan intervals.

    The scan interval c is the median of CADENCE_SAMPLES evenly spaced intervals
    (median_interval); an interval strictly between c/2 and 3c/2 is one scan. That separates
    one scan from two (a skipped scan) and covers jittered times. A decrease in the times gives
    a negative interval, never one scan.

    Args:
        ticks: 1D int64 array of the block's ticks (at least 2).
        scratch_pivots: 1D int64 scratch array of at least CADENCE_SAMPLES.

    Returns:
        A tuple of (scan, lower, upper, num_one_scan, one_scan_total, regular): an interval d
        is one scan if lower < d < upper; the number of one-scan intervals and their sum (which
        wraps, so is meaningful only for a scan below ONE_SCAN_SUM_LIMIT); and whether every
        interval is equal. All 0 (and not regular) if the scan isn't in (0, 2^62).
    """
    num_intervals = ticks.shape[0] - 1
    scan = median_interval(ticks, scratch_pivots)
    # Strictly between scan/2 and 3 scan/2 in integers: no rounding, and no overflow below 2^62
    if not 0 < scan < (1 << 62):
        return 0, 0, 0, 0, 0, False
    lower, upper = scan // 2, scan + (scan + 1) // 2
    # Offset slices from index 0 allow SIMD vectorization without negative index checks
    ticks_next, ticks_prev = ticks[1:], ticks[:num_intervals]
    first_interval = ticks_next[0] - ticks_prev[0]
    num_one_scan = 0
    one_scan_total = 0
    differing_bits = 0
    for interval_idx in range(num_intervals):
        interval = ticks_next[interval_idx] - ticks_prev[interval_idx]
        is_one_scan = (interval > lower) & (interval < upper)
        num_one_scan += is_one_scan
        one_scan_total += interval * is_one_scan
        differing_bits |= interval ^ first_interval
    return scan, lower, upper, num_one_scan, one_scan_total, differing_bits == 0


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


@njit(nogil=True, cache=True)
def time_quanta(
    ticks: np.ndarray,
    sample_offsets: np.ndarray,
    time_error: float,
    scratch_pivots: np.ndarray,
    out_quanta: np.ndarray,
    out_regular: np.ndarray,
) -> None:
    """Chooses each block's time quantum: the step its ticks are rounded to.

    The block's interval is its scan interval c (one_scan), or, if at least CADENCE_SHARE of its
    intervals are one scan and c is below ONE_SCAN_SUM_LIMIT, their mean: stable to about 0.2%
    on a 5% jittered clock, where the median of 15 wanders by 2%.

    Args:
        ticks: 1D int64 array of every sample's tick.
        sample_offsets: 1D int64 array of the blocks' sample offsets (num_blocks + 1).
        time_error: The time error e (> 0).
        scratch_pivots: 1D int64 scratch array of at least CADENCE_SAMPLES.
        out_quanta: Output 1D int64 array of the blocks' quanta: 0 where there is none
            (fewer than 2 samples, a median interval of 0, or no step above 1).
        out_regular: Output 1D bool array: the block is regular. It stores as cheaply as it can
            already, so rounding it could only move its times.
    """
    num_blocks = sample_offsets.shape[0] - 1
    for block_idx in range(num_blocks):
        first_sample = sample_offsets[block_idx]
        block_len = sample_offsets[block_idx + 1] - first_sample
        out_quanta[block_idx] = 0
        out_regular[block_idx] = False
        if block_len < 2:
            continue
        times = ticks[first_sample:first_sample + block_len]
        scan, _, _, num_one_scan, one_scan_total, regular = one_scan(times, scratch_pivots)
        if scan == 0:
            continue
        reference = float(scan)
        if scan < ONE_SCAN_SUM_LIMIT and num_one_scan >= CADENCE_SHARE * (block_len - 1):
            reference = one_scan_total / num_one_scan
        quantum = time_quantum(reference, time_error)
        if quantum <= 1:
            continue
        out_quanta[block_idx] = quantum
        out_regular[block_idx] = regular


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


@njit(nogil=True, cache=True)
def time_phases(
    ticks: np.ndarray, sample_offsets: np.ndarray, quanta: np.ndarray, seed_ticks: np.ndarray, out_phases: np.ndarray
) -> None:
    """Chooses each rounded block's phase: its grid is phase + k quantum.

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
        ticks: 1D int64 array of every sample's tick.
        sample_offsets: 1D int64 array of the blocks' sample offsets (num_blocks + 1).
        quanta: 1D int64 array of the blocks' quanta (0 or 1: not rounded).
        seed_ticks: 1D int64 array: a block's previous tick (the last of a stored block before
            it), or INT64_MIN to carry on from the block before it here.
        out_phases: Output 1D int64 array of the blocks' phases (0 where not rounded).
    """
    num_blocks = sample_offsets.shape[0] - 1
    previous = np.int64(INT64_MIN)
    residues = np.empty(PHASE_SAMPLES, np.int64)
    for block_idx in range(num_blocks):
        first_sample = sample_offsets[block_idx]
        block_len = sample_offsets[block_idx + 1] - first_sample
        out_phases[block_idx] = 0
        if seed_ticks[block_idx] != INT64_MIN:
            previous = seed_ticks[block_idx]
        if block_len == 0:
            continue
        times = ticks[first_sample:first_sample + block_len]
        quantum = np.int64(quanta[block_idx])
        if quantum <= 1 or quantum >= PHASE_QUANTUM_LIMIT:
            previous = times[block_len - 1] if quantum <= 1 else _round_tick(times[block_len - 1], quantum, np.int64(0))
            continue
        step = max(quantum // PHASE_STEPS, np.int64(1))
        num_candidates = quantum // step
        num_sampled = min(PHASE_SAMPLES, block_len)
        sampled = residues[:num_sampled]
        # The sampled ticks' offsets into the grid of phase 0, once
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
        phase = best_phase
        if previous_cost <= best_cost + num_sampled * step // 2:
            phase = previous_phase
        out_phases[block_idx] = phase
        previous = _round_tick(times[block_len - 1], quantum, phase)


FLOAT_ROUNDING_SPAN: int = 1 << 48
"""Largest span of a block's ticks (from its first, rounded) that rounds in float64. The float64
quotient offset/quantum is off by at most about offset x 2^-52 quanta, so its rounded multiple is
off by at most offset x 2^-52 ticks: under 1/16 tick below 2^48. A tick that isn't a tie is at
least half a tick from the rounding point, so it rounds as exact arithmetic would; a tie may go
either way, both within half a quantum."""


@njit(nogil=True, cache=True)
def snap_times(
    ticks: np.ndarray, sample_offsets: np.ndarray, quanta: np.ndarray, phases: np.ndarray, out_ticks: np.ndarray
) -> tuple[int, int]:
    """Rounds each block's ticks to the nearest point of its grid, keeping them in order.

    A block's grid is its phase plus multiples of its quantum, from tick 0 (the epoch), so
    series on the same clock share a grid; ties round up. A tick that can't be rounded without leaving the int64 range stays as it is.
    Rounding with one quantum keeps ticks in order. Where neighbouring blocks' quanta differ
    and the earlier block's last ticks round past the later one's first, the block with the
    larger quantum gives way: the earlier block's trailing ticks are lowered to it, or the
    later block's leading ticks raised to the earlier one's last. Either way a tick stays
    within half its own block's quantum of where it was.

    Args:
        ticks: 1D int64 array of every sample's tick (no NaT).
        sample_offsets: 1D int64 array of the blocks' sample offsets (num_blocks + 1).
        quanta: 1D int64 array of the blocks' quanta (0 or 1 keeps a block's ticks).
        phases: 1D int64 array of the blocks' phases (time_phases), 0 <= phase < quantum.
        out_ticks: Output 1D int64 array of every sample's rounded tick.

    Returns:
        A tuple of (status, sample_idx): OK, or DECREASING and the index of the first tick
        below its predecessor (checked on the given ticks: rounding could hide a decrease).
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
    num_blocks = sample_offsets.shape[0] - 1
    for block_idx in range(num_blocks):
        first_sample = sample_offsets[block_idx]
        block_len = sample_offsets[block_idx + 1] - first_sample
        if block_len == 0:
            continue
        times = ticks[first_sample:first_sample + block_len]
        out = out_ticks[first_sample:first_sample + block_len]
        quantum = np.int64(quanta[block_idx])
        if quantum <= 1:
            out[:] = times
            continue
        phase = np.int64(phases[block_idx])
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
    # Rounding keeps each block in order: only ticks next to a boundary can cross it
    previous_block = -1
    for block_idx in range(num_blocks):
        first_sample = sample_offsets[block_idx]
        if sample_offsets[block_idx + 1] == first_sample:
            continue
        if previous_block >= 0:
            previous_first = sample_offsets[previous_block]
            sample_idx = first_sample - 1
            # The earlier block rounded up past this one with the larger quantum: lower its
            # trailing ticks
            if out_ticks[sample_idx] > out_ticks[first_sample] and max(quanta[previous_block], 1) > max(
                quanta[block_idx], 1
            ):
                while sample_idx >= previous_first and out_ticks[sample_idx] > out_ticks[first_sample]:
                    out_ticks[sample_idx] = out_ticks[first_sample]
                    sample_idx -= 1
        previous_block = block_idx
    # Then raise any block's leading ticks still below the previous block's last: this block
    # rounded down past it (or, on blocks of a few ticks, the lowering went too far back)
    previous = np.int64(INT64_MIN)
    for block_idx in range(num_blocks):
        first_sample = sample_offsets[block_idx]
        end = sample_offsets[block_idx + 1]
        sample_idx = first_sample
        while sample_idx < end and out_ticks[sample_idx] < previous:
            out_ticks[sample_idx] = previous
            sample_idx += 1
        if end > first_sample:
            previous = out_ticks[end - 1]
    return OK, 0
