# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Exact timestamp encoding/decoding and cadence detection kernels.

Timestamps are int64 ticks in a block group's time unit (seconds through nanoseconds),
naive (no time zone), and non-decreasing within and across blocks. Each block stores:

- ``time_start``: first timestamp in the block (0 if empty).
- ``time_step``: GCD of consecutive within-block timestamp deltas (0 if all equal or < 2 samples).
- ``time_ref_quotient``: reference quotient ``(time[i] - time[i - 1]) / time_step``.
- ``time_residuals`` (irregular blocks only): per-sample zigzagged
  ``(quotient - time_ref_quotient) mod 2^64``, with 0 at sample 0.

Blocks with identical consecutive deltas are regular and store no time residuals. Lossy
timestamp rounding (``Params.time_error > 0``) lives in ``_time_rounding``.
"""

from collections.abc import Callable
from typing import cast

import numpy as np
from numba import njit
from numba.cpython.unsafe.numbers import trailing_zeros as _trailing_zeros_intrinsic

from ._format import BLOCK_FLAG_IRREGULAR_TIME, BLOCK_FLAG_WIDE_TIME

_trailing_zeros = cast("Callable[[int | np.integer], int]", _trailing_zeros_intrinsic)
"""Numba intrinsic counting trailing zero bits, typed with its jitted call signature."""

INT64_MAX: int = 0x7FFFFFFFFFFFFFFF
"""Largest valid int64 timestamp tick."""

INT64_MIN: int = -0x8000000000000000
"""Smallest int64 value, reserved by NumPy datetime64 as NaT (never a valid tick)."""

CADENCE_SAMPLES: int = 15
"""Number of evenly spaced intervals whose median estimates a block's scan interval."""

CADENCE_SHARE: float = 0.9
"""Minimum fraction of one-scan intervals required to classify a block as having a regular cadence."""

# Status codes returned by encode_times and expand_times (plain ints for Numba).
OK: int = 0
"""Timestamps are valid."""

DECREASING: int = 1
"""Encoding error: a timestamp is strictly smaller than its predecessor."""

BAD_STEP: int = 2
"""Decoding error: a block's time step is negative, or zero on an irregular block."""

BAD_FIRST_RESIDUAL: int = 3
"""Decoding error: an irregular block's sample-0 time residual is non-zero."""

OVERFLOW: int = 4
"""Decoding error: reconstructed timestamp exceeds ``INT64_MAX``."""


@njit(inline="always")
def _gcd(first: np.uint64, second: np.uint64) -> np.uint64:
    """Computes the greatest common divisor of two uint64 values.

    Args:
        first: First uint64 operand (``_gcd(0, second) == second``).
        second: Second uint64 operand.

    Returns:
        Greatest common divisor of ``first`` and ``second``.
    """
    while second != 0:
        first, second = second, first % second
    return first


@njit(inline="always")
def _odd_inverse(odd: np.uint64) -> np.uint64:
    """Computes the multiplicative inverse of an odd uint64 modulo ``2^64``.

    Args:
        odd: Odd uint64 integer.

    Returns:
        uint64 ``inv`` satisfying ``(odd * inv) mod 2^64 == 1``.
    """
    # Newton's method mod 2^64: odd * odd == 1 (mod 8) gives 3 correct bits initially,
    # and 5 iterations double precision to 64 bits (3 -> 6 -> 12 -> 24 -> 48 -> 64).
    inverse = odd
    for _ in range(5):
        inverse *= np.uint64(2) - odd * inverse
    return inverse


@njit(inline="always")
def _deltas_gcd(times: np.ndarray, num_deltas: int) -> np.uint64:
    """Computes the GCD of the first ``num_deltas`` consecutive timestamp deltas.

    Args:
        times: 1D int64 array of non-decreasing timestamps.
        num_deltas: Number of consecutive deltas to inspect (``<= len(times) - 1``).

    Returns:
        GCD of the inspected deltas, or 0 if all inspected deltas are 0.
    """
    step_gcd = np.uint64(0)
    for sample_idx in range(1, num_deltas + 1):
        delta = np.uint64(times[sample_idx]) - np.uint64(times[sample_idx - 1])
        # Fast path: skip Euclidean division when delta is already divisible by step_gcd.
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
    """Divides every consecutive timestamp delta by ``step`` and checks exact divisibility.

    The returned ``minimum`` and ``total`` are meaningful only when ``divisible`` is True.

    Args:
        times: 1D int64 array of non-decreasing timestamps.
        step: Positive uint64 candidate divisor.
        out_quotients: Preallocated 1D uint64 array receiving quotients at indices ``1..n-1``.

    Returns:
        Tuple ``(divisible, minimum, total)`` where ``divisible`` is True iff ``step`` divides
        every delta exactly, ``minimum`` is the smallest quotient, and ``total`` is the sum
        of all quotients.
    """
    # Factor step = odd * 2^shift and replace 64-bit hardware division with a right shift and
    # multiplication by the modular inverse of odd; exact divisibility holds iff all shifted-out
    # low bits are zero and every quotient stays within floor((2^64 - 1) / odd).
    minimum = np.uint64(0xFFFFFFFFFFFFFFFF)
    total = np.uint64(0)
    if step == 1:
        # Common nanosecond case: quotients equal raw deltas directly.
        for sample_idx in range(1, times.shape[0]):
            quotient = np.uint64(times[sample_idx]) - np.uint64(times[sample_idx - 1])
            out_quotients[sample_idx] = quotient
            minimum = min(minimum, quotient)
            total += quotient
        return True, minimum, total
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
        total += quotient
    return bool(low_bits == 0 and largest <= multiple_limit), minimum, total


@njit(inline="always")
def _reference_quotient(quotients: np.ndarray, minimum: np.uint64, total: np.uint64) -> np.uint64:
    """Selects the reference quotient used to center an irregular block's time residuals.

    Args:
        quotients: 1D uint64 array of block quotients (valid entries at indices ``1..n-1``).
        minimum: Minimum quotient in the block.
        total: Sum of all quotients in the block.

    Returns:
        Selected uint64 reference quotient (rounded mean for tight distributions, else minimum).
    """
    # Compare variance against squared distance from the minimum to decide whether centering on
    # the mean produces smaller zigzagged residuals than anchoring at the minimum. Accumulate
    # float64 sums in four fixed-order lanes to break serial FP latency without fastmath
    # (which would make encoded bytes depend on the CPU's SIMD width).
    num_quotients = quotients.shape[0] - 1
    first = np.float64(quotients[1])
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
        # Compute rounded mean without overflowing total + num_quotients // 2.
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
    """Encodes each block's timestamps into start, step, reference quotient, and optional residuals.

    Clears and updates ``BLOCK_FLAG_IRREGULAR_TIME`` and ``BLOCK_FLAG_WIDE_TIME`` in
    ``out_block_flags`` while preserving all other flag bits. Empty blocks receive 0 for start,
    step, and reference; single-sample blocks receive 0 for step and reference. Also verifies
    monotonic non-decreasing order within and across all non-empty blocks.

    Args:
        ticks: 1D int64 array of all samples' timestamp ticks.
        sample_offsets: 1D int64 array of cumulative per-block sample offsets (``num_blocks + 1``).
        out_block_flags: In-out 1D uint8 array of per-block flags.
        out_time_starts: Output 1D int64 array receiving block start timestamps.
        out_time_steps: Output 1D int64 array receiving per-block delta GCDs.
        out_time_refs: Output 1D uint64 array receiving per-block reference quotients.
        out_time_residuals: Output 1D uint64 array receiving zigzagged residuals for irregular
            blocks (slices for regular blocks are left untouched).

    Returns:
        Tuple ``(OK, 0)`` on success, or ``(DECREASING, sample_idx)`` naming the first flat
        sample index whose timestamp is smaller than its predecessor.
    """
    # For each block: verify monotonic order and check regularity in one branch-free pass.
    # Regular blocks store only (start, step, ref). Irregular blocks estimate step GCD from
    # the first 8 deltas (falling back to full GCD if needed), pick a reference quotient, and
    # zigzag-encode residuals in place, setting BLOCK_FLAG_WIDE_TIME if any exceeds 32 bits.
    num_blocks = sample_offsets.shape[0] - 1
    int64_max = np.uint64(INT64_MAX)
    previous_last = -1
    for block_idx in range(num_blocks):
        first_sample = sample_offsets[block_idx]
        block_len = sample_offsets[block_idx + 1] - first_sample
        out_block_flags[block_idx] &= ~(BLOCK_FLAG_IRREGULAR_TIME | BLOCK_FLAG_WIDE_TIME)
        if block_len:
            # Verify monotonic ordering across non-empty block boundaries.
            if previous_last >= 0 and ticks[first_sample] < ticks[previous_last]:
                return DECREASING, first_sample
            previous_last = first_sample + block_len - 1
        if block_len < 2:
            # Fewer than 2 samples means no deltas: step and reference are 0.
            out_time_starts[block_idx] = ticks[first_sample] if block_len else 0
            out_time_steps[block_idx] = 0
            out_time_refs[block_idx] = 0
            continue
        times = ticks[first_sample : first_sample + block_len]
        out_time_starts[block_idx] = times[0]
        # Branch-free regularity and monotonicity scan so LLVM vectorizes it; locate any
        # decrease in a cold fallback loop.
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
            # Regular block: all deltas equal. A single delta exceeding INT64_MAX is encoded
            # as ref = delta over step = 1 so step always fits in signed int64.
            if first_delta > int64_max:
                out_time_steps[block_idx] = 1
                out_time_refs[block_idx] = first_delta
            else:
                out_time_steps[block_idx] = np.int64(first_delta)
                out_time_refs[block_idx] = np.uint64(first_delta != 0)
            continue
        quotients = out_time_residuals[first_sample : first_sample + block_len]
        quotients[0] = 0
        # Optimistically test the GCD of the first 8 deltas while dividing all deltas in one pass.
        step_gcd = _deltas_gcd(times, min(8, block_len - 1))
        divisible = False
        if 0 < step_gcd <= int64_max:
            divisible, minimum, total = _divide_deltas(times, step_gcd, quotients)
        if not divisible:
            step_gcd = _deltas_gcd(times, block_len - 1)
            if step_gcd > int64_max:
                # Only possible when a delta spans more than half the int64 range: fall back to step 1.
                step_gcd = np.uint64(1)
            _, minimum, total = _divide_deltas(times, step_gcd, quotients)
        reference = _reference_quotient(quotients, minimum, total)
        # Zigzag-encode residuals mod 2^64 in place and track whether any residual exceeds 32 bits.
        all_bits = np.uint64(0)
        for sample_idx in range(1, block_len):
            residual = np.int64(quotients[sample_idx] - reference)
            quotients[sample_idx] = np.uint64((residual << 1) ^ (residual >> 63))
            all_bits |= quotients[sample_idx]
        out_time_steps[block_idx] = np.int64(step_gcd)
        out_time_refs[block_idx] = reference
        out_block_flags[block_idx] |= BLOCK_FLAG_IRREGULAR_TIME
        if all_bits >> np.uint64(32):
            out_block_flags[block_idx] |= BLOCK_FLAG_WIDE_TIME
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
    """Reconstructs timestamp ticks for the specified blocks from their encoded time columns.

    Args:
        block_flags: 1D uint8 array of per-block flags.
        sample_offsets: 1D int64 array of cumulative per-block sample offsets (``num_blocks + 1``).
        time_starts: 1D int64 array of absolute block start timestamps.
        time_steps: 1D int64 array of per-block time steps.
        time_refs: 1D uint64 array of per-block reference quotients.
        time_residuals: 1D uint64 array of zigzagged time residuals (read only for irregular blocks).
        block_ids: 1D int64 array of block indices to expand.
        out_times: Preallocated 1D int64 array sized for all samples; only slices belonging to
            ``block_ids`` are written.

    Returns:
        Tuple ``(OK, 0)`` on success, or ``(status, block_idx)`` with ``BAD_STEP``,
        ``BAD_FIRST_RESIDUAL``, or ``OVERFLOW`` for the first invalid block.
    """
    # For each requested block, validate step and compute the maximum safe cumulative quotient
    # (quotient_limit) with one division per block. Regular blocks expand linearly; irregular
    # blocks reverse zigzag encoding and integrate quotients while checking overflow at each step.
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
        times = out_times[first_sample : first_sample + block_len]
        if step == 0:
            times[:] = time_starts[block_idx]
            continue
        step_unsigned = np.uint64(step)
        # start + step * quotient_sum <= INT64_MAX iff quotient_sum <= (INT64_MAX - start) // step.
        quotient_limit = (int64_max - start) // step_unsigned
        times[0] = time_starts[block_idx]
        if not irregular:
            if reference > quotient_limit // np.uint64(block_len - 1):
                return OVERFLOW, block_idx
            increment = reference * step_unsigned
            for sample_idx in range(1, block_len):
                times[sample_idx] = np.int64(start + np.uint64(sample_idx) * increment)
            continue
        residuals = time_residuals[first_sample : first_sample + block_len]
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
def sampled_intervals(ticks: np.ndarray, scratch_pivots: np.ndarray) -> np.ndarray:
    """Samples ``CADENCE_SAMPLES`` evenly spaced consecutive intervals from ``ticks``, sorted.

    Args:
        ticks: 1D int64 array of block timestamps (at least 2 samples).
        scratch_pivots: Preallocated 1D int64 scratch buffer of length ``>= CADENCE_SAMPLES``.

    Returns:
        View ``scratch_pivots[:CADENCE_SAMPLES]`` containing the sampled intervals in ascending order.
    """
    # Insertion-sort directly into scratch_pivots to avoid heap allocation inside Numba kernels.
    num_intervals = ticks.shape[0] - 1
    pivots = scratch_pivots[:CADENCE_SAMPLES]
    for pivot_idx in range(CADENCE_SAMPLES):
        interval_idx = pivot_idx * (num_intervals - 1) // (CADENCE_SAMPLES - 1)
        pivot = ticks[interval_idx + 1] - ticks[interval_idx]
        insert_idx = pivot_idx
        while insert_idx > 0 and pivots[insert_idx - 1] > pivot:
            pivots[insert_idx] = pivots[insert_idx - 1]
            insert_idx -= 1
        pivots[insert_idx] = pivot
    return pivots


@njit(inline="always")
def median_interval(ticks: np.ndarray, scratch_pivots: np.ndarray) -> int:
    """Estimates a block's median interval from ``CADENCE_SAMPLES`` evenly spaced intervals.

    Args:
        ticks: 1D int64 array of block timestamps (at least 2 samples).
        scratch_pivots: Preallocated 1D int64 scratch buffer of length ``>= CADENCE_SAMPLES``.

    Returns:
        Median of the sampled intervals.
    """
    return sampled_intervals(ticks, scratch_pivots)[CADENCE_SAMPLES // 2]


@njit(inline="always")
def detect_scan_interval(ticks: np.ndarray, scratch_pivots: np.ndarray) -> tuple[int, int, int, int]:
    """Detects a block's dominant single-scan interval and counts how many intervals match it.

    An interval ``d`` counts as one scan when strictly inside ``(scan / 2, 3 * scan / 2)``,
    separating single scans (including clock jitter) from skipped scans (2+ periods) and
    decreasing/duplicate timestamps.

    Args:
        ticks: 1D int64 array of block timestamps (at least 2 samples).
        scratch_pivots: Preallocated 1D int64 scratch buffer of length ``>= CADENCE_SAMPLES``.

    Returns:
        Tuple ``(scan, lower, upper, num_one_scan)`` where an interval ``d`` is one scan iff
        ``lower < d < upper``. Returns ``(0, 0, 0, 0)`` if ``scan`` is not in ``(0, 2^62)``.
    """
    # Compute integer half-open bounds (scan // 2, scan + (scan + 1) // 2) for exact
    # (scan / 2, 3 * scan / 2) membership without floating-point rounding, then count matches
    # over zero-based slices so LLVM vectorizes the loop.
    num_intervals = ticks.shape[0] - 1
    scan = median_interval(ticks, scratch_pivots)
    if not 0 < scan < (1 << 62):
        return 0, 0, 0, 0
    lower, upper = scan // 2, scan + (scan + 1) // 2
    ticks_next, ticks_prev = ticks[1:], ticks[:num_intervals]
    num_one_scan = 0
    for interval_idx in range(num_intervals):
        interval = ticks_next[interval_idx] - ticks_prev[interval_idx]
        num_one_scan += (interval > lower) & (interval < upper)
    return scan, lower, upper, num_one_scan
