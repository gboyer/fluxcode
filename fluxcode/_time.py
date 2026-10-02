# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Time axis kernels: per-block regularity, GCD and reference analysis, and exact reconstruction.

Timestamps are int64 ticks in the unit's time unit (seconds to nanoseconds), naive
(no time zone). Within a block they must be non-decreasing. Every block stores a start,
a step (the GCD of its time deltas) and a reference quotient; sample i > 0 has the
quotient (time[i] - time[i-1]) / step, stored as the zigzagged residual
quotient - reference (mod 2^64; 0 for sample 0). A block whose deltas are all equal is
regular: every quotient equals the reference (1, or 0 when all times are equal), so it
stores no residuals. Any other block is irregular and stores them.
"""

import enum
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


class TimeStatus(enum.IntEnum):
    """Outcome of time axis analysis (encoding) or expansion (decoding)."""

    OK = 0
    DECREASING = 1
    BAD_STEP = 2
    BAD_FIRST_RESIDUAL = 3
    OVERFLOW = 4


OK: int = int(TimeStatus.OK)
"""The times are valid."""

DECREASING: int = int(TimeStatus.DECREASING)
"""Encoding: a time is below the previous time in its block."""

BAD_STEP: int = int(TimeStatus.BAD_STEP)
"""Decoding: a time step is negative, or zero on an irregular block."""

BAD_FIRST_RESIDUAL: int = int(TimeStatus.BAD_FIRST_RESIDUAL)
"""Decoding: an irregular block's first time residual is not 0."""

OVERFLOW: int = int(TimeStatus.OVERFLOW)
"""Decoding: a time exceeds int64 maximum."""


@njit(inline="always")
def _gcd(first: np.uint64, second: np.uint64) -> np.uint64:
    """Greatest common divisor of two uint64 values (gcd(0, x) = x)."""
    while second != 0:
        first, second = second, first % second
    return first


@njit(inline="always")
def _odd_inverse(odd: np.uint64) -> np.uint64:
    """Multiplicative inverse of an odd number mod 2^64, by Newton's iteration.

    odd * odd = 1 mod 8, so odd is its own inverse to 3 bits; each step doubles the correct
    bits (3, 6, 12, 24, 48, 96).
    """
    inverse = odd
    for _ in range(5):
        inverse *= np.uint64(2) - odd * inverse
    return inverse


@njit(inline="always")
def _deltas_gcd(times: np.ndarray, num_deltas: int) -> np.uint64:
    """GCD of the first num_deltas time deltas (0 if they are all 0).

    Most deltas are multiples of the GCD so far, which skips the Euclid steps.
    """
    step_gcd = np.uint64(0)
    for sample_idx in range(1, num_deltas + 1):
        delta = np.uint64(times[sample_idx]) - np.uint64(times[sample_idx - 1])
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

    Exact division without a 64-bit divide: shift out the power of two, then multiply by the
    odd part's inverse mod 2^64. That is exact for multiples of the odd part, and a delta is
    one iff its shifted-out bits are 0 and the product is at most (2^64 - 1) // odd.

    Returns:
        A tuple of (divisible, minimum, total): whether step divides every delta, and the
        minimum and sum of the quotients written to out_quotients[1:] (valid if divisible).
    """
    minimum = np.uint64(0xFFFFFFFFFFFFFFFF)
    total = np.uint64(0)
    if step == 1:
        # The deltas themselves (the common ns case): no multiply, always divisible
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
        # The quotients sum to (last time - first time) / step, below 2^64: no overflow
        total += quotient
    return bool(low_bits == 0 and largest <= multiple_limit), minimum, total


@njit(inline="always")
def _reference_quotient(quotients: np.ndarray, minimum: np.uint64, total: np.uint64) -> np.uint64:
    """The reference that the block's quotients (from index 1, with the given minimum and
    total) are stored relative to.

    The rounded mean suits quotients spread around a center (clock jitter); the minimum
    suits skewed ones (gaps, events, deadband logging), which the mean would shift away
    from most of them. Residuals from the mean are about 2 sigma once zigzagged; from the
    minimum they are non-negative, so zigzag only moves them up one bit plane (a free,
    all-zero plane 0), and their size is about sqrt(sigma^2 + (mean - min)^2). So the mean
    wins when 3 sigma^2 < (mean - min)^2. The float statistics only choose between two
    exact integer references, shifted by the first quotient to keep their precision.
    """
    num_quotients = quotients.shape[0] - 1
    first = np.float64(quotients[1])
    # Float sums in four fixed-order lanes: float addition doesn't reassociate, so one running
    # sum is a serial chain, while fastmath would make the result (and so the unit bytes)
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
