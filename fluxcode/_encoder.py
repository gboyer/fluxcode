# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Encoding stages and numba kernels for float64 series compression.

Provides per-block analysis, noise floor gating, decimal detection, power-of-two
quantization, predictor order selection, residual differentiation mod 2^16, and
per-unit bit target allocation. Composed per unit in `encode_unit`.
"""

import math
from collections.abc import Callable
from typing import cast

import numpy as np
from numba import njit
from numba.cpython.unsafe.numbers import leading_zeros as _leading_zeros

from . import _extreme_magnitudes as xm
from ._format import (
    BLOCK_FLAG_DECIMAL,
    BLOCK_FLAG_NONFINITE,
    E_MAX,
    E_MIN,
    P_MAX,
    P_MIN,
    SHORT_BLOCK_LEN,
)
from ._noise import noise
from ._nonfinite import fill_nonfinite, noise_finite

leading_zeros = cast("Callable[[int | np.integer], int]", _leading_zeros)
"""Numba intrinsic counting leading zero bits, typed as its jitted call signature."""

NO_DECIMAL: int = 99
"""Sentinel indicating no decimal grid coarser than the baseline step was found."""

NOISE_RHO: float = -0.6
"""Autocorrelation threshold for noise gating (white noise gives rho = -2/3)."""

NOISE_MIN_LEN: int = 256
"""Smallest block the noise floor applies to. Below it the rho test can't tell white noise (rho
= -2/3) from a random walk (-1/2): the estimate's standard error is about
sqrt((1 - 3 rho^2 + 4 rho^4) / m) on m differences (0.044 at 256 samples, so -0.6 is 2.25 of them
from -1/2). Random walks pass the gate 9% of the time at 64 samples, 0.8% at 256 and none at 1000,
while white noise still passes 96% of the time at 256 (SPEC §3.1a)."""

PICK_LEN: int = 250
"""Number of initial samples evaluated to select the predictor difference order."""

FLOAT64_EXACT_INT_LIMIT: float = 2.0 ** 52
"""Scaled-magnitude limit for decimal candidates: below 2^52 a double's spacing is at most
0.5, so floor(v + 0.5) and the tolerance check are meaningful."""

NUM_HIST_CHAINS: int = 4
"""Number of interleaved histogram accumulator chains to prevent pipeline write stalls."""

NUM_BIT_LENGTH_BINS: int = 17
"""Total bit-length classes (0 through 16) for zigzagged uint16 residuals."""

UINT16_MASK: int = 0xFFFF
"""Bitmask for 16-bit unsigned integers (65535)."""

INT16_WRAP_OFFSET: int = 32768
"""Half-range offset (2^15) used for modular symmetric signed int16 wrapping."""



@njit(nogil=True, cache=True, fastmath=True)
def _min_max(samples: np.ndarray) -> tuple[float, float]:
    """Computes minimum and maximum over finite samples using four parallel chains.

    Args:
        samples: 1D float64 array of finite values.

    Returns:
        A tuple of (min_val, max_val).
    """
    num_samples = samples.shape[0]
    # Four independent accumulator chains for instruction-level parallelism
    min_chain0 = min_chain1 = min_chain2 = min_chain3 = samples[0]
    max_chain0 = max_chain1 = max_chain2 = max_chain3 = samples[0]
    for sample_idx in range(0, num_samples - 3, 4):
        val0 = samples[sample_idx]
        val1 = samples[sample_idx + 1]
        val2 = samples[sample_idx + 2]
        val3 = samples[sample_idx + 3]
        min_chain0 = min(min_chain0, val0)
        min_chain1 = min(min_chain1, val1)
        min_chain2 = min(min_chain2, val2)
        min_chain3 = min(min_chain3, val3)
        max_chain0 = max(max_chain0, val0)
        max_chain1 = max(max_chain1, val1)
        max_chain2 = max(max_chain2, val2)
        max_chain3 = max(max_chain3, val3)
    # Process remaining elements
    for sample_idx in range(num_samples - num_samples % 4, num_samples):
        min_chain0 = min(min_chain0, samples[sample_idx])
        max_chain0 = max(max_chain0, samples[sample_idx])
    final_min = min(min(min_chain0, min_chain1), min(min_chain2, min_chain3))
    final_max = max(max(max_chain0, max_chain1), max(max_chain2, max_chain3))
    return final_min, final_max


@njit(nogil=True, cache=True, fastmath={"reassoc", "nsz"})
def _sum(samples: np.ndarray) -> float:
    """Computes sum of array elements with fastmath reassociation enabled.

    Args:
        samples: 1D float64 array.

    Returns:
        Accumulated sum.
    """
    accum_sum = 0.0
    for sample_idx in range(samples.shape[0]):
        # Vectorized floating-point sum
        accum_sum += samples[sample_idx]
    return accum_sum


@njit(nogil=True, cache=True)
def block_stats(samples: np.ndarray) -> tuple[float, float, float, int]:
    """Computes block minimum, maximum, mean, and identifies non-finite samples.

    Args:
        samples: 1D float64 array of samples.

    Returns:
        A tuple of (lo, hi, mean, bad_index):
            lo: Minimum value (or 0.0 if non-finite sample present).
            hi: Maximum value (or 0.0 if non-finite sample present).
            mean: Arithmetic mean of samples.
            bad_index: Index of first non-finite sample, or -1 if all are finite.
    """
    num_samples = samples.shape[0]
    total_sum = _sum(samples)
    # Check if sum is non-finite: indicates non-finite sample or float overflow
    if not math.isfinite(total_sum):
        for sample_idx in range(num_samples):
            if not math.isfinite(samples[sample_idx]):
                # Return index of first non-finite sample found
                return 0.0, 0.0, 0.0, sample_idx
        # All samples are finite, but direct sum overflowed
        min_val, max_val = _min_max(samples)
        return min_val, max_val, xm.mean_by_division(samples, num_samples), -1
    min_val, max_val = _min_max(samples)
    return min_val, max_val, total_sum / num_samples, -1


@njit(nogil=True, cache=True)
def exponent(range_span: float, bits: int) -> int:
    """Determines the finest quantization exponent e yielding at most `bits` bits.

    Guarantees that round(rng * 2^-e) < 2^bits.

    Args:
        range_span: Span of finite values (hi - lo).
        bits: Maximum bit width permitted across the range.

    Returns:
        Integer exponent e. Returns 0 for non-positive ranges (constant blocks).
    """
    if not range_span > 0:
        return 0
    # Decompose range into mantissa in [0.5, 1.0) and exponent k
    _, scale_exp = math.frexp(range_span)
    target_exp = scale_exp - bits
    # Bump exponent if upper boundary rounds up to 2^bits
    if math.floor(math.ldexp(range_span, -target_exp) + 0.5) >= (1 << bits):
        target_exp += 1
    return target_exp


@njit(nogil=True, cache=True)
def _unsnapped_exponent(lower_bound: float, upper_bound: float, bits: int) -> int:
    """The finest exponent e with round((hi - lo) * 2^-e) < 2^bits, clamped to [E_MIN, E_MAX]
    (the grid anchored at lo); range_exponent starts from it."""
    range_span = upper_bound - lower_bound
    # Direct range calculation for ordinary ranges
    if range_span < xm.WIDE_RANGE:
        return max(exponent(range_span, bits), E_MIN)
    # Use exact half-range calculation when hi - lo could overflow float64
    return min(exponent(xm.half_range(lower_bound, upper_bound), bits) + 1, E_MAX)


@njit(inline="always")
def grid_index(value: float, quant_exp: int) -> float:
    """rint(value * 2^-e): the index of the nearest point of the absolute grid of step 2^e,
    ties to even, as a float holding an integer.

    value * 2^-e is exact (a power-of-two scale) wherever it matters: it can't overflow for a
    sample of a block whose step is 2^e (the step is at least about 2^-69 of its magnitude),
    and when it underflows, the index is 0 either way.
    """
    if quant_exp >= xm.E_TINY:
        return np.rint(value * math.ldexp(1.0, -quant_exp))
    # 2^-e overflows float64: scale in two exact steps
    return np.rint(value * math.ldexp(1.0, -xm.E_TINY) * math.ldexp(1.0, xm.E_TINY - quant_exp))


@njit(nogil=True, cache=True)
def range_exponent(lower_bound: float, upper_bound: float, bits: int) -> int:
    """The finest exponent e whose snapped grid fits: rint(hi * 2^-e) - rint(lo * 2^-e) <= L,
    with L = 2^bits for bits < 16 and 65,535 for 16 (q is stored mod 2^16).

    The grid points nearest lo and hi can sit up to half a step outside [lo, hi], so the
    snapped grid can need one more level than the range alone: the rule accounts for it. It
    is within one level of _unsnapped_exponent and never coarser, except at bits = 16 when
    the snap would reach q = 65,536 (one level coarser then).

    Args:
        lower_bound: Lower bound.
        upper_bound: Upper bound.
        bits: The bit budget B.

    Returns:
        Quantization exponent in [E_MIN, E_MAX] (0 for an empty range).
    """
    if not upper_bound > lower_bound:
        return 0
    max_index = (1 << bits) if bits < 16 else (1 << 16) - 1
    # One level coarser than the unsnapped rule always fits: halving the range in steps
    # leaves room for the snap. Finer levels fit at most one level below it (only for bits < 16).
    quant_exp = min(_unsnapped_exponent(lower_bound, upper_bound, bits) + 1, E_MAX)
    while quant_exp > E_MIN and (
        grid_index(upper_bound, quant_exp - 1) - grid_index(lower_bound, quant_exp - 1) <= max_index
    ):
        quant_exp -= 1
    return quant_exp


@njit(nogil=True, cache=True)
def snap_anchor(lower_bound: float, upper_bound: float, quant_exp: int) -> tuple[float, float]:
    """The anchor of a power-of-two block: the grid point nearest its minimum.

    Args:
        lower_bound: Block minimum lo.
        upper_bound: Block maximum hi.
        quant_exp: Quantization exponent e.

    Returns:
        A tuple of (anchor, K): anchor = K * 2^e with K = rint(lo * 2^-e). The exceptions are
        a constant block (e is only a placeholder there) and a snapped anchor that isn't
        finite (near +-DBL_MAX): (lo, NaN), quantized relative to lo instead.
    """
    if not upper_bound > lower_bound:
        return lower_bound, math.nan
    base_index = grid_index(lower_bound, quant_exp)
    anchor = math.ldexp(base_index, quant_exp)
    if not math.isfinite(anchor):
        return lower_bound, math.nan
    return anchor + 0.0, base_index


@njit(nogil=True, cache=True)
def range_scale(lower_bound: float, upper_bound: float) -> int:
    """Computes binary scale exponent k such that (hi - lo) in [2^(k-1), 2^k).

    Args:
        lower_bound: Lower bound.
        upper_bound: Upper bound (hi > lo).

    Returns:
        Integer scale exponent k.
    """
    range_span = upper_bound - lower_bound
    if range_span < xm.WIDE_RANGE:
        # Extract binary exponent from range directly
        return math.frexp(range_span)[1]
    # Halve range to avoid overflow, then add 1 back
    return math.frexp(xm.half_range(lower_bound, upper_bound))[1] + 1


@njit(nogil=True, cache=True)
def plan_step(
    lower_bound: float,
    upper_bound: float,
    noise_sigma: float,
    noise_rho: float,
    noise_factor: float,
    min_bits: int,
    max_bits: int,
) -> int:
    """Plans quantization exponent before decimal detection and target budget.

    Starts at finest step (max_bits), coarsens based on noise floor if gate
    conditions pass, and clamps to coarsest allowable step (min_bits).

    Args:
        lower_bound: Lower bound of block.
        upper_bound: Upper bound of block.
        noise_sigma: Estimated white-noise standard deviation.
        noise_rho: Lag-1 autocorrelation of second differences.
        noise_factor: Noise floor multiplier f (0.0 to disable).
        min_bits: Hard minimum precision bits.
        max_bits: Hard maximum precision bits.

    Returns:
        Integer quantization exponent e.
    """
    if not upper_bound > lower_bound:
        return 0
    # Start with finest allowed step at max_bits
    planned_exp = range_exponent(lower_bound, upper_bound, max_bits)
    # Check noise gate: white noise gives rho < -0.6
    if noise_factor > 0 and noise_rho < NOISE_RHO and noise_sigma > 0:
        # Coarsen step up to f * sigma
        planned_exp = max(planned_exp, math.floor(math.log2(noise_factor * noise_sigma)))
    # Clamp to coarsest allowed step at min_bits
    return min(planned_exp, range_exponent(lower_bound, upper_bound, min_bits))


@njit(nogil=True, cache=True)
def detect_decimal(
    samples: np.ndarray,
    lower_bound: float,
    upper_bound: float,
    pow2_step: float,
    out_quantized: np.ndarray,
) -> tuple[int, np.int64]:
    """Finds coarsest decimal step 10^p > ps fitting all samples within tolerance.

    Candidate steps must be coarser than power-of-two step ps and satisfy
    |x[i] - round(x[i] * 10^-p) * 10^p| <= ps / 4 for all samples.

    Args:
        samples: 1D float64 array of samples.
        lower_bound: Minimum sample value.
        upper_bound: Maximum sample value.
        pow2_step: Power-of-two quantization step size 2^e.
        out_quantized: Output 1D int32 array populated with quantized grid offsets on success.

    Returns:
        A tuple of (p, K0):
            p: Decimal exponent in [-22, 22], or NO_DECIMAL if no grid matches.
            K0: Grid index of lower_bound, floor(lo * 10^-p + 0.5) (0 if no grid matches).
    """
    # Tolerance is 1/4 of the power-of-two step size
    tolerance = 0.25 * pow2_step
    max_abs_val = max(abs(lower_bound), abs(upper_bound))
    # Coarsest plausible decimal exponent spanning the block's range
    range_span = upper_bound - lower_bound
    decimal_exp = min(math.floor(math.log10(range_span)), P_MAX)
    while decimal_exp >= P_MIN and 10.0 ** decimal_exp > pow2_step:
        # Scale factor 10^-p (exact integer representation for p <= 0)
        inv_decimal_step = 10.0 ** -decimal_exp
        # Stop if the scaled magnitude reaches 2^52 (finer candidates only get larger)
        if not max_abs_val * inv_decimal_step < FLOAT64_EXACT_INT_LIMIT:
            break
        scaled_tolerance = tolerance * inv_decimal_step
        # Base grid point corresponding to lo
        base_grid_offset = np.int64(math.floor(lower_bound * inv_decimal_step + 0.5))
        grid_matches = True
        for sample_idx in range(samples.shape[0]):
            scaled_sample = samples[sample_idx] * inv_decimal_step
            rounded_grid_point = math.floor(scaled_sample + 0.5)
            # Stop scan on first off-grid sample
            if abs(scaled_sample - rounded_grid_point) > scaled_tolerance:
                grid_matches = False
                break
            # Quantize on decimal grid
            out_quantized[sample_idx] = np.int32(np.int64(rounded_grid_point) - base_grid_offset)
        if grid_matches:
            return decimal_exp, base_grid_offset
        decimal_exp -= 1
    return NO_DECIMAL, np.int64(0)


@njit(nogil=True, cache=True)
def quantize(
    samples: np.ndarray, lower_bound: float, quant_exp: int, out_quantized: np.ndarray
) -> None:
    """Quantizes samples onto the power-of-two grid anchored at lo: floor((x - lo) * 2^-e + 0.5).

    Used only for the blocks snap_anchor exempts; others use quantize_grid.

    Args:
        samples: 1D float64 array of samples.
        lower_bound: Minimum value in the block.
        quant_exp: Quantization exponent.
        out_quantized: Output 1D int32 array receiving quantized integers.
    """
    # Huge exponent: quantize at half-scale to prevent (x - lo) overflow. (The exempt blocks are
    # constant, with e = 0, or near +-DBL_MAX: never a subnormal step.)
    if quant_exp >= xm.E_HUGE:
        xm.quantize_huge(samples, lower_bound, quant_exp, out_quantized)
    else:
        # Standard power-of-two step inverse: 2^-e
        step_inv = math.ldexp(1.0, -quant_exp)
        for sample_idx in range(samples.shape[0]):
            # Round to nearest integer with ties rounding toward +infinity
            diff = samples[sample_idx] - lower_bound
            out_quantized[sample_idx] = np.int32(math.floor(diff * step_inv + 0.5))


@njit(nogil=True, cache=True)
def quantize_grid(samples: np.ndarray, quant_exp: int, base_index: float, out_quantized: np.ndarray) -> None:
    """Quantizes samples onto the absolute power-of-two grid: rint(x * 2^-e) - K.

    K = rint(lo * 2^-e) with the same rounding, so the minimum maps to 0 and every q >= 0.
    The difference of two nearby integers held in floats is exact.

    Args:
        samples: 1D float64 array of samples.
        quant_exp: Quantization exponent e.
        base_index: K, the anchor's grid index (snap_anchor).
        out_quantized: Output 1D int32 array receiving quantized integers.
    """
    if quant_exp >= xm.E_TINY:
        step_inv = math.ldexp(1.0, -quant_exp)
        for sample_idx in range(samples.shape[0]):
            out_quantized[sample_idx] = np.int32(np.rint(samples[sample_idx] * step_inv) - base_index)
    else:
        for sample_idx in range(samples.shape[0]):
            out_quantized[sample_idx] = np.int32(grid_index(samples[sample_idx], quant_exp) - base_index)


@njit(nogil=True, cache=True)
def quantize_block(
    samples: np.ndarray, lower_bound: float, upper_bound: float, quant_exp: int, out_quantized: np.ndarray
) -> float:
    """Quantizes a block on its power-of-two grid and returns the block's anchor.

    The grid is snapped (snap_anchor) unless the block is an exception, which is quantized
    relative to its minimum.
    """
    anchor, base_index = snap_anchor(lower_bound, upper_bound, quant_exp)
    if math.isnan(base_index):
        quantize(samples, lower_bound, quant_exp, out_quantized)
    else:
        quantize_grid(samples, quant_exp, base_index, out_quantized)
    return anchor


@njit(nogil=True, cache=True)
def pick_order(quantized_samples: np.ndarray, num_samples: int, orders_mask: int) -> int:
    """Selects predictor order (0-3) minimizing difference variance on q[:m].

    Args:
        quantized_samples: 1D int32 array of quantized integers.
        num_samples: Number of samples evaluated (min(PICK_LEN, len(q))).
        orders_mask: Bitmask of allowable orders (bit k enables order k).

    Returns:
        Selected difference order (0, 1, 2, or 3). Ties favor lower orders.
    """
    sum_q = sum_sq_q = sum_sq_diff1 = sum_sq_diff2 = sum_sq_diff3 = np.int64(0)
    for sample_idx in range(3):
        val = np.int64(quantized_samples[sample_idx])
        sum_q += val
        sum_sq_q += val * val
    for sample_idx in range(1, 3):
        diff = np.int64(quantized_samples[sample_idx]) - quantized_samples[sample_idx - 1]
        sum_sq_diff1 += diff * diff
    diff_order2 = (
        np.int64(quantized_samples[2])
        - 2 * np.int64(quantized_samples[1])
        + quantized_samples[0]
    )
    sum_sq_diff2 += diff_order2 * diff_order2
    # Offset slices from index 0 avoid numba's negative-index checks, so the loop vectorizes
    slice_0 = quantized_samples[3:num_samples]
    slice_1 = quantized_samples[2:num_samples - 1]
    slice_2 = quantized_samples[1:num_samples - 2]
    slice_3 = quantized_samples[:num_samples - 3]
    for sample_idx in range(num_samples - 3):
        sample_a = slice_0[sample_idx]
        sample_b = slice_1[sample_idx]
        sample_c = slice_2[sample_idx]
        sample_d = slice_3[sample_idx]
        # Differences in int32 (|d3| < 2^19), widened to int64 for squaring
        diff1 = np.int32(sample_a - sample_b)
        diff2 = np.int32(sample_a - 2 * sample_b + sample_c)
        diff3 = np.int32(sample_a - 3 * sample_b + 3 * sample_c - sample_d)
        sum_q += sample_a
        sum_sq_q += np.int64(sample_a) * np.int64(sample_a)
        sum_sq_diff1 += np.int64(diff1) * np.int64(diff1)
        sum_sq_diff2 += np.int64(diff2) * np.int64(diff2)
        sum_sq_diff3 += np.int64(diff3) * np.int64(diff3)
    # Telescoping sums give exact mean differences in O(1)
    sum_diff1 = np.int64(quantized_samples[num_samples - 1]) - quantized_samples[0]
    sum_diff2 = (
        (np.int64(quantized_samples[num_samples - 1]) - quantized_samples[num_samples - 2])
        - (np.int64(quantized_samples[1]) - quantized_samples[0])
    )
    sum_diff3 = (
        (np.int64(quantized_samples[num_samples - 1]) - 2 * np.int64(quantized_samples[num_samples - 2]) + quantized_samples[num_samples - 3])
        - (np.int64(quantized_samples[2]) - 2 * np.int64(quantized_samples[1]) + quantized_samples[0])
    )
    best_order, min_variance = 0, 1e300
    candidates = (
        (0, sum_q, sum_sq_q),
        (1, sum_diff1, sum_sq_diff1),
        (2, sum_diff2, sum_sq_diff2),
        (3, sum_diff3, sum_sq_diff3),
    )
    for order_idx, sum_diff_val, sum_sq_diff_val in candidates:
        # Skip orders not permitted by caller's bitmask
        if not (orders_mask >> order_idx) & 1:
            continue
        valid_count = num_samples - order_idx
        # Variance formula: (m * sum(d^2) - sum(d)^2) / m^2
        variance = (valid_count * float(sum_sq_diff_val) - float(sum_diff_val) ** 2) / (valid_count * valid_count)
        if variance < min_variance:
            best_order, min_variance = order_idx, variance
    return best_order


@njit(inline="always")
def _wrap16(diff_val: int) -> np.int16:
    """Wraps integer difference into signed int16 range [-32768, 32767] mod 2^16.

    Args:
        diff_val: Integer difference value.

    Returns:
        Modularly wrapped signed 16-bit integer.
    """
    # Modular reduction: ((diff_val + INT16_WRAP_OFFSET) mod 65536) - INT16_WRAP_OFFSET
    return np.int16(((diff_val + INT16_WRAP_OFFSET) & UINT16_MASK) - INT16_WRAP_OFFSET)


@njit(nogil=True, cache=True)
def residual(quantized_samples: np.ndarray, order: int, out_residuals: np.ndarray) -> None:
    """Computes order-k differences mod 2^16 of q with k prepended zeros.

    Args:
        quantized_samples: 1D int32 array of quantized integers.
        order: Difference order (0, 1, 2, or 3).
        out_residuals: Output 1D int16 array receiving wrapped differences.
    """
    num_samples = quantized_samples.shape[0]
    out_residuals[0] = _wrap16(quantized_samples[0])
    if order == 0:
        for sample_idx in range(num_samples):
            out_residuals[sample_idx] = _wrap16(quantized_samples[sample_idx])
    elif order == 1:
        # First difference: q[i] - q[i-1]
        slice_curr = quantized_samples[1:]
        slice_prev = quantized_samples[:num_samples - 1]
        out_slice = out_residuals[1:]
        for sample_idx in range(num_samples - 1):
            out_slice[sample_idx] = _wrap16(slice_curr[sample_idx] - slice_prev[sample_idx])
    elif order == 2:
        # Second difference with prepended zeros: v[1] = q[1] - 2*q[0]
        out_residuals[1] = _wrap16(quantized_samples[1] - 2 * quantized_samples[0])
        slice_curr = quantized_samples[2:]
        slice_prev1 = quantized_samples[1:num_samples - 1]
        slice_prev2 = quantized_samples[:num_samples - 2]
        out_slice = out_residuals[2:]
        for sample_idx in range(num_samples - 2):
            out_slice[sample_idx] = _wrap16(
                slice_curr[sample_idx] - 2 * slice_prev1[sample_idx] + slice_prev2[sample_idx]
            )
    else:
        # Third difference with prepended zeros
        out_residuals[1] = _wrap16(quantized_samples[1] - 3 * quantized_samples[0])
        out_residuals[2] = _wrap16(quantized_samples[2] - 3 * quantized_samples[1] + 3 * quantized_samples[0])
        slice_curr = quantized_samples[3:]
        slice_prev1 = quantized_samples[2:num_samples - 1]
        slice_prev2 = quantized_samples[1:num_samples - 2]
        slice_prev3 = quantized_samples[:num_samples - 3]
        out_slice = out_residuals[3:]
        for sample_idx in range(num_samples - 3):
            out_slice[sample_idx] = _wrap16(
                slice_curr[sample_idx]
                - 3 * slice_prev1[sample_idx]
                + 3 * slice_prev2[sample_idx]
                - slice_prev3[sample_idx]
            )


@njit(nogil=True, cache=True, fastmath=True)
def estimate_bits(residuals: np.ndarray) -> float:
    """Estimates compressed bits per sample via class entropy of residuals.

    Args:
        residuals: 1D int16 array of differences.

    Returns:
        Estimated bits per sample.
    """
    num_samples = residuals.shape[0]
    # Interleaved sub-histograms, so consecutive increments of the same bin don't wait on each other
    bit_length_hist = np.zeros(NUM_HIST_CHAINS * NUM_BIT_LENGTH_BINS, np.int64)
    for sample_idx in range(num_samples):
        residual_val = int(residuals[sample_idx])
        # Zigzag residual to uint16
        zigzag_val = np.uint32(((residual_val << 1) ^ (residual_val >> 15)) & UINT16_MASK)
        # Compute bit length L (0 for 0, up to 16)
        bit_length = 32 - leading_zeros(zigzag_val)
        bit_length_hist[(sample_idx & (NUM_HIST_CHAINS - 1)) * NUM_BIT_LENGTH_BINS + bit_length] += 1
    total_entropy_bits = 0.0
    for length_bin in range(NUM_BIT_LENGTH_BINS):
        bin_count = (
            bit_length_hist[length_bin]
            + bit_length_hist[NUM_BIT_LENGTH_BINS + length_bin]
            + bit_length_hist[2 * NUM_BIT_LENGTH_BINS + length_bin]
            + bit_length_hist[3 * NUM_BIT_LENGTH_BINS + length_bin]
        )
        if bin_count > 0:
            prob = bin_count / num_samples
            # Entropy of bit length L plus bits below the leading 1
            total_entropy_bits += prob * (max(length_bin - 1, 0) - math.log2(prob))
    return total_entropy_bits



@njit(nogil=True, cache=True)
def encode_block(
    samples: np.ndarray,
    lower_bound: float,
    upper_bound: float,
    quant_exp: int,
    decimal: bool,
    orders_mask: int,
    pick_len: int,
    scratch_quantized: np.ndarray,
    out_residuals: np.ndarray,
) -> tuple[int, int, np.int64, float]:
    """Encodes one block: quantizes, picks predictor order, and computes residuals.

    Args:
        samples: 1D float64 array of samples.
        lower_bound: Minimum value.
        upper_bound: Maximum value.
        quant_exp: Quantization exponent.
        decimal: Whether decimal detection is enabled.
        orders_mask: Allowed predictor orders bitmask.
        pick_len: Sample count evaluated for order selection.
        scratch_quantized: Preallocated 1D int32 scratch buffer for quantized values.
        out_residuals: Output 1D int16 array receiving residuals.

    Returns:
        A tuple of (flags, param, decimal_base, anchor): decimal_base is the grid index K0
        of lower_bound on a detected decimal grid (the block's anchor), 0 on the power-of-two
        grid; anchor is the power-of-two grid's anchor (unused on a decimal grid).
    """
    flags = 0
    anchor = lower_bound
    param_val = quant_exp
    decimal_base = np.int64(0)
    decimal_detected = False
    # Attempt decimal grid detection if enabled and range is within 2^1023
    range_span = upper_bound - lower_bound
    if decimal and upper_bound > lower_bound and range_span < xm.WIDE_RANGE:
        decimal_exp, decimal_base = detect_decimal(
            samples, lower_bound, upper_bound, math.ldexp(1.0, quant_exp), scratch_quantized
        )
        if decimal_exp != NO_DECIMAL:
            decimal_detected = True
            param_val = decimal_exp
            flags |= BLOCK_FLAG_DECIMAL
    # Fall back to power-of-two quantization if decimal grid not detected
    if not decimal_detected:
        anchor = quantize_block(samples, lower_bound, upper_bound, quant_exp, scratch_quantized)
    # Select difference order minimizing residual variance
    selected_order = pick_order(scratch_quantized, pick_len, orders_mask)
    # Compute modular differences mod 2^16
    residual(scratch_quantized, selected_order, out_residuals)
    return flags | selected_order, param_val, decimal_base, anchor


@njit(inline="always")
def _store_anchor(
    out_value_anchors: np.ndarray,
    anchor_floats: np.ndarray,
    block_idx: int,
    flags: int,
    anchor: float,
    decimal_base: np.int64,
) -> None:
    """Writes block b's anchor: the decimal grid index K0, or the float64 bits of the
    power-of-two anchor.

    Args:
        out_value_anchors: Output 1D int64 array of length N receiving anchors.
        anchor_floats: out_value_anchors viewed as float64, made once per unit rather
            than per block.
        block_idx: Zero-based block index.
        flags: Block header byte; BLOCK_FLAG_DECIMAL selects the decimal anchor.
        anchor: The power-of-two grid's anchor, stored as float64 bits.
        decimal_base: Grid index K0 of lo on a detected decimal grid.
    """
    if flags & BLOCK_FLAG_DECIMAL:
        out_value_anchors[block_idx] = decimal_base
    else:
        anchor_floats[block_idx] = anchor


@njit(nogil=True, cache=True)
def allocate_target(
    estimated_bits: np.ndarray,
    current_exponents: np.ndarray,
    coarsest_exponents: np.ndarray,
    block_weights: np.ndarray,
    target_bits: float,
    out_bit_reductions: np.ndarray,
) -> bool:
    """Allocates bit reductions across blocks to meet a per-unit bit target.

    Args:
        estimated_bits: 1D float64 array of per-block estimated bits per sample.
        current_exponents: 1D int64 array of current per-block quantization exponents.
        coarsest_exponents: 1D int64 array of coarsest allowable exponents per block.
        block_weights: 1D int64 array of each block's sample count in the budget (0 for a
            block outside it, which is never coarsened).
        target_bits: Target bits per sample across the budgeted samples.
        out_bit_reductions: Output 1D int64 array receiving bit coarsening increments.

    Returns:
        True if any block is allocated a coarsening step, False otherwise.
    """
    num_blocks = estimated_bits.shape[0]
    total_bits = 0.0
    total_weight = 0
    for block_idx in range(num_blocks):
        weight = block_weights[block_idx]
        # Maximum bits block b can give up without exceeding coarsest_exponents
        headroom = coarsest_exponents[block_idx] - current_exponents[block_idx]
        max_give = math.floor(max(estimated_bits[block_idx] - 1.0, 0.0) + 0.5)
        out_bit_reductions[block_idx] = min(max_give, headroom) if weight > 0 else 0
        total_bits += estimated_bits[block_idx] * weight
        total_weight += weight
    excess_bits = total_bits - target_bits * total_weight
    # Return immediately if already within target budget
    if excess_bits <= 0:
        return False
    uniform_cap = 1
    # Search for smallest uniform cap that covers the excess
    while uniform_cap < 16:
        saved_bits = 0
        for block_idx in range(num_blocks):
            saved_bits += min(uniform_cap, out_bit_reductions[block_idx]) * block_weights[block_idx]
        if saved_bits >= excess_bits:
            break
        uniform_cap += 1
    any_coarsened = False
    for block_idx in range(num_blocks):
        out_bit_reductions[block_idx] = min(uniform_cap, out_bit_reductions[block_idx])
        any_coarsened = any_coarsened or out_bit_reductions[block_idx] > 0
    return any_coarsened


@njit(nogil=True, cache=True)
def _apply_target_reallocation(
    samples: np.ndarray,
    sample_offsets: np.ndarray,
    bit_reductions: np.ndarray,
    block_exponents: np.ndarray,
    base_lower_bounds: np.ndarray,
    base_upper_bounds: np.ndarray,
    estimated_bits_per_block: np.ndarray,
    decimal: bool,
    orders_mask: int,
    pick_len: int,
    scratch_quantized: np.ndarray,
    scratch_held: np.ndarray,
    scratch_residuals: np.ndarray,
    out_block_flags: np.ndarray,
    out_grid_params: np.ndarray,
    out_value_anchors: np.ndarray,
    out_residuals: np.ndarray,
    out_codes: np.ndarray,
) -> None:
    """Re-encodes blocks selected by target allocation and rolls back if bits do not drop."""
    num_blocks = sample_offsets.shape[0] - 1
    anchor_floats = out_value_anchors.view(np.float64)
    for block_idx in range(num_blocks):
        if bit_reductions[block_idx] > 0:
            first_sample = sample_offsets[block_idx]
            block_len = sample_offsets[block_idx + 1] - first_sample
            block_residuals = out_residuals[first_sample:first_sample + block_len]
            prev_flags = out_block_flags[block_idx]
            prev_param = out_grid_params[block_idx]
            prev_anchor = out_value_anchors[block_idx]
            backup_residuals = scratch_residuals[:block_len]
            backup_residuals[:] = block_residuals
            block_samples = samples[first_sample:first_sample + block_len]
            has_nonfinite = prev_flags & BLOCK_FLAG_NONFINITE
            if has_nonfinite:
                fill_nonfinite(block_samples, scratch_held[:block_len], out_codes[first_sample:first_sample + block_len])
                block_samples = scratch_held[:block_len]
            target_exp = min(block_exponents[block_idx] + bit_reductions[block_idx], E_MAX)
            out_block_flags[block_idx], out_grid_params[block_idx], decimal_base, anchor = encode_block(
                block_samples,
                base_lower_bounds[block_idx],
                base_upper_bounds[block_idx],
                target_exp,
                decimal,
                orders_mask,
                min(pick_len, block_len),
                scratch_quantized[:block_len],
                block_residuals,
            )
            _store_anchor(
                out_value_anchors,
                anchor_floats,
                block_idx,
                out_block_flags[block_idx],
                anchor,
                decimal_base,
            )
            if has_nonfinite:
                out_block_flags[block_idx] |= BLOCK_FLAG_NONFINITE
            # Roll back if re-quantization failed to reduce estimated bits
            if estimate_bits(block_residuals) >= estimated_bits_per_block[block_idx]:
                out_block_flags[block_idx] = prev_flags
                out_grid_params[block_idx] = prev_param
                out_value_anchors[block_idx] = prev_anchor
                block_residuals[:] = backup_residuals


@njit(nogil=True, cache=True)
def encode_unit(
    samples: np.ndarray,
    sample_offsets: np.ndarray,
    min_bits: int,
    max_bits: int,
    orders_mask: int,
    noise_factor: float,
    target_bits: float,
    decimal: bool,
    pick_len: int,
    out_block_flags: np.ndarray,
    out_grid_params: np.ndarray,
    out_value_anchors: np.ndarray,
    out_residuals: np.ndarray,
    out_codes: np.ndarray,
    out_block_minima: np.ndarray,
    out_block_maxima: np.ndarray,
    out_block_means: np.ndarray,
) -> None:
    """Encodes all blocks of a unit and computes index column metadata.

    Blocks of at most SHORT_BLOCK_LEN samples skip the analysis: they take the finest step
    (a decimal grid if one is detected, else the power-of-two grid) and order 0, and stay
    outside the target. An empty block gets
    zero flags, grid parameter and anchor, and NaN statistics.

    Args:
        samples: 1D float64 array of every sample of the unit.
        sample_offsets: 1D int64 array of the blocks' sample offsets (N + 1).
        min_bits: Hard lower bound on quantization bits.
        max_bits: Hard upper bound on quantization bits.
        orders_mask: Bitmask of allowable predictor difference orders.
        noise_factor: Noise floor multiplier f (0.0 if disabled).
        target_bits: Target bits per sample (0.0 if disabled).
        decimal: Whether decimal detection is active.
        pick_len: Sample count used for predictor order selection.
        out_block_flags: Output 1D uint8 array of length N receiving header bytes.
        out_grid_params: Output 1D int64 array of length N receiving parameters.
        out_value_anchors: Output 1D int64 array of length N receiving anchors (float64 bits of
            the power-of-two anchor, the grid point nearest the block minimum, or the decimal
            grid index of the minimum).
        out_residuals: Output 1D int16 array of every sample receiving residuals.
        out_codes: Output 1D uint8 array of every sample receiving sample codes (only
            flagged blocks' are written).
        out_block_minima: Output 1D float64 array of length N receiving block minima.
        out_block_maxima: Output 1D float64 array of length N receiving block maxima.
        out_block_means: Output 1D float64 array of length N receiving block means.
    """
    num_blocks = sample_offsets.shape[0] - 1
    max_len = 0
    for block_idx in range(num_blocks):
        max_len = max(max_len, sample_offsets[block_idx + 1] - sample_offsets[block_idx])
    scratch_quantized = np.empty(max_len, np.int32)
    scratch_held = np.empty(max_len)
    scratch_noise_diffs = np.empty(max_len)
    scratch_noise_weights = np.empty(max_len)
    use_target = target_bits > 0
    block_exponents = np.empty(num_blocks, np.int64)
    coarsest_exponents = np.empty(num_blocks, np.int64)
    block_weights = np.zeros(num_blocks, np.int64)
    base_lower_bounds = np.empty(num_blocks)
    base_upper_bounds = np.empty(num_blocks)
    estimated_bits_per_block = np.zeros(num_blocks)
    anchor_floats = out_value_anchors.view(np.float64)
    for block_idx in range(num_blocks):
        first_sample = sample_offsets[block_idx]
        block_len = sample_offsets[block_idx + 1] - first_sample
        block_exponents[block_idx] = coarsest_exponents[block_idx] = 0
        if block_len == 0:
            out_block_flags[block_idx] = 0
            out_grid_params[block_idx] = 0
            out_value_anchors[block_idx] = 0
            out_block_minima[block_idx] = out_block_maxima[block_idx] = out_block_means[block_idx] = math.nan
            continue
        block_samples = samples[first_sample:first_sample + block_len]
        block_codes = out_codes[first_sample:first_sample + block_len]
        block_residuals = out_residuals[first_sample:first_sample + block_len]
        block_quantized = scratch_quantized[:block_len]
        block_min, block_max, block_mean, bad_sample_idx = block_stats(block_samples)
        has_nonfinite = bad_sample_idx >= 0
        finite_count = block_len
        if has_nonfinite:
            # Impute non-finite values with sample-and-held finite values
            block_min, block_max, block_mean, finite_count = fill_nonfinite(
                block_samples, scratch_held[:block_len], block_codes
            )
            block_samples = scratch_held[:block_len]
        # Normalize -0.0 to +0.0 to match decoder behavior
        block_min = block_min + 0.0
        block_max = block_max + 0.0
        block_mean = block_mean + 0.0
        out_block_minima[block_idx] = block_min
        out_block_maxima[block_idx] = block_max
        out_block_means[block_idx] = block_mean
        # All non-finite block defaults to zero range
        if has_nonfinite and finite_count == 0:
            block_min = block_max = 0.0
        if block_len <= SHORT_BLOCK_LEN:
            # Too short to analyze: the finest step, stored without differences. Decimal
            # detection needs no analysis, so decimal data stays exact
            fine_exp = range_exponent(block_min, block_max, max_bits)
            decimal_exp, decimal_base = NO_DECIMAL, np.int64(0)
            if decimal and block_max > block_min and block_max - block_min < xm.WIDE_RANGE:
                decimal_exp, decimal_base = detect_decimal(
                    block_samples, block_min, block_max, math.ldexp(1.0, fine_exp), block_quantized
                )
            if decimal_exp != NO_DECIMAL:
                out_block_flags[block_idx] = BLOCK_FLAG_DECIMAL
                out_grid_params[block_idx] = decimal_exp
                out_value_anchors[block_idx] = decimal_base
            else:
                out_block_flags[block_idx] = 0
                out_grid_params[block_idx] = fine_exp
                anchor_floats[block_idx] = quantize_block(block_samples, block_min, block_max, fine_exp, block_quantized)
            residual(block_quantized, 0, block_residuals)
            if has_nonfinite:
                out_block_flags[block_idx] |= BLOCK_FLAG_NONFINITE
            continue
        base_lower_bounds[block_idx] = block_min
        base_upper_bounds[block_idx] = block_max
        noise_sigma = noise_rho = 0.0
        if noise_factor > 0 and block_max > block_min and block_len >= NOISE_MIN_LEN:
            if not has_nonfinite:
                noise_sigma, noise_rho = noise(block_samples, range_scale(block_min, block_max))
            else:
                # Estimate noise floor using only finite sample triplets
                noise_sigma, noise_rho = noise_finite(
                    block_samples,
                    block_codes,
                    range_scale(block_min, block_max),
                    block_len // 2,
                    scratch_noise_diffs,
                    scratch_noise_weights,
                )
        # Plan quantization exponent
        planned_exp = plan_step(
            block_min, block_max, noise_sigma, noise_rho, noise_factor, min_bits, max_bits
        )
        block_exponents[block_idx] = planned_exp
        out_block_flags[block_idx], out_grid_params[block_idx], decimal_base, anchor = encode_block(
            block_samples,
            block_min,
            block_max,
            planned_exp,
            decimal,
            orders_mask,
            min(pick_len, block_len),
            block_quantized,
            block_residuals,
        )
        _store_anchor(out_value_anchors, anchor_floats, block_idx, out_block_flags[block_idx], anchor, decimal_base)
        if has_nonfinite:
            out_block_flags[block_idx] |= BLOCK_FLAG_NONFINITE
        if use_target:
            if out_block_flags[block_idx] & BLOCK_FLAG_DECIMAL:
                # Decimal grid is already 10^p; track equivalent binary exponent
                block_exponents[block_idx] = math.floor(math.log2(10.0 ** out_grid_params[block_idx]))
            coarsest_exponents[block_idx] = max(
                range_exponent(block_min, block_max, min_bits), block_exponents[block_idx]
            )
            estimated_bits_per_block[block_idx] = estimate_bits(block_residuals)
            block_weights[block_idx] = block_len
    # Apply soft per-unit bit target if enabled
    if use_target:
        bit_reductions = np.zeros(num_blocks, np.int64)
        if allocate_target(
            estimated_bits_per_block,
            block_exponents,
            coarsest_exponents,
            block_weights,
            target_bits,
            bit_reductions,
        ):
            _apply_target_reallocation(
                samples,
                sample_offsets,
                bit_reductions,
                block_exponents,
                base_lower_bounds,
                base_upper_bounds,
                estimated_bits_per_block,
                decimal,
                orders_mask,
                pick_len,
                scratch_quantized,
                scratch_held,
                np.empty(max_len, np.int16),
                out_block_flags,
                out_grid_params,
                out_value_anchors,
                out_residuals,
                out_codes,
            )
