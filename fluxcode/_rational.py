# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Scaled grids: the index gcd and the rational step search (ENCODER.md §5.4).

A scaled block's values are (K0 + N q) * B^p / D, B = 10 or 2 (SPEC.md §5). The encoder finds N
and D two ways:

- the index gcd: N is the gcd of the indices a block already has on its decimal or power-of-two
  grid (D = 1). No search; it stops at the first index that brings the gcd to 1.
- the rational search, for blocks on neither grid: an approximate gcd of the values gives a step
  S, and N / D is the simplest fraction within S's uncertainty, checked exactly.

Each block is searched on its own: nothing carries over from the block before.
"""

import math

import numpy as np
from numba import njit

from ._format import (
    INDEX_LIMIT,
    P_MIN,
    SCALE_MASK,
    SCALED_E_MAX,
    SCALED_E_MIN,
    SCALED_P_MAX,
    scale_valid,
)

PREFIX_LEN: int = 64
"""Samples the rational search seeds its step from, and the length of its spot-check windows (and
of the index gcd's first pass)."""

MAX_SPLIT: int = 16
"""The prefix's approximate gcd is tried as 1 to MAX_SPLIT steps (its differences may all share a
factor)."""

MAX_DECIMAL_FOLD: int = 4
"""Largest decimal p > 0 folded into N (10^5 > 65,535)."""

MAX_Q: int = 0xFFFF
"""Largest grid index q of a block (stored mod 2^16)."""


@njit(inline="always")
def _gcd(a: int | np.integer, b: int | np.integer) -> int | np.integer:
    """gcd of two non-negative integers (Euclid)."""
    while b:
        a, b = b, a % b
    return a


@njit(nogil=True, cache=True)
def index_gcd(quantized: np.ndarray) -> int | np.integer:
    """The gcd of a block's grid indices (all >= 0), stopping as soon as it is 1.

    A gcd that is still a power of two after the first PREFIX_LEN indices (an exact coarser
    power-of-two grid: integers, decoded data) is checked against the rest with one branch-free
    OR, which vectorizes: 0.16 against 0.65 us per 1000 samples with a remainder each.

    Args:
        quantized: 1D int32 array of the block's indices q.

    Returns:
        The gcd (0 if every index is 0).
    """
    gcd: int | np.integer = np.int64(0)
    num_samples = quantized.shape[0]
    for sample_idx in range(num_samples):
        if sample_idx == PREFIX_LEN and gcd & (gcd - 1) == 0:
            bits = np.int64(0)
            for rest_idx in range(PREFIX_LEN, num_samples):
                bits |= quantized[rest_idx]
            if bits & (gcd - 1) == 0:
                return gcd
        index = np.int64(quantized[sample_idx])
        if gcd and index % gcd == 0:
            continue
        gcd = _gcd(gcd, index) if gcd else index
        if gcd == 1:
            return 1
    return gcd


@njit(nogil=True, cache=True)
def divide(quantized: np.ndarray, divisor: int | np.integer) -> None:
    """Divides every index by divisor, which divides them all, in place (a power of two by a shift:
    0.17 against 0.64 us per 1000 samples)."""
    if divisor & (divisor - 1) == 0:
        shift = 0
        while (np.int64(1) << shift) < divisor:
            shift += 1
        for sample_idx in range(quantized.shape[0]):
            quantized[sample_idx] >>= shift
        return
    for sample_idx in range(quantized.shape[0]):
        quantized[sample_idx] //= divisor


@njit(nogil=True, cache=True)
def simplest_between(lower: float, upper: float, max_den: int) -> tuple[int, int]:
    """The fraction with the smallest denominator in [lower, upper] (0 < lower <= upper).

    Continued fractions: if an integer lies in the interval, the smallest one is the answer;
    otherwise both ends share their integer part a, and the answer is a + 1 / (the simplest
    fraction between the reciprocals of their fractional parts). The convergents accumulate
    the partial quotients.

    Args:
        lower: Lower end.
        upper: Upper end.
        max_den: Largest denominator accepted.

    Returns:
        (numerator, denominator), or (0, 0) if the simplest fraction's denominator exceeds max_den.
    """
    num_prev, den_prev, num, den = 0, 1, 1, 0
    for _ in range(64):
        whole = math.floor(lower)
        if whole == lower or whole + 1 <= upper:
            last = int(whole) if whole == lower else int(whole) + 1
            result_num, result_den = last * num + num_prev, last * den + den_prev
            if result_den > max_den:
                return 0, 0
            return result_num, result_den
        num_prev, den_prev, num, den = num, den, int(whole) * num + num_prev, int(whole) * den + den_prev
        if den > max_den:
            return 0, 0
        lower, upper = 1.0 / (upper - whole), 1.0 / (lower - whole)
    return 0, 0


@njit(nogil=True, cache=True)
def rational_grid(
    samples: np.ndarray, lower_bound: float, decimal: bool, param: int, num: int, den: int, tolerance: float,
    out_quantized: np.ndarray,
) -> tuple[bool, np.int64]:
    """Quantizes a block on the scaled grid of step num / den * B^param, if every sample fits.

    The fine grid's index of each sample is k = round(x * den / B^param); every sample must be
    within tolerance of its point, and every k - K0 (K0 the minimum's) a multiple of num.

    Args:
        samples: 1D float64 array of the block's samples.
        lower_bound: The block's minimum.
        decimal: Whether B is 10 (else 2).
        param: The exponent p (B = 10, p <= 0) or e (B = 2).
        num: N.
        den: D.
        tolerance: How far a sample may be from its grid point.
        out_quantized: Output 1D int32 array receiving q = (k - K0) / num.

    Returns:
        (fits, K0).
    """
    scale = den * 10.0 ** -param if decimal else den * math.ldexp(1.0, -param)
    base_index = math.floor(lower_bound * scale + 0.5)
    scaled_tolerance = tolerance * scale
    if not abs(base_index) < INDEX_LIMIT:
        return False, np.int64(0)
    base_int = np.int64(base_index)
    for sample_idx in range(samples.shape[0]):
        scaled = samples[sample_idx] * scale
        index = math.floor(scaled + 0.5)
        # Stop at the first sample off the grid
        if abs(scaled - index) > scaled_tolerance or not abs(index) < INDEX_LIMIT:
            return False, np.int64(0)
        offset = np.int64(index) - base_int
        if offset % num:
            return False, np.int64(0)
        quantized = offset // num
        if quantized > MAX_Q:
            return False, np.int64(0)
        out_quantized[sample_idx] = np.int32(quantized)
    return True, base_int


@njit(inline="always")
def _track(
    samples: np.ndarray, first: int, stop: int, step: float, error: float, out_index: np.ndarray,
) -> tuple[bool, float, float, float, float, float]:
    """Each sample's step count k from x[first], tracked from consecutive differences (each small,
    so a slightly-off step doesn't drift), with the least-squares sums of u = x - x[first] on k and
    the largest |k|.

    Args:
        samples: 1D float64 array of the block's samples.
        first: The first sample (k = 0).
        stop: One past the last sample.
        step: The step.
        error: How far a difference may be from a multiple of step.
        out_index: Output 1D float64 array receiving k at each sample's index.

    Returns:
        (every difference within error of a multiple, sum k, sum k^2, sum u, sum k u, largest |k|).
    """
    inv_step = 1.0 / step
    index = sum_k = sum_kk = sum_u = sum_ku = largest = 0.0
    out_index[first] = 0.0
    for sample_idx in range(first + 1, stop):
        diff = samples[sample_idx] - samples[sample_idx - 1]
        steps = math.floor(diff * inv_step + 0.5)
        if abs(diff - steps * step) > error:
            return False, 0.0, 0.0, 0.0, 0.0, 0.0
        index += steps
        largest = max(largest, abs(index))
        offset = samples[sample_idx] - samples[first]
        out_index[sample_idx] = index
        sum_k += index
        sum_kk += index * index
        sum_u += offset
        sum_ku += index * offset
    return True, sum_k, sum_kk, sum_u, sum_ku, largest


@njit(inline="always")
def _largest_residual(
    samples: np.ndarray, first: int, stop: int, index: np.ndarray, intercept: float, step: float, limit: float,
) -> float:
    """The largest |x - x[first] - intercept - k step|, or inf as soon as one exceeds limit."""
    largest = 0.0
    for sample_idx in range(first, stop):
        residual = abs(samples[sample_idx] - samples[first] - intercept - index[sample_idx] * step)
        if residual > limit:
            return math.inf
        largest = max(largest, residual)
    return largest


@njit(inline="always")
def _approx_gcd(larger: float, smaller: float, error: float, floor_step: float) -> float:
    """Euclid on reals: the remainder to the nearest multiple, until it is within error of 0.

    With a step S > floor_step dividing both, every remainder is within error of 0 or of a multiple
    of S: one between error and floor_step proves there's none, and ends the search (0 is returned).
    """
    if smaller > larger:
        larger, smaller = smaller, larger
    while smaller > error:
        if not smaller > floor_step:
            return 0.0
        remainder = abs(larger - math.floor(larger / smaller + 0.5) * smaller)
        larger, smaller = smaller, remainder
    return larger


@njit(nogil=True, cache=True)
def fit_step(samples: np.ndarray, pow2_step: float, scratch_index: np.ndarray) -> tuple[float, float, float]:
    """The coarsest step S > pow2_step with every sample within pow2_step / 4 of a + k S.

    1. Seed: the approximate gcd of the prefix's differences above 2 tol (the block's, if the
       prefix has none), stopping as soon as it reaches pow2_step: no coarser grid (a continuous
       block takes a few differences).
    2. The seed, fitted to the prefix (least squares through x[0]); a prefix that drifts from it
       drifts from any finer step too.
    3. For S0 = seed / 1 ... MAX_SPLIT (the prefix's differences may share a factor): windows at
       the end and the middle (a difference off S0 tries the next split; drift, a smooth ramp,
       ends the search), then the whole block: a and S by least squares, every sample within tol.

    Args:
        samples: 1D float64 array of the block's samples.
        pow2_step: The block's power-of-two step: S must be coarser.
        scratch_index: 1D float64 scratch array of at least len(samples).

    Returns:
        (S, the largest |x - x[0] - a - k S|, the largest |k|), or S = 0 if none.
    """
    num_samples = samples.shape[0]
    tolerance = pow2_step / 4
    # Two samples within tol of the grid: a difference is within 2 tol of a multiple of S
    error = 2 * tolerance
    seed = inv_seed = 0.0
    for sample_idx in range(1, num_samples):
        diff = abs(samples[sample_idx] - samples[sample_idx - 1])
        # A multiple of the seed leaves it as it is: Euclid only for the others
        if diff > error and abs(diff - math.floor(diff * inv_seed + 0.5) * seed) > error:
            seed = diff if seed == 0.0 else _approx_gcd(seed, diff, error, pow2_step)
            if not seed > pow2_step:
                return 0.0, 0.0, 0.0
            inv_seed = 1.0 / seed
        if sample_idx >= PREFIX_LEN - 1 and seed > 0.0:
            break  # the prefix has a step: seed from it
    if seed == 0.0:
        return 0.0, 0.0, 0.0
    prefix = min(PREFIX_LEN, num_samples)
    _, _, sum_kk, _, sum_ku, _ = _track(samples, 0, prefix, seed, error, scratch_index)
    if sum_kk > 0:
        seed = sum_ku / sum_kk
    if _largest_residual(samples, 0, prefix, scratch_index, 0.0, seed, error) == math.inf:
        return 0.0, 0.0, 0.0
    windows = (max(num_samples - PREFIX_LEN, 0), num_samples // 2)
    for split in range(1, MAX_SPLIT + 1):
        step = seed / split
        if not step > pow2_step:
            break
        off_grid = False
        for first in windows:
            stop = min(first + PREFIX_LEN, num_samples)
            if not _track(samples, first, stop, step, error, scratch_index)[0]:
                off_grid = True
                break
            if _largest_residual(samples, first, stop, scratch_index, 0.0, step, error) == math.inf:
                return 0.0, 0.0, 0.0
        if off_grid:
            continue
        on_grid, sum_k, sum_kk, sum_u, sum_ku, largest_index = _track(
            samples, 0, num_samples, step, error, scratch_index
        )
        det = num_samples * sum_kk - sum_k * sum_k
        if not (on_grid and det > 0):
            continue
        fitted = (num_samples * sum_ku - sum_k * sum_u) / det
        intercept = (sum_u - fitted * sum_k) / num_samples
        largest = _largest_residual(samples, 0, num_samples, scratch_index, intercept, fitted, tolerance)
        if largest < math.inf and fitted > pow2_step:
            return fitted, largest, largest_index
    return 0.0, 0.0, 0.0


@njit(nogil=True, cache=True)
def detect_rational(
    samples: np.ndarray,
    lower_bound: float,
    upper_bound: float,
    pow2_step: float,
    out_quantized: np.ndarray,
    scratch_index: np.ndarray,
) -> tuple[bool, bool, int, int, int, np.int64]:
    """Finds a scaled grid N / D * B^p coarser than pow2_step holding every sample within a quarter of it.

    fit_step gives S, and S's uncertainty (twice the largest residual plus an ulp, over the largest
    index); for B = 10, then 2, with B^p <= S < B^(p+1), N / D is the simplest fraction within that
    of S / B^p. A decimal p from 1 to 4 is folded into N. The grid must hold the block exactly
    (rational_grid).

    Args:
        samples: 1D float64 array of the block's samples (non-constant).
        lower_bound: The block's minimum.
        upper_bound: The block's maximum.
        pow2_step: The block's power-of-two step 2^e.
        out_quantized: Output 1D int32 array receiving the indices q on success.
        scratch_index: 1D float64 scratch array of at least len(samples).

    Returns:
        (found, decimal, p, N, D, K0).
    """
    step, largest_residual, largest_index = fit_step(samples, pow2_step, scratch_index)
    if step == 0.0:
        return False, False, 0, 0, 0, np.int64(0)
    ulp = max(abs(lower_bound), abs(upper_bound)) * 2.0 ** -52
    uncertainty = 2 * (largest_residual + ulp) / max(largest_index, 1.0)
    for base in (10, 2):
        decimal = base == 10
        param = math.floor(math.log(step) / math.log(base))
        # Out of the scaled grids' range: also keeps the power from underflowing
        if param < (P_MIN if decimal else SCALED_E_MIN) or param > (MAX_DECIMAL_FOLD if decimal else SCALED_E_MAX):
            continue
        unit = float(base) ** param
        if unit > step:  # log rounding at an exact power
            param -= 1
            unit = float(base) ** param
        num, den = simplest_between((step - uncertainty) / unit, (step + uncertainty) / unit, SCALE_MASK)
        if num == 0:
            continue
        if decimal and param > SCALED_P_MAX:
            num *= 10 ** param
            param = SCALED_P_MAX
        if not scale_valid(decimal, param, num, den):
            continue
        fits, base_index = rational_grid(
            samples, lower_bound, decimal, param, num, den, pow2_step / 4, out_quantized
        )
        if fits:
            return True, decimal, param, num, den, base_index
    return False, False, 0, 0, 0, np.int64(0)
