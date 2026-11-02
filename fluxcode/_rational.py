# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Scaled grids: the index gcd and the rational step search (ENCODER.md §5.4).

A scaled block's values are (K0 + N q) * B^p / D, B = 10 or 2 (SPEC.md §5). The encoder finds N
and D two ways:

- the index gcd: N is the gcd of the indices a block already has on its decimal or power-of-two
  grid (D = 1). No search; it stops at the first index that brings the gcd to 1.
- the rational search, for blocks on neither grid: an approximate gcd of the values gives a step
  S, and N / D is the simplest fraction within S's uncertainty, checked exactly.
"""

import math

import numpy as np
from numba import njit

from ._format import P_MIN, SCALED_E_MAX, SCALED_E_MIN, SCALED_P_MAX

MAX_SCALE: int = 0xFFFF
"""Largest N and D: grid_scale holds each in 16 bits."""

PREFIX_LEN: int = 64
"""Samples the rational search seeds its step from and first tests it on."""

MAX_SPLIT: int = 16
"""The prefix's approximate gcd is tried as 1 to MAX_SPLIT steps (its differences may all share a
factor)."""

EXACT_INT_LIMIT: int = 1 << 53
"""Integers below this are exact doubles."""

INDEX_LIMIT: float = 2.0 ** 52
"""Bound on a scaled grid's index magnitudes (the format's anchor bound)."""

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

    Args:
        quantized: 1D int32 array of the block's indices q.

    Returns:
        The gcd (0 if every index is 0).
    """
    gcd: int | np.integer = np.int64(0)
    # gcd - 1 masks the remainder while gcd is a power of two (a mask, not a division)
    mask: int | np.integer = np.int64(-1)
    num_samples = quantized.shape[0]
    for sample_idx in range(num_samples):
        if sample_idx == PREFIX_LEN and mask >= 0:
            # Still a power of two: OR the rest in one branch-free (vectorized) pass
            bits = np.int64(0)
            for rest_idx in range(PREFIX_LEN, num_samples):
                bits |= quantized[rest_idx]
            if bits & mask == 0:
                return gcd
        index = np.int64(quantized[sample_idx])
        if mask >= 0:
            if index & mask == 0:
                continue
        elif gcd and index % gcd == 0:
            continue
        gcd = _gcd(gcd, index) if gcd else index
        if gcd == 1:
            return 1
        mask = gcd - 1 if gcd & (gcd - 1) == 0 else np.int64(-1)
    return gcd


@njit(nogil=True, cache=True)
def divide(quantized: np.ndarray, divisor: int | np.integer) -> None:
    """Divides every index by divisor, which divides them all, in place."""
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


@njit(inline="always")
def scale_fits(base: int, param: int, num: int, den: int) -> bool:
    """Whether a scaled grid's exponent, N and D are in the format's range (SPEC.md §6.4)."""
    if not (1 <= num <= MAX_SCALE and 1 <= den <= MAX_SCALE):
        return False
    if base == 10:
        return P_MIN <= param <= SCALED_P_MAX and bool(np.int64(5) ** -param < EXACT_INT_LIMIT // den)
    return SCALED_E_MIN <= param <= SCALED_E_MAX


@njit(nogil=True, cache=True)
def rational_grid(
    samples: np.ndarray, lower_bound: float, base: int, param: int, num: int, den: int, tolerance: float,
    out_quantized: np.ndarray,
) -> tuple[bool, np.int64]:
    """Quantizes a block on the scaled grid of step num / den * base^param, if every sample fits.

    The fine grid's index of each sample is k = round(x * den / base^param); every sample must be
    within tolerance of its point, and every k - K0 (K0 the minimum's) a multiple of num.

    Args:
        samples: 1D float64 array of the block's samples.
        lower_bound: The block's minimum.
        base: 10 or 2.
        param: The exponent p (base 10, p <= 0) or e (base 2).
        num: N.
        den: D.
        tolerance: How far a sample may be from its grid point.
        out_quantized: Output 1D int32 array receiving q = (k - K0) / num.

    Returns:
        (fits, K0).
    """
    if base == 10:
        scale = den * 10.0 ** -param
    else:
        scale = den * math.ldexp(1.0, -param)
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


WINDOW_FITS: int = 0
"""_window_check: every sample of the window fits."""

WINDOW_DRIFTS: int = 1
"""_window_check: every difference is a multiple of the step, but the positions drift away from
them (a smooth ramp): no finer step fits either."""

WINDOW_OFF_GRID: int = 2
"""_window_check: a difference isn't a multiple of the step: a finer one may fit."""


@njit(inline="always")
def _window_check(samples: np.ndarray, first: int, step: float, tolerance: float) -> int:
    """Whether PREFIX_LEN samples from first sit within 2 * tolerance of x[first] + k step, k
    tracked from consecutive differences (WINDOW_FITS, WINDOW_DRIFTS or WINDOW_OFF_GRID)."""
    inv_step = 1.0 / step
    index = 0.0
    for sample_idx in range(first + 1, min(first + PREFIX_LEN, samples.shape[0])):
        diff = samples[sample_idx] - samples[sample_idx - 1]
        steps = math.floor(diff * inv_step + 0.5)
        if abs(diff - steps * step) > 2 * tolerance:
            return WINDOW_OFF_GRID
        index += steps
        if abs(samples[sample_idx] - samples[first] - index * step) > 2 * tolerance:
            return WINDOW_DRIFTS
    return WINDOW_FITS


@njit(inline="always")
def _prefix_fits(samples: np.ndarray, step: float, tolerance: float) -> bool:
    """Whether the prefix sits within 2 * tolerance of x[0] + k step."""
    return _window_check(samples, 0, step, tolerance) == WINDOW_FITS


@njit(inline="always")
def _refine_on_prefix(samples: np.ndarray, step: float) -> float:
    """The step fitted (least squares through x[0]) to the prefix's k, tracked with step."""
    inv_step = 1.0 / step
    index = sum_kk = sum_ku = 0.0
    for sample_idx in range(1, min(PREFIX_LEN, samples.shape[0])):
        index += math.floor((samples[sample_idx] - samples[sample_idx - 1]) * inv_step + 0.5)
        sum_kk += index * index
        sum_ku += index * (samples[sample_idx] - samples[0])
    return sum_ku / sum_kk if sum_kk > 0 else step


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
def fit_step(samples: np.ndarray, pow2_step: float) -> tuple[float, float, float]:
    """The coarsest step S > pow2_step with every sample within pow2_step / 4 of a + k S.

    Seeds S from the approximate gcd of the differences above 2 tol in the prefix (or the block,
    if the prefix has none), stopping as soon as it reaches pow2_step (no coarser grid: a continuous
    block takes a few differences), tried as 1 to MAX_SPLIT steps. Each candidate's k is tracked from consecutive differences (small, so a
    slightly-off S doesn't drift), checked on the prefix, fitted (a and S by least squares) and
    verified on the whole block.

    Args:
        samples: 1D float64 array of the block's samples.
        pow2_step: The block's power-of-two step: S must be coarser.

    Returns:
        (S, the largest |x - a - k S|, the largest |k|), or S = 0 if none.
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
    # Every prefix difference is a multiple of the seed: a prefix that drifts from it drifts from
    # any finer step too
    seed = _refine_on_prefix(samples, seed)
    if not _prefix_fits(samples, seed, tolerance):
        return 0.0, 0.0, 0.0
    for split in range(1, MAX_SPLIT + 1):
        step = seed / split
        if not step > pow2_step:
            break
        # Spot checks before the full passes: windows at the end and the middle. A difference
        # off the step may fit a finer one; drift (a smooth ramp) fits none
        end_check = _window_check(samples, max(num_samples - PREFIX_LEN, 0), step, tolerance)
        middle_check = end_check if end_check else _window_check(samples, num_samples // 2, step, tolerance)
        if middle_check == WINDOW_DRIFTS:
            return 0.0, 0.0, 0.0
        if middle_check == WINDOW_OFF_GRID:
            continue
        # Least squares of u = x - x[0] on k, with an offset
        inv_step = 1.0 / step
        index = sum_k = sum_kk = sum_u = sum_ku = 0.0
        for sample_idx in range(num_samples):
            if sample_idx:
                index += math.floor((samples[sample_idx] - samples[sample_idx - 1]) * inv_step + 0.5)
            offset = samples[sample_idx] - samples[0]
            sum_k += index
            sum_kk += index * index
            sum_u += offset
            sum_ku += index * offset
        det = num_samples * sum_kk - sum_k * sum_k
        if not det > 0:
            continue
        fitted = (num_samples * sum_ku - sum_k * sum_u) / det
        intercept = (sum_u - fitted * sum_k) / num_samples
        index = largest_residual = largest_index = 0.0
        fits = True
        for sample_idx in range(num_samples):
            if sample_idx:
                index += math.floor((samples[sample_idx] - samples[sample_idx - 1]) * inv_step + 0.5)
            residual = abs(samples[sample_idx] - samples[0] - intercept - index * fitted)
            if residual > tolerance:
                fits = False
                break
            largest_residual = max(largest_residual, residual)
            largest_index = max(largest_index, abs(index))
        if fits and fitted > pow2_step:
            return fitted, largest_residual, largest_index
    return 0.0, 0.0, 0.0


@njit(nogil=True, cache=True)
def detect_rational(
    samples: np.ndarray,
    lower_bound: float,
    upper_bound: float,
    pow2_step: float,
    out_quantized: np.ndarray,
    hint: np.ndarray,
    failed_step: np.ndarray,
) -> tuple[bool, int, int, int, int, np.int64]:
    """Finds a scaled grid N / D * B^p coarser than pow2_step holding every sample within a quarter of it.

    1. The previous block's grid (hint), if it holds this block with indices of gcd 1 (no coarser
       multiple of it would): what the search below finds, without the search.
    2. A block whose prefix fits the step the previous block fitted but couldn't rationalize
       (failed_step) is skipped.
    3. fit_step gives S, and S's uncertainty (twice the largest residual plus an ulp, over the
       largest index); for B = 10, then 2, with B^p <= S < B^(p+1), N / D is the simplest fraction
       within that of S / B^p. A decimal p > 0 is folded into N. The grid must hold the block
       exactly (rational_grid).

    Args:
        samples: 1D float64 array of the block's samples (non-constant).
        lower_bound: The block's minimum.
        upper_bound: The block's maximum.
        pow2_step: The block's power-of-two step 2^e.
        out_quantized: Output 1D int32 array receiving the indices q on success.
        hint: In-out 1D int64 array (base, p, N, D) of the last grid found (N = 0: none).
        failed_step: In-out 1D float64 array of one: the last fitted step that didn't rationalize.

    Returns:
        (found, base, p, N, D, K0).
    """
    tolerance = pow2_step / 4
    base, param, num, den = int(hint[0]), int(hint[1]), int(hint[2]), int(hint[3])
    if num and num / den * float(base) ** param > pow2_step:
        fits, base_index = rational_grid(samples, lower_bound, base, param, num, den, tolerance, out_quantized)
        if fits and index_gcd(out_quantized) == 1:
            return True, base, param, num, den, base_index
    if failed_step[0] > pow2_step and _prefix_fits(samples, failed_step[0], tolerance):
        return False, 0, 0, 0, 0, np.int64(0)
    step, largest_residual, largest_index = fit_step(samples, pow2_step)
    if step == 0.0:
        return False, 0, 0, 0, 0, np.int64(0)
    ulp = max(abs(lower_bound), abs(upper_bound)) * 2.0 ** -52
    uncertainty = 2 * (largest_residual + ulp) / max(largest_index, 1.0)
    for base in (10, 2):
        param = math.floor(math.log(step) / math.log(base))
        # Out of the scaled grids' range (a decimal p > 0 folds into N below): also keeps the
        # power from underflowing
        if param < (P_MIN if base == 10 else SCALED_E_MIN) or (base == 2 and param > SCALED_E_MAX):
            continue
        unit = float(base) ** param
        if unit > step:  # log rounding at an exact power
            param -= 1
            unit = float(base) ** param
        num, den = simplest_between((step - uncertainty) / unit, (step + uncertainty) / unit, MAX_SCALE)
        if num == 0:
            continue
        if base == 10 and param > SCALED_P_MAX:
            num *= 10 ** param
            param = SCALED_P_MAX
        if not scale_fits(base, param, num, den):
            continue
        fits, base_index = rational_grid(samples, lower_bound, base, param, num, den, tolerance, out_quantized)
        if fits:
            hint[0], hint[1], hint[2], hint[3] = base, param, num, den
            return True, base, param, num, den, base_index
    failed_step[0] = step
    return False, 0, 0, 0, 0, np.int64(0)
