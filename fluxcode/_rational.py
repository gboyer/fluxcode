# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Scaled quantization grids: index GCD reduction and rational step search.

A scaled block reconstructs values as ``(K0 + N * q) * B^p / D`` with base ``B`` in ``{2, 10}``.
The encoder discovers ``N`` and ``D`` independently per block via two mechanisms:

- **Index GCD** (``index_gcd``, ``divide_indices_in_place``): divides out any common factor
  ``N > 1`` already present in a block's binary or decimal grid indices (``D = 1``).
- **Rational step search** (``fit_rational_step``, ``detect_rational_grid``): estimates a
  coarser step ``S`` via approximate Euclidean GCD on sample differences, refines ``S`` by
  least squares, finds the simplest rational ``N / D`` within ``S``'s uncertainty via
  continued fractions, and verifies every sample against the candidate grid.
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
"""Leading sample count used for GCD fast-path checks, step seeding, and spot-check windows."""

MAX_SPLIT: int = 16
"""Maximum divisor tried when splitting the prefix seed step (handles shared prefix factors)."""

MAX_DECIMAL_FOLD: int = 4
"""Largest positive decimal exponent ``p`` folded into ``N`` (``10^5 > 65535``)."""

MAX_Q: int = 0xFFFF
"""Maximum unsigned 16-bit grid offset ``q`` representable in a block."""


@njit(inline="always")
def _gcd(a: int | np.integer, b: int | np.integer) -> int | np.integer:
    """Computes the greatest common divisor of two non-negative integers via Euclid's algorithm.

    Args:
        a: First non-negative integer.
        b: Second non-negative integer.

    Returns:
        Greatest common divisor of ``a`` and ``b``.
    """
    while b:
        a, b = b, a % b
    return a


@njit(nogil=True, cache=True)
def index_gcd(quantized: np.ndarray) -> int | np.integer:
    """Computes the greatest common divisor of a block's non-negative grid indices.

    Args:
        quantized: 1D int32 array of non-negative grid offsets ``q``.

    Returns:
        The GCD across all entries, or ``0`` if every entry is 0.
    """
    # Scan indices and exit as soon as the running GCD reaches 1. If the GCD after PREFIX_LEN
    # samples is a power of two, verify the remainder with a branch-free bitwise OR so LLVM
    # vectorizes the common case (exact power-of-two data) without per-element modulo.
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
def divide_indices_in_place(quantized: np.ndarray, divisor: int | np.integer) -> None:
    """Divides every grid index in ``quantized`` by an exact positive ``divisor`` in place.

    Args:
        quantized: 1D int32 array of grid indices, each an exact multiple of ``divisor``.
        divisor: Positive integer divisor.
    """
    # Use a right shift when divisor is a power of two so the loop vectorizes cleanly.
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
def simplest_fraction_between(lower: float, upper: float, max_den: int) -> tuple[int, int]:
    """Finds the rational number with the smallest denominator in ``[lower, upper]``.

    Args:
        lower: Positive lower bound of the target interval.
        upper: Upper bound of the target interval (``lower <= upper``).
        max_den: Maximum allowed denominator.

    Returns:
        Tuple ``(scale_num, scale_den)`` in lowest terms with ``scale_den <= max_den``,
        or ``(0, 0)`` if no fraction in the interval has denominator at most ``max_den``.
    """
    # Standard continued-fraction interval narrowing: if an integer falls in [lower, upper],
    # pick the smallest such integer as the final partial quotient; otherwise strip the shared
    # integer part and invert the remaining fractional interval.
    num_prev, den_prev, scale_num, scale_den = 0, 1, 1, 0
    for _ in range(64):
        whole = math.floor(lower)
        if whole == lower or whole + 1 <= upper:
            last = int(whole) if whole == lower else int(whole) + 1
            result_num, result_den = last * scale_num + num_prev, last * scale_den + den_prev
            if result_den > max_den:
                return 0, 0
            return result_num, result_den
        num_prev, den_prev, scale_num, scale_den = (
            scale_num,
            scale_den,
            int(whole) * scale_num + num_prev,
            int(whole) * scale_den + den_prev,
        )
        if scale_den > max_den:
            return 0, 0
        lower, upper = 1.0 / (upper - whole), 1.0 / (lower - whole)
    return 0, 0


@njit(nogil=True, cache=True)
def try_rational_grid(
    samples: np.ndarray,
    lower_bound: float,
    decimal: bool,
    grid_exp: int,
    scale_num: int,
    scale_den: int,
    tolerance: float,
    out_quantized: np.ndarray,
) -> tuple[bool, np.int64]:
    """Attempts to quantize ``samples`` onto the scaled grid ``(scale_num / scale_den) * B^grid_exp``.

    Args:
        samples: 1D float64 array of block samples.
        lower_bound: Minimum sample value in the block.
        decimal: True for base 10 (``grid_exp <= 0``), False for base 2.
        grid_exp: Decimal exponent ``p`` or binary exponent ``e``.
        scale_num: Numerator ``N`` (step in fine grid points).
        scale_den: Denominator ``D``.
        tolerance: Maximum allowed distance from each sample to its grid point.
        out_quantized: Output 1D int32 array receiving ``q = (k - K0) / scale_num`` on success.

    Returns:
        Tuple ``(fits, anchor_index)`` where ``fits`` is True iff every sample lies within
        ``tolerance`` of a valid grid point with ``q`` in ``[0, MAX_Q]`` and ``|K0| < INDEX_LIMIT``.
    """
    # Map each sample to its fine-grid index k = round(x * D / B^grid_exp). For decimal grids,
    # D * 10^-p is exact (guaranteed by scale_valid); for binary grids, multiply by 2^-e first
    # because D * 2^-e can overflow float64 when the fine indices themselves stay small.
    first, second = (scale_den * 10.0**-grid_exp, 1.0) if decimal else (math.ldexp(1.0, -grid_exp), float(scale_den))
    scaled_tolerance = tolerance * first * second
    base_scaled = lower_bound * first * second
    # Guard in float64 before int64 conversion so infinite or huge indices cannot overflow.
    if not abs(base_scaled) < INDEX_LIMIT:
        return False, np.int64(0)
    anchor_index = np.int64(math.floor(base_scaled + 0.5))
    for sample_idx in range(samples.shape[0]):
        scaled = samples[sample_idx] * first * second
        if not abs(scaled) < INDEX_LIMIT:
            return False, np.int64(0)
        index = math.floor(scaled + 0.5)
        if abs(scaled - index) > scaled_tolerance:
            return False, np.int64(0)
        offset = np.int64(index) - anchor_index
        if offset % scale_num:
            return False, np.int64(0)
        quantized = offset // scale_num
        if quantized > MAX_Q:
            return False, np.int64(0)
        out_quantized[sample_idx] = np.int32(quantized)
    return True, anchor_index


@njit(inline="always")
def _track(
    samples: np.ndarray,
    first: int,
    stop: int,
    step: float,
    error: float,
    out_index: np.ndarray,
) -> tuple[bool, float, float, float, float, float]:
    """Tracks cumulative integer step counts from consecutive differences over ``samples[first:stop]``.

    Args:
        samples: 1D float64 array of block samples.
        first: Inclusive start sample index (assigned step count 0).
        stop: Exclusive end sample index.
        step: Candidate grid step size.
        error: Maximum allowed deviation of each consecutive difference from a multiple of ``step``.
        out_index: Output 1D float64 array populated with step counts ``k`` at ``[first:stop]``.

    Returns:
        Tuple ``(on_grid, sum_k, sum_kk, sum_u, sum_ku, largest_abs_k)`` where ``u = x - x[first]``
        and the sums provide the sufficient statistics for ordinary least-squares step refinement.
    """
    # Track integer steps incrementally from adjacent differences so small step errors do not
    # accumulate into a wrong rounded index before least-squares refinement.
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
    samples: np.ndarray,
    first: int,
    stop: int,
    index: np.ndarray,
    intercept: float,
    step: float,
    limit: float,
) -> float:
    """Computes the maximum absolute grid residual over ``samples[first:stop]`` with early exit.

    Args:
        samples: 1D float64 array of block samples.
        first: Inclusive start sample index.
        stop: Exclusive end sample index.
        index: 1D float64 array of tracked step counts ``k``.
        intercept: Fitted offset relative to ``samples[first]``.
        step: Fitted step size.
        limit: Maximum allowable residual magnitude.

    Returns:
        Maximum ``|x[i] - x[first] - intercept - index[i] * step|``, or ``inf`` as soon as any
        sample exceeds ``limit``.
    """
    largest = 0.0
    for sample_idx in range(first, stop):
        residual = abs(samples[sample_idx] - samples[first] - intercept - index[sample_idx] * step)
        if residual > limit:
            return math.inf
        largest = max(largest, residual)
    return largest


@njit(inline="always")
def _approx_gcd(larger: float, smaller: float, error: float, floor_step: float) -> float:
    """Computes an approximate Euclidean GCD of two positive floats down to ``error``.

    Args:
        larger: First positive float.
        smaller: Second positive float.
        error: Remainder cutoff below which Euclid terminates.
        floor_step: Minimum acceptable GCD; any remainder in ``(error, floor_step]`` aborts.

    Returns:
        Approximate common step above ``floor_step``, or ``0.0`` if none exists.
    """
    if smaller > larger:
        larger, smaller = smaller, larger
    while smaller > error:
        if not smaller > floor_step:
            return 0.0
        remainder = abs(larger - math.floor(larger / smaller + 0.5) * smaller)
        # Guard against float64 rounding above 2^53 where remainder could exceed 0.5 * smaller and cycle.
        if not remainder <= 0.5 * smaller:
            return 0.0
        larger, smaller = smaller, remainder
    return larger


@njit(nogil=True, cache=True)
def fit_rational_step(samples: np.ndarray, pow2_step: float, scratch_index: np.ndarray) -> tuple[float, float, float]:
    """Finds the coarsest step ``S > pow2_step`` fitting all samples within ``pow2_step / 4`` of ``a + k * S``.

    Args:
        samples: 1D float64 array of block samples.
        pow2_step: Baseline power-of-two step (candidate ``S`` must be strictly coarser).
        scratch_index: Preallocated 1D float64 scratch buffer of at least ``len(samples)``.

    Returns:
        Tuple ``(fitted_step, largest_residual, largest_abs_index)``, or ``(0.0, 0.0, 0.0)`` if no
        coarser step fits the block.
    """
    # 1. Seed S via approximate GCD of consecutive differences over the prefix (or whole block
    #    if the prefix is constant), aborting immediately if S drops to pow2_step.
    # 2. Refine the seed on the prefix by least squares through x[0]; drift here rules out any grid.
    # 3. Try splits seed / 1..MAX_SPLIT (in case the prefix differences shared a common factor),
    #    spot-checking end and middle windows before fitting the full block by least squares.
    num_samples = samples.shape[0]
    tolerance = pow2_step / 4
    # Two samples within tol of the grid differ by at most 2 * tol from a multiple of S.
    error = 2 * tolerance
    seed = inv_seed = 0.0
    for sample_idx in range(1, num_samples):
        diff = abs(samples[sample_idx] - samples[sample_idx - 1])
        # Skip differences that are already multiples of the current seed.
        if diff > error and abs(diff - math.floor(diff * inv_seed + 0.5) * seed) > error:
            seed = diff if seed == 0.0 else _approx_gcd(seed, diff, error, pow2_step)
            if not seed > pow2_step:
                return 0.0, 0.0, 0.0
            inv_seed = 1.0 / seed
        if sample_idx >= PREFIX_LEN - 1 and seed > 0.0:
            break
    if seed == 0.0:
        return 0.0, 0.0, 0.0
    # Refine the seed on the prefix and reject drifting blocks early.
    prefix = min(PREFIX_LEN, num_samples)
    on_grid, _, sum_kk, _, sum_ku, _ = _track(samples, 0, prefix, seed, error, scratch_index)
    if not on_grid:
        return 0.0, 0.0, 0.0
    if sum_kk > 0:
        seed = sum_ku / sum_kk
    if _largest_residual(samples, 0, prefix, scratch_index, 0.0, seed, error) == math.inf:
        return 0.0, 0.0, 0.0
    # Test candidate splits against tail and middle windows before a full-block least-squares fit.
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
def detect_rational_grid(
    samples: np.ndarray,
    lower_bound: float,
    upper_bound: float,
    pow2_step: float,
    out_quantized: np.ndarray,
    scratch_index: np.ndarray,
) -> tuple[bool, bool, int, int, int, np.int64]:
    """Searches for a scaled rational grid ``(N / D) * B^p`` coarser than ``pow2_step``.

    Args:
        samples: 1D float64 array of non-constant block samples.
        lower_bound: Minimum sample value in the block.
        upper_bound: Maximum sample value in the block.
        pow2_step: Baseline power-of-two step ``2^e``.
        out_quantized: Output 1D int32 array receiving grid offsets ``q`` on success.
        scratch_index: Preallocated 1D float64 scratch buffer of at least ``len(samples)``.

    Returns:
        Tuple ``(found, decimal, grid_exp, scale_num, scale_den, anchor_index)``. When ``found``
        is False, all remaining fields are zero/False.
    """
    # Fit an unconstrained step S and bound its uncertainty from the largest residual and float64
    # ULP. Try base 10 first, then base 2: find the simplest fraction N / D inside S's uncertainty
    # interval, fold small positive decimal powers into N, and verify every sample on the grid.
    step, largest_residual, largest_index = fit_rational_step(samples, pow2_step, scratch_index)
    if step == 0.0:
        return False, False, 0, 0, 0, np.int64(0)
    ulp = max(abs(lower_bound), abs(upper_bound)) * 2.0**-52
    uncertainty = 2 * (largest_residual + ulp) / max(largest_index, 1.0)
    for base in (10, 2):
        decimal = base == 10
        grid_exp = math.floor(math.log(step) / math.log(base))
        if grid_exp < (P_MIN if decimal else SCALED_E_MIN) or grid_exp > (
            MAX_DECIMAL_FOLD if decimal else SCALED_E_MAX
        ):
            continue
        unit = float(base) ** grid_exp
        # Correct log rounding when step sits just below an exact power of base.
        if unit > step:
            grid_exp -= 1
            unit = float(base) ** grid_exp
        scale_num, scale_den = simplest_fraction_between(
            (step - uncertainty) / unit, (step + uncertainty) / unit, SCALE_MASK
        )
        if scale_num == 0:
            continue
        # Scaled decimal grids require grid_exp <= 0; fold p in 1..MAX_DECIMAL_FOLD into scale_num.
        if decimal and grid_exp > SCALED_P_MAX:
            scale_num *= 10**grid_exp
            grid_exp = SCALED_P_MAX
        if not scale_valid(decimal, grid_exp, scale_num, scale_den):
            continue
        fits, anchor_index = try_rational_grid(
            samples, lower_bound, decimal, grid_exp, scale_num, scale_den, pow2_step / 4, out_quantized
        )
        if fits:
            return True, decimal, grid_exp, scale_num, scale_den, anchor_index
    return False, False, 0, 0, 0, np.int64(0)
