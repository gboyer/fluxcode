# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Quantization grids and step planning for float64 block encoding.

Provides power-of-two exponent sizing, snapped anchors, decimal grid detection,
power-of-two quantization, and step planning helpers shared by `_encoder`.
"""

import math

import numpy as np
from numba import njit

from . import _extreme_magnitudes as xm
from ._format import (
    BLOCK_FLAG_DECIMAL,
    BLOCK_FLAG_SCALED,
    E_MAX,
    E_MIN,
    INDEX_LIMIT,
    P_MAX,
    P_MIN,
    SCALE_MASK,
    SCALE_SHIFT,
)

NO_DECIMAL: int = 99
"""Sentinel indicating no decimal grid coarser than the baseline step was found."""

GRIDS_BINARY: int = 0
"""Params.grids "binary": power-of-two grids only."""

GRIDS_DECIMAL: int = 1
"""Params.grids "decimal": decimal grids too (§5.3)."""

GRIDS_RATIONAL: int = 2
"""Params.grids "rational": scaled grids too (§5.4)."""

GRIDS_LEVELS: dict[str, int] = {"binary": GRIDS_BINARY, "decimal": GRIDS_DECIMAL, "rational": GRIDS_RATIONAL}
"""Params.grids to its level: each level adds grids to the one before."""

NOISE_RHO: float = -0.6
"""Autocorrelation threshold for noise gating (white noise gives rho = -2/3)."""

FLOAT64_EXACT_INT_LIMIT: float = 2.0 ** 52
"""Scaled-magnitude limit for decimal candidates: below 2^52 a double's spacing is at most
0.5, so floor(v + 0.5) and the tolerance check are meaningful."""

TRIM_STRIDE: int = 4
"""The trimmed mean of an order-0 block reads every TRIM_STRIDE-th index."""

TRIM_DEVIATIONS: float = 3.0
"""An order-0 block centers on the mean of its indices within this many mean absolute deviations
of their mean (center_anchor)."""

INT16_MAX: int = 32767
"""Largest signed 16-bit index (SPEC §4: q is read as int16)."""

INT16_MIN: int = -32768
"""Smallest signed 16-bit index."""


@njit(nogil=True, cache=True)
def exponent(range_span: float, bits: int) -> int:
    """Determines the finest quantization exponent yielding at most `bits` bits.

    Guarantees that round(range_span * 2^-exponent) < 2^bits.

    Args:
        range_span: Span of the finite values (maximum - minimum).
        bits: Maximum bit width permitted across the range.

    Returns:
        The exponent, or 0 for a non-positive range (a constant block).
    """
    if not range_span > 0:
        return 0
    # Decompose the range into a mantissa in [0.5, 1.0) and an exponent
    _, scale_exp = math.frexp(range_span)
    target_exp = scale_exp - bits
    # Bump exponent if upper boundary rounds up to 2^bits
    if math.floor(math.ldexp(range_span, -target_exp) + 0.5) >= (1 << bits):
        target_exp += 1
    return target_exp


@njit(nogil=True, cache=True)
def _unsnapped_exponent(lower_bound: float, upper_bound: float, bits: int) -> int:
    """Finds the finest quantization exponent fitting a range into `bits` bits.

    Computes the finest exponent with round((upper_bound - lower_bound) * 2^-exponent) <
    2^bits, clamped to [E_MIN, E_MAX].

    Args:
        lower_bound: Minimum sample value in the block.
        upper_bound: Maximum sample value in the block.
        bits: Bit budget for the quantized range.

    Returns:
        Quantization exponent clamped to [E_MIN, E_MAX].
    """
    range_span = upper_bound - lower_bound
    # Direct range calculation for ordinary ranges
    if range_span < xm.WIDE_RANGE:
        return max(exponent(range_span, bits), E_MIN)
    # Use the half-range when upper_bound - lower_bound could overflow float64
    return min(exponent(xm.half_range(lower_bound, upper_bound), bits) + 1, E_MAX)


@njit(inline="always")
def grid_index(value: float, quant_exp: int) -> float:
    """Computes the nearest grid point index on an absolute power-of-two grid.

    Calculates round-to-nearest-even of value * 2^-quant_exp as a float integer.

    Args:
        value: Float64 sample value.
        quant_exp: Quantization exponent.

    Returns:
        Nearest grid point index as a float holding an integer value.
    """
    if quant_exp >= xm.E_TINY:
        return np.rint(value * math.ldexp(1.0, -quant_exp))
    # 2^-e overflows float64: scale in two exact steps
    return np.rint(value * math.ldexp(1.0, -xm.E_TINY) * math.ldexp(1.0, xm.E_TINY - quant_exp))


@njit(nogil=True, cache=True)
def range_exponent(lower_bound: float, upper_bound: float, bits: int) -> int:
    """Finds the finest exponent whose snapped grid fits the range in `bits` bits.

    The grid fits if rint(upper_bound * 2^-exponent) - rint(lower_bound * 2^-exponent) <= L,
    with L = 2^bits for bits < 16 and 65,535 for bits = 16 (quantized values are stored
    mod 2^16).

    The grid points nearest the bounds can sit up to half a step outside the range, so the
    snapped grid can need one more level than the range alone: the rule accounts for it. It
    is within one level of _unsnapped_exponent and never coarser, except at bits = 16 when
    the snap would reach 65,536 (one level coarser then).

    Args:
        lower_bound: Minimum sample value in the block.
        upper_bound: Maximum sample value in the block.
        bits: Bit budget for the quantized range.

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
    """Finds the anchor of a power-of-two block: the grid point nearest its minimum.

    Args:
        lower_bound: Block minimum.
        upper_bound: Block maximum.
        quant_exp: Quantization exponent.

    Returns:
        A tuple of (anchor, base_index): anchor = base_index * 2^quant_exp, with base_index =
        rint(lower_bound * 2^-quant_exp). The exceptions are a constant block (quant_exp is
        only a placeholder there) and a snapped anchor that isn't finite (near +-DBL_MAX):
        (lower_bound, NaN), quantized relative to lower_bound instead.
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
    """Computes the binary scale exponent k with the range in [2^(k-1), 2^k).

    Args:
        lower_bound: Minimum sample value in the block.
        upper_bound: Maximum sample value in the block (above lower_bound).

    Returns:
        The scale exponent k.
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
        lower_bound: Minimum sample value in the block.
        upper_bound: Maximum sample value in the block.
        noise_sigma: Estimated white-noise standard deviation.
        noise_rho: Lag-1 autocorrelation of second differences.
        noise_factor: Noise floor multiplier (0.0 to disable).
        min_bits: Hard minimum precision bits.
        max_bits: Hard maximum precision bits.

    Returns:
        The quantization exponent.
    """
    if not upper_bound > lower_bound:
        return 0
    # Start with finest allowed step at max_bits
    planned_exp = range_exponent(lower_bound, upper_bound, max_bits)
    # Check noise gate: white noise gives rho = -2/3; the gate is rho < NOISE_RHO
    if noise_factor > 0 and noise_rho < NOISE_RHO and noise_sigma > 0:
        # Coarsen the step up to noise_factor * noise_sigma
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
    """Finds the coarsest decimal step that fits all samples within tolerance.

    A candidate step 10^p must be coarser than pow2_step and, for every sample,
    |sample - round(sample * 10^-p) * 10^p| <= pow2_step / 4.

    Args:
        samples: 1D float64 array of samples.
        lower_bound: Minimum sample value.
        upper_bound: Maximum sample value.
        pow2_step: Power-of-two quantization step size 2^e.
        out_quantized: Output 1D int32 array populated with quantized grid offsets on success.

    Returns:
        A tuple of (decimal_exp, base_grid_offset):
            decimal_exp: Decimal exponent p in [-22, 22], or NO_DECIMAL if no grid matches.
            base_grid_offset: Grid index of lower_bound, floor(lower_bound * 10^-p + 0.5)
                (0 if no grid matches).
    """
    # Tolerance is 1/4 of the power-of-two step size
    tolerance = 0.25 * pow2_step
    max_abs_val = max(abs(lower_bound), abs(upper_bound))
    # Coarsest plausible decimal exponent: samples within tolerance of two grid points are at most
    # range + 2 tolerance apart (a range of one step can come out a hair under it, 1e-6 as 9.99...e-7)
    range_span = upper_bound - lower_bound
    decimal_exp = min(math.floor(math.log10(range_span + 2.0 * tolerance)), P_MAX)
    while decimal_exp >= P_MIN and 10.0 ** decimal_exp > pow2_step:
        # Scale factor 10^-decimal_exp (an exact double for p <= 0)
        inv_decimal_step = 10.0 ** -decimal_exp
        # Stop if the scaled magnitude reaches 2^52 (finer candidates only get larger)
        if not max_abs_val * inv_decimal_step < FLOAT64_EXACT_INT_LIMIT:
            break
        scaled_tolerance = tolerance * inv_decimal_step
        # Grid point of the block minimum
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
def coarse_binary_exponent(samples: np.ndarray, quant_exp: int, decimal_step: float) -> int:
    """The coarsest power-of-two grid holding the block exactly, if it is coarser than decimal_step.

    Values on a coarse power-of-two grid are also on finer decimal ones (0.25 on 0.01), whose
    indices carry an odd factor (x 25) that bit planes can't absorb. On the coarsest power-of-two
    grid holding them they are compact and stored exactly.

    decimal_step is a detected 10^p > 2^quant_exp, so |sample| / 10^p < 2^52 for every sample:
    the scaled samples below are exact and their integer test is meaningful.

    Args:
        samples: 1D float64 array of samples of a non-constant block.
        quant_exp: The block's power-of-two exponent.
        decimal_step: The detected decimal step 10^p.

    Returns:
        The exponent of the coarsest power of two that every sample is a multiple of, if that is
        coarser than decimal_step; else quant_exp.
    """
    coarse_exp = quant_exp + 1
    while math.ldexp(1.0, coarse_exp) <= decimal_step:
        coarse_exp += 1
    inv_step = math.ldexp(1.0, -coarse_exp)
    index_bits = np.int64(0)
    for sample_idx in range(samples.shape[0]):
        scaled = samples[sample_idx] * inv_step
        # Stop at the first sample off the coarse grid (sample 0 for ordinary decimal data)
        if scaled != math.floor(scaled):
            return quant_exp
        index_bits |= np.int64(scaled)
    # The indices' common trailing zero bits make the grid coarser still (a non-constant block
    # has a nonzero index)
    while index_bits & 1 == 0:
        index_bits >>= 1
        coarse_exp += 1
    return coarse_exp


@njit(nogil=True, cache=True)
def moved_off_binary_grid(
    samples: np.ndarray, binary_exp: int, decimal_exp: int, base_index: np.int64, quantized: np.ndarray
) -> bool:
    """Whether the block is exactly on the 2^binary_exp grid and the decimal grid would move it.

    x * 2^-e is exact unless it overflows, and a sample that large is a multiple anyway. A sample
    on the decimal grid decodes to itself: (base_index + q) / 10^-p, or times 10^p (§6.2).
    """
    inv_step = math.ldexp(1.0, -binary_exp)
    divide = decimal_exp < 0
    power = 10.0 ** abs(decimal_exp)
    moved = False
    for sample_idx in range(samples.shape[0]):
        sample = samples[sample_idx]
        scaled = sample * inv_step
        # Stop at the first sample off the grid (sample 0 for ordinary decimal data)
        if scaled != math.floor(scaled):
            return False
        index = float(base_index + quantized[sample_idx])
        moved |= (index / power if divide else index * power) != sample
    return moved


@njit(nogil=True, cache=True)
def decimal_grid(
    samples: np.ndarray,
    lower_bound: float,
    upper_bound: float,
    quant_exp: int,
    out_quantized: np.ndarray,
) -> tuple[int, np.int64, int]:
    """The block's decimal grid (§5.3), if it takes one, else its power-of-two exponent.

    A block exactly on a power-of-two grid coarser than the decimal grid takes the coarsest such
    (the decimal grid would store it with an odd factor: 0.25 on 0.01, x 25). A block exactly on
    its planned power-of-two grid stays there when the decimal grid would move it (detection
    allows a quarter step): data decoded from a power-of-two block re-encode to themselves.

    Args:
        samples: 1D float64 array of a non-constant block's samples, range below 2^1023.
        lower_bound: Minimum value.
        upper_bound: Maximum value.
        quant_exp: The planned power-of-two exponent.
        out_quantized: Output 1D int32 array receiving the decimal grid offsets on a decimal grid.

    Returns:
        A tuple of (decimal_exp, grid_index, quant_exp): the decimal exponent and the minimum's
        index on it, or NO_DECIMAL, 0 and the power-of-two exponent (quant_exp or coarser).
    """
    decimal_exp, grid_index = detect_decimal(samples, lower_bound, upper_bound, math.ldexp(1.0, quant_exp), out_quantized)
    if decimal_exp == NO_DECIMAL:
        return NO_DECIMAL, np.int64(0), quant_exp
    binary_exp = coarse_binary_exponent(samples, quant_exp, 10.0 ** decimal_exp)
    # With p >= 0 and e <= p + 1, a sample on 2^e is a multiple of 2^min(e, p), as its decimal grid
    # point is, and within 2^e / 4 of it (detect_decimal): on it. Integers skip the check
    moves = decimal_exp < 0 or quant_exp > decimal_exp + 1
    if binary_exp != quant_exp or (
        moves and moved_off_binary_grid(samples, quant_exp, decimal_exp, grid_index, out_quantized)
    ):
        return NO_DECIMAL, np.int64(0), binary_exp
    return decimal_exp, grid_index, quant_exp


@njit(nogil=True, cache=True)
def quantize(
    samples: np.ndarray, lower_bound: float, quant_exp: int, out_quantized: np.ndarray
) -> None:
    """Quantizes samples onto the power-of-two grid anchored at the block minimum.

    Each sample becomes floor((sample - lower_bound) * 2^-quant_exp + 0.5).

    Used only for the blocks snap_anchor exempts; others use quantize_grid.

    Args:
        samples: 1D float64 array of samples.
        lower_bound: Minimum value in the block.
        quant_exp: Quantization exponent.
        out_quantized: Output 1D int32 array receiving quantized integers.
    """
    # Huge exponent: quantize at half-scale to prevent (sample - lower_bound) overflow. (The
    # exempt blocks are constant, with exponent 0, or near +-DBL_MAX: never a subnormal step.)
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
    """Quantizes samples onto the absolute power-of-two grid.

    Each sample becomes rint(sample * 2^-quant_exp) - base_index, where base_index is
    rint(minimum * 2^-quant_exp) with the same rounding: the minimum maps to 0 and every
    quantized value is non-negative. The difference of two nearby integers held in floats
    is exact.

    Args:
        samples: 1D float64 array of samples.
        quant_exp: Quantization exponent.
        base_index: The anchor's grid index (snap_anchor).
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

    Args:
        samples: 1D float64 array of sample values.
        lower_bound: Minimum sample value in the block.
        upper_bound: Maximum sample value in the block.
        quant_exp: Quantization exponent.
        out_quantized: Output 1D int32 array receiving quantized values.

    Returns:
        The power-of-two anchor value as a float64.
    """
    anchor, base_index = snap_anchor(lower_bound, upper_bound, quant_exp)
    if math.isnan(base_index):
        quantize(samples, lower_bound, quant_exp, out_quantized)
    else:
        quantize_grid(samples, quant_exp, base_index, out_quantized)
    return anchor


@njit(inline="always")
def _common_twos(gcd: int | np.integer, base_index: np.int64) -> int:
    """The number of trailing zero bits common to gcd (> 0) and base_index."""
    bits = gcd | base_index if base_index else gcd
    twos = 0
    while bits & 1 == 0:
        bits >>= 1
        twos += 1
    return twos


@njit(inline="always")
def stored_step(flags: int, param: int, scale: np.int64) -> float:
    """A block's grid step: 2^e, 10^p, or N / D times either."""
    step = 10.0 ** param if flags & BLOCK_FLAG_DECIMAL else math.ldexp(1.0, param)
    if flags & BLOCK_FLAG_SCALED:
        step *= (scale & SCALE_MASK) / (scale >> SCALE_SHIFT)
    return step


@njit(inline="always")
def planned_multiple(step: float, planned_exp: int) -> int:
    """step in units of 2^planned_exp, if an integer M up to 2^16 (every power-of-two grid, a decimal
    10^p with p >= 0, N 2^e), else 0.

    The block's indices on the planned grid are then M q exactly: the minimum is on both grids.
    """
    ratio = math.ldexp(step, -planned_exp)
    return int(ratio) if ratio == math.floor(ratio) and 1 <= ratio <= SCALE_MASK + 1 else 0


@njit(nogil=True, cache=True)
def center_anchor(
    quantized: np.ndarray,
    order: int,
    flags: int,
    param: int,
    grid_index: np.int64,
    anchor: float,
    scale: np.int64,
) -> tuple[np.int64, float]:
    """Moves a block's anchor to its reference index c and its indices to q - c, signed 16-bit (SPEC §4).

    c is the trimmed mean index for order 0 (the residuals are then the indices themselves,
    centered on the block), else the first index (the first residual is then 0), clamped so that
    every q - c fits in [-32768, 32767] and the anchor stays exact: an integer anchor (decimal or
    scaled grid) moves by c grid steps (N c fine points), a power-of-two one to the grid point
    (k + c) 2^e when its index k = anchor / 2^e is an integer under 2^52; an unsnapped anchor (near
    +-DBL_MAX) moves only as far as the 16 bits require, by an exact sum there (e >= 971).

    Args:
        quantized: 1D int32 array of the block's indices from its minimum (0 to 65,535), shifted in place.
        order: The block's predictor order.
        flags, param, grid_index, anchor, scale: The block's grid, as encode_block returns them.

    Returns:
        A tuple of (grid_index, anchor), moved to index c.
    """
    num_samples = quantized.shape[0]
    if num_samples == 0:
        return grid_index, anchor
    # int32 max and sum (|sum| < 2^16 * 2^16 fits int64 only: widen per 2^15 samples) vectorize
    max_q = np.int32(0)
    total = np.int64(0)
    for chunk_start in range(0, num_samples, 1 << 15):
        chunk = quantized[chunk_start:min(chunk_start + (1 << 15), num_samples)]
        chunk_total = np.int32(0)
        chunk_max = np.int32(0)
        for sample_idx in range(chunk.shape[0]):
            chunk_max = max(chunk_max, chunk[sample_idx])
            chunk_total += chunk[sample_idx] >> 1
        max_q = max(max_q, chunk_max)
        total += np.int64(chunk_total)
    max_index = np.int64(max_q)
    low, high = max(max_index - INT16_MAX, np.int64(0)), min(max_index, np.int64(-INT16_MIN))
    # The mean from halved indices (an int32 sum of 2^15 of them can't overflow): within one of it
    wanted = (2 * total + num_samples // 2) // num_samples if order == 0 else np.int64(quantized[0])
    if order == 0:
        # Trimmed: spikes would pull the plain mean off the bulk of the block (integer loops vectorize)
        mean = np.int32(wanted)
        # On every 4th index: a center needn't be exact, and this keeps the pass cheap
        sub = quantized[::TRIM_STRIDE]
        abs_dev = np.int64(0)
        for sample_idx in range(sub.shape[0]):
            abs_dev += abs(sub[sample_idx] - mean)
        limit = TRIM_DEVIATIONS * abs_dev / sub.shape[0]
        kept_sum = np.int64(0)
        kept = np.int64(0)
        for sample_idx in range(sub.shape[0]):
            if abs(sub[sample_idx] - mean) <= limit:
                kept_sum += sub[sample_idx]
                kept += 1
        if kept > 0:
            wanted = (2 * kept_sum + kept) // (2 * kept)
    shift = min(max(wanted, low), high)
    if flags & (BLOCK_FLAG_DECIMAL | BLOCK_FLAG_SCALED):
        num = scale & SCALE_MASK if flags & BLOCK_FLAG_SCALED else np.int64(1)
        # The anchor stays an index under 2^52, as the samples' own indices are (§5.3, §5.4)
        if abs(grid_index + num * shift) >= INDEX_LIMIT:
            shift = low
        grid_index = grid_index + num * shift
    else:
        anchor_index = math.ldexp(anchor, -param)
        if anchor_index == math.floor(anchor_index) and abs(anchor_index) < FLOAT64_EXACT_INT_LIMIT:
            anchor = math.ldexp(anchor_index + float(shift), param) + 0.0
        else:
            # Unsnapped (or beyond exact indices): only the shift the 16 bits need, at half scale
            shift = low
            if shift:
                anchor = 2.0 * (0.5 * anchor + float(shift) * math.ldexp(1.0, param - 1))
    if shift:
        shift32 = np.int32(shift)
        for sample_idx in range(num_samples):
            quantized[sample_idx] -= shift32
    return grid_index, anchor
