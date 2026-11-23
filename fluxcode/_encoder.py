# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Encoding stages and numba kernels for float64 series compression.

Provides per-block analysis, noise floor gating, decimal detection, power-of-two
quantization, predictor order selection, residual differentiation mod 2^16, and
per-block-group bit target allocation. Composed per block group in `encode_group`.
"""

import math
from collections.abc import Callable
from typing import cast

import numpy as np
from numba import njit
from numba.cpython.unsafe.numbers import leading_zeros as _leading_zeros

from . import _extreme_magnitudes as xm
from . import _noise as _noise_mod
from . import _rational
from ._format import (
    BLOCK_FLAG_DECIMAL,
    BLOCK_FLAG_IRREGULAR_TIME,
    BLOCK_FLAG_NONFINITE,
    BLOCK_FLAG_ORDER,
    BLOCK_FLAG_SCALED,
    CODE_ESCAPE,
    CODE_FINITE,
    E_MAX,
    E_MIN,
    INDEX_LIMIT,
    P_MAX,
    P_MIN,
    SCALE_MASK,
    SCALE_SHIFT,
    SCALED_E_MAX,
    SCALED_E_MIN,
    SCALED_P_MAX,
    SHORT_BLOCK_LEN,
)
from ._noise import MAD_TO_SD, block_noise
from ._nonfinite import fill_nonfinite, hold_coded
from ._time import CADENCE_SAMPLES

leading_zeros = cast("Callable[[int | np.integer], int]", _leading_zeros)
"""Numba intrinsic counting leading zero bits, typed as its jitted call signature."""

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

FLOOR_MARGIN: float = 1.02
"""How far past a power of two the noise floor's step may be for a block exactly on the grid one
finer to keep it (keep_floor_grid): above the 0.26% quantizing adds to sigma, with room for the
estimate's own movement."""

NOISE_GATE: float = _noise_mod.NOISE_GATE
"""The noise gate (_noise.NOISE_GATE)."""

NOISE_MIN_LEN: int = 128
"""Smallest block the noise floor applies to. Below it one clean step or spike is most of the
12th differences (at 64 samples a step of 10 read sigma 0.004, at 32 samples 0.78); from 128 up
such shapes read far under the B-bit step and white noise passes the gate 96% of the time (ENCODER
§5.2). Irregular times keep their own minimum (_noise.MIN_ONE_SCAN_TRIPLETS), as their chords'
gate must still tell a random walk from white noise."""

PICK_LEN: int = 256
"""Blocks up to this many samples pick their predictor order on every sample; longer ones on
PICK_WINDOWS evenly spread windows of PICK_WINDOW_LEN (the same 256 samples)."""

PICK_WINDOWS: int = 8
"""Windows a long block picks its order on."""

PICK_WINDOW_LEN: int = 32
"""Samples per order-pick window."""

FLOAT64_EXACT_INT_LIMIT: float = 2.0 ** 52
"""Scaled-magnitude limit for decimal candidates: below 2^52 a double's spacing is at most
0.5, so floor(v + 0.5) and the tolerance check are meaningful."""

NUM_HIST_CHAINS: int = 4
"""Number of interleaved histogram accumulator chains to prevent pipeline write stalls."""

NUM_BIT_LENGTH_BINS: int = 17
"""Total bit-length classes (0 through 16) for zigzagged uint16 residuals."""

UINT16_MASK: int = 0xFFFF
"""Bitmask for 16-bit unsigned integers (65535)."""

ESCAPE_MIN_LEN: int = 16
"""Smallest block whose outliers may be escaped."""

ESCAPE_TRIGGER: float = 0.05
"""A block is looked at for outliers when its mean is within this share of its range from one end."""

ESCAPE_RANGE_SPREAD: float = 16.0
"""... or when its range is over this many standard deviations."""

ESCAPE_DEVIATIONS: float = 16.0
"""Outliers are beyond this many robust deviations of the block's trimmed center."""

ESCAPE_MAX_SHARE: int = 16
"""At most 1/ESCAPE_MAX_SHARE of a block is escaped."""

ESCAPE_MIN_GAIN: int = 2
"""Escapes are taken when the bulk's step is at least 2^ESCAPE_MIN_GAIN times finer."""

TRIM_STRIDE: int = 4
"""The trimmed mean of an order-0 block reads every TRIM_STRIDE-th index."""

TRIM_DEVIATIONS: float = 3.0
"""An order-0 block centers on the mean of its indices within this many mean absolute deviations
of their mean (center)."""

INT16_MAX: int = 32767
"""Largest signed 16-bit index (SPEC §4: q is read as int16)."""

INT16_MIN: int = -32768
"""Smallest signed 16-bit index."""

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
        A tuple of (min_val, max_val, mean_val, bad_index):
            min_val: Minimum value (0.0 if a non-finite sample is present).
            max_val: Maximum value (0.0 if a non-finite sample is present).
            mean_val: Arithmetic mean of samples (0.0 if a non-finite sample is present).
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


NO_FLOOR_EXP: int = E_MIN - 64
"""floor_exponent without a noise floor: finer than any step."""

TOLERANCE_STEP_BITS: int = 4
"""A tolerance step is N points of 2^(e - TOLERANCE_STEP_BITS), N from 17 to 31: within 6% of 2 tolerance."""


def tolerance_exponent(tolerance: float) -> int:
    """The exponent of the largest power-of-two step whose half is at most tolerance (Params.tolerance);
    a very negative one (no effect) for 0."""
    if not tolerance > 0:
        return E_MIN - 64
    return math.floor(math.log2(2.0 * tolerance))


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
        noise_rho: The noise gate's statistic (_noise.block_noise): over NOISE_GATE passes.
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
    # Coarsen the step up to noise_factor * noise_sigma where the noise gate passes
    planned_exp = max(planned_exp, floor_exponent(noise_factor, noise_sigma, noise_rho))
    # Clamp to coarsest allowed step at min_bits
    return min(planned_exp, range_exponent(lower_bound, upper_bound, min_bits))


@njit(nogil=True, cache=True)
def keep_floor_grid(samples: np.ndarray, planned_exp: int, floor_step: float) -> int:
    """The planned exponent, or one finer for a block already exactly on that finer grid when the
    noise floor's step f·sigma is under FLOOR_MARGIN past 2^planned_exp.

    Decoded data sits on its step's grid, and quantizing it raised sigma slightly (at most 0.26%):
    a floor that was just under a power of two can cross it on re-encode. Keeping the finer grid
    makes decoded data a fixed point; data not exactly on it (any original data in practice) and
    floors clear of the boundary are unaffected.
    """
    if not floor_step > 0 or math.floor(math.log2(floor_step)) != planned_exp:
        return planned_exp
    if floor_step >= math.ldexp(1.0, planned_exp) * FLOOR_MARGIN:
        return planned_exp
    inv_step = math.ldexp(1.0, 1 - planned_exp)
    for sample_idx in range(samples.shape[0]):
        scaled = samples[sample_idx] * inv_step
        if scaled != math.floor(scaled) or not math.isfinite(scaled):
            return planned_exp
    return planned_exp - 1


@njit(nogil=True, cache=True)
def floor_exponent(noise_factor: float, noise_sigma: float, noise_rho: float) -> int:
    """The noise floor's exponent floor(log2(f·sigma)) if the noise gate passes, else NO_FLOOR_EXP."""
    if noise_factor > 0 and noise_rho > NOISE_GATE and noise_sigma > 0:
        return math.floor(math.log2(noise_factor * noise_sigma))
    return NO_FLOOR_EXP


@njit(nogil=True, cache=True)
def plan_block_step(
    samples: np.ndarray, lower_bound: float, upper_bound: float, noise_sigma: float, noise_rho: float,
    noise_factor: float, min_bits: int, max_bits: int, tolerance_exp: int, tolerance_step: float,
) -> tuple[int, float]:
    """A block's planned step (ENCODER §5.1), before grid detection and the target: its exponent,
    and the scaled step tolerance sets (0.0 if it doesn't).

    The coarsest of the finest step its range fits in max_bits, the noise floor's (plan_step) and
    tolerance's (2^tolerance_exp), capped at min_bits' coarsest. Where tolerance sets it under that
    cap, encode_block may take all of its allowance: a scaled step tolerance_step (tolerance_grid).
    A floored block keeps a finer grid it is already on (keep_floor_grid), never finer than its
    range fits in max_bits.

    Args:
        samples: 1D float64 array of the block's samples (held where coded).
        lower_bound, upper_bound: Their minimum and maximum.
        noise_sigma, noise_rho: The block's noise estimate (_noise.block_noise).
        noise_factor: Params.noise_floor_sigma (0.0 if off).
        min_bits, max_bits: Params.min_quantize_bits, Params.max_quantize_bits.
        tolerance_exp: tolerance_exponent(Params.tolerance) (very negative without one).
        tolerance_step: 2 Params.tolerance (0.0 without one).

    Returns:
        A tuple of (exponent, tolerance_step or 0.0); (0, 0.0) for a constant block.
    """
    planned_exp = plan_step(lower_bound, upper_bound, noise_sigma, noise_rho, noise_factor, min_bits, max_bits)
    block_tolerance_step = 0.0
    if upper_bound > lower_bound and tolerance_exp > planned_exp:
        # The caller's absolute error, within min_quantize_bits' coarsest step
        coarsest_exp = range_exponent(lower_bound, upper_bound, min_bits)
        planned_exp = min(tolerance_exp, coarsest_exp)
        if tolerance_exp < coarsest_exp:
            # All of it (a scaled step up to 2 tolerance, still under 2^coarsest_exp)
            block_tolerance_step = tolerance_step
    if noise_rho > NOISE_GATE:
        # Never finer than the range fits in max_bits (a block whose floor is its range's own step)
        planned_exp = max(keep_floor_grid(samples, planned_exp, noise_factor * noise_sigma),
                          range_exponent(lower_bound, upper_bound, max_bits) if upper_bound > lower_bound else planned_exp)
    return planned_exp, block_tolerance_step


@njit(nogil=True, cache=True)
def tolerance_grid(lower_bound: float, quant_exp: int, tolerance_step: float) -> tuple[np.int64, int, float]:
    """Tolerance's scaled grid for a block whose step it sets: N points of the fine grid
    2^(quant_exp - TOLERANCE_STEP_BITS), N from 17 to 31 the most that fit in tolerance_step, on the
    absolute grid of multiples of the step (so decoded data lies on it and re-encodes to itself).

    Only encode_block's rational search takes it (Params.grids "rational"): "binary" and "decimal"
    allow only power-of-two and decimal steps, and a block whose indices share a factor already has
    an exact grid at least as coarse (an odd factor of 3 or more times 2^quant_exp).

    Returns:
        A tuple of (N, fine exponent, the minimum's fine index rounded down to a multiple of N), or
        N = 0 where it doesn't apply (tolerance didn't set the step, or the scaled grid's limits).
    """
    fine_exp = quant_exp - TOLERANCE_STEP_BITS
    if not (tolerance_step > 0 and quant_exp == math.floor(math.log2(tolerance_step))):
        return np.int64(0), fine_exp, 0.0
    fine_step = math.ldexp(1.0, fine_exp)
    num = np.int64(tolerance_step / fine_step)
    base = math.floor(lower_bound / (num * fine_step)) * num
    if not (num > (1 << TOLERANCE_STEP_BITS) and fine_exp >= SCALED_E_MIN and abs(base) < INDEX_LIMIT):
        return np.int64(0), fine_exp, 0.0
    return num, fine_exp, float(base)


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
def _order_sums(segment: np.ndarray) -> tuple[np.int64, np.int64, np.int64, np.int64, np.int64, np.int64, np.int64, np.int64]:
    """Sums of the order-0 to order-3 differences of one contiguous segment (at least 4 samples).

    Returns:
        A tuple of (sum_q, sum_sq_q, sum_d1, sum_sq_d1, sum_d2, sum_sq_d2, sum_d3, sum_sq_d3) over the
        segment's len, len - 1, len - 2 and len - 3 differences.
    """
    num_samples = segment.shape[0]
    sum_q = sum_sq_q = sum_sq_diff1 = sum_sq_diff2 = sum_sq_diff3 = np.int64(0)
    for sample_idx in range(3):
        val = np.int64(segment[sample_idx])
        sum_q += val
        sum_sq_q += val * val
    for sample_idx in range(1, 3):
        diff = np.int64(segment[sample_idx]) - segment[sample_idx - 1]
        sum_sq_diff1 += diff * diff
    diff_order2 = np.int64(segment[2]) - 2 * np.int64(segment[1]) + segment[0]
    sum_sq_diff2 += diff_order2 * diff_order2
    # Offset slices from index 0 avoid numba's negative-index checks, so the loop vectorizes
    slice_0 = segment[3:num_samples]
    slice_1 = segment[2:num_samples - 1]
    slice_2 = segment[1:num_samples - 2]
    slice_3 = segment[:num_samples - 3]
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
    # Telescoping sums give exact sums of differences in O(1)
    last = num_samples - 1
    sum_diff1 = np.int64(segment[last]) - segment[0]
    sum_diff2 = (np.int64(segment[last]) - segment[last - 1]) - (np.int64(segment[1]) - segment[0])
    sum_diff3 = (
        (np.int64(segment[last]) - 2 * np.int64(segment[last - 1]) + segment[last - 2])
        - (np.int64(segment[2]) - 2 * np.int64(segment[1]) + segment[0])
    )
    return sum_q, sum_sq_q, sum_diff1, sum_sq_diff1, sum_diff2, sum_sq_diff2, sum_diff3, sum_sq_diff3


@njit(nogil=True, cache=True)
def pick_order(quantized_samples: np.ndarray, orders_mask: int) -> int:
    """Selects the predictor order (0-3) minimizing difference variance.

    A block of up to PICK_LEN samples is measured whole; a longer one on PICK_WINDOWS windows of
    PICK_WINDOW_LEN samples spread evenly over it (differences within each window), so a block
    whose character changes part way (a move, a ramp, a burst) is judged on all of it.

    Args:
        quantized_samples: 1D int32 array of the block's quantized integers (at least 4).
        orders_mask: Bitmask of allowable orders (bit k enables order k).

    Returns:
        Selected difference order (0, 1, 2, or 3). Ties favor lower orders.
    """
    num_samples = quantized_samples.shape[0]
    num_windows, window_len = 1, num_samples
    if num_samples > PICK_LEN:
        num_windows, window_len = PICK_WINDOWS, PICK_WINDOW_LEN
    sums = np.zeros(8, np.int64)
    for window_idx in range(num_windows):
        window_start = window_idx * (num_samples - window_len) // max(num_windows - 1, 1)
        window_sums = _order_sums(quantized_samples[window_start:window_start + window_len])
        for sum_idx in range(8):
            sums[sum_idx] += window_sums[sum_idx]
    best_order, min_variance = 0, 1e300
    for order_idx in range(4):
        # Skip orders not permitted by caller's bitmask
        if not (orders_mask >> order_idx) & 1:
            continue
        valid_count = num_windows * (window_len - order_idx)
        sum_diff_val, sum_sq_diff_val = sums[2 * order_idx], sums[2 * order_idx + 1]
        # Variance of the differences: (n * sum(d^2) - sum(d)^2) / n^2, n = valid_count
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
    """Computes differences of the given order, mod 2^16, of the quantized samples.

    The samples are preceded by zeros, so the first `order` residuals are partial
    differences.

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
        # First difference: sample - previous sample
        slice_curr = quantized_samples[1:]
        slice_prev = quantized_samples[:num_samples - 1]
        out_slice = out_residuals[1:]
        for sample_idx in range(num_samples - 1):
            out_slice[sample_idx] = _wrap16(slice_curr[sample_idx] - slice_prev[sample_idx])
    elif order == 2:
        # Second difference with prepended zeros: residual 1 is sample 1 - 2 * sample 0
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


@njit(inline="always")
def _common_twos(gcd: int | np.integer, base_index: np.int64) -> int:
    """The number of trailing zero bits common to gcd (> 0) and base_index."""
    bits = gcd | base_index if base_index else gcd
    twos = 0
    while bits & 1 == 0:
        bits >>= 1
        twos += 1
    return twos


@njit(nogil=True, cache=True)
def zero_share(residuals: np.ndarray) -> float:
    """The share of a block's residuals that are 0."""
    zeros = 0
    for sample_idx in range(residuals.shape[0]):
        zeros += residuals[sample_idx] == 0
    return zeros / max(residuals.shape[0], 1)


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


@njit(nogil=True, cache=True, fastmath=True)
def _sum_squares(samples: np.ndarray, mean: float) -> float:
    """Sum of squared deviations from mean (fastmath: a reassociated sum vectorizes; a trigger only)."""
    total = 0.0
    for sample_idx in range(samples.shape[0]):
        dev = samples[sample_idx] - mean
        total += dev * dev
    return total


@njit(inline="always")
def _on_decimal_grid(samples: np.ndarray, codes: np.ndarray, coded: bool, decimal_exp: int) -> bool:
    """Whether every uncoded sample is the double nearest k 10^p (as the decoder makes it)."""
    for sample_idx in range(samples.shape[0]):
        if coded and codes[sample_idx] != CODE_FINITE:
            continue
        value = samples[sample_idx]
        if decimal_exp < 0:
            divisor = 10.0 ** -decimal_exp
            if float(round(value * divisor)) / divisor != value:
                return False
        else:
            multiplier = 10.0 ** decimal_exp
            if float(round(value / multiplier)) * multiplier != value:
                return False
    return True


@njit(inline="always")
def _on_binary_grid(samples: np.ndarray, codes: np.ndarray, coded: bool, exponent: int) -> bool:
    """Whether every uncoded sample is a multiple of 2^exponent."""
    for sample_idx in range(samples.shape[0]):
        if coded and codes[sample_idx] != CODE_FINITE:
            continue
        scaled = math.ldexp(samples[sample_idx], -exponent)
        if scaled != math.floor(scaled):
            return False
    return True


@njit(nogil=True, cache=True)
def escape_outliers(
    samples: np.ndarray, held: np.ndarray, codes: np.ndarray, coded: bool, mean: float, lower_bound: float,
    upper_bound: float, floor_exp: int, max_bits: int, grids: int, scratch_quantized: np.ndarray,
) -> tuple[bool, float, float]:
    """Escapes a block's outliers when that refines the rest of it (ENCODER §5.1.1).

    A glitch (an error code, an overrange reading) stretches the block's range, and with it the
    step of every other sample. Escaped samples get code 10b (their values go to the escapes field)
    and are held; the block is then quantized on the bulk's range. Tried when the mean is within
    ESCAPE_TRIGGER of the range from one end (a few points far from the rest), or the range is over
    ESCAPE_RANGE_SPREAD standard deviations (outliers on both sides). Taken only if it pays: at
    most 1/ESCAPE_MAX_SHARE of the block beyond ESCAPE_DEVIATIONS robust standard deviations of a
    trimmed center, the bulk's step at least 4 times finer than the block's and than its noise
    floor's (or tolerance's), and the block not already exact at its full range (a decimal grid).

    Args:
        samples: 1D float64 array of the block's samples (held where already coded).
        held: 1D float64 array (not samples) receiving the held samples when escapes are taken.
        codes: 1D uint8 array of the block's codes (written in full if escapes are taken).
        coded: Whether codes already holds the block's non-finite codes.
        mean, lower_bound, upper_bound: The block's finite mean, minimum and maximum.
        floor_exp: The coarsest exponent the noise floor or tolerance allow the block (its noise
            estimate shrugs off outliers), or NO_FLOOR_EXP without either.
        max_bits, grids: Params.
        scratch_quantized: 1D int32 scratch of the block's length.

    Returns:
        A tuple of (escaped, bulk_min, bulk_max).
    """
    num_samples = samples.shape[0]
    span = upper_bound - lower_bound
    if num_samples < ESCAPE_MIN_LEN or not span > 0 or not math.isfinite(span):
        return False, lower_bound, upper_bound
    old_exp = range_exponent(lower_bound, upper_bound, max_bits)
    # The noise floor sets the step within 2 levels of the full range's anyway: nothing to gain
    if old_exp - floor_exp < ESCAPE_MIN_GAIN:
        return False, lower_bound, upper_bound
    if not (mean - lower_bound < ESCAPE_TRIGGER * span or upper_bound - mean < ESCAPE_TRIGGER * span):
        # Outliers on both sides leave the mean mid-range, but stretch the range far past the spread
        # (a sine's range is 2.8 standard deviations, Gaussian noise's 6-7 on 1000 samples)
        squares = 0.0
        count = num_samples
        if coded:
            count = 0
            for sample_idx in range(num_samples):
                if codes[sample_idx] == CODE_FINITE:
                    dev = samples[sample_idx] - mean
                    squares += dev * dev
                    count += 1
        else:
            squares = _sum_squares(samples, mean)
        if not span * span > ESCAPE_RANGE_SPREAD * ESCAPE_RANGE_SPREAD * squares / max(count, 1):
            return False, lower_bound, upper_bound
    # A trimmed center and mean deviation of the uncoded samples
    center_val = mean
    deviation = 0.0
    for _ in range(3):
        total = 0.0
        count = 0
        for sample_idx in range(num_samples):
            if not coded or codes[sample_idx] == CODE_FINITE:
                total += abs(samples[sample_idx] - center_val)
                count += 1
        if count == 0:
            return False, lower_bound, upper_bound
        limit = 4.0 * MAD_TO_SD * total / count
        kept_sum = 0.0
        kept = 0
        deviation = 0.0
        for sample_idx in range(num_samples):
            if not coded or codes[sample_idx] == CODE_FINITE:
                dev = abs(samples[sample_idx] - center_val)
                if dev <= limit:
                    kept_sum += samples[sample_idx]
                    deviation += dev
                    kept += 1
        if kept == 0:
            return False, lower_bound, upper_bound
        center_val = kept_sum / kept
        deviation /= kept
    threshold = ESCAPE_DEVIATIONS * MAD_TO_SD * deviation
    if not threshold > 0:
        return False, lower_bound, upper_bound
    num_escaped = 0
    bulk_min, bulk_max = math.inf, -math.inf
    for sample_idx in range(num_samples):
        if coded and codes[sample_idx] != CODE_FINITE:
            continue
        value = samples[sample_idx]
        if abs(value - center_val) > threshold:
            num_escaped += 1
        else:
            bulk_min = min(bulk_min, value)
            bulk_max = max(bulk_max, value)
    if num_escaped == 0 or num_escaped * ESCAPE_MAX_SHARE > num_samples or not bulk_max >= bulk_min:
        return False, lower_bound, upper_bound
    new_exp = range_exponent(bulk_min, bulk_max, max_bits) if bulk_max > bulk_min else old_exp - ESCAPE_MIN_GAIN
    # The noise floor would set the step anyway: nothing to gain
    if old_exp - max(new_exp, floor_exp) < ESCAPE_MIN_GAIN:
        return False, lower_bound, upper_bound
    # Already exact at the full range (every sample on the binary grid 2^old_exp, or on a decimal grid
    # detection finds): nothing to gain
    if _on_binary_grid(samples, codes, coded, old_exp):
        return False, lower_bound, upper_bound
    if grids >= GRIDS_DECIMAL and span < xm.WIDE_RANGE:
        decimal_exp, _, _ = decimal_grid(samples, lower_bound, upper_bound, old_exp, scratch_quantized)
        if decimal_exp != NO_DECIMAL and _on_decimal_grid(samples, codes, coded, decimal_exp):
            return False, lower_bound, upper_bound
    # Hold the escaped samples (in held, apart from samples), and the coded ones again: a non-finite
    # sample after a glitch held the glitch's value, which must not stay in the bulk
    held[:] = samples
    previous = center_val
    for sample_idx in range(num_samples):
        uncoded = not coded or codes[sample_idx] == CODE_FINITE
        if uncoded and abs(samples[sample_idx] - center_val) <= threshold:
            previous = samples[sample_idx]
        else:
            held[sample_idx] = previous
    if not coded:
        codes[:] = CODE_FINITE
    for sample_idx in range(num_samples):
        if codes[sample_idx] == CODE_FINITE and abs(samples[sample_idx] - center_val) > threshold:
            codes[sample_idx] = CODE_ESCAPE
    return True, bulk_min, bulk_max


@njit(nogil=True, cache=True)
def encode_block(
    samples: np.ndarray,
    lower_bound: float,
    upper_bound: float,
    quant_exp: int,
    grids: int,
    orders_mask: int,
    scratch_quantized: np.ndarray,
    scratch_rational: np.ndarray,
    scratch_index: np.ndarray,
    out_residuals: np.ndarray,
    tolerance_step: float = 0.0,
) -> tuple[int, int, np.int64, float, np.int64]:
    """Encodes one block: quantizes, scales the grid, picks predictor order, and computes residuals.

    The grid (§5.3, §5.4), by grids level: GRIDS_DECIMAL and up, a decimal grid if one is detected
    (unless the block is exactly on a power-of-two grid that is coarser or that the decimal grid
    would move it off), else the power-of-two grid; GRIDS_RATIONAL, the
    index gcd N of its indices scales it (on a power-of-two grid, the power of two N shares with
    the anchor's index coarsens the exponent instead), and a power-of-two block with no common
    factor tries the rational search, or tolerance's scaled grid where tolerance_step is given and
    the search finds no coarser one (tolerance_grid). Then the order (pick_order), the anchor moved
    to the center (center), and the residuals.

    Args:
        samples: 1D float64 array of samples.
        lower_bound: Minimum value.
        upper_bound: Maximum value.
        quant_exp: Quantization exponent.
        grids: GRIDS_BINARY, GRIDS_DECIMAL or GRIDS_RATIONAL (Params.grids).
        orders_mask: Allowed predictor orders bitmask.
        scratch_quantized: Preallocated 1D int32 scratch buffer for quantized values.
        scratch_rational: Preallocated 1D int32 scratch buffer for the rational search's indices.
        scratch_index: Preallocated 1D float64 scratch buffer for the rational search's step counts.
        out_residuals: Output 1D int16 array receiving residuals.
        tolerance_step: The scaled step tolerance sets (plan_block_step), or 0.0.

    Returns:
        A tuple of (flags, param, grid_index, anchor, scale): grid_index is the integer anchor of
        a decimal or scaled grid, anchor the power-of-two grid's float anchor (unused on the
        others), both at the block's reference index (center); scale is grid_scale, N | D << 16
        (0 when unscaled).
    """
    flags = 0
    anchor = lower_bound
    param_val = quant_exp
    grid_index = np.int64(0)
    scale = np.int64(0)
    decimal_detected = False
    non_constant = upper_bound > lower_bound
    range_span = upper_bound - lower_bound
    # Attempt decimal grid detection if enabled and range is within 2^1023
    if grids >= GRIDS_DECIMAL and non_constant and range_span < xm.WIDE_RANGE:
        decimal_exp, grid_index, quant_exp = decimal_grid(
            samples, lower_bound, upper_bound, quant_exp, scratch_quantized
        )
        param_val = quant_exp
        if decimal_exp != NO_DECIMAL:
            decimal_detected = True
            param_val = decimal_exp
            flags |= BLOCK_FLAG_DECIMAL
    if decimal_detected:
        gcd = _rational.index_gcd(scratch_quantized) if grids >= GRIDS_RATIONAL else 0
        # A decimal p > 0 folds into N, so the grid divides by D * 10^-p (SPEC §5): p <= 4 (10^5 > 2^16)
        if gcd > 1 and param_val <= SCALED_P_MAX + _rational.MAX_DECIMAL_FOLD:
            fold = np.int64(10) ** max(param_val - SCALED_P_MAX, 0)
            if gcd * fold <= SCALE_MASK and abs(grid_index * fold) < INDEX_LIMIT:
                _rational.divide(scratch_quantized, gcd)
                scale = (gcd * fold) | (np.int64(1) << SCALE_SHIFT)  # D = 1
                grid_index *= fold
                param_val = min(param_val, SCALED_P_MAX)
                flags |= BLOCK_FLAG_SCALED
    else:
        anchor = quantize_block(samples, lower_bound, upper_bound, quant_exp, scratch_quantized)
        base_index = snap_anchor(lower_bound, upper_bound, quant_exp)[1]
        # Exceptions (constant blocks, unfinite snapped anchors) keep their grid
        if grids >= GRIDS_RATIONAL and non_constant and not math.isnan(base_index):
            gcd = _rational.index_gcd(scratch_quantized)
            base_int = np.int64(base_index)
            if gcd > 1:
                # The power of two shared with the anchor's index coarsens the exponent: the
                # grid stays absolute and unscaled
                twos = min(_common_twos(gcd, base_int), E_MAX - quant_exp)
                if twos:
                    _rational.divide(scratch_quantized, np.int64(1) << twos)
                    gcd >>= twos
                    base_int >>= twos
                    quant_exp += twos
                    param_val = quant_exp
                if gcd > 1 and SCALED_E_MIN <= quant_exp <= SCALED_E_MAX and abs(base_int) < INDEX_LIMIT:
                    _rational.divide(scratch_quantized, gcd)
                    scale = gcd | (np.int64(1) << SCALE_SHIFT)  # D = 1
                    grid_index = base_int
                    flags |= BLOCK_FLAG_SCALED
            # No common factor: search (not for subnormal steps, which no scaled grid has, and
            # whose 1 / step would overflow)
            elif range_span < xm.WIDE_RANGE and quant_exp >= SCALED_E_MIN:
                found, decimal, param, num, den, rational_index = _rational.detect_rational(
                    samples, lower_bound, upper_bound, math.ldexp(1.0, quant_exp), scratch_rational, scratch_index,
                )
                # Params.tolerance set the step: all of its allowance, N fine points of 2^(e - 4)
                tolerance_num, tolerance_fine_exp, tolerance_base = tolerance_grid(lower_bound, quant_exp, tolerance_step)
                if found and tolerance_num:
                    # An exact grid only if it is coarser than tolerance's step; the same step (decoded data
                    # found again) keeps tolerance's form, so decoded data re-encodes to itself
                    found_step = float(num) / den * (10.0 ** param if decimal else math.ldexp(1.0, param))
                    found = bool(found_step > tolerance_num * math.ldexp(1.0, tolerance_fine_exp) * (1.0 + 2.0 ** -40))
                if found:
                    flags |= BLOCK_FLAG_SCALED | (BLOCK_FLAG_DECIMAL if decimal else 0)
                    param_val = param
                    scale = np.int64(num) | (np.int64(den) << SCALE_SHIFT)
                    grid_index = rational_index
                    scratch_quantized[:] = scratch_rational
                elif tolerance_num:
                    inv_step = 1.0 / (tolerance_num * math.ldexp(1.0, tolerance_fine_exp))
                    origin = tolerance_base * math.ldexp(1.0, tolerance_fine_exp)
                    for sample_idx in range(samples.shape[0]):
                        scratch_quantized[sample_idx] = np.int32(math.floor((samples[sample_idx] - origin) * inv_step + 0.5))
                    flags |= BLOCK_FLAG_SCALED
                    param_val = tolerance_fine_exp
                    scale = tolerance_num | (np.int64(1) << SCALE_SHIFT)
                    grid_index = np.int64(tolerance_base)
    # Select difference order minimizing residual variance (a shift of q doesn't change it)
    selected_order = pick_order(scratch_quantized[:samples.shape[0]], orders_mask)
    grid_index, anchor = center(scratch_quantized, selected_order, flags, param_val, grid_index, anchor, scale)
    # Compute modular differences mod 2^16
    residual(scratch_quantized, selected_order, out_residuals)
    return flags | selected_order, param_val, grid_index, anchor, scale


@njit(nogil=True, cache=True)
def center(
    quantized: np.ndarray, order: int, flags: int, param: int, grid_index: np.int64, anchor: float, scale: np.int64,
) -> tuple[np.int64, float]:
    """Moves a block's anchor to its reference index c and its indices to q - c, signed 16-bit (SPEC §4).

    c is the rounded mean index for order 0 (the residuals are then the indices themselves,
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


@njit(inline="always")
def _store_anchor(
    out_value_anchors: np.ndarray,
    anchor_floats: np.ndarray,
    block_idx: int,
    flags: int,
    anchor: float,
    decimal_base: np.int64,
) -> None:
    """Writes a block's anchor: its decimal or scaled grid index, or the float64 bits of its power-of-two anchor.

    Args:
        out_value_anchors: Output 1D int64 array of length num_blocks receiving anchors.
        anchor_floats: out_value_anchors viewed as float64, made once per block group rather
            than per block.
        block_idx: Zero-based block index.
        flags: Block header byte; BLOCK_FLAG_DECIMAL or BLOCK_FLAG_SCALED select the integer anchor.
        anchor: The power-of-two grid's anchor, stored as float64 bits.
        decimal_base: Grid index of the block minimum on a decimal or scaled grid.
    """
    if flags & (BLOCK_FLAG_DECIMAL | BLOCK_FLAG_SCALED):
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
    """Allocates bit reductions across blocks to meet a per-block-group bit target.

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
        # Maximum bits block block_idx can give up without exceeding coarsest_exponents
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
    grids: int,
    orders_mask: int,
    scratch_quantized: np.ndarray,
    scratch_rational: np.ndarray,
    scratch_index: np.ndarray,
    scratch_held: np.ndarray,
    scratch_residuals: np.ndarray,
    out_block_flags: np.ndarray,
    out_grid_params: np.ndarray,
    out_grid_scales: np.ndarray,
    out_value_anchors: np.ndarray,
    out_residuals: np.ndarray,
    out_codes: np.ndarray,
) -> None:
    """Re-encodes blocks selected by target allocation and rolls back if bits do not drop.

    Args:
        samples: 1D float64 array of all samples in the block group.
        sample_offsets: 1D int64 array of sample offsets for each block.
        bit_reductions: 1D int64 array of bit reduction increments per block.
        block_exponents: 1D int64 array of current quantization exponents per block.
        base_lower_bounds: 1D float64 array of lower bounds per block.
        base_upper_bounds: 1D float64 array of upper bounds per block.
        estimated_bits_per_block: 1D float64 array of estimated bits before coarsening.
        grids: The grids level (GRIDS_BINARY, GRIDS_DECIMAL or GRIDS_RATIONAL).
        orders_mask: Bitmask of allowable predictor difference orders.
        scratch_quantized: 1D int32 scratch buffer.
        scratch_rational: 1D int32 scratch buffer for the rational search.
        scratch_index: 1D float64 scratch buffer for the rational search.
        scratch_held: 1D float64 scratch buffer for imputed non-finite samples.
        scratch_residuals: 1D int16 scratch buffer for rolling back residuals.
        out_block_flags: In-out 1D uint8 array of block flags.
        out_grid_params: In-out 1D int64 array of grid parameters.
        out_grid_scales: In-out 1D int64 array of grid scales.
        out_value_anchors: In-out 1D int64 array of value anchors.
        out_residuals: In-out 1D int16 array of residuals.
        out_codes: In-out 1D uint8 array of non-finite sample codes.
    """
    num_blocks = sample_offsets.shape[0] - 1
    anchor_floats = out_value_anchors.view(np.float64)
    for block_idx in range(num_blocks):
        if bit_reductions[block_idx] > 0:
            first_sample = sample_offsets[block_idx]
            block_len = sample_offsets[block_idx + 1] - first_sample
            block_residuals = out_residuals[first_sample:first_sample + block_len]
            prev_flags = out_block_flags[block_idx]
            prev_param = out_grid_params[block_idx]
            prev_scale = out_grid_scales[block_idx]
            prev_anchor = out_value_anchors[block_idx]
            backup_residuals = scratch_residuals[:block_len]
            backup_residuals[:] = block_residuals
            block_samples = samples[first_sample:first_sample + block_len]
            has_nonfinite = prev_flags & BLOCK_FLAG_NONFINITE
            if has_nonfinite:
                # Hold the coded samples (non-finite or escaped) as the first pass did
                hold_coded(block_samples, out_codes[first_sample:first_sample + block_len], scratch_held[:block_len])
                block_samples = scratch_held[:block_len]
            target_exp = min(block_exponents[block_idx] + bit_reductions[block_idx], E_MAX)
            (
                out_block_flags[block_idx], out_grid_params[block_idx], decimal_base, anchor,
                out_grid_scales[block_idx],
            ) = encode_block(
                block_samples,
                base_lower_bounds[block_idx],
                base_upper_bounds[block_idx],
                target_exp,
                grids,
                orders_mask,
                scratch_quantized[:block_len],
                scratch_rational[:block_len],
                scratch_index[:block_len],
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
                out_grid_scales[block_idx] = prev_scale
                out_value_anchors[block_idx] = prev_anchor
                block_residuals[:] = backup_residuals


@njit(nogil=True, cache=True)
def encode_group(
    samples: np.ndarray,
    sample_offsets: np.ndarray,
    ticks: np.ndarray,
    time_flags: np.ndarray,
    min_bits: int,
    max_bits: int,
    orders_mask: int,
    noise_factor: float,
    target_bits: float,
    grids: int,
    tolerance_exp: int,
    tolerance_step: float,
    out_block_flags: np.ndarray,
    out_grid_params: np.ndarray,
    out_grid_scales: np.ndarray,
    out_value_anchors: np.ndarray,
    out_residuals: np.ndarray,
    out_codes: np.ndarray,
    out_block_minima: np.ndarray,
    out_block_maxima: np.ndarray,
    out_block_means: np.ndarray,
) -> None:
    """Encodes all blocks of a block group: their rows and summary statistics.

    Blocks of at most SHORT_BLOCK_LEN samples skip the analysis: they take the finest step
    (a decimal grid if one is detected, else the power-of-two grid) and order 0, and stay
    outside the target. An empty block gets zero flags, grid parameter and anchor, and NaN
    statistics. A block with irregular times gets a noise floor only if it has a cadence, and
    estimates it on consecutive scans (_noise.cadence).

    Args:
        samples: 1D float64 array of every sample of the block group.
        sample_offsets: 1D int64 array of the blocks' sample offsets (num_blocks + 1).
        ticks: 1D int64 array of every sample's tick, or an empty array without a time axis.
        time_flags: 1D uint8 array of the blocks' time flags (BLOCK_FLAG_IRREGULAR_TIME, as the
            time rows set them), or an empty array without a time axis.
        min_bits: Hard lower bound on quantization bits.
        max_bits: Hard upper bound on quantization bits.
        orders_mask: Bitmask of allowable predictor difference orders.
        noise_factor: Noise floor multiplier (0.0 if disabled).
        target_bits: Target bits per sample (0.0 if disabled).
        grids: The grids level (GRIDS_BINARY, GRIDS_DECIMAL or GRIDS_RATIONAL; Params.grids).
        tolerance_exp: tolerance_exponent(Params.tolerance): the exponent tolerance allows.
        tolerance_step: 2 Params.tolerance, the scaled step it allows (0.0 without one).
        out_block_flags: Output 1D uint8 array of length num_blocks receiving header bytes.
        out_grid_params: Output 1D int64 array of length num_blocks receiving parameters.
        out_grid_scales: Output 1D int64 array of length num_blocks receiving grid scales (N | D << 16,
            0 on unscaled blocks).
        out_value_anchors: Output 1D int64 array of length num_blocks receiving anchors (float64 bits of
            the power-of-two anchor, or the decimal or scaled grid's integer index), at each
            block's reference index (center).
        out_residuals: Output 1D int16 array of every sample receiving residuals.
        out_codes: Output 1D uint8 array of every sample receiving sample codes (only
            flagged blocks' codes are written).
        out_block_minima: Output 1D float64 array of length num_blocks receiving block minima.
        out_block_maxima: Output 1D float64 array of length num_blocks receiving block maxima.
        out_block_means: Output 1D float64 array of length num_blocks receiving block means.
    """
    num_blocks = sample_offsets.shape[0] - 1
    max_len = 0
    for block_idx in range(num_blocks):
        max_len = max(max_len, sample_offsets[block_idx + 1] - sample_offsets[block_idx])
    scratch_quantized = np.empty(max_len, np.int32)
    scratch_rational = np.empty(max_len, np.int32)
    scratch_index = np.empty(max_len)
    scratch_held = np.empty(max_len)
    scratch_noise_diffs = np.empty(max_len)
    scratch_noise_weights = np.empty(max_len)
    timed = ticks.shape[0] > 0
    scratch_pivots = np.empty(CADENCE_SAMPLES, np.int64)
    no_codes = np.empty(0, np.uint8)
    no_ticks = ticks[:0]
    use_target = target_bits > 0
    block_exponents = np.empty(num_blocks, np.int64)
    coarsest_exponents = np.empty(num_blocks, np.int64)
    block_weights = np.zeros(num_blocks, np.int64)
    base_lower_bounds = np.empty(num_blocks)
    base_upper_bounds = np.empty(num_blocks)
    estimated_bits_per_block = np.zeros(num_blocks)
    budget_bits = np.zeros(num_blocks)
    scratch_budget = np.empty(max_len if use_target else 0, np.int16)
    anchor_floats = out_value_anchors.view(np.float64)
    for block_idx in range(num_blocks):
        first_sample = sample_offsets[block_idx]
        block_len = sample_offsets[block_idx + 1] - first_sample
        block_exponents[block_idx] = coarsest_exponents[block_idx] = 0
        out_grid_scales[block_idx] = 0
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
            if block_max > block_min:
                fine_exp = min(max(fine_exp, tolerance_exp), range_exponent(block_min, block_max, min_bits))
            decimal_exp, decimal_base = NO_DECIMAL, np.int64(0)
            if grids >= GRIDS_DECIMAL and block_max > block_min and block_max - block_min < xm.WIDE_RANGE:
                decimal_exp, decimal_base, fine_exp = decimal_grid(
                    block_samples, block_min, block_max, fine_exp, block_quantized
                )
            if decimal_exp != NO_DECIMAL:
                out_block_flags[block_idx] = BLOCK_FLAG_DECIMAL
                out_grid_params[block_idx] = decimal_exp
                out_value_anchors[block_idx], _ = center(
                    block_quantized, 0, BLOCK_FLAG_DECIMAL, decimal_exp, decimal_base, 0.0, np.int64(0)
                )
            else:
                out_block_flags[block_idx] = 0
                out_grid_params[block_idx] = fine_exp
                anchor = quantize_block(block_samples, block_min, block_max, fine_exp, block_quantized)
                _, anchor_floats[block_idx] = center(block_quantized, 0, 0, fine_exp, np.int64(0), anchor, np.int64(0))
            residual(block_quantized, 0, block_residuals)
            if has_nonfinite:
                out_block_flags[block_idx] |= BLOCK_FLAG_NONFINITE
            continue
        base_lower_bounds[block_idx] = block_min
        base_upper_bounds[block_idx] = block_max
        noise_sigma = noise_rho = 0.0
        # Only irregular times steer the noise floor: regular ones are as no times
        irregular = timed and (time_flags[block_idx] & BLOCK_FLAG_IRREGULAR_TIME) != 0
        if noise_factor > 0 and block_max > block_min and block_len >= NOISE_MIN_LEN:
            noise_sigma, noise_rho = block_noise(
                block_samples,
                block_codes if has_nonfinite else no_codes,
                ticks[first_sample:first_sample + block_len] if irregular else no_ticks,
                range_scale(block_min, block_max),
                scratch_pivots,
                scratch_noise_diffs,
                scratch_noise_weights,
                # A sigma under this can't coarsen the finest step: its block skips the gate's checks
                math.ldexp(1.0, range_exponent(block_min, block_max, max_bits)) / noise_factor,
            )
        # Outliers stored apart (escapes), when that refines the rest of the block (§5.1.1)
        if finite_count > 0 and block_len >= ESCAPE_MIN_LEN:
            escaped, bulk_min, bulk_max = escape_outliers(
                block_samples, scratch_index[:block_len], block_codes, has_nonfinite, block_mean, block_min, block_max,
                max(floor_exponent(noise_factor, noise_sigma, noise_rho), tolerance_exp),
                max_bits, grids,
                block_quantized,
            )
            if escaped:
                scratch_held[:block_len] = scratch_index[:block_len]
                block_samples = scratch_held[:block_len]
                block_min, block_max = bulk_min + 0.0, bulk_max + 0.0
                base_lower_bounds[block_idx], base_upper_bounds[block_idx] = block_min, block_max
                has_nonfinite = True
                # The first estimate saw the escaped samples (a dropout run can hide the noise, glitches on
                # both sides inflate it): again, without them
                if noise_factor > 0 and block_len >= NOISE_MIN_LEN and block_max > block_min:
                    noise_sigma, noise_rho = block_noise(
                        block_samples, block_codes, ticks[first_sample:first_sample + block_len] if irregular else no_ticks,
                        range_scale(block_min, block_max), scratch_pivots, scratch_noise_diffs, scratch_noise_weights,
                        math.ldexp(1.0, range_exponent(block_min, block_max, max_bits)) / noise_factor,
                    )
        planned_exp, block_tolerance_step = plan_block_step(
            block_samples, block_min, block_max, noise_sigma, noise_rho, noise_factor, min_bits, max_bits, tolerance_exp,
            tolerance_step,
        )
        block_exponents[block_idx] = planned_exp
        (
            out_block_flags[block_idx], out_grid_params[block_idx], decimal_base, anchor, out_grid_scales[block_idx],
        ) = encode_block(
            block_samples,
            block_min,
            block_max,
            planned_exp,
            grids,
            orders_mask,
            block_quantized,
            scratch_rational[:block_len],
            scratch_index[:block_len],
            block_residuals,
            block_tolerance_step,
        )
        _store_anchor(out_value_anchors, anchor_floats, block_idx, out_block_flags[block_idx], anchor, decimal_base)
        if has_nonfinite:
            out_block_flags[block_idx] |= BLOCK_FLAG_NONFINITE
        if use_target:
            # Every block counts from its planned exponent, also on an exact coarser grid (a decimal
            # grid, the index gcd's, a scaled grid), with the budget below in the same units (§5.8)
            coarsest_exponents[block_idx] = max(
                range_exponent(block_min, block_max, min_bits), block_exponents[block_idx]
            )
            estimated_bits_per_block[block_idx] = estimate_bits(block_residuals)
            # The budget counts a block on an exact coarser grid as at its planned exponent, so the
            # cap coarsens it as it would a block without, and decoded data, back on that grid,
            # re-encodes to itself. A step M * 2^planned (every power-of-two grid, N 2^e): the
            # residuals of M q, the indices on the planned grid (every power-of-two grid, a decimal
            # p >= 0, N 2^e); other grids: their nonzero residuals as if they had the log2 of the
            # step's ratio in low bits
            step = stored_step(out_block_flags[block_idx], out_grid_params[block_idx], out_grid_scales[block_idx])
            multiple = planned_multiple(step, planned_exp)
            if multiple == 1:
                budget_bits[block_idx] = estimated_bits_per_block[block_idx]
            elif multiple > 1:
                planned_quantized = scratch_rational[:block_len]
                for sample_idx in range(block_len):
                    planned_quantized[sample_idx] = block_quantized[sample_idx] * multiple
                planned_residuals = scratch_budget[:block_len]
                residual(planned_quantized, out_block_flags[block_idx] & BLOCK_FLAG_ORDER, planned_residuals)
                budget_bits[block_idx] = estimate_bits(planned_residuals)
            else:
                budget_bits[block_idx] = estimated_bits_per_block[block_idx] + max(
                    math.log2(step) - planned_exp, 0.0
                ) * (1.0 - zero_share(block_residuals))
            block_weights[block_idx] = block_len
    # Apply soft per-block-group bit target if enabled
    if use_target:
        bit_reductions = np.zeros(num_blocks, np.int64)
        if allocate_target(
            budget_bits,
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
                grids,
                orders_mask,
                scratch_quantized,
                scratch_rational,
                scratch_index,
                scratch_held,
                np.empty(max_len, np.int16),
                out_block_flags,
                out_grid_params,
                out_grid_scales,
                out_value_anchors,
                out_residuals,
                out_codes,
            )
