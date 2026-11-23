# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Quantization grids, anchor snapping, and step planning for float64 block encoding.

Provides power-of-two exponent sizing, snapped and relative quantization, decimal grid
detection, noise-floor and absolute-tolerance step planning, and anchor centering shared
by ``_encoder`` and ``_escapes``.
"""

import math

import numpy as np
from numba import njit

from . import _extreme_magnitudes as xm
from . import _noise as _noise_mod
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
    SCALED_E_MAX,
    SCALED_E_MIN,
)

NO_DECIMAL: int = 99
"""Sentinel exponent indicating that no decimal grid coarser than the baseline step matched."""

GRIDS_BINARY: int = 0
"""Grid mode ``"binary"``: power-of-two grids only."""

GRIDS_DECIMAL: int = 1
"""Grid mode ``"decimal"``: power-of-two and decimal (``10^p``) grids."""

GRIDS_RATIONAL: int = 2
"""Grid mode ``"rational"``: power-of-two, decimal, and scaled rational grids."""

GRIDS_LEVELS: dict[str, int] = {
    "binary": GRIDS_BINARY,
    "decimal": GRIDS_DECIMAL,
    "rational": GRIDS_RATIONAL,
}
"""Mapping from ``Params.grids`` string names to ordered grid-capability levels."""

NOISE_GATE: float = _noise_mod.NOISE_GATE
"""Whiteness gate threshold: lag-1 autocorrelation share above this value enables the noise floor."""

NO_FLOOR_EXP: int = E_MIN - 64
"""Sentinel exponent representing an inactive noise or tolerance floor (finer than any valid step)."""

FLOOR_MARGIN: float = 1.02
"""Hysteresis ratio past a power of two within which a block already on the finer grid keeps it."""

TOLERANCE_STEP_BITS: int = 4
"""Subdivision bits for tolerance scaled steps: ``N * 2^(e - 4)`` with ``N`` in ``[17, 31]``."""

FLOAT64_EXACT_INT_LIMIT: float = 2.0**52
"""Magnitude bound below which float64 spacing is at most 0.5, keeping integer rounding exact."""

TRIM_STRIDE: int = 4
"""Subsampling stride used when estimating the trimmed mean anchor for order-0 blocks."""

TRIM_DEVIATIONS: float = 3.0
"""Inlier cutoff in mean absolute deviations around the mean for order-0 anchor centering."""

INT16_MAX: int = 32767
"""Maximum signed 16-bit quantized grid index."""

INT16_MIN: int = -32768
"""Minimum signed 16-bit quantized grid index."""


def tolerance_exponent(abs_tol: float) -> int:
    """Computes the coarsest power-of-two exponent whose half-step does not exceed ``abs_tol``.

    Args:
        abs_tol: Non-negative maximum allowed quantization error (0 disables the bound).

    Returns:
        Power-of-two exponent ``e`` with ``2^(e - 1) <= abs_tol`` clamped at ``E_MAX``,
        or ``NO_FLOOR_EXP`` when ``abs_tol <= 0``.
    """
    if not abs_tol > 0:
        return NO_FLOOR_EXP
    return min(math.floor(math.log2(abs_tol)) + 1, E_MAX)


def tolerance_step(abs_tol: float) -> float:
    """Computes the maximum step ``2 * abs_tol`` allowed by an absolute error bound.

    Args:
        abs_tol: Non-negative maximum allowed quantization error.

    Returns:
        Finite positive ``2.0 * abs_tol``, or ``0.0`` when ``abs_tol <= 0`` or ``2 * abs_tol``
        overflows float64.
    """
    if not abs_tol > 0:
        return 0.0
    step = 2.0 * float(abs_tol)
    return step if math.isfinite(step) else 0.0


@njit(nogil=True, cache=True)
def min_exponent_for_span(range_span: float, bits: int) -> int:
    """Computes the finest power-of-two exponent whose step fits ``range_span`` in ``bits`` bits.

    Guarantees ``round(range_span * 2^-exponent) < 2^bits`` for any positive ``range_span``.

    Args:
        range_span: Span of the finite values (``upper_bound - lower_bound``).
        bits: Maximum bit width permitted across the span.

    Returns:
        The finest power-of-two exponent fitting the span, or ``0`` for a non-positive span.
    """
    if not range_span > 0:
        return 0
    # Decompose span = mantissa * 2^scale_exp with mantissa in [0.5, 1.0); scale_exp - bits
    # puts the scaled span in [2^(bits - 1), 2^bits), bumping by 1 if rounding reaches 2^bits.
    _, scale_exp = math.frexp(range_span)
    target_exp = scale_exp - bits
    if math.floor(math.ldexp(range_span, -target_exp) + 0.5) >= (1 << bits):
        target_exp += 1
    return target_exp


@njit(nogil=True, cache=True)
def _unsnapped_exponent_for_range(lower_bound: float, upper_bound: float, bits: int) -> int:
    """Computes the finest unsnapped exponent fitting ``[lower_bound, upper_bound]`` in ``bits`` bits.

    Args:
        lower_bound: Minimum finite sample value in the block.
        upper_bound: Maximum finite sample value in the block.
        bits: Bit budget for the quantized range.

    Returns:
        Power-of-two exponent clamped to ``[E_MIN, E_MAX]``.
    """
    # Compute the span directly for normal ranges; halve endpoints when subtraction could overflow.
    range_span = upper_bound - lower_bound
    if range_span < xm.WIDE_RANGE:
        return max(min_exponent_for_span(range_span, bits), E_MIN)
    return min(min_exponent_for_span(xm.half_range(lower_bound, upper_bound), bits) + 1, E_MAX)


@njit(inline="always")
def grid_index(value: float, quant_exp: int) -> float:
    """Maps a float64 value to its nearest integer index on the absolute grid ``k * 2^quant_exp``.

    Uses round-to-nearest-even (``np.rint``) and returns the integer index as a float64.

    Args:
        value: Finite float64 sample value.
        quant_exp: Power-of-two quantization exponent.

    Returns:
        Nearest integer grid index represented as a float64.
    """
    if quant_exp >= xm.E_TINY:
        return float(np.rint(value * math.ldexp(1.0, -quant_exp)))
    # When quant_exp < E_TINY, 2^-quant_exp overflows float64; apply two exact power-of-two factors.
    return float(np.rint(value * math.ldexp(1.0, -xm.E_TINY) * math.ldexp(1.0, xm.E_TINY - quant_exp)))


@njit(nogil=True, cache=True)
def snapped_exponent_for_range(lower_bound: float, upper_bound: float, bits: int) -> int:
    """Computes the finest exponent whose absolute power-of-two grid fits the range in ``bits`` bits.

    The snapped grid fits when ``grid_index(upper_bound, e) - grid_index(lower_bound, e) <= L``,
    where ``L = 2^bits`` for ``bits < 16`` and ``65535`` for ``bits == 16``.

    Args:
        lower_bound: Minimum sample value in the block.
        upper_bound: Maximum sample value in the block.
        bits: Bit budget for the quantized range (``1..16``).

    Returns:
        Quantization exponent in ``[E_MIN, E_MAX]``, or ``0`` for a constant block.
    """
    # Snapping both endpoints to an absolute grid can widen the span by up to one step.
    # Start one level coarser than the unsnapped exponent (guaranteed to fit) and step finer
    # while the snapped index span stays within the bit budget.
    if not upper_bound > lower_bound:
        return 0
    max_index = (1 << bits) if bits < 16 else (1 << 16) - 1
    quant_exp = min(_unsnapped_exponent_for_range(lower_bound, upper_bound, bits) + 1, E_MAX)
    while quant_exp > E_MIN and (
        grid_index(upper_bound, quant_exp - 1) - grid_index(lower_bound, quant_exp - 1) <= max_index
    ):
        quant_exp -= 1
    return quant_exp


@njit(nogil=True, cache=True)
def snap_anchor(lower_bound: float, upper_bound: float, quant_exp: int) -> tuple[float, float]:
    """Computes the snapped grid anchor nearest ``lower_bound`` for a power-of-two block.

    Args:
        lower_bound: Minimum sample value in the block.
        upper_bound: Maximum sample value in the block.
        quant_exp: Power-of-two quantization exponent.

    Returns:
        Tuple ``(anchor, anchor_index)`` where ``anchor = anchor_index * 2^quant_exp`` and
        ``anchor_index = grid_index(lower_bound, quant_exp)``. For constant blocks or when the
        snapped anchor overflows float64, returns ``(lower_bound, NaN)`` to signal relative
        quantization.
    """
    if not upper_bound > lower_bound:
        return lower_bound, math.nan
    anchor_index = grid_index(lower_bound, quant_exp)
    anchor = math.ldexp(anchor_index, quant_exp)
    if not math.isfinite(anchor):
        return lower_bound, math.nan
    # Normalize -0.0 to +0.0 so the stored anchor matches canonical zero.
    return anchor + 0.0, anchor_index


@njit(nogil=True, cache=True)
def span_binary_exponent(lower_bound: float, upper_bound: float) -> int:
    """Computes the binary scale exponent ``k`` such that ``upper_bound - lower_bound`` lies in ``[2^(k-1), 2^k)``.

    Args:
        lower_bound: Minimum sample value in the block.
        upper_bound: Maximum sample value in the block (must exceed ``lower_bound``).

    Returns:
        Binary exponent ``k`` of the range span.
    """
    # Use half_range when direct subtraction could overflow float64 near +/-DBL_MAX.
    range_span = upper_bound - lower_bound
    if range_span < xm.WIDE_RANGE:
        return math.frexp(range_span)[1]
    return math.frexp(xm.half_range(lower_bound, upper_bound))[1] + 1


@njit(nogil=True, cache=True)
def floor_exponent(noise_factor: float, noise_sigma: float, noise_rho: float) -> int:
    """Computes the noise-floor quantization exponent when the whiteness gate passes.

    Args:
        noise_factor: Noise floor multiplier (0.0 disables the noise floor).
        noise_sigma: Estimated white-noise standard deviation.
        noise_rho: Whiteness gate statistic (values above ``NOISE_GATE`` pass).

    Returns:
        ``floor(log2(noise_factor * noise_sigma))`` when the noise floor is active and the
        whiteness gate passes, else ``NO_FLOOR_EXP``.
    """
    if noise_factor > 0 and noise_rho > NOISE_GATE and noise_sigma > 0:
        return math.floor(math.log2(noise_factor * noise_sigma))
    return NO_FLOOR_EXP


@njit(nogil=True, cache=True)
def preserve_finer_grid(samples: np.ndarray, planned_exp: int, floor_step: float) -> int:
    """Applies boundary hysteresis so already-quantized blocks stay a fixed point on re-encoding.

    Args:
        samples: 1D float64 array of block samples.
        planned_exp: Planned power-of-two exponent.
        floor_step: Unrounded noise-floor step ``noise_factor * noise_sigma``.

    Returns:
        ``planned_exp - 1`` if ``floor_step`` lies within ``FLOOR_MARGIN`` above ``2^planned_exp``
        and every sample is already an exact multiple of ``2^(planned_exp - 1)``; otherwise
        ``planned_exp``.
    """
    # Quantizing onto a grid raises noise sigma slightly (~0.26%), which can push a marginal floor
    # just over a power-of-two boundary on re-encode. Keep the finer exponent when the block is
    # already exact on it and the floor only barely crossed the boundary.
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
def plan_base_exponent(
    lower_bound: float,
    upper_bound: float,
    noise_sigma: float,
    noise_rho: float,
    noise_factor: float,
    min_bits: int,
    max_bits: int,
) -> int:
    """Plans a block's power-of-two quantization exponent from bit bounds and noise floor.

    Args:
        lower_bound: Minimum sample value in the block.
        upper_bound: Maximum sample value in the block.
        noise_sigma: Estimated white-noise standard deviation.
        noise_rho: Whiteness gate statistic (values above ``NOISE_GATE`` pass).
        noise_factor: Noise floor multiplier (0.0 disables the noise floor).
        min_bits: Hard minimum precision bits.
        max_bits: Hard maximum precision bits.

    Returns:
        Planned power-of-two exponent, or ``0`` for a constant block.
    """
    # Start at the finest exponent allowed by max_bits, coarsen to the noise floor if the
    # whiteness gate passes, and clamp to the coarsest exponent allowed by min_bits.
    if not upper_bound > lower_bound:
        return 0
    planned_exp = snapped_exponent_for_range(lower_bound, upper_bound, max_bits)
    planned_exp = max(planned_exp, floor_exponent(noise_factor, noise_sigma, noise_rho))
    return min(planned_exp, snapped_exponent_for_range(lower_bound, upper_bound, min_bits))


@njit(nogil=True, cache=True)
def plan_block_step(
    samples: np.ndarray,
    lower_bound: float,
    upper_bound: float,
    noise_sigma: float,
    noise_rho: float,
    noise_factor: float,
    min_bits: int,
    max_bits: int,
    tolerance_exp: int = NO_FLOOR_EXP,
    tol_step: float = 0.0,
) -> tuple[int, float]:
    """Plans a block's base quantization exponent and optional absolute-tolerance step.

    Args:
        samples: 1D float64 array of block samples.
        lower_bound: Minimum sample value in the block.
        upper_bound: Maximum sample value in the block.
        noise_sigma: Estimated white-noise standard deviation.
        noise_rho: Whiteness gate statistic.
        noise_factor: Noise floor multiplier (0.0 disables the noise floor).
        min_bits: Hard minimum precision bits.
        max_bits: Hard maximum precision bits.
        tolerance_exp: Power-of-two exponent from ``tolerance_exponent`` (or ``NO_FLOOR_EXP``).
        tol_step: Maximum step ``2 * abs_tol`` from ``tolerance_step`` (or ``0.0``).

    Returns:
        Tuple ``(planned_exp, block_tol_step)`` where ``block_tol_step`` is non-zero only when
        ``tolerance_exp`` actively coarsened the step beyond the base plan.
    """
    # Compute the base exponent from bit bounds and noise floor. If absolute_tolerance is coarser,
    # override even min_bits; otherwise apply floor hysteresis when the noise floor set the step.
    planned_exp = plan_base_exponent(
        lower_bound, upper_bound, noise_sigma, noise_rho, noise_factor, min_bits, max_bits
    )
    block_tol_step = 0.0
    if upper_bound > lower_bound and tolerance_exp > planned_exp:
        planned_exp = min(tolerance_exp, E_MAX)
        block_tol_step = tol_step
    if noise_rho > NOISE_GATE and upper_bound > lower_bound and planned_exp > tolerance_exp:
        planned_exp = max(
            preserve_finer_grid(samples, planned_exp, noise_factor * noise_sigma),
            snapped_exponent_for_range(lower_bound, upper_bound, max_bits),
        )
    return planned_exp, block_tol_step


@njit(nogil=True, cache=True)
def tolerance_scaled_grid(lower_bound: float, quant_exp: int, tol_step: float) -> tuple[np.int64, int, float]:
    """Computes a scaled binary grid closely matching an absolute-tolerance step ``tol_step``.

    Represented as ``N * 2^(quant_exp - TOLERANCE_STEP_BITS)`` with ``N`` in ``[17, 31]``,
    anchored on multiples of the step so decoded blocks re-encode identically.

    Args:
        lower_bound: Minimum sample value in the block.
        quant_exp: Planned power-of-two exponent.
        tol_step: Maximum allowed step ``2 * abs_tol``.

    Returns:
        Tuple ``(scale_num, fine_exp, anchor_index)``, or ``(0, fine_exp, 0.0)`` if the scaled
        tolerance grid is not applicable or its anchor exceeds ``INDEX_LIMIT``.
    """
    # Only engage when tol_step lies strictly between 2^quant_exp and 2^(quant_exp + 1) and the
    # subdivided fine exponent fits within the scaled binary exponent bounds.
    fine_exp = quant_exp - TOLERANCE_STEP_BITS
    if not (tol_step > 0 and quant_exp == math.floor(math.log2(tol_step))):
        return np.int64(0), fine_exp, 0.0
    if not (SCALED_E_MIN <= fine_exp <= SCALED_E_MAX):
        return np.int64(0), fine_exp, 0.0
    fine_step = math.ldexp(1.0, fine_exp)
    scale_num = np.int64(tol_step / fine_step)
    if scale_num <= (1 << TOLERANCE_STEP_BITS):
        return np.int64(0), fine_exp, 0.0
    # Anchor on the largest multiple of step <= lower_bound so reconstructed points are grid-aligned.
    step = float(scale_num) * fine_step
    anchor_index = math.floor(lower_bound / step) * scale_num
    if not (math.isfinite(anchor_index) and abs(anchor_index) < INDEX_LIMIT):
        return np.int64(0), fine_exp, 0.0
    return scale_num, fine_exp, float(anchor_index)


@njit(nogil=True, cache=True)
def on_tolerance_grid(samples: np.ndarray, scale_num: np.int64, fine_exp: int) -> bool:
    """Checks whether every sample is an exact multiple of ``scale_num * 2^fine_exp``.

    Args:
        samples: 1D float64 array of block samples.
        scale_num: Positive integer step multiplier ``N``.
        fine_exp: Fine power-of-two exponent.

    Returns:
        True if every sample lies exactly on the scaled grid, False otherwise.
    """
    for sample_idx in range(samples.shape[0]):
        scaled = math.ldexp(samples[sample_idx], -fine_exp)
        if scaled != math.floor(scaled) or not math.isfinite(scaled) or abs(scaled) >= 2.0 * FLOAT64_EXACT_INT_LIMIT:
            return False
        if np.int64(scaled) % scale_num != 0:
            return False
    return True


@njit(nogil=True, cache=True)
def on_binary_grid(samples: np.ndarray, quant_exp: int) -> bool:
    """Checks whether every sample is an exact multiple of ``2^quant_exp``.

    Args:
        samples: 1D float64 array of block samples.
        quant_exp: Power-of-two quantization exponent.

    Returns:
        True if every sample lies exactly on ``k * 2^quant_exp``, False otherwise.
    """
    for sample_idx in range(samples.shape[0]):
        scaled = math.ldexp(samples[sample_idx], -quant_exp)
        if scaled != math.floor(scaled) or not math.isfinite(scaled):
            return False
    return True


@njit(nogil=True, cache=True)
def moved_off_tolerance_grid(
    samples: np.ndarray,
    scale_num: np.int64,
    fine_exp: int,
    decimal_exp: int,
    anchor_index: np.int64,
    quantized: np.ndarray,
) -> bool:
    """Checks whether a detected decimal grid would perturb samples already exact on a tolerance grid.

    Args:
        samples: 1D float64 array of block samples.
        scale_num: Tolerance grid step multiplier ``N``.
        fine_exp: Tolerance grid fine exponent.
        decimal_exp: Candidate decimal exponent ``p``.
        anchor_index: Candidate decimal grid anchor index ``K0``.
        quantized: 1D int32 array of candidate decimal grid offsets ``q``.

    Returns:
        True if every sample is exact on ``k * scale_num * 2^fine_exp`` and at least one sample
        would decode to a different float64 under the decimal grid.
    """
    if not on_tolerance_grid(samples, scale_num, fine_exp):
        return False
    divide = decimal_exp < 0
    power = 10.0 ** abs(decimal_exp)
    moved = False
    for sample_idx in range(samples.shape[0]):
        index = float(anchor_index + quantized[sample_idx])
        moved |= (index / power if divide else index * power) != samples[sample_idx]
    return moved


@njit(nogil=True, cache=True)
def find_decimal_grid(
    samples: np.ndarray,
    lower_bound: float,
    upper_bound: float,
    pow2_step: float,
    out_quantized: np.ndarray,
) -> tuple[int, np.int64]:
    """Finds the coarsest decimal step ``10^p > pow2_step`` fitting all samples within ``pow2_step / 4``.

    Args:
        samples: 1D float64 array of block samples.
        lower_bound: Minimum sample value.
        upper_bound: Maximum sample value.
        pow2_step: Baseline power-of-two step ``2^e``.
        out_quantized: Output 1D int32 array populated with decimal grid offsets on match.

    Returns:
        Tuple ``(decimal_exp, anchor_index)`` where ``decimal_exp`` is in ``[P_MIN, P_MAX]``
        and ``anchor_index = floor(lower_bound * 10^-decimal_exp + 0.5)``, or
        ``(NO_DECIMAL, 0)`` if no qualifying decimal grid matches.
    """
    # Search downward from the coarsest decimal exponent that can span the range (with 2*tol
    # slack for float rounding) down to the finest 10^p strictly coarser than pow2_step.
    # Stop early if scaled magnitudes reach 2^52, where float64 spacing exceeds 0.5.
    tolerance = 0.25 * pow2_step
    max_abs_val = max(abs(lower_bound), abs(upper_bound))
    range_span = upper_bound - lower_bound
    decimal_exp = min(math.floor(math.log10(range_span + 2.0 * tolerance)), P_MAX)
    while decimal_exp >= P_MIN and 10.0**decimal_exp > pow2_step:
        inv_decimal_step = 10.0**-decimal_exp
        if not max_abs_val * inv_decimal_step < FLOAT64_EXACT_INT_LIMIT:
            break
        scaled_tolerance = tolerance * inv_decimal_step
        anchor_index = np.int64(math.floor(lower_bound * inv_decimal_step + 0.5))
        grid_matches = True
        for sample_idx in range(samples.shape[0]):
            scaled_sample = samples[sample_idx] * inv_decimal_step
            rounded_grid_point = math.floor(scaled_sample + 0.5)
            # Abort on the first off-grid sample; ordinary binary data fails immediately.
            if abs(scaled_sample - rounded_grid_point) > scaled_tolerance:
                grid_matches = False
                break
            out_quantized[sample_idx] = np.int32(np.int64(rounded_grid_point) - anchor_index)
        if grid_matches:
            return decimal_exp, anchor_index
        decimal_exp -= 1
    return NO_DECIMAL, np.int64(0)


@njit(nogil=True, cache=True)
def coarse_binary_exponent(samples: np.ndarray, quant_exp: int, decimal_step: float) -> int:
    """Finds the coarsest exact power-of-two grid holding ``samples`` if coarser than ``decimal_step``.

    Args:
        samples: 1D float64 array of a non-constant block's samples.
        quant_exp: Baseline power-of-two exponent.
        decimal_step: Candidate decimal step ``10^p > 2^quant_exp``.

    Returns:
        Coarsest exponent ``e`` with ``2^e > decimal_step`` such that every sample is an exact
        multiple of ``2^e``; otherwise ``quant_exp``.
    """
    # Values on a coarse binary grid (e.g. 0.25) also match finer decimal grids (0.01), which
    # introduces an unnecessary odd factor (x25). Check if all samples are exact multiples of
    # the smallest power of two above decimal_step, then fold any shared trailing zero bits.
    coarse_exp = quant_exp + 1
    while math.ldexp(1.0, coarse_exp) <= decimal_step:
        coarse_exp += 1
    inv_step = math.ldexp(1.0, -coarse_exp)
    index_bits = np.int64(0)
    for sample_idx in range(samples.shape[0]):
        scaled = samples[sample_idx] * inv_step
        if scaled != math.floor(scaled):
            return quant_exp
        index_bits |= np.int64(scaled)
    # Non-constant blocks have at least one non-zero scaled index, so this loop always terminates.
    while index_bits & 1 == 0:
        index_bits >>= 1
        coarse_exp += 1
    return coarse_exp


@njit(nogil=True, cache=True)
def moved_off_binary_grid(
    samples: np.ndarray,
    binary_exp: int,
    decimal_exp: int,
    anchor_index: np.int64,
    quantized: np.ndarray,
) -> bool:
    """Checks whether ``samples`` are exact on ``2^binary_exp`` and the decimal grid would perturb them.

    Args:
        samples: 1D float64 array of block samples.
        binary_exp: Planned power-of-two exponent.
        decimal_exp: Candidate decimal exponent ``p``.
        anchor_index: Candidate decimal grid anchor index ``K0``.
        quantized: 1D int32 array of candidate decimal grid offsets ``q``.

    Returns:
        True if every sample is an exact multiple of ``2^binary_exp`` and at least one sample
        would reconstruct to a different float64 on the decimal grid.
    """
    inv_step = math.ldexp(1.0, -binary_exp)
    divide = decimal_exp < 0
    power = 10.0 ** abs(decimal_exp)
    moved = False
    for sample_idx in range(samples.shape[0]):
        sample = samples[sample_idx]
        scaled = sample * inv_step
        # Stop at the first non-binary sample (typically sample 0 for decimal data).
        if scaled != math.floor(scaled):
            return False
        index = float(anchor_index + quantized[sample_idx])
        moved |= (index / power if divide else index * power) != sample
    return moved


@njit(nogil=True, cache=True)
def choose_decimal_or_binary_grid(
    samples: np.ndarray,
    lower_bound: float,
    upper_bound: float,
    quant_exp: int,
    out_quantized: np.ndarray,
) -> tuple[int, np.int64, int]:
    """Selects a decimal grid when appropriate, or returns a binary grid exponent.

    Prefers an exact power-of-two grid coarser than the detected decimal grid, and rejects a
    decimal grid that would perturb samples already exact on ``2^quant_exp`` (preserving fixed
    points on re-encoding).

    Args:
        samples: 1D float64 array of a non-constant block's samples (span below ``2^1023``).
        lower_bound: Minimum sample value.
        upper_bound: Maximum sample value.
        quant_exp: Planned power-of-two exponent.
        out_quantized: Output 1D int32 array receiving decimal offsets when a decimal grid is chosen.

    Returns:
        Tuple ``(decimal_exp, anchor_index, binary_exp)``. When a decimal grid is selected,
        ``decimal_exp != NO_DECIMAL`` and ``out_quantized`` holds its offsets. Otherwise
        ``decimal_exp == NO_DECIMAL``, ``anchor_index == 0``, and ``binary_exp`` is ``quant_exp``
        or a coarser exact binary exponent.
    """
    # Detect the coarsest matching decimal grid, then guard against two binary-grid cases:
    # (1) samples already lie on an even coarser binary grid, or (2) samples are exact on
    # 2^quant_exp and the decimal grid's quarter-step slack would move them.
    decimal_exp, anchor_index = find_decimal_grid(
        samples, lower_bound, upper_bound, math.ldexp(1.0, quant_exp), out_quantized
    )
    if decimal_exp == NO_DECIMAL:
        return NO_DECIMAL, np.int64(0), quant_exp
    binary_exp = coarse_binary_exponent(samples, quant_exp, 10.0**decimal_exp)
    # When decimal_exp >= 0 and quant_exp <= decimal_exp + 1, any multiple of 2^quant_exp within
    # 2^quant_exp / 4 of a multiple of 10^decimal_exp must already equal that multiple.
    could_move = decimal_exp < 0 or quant_exp > decimal_exp + 1
    if binary_exp != quant_exp or (
        could_move and moved_off_binary_grid(samples, quant_exp, decimal_exp, anchor_index, out_quantized)
    ):
        return NO_DECIMAL, np.int64(0), binary_exp
    return decimal_exp, anchor_index, quant_exp


@njit(nogil=True, cache=True)
def quantize_relative(
    samples: np.ndarray, lower_bound: float, quant_exp: int, out_quantized: np.ndarray
) -> None:
    """Quantizes samples onto a power-of-two grid anchored directly at ``lower_bound``.

    Used for blocks where ``snap_anchor`` returns a NaN anchor index (constant blocks and
    blocks near ``+/-DBL_MAX``).

    Args:
        samples: 1D float64 array of block samples.
        lower_bound: Anchor value (block minimum).
        quant_exp: Power-of-two quantization exponent.
        out_quantized: Output 1D int32 array receiving quantized grid offsets.
    """
    # Use half-scale subtraction for huge exponents where sample - lower_bound can overflow.
    if quant_exp >= xm.E_HUGE:
        xm.quantize_huge_exponent(samples, lower_bound, quant_exp, out_quantized)
    else:
        step_inv = math.ldexp(1.0, -quant_exp)
        for sample_idx in range(samples.shape[0]):
            diff = samples[sample_idx] - lower_bound
            out_quantized[sample_idx] = np.int32(math.floor(diff * step_inv + 0.5))


@njit(nogil=True, cache=True)
def quantize_snapped(
    samples: np.ndarray, quant_exp: int, anchor_index: float, out_quantized: np.ndarray
) -> None:
    """Quantizes samples onto the absolute power-of-two grid offset from ``anchor_index``.

    Each sample becomes ``grid_index(sample, quant_exp) - anchor_index``, mapping the block
    minimum to 0 and all samples to non-negative integers.

    Args:
        samples: 1D float64 array of block samples.
        quant_exp: Power-of-two quantization exponent.
        anchor_index: Anchor's grid index from ``snap_anchor``.
        out_quantized: Output 1D int32 array receiving quantized grid offsets.
    """
    # Hoist the single-step reciprocal for normal exponents; fall back to two-stage scaling
    # via grid_index for subnormal steps below E_TINY.
    if quant_exp >= xm.E_TINY:
        step_inv = math.ldexp(1.0, -quant_exp)
        for sample_idx in range(samples.shape[0]):
            out_quantized[sample_idx] = np.int32(np.rint(samples[sample_idx] * step_inv) - anchor_index)
    else:
        for sample_idx in range(samples.shape[0]):
            out_quantized[sample_idx] = np.int32(grid_index(samples[sample_idx], quant_exp) - anchor_index)


@njit(nogil=True, cache=True)
def quantize_power_of_two(
    samples: np.ndarray, lower_bound: float, upper_bound: float, quant_exp: int, out_quantized: np.ndarray
) -> float:
    """Quantizes a block onto its power-of-two grid and returns the block's float64 anchor.

    Uses an absolute snapped grid when ``snap_anchor`` succeeds, otherwise quantizes relative
    to ``lower_bound``.

    Args:
        samples: 1D float64 array of block samples.
        lower_bound: Minimum sample value in the block.
        upper_bound: Maximum sample value in the block.
        quant_exp: Power-of-two quantization exponent.
        out_quantized: Output 1D int32 array receiving quantized grid offsets.

    Returns:
        Block anchor value as a float64.
    """
    anchor, anchor_index = snap_anchor(lower_bound, upper_bound, quant_exp)
    if math.isnan(anchor_index):
        quantize_relative(samples, lower_bound, quant_exp, out_quantized)
    else:
        quantize_snapped(samples, quant_exp, anchor_index, out_quantized)
    return anchor


@njit(inline="always")
def common_trailing_zeros(gcd: int | np.integer, anchor_index: np.int64) -> int:
    """Counts trailing zero bits shared by ``gcd`` (positive) and ``anchor_index``.

    Args:
        gcd: Positive integer GCD of block grid offsets.
        anchor_index: Integer grid index of the block anchor.

    Returns:
        Number of common trailing zero bits.
    """
    bits = gcd | anchor_index if anchor_index else gcd
    twos = 0
    while bits & 1 == 0:
        bits >>= 1
        twos += 1
    return twos


@njit(inline="always")
def block_grid_step(flags: int, grid_exp: int, grid_scale: np.int64) -> float:
    """Computes the effective quantization step of an encoded block.

    Args:
        flags: Block flags byte (``BLOCK_FLAG_DECIMAL``, ``BLOCK_FLAG_SCALED``).
        grid_exp: Stored grid exponent (power-of-two ``e`` or decimal ``p``).
        grid_scale: Packed ``N | (D << 16)`` scale (ignored when unscaled).

    Returns:
        Grid step size ``2^e``, ``10^p``, or ``(N / D)`` times either.
    """
    step = 10.0**grid_exp if flags & BLOCK_FLAG_DECIMAL else math.ldexp(1.0, grid_exp)
    if flags & BLOCK_FLAG_SCALED:
        step *= (grid_scale & SCALE_MASK) / (grid_scale >> SCALE_SHIFT)
    return step


@njit(inline="always")
def step_multiple_of_planned(step: float, planned_exp: int) -> int:
    """Expresses ``step`` as an integer multiple of ``2^planned_exp`` when possible.

    Args:
        step: Block's effective grid step.
        planned_exp: Block's planned power-of-two exponent.

    Returns:
        Integer multiple ``M`` in ``[1, 65536]`` if ``step == M * 2^planned_exp`` exactly,
        otherwise ``0``.
    """
    ratio = math.ldexp(step, -planned_exp)
    return int(ratio) if ratio == math.floor(ratio) and 1 <= ratio <= SCALE_MASK + 1 else 0


@njit(nogil=True, cache=True)
def center_anchor(
    quantized: np.ndarray,
    order: int,
    flags: int,
    grid_exp: int,
    anchor_index: np.int64,
    anchor: float,
    grid_scale: np.int64,
) -> tuple[np.int64, float]:
    """Shifts a block's anchor to its reference index and centers ``quantized`` into signed int16.

    For order 0, targets the trimmed mean index so residuals center around 0; for orders 1-3,
    targets ``quantized[0]`` so the first residual is 0. Clamps the shift so every shifted
    index lies in ``[-32768, 32767]`` and the updated anchor stays exact.

    Args:
        quantized: 1D int32 array of non-negative grid offsets (``0..65535``), shifted in place.
        order: Selected predictor difference order (``0..3``).
        flags: Block flags byte.
        grid_exp: Block grid exponent.
        anchor_index: Integer anchor index ``K0`` for decimal or scaled grids.
        anchor: Float64 anchor value for unscaled power-of-two grids.
        grid_scale: Packed ``N | (D << 16)`` scale (0 if unscaled).

    Returns:
        Tuple ``(new_anchor_index, new_anchor)`` after applying the chosen shift.
    """
    # 1. Scan max index and halved sum in 32K chunks (stays in int32 for SIMD vectorization).
    # 2. Pick the target shift: trimmed mean on a strided subsample for order 0, else q[0],
    #    clamped to [max_q - 32767, min(max_q, 32768)].
    # 3. Advance the integer or float64 anchor while preserving exactness, then subtract shift.
    num_samples = quantized.shape[0]
    if num_samples == 0:
        return anchor_index, anchor
    max_q = np.int32(0)
    total = np.int64(0)
    # Sum halved indices in 2^15-sample chunks so the inner accumulator stays in int32.
    for chunk_start in range(0, num_samples, 1 << 15):
        chunk = quantized[chunk_start : min(chunk_start + (1 << 15), num_samples)]
        chunk_total = np.int32(0)
        chunk_max = np.int32(0)
        for sample_idx in range(chunk.shape[0]):
            chunk_max = max(chunk_max, chunk[sample_idx])
            chunk_total += chunk[sample_idx] >> 1
        max_q = max(max_q, chunk_max)
        total += np.int64(chunk_total)
    max_index = np.int64(max_q)
    low, high = max(max_index - INT16_MAX, np.int64(0)), min(max_index, np.int64(-INT16_MIN))
    wanted = (2 * total + num_samples // 2) // num_samples if order == 0 else np.int64(quantized[0])
    if order == 0:
        # Refine with a trimmed mean over every TRIM_STRIDE-th sample so spikes do not pull the center.
        mean = np.int32(wanted)
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
        scale_num = grid_scale & SCALE_MASK if flags & BLOCK_FLAG_SCALED else np.int64(1)
        # Keep the integer anchor strictly below INDEX_LIMIT (2^52) so reconstruction stays exact.
        if abs(anchor_index + scale_num * shift) >= INDEX_LIMIT:
            shift = low
        anchor_index = anchor_index + scale_num * shift
    else:
        raw_anchor_index = math.ldexp(anchor, -grid_exp)
        if raw_anchor_index == math.floor(raw_anchor_index) and abs(raw_anchor_index) < FLOAT64_EXACT_INT_LIMIT:
            anchor = math.ldexp(raw_anchor_index + float(shift), grid_exp) + 0.0
        else:
            # Unsnapped or huge-index anchor: shift only by the minimum required for int16 fit.
            shift = low
            if shift:
                anchor = 2.0 * (0.5 * anchor + float(shift) * math.ldexp(1.0, grid_exp - 1))
    if shift:
        shift32 = np.int32(shift)
        for sample_idx in range(num_samples):
            quantized[sample_idx] -= shift32
    return anchor_index, anchor
