# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Overflow-safe numerical routines for subnormal and extreme IEEE-754 float64 magnitudes.

Handles blocks whose dynamic range approaches or exceeds ``2^1023`` (where ``max - min`` or
``sample - anchor`` overflows float64), whose quantization step is subnormal (where ``2^-e``
overflows float64), or whose sample sum overflows float64. Each routine matches the standard
float64 formula wherever that formula does not overflow.
"""

import math

import numpy as np
from numba import njit

WIDE_RANGE: float = 2.0**1023
"""Value span at or above which ``upper_bound - lower_bound`` may overflow float64."""

E_TINY: int = -1023
"""Power-of-two exponent below which ``2^-e`` overflows float64 (requires two-stage scaling)."""

E_HUGE: int = 1008
"""Power-of-two exponent at or above which ``sample - anchor`` may overflow float64."""

E_WIDE: int = 971
"""Power-of-two exponent at or above which ``anchor + q * 2^e`` may exceed ``DBL_MAX``."""

SCALE_LIMIT: int = 1000
"""Maximum power-of-two exponent magnitude applied in a single scaling pass."""

DBL_MAX: float = float(np.finfo(np.float64).max)
"""Largest finite IEEE-754 double-precision value."""


@njit(nogil=True, cache=True)
def half_range(lower_bound: float, upper_bound: float) -> float:
    """Computes ``(upper_bound - lower_bound) / 2`` without intermediate overflow.

    Args:
        lower_bound: Minimum finite value in the range.
        upper_bound: Maximum finite value in the range.

    Returns:
        Half of the span as a finite float64.
    """
    # Halving each endpoint before subtracting keeps the difference finite even near +/-DBL_MAX.
    return 0.5 * upper_bound - 0.5 * lower_bound


@njit(nogil=True, cache=True)
def mean_by_division(samples: np.ndarray, finite_count: int) -> float:
    """Computes the arithmetic mean of finite samples by dividing each term before summing.

    Fallback used when direct summation overflows float64 to ``+/-inf``.

    Args:
        samples: 1D float64 array of samples.
        finite_count: Number of finite values in ``samples`` (must be > 0).

    Returns:
        Finite float64 arithmetic mean of the finite samples.
    """
    accum_mean = 0.0
    for sample_idx in range(samples.shape[0]):
        if math.isfinite(samples[sample_idx]):
            accum_mean += samples[sample_idx] / finite_count
    return accum_mean


@njit(nogil=True, cache=True)
def _scaled(samples: np.ndarray, scale_factor: float) -> np.ndarray:
    """Multiplies ``samples`` by ``scale_factor`` in a separate non-fastmath kernel.

    Args:
        samples: 1D float64 array.
        scale_factor: Finite float64 power-of-two multiplier.

    Returns:
        Element-wise product ``samples * scale_factor``.
    """
    # Compiled without fastmath so LLVM cannot reassociate two-stage scaling factors into one
    # overflowing or subnormal multiplier.
    return samples * scale_factor


@njit(nogil=True, cache=True)
def prescale(samples: np.ndarray, scale_exp: int) -> tuple[np.ndarray, float]:
    """Splits scaling by ``2^-scale_exp`` into two normal power-of-two factors.

    Used when ``|scale_exp| > SCALE_LIMIT``, where ``2^-scale_exp`` would overflow to infinity
    or underflow to a subnormal.

    Args:
        samples: 1D float64 array.
        scale_exp: Target exponent such that the desired total factor is ``2^-scale_exp``.

    Returns:
        Tuple ``(scaled_samples, second_factor)`` satisfying
        ``scaled_samples * second_factor == samples * 2^-scale_exp``.
    """
    # Apply +/-SCALE_LIMIT first so both stages use normal powers of two with exact rounding.
    first_exponent = -SCALE_LIMIT if scale_exp > 0 else SCALE_LIMIT
    second_factor = math.ldexp(1.0, -scale_exp - first_exponent)
    scaled_samples = _scaled(samples, math.ldexp(1.0, first_exponent))
    return scaled_samples, second_factor


@njit(nogil=True, cache=True)
def quantize_huge_exponent(samples: np.ndarray, anchor: float, quant_exp: int, out_quantized: np.ndarray) -> None:
    """Quantizes samples for large exponents where ``sample - anchor`` may overflow float64.

    Evaluates ``floor((sample / 2 - anchor / 2) * 2^(1 - quant_exp) + 0.5)``, which produces
    the same rounded grid indices as ``(sample - anchor) * 2^-quant_exp`` without overflow.

    Args:
        samples: 1D float64 array of samples to quantize.
        anchor: Block anchor value.
        quant_exp: Power-of-two quantization exponent (at or above ``E_HUGE``).
        out_quantized: Preallocated 1D int32 array receiving quantized grid indices.
    """
    # Compute the difference at half scale where it cannot overflow, then multiply by 2^(1 - e).
    half_scale_factor = math.ldexp(1.0, 1 - quant_exp)
    half_anchor = 0.5 * anchor
    for sample_idx in range(samples.shape[0]):
        half_diff = 0.5 * samples[sample_idx] - half_anchor
        out_quantized[sample_idx] = np.int32(math.floor(half_diff * half_scale_factor + 0.5))


@njit(nogil=True, cache=True)
def dequantize_pow2_wide_exponent(
    quantized_samples: np.ndarray, anchor: float, quant_exp: int, out_samples: np.ndarray
) -> None:
    """Dequantizes power-of-two samples when ``anchor + q * 2^quant_exp`` could exceed ``DBL_MAX``.

    Evaluates the reconstruction at half scale and clamps to ``[-DBL_MAX, DBL_MAX]``, while
    preserving ``anchor`` verbatim when ``q == 0``.

    Args:
        quantized_samples: 1D int32 array of quantized grid indices ``q``.
        anchor: Block anchor value (the reconstructed value for ``q == 0``).
        quant_exp: Power-of-two quantization exponent (at or above ``E_WIDE``).
        out_samples: Preallocated 1D float64 array receiving reconstructed samples.
    """
    # Use half-step 2^(e - 1) and half-anchor so intermediate products stay finite, then double
    # and clamp to +/-DBL_MAX. Preserve anchor directly at q == 0 in case anchor is subnormal.
    half_step_size = math.ldexp(1.0, quant_exp - 1)
    half_anchor = 0.5 * anchor
    for sample_idx in range(quantized_samples.shape[0]):
        grid_offset = quantized_samples[sample_idx]
        reconstructed = max(min(2.0 * (half_anchor + grid_offset * half_step_size), DBL_MAX), -DBL_MAX)
        out_samples[sample_idx] = anchor if grid_offset == 0 else reconstructed
