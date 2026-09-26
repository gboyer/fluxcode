# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Overflow-safe operations for subnormal and extreme float64 magnitudes.

Provides specialized numerical routines for blocks spanning subnormal ranges,
ranges near or exceeding 2^1023 (where `hi - lo` overflows float64), and sums
whose accumulated values exceed float64 capacity. Each routine gives the same
result as the plain float64 formula wherever that formula doesn't overflow
(the mean fallback approximately: it sums x / cnt).
"""

import math

import numpy as np
from numba import njit

WIDE_RANGE: float = 2.0 ** 1023
"""Range at or above which hi - lo may overflow float64 (measured as hi/2 - lo/2 instead)."""

E_TINY: int = -1023
"""Exponent below which the inverse quantization step 2^-e overflows float64."""

E_HUGE: int = 1008
"""Exponent at or above which sample differences x - lo can overflow float64."""

E_WIDE: int = 971
"""Decoder exponent threshold where lo + q * 2^e can exceed DBL_MAX."""

SCALE_LIMIT: int = 1000
"""Maximum exponent magnitude for single-step power-of-two scaling."""

DBL_MAX: float = float(np.finfo(np.float64).max)
"""Maximum finite IEEE-754 double-precision floating-point value."""


@njit(nogil=True, cache=True)
def half_range(lower_bound: float, upper_bound: float) -> float:
    """Computes (hi - lo) / 2 without intermediate overflow.

    Halving is exact for any double that isn't subnormal. In a range this wide
    (used only at or above 2^1023), a subnormal side is far below the other
    side's ulp, so the result is (hi - lo) / 2 rounded once.

    Args:
        lower_bound: Lower bound of the range.
        upper_bound: Upper bound of the range.

    Returns:
        Half of the range (hi - lo) / 2 as a finite float64.
    """
    # 0.5 * hi - 0.5 * lo never overflows
    return 0.5 * upper_bound - 0.5 * lower_bound


@njit(nogil=True, cache=True)
def mean_by_division(samples: np.ndarray, finite_count: int) -> float:
    """Computes the arithmetic mean by dividing each finite sample before summing.

    Used when direct summation of samples overflows float64 to +/-inf.

    Args:
        samples: 1D float64 array of samples.
        finite_count: Number of finite samples present in samples (must be > 0).

    Returns:
        Arithmetic mean of the finite samples as a finite float64.
    """
    accum_mean = 0.0
    for sample_idx in range(samples.shape[0]):
        if math.isfinite(samples[sample_idx]):
            # Accumulate pre-divided terms to prevent overflow
            accum_mean += samples[sample_idx] / finite_count
    return accum_mean


@njit(nogil=True, cache=True)
def _scaled(samples: np.ndarray, scale_factor: float) -> np.ndarray:
    """Multiplies an array by a scalar factor outside fastmath optimizations.

    Compiled without fastmath to prevent LLVM from reassociating multi-step
    scaling factors into a single overflowing multiplier.

    Args:
        samples: 1D float64 array.
        scale_factor: Finite float64 scale factor.

    Returns:
        Element-wise product samples * scale_factor.
    """
    # Preserves separate scaling pass so LLVM does not fold factors
    return samples * scale_factor


@njit(nogil=True, cache=True)
def prescale(samples: np.ndarray, scale_exp: int) -> tuple[np.ndarray, float]:
    """Splits extreme power-of-two scaling into two safe normal factors.

    Used when |scale_exp| > SCALE_LIMIT, where direct evaluation of
    2^-scale_exp would overflow to infinity or underflow to a subnormal.

    Args:
        samples: 1D float64 array.
        scale_exp: Total power-of-two exponent to scale by.

    Returns:
        A tuple (scaled_samples, scale_factor_2) such that
        scaled_samples * scale_factor_2 == samples * 2^-scale_exp.
    """
    # Clamp first stage to +/-1000 to keep scale factor normal
    first_exponent = -SCALE_LIMIT if scale_exp > 0 else SCALE_LIMIT
    second_factor = math.ldexp(1.0, -scale_exp - first_exponent)
    # Apply first scale factor outside fastmath
    scaled_samples = _scaled(samples, math.ldexp(1.0, first_exponent))
    return scaled_samples, second_factor


@njit(nogil=True, cache=True)
def quantize_tiny(
    samples: np.ndarray, lower_bound: float, quant_exp: int, out_quantized: np.ndarray
) -> None:
    """Quantizes samples for subnormal exponents where 2^-e overflows float64.

    Applies the scale factor 2^-e as two successive exact power-of-two
    multiplications: s1 = 2^1023 and s2 = 2^(-e - 1023).

    Args:
        samples: 1D float64 array of samples to quantize.
        lower_bound: Minimum sample value in the block.
        quant_exp: Quantization exponent strictly less than E_TINY (-1023).
        out_quantized: Output 1D int32 array receiving quantized values.
    """
    # Split 2^-e into two exact normal power-of-two factors
    scale_factor_1 = math.ldexp(1.0, -E_TINY)
    scale_factor_2 = math.ldexp(1.0, E_TINY - quant_exp)
    for sample_idx in range(samples.shape[0]):
        # Multiply successively to avoid intermediate overflow
        diff = samples[sample_idx] - lower_bound
        out_quantized[sample_idx] = np.int32(math.floor(diff * scale_factor_1 * scale_factor_2 + 0.5))


@njit(nogil=True, cache=True)
def quantize_huge(
    samples: np.ndarray, lower_bound: float, quant_exp: int, out_quantized: np.ndarray
) -> None:
    """Quantizes samples for wide exponents where x - lo can overflow float64.

    Evaluates the quantization formula at half-scale: (x/2 - lo/2) * 2^(1 - e),
    which produces the exact same rounding as (x - lo) * 2^-e.

    Args:
        samples: 1D float64 array of samples to quantize.
        lower_bound: Minimum sample value in the block.
        quant_exp: Quantization exponent at or above E_HUGE (1008).
        out_quantized: Output 1D int32 array receiving quantized values.
    """
    # Step at half-scale: 2^(1 - e) = 2 * 2^-e
    half_scale_factor = math.ldexp(1.0, 1 - quant_exp)
    half_lower_bound = 0.5 * lower_bound
    for sample_idx in range(samples.shape[0]):
        # Evaluate difference at half scale: 0.5 * x[i] - 0.5 * lo
        scaled_diff = 0.5 * samples[sample_idx] - half_lower_bound
        out_quantized[sample_idx] = np.int32(math.floor(scaled_diff * half_scale_factor + 0.5))


@njit(nogil=True, cache=True)
def dequantize_pow2_wide(
    quantized_samples: np.ndarray, lower_bound: float, quant_exp: int, out_samples: np.ndarray
) -> None:
    """Dequantizes samples for wide exponents where lo + q * 2^e can exceed DBL_MAX.

    Evaluates reconstruction at half-scale and clamps to DBL_MAX to prevent
    rounding to infinity. Preserves lo exactly when q is zero.

    Args:
        quantized_samples: 1D int32 array of quantized integers.
        lower_bound: Block minimum float64 value.
        quant_exp: Quantization exponent at or above E_WIDE (971).
        out_samples: Output 1D float64 array receiving reconstructed samples.
    """
    # Half-step 2^(e - 1) avoids intermediate overflow of q * 2^e
    half_step_size = math.ldexp(1.0, quant_exp - 1)
    half_lower_bound = 0.5 * lower_bound
    for sample_idx in range(quantized_samples.shape[0]):
        grid_offset = quantized_samples[sample_idx]
        # Reconstruct at half-scale and clamp to maximum finite double
        reconstructed_val = min(2.0 * (half_lower_bound + grid_offset * half_step_size), DBL_MAX)
        # Preserve lo exactly when q == 0 (lo / 2 may round if lo is subnormal)
        out_samples[sample_idx] = lower_bound if grid_offset == 0 else reconstructed_val
