# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Overflow-safe operations for subnormal and extreme float64 magnitudes.

Provides specialized numerical routines for blocks spanning subnormal ranges,
ranges near or exceeding 2^1023 (where maximum - minimum overflows float64), and sums
whose accumulated values exceed float64 capacity. Each routine gives the same
result as the plain float64 formula wherever that formula doesn't overflow
(the mean fallback approximately: it sums each sample divided by the count).
"""

import math

import numpy as np
from numba import njit

WIDE_RANGE: float = 2.0 ** 1023
"""Range at or above which maximum - minimum may overflow float64 (measured by halves instead)."""

E_TINY: int = -1023
"""Exponent below which the inverse quantization step 2^-e overflows float64 (the encoder
scales in two steps there)."""

E_HUGE: int = 1008
"""Exponent at or above which sample - minimum can overflow float64."""

E_WIDE: int = 971
"""Decoder exponent at or above which minimum + q * 2^exponent can exceed DBL_MAX."""

SCALE_LIMIT: int = 1000
"""Maximum exponent magnitude for single-step power-of-two scaling."""

DBL_MAX: float = float(np.finfo(np.float64).max)
"""Maximum finite IEEE-754 double-precision floating-point value."""


@njit(nogil=True, cache=True)
def half_range(lower_bound: float, upper_bound: float) -> float:
    """Computes (upper_bound - lower_bound) / 2 without intermediate overflow.

    Halving is exact for any double that isn't subnormal. In a range this wide
    (used only at or above 2^1023), a subnormal bound is far below the other
    bound's ulp, so the result is the exact half-range rounded once.

    Args:
        lower_bound: Lower bound of the range.
        upper_bound: Upper bound of the range.

    Returns:
        Half of the range as a finite float64.
    """
    # 0.5 * upper_bound - 0.5 * lower_bound never overflows
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
        A tuple (scaled_samples, second_factor) such that
        scaled_samples * second_factor == samples * 2^-scale_exp.
    """
    # Clamp first stage to +/-1000 to keep scale factor normal
    first_exponent = -SCALE_LIMIT if scale_exp > 0 else SCALE_LIMIT
    second_factor = math.ldexp(1.0, -scale_exp - first_exponent)
    # Apply first scale factor outside fastmath
    scaled_samples = _scaled(samples, math.ldexp(1.0, first_exponent))
    return scaled_samples, second_factor


@njit(nogil=True, cache=True)
def quantize_huge(
    samples: np.ndarray, lower_bound: float, quant_exp: int, out_quantized: np.ndarray
) -> None:
    """Quantizes samples for wide exponents where sample - lower_bound can overflow float64.

    Evaluates the quantization formula at half-scale:
    (sample / 2 - lower_bound / 2) * 2^(1 - quant_exp), which rounds exactly as
    (sample - lower_bound) * 2^-quant_exp does.

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
        # Evaluate the difference at half scale, where it can't overflow
        scaled_diff = 0.5 * samples[sample_idx] - half_lower_bound
        out_quantized[sample_idx] = np.int32(math.floor(scaled_diff * half_scale_factor + 0.5))


@njit(nogil=True, cache=True)
def dequantize_pow2_wide(
    quantized_samples: np.ndarray, lower_bound: float, quant_exp: int, out_samples: np.ndarray
) -> None:
    """Dequantizes samples for wide exponents where lower_bound + q * 2^quant_exp can exceed DBL_MAX.

    Evaluates the reconstruction at half-scale and clamps to DBL_MAX to prevent
    rounding to infinity. Preserves lower_bound exactly when q is zero.

    Args:
        quantized_samples: 1D int32 array of quantized integers.
        lower_bound: Anchor of the block (grid point nearest its minimum; the minimum for exempt blocks).
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
        # Keep the lower bound exact when q == 0 (halving may round a subnormal)
        out_samples[sample_idx] = lower_bound if grid_offset == 0 else reconstructed_val
