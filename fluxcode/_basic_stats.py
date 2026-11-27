# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Basic numerical and summary-statistics kernels used by the encoder and escapes."""

import math
from collections.abc import Callable
from typing import cast

import numpy as np
from numba import njit
from numba.cpython.unsafe.numbers import leading_zeros as _leading_zeros

from . import _extreme_magnitudes as xm

leading_zeros = cast("Callable[[int | np.integer], int]", _leading_zeros)
"""Numba intrinsic counting leading zero bits, typed as its jitted call signature."""

NUM_HIST_CHAINS: int = 4
"""Number of interleaved histogram chains used to avoid store-to-load stalls."""

NUM_BIT_LENGTH_BINS: int = 17
"""Bit-length classes (0..16) for zigzag-encoded uint16 residuals."""

UINT16_MASK: int = 0xFFFF
"""Bitmask for 16-bit unsigned integers (65535)."""


@njit(nogil=True, cache=True, fastmath=True)
def _min_max(samples: np.ndarray) -> tuple[float, float]:
    """Computes the minimum and maximum of a non-empty finite float64 array.

    Args:
        samples: Non-empty 1D float64 array of finite values.

    Returns:
        Tuple ``(min_val, max_val)``.
    """
    # Unroll across four independent chains to expose instruction-level parallelism.
    num_samples = samples.shape[0]
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
    for sample_idx in range(num_samples - num_samples % 4, num_samples):
        min_chain0 = min(min_chain0, samples[sample_idx])
        max_chain0 = max(max_chain0, samples[sample_idx])
    final_min = min(min(min_chain0, min_chain1), min(min_chain2, min_chain3))
    final_max = max(max(max_chain0, max_chain1), max(max_chain2, max_chain3))
    return final_min, final_max


@njit(nogil=True, cache=True, fastmath={"reassoc", "nsz"})
def _sum(samples: np.ndarray) -> float:
    """Computes the sum of ``samples`` with reassociation enabled for SIMD reduction.

    Args:
        samples: 1D float64 array.

    Returns:
        Accumulated float sum (non-finite if any sample is non-finite or sum overflows).
    """
    accum_sum = 0.0
    for sample_idx in range(samples.shape[0]):
        accum_sum += samples[sample_idx]
    return accum_sum


@njit(nogil=True, cache=True)
def block_stats(samples: np.ndarray) -> tuple[float, float, float, int]:
    """Computes block minimum, maximum, mean, and detects any non-finite sample.

    Args:
        samples: Non-empty 1D float64 array of block samples.

    Returns:
        Tuple ``(min_val, max_val, mean_val, bad_index)`` where ``bad_index`` is ``-1``
        if all samples are finite, or the index of the first non-finite sample (with
        zeros for the stats).
    """
    # Use the sum as a fast finite check: if the sum is finite, all samples are finite;
    # otherwise scan for a non-finite sample and fall back to overflow-safe division.
    num_samples = samples.shape[0]
    total_sum = _sum(samples)
    if not math.isfinite(total_sum):
        for sample_idx in range(num_samples):
            if not math.isfinite(samples[sample_idx]):
                return 0.0, 0.0, 0.0, sample_idx
        min_val, max_val = _min_max(samples)
        return min_val, max_val, xm.mean_by_division(samples, num_samples), -1
    min_val, max_val = _min_max(samples)
    return min_val, max_val, total_sum / num_samples, -1


@njit(nogil=True, cache=True, fastmath=True)
def sum_squares(samples: np.ndarray, mean: float) -> float:
    """Computes the sum of squared deviations from ``mean`` over all samples.

    Args:
        samples: 1D float64 array of finite samples.
        mean: Reference mean value.

    Returns:
        Sum of ``(samples[i] - mean)^2``.
    """
    # Separate fastmath kernel so LLVM vectorizes the clean-block variance check.
    total = 0.0
    for sample_idx in range(samples.shape[0]):
        dev = samples[sample_idx] - mean
        total += dev * dev
    return total


@njit(nogil=True, cache=True, fastmath=True)
def estimate_residual_bits(residuals: np.ndarray) -> float:
    """Estimates compressed bits per sample from residual bit-length class entropy.

    Args:
        residuals: 1D int16 array of difference residuals.

    Returns:
        Estimated bits per sample.
    """
    # Histogram zigzagged residual bit lengths (0..16) across NUM_HIST_CHAINS
    # interleaved tables to avoid pipeline stalls on runs of equal bit lengths.
    num_samples = residuals.shape[0]
    bit_length_hist = np.zeros(NUM_HIST_CHAINS * NUM_BIT_LENGTH_BINS, np.int64)
    for sample_idx in range(num_samples):
        residual_val = int(residuals[sample_idx])
        zigzag_val = np.uint32(
            ((residual_val << 1) ^ (residual_val >> 15)) & UINT16_MASK
        )
        bit_length = 32 - leading_zeros(zigzag_val)
        chain = sample_idx & (NUM_HIST_CHAINS - 1)
        bit_length_hist[chain * NUM_BIT_LENGTH_BINS + bit_length] += 1
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
            total_entropy_bits += prob * (max(length_bin - 1, 0) - math.log2(prob))
    return total_entropy_bits


@njit(nogil=True, cache=True)
def zero_residual_fraction(residuals: np.ndarray) -> float:
    """Computes the fraction of residuals equal to zero.

    Args:
        residuals: 1D int16 array of difference residuals.

    Returns:
        Fraction in ``[0.0, 1.0]``.
    """
    zeros = 0
    for sample_idx in range(residuals.shape[0]):
        zeros += residuals[sample_idx] == 0
    return float(zeros / max(residuals.shape[0], 1))
