# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""White-noise level and autocorrelation estimation on second differences.

Provides robust estimation of high-frequency white noise standard deviation
(sigma) and the lag-1 autocorrelation coefficient (rho) of second differences.
Used by the encoder's noise floor gate to coarsen quantization steps on blocks
dominated by white measurement noise without degrading deterministic or
random-walk signals.
"""

import math

import numpy as np
from numba import njit

from . import _extreme_magnitudes as xm

MAD_TO_SD: float = math.sqrt(math.pi / 2)
"""Ratio of standard deviation to mean absolute deviation for a normal distribution."""

CLIP_SIGMAS: float = 4.0
"""Threshold in robust standard deviations beyond which differences are clipped."""

CLIP_PASSES: int = 2
"""Number of iterative outlier clipping passes used in scale re-estimation."""

SIGMA_GAIN: float = math.sqrt(6.0)
"""Ratio of second-difference standard deviation to white-noise standard deviation."""


@njit(nogil=True, cache=True, fastmath=True)
def noise(samples: np.ndarray, scale_exp: int) -> tuple[float, float]:
    """Estimates white-noise standard deviation and lag-1 autocorrelation.

    Evaluates second differences of finite samples scaled by 2^-scale_exp (which
    brings the block's range into [0.5, 1.0)) to prevent intermediate overflow or
    underflow. Computes a robust standard deviation via iterated MAD with outlier
    clipping at CLIP_SIGMAS sigmas, then determines the lag-1 autocorrelation of the
    clipped differences.

    Args:
        samples: 1D float64 array of finite samples.
        scale_exp: Power-of-two scaling exponent that maps the block range to
            [0.5, 1.0).

    Returns:
        A tuple of (sigma, rho):
            sigma: Estimated white-noise standard deviation in original signal
                block groups, or 0.0 if variance is negligible or underflows.
            rho: Lag-1 autocorrelation of clipped second differences (typically
                -2/3 for white noise, -1/2 for random walks, and positive for
                smooth signals). Returns (0.0, 0.0) if fewer than 4 samples are
                present.
    """
    num_samples = samples.shape[0]
    num_diffs = num_samples - 2
    # Need at least two second differences to compute lag-1 autocorrelation
    if num_diffs < 2:
        return 0.0, 0.0
    # Use single-step scaling within normal range, or two-step prescaling for extreme exponents
    if abs(scale_exp) <= xm.SCALE_LIMIT:
        scale_factor = math.ldexp(1.0, -scale_exp)
    else:
        samples, scale_factor = xm.prescale(samples, scale_exp)
    diffs = np.empty(num_diffs)
    mean_diff = 0.0
    for diff_idx in range(num_diffs):
        # Second difference: samples[i + 2] - 2 * samples[i + 1] + samples[i]
        diff_val = (
            samples[diff_idx + 2] * scale_factor
            - 2.0 * (samples[diff_idx + 1] * scale_factor)
            + samples[diff_idx] * scale_factor
        )
        diffs[diff_idx] = diff_val
        mean_diff += diff_val
    # Subtract mean second difference to remove pure parabolic trends (quadratics)
    mean_diff /= num_diffs
    mean_abs_dev = 0.0
    for diff_idx in range(num_diffs):
        diffs[diff_idx] -= mean_diff
        # Accumulate mean absolute deviation (MAD)
        mean_abs_dev += abs(diffs[diff_idx])
    mean_abs_dev /= num_diffs
    # Iteratively re-estimate MAD by clipping values beyond CLIP_SIGMAS robust sigmas
    for _ in range(CLIP_PASSES):
        clip_threshold = CLIP_SIGMAS * MAD_TO_SD * mean_abs_dev
        clipped_sum = 0.0
        kept_count = np.int64(0)
        for diff_idx in range(num_diffs):
            abs_diff = abs(diffs[diff_idx])
            # Keep sample if within clip boundary
            is_within_clip = abs_diff <= clip_threshold
            clipped_sum += abs_diff * is_within_clip
            kept_count += is_within_clip
        mean_abs_dev = float(clipped_sum / kept_count) if kept_count > 0 else 0.0
    robust_std = MAD_TO_SD * mean_abs_dev
    # Abort if variation is zero or non-positive
    if not robust_std > 0:
        return 0.0, 0.0
    clip_threshold = CLIP_SIGMAS * robust_std
    # Clip extreme difference spikes to +/- CLIP_SIGMAS sigmas
    for diff_idx in range(num_diffs):
        diffs[diff_idx] = min(max(diffs[diff_idx], -clip_threshold), clip_threshold)
    # Autocorrelation denominator: the sum of the squared differences
    sum_sq_diffs = diffs[0] * diffs[0]
    lag1_cross_prod = 0.0
    # Offset views from index 0 allow SIMD vectorization without negative index checks
    diffs_shifted, diffs_base = diffs[1:], diffs[:num_diffs - 1]
    for diff_idx in range(num_diffs - 1):
        # Accumulate squared norm of shifted diffs and lag-1 cross product
        sum_sq_diffs += diffs_shifted[diff_idx] * diffs_shifted[diff_idx]
        lag1_cross_prod += diffs_shifted[diff_idx] * diffs_base[diff_idx]
    # Check for underflow where differences are below ~1e-154 of the range
    if not sum_sq_diffs > 0:
        return 0.0, 0.0
    # Scale robust standard deviation back to signal block groups and divide by sqrt(6)
    return math.ldexp(robust_std, scale_exp) / SIGMA_GAIN, lag1_cross_prod / sum_sq_diffs
