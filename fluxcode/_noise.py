# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""White-noise level and autocorrelation estimation on second differences.

Provides robust estimation of high-frequency white noise standard deviation
(sigma) and the lag-1 autocorrelation coefficient (rho) of second differences.
Used by the encoder's noise floor gate to coarsen quantization steps on blocks
dominated by white measurement noise without degrading deterministic or
random-walk signals.

A block with irregular times gets a noise floor only if it has a cadence: nearly all of its
intervals one scan apart (`cadence`). That covers jittered clocks and gaps; a swinging-door or
deadband archive keeps only the points a straight line or the last value can't predict, so its
points look like white noise without being noise, and it gets none unless it kept nearly every
scan. Neither do events or any other spread of intervals. On such a block only triplets of
consecutive scans count.
"""

import math

import numpy as np
from numba import njit

from . import _extreme_magnitudes as xm
from ._format import CODE_FINITE

MAD_TO_SD: float = math.sqrt(math.pi / 2)
"""Ratio of standard deviation to mean absolute deviation for a normal distribution."""

CLIP_SIGMAS: float = 4.0
"""Threshold in robust standard deviations beyond which differences are clipped."""

CLIP_PASSES: int = 2
"""Number of iterative outlier clipping passes used in scale re-estimation."""

SIGMA_GAIN: float = math.sqrt(6.0)
"""Ratio of second-difference standard deviation to white-noise standard deviation."""

NOISE_WINDOW: int = 256
"""Valid second differences per window of the noise level (up to twice as many are one window)."""

QUIET_RATIO: float = 0.75
"""The noise level is the windows' mean spread, or the quietest window's over QUIET_RATIO if
that is lower: noise that varies within a block is floored by at most 4/3 of its quietest part.
On steady white noise the quietest of up to 255 windows never measured below 0.77 of the mean,
so steady noise reads unbiased (the plain minimum read 5% low over 3 windows, 18% over 255)."""

CADENCE_SAMPLES: int = 15
"""Evenly spaced intervals whose median is a block's scan interval. Where nearly all intervals
are one scan, so is the median of any 15 of them; where it is not, the block gets no floor."""

CADENCE_SHARE: float = 0.9
"""Share of a block's intervals that must be one scan for a noise floor."""

MIN_ONE_SCAN_TRIPLETS: int = 254
"""Fewest triplets of consecutive scans an irregular block needs for a noise floor: as many as
the second differences of the smallest block that gets one (_encoder.NOISE_MIN_LEN, 256)."""


@njit(nogil=True, cache=True, fastmath=True)
def _robust_noise(diffs: np.ndarray, mean_diff: float) -> tuple[float, float]:
    """The robust spread and lag-1 autocorrelation of second differences.

    Splits the mean-removed differences into windows of NOISE_WINDOW or more; in each, a robust
    standard deviation comes from the mean absolute deviation re-estimated CLIP_PASSES times
    without the differences beyond CLIP_SIGMAS, and the window's differences are clipped to
    CLIP_SIGMAS of it. The spread combines the windows' (QUIET_RATIO); rho is the lag-1
    autocorrelation of the clipped differences.

    Args:
        diffs: 1D float64 array of at least 2 second differences, overwritten with the clipped,
            mean-removed differences.
        mean_diff: Their mean.

    Returns:
        A tuple of (spread, rho), or (0.0, 0.0) if a window has no spread or the clipped
        squares underflow.
    """
    num_diffs = diffs.shape[0]
    num_windows = max(1, num_diffs // NOISE_WINDOW)
    smallest_spread = 0.0
    total_spread = 0.0
    for window_idx in range(num_windows):
        # Slices from index 0 allow SIMD vectorization without negative index checks
        window_start = num_diffs * window_idx // num_windows
        window_len = num_diffs * (window_idx + 1) // num_windows - window_start
        window_diffs = diffs[window_start:window_start + window_len]
        mean_abs_dev = 0.0
        for diff_idx in range(window_len):
            window_diffs[diff_idx] -= mean_diff
            # Accumulate mean absolute deviation (MAD)
            mean_abs_dev += abs(window_diffs[diff_idx])
        mean_abs_dev /= window_len
        # Iteratively re-estimate MAD by clipping values beyond CLIP_SIGMAS robust sigmas
        for _ in range(CLIP_PASSES):
            clip_threshold = CLIP_SIGMAS * MAD_TO_SD * mean_abs_dev
            clipped_sum = 0.0
            kept_count = np.int64(0)
            for diff_idx in range(window_len):
                abs_diff = abs(window_diffs[diff_idx])
                # Keep sample if within clip boundary
                is_within_clip = abs_diff <= clip_threshold
                clipped_sum += abs_diff * is_within_clip
                kept_count += is_within_clip
            mean_abs_dev = float(clipped_sum / kept_count) if kept_count > 0 else 0.0
        spread = MAD_TO_SD * mean_abs_dev
        # A window without variation has no white noise to floor
        if not spread > 0:
            return 0.0, 0.0
        # The first window's spread, then the smallest (no infinities under fastmath)
        smallest_spread = spread if window_idx == 0 else min(smallest_spread, spread)
        total_spread += spread
        clip_threshold = CLIP_SIGMAS * spread
        # Clip extreme difference spikes to +/- CLIP_SIGMAS sigmas of their window
        for diff_idx in range(window_len):
            window_diffs[diff_idx] = min(max(window_diffs[diff_idx], -clip_threshold), clip_threshold)
    # Autocorrelation denominator: the sum of the squared differences
    sum_sq_diffs = diffs[0] * diffs[0]
    lag1_cross_prod = 0.0
    diffs_shifted, diffs_base = diffs[1:], diffs[:num_diffs - 1]
    for diff_idx in range(num_diffs - 1):
        # Accumulate squared norm of shifted diffs and lag-1 cross product
        sum_sq_diffs += diffs_shifted[diff_idx] * diffs_shifted[diff_idx]
        lag1_cross_prod += diffs_shifted[diff_idx] * diffs_base[diff_idx]
    # Check for underflow where differences are below ~1e-154 of the range
    if not sum_sq_diffs > 0:
        return 0.0, 0.0
    # Normalize covariance by pair count and variance by difference count, as _robust_noise_weighted
    rho = (lag1_cross_prod / (num_diffs - 1)) / (sum_sq_diffs / num_diffs)
    return min(total_spread / num_windows, smallest_spread / QUIET_RATIO), rho


@njit(nogil=True, cache=True, fastmath=True)
def _robust_noise_weighted(
    diffs: np.ndarray, weights: np.ndarray, num_valid: float, mean_diff: float
) -> tuple[float, float]:
    """The spread and lag-1 autocorrelation of _robust_noise on the valid differences only.

    Windows hold equal shares of the valid differences; the autocorrelation pairs adjacent
    valid differences.

    Args:
        diffs: 1D float64 array of second differences (0 where invalid), overwritten with the
            clipped, mean-removed differences (still 0 where invalid).
        weights: 1D float64 array of 1.0 (valid) or 0.0 per difference.
        num_valid: Number of valid differences (at least 2).
        mean_diff: Mean of the valid differences.

    Returns:
        A tuple of (spread, rho) as _robust_noise's, or (0.0, 0.0) if a window has no spread or
        the clipped squares underflow.
    """
    num_diffs = diffs.shape[0]
    num_windows = max(1, int(num_valid) // NOISE_WINDOW)
    smallest_spread = 0.0
    total_spread = 0.0
    window_start = 0
    valid_seen = 0.0
    for window_idx in range(num_windows):
        # A window ends once it holds its share of the valid differences; the last one at the end
        window_end = num_diffs
        if window_idx < num_windows - 1:
            target = math.floor(num_valid * (window_idx + 1) / num_windows)
            window_end = window_start
            while valid_seen < target:
                # At least the shortfall's worth of differences remain: add them in one sum
                shortfall = int(target - valid_seen)
                valid_seen += weights[window_end:window_end + shortfall].sum()
                window_end += shortfall
        # Slices from index 0 allow SIMD vectorization without negative index checks
        window_diffs = diffs[window_start:window_end]
        window_weights = weights[window_start:window_end]
        window_len = window_end - window_start
        window_valid = 0.0
        mean_abs_dev = 0.0
        for diff_idx in range(window_len):
            # Remove the mean from valid differences, keeping invalid ones at 0
            window_diffs[diff_idx] = (window_diffs[diff_idx] - mean_diff) * window_weights[diff_idx]
            window_valid += window_weights[diff_idx]
            mean_abs_dev += abs(window_diffs[diff_idx])
        mean_abs_dev /= window_valid
        # Iterative outlier clipping on valid differences
        for _ in range(CLIP_PASSES):
            clip_threshold = CLIP_SIGMAS * MAD_TO_SD * mean_abs_dev
            clipped_sum = 0.0
            kept_count = 0.0
            for diff_idx in range(window_len):
                abs_diff = abs(window_diffs[diff_idx])
                # Keep a difference only if within the clip threshold and valid
                is_kept = (abs_diff <= clip_threshold) * window_weights[diff_idx]
                clipped_sum += abs_diff * is_kept
                kept_count += is_kept
            mean_abs_dev = clipped_sum / kept_count if kept_count > 0 else 0.0
        spread = MAD_TO_SD * mean_abs_dev
        if not spread > 0:
            return 0.0, 0.0
        # The first window's spread, then the smallest (no infinities under fastmath)
        smallest_spread = spread if window_idx == 0 else min(smallest_spread, spread)
        total_spread += spread
        clip_threshold = CLIP_SIGMAS * spread
        for diff_idx in range(window_len):
            window_diffs[diff_idx] = min(max(window_diffs[diff_idx], -clip_threshold), clip_threshold)
        window_start = window_end
    sum_sq_diffs = diffs[0] * diffs[0]
    lag1_cross_prod = 0.0
    valid_pairs = 0.0
    diffs_shifted, diffs_base = diffs[1:], diffs[:num_diffs - 1]
    weights_shifted, weights_base = weights[1:], weights[:num_diffs - 1]
    for diff_idx in range(num_diffs - 1):
        sum_sq_diffs += diffs_shifted[diff_idx] * diffs_shifted[diff_idx]
        lag1_cross_prod += diffs_shifted[diff_idx] * diffs_base[diff_idx]
        # Count only pairs of adjacent differences that are both valid
        valid_pairs += weights_shifted[diff_idx] * weights_base[diff_idx]
    if not sum_sq_diffs > 0 or valid_pairs == 0:
        return 0.0, 0.0
    # Normalize covariance by pair count and variance by valid difference count
    rho = (lag1_cross_prod / valid_pairs) / (sum_sq_diffs / num_valid)
    return min(total_spread / num_windows, smallest_spread / QUIET_RATIO), rho


@njit(nogil=True, cache=True, fastmath=True)
def noise(samples: np.ndarray, scale_exp: int, scratch_diffs: np.ndarray) -> tuple[float, float]:
    """Estimates white-noise standard deviation and lag-1 autocorrelation.

    Evaluates second differences of finite samples scaled by 2^-scale_exp (which brings the
    block's range into [0.5, 1.0)) to prevent intermediate overflow or underflow, removes their
    mean (a parabolic trend) and takes their robust spread and autocorrelation (_robust_noise).

    Args:
        samples: 1D float64 array of finite samples.
        scale_exp: Power-of-two scaling exponent that maps the block range to [0.5, 1.0).
        scratch_diffs: 1D float64 scratch array of at least len(samples) - 2.

    Returns:
        A tuple of (sigma, rho):
            sigma: Estimated white-noise standard deviation in original signal units, or 0.0
                if a window's variation is negligible or underflows.
            rho: Lag-1 autocorrelation of clipped second differences (typically -2/3 for white
                noise, -1/2 for random walks, and positive for smooth signals). Returns
                (0.0, 0.0) if fewer than 4 samples are present.
    """
    num_diffs = samples.shape[0] - 2
    # Need at least two second differences to compute lag-1 autocorrelation
    if num_diffs < 2:
        return 0.0, 0.0
    # Use single-step scaling within normal range, or two-step prescaling for extreme exponents
    if abs(scale_exp) <= xm.SCALE_LIMIT:
        scale_factor = math.ldexp(1.0, -scale_exp)
    else:
        samples, scale_factor = xm.prescale(samples, scale_exp)
    diffs = scratch_diffs[:num_diffs]
    sum_diffs = 0.0
    for diff_idx in range(num_diffs):
        # Second difference: samples[i + 2] - 2 * samples[i + 1] + samples[i]
        diff_val = (
            samples[diff_idx + 2] * scale_factor
            - 2.0 * (samples[diff_idx + 1] * scale_factor)
            + samples[diff_idx] * scale_factor
        )
        diffs[diff_idx] = diff_val
        sum_diffs += diff_val
    # Subtract mean second difference to remove pure parabolic trends (quadratics)
    spread, rho = _robust_noise(diffs, sum_diffs / num_diffs)
    # Scale robust standard deviation back to signal units and divide by sqrt(6)
    return math.ldexp(spread, scale_exp) / SIGMA_GAIN, rho


@njit(nogil=True, cache=True)
def cadence(ticks: np.ndarray, scratch_pivots: np.ndarray) -> tuple[int, int, float]:
    """A block's one-scan window, if nearly all of its intervals are one scan.

    The scan interval c is the median of CADENCE_SAMPLES evenly spaced intervals; an interval
    strictly between c/2 and 3c/2 is one scan. That separates one scan from two (a skipped
    scan) and covers jittered times. A decrease in the times wraps to a negative interval
    (they are checked elsewhere), so never one scan.

    Args:
        ticks: 1D int64 array of the block's ticks (at least 2).
        scratch_pivots: 1D int64 scratch array of at least CADENCE_SAMPLES.

    Returns:
        A tuple of (lower, upper, share): an interval d is one scan if lower < d < upper, and
        share is the share of one-scan intervals. If it is below CADENCE_SHARE, the window is
        (0, 0), which holds no interval.
    """
    num_intervals = ticks.shape[0] - 1
    # Offset slices from index 0 allow SIMD vectorization without negative index checks
    ticks_next, ticks_prev = ticks[1:], ticks[:num_intervals]
    # Insertion sort of the evenly spaced intervals (a library sort would allocate)
    pivots = scratch_pivots[:CADENCE_SAMPLES]
    for pivot_idx in range(CADENCE_SAMPLES):
        interval_idx = pivot_idx * (num_intervals - 1) // (CADENCE_SAMPLES - 1)
        pivot = ticks_next[interval_idx] - ticks_prev[interval_idx]
        insert_idx = pivot_idx
        while insert_idx > 0 and pivots[insert_idx - 1] > pivot:
            pivots[insert_idx] = pivots[insert_idx - 1]
            insert_idx -= 1
        pivots[insert_idx] = pivot
    scan = pivots[CADENCE_SAMPLES // 2]
    # Strictly between scan/2 and 3 scan/2 in integers: no rounding, and no overflow below 2^62
    if not 0 < scan < (1 << 62):
        return 0, 0, 0.0
    lower, upper = scan // 2, scan + (scan + 1) // 2
    num_one_scan = 0
    for interval_idx in range(num_intervals):
        interval = ticks_next[interval_idx] - ticks_prev[interval_idx]
        num_one_scan += (interval > lower) & (interval < upper)
    share = num_one_scan / num_intervals
    if share < CADENCE_SHARE:
        return 0, 0, share
    return lower, upper, share


@njit(nogil=True, cache=True, fastmath=True)
def noise_weighted(
    samples: np.ndarray,
    codes: np.ndarray,
    ticks: np.ndarray,
    lower: int,
    upper: int,
    scale_exp: int,
    min_valid: int,
    scratch_diffs: np.ndarray,
    scratch_weights: np.ndarray,
) -> tuple[float, float]:
    """Estimates white-noise parameters on the valid triplets of a block.

    A triplet (three consecutive samples) is valid if all three are finite (when codes are
    given) and both its intervals are one scan, lower < d < upper (when ticks are given). With
    ticks, its difference is the middle sample's from the straight line through its
    neighbours, scaled to the variance of a regular second difference on white noise: for
    intervals a (before) and b (after), sqrt(6) (b x0 + a x2 - (a + b) x1) / sqrt((a + b)^2 +
    a^2 + b^2). A slope cancels however the intervals differ, white noise of sigma gives
    6 sigma^2 for any intervals, and equal intervals give the plain second difference.
    If every triplet is valid, the plain estimator takes the differences (_robust_noise).

    Args:
        samples: 1D float64 array of the block's samples (non-finite ones sample-and-held).
        codes: 1D uint8 array of the samples' 2-bit codes, or an empty array if all are finite.
        ticks: 1D int64 array of the samples' ticks, or an empty array to use plain second
            differences.
        lower: Exclusive lower bound of a one-scan interval (read with ticks).
        upper: Exclusive upper bound of a one-scan interval (read with ticks).
        scale_exp: Power-of-two scaling exponent that normalizes range to [0.5, 1.0).
        min_valid: Minimum number of valid triplets required to gate.
        scratch_diffs: 1D float64 scratch array of at least len(samples) - 2.
        scratch_weights: 1D float64 scratch array of at least len(samples) - 2.

    Returns:
        A tuple of (sigma, rho):
            sigma: Estimated white-noise standard deviation in original units.
            rho: Lag-1 autocorrelation of valid differences.
            Both are 0.0 if there are fewer than min_valid (and at least 2) valid triplets,
            or if their variance is negligible.
    """
    num_diffs = samples.shape[0] - 2
    if num_diffs < 2:
        return 0.0, 0.0
    # Apply power-of-two scaling to prevent intermediate overflow/underflow
    if abs(scale_exp) <= xm.SCALE_LIMIT:
        scale_factor = math.ldexp(1.0, -scale_exp)
    else:
        samples, scale_factor = xm.prescale(samples, scale_exp)
    has_codes = codes.shape[0] > 0
    timed = ticks.shape[0] > 0
    diffs = scratch_diffs[:num_diffs]
    weights = scratch_weights[:num_diffs]
    # Offset slices from index 0 allow SIMD vectorization without negative index checks
    samples_0, samples_1, samples_2 = samples[:num_diffs], samples[1:num_diffs + 1], samples[2:num_diffs + 2]
    codes_0, codes_1, codes_2 = codes[:num_diffs], codes[1:num_diffs + 1], codes[2:num_diffs + 2]
    ticks_0, ticks_1, ticks_2 = ticks[:num_diffs], ticks[1:num_diffs + 1], ticks[2:num_diffs + 2]
    num_valid = 0.0
    sum_diffs = 0.0
    for diff_idx in range(num_diffs):
        is_valid = True
        if has_codes:
            is_valid = (codes_0[diff_idx] | codes_1[diff_idx] | codes_2[diff_idx]) == CODE_FINITE
        first = samples_0[diff_idx] * scale_factor
        middle = samples_1[diff_idx] * scale_factor
        last = samples_2[diff_idx] * scale_factor
        if timed:
            before = ticks_1[diff_idx] - ticks_0[diff_idx]
            after = ticks_2[diff_idx] - ticks_1[diff_idx]
            is_valid &= (before > lower) & (before < upper) & (after > lower) & (after < upper)
            # (a + b) times the middle's distance below the line through its neighbours, over
            # sqrt((a + b)^2 + a^2 + b^2) / sqrt(6): the line weighs each by the other's interval
            after_f, before_f = np.float64(after), np.float64(before)
            span = after_f + before_f
            norm = math.sqrt(max(span * span + after_f * after_f + before_f * before_f, 1.0))
            diff_val = SIGMA_GAIN * (after_f * first + before_f * last - span * middle) / norm
        else:
            diff_val = last - 2.0 * middle + first
        # Selects rather than branches, so that the loop vectorizes
        diff_val = diff_val if is_valid else 0.0
        weight = 1.0 if is_valid else 0.0
        diffs[diff_idx] = diff_val
        weights[diff_idx] = weight
        num_valid += weight
        sum_diffs += diff_val
    if num_valid < max(min_valid, 2):
        return 0.0, 0.0
    if num_valid == num_diffs:
        spread, rho = _robust_noise(diffs, sum_diffs / num_valid)
    else:
        spread, rho = _robust_noise_weighted(diffs, weights, num_valid, sum_diffs / num_valid)
    return math.ldexp(spread, scale_exp) / SIGMA_GAIN, rho


@njit(nogil=True, cache=True)
def block_noise(
    samples: np.ndarray,
    codes: np.ndarray,
    ticks: np.ndarray,
    scale_exp: int,
    scratch_pivots: np.ndarray,
    scratch_diffs: np.ndarray,
    scratch_weights: np.ndarray,
) -> tuple[float, float]:
    """Estimates a block's white noise for the noise floor.

    A block with regular times (or none) and finite samples uses noise. With non-finite samples,
    only triplets of finite ones count (at least half the block). With irregular times, only a
    block with a cadence gets a floor, and only triplets of consecutive scans count (at least
    MIN_ONE_SCAN_TRIPLETS).

    Args:
        samples: 1D float64 array of the block's finite (or sample-and-held) samples.
        codes: 1D uint8 array of the samples' 2-bit codes, or an empty array if all are finite.
        ticks: 1D int64 array of the samples' ticks if they are irregular (not all intervals
            equal), else an empty array.
        scale_exp: Power-of-two scaling exponent that maps the block range to [0.5, 1.0).
        scratch_pivots: 1D int64 scratch array of at least CADENCE_SAMPLES (used with ticks).
        scratch_diffs: 1D float64 scratch array of at least len(samples).
        scratch_weights: 1D float64 scratch array of at least len(samples).

    Returns:
        A tuple of (sigma, rho), as noise's; (0.0, 0.0) for irregular times without a cadence.
    """
    irregular = ticks.shape[0] > 0
    if not irregular and codes.shape[0] == 0:
        return noise(samples, scale_exp, scratch_diffs)
    lower, upper = 0, 0
    if irregular:
        lower, upper, _ = cadence(ticks, scratch_pivots)
        if upper == 0:
            return 0.0, 0.0
    return noise_weighted(
        samples,
        codes,
        ticks,
        lower,
        upper,
        scale_exp,
        MIN_ONE_SCAN_TRIPLETS if irregular else samples.shape[0] // 2,
        scratch_diffs,
        scratch_weights,
    )
