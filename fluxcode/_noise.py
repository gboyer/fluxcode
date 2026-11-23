# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""White-noise level estimation for the noise floor (ENCODER §5.2).

Organized around standard DSP stages:

1. High-pass 12th-difference estimator (`noise`) with a clean-block early exit on the head.
2. Autocorrelation whiteness checks: lag-1 gate, lag-2 check against near-Nyquist tones,
   and long-lag decay check against periodic tones.
3. Lower-subband (anti-alias) estimator (`_lower_subband`, `_lower_subband_masked`) for
   band-limited DAQs (e.g., sigma-delta converters) whose top band sits in the stopband,
   plus a first-difference step cap (`_first_difference_cap`).
4. Separate paths for masked blocks with NaNs or escaped outliers (`noise_masked`, which
   applies the same whiteness, long-lag decay, lower-subband, and step-cap checks over
   valid windows, falling back to 2nd differences when gaps are dense) and irregular
   timestamps (`noise_irregular`, using second divided differences on consecutive one-scan
   triplets when the block has a cadence).
"""

import math

import numpy as np
from numba import njit

from . import _extreme_magnitudes as xm
from ._format import CODE_FINITE
from ._time import CADENCE_SHARE, one_scan

NO_CODES = np.zeros(0, np.uint8)
"""Empty code array: every sample is finite and unescaped."""

MAD_TO_SD: float = math.sqrt(math.pi / 2)
"""Ratio of standard deviation to mean absolute deviation for a normal distribution."""

CLIP_SIGMAS: float = 4.0
"""Threshold in robust standard deviations beyond which differences are clipped."""

CLIP_PASSES: int = 2
"""Number of iterative outlier clipping passes used in scale re-estimation."""

NOISE_ORDER: int = 12
"""Order k of the high-pass differences on regular (or untimed) blocks: its squared
response (2 sin(pi f))^(2k) / C(2k, k) reads the spectral density at the top of the band,
rejecting smooth signal content below ~1/3 of the sample rate."""

NOISE_COEFS = tuple((-1) ** (NOISE_ORDER - j) * math.comb(NOISE_ORDER, j) for j in range(NOISE_ORDER + 1))
"""12th-difference binomial coefficients, oldest sample first."""

SIGMA_GAIN: float = math.sqrt(math.comb(2 * NOISE_ORDER, NOISE_ORDER))
"""Ratio of 12th-difference standard deviation to white-noise standard deviation."""

WHITE_RHO: float = -NOISE_ORDER / (NOISE_ORDER + 1)
"""Lag-1 autocorrelation of white noise after 12th differencing (-12/13)."""

WHITE_RHO2: float = math.comb(2 * NOISE_ORDER, NOISE_ORDER - 2) / math.comb(2 * NOISE_ORDER, NOISE_ORDER)
"""Lag-2 autocorrelation of white noise after 12th differencing (C(24, 10) / C(24, 12) ~ 0.725)."""

RHO2_TOLERANCE: float = 0.2
"""Relative tolerance on lag-2 autocorrelation: rejects near-Nyquist tones that alternate
every sample or few (rho1 ~ -1, rho2 ~ +1)."""

NOISE_GATE: float = 0.9
"""Lag-1 autocorrelation gate as a share of white noise's theoretical rho (-12/13 for 12th
differences; -2/3 for second divided differences)."""


def _subband_kernel(center: float) -> np.ndarray:
    """12th-difference kernel modulated to `center` of the sample rate, with 4 extra zeros
    at DC (1 - z^-1)^4 to keep slow signal content out."""
    taps = np.array([(-1) ** (NOISE_ORDER - j) * math.comb(NOISE_ORDER, j) for j in range(NOISE_ORDER + 1)], np.float64)
    taps = taps * np.cos(2 * np.pi * (0.5 - center) * np.arange(NOISE_ORDER + 1))
    return np.convolve(taps, np.array([1.0, -4.0, 6.0, -4.0, 1.0]))


SUBBAND_CENTERS = (0.375, 0.25, 0.125)
"""Lower subband center frequencies (cycles/sample) tried when the top band fails the gate,
for anti-alias filtered DAQs whose stopband hides the noise plateau."""

SUBBAND_KERNELS = tuple(_subband_kernel(center) for center in SUBBAND_CENTERS)
"""17-tap FIR kernels for each lower subband."""

SUBBAND_GAINS = tuple(float(np.sqrt(np.dot(kernel, kernel))) for kernel in SUBBAND_KERNELS)
"""Output standard deviation of each subband kernel on unit white noise."""

HEAD_DIFFS: int = 128
"""Number of leading 12th differences inspected for the clean-block early exit."""

HEAD_MARGIN: float = 4.0
"""Safety multiplier on the head spread before exiting early on clean blocks."""

LOWER_BAND_REACH: float = 256.0
"""Lower subbands are tried only when the top band's level exceeds min_sigma / LOWER_BAND_REACH."""

PLATEAU_RATIO: float = 1.25
"""A lower subband is accepted only if its level is at most this factor above the next lower
subband (allows anti-alias roll-off toward Nyquist, rejects rising signal spectra)."""

DECAY_LAGS: tuple[int, ...] = (20, 21, 22, 23)
"""Autocorrelation lags beyond every kernel's support (13 and 17 taps): white noise is
uncorrelated here while periodic tones remain correlated."""

DECAY_LIMIT: float = 0.25
"""Maximum |autocorrelation| at DECAY_LAGS for noise (raised to DECAY_SCALE / sqrt(n) on
short blocks with fewer than 1024 differences)."""

DECAY_SCALE: float = 8.0
"""Short-block scaling (~4 standard errors) for the long-lag autocorrelation decay bound."""

CAP_RATIO: float = 1.25
"""First-difference step cap applies only when it reads at least 1/CAP_RATIO below the
high-pass estimate, avoiding downward bias on true white noise."""

CAP_WINDOWS: int = 8
"""Number of spread windows sampled by the first-difference step cap."""

CAP_WINDOW_LEN: int = 32
"""First differences per step-cap window."""

DIVIDED_DIFF_GAIN: float = math.sqrt(6.0)
"""Ratio of second divided-difference standard deviation to white-noise standard deviation."""

DIVIDED_DIFF_WHITE_RHO: float = -2.0 / 3.0
"""Lag-1 autocorrelation of white noise's second differences."""

NOISE_WINDOW: int = 256
"""Valid differences per spread window (up to twice as many form one window)."""

MIN_SUBBAND_OUT: int = NOISE_WINDOW // 2
"""Minimum valid subband filter outputs needed to evaluate a lower subband (128)."""

QUIET_RATIO: float = 0.75
"""The noise level is the windows' mean spread, or the quietest window's over QUIET_RATIO if
that is lower: noise that varies within a block is floored by at most 4/3 of its quietest part."""

MIN_ONE_SCAN_TRIPLETS: int = 254
"""Fewest triplets of consecutive scans an irregular block needs for a noise floor (2nd
differences of 256 samples; higher than regular blocks' 128 so 2nd differences can separate
white noise from a random walk)."""


# ---------------------------------------------------------------------------
# Robust spread & lag-1 autocorrelation
# ---------------------------------------------------------------------------

@njit(nogil=True, cache=True, fastmath=True)
def _robust_noise(diffs: np.ndarray, mean_diff: float) -> tuple[float, float]:
    """Robust spread and lag-1 autocorrelation of contiguous differences.

    Splits the mean-removed differences into windows of NOISE_WINDOW or more; in each, a robust
    standard deviation comes from the MAD re-estimated CLIP_PASSES times without values beyond
    CLIP_SIGMAS, and the window's differences are clipped to +/- CLIP_SIGMAS of it.
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
            mean_abs_dev += abs(window_diffs[diff_idx])
        mean_abs_dev /= window_len
        for _ in range(CLIP_PASSES):
            clip_threshold = CLIP_SIGMAS * MAD_TO_SD * mean_abs_dev
            clipped_sum = 0.0
            kept_count = np.int64(0)
            for diff_idx in range(window_len):
                abs_diff = abs(window_diffs[diff_idx])
                is_within_clip = abs_diff <= clip_threshold
                clipped_sum += abs_diff * is_within_clip
                kept_count += is_within_clip
            mean_abs_dev = float(clipped_sum / kept_count) if kept_count > 0 else 0.0
        spread = MAD_TO_SD * mean_abs_dev
        if not spread > 0:
            return 0.0, 0.0
        smallest_spread = spread if window_idx == 0 else min(smallest_spread, spread)
        total_spread += spread
        clip_threshold = CLIP_SIGMAS * spread
        for diff_idx in range(window_len):
            window_diffs[diff_idx] = min(max(window_diffs[diff_idx], -clip_threshold), clip_threshold)
    sum_sq_diffs = diffs[0] * diffs[0]
    lag1_cross_prod = 0.0
    diffs_shifted, diffs_base = diffs[1:], diffs[:num_diffs - 1]
    for diff_idx in range(num_diffs - 1):
        sum_sq_diffs += diffs_shifted[diff_idx] * diffs_shifted[diff_idx]
        lag1_cross_prod += diffs_shifted[diff_idx] * diffs_base[diff_idx]
    if not sum_sq_diffs > 0:
        return 0.0, 0.0
    rho = (lag1_cross_prod / (num_diffs - 1)) / (sum_sq_diffs / num_diffs)
    return min(total_spread / num_windows, smallest_spread / QUIET_RATIO), rho


@njit(nogil=True, cache=True, fastmath=True)
def _robust_noise_weighted(
    diffs: np.ndarray, weights: np.ndarray, num_valid: float, mean_diff: float
) -> tuple[float, float]:
    """Robust spread and lag-1 autocorrelation on masked differences (weight 1.0 valid, 0.0 invalid)."""
    num_diffs = diffs.shape[0]
    num_windows = max(1, int(num_valid) // NOISE_WINDOW)
    smallest_spread = 0.0
    total_spread = 0.0
    window_start = 0
    valid_seen = 0.0
    for window_idx in range(num_windows):
        window_end = num_diffs
        if window_idx < num_windows - 1:
            target = math.floor(num_valid * (window_idx + 1) / num_windows)
            window_end = window_start
            while valid_seen < target:
                shortfall = int(target - valid_seen)
                valid_seen += weights[window_end:window_end + shortfall].sum()
                window_end += shortfall
        window_diffs = diffs[window_start:window_end]
        window_weights = weights[window_start:window_end]
        window_len = window_end - window_start
        window_valid = 0.0
        mean_abs_dev = 0.0
        for diff_idx in range(window_len):
            window_diffs[diff_idx] = (window_diffs[diff_idx] - mean_diff) * window_weights[diff_idx]
            window_valid += window_weights[diff_idx]
            mean_abs_dev += abs(window_diffs[diff_idx])
        mean_abs_dev /= window_valid
        for _ in range(CLIP_PASSES):
            clip_threshold = CLIP_SIGMAS * MAD_TO_SD * mean_abs_dev
            clipped_sum = 0.0
            kept_count = 0.0
            for diff_idx in range(window_len):
                abs_diff = abs(window_diffs[diff_idx])
                is_kept = (abs_diff <= clip_threshold) * window_weights[diff_idx]
                clipped_sum += abs_diff * is_kept
                kept_count += is_kept
            mean_abs_dev = clipped_sum / kept_count if kept_count > 0 else 0.0
        spread = MAD_TO_SD * mean_abs_dev
        if not spread > 0:
            return 0.0, 0.0
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
        valid_pairs += weights_shifted[diff_idx] * weights_base[diff_idx]
    if not sum_sq_diffs > 0 or valid_pairs == 0:
        return 0.0, 0.0
    rho = (lag1_cross_prod / valid_pairs) / (sum_sq_diffs / num_valid)
    return min(total_spread / num_windows, smallest_spread / QUIET_RATIO), rho


@njit(nogil=True, cache=True, fastmath=True)
def _weighted_spread(
    diffs: np.ndarray, weights: np.ndarray, num_valid: float, sum_diffs: float, scale_exp: int, gain: float,
    white_rho: float,
) -> tuple[float, float]:
    """Computes (sigma, rho_share) from valid differences of a given gain and theoretical white rho."""
    if num_valid == diffs.shape[0]:
        spread, rho = _robust_noise(diffs, sum_diffs / num_valid)
    else:
        spread, rho = _robust_noise_weighted(diffs, weights, num_valid, sum_diffs / num_valid)
    return math.ldexp(spread / gain, scale_exp), rho / white_rho


# ---------------------------------------------------------------------------
# Whiteness checks (lag-2 & long-lag decay)
# ---------------------------------------------------------------------------

@njit(nogil=True, cache=True, fastmath=True)
def _decayed(clipped: np.ndarray) -> bool:
    """Whether autocorrelations at DECAY_LAGS are all below DECAY_LIMIT (noise, not a tone)."""
    num_diffs = clipped.shape[0]
    longest = DECAY_LAGS[len(DECAY_LAGS) - 1]
    if num_diffs <= 2 * longest:
        return False
    sum_sq = 0.0
    for diff_idx in range(num_diffs):
        sum_sq += clipped[diff_idx] * clipped[diff_idx]
    if not sum_sq > 0:
        return False
    limit = max(DECAY_LIMIT, DECAY_SCALE / math.sqrt(num_diffs))
    for lag in DECAY_LAGS:
        cross = 0.0
        later, earlier = clipped[lag:], clipped[:num_diffs - lag]
        for diff_idx in range(num_diffs - lag):
            cross += later[diff_idx] * earlier[diff_idx]
        if abs(cross / sum_sq) >= limit:
            return False
    return True


@njit(nogil=True, cache=True, fastmath=True)
def _white_shape(clipped: np.ndarray) -> bool:
    """Fused single-pass lag-2 whiteness and long-lag decay check on clean clipped differences."""
    num_diffs = clipped.shape[0]
    first = DECAY_LAGS[len(DECAY_LAGS) - 1]
    if num_diffs <= 2 * first:
        return False
    count = num_diffs - first
    base = clipped[first:]
    lag2s = clipped[first - 2:num_diffs - 2]
    lag_a = clipped[first - DECAY_LAGS[0]:num_diffs - DECAY_LAGS[0]]
    lag_b = clipped[first - DECAY_LAGS[1]:num_diffs - DECAY_LAGS[1]]
    lag_c = clipped[first - DECAY_LAGS[2]:num_diffs - DECAY_LAGS[2]]
    lag_d = clipped[:count]
    sum_sq = cross2 = cross_a = cross_b = cross_c = cross_d = 0.0
    for idx in range(count):
        value = base[idx]
        sum_sq += value * value
        cross2 += value * lag2s[idx]
        cross_a += value * lag_a[idx]
        cross_b += value * lag_b[idx]
        cross_c += value * lag_c[idx]
        cross_d += value * lag_d[idx]
    if not sum_sq > 0:
        return False
    if not abs((cross2 / sum_sq) / WHITE_RHO2 - 1.0) < RHO2_TOLERANCE:
        return False
    limit = max(DECAY_LIMIT, DECAY_SCALE / math.sqrt(num_diffs))
    longest = max(max(abs(cross_a), abs(cross_b)), max(abs(cross_c), abs(cross_d)))
    return longest < limit * sum_sq


@njit(nogil=True, cache=True, fastmath=True)
def _decayed_masked(clipped: np.ndarray, weights: np.ndarray, num_valid: float) -> bool:
    """Long-lag autocorrelation decay check over valid pairs of masked clipped differences."""
    num_diffs = clipped.shape[0]
    longest = DECAY_LAGS[len(DECAY_LAGS) - 1]
    if num_diffs <= 2 * longest or num_valid <= 2 * longest:
        return False
    limit = max(DECAY_LIMIT, DECAY_SCALE / math.sqrt(num_valid))
    for lag in DECAY_LAGS:
        sum_sq = 0.0
        cross = 0.0
        for diff_idx in range(lag, num_diffs):
            pair = weights[diff_idx] * weights[diff_idx - lag]
            sum_sq += clipped[diff_idx] * clipped[diff_idx] * pair
            cross += clipped[diff_idx] * clipped[diff_idx - lag] * pair
        if not sum_sq > 0 or abs(cross / sum_sq) >= limit:
            return False
    return True


@njit(nogil=True, cache=True, fastmath=True)
def _white_shape_masked(clipped: np.ndarray, weights: np.ndarray, num_valid: float) -> bool:
    """Lag-2 whiteness and long-lag decay check over valid pairs of masked clipped differences."""
    num_diffs = clipped.shape[0]
    sum_sq = 0.0
    lag2 = 0.0
    for diff_idx in range(2, num_diffs):
        pair = weights[diff_idx] * weights[diff_idx - 2]
        sum_sq += clipped[diff_idx] * clipped[diff_idx] * pair
        lag2 += clipped[diff_idx] * clipped[diff_idx - 2] * pair
    if not sum_sq > 0:
        return False
    if not abs((lag2 / sum_sq) / WHITE_RHO2 - 1.0) < RHO2_TOLERANCE:
        return False
    return _decayed_masked(clipped, weights, num_valid)


# ---------------------------------------------------------------------------
# Lower-subband (anti-alias) estimator & first-difference step cap
# ---------------------------------------------------------------------------

@njit(nogil=True, cache=True, fastmath=True)
def _subband_level(samples: np.ndarray, scale_factor: float, band: int, scratch_diffs: np.ndarray) -> tuple[float, bool]:
    """A lower subband's noise level (scaled units) and whether its output decays like noise."""
    kernel = SUBBAND_KERNELS[band]
    num_taps = kernel.shape[0]
    num_out = samples.shape[0] - num_taps + 1
    if num_out < MIN_SUBBAND_OUT:
        return 0.0, False
    out = scratch_diffs[:num_out]
    total = 0.0
    for out_idx in range(num_out):
        out[out_idx] = 0.0
    for tap in range(num_taps):
        coef = kernel[tap] * scale_factor
        shifted = samples[tap:tap + num_out]
        for out_idx in range(num_out):
            out[out_idx] += coef * shifted[out_idx]
    for out_idx in range(num_out):
        total += out[out_idx]
    spread, _ = _robust_noise(out, total / num_out)
    if not spread > 0:
        return 0.0, False
    return spread / SUBBAND_GAINS[band], _decayed(out)


@njit(nogil=True, cache=True, fastmath=True)
def _lower_subband(samples: np.ndarray, scale_factor: float, scratch_diffs: np.ndarray) -> float:
    """Noise level of the highest lower subband that decays like noise and sits on a plateau
    (at most PLATEAU_RATIO above the next lower subband), in scaled units; 0.0 if none."""
    level, decayed = _subband_level(samples, scale_factor, 0, scratch_diffs)
    for band in range(1, len(SUBBAND_CENTERS) + 1):
        if band < len(SUBBAND_CENTERS):
            next_level, next_decayed = _subband_level(samples, scale_factor, band, scratch_diffs)
        else:
            next_level, next_decayed = math.inf, False
        if decayed and level > 0 and level <= PLATEAU_RATIO * next_level:
            return level
        level, decayed = next_level, next_decayed
    return 0.0


@njit(nogil=True, cache=True, fastmath=True)
def _subband_level_masked(
    samples: np.ndarray, codes: np.ndarray, scale_factor: float, band: int, min_valid: int,
    scratch_diffs: np.ndarray, scratch_weights: np.ndarray,
) -> tuple[float, bool]:
    """Masked lower-subband level over all-finite 17-sample windows (scaled units)."""
    kernel = SUBBAND_KERNELS[band]
    num_taps = kernel.shape[0]
    num_out = samples.shape[0] - num_taps + 1
    if num_out < MIN_SUBBAND_OUT:
        return 0.0, False
    out = scratch_diffs[:num_out]
    weights = scratch_weights[:num_out]
    last_bad = -1
    for sample_idx in range(num_taps - 1):
        if codes[sample_idx] != CODE_FINITE:
            last_bad = sample_idx
    num_valid = 0.0
    for out_idx in range(num_out):
        if codes[out_idx + num_taps - 1] != CODE_FINITE:
            last_bad = out_idx + num_taps - 1
        weight = 1.0 if last_bad < out_idx else 0.0
        weights[out_idx] = weight
        num_valid += weight
        out[out_idx] = 0.0
    if num_valid < max(min_valid, MIN_SUBBAND_OUT):
        return 0.0, False
    for tap in range(num_taps):
        coef = kernel[tap] * scale_factor
        shifted = samples[tap:tap + num_out]
        for out_idx in range(num_out):
            out[out_idx] += coef * shifted[out_idx]
    total = 0.0
    for out_idx in range(num_out):
        out[out_idx] *= weights[out_idx]
        total += out[out_idx]
    if num_valid == num_out:
        spread, _ = _robust_noise(out, total / num_valid)
    else:
        spread, _ = _robust_noise_weighted(out, weights, num_valid, total / num_valid)
    if not spread > 0:
        return 0.0, False
    return spread / SUBBAND_GAINS[band], _decayed_masked(out, weights, num_valid)


@njit(nogil=True, cache=True, fastmath=True)
def _lower_subband_masked(
    samples: np.ndarray, codes: np.ndarray, scale_factor: float, min_valid: int,
    scratch_diffs: np.ndarray, scratch_weights: np.ndarray,
) -> float:
    """Masked counterpart of _lower_subband for blocks with NaNs or escaped samples."""
    level, decayed = _subband_level_masked(samples, codes, scale_factor, 0, min_valid, scratch_diffs, scratch_weights)
    for band in range(1, len(SUBBAND_CENTERS) + 1):
        if band < len(SUBBAND_CENTERS):
            next_level, next_decayed = _subband_level_masked(
                samples, codes, scale_factor, band, min_valid, scratch_diffs, scratch_weights
            )
        else:
            next_level, next_decayed = math.inf, False
        if decayed and level > 0 and level <= PLATEAU_RATIO * next_level:
            return level
        level, decayed = next_level, next_decayed
    return 0.0


@njit(nogil=True, cache=True, fastmath=True)
def _first_difference_cap(
    samples: np.ndarray, codes: np.ndarray, scale_factor: float, scratch_diffs: np.ndarray,
) -> float:
    """Robust spread of first differences over sqrt(2) (scaled units): caps the 12th-difference
    estimate when dense steps ring through the 13-tap kernel."""
    num_samples = samples.shape[0]
    has_codes = codes.shape[0] > 0
    num_windows, window_len = 1, num_samples - 1
    if num_samples - 1 > CAP_WINDOWS * CAP_WINDOW_LEN:
        num_windows, window_len = CAP_WINDOWS, CAP_WINDOW_LEN
    num_diffs = num_windows * window_len
    if num_diffs < 2:
        return 0.0
    diffs = scratch_diffs[:num_diffs]
    num_valid = 0
    sum_diffs = 0.0
    for window_idx in range(num_windows):
        start = window_idx * (num_samples - 1 - window_len) // max(num_windows - 1, 1)
        for offset in range(window_len):
            sample_idx = start + offset
            diff_val = samples[sample_idx + 1] * scale_factor - samples[sample_idx] * scale_factor
            if has_codes and (codes[sample_idx] | codes[sample_idx + 1]) != CODE_FINITE:
                continue
            diffs[num_valid] = diff_val
            sum_diffs += diff_val
            num_valid += 1
    if num_valid < 2:
        return 0.0
    diffs = diffs[:num_valid]
    plain_mean = sum_diffs / num_valid
    abs_dev = 0.0
    for diff_idx in range(num_valid):
        abs_dev += abs(diffs[diff_idx] - plain_mean)
    threshold = CLIP_SIGMAS * MAD_TO_SD * abs_dev / num_valid
    kept_sum = 0.0
    kept = 0
    for diff_idx in range(num_valid):
        inside = abs(diffs[diff_idx] - plain_mean) <= threshold
        kept_sum += diffs[diff_idx] * inside
        kept += inside
    trimmed_mean = kept_sum / kept if kept > 0 else plain_mean
    for _ in range(CLIP_PASSES):
        clipped_sum = 0.0
        clipped = 0
        for diff_idx in range(num_valid):
            dev = abs(diffs[diff_idx] - trimmed_mean)
            inside = dev <= threshold
            clipped_sum += dev * inside
            clipped += inside
        if clipped == 0:
            return 0.0
        threshold = CLIP_SIGMAS * MAD_TO_SD * clipped_sum / clipped
    return MAD_TO_SD * (clipped_sum / clipped) / math.sqrt(2.0)


# ---------------------------------------------------------------------------
# High-pass 12th-difference estimator (clean blocks)
# ---------------------------------------------------------------------------

@njit(nogil=True, cache=True, fastmath=True)
def _twelfth_differences(samples: np.ndarray, scale_factor: float, out: np.ndarray, count: int) -> float:
    """Writes the first `count` 12th differences of the scaled samples; returns their sum."""
    t0, t1, t2, t3 = samples[:count], samples[1:count + 1], samples[2:count + 2], samples[3:count + 3]
    t4, t5, t6 = samples[4:count + 4], samples[5:count + 5], samples[6:count + 6]
    t7, t8, t9 = samples[7:count + 7], samples[8:count + 8], samples[9:count + 9]
    t10, t11, t12 = samples[10:count + 10], samples[11:count + 11], samples[12:count + 12]
    total = 0.0
    # Scale each sample first (exact power of two): unscaled sums could overflow
    for diff_idx in range(count):
        diff_val = (
            (t0[diff_idx] * scale_factor + t12[diff_idx] * scale_factor)
            - 12.0 * (t1[diff_idx] * scale_factor + t11[diff_idx] * scale_factor)
            + 66.0 * (t2[diff_idx] * scale_factor + t10[diff_idx] * scale_factor)
            - 220.0 * (t3[diff_idx] * scale_factor + t9[diff_idx] * scale_factor)
            + 495.0 * (t4[diff_idx] * scale_factor + t8[diff_idx] * scale_factor)
            - 792.0 * (t5[diff_idx] * scale_factor + t7[diff_idx] * scale_factor)
            + 924.0 * (t6[diff_idx] * scale_factor)
        )
        out[diff_idx] = diff_val
        total += diff_val
    return total


@njit(nogil=True, cache=True, fastmath=True)
def noise(samples: np.ndarray, scale_exp: int, scratch_diffs: np.ndarray, min_sigma: float = 0.0) -> tuple[float, float]:
    """Estimates a clean block's white-noise standard deviation and whiteness gate share.

    Evaluates 12th differences scaled by 2^-scale_exp, tests lag-1/lag-2 whiteness and
    long-lag decay, falls back to lower subbands under an anti-alias filter's stopband,
    and caps the level with first differences.

    Args:
        samples: 1D float64 array of finite samples.
        scale_exp: Power-of-two scaling exponent mapping the block range into [0.5, 1.0).
        scratch_diffs: 1D float64 scratch array of at least len(samples).
        min_sigma: Below this level the noise floor cannot coarsen the block's finest step;
            enables the clean-block head early exit and skips lower-subband/cap checks.

    Returns:
        (sigma, rho_share): sigma in signal units (0.0 for none), and lag-1 autocorrelation
        as a share of white noise's theoretical rho (> NOISE_GATE passes; 1.0 when a lower
        subband sets sigma).
    """
    num_diffs = samples.shape[0] - NOISE_ORDER
    if num_diffs < 2:
        return 0.0, 0.0
    if abs(scale_exp) <= xm.SCALE_LIMIT:
        scale_factor = math.ldexp(1.0, -scale_exp)
    else:
        samples, scale_factor = xm.prescale(samples, scale_exp)
    diffs = scratch_diffs[:num_diffs]
    # Clean-block early exit on the leading HEAD_DIFFS differences
    if min_sigma > 0 and num_diffs >= 2 * HEAD_DIFFS:
        head_mean = _twelfth_differences(samples, scale_factor, diffs, HEAD_DIFFS) / HEAD_DIFFS
        deviation = 0.0
        for diff_idx in range(HEAD_DIFFS):
            deviation += abs(diffs[diff_idx] - head_mean)
        head_spread = MAD_TO_SD * deviation / HEAD_DIFFS
        if math.ldexp(head_spread / SIGMA_GAIN, scale_exp) * LOWER_BAND_REACH * HEAD_MARGIN < min_sigma:
            return 0.0, 0.0
    sum_diffs = _twelfth_differences(samples, scale_factor, diffs, num_diffs)
    spread, rho = _robust_noise(diffs, sum_diffs / num_diffs)
    sigma = spread / SIGMA_GAIN
    rho_share = rho / WHITE_RHO
    top_white = rho_share > NOISE_GATE and _white_shape(diffs)
    if not top_white:
        if math.ldexp(sigma, scale_exp) * LOWER_BAND_REACH < min_sigma:
            return 0.0, 0.0
        sigma = _lower_subband(samples, scale_factor, scratch_diffs)
        if not math.ldexp(sigma, scale_exp) >= min_sigma or not sigma > 0:
            return 0.0, 0.0
        rho_share = 1.0
    if rho_share > NOISE_GATE and math.ldexp(sigma, scale_exp) >= min_sigma:
        sigma = min(sigma, CAP_RATIO * _first_difference_cap(samples, NO_CODES, scale_factor, scratch_diffs))
    return math.ldexp(sigma, scale_exp), rho_share


# ---------------------------------------------------------------------------
# Masked-sample path (blocks with NaNs or escaped outliers)
# ---------------------------------------------------------------------------

@njit(nogil=True, cache=True, fastmath=True)
def _masked_differences(
    samples: np.ndarray, codes: np.ndarray, scale_factor: float, order: int, scratch_diffs: np.ndarray,
    scratch_weights: np.ndarray,
) -> tuple[int, float, float]:
    """Writes order-k differences (k = 2 or NOISE_ORDER) of scaled samples, valid (weight 1.0)
    where all k + 1 samples in the window have CODE_FINITE, else 0.0 with weight 0.0."""
    num_diffs = samples.shape[0] - order
    if num_diffs < 2:
        return 0, 0.0, 0.0
    has_codes = codes.shape[0] > 0
    num_valid = 0.0
    sum_diffs = 0.0
    last_bad = -1
    if has_codes:
        for sample_idx in range(order):
            if codes[sample_idx] != CODE_FINITE:
                last_bad = sample_idx
    for diff_idx in range(num_diffs):
        if has_codes and codes[diff_idx + order] != CODE_FINITE:
            last_bad = diff_idx + order
        diff_val = 0.0
        if order == 2:
            diff_val = (
                samples[diff_idx + 2] * scale_factor
                - 2.0 * (samples[diff_idx + 1] * scale_factor)
                + samples[diff_idx] * scale_factor
            )
        else:
            for j in range(order + 1):
                diff_val += NOISE_COEFS[j] * (samples[diff_idx + j] * scale_factor)
        is_valid = last_bad < diff_idx
        diff_val = diff_val if is_valid else 0.0
        weight = 1.0 if is_valid else 0.0
        scratch_diffs[diff_idx] = diff_val
        scratch_weights[diff_idx] = weight
        num_valid += weight
        sum_diffs += diff_val
    return num_diffs, num_valid, sum_diffs


@njit(nogil=True, cache=True, fastmath=True)
def noise_masked(
    samples: np.ndarray,
    codes: np.ndarray,
    scale_exp: int,
    min_valid: int,
    scratch_diffs: np.ndarray,
    scratch_weights: np.ndarray,
    min_sigma: float = 0.0,
) -> tuple[float, float]:
    """Estimates white-noise parameters for a regularly-sampled block with masked samples
    (NaNs or escaped outliers).

    When at least `min_valid` 12th-difference windows are unmasked, runs the full pipeline
    (lag-1/lag-2 whiteness, long-lag decay, lower subbands under an anti-alias stopband,
    and first-difference step cap). When gaps are too dense for 13-sample windows, falls
    back to 2nd differences over unmasked triplets.
    """
    if abs(scale_exp) <= xm.SCALE_LIMIT:
        scale_factor = math.ldexp(1.0, -scale_exp)
    else:
        samples, scale_factor = xm.prescale(samples, scale_exp)
    num_diffs, num_valid, sum_diffs = _masked_differences(
        samples, codes, scale_factor, NOISE_ORDER, scratch_diffs, scratch_weights
    )
    if num_valid >= max(min_valid, 2):
        diffs = scratch_diffs[:num_diffs]
        weights = scratch_weights[:num_diffs]
        sigma, rho_share = _weighted_spread(diffs, weights, num_valid, sum_diffs, scale_exp, SIGMA_GAIN, WHITE_RHO)
        top_white = rho_share > NOISE_GATE and _white_shape_masked(diffs, weights, num_valid)
        if not top_white:
            if sigma * LOWER_BAND_REACH < min_sigma:
                return 0.0, 0.0
            sub_sigma = _lower_subband_masked(samples, codes, scale_factor, min_valid, scratch_diffs, scratch_weights)
            sigma = math.ldexp(sub_sigma, scale_exp)
            if not sigma >= min_sigma or not sigma > 0:
                return 0.0, 0.0
            rho_share = 1.0
        if rho_share > NOISE_GATE and sigma >= min_sigma:
            cap = _first_difference_cap(samples, codes, scale_factor, scratch_diffs)
            sigma = min(sigma, math.ldexp(CAP_RATIO * cap, scale_exp))
        return sigma, rho_share
    # Dense gaps: fall back to 2nd differences of unmasked triplets
    num_diffs, num_valid, sum_diffs = _masked_differences(
        samples, codes, scale_factor, 2, scratch_diffs, scratch_weights
    )
    if num_valid < max(min_valid, 2):
        return 0.0, 0.0
    return _weighted_spread(
        scratch_diffs[:num_diffs], scratch_weights[:num_diffs], num_valid, sum_diffs, scale_exp,
        DIVIDED_DIFF_GAIN, DIVIDED_DIFF_WHITE_RHO,
    )


# ---------------------------------------------------------------------------
# Irregular-timestamp path (cadence & second divided differences)
# ---------------------------------------------------------------------------

@njit(nogil=True, cache=True)
def cadence(ticks: np.ndarray, scratch_pivots: np.ndarray) -> tuple[int, int, float]:
    """A block's one-scan window, if nearly all of its intervals are one scan.

    The scan interval c is the median of CADENCE_SAMPLES evenly spaced intervals; an interval
    strictly between c/2 and 3c/2 is one scan.
    """
    _, lower, upper, num_one_scan = one_scan(ticks, scratch_pivots)
    if upper == 0:
        return 0, 0, 0.0
    num_intervals = ticks.shape[0] - 1
    share = num_one_scan / num_intervals
    if share < CADENCE_SHARE:
        return 0, 0, share
    return lower, upper, share


@njit(nogil=True, cache=True, fastmath=True)
def noise_irregular(
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
    """Estimates white-noise parameters on irregular timestamps using second divided differences.

    A triplet (three consecutive samples) is valid if all three are unmasked (when codes are
    given) and both intervals lie in (lower, upper). For intervals a (before) and b (after),
    the normalized second divided difference is:
        sqrt(6) * (b * x0 + a * x2 - (a + b) * x1) / sqrt((a + b)^2 + a^2 + b^2)
    which cancels any linear slope and has variance 6 * sigma^2 on white noise.
    """
    num_diffs = samples.shape[0] - 2
    if num_diffs < 2:
        return 0.0, 0.0
    if abs(scale_exp) <= xm.SCALE_LIMIT:
        scale_factor = math.ldexp(1.0, -scale_exp)
    else:
        samples, scale_factor = xm.prescale(samples, scale_exp)
    has_codes = codes.shape[0] > 0
    diffs = scratch_diffs[:num_diffs]
    weights = scratch_weights[:num_diffs]
    num_valid = 0.0
    sum_diffs = 0.0
    for diff_idx in range(num_diffs):
        is_valid = True
        if has_codes:
            is_valid = (codes[diff_idx] | codes[diff_idx + 1] | codes[diff_idx + 2]) == CODE_FINITE
        before = ticks[diff_idx + 1] - ticks[diff_idx]
        after = ticks[diff_idx + 2] - ticks[diff_idx + 1]
        is_valid &= (before > lower) & (before < upper) & (after > lower) & (after < upper)
        after_f, before_f = np.float64(after), np.float64(before)
        span = after_f + before_f
        norm = math.sqrt(max(span * span + after_f * after_f + before_f * before_f, 1.0))
        first = samples[diff_idx] * scale_factor
        middle = samples[diff_idx + 1] * scale_factor
        last = samples[diff_idx + 2] * scale_factor
        diff_val = DIVIDED_DIFF_GAIN * (after_f * first + before_f * last - span * middle) / norm
        diff_val = diff_val if is_valid else 0.0
        weight = 1.0 if is_valid else 0.0
        diffs[diff_idx] = diff_val
        weights[diff_idx] = weight
        num_valid += weight
        sum_diffs += diff_val
    if num_valid < max(min_valid, 2):
        return 0.0, 0.0
    return _weighted_spread(diffs, weights, num_valid, sum_diffs, scale_exp, DIVIDED_DIFF_GAIN, DIVIDED_DIFF_WHITE_RHO)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@njit(nogil=True, cache=True)
def block_noise(
    samples: np.ndarray,
    codes: np.ndarray,
    ticks: np.ndarray,
    scale_exp: int,
    scratch_pivots: np.ndarray,
    scratch_diffs: np.ndarray,
    scratch_weights: np.ndarray,
    min_sigma: float = 0.0,
) -> tuple[float, float]:
    """Estimates a block's white-noise standard deviation and whiteness gate share.

    Dispatches to:
    - `noise` for clean regular/untimed blocks (zero mask overhead);
    - `noise_masked` for regular/untimed blocks with NaNs or escaped outliers;
    - `noise_irregular` for irregular-timestamp blocks that have a cadence.
    """
    irregular = ticks.shape[0] > 0
    if not irregular:
        if codes.shape[0] == 0:
            return noise(samples, scale_exp, scratch_diffs, min_sigma)
        return noise_masked(samples, codes, scale_exp, samples.shape[0] // 2, scratch_diffs, scratch_weights, min_sigma)
    lower, upper, _ = cadence(ticks, scratch_pivots)
    if upper == 0:
        return 0.0, 0.0
    return noise_irregular(
        samples, codes, ticks, lower, upper, scale_exp, MIN_ONE_SCAN_TRIPLETS, scratch_diffs, scratch_weights
    )
