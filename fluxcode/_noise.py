# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""White-noise level estimation and whiteness gating for quantization noise floors.

Organized around standard digital signal processing stages:

1. High-pass 12th-difference estimator (``estimate_clean_noise``) with a clean-block early
   exit on the leading differences.
2. Autocorrelation whiteness verification: lag-1 gate, lag-2 check rejecting near-Nyquist
   oscillations, and long-lag decay check rejecting periodic tones.
3. Lower-subband anti-alias estimator (``_lower_subband``, ``_lower_subband_masked``) for
   band-limited DAQs whose top band sits inside a filter stopband, plus a first-difference
   step cap (``_first_difference_cap``).
4. Dedicated paths for masked blocks containing NaNs or escaped outliers
   (``estimate_masked_noise``, falling back to 2nd differences when gaps are dense) and
   irregular timestamps (``estimate_irregular_noise``, using second divided differences on
   consecutive single-scan triplets when the block has a regular cadence).
"""

import math

import numpy as np
from numba import njit

from . import _extreme_magnitudes as xm
from ._format import CODE_ON_GRID
from ._time import CADENCE_SHARE, detect_scan_interval

NO_CODES = np.zeros(0, np.uint8)
"""Empty sample-code array indicating that every sample is finite and unescaped."""

MAD_TO_SD: float = math.sqrt(math.pi / 2)
"""Ratio of standard deviation to mean absolute deviation for a Gaussian distribution."""

CLIP_SIGMAS: float = 4.0
"""Threshold in robust standard deviations beyond which differences are clipped."""

CLIP_PASSES: int = 2
"""Number of iterative outlier-clipping passes used when re-estimating scale."""

NOISE_ORDER: int = 12
"""High-pass difference order on regular blocks; suppresses smooth signals below ~fs/3."""

NOISE_COEFS = tuple((-1) ** (NOISE_ORDER - j) * math.comb(NOISE_ORDER, j) for j in range(NOISE_ORDER + 1))
"""12th-difference binomial coefficients, oldest sample first."""

SIGMA_GAIN: float = math.sqrt(math.comb(2 * NOISE_ORDER, NOISE_ORDER))
"""Ratio of 12th-difference standard deviation to white-noise standard deviation."""

EXPECTED_WHITE_LAG1_AUTOCORR: float = -NOISE_ORDER / (NOISE_ORDER + 1)
"""Theoretical lag-1 autocorrelation of white noise after 12th differencing (-12/13)."""

EXPECTED_WHITE_LAG2_AUTOCORR: float = math.comb(2 * NOISE_ORDER, NOISE_ORDER - 2) / math.comb(
    2 * NOISE_ORDER, NOISE_ORDER
)
"""Theoretical lag-2 autocorrelation of white noise after 12th differencing (C(24, 10) / C(24, 12))."""

RHO2_TOLERANCE: float = 0.2
"""Relative tolerance on lag-2 autocorrelation to reject alternating near-Nyquist tones."""

NOISE_GATE: float = 0.9
"""Lag-1 autocorrelation gate as a fraction of white noise's theoretical lag-1 autocorrelation."""


def _subband_kernel(center: float) -> np.ndarray:
    """Builds a 17-tap bandpass FIR kernel centered at ``center`` cycles/sample.

    Args:
        center: Target center frequency in ``(0, 0.5)`` cycles per sample.

    Returns:
        1D float64 array of 17 FIR filter taps with four extra zeros at DC.
    """
    # Modulate the 12th-difference high-pass kernel down to `center`, then convolve with
    # (1 - z^-1)^4 so low-frequency signal trends stay strongly attenuated.
    taps = np.array([(-1) ** (NOISE_ORDER - j) * math.comb(NOISE_ORDER, j) for j in range(NOISE_ORDER + 1)], np.float64)
    taps = taps * np.cos(2 * np.pi * (0.5 - center) * np.arange(NOISE_ORDER + 1))
    return np.convolve(taps, np.array([1.0, -4.0, 6.0, -4.0, 1.0]))


SUBBAND_CENTERS = (0.375, 0.25, 0.125)
"""Lower-subband center frequencies (cycles/sample) tried when the top band fails the gate."""

SUBBAND_KERNELS = tuple(_subband_kernel(center) for center in SUBBAND_CENTERS)
"""17-tap FIR kernels for each lower subband."""

SUBBAND_GAINS = tuple(float(np.sqrt(np.dot(kernel, kernel))) for kernel in SUBBAND_KERNELS)
"""Output standard deviation of each lower-subband kernel on unit white noise."""

HEAD_DIFFS: int = 128
"""Number of leading 12th differences inspected for the clean-block early exit."""

HEAD_MARGIN: float = 4.0
"""Safety multiplier on the head spread before exiting early on clean blocks."""

LOWER_BAND_REACH: float = 256.0
"""Lower subbands are tried only when the top band's level exceeds ``min_sigma / LOWER_BAND_REACH``."""

PLATEAU_RATIO: float = 1.25
"""Maximum ratio of a subband's level to the next lower subband for a flat noise plateau."""

DECAY_LAGS: tuple[int, ...] = (20, 21, 22, 23)
"""Autocorrelation lags beyond every filter's support (13 and 17 taps) to reject periodic tones."""

DECAY_LIMIT: float = 0.25
"""Maximum allowed ``|autocorrelation|`` at ``DECAY_LAGS`` for white noise on long blocks."""

DECAY_SCALE: float = 8.0
"""Short-block scaling factor (~4 standard errors) for the long-lag autocorrelation bound."""

CAP_RATIO: float = 1.25
"""Slack multiplier on the first-difference step cap to avoid downward bias on true white noise."""

CAP_WINDOWS: int = 8
"""Number of evenly spread windows sampled by the first-difference step cap."""

CAP_WINDOW_LEN: int = 32
"""First differences per window in the first-difference step cap."""

DIVIDED_DIFF_GAIN: float = math.sqrt(6.0)
"""Ratio of normalized second divided-difference standard deviation to white-noise sigma."""

DIVIDED_DIFF_WHITE_RHO: float = -2.0 / 3.0
"""Theoretical lag-1 autocorrelation of white noise after 2nd differencing."""

NOISE_WINDOW: int = 256
"""Target number of valid differences per local spread window."""

MIN_SUBBAND_OUT: int = NOISE_WINDOW // 2
"""Minimum valid subband filter outputs required to evaluate a lower subband (128)."""

QUIET_RATIO: float = 0.75
"""Flooring ratio limiting a variable-noise block to at most ``1 / QUIET_RATIO`` of its quietest window."""

MIN_ONE_SCAN_TRIPLETS: int = 254
"""Minimum consecutive single-scan triplets required for an irregular block's noise floor."""


# ---------------------------------------------------------------------------
# Robust spread & lag-1 autocorrelation
# ---------------------------------------------------------------------------


@njit(nogil=True, cache=True, fastmath=True)
def _robust_noise(diffs: np.ndarray, mean_diff: float) -> tuple[float, float]:
    """Computes robust spread and lag-1 autocorrelation of contiguous differences in place.

    Centers and clips ``diffs`` to ``+/- CLIP_SIGMAS`` of each window's robust spread so
    subsequent whiteness checks operate on outlier-clipped differences.

    Args:
        diffs: 1D float64 array of filter differences, centered and clipped in place.
        mean_diff: Mean of ``diffs``.

    Returns:
        Tuple ``(spread, lag1_rho)``, or ``(0.0, 0.0)`` if any window has zero spread or energy.
    """
    # Partition differences into windows of at least NOISE_WINDOW samples, estimate each
    # window's robust SD via CLIP_PASSES of MAD clipping, clip the window in place, and
    # compute lag-1 autocorrelation across the clipped sequence.
    num_diffs = diffs.shape[0]
    num_windows = max(1, num_diffs // NOISE_WINDOW)
    smallest_spread = 0.0
    total_spread = 0.0
    for window_idx in range(num_windows):
        # Zero-based slices avoid negative-index checks and allow LLVM SIMD vectorization.
        window_start = num_diffs * window_idx // num_windows
        window_len = num_diffs * (window_idx + 1) // num_windows - window_start
        window_diffs = diffs[window_start : window_start + window_len]
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
    # Accumulate lag-1 cross-product and total energy over clipped differences.
    sum_sq_diffs = diffs[0] * diffs[0]
    lag1_cross_prod = 0.0
    diffs_shifted, diffs_base = diffs[1:], diffs[: num_diffs - 1]
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
    """Computes robust spread and lag-1 autocorrelation on masked differences in place.

    Args:
        diffs: 1D float64 array of differences, centered and clipped in place.
        weights: 1D float64 mask (``1.0`` for valid entries, ``0.0`` for invalid entries).
        num_valid: Total count of valid entries (sum of ``weights``, must be > 0).
        mean_diff: Mean over valid entries of ``diffs``.

    Returns:
        Tuple ``(spread, lag1_rho)``, or ``(0.0, 0.0)`` if any window has zero spread or no valid pairs.
    """
    # Partition by equal valid-sample counts rather than raw index counts so gaps do not create
    # sparse windows, then run the same iterative MAD clipping and lag-1 autocorrelation as
    # _robust_noise over masked entries and adjacent valid pairs.
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
            # Advance by at least the remaining valid-count shortfall (each entry contributes <= 1).
            while valid_seen < target:
                shortfall = int(target - valid_seen)
                valid_seen += weights[window_end : window_end + shortfall].sum()
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
    # Accumulate lag-1 cross-product only over adjacent pairs where both entries are valid.
    sum_sq_diffs = diffs[0] * diffs[0]
    lag1_cross_prod = 0.0
    valid_pairs = 0.0
    diffs_shifted, diffs_base = diffs[1:], diffs[: num_diffs - 1]
    weights_shifted, weights_base = weights[1:], weights[: num_diffs - 1]
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
    diffs: np.ndarray,
    weights: np.ndarray,
    num_valid: float,
    sum_diffs: float,
    scale_exp: int,
    gain: float,
    white_rho: float,
) -> tuple[float, float]:
    """Converts raw difference statistics into signal-unit ``sigma`` and normalized ``rho_share``.

    Args:
        diffs: 1D float64 array of scaled differences (modified in place).
        weights: 1D float64 validity mask.
        num_valid: Count of valid entries in ``diffs``.
        sum_diffs: Sum of valid entries in ``diffs``.
        scale_exp: Power-of-two exponent restoring signal units.
        gain: Filter RMS gain on unit white noise.
        white_rho: Theoretical lag-1 autocorrelation of white noise for this filter.

    Returns:
        Tuple ``(sigma, rho_share)``.
    """
    if num_valid == diffs.shape[0]:
        spread, rho = _robust_noise(diffs, sum_diffs / num_valid)
    else:
        spread, rho = _robust_noise_weighted(diffs, weights, num_valid, sum_diffs / num_valid)
    return math.ldexp(spread / gain, scale_exp), rho / white_rho


# ---------------------------------------------------------------------------
# Whiteness checks (lag-2 & long-lag decay)
# ---------------------------------------------------------------------------


@njit(nogil=True, cache=True, fastmath=True)
def _autocorr_decays_at_long_lags(clipped: np.ndarray) -> bool:
    """Checks whether autocorrelations at ``DECAY_LAGS`` stay below the white-noise decay bound.

    Args:
        clipped: 1D float64 array of centered, clipped filter differences.

    Returns:
        True if the sequence has enough samples and every long-lag autocorrelation is below the
        decay bound; False otherwise.
    """
    # Lags 20..23 lie strictly outside the 13- and 17-tap kernel supports, so white noise is
    # uncorrelated there while periodic tones retain large autocorrelation.
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
        later, earlier = clipped[lag:], clipped[: num_diffs - lag]
        for diff_idx in range(num_diffs - lag):
            cross += later[diff_idx] * earlier[diff_idx]
        if abs(cross / sum_sq) >= limit:
            return False
    return True


@njit(nogil=True, cache=True, fastmath=True)
def _passes_lag2_and_long_lag_whiteness(clipped: np.ndarray) -> bool:
    """Verifies lag-2 whiteness and long-lag autocorrelation decay on clean clipped differences.

    Args:
        clipped: 1D float64 array of centered, clipped 12th differences.

    Returns:
        True if lag-2 autocorrelation matches ``EXPECTED_WHITE_LAG2_AUTOCORR`` within
        ``RHO2_TOLERANCE`` and all ``DECAY_LAGS`` autocorrelations stay below the decay bound.
    """
    # Fuse energy, lag-2, and all four long-lag cross-products into a single pass over
    # zero-based slices so LLVM vectorizes all six reductions together.
    num_diffs = clipped.shape[0]
    first = DECAY_LAGS[len(DECAY_LAGS) - 1]
    if num_diffs <= 2 * first:
        return False
    count = num_diffs - first
    base = clipped[first:]
    lag2s = clipped[first - 2 : num_diffs - 2]
    lag_a = clipped[first - DECAY_LAGS[0] : num_diffs - DECAY_LAGS[0]]
    lag_b = clipped[first - DECAY_LAGS[1] : num_diffs - DECAY_LAGS[1]]
    lag_c = clipped[first - DECAY_LAGS[2] : num_diffs - DECAY_LAGS[2]]
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
    if not abs((cross2 / sum_sq) / EXPECTED_WHITE_LAG2_AUTOCORR - 1.0) < RHO2_TOLERANCE:
        return False
    limit = max(DECAY_LIMIT, DECAY_SCALE / math.sqrt(num_diffs))
    longest = max(max(abs(cross_a), abs(cross_b)), max(abs(cross_c), abs(cross_d)))
    return longest < limit * sum_sq


@njit(nogil=True, cache=True, fastmath=True)
def _masked_autocorr_decays_at_long_lags(clipped: np.ndarray, weights: np.ndarray, num_valid: float) -> bool:
    """Checks long-lag autocorrelation decay over valid pairs of masked clipped differences.

    Args:
        clipped: 1D float64 array of centered, clipped filter differences.
        weights: 1D float64 validity mask (``1.0`` valid, ``0.0`` invalid).
        num_valid: Total count of valid differences.

    Returns:
        True if all ``DECAY_LAGS`` autocorrelations over valid pairs lie below the decay bound.
    """
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
def _passes_masked_whiteness(clipped: np.ndarray, weights: np.ndarray, num_valid: float) -> bool:
    """Verifies lag-2 whiteness and long-lag decay over valid pairs of masked clipped differences.

    Args:
        clipped: 1D float64 array of centered, clipped 12th differences.
        weights: 1D float64 validity mask (``1.0`` valid, ``0.0`` invalid).
        num_valid: Total count of valid differences.

    Returns:
        True if lag-2 autocorrelation and long-lag decay both pass the white-noise checks.
    """
    num_diffs = clipped.shape[0]
    sum_sq = 0.0
    lag2 = 0.0
    for diff_idx in range(2, num_diffs):
        pair = weights[diff_idx] * weights[diff_idx - 2]
        sum_sq += clipped[diff_idx] * clipped[diff_idx] * pair
        lag2 += clipped[diff_idx] * clipped[diff_idx - 2] * pair
    if not sum_sq > 0:
        return False
    if not abs((lag2 / sum_sq) / EXPECTED_WHITE_LAG2_AUTOCORR - 1.0) < RHO2_TOLERANCE:
        return False
    return _masked_autocorr_decays_at_long_lags(clipped, weights, num_valid)


# ---------------------------------------------------------------------------
# Lower-subband (anti-alias) estimator & first-difference step cap
# ---------------------------------------------------------------------------


@njit(nogil=True, cache=True, fastmath=True)
def _subband_level(samples: np.ndarray, scale_factor: float, band: int, scratch_diffs: np.ndarray) -> tuple[float, bool]:
    """Computes a lower subband's noise level and long-lag decay status on clean samples.

    Args:
        samples: 1D float64 array of finite block samples.
        scale_factor: Power-of-two scaling multiplier mapping samples near unit range.
        band: Subband index into ``SUBBAND_KERNELS``.
        scratch_diffs: Preallocated 1D float64 scratch buffer of at least ``len(samples)``.

    Returns:
        Tuple ``(level_scaled, decayed)`` where ``level_scaled`` is the estimated noise sigma in
        scaled units (0.0 if insufficient outputs or zero spread).
    """
    # Convolve tap-by-tap (outer loop over taps, inner loop over samples) so each inner pass
    # is a contiguous SAXPY that vectorizes cleanly.
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
        shifted = samples[tap : tap + num_out]
        for out_idx in range(num_out):
            out[out_idx] += coef * shifted[out_idx]
    for out_idx in range(num_out):
        total += out[out_idx]
    spread, _ = _robust_noise(out, total / num_out)
    if not spread > 0:
        return 0.0, False
    return spread / SUBBAND_GAINS[band], _autocorr_decays_at_long_lags(out)


@njit(nogil=True, cache=True, fastmath=True)
def _lower_subband(samples: np.ndarray, scale_factor: float, scratch_diffs: np.ndarray) -> float:
    """Estimates noise sigma from the highest lower subband sitting on a flat noise plateau.

    Args:
        samples: 1D float64 array of finite block samples.
        scale_factor: Power-of-two scaling multiplier.
        scratch_diffs: Preallocated 1D float64 scratch buffer of at least ``len(samples)``.

    Returns:
        Noise level in scaled units for the highest qualifying subband, or ``0.0`` if none.
    """
    # Walk from highest lower subband (0.375 fs) downward; accept the first subband that decays
    # at long lags and does not rise above PLATEAU_RATIO times the next lower subband.
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
    samples: np.ndarray,
    codes: np.ndarray,
    scale_factor: float,
    band: int,
    min_valid: int,
    scratch_diffs: np.ndarray,
    scratch_weights: np.ndarray,
) -> tuple[float, bool]:
    """Computes a lower subband's noise level and decay status over unmasked 17-sample windows.

    Args:
        samples: 1D float64 array of block samples (with masked entries imputed).
        codes: 1D uint8 array of per-sample exception codes.
        scale_factor: Power-of-two scaling multiplier.
        band: Subband index into ``SUBBAND_KERNELS``.
        min_valid: Minimum valid filter outputs required.
        scratch_diffs: Preallocated 1D float64 scratch buffer.
        scratch_weights: Preallocated 1D float64 scratch buffer for window validity weights.

    Returns:
        Tuple ``(level_scaled, decayed)`` in scaled units.
    """
    kernel = SUBBAND_KERNELS[band]
    num_taps = kernel.shape[0]
    num_out = samples.shape[0] - num_taps + 1
    if num_out < MIN_SUBBAND_OUT:
        return 0.0, False
    out = scratch_diffs[:num_out]
    weights = scratch_weights[:num_out]
    # Track the most recent non-CODE_ON_GRID sample to mark all-valid 17-sample windows in O(N).
    last_bad = -1
    for sample_idx in range(num_taps - 1):
        if codes[sample_idx] != CODE_ON_GRID:
            last_bad = sample_idx
    num_valid = 0.0
    for out_idx in range(num_out):
        if codes[out_idx + num_taps - 1] != CODE_ON_GRID:
            last_bad = out_idx + num_taps - 1
        weight = 1.0 if last_bad < out_idx else 0.0
        weights[out_idx] = weight
        num_valid += weight
        out[out_idx] = 0.0
    if num_valid < max(min_valid, MIN_SUBBAND_OUT):
        return 0.0, False
    for tap in range(num_taps):
        coef = kernel[tap] * scale_factor
        shifted = samples[tap : tap + num_out]
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
    return spread / SUBBAND_GAINS[band], _masked_autocorr_decays_at_long_lags(out, weights, num_valid)


@njit(nogil=True, cache=True, fastmath=True)
def _lower_subband_masked(
    samples: np.ndarray,
    codes: np.ndarray,
    scale_factor: float,
    min_valid: int,
    scratch_diffs: np.ndarray,
    scratch_weights: np.ndarray,
) -> float:
    """Masked counterpart of ``_lower_subband`` for blocks with NaNs or escaped outliers.

    Args:
        samples: 1D float64 array of block samples.
        codes: 1D uint8 array of per-sample exception codes.
        scale_factor: Power-of-two scaling multiplier.
        min_valid: Minimum valid filter outputs required.
        scratch_diffs: Preallocated 1D float64 scratch buffer.
        scratch_weights: Preallocated 1D float64 scratch buffer.

    Returns:
        Noise level in scaled units for the highest qualifying subband, or ``0.0`` if none.
    """
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
    samples: np.ndarray, codes: np.ndarray, scale_factor: float, scratch_diffs: np.ndarray
) -> float:
    """Estimates an upper noise bound from robust first-difference spread divided by ``sqrt(2)``.

    Caps the 12th-difference estimate when dense step discontinuities ring through the 13-tap
    kernel and inflate high-pass spread.

    Args:
        samples: 1D float64 array of block samples.
        codes: 1D uint8 array of sample codes (or empty when all samples are on-grid).
        scale_factor: Power-of-two scaling multiplier.
        scratch_diffs: Preallocated 1D float64 scratch buffer.

    Returns:
        Robust first-difference sigma estimate in scaled units (0.0 if fewer than 2 valid diffs).
    """
    # Collect valid first differences from up to CAP_WINDOWS evenly spread windows, compute a
    # trimmed mean to reject large step jumps, then refine the MAD over CLIP_PASSES passes.
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
            if has_codes and (codes[sample_idx] | codes[sample_idx + 1]) != CODE_ON_GRID:
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
    """Computes the first ``count`` scaled 12th differences of ``samples`` into ``out``.

    Args:
        samples: 1D float64 array of finite samples.
        scale_factor: Exact power-of-two scaling multiplier.
        out: Output 1D float64 array of length at least ``count``.
        count: Number of 12th differences to compute.

    Returns:
        Sum of the ``count`` computed differences.
    """
    # Pair symmetric binomial coefficients and use zero-based offset slices so LLVM vectorizes
    # the stencil without negative-index bounds checks; scale each sample before summing to
    # prevent float64 overflow on large-magnitude blocks.
    t0, t1, t2, t3 = samples[:count], samples[1 : count + 1], samples[2 : count + 2], samples[3 : count + 3]
    t4, t5, t6 = samples[4 : count + 4], samples[5 : count + 5], samples[6 : count + 6]
    t7, t8, t9 = samples[7 : count + 7], samples[8 : count + 8], samples[9 : count + 9]
    t10, t11, t12 = samples[10 : count + 10], samples[11 : count + 11], samples[12 : count + 12]
    total = 0.0
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
def estimate_clean_noise(
    samples: np.ndarray, scale_exp: int, scratch_diffs: np.ndarray, min_sigma: float = 0.0
) -> tuple[float, float]:
    """Estimates white-noise standard deviation and whiteness gate share for a clean regular block.

    Args:
        samples: 1D float64 array of finite, unmasked samples.
        scale_exp: Binary scale exponent mapping the block range into ``[0.5, 1.0)``.
        scratch_diffs: Preallocated 1D float64 scratch buffer of at least ``len(samples)``.
        min_sigma: Noise level below which the floor cannot coarsen the block's finest step;
            enables the clean-block head early exit and skips lower-subband/cap work when below.

    Returns:
        Tuple ``(sigma, rho_share)`` where ``sigma`` is in signal units (0.0 if rejected) and
        ``rho_share`` exceeds ``NOISE_GATE`` when the whiteness gate passes (1.0 when a lower
        subband sets ``sigma``).
    """
    # 1. Prescale into near-unit range; exit early if the leading HEAD_DIFFS 12th differences
    #    are far below min_sigma (fast path for smooth/clean signals).
    # 2. Compute full 12th differences and test lag-1, lag-2, and long-lag whiteness.
    # 3. If the top band fails (e.g. anti-alias stopband), search lower subbands for a noise
    #    plateau; finally cap the estimate using first differences against dense step ringing.
    num_diffs = samples.shape[0] - NOISE_ORDER
    if num_diffs < 2:
        return 0.0, 0.0
    if abs(scale_exp) <= xm.SCALE_LIMIT:
        scale_factor = math.ldexp(1.0, -scale_exp)
    else:
        samples, scale_factor = xm.prescale(samples, scale_exp)
    diffs = scratch_diffs[:num_diffs]
    # Fast early exit on clean blocks using only the first HEAD_DIFFS differences.
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
    rho_share = rho / EXPECTED_WHITE_LAG1_AUTOCORR
    top_white = rho_share > NOISE_GATE and _passes_lag2_and_long_lag_whiteness(diffs)
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
    samples: np.ndarray,
    codes: np.ndarray,
    scale_factor: float,
    order: int,
    scratch_diffs: np.ndarray,
    scratch_weights: np.ndarray,
) -> tuple[int, float, float]:
    """Computes masked order-``k`` differences (``k = 2`` or ``NOISE_ORDER``) of scaled samples.

    A difference is valid (weight ``1.0``) iff all ``order + 1`` samples in its window have
    ``CODE_ON_GRID``; otherwise both the difference and weight are written as ``0.0``.

    Args:
        samples: 1D float64 array of block samples.
        codes: 1D uint8 array of sample codes (or empty when all samples are on-grid).
        scale_factor: Exact power-of-two scaling multiplier.
        order: Difference order (``2`` or ``NOISE_ORDER``).
        scratch_diffs: Output 1D float64 array of differences.
        scratch_weights: Output 1D float64 array of validity weights.

    Returns:
        Tuple ``(num_diffs, num_valid, sum_diffs)``.
    """
    num_diffs = samples.shape[0] - order
    if num_diffs < 2:
        return 0, 0.0, 0.0
    has_codes = codes.shape[0] > 0
    num_valid = 0.0
    sum_diffs = 0.0
    last_bad = -1
    if has_codes:
        for sample_idx in range(order):
            if codes[sample_idx] != CODE_ON_GRID:
                last_bad = sample_idx
    for diff_idx in range(num_diffs):
        if has_codes and codes[diff_idx + order] != CODE_ON_GRID:
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
def estimate_masked_noise(
    samples: np.ndarray,
    codes: np.ndarray,
    scale_exp: int,
    min_valid: int,
    scratch_diffs: np.ndarray,
    scratch_weights: np.ndarray,
    min_sigma: float = 0.0,
) -> tuple[float, float]:
    """Estimates white-noise parameters for a regular block containing NaNs or escaped outliers.

    Args:
        samples: 1D float64 array of block samples (with masked entries imputed).
        codes: 1D uint8 array of per-sample exception codes.
        scale_exp: Binary scale exponent mapping the block range into ``[0.5, 1.0)``.
        min_valid: Minimum number of valid difference windows required.
        scratch_diffs: Preallocated 1D float64 scratch buffer.
        scratch_weights: Preallocated 1D float64 scratch buffer.
        min_sigma: Noise level below which the floor cannot coarsen the block's finest step.

    Returns:
        Tuple ``(sigma, rho_share)`` in signal units.
    """
    # Prefer the full 12th-difference pipeline (with whiteness checks, lower-subband fallback,
    # and first-difference cap) when enough 13-sample windows are unmasked; otherwise fall back
    # to 2nd differences over unmasked triplets.
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
        sigma, rho_share = _weighted_spread(
            diffs, weights, num_valid, sum_diffs, scale_exp, SIGMA_GAIN, EXPECTED_WHITE_LAG1_AUTOCORR
        )
        top_white = rho_share > NOISE_GATE and _passes_masked_whiteness(diffs, weights, num_valid)
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
    # Dense gaps: fall back to 2nd differences over unmasked 3-sample windows.
    num_diffs, num_valid, sum_diffs = _masked_differences(
        samples, codes, scale_factor, 2, scratch_diffs, scratch_weights
    )
    if num_valid < max(min_valid, 2):
        return 0.0, 0.0
    return _weighted_spread(
        scratch_diffs[:num_diffs],
        scratch_weights[:num_diffs],
        num_valid,
        sum_diffs,
        scale_exp,
        DIVIDED_DIFF_GAIN,
        DIVIDED_DIFF_WHITE_RHO,
    )


# ---------------------------------------------------------------------------
# Irregular-timestamp path (cadence & second divided differences)
# ---------------------------------------------------------------------------


@njit(nogil=True, cache=True)
def cadence(ticks: np.ndarray, scratch_pivots: np.ndarray) -> tuple[int, int, float]:
    """Determines a block's single-scan interval window if nearly all intervals follow one cadence.

    Args:
        ticks: 1D int64 array of non-decreasing block timestamps.
        scratch_pivots: Preallocated 1D int64 scratch buffer of length ``CADENCE_SAMPLES``.

    Returns:
        Tuple ``(lower, upper, share)`` where intervals strictly in ``(lower, upper)`` represent
        one scan and ``share`` is their fraction of all intervals. When ``share < CADENCE_SHARE``
        or no positive cadence exists, returns ``(0, 0, share)``.
    """
    _, lower, upper, num_one_scan = detect_scan_interval(ticks, scratch_pivots)
    if upper == 0:
        return 0, 0, 0.0
    num_intervals = ticks.shape[0] - 1
    share = num_one_scan / num_intervals
    if share < CADENCE_SHARE:
        return 0, 0, share
    return lower, upper, share


@njit(nogil=True, cache=True, fastmath=True)
def estimate_irregular_noise(
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
    """Estimates white-noise parameters on irregular timestamps via normalized 2nd divided differences.

    A triplet is valid iff all three samples are unmasked (when ``codes`` is non-empty) and both
    adjacent time intervals lie strictly in ``(lower, upper)``.

    Args:
        samples: 1D float64 array of block samples.
        codes: 1D uint8 array of sample codes (or empty when all samples are on-grid).
        ticks: 1D int64 array of block timestamps.
        lower: Exclusive lower bound on single-scan intervals.
        upper: Exclusive upper bound on single-scan intervals.
        scale_exp: Binary scale exponent mapping the block range into ``[0.5, 1.0)``.
        min_valid: Minimum valid single-scan triplets required.
        scratch_diffs: Preallocated 1D float64 scratch buffer.
        scratch_weights: Preallocated 1D float64 scratch buffer.

    Returns:
        Tuple ``(sigma, rho_share)`` in signal units, or ``(0.0, 0.0)`` if too few valid triplets.
    """
    # For intervals a (before) and b (after), sqrt(6) * (b*x0 + a*x2 - (a+b)*x1) / sqrt((a+b)^2 + a^2 + b^2)
    # annihilates any linear trend and has variance 6 * sigma^2 on white noise.
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
            is_valid = (codes[diff_idx] | codes[diff_idx + 1] | codes[diff_idx + 2]) == CODE_ON_GRID
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
def estimate_block_noise(
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

    Args:
        samples: 1D float64 array of block samples (imputed where masked).
        codes: 1D uint8 array of sample codes, or empty if every sample is on-grid.
        ticks: 1D int64 array of irregular timestamps, or empty for regular/untimed blocks.
        scale_exp: Binary scale exponent mapping the block range into ``[0.5, 1.0)``.
        scratch_pivots: Preallocated 1D int64 scratch buffer for cadence detection.
        scratch_diffs: Preallocated 1D float64 scratch buffer.
        scratch_weights: Preallocated 1D float64 scratch buffer.
        min_sigma: Noise level below which the floor cannot coarsen the block's finest step.

    Returns:
        Tuple ``(sigma, rho_share)`` in signal units.
    """
    # Dispatch to the clean regular estimator, masked regular estimator, or irregular-timestamp
    # estimator (gated on having a dominant single-scan cadence).
    irregular = ticks.shape[0] > 0
    if not irregular:
        if codes.shape[0] == 0:
            return estimate_clean_noise(samples, scale_exp, scratch_diffs, min_sigma)
        return estimate_masked_noise(
            samples, codes, scale_exp, samples.shape[0] // 2, scratch_diffs, scratch_weights, min_sigma
        )
    lower, upper, _ = cadence(ticks, scratch_pivots)
    if upper == 0:
        return 0.0, 0.0
    return estimate_irregular_noise(
        samples, codes, ticks, lower, upper, scale_exp, MIN_ONE_SCAN_TRIPLETS, scratch_diffs, scratch_weights
    )
