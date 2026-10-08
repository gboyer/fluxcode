# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""White-noise level and autocorrelation estimation on second differences.

Provides robust estimation of high-frequency white noise standard deviation
(sigma) and the lag-1 autocorrelation coefficient (rho) of second differences.
Used by the encoder's noise floor gate to coarsen quantization steps on blocks
dominated by white measurement noise without degrading deterministic or
random-walk signals.

On a block with irregular times, only triplets of consecutive scans count: samples one scan
interval apart, the interval being the short end of the block's intervals. A swinging-door
archive keeps only the points a straight line can't predict, so its sparse points look like
white noise without being noise; where it kept consecutive scans, they are the sensor's raw
samples. The share of one-scan intervals also scales the noise floor (`cadence_factor`): a
sparse archive gets none, one that kept most scans all of it.
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
"""Valid second differences per window of the noise level: sigma is the smallest window's, so
noise that varies within a block is floored by its quietest part. Up to twice as many are one
window."""

CADENCE_QUANTILE: float = 0.01
"""Quantile of a block's non-zero time intervals taken as its scan interval: the short end, so
that a sparse archive's typical gap isn't mistaken for it, but not the minimum, which one burst
or near-duplicate time would drag down."""

CADENCE_BIN_SHIFT: int = 49
"""The quantile is found in bins of 2^49 in float64 bit patterns: an eighth of an octave."""

CADENCE_BINS: int = 256
"""Eighth-octave bins above the shortest interval (32 octaves); longer intervals share the last."""

NUM_COUNT_CHAINS: int = 4
"""Interleaved histogram chains, so that runs of the same bin don't stall on each other."""

CADENCE_TOLERANCE: float = 0.5
"""An interval within this fraction of the scan interval is one scan. Generous: it only has to
separate one scan (1x) from a skipped one (2x), and so covers timestamps jittered by 20%."""

CADENCE_SHARE_OFF: float = 0.5
"""Share of one-scan intervals at or below which a block gets no noise floor."""

CADENCE_SHARE_FULL: float = 0.9
"""Share of one-scan intervals from which a block gets the full noise floor; in between it
ramps linearly."""

MIN_ONE_SCAN_TRIPLETS: int = 254
"""Fewest triplets of consecutive scans an irregular block needs for a noise floor: as many as
the second differences of the smallest block that gets one (_encoder.NOISE_MIN_LEN, 256)."""


@njit(nogil=True, cache=True, fastmath=True)
def noise(samples: np.ndarray, scale_exp: int) -> tuple[float, float]:
    """Estimates white-noise standard deviation and lag-1 autocorrelation.

    Evaluates second differences of finite samples scaled by 2^-scale_exp (which
    brings the block's range into [0.5, 1.0)) to prevent intermediate overflow or
    underflow, and removes their mean (a parabolic trend). Splits them into windows of
    NOISE_WINDOW or more; in each, a robust standard deviation comes from the mean absolute
    deviation re-estimated CLIP_PASSES times without the differences beyond CLIP_SIGMAS, and
    the window's differences are clipped to CLIP_SIGMAS of it. Sigma is the smallest window's;
    rho is the lag-1 autocorrelation of the clipped differences. noise_weighted is the same
    estimator on a subset of the differences.

    Args:
        samples: 1D float64 array of finite samples.
        scale_exp: Power-of-two scaling exponent that maps the block range to
            [0.5, 1.0).

    Returns:
        A tuple of (sigma, rho):
            sigma: Estimated white-noise standard deviation in original signal
                units, or 0.0 if a window's variation is negligible or underflows.
            rho: Lag-1 autocorrelation of clipped second differences (typically
                -2/3 for white noise, -1/2 for random walks, and positive for
                smooth signals). Returns (0.0, 0.0) if fewer than 4 samples are
                present.
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
    num_windows = max(1, num_diffs // NOISE_WINDOW)
    smallest_spread = 0.0
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
    # Normalize covariance by pair count and variance by difference count, as noise_weighted
    rho = (lag1_cross_prod / (num_diffs - 1)) / (sum_sq_diffs / num_diffs)
    # Scale robust standard deviation back to signal units and divide by sqrt(6)
    return math.ldexp(smallest_spread, scale_exp) / SIGMA_GAIN, rho


@njit(nogil=True, cache=True, fastmath=True)
def _robust_noise_weighted(
    diffs: np.ndarray, weights: np.ndarray, num_valid: float, mean_diff: float
) -> tuple[float, float]:
    """The spread and lag-1 autocorrelation of noise() on the valid differences only.

    Windows hold equal shares of the valid differences; the autocorrelation pairs adjacent
    valid differences.

    Args:
        diffs: 1D float64 array of second differences (0 where invalid), overwritten with the
            clipped, mean-removed differences (still 0 where invalid).
        weights: 1D float64 array of 1.0 (valid) or 0.0 per difference.
        num_valid: Number of valid differences (at least 2).
        mean_diff: Mean of the valid differences.

    Returns:
        A tuple of (spread, rho): the smallest window's robust standard deviation of the
        differences and their lag-1 autocorrelation, or (0.0, 0.0) if a window has no spread
        or the clipped squares underflow.
    """
    num_diffs = diffs.shape[0]
    num_windows = max(1, int(num_valid) // NOISE_WINDOW)
    smallest_spread = 0.0
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
    return smallest_spread, (lag1_cross_prod / valid_pairs) / (sum_sq_diffs / num_valid)


@njit(nogil=True, cache=True)
def regular_times(ticks: np.ndarray) -> bool:
    """Whether a block's ticks are equally spaced (a block of fewer than 3 always is).

    Args:
        ticks: 1D int64 array of the block's ticks.

    Returns:
        True if every interval equals the first.
    """
    if ticks.shape[0] < 3:
        return True
    first_interval = ticks[1] - ticks[0]
    differing_bits = np.int64(0)
    # Branch-free so that it vectorizes; wrapping differences compare exactly
    for tick_idx in range(2, ticks.shape[0]):
        differing_bits |= (ticks[tick_idx] - ticks[tick_idx - 1]) ^ first_interval
    return differing_bits == 0


@njit(inline="always")
def _from_bits(bits: int) -> float:
    """The float64 whose bit pattern is bits."""
    return np.array([bits]).view(np.float64)[0]


@njit(inline="always")
def _to_bits(value: float) -> int:
    """The bit pattern of a float64, as an int64 (monotonic in the value for positive floats)."""
    return np.array([value]).view(np.int64)[0]


@njit(inline="always")
def _interval(ticks_next: np.ndarray, ticks_prev: np.ndarray, interval_idx: int) -> float:
    """The interval before ticks_next[interval_idx], as float64.

    uint64: a non-decreasing pair's difference fits; float64 keeps 53 bits of it.
    """
    return np.float64(np.uint64(ticks_next[interval_idx]) - np.uint64(ticks_prev[interval_idx]))


@njit(nogil=True, cache=True, fastmath=True)
def cadence(ticks: np.ndarray, out_intervals: np.ndarray) -> tuple[float, float]:
    """Finds a block's one-scan intervals, their share and their mean.

    The scan interval is the CADENCE_QUANTILE quantile of the block's non-zero intervals,
    rounded up to the next eighth of an octave above the shortest (by at most 12.5%, well within
    CADENCE_TOLERANCE); an interval within CADENCE_TOLERANCE of it is one scan. A decrease in the
    times makes its interval huge (they are checked elsewhere), so never one scan.

    Args:
        ticks: 1D int64 array of the block's ticks (at least 2).
        out_intervals: Output 1D float64 array of len(ticks) - 1 receiving each interval if it
            is one scan, else 0.0.

    Returns:
        A tuple of (share, mean): the share of one-scan intervals and their mean length (both
        0.0 if there are none).
    """
    num_intervals = ticks.shape[0] - 1
    intervals = out_intervals[:num_intervals]
    # Offset slices from index 0 allow SIMD vectorization without negative index checks
    ticks_next, ticks_prev = ticks[1:], ticks[:num_intervals]
    # The shortest non-zero interval, in uint64 (zero wraps to the largest)
    one = np.uint64(1)
    shortest_minus_one = ~np.uint64(0)
    for interval_idx in range(num_intervals):
        gap = np.uint64(ticks_next[interval_idx]) - np.uint64(ticks_prev[interval_idx])
        shortest_minus_one = min(shortest_minus_one, gap - one)
    if shortest_minus_one == ~np.uint64(0):
        intervals[:] = 0.0
        return 0.0, 0.0
    shortest_bits = _to_bits(np.float64(shortest_minus_one + one))
    # Usually the quantile is within the first bin: classify against its edge while counting it
    first_edge = _from_bits(shortest_bits + (1 << CADENCE_BIN_SHIFT))
    num_nonzero = 0
    first_count = 0
    num_one_scan = 0
    one_scan_total = 0.0
    lower, upper = (1.0 - CADENCE_TOLERANCE) * first_edge, (1.0 + CADENCE_TOLERANCE) * first_edge
    for interval_idx in range(num_intervals):
        length = _interval(ticks_next, ticks_prev, interval_idx)
        num_nonzero += int(length > 0)
        first_count += int((length > 0) & (length < first_edge))
        is_one_scan = (lower <= length) & (length <= upper)
        kept = length if is_one_scan else 0.0
        intervals[interval_idx] = kept
        num_one_scan += int(is_one_scan)
        one_scan_total += kept
    rank = int(CADENCE_QUANTILE * (num_nonzero - 1))
    if first_count <= rank:
        # Histogram of eighth-octave bins above the shortest interval (float bits are monotonic
        # in the value), in interleaved chains so that repeated bins don't stall on each other
        counts = np.zeros(NUM_COUNT_CHAINS * CADENCE_BINS, np.int32)
        interval_bits = intervals.view(np.int64)
        for interval_idx in range(num_intervals):
            intervals[interval_idx] = _interval(ticks_next, ticks_prev, interval_idx)
            # Zero intervals fall below the shortest and are not counted
            bin_idx = (interval_bits[interval_idx] - shortest_bits) >> CADENCE_BIN_SHIFT
            if bin_idx >= 0:
                chain = interval_idx % NUM_COUNT_CHAINS
                counts[chain * CADENCE_BINS + min(bin_idx, CADENCE_BINS - 1)] += 1
        seen = 0
        scan_bin = CADENCE_BINS - 1
        for bin_idx in range(CADENCE_BINS):
            for chain in range(NUM_COUNT_CHAINS):
                seen += counts[chain * CADENCE_BINS + bin_idx]
            if seen > rank:
                scan_bin = bin_idx
                break
        # The upper edge of the bin
        scan = _from_bits(shortest_bits + ((scan_bin + 1) << CADENCE_BIN_SHIFT))
        lower, upper = (1.0 - CADENCE_TOLERANCE) * scan, (1.0 + CADENCE_TOLERANCE) * scan
        num_one_scan = 0
        one_scan_total = 0.0
        for interval_idx in range(num_intervals):
            length = intervals[interval_idx]
            is_one_scan = (lower <= length) & (length <= upper)
            kept = length if is_one_scan else 0.0
            intervals[interval_idx] = kept
            num_one_scan += int(is_one_scan)
            one_scan_total += kept
    if num_one_scan == 0:
        return 0.0, 0.0
    return num_one_scan / num_intervals, one_scan_total / num_one_scan


@njit(nogil=True, cache=True)
def cadence_factor(share: float) -> float:
    """Scales the noise floor by a block's share of one-scan intervals.

    Args:
        share: Share of one-scan intervals, 0.0 to 1.0.

    Returns:
        0.0 at or below CADENCE_SHARE_OFF, 1.0 from CADENCE_SHARE_FULL, linear in between.
    """
    ramp = (share - CADENCE_SHARE_OFF) / (CADENCE_SHARE_FULL - CADENCE_SHARE_OFF)
    return min(max(ramp, 0.0), 1.0)


@njit(nogil=True, cache=True, fastmath=True)
def noise_weighted(
    samples: np.ndarray,
    codes: np.ndarray,
    intervals: np.ndarray,
    mean_interval: float,
    scale_exp: int,
    min_valid: int,
    scratch_diffs: np.ndarray,
    scratch_weights: np.ndarray,
) -> tuple[float, float]:
    """Estimates white-noise parameters on the valid triplets of a block.

    A triplet (three consecutive samples) is valid if all three are finite (when codes are
    given) and both its intervals are one scan (when intervals are given). With intervals, its
    difference is the middle sample's from the straight line through its neighbours, times the
    sum of its intervals over the mean one: a slope cancels however the intervals differ, and on
    equal intervals it is the plain second difference. On white noise its variance is
    (b^2 + a^2 + (a + b)^2) / m^2 sigma^2 for intervals a and b and mean m: 6 sigma^2 on average
    times 1 + (2/3) var / m^2, which the interval spread of one-scan triplets keeps under 6%
    (3% in sigma) and jittered timestamps far under.

    Args:
        samples: 1D float64 array of the block's samples (non-finite ones sample-and-held).
        codes: 1D uint8 array of the samples' 2-bit codes, or an empty array if all are finite.
        intervals: 1D float64 array of the block's one-scan intervals (0.0 for the others, as
            cadence gives them), or an empty array to use plain second differences.
        mean_interval: Mean one-scan interval (read with intervals).
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
    timed = intervals.shape[0] > 0
    interval_scale = 1.0 / mean_interval if timed else 1.0
    diffs = scratch_diffs[:num_diffs]
    weights = scratch_weights[:num_diffs]
    # Offset slices from index 0 allow SIMD vectorization without negative index checks
    samples_0, samples_1, samples_2 = samples[:num_diffs], samples[1:num_diffs + 1], samples[2:num_diffs + 2]
    codes_0, codes_1, codes_2 = codes[:num_diffs], codes[1:num_diffs + 1], codes[2:num_diffs + 2]
    intervals_before, intervals_after = intervals[:num_diffs], intervals[1:num_diffs + 1]
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
            before = intervals_before[diff_idx]
            after = intervals_after[diff_idx]
            is_valid &= (before > 0) & (after > 0)
            # (a + b) (chord - middle), over the mean interval
            diff_val = (after * first + before * last - (before + after) * middle) * interval_scale
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
    spread, rho = _robust_noise_weighted(diffs, weights, num_valid, sum_diffs / num_valid)
    return math.ldexp(spread, scale_exp) / SIGMA_GAIN, rho


@njit(nogil=True, cache=True)
def block_noise(
    samples: np.ndarray,
    codes: np.ndarray,
    ticks: np.ndarray,
    scale_exp: int,
    scratch_intervals: np.ndarray,
    scratch_diffs: np.ndarray,
    scratch_weights: np.ndarray,
) -> tuple[float, float, float]:
    """Estimates a block's white noise for the noise floor, and how much of the floor it gets.

    A block with regular times (or none) and finite samples uses noise. With non-finite samples,
    only triplets of finite ones count (at least half the block). With irregular times, only
    triplets of consecutive scans (at least MIN_ONE_SCAN_TRIPLETS), and the floor is scaled by
    the share of one-scan intervals (cadence_factor).

    Args:
        samples: 1D float64 array of the block's finite (or sample-and-held) samples.
        codes: 1D uint8 array of the samples' 2-bit codes, or an empty array if all are finite.
        ticks: 1D int64 array of the samples' ticks, or an empty array without a time axis.
        scale_exp: Power-of-two scaling exponent that maps the block range to [0.5, 1.0).
        scratch_intervals: 1D float64 scratch array of at least len(samples) - 1 (used with
            ticks).
        scratch_diffs: 1D float64 scratch array of at least len(samples).
        scratch_weights: 1D float64 scratch array of at least len(samples).

    Returns:
        A tuple of (sigma, rho, scale): the white-noise estimate (as noise) and the factor
        that scales the noise floor (1.0 unless the times are irregular).
    """
    irregular = ticks.shape[0] > 0 and not regular_times(ticks)
    scale = 1.0
    intervals = scratch_intervals[:0]
    mean_interval = 1.0
    if irregular:
        # Scale the floor by the share of consecutive scans (none for a sparse archive)
        intervals = scratch_intervals[:ticks.shape[0] - 1]
        share, mean_interval = cadence(ticks, intervals)
        scale = cadence_factor(share)
        if scale == 0:
            return 0.0, 0.0, 0.0
    if codes.shape[0] == 0 and not irregular:
        sigma, rho = noise(samples, scale_exp)
    else:
        sigma, rho = noise_weighted(
            samples,
            codes,
            intervals,
            mean_interval,
            scale_exp,
            MIN_ONE_SCAN_TRIPLETS if irregular else samples.shape[0] // 2,
            scratch_diffs,
            scratch_weights,
        )
    return sigma, rho, scale
