# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Value error-bound, decimal-exactness, and block-statistics checks for ``fluxcode.verify``."""

import numpy as np
from numba import njit

from ._format import BLOCK_FLAG_DECIMAL, BLOCK_FLAG_SCALED
from ._types import EncodedGroup, Params, UpdatedGroup
from ._verify_common import (
    MAX_REPORTED,
    NOISE_FLOOR_MIN_SAMPLES,
    Violation,
    report_blocks,
    report_samples,
)
from ._verify_decode import Decoded


def check_values(
    found: list[Violation],
    given: np.ndarray,
    new: Decoded,
    sizes: np.ndarray,
    encoded: np.ndarray,
    params: Params,
    result: EncodedGroup | UpdatedGroup,
) -> None:
    """Checks non-finite preservation, range/fine/step error bounds, and per-block statistics."""
    # Evaluate all errors and ranges at half scale (x / 2) so blocks spanning [-DBL_MAX, +DBL_MAX]
    # never overflow during subtraction.
    half_error, lo, hi, magnitude, count, mean, nonfinite_bad, nonfinite_first = _value_stats(
        given, new.values, new.offsets
    )
    report_blocks(
        found, "nonfinite", encoded, nonfinite_bad, nonfinite_first, "non-finite samples didn't decode as themselves"
    )
    with np.errstate(all="ignore"):
        half_range = np.where(count > 0, hi / 2 - lo / 2, 0.0)
        # Allow 4 ULPs (2 at half scale) for quantization + reconstruction rounding.
        half_ulps = 2 * np.spacing(magnitude)
        grid_half = _grid_bounds(new) / 2
        bulk_error, escaped_error = _split_errors(given, new, half_error)
        bounds = [
            (
                "range_bound",
                np.maximum(half_range / (2.0**params.min_quantize_bits - 0.5), params.absolute_tolerance / 2),
                "range / (2^min_quantize_bits - 1/2), or absolute_tolerance",
                half_error,
            ),
            ("step_bound", grid_half, "the stored grid's", bulk_error),
            (
                "step_bound",
                np.maximum(grid_half, half_range / (2.0**params.max_quantize_bits - 1) * 2),
                "an escaped sample's: the full range's at max_quantize_bits",
                escaped_error,
            ),
        ]
        if params.target_bits_per_sample is None:
            fine = (params.noise_floor_sigma == 0) | (sizes < NOISE_FLOOR_MIN_SAMPLES)
            fine_bound = np.where(
                fine,
                np.maximum(half_range / (2.0**params.max_quantize_bits - 0.5), params.absolute_tolerance / 2),
                np.inf,
            )
            bounds.append(
                ("fine_bound", fine_bound, "range / (2^max_quantize_bits - 1/2), or absolute_tolerance", half_error)
            )
        for check, half_bound, what, block_error in bounds:
            over = encoded & (count > 0) & ~(block_error <= half_bound + half_ulps)
            for block_idx in np.flatnonzero(over)[:MAX_REPORTED].tolist():
                found.append(
                    Violation(
                        check,
                        block_idx,
                        f"max error {2 * block_error[block_idx]:.17g} over {what} {2 * half_bound[block_idx]:.17g}",
                    )
                )
    _check_stats(found, result, encoded, count, lo, hi, mean, magnitude)


def check_decimals(
    found: list[Violation],
    given: np.ndarray,
    new: Decoded,
    sizes: np.ndarray,
    encoded: np.ndarray,
    params: Params,
    decimals: int,
) -> None:
    """Checks bit-exact decimal preservation on blocks where ENCODER.md §6 guarantees it."""
    # A block is guaranteed bit-exact when all finite samples are exact k / 10^decimals with
    # |k| < 2^52 spanning < 2^(max_quantize_bits - 1) steps, decimal grids are enabled,
    # absolute_tolerance does not exceed half a decimal step, and neither target nor noise floor applies.
    if not -22 <= decimals <= 22:
        raise ValueError(f"decimals must be from -22 to 22, got {decimals}")
    if params.target_bits_per_sample is not None or params.grids == "binary" or not given.shape[0]:
        return
    if params.absolute_tolerance > 10.0**-decimals / 2:
        return
    finite = np.isfinite(given)
    with np.errstate(all="ignore"):
        k = np.round(given * 10.0**decimals) if decimals >= 0 else np.round(given / 10.0**-decimals)
        nearest = k / 10.0**decimals if decimals >= 0 else k * 10.0**-decimals
    off = finite & ~((nearest == given) & (np.abs(k) < 2.0**52))
    block_ids = np.repeat(np.arange(sizes.shape[0]), sizes)
    k = np.where(finite, k, np.nan)
    k_lo = np.full(sizes.shape[0], np.inf)
    k_hi = np.full(sizes.shape[0], -np.inf)
    np.fmin.at(k_lo, block_ids, k)
    np.fmax.at(k_hi, block_ids, k)
    checked = (
        encoded
        & (np.bincount(block_ids, off, sizes.shape[0]) == 0)
        & (k_hi - k_lo < 2.0 ** (params.max_quantize_bits - 1))
        & ((params.noise_floor_sigma == 0) | (sizes < NOISE_FLOOR_MIN_SAMPLES))
    )
    bad = np.repeat(checked, sizes) & finite & (new.values != given)
    report_samples(found, "decimal", bad, new.offsets, "decimal samples of a block guaranteed exact changed")


def _split_errors(given: np.ndarray, new: Decoded, half_error: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Splits each block's half-scale max error into bulk-sample error and escaped-sample error."""
    bulk = half_error.copy()
    escaped = np.zeros_like(half_error)
    for block_idx in np.flatnonzero(new.escaped).tolist():
        first, end = int(new.offsets[block_idx]), int(new.offsets[block_idx + 1])
        with np.errstate(all="ignore"):
            error = np.abs(given[first:end] / 2 - new.values[first:end] / 2)
        error = np.where(np.isfinite(given[first:end]) & np.isfinite(new.values[first:end]), error, 0.0)
        which = new.escaped_samples[first:end]
        bulk[block_idx] = error[~which].max(initial=0.0)
        escaped[block_idx] = error[which].max(initial=0.0)
    return bulk, escaped


@njit(nogil=True, cache=True)
def _value_stats(
    given: np.ndarray, got: np.ndarray, offsets: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Computes per-block half-scale max error, finite min/max/magnitude/count/mean, and non-finite mismatches."""
    # Run a fast branch-free pass first; fall back to _careful only on blocks containing non-finite
    # values or overflowing sums.
    num_blocks = offsets.shape[0] - 1
    half_error = np.zeros(num_blocks)
    lo = np.full(num_blocks, np.inf)
    hi = np.full(num_blocks, -np.inf)
    magnitude = np.zeros(num_blocks)
    count = np.zeros(num_blocks, np.int64)
    mean = np.zeros(num_blocks)
    bad = np.zeros(num_blocks, np.int64)
    first_bad = np.full(num_blocks, -1, np.int64)
    for block_idx in range(num_blocks):
        first, end = offsets[block_idx], offsets[block_idx + 1]
        if first == end:
            continue
        _all_finite(given, got, first, end, block_idx, half_error, lo, hi, mean)
        if np.isfinite(half_error[block_idx]) and np.isfinite(mean[block_idx]):
            count[block_idx] = end - first
        else:
            _careful(given, got, first, end, block_idx, half_error, lo, hi, count, mean, bad, first_bad)
        if count[block_idx]:
            magnitude[block_idx] = max(-lo[block_idx], hi[block_idx])
    return half_error, lo, hi, magnitude, count, mean, bad, first_bad


@njit(nogil=True, cache=True, fastmath={"reassoc", "nsz"})
def _all_finite(
    given: np.ndarray,
    got: np.ndarray,
    first: int,
    end: int,
    block_idx: int,
    half_error: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    mean: np.ndarray,
) -> None:
    """Fast branch-free pass assuming all samples are finite; poisons ``half_error`` with NaN if not."""
    error, low, high, total, poison = 0.0, np.inf, -np.inf, 0.0, 0.0
    for sample_idx in range(first, end):
        x, y = given[sample_idx], got[sample_idx]
        error = max(error, abs(0.5 * y - 0.5 * x))
        low = min(low, x)
        high = max(high, x)
        total += x
        poison += (x - x) + (y - y)
    half_error[block_idx] = error + poison
    lo[block_idx], hi[block_idx] = low, high
    mean[block_idx] = total / (end - first)


@njit(nogil=True, cache=True)
def _careful(
    given: np.ndarray,
    got: np.ndarray,
    first: int,
    end: int,
    block_idx: int,
    half_error: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    count: np.ndarray,
    mean: np.ndarray,
    bad: np.ndarray,
    first_bad: np.ndarray,
) -> None:
    """Fallback statistics pass handling non-finite samples and overflowing finite sums."""
    half_error[block_idx], lo[block_idx], hi[block_idx], count[block_idx] = 0.0, np.inf, -np.inf, 0
    total = 0.0
    for sample_idx in range(first, end):
        x, y = given[sample_idx], got[sample_idx]
        if np.isfinite(x):
            if np.isfinite(y):
                half_error[block_idx] = max(half_error[block_idx], abs(0.5 * y - 0.5 * x))
            else:
                bad[block_idx] += 1
                if first_bad[block_idx] < 0:
                    first_bad[block_idx] = sample_idx
            lo[block_idx] = min(lo[block_idx], x)
            hi[block_idx] = max(hi[block_idx], x)
            count[block_idx] += 1
            total += x
        elif not (x == y or (np.isnan(x) and np.isnan(y))):
            bad[block_idx] += 1
            if first_bad[block_idx] < 0:
                first_bad[block_idx] = sample_idx
    mean[block_idx] = total / max(count[block_idx], 1)
    if not np.isfinite(mean[block_idx]):
        # Avoid overflow by dividing each finite sample before summing.
        mean[block_idx] = 0.0
        for sample_idx in range(first, end):
            if np.isfinite(given[sample_idx]):
                mean[block_idx] += given[sample_idx] / count[block_idx]


def _check_stats(
    found: list[Violation],
    result: EncodedGroup | UpdatedGroup,
    encoded: np.ndarray,
    count: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    mean: np.ndarray,
    magnitude: np.ndarray,
) -> None:
    """Checks returned ``block_min``, ``block_max``, and ``block_mean`` for all encoded blocks."""
    blocks = np.flatnonzero(encoded)
    stats = [np.asarray(result.block_min), np.asarray(result.block_max), np.asarray(result.block_mean)]
    if any(stat.shape != blocks.shape for stat in stats):
        found.append(Violation("stats", -1, f"{[stat.shape for stat in stats]} statistics for {blocks.shape[0]} blocks"))
        return
    some = count[blocks] > 0
    expected = [np.where(some, lo[blocks], np.nan), np.where(some, hi[blocks], np.nan)]
    with np.errstate(invalid="ignore"):
        # Fastmath summation may reorder terms, causing up to 2 * count ULPs of rounding error.
        mean_tolerance = 2.0 * count[blocks] * np.spacing(magnitude[blocks])
        mean_ok = np.where(some, np.abs(stats[2] - mean[blocks]) <= mean_tolerance, np.isnan(stats[2]))
    for name, stat, want in zip(("block_min", "block_max"), stats[:2], expected, strict=True):
        for pos in np.flatnonzero(~((stat == want) | (np.isnan(stat) & np.isnan(want))))[:MAX_REPORTED].tolist():
            found.append(Violation("stats", int(blocks[pos]), f"{name} {stat[pos]:.17g}, not {want[pos]:.17g}"))
    for pos in np.flatnonzero(~mean_ok)[:MAX_REPORTED].tolist():
        want_mean = mean[blocks[pos]] if some[pos] else np.nan
        found.append(Violation("stats", int(blocks[pos]), f"block_mean {stats[2][pos]:.17g}, not {want_mean:.17g}"))


def _grid_bounds(new: Decoded) -> np.ndarray:
    """Computes each block's maximum allowed sample error implied by its stored grid metadata."""
    # Power-of-two and scaled grids allow half a grid step; unscaled decimal grids are only
    # accepted within a quarter step of the planned binary step (which is finer than 10^p).
    decimal = (new.flags & BLOCK_FLAG_DECIMAL) != 0
    scaled = (new.flags & BLOCK_FLAG_SCALED) != 0
    with np.errstate(over="ignore"):
        base = np.where(
            decimal, 10.0 ** np.clip(new.exponents, -22, 22), np.ldexp(1.0, np.clip(new.exponents, -1074, 1023))
        )
    numerators, denominators = new.scales & 0xFFFF, new.scales >> 16
    step = np.where(scaled, base * numerators / np.maximum(denominators, 1), base)
    return np.asarray(np.where(decimal & ~scaled, step / 4, step / 2), dtype=np.float64)
