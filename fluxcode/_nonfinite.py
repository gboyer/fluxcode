# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Handling of non-finite float64 values (NaN, +inf, -inf).

Provides sample-and-hold imputation and 2-bit code categorization for blocks
containing non-finite values during encoding, and exact restoration of canonical
NaN and signed infinities during decoding. Blocks without non-finite values do
not invoke this module.
"""

import math

import numpy as np
from numba import njit

from ._bitpacking import get_code
from ._format import (
    CODE_FINITE,
    CODE_NAN,
    CODE_NEG_INF,
    CODE_POS_INF,
)


@njit(nogil=True, cache=True)
def fill_nonfinite(
    samples: np.ndarray, out_held_samples: np.ndarray, out_codes: np.ndarray
) -> tuple[float, float, float, int]:
    """Imputes non-finite samples with held values and classifies sample categories.

    Replaces each non-finite sample in samples with the preceding finite value
    (or the first finite value in the block for leading non-finite runs). Records
    2-bit classification codes (finite, NaN, +inf, -inf) and calculates block
    statistics (min, max, mean, count) over finite samples only.

    Args:
        samples: 1D float64 array of input samples.
        out_held_samples: Output 1D float64 array receiving imputed sample-and-held data.
        out_codes: Output 1D uint8 array receiving 2-bit sample classification codes.

    Returns:
        A tuple of (min_val, max_val, mean_val, finite_count):
            min_val: Minimum finite value, or nan if no finite samples exist.
            max_val: Maximum finite value, or nan if no finite samples exist.
            mean_val: Arithmetic mean of finite samples, or nan if none exist.
            finite_count: Number of finite samples found in samples.
    """
    num_samples = samples.shape[0]
    first_finite_idx = -1
    # Locate index of first finite sample to initialize leading non-finite runs
    for sample_idx in range(num_samples):
        if math.isfinite(samples[sample_idx]):
            first_finite_idx = sample_idx
            break
    # Default to 0.0 if entire block is non-finite
    prev_finite_val = samples[first_finite_idx] if first_finite_idx >= 0 else 0.0
    min_val = max_val = prev_finite_val
    accum_sum = 0.0
    finite_count = 0
    for sample_idx in range(num_samples):
        val = samples[sample_idx]
        if math.isfinite(val):
            prev_finite_val = val
            # Track finite extremes and sum
            min_val = min(min_val, val)
            max_val = max(max_val, val)
            accum_sum += val
            finite_count += 1
            out_codes[sample_idx] = CODE_FINITE
        else:
            # Classify non-finite category into 2-bit code
            out_codes[sample_idx] = CODE_NAN if math.isnan(val) else (CODE_POS_INF if val > 0 else CODE_NEG_INF)
        # Store held finite value
        out_held_samples[sample_idx] = prev_finite_val
    # Return NaNs if block contains no finite samples
    if finite_count == 0:
        return math.nan, math.nan, math.nan, 0
    mean_val = accum_sum / finite_count
    # Sum overflowed float64 capacity: fall back to division before summation
    if not math.isfinite(mean_val):
        mean_val = _mean_by_division_coded(samples, out_codes, finite_count)
    return min_val, max_val, mean_val, finite_count


@njit(nogil=True, cache=True)
def _mean_by_division_coded(samples: np.ndarray, codes: np.ndarray, finite_count: int) -> float:
    """Computes mean over finite samples by division when direct sum overflows.

    Filters samples by checking classification codes rather than isfinite, which keeps
    the loop fast.

    Args:
        samples: 1D float64 array of samples.
        codes: 1D uint8 array of 2-bit classification codes.
        finite_count: Number of finite samples (must be > 0).

    Returns:
        Arithmetic mean of finite samples.
    """
    accum_mean = 0.0
    for sample_idx in range(samples.shape[0]):
        # Only accumulate samples classified as finite
        if codes[sample_idx] == CODE_FINITE:
            accum_mean += samples[sample_idx] / finite_count
    return accum_mean


@njit(inline="always")
def _restore_octet(
    code_planes: np.ndarray, plane_byte_offset: int, octet_idx: int, num_valid: int, in_out_samples: np.ndarray
) -> None:
    """Restores non-finite samples for an 8-sample octet from 2-bit code planes.

    Args:
        code_planes: 2D uint8 array of shape (2, code_octets) holding code planes.
        plane_byte_offset: Starting byte offset of the block in code planes.
        octet_idx: Zero-based octet index within the block.
        num_valid: Number of valid samples in this octet (1 to 8).
        in_out_samples: In-out 1D float64 array modified in-place with non-finite values.
    """
    plane0_byte = code_planes[0, plane_byte_offset + octet_idx]
    plane1_byte = code_planes[1, plane_byte_offset + octet_idx]
    # Fast path: skip 8-sample octet if both plane bytes are zero (all finite)
    if plane0_byte | plane1_byte:
        for bit_idx in range(num_valid):
            code = get_code(plane0_byte, plane1_byte, bit_idx)
            # Overwrite sample with canonical quiet NaN or signed infinity
            sample_offset = 8 * octet_idx + bit_idx
            if code == CODE_NAN:
                in_out_samples[sample_offset] = np.nan
            elif code == CODE_POS_INF:
                in_out_samples[sample_offset] = np.inf
            elif code == CODE_NEG_INF:
                in_out_samples[sample_offset] = -np.inf


@njit(nogil=True, cache=True)
def restore_nonfinite(code_planes: np.ndarray, plane_byte_offset: int, in_out_samples: np.ndarray) -> None:
    """Overwrites imputed samples with exact non-finite values from code planes.

    Reconstructs canonical quiet NaNs and signed infinities for a flagged block.
    Skips 8-sample octets where all samples are finite.

    Args:
        code_planes: 2D uint8 array of shape (2, code octets) holding code planes.
        plane_byte_offset: The block's first byte in each code plane.
        in_out_samples: In-out 1D float64 array of dequantized samples modified in-place.
    """
    num_samples = in_out_samples.shape[0]
    num_full_octets = num_samples // 8
    # Full octets with a constant bit count, then a partial last octet
    for octet_idx in range(num_full_octets):
        _restore_octet(code_planes, plane_byte_offset, octet_idx, 8, in_out_samples)
    if num_samples % 8:
        _restore_octet(code_planes, plane_byte_offset, num_full_octets, num_samples % 8, in_out_samples)
