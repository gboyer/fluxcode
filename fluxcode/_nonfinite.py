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

from . import _extreme_magnitudes as xm
from ._bitpacking import code_planes_view, get_code
from ._format import (
    CODE_ESCAPE,
    CODE_FINITE,
    CODE_NAN,
    CODE_REPEAT,
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
            out_codes[sample_idx] = CODE_NAN if math.isnan(val) else CODE_ESCAPE  # +-inf: escaped values
        # Store held finite value
        out_held_samples[sample_idx] = prev_finite_val
    # Return NaNs if block contains no finite samples
    if finite_count == 0:
        return math.nan, math.nan, math.nan, 0
    mean_val = accum_sum / finite_count
    # Sum overflowed float64 capacity: fall back to division before summation
    if not math.isfinite(mean_val):
        mean_val = xm.mean_by_division(samples, finite_count)
    return min_val, max_val, mean_val, finite_count


@njit(nogil=True, cache=True)
def hold_coded(samples: np.ndarray, codes: np.ndarray, out_held: np.ndarray) -> None:
    """Sample-and-holds every coded (non-finite or escaped) sample from the previous uncoded one
    (the first uncoded one for a leading run; 0.0 if none)."""
    num_samples = samples.shape[0]
    previous = 0.0
    for sample_idx in range(num_samples):
        if codes[sample_idx] == CODE_FINITE:
            previous = samples[sample_idx]
            break
    for sample_idx in range(num_samples):
        if codes[sample_idx] == CODE_FINITE:
            previous = samples[sample_idx]
        out_held[sample_idx] = previous


@njit(nogil=True, cache=True)
def restore_nonfinite(
    code_planes: np.ndarray, plane_byte_offset: int, escapes: np.ndarray, in_out_samples: np.ndarray
) -> None:
    """Overwrites a flagged block's coded samples: NaN, the next escaped value, or the previous again.

    Args:
        code_planes: 2D uint8 array of shape (2, code octets) holding code planes.
        plane_byte_offset: The block's first byte in each code plane.
        escapes: 1D float64 array of the block's escaped values, in sample order.
        in_out_samples: In-out 1D float64 array of dequantized samples modified in-place.
    """
    num_samples = in_out_samples.shape[0]
    next_escape = 0
    for octet_idx in range((num_samples + 7) // 8):
        plane0_byte = code_planes[0, plane_byte_offset + octet_idx]
        plane1_byte = code_planes[1, plane_byte_offset + octet_idx]
        # Skip an octet whose samples are all uncoded
        if not plane0_byte | plane1_byte:
            continue
        for bit_idx in range(min(8, num_samples - 8 * octet_idx)):
            code = get_code(plane0_byte, plane1_byte, bit_idx)
            sample_offset = 8 * octet_idx + bit_idx
            if code == CODE_NAN:
                in_out_samples[sample_offset] = np.nan
            elif code == CODE_ESCAPE:
                in_out_samples[sample_offset] = escapes[next_escape]
                next_escape += 1
            elif code == CODE_REPEAT:
                in_out_samples[sample_offset] = escapes[next_escape - 1]


@njit(nogil=True, cache=True)
def escape_offsets(
    raw_group: np.ndarray, sample_offsets: np.ndarray, octet_offsets: np.ndarray, code_offsets: np.ndarray,
    has_time: bool, out_offsets: np.ndarray,
) -> bool:
    """Each block's first escaped value (out_offsets, num_blocks + 1), from the escape codes of the
    flagged blocks; False if a block repeats an escape before having one (corrupt)."""
    num_blocks = sample_offsets.shape[0] - 1
    num_octets = int(octet_offsets[num_blocks])
    code_planes = code_planes_view(raw_group, num_blocks, num_octets, int(code_offsets[num_blocks]), has_time)
    out_offsets[0] = 0
    for block_idx in range(num_blocks):
        count = 0
        if code_offsets[block_idx + 1] > code_offsets[block_idx]:
            seen = False
            for octet_idx in range(code_offsets[block_idx + 1] - code_offsets[block_idx]):
                plane0_byte = code_planes[0, code_offsets[block_idx] + octet_idx]
                plane1_byte = code_planes[1, code_offsets[block_idx] + octet_idx]
                if not plane0_byte | plane1_byte:
                    continue
                # Only the block's samples: decoders ignore padding bits
                block_len = sample_offsets[block_idx + 1] - sample_offsets[block_idx]
                for bit_idx in range(min(8, block_len - 8 * octet_idx)):
                    code = get_code(plane0_byte, plane1_byte, bit_idx)
                    if code == CODE_ESCAPE:
                        count += 1
                        seen = True
                    elif code == CODE_REPEAT and not seen:
                        return False
        out_offsets[block_idx + 1] = out_offsets[block_idx] + count
    return True
