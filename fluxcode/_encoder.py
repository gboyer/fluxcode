# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Encoding stages and numba kernels for float64 series compression.

Provides per-block analysis, noise floor gating, decimal detection, power-of-two
quantization, predictor order selection, residual differentiation mod 2^16, and
per-block-group bit target allocation. Composed per block group in `encode_group`.
"""

import math
from collections.abc import Callable
from typing import cast

import numpy as np
from numba import njit
from numba.cpython.unsafe.numbers import leading_zeros as _leading_zeros

from . import _extreme_magnitudes as xm
from . import _rational
from ._escapes import ESCAPE_MIN_LEN, escape_outliers, fill_nonfinite, hold_coded
from ._format import (
    BLOCK_FLAG_CODED,
    BLOCK_FLAG_DECIMAL,
    BLOCK_FLAG_IRREGULAR_TIME,
    BLOCK_FLAG_ORDER,
    BLOCK_FLAG_SCALED,
    E_MAX,
    INDEX_LIMIT,
    SCALE_MASK,
    SCALE_SHIFT,
    SCALED_E_MAX,
    SCALED_E_MIN,
    SCALED_P_MAX,
    SHORT_BLOCK_LEN,
)
from ._grids import (
    GRIDS_DECIMAL,
    GRIDS_RATIONAL,
    NO_DECIMAL,
    _common_twos,
    center_anchor,
    decimal_grid,
    floor_exponent,
    moved_off_tolerance_grid,
    plan_block_step,
    planned_multiple,
    quantize_block,
    range_exponent,
    range_scale,
    snap_anchor,
    stored_step,
    tolerance_grid,
)
from ._noise import block_noise
from ._time import CADENCE_SAMPLES

leading_zeros = cast("Callable[[int | np.integer], int]", _leading_zeros)
"""Numba intrinsic counting leading zero bits, typed as its jitted call signature."""

NOISE_MIN_LEN: int = 128
"""Smallest block the noise floor applies to. Below it a single step or spike dominates the
12th differences; from 128 up, clean signals read far below the B-bit step while white noise
passes the gate 96% of the time (ENCODER §5.2). Irregular times keep their own minimum
(_noise.MIN_ONE_SCAN_TRIPLETS = 254) for second divided differences."""

PICK_LEN: int = 256
"""Blocks up to this many samples pick their predictor order on every sample; longer ones on
PICK_WINDOWS evenly spread windows of PICK_WINDOW_LEN (the same 256 samples)."""

PICK_WINDOWS: int = 8
"""Windows a long block picks its order on."""

PICK_WINDOW_LEN: int = 32
"""Samples per order-pick window."""

NUM_HIST_CHAINS: int = 4
"""Number of interleaved histogram accumulator chains to prevent pipeline write stalls."""

NUM_BIT_LENGTH_BINS: int = 17
"""Total bit-length classes (0 through 16) for zigzagged uint16 residuals."""

UINT16_MASK: int = 0xFFFF
"""Bitmask for 16-bit unsigned integers (65535)."""

INT16_WRAP_OFFSET: int = 32768
"""Half-range offset (2^15) used for modular symmetric signed int16 wrapping."""


@njit(nogil=True, cache=True, fastmath=True)
def _min_max(samples: np.ndarray) -> tuple[float, float]:
    """Computes minimum and maximum over finite samples using four parallel chains.

    Args:
        samples: 1D float64 array of finite values.

    Returns:
        A tuple of (min_val, max_val).
    """
    num_samples = samples.shape[0]
    # Four independent accumulator chains for instruction-level parallelism
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
    # Process remaining elements
    for sample_idx in range(num_samples - num_samples % 4, num_samples):
        min_chain0 = min(min_chain0, samples[sample_idx])
        max_chain0 = max(max_chain0, samples[sample_idx])
    final_min = min(min(min_chain0, min_chain1), min(min_chain2, min_chain3))
    final_max = max(max(max_chain0, max_chain1), max(max_chain2, max_chain3))
    return final_min, final_max


@njit(nogil=True, cache=True, fastmath={"reassoc", "nsz"})
def _sum(samples: np.ndarray) -> float:
    """Computes sum of array elements with fastmath reassociation enabled.

    Args:
        samples: 1D float64 array.

    Returns:
        Accumulated sum.
    """
    accum_sum = 0.0
    for sample_idx in range(samples.shape[0]):
        # Vectorized floating-point sum
        accum_sum += samples[sample_idx]
    return accum_sum


@njit(nogil=True, cache=True)
def block_stats(samples: np.ndarray) -> tuple[float, float, float, int]:
    """Computes block minimum, maximum, mean, and identifies non-finite samples.

    Args:
        samples: 1D float64 array of samples.

    Returns:
        A tuple of (min_val, max_val, mean_val, bad_index):
            min_val: Minimum value (0.0 if a non-finite sample is present).
            max_val: Maximum value (0.0 if a non-finite sample is present).
            mean_val: Arithmetic mean of samples (0.0 if a non-finite sample is present).
            bad_index: Index of first non-finite sample, or -1 if all are finite.
    """
    num_samples = samples.shape[0]
    total_sum = _sum(samples)
    # Check if sum is non-finite: indicates non-finite sample or float overflow
    if not math.isfinite(total_sum):
        for sample_idx in range(num_samples):
            if not math.isfinite(samples[sample_idx]):
                # Return index of first non-finite sample found
                return 0.0, 0.0, 0.0, sample_idx
        # All samples are finite, but direct sum overflowed
        min_val, max_val = _min_max(samples)
        return min_val, max_val, xm.mean_by_division(samples, num_samples), -1
    min_val, max_val = _min_max(samples)
    return min_val, max_val, total_sum / num_samples, -1


@njit(inline="always")
def _order_sums(segment: np.ndarray) -> tuple[np.int64, np.int64, np.int64, np.int64, np.int64, np.int64, np.int64, np.int64]:
    """Sums of the order-0 to order-3 differences of one contiguous segment (at least 4 samples).

    Returns:
        A tuple of (sum_q, sum_sq_q, sum_d1, sum_sq_d1, sum_d2, sum_sq_d2, sum_d3, sum_sq_d3) over the
        segment's len, len - 1, len - 2 and len - 3 differences.
    """
    num_samples = segment.shape[0]
    sum_q = sum_sq_q = sum_sq_diff1 = sum_sq_diff2 = sum_sq_diff3 = np.int64(0)
    for sample_idx in range(3):
        val = np.int64(segment[sample_idx])
        sum_q += val
        sum_sq_q += val * val
    for sample_idx in range(1, 3):
        diff = np.int64(segment[sample_idx]) - segment[sample_idx - 1]
        sum_sq_diff1 += diff * diff
    diff_order2 = np.int64(segment[2]) - 2 * np.int64(segment[1]) + segment[0]
    sum_sq_diff2 += diff_order2 * diff_order2
    # Offset slices from index 0 avoid numba's negative-index checks, so the loop vectorizes
    slice_0 = segment[3:num_samples]
    slice_1 = segment[2:num_samples - 1]
    slice_2 = segment[1:num_samples - 2]
    slice_3 = segment[:num_samples - 3]
    for sample_idx in range(num_samples - 3):
        sample_a = slice_0[sample_idx]
        sample_b = slice_1[sample_idx]
        sample_c = slice_2[sample_idx]
        sample_d = slice_3[sample_idx]
        # Differences in int32 (|d3| < 2^19), widened to int64 for squaring
        diff1 = np.int32(sample_a - sample_b)
        diff2 = np.int32(sample_a - 2 * sample_b + sample_c)
        diff3 = np.int32(sample_a - 3 * sample_b + 3 * sample_c - sample_d)
        sum_q += sample_a
        sum_sq_q += np.int64(sample_a) * np.int64(sample_a)
        sum_sq_diff1 += np.int64(diff1) * np.int64(diff1)
        sum_sq_diff2 += np.int64(diff2) * np.int64(diff2)
        sum_sq_diff3 += np.int64(diff3) * np.int64(diff3)
    # Telescoping sums give exact sums of differences in O(1)
    last = num_samples - 1
    sum_diff1 = np.int64(segment[last]) - segment[0]
    sum_diff2 = (np.int64(segment[last]) - segment[last - 1]) - (np.int64(segment[1]) - segment[0])
    sum_diff3 = (
        (np.int64(segment[last]) - 2 * np.int64(segment[last - 1]) + segment[last - 2])
        - (np.int64(segment[2]) - 2 * np.int64(segment[1]) + segment[0])
    )
    return sum_q, sum_sq_q, sum_diff1, sum_sq_diff1, sum_diff2, sum_sq_diff2, sum_diff3, sum_sq_diff3


@njit(nogil=True, cache=True)
def pick_order(quantized_samples: np.ndarray, orders_mask: int) -> int:
    """Selects the predictor order (0-3) minimizing difference variance.

    A block of up to PICK_LEN samples is measured whole; a longer one on PICK_WINDOWS windows of
    PICK_WINDOW_LEN samples spread evenly over it (differences within each window), so a block
    whose character changes part way (a move, a ramp, a burst) is judged on all of it.

    Args:
        quantized_samples: 1D int32 array of the block's quantized integers (at least 4).
        orders_mask: Bitmask of allowable orders (bit k enables order k).

    Returns:
        Selected difference order (0, 1, 2, or 3). Ties favor lower orders.
    """
    num_samples = quantized_samples.shape[0]
    num_windows, window_len = 1, num_samples
    if num_samples > PICK_LEN:
        num_windows, window_len = PICK_WINDOWS, PICK_WINDOW_LEN
    sums = np.zeros(8, np.int64)
    for window_idx in range(num_windows):
        window_start = window_idx * (num_samples - window_len) // max(num_windows - 1, 1)
        window_sums = _order_sums(quantized_samples[window_start:window_start + window_len])
        for sum_idx in range(8):
            sums[sum_idx] += window_sums[sum_idx]
    best_order, min_variance = 0, 1e300
    for order_idx in range(4):
        # Skip orders not permitted by caller's bitmask
        if not (orders_mask >> order_idx) & 1:
            continue
        valid_count = num_windows * (window_len - order_idx)
        sum_diff_val, sum_sq_diff_val = sums[2 * order_idx], sums[2 * order_idx + 1]
        # Variance of the differences: (n * sum(d^2) - sum(d)^2) / n^2, n = valid_count
        variance = (valid_count * float(sum_sq_diff_val) - float(sum_diff_val) ** 2) / (valid_count * valid_count)
        if variance < min_variance:
            best_order, min_variance = order_idx, variance
    return best_order


@njit(inline="always")
def _wrap16(diff_val: int) -> np.int16:
    """Wraps integer difference into signed int16 range [-32768, 32767] mod 2^16.

    Args:
        diff_val: Integer difference value.

    Returns:
        Modularly wrapped signed 16-bit integer.
    """
    # Modular reduction: ((diff_val + INT16_WRAP_OFFSET) mod 65536) - INT16_WRAP_OFFSET
    return np.int16(((diff_val + INT16_WRAP_OFFSET) & UINT16_MASK) - INT16_WRAP_OFFSET)


@njit(nogil=True, cache=True)
def residual(quantized_samples: np.ndarray, order: int, out_residuals: np.ndarray) -> None:
    """Computes differences of the given order, mod 2^16, of the quantized samples.

    The samples are preceded by zeros, so the first `order` residuals are partial
    differences.

    Args:
        quantized_samples: 1D int32 array of quantized integers.
        order: Difference order (0, 1, 2, or 3).
        out_residuals: Output 1D int16 array receiving wrapped differences.
    """
    num_samples = quantized_samples.shape[0]
    out_residuals[0] = _wrap16(quantized_samples[0])
    if order == 0:
        for sample_idx in range(num_samples):
            out_residuals[sample_idx] = _wrap16(quantized_samples[sample_idx])
    elif order == 1:
        # First difference: sample - previous sample
        slice_curr = quantized_samples[1:]
        slice_prev = quantized_samples[:num_samples - 1]
        out_slice = out_residuals[1:]
        for sample_idx in range(num_samples - 1):
            out_slice[sample_idx] = _wrap16(slice_curr[sample_idx] - slice_prev[sample_idx])
    elif order == 2:
        # Second difference with prepended zeros: residual 1 is sample 1 - 2 * sample 0
        out_residuals[1] = _wrap16(quantized_samples[1] - 2 * quantized_samples[0])
        slice_curr = quantized_samples[2:]
        slice_prev1 = quantized_samples[1:num_samples - 1]
        slice_prev2 = quantized_samples[:num_samples - 2]
        out_slice = out_residuals[2:]
        for sample_idx in range(num_samples - 2):
            out_slice[sample_idx] = _wrap16(
                slice_curr[sample_idx] - 2 * slice_prev1[sample_idx] + slice_prev2[sample_idx]
            )
    else:
        # Third difference with prepended zeros
        out_residuals[1] = _wrap16(quantized_samples[1] - 3 * quantized_samples[0])
        out_residuals[2] = _wrap16(quantized_samples[2] - 3 * quantized_samples[1] + 3 * quantized_samples[0])
        slice_curr = quantized_samples[3:]
        slice_prev1 = quantized_samples[2:num_samples - 1]
        slice_prev2 = quantized_samples[1:num_samples - 2]
        slice_prev3 = quantized_samples[:num_samples - 3]
        out_slice = out_residuals[3:]
        for sample_idx in range(num_samples - 3):
            out_slice[sample_idx] = _wrap16(
                slice_curr[sample_idx]
                - 3 * slice_prev1[sample_idx]
                + 3 * slice_prev2[sample_idx]
                - slice_prev3[sample_idx]
            )


@njit(nogil=True, cache=True, fastmath=True)
def estimate_bits(residuals: np.ndarray) -> float:
    """Estimates compressed bits per sample via class entropy of residuals.

    Args:
        residuals: 1D int16 array of differences.

    Returns:
        Estimated bits per sample.
    """
    num_samples = residuals.shape[0]
    # Interleaved sub-histograms, so consecutive increments of the same bin don't wait on each other
    bit_length_hist = np.zeros(NUM_HIST_CHAINS * NUM_BIT_LENGTH_BINS, np.int64)
    for sample_idx in range(num_samples):
        residual_val = int(residuals[sample_idx])
        # Zigzag residual to uint16
        zigzag_val = np.uint32(((residual_val << 1) ^ (residual_val >> 15)) & UINT16_MASK)
        # Compute bit length L (0 for 0, up to 16)
        bit_length = 32 - leading_zeros(zigzag_val)
        bit_length_hist[(sample_idx & (NUM_HIST_CHAINS - 1)) * NUM_BIT_LENGTH_BINS + bit_length] += 1
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
            # Entropy of bit length L plus bits below the leading 1
            total_entropy_bits += prob * (max(length_bin - 1, 0) - math.log2(prob))
    return total_entropy_bits


@njit(nogil=True, cache=True)
def zero_share(residuals: np.ndarray) -> float:
    """The share of a block's residuals that are 0."""
    zeros = 0
    for sample_idx in range(residuals.shape[0]):
        zeros += residuals[sample_idx] == 0
    return zeros / max(residuals.shape[0], 1)


@njit(nogil=True, cache=True)
def encode_block(
    samples: np.ndarray,
    lower_bound: float,
    upper_bound: float,
    quant_exp: int,
    grids: int,
    orders_mask: int,
    scratch_quantized: np.ndarray,
    scratch_rational: np.ndarray,
    scratch_index: np.ndarray,
    out_residuals: np.ndarray,
    tol_step: float = 0.0,
) -> tuple[int, int, np.int64, float, np.int64]:
    """Encodes one block: quantizes, scales the grid, picks predictor order, and computes residuals.

    The grid (§5.3, §5.4), by grids level: GRIDS_DECIMAL and up, a decimal grid if one is detected
    (unless the block is exactly on a power-of-two grid that is coarser or that the decimal grid
    would move it off), else the power-of-two grid; GRIDS_RATIONAL, the
    index gcd N of its indices scales it (on a power-of-two grid, the power of two N shares with
    the anchor's index coarsens the exponent instead), and a power-of-two block with no common
    factor tries the rational search.

    Args:
        samples: 1D float64 array of samples.
        lower_bound: Minimum value.
        upper_bound: Maximum value.
        quant_exp: Quantization exponent.
        grids: GRIDS_BINARY, GRIDS_DECIMAL or GRIDS_RATIONAL (Params.grids).
        orders_mask: Allowed predictor orders bitmask.
        scratch_quantized: Preallocated 1D int32 scratch buffer for quantized values.
        scratch_rational: Preallocated 1D int32 scratch buffer for the rational search's indices.
        scratch_index: Preallocated 1D float64 scratch buffer for the rational search's step counts.
        out_residuals: Output 1D int16 array receiving residuals.

    Returns:
        A tuple of (flags, param, grid_index, anchor, scale): grid_index is the integer anchor of
        a decimal or scaled grid (the minimum's index), anchor the power-of-two grid's float
        anchor (unused on the others), scale N | D << 16 (0 when unscaled).
    """
    flags = 0
    anchor = lower_bound
    param_val = quant_exp
    grid_index = np.int64(0)
    scale = np.int64(0)
    decimal_detected = False
    non_constant = upper_bound > lower_bound
    range_span = upper_bound - lower_bound
    # Attempt decimal grid detection if enabled and range is within 2^1023
    if grids >= GRIDS_DECIMAL and non_constant and range_span < xm.WIDE_RANGE:
        decimal_exp, grid_index, quant_exp = decimal_grid(
            samples, lower_bound, upper_bound, quant_exp, scratch_quantized
        )
        if decimal_exp != NO_DECIMAL and tol_step > 0 and grids >= GRIDS_RATIONAL:
            tol_num, tol_fine_exp, _ = tolerance_grid(lower_bound, quant_exp, tol_step)
            if tol_num and moved_off_tolerance_grid(
                samples, tol_num, tol_fine_exp, decimal_exp, grid_index, scratch_quantized
            ):
                decimal_exp, grid_index = NO_DECIMAL, np.int64(0)
        param_val = quant_exp
        if decimal_exp != NO_DECIMAL:
            decimal_detected = True
            param_val = decimal_exp
            flags |= BLOCK_FLAG_DECIMAL
    if decimal_detected:
        gcd = _rational.index_gcd(scratch_quantized) if grids >= GRIDS_RATIONAL else 0
        # A decimal p > 0 folds into N, so the grid divides by D * 10^-p (SPEC §5): p <= 4 (10^5 > 2^16)
        if gcd > 1 and param_val <= SCALED_P_MAX + _rational.MAX_DECIMAL_FOLD:
            fold = np.int64(10) ** max(param_val - SCALED_P_MAX, 0)
            if gcd * fold <= SCALE_MASK and abs(grid_index * fold) < INDEX_LIMIT:
                _rational.divide(scratch_quantized, gcd)
                scale = (gcd * fold) | (np.int64(1) << SCALE_SHIFT)  # D = 1
                grid_index *= fold
                param_val = min(param_val, SCALED_P_MAX)
                flags |= BLOCK_FLAG_SCALED
    else:
        anchor = quantize_block(samples, lower_bound, upper_bound, quant_exp, scratch_quantized)
        base_index = snap_anchor(lower_bound, upper_bound, quant_exp)[1]
        # Exceptions (constant blocks, unfinite snapped anchors) keep their grid
        if grids >= GRIDS_RATIONAL and non_constant and not math.isnan(base_index):
            gcd = _rational.index_gcd(scratch_quantized)
            base_int = np.int64(base_index)
            if gcd > 1:
                # The power of two shared with the anchor's index coarsens the exponent: the
                # grid stays absolute and unscaled
                twos = min(_common_twos(gcd, base_int), E_MAX - quant_exp)
                if twos:
                    _rational.divide(scratch_quantized, np.int64(1) << twos)
                    gcd >>= twos
                    base_int >>= twos
                    quant_exp += twos
                    param_val = quant_exp
                if gcd > 1 and SCALED_E_MIN <= quant_exp <= SCALED_E_MAX and abs(base_int) < INDEX_LIMIT:
                    _rational.divide(scratch_quantized, gcd)
                    scale = gcd | (np.int64(1) << SCALE_SHIFT)  # D = 1
                    grid_index = base_int
                    flags |= BLOCK_FLAG_SCALED
            # No common factor (and not already constant on 2^quant_exp): search rational / tolerance grid
            elif gcd == 1 and range_span < xm.WIDE_RANGE and quant_exp >= SCALED_E_MIN:
                found, decimal, param, num, den, rational_index = _rational.detect_rational(
                    samples, lower_bound, upper_bound, math.ldexp(1.0, quant_exp), scratch_rational, scratch_index,
                )
                tol_num, tol_fine_exp, tol_base = tolerance_grid(lower_bound, quant_exp, tol_step)
                if found and tol_num:
                    # Keep tolerance's form unless an exact rational grid is strictly coarser (fixed points)
                    found_step = float(num) / den * (10.0 ** param if decimal else math.ldexp(1.0, param))
                    found = bool(found_step > tol_num * math.ldexp(1.0, tol_fine_exp) * (1.0 + 2.0 ** -40))
                if found:
                    flags |= BLOCK_FLAG_SCALED | (BLOCK_FLAG_DECIMAL if decimal else 0)
                    param_val = param
                    scale = np.int64(num) | (np.int64(den) << SCALE_SHIFT)
                    grid_index = rational_index
                    scratch_quantized[:] = scratch_rational
                elif tol_num:
                    inv_step = 1.0 / (tol_num * math.ldexp(1.0, tol_fine_exp))
                    origin = tol_base * math.ldexp(1.0, tol_fine_exp)
                    for sample_idx in range(samples.shape[0]):
                        scratch_quantized[sample_idx] = np.int32(math.floor((samples[sample_idx] - origin) * inv_step + 0.5))
                    flags |= BLOCK_FLAG_SCALED
                    param_val = tol_fine_exp
                    scale = tol_num | (np.int64(1) << SCALE_SHIFT)
                    grid_index = np.int64(tol_base)
    # Select difference order minimizing residual variance (a shift of q doesn't change it)
    selected_order = pick_order(scratch_quantized[:samples.shape[0]], orders_mask)
    grid_index, anchor = center_anchor(scratch_quantized, selected_order, flags, param_val, grid_index, anchor, scale)
    # Compute modular differences mod 2^16
    residual(scratch_quantized, selected_order, out_residuals)
    return flags | selected_order, param_val, grid_index, anchor, scale


@njit(inline="always")
def _store_anchor(
    out_value_anchors: np.ndarray,
    anchor_floats: np.ndarray,
    block_idx: int,
    flags: int,
    anchor: float,
    decimal_base: np.int64,
) -> None:
    """Writes a block's anchor: its decimal or scaled grid index, or the float64 bits of its power-of-two anchor.

    Args:
        out_value_anchors: Output 1D int64 array of length num_blocks receiving anchors.
        anchor_floats: out_value_anchors viewed as float64, made once per block group rather
            than per block.
        block_idx: Zero-based block index.
        flags: Block header byte; BLOCK_FLAG_DECIMAL or BLOCK_FLAG_SCALED select the integer anchor.
        anchor: The power-of-two grid's anchor, stored as float64 bits.
        decimal_base: Grid index of the block minimum on a decimal or scaled grid.
    """
    if flags & (BLOCK_FLAG_DECIMAL | BLOCK_FLAG_SCALED):
        out_value_anchors[block_idx] = decimal_base
    else:
        anchor_floats[block_idx] = anchor


@njit(nogil=True, cache=True)
def allocate_target(
    estimated_bits: np.ndarray,
    current_exponents: np.ndarray,
    coarsest_exponents: np.ndarray,
    block_weights: np.ndarray,
    target_bits: float,
    out_bit_reductions: np.ndarray,
) -> bool:
    """Allocates bit reductions across blocks to meet a per-block-group bit target.

    Args:
        estimated_bits: 1D float64 array of per-block estimated bits per sample.
        current_exponents: 1D int64 array of current per-block quantization exponents.
        coarsest_exponents: 1D int64 array of coarsest allowable exponents per block.
        block_weights: 1D int64 array of each block's sample count in the budget (0 for a
            block outside it, which is never coarsened).
        target_bits: Target bits per sample across the budgeted samples.
        out_bit_reductions: Output 1D int64 array receiving bit coarsening increments.

    Returns:
        True if any block is allocated a coarsening step, False otherwise.
    """
    num_blocks = estimated_bits.shape[0]
    total_bits = 0.0
    total_weight = 0
    for block_idx in range(num_blocks):
        weight = block_weights[block_idx]
        # Maximum bits block block_idx can give up without exceeding coarsest_exponents
        headroom = coarsest_exponents[block_idx] - current_exponents[block_idx]
        max_give = math.floor(max(estimated_bits[block_idx] - 1.0, 0.0) + 0.5)
        out_bit_reductions[block_idx] = min(max_give, headroom) if weight > 0 else 0
        total_bits += estimated_bits[block_idx] * weight
        total_weight += weight
    excess_bits = total_bits - target_bits * total_weight
    # Return immediately if already within target budget
    if excess_bits <= 0:
        return False
    uniform_cap = 1
    # Search for smallest uniform cap that covers the excess
    while uniform_cap < 16:
        saved_bits = 0
        for block_idx in range(num_blocks):
            saved_bits += min(uniform_cap, out_bit_reductions[block_idx]) * block_weights[block_idx]
        if saved_bits >= excess_bits:
            break
        uniform_cap += 1
    any_coarsened = False
    for block_idx in range(num_blocks):
        out_bit_reductions[block_idx] = min(uniform_cap, out_bit_reductions[block_idx])
        any_coarsened = any_coarsened or out_bit_reductions[block_idx] > 0
    return any_coarsened


@njit(nogil=True, cache=True)
def _apply_target_reallocation(
    samples: np.ndarray,
    sample_offsets: np.ndarray,
    bit_reductions: np.ndarray,
    block_exponents: np.ndarray,
    base_lower_bounds: np.ndarray,
    base_upper_bounds: np.ndarray,
    estimated_bits_per_block: np.ndarray,
    grids: int,
    orders_mask: int,
    scratch_quantized: np.ndarray,
    scratch_rational: np.ndarray,
    scratch_index: np.ndarray,
    scratch_held: np.ndarray,
    scratch_residuals: np.ndarray,
    out_block_flags: np.ndarray,
    out_grid_params: np.ndarray,
    out_grid_scales: np.ndarray,
    out_value_anchors: np.ndarray,
    out_residuals: np.ndarray,
    out_codes: np.ndarray,
) -> None:
    """Re-encodes blocks selected by target allocation and rolls back if bits do not drop.

    Args:
        samples: 1D float64 array of all samples in the block group.
        sample_offsets: 1D int64 array of sample offsets for each block.
        bit_reductions: 1D int64 array of bit reduction increments per block.
        block_exponents: 1D int64 array of current quantization exponents per block.
        base_lower_bounds: 1D float64 array of lower bounds per block.
        base_upper_bounds: 1D float64 array of upper bounds per block.
        estimated_bits_per_block: 1D float64 array of estimated bits before coarsening.
        grids: The grids level (GRIDS_BINARY, GRIDS_DECIMAL or GRIDS_RATIONAL).
        orders_mask: Bitmask of allowable predictor difference orders.
        scratch_quantized: 1D int32 scratch buffer.
        scratch_rational: 1D int32 scratch buffer for the rational search.
        scratch_index: 1D float64 scratch buffer for the rational search.
        scratch_held: 1D float64 scratch buffer for imputed non-finite samples.
        scratch_residuals: 1D int16 scratch buffer for rolling back residuals.
        out_block_flags: In-out 1D uint8 array of block flags.
        out_grid_params: In-out 1D int64 array of grid parameters.
        out_grid_scales: In-out 1D int64 array of grid scales.
        out_value_anchors: In-out 1D int64 array of value anchors.
        out_residuals: In-out 1D int16 array of residuals.
        out_codes: In-out 1D uint8 array of non-finite sample codes.
    """
    num_blocks = sample_offsets.shape[0] - 1
    anchor_floats = out_value_anchors.view(np.float64)
    for block_idx in range(num_blocks):
        if bit_reductions[block_idx] > 0:
            first_sample = sample_offsets[block_idx]
            block_len = sample_offsets[block_idx + 1] - first_sample
            block_residuals = out_residuals[first_sample:first_sample + block_len]
            prev_flags = out_block_flags[block_idx]
            prev_param = out_grid_params[block_idx]
            prev_scale = out_grid_scales[block_idx]
            prev_anchor = out_value_anchors[block_idx]
            backup_residuals = scratch_residuals[:block_len]
            backup_residuals[:] = block_residuals
            block_samples = samples[first_sample:first_sample + block_len]
            has_coded = prev_flags & BLOCK_FLAG_CODED
            if has_coded:
                # Hold coded samples (non-finite or escaped) as the first pass did
                hold_coded(block_samples, out_codes[first_sample:first_sample + block_len], scratch_held[:block_len])
                block_samples = scratch_held[:block_len]
            target_exp = min(block_exponents[block_idx] + bit_reductions[block_idx], E_MAX)
            (
                out_block_flags[block_idx], out_grid_params[block_idx], decimal_base, anchor,
                out_grid_scales[block_idx],
            ) = encode_block(
                block_samples,
                base_lower_bounds[block_idx],
                base_upper_bounds[block_idx],
                target_exp,
                grids,
                orders_mask,
                scratch_quantized[:block_len],
                scratch_rational[:block_len],
                scratch_index[:block_len],
                block_residuals,
            )
            _store_anchor(
                out_value_anchors,
                anchor_floats,
                block_idx,
                out_block_flags[block_idx],
                anchor,
                decimal_base,
            )
            if has_coded:
                out_block_flags[block_idx] |= BLOCK_FLAG_CODED
            # Roll back if re-quantization failed to reduce estimated bits
            if estimate_bits(block_residuals) >= estimated_bits_per_block[block_idx]:
                out_block_flags[block_idx] = prev_flags
                out_grid_params[block_idx] = prev_param
                out_grid_scales[block_idx] = prev_scale
                out_value_anchors[block_idx] = prev_anchor
                block_residuals[:] = backup_residuals


@njit(nogil=True, cache=True)
def encode_group(
    samples: np.ndarray,
    sample_offsets: np.ndarray,
    ticks: np.ndarray,
    time_flags: np.ndarray,
    min_bits: int,
    max_bits: int,
    orders_mask: int,
    noise_factor: float,
    target_bits: float,
    grids: int,
    tolerance_exp: int,
    tol_step: float,
    out_block_flags: np.ndarray,
    out_grid_params: np.ndarray,
    out_grid_scales: np.ndarray,
    out_value_anchors: np.ndarray,
    out_residuals: np.ndarray,
    out_codes: np.ndarray,
    out_block_minima: np.ndarray,
    out_block_maxima: np.ndarray,
    out_block_means: np.ndarray,
) -> None:
    """Encodes all blocks of a block group: their rows and summary statistics.

    Blocks of at most SHORT_BLOCK_LEN samples skip the analysis: they take the finest step
    (a decimal grid if one is detected, else the power-of-two grid) and order 0, and stay
    outside the target. An empty block gets zero flags, grid parameter and anchor, and NaN
    statistics. A block with irregular times gets a noise floor only if it has a cadence, and
    estimates it on consecutive scans (_noise.cadence).

    Args:
        samples: 1D float64 array of every sample of the block group.
        sample_offsets: 1D int64 array of the blocks' sample offsets (num_blocks + 1).
        ticks: 1D int64 array of every sample's tick, or an empty array without a time axis.
        time_flags: 1D uint8 array of the blocks' time flags (BLOCK_FLAG_IRREGULAR_TIME, as the
            time rows set them), or an empty array without a time axis.
        min_bits: Hard lower bound on quantization bits.
        max_bits: Hard upper bound on quantization bits.
        orders_mask: Bitmask of allowable predictor difference orders.
        noise_factor: Noise floor multiplier (0.0 if disabled).
        target_bits: Target bits per sample (0.0 if disabled).
        grids: The grids level (GRIDS_BINARY, GRIDS_DECIMAL or GRIDS_RATIONAL; Params.grids).
        out_block_flags: Output 1D uint8 array of length num_blocks receiving header bytes.
        out_grid_params: Output 1D int64 array of length num_blocks receiving parameters.
        out_grid_scales: Output 1D int64 array of length num_blocks receiving grid scales (N | D << 16,
            0 on unscaled blocks).
        out_value_anchors: Output 1D int64 array of length num_blocks receiving anchors (float64 bits of
            the power-of-two anchor, the grid point nearest the block minimum, or the decimal
            grid index of the minimum).
        out_residuals: Output 1D int16 array of every sample receiving residuals.
        out_codes: Output 1D uint8 array of every sample receiving sample codes (only
            flagged blocks' codes are written).
        out_block_minima: Output 1D float64 array of length num_blocks receiving block minima.
        out_block_maxima: Output 1D float64 array of length num_blocks receiving block maxima.
        out_block_means: Output 1D float64 array of length num_blocks receiving block means.
    """
    num_blocks = sample_offsets.shape[0] - 1
    max_len = 0
    for block_idx in range(num_blocks):
        max_len = max(max_len, sample_offsets[block_idx + 1] - sample_offsets[block_idx])
    scratch_quantized = np.empty(max_len, np.int32)
    scratch_rational = np.empty(max_len, np.int32)
    scratch_index = np.empty(max_len)
    scratch_held = np.empty(max_len)
    scratch_noise_diffs = np.empty(max_len)
    scratch_noise_weights = np.empty(max_len)
    timed = ticks.shape[0] > 0
    scratch_pivots = np.empty(CADENCE_SAMPLES, np.int64)
    no_codes = np.empty(0, np.uint8)
    no_ticks = ticks[:0]
    use_target = target_bits > 0
    block_exponents = np.empty(num_blocks, np.int64)
    coarsest_exponents = np.empty(num_blocks, np.int64)
    block_weights = np.zeros(num_blocks, np.int64)
    base_lower_bounds = np.empty(num_blocks)
    base_upper_bounds = np.empty(num_blocks)
    estimated_bits_per_block = np.zeros(num_blocks)
    budget_bits = np.zeros(num_blocks)
    scratch_budget = np.empty(max_len if use_target else 0, np.int16)
    anchor_floats = out_value_anchors.view(np.float64)
    for block_idx in range(num_blocks):
        first_sample = sample_offsets[block_idx]
        block_len = sample_offsets[block_idx + 1] - first_sample
        block_exponents[block_idx] = coarsest_exponents[block_idx] = 0
        out_grid_scales[block_idx] = 0
        if block_len == 0:
            out_block_flags[block_idx] = 0
            out_grid_params[block_idx] = 0
            out_value_anchors[block_idx] = 0
            out_block_minima[block_idx] = out_block_maxima[block_idx] = out_block_means[block_idx] = math.nan
            continue
        block_samples = samples[first_sample:first_sample + block_len]
        block_codes = out_codes[first_sample:first_sample + block_len]
        block_residuals = out_residuals[first_sample:first_sample + block_len]
        block_quantized = scratch_quantized[:block_len]
        block_min, block_max, block_mean, bad_sample_idx = block_stats(block_samples)
        has_coded = bad_sample_idx >= 0
        finite_count = block_len
        if has_coded:
            # Impute non-finite values with sample-and-held finite values
            block_min, block_max, block_mean, finite_count = fill_nonfinite(
                block_samples, scratch_held[:block_len], block_codes
            )
            block_samples = scratch_held[:block_len]
        # Normalize -0.0 to +0.0 to match decoder behavior
        block_min = block_min + 0.0
        block_max = block_max + 0.0
        block_mean = block_mean + 0.0
        out_block_minima[block_idx] = block_min
        out_block_maxima[block_idx] = block_max
        out_block_means[block_idx] = block_mean
        # All non-finite block defaults to zero range
        if has_coded and finite_count == 0:
            block_min = block_max = 0.0
        if block_len <= SHORT_BLOCK_LEN:
            # Too short to analyze: the finest step (or absolute_tolerance if coarser), stored without differences.
            fine_exp = range_exponent(block_min, block_max, max_bits)
            if block_max > block_min and tolerance_exp > fine_exp:
                fine_exp = min(tolerance_exp, E_MAX)
            decimal_exp, decimal_base = NO_DECIMAL, np.int64(0)
            if grids >= GRIDS_DECIMAL and block_max > block_min and block_max - block_min < xm.WIDE_RANGE:
                decimal_exp, decimal_base, fine_exp = decimal_grid(
                    block_samples, block_min, block_max, fine_exp, block_quantized
                )
            if decimal_exp != NO_DECIMAL:
                out_block_flags[block_idx] = BLOCK_FLAG_DECIMAL
                out_grid_params[block_idx] = decimal_exp
                out_value_anchors[block_idx], _ = center_anchor(
                    block_quantized, 0, BLOCK_FLAG_DECIMAL, decimal_exp, decimal_base, 0.0, np.int64(0)
                )
            else:
                out_block_flags[block_idx] = 0
                out_grid_params[block_idx] = fine_exp
                anchor = quantize_block(block_samples, block_min, block_max, fine_exp, block_quantized)
                _, anchor_floats[block_idx] = center_anchor(
                    block_quantized, 0, 0, fine_exp, np.int64(0), anchor, np.int64(0)
                )
            residual(block_quantized, 0, block_residuals)
            if has_coded:
                out_block_flags[block_idx] |= BLOCK_FLAG_CODED
            continue
        base_lower_bounds[block_idx] = block_min
        base_upper_bounds[block_idx] = block_max
        noise_sigma = noise_rho = 0.0
        # Only irregular times steer the noise floor: regular ones are as no times
        irregular = timed and (time_flags[block_idx] & BLOCK_FLAG_IRREGULAR_TIME) != 0
        if noise_factor > 0 and block_max > block_min and block_len >= NOISE_MIN_LEN:
            noise_sigma, noise_rho = block_noise(
                block_samples,
                block_codes if has_coded else no_codes,
                ticks[first_sample:first_sample + block_len] if irregular else no_ticks,
                range_scale(block_min, block_max),
                scratch_pivots,
                scratch_noise_diffs,
                scratch_noise_weights,
                math.ldexp(1.0, range_exponent(block_min, block_max, max_bits)) / noise_factor,
            )
        # Outliers stored apart (escapes), when that refines the rest of the block (§5.1.1)
        if finite_count > 0 and block_len >= ESCAPE_MIN_LEN:
            escaped, bulk_min, bulk_max = escape_outliers(
                block_samples, scratch_index[:block_len], block_codes, has_coded, block_mean, block_min, block_max,
                max(floor_exponent(noise_factor, noise_sigma, noise_rho), tolerance_exp), max_bits, grids, block_quantized,
            )
            if escaped:
                scratch_held[:block_len] = scratch_index[:block_len]
                block_samples = scratch_held[:block_len]
                block_min, block_max = bulk_min + 0.0, bulk_max + 0.0
                base_lower_bounds[block_idx], base_upper_bounds[block_idx] = block_min, block_max
                has_coded = True
                # Re-estimate noise without the escaped samples (dropouts/glitches skew the first estimate)
                if noise_factor > 0 and block_len >= NOISE_MIN_LEN and block_max > block_min:
                    noise_sigma, noise_rho = block_noise(
                        block_samples, block_codes, ticks[first_sample:first_sample + block_len] if irregular else no_ticks,
                        range_scale(block_min, block_max), scratch_pivots, scratch_noise_diffs, scratch_noise_weights,
                        math.ldexp(1.0, range_exponent(block_min, block_max, max_bits)) / noise_factor,
                    )
        # Plan quantization exponent
        planned_exp, block_tol_step = plan_block_step(
            block_samples, block_min, block_max, noise_sigma, noise_rho, noise_factor, min_bits, max_bits,
            tolerance_exp, tol_step,
        )
        block_exponents[block_idx] = planned_exp
        (
            out_block_flags[block_idx], out_grid_params[block_idx], decimal_base, anchor, out_grid_scales[block_idx],
        ) = encode_block(
            block_samples,
            block_min,
            block_max,
            planned_exp,
            grids,
            orders_mask,
            block_quantized,
            scratch_rational[:block_len],
            scratch_index[:block_len],
            block_residuals,
            block_tol_step,
        )
        _store_anchor(out_value_anchors, anchor_floats, block_idx, out_block_flags[block_idx], anchor, decimal_base)
        if has_coded:
            out_block_flags[block_idx] |= BLOCK_FLAG_CODED
        if use_target:
            # Every block counts from its planned exponent, also on an exact coarser grid (a decimal
            # grid, the index gcd's, a scaled grid), with the budget below in the same units (§5.8)
            coarsest_exponents[block_idx] = max(
                range_exponent(block_min, block_max, min_bits), block_exponents[block_idx]
            )
            estimated_bits_per_block[block_idx] = estimate_bits(block_residuals)
            # The budget counts a block on an exact coarser grid as at its planned exponent, so the
            # cap coarsens it as it would a block without, and decoded data, back on that grid,
            # re-encodes to itself. A step M * 2^planned (every power-of-two grid, N 2^e): the
            # residuals of M q, the indices on the planned grid (every power-of-two grid, a decimal
            # p >= 0, N 2^e); other grids: their nonzero residuals as if they had the log2 of the
            # step's ratio in low bits
            step = stored_step(out_block_flags[block_idx], out_grid_params[block_idx], out_grid_scales[block_idx])
            multiple = planned_multiple(step, planned_exp)
            if multiple == 1:
                budget_bits[block_idx] = estimated_bits_per_block[block_idx]
            elif multiple > 1:
                planned_quantized = scratch_rational[:block_len]
                for sample_idx in range(block_len):
                    planned_quantized[sample_idx] = block_quantized[sample_idx] * multiple
                planned_residuals = scratch_budget[:block_len]
                residual(planned_quantized, out_block_flags[block_idx] & BLOCK_FLAG_ORDER, planned_residuals)
                budget_bits[block_idx] = estimate_bits(planned_residuals)
            else:
                budget_bits[block_idx] = estimated_bits_per_block[block_idx] + max(
                    math.log2(step) - planned_exp, 0.0
                ) * (1.0 - zero_share(block_residuals))
            block_weights[block_idx] = block_len
    # Apply soft per-block-group bit target if enabled
    if use_target:
        bit_reductions = np.zeros(num_blocks, np.int64)
        if allocate_target(
            budget_bits,
            block_exponents,
            coarsest_exponents,
            block_weights,
            target_bits,
            bit_reductions,
        ):
            _apply_target_reallocation(
                samples,
                sample_offsets,
                bit_reductions,
                block_exponents,
                base_lower_bounds,
                base_upper_bounds,
                estimated_bits_per_block,
                grids,
                orders_mask,
                scratch_quantized,
                scratch_rational,
                scratch_index,
                scratch_held,
                np.empty(max_len, np.int16),
                out_block_flags,
                out_grid_params,
                out_grid_scales,
                out_value_anchors,
                out_residuals,
                out_codes,
            )
