# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Per-block and per-group encoding kernels for float64 time-series compression.

Combines block summary statistics, non-finite/outlier escaping, white-noise floor
planning, grid selection (binary, decimal, GCD-scaled, rational, and tolerance-scaled),
predictor order selection, modular 16-bit differencing, and soft per-group bit-budget
reallocation in ``encode_group``.
"""

import math
from typing import NamedTuple

import numpy as np
from numba import njit

from . import _extreme_magnitudes as xm
from . import _rational
from ._basic_stats import (
    UINT16_MASK,
    block_stats,
    estimate_residual_bits,
    zero_residual_fraction,
)
from ._escapes import (
    ESCAPE_MIN_LEN,
    escape_outliers,
    hold_exception_samples,
    replace_nonfinite,
)
from ._format import (
    BLOCK_FLAG_DECIMAL,
    BLOCK_FLAG_IRREGULAR_TIME,
    BLOCK_FLAG_ORDER,
    BLOCK_FLAG_SAMPLE_CODES,
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
    block_grid_step,
    center_anchor,
    choose_decimal_or_binary_grid,
    common_trailing_zeros,
    floor_exponent,
    moved_off_tolerance_grid,
    on_binary_grid,
    plan_block_step,
    quantize_power_of_two,
    snap_anchor,
    snapped_exponent_for_range,
    span_binary_exponent,
    step_multiple_of_planned,
    tolerance_scaled_grid,
)
from ._noise import estimate_block_noise
from ._time import CADENCE_SAMPLES

NOISE_MIN_LEN: int = 128
"""Minimum regular block length eligible for noise-floor estimation."""

PICK_LEN: int = 256
"""Maximum block length evaluated on every sample when selecting predictor order."""

PICK_WINDOWS: int = 8
"""Number of evenly spread windows evaluated for predictor order on longer blocks."""

PICK_WINDOW_LEN: int = 32
"""Samples per predictor-order evaluation window on longer blocks."""

INT16_WRAP_OFFSET: int = 32768
"""Half-range offset (2^15) used for symmetric modular wrapping into signed int16."""


class _EncoderState(NamedTuple):
    """Shared scratch buffers and group output arrays passed across encoder passes."""

    quantized: np.ndarray
    rational: np.ndarray
    rational_index: np.ndarray
    held: np.ndarray
    block_flags: np.ndarray
    grid_exponents: np.ndarray
    grid_scales: np.ndarray
    block_anchors: np.ndarray
    residuals: np.ndarray
    sample_codes: np.ndarray


# ======================================================================================
# Predictor order selection & modular differencing
# ======================================================================================


@njit(inline="always")
def _order_sums(
    segment: np.ndarray,
) -> tuple[
    np.int64,
    np.int64,
    np.int64,
    np.int64,
    np.int64,
    np.int64,
    np.int64,
    np.int64,
]:
    """Computes exact sum and sum-of-squares for difference orders 0..3 on a segment."""
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
    # Zero-based offset slices avoid negative-index checks so LLVM vectorizes the loop.
    slice_0 = segment[3:num_samples]
    slice_1 = segment[2 : num_samples - 1]
    slice_2 = segment[1 : num_samples - 2]
    slice_3 = segment[: num_samples - 3]
    for sample_idx in range(num_samples - 3):
        sample_a = slice_0[sample_idx]
        sample_b = slice_1[sample_idx]
        sample_c = slice_2[sample_idx]
        sample_d = slice_3[sample_idx]
        # Differences stay within int32 (|d3| < 2^19) before widening to int64.
        diff1 = np.int32(sample_a - sample_b)
        diff2 = np.int32(sample_a - 2 * sample_b + sample_c)
        diff3 = np.int32(sample_a - 3 * sample_b + 3 * sample_c - sample_d)
        sum_q += sample_a
        sum_sq_q += np.int64(sample_a) * np.int64(sample_a)
        sum_sq_diff1 += np.int64(diff1) * np.int64(diff1)
        sum_sq_diff2 += np.int64(diff2) * np.int64(diff2)
        sum_sq_diff3 += np.int64(diff3) * np.int64(diff3)
    # Telescoping boundary identities give exact sums of orders 1-3 in O(1).
    last = num_samples - 1
    sum_diff1 = np.int64(segment[last]) - segment[0]
    sum_diff2 = (np.int64(segment[last]) - segment[last - 1]) - (
        np.int64(segment[1]) - segment[0]
    )
    sum_diff3 = (
        np.int64(segment[last]) - 2 * np.int64(segment[last - 1]) + segment[last - 2]
    ) - (np.int64(segment[2]) - 2 * np.int64(segment[1]) + segment[0])
    return (
        sum_q,
        sum_sq_q,
        sum_diff1,
        sum_sq_diff1,
        sum_diff2,
        sum_sq_diff2,
        sum_diff3,
        sum_sq_diff3,
    )


@njit(nogil=True, cache=True)
def select_predictor_order(quantized_samples: np.ndarray, orders_mask: int) -> int:
    """Selects the allowed predictor difference order (0-3) minimizing variance."""
    num_samples = quantized_samples.shape[0]
    num_windows, window_len = 1, num_samples
    if num_samples > PICK_LEN:
        num_windows, window_len = PICK_WINDOWS, PICK_WINDOW_LEN
    sums = np.zeros(8, np.int64)
    for window_idx in range(num_windows):
        window_start = (
            window_idx * (num_samples - window_len) // max(num_windows - 1, 1)
        )
        window_sums = _order_sums(
            quantized_samples[window_start : window_start + window_len]
        )
        for sum_idx in range(8):
            sums[sum_idx] += window_sums[sum_idx]
    best_order, min_variance = 0, 1e300
    for order_idx in range(4):
        if not (orders_mask >> order_idx) & 1:
            continue
        valid_count = num_windows * (window_len - order_idx)
        sum_diff_val, sum_sq_diff_val = sums[2 * order_idx], sums[2 * order_idx + 1]
        variance = (valid_count * float(sum_sq_diff_val) - float(sum_diff_val) ** 2) / (
            valid_count * valid_count
        )
        if variance < min_variance:
            best_order, min_variance = order_idx, variance
    return best_order


@njit(inline="always")
def _wrap16(diff_val: int) -> np.int16:
    """Wraps an integer difference modularly into signed int16 ``[-32768, 32767]``."""
    return np.int16(((diff_val + INT16_WRAP_OFFSET) & UINT16_MASK) - INT16_WRAP_OFFSET)


@njit(nogil=True, cache=True)
def compute_residuals(
    quantized_samples: np.ndarray, order: int, out_residuals: np.ndarray
) -> None:
    """Computes ``order``-th differences mod ``2^16`` of ``quantized_samples``."""
    num_samples = quantized_samples.shape[0]
    out_residuals[0] = _wrap16(quantized_samples[0])
    if order == 0:
        for sample_idx in range(num_samples):
            out_residuals[sample_idx] = _wrap16(quantized_samples[sample_idx])
    elif order == 1:
        slice_curr = quantized_samples[1:]
        slice_prev = quantized_samples[: num_samples - 1]
        out_slice = out_residuals[1:]
        for sample_idx in range(num_samples - 1):
            out_slice[sample_idx] = _wrap16(
                slice_curr[sample_idx] - slice_prev[sample_idx]
            )
    elif order == 2:
        out_residuals[1] = _wrap16(quantized_samples[1] - 2 * quantized_samples[0])
        slice_curr = quantized_samples[2:]
        slice_prev1 = quantized_samples[1 : num_samples - 1]
        slice_prev2 = quantized_samples[: num_samples - 2]
        out_slice = out_residuals[2:]
        for sample_idx in range(num_samples - 2):
            out_slice[sample_idx] = _wrap16(
                slice_curr[sample_idx]
                - 2 * slice_prev1[sample_idx]
                + slice_prev2[sample_idx]
            )
    else:
        out_residuals[1] = _wrap16(quantized_samples[1] - 3 * quantized_samples[0])
        out_residuals[2] = _wrap16(
            quantized_samples[2] - 3 * quantized_samples[1] + 3 * quantized_samples[0]
        )
        slice_curr = quantized_samples[3:]
        slice_prev1 = quantized_samples[2 : num_samples - 1]
        slice_prev2 = quantized_samples[1 : num_samples - 2]
        slice_prev3 = quantized_samples[: num_samples - 3]
        out_slice = out_residuals[3:]
        for sample_idx in range(num_samples - 3):
            out_slice[sample_idx] = _wrap16(
                slice_curr[sample_idx]
                - 3 * slice_prev1[sample_idx]
                + 3 * slice_prev2[sample_idx]
                - slice_prev3[sample_idx]
            )


# ======================================================================================
# Single-block grid selection & encoding
# ======================================================================================


@njit(inline="always")
def _try_scale_decimal_grid(
    quantized: np.ndarray, grid_exp: int, anchor_index: np.int64
) -> tuple[int, int, np.int64, np.int64]:
    """Divides out a common integer factor from decimal grid indices when possible."""
    gcd = _rational.index_gcd(quantized)
    if gcd > 1 and grid_exp <= SCALED_P_MAX + _rational.MAX_DECIMAL_FOLD:
        fold = np.int64(10) ** max(grid_exp - SCALED_P_MAX, 0)
        if gcd * fold <= SCALE_MASK and abs(anchor_index * fold) < INDEX_LIMIT:
            _rational.divide_indices_in_place(quantized, gcd)
            scale = (gcd * fold) | (np.int64(1) << SCALE_SHIFT)
            return (
                BLOCK_FLAG_SCALED,
                min(grid_exp, SCALED_P_MAX),
                anchor_index * fold,
                scale,
            )
    return 0, grid_exp, anchor_index, np.int64(0)


@njit(inline="always")
def _quantize_tolerance_grid(
    samples: np.ndarray,
    tol_num: np.int64,
    tol_fine_exp: int,
    tol_base: float,
    out_quantized: np.ndarray,
) -> tuple[int, int, np.int64, np.int64]:
    """Quantizes ``samples`` onto an absolute-tolerance scaled binary grid."""
    inv_step = 1.0 / (tol_num * math.ldexp(1.0, tol_fine_exp))
    origin = tol_base * math.ldexp(1.0, tol_fine_exp)
    for sample_idx in range(samples.shape[0]):
        out_quantized[sample_idx] = np.int32(
            math.floor((samples[sample_idx] - origin) * inv_step + 0.5)
        )
    return (
        BLOCK_FLAG_SCALED,
        tol_fine_exp,
        np.int64(tol_base),
        tol_num | (np.int64(1) << SCALE_SHIFT),
    )


@njit(inline="always")
def _try_scale_binary_grid(
    samples: np.ndarray,
    lower_bound: float,
    upper_bound: float,
    quant_exp: int,
    base_int: np.int64,
    tol_step: float,
    scratch_quantized: np.ndarray,
    scratch_rational: np.ndarray,
    scratch_index: np.ndarray,
) -> tuple[int, int, np.int64, np.int64]:
    """Attempts GCD reduction, rational step search, or tolerance scaling."""
    range_span = upper_bound - lower_bound
    gcd = _rational.index_gcd(scratch_quantized)
    if gcd > 1 and (tol_step == 0.0 or on_binary_grid(samples, quant_exp)):
        twos = min(common_trailing_zeros(gcd, base_int), E_MAX - quant_exp)
        if twos:
            _rational.divide_indices_in_place(scratch_quantized, np.int64(1) << twos)
            gcd >>= twos
            base_int >>= twos
            quant_exp += twos
        if (
            gcd > 1
            and SCALED_E_MIN <= quant_exp <= SCALED_E_MAX
            and abs(base_int) < INDEX_LIMIT
        ):
            _rational.divide_indices_in_place(scratch_quantized, gcd)
            return (
                BLOCK_FLAG_SCALED,
                quant_exp,
                base_int,
                gcd | (np.int64(1) << SCALE_SHIFT),
            )
        return 0, quant_exp, np.int64(0), np.int64(0)
    if gcd >= 1 and range_span < xm.WIDE_RANGE and quant_exp >= SCALED_E_MIN:
        found, decimal, param, scale_num, scale_den, rational_index = (
            _rational.detect_rational_grid(
                samples,
                lower_bound,
                upper_bound,
                math.ldexp(1.0, quant_exp),
                scratch_rational,
                scratch_index,
            )
        )
        tol_num, tol_fine_exp, tol_base = tolerance_scaled_grid(
            lower_bound, quant_exp, tol_step
        )
        if found and tol_num:
            found_step = (
                float(scale_num)
                / scale_den
                * (10.0**param if decimal else math.ldexp(1.0, param))
            )
            found = bool(
                found_step > tol_num * math.ldexp(1.0, tol_fine_exp) * (1.0 + 2.0**-40)
            )
        if found:
            scratch_quantized[:] = scratch_rational
            extra_flags = BLOCK_FLAG_SCALED | (BLOCK_FLAG_DECIMAL if decimal else 0)
            scale = np.int64(scale_num) | (np.int64(scale_den) << SCALE_SHIFT)
            return extra_flags, param, rational_index, scale
        if tol_num:
            return _quantize_tolerance_grid(
                samples, tol_num, tol_fine_exp, tol_base, scratch_quantized
            )
    return 0, quant_exp, np.int64(0), np.int64(0)


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
    """Encodes a single block: selects grid, quantizes, picks order, and diffs."""
    flags = 0
    anchor = lower_bound
    grid_exp = quant_exp
    anchor_index = np.int64(0)
    grid_scale = np.int64(0)
    decimal_detected = False
    non_constant = upper_bound > lower_bound
    range_span = upper_bound - lower_bound
    if grids >= GRIDS_DECIMAL and non_constant and range_span < xm.WIDE_RANGE:
        decimal_exp, anchor_index, quant_exp = choose_decimal_or_binary_grid(
            samples, lower_bound, upper_bound, quant_exp, scratch_quantized
        )
        if decimal_exp != NO_DECIMAL and tol_step > 0 and grids >= GRIDS_RATIONAL:
            tol_num, tol_fine_exp, _ = tolerance_scaled_grid(
                lower_bound, quant_exp, tol_step
            )
            if tol_num and moved_off_tolerance_grid(
                samples,
                tol_num,
                tol_fine_exp,
                decimal_exp,
                anchor_index,
                scratch_quantized,
            ):
                decimal_exp, anchor_index = NO_DECIMAL, np.int64(0)
        grid_exp = quant_exp
        if decimal_exp != NO_DECIMAL:
            decimal_detected = True
            grid_exp = decimal_exp
            flags |= BLOCK_FLAG_DECIMAL
    if decimal_detected:
        if grids >= GRIDS_RATIONAL:
            extra_flags, grid_exp, anchor_index, grid_scale = _try_scale_decimal_grid(
                scratch_quantized, grid_exp, anchor_index
            )
            flags |= extra_flags
    else:
        anchor = quantize_power_of_two(
            samples, lower_bound, upper_bound, quant_exp, scratch_quantized
        )
        base_index = snap_anchor(lower_bound, upper_bound, quant_exp)[1]
        if grids >= GRIDS_RATIONAL and non_constant and not math.isnan(base_index):
            extra_flags, grid_exp, anchor_index, grid_scale = _try_scale_binary_grid(
                samples,
                lower_bound,
                upper_bound,
                quant_exp,
                np.int64(base_index),
                tol_step,
                scratch_quantized,
                scratch_rational,
                scratch_index,
            )
            flags |= extra_flags
    # Predictor variance is invariant to constant shifts, so select order first.
    selected_order = select_predictor_order(
        scratch_quantized[: samples.shape[0]], orders_mask
    )
    anchor_index, anchor = center_anchor(
        scratch_quantized,
        selected_order,
        flags,
        grid_exp,
        anchor_index,
        anchor,
        grid_scale,
    )
    compute_residuals(scratch_quantized, selected_order, out_residuals)
    return flags | selected_order, grid_exp, anchor_index, anchor, grid_scale


@njit(inline="always")
def _store_anchor(
    out_block_anchors: np.ndarray,
    anchor_floats: np.ndarray,
    block_idx: int,
    flags: int,
    anchor: float,
    anchor_index: np.int64,
) -> None:
    """Stores a block's anchor as either an int64 grid index or float64 bits."""
    if flags & (BLOCK_FLAG_DECIMAL | BLOCK_FLAG_SCALED):
        out_block_anchors[block_idx] = anchor_index
    else:
        anchor_floats[block_idx] = anchor


@njit(inline="always")
def _encode_short_block(
    block_samples: np.ndarray,
    block_min: float,
    block_max: float,
    max_bits: int,
    grids: int,
    tolerance_exp: int,
    has_sample_codes: bool,
    block_quantized: np.ndarray,
    block_residuals: np.ndarray,
) -> tuple[int, int, np.int64, float]:
    """Encodes a short block (``<= SHORT_BLOCK_LEN``) at order 0."""
    fine_exp = snapped_exponent_for_range(block_min, block_max, max_bits)
    if block_max > block_min and tolerance_exp > fine_exp:
        fine_exp = min(tolerance_exp, E_MAX)
    decimal_exp, anchor_index = NO_DECIMAL, np.int64(0)
    if (
        grids >= GRIDS_DECIMAL
        and block_max > block_min
        and block_max - block_min < xm.WIDE_RANGE
    ):
        decimal_exp, anchor_index, fine_exp = choose_decimal_or_binary_grid(
            block_samples, block_min, block_max, fine_exp, block_quantized
        )
    if decimal_exp != NO_DECIMAL:
        flags = BLOCK_FLAG_DECIMAL
        grid_exp = decimal_exp
        anchor_index, anchor = center_anchor(
            block_quantized,
            0,
            BLOCK_FLAG_DECIMAL,
            decimal_exp,
            anchor_index,
            0.0,
            np.int64(0),
        )
    else:
        flags = 0
        grid_exp = fine_exp
        anchor = quantize_power_of_two(
            block_samples, block_min, block_max, fine_exp, block_quantized
        )
        anchor_index, anchor = center_anchor(
            block_quantized, 0, 0, fine_exp, np.int64(0), anchor, np.int64(0)
        )
    compute_residuals(block_quantized, 0, block_residuals)
    if has_sample_codes:
        flags |= BLOCK_FLAG_SAMPLE_CODES
    return flags, grid_exp, anchor_index, anchor


# ======================================================================================
# Soft bit-target budget allocation
# ======================================================================================


@njit(nogil=True, cache=True)
def allocate_bit_reductions(
    estimated_bits: np.ndarray,
    current_exponents: np.ndarray,
    coarsest_exponents: np.ndarray,
    block_weights: np.ndarray,
    target_bits: float,
    out_bit_reductions: np.ndarray,
) -> bool:
    """Allocates per-block bit coarsening increments to meet a soft group bit target."""
    num_blocks = estimated_bits.shape[0]
    total_bits = 0.0
    total_weight = 0
    for block_idx in range(num_blocks):
        weight = block_weights[block_idx]
        headroom = coarsest_exponents[block_idx] - current_exponents[block_idx]
        max_give = math.floor(max(estimated_bits[block_idx] - 1.0, 0.0) + 0.5)
        out_bit_reductions[block_idx] = min(max_give, headroom) if weight > 0 else 0
        total_bits += estimated_bits[block_idx] * weight
        total_weight += weight
    excess_bits = total_bits - target_bits * total_weight
    if excess_bits <= 0:
        return False
    uniform_cap = 1
    while uniform_cap < 16:
        saved_bits = 0
        for block_idx in range(num_blocks):
            saved_bits += (
                min(uniform_cap, out_bit_reductions[block_idx])
                * block_weights[block_idx]
            )
        if saved_bits >= excess_bits:
            break
        uniform_cap += 1
    any_coarsened = False
    for block_idx in range(num_blocks):
        out_bit_reductions[block_idx] = min(uniform_cap, out_bit_reductions[block_idx])
        any_coarsened = any_coarsened or out_bit_reductions[block_idx] > 0
    return any_coarsened


@njit(inline="always")
def _budget_bits_at_planned_step(
    flags: int,
    grid_exp: int,
    grid_scale: np.int64,
    planned_exp: int,
    block_quantized: np.ndarray,
    block_residuals: np.ndarray,
    estimated_bits: float,
    scratch_quantized: np.ndarray,
    scratch_residuals: np.ndarray,
) -> float:
    """Computes a block's budget bit-rate expressed at its planned power-of-two step."""
    step = block_grid_step(flags, grid_exp, grid_scale)
    multiple = step_multiple_of_planned(step, planned_exp)
    if multiple == 1:
        return estimated_bits
    if multiple > 1:
        for sample_idx in range(block_quantized.shape[0]):
            scratch_quantized[sample_idx] = block_quantized[sample_idx] * multiple
        compute_residuals(
            scratch_quantized, flags & BLOCK_FLAG_ORDER, scratch_residuals
        )
        return estimate_residual_bits(scratch_residuals)
    return estimated_bits + max(math.log2(step) - planned_exp, 0.0) * (
        1.0 - zero_residual_fraction(block_residuals)
    )


@njit(nogil=True, cache=True)
def _reencode_with_bit_reductions(
    samples: np.ndarray,
    sample_offsets: np.ndarray,
    bit_reductions: np.ndarray,
    block_exponents: np.ndarray,
    base_lower_bounds: np.ndarray,
    base_upper_bounds: np.ndarray,
    estimated_bits_per_block: np.ndarray,
    grids: int,
    orders_mask: int,
    scratch_residuals: np.ndarray,
    state: _EncoderState,
) -> None:
    """Re-encodes blocks for bit-target allocation, rolling back if not smaller."""
    num_blocks = sample_offsets.shape[0] - 1
    anchor_floats = state.block_anchors.view(np.float64)
    for block_idx in range(num_blocks):
        if bit_reductions[block_idx] <= 0:
            continue
        first_sample = sample_offsets[block_idx]
        block_len = sample_offsets[block_idx + 1] - first_sample
        block_residuals = state.residuals[first_sample : first_sample + block_len]
        prev_flags = state.block_flags[block_idx]
        prev_exp = state.grid_exponents[block_idx]
        prev_scale = state.grid_scales[block_idx]
        prev_anchor = state.block_anchors[block_idx]
        backup_residuals = scratch_residuals[:block_len]
        backup_residuals[:] = block_residuals
        block_samples = samples[first_sample : first_sample + block_len]
        has_sample_codes = prev_flags & BLOCK_FLAG_SAMPLE_CODES
        if has_sample_codes:
            # Re-replace exception samples identically to the first pass.
            hold_exception_samples(
                block_samples,
                state.sample_codes[first_sample : first_sample + block_len],
                state.held[:block_len],
            )
            block_samples = state.held[:block_len]
        target_exp = min(block_exponents[block_idx] + bit_reductions[block_idx], E_MAX)
        (
            state.block_flags[block_idx],
            state.grid_exponents[block_idx],
            anchor_index,
            anchor,
            state.grid_scales[block_idx],
        ) = encode_block(
            block_samples,
            base_lower_bounds[block_idx],
            base_upper_bounds[block_idx],
            target_exp,
            grids,
            orders_mask,
            state.quantized[:block_len],
            state.rational[:block_len],
            state.rational_index[:block_len],
            block_residuals,
        )
        _store_anchor(
            state.block_anchors,
            anchor_floats,
            block_idx,
            state.block_flags[block_idx],
            anchor,
            anchor_index,
        )
        if has_sample_codes:
            state.block_flags[block_idx] |= BLOCK_FLAG_SAMPLE_CODES
        if (
            estimate_residual_bits(block_residuals)
            >= estimated_bits_per_block[block_idx]
        ):
            state.block_flags[block_idx] = prev_flags
            state.grid_exponents[block_idx] = prev_exp
            state.grid_scales[block_idx] = prev_scale
            state.block_anchors[block_idx] = prev_anchor
            block_residuals[:] = backup_residuals


# ======================================================================================
# Block-group encoding driver
# ======================================================================================


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
    out_grid_exponents: np.ndarray,
    out_grid_scales: np.ndarray,
    out_block_anchors: np.ndarray,
    out_residuals: np.ndarray,
    out_sample_codes: np.ndarray,
    out_block_minima: np.ndarray,
    out_block_maxima: np.ndarray,
    out_block_means: np.ndarray,
) -> None:
    """Encodes all blocks of a block group and computes per-block summary statistics.

    Empty blocks receive zero flags/exponent/anchor and NaN statistics. Blocks of at
    most ``SHORT_BLOCK_LEN`` samples skip noise, outlier, and target-budget analysis
    and encode at order 0.
    """
    # Allocate per-group scratch buffers once, then run each block through:
    # (1) summary stats & non-finite replacement, (2) short-block fast path, (3) noise
    # estimation & outlier escaping, (4) step planning & block encoding, and (5)
    # optional soft group bit-target reallocation across all budgeted blocks.
    num_blocks = sample_offsets.shape[0] - 1
    max_len = 0
    for block_idx in range(num_blocks):
        max_len = max(
            max_len, sample_offsets[block_idx + 1] - sample_offsets[block_idx]
        )
    state = _EncoderState(
        np.empty(max_len, np.int32),
        np.empty(max_len, np.int32),
        np.empty(max_len),
        np.empty(max_len),
        out_block_flags,
        out_grid_exponents,
        out_grid_scales,
        out_block_anchors,
        out_residuals,
        out_sample_codes,
    )
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
    anchor_floats = out_block_anchors.view(np.float64)
    for block_idx in range(num_blocks):
        first_sample = sample_offsets[block_idx]
        block_len = sample_offsets[block_idx + 1] - first_sample
        block_exponents[block_idx] = coarsest_exponents[block_idx] = 0
        out_grid_scales[block_idx] = 0
        if block_len == 0:
            out_block_flags[block_idx] = 0
            out_grid_exponents[block_idx] = 0
            out_block_anchors[block_idx] = 0
            out_block_minima[block_idx] = math.nan
            out_block_maxima[block_idx] = math.nan
            out_block_means[block_idx] = math.nan
            continue
        block_samples = samples[first_sample : first_sample + block_len]
        block_codes = out_sample_codes[first_sample : first_sample + block_len]
        block_residuals = out_residuals[first_sample : first_sample + block_len]
        block_quantized = state.quantized[:block_len]
        block_min, block_max, block_mean, bad_sample_idx = block_stats(block_samples)
        has_sample_codes = bad_sample_idx >= 0
        finite_count = block_len
        if has_sample_codes:
            block_min, block_max, block_mean, finite_count = replace_nonfinite(
                block_samples, state.held[:block_len], block_codes
            )
            block_samples = state.held[:block_len]
        # Canonicalize -0.0 to +0.0 so summary stats match decoder output.
        block_min = block_min + 0.0
        block_max = block_max + 0.0
        block_mean = block_mean + 0.0
        out_block_minima[block_idx] = block_min
        out_block_maxima[block_idx] = block_max
        out_block_means[block_idx] = block_mean
        if has_sample_codes and finite_count == 0:
            block_min = block_max = 0.0
        if block_len <= SHORT_BLOCK_LEN:
            flags, grid_exp, anchor_index, anchor = _encode_short_block(
                block_samples,
                block_min,
                block_max,
                max_bits,
                grids,
                tolerance_exp,
                has_sample_codes,
                block_quantized,
                block_residuals,
            )
            out_block_flags[block_idx] = flags
            out_grid_exponents[block_idx] = grid_exp
            _store_anchor(
                out_block_anchors, anchor_floats, block_idx, flags, anchor, anchor_index
            )
            continue
        base_lower_bounds[block_idx] = block_min
        base_upper_bounds[block_idx] = block_max
        noise_sigma = noise_rho = 0.0
        irregular = timed and (time_flags[block_idx] & BLOCK_FLAG_IRREGULAR_TIME) != 0
        if noise_factor > 0 and block_max > block_min and block_len >= NOISE_MIN_LEN:
            noise_sigma, noise_rho = estimate_block_noise(
                block_samples,
                block_codes if has_sample_codes else no_codes,
                ticks[first_sample : first_sample + block_len]
                if irregular
                else no_ticks,
                span_binary_exponent(block_min, block_max),
                scratch_pivots,
                scratch_noise_diffs,
                scratch_noise_weights,
                math.ldexp(
                    1.0, snapped_exponent_for_range(block_min, block_max, max_bits)
                )
                / noise_factor,
            )
        if finite_count > 0 and block_len >= ESCAPE_MIN_LEN:
            escaped, bulk_min, bulk_max = escape_outliers(
                block_samples,
                state.rational_index[:block_len],
                block_codes,
                has_sample_codes,
                block_mean,
                block_min,
                block_max,
                max(
                    floor_exponent(noise_factor, noise_sigma, noise_rho), tolerance_exp
                ),
                max_bits,
                grids,
                block_quantized,
            )
            if escaped:
                state.held[:block_len] = state.rational_index[:block_len]
                block_samples = state.held[:block_len]
                block_min, block_max = bulk_min + 0.0, bulk_max + 0.0
                base_lower_bounds[block_idx] = block_min
                base_upper_bounds[block_idx] = block_max
                has_sample_codes = True
                if (
                    noise_factor > 0
                    and block_len >= NOISE_MIN_LEN
                    and block_max > block_min
                ):
                    noise_sigma, noise_rho = estimate_block_noise(
                        block_samples,
                        block_codes,
                        ticks[first_sample : first_sample + block_len]
                        if irregular
                        else no_ticks,
                        span_binary_exponent(block_min, block_max),
                        scratch_pivots,
                        scratch_noise_diffs,
                        scratch_noise_weights,
                        math.ldexp(
                            1.0,
                            snapped_exponent_for_range(block_min, block_max, max_bits),
                        )
                        / noise_factor,
                    )
        planned_exp, block_tol_step = plan_block_step(
            block_samples,
            block_min,
            block_max,
            noise_sigma,
            noise_rho,
            noise_factor,
            min_bits,
            max_bits,
            tolerance_exp,
            tol_step,
        )
        block_exponents[block_idx] = planned_exp
        (
            out_block_flags[block_idx],
            out_grid_exponents[block_idx],
            anchor_index,
            anchor,
            out_grid_scales[block_idx],
        ) = encode_block(
            block_samples,
            block_min,
            block_max,
            planned_exp,
            grids,
            orders_mask,
            block_quantized,
            state.rational[:block_len],
            state.rational_index[:block_len],
            block_residuals,
            block_tol_step,
        )
        _store_anchor(
            out_block_anchors,
            anchor_floats,
            block_idx,
            out_block_flags[block_idx],
            anchor,
            anchor_index,
        )
        if has_sample_codes:
            out_block_flags[block_idx] |= BLOCK_FLAG_SAMPLE_CODES
        if use_target:
            coarsest_exponents[block_idx] = max(
                snapped_exponent_for_range(block_min, block_max, min_bits),
                block_exponents[block_idx],
            )
            estimated_bits_per_block[block_idx] = estimate_residual_bits(
                block_residuals
            )
            budget_bits[block_idx] = _budget_bits_at_planned_step(
                out_block_flags[block_idx],
                out_grid_exponents[block_idx],
                out_grid_scales[block_idx],
                planned_exp,
                block_quantized,
                block_residuals,
                estimated_bits_per_block[block_idx],
                state.rational[:block_len],
                scratch_budget[:block_len],
            )
            block_weights[block_idx] = block_len
    if use_target:
        bit_reductions = np.zeros(num_blocks, np.int64)
        if allocate_bit_reductions(
            budget_bits,
            block_exponents,
            coarsest_exponents,
            block_weights,
            target_bits,
            bit_reductions,
        ):
            _reencode_with_bit_reductions(
                samples,
                sample_offsets,
                bit_reductions,
                block_exponents,
                base_lower_bounds,
                base_upper_bounds,
                estimated_bits_per_block,
                grids,
                orders_mask,
                scratch_budget,
                state,
            )
