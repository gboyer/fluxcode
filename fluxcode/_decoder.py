# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Decoding kernels and validation of an uncompressed block group body.

Decodes a block in a single pass: unshuffling the planes, reversing the zigzag,
integrating the residuals (modular prefix sums) and dequantizing on the power-of-two or
decimal grid. Restores non-finite values (canonical quiet NaN and signed infinities)
when flagged.
"""

import math

import numpy as np
from numba import njit

from . import _extreme_magnitudes as xm
from ._bitpacking import (
    byte_planes_view,
    code_planes_view,
    planes_view,
    unshuffle_block,
    unzigzag16,
)
from ._format import (
    BLOCK_FLAG_DECIMAL,
    BLOCK_FLAG_IRREGULAR_TIME,
    BLOCK_FLAG_LONG_TIME,
    BLOCK_FLAG_NONFINITE,
    BLOCK_FLAG_ORDER,
    BLOCK_FLAG_RESERVED,
    E_MAX,
    E_MIN,
    P_MAX,
    P_MIN,
    get_int16,
    get_int64,
    grid_params_start,
    value_anchor_start,
)
from ._nonfinite import restore_nonfinite

# Validation statuses of uncompressed block groups (plain ints: numba kernels use them)
OK: int = 0
"""Validation status indicating block group headers and parameters are valid."""

BAD_FLAGS: int = 1
"""Validation status indicating reserved or unsupported bits in a block header."""

BAD_PARAM: int = 2
"""Validation status indicating an out-of-range exponent or parameter."""

BAD_ANCHOR: int = 3
"""Validation status indicating a non-finite float anchor or an out-of-range decimal grid index."""

MAX_DECIMAL_ANCHOR: int = 1 << 52
"""Bound on the magnitude of a decimal grid index: the index plus q stays an exact double below 2^53."""


@njit(nogil=True, cache=True)
def unzigzag(low_bytes: np.ndarray, high_bytes: np.ndarray, out_residuals: np.ndarray) -> None:
    """Decodes signed residuals (int32 storage) from low and high zigzag byte arrays.

    Args:
        low_bytes: 1D uint8 array of lower zigzag bytes.
        high_bytes: 1D uint8 array of upper zigzag bytes.
        out_residuals: Output 1D int32 array receiving decoded signed residuals.
    """
    for sample_idx in range(out_residuals.shape[0]):
        # Unzigzag little-endian byte pair into signed 16-bit integer
        out_residuals[sample_idx] = unzigzag16(low_bytes, high_bytes, sample_idx)


@njit(nogil=True, cache=True)
def integrate(residuals_in_out: np.ndarray, order: int) -> None:
    """Reconstructs quantized indices by integrating residuals mod 2^16 in-place.

    Computes `order` successive prefix sums modulo 65536. For order 0, masks the
    elements to the uint16 range.

    Args:
        residuals_in_out: In-out 1D int32 array containing residuals, modified in-place to hold
            quantized grid offsets q in [0, 65535].
        order: Difference order (0, 1, 2, or 3).
    """
    num_samples = residuals_in_out.shape[0]
    if order == 0:
        for sample_idx in range(num_samples):
            # Mask to 16-bit unsigned range
            residuals_in_out[sample_idx] &= 0xFFFF
    elif order == 1:
        sum_order1 = np.int32(0)
        for sample_idx in range(num_samples):
            # First-order prefix sum, mod 2^16
            sum_order1 = (sum_order1 + residuals_in_out[sample_idx]) & 0xFFFF
            residuals_in_out[sample_idx] = sum_order1
    elif order == 2:
        sum_order1 = np.int32(0)
        sum_order2 = np.int32(0)
        for sample_idx in range(num_samples):
            # Second-order double prefix sum mod 2^16
            sum_order1 = (sum_order1 + residuals_in_out[sample_idx]) & 0xFFFF
            sum_order2 = (sum_order2 + sum_order1) & 0xFFFF
            residuals_in_out[sample_idx] = sum_order2
    else:
        sum_order1 = np.int32(0)
        sum_order2 = np.int32(0)
        sum_order3 = np.int32(0)
        for sample_idx in range(num_samples):
            # Third-order triple prefix sum mod 2^16
            sum_order1 = (sum_order1 + residuals_in_out[sample_idx]) & 0xFFFF
            sum_order2 = (sum_order2 + sum_order1) & 0xFFFF
            sum_order3 = (sum_order3 + sum_order2) & 0xFFFF
            residuals_in_out[sample_idx] = sum_order3


@njit(nogil=True, cache=True)
def dequantize_pow2(
    quantized_samples: np.ndarray, lower_bound: float, quant_exp: int, out_samples: np.ndarray
) -> None:
    """Reconstructs float64 samples on a power-of-two grid: lower_bound + q * 2^quant_exp.

    For exponents at or above E_WIDE (971), delegates to an overflow-safe routine that
    evaluates at half-scale and clamps to DBL_MAX.

    Args:
        quantized_samples: 1D int32 array of quantized grid offsets.
        lower_bound: Anchor of the block: the grid point nearest its minimum (the minimum itself
            for a constant block or a non-finite snapped anchor); the value for q = 0.
        quant_exp: Power-of-two quantization exponent.
        out_samples: Output 1D float64 array receiving reconstructed samples.
    """
    # Evaluate at half-scale for extreme ranges near DBL_MAX
    if quant_exp >= xm.E_WIDE:
        xm.dequantize_pow2_wide(quantized_samples, lower_bound, quant_exp, out_samples)
        return
    # Compute power-of-two step size: 2^e
    step_size = math.ldexp(1.0, quant_exp)
    for sample_idx in range(quantized_samples.shape[0]):
        # Reconstruct the sample: lower_bound + q * step
        out_samples[sample_idx] = lower_bound + quantized_samples[sample_idx] * step_size


@njit(nogil=True, cache=True)
def dequantize_decimal(
    quantized_samples: np.ndarray, base_grid_offset: int, decimal_exp: int, out_samples: np.ndarray
) -> None:
    """Reconstructs float64 samples on an exact decimal grid: (base_grid_offset + q) * 10^decimal_exp.

    The result is the double nearest the decimal value, as decimal string parsing gives:
    the integer is divided by 10^-decimal_exp when the exponent is negative and multiplied
    by 10^decimal_exp otherwise (both powers of ten are exact doubles).

    Args:
        quantized_samples: 1D int32 array of quantized grid offsets.
        base_grid_offset: Grid index of the block minimum (the block's anchor).
        decimal_exp: Decimal exponent in [-22, 22].
        out_samples: Output 1D float64 array receiving reconstructed samples.
    """
    if decimal_exp < 0:
        # Exact integer power of 10 for negative exponents: divide directly
        divisor = 10.0 ** -decimal_exp
        for sample_idx in range(quantized_samples.shape[0]):
            out_samples[sample_idx] = float(base_grid_offset + quantized_samples[sample_idx]) / divisor
    else:
        # Positive decimal exponent: multiply by exact power of 10
        multiplier = 10.0 ** decimal_exp
        for sample_idx in range(quantized_samples.shape[0]):
            out_samples[sample_idx] = float(base_grid_offset + quantized_samples[sample_idx]) * multiplier


@njit(nogil=True, cache=True)
def check_group(raw_group: np.ndarray, sample_offsets: np.ndarray, has_time: bool) -> tuple[int, int]:
    """Validates block flags, grid parameters and value anchors across all blocks in a block group.

    Checks, block by block, that the reserved flag bits (6-7) are zero, that the irregular
    time bit (4) is set only in block groups with a time axis and the long time bit (5) only with
    bit 4, that grid parameters fall within the format's limits, and that value anchors
    are finite floats (power-of-two blocks) or grid indices below 2^52 (decimal). An empty
    block's flags, grid parameter and anchor must be 0.

    Args:
        raw_group: 1D uint8 array containing uncompressed block group bytes.
        sample_offsets: 1D int64 array of the blocks' sample offsets (num_blocks + 1).
        has_time: Whether the block group has a time axis.

    Returns:
        A tuple of (status, failing_block_idx):
            status: Validation outcome (OK, BAD_FLAGS, BAD_PARAM, or BAD_ANCHOR).
            failing_block_idx: Zero-based index of the first failing block (0 if OK).
    """
    num_blocks = sample_offsets.shape[0] - 1
    anchor_bits = np.empty(1, np.int64)
    anchor_float = anchor_bits.view(np.float64)
    for block_idx in range(num_blocks):
        flags = raw_group[block_idx]
        param_val = int(get_int16(raw_group, grid_params_start(num_blocks), num_blocks, block_idx))
        anchor_bits[0] = get_int64(raw_group, value_anchor_start(num_blocks), num_blocks, block_idx)
        if sample_offsets[block_idx + 1] == sample_offsets[block_idx]:
            # An empty block stores nothing: its columns are 0
            if flags:
                return BAD_FLAGS, block_idx
            if param_val:
                return BAD_PARAM, block_idx
            if anchor_bits[0]:
                return BAD_ANCHOR, block_idx
            continue
        # Reserved bits 6-7 must be zero, bit 4 needs a time axis and bit 5 needs bit 4
        irregular_time = flags & BLOCK_FLAG_IRREGULAR_TIME
        if flags & BLOCK_FLAG_RESERVED or (irregular_time and not has_time):
            return BAD_FLAGS, block_idx
        if flags & BLOCK_FLAG_LONG_TIME and not irregular_time:
            return BAD_FLAGS, block_idx
        if flags & BLOCK_FLAG_DECIMAL:
            # Check decimal exponent range [-22, 22]
            if param_val < P_MIN or param_val > P_MAX:
                return BAD_PARAM, block_idx
            if not -MAX_DECIMAL_ANCHOR < anchor_bits[0] < MAX_DECIMAL_ANCHOR:
                return BAD_ANCHOR, block_idx
        # Check power-of-two exponent range [-1074, 1023]
        elif param_val < E_MIN or param_val > E_MAX:
            return BAD_PARAM, block_idx
        elif not math.isfinite(anchor_float[0]):
            return BAD_ANCHOR, block_idx
    return OK, 0


@njit(nogil=True, cache=True)
def decode_group(
    raw_group: np.ndarray,
    sample_offsets: np.ndarray,
    octet_offsets: np.ndarray,
    code_offsets: np.ndarray,
    block_ids: np.ndarray,
    out_samples: np.ndarray,
    byte_planes: bool,
    has_time: bool,
) -> None:
    """Decodes the given blocks of an uncompressed body into the output array.

    Each block is unshuffled, unzigzagged, integrated, dequantized and has its non-finite
    samples restored.

    Args:
        raw_group: 1D uint8 array of uncompressed body bytes (must pass check_group).
        sample_offsets: 1D int64 array of sample offsets for each block.
        octet_offsets: 1D int64 array of 8-sample octet offsets for each block.
        code_offsets: 1D int64 array of non-finite code plane byte offsets for each block.
        block_ids: 1D int64 array of the blocks to decode.
        out_samples: Output 1D float64 array of every sample of the block group: each decoded
            block's samples are written at its sample offsets (others are left unmodified).
        byte_planes: Whether the residuals are stored as byte planes (group flags bit 0).
        has_time: Whether the block group has a time axis (time columns before the residuals).
    """
    num_blocks = sample_offsets.shape[0] - 1
    num_octets = int(octet_offsets[num_blocks])
    max_octets = 0
    for block_idx in block_ids:
        max_octets = max(max_octets, octet_offsets[block_idx + 1] - octet_offsets[block_idx])
    scratch_low_bytes = np.empty(8 * max_octets, np.uint8)
    scratch_high_bytes = np.empty(8 * max_octets, np.uint8)
    scratch_residuals = np.empty(8 * max_octets, np.int32)
    anchor_bits = np.empty(1, np.int64)
    anchor_float = anchor_bits.view(np.float64)
    # Obtain views into residual and code bit planes
    bit_planes = planes_view(raw_group, num_blocks, num_octets, has_time)
    byte_planes_2d = byte_planes_view(raw_group, num_blocks, num_octets, has_time)
    code_planes = code_planes_view(raw_group, num_blocks, num_octets, int(code_offsets[num_blocks]), has_time)
    for block_idx in block_ids:
        first_sample = sample_offsets[block_idx]
        block_len = sample_offsets[block_idx + 1] - first_sample
        if block_len == 0:
            continue
        flags = raw_group[block_idx]
        param_val = int(get_int16(raw_group, grid_params_start(num_blocks), num_blocks, block_idx))
        # Anchor: float64 bits of the grid point nearest the minimum (power-of-two blocks) or the decimal grid index
        anchor_bits[0] = get_int64(raw_group, value_anchor_start(num_blocks), num_blocks, block_idx)
        block_residuals = scratch_residuals[:block_len]
        block_out = out_samples[first_sample:first_sample + block_len]
        if byte_planes:
            # Byte planes: the block's low and high zigzag bytes are contiguous
            sample_start = 8 * octet_offsets[block_idx]
            unzigzag(
                byte_planes_2d[0, sample_start:sample_start + block_len],
                byte_planes_2d[1, sample_start:sample_start + block_len],
                block_residuals,
            )
        else:
            # Gather bit planes into low and high zigzag bytes
            unshuffle_block(
                bit_planes, octet_offsets[block_idx], octet_offsets[block_idx + 1] - octet_offsets[block_idx],
                scratch_low_bytes, scratch_high_bytes,
            )
            # Unzigzag into signed differences
            unzigzag(scratch_low_bytes, scratch_high_bytes, block_residuals)
        # Integrate mod 2^16 by predictor order (bits 0-1)
        integrate(block_residuals, flags & BLOCK_FLAG_ORDER)
        # Dequantize according to grid type (decimal or power-of-two)
        if flags & BLOCK_FLAG_DECIMAL:
            dequantize_decimal(block_residuals, anchor_bits[0], param_val, block_out)
        else:
            dequantize_pow2(block_residuals, anchor_float[0], param_val, block_out)
        # Restore non-finite samples (NaN, +/-inf) from code planes
        if flags & BLOCK_FLAG_NONFINITE:
            restore_nonfinite(code_planes, code_offsets[block_idx], block_out)
