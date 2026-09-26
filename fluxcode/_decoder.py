# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Decoding kernels and unit validation for compressed series.

Fuses bit-plane unshuffling, zigzag reversal, modular prefix-sum integration,
and grid dequantization (power-of-two or decimal) into a single pass per block.
Restores non-finite values (canonical quiet NaN and signed infinities) when flagged.
"""

import enum
import math

import numpy as np
from numba import njit

from . import _extreme_magnitudes as xm
from ._format import (
    E_MAX,
    E_MIN,
    HEAD_DECIMAL,
    HEAD_NONFINITE,
    HEAD_ORDER,
    HEAD_RESERVED,
    P_MAX,
    P_MIN,
    anchor_start,
    byte_planes_view,
    code_planes_view,
    count_flagged,
    get_int64,
    param_start,
    planes_view,
    unshuffle_block,
    unzigzag16,
)
from ._nonfinite import restore_nonfinite


class ValidationStatus(enum.IntEnum):
    """Header and parameter validation status for uncompressed units."""

    OK = 0
    BAD_HEAD = 1
    BAD_PARAM = 2
    BAD_ANCHOR = 3


OK: int = int(ValidationStatus.OK)
"""Validation status indicating unit headers and parameters are valid."""

BAD_HEAD: int = int(ValidationStatus.BAD_HEAD)
"""Validation status indicating reserved or unsupported bits in a block header."""

BAD_PARAM: int = int(ValidationStatus.BAD_PARAM)
"""Validation status indicating an out-of-range exponent or parameter."""

BAD_ANCHOR: int = int(ValidationStatus.BAD_ANCHOR)
"""Validation status indicating a non-finite float anchor or an out-of-range decimal grid index."""

MAX_DECIMAL_ANCHOR: int = 1 << 52
"""Largest decimal grid index magnitude: K0 + q stays an exact double below 2^53."""



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

    Computes `order` iterative prefix sums modulo 65536 on v. For order 0,
    masks elements to uint16 range.

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
            # First-order prefix sum: q[i] = (q[i-1] + v[i]) mod 2^16
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
    """Reconstructs float64 samples on a power-of-two grid: lo + q * 2^e.

    For exponents e >= E_WIDE (971), delegates to an overflow-safe routine that
    evaluates at half-scale and clamps to DBL_MAX.

    Args:
        quantized_samples: 1D int32 array of quantized grid offsets.
        lower_bound: Minimum value of the block (exact reconstruction point for q = 0).
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
        # Reconstruct sample: lo + q * step
        out_samples[sample_idx] = lower_bound + quantized_samples[sample_idx] * step_size


@njit(nogil=True, cache=True)
def dequantize_decimal(
    quantized_samples: np.ndarray, base_grid_offset: int, decimal_exp: int, out_samples: np.ndarray
) -> None:
    """Reconstructs float64 samples on an exact decimal grid: (K0 + q) * 10^p.

    Ensures bit-identical results to decimal string parsing by dividing by
    10^-p when p < 0 or multiplying by 10^p when p >= 0.

    Args:
        quantized_samples: 1D int32 array of quantized grid offsets.
        base_grid_offset: Grid index K0 of the block minimum (the block's anchor).
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
def check_unit(raw_unit: np.ndarray, num_blocks: int) -> tuple[int, int]:
    """Validates header bytes, parameters and anchors across all blocks in a unit.

    Inspects each block to verify that reserved header bits (4-7) are zero, that
    exponents/parameters fall within permissible format limits, and that anchors
    are finite floats (power-of-two blocks) or grid indices below 2^52 (decimal).

    Args:
        raw_unit: 1D uint8 array containing uncompressed unit bytes.
        num_blocks: Number of blocks in the unit.

    Returns:
        A tuple of (status, b):
            status: Validation outcome (OK, BAD_HEAD, BAD_PARAM, or BAD_ANCHOR).
            b: Zero-based index of the first failing block (0 if OK).
    """
    anchor_bits = np.empty(1, np.int64)
    anchor_float = anchor_bits.view(np.float64)
    for block_idx in range(num_blocks):
        header_byte = raw_unit[block_idx]
        param_val = int(get_int64(raw_unit, param_start(num_blocks), num_blocks, block_idx))
        anchor_bits[0] = get_int64(raw_unit, anchor_start(num_blocks), num_blocks, block_idx)
        # Check that reserved bits 4-7 are all zero
        if header_byte & HEAD_RESERVED:
            return BAD_HEAD, block_idx
        if header_byte & HEAD_DECIMAL:
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
def decode_unit(raw_unit: np.ndarray, out_blocks: np.ndarray, byte_planes: bool) -> None:
    """Decodes all blocks of an uncompressed body into output array.

    Fuses unshuffling, unzigzagging, integration, dequantization, and
    non-finite sample restoration.

    Args:
        raw_unit: 1D uint8 array of uncompressed body bytes (must pass check_unit).
        out_blocks: Output 2D float64 array of shape (N, n) receiving decoded samples.
        byte_planes: Whether the residuals are stored as byte planes (unit flags bit 0).
    """
    num_blocks, block_len = out_blocks.shape
    # Anchors: float64 bits of the minimum (power-of-two blocks) or the decimal grid index
    anchor_bits = np.empty(num_blocks, np.int64)
    for block_idx in range(num_blocks):
        anchor_bits[block_idx] = get_int64(raw_unit, anchor_start(num_blocks), num_blocks, block_idx)
    anchor_floats = anchor_bits.view(np.float64)
    scratch_low_bytes = np.empty(block_len, np.uint8)
    scratch_high_bytes = np.empty(block_len, np.uint8)
    block_residuals = np.empty(block_len, np.int32)
    # Obtain views into residual and code bit planes
    bit_planes = planes_view(raw_unit, num_blocks, block_len)
    byte_planes_2d = byte_planes_view(raw_unit, num_blocks, block_len)
    code_planes = code_planes_view(raw_unit, num_blocks, block_len, count_flagged(raw_unit, num_blocks))
    flagged_block_counter = 0
    for block_idx in range(num_blocks):
        header_byte = raw_unit[block_idx]
        param_val = int(get_int64(raw_unit, param_start(num_blocks), num_blocks, block_idx))
        if byte_planes:
            # Byte planes: the block's low and high zigzag bytes are contiguous
            sample_start = block_idx * block_len
            unzigzag(
                byte_planes_2d[0, sample_start:sample_start + block_len],
                byte_planes_2d[1, sample_start:sample_start + block_len],
                block_residuals,
            )
        else:
            # Gather bit planes into low and high zigzag bytes
            unshuffle_block(bit_planes, block_idx, scratch_low_bytes, scratch_high_bytes)
            # Unzigzag into signed differences
            unzigzag(scratch_low_bytes, scratch_high_bytes, block_residuals)
        # Integrate mod 2^16 by predictor order (bits 0-1)
        integrate(block_residuals, header_byte & HEAD_ORDER)
        # Dequantize according to grid type (decimal or power-of-two)
        if header_byte & HEAD_DECIMAL:
            dequantize_decimal(block_residuals, anchor_bits[block_idx], param_val, out_blocks[block_idx])
        else:
            dequantize_pow2(block_residuals, anchor_floats[block_idx], param_val, out_blocks[block_idx])
        # Restore non-finite samples (NaN, +/-inf) from code planes
        if header_byte & HEAD_NONFINITE:
            restore_nonfinite(code_planes, flagged_block_counter, out_blocks[block_idx])
            flagged_block_counter += 1
