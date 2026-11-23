# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Validation and decoding kernels for an uncompressed block group body.

Validates per-block metadata columns (flags, exponents, scales, anchors) and decodes
requested blocks by unshuffling residual planes, reversing zigzag encoding, integrating
modular differences, dequantizing onto the block's grid (power-of-two, decimal, or scaled),
and restoring NaN and escaped samples when flagged.
"""

import math

import numpy as np
from numba import njit

from . import _extreme_magnitudes as xm
from ._bitpacking import (
    residual_bit_planes_view,
    residual_byte_planes_view,
    sample_code_planes_view,
    unshuffle_residual_block,
    unzigzag_sample,
)
from ._escapes import restore_coded
from ._format import (
    BLOCK_FLAG_DECIMAL,
    BLOCK_FLAG_IRREGULAR_TIME,
    BLOCK_FLAG_ORDER,
    BLOCK_FLAG_RESERVED,
    BLOCK_FLAG_SAMPLE_CODES,
    BLOCK_FLAG_SCALED,
    BLOCK_FLAG_WIDE_TIME,
    E_MAX,
    E_MIN,
    INDEX_LIMIT,
    P_MAX,
    P_MIN,
    SAMPLES_PER_PLANE_BYTE,
    SCALE_MASK,
    SCALE_SHIFT,
    block_anchors_start,
    get_int16,
    get_int64,
    get_uint32,
    grid_exponents_start,
    grid_scales_start,
    scale_valid,
)

# Validation status codes returned by validate_body_columns (plain ints for Numba).
VALID_OK: int = 0
"""All per-block flags, exponents, scales, and anchors are valid."""

INVALID_FLAGS: int = 1
"""A block sets reserved or incompatible flag bits."""

INVALID_EXPONENT: int = 2
"""A block's quantization exponent is outside the valid range for its grid type."""

INVALID_ANCHOR: int = 3
"""A block's anchor is non-finite (power-of-two) or outside ``(-2^52, 2^52)`` (decimal/scaled)."""

INVALID_SCALE: int = 4
"""An unscaled block has a non-zero scale, or a scaled block's ``(exponent, N, D)`` is invalid."""


@njit(nogil=True, cache=True)
def unzigzag_bytes(low_bytes: np.ndarray, high_bytes: np.ndarray, out_residuals: np.ndarray) -> None:
    """Decodes signed residuals (stored in int32) from zigzagged low and high byte arrays.

    Args:
        low_bytes: 1D uint8 array of low zigzag bytes.
        high_bytes: 1D uint8 array of high zigzag bytes.
        out_residuals: Preallocated 1D int32 array receiving decoded signed 16-bit residuals.
    """
    for sample_idx in range(out_residuals.shape[0]):
        out_residuals[sample_idx] = unzigzag_sample(low_bytes, high_bytes, sample_idx)


@njit(nogil=True, cache=True)
def integrate_residuals(residuals_in_out: np.ndarray, order: int) -> None:
    """Reconstructs signed int16 grid indices by integrating residuals mod 2^16 in place.

    Applies ``order`` successive modular prefix sums and sign-extends the final result into
    ``[-32768, 32767]``.

    Args:
        residuals_in_out: 1D int32 array of residuals, overwritten in place with grid indices ``q``.
        order: Predictor difference order in ``0..3``.
    """
    # Order 0 residuals are already signed grid indices; orders 1-3 accumulate 1..3 running
    # sums mod 65536 and map [0..65535] back to signed [-32768..32767].
    num_samples = residuals_in_out.shape[0]
    if order == 0:
        return
    elif order == 1:
        sum1 = np.int32(0)
        for sample_idx in range(num_samples):
            sum1 = (sum1 + residuals_in_out[sample_idx]) & 0xFFFF
            residuals_in_out[sample_idx] = ((sum1 + 32768) & 0xFFFF) - 32768
    elif order == 2:
        sum1 = np.int32(0)
        sum2 = np.int32(0)
        for sample_idx in range(num_samples):
            sum1 = (sum1 + residuals_in_out[sample_idx]) & 0xFFFF
            sum2 = (sum2 + sum1) & 0xFFFF
            residuals_in_out[sample_idx] = ((sum2 + 32768) & 0xFFFF) - 32768
    else:
        sum1 = np.int32(0)
        sum2 = np.int32(0)
        sum3 = np.int32(0)
        for sample_idx in range(num_samples):
            sum1 = (sum1 + residuals_in_out[sample_idx]) & 0xFFFF
            sum2 = (sum2 + sum1) & 0xFFFF
            sum3 = (sum3 + sum2) & 0xFFFF
            residuals_in_out[sample_idx] = ((sum3 + 32768) & 0xFFFF) - 32768


@njit(nogil=True, cache=True)
def dequantize_pow2(
    quantized_samples: np.ndarray, anchor: float, quant_exp: int, out_samples: np.ndarray
) -> None:
    """Reconstructs float64 samples on a power-of-two grid: ``anchor + q * 2^quant_exp``.

    Delegates to ``dequantize_pow2_wide_exponent`` when ``quant_exp >= E_WIDE`` to avoid
    intermediate overflow near ``DBL_MAX``.

    Args:
        quantized_samples: 1D int32 array of signed grid indices ``q``.
        anchor: Block anchor value (the reconstructed value for ``q == 0``).
        quant_exp: Power-of-two quantization exponent in ``[E_MIN, E_MAX]``.
        out_samples: Preallocated 1D float64 array receiving reconstructed samples.
    """
    if quant_exp >= xm.E_WIDE:
        xm.dequantize_pow2_wide_exponent(quantized_samples, anchor, quant_exp, out_samples)
        return
    step_size = math.ldexp(1.0, quant_exp)
    for sample_idx in range(quantized_samples.shape[0]):
        out_samples[sample_idx] = anchor + quantized_samples[sample_idx] * step_size


@njit(nogil=True, cache=True)
def dequantize_decimal(
    quantized_samples: np.ndarray, anchor_index: int, decimal_exp: int, out_samples: np.ndarray
) -> None:
    """Reconstructs float64 samples on an exact decimal grid: ``(anchor_index + q) * 10^decimal_exp``.

    Uses a single IEEE-754 division by ``10^-decimal_exp`` (for negative exponents) or
    multiplication by ``10^decimal_exp`` (for non-negative exponents) so every result is the
    nearest float64 to the exact decimal value.

    Args:
        quantized_samples: 1D int32 array of signed grid indices ``q``.
        anchor_index: Signed int64 anchor index ``K0`` on the ``10^decimal_exp`` grid.
        decimal_exp: Decimal exponent in ``[P_MIN, P_MAX]``.
        out_samples: Preallocated 1D float64 array receiving reconstructed samples.
    """
    if decimal_exp < 0:
        divisor = 10.0 ** -decimal_exp
        for sample_idx in range(quantized_samples.shape[0]):
            out_samples[sample_idx] = float(anchor_index + quantized_samples[sample_idx]) / divisor
    else:
        multiplier = 10.0**decimal_exp
        for sample_idx in range(quantized_samples.shape[0]):
            out_samples[sample_idx] = float(anchor_index + quantized_samples[sample_idx]) * multiplier


@njit(nogil=True, cache=True)
def dequantize_scaled(
    quantized_samples: np.ndarray,
    anchor_index: int | np.integer,
    scale_num: int | np.integer,
    scale_den: int | np.integer,
    exponent: int,
    decimal: bool,
    out_samples: np.ndarray,
) -> None:
    """Reconstructs float64 samples on a scaled rational grid: ``(anchor_index + scale_num * q) * B^exponent / scale_den``.

    Evaluates each sample with a single rounding: ``anchor_index + scale_num * q`` is an exact
    integer below ``2^53``; for decimal grids (``exponent <= 0``) it divides by exact double
    ``scale_den * 10^-exponent``; for binary grids it scales by exact ``2^exponent`` and divides
    by ``scale_den``.

    Args:
        quantized_samples: 1D int32 array of signed grid indices ``q``.
        anchor_index: Signed int64 anchor index ``K0`` on the fine grid ``B^exponent / scale_den``.
        scale_num: Numerator ``N`` (step in fine grid points).
        scale_den: Denominator ``D``.
        exponent: Decimal power ``p`` or power-of-two exponent ``e``.
        decimal: True for base-10 grid, False for base-2 grid.
        out_samples: Preallocated 1D float64 array receiving reconstructed samples.
    """
    if decimal:
        divisor = scale_den * 10.0 ** -exponent
        for sample_idx in range(quantized_samples.shape[0]):
            out_samples[sample_idx] = (
                float(anchor_index + scale_num * np.int64(quantized_samples[sample_idx])) / divisor
            )
    else:
        scale = math.ldexp(1.0, exponent)
        for sample_idx in range(quantized_samples.shape[0]):
            out_samples[sample_idx] = (
                float(anchor_index + scale_num * np.int64(quantized_samples[sample_idx])) * scale
            ) / scale_den


@njit(nogil=True, cache=True)
def validate_body_columns(body: np.ndarray, sample_offsets: np.ndarray, has_time: bool) -> tuple[int, int]:
    """Validates per-block flags, exponents, scales, and anchors across all blocks in a body.

    Args:
        body: 1D uint8 array of uncompressed body bytes.
        sample_offsets: 1D int64 array of cumulative sample offsets (length ``num_blocks + 1``).
        has_time: Whether the block group has a time axis.

    Returns:
        Tuple ``(status, failing_block_idx)`` where ``status`` is ``VALID_OK`` (with index 0)
        or one of ``INVALID_FLAGS``, ``INVALID_EXPONENT``, ``INVALID_ANCHOR``, ``INVALID_SCALE``
        with the first failing block index.
    """
    # Inspect each block's metadata columns in order: enforce all-zero columns on empty blocks,
    # reject reserved or inconsistent time flags, then verify scale, exponent, and anchor bounds
    # according to whether the block uses a scaled, decimal, or power-of-two grid.
    num_blocks = sample_offsets.shape[0] - 1
    anchor_bits = np.empty(1, np.int64)
    anchor_float = anchor_bits.view(np.float64)
    for block_idx in range(num_blocks):
        flags = body[block_idx]
        exponent = int(get_int16(body, grid_exponents_start(num_blocks), num_blocks, block_idx))
        anchor_bits[0] = get_int64(body, block_anchors_start(num_blocks), num_blocks, block_idx)
        scale = get_uint32(body, grid_scales_start(num_blocks), num_blocks, block_idx)
        if sample_offsets[block_idx + 1] == sample_offsets[block_idx]:
            # Empty blocks must store zero in all metadata columns.
            if flags:
                return INVALID_FLAGS, block_idx
            if exponent:
                return INVALID_EXPONENT, block_idx
            if scale:
                return INVALID_SCALE, block_idx
            if anchor_bits[0]:
                return INVALID_ANCHOR, block_idx
            continue
        # Reserved bit 7 must be 0; irregular time requires a time axis; wide time requires irregular time.
        irregular_time = flags & BLOCK_FLAG_IRREGULAR_TIME
        if flags & BLOCK_FLAG_RESERVED or (irregular_time and not has_time):
            return INVALID_FLAGS, block_idx
        if flags & BLOCK_FLAG_WIDE_TIME and not irregular_time:
            return INVALID_FLAGS, block_idx
        if flags & BLOCK_FLAG_SCALED:
            if not scale_valid(flags & BLOCK_FLAG_DECIMAL != 0, exponent, scale & SCALE_MASK, scale >> SCALE_SHIFT):
                return INVALID_SCALE, block_idx
            if not -INDEX_LIMIT < anchor_bits[0] < INDEX_LIMIT:
                return INVALID_ANCHOR, block_idx
            continue
        if scale:
            return INVALID_SCALE, block_idx
        if flags & BLOCK_FLAG_DECIMAL:
            if exponent < P_MIN or exponent > P_MAX:
                return INVALID_EXPONENT, block_idx
            if not -INDEX_LIMIT < anchor_bits[0] < INDEX_LIMIT:
                return INVALID_ANCHOR, block_idx
        elif exponent < E_MIN or exponent > E_MAX:
            return INVALID_EXPONENT, block_idx
        elif not math.isfinite(anchor_float[0]):
            return INVALID_ANCHOR, block_idx
    return VALID_OK, 0


@njit(nogil=True, cache=True)
def decode_blocks(
    body: np.ndarray,
    sample_offsets: np.ndarray,
    plane_byte_offsets: np.ndarray,
    code_plane_offsets: np.ndarray,
    block_ids: np.ndarray,
    out_samples: np.ndarray,
    byte_planes: bool,
    has_time: bool,
    escaped_values: np.ndarray,
    escape_offsets: np.ndarray,
) -> None:
    """Decodes the requested blocks of an uncompressed body into ``out_samples``.

    Args:
        body: 1D uint8 array of uncompressed body bytes (must pass ``validate_body_columns``).
        sample_offsets: ``BodyLayout.sample_offsets``.
        plane_byte_offsets: ``BodyLayout.plane_byte_offsets``.
        code_plane_offsets: ``BodyLayout.code_plane_offsets``.
        block_ids: 1D int64 array of block indices to decode.
        out_samples: Preallocated 1D float64 array sized for all samples in the group; only
            slices belonging to ``block_ids`` are written.
        byte_planes: True if residual planes are stored as 2 byte planes; False for 16 bit planes.
        has_time: Whether the body includes time metadata columns before the residuals.
        escaped_values: 1D float64 array of escaped values from the body trailer.
        escape_offsets: 1D int64 cumulative offset array into ``escaped_values`` per block.
    """
    # Preallocate reusable per-block scratch buffers sized to the largest requested block, then
    # decode each block: unshuffle (if bit-planed) -> unzigzag -> integrate -> dequantize ->
    # restore NaN and escaped samples.
    num_blocks = sample_offsets.shape[0] - 1
    num_plane_bytes = int(plane_byte_offsets[num_blocks])
    max_plane_bytes = 0
    for block_idx in block_ids:
        max_plane_bytes = max(max_plane_bytes, plane_byte_offsets[block_idx + 1] - plane_byte_offsets[block_idx])
    scratch_low_bytes = np.empty(SAMPLES_PER_PLANE_BYTE * max_plane_bytes, np.uint8)
    scratch_high_bytes = np.empty(SAMPLES_PER_PLANE_BYTE * max_plane_bytes, np.uint8)
    scratch_residuals = np.empty(SAMPLES_PER_PLANE_BYTE * max_plane_bytes, np.int32)
    # 1-element int64/float64 view pair to reinterpret raw anchor bits without copying.
    anchor_bits = np.empty(1, np.int64)
    anchor_float = anchor_bits.view(np.float64)
    bit_planes = residual_bit_planes_view(body, num_blocks, num_plane_bytes, has_time)
    byte_planes_2d = residual_byte_planes_view(body, num_blocks, num_plane_bytes, has_time)
    code_planes = sample_code_planes_view(
        body, num_blocks, num_plane_bytes, int(code_plane_offsets[num_blocks]), has_time
    )
    for block_idx in block_ids:
        first_sample = sample_offsets[block_idx]
        block_len = sample_offsets[block_idx + 1] - first_sample
        if block_len == 0:
            continue
        flags = body[block_idx]
        exponent = int(get_int16(body, grid_exponents_start(num_blocks), num_blocks, block_idx))
        anchor_bits[0] = get_int64(body, block_anchors_start(num_blocks), num_blocks, block_idx)
        block_residuals = scratch_residuals[:block_len]
        block_out = out_samples[first_sample : first_sample + block_len]
        if byte_planes:
            # Byte-plane mode: low and high zigzag bytes are already contiguous per block.
            sample_start = SAMPLES_PER_PLANE_BYTE * plane_byte_offsets[block_idx]
            unzigzag_bytes(
                byte_planes_2d[0, sample_start : sample_start + block_len],
                byte_planes_2d[1, sample_start : sample_start + block_len],
                block_residuals,
            )
        else:
            # Bit-plane mode: transpose 16 bit planes back into low/high bytes before unzigzagging.
            unshuffle_residual_block(
                bit_planes,
                plane_byte_offsets[block_idx],
                plane_byte_offsets[block_idx + 1] - plane_byte_offsets[block_idx],
                scratch_low_bytes,
                scratch_high_bytes,
            )
            unzigzag_bytes(scratch_low_bytes, scratch_high_bytes, block_residuals)
        integrate_residuals(block_residuals, flags & BLOCK_FLAG_ORDER)
        if flags & BLOCK_FLAG_SCALED:
            scale = get_uint32(body, grid_scales_start(num_blocks), num_blocks, block_idx)
            dequantize_scaled(
                block_residuals,
                anchor_bits[0],
                scale & SCALE_MASK,
                scale >> SCALE_SHIFT,
                exponent,
                flags & BLOCK_FLAG_DECIMAL != 0,
                block_out,
            )
        elif flags & BLOCK_FLAG_DECIMAL:
            dequantize_decimal(block_residuals, anchor_bits[0], exponent, block_out)
        else:
            dequantize_pow2(block_residuals, anchor_float[0], exponent, block_out)
        if flags & BLOCK_FLAG_SAMPLE_CODES:
            restore_coded(
                code_planes,
                code_plane_offsets[block_idx],
                escaped_values[escape_offsets[block_idx] : escape_offsets[block_idx + 1]],
                block_out,
            )
