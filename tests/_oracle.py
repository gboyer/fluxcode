# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Readers and writers only the tests use, as oracles for the package's own paths: a bit-plane body
writer built independently with numpy, a row reader that works for both layouts, and the bit-plane
share of wide residuals."""

import numpy as np
from numba import njit

from fluxcode import _bitpacking, _format, _group
from fluxcode._bitpacking import (
    byte_planes_view,
    code_planes_view,
    get_code,
    planes_view,
    unshuffle_block,
    unzigzag16,
)
from fluxcode._format import (
    BLOCK_FLAG_NONFINITE,
    BYTES_PER_FLAGS,
    BYTES_PER_SIZE,
    GroupRows,
    TimeRows,
    allocate_time_rows,
    get_int16,
    get_int64,
    grid_params_start,
    group_size,
    layout,
    read_layout,
    value_anchor_start,
)


def write_group(block_flags, block_sizes, grid_params, value_anchors, residuals, codes=None, byte_planes=False,
                time_rows: TimeRows | None = None, offsets=None):
    """Serializes rows into a body. The residuals are 16 bit planes unless byte_planes, packed here
    with numpy (everything else is the package's writer's)."""
    block_sizes = np.asarray(block_sizes, np.int64)
    if offsets is None:
        offsets = layout(block_flags, block_sizes)
    if codes is None:
        if offsets.code_offsets[-1]:
            raise ValueError("flagged blocks need codes")
        codes = np.zeros(0, np.uint8)
    body = _bitpacking.write_group(block_flags, block_sizes, grid_params, value_anchors, residuals, codes, time_rows,
                                   offsets)
    if byte_planes:
        return body
    num_blocks = block_flags.shape[0]
    num_octets = int(offsets.octet_offsets[-1])
    start = _format.residual_start(num_blocks, time_rows is not None)
    stop = start + 16 * num_octets
    zigzag = (residuals.astype(np.int32) << 1) ^ (residuals.astype(np.int32) >> 15)
    zigzag = (zigzag & 0xFFFF).astype(np.uint16)
    planes = body[start:stop].reshape(16, num_octets)
    for block_idx in range(num_blocks):
        first, last = offsets.sample_offsets[block_idx], offsets.sample_offsets[block_idx + 1]
        octets = offsets.octet_offsets[block_idx + 1] - offsets.octet_offsets[block_idx]
        padded = np.zeros(8 * octets, np.uint16)
        padded[:last - first] = zigzag[first:last]
        for plane_idx in range(16):
            bits = ((padded >> plane_idx) & 1).astype(np.uint8)
            planes[plane_idx, offsets.octet_offsets[block_idx]:offsets.octet_offsets[block_idx + 1]] = (
                np.packbits(bits, bitorder="little")
            )
    return body


def wide_share(raw_group, num_blocks, num_octets, has_time, byte_planes, bit_idx):
    """The share of residual slots whose zigzagged residual reaches 2^bit_idx, from either layout
    (the package's wide_share reads byte planes only)."""
    if byte_planes:
        return _bitpacking.wide_share(raw_group, num_blocks, num_octets, has_time, bit_idx)
    if not num_octets:
        return 0.0
    planes = planes_view(raw_group, num_blocks, num_octets, has_time)
    wide = np.bitwise_or.reduce(planes[bit_idx:], axis=0)
    return int(np.unpackbits(wide).sum()) / (8 * num_octets)


@njit(nogil=True)
def get_codes(code_planes, plane_byte_offset, out_codes):
    """Unpacks a flagged block's 2-bit non-finite codes from the code planes."""
    for sample_idx in range(out_codes.shape[0]):
        octet_idx = plane_byte_offset + sample_idx // 8
        out_codes[sample_idx] = get_code(code_planes[0, octet_idx], code_planes[1, octet_idx], sample_idx % 8)


@njit(nogil=True)
def _read_rows(raw_group, byte_planes, has_time, sample_offsets, octet_offsets, code_offsets, out_grid_params,
               out_value_anchors, out_residuals, out_codes):
    num_blocks = out_grid_params.shape[0]
    max_octets = 0
    for block_idx in range(num_blocks):
        max_octets = max(max_octets, octet_offsets[block_idx + 1] - octet_offsets[block_idx])
    scratch_low_bytes = np.empty(8 * max_octets, np.uint8)
    scratch_high_bytes = np.empty(8 * max_octets, np.uint8)
    num_octets = int(octet_offsets[num_blocks])
    planes = planes_view(raw_group, num_blocks, num_octets, has_time)
    bplanes = byte_planes_view(raw_group, num_blocks, num_octets, has_time)
    cplanes = code_planes_view(raw_group, num_blocks, num_octets, int(code_offsets[num_blocks]), has_time)
    for block_idx in range(num_blocks):
        out_grid_params[block_idx] = get_int16(raw_group, grid_params_start(num_blocks), num_blocks, block_idx)
        out_value_anchors[block_idx] = get_int64(raw_group, value_anchor_start(num_blocks), num_blocks, block_idx)
        first_sample = sample_offsets[block_idx]
        block_len = sample_offsets[block_idx + 1] - first_sample
        if byte_planes:
            sample_start = 8 * octet_offsets[block_idx]
            scratch_low_bytes[:block_len] = bplanes[0, sample_start:sample_start + block_len]
            scratch_high_bytes[:block_len] = bplanes[1, sample_start:sample_start + block_len]
        else:
            unshuffle_block(
                planes, octet_offsets[block_idx], octet_offsets[block_idx + 1] - octet_offsets[block_idx],
                scratch_low_bytes, scratch_high_bytes,
            )
        for sample_idx in range(block_len):
            out_residuals[first_sample + sample_idx] = unzigzag16(scratch_low_bytes, scratch_high_bytes, sample_idx)
        if raw_group[block_idx] & BLOCK_FLAG_NONFINITE:
            get_codes(cplanes, code_offsets[block_idx], out_codes[first_sample:first_sample + block_len])


def read_group(raw_group, num_blocks, byte_planes=False, has_time=False) -> GroupRows:
    """Deserializes a body (either layout) into its rows.

    Raises:
        ValueError: If the buffer size does not match its block flags and sizes, or a time column is invalid.
    """
    raw_arr = np.frombuffer(raw_group, np.uint8) if not isinstance(raw_group, np.ndarray) else raw_group
    if raw_arr.shape[0] < (BYTES_PER_FLAGS + BYTES_PER_SIZE) * num_blocks:
        raise ValueError(f"a body of {raw_arr.shape[0]} bytes doesn't hold {num_blocks} blocks")
    block_flags = raw_arr[:num_blocks].copy()
    block_sizes, offsets = read_layout(raw_arr, num_blocks)
    if raw_arr.shape[0] != group_size(num_blocks, offsets, has_time):
        raise ValueError(f"a body of {raw_arr.shape[0]} bytes doesn't hold its {num_blocks} blocks")
    num_samples = int(offsets.sample_offsets[-1])
    grid_params = np.empty(num_blocks, np.int64)
    value_anchors = np.empty(num_blocks, np.int64)
    residuals = np.empty(num_samples, np.int16)
    codes = np.zeros(num_samples, np.uint8)
    _read_rows(raw_arr, byte_planes, has_time, offsets.sample_offsets, offsets.octet_offsets, offsets.code_offsets,
               grid_params, value_anchors, residuals, codes)
    time_rows = None
    if has_time:
        # Zeroed: read_time_rows leaves regular blocks' residuals untouched
        time_rows = allocate_time_rows(num_blocks, num_samples)
        time_rows.residuals[:] = 0
        everyone = np.arange(num_blocks, dtype=np.int64)
        _bitpacking.check_time_rows_status(*_bitpacking.read_time_rows(raw_arr, *offsets, *time_rows, everyone))
    return GroupRows(block_flags, block_sizes, grid_params, value_anchors, residuals, codes, time_rows)


def read_rows(parsed: _group.ParsedGroup) -> GroupRows:
    """Every block's rows (residuals and codes, without dequantizing) and time rows."""
    return read_group(parsed.raw_body, parsed.header.num_blocks, parsed.header.byte_planes, parsed.has_time)
