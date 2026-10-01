# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Packing rows into body bytes and back (_bitpacking): bit and byte planes, code planes, time
residual planes, their padding, and the body writers and readers."""

import numpy as np
import pytest

from fluxcode import _bitpacking, _format
from fluxcode._format import (
    HEAD_DECIMAL,
    HEAD_IRREGULAR_TIME,
    HEAD_LONG_TIME,
    HEAD_NONFINITE,
)

RNG = np.random.default_rng(0)

SIZES = [*range(18), 255, 256, 257]
"""Block sizes around the 8-sample groups, plus a few long ones."""


def random_rows(rng, sizes, has_time, nonfinite=0.3, irregular=0.5, long=0.3):
    """Valid UnitRows for blocks of the given sizes: random flags, columns, residuals and codes,
    and (with has_time) regular, irregular and long time blocks."""
    sizes = np.asarray(sizes, np.int64)
    num_blocks, num_samples = sizes.shape[0], int(sizes.sum())
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    filled = sizes > 0
    flags = (rng.integers(0, 4, num_blocks) | rng.choice([0, HEAD_DECIMAL], num_blocks)).astype(np.uint8)
    flags |= np.where(filled & (rng.random(num_blocks) < nonfinite), HEAD_NONFINITE, 0).astype(np.uint8)
    params = np.where(filled, rng.integers(-2 ** 40, 2 ** 40, num_blocks), 0)
    anchors = np.where(filled, rng.integers(-2 ** 63, 2 ** 63 - 1, num_blocks), 0)
    residuals = rng.integers(-2 ** 15, 2 ** 15, num_samples).astype(np.int16)
    codes = np.zeros(num_samples, np.uint8)
    for block_idx in np.flatnonzero(flags & HEAD_NONFINITE):
        codes[offsets[block_idx]:offsets[block_idx + 1]] = rng.integers(0, 4, sizes[block_idx])
    flags[~filled] = 0
    time_rows = None
    if has_time:
        starts = np.where(filled, rng.integers(-2 ** 62, 0) + np.cumsum(rng.integers(0, 10 ** 12, num_blocks)), 0)
        steps = np.where(sizes >= 2, rng.integers(1, 10 ** 6, num_blocks), 0)
        refs = np.where(sizes >= 2, rng.integers(0, 1000, num_blocks), 0).astype(np.uint64)
        time_residuals = np.zeros(num_samples, np.uint64)
        for block_idx in np.flatnonzero((sizes >= 2) & (rng.random(num_blocks) < irregular)):
            flags[block_idx] |= HEAD_IRREGULAR_TIME
            block = time_residuals[offsets[block_idx]:offsets[block_idx + 1]]
            block[1:] = rng.integers(0, 2 ** 20, block.shape[0] - 1)
            if rng.random() < long:
                flags[block_idx] |= HEAD_LONG_TIME
                block[rng.integers(1, block.shape[0])] = rng.integers(2 ** 32, 2 ** 64, dtype=np.uint64)
        time_rows = _format.TimeRows(starts.astype(np.int64), steps.astype(np.int64), refs, time_residuals)
    return _format.UnitRows(flags, sizes, params.astype(np.int64), anchors.astype(np.int64), residuals, codes,
                            time_rows)


def assert_rows_equal(got, want):
    for name, got_field, want_field in zip(_format.UnitRows._fields, got, want):
        if name == "time_rows":
            assert (got_field is None) == (want_field is None)
            if want_field is not None:
                for got_column, want_column in zip(got_field, want_field):
                    np.testing.assert_array_equal(got_column, want_column)
        else:
            np.testing.assert_array_equal(got_field, want_field, err_msg=name)


def assert_padding_zero(body, rows, byte_planes, has_time):
    """Every plane field pads each block's last byte with zero bits (byte planes: zero bytes)."""
    num_blocks, sizes = rows.block_flags.shape[0], rows.block_sizes
    lay = _format.layout(rows.block_flags, sizes)
    groups, code_groups = int(lay.group_offsets[-1]), int(lay.code_offsets[-1])
    fields = [(_bitpacking.code_planes_view(body, num_blocks, groups, code_groups, has_time), lay.code_offsets)]
    if has_time:
        short, long = _bitpacking.time_planes_views(body, num_blocks, groups, code_groups, int(lay.short_offsets[-1]),
                                                    int(lay.long_offsets[-1]))
        fields += [(short, lay.short_offsets), (long, lay.long_offsets)]
    if byte_planes:
        planes = _bitpacking.byte_planes_view(body, num_blocks, groups, has_time)
        for block_idx in range(num_blocks):
            assert not planes[:, 8 * lay.group_offsets[block_idx] + sizes[block_idx]:8 * lay.group_offsets[block_idx + 1]].any()
    else:
        fields.append((_bitpacking.planes_view(body, num_blocks, groups, has_time), lay.group_offsets))
    for field, field_offsets in fields:
        for block_idx in range(num_blocks):
            block_bytes = field[:, field_offsets[block_idx]:field_offsets[block_idx + 1]]
            if block_bytes.shape[1]:
                assert not np.unpackbits(block_bytes, axis=1, bitorder="little")[:, sizes[block_idx]:].any()


@pytest.mark.parametrize("has_time", [False, True])
@pytest.mark.parametrize("byte_planes", [False, True])
@pytest.mark.parametrize("size", SIZES)
def test_round_trip(size, byte_planes, has_time):
    """Blocks of each size, between neighbours of other sizes, round-trip in both plane modes."""
    rng = np.random.default_rng(size)
    for _ in range(3):
        rows = random_rows(rng, [size, 5, size, 0, size, 9], has_time)
        body = _bitpacking.write_unit(*rows[:6], byte_planes=byte_planes, time_rows=rows.time_rows)
        assert body.shape[0] == _format.unit_size(6, _format.layout(rows.block_flags, rows.block_sizes), has_time)
        assert_rows_equal(_bitpacking.read_unit(body, 6, byte_planes, has_time), rows)
        assert_padding_zero(body, rows, byte_planes, has_time)


@pytest.mark.parametrize("has_time", [False, True])
@pytest.mark.parametrize("byte_planes", [False, True])
def test_round_trip_largest_block(byte_planes, has_time):
    rows = random_rows(np.random.default_rng(1), [_format.MAX_BLOCK_LEN, 1, 0, 3], has_time, nonfinite=1,
                       irregular=1, long=1)
    assert rows.block_flags[0] & HEAD_NONFINITE
    if has_time:
        assert rows.block_flags[0] & HEAD_LONG_TIME and rows.block_flags[3] & HEAD_IRREGULAR_TIME
    body = _bitpacking.write_unit(*rows[:6], byte_planes=byte_planes, time_rows=rows.time_rows)
    assert_rows_equal(_bitpacking.read_unit(body, 4, byte_planes, has_time), rows)
    assert_padding_zero(body, rows, byte_planes, has_time)


def test_flags_select_the_fields():
    """Only flagged blocks take code planes, only irregular ones time planes, only long ones the
    upper 32 time planes."""
    rows = random_rows(np.random.default_rng(2), [20] * 40, True)
    lay = _format.layout(rows.block_flags, rows.block_sizes)
    groups = np.diff(lay.group_offsets)
    for flag, field_offsets in [(HEAD_NONFINITE, lay.code_offsets), (HEAD_IRREGULAR_TIME, lay.short_offsets),
                                (HEAD_LONG_TIME, lay.long_offsets)]:
        flagged = (rows.block_flags & flag) > 0
        assert 0 < flagged.sum() < 40
        np.testing.assert_array_equal(np.diff(field_offsets), np.where(flagged, groups, 0))


def _columns(raw, N):
    """The block_flags, block_sizes, grid_params and value_anchor columns of a body."""
    sizes = raw[N:2 * N].astype(np.int64) | (raw[2 * N:3 * N].astype(np.int64) << 8)
    param = raw[3 * N:11 * N].reshape(8, N).T.copy().view(np.int64).ravel()
    anchor = raw[11 * N:19 * N].reshape(8, N).T.copy().view(np.int64).ravel()
    return raw[:N], sizes, param, anchor


@pytest.mark.parametrize("sizes", [[64] * 3, [61] * 3, [5] * 3, [64, 0, 5], [1, 300, 17]])
def test_shuffle_round_trip_and_layout(sizes):
    N = len(sizes)
    sizes = np.array(sizes)
    resid = RNG.integers(-2 ** 15, 2 ** 15, sizes.sum()).astype(np.int16)
    head = np.array([1, 0 if not sizes[1] else 2, 3], np.uint8)
    param = np.array([-5, 0 if not sizes[1] else 17, -(2 ** 40)], np.int64)
    anchor = np.array([-1.5, 0.0 if not sizes[1] else 1e300, 0.0]).view(np.int64)
    raw = _bitpacking.write_unit(head, sizes, param, anchor, resid)
    lay = _format.layout(head, sizes)
    assert raw.shape[0] == _format.unit_size(N, lay)
    for got, want in zip(_columns(raw, N), (head, sizes, param, anchor)):
        np.testing.assert_array_equal(got, want)
    s = resid.astype(np.int32)
    u = ((s << 1) ^ (s >> 15)) & 0xFFFF
    planes = raw[19 * N:].reshape(16, -1)
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    for b in range(N):
        block_planes = planes[:, lay.group_offsets[b]:lay.group_offsets[b + 1]]
        bits = np.unpackbits(block_planes, axis=1, bitorder="little")  # (16, 8 * groups): bit j of u[i]
        block_u = u[offsets[b]:offsets[b + 1]]
        np.testing.assert_array_equal(bits[:, :sizes[b]], ((block_u[None] >> np.arange(16)[:, None]) & 1).astype(np.uint8))
        assert not bits[:, sizes[b]:].any()  # each block's last group is padded with zero bits
    rows = _bitpacking.read_unit(raw, N)
    for got, want in zip(rows[:5], (head, sizes, param, anchor, resid)):
        np.testing.assert_array_equal(got, want)
    with pytest.raises(ValueError, match="doesn't hold"):
        _bitpacking.read_unit(raw[:-1], N)


def test_byte_planes_round_trip_and_layout():
    N, n = 3, 64
    sizes = np.full(N, n)
    resid = RNG.integers(-2 ** 15, 2 ** 15, N * n).astype(np.int16)
    head = np.array([1, 2, 3], np.uint8)
    param = np.array([-5, 17, -(2 ** 40)], np.int64)
    anchor = np.array([-1.5, 1e300, 0.0]).view(np.int64)
    raw = _bitpacking.write_unit(head, sizes, param, anchor, resid, byte_planes=True)
    assert raw.shape[0] == _format.unit_size(N, _format.layout(head, sizes))
    np.testing.assert_array_equal(raw[:19 * N], _bitpacking.write_unit(head, sizes, param, anchor, resid)[:19 * N])
    s = resid.astype(np.int32)
    u = ((s << 1) ^ (s >> 15)) & 0xFFFF
    planes = raw[19 * N:].reshape(2, N * n)  # every low byte, then every high byte
    np.testing.assert_array_equal(planes[0], u & 0xFF)
    np.testing.assert_array_equal(planes[1], u >> 8)
    rows = _bitpacking.read_unit(raw, N, byte_planes=True)
    for got, want in zip(rows[:5], (head, sizes, param, anchor, resid)):
        np.testing.assert_array_equal(got, want)


def test_bit_order_vector():
    """§7.4: u[3] = 0x0020, u[6] = 0x0400 -> plane 5 byte 0 = 0b00001000, plane 10 byte 0 = 0b01000000."""
    v = np.zeros(8, np.int16)
    v[3] = 0x0010  # zigzag(16) = 32 = 0x0020
    v[6] = 0x0200  # zigzag(512) = 1024 = 0x0400
    raw = _bitpacking.write_unit(np.zeros(1, np.uint8), np.array([8]), np.zeros(1, np.int64), np.zeros(1, np.int64), v)
    planes = raw[19:].reshape(16, 1)  # after head (1), size (2), param (8) and anchor (8)
    expect = np.zeros(16, np.uint8)
    expect[5], expect[10] = 0b00001000, 0b01000000
    np.testing.assert_array_equal(planes[:, 0], expect)
