# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Packing rows into body bytes and back (_bitpacking): bit and byte planes, code planes, time
residual planes, their padding, and the body writers and readers."""

import numpy as np
import pytest

from fluxcode import _bitpacking, _format
from fluxcode._format import (
    BLOCK_FLAG_DECIMAL,
    BLOCK_FLAG_IRREGULAR_TIME,
    BLOCK_FLAG_LONG_TIME,
    BLOCK_FLAG_NONFINITE,
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
    flags = (rng.integers(0, 4, num_blocks) | rng.choice([0, BLOCK_FLAG_DECIMAL], num_blocks)).astype(np.uint8)
    flags |= np.where(filled & (rng.random(num_blocks) < nonfinite), BLOCK_FLAG_NONFINITE, 0).astype(np.uint8)
    params = np.where(filled, rng.integers(-2 ** 15, 2 ** 15, num_blocks), 0)
    anchors = np.where(filled, rng.integers(-2 ** 63, 2 ** 63 - 1, num_blocks), 0)
    residuals = rng.integers(-2 ** 15, 2 ** 15, num_samples).astype(np.int16)
    codes = np.zeros(num_samples, np.uint8)
    for block_idx in np.flatnonzero(flags & BLOCK_FLAG_NONFINITE):
        codes[offsets[block_idx]:offsets[block_idx + 1]] = rng.integers(0, 4, sizes[block_idx])
    flags[~filled] = 0
    time_rows = None
    if has_time:
        starts = np.where(filled, rng.integers(-2 ** 62, 0) + np.cumsum(rng.integers(0, 10 ** 12, num_blocks)), 0)
        steps = np.where(sizes >= 2, rng.integers(1, 10 ** 6, num_blocks), 0)
        refs = np.where(sizes >= 2, rng.integers(0, 1000, num_blocks), 0).astype(np.uint64)
        time_residuals = np.zeros(num_samples, np.uint64)
        for block_idx in np.flatnonzero((sizes >= 2) & (rng.random(num_blocks) < irregular)):
            flags[block_idx] |= BLOCK_FLAG_IRREGULAR_TIME
            block = time_residuals[offsets[block_idx]:offsets[block_idx + 1]]
            block[1:] = rng.integers(0, 2 ** 20, block.shape[0] - 1)
            if rng.random() < long:
                flags[block_idx] |= BLOCK_FLAG_LONG_TIME
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
    assert rows.block_flags[0] & BLOCK_FLAG_NONFINITE
    if has_time:
        assert rows.block_flags[0] & BLOCK_FLAG_LONG_TIME and rows.block_flags[3] & BLOCK_FLAG_IRREGULAR_TIME
    body = _bitpacking.write_unit(*rows[:6], byte_planes=byte_planes, time_rows=rows.time_rows)
    assert_rows_equal(_bitpacking.read_unit(body, 4, byte_planes, has_time), rows)
    assert_padding_zero(body, rows, byte_planes, has_time)


def test_flags_select_the_fields():
    """Only flagged blocks take code planes, only irregular ones time planes, only long ones the
    upper 32 time planes."""
    rows = random_rows(np.random.default_rng(2), [20] * 40, True)
    lay = _format.layout(rows.block_flags, rows.block_sizes)
    groups = np.diff(lay.group_offsets)
    for flag, field_offsets in [(BLOCK_FLAG_NONFINITE, lay.code_offsets), (BLOCK_FLAG_IRREGULAR_TIME, lay.short_offsets),
                                (BLOCK_FLAG_LONG_TIME, lay.long_offsets)]:
        flagged = (rows.block_flags & flag) > 0
        assert 0 < flagged.sum() < 40
        np.testing.assert_array_equal(np.diff(field_offsets), np.where(flagged, groups, 0))


def _columns(raw, N):
    """The block_flags, block_sizes, grid_params and value_anchor columns of a body."""
    sizes = raw[N:2 * N].astype(np.int64) | (raw[2 * N:3 * N].astype(np.int64) << 8)
    param = raw[3 * N:5 * N].reshape(2, N).T.copy().view(np.int16).ravel().astype(np.int64)
    anchor = raw[5 * N:13 * N].reshape(8, N).T.copy().view(np.int64).ravel()
    return raw[:N], sizes, param, anchor


@pytest.mark.parametrize("sizes", [[64] * 3, [61] * 3, [5] * 3, [64, 0, 5], [1, 300, 17]])
def test_shuffle_round_trip_and_layout(sizes):
    N = len(sizes)
    sizes = np.array(sizes)
    resid = RNG.integers(-2 ** 15, 2 ** 15, sizes.sum()).astype(np.int16)
    flags = np.array([1, 0 if not sizes[1] else 2, 3], np.uint8)
    param = np.array([-5, 0 if not sizes[1] else 17, -1074], np.int64)
    anchor = np.array([-1.5, 0.0 if not sizes[1] else 1e300, 0.0]).view(np.int64)
    raw = _bitpacking.write_unit(flags, sizes, param, anchor, resid)
    lay = _format.layout(flags, sizes)
    assert raw.shape[0] == _format.unit_size(N, lay)
    for got, want in zip(_columns(raw, N), (flags, sizes, param, anchor)):
        np.testing.assert_array_equal(got, want)
    s = resid.astype(np.int32)
    u = ((s << 1) ^ (s >> 15)) & 0xFFFF
    planes = raw[13 * N:].reshape(16, -1)
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    for b in range(N):
        block_planes = planes[:, lay.group_offsets[b]:lay.group_offsets[b + 1]]
        bits = np.unpackbits(block_planes, axis=1, bitorder="little")  # (16, 8 * groups): bit j of u[i]
        block_u = u[offsets[b]:offsets[b + 1]]
        np.testing.assert_array_equal(bits[:, :sizes[b]], ((block_u[None] >> np.arange(16)[:, None]) & 1).astype(np.uint8))
        assert not bits[:, sizes[b]:].any()  # each block's last group is padded with zero bits
    rows = _bitpacking.read_unit(raw, N)
    for got, want in zip(rows[:5], (flags, sizes, param, anchor, resid)):
        np.testing.assert_array_equal(got, want)
    with pytest.raises(ValueError, match="doesn't hold"):
        _bitpacking.read_unit(raw[:-1], N)


def test_byte_planes_round_trip_and_layout():
    N, n = 3, 64
    sizes = np.full(N, n)
    resid = RNG.integers(-2 ** 15, 2 ** 15, N * n).astype(np.int16)
    flags = np.array([1, 2, 3], np.uint8)
    param = np.array([-5, 17, -1074], np.int64)
    anchor = np.array([-1.5, 1e300, 0.0]).view(np.int64)
    raw = _bitpacking.write_unit(flags, sizes, param, anchor, resid, byte_planes=True)
    assert raw.shape[0] == _format.unit_size(N, _format.layout(flags, sizes))
    np.testing.assert_array_equal(raw[:13 * N], _bitpacking.write_unit(flags, sizes, param, anchor, resid)[:13 * N])
    s = resid.astype(np.int32)
    u = ((s << 1) ^ (s >> 15)) & 0xFFFF
    planes = raw[13 * N:].reshape(2, N * n)  # every low byte, then every high byte
    np.testing.assert_array_equal(planes[0], u & 0xFF)
    np.testing.assert_array_equal(planes[1], u >> 8)
    rows = _bitpacking.read_unit(raw, N, byte_planes=True)
    for got, want in zip(rows[:5], (flags, sizes, param, anchor, resid)):
        np.testing.assert_array_equal(got, want)


def test_bit_order_vector():
    """§9.4: u[3] = 0x0020, u[6] = 0x0400 -> plane 5 byte 0 = 0b00001000, plane 10 byte 0 = 0b01000000."""
    v = np.zeros(8, np.int16)
    v[3] = 0x0010  # zigzag(16) = 32 = 0x0020
    v[6] = 0x0200  # zigzag(512) = 1024 = 0x0400
    raw = _bitpacking.write_unit(np.zeros(1, np.uint8), np.array([8]), np.zeros(1, np.int64), np.zeros(1, np.int64), v)
    planes = raw[13:].reshape(16, 1)  # after flags (1), size (2), param (2) and anchor (8)
    expect = np.zeros(16, np.uint8)
    expect[5], expect[10] = 0b00001000, 0b01000000
    np.testing.assert_array_equal(planes[:, 0], expect)


def merge_rows(old, new_ids, new):
    """The rows of old with blocks new_ids replaced or appended by new's (reference for splice_body)."""
    num_blocks = max(old.block_flags.shape[0], int(new_ids[-1]) + 1) if len(new_ids) else old.block_flags.shape[0]
    old_offsets = np.concatenate([[0], np.cumsum(old.block_sizes)])
    new_offsets = np.concatenate([[0], np.cumsum(new.block_sizes)])
    rank = {int(b): r for r, b in enumerate(new_ids)}

    def source(block_idx):
        if block_idx in rank:
            return new, rank[block_idx], new_offsets
        return old, block_idx, old_offsets

    def column(name, time=False):
        return np.array([(getattr(rows.time_rows, name) if time else getattr(rows, name))[idx]
                         for rows, idx, _ in map(source, range(num_blocks))],
                        (getattr(old.time_rows, name) if time else getattr(old, name)).dtype)

    def flat(name, time=False):
        pieces = [(getattr(rows.time_rows, name) if time else getattr(rows, name))[offsets[idx]:offsets[idx + 1]]
                  for rows, idx, offsets in map(source, range(num_blocks))]
        return np.concatenate(pieces).astype((getattr(old.time_rows, name) if time else getattr(old, name)).dtype)

    time_rows = None
    if old.time_rows is not None:
        time_rows = _format.TimeRows(column("starts", True), column("steps", True), column("refs", True),
                                     flat("residuals", True))
    return _format.UnitRows(column("block_flags"), column("block_sizes"), column("grid_params"),
                            column("value_anchors"), flat("residuals"), flat("codes"), time_rows)


def splice_case(rng, num_old, has_time, sizes=(0, 1, 7, 8, 9, 300)):
    """Random old rows, random new block ids (replacing, appending, filling gaps) and new rows."""
    old_sizes = rng.choice(sizes, num_old)
    old = random_rows(rng, old_sizes, has_time)
    num_blocks = num_old + int(rng.integers(0, 4))
    num_new = int(rng.integers(0, min(num_blocks, 6) + 1))
    new_ids = np.sort(rng.choice(num_blocks, num_new, replace=False))
    new_ids = np.union1d(new_ids, np.arange(num_old, num_blocks)).astype(np.int64)  # appended blocks are new
    new = random_rows(rng, rng.choice(sizes, new_ids.shape[0]), has_time)
    if has_time:
        # Starts that grow with the block index, old or new: never decreasing however they merge
        for rows, ids in [(old, np.arange(num_old)), (new, new_ids)]:
            rows.time_rows.starts[:] = np.where(rows.block_sizes, ids * 10 ** 9 + rng.integers(0, 10 ** 6), 0) - 2 ** 61
            rows.time_rows.starts[rows.block_sizes == 0] = 0
    return old, new_ids, new


def sample_offsets(sizes):
    return np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)


@pytest.mark.parametrize("has_time", [False, True])
@pytest.mark.parametrize("seed", range(25))
def test_splice_body_matches_write_unit(seed, has_time):
    rng = np.random.default_rng(seed)
    old, new_ids, new = splice_case(rng, int(rng.integers(1, 12)), has_time)
    old_body = _bitpacking.write_unit(*old[:6], byte_planes=True, time_rows=old.time_rows)
    old_layout = _format.layout(old.block_flags, old.block_sizes)
    if has_time:
        # The splice reads only the old columns: check they come back from the body as given
        starts, steps, refs = (np.empty_like(column) for column in old.time_rows[:3])
        assert _bitpacking.read_time_columns(old_body, *old_layout, starts, steps, refs) == (0, 0)
        np.testing.assert_array_equal(starts, old.time_rows.starts)
    body, offsets = _bitpacking.splice_body(old_body, old_layout, old.time_rows.starts if has_time else None, new_ids,
                                            new, sample_offsets(new.block_sizes))
    merged = merge_rows(old, new_ids, new)
    expected = _bitpacking.write_unit(*merged[:6], byte_planes=True, time_rows=merged.time_rows)
    np.testing.assert_array_equal(body, expected)
    for got, want in zip(offsets, _format.layout(merged.block_flags, merged.block_sizes), strict=True):
        np.testing.assert_array_equal(got, want)


def test_splice_body_large_blocks():
    """Long blocks, every flag, the largest block size, and a replaced block that changes size."""
    rng = np.random.default_rng(7)
    old = random_rows(rng, [1000, _format.MAX_BLOCK_LEN, 257, 0, 1000], True, nonfinite=0.7, irregular=1, long=0.5)
    old.time_rows.starts[:] = np.where(old.block_sizes, np.arange(5) * 10 ** 12, 0)
    new_ids = np.array([2, 6], np.int64)
    gap_fill = np.array([2, 5, 6], np.int64)
    new = random_rows(rng, [4000, 0, 1], True, nonfinite=1, irregular=1, long=1)
    new.time_rows.starts[:] = [2 * 10 ** 12 + 1, 0, 7 * 10 ** 12]
    old_body = _bitpacking.write_unit(*old[:6], byte_planes=True, time_rows=old.time_rows)
    old_layout = _format.layout(old.block_flags, old.block_sizes)
    body, _ = _bitpacking.splice_body(old_body, old_layout, old.time_rows.starts, gap_fill, new,
                                      sample_offsets(new.block_sizes))
    merged = merge_rows(old, gap_fill, new)
    np.testing.assert_array_equal(
        body, _bitpacking.write_unit(*merged[:6], byte_planes=True, time_rows=merged.time_rows))
    with pytest.raises(ValueError, match="past the old unit's end"):
        _bitpacking.splice_body(old_body, old_layout, old.time_rows.starts, new_ids,
                                new._replace(block_flags=new.block_flags[:2]), sample_offsets(new.block_sizes))


@pytest.mark.parametrize("has_time", [False, True])
@pytest.mark.parametrize("seed", range(10))
def test_plane_conversion_matches_write_unit(seed, has_time):
    """One transpose of the residual region gives the other layout's body, for all blocks at once."""
    rng = np.random.default_rng(seed)
    rows = random_rows(rng, rng.choice([0, 1, 7, 8, 9, 300], int(rng.integers(0, 12))), has_time)
    byte_body = _bitpacking.write_unit(*rows[:6], byte_planes=True, time_rows=rows.time_rows)
    bit_body = _bitpacking.write_unit(*rows[:6], byte_planes=False, time_rows=rows.time_rows)
    num_blocks = rows.block_flags.shape[0]
    num_groups = int(_format.layout(rows.block_flags, rows.block_sizes).group_offsets[-1])
    np.testing.assert_array_equal(_bitpacking.to_bit_planes(byte_body, num_blocks, num_groups, has_time), bit_body)
    np.testing.assert_array_equal(_bitpacking.to_byte_planes(bit_body, num_blocks, num_groups, has_time), byte_body)
