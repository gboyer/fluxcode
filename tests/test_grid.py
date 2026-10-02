# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""The snapped power-of-two grid: anchor = rint(min / 2^e) * 2^e, q = rint(x / 2^e) - K with
ties to even, and e the finest step whose snapped grid fits (SPEC §5.1, §5.4)."""

import numpy as np
import pytest
from _series import unit_rows

import fluxcode
from fluxcode import Params, _encoder
from fluxcode._format import BLOCK_FLAG_DECIMAL

RAW = Params(noise_floor_sigma=None, decimal_detection=False)
"""Just the power-of-two grid: no noise floor, no decimal grids."""

DBL_MAX = float(np.finfo(np.float64).max)


def encode_block(x, params=RAW):
    """(unit, e, anchor) of x as one block."""
    unit = fluxcode.encode_blocks(x, [len(x)], params).unit
    rows = unit_rows(unit)
    return unit, int(rows.grid_params[0]), float(rows.value_anchors.view(np.float64)[0])


def decode(unit):
    return fluxcode.decode_unit(unit).values


@pytest.mark.parametrize("e", [-20, -10, -3, 0, 5])
@pytest.mark.parametrize("base", [-7, -6, 0, 3, 4])
def test_minimum_halfway_between_grid_points(e, base):
    """A minimum exactly halfway between two grid points gets q = 0, not -1 (which would wrap to
    65,535 and decode as the block's maximum): K and q round alike."""
    step = 2.0 ** e
    lo = (base + 0.5) * step
    x = lo + np.random.default_rng(e + 100).uniform(0, 40_000, 1000) * step
    x[[0, -1]] = lo, lo + 40_000 * step
    unit, got_e, anchor = encode_block(x)
    assert got_e == e
    assert anchor == np.rint(lo / step) * step  # ties to even
    q = np.empty(1000, np.int32)
    _encoder.quantize_grid(x, e, np.rint(lo / step), q)
    assert q[0] == 0 and q.min() == 0
    y = decode(unit)
    assert np.abs(y - x).max() <= step / 2
    assert y[0] == anchor


def test_fencepost_power_of_two_range():
    """[1, 2] gets 2^B steps at B < 16 (grid-aligned data in it is exact), and 2^15 at B = 16,
    where 2^16 + 1 levels don't fit."""
    assert _encoder.range_exponent(1.0, 2.0, 12) == -12
    assert _encoder.range_exponent(1.0, 2.0, 16) == -15
    assert _encoder.range_exponent(0.0, 1.0, 12) == -12
    k = np.random.default_rng(0).integers(0, 4097, 1000)
    k[:2] = 0, 4096
    x = 1 + k / 4096
    unit, e, _ = encode_block(x, Params(max_quantize_bits=12, noise_floor_sigma=None, decimal_detection=False))
    assert e == -12
    np.testing.assert_array_equal(decode(unit), x)


def test_storage_edge_at_16_bits():
    """A range within half a step of 65,535 steps, placed so the snapped grid would need
    q = 65,536: one level coarser. q never reaches 2^16 anywhere near the edge."""
    step = 2.0 ** -10
    lo, hi = 0.4 * step, 65_535.6 * step  # rint: 0 and 65,536
    assert _encoder.exponent(hi - lo, 16) == -10  # the unsnapped rule fits
    assert _encoder.range_exponent(lo, hi, 16) == -9
    x = np.linspace(lo, hi, 1000)
    unit, e, _ = encode_block(x, Params(noise_floor_sigma=None, decimal_detection=False, min_quantize_bits=16))
    assert e == -9
    assert np.abs(decode(unit) - x).max() <= 2.0 ** -10
    rng = np.random.default_rng(1)
    q = np.empty(2, np.int32)
    for _ in range(20_000):
        lo = rng.uniform(-3, 3) * step
        hi = lo + (65_535 + rng.uniform(-1, 1)) * step
        e = _encoder.range_exponent(lo, hi, 16)
        assert _encoder.quantize_block(np.array([lo, hi]), lo, hi, e, q) == np.rint(lo / 2.0 ** e) * 2.0 ** e
        assert q[0] == 0 and q[1] <= 65_535


@pytest.mark.parametrize("bits", [1, 2, 6, 12, 15, 16])
def test_range_exponent_is_never_coarser_than_unsnapped(bits):
    """The snapped rule picks at most one level coarser than the unsnapped one, and only at 16
    bits (the storage edge); one level finer only below 16 bits."""
    rng = np.random.default_rng(bits)
    for _ in range(5_000):
        lo = rng.normal() * 10.0 ** rng.integers(-5, 5)
        hi = lo + rng.uniform(0, 1) * 10.0 ** rng.integers(-5, 5)
        old, new = _encoder._unsnapped_exponent(lo, hi, bits), _encoder.range_exponent(lo, hi, bits)
        assert old - (bits < 16) <= new <= old + (bits == 16)
        limit = 2 ** bits if bits < 16 else 2 ** 16 - 1
        assert np.rint(hi / 2.0 ** new) - np.rint(lo / 2.0 ** new) <= limit
        assert np.rint(hi / 2.0 ** (new - 1)) - np.rint(lo / 2.0 ** (new - 1)) > limit


@pytest.mark.parametrize("value", [0.3, -0.3, 1e-310, 5e-324, 1e300, -DBL_MAX, DBL_MAX, 123456.789])
@pytest.mark.parametrize("size", [1, 5, 1000])
def test_constant_blocks_decode_exactly(value, size):
    """Range 0: e is only a placeholder, so the anchor is the value itself."""
    x = np.full(size, value)
    unit, _, anchor = encode_block(x)
    assert anchor == value
    np.testing.assert_array_equal(decode(unit), x)


def _blocks(rng):
    """Blocks of every kind: noisy, smooth, steps, every short size, extreme magnitudes."""
    t = np.arange(1000)
    yield rng.normal(size=1000) * 3 + 0.123
    yield np.cumsum(rng.normal(size=1000))
    yield 100 * np.sin(t / 37 + rng.uniform())
    yield np.repeat(rng.normal(size=10), 100) * 1e3
    for size in range(1, 9):
        yield rng.normal(size=size) * 7
    for scale in (1e-310, 1e-300, 1e300, 1.7e308):
        yield np.cumsum(rng.normal(size=1000)) / 40 * scale


@pytest.mark.parametrize("params", [RAW, Params(), Params(max_quantize_bits=9, decimal_detection=False)],
                         ids=["raw", "default", "9 bits"])
def test_decoded_values_are_fixed_points(params):
    """decode(encode(y)) == y bit for bit for y = decode(encode(x)), wherever the re-encode
    picks the same step: decoded values sit on the absolute grid."""
    rng = np.random.default_rng(3)
    checked = 0
    for x in _blocks(rng):
        unit, e, _ = encode_block(x, params)
        y = decode(unit)
        unit2, e2, _ = encode_block(y, params)
        flags = unit_rows(unit2).block_flags[0]
        if e2 == e or flags & BLOCK_FLAG_DECIMAL:
            np.testing.assert_array_equal(decode(unit2), y)
            checked += 1
    assert checked >= 15


def test_no_tie_bias_when_coarsening():
    """Re-encoding decoded values one level coarser: half of them sit exactly halfway between
    the coarser grid's points. Ties to even keep the mean; ties up would shift it by a quarter
    of the finer step."""
    x = np.cumsum(np.random.default_rng(5).normal(size=1000))
    unit, e, _ = encode_block(x)
    y = decode(unit)
    coarse = e + 1
    q = np.empty(1000, np.int32)
    base = np.rint(y.min() / 2.0 ** coarse)
    _encoder.quantize_grid(y, coarse, base, q)
    shift = (base + q) * 2.0 ** coarse - y
    assert np.mean(np.abs(shift) == 2.0 ** e) > 0.3  # many ties
    assert abs(shift.mean()) < 0.05 * 2.0 ** e
    ties_up = np.floor(y / 2.0 ** coarse + 0.5) * 2.0 ** coarse - y
    assert ties_up.mean() > 0.15 * 2.0 ** e  # what the test would catch


START = np.datetime64("2026-03-01T12:00", "ms")
SECOND = np.timedelta64(1, "s")


def test_straddling_updates_keep_a_bounded_error():
    """Repeated update_time_blocks on non-decimal data, the replaced samples moving the block's
    min and max (range 1-4x): the kept samples' error stays under one step of the coarsest grid
    the block used, and doesn't change at all while the step doesn't."""
    rng = np.random.default_rng(9)
    t = START + np.arange(1000) * np.timedelta64(10, "ms")  # 100 samples per one-second block
    x = np.cumsum(rng.normal(size=1000))
    unit = fluxcode.encode_time_blocks(x, t, RAW, start_time=START, block_duration=SECOND).unit
    lo, hi = START + 5 * SECOND + np.timedelta64(500, "ms"), START + 6 * SECOND  # second half of block 5
    kept = (t >= START + 5 * SECOND) & (t < lo)
    replaced = (t >= lo) & (t < hi)
    original = x[kept]
    steps = []
    previous = None
    for _ in range(200):
        center, spread = x[kept].mean(), np.ptp(x[kept])
        new = center + rng.uniform(-2, 2) * spread * rng.uniform(0.5, 2, replaced.sum())
        unit = fluxcode.update_time_blocks(unit, new, t[replaced], RAW, update_ranges=[(lo, hi)],
                                           start_time=START, block_duration=SECOND).unit
        e = int(unit_rows(unit).grid_params[5])
        y = decode(unit)[kept]
        steps.append(2.0 ** e)
        assert np.abs(y - original).max() < max(steps)
        if previous is not None and previous[0] == e:
            np.testing.assert_array_equal(y, previous[1])
        previous = (e, y)
    assert len(set(steps)) > 2  # the step moved both ways
