# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Each numba stage against a small numpy reference."""

import math

import numpy as np
import pytest

from fluxcode import _decoder, _encoder, _noise, _nonfinite

RNG = np.random.default_rng(0)


def some_blocks():
    t = np.arange(1000.0)
    return [RNG.normal(size=1000), np.cumsum(RNG.normal(size=1000)), 100 * np.sin(t / 7), t ** 2 / 1000,
            np.round(RNG.normal(size=1000), 2), np.full(1000, 3.5)]


@pytest.mark.parametrize("i", range(6))
def test_block_stats(i):
    x = some_blocks()[i]
    lo, hi, mean, bad = _encoder.block_stats(x)
    assert (lo, hi, bad) == (x.min(), x.max(), -1)
    assert np.isclose(mean, x.mean(), rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("where", [0, 517, 999])
def test_block_stats_non_finite(value, where):
    x = RNG.normal(size=1000)
    x[where] = value
    x[where + 1:] = value if where < 999 else x[where + 1:]
    assert _encoder.block_stats(x)[3] == where


def test_block_stats_sum_overflow_is_not_non_finite():
    x = np.full(1000, 1e308)
    _, _, mean, bad = _encoder.block_stats(x)
    assert bad == -1 and mean == pytest.approx(1e308)


@pytest.mark.parametrize("bits", range(1, 17))
def test_exponent(bits):
    for rng in [1e-9, 0.3, 1.0, 1.0 - 2 ** -20, 1000.0, 65535.0, 65535.6, 7.0e12]:
        e = _encoder.exponent(rng, bits)
        assert round(rng / 2.0 ** e) < 2 ** bits  # fits
        assert round(rng / 2.0 ** (e - 1)) >= 2 ** bits  # and is the finest that does


def test_plan_step_precedence():
    rng = 100.0
    fine, coarse = _encoder.exponent(rng, 16), _encoder.exponent(rng, 6)
    lo, hi = -40.0, 60.0
    assert _encoder.plan_step(lo, hi, 0.0, 0.0, 0.0, 6, 16) == fine
    assert _encoder.plan_step(lo, hi, 1.0, -0.7, 0.25, 6, 16) == -2  # floor(log2(0.25))
    assert _encoder.plan_step(lo, hi, 1.0, -0.5, 0.25, 6, 16) == fine  # not gated
    assert _encoder.plan_step(lo, hi, 1e6, -0.7, 0.25, 6, 16) == coarse  # min bits caps the noise floor
    assert _encoder.plan_step(5.0, 5.0, 1.0, -0.7, 0.25, 6, 16) == 0


def test_quantize():
    x = some_blocks()[1]
    q = np.empty(1000, np.int32)
    _encoder.quantize(x, x.min(), -10, q)
    np.testing.assert_array_equal(q, np.floor((x - x.min()) * 2.0 ** 10 + 0.5))


@pytest.mark.parametrize("p", [-3, -1, 0, 2])
def test_detect_decimal(p):
    K = np.cumsum(RNG.integers(-5, 6, 1000)) + 5000
    x = K * 10.0 ** p if p >= 0 else K / 10.0 ** -p
    q = np.empty(1000, np.int32)
    rng = x.max() - x.min()
    got, base = _encoder.detect_decimal(x, x.min(), x.max(), 2.0 ** _encoder.exponent(rng, 16), q)
    assert got == p
    assert base == K.min()  # the grid index of the minimum: the block's anchor
    np.testing.assert_array_equal(q, K - K.min())


def test_detect_decimal_rejects_continuous():
    x = some_blocks()[0]
    q = np.empty(1000, np.int32)
    ps = 2.0 ** _encoder.exponent(x.max() - x.min(), 16)
    assert _encoder.detect_decimal(x, x.min(), x.max(), ps, q)[0] == _encoder.NO_DECIMAL


def ref_order(q, m, orders):
    var = {k: np.var(np.diff(q[:m].astype(np.int64), k)) for k in sorted(orders)}
    best = min(var.values())
    return min(k for k in var if var[k] <= best * (1 + 1e-12))


@pytest.mark.parametrize("m", [4, 250, 1000])
@pytest.mark.parametrize("orders", [{0, 1, 2, 3}, {1, 2}, {0, 3}])
@pytest.mark.parametrize("i", range(5))
def test_pick_order(i, orders, m):
    x = some_blocks()[i]
    q = np.empty(1000, np.int32)
    _encoder.quantize(x, x.min(), _encoder.exponent(x.max() - x.min(), 16), q)
    assert _encoder.pick_order(q, m, sum(1 << k for k in orders)) == ref_order(q, m, orders)


@pytest.mark.parametrize("order", range(4))
def test_residual_and_integrate(order):
    q = RNG.integers(0, 2 ** 16, 1000).astype(np.int32)
    v = np.empty(1000, np.int16)
    _encoder.residual(q, order, v)
    ref = np.diff(np.concatenate([np.zeros(order, np.int64), q]), order)
    np.testing.assert_array_equal(v, ((ref + 32768) % 65536 - 32768).astype(np.int16))
    w = v.astype(np.int32)
    _decoder.integrate(w, order)
    np.testing.assert_array_equal(w, q)


def test_dequantize():
    q = np.arange(10, dtype=np.int32)
    y = np.empty(10)
    _decoder.dequantize_pow2(q, -3.0, -2, y)
    np.testing.assert_array_equal(y, -3.0 + 0.25 * np.arange(10))
    _decoder.dequantize_decimal(q, 12, -1, y)  # anchor: the grid index K0 = 12 of 1.2
    np.testing.assert_array_equal(y, (12 + np.arange(10)) / 10.0)


def ref_class_entropy(v):
    s = v.astype(np.int64)
    u = ((s << 1) ^ (s >> 63)) & 0xFFFF
    L = np.array([int(k).bit_length() for k in u])
    p = np.bincount(L, minlength=17) / len(L)
    nz = p > 0
    return float(-(p[nz] * np.log2(p[nz])).sum() + (p * np.maximum(np.arange(17) - 1, 0)).sum())


@pytest.mark.parametrize("scale", [0, 1, 30, 3000])
def test_estimate_bits(scale):
    v = np.round(RNG.normal(0, scale, 1000)).astype(np.int16) if scale else np.zeros(1000, np.int16)
    assert math.isclose(_encoder.estimate_bits(v), ref_class_entropy(v), abs_tol=1e-9)


def test_allocate_target():
    h = np.array([10.0, 10.0, 0.5, 6.0, 1.2])
    e = np.zeros(5, np.int64)
    ec = np.full(5, 8, np.int64)
    w = np.ones(5, np.int64)
    k = np.zeros(5, np.int64)
    # budget 5 * 5 = 25 of 27.7; blocks can give 8 (clamped from 9), 8, 0, 5, 0 bits
    assert _encoder.allocate_target(h, e, ec, w, 5.0, k)
    np.testing.assert_array_equal(k, [1, 1, 0, 1, 0])  # k = 1 saves 3 >= 2.7
    assert _encoder.allocate_target(h, e, ec, w, 3.0, k)  # excess 12.7: k = 5 saves 13, k = 4 saves 12
    np.testing.assert_array_equal(k, [5, 5, 0, 5, 0])
    assert not _encoder.allocate_target(h, e, ec, w, 6.0, k)  # under budget
    ec = np.array([1, 8, 8, 8, 8])
    _encoder.allocate_target(h, e, ec, w, 3.0, k)  # block 0 gives only 1 bit: k = 6 saves 12, k = 7 saves 13
    np.testing.assert_array_equal(k, [1, 7, 0, 5, 0])
    # Weighted by block size: block 1 is 3 blocks' worth, block 3 outside the budget (weight 0)
    w = np.array([1, 3, 1, 0, 1])
    ec = np.full(5, 8, np.int64)
    # bits 10 + 30 + 0.5 + 1.2 = 41.7 over 6 samples, budget 30: k = 3 saves 3 + 9 = 12 >= 11.7
    assert _encoder.allocate_target(h, e, ec, w, 5.0, k)
    np.testing.assert_array_equal(k, [3, 3, 0, 0, 0])


# Extreme ranges, against exact rational arithmetic (fractions.Fraction holds any double exactly)

from fractions import Fraction

DBL_MAX = float(np.finfo(np.float64).max)
TINY = 5e-324


def ref_exponent(lo, hi, bits):
    """Smallest e with round((hi - lo) / 2^e) < 2^bits, clamped to [-1074, 1023], exactly."""
    r = Fraction(hi) - Fraction(lo)
    e = -1074
    while e < 1023 and math.floor(r / Fraction(2) ** e + Fraction(1, 2)) >= 2 ** bits:
        e += 1
    return e


def ref_quantize(x, lo, e):
    s = Fraction(2) ** -e
    return [math.floor((Fraction(v) - Fraction(lo)) * s + Fraction(1, 2)) for v in x]


EXTREME = [(0.0, TINY), (0.0, 7 * TINY), (-3 * TINY, 1000 * TINY), (0.0, 2.0 ** -1010), (1e-300, 1e-300 + 2e-310),
           (-DBL_MAX, DBL_MAX), (-DBL_MAX, 0.0), (0.0, DBL_MAX), (-1e308, 1e308), (-2.0 ** 1022, 2.0 ** 1022),
           (1e300, DBL_MAX), (-5.0, DBL_MAX)]


@pytest.mark.parametrize("bits", [1, 6, 16])
@pytest.mark.parametrize("lo,hi", EXTREME)
def test_range_exponent_extremes(lo, hi, bits):
    assert _encoder.range_exponent(lo, hi, bits) == ref_exponent(lo, hi, bits)


@pytest.mark.parametrize("bits", [1, 6, 16])
@pytest.mark.parametrize("lo,hi", EXTREME)
def test_quantize_and_dequantize_extremes(lo, hi, bits):
    """q matches exact arithmetic on every path (tiny, normal, huge), q fits in 16 bits, and the
    decoder gets within half a step without overflowing."""
    rng = np.random.default_rng(bits)
    t = rng.uniform(0, 1, 200)
    x = np.clip(lo + t * hi - t * lo, lo, hi)  # lo + t (hi - lo) without overflowing hi - lo
    x[:2] = lo, hi
    e = _encoder.range_exponent(lo, hi, bits)
    q = np.empty(200, np.int32)
    _encoder.quantize(x, lo, e, q)
    assert q.tolist() == ref_quantize(x, lo, e)
    assert ((0 <= q) & (q < 2 ** 16)).all()
    y = np.empty(200)
    _decoder.dequantize_pow2(q, lo, e, y)
    assert np.isfinite(y).all() and y[0] == lo
    half = Fraction(2) ** (e - 1)
    for xi, yi in zip(x, y):
        assert abs(Fraction(yi) - Fraction(xi)) <= half


@pytest.mark.parametrize("e", [971, 1000, 1007])
def test_wide_dequantize_matches_plain_where_finite(e):
    """The half-scale decoder path is bit-identical to lo + q * 2^e wherever that doesn't overflow."""
    rng = np.random.default_rng(e)
    for lo in [-DBL_MAX, -1e300, -TINY, 0.0, 3 * TINY, 1e200]:
        q = rng.integers(0, 2 ** 16, 5000).astype(np.int32)
        q[0] = 0
        y = np.empty(5000)
        _decoder.dequantize_pow2(q, lo, e, y)
        with np.errstate(over="ignore"):
            plain = lo + q * 2.0 ** e
        ok = np.isfinite(plain)
        np.testing.assert_array_equal(y[ok], plain[ok])
        assert (y[~ok] == DBL_MAX).all()


def test_noise_scale_extremes():
    """The noise estimate scales exactly: the same white noise gives the same rho at any scale."""
    x = np.random.default_rng(0).normal(size=1000)
    x = x / np.abs(x).max()
    ref = _noise.noise(x, _encoder.range_scale(x.min(), x.max()))
    for k in (-1060, -1000, 1000, 1023):
        y = np.ldexp(x, k)
        sigma, rho = _noise.noise(y, _encoder.range_scale(y.min(), y.max()))
        if k > -1050:  # at -1060 the samples themselves are rounded to the subnormal grid
            assert rho == ref[1] and sigma == math.ldexp(ref[0], k)
        else:
            assert rho < -0.6


def ref_noise_finite(y, codes):
    """numpy reference: second differences over three consecutive finite samples; lag-1 pairs of
    consecutive ones; robust spread as in noise()."""
    ok = (codes[:-2] | codes[1:-1] | codes[2:]) == 0
    d = y[2:] - 2 * y[1:-1] + y[:-2]
    dv = d[ok] - d[ok].mean()
    k = math.sqrt(math.pi / 2)
    mad = np.abs(dv).mean()
    for _ in range(2):
        a = np.abs(dv)
        mad = a[a <= 4 * k * mad].mean()
    sd = k * mad
    dc = np.zeros(len(d))
    dc[ok] = np.clip(dv, -4 * sd, 4 * sd)
    pairs = (ok[1:] & ok[:-1]).sum()
    return sd / math.sqrt(6), (np.sum(dc[1:] * dc[:-1]) / pairs) / (np.sum(dc ** 2) / ok.sum())


@pytest.mark.parametrize("kind", ["random-walk", "noisy-sine", "chirp", "sin-4.12hz", "impulses", "gauss-spikes"])
@pytest.mark.parametrize("frac", [0.001, 0.01, 0.05, 0.2])
@pytest.mark.parametrize("scale_k", [0, -1060, 1000])
def test_noise_finite_matches_reference(kind, frac, scale_k):
    from _signals import minute
    rng = np.random.default_rng(int(frac * 1000) + abs(scale_k))
    X = np.ldexp(minute(kind, 4).reshape(60, 1000)[:10] / 128, scale_k)  # ranges up to ~2^1000 / ~2^-1060
    d, w = np.empty(1000), np.empty(1000)
    for x in X:
        y = x.copy()
        y[rng.random(1000) < frac] = np.nan
        xf, codes = np.empty(1000), np.empty(1000, np.uint8)
        lo, hi, _, _ = _nonfinite.fill_nonfinite(y, xf, codes)
        k = _encoder.range_scale(lo, hi)
        sigma, rho = _nonfinite.noise_finite(xf, codes, k, 0, d, w)
        yr = np.ldexp(y, -k)  # exact rescale for the reference
        rs, rr = ref_noise_finite(yr, codes)
        if scale_k == -1060:  # subnormal samples are already rounded: compare the gate only
            assert (rho < -0.6) == (rr < -0.6)
            continue
        assert rho == pytest.approx(rr, abs=1e-9)
        assert sigma == pytest.approx(math.ldexp(rs, k), rel=1e-9)
