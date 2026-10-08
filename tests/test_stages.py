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
    """Smallest e with rint(hi / 2^e) - rint(lo / 2^e) <= L(bits), clamped to [-1074, 1023],
    exactly (round() of a Fraction rounds half to even, like rint)."""
    limit = 2 ** bits if bits < 16 else 2 ** 16 - 1
    e = -1074
    while e < 1023 and round(Fraction(hi) / Fraction(2) ** e) - round(Fraction(lo) / Fraction(2) ** e) > limit:
        e += 1
    return e


def ref_quantize(x, lo, e):
    """(anchor, q) exactly: the snapped grid, or (lo, relative q) when the snapped anchor overflows."""
    step = Fraction(2) ** e
    base = round(Fraction(lo) / step)
    if abs(base * step) <= Fraction(DBL_MAX):
        return float(base * step), [round(Fraction(v) / step) - base for v in x]
    return lo, [math.floor((Fraction(v) - Fraction(lo)) / step + Fraction(1, 2)) for v in x]


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
    """q and the anchor match exact arithmetic on every path (tiny, normal, huge), q fits in 16
    bits and is 0 at the minimum, and the decoder gets within half a step without overflowing."""
    rng = np.random.default_rng(bits)
    t = rng.uniform(0, 1, 200)
    x = np.clip(lo + t * hi - t * lo, lo, hi)  # lo + t (hi - lo) without overflowing hi - lo
    x[:2] = lo, hi
    e = _encoder.range_exponent(lo, hi, bits)
    q = np.empty(200, np.int32)
    anchor = _encoder.quantize_block(x, lo, hi, e, q)
    assert (anchor, q.tolist()) == ref_quantize(x, lo, e)
    assert ((0 <= q) & (q < 2 ** 16)).all() and q[0] == 0
    y = np.empty(200)
    _decoder.dequantize_pow2(q, anchor, e, y)
    assert np.isfinite(y).all()
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


def ref_robust_noise(d, ok):
    """numpy reference of robust_noise: the mean-removed valid differences in windows of at least
    256 valid ones, each with its clipped MAD spread and clipped to 4 of it; the smallest spread,
    and the lag-1 autocorrelation of adjacent valid pairs."""
    dv = np.where(ok, d - d[ok].mean(), 0.0)
    num_valid = int(ok.sum())
    num_windows = max(1, num_valid // 256)
    counts = np.cumsum(ok)
    ends = [int(np.searchsorted(counts, math.floor(num_valid * (w + 1) / num_windows))) + 1
            for w in range(num_windows - 1)] + [len(d)]
    k = math.sqrt(math.pi / 2)
    spreads, start = [], 0
    for end in ends:
        a = np.abs(dv[start:end][ok[start:end]])
        mad = a.mean()
        for _ in range(2):
            mad = a[a <= 4 * k * mad].mean()
        spreads.append(k * mad)
        dv[start:end] = np.clip(dv[start:end], -4 * k * mad, 4 * k * mad)
        start = end
    pairs = (ok[1:] & ok[:-1]).sum()
    return min(spreads), (np.sum(dv[1:] * dv[:-1]) / pairs) / (np.sum(dv ** 2) / num_valid)


@pytest.mark.parametrize("kind", ["random-walk", "noisy-sine", "chirp", "sin-4.12hz", "impulses", "gauss-spikes"])
@pytest.mark.parametrize("frac", [0.001, 0.01, 0.05, 0.2])
@pytest.mark.parametrize("scale_k", [0, -1060, 1000])
def test_noise_weighted_matches_reference(kind, frac, scale_k):
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
        sigma, rho = _noise.noise_weighted(xf, codes, np.zeros(0), 1.0, k, 0, d, w)
        yr = np.ldexp(y, -k)  # exact rescale for the reference
        ok = (codes[:-2] | codes[1:-1] | codes[2:]) == 0
        spread, rr = ref_robust_noise(yr[2:] - 2 * yr[1:-1] + yr[:-2], ok)
        if scale_k == -1060:  # subnormal samples are already rounded: compare the gate only
            assert (rho < -0.6) == (rr < -0.6)
            continue
        assert rho == pytest.approx(rr, abs=1e-9)
        assert sigma == pytest.approx(math.ldexp(spread, k) / math.sqrt(6), rel=1e-9)


def test_noise_weighted_with_equal_intervals_is_plain():
    """On equal intervals the chord difference is the plain second difference, and with every
    triplet valid noise_weighted is noise."""
    x = np.random.default_rng(5).normal(size=1000)
    k = _encoder.range_scale(x.min(), x.max())
    d, w = np.empty(1000), np.empty(1000)
    no_codes = np.zeros(0, np.uint8)
    ref_sigma, ref_rho = _noise.noise(x, k)
    for intervals in (np.full(999, 7.0), np.zeros(0)):
        sigma, rho = _noise.noise_weighted(x, no_codes, intervals, 7.0, k, 0, d, w)
        assert sigma == pytest.approx(ref_sigma, rel=1e-12) and rho == pytest.approx(ref_rho, rel=1e-12)


def test_noise_weighted_cancels_slopes_across_unequal_intervals():
    """A ramp sampled at jittered times has no noise of its own: the chord difference cancels
    it, where the plain second difference sees the jitter as noise."""
    rng = np.random.default_rng(6)
    ticks = np.arange(1000, dtype=np.int64) * 1000 + rng.integers(-200, 201, 1000)
    noisy = 0.001 * ticks.astype(float) + rng.normal(0, 0.01, 1000)
    d, w = np.empty(1000), np.empty(1000)
    k = _encoder.range_scale(noisy.min(), noisy.max())
    intervals = np.diff(ticks).astype(float)
    sigma, rho = _noise.noise_weighted(noisy, np.zeros(0, np.uint8), intervals, intervals.mean(), k, 0, d, w)
    assert sigma == pytest.approx(0.01, rel=0.1) and rho < -0.6
    assert _noise.noise(noisy, k)[0] > 5 * sigma


def test_robust_noise_takes_the_quietest_window():
    """Noise that varies within a block is floored by its quietest part."""
    rng = np.random.default_rng(7)
    x = np.r_[rng.normal(0, 0.01, 600), rng.normal(0, 1, 600)]
    sigma, rho = _noise.noise(x, _encoder.range_scale(x.min(), x.max()))
    assert sigma == pytest.approx(0.01, rel=0.15) and rho < -0.6


def test_regular_times():
    assert _noise.regular_times(np.arange(5, dtype=np.int64) * 3)
    assert _noise.regular_times(np.array([5, 9], np.int64))
    assert _noise.regular_times(np.zeros(4, np.int64))
    assert not _noise.regular_times(np.array([0, 3, 6, 10], np.int64))


def share_of(ticks):
    """A block's share of one-scan intervals, and which they are."""
    ticks = np.asarray(ticks, np.int64)
    intervals = np.empty(ticks.size - 1)
    return _noise.cadence(ticks, intervals)[0], intervals > 0


def test_cadence_share():
    rng = np.random.default_rng(8)
    scans = np.arange(5000, dtype=np.int64) * 1000
    # Jittered by up to 20% of the interval: every interval is one scan
    assert share_of(scans + rng.integers(0, 201, scans.size))[0] == 1.0
    # A tenth of the scans dropped: their intervals are two scans
    kept = scans[rng.random(scans.size) >= 0.1]
    share, one_scan = share_of(kept)
    np.testing.assert_array_equal(one_scan, np.diff(kept) == 1000)
    assert 0.85 < share < 0.95
    # A sparse archive keeps three scans in ten: about a third of its intervals are one scan
    assert 0.2 < share_of(scans[rng.random(scans.size) < 0.3])[0] < 0.4
    # A few bursts (1 ms apart) don't move the scan interval; zero intervals are never one scan
    burst = np.sort(np.r_[scans, scans[::1000] + 1])
    assert share_of(burst)[0] > 0.99
    assert share_of(np.zeros(10))[0] == 0.0
    jittered = scans + rng.integers(0, 201, scans.size)
    intervals = np.empty(scans.size - 1)
    assert _noise.cadence(jittered, intervals)[1] == pytest.approx(np.diff(jittered).mean())


def test_cadence_factor_ramp():
    factors = [_noise.cadence_factor(share) for share in (0.0, 0.5, 0.6, 0.7, 0.9, 1.0)]
    assert factors == pytest.approx([0.0, 0.0, 0.25, 0.5, 1.0, 1.0])
