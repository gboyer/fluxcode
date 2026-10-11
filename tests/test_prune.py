# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
import itertools

import numpy as np
import pytest

import fluxprune


def exact_mse(t, x, keep):
    """Time-weighted mean squared difference of the kept points' straight lines against the input's,
    by brute force: on each input segment both are linear, so the integral is exact. Each segment
    j -> j+1 lies on the chord between the kept points around it (which also settles repeated times)."""
    t = np.asarray(t, np.float64)
    total = 0.0
    for j in range(len(t) - 1):
        h = t[j + 1] - t[j]
        if h == 0:
            continue
        a = keep[np.searchsorted(keep, j, side="right") - 1]
        b = keep[np.searchsorted(keep, j + 1)]
        s = (x[b] - x[a]) / (t[b] - t[a])
        d0 = x[j] - (x[a] + s * (t[j] - t[a]))
        d1 = x[j + 1] - (x[a] + s * (t[j + 1] - t[a]))
        total += h * (d0 * d0 + d0 * d1 + d1 * d1) / 3
    return total / (t[-1] - t[0])


def series(seed, n=400):
    rng = np.random.default_rng(seed)
    t = np.cumsum(rng.integers(1, 20, n)) * 1_000_000_000 + 1_790_035_200_000_000_000
    x = np.cumsum(rng.normal(size=n)) + 5 * np.sin(np.arange(n) / 30)
    return t, x


@pytest.mark.parametrize("seed", range(5))
def test_rmse_matches_brute_force(seed):
    t, x = series(seed)
    r = fluxprune.rank(t, x)
    span = x.max() - x.min()
    for k in (r.min_points, 5, 20, 50, 200, 399, 400):
        keep = r.keep(k)
        assert len(keep) == k
        want = np.sqrt(exact_mse(t - t[0], x, keep)) / span
        assert r.rmse(k) == pytest.approx(want, rel=1e-9, abs=1e-12)
    curve = r.rmse()
    assert np.isnan(curve[: r.min_points]).all() and curve[400] == 0


def test_required_and_nested():
    t, x = series(9)
    r = fluxprune.rank(t, x)
    assert set(r.required) == {0, len(x) - 1, int(np.argmin(x)), int(np.argmax(x))}
    prev = set(r.keep(r.min_points))
    assert prev == set(r.required)
    for k in range(r.min_points + 1, len(x) + 1, 7):
        cur = set(r.keep(k))
        assert prev <= cur
        prev = cur


def test_straight_lines_cost_nothing():
    t = np.arange(100.0)
    x = np.where(t < 50, t, 100 - t)  # a tent: 3 points describe it
    r = fluxprune.rank(t, x)
    assert r.rmse(3) == 0
    assert list(r.keep(3)) == [0, 50, 99]


def test_datetime_and_float_times_agree():
    t, x = series(3)
    a = fluxprune.rank(t.view("datetime64[ns]"), x)
    b = fluxprune.rank((t - t[0]) / 1e9, x)
    assert np.array_equal(a.removed, b.removed)


def test_non_finite_values_are_kept_with_neighbours():
    t, x = series(4, 100)
    x[[10, 60]] = [np.nan, np.inf]
    r = fluxprune.rank(t, x)
    assert {9, 10, 11, 59, 60, 61} <= set(r.required)
    assert np.isfinite(r.rmse()[r.min_points:]).all()


def test_repeated_times_and_degenerate_inputs():
    t = np.array([0, 1, 1, 2, 3, 3, 4], np.int64)
    x = np.array([0, 0, 5, 5, 1, 0, 0.0])
    r = fluxprune.rank(t, x)
    for k in range(r.min_points, 8):
        assert r.rmse(k) == pytest.approx(np.sqrt(exact_mse(t, x, r.keep(k))) / 5, abs=1e-12)
    for t, x in [([], []), ([0], [1.0]), ([0, 1], [1.0, 2.0]), ([0, 1, 2], [3.0, 3, 3]), ([5, 5, 5, 5], [1.0, 2, 3, 4])]:
        r = fluxprune.rank(np.array(t, np.int64), np.array(x))
        assert set(r.keep(0)) >= {0, len(x) - 1} - {-1} if len(x) else len(r.keep(0)) == 0
    with pytest.raises(ValueError):
        fluxprune.rank(np.array([0, 2, 1]), np.zeros(3))


@pytest.mark.parametrize("seed", range(3))
def test_refine_lowers_error_and_keeps_required(seed):
    t, x = series(seed)
    r = fluxprune.rank(t, x)
    span = x.max() - x.min()
    for k in (10, 40, 120):
        keep = r.keep(k)
        better = r.refine(keep)
        assert len(better) == k and np.all(np.diff(better) > 0) and set(r.required) <= set(better)
        assert r.error(keep) == pytest.approx(r.rmse(k), rel=1e-9, abs=1e-12)
        assert r.error(better) <= r.error(keep) + 1e-12
        assert r.error(better) == pytest.approx(np.sqrt(exact_mse(t - t[0], x, better)) / span, rel=1e-9, abs=1e-12)
    k = r.count()
    assert np.array_equal(fluxprune.prune(t, x).index, r.refine(r.keep(k)))


def test_error_per_point():
    t, x = series(5)
    r = fluxprune.rank(t, x)
    e = r.rmse()
    for theta in (1e-4, 1e-3, 1e-2):
        k = r.count(error_per_point=theta)
        assert e[k] < theta * (k - r.min_points + 1)
        assert all(e[j] >= theta * (j - r.min_points + 1) for j in range(r.min_points, k))
    assert r.count(error_per_point=1e-3) > r.count(error_per_point=1e-2)
    with pytest.raises(ValueError):
        r.count(error_per_point=0)


def max_deviation(t, x, keep):
    """Largest distance in value of an input point from the kept points' lines, by brute force (a point at
    the time of a vertical step: its distance from the step's span of values)."""
    t = np.asarray(t, np.float64)
    worst = 0.0
    for a, b in itertools.pairwise(keep):
        for j in range(a + 1, b):
            if t[b] > t[a]:
                d = abs(x[j] - (x[a] + (x[b] - x[a]) * (t[j] - t[a]) / (t[b] - t[a])))
            else:
                d = max(0.0, x[j] - max(x[a], x[b]), min(x[a], x[b]) - x[j])
            worst = max(worst, d)
    return worst


@pytest.mark.parametrize("seed", range(3))
def test_prune_reports_its_errors(seed):
    t, x = series(seed)
    p = fluxprune.prune(t, x)
    assert isinstance(p, fluxprune.Pruned) and not p.capped and len(p) == len(p.index)
    assert p.rmse == pytest.approx(np.sqrt(exact_mse(t - t[0], x, p.index)), rel=1e-9)
    assert p.max_error == pytest.approx(max_deviation(t, x, p.index), rel=1e-9)
    assert len(fluxprune.prune(t, x, error_per_point=1e-4)) > len(p) > len(fluxprune.prune(t, x, error_per_point=1e-2))


@pytest.mark.parametrize("seed", range(4))
def test_max_error(seed):
    t, x = series(seed)
    auto = fluxprune.prune(t, x)
    previous = len(x) + 1
    for tol in (0.5, 1.0, 2.0, 4.0):
        p = fluxprune.prune(t, x, max_error=tol)
        assert not p.capped
        assert p.max_error <= tol and max_deviation(t, x, p.index) <= tol * (1 + 1e-9)
        assert set(auto.index) <= set(p.index)  # it only adds points
        assert len(p) >= len(auto) and len(p) <= previous
        previous = len(p)
    assert len(fluxprune.prune(t, x, max_error=0)) == len(x)  # a random walk has no straight stretch
    assert np.array_equal(fluxprune.prune(t, x, max_error=1e9).index, auto.index)


def test_max_points():
    t, x = series(7)
    want = fluxprune.prune(t, x, max_error=1.0)
    alone = fluxprune.rank(t, x).prune(error_per_point=1e9, max_error=1.0)  # max_error with no automatic points
    for cap in (4, 10, 50, len(alone) - 1):
        p = fluxprune.prune(t, x, max_error=1.0, max_points=cap)
        assert len(p) == cap and p.capped and p.max_error > 1.0  # the guard fills the cap, worst first
    p = fluxprune.prune(t, x, max_error=1.0, max_points=len(want) - 1)
    assert len(p) < len(want) and p.capped and p.max_error <= 1.0  # automatic points gave way
    assert np.array_equal(fluxprune.prune(t, x, max_error=1.0, max_points=len(want)).index, want.index)
    auto = fluxprune.prune(t, x)
    p = fluxprune.prune(t, x, max_points=len(auto) - 3)
    assert len(p) == len(auto) - 3 and p.capped
    assert not fluxprune.prune(t, x, max_points=len(auto)).capped
    # when the cap is reached, the automatic points give way to the max_error guard first
    loose = fluxprune.prune(t, x, max_error=8.0)
    p = fluxprune.prune(t, x, error_per_point=1e-4, max_error=8.0, max_points=len(loose) + 5)
    assert p.max_error <= 8.0 and len(p) <= len(loose) + 5
    with pytest.raises(ValueError):
        fluxprune.prune(t, x, max_points=1)


def test_required_points_beyond_the_cap():
    t = np.arange(50.0)
    x = np.sin(t)
    x[10:40:3] = np.nan
    p = fluxprune.prune(t, x, max_points=4)
    assert len(p) == 4 and p.capped
    assert set(p.index) == {0, 49, int(np.nanargmin(x)), int(np.nanargmax(x))}
    p = fluxprune.prune(t, x, max_points=3, keep=[25])
    assert set(p.index) == {0, 49, int(np.nanargmin(x))}
    assert {9, 10, 11} <= set(fluxprune.prune(t, x).index)


def test_keep():
    t, x = series(8)
    extra = [17, 123, 200, 333]
    p = fluxprune.prune(t, x, keep=extra)
    assert set(extra) <= set(p.index)
    assert set(extra) <= set(fluxprune.rank(t, x, keep=extra).keep(0))
    with pytest.raises(ValueError):
        fluxprune.rank(t, x, keep=[400])


def peaks_oracle(t, x, period, prominence):
    """Brute force of peaks() without a cap."""
    n = len(x)
    t = np.asarray(t, np.float64)
    half = period / 2 if period else 0.0

    def extremes(y):
        found = []
        for i in range(1, n - 1):
            if not y[i] > y[i - 1]:
                continue
            k = i + 1
            while k < n - 1 and y[k] == y[i]:
                k += 1
            if not y[k] < y[i]:
                continue
            left = next((j for j in range(i - 1, -1, -1) if y[j] > y[i]), -1)
            right = next((j for j in range(i + 1, n) if y[j] > y[i]), n)
            bases = [y[left + 1:i + 1].min(), y[i:right].min()]
            known = [b for b, side in zip(bases, (left >= 0, right < n)) if side]  # an end cuts a side
            prom = y[i] - (max(known) if len(known) == 2 else known[0] if known else min(bases))
            if prominence is None or prom >= prominence:
                found.append(i)
        kept = []
        for i in sorted(found, key=lambda i: (-y[i], i)):  # highest first; ties: the earlier
            if all(abs(t[i] - t[j]) >= half for j in kept):
                kept.append(i)
        return sorted(kept)

    marks = sorted([(i, True) for i in extremes(x)] + [(i, False) for i in extremes(-x)])
    out = {0, n - 1}
    for s, (i, top) in enumerate(marks):
        if s and marks[s - 1][1] == top and i - marks[s - 1][0] > 1:
            a = marks[s - 1][0]
            out.add(a + 1 + int(np.argmin(x[a + 1:i]) if top else np.argmax(x[a + 1:i])))
        out.add(i)
    return sorted(out)


@pytest.mark.parametrize("seed", range(4))
def test_peaks_match_brute_force(seed):
    rng = np.random.default_rng(seed)
    n = 300
    t = np.cumsum(rng.integers(1, 4, n)).astype(np.float64)
    x = np.round(np.sin(t / 15) * 5 + rng.normal(0, 0.7, n))  # rounded: flat tops and ties
    for period, prominence in ((None, None), (None, 2.0), (40.0, None), (40.0, 1.0), (90.0, 3.0)):
        got = fluxprune.peaks(t, x, period=period, prominence=prominence)
        assert list(got) == peaks_oracle(t, x, period, prominence), (period, prominence)


def test_peaks_of_a_periodic_signal():
    rng = np.random.default_rng(1)
    t = np.arange(0, 3600, dtype=np.int64) * 1_000_000_000
    period = 60 + 8 * np.sin(np.arange(3600) / 500)  # drifts between 52 and 68 s
    phase = np.cumsum(2 * np.pi / period)
    x = 10 * np.sin(phase) + rng.normal(0, 0.3, 3600)
    got = fluxprune.peaks(t.view("datetime64[ns]"), x, period=np.timedelta64(60, "s"), prominence=5)
    cycles = int(phase[-1] / (2 * np.pi))
    assert abs(len(got) - 2 - 2 * cycles) <= 2  # a peak and a trough per cycle, nothing else
    kinds = np.sign(x[got[1:-1]])
    assert np.all(kinds[1:] != kinds[:-1])  # alternating
    same = fluxprune.peaks(t / 1e9, x, period=60.0, prominence=5)
    assert np.array_equal(got, same)
    noisy = fluxprune.peaks(t / 1e9, x, period=60.0)
    assert len(noisy) >= len(got)
    with pytest.raises(ValueError):
        fluxprune.peaks(t / 1e9, x, period=np.timedelta64(60, "s"))


def test_peaks_cap_and_edges():
    rng = np.random.default_rng(2)
    t = np.arange(2000.0)
    x = np.sin(t / 20) * np.linspace(1, 10, 2000) + rng.normal(0, 0.1, 2000)
    full = fluxprune.peaks(t, x, period=100)
    for cap in (2, 7, 20, len(full) - 1):
        got = fluxprune.peaks(t, x, period=100, max_points=cap)
        assert 2 <= len(got) <= cap and {0, 1999} <= set(got)
    top = fluxprune.peaks(t, x, period=100, max_points=12)
    assert np.mean(np.abs(x[top[1:-1]])) > np.mean(np.abs(x[full[1:-1]]))  # the most prominent win
    x[[100, 500, 501]] = np.nan
    got = fluxprune.peaks(t, x, period=100)
    assert np.isfinite(x[got[1:-1]]).all()
    assert list(fluxprune.peaks(t[:2], x[:2])) == [0, 1] and len(fluxprune.peaks([], [])) == 0
    p = fluxprune.prune(t, x, keep=got)
    assert set(got) <= set(p.index)
