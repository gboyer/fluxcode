# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
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
def test_count_rules(seed):
    t, x = series(seed)
    r = fluxprune.rank(t, x)
    assert r.count(points=50) == 50
    k = r.count(rmse=0.02)
    assert r.rmse(k) <= 0.02 and (k == r.min_points or r.rmse(k - 1) > 0.02)
    assert all(r.rmse(j) <= 0.02 for j in range(k, 401))
    assert r.count(points=50, rmse=0.02) == round(np.sqrt(50 * k))
    assert r.count(rmse=1.0) == r.min_points and r.count(rmse=0) == 400
    with pytest.raises(ValueError):
        r.count()


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
    assert np.array_equal(fluxprune.prune(t, x, points=40), r.refine(r.keep(40)))


def test_noise_floor():
    rng = np.random.default_rng(0)
    t = np.cumsum(rng.integers(1, 5, 3600))
    x = rng.normal(size=3600)
    r = fluxprune.rank(t, x)
    assert r.noise() * (x.max() - x.min()) == pytest.approx(1.0, rel=0.05)
    assert r.count(rmse=0.01) > 2500  # chasing the noise keeps most of it
    assert r.count(rmse=0.01, noise_floor=True) < 100
    clean = fluxprune.rank(t, np.sin(t / 300))
    assert clean.noise() < 1e-3
    assert clean.count(rmse=0.01, noise_floor=True) == clean.count(rmse=0.01)
    assert fluxprune.rank(t[:10], x[:10]).noise() == 0


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
        r.count(points=10, error_per_point=1e-3)


def test_scale():
    t, x = series(6)
    span = x.max() - x.min()
    r, s = fluxprune.rank(t, x), fluxprune.rank(t, x, scale=10 * span)
    np.testing.assert_array_equal(r.removed, s.removed)  # the ranking doesn't depend on it
    np.testing.assert_allclose(s.rmse()[r.min_points:], r.rmse()[r.min_points:] / 10, rtol=1e-12, atol=1e-15)
    assert s.rmse(50) == pytest.approx(np.sqrt(exact_mse(t - t[0], x, r.keep(50))) / (10 * span), rel=1e-9)
    assert s.count(rmse=0.001) == r.count(rmse=0.01)
    # the same series on a large offset: the range is unchanged, a fixed scale gives the same answer
    np.testing.assert_array_equal(fluxprune.prune(t, x + 2000, error_per_point=1e-3, scale=span),
                                  fluxprune.prune(t, x, error_per_point=1e-3))
    for bad in (0.0, -1.0, np.inf, np.nan):
        with pytest.raises(ValueError):
            fluxprune.rank(t, x, scale=bad)
