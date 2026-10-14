# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
import itertools
import warnings

import numpy as np
import pytest

import fluxprune
from fluxprune._api import rank


def exact_mse(t, x, keep, values=None):
    """Time-weighted mean squared difference of the kept points' straight lines (at `values`, or the
    samples) against the input's, by brute force: on each input segment both are linear, so the integral
    is exact. Each segment j -> j+1 lies on the line between the kept points around it (which also settles
    repeated times)."""
    t = np.asarray(t, np.float64)
    v = dict(zip(keep.tolist(), x[keep] if values is None else values))
    total = 0.0
    for j in range(len(t) - 1):
        h = t[j + 1] - t[j]
        if h == 0:
            continue
        a = keep[np.searchsorted(keep, j, side="right") - 1]
        b = keep[np.searchsorted(keep, j + 1)]
        s = (v[b] - v[a]) / (t[b] - t[a])
        d0 = x[j] - (v[a] + s * (t[j] - t[a]))
        d1 = x[j + 1] - (v[a] + s * (t[j + 1] - t[a]))
        total += h * (d0 * d0 + d0 * d1 + d1 * d1) / 3
    return total / (t[-1] - t[0])


def max_deviation(t, x, keep, values=None):
    """Largest distance in value of an input point from the kept points' lines, by brute force (a point at
    the time of a vertical step: its distance from the step's span of values)."""
    t = np.asarray(t, np.float64)
    v = dict(zip(keep.tolist(), x[keep] if values is None else values))
    worst = 0.0
    for a, b in itertools.pairwise(keep):
        for j in range(a + 1, b):
            if t[b] > t[a]:
                d = abs(x[j] - (v[a] + (v[b] - v[a]) * (t[j] - t[a]) / (t[b] - t[a])))
            else:
                d = max(0.0, x[j] - max(v[a], v[b]), min(v[a], v[b]) - x[j])
            worst = max(worst, d)
    return worst


def series(seed, n=400):
    rng = np.random.default_rng(seed)
    t = np.cumsum(rng.integers(1, 20, n)) * 1_000_000_000 + 1_790_035_200_000_000_000
    x = np.cumsum(rng.normal(size=n)) + 5 * np.sin(np.arange(n) / 30)
    return t, x


def index(t, p):
    """The kept points' indices, from their times (unique in series())."""
    return np.searchsorted(t, p.times)


# the rmse pass, step by step (fluxprune._api.rank)

@pytest.mark.parametrize("seed", range(5))
def test_rmse_matches_brute_force(seed):
    t, x = series(seed)
    r = rank(t, x)
    span = x.max() - x.min()
    for k in (r.min_points, 5, 20, 50, 200, 399, 400):
        keep = r.top(k)
        assert len(keep) == k
        want = np.sqrt(exact_mse(t - t[0], x, keep)) / span
        assert r.relative_rmse(k) == pytest.approx(want, rel=1e-9, abs=1e-12)
    curve = r.relative_rmse()
    assert np.isnan(curve[: r.min_points]).all() and curve[400] == 0


def test_required_and_nested():
    t, x = series(9)
    r = rank(t, x)
    assert set(r.required) == {0, len(x) - 1, int(np.argmin(x)), int(np.argmax(x))} and r.gaps == 0
    prev = set(r.top(r.min_points))
    assert prev == set(r.required)
    for k in range(r.min_points + 1, len(x) + 1, 7):
        cur = set(r.top(k))
        assert prev <= cur
        prev = cur


def test_straight_lines_cost_nothing():
    t = np.arange(100.0)
    x = np.where(t < 50, t, 100 - t)  # a tent: 3 points describe it
    r = rank(t, x)
    assert r.relative_rmse(3) == 0
    assert list(r.top(3)) == [0, 50, 99]


def test_datetime_and_float_times_agree():
    t, x = series(3)
    a = fluxprune.prune(t.view("datetime64[ns]"), x)
    b = fluxprune.prune((t - t[0]) / 1e9, x)
    assert a.times.dtype == np.dtype("datetime64[ns]")
    assert np.array_equal(a.times.view(np.int64), t[index(t, a)]) and np.array_equal(a.values, b.values)


def test_repeated_times_and_degenerate_inputs():
    t = np.array([0, 1, 1, 2, 3, 3, 4], np.int64)
    x = np.array([0, 0, 5, 5, 1, 0, 0.0])
    r = rank(t, x)
    for k in range(r.min_points, 8):
        assert r.relative_rmse(k) == pytest.approx(np.sqrt(exact_mse(t, x, r.top(k))) / 5, abs=1e-12)
    for t, x in [([], []), ([0], [1.0]), ([0, 1], [1.0, 2.0]), ([0, 1, 2], [3.0, 3, 3]), ([5, 5, 5, 5], [1.0, 2, 3, 4])]:
        p = fluxprune.prune(np.array(t, np.int64), np.array(x))
        assert len(p) == min(len(x), 2) or len(x) == 4 and len(p) >= 2
        assert p.rmse == 0 and p.max_error == 0 and p.rmse_met and p.max_error_met and p.max_points_met
    with pytest.raises(ValueError):
        fluxprune.prune(np.array([0, 2, 1]), np.zeros(3))


@pytest.mark.parametrize("seed", range(3))
def test_refine_lowers_error_and_keeps_required(seed):
    t, x = series(seed)
    r = rank(t, x)
    span = x.max() - x.min()
    for k in (10, 40, 120):
        keep = r.top(k)
        better = r.refine(keep)
        assert len(better) == k and np.all(np.diff(better) > 0) and set(r.required) <= set(better)
        assert r.relative_error(keep) == pytest.approx(r.relative_rmse(k), rel=1e-9, abs=1e-12)
        assert r.relative_error(better) <= r.relative_error(keep) + 1e-12
        assert r.relative_error(better) == pytest.approx(np.sqrt(exact_mse(t - t[0], x, better)) / span, rel=1e-9)
    assert np.array_equal(index(t, fluxprune.prune(t, x)), r.refine(r.top(r.count())))


def test_error_per_point():
    t, x = series(5)
    r = rank(t, x)
    raw = r.relative_rmse()
    sigma = r.noise / (x.max() - x.min())
    assert sigma > 0  # a random walk's unit steps read as a little noise
    for floor, e in ((True, np.sqrt(np.maximum(raw**2 - sigma**2, 0))), (False, raw)):
        for theta in (1e-4, 1e-3, 1e-2):
            k = r.count(error_per_point=theta, noise_floor=floor)
            assert e[k] < theta * (k - r.min_points + 1)
            assert all(e[j] >= theta * (j - r.min_points + 1) for j in range(r.min_points, k))
    assert r.count(error_per_point=1e-3) > r.count(error_per_point=1e-2)
    assert r.count(error_per_point=0) == r.n and r.count(error_per_point=np.inf) == r.min_points
    for bad in (-1e-3, np.nan):
        with pytest.raises(ValueError):
            r.count(error_per_point=bad)
    # the ends of the range: the best max_points, and the max_error pass alone
    best = fluxprune.prune(t, x, error_per_point=0, max_points=30)
    assert np.array_equal(index(t, best), r.refine(r.top(30))) and not best.rmse_met
    alone = fluxprune.prune(t, x, error_per_point=np.inf, max_error=2.0)
    assert set(index(t, alone)) >= set(r.required) and alone.max_error <= 2.0 and alone.rmse_met
    assert len(alone) < len(fluxprune.prune(t, x, max_error=2.0))


# prune()

@pytest.mark.parametrize("seed", range(3))
def test_prune_reports_its_errors(seed):
    t, x = series(seed)
    p = fluxprune.prune(t, x)
    keep = index(t, p)
    assert isinstance(p, fluxprune.Pruned) and p.rmse_met and p.max_error_met and p.max_points_met
    assert np.array_equal(p.values, x[keep])
    assert p.rmse == pytest.approx(np.sqrt(exact_mse(t - t[0], x, keep)), rel=1e-9)
    assert p.max_error == pytest.approx(max_deviation(t, x, keep), rel=1e-9)
    assert len(fluxprune.prune(t, x, error_per_point=0)) > len(p) > len(fluxprune.prune(t, x, error_per_point=1e-3))


def test_default_error_per_point():
    """None: 0.1% of the range at a tenth of the input's points."""
    t, x = series(4)
    r = rank(t, x)
    assert fluxprune._api.default_error_per_point(400) == pytest.approx(0.001 / 40)
    assert r.count() == r.count(error_per_point=0.001 / 40)
    assert np.array_equal(fluxprune.prune(t, x).times, fluxprune.prune(t, x, error_per_point=0.001 / 40).times)
    gap = x.copy()
    gap[50:60] = np.nan  # the input's length, not the keepable points'
    assert rank(t, gap).count() == rank(t, gap).count(error_per_point=0.001 / 40)


@pytest.mark.filterwarnings("ignore:.*below the noise")  # a random walk's unit steps
@pytest.mark.parametrize("seed", range(4))
def test_max_error(seed):
    t, x = series(seed)
    auto = fluxprune.prune(t, x)
    previous = len(x) + 1
    for tol in (0.5, 1.0, 2.0, 4.0):
        p = fluxprune.prune(t, x, max_error=tol)
        assert p.max_error_met and p.rmse_met
        assert p.max_error <= tol and max_deviation(t, x, index(t, p)) <= tol * (1 + 1e-9)
        assert set(auto.times) <= set(p.times)  # it only adds points
        assert len(p) >= len(auto) and len(p) <= previous
        previous = len(p)
    assert len(fluxprune.prune(t, x, max_error=0)) == len(x)  # a random walk has no straight stretch
    assert np.array_equal(fluxprune.prune(t, x, max_error=1e9).times, auto.times)


def test_max_points():
    t, x = series(7)
    want = fluxprune.prune(t, x, max_error=1.0)
    alone = fluxprune.prune(t, x, error_per_point=np.inf, max_error=1.0)  # max_error with no rmse pass
    worse = []
    for cap in (6, 10, 25, 50, len(alone) - 1):
        p = fluxprune.prune(t, x, max_error=1.0, max_points=cap)
        assert len(p) <= cap and not p.rmse_met and not p.max_error_met and p.max_error > 1.0 and p.max_points_met
        # relaxed to about the tightest bound that fits (the pass's count isn't quite monotone in the bound)
        assert len(fluxprune.prune(t, x, error_per_point=np.inf, max_error=0.9 * p.max_error)) > cap
        worse.append(worst_first(t, x, rank(t, x).required, cap) / p.max_error)
    assert np.mean(worse) > 1  # tighter than splitting the worst segment at its worst point
    p = fluxprune.prune(t, x, max_error=1.0, max_points=len(want) - 1)
    assert len(p) < len(want) and not p.rmse_met and p.max_error_met  # the rmse pass gave way
    assert np.array_equal(fluxprune.prune(t, x, max_error=1.0, max_points=len(want)).times, want.times)
    auto = fluxprune.prune(t, x)
    p = fluxprune.prune(t, x, max_points=len(auto) - 3)
    assert len(p) == len(auto) - 3 and not p.rmse_met and p.max_error_met
    assert fluxprune.prune(t, x, max_points=len(auto)).rmse_met
    # when the cap is reached, the rmse pass gives way to max_error first
    loose = fluxprune.prune(t, x, max_error=8.0)
    p = fluxprune.prune(t, x, error_per_point=1e-4, max_error=8.0, max_points=len(loose) + 5)
    assert p.max_error <= 8.0 and len(p) <= len(loose) + 5
    with pytest.raises(ValueError):
        fluxprune.prune(t, x, max_points=3)


def worst_first(t, x, required, cap):
    """Largest error after splitting the segment that strays furthest at its worst point until there are cap points
    (Douglas-Peucker, worst first): the reference a binding cap is compared with."""
    keep = sorted(int(i) for i in required)
    while len(keep) < cap:
        worst = (0.0, None)
        for a, b in itertools.pairwise(keep):
            for j in range(a + 1, b):
                d = abs(x[j] - (x[a] + (x[b] - x[a]) * (t[j] - t[a]) / (t[b] - t[a])))
                worst = max(worst, (d, j), key=lambda w: w[0])
        if worst[1] is None:
            break
        keep = sorted([*keep, worst[1]])
    return max_deviation(t, x, np.array(keep))


def test_noise_warning():
    rng = np.random.default_rng(3)
    t = np.arange(3600.0)
    x = rng.normal(0, 1, 3600)
    with pytest.warns(UserWarning, match="below the noise"):
        fluxprune.prune(t, x, max_error=0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fluxprune.prune(t, x, max_error=3.0)
        fluxprune.prune(t, x)


# the noise floor

def noise_estimate(t, x):
    """σ from each point's distance to its neighbours' chord, by brute force."""
    t = np.asarray(t, np.float64)
    r = []
    for i in range(1, len(x) - 1):
        w = (t[i] - t[i - 1]) / (t[i + 1] - t[i - 1])
        r.append(abs(x[i] - x[i - 1] - w * (x[i + 1] - x[i - 1])) / np.sqrt(1 + w * w + (1 - w) ** 2))
    return 1.4826 * np.median(r)


@pytest.mark.parametrize("irregular", [False, True])
def test_noise_estimate(irregular):
    rng = np.random.default_rng(7)
    t = np.cumsum(rng.uniform(0.2, 1.8, 3600)) if irregular else np.arange(3600.0)
    x = 50 + 0.01 * t + rng.normal(0, 2, 3600)
    r = rank(t, x)
    assert r.noise == pytest.approx(noise_estimate(t, x), rel=1e-9)
    assert r.noise == pytest.approx(2.0, rel=0.05)
    spikes = x + sum(800 / (1 + ((t - c) / 2) ** 2) for c in np.arange(100.0, t[-1], 180))
    assert rank(t, spikes).noise == pytest.approx(2.0, rel=0.15)  # the median ignores the spikes
    assert rank(t, np.round(np.sin(t / 300), 1)).noise == 0.0  # quantized steps: no noise
    with_gap = x.copy()
    with_gap[100:110] = np.nan
    assert rank(t, with_gap).noise == pytest.approx(r.noise, rel=0.02)


def test_noise_floor():
    rng = np.random.default_rng(11)
    t = np.arange(3600.0)
    quiet = 50 + rng.normal(0, 2, 3600)
    theta = 0.0015  # a coarse rate: at the default, without the floor, the noise keeps most of the points
    on, off = fluxprune.prune(t, quiet, theta), fluxprune.prune(t, quiet, theta, noise_floor=False)
    assert on.noise == pytest.approx(2.0, rel=0.05) and off.noise == on.noise
    assert len(on) * 4 < len(off)  # a tenth or so: it doesn't trace the noise
    assert on.rmse < 1.2 * off.rmse  # against the samples, what's left is mostly the noise either way
    under = np.zeros(2)  # against the signal under the noise, fewer points are closer on average
    for seed in range(10):
        q = 50 + np.random.default_rng(seed).normal(0, 2, 3600)
        for j, floor in enumerate((True, False)):
            p = fluxprune.prune(t, q, theta, noise_floor=floor)
            under[j] += np.sqrt(np.mean((np.interp(t, p.times, p.values) - 50) ** 2))
    assert under[0] < 0.9 * under[1]
    assert len(fluxprune.prune(t, quiet)) < 40 and len(fluxprune.prune(t, quiet, noise_floor=False)) > 2000
    sine = np.sin(2 * np.pi * t / 1200)  # smooth: no noise, no change
    assert np.array_equal(fluxprune.prune(t, sine).times, fluxprune.prune(t, sine, noise_floor=False).times)
    # narrow spikes on the noise: cheap in the rmse, kept by a max_error of a few σ
    truth = 50 + sum(800 / (1 + ((t - c) / 2) ** 2) for c in np.arange(300.0, 3600, 600))
    x = truth + rng.normal(0, 2, 3600)
    p = fluxprune.prune(t, x, max_error=20)
    line = np.interp(t, p.times, p.values)
    for c in np.arange(300.0, 3600, 600):
        near = np.abs(t - c) <= 5
        assert truth[near].max() - line[near].max() < 20


# gaps: runs of non-finite values

def test_gaps_hold_forward():
    t = np.arange(30.0)
    x = np.sin(t / 3)
    x[5:9] = np.nan
    x[12] = np.inf
    x[13:15] = np.nan  # inf then nan: two starts
    x[29] = np.nan  # the last point
    r = rank(t, x)
    assert set(r.required) >= {4, 5, 9, 11, 12, 13, 15, 28, 29}
    assert not {6, 7, 8, 14} & set(r.removed) and not {6, 7, 8, 14} & set(r.required)  # the inside is never kept
    p = fluxprune.prune(t, x)
    kept = np.searchsorted(t, p.times)
    assert not {6, 7, 8, 14} & set(kept) and {5, 12, 13, 29} <= set(kept)
    assert np.array_equal(p.values, x[kept], equal_nan=True)
    assert np.isfinite(p.rmse) and np.isfinite(p.max_error)
    every = fluxprune.prune(t, x, error_per_point=0)  # every keepable point: no error, and the holds have none
    assert len(every) == 30 - 4 and every.rmse == pytest.approx(0, abs=1e-12) and every.max_error == 0


def test_gaps_beat_the_cap():
    t = np.arange(200.0)
    x = np.sin(t / 7)
    x[10:190:12] = np.nan  # 15 gaps of one point: 3 points each, 45 in all
    r = rank(t, x)
    near = {int(i) + d for i in np.flatnonzero(np.isnan(x)) for d in (-1, 0, 1)}
    base = {0, 199, int(np.nanargmin(x)), int(np.nanargmax(x))}
    assert set(r.required) == near | base and r.gaps == len(near - base)
    with pytest.warns(UserWarning, match="max_points"):
        p = fluxprune.prune(t, x, max_points=20)
    assert not p.max_points_met and len(p) == r.min_points
    p = fluxprune.prune(t, x, max_points=r.min_points)
    assert p.max_points_met and len(p) == r.min_points


# refit

def refit_oracle(t, x, keep, pinned):
    """Values at the kept points minimising the time-weighted squared error, pinned ones at their samples: least
    squares at two Gauss-Legendre nodes per input segment (dense; exact, since the residual is linear there)."""
    t = np.asarray(t, np.float64)
    rows, ys = [], []
    for j in range(len(t) - 1):
        for u in (0.5 - 0.5 / np.sqrt(3), 0.5 + 0.5 / np.sqrt(3)):
            tt = t[j] + u * (t[j + 1] - t[j])
            yy = x[j] + u * (x[j + 1] - x[j])
            s = np.searchsorted(t[keep], tt, side="right") - 1
            row = np.zeros(len(keep))
            w = (tt - t[keep[s]]) / (t[keep[s + 1]] - t[keep[s]])
            row[s], row[s + 1] = 1 - w, w
            rows.append(row * np.sqrt(t[j + 1] - t[j]))
            ys.append(yy * np.sqrt(t[j + 1] - t[j]))
    A, b = np.array(rows), np.array(ys)
    fix = np.isin(keep, pinned)
    c = x[keep].astype(float)
    c[~fix] = np.linalg.lstsq(A[:, ~fix], b - A[:, fix] @ c[fix], rcond=None)[0]
    return c


@pytest.mark.parametrize("seed", range(3))
def test_refit(seed):
    t, x = series(seed, 120)
    t = (t - t[0]) / 1e9
    plain = fluxprune.prune(t, x)
    p = fluxprune.prune(t, x, refit=True)
    keep = np.searchsorted(t, p.times)
    assert np.array_equal(p.times, plain.times) and p.rmse <= plain.rmse
    pinned = rank(t, x).required
    assert np.array_equal(p.values[np.isin(keep, pinned)], x[pinned])
    assert p.values == pytest.approx(refit_oracle(t, x, keep, pinned), rel=1e-8, abs=1e-10)
    assert p.rmse == pytest.approx(np.sqrt(exact_mse(t, x, keep, p.values)), rel=1e-9)
    assert p.max_error == pytest.approx(max_deviation(t, x, keep, p.values), rel=1e-9)
    q = fluxprune.prune(t, x, max_error=1.0, refit=True)
    assert q.max_error_met == (q.max_error <= 1.0)


def test_refit_with_gaps():
    t = np.arange(100.0)
    x = np.sin(t / 9) * 3
    x[40:45] = np.nan
    p = fluxprune.prune(t, x, refit=True)
    keep = np.searchsorted(t, p.times)
    assert np.isnan(p.values[keep == 40]).all() and np.isfinite(p.values[keep != 40]).all()
    assert np.isfinite(p.rmse) and p.rmse <= fluxprune.prune(t, x).rmse


# the max_error pass: farthest reach

def chords(t, x, tol, a):
    """(j, whether the chord a -> j stays within tol of every point between) for j after a, until no line from a
    could (the slope bounds cross)."""
    lo, hi = -np.inf, np.inf
    for j in range(a + 1, len(x)):
        du = t[j] - t[a]
        s = (x[j] - x[a]) / du
        yield j, lo <= s <= hi
        lo, hi = max(lo, (x[j] - tol - x[a]) / du), min(hi, (x[j] + tol - x[a]) / du)
        if lo > hi:
            return


def door_count(t, x, tol):
    """Points swinging door's rule keeps between the ends: each line ends at the first point it can't reach."""
    a, count = 0, 1
    while a < len(x) - 1:
        reach = a + 1
        for j, ok in chords(t, x, tol, a):
            if not ok:
                break
            reach = j
        a, count = reach, count + 1
    return count


def fewest_count(t, x, tol):
    """The fewest points, the ends included, whose lines stay within tol of every point (BFS over chords)."""
    dist = [1] + [len(x) + 1] * (len(x) - 1)
    for a in range(len(x)):
        for j, ok in chords(t, x, tol, a):
            if ok:
                dist[j] = min(dist[j], dist[a] + 1)
    return dist[-1]


@pytest.mark.filterwarnings("ignore:.*below the noise")
@pytest.mark.parametrize("seed", range(4))
def test_max_error_pass_reaches_far(seed):
    rng = np.random.default_rng(seed)
    t = np.cumsum(rng.integers(1, 4, 300)).astype(np.float64)
    x = np.sin(t / 40) * 5 + rng.normal(0, 0.5, 300)
    for tol in (0.5, 1.0, 2.0):
        got = fluxprune.prune(t, x, error_per_point=np.inf, max_error=tol)  # the required points and the pass
        assert got.max_error <= tol and max_deviation(t, x, np.searchsorted(t, got.times)) <= tol * (1 + 1e-9)
        assert len(got) >= fewest_count(t, x, tol)
        assert len(got) <= door_count(t, x, tol) + 2  # the min and max are kept as well


def test_max_error_pass_with_repeated_times():
    t = np.array([0, 1, 1, 2, 3, 3, 3, 4, 5, 6], np.int64)
    x = np.array([0, 0, 4, 4, 1, 3, 0, 0, 2, 0.0])
    r = rank(t, x)
    for tol in (0.1, 0.5, 1.5):
        keep, _ = r.select(error_per_point=np.inf, max_error=tol)
        worst = r.errors(keep)[1]
        assert worst * 4 <= tol + 1e-12 and max_deviation(t, x, keep) <= tol + 1e-12


def test_a_gap_at_the_start_next_to_the_maximum():
    t = np.arange(20.0)
    x = np.sin(t / 3)
    x[0], x[1] = np.nan, 5.0  # the gap's finite neighbour is the maximum: no gap point beyond the four
    r = rank(t, x)
    assert r.gaps == 0 and set(r.required) == {0, 1, 19, int(np.nanargmin(x))}
    p = fluxprune.prune(t, x)
    assert np.isnan(p.values[0]) and np.isfinite(p.rmse) and np.isfinite(p.max_error)
