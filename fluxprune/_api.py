# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""prune(): the points that carry a series' shape, with optional guards; peaks(): its peaks and troughs.
rank() and Ranking: the same, step by step."""

import datetime
from dataclasses import dataclass, field

import numpy as np

from ._guard import errors_kernel, fill_kernel
from ._peaks import alternate_kernel, distance_kernel, maxima_kernel, prominence_kernel
from ._rank import rank_kernel, refine_kernel

ERROR_PER_POINT = 0.0015
"""The default exchange rate of error for points: about 1% of the points of 1 s scans in 1 h blocks of
slow process data (report/prune.html). For blocks H times as long, error_per_point / H keeps about the
same density of points."""


@dataclass(frozen=True, eq=False)
class Pruned:
    """What prune() kept."""

    index: np.ndarray
    """Sorted indices of the kept points."""
    max_error: float
    """The largest distance, in value, of any input point from the kept points' straight lines."""
    rmse: float
    """The time-weighted RMSE of the kept points' lines against the input's, in value units."""
    capped: bool
    """max_points kept the result from what the other parameters asked for: fewer points than the
    error_per_point rule wanted, or a max_error not met."""

    def __len__(self):
        return len(self.index)


@dataclass(frozen=True, eq=False)
class Ranking:
    """Every point of a series ordered by significance, and the error of keeping each count.

    Keeping k points keeps the `required` ones (first, last, minimum, maximum, any `keep` given to
    rank(), and any non-finite value with its neighbours) and the k - len(required) last removed;
    a smaller set is always a subset of a larger one.

    Errors are time-weighted RMSE of the kept points' straight lines against the input's, relative
    to the range of the finite values (max - min): `rmse(k)`.
    """

    n: int
    required: np.ndarray
    """Indices always kept, sorted."""
    removed: np.ndarray
    """The other indices in removal order: least significant first."""
    sq_error: np.ndarray
    """sq_error[r]: mean squared error (relative to the range squared) after the first r removals."""
    _tau: np.ndarray = field(repr=False)
    _y: np.ndarray = field(repr=False)
    _priority: np.ndarray = field(repr=False)
    """The required indices in the order a max_points below their number keeps them."""
    _span: float = field(repr=False)

    @property
    def min_points(self) -> int:
        return len(self.required)

    def rmse(self, k=None):
        """Relative RMSE of keeping k points (min_points <= k <= n), or of every count as an array
        indexed by k (nan below min_points)."""
        e = np.sqrt(np.maximum(self.sq_error, 0.0))
        if k is not None:
            return float(e[self.n - self._clip(k)])
        out = np.full(self.n + 1, np.nan)
        out[self.min_points:] = e[::-1]
        return out

    def keep(self, k):
        """Sorted indices of the k most significant points (k is clipped to [min_points, n])."""
        k = self._clip(k)
        mask = np.ones(self.n, bool)
        mask[self.removed[: self.n - k]] = False
        return np.flatnonzero(mask)

    def count(self, error_per_point=ERROR_PER_POINT):
        """The first count k where the relative RMSE falls below error_per_point times the points added
        beyond the required ones: where the error left and the points spent, at that exchange rate,
        meet. More points for a complex block, fewer for a simple one, sublinearly; it hardly depends
        on the sampling rate (report/prune.html, "amortized")."""
        if not error_per_point > 0:
            raise ValueError("error_per_point must be positive")
        e = self.rmse()[self.min_points:]
        ok = np.flatnonzero(e < error_per_point * np.arange(1, len(e) + 1))
        return self.min_points + int(ok[0]) if len(ok) else self.n

    def refine(self, keep, passes=2):
        """keep (sorted indices of at least the required points) with each other point moved to where
        it fits best between its neighbours, `passes` times over: about 10-30% lower error, no longer
        nested. O(n) per pass."""
        keep = np.array(keep, dtype=np.int64)
        if len(self._tau):
            fixed = np.zeros(self.n, np.bool_)
            fixed[self.required] = True
            refine_kernel(self._tau, self._y, keep, fixed, passes)
        return keep

    def error(self, keep):
        """Relative time-weighted RMSE of any sorted subset that includes the first and last points."""
        if not len(self._tau):
            return 0.0
        return float(np.sqrt(errors_kernel(self._tau, self._y, np.asarray(keep, np.int64))[0]))

    def prune(self, error_per_point=ERROR_PER_POINT, max_points=None, max_error=None, passes=2):
        """The points prune() keeps (see there), from this ranking."""
        cap = self.n if max_points is None else int(max_points)
        if cap < 2:
            raise ValueError("max_points must be at least 2 (the first and last points)")
        if max_error is not None and not (np.isfinite(max_error) and max_error >= 0):
            raise ValueError("max_error must be finite and non-negative")
        if self.min_points > cap:
            return self._result(np.sort(self._priority[:cap]), True)
        wanted = self.count(error_per_point)
        k = min(wanted, cap)
        if max_error is None or not len(self._tau):
            return self._result(self.refine(self.keep(k), passes), k < wanted)
        tol = max_error / self._span

        def build(k, tol=tol):
            return fill_kernel(self._tau, self._y, self.refine(self.keep(k), passes), tol)

        # 1. can max_error be met within the cap at all, from the required points alone?
        idx = build(self.min_points)
        if len(idx) > cap:
            return self._result(self._relax(tol, cap), True)
        # 2. it can: the automatic points take the rest, the least significant giving way first
        full = build(k)
        if len(full) <= cap:
            return self._result(full, k < wanted)
        lo, hi = self.min_points, k
        while hi - lo > 1:
            mid = (lo + hi) // 2
            mid_idx = build(mid)
            if len(mid_idx) <= cap:
                lo, idx = mid, mid_idx
            else:
                hi = mid
        return self._result(idx, True)

    def _relax(self, tol, cap):
        """The swinging-door fill of the required points at about the smallest tolerance above tol that fits
        in cap points (a bisection to 0.1%; the fill's count isn't quite monotone in the tolerance): about
        the tightest bound on every point's error the cap allows."""
        base = self.keep(self.min_points)
        hi = errors_kernel(self._tau, self._y, base)[1]  # the fill adds nothing at the required points' error
        best = base
        lo = tol if tol > 0 else hi * 1e-12
        while hi > lo * 1.001:
            mid = np.sqrt(lo * hi)
            idx = fill_kernel(self._tau, self._y, base, mid)
            if len(idx) <= cap:
                hi, best = mid, idx
            else:
                lo = mid
        return best

    def _result(self, idx, capped):
        if not len(self._tau):
            return Pruned(idx, 0.0, 0.0, capped)
        sq, worst = errors_kernel(self._tau, self._y, idx)
        return Pruned(idx, float(worst * self._span), float(np.sqrt(sq) * self._span), capped)

    def _clip(self, k):
        return int(min(max(int(k), self.min_points), self.n))


def _offsets(times, n):
    """(times as float64 offsets from the first, the datetime/timedelta unit or None), validated."""
    t = np.asarray(times)
    if t.shape != (n,):
        raise ValueError("times and values must be 1-D arrays of the same length")
    unit = None
    if np.issubdtype(t.dtype, np.datetime64) or np.issubdtype(t.dtype, np.timedelta64):
        unit = np.datetime_data(t.dtype)[0]
        t = t.view(np.int64)
    if n and np.issubdtype(t.dtype, np.integer):
        dt = np.diff(t)
        if np.any(dt < 0):
            raise ValueError("times must be non-decreasing")
        return np.concatenate(([0.0], np.cumsum(dt, dtype=np.float64))), unit  # exact steps, no int64 overflow
    tau = np.asarray(t, dtype=np.float64)
    if n and (not np.all(np.isfinite(tau)) or np.any(np.diff(tau) < 0)):
        raise ValueError("times must be finite and non-decreasing")
    return (tau - tau[0] if n else tau), unit


def rank(times, values, keep=None):
    """Rank the points (times[i], values[i]) of a series read as straight lines between them.

    times: non-decreasing, datetime64 or numbers (repeated times are allowed: a vertical step).
    values: floats, same length; non-finite values are kept with their neighbours, and no
    straight line spans them.
    keep: indices to keep whatever else is kept, like the first, last, minimum and maximum points
    (peaks() gives some); if max_points is lower than all of these, those listed first win.
    """
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("times and values must be 1-D arrays of the same length")
    n = len(x)
    tau, _ = _offsets(times, n)

    finite = np.isfinite(x)
    priority = [0, n - 1] if n else []
    if finite.any():
        fi = np.flatnonzero(finite)
        priority += [int(fi[np.argmin(x[fi])]), int(fi[np.argmax(x[fi])])]
        lo, hi = x[fi].min(), x[fi].max()
    else:
        lo, hi = 0.0, 0.0
    if keep is not None:
        keep = np.asarray(keep, dtype=np.int64).ravel()
        if len(keep) and (keep.min() < 0 or keep.max() >= n):
            raise ValueError("keep must be indices into values")
        priority += keep.tolist()
    bad = ~finite
    priority += np.flatnonzero(bad | np.roll(bad, 1) | np.roll(bad, -1)).tolist()  # the wrap-around marks the ends again
    priority = np.array(list(dict.fromkeys(priority)), dtype=np.int64)
    required = np.zeros(n, bool)
    required[priority] = True

    duration = tau[-1] if n else 0.0
    if n < 3 or duration <= 0 or hi <= lo:
        # nothing to weigh by (all at one time) or nothing to lose (constant): any removal is free
        rest = np.flatnonzero(~required)
        return Ranking(n, np.flatnonzero(required), rest, np.zeros(len(rest) + 1), np.empty(0), np.empty(0),
                       priority, 0.0)
    y = np.where(finite, (x - lo) / (hi - lo), 0.0)
    tau = tau / duration
    order, delta = rank_kernel(tau, y, ~required)
    return Ranking(n, np.flatnonzero(required), order, np.concatenate(([0.0], np.cumsum(delta))), tau, y,
                   priority, float(hi - lo))


def prune(times, values, error_per_point=ERROR_PER_POINT, max_points=None, max_error=None, keep=None, passes=2):
    """The points to keep of a series read as straight lines between them: a Pruned with their
    sorted indices, the largest and the RMS error they leave, and whether max_points capped them.

    error_per_point: how many points, automatically: the first count where the time-weighted RMSE
        left, relative to the block's range (max - min), falls below this many times the points
        added. For blocks H times as long as another, error_per_point / H keeps about the same
        density of points.
    max_points: a hard cap on the points kept (at least 2).
    max_error: no input point strays further than this, in value units, from the kept lines: each
        segment of the automatic points that strays further is filled by swinging door (about the
        fewest points that bound its error). If that exceeds max_points, the automatic points give
        way, least significant first, while max_error still holds; if max_error can't be met within
        max_points even without them, it is relaxed to the smallest bound the cap allows (no automatic
        points; `max_error` of the result reports the bound, and `capped` says so).
    keep: indices that are kept whatever else is (peaks(), for example).

    The first, last, minimum and maximum points are always kept (within max_points, in that order),
    as are non-finite values with their neighbours; `passes` refinement passes then move each
    automatic point to where it fits best between its neighbours.
    """
    return rank(times, values, keep).prune(error_per_point, max_points, max_error, passes)


def _period(period, unit):
    if isinstance(period, (np.timedelta64, datetime.timedelta)):
        if unit is None:
            raise ValueError("a timedelta period needs datetime64 or timedelta64 times")
        return float(np.timedelta64(period) / np.timedelta64(1, unit))
    return float(period)


def peaks(times, values, period=None, prominence=None, max_points=None):
    """Sorted indices of the peaks and troughs of a series, alternating, with its first and last points.

    prominence: the least a peak (a local maximum) rises above the higher of the two lowest points
        separating it from a higher point on either side, in value units, troughs mirrored: rejects
        noise, on flat stretches and on a peak's top: about 6 times the noise's σ or more. A side with
        no higher point before the end of the series doesn't count, since a block's edge cuts it.
    period: the expected time between peaks (a timedelta for datetime64 times, otherwise in the times'
        units): of two peaks (after the prominence test) less than half a period apart, only the higher
        is kept, so a period that varies down to half of it keeps every peak. Without a prominence it
        keeps a noise bump every half period on a flat stretch.
    max_points: a cap; the most prominent peaks and troughs win.

    Between two peaks with no trough left between them, the lowest point is kept as their trough,
    and the highest point between two troughs as their peak. Non-finite values are skipped.
    O(n log n). Pass the result as prune(keep=...) to keep these and prune the rest.
    """
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("times and values must be 1-D arrays of the same length")
    n = len(x)
    tau, unit = _offsets(times, n)
    half = 0.0
    if period is not None:
        half = _period(period, unit) / 2
        if not (np.isfinite(half) and half > 0):
            raise ValueError("period must be positive")
    if prominence is not None and not prominence >= 0:
        raise ValueError("prominence must be non-negative")
    if max_points is not None and max_points < 2:
        raise ValueError("max_points must be at least 2 (the first and last points)")
    ends = np.array([0, n - 1] if n else [], dtype=np.int64)
    fi = np.flatnonzero(np.isfinite(x))
    if len(fi) < 3:
        return np.unique(np.concatenate((ends, fi)))
    xs, ts = x[fi], tau[fi]
    found = []
    for y in (xs, -xs):
        top = np.flatnonzero(maxima_kernel(y))
        prom = prominence_kernel(y, top)
        if prominence is not None:
            top, prom = top[prom >= prominence], prom[prom >= prominence]
        if half > 0:
            sel = distance_kernel(ts, y, top, half)
            top, prom = top[sel], prom[sel]
        found.append((top, prom))
    (top, ptop), (bottom, pbottom) = found
    idx = np.concatenate((top, bottom))
    is_peak = np.concatenate((np.ones(len(top), np.bool_), np.zeros(len(bottom), np.bool_)))
    prom = np.concatenate((ptop, pbottom))

    def assemble(sel):
        order = np.argsort(idx[sel])
        alt = alternate_kernel(xs, idx[sel][order], is_peak[sel][order])
        return np.unique(np.concatenate((ends, fi[alt])))

    out = assemble(np.arange(len(idx)))
    if max_points is None or len(out) <= max_points:
        return out
    by_prominence = np.argsort(-prom, kind="stable")
    lo, hi = 0, len(idx)  # the most extremes whose assembly fits: lo fits, hi doesn't
    best = ends[:max_points]
    while hi - lo > 1:
        mid = (lo + hi) // 2
        cand = assemble(by_prominence[:mid])
        if len(cand) <= max_points:
            lo, best = mid, cand
        else:
            hi = mid
    return best
