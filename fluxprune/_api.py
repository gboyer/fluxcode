# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""rank(): order every point by how much it matters; Ranking: pick how many to keep."""

from dataclasses import dataclass, field

import numpy as np

from ._rank import rank_kernel, refine_kernel

NOISE_AT_HALF = 0.29
"""The error of keeping half the points of white noise, over its standard deviation: 0.29 from 64
points (± 0.04) to 43,200 (± 0.002), regular or irregular times."""

NOISE_MIN_POINTS = 64
"""Fewer points than this give no noise estimate (noise() is 0)."""


@dataclass(frozen=True, eq=False)
class Ranking:
    """Every point of a series ordered by significance, and the error of keeping each count.

    Keeping k points keeps the `required` ones (first, last, minimum, maximum, and any
    non-finite value with its neighbours) and the k - len(required) last removed; the result
    is the same for every k that a single bottom-up pass gives, so a smaller set is always a
    subset of a larger one.

    Errors are time-weighted RMSE of the kept points' straight lines against the input's,
    relative to the range of the finite values (max - min): `rmse(k)`.
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

    def noise(self):
        """The white-noise level of the series relative to its range, estimated from its own error curve:
        keeping the best half of white noise's points leaves NOISE_AT_HALF of its standard deviation.
        Correlated noise reads lower (it partly looks like signal); a swinging-door archive reads about
        its CompDev. 0 below NOISE_MIN_POINTS points."""
        return self.rmse(self.n // 2) / NOISE_AT_HALF if self.n >= NOISE_MIN_POINTS else 0.0

    def count(self, points=None, rmse=None, noise_floor=False, error_per_point=None):
        """How many points to keep:

        - points: that many;
        - rmse: the fewest whose relative RMSE (and that of every larger count) is at most rmse;
        - points and rmse: their geometric mean, sqrt(points * k_rmse): more points for a complex
          block, fewer for a simple one, but sublinearly in either direction;
        - error_per_point (alone): the first count k where the RMSE falls below error_per_point times
          the points added beyond the required ones, i.e. where the error left and the points spent,
          at that exchange rate, meet. One parameter, no noise estimate, and it hardly depends on the
          sampling rate (report/prune.html, "amortized").

        noise_floor: never aim below the noise (rmse is raised to noise()), so that noise doesn't take
        points for nothing.
        """
        if error_per_point is not None:
            if points is not None or rmse is not None:
                raise ValueError("error_per_point is a rule of its own: give it alone")
            e = self.rmse()[self.min_points:]
            ok = np.flatnonzero(e < error_per_point * np.arange(1, len(e) + 1))
            return self.min_points + int(ok[0]) if len(ok) else self.n
        if points is None and rmse is None:
            raise ValueError("give points, rmse or both, or error_per_point")
        if rmse is None:
            return self._clip(points)
        if noise_floor:
            rmse = max(rmse, self.noise())
        e = np.sqrt(np.maximum(self.sq_error, 0.0))  # e[r]: after r removals
        over = np.flatnonzero(e > rmse)
        k = self.n - (over[0] - 1 if len(over) else len(e) - 1)
        if points is None:
            return k
        return self._clip(round(np.sqrt(self._clip(points) * k)))

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
        tau, y = self._tau, self._y
        keep = np.asarray(keep)
        j = np.arange(self.n - 1)
        a = keep[np.searchsorted(keep, j, side="right") - 1]
        b = keep[np.searchsorted(keep, j + 1)]
        span = tau[b] - tau[a]
        s = np.divide(y[b] - y[a], span, out=np.zeros(len(j)), where=span > 0)
        d0 = y[j] - y[a] - s * (tau[j] - tau[a])
        d1 = y[j + 1] - y[a] - s * (tau[j + 1] - tau[a])
        sq = np.sum(np.diff(tau) * (d0 * d0 + d0 * d1 + d1 * d1)) / 3.0
        return float(np.sqrt(max(sq, 0.0)))

    def _clip(self, k):
        return int(min(max(int(k), self.min_points), self.n))


def rank(times, values):
    """Rank the points (times[i], values[i]) of a series read as straight lines between them.

    times: non-decreasing, datetime64 or numbers (repeated times are allowed: a vertical step).
    values: floats, same length; non-finite values are kept with their neighbours, and no
    straight line spans them.
    """
    x = np.asarray(values, dtype=np.float64)
    t = np.asarray(times)
    if t.shape != x.shape or x.ndim != 1:
        raise ValueError("times and values must be 1-D arrays of the same length")
    n = len(x)
    if np.issubdtype(t.dtype, np.datetime64) or np.issubdtype(t.dtype, np.timedelta64):
        t = t.view(np.int64)
    if n and np.issubdtype(t.dtype, np.integer):
        dt = np.diff(t)
        if np.any(dt < 0):
            raise ValueError("times must be non-decreasing")
        tau = np.concatenate(([0.0], np.cumsum(dt, dtype=np.float64)))  # exact steps, no int64 overflow
    else:
        tau = np.asarray(t, dtype=np.float64)
        if n and (not np.all(np.isfinite(tau)) or np.any(np.diff(tau) < 0)):
            raise ValueError("times must be finite and non-decreasing")
        tau = tau - tau[0] if n else tau

    finite = np.isfinite(x)
    required = np.zeros(n, bool)
    if n:
        required[[0, -1]] = True
    if finite.any():
        fi = np.flatnonzero(finite)
        required[fi[np.argmin(x[fi])]] = True
        required[fi[np.argmax(x[fi])]] = True
        lo, hi = x[fi].min(), x[fi].max()
    else:
        lo, hi = 0.0, 0.0
    bad = ~finite
    required |= bad | np.roll(bad, 1) | np.roll(bad, -1)  # the wrap-around only marks the ends again

    duration = tau[-1] if n else 0.0
    if n < 3 or duration <= 0 or hi <= lo:
        # nothing to weigh by (all at one time) or nothing to lose (constant): any removal is free
        req = np.flatnonzero(required)
        rest = np.flatnonzero(~required)
        return Ranking(n, req, rest, np.zeros(len(rest) + 1), np.empty(0), np.empty(0))
    y = np.where(finite, (x - lo) / (hi - lo), 0.0)
    tau = tau / duration
    order, delta = rank_kernel(tau, y, ~required)
    return Ranking(n, np.flatnonzero(required), order, np.concatenate(([0.0], np.cumsum(delta))), tau, y)


def prune(times, values, points=None, rmse=None, noise_floor=False, error_per_point=None, passes=2):
    """Sorted indices of the points to keep: Ranking.count(...) of them, refined.

    points: how many points to keep; rmse: the time-weighted RMSE to stay within, relative to the
    range of the values (max - min); both: their geometric mean; noise_floor: raise rmse to the
    series' own noise level; error_per_point: stop where the RMSE left is below this many times the
    points added. The first, last, minimum and maximum points are always kept.
    """
    r = rank(times, values)
    return r.refine(r.keep(r.count(points, rmse, noise_floor, error_per_point)), passes)
