# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""prune(): the points that carry a series' shape, read as straight lines between them, with
optional guards. rank() and Ranking are its steps, for experimental/; not part of the API."""

import warnings
from dataclasses import dataclass, field

import numpy as np

from ._guard import errors_kernel, max_error_kernel
from ._noise import noise_kernel
from ._rank import rank_kernel, refine_kernel
from ._refit import refit_kernel

ERROR_PER_POINT = 0.0015
"""The default exchange rate of error for points: about 0.6% of the points of 1 s scans in 1 h blocks of
slow process data, 1% without the noise floor (report/prune.html). It is per block: for blocks H times as
long, start from error_per_point / H. That keeps fewer points per hour, not the same density (about a third as many in
12 h blocks as in 1 h blocks on the report's corpus): a longer block has a wider range, which the error
is relative to, and it needs one set of required points instead of H."""

NOISE_MIN_POINTS = 64
"""max_error's warning that it is below the noise needs this many points for the noise estimate."""


@dataclass(frozen=True, eq=False)
class Pruned:
    """What prune() kept, the errors it leaves, and which of the asks were met."""

    times: np.ndarray
    """The kept points' times, sorted, in the input's dtype."""
    values: np.ndarray
    """Their values: the samples, or with refit=True the refit ones (the first, last, minimum and
    maximum points and the points around gaps keep their samples)."""
    noise: float
    """The input's white-noise σ, estimated (value units): the rmse pass's noise floor."""
    rmse: float
    """The time-weighted RMSE of the kept points' lines against the input's, in value units."""
    max_error: float
    """The largest distance, in value, of any input point from the kept points' lines."""
    rmse_met: bool
    """The rmse pass kept every point error_per_point asked for: max_points didn't cut it."""
    max_error_met: bool
    """No input point strays further than max_error from the kept lines (True without max_error)."""
    max_points_met: bool
    """At most max_points were kept: False only when the points around gaps alone exceed it."""

    def __len__(self):
        return len(self.times)


@dataclass(frozen=True, eq=False)
class Ranking:
    """Every point of a series ordered by significance, and the error of keeping each count.

    Indices are into the input. The inside of a run of non-finite values (all but its first point) is
    never kept, since a kept non-finite value holds until the next kept point: `n` counts the points
    that can be. Keeping k points keeps the `required` ones (first, last, minimum, maximum, and the
    first point of each gap with the finite points just before and after it) and the
    k - len(required) last removed; a smaller set is always a subset of a larger one.

    Errors here are time-weighted RMSE of the kept points' lines against the input's, relative to the
    range of the finite values (max - min), as the error_per_point rule reads them:
    `relative_rmse(k)`, `relative_error(keep)`. prune() reports them in value units.
    """

    n: int
    gaps: int
    """How many of the required points are there for gaps (beyond first, last, minimum and maximum)."""
    _required: np.ndarray = field(repr=False)  # indices into the keepable points, as are the next two
    _removed: np.ndarray = field(repr=False)
    _sq_error: np.ndarray = field(repr=False)  # [r]: relative mean squared error after r removals
    _tau: np.ndarray = field(repr=False)  # empty when no removal costs anything
    _y: np.ndarray = field(repr=False)  # relative to the range; non-finite values as they are
    _live: np.ndarray = field(repr=False)  # the keepable points' indices into the input
    _span: float = field(repr=False)
    _noise: float = field(default=0.0, repr=False)  # the white-noise σ, relative to the range

    @property
    def noise(self) -> float:
        """The input's white-noise σ, estimated (value units): each point's distance from its
        neighbours' chord, its median scaled for normal noise (_noise.py)."""
        return self._noise * self._span

    @property
    def min_points(self) -> int:
        return len(self._required)

    @property
    def required(self):
        """Indices always kept, sorted."""
        return self._live[self._required]

    @property
    def removed(self):
        """The other keepable indices in removal order: least significant first."""
        return self._live[self._removed]

    def relative_rmse(self, k=None):
        """Relative RMSE of keeping k points (min_points <= k <= n), or of every count as an array
        indexed by k (nan below min_points)."""
        if k is not None:
            return float(np.sqrt(max(self._sq_error[self.n - self._clip(k)], 0.0)))
        e = np.sqrt(np.maximum(self._sq_error, 0.0))
        out = np.full(self.n + 1, np.nan)
        out[self.min_points:] = e[::-1]
        return out

    def top(self, k):
        """Sorted indices of the k most significant points (k is clipped to [min_points, n])."""
        return self._live[self._top(k)]

    def count(self, error_per_point=ERROR_PER_POINT, noise_floor=True):
        """The first count k where the relative RMSE falls below error_per_point times the points added
        beyond the required ones: where the error left and the points spent, at that exchange rate,
        meet. More points for a complex block, fewer for a simple one, sublinearly; it hardly depends
        on the sampling rate (report/prune.html, "amortized"). At 0 points are free: every point (n);
        at inf none is worth its price: the required points alone (min_points).

        noise_floor: read the RMSE as √(e² - σ²), σ the estimated white noise (`noise`): an estimate of
        the error against the signal under the noise, so points that only trace the noise aren't
        bought. Lines through noisy samples leave about σ against them however many are kept, until
        nearly all are."""
        if not error_per_point >= 0:
            raise ValueError("error_per_point must be non-negative (0 to inf)")
        e = self.relative_rmse()[self.min_points:]
        if noise_floor and self._noise > 0:
            e = np.sqrt(np.maximum(e * e - self._noise * self._noise, 0.0))
        ok = np.flatnonzero(e < error_per_point * np.arange(1, len(e) + 1))
        return self.min_points + int(ok[0]) if len(ok) else self.n

    def refine(self, keep, passes=2):
        """keep (sorted indices of at least the required points) with each other point moved to where
        it fits best between its neighbours, `passes` times over: about 10-30% lower error, no longer
        nested. O(n) per pass."""
        return self._live[self._refine(self._local(keep), passes)]

    def relative_error(self, keep):
        """Relative time-weighted RMSE of any sorted subset that includes the required points."""
        return self.errors(keep)[0]

    def errors(self, keep, values=None):
        """(time-weighted RMSE, largest deviation of any point) of the kept points' lines, at `values`
        (from refit()) or their samples, relative to the range."""
        keep = self._local(keep)
        if not len(self._tau):
            return 0.0, 0.0
        sq, worst = errors_kernel(self._tau, self._y, keep, self._y[keep] if values is None else values)
        return float(np.sqrt(sq)), float(worst)

    def select(self, error_per_point=ERROR_PER_POINT, max_points=None, max_error=None, noise_floor=True):
        """(the sorted indices prune() keeps, whether the rmse pass kept all it asked for)."""
        keep, rmse_met = self._select(error_per_point, max_points, max_error, noise_floor)
        return self._live[keep], rmse_met

    def refit(self, keep):
        """Values (relative to the range, from the minimum) at the kept points whose lines fit the
        input best in time-weighted L2, the required points keeping their samples. O(n)."""
        keep = self._local(keep)
        if not len(self._tau):
            return self._y[keep]
        fixed = np.zeros(self.n, np.bool_)
        fixed[self._required] = True
        return refit_kernel(self._tau, self._y, keep, fixed)

    # the same in indices into the keepable points

    def _local(self, keep):
        return np.searchsorted(self._live, np.asarray(keep, np.int64))

    def _clip(self, k):
        return int(min(max(int(k), self.min_points), self.n))

    def _top(self, k):
        mask = np.ones(self.n, bool)
        mask[self._removed[: self.n - self._clip(k)]] = False
        return np.flatnonzero(mask)

    def _refine(self, keep, passes=2):
        keep = np.array(keep, dtype=np.int64)
        if len(self._tau):
            fixed = np.zeros(self.n, np.bool_)
            fixed[self._required] = True
            refine_kernel(self._tau, self._y, keep, fixed, passes)
        return keep

    def _select(self, error_per_point, max_points, max_error, noise_floor=True):
        cap = self.n if max_points is None else int(max_points)
        wanted = self.count(error_per_point, noise_floor)
        if self.min_points > cap:  # the gaps alone exceed it: they win, and nothing is added
            return self._required, wanted == self.min_points
        k = min(wanted, cap)
        if max_error is None or not len(self._tau):
            return self._refine(self._top(k)), k == wanted
        tol = max_error / self._span

        def build(k):
            return max_error_kernel(self._tau, self._y, self._refine(self._top(k)), tol)

        # 1. can max_error be met within the cap at all, from the required points alone?
        keep = build(self.min_points)
        if len(keep) > cap:
            return self._relax(tol, cap), wanted == self.min_points
        # 2. it can: the rmse pass takes the rest, its least significant points giving way first
        full = build(k)
        if len(full) <= cap:
            return full, k == wanted
        lo, hi = self.min_points, k
        while hi - lo > 1:
            mid = (lo + hi) // 2
            mid_keep = build(mid)
            if len(mid_keep) <= cap:
                lo, keep = mid, mid_keep
            else:
                hi = mid
        return keep, False

    def _relax(self, tol, cap):
        """The max_error pass from the required points at about the smallest tolerance above tol that
        fits in cap points (a bisection to 0.1%; its count isn't quite monotone in the tolerance): about
        the tightest bound on every point's error the cap allows."""
        base = self._required
        hi = errors_kernel(self._tau, self._y, base, self._y[base])[1]  # the pass adds nothing at this bound
        best = base
        lo = tol if tol > 0 else hi * 1e-12
        while hi > lo * 1.001:
            mid = np.sqrt(lo * hi)
            keep = max_error_kernel(self._tau, self._y, base, mid)
            if len(keep) <= cap:
                hi, best = mid, keep
            else:
                lo = mid
        return best


def _offsets(times, n):
    """Times as float64 offsets from the first, validated."""
    t = np.asarray(times)
    if t.shape != (n,):
        raise ValueError("times and values must be 1-D arrays of the same length")
    if np.issubdtype(t.dtype, np.datetime64) or np.issubdtype(t.dtype, np.timedelta64):
        t = t.view(np.int64)
    if n and np.issubdtype(t.dtype, np.integer):
        dt = np.diff(t)
        if np.any(dt < 0):
            raise ValueError("times must be non-decreasing")
        return np.concatenate(([0.0], np.cumsum(dt, dtype=np.float64)))  # exact steps, no int64 overflow
    tau = np.asarray(t, dtype=np.float64)
    if n and (not np.all(np.isfinite(tau)) or np.any(np.diff(tau) < 0)):
        raise ValueError("times must be finite and non-decreasing")
    return tau - tau[0] if n else tau


def rank(times, values):
    """Rank the points (times[i], values[i]) of a series read as straight lines between them.

    times: non-decreasing, datetime64 or numbers (repeated times are allowed: a vertical step).
    values: floats, same length. A run of non-finite values is a gap: its first point and the finite
    points around it are kept, and a kept non-finite value holds until the next kept point.
    """
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("times and values must be 1-D arrays of the same length")
    n = len(x)
    tau = _offsets(times, n)

    finite = np.isfinite(x)
    base = np.zeros(n, bool)  # first, last, minimum, maximum
    if n:
        base[[0, n - 1]] = True
    lo = hi = 0.0
    every = bool(finite.all())
    if every:  # no gaps, the usual case: none of their masks
        live = np.arange(n)
        if n:
            low, high = int(np.argmin(x)), int(np.argmax(x))
            base[low] = base[high] = True
            lo, hi = x[low], x[high]
        req = base
        gaps = 0
    else:
        kind = np.where(finite, 0, np.where(np.isnan(x), 1, np.where(x > 0, 2, 3)))
        start = ~finite
        start[1:] &= kind[1:] != kind[:-1]  # each gap's first point, and where it changes between nan and ±inf
        near = np.zeros(n, bool)  # the finite points just before and after a gap
        near[:-1] |= finite[:-1] & ~finite[1:]
        near[1:] |= finite[1:] & ~finite[:-1]
        if finite.any():
            fi = np.flatnonzero(finite)
            base[fi[np.argmin(x[fi])]] = base[fi[np.argmax(x[fi])]] = True
            lo, hi = x[fi].min(), x[fi].max()
        required = base | start | near
        live = np.flatnonzero(finite | start)  # the inside of a gap is never kept
        req = required[live]
        gaps = int(np.count_nonzero(required & ~base))

    m = len(live)
    duration = tau[live[-1]] - tau[live[0]] if m else 0.0
    if m < 3 or duration <= 0 or hi <= lo:
        # nothing to weigh by (all at one time) or nothing to lose (constant): any removal is free
        rest = np.flatnonzero(~req)
        return Ranking(m, gaps, np.flatnonzero(req), rest, np.zeros(len(rest) + 1), np.empty(0),
                       np.zeros(m), live, 0.0)
    y = (x[live] - lo) / (hi - lo)
    t = (tau[live] - tau[live[0]]) / duration
    order, delta = rank_kernel(t, y if every else np.where(np.isfinite(y), y, 0.0), ~req)
    return Ranking(m, gaps, np.flatnonzero(req), order, np.concatenate(([0.0], np.cumsum(delta))), t, y, live,
                   float(hi - lo), float(noise_kernel(t, y)))


def prune(times, values, error_per_point=ERROR_PER_POINT, max_points=None, max_error=None, refit=False,
          noise_floor=True):
    """The points to keep of a series read as straight lines between them: a Pruned with their times and
    values, the RMS and largest error they leave, and which of the asks were met.

    Two passes choose them. The rmse pass ranks every point by how much it lowers the time-weighted RMSE
    and keeps as many as error_per_point asks for, above the noise floor; the max_error pass then adds points wherever a kept
    line strays further than max_error from the input.

    error_per_point: the rmse pass keeps the first count where the time-weighted RMSE left, relative to
        the block's range (max - min), falls below this many times the points added. 0 keeps every point
        (with max_points: the best max_points); inf none beyond the required ones (with max_error: the
        max_error pass alone). It is per block: for blocks H times as long, start from
        error_per_point / H (ERROR_PER_POINT has how far that is from the same density).
    noise_floor: the rmse pass reads its RMSE above the input's white noise, √(rmse² - σ²) with σ
        estimated from each point's distance to its neighbours' chord (`noise` of the result), so it
        doesn't keep points that only trace the noise: a quiet block of noise keeps about a tenth as
        many. Smooth and quantized signals hardly change (σ is about 0). Narrow spikes on the noise
        are cheap in a time-weighted RMSE either way: max_error (a few σ) keeps them.
    max_points: a hard cap on the points kept (at least 4: the first, last, minimum and maximum).
    max_error: no input point strays further than this, in value units, from the kept lines: each line
        that strays further is split by farthest reach (from each kept point, the line to the farthest
        point it can reach within max_error; 4% more points than the fewest on slow process data). If
        that exceeds max_points, the rmse pass gives way, least significant points first, while
        max_error still holds; if max_error can't be met within max_points even without the rmse pass,
        it is relaxed to the smallest bound the cap allows (`max_error` of the result is the bound
        reached). A max_error below the noise's σ keeps about half the points or more, and warns.
        fluxprune/README.md, "How the guards work", has the details and their costs.
    refit: experimental: instead of the samples, values at the kept times that fit the input best in
        time-weighted L2 (lower RMSE, mostly on smooth or step-like signals); the first, last, minimum
        and maximum points and the points around gaps keep their samples. max_error isn't checked
        again: `max_error_met` says whether it still holds.

    The first, last, minimum and maximum points are always kept. A run of non-finite values (a gap)
    keeps its first point and the finite points just before and after it, past max_points if need be
    (with a warning, and `max_points_met` False); a kept non-finite value holds until the next kept
    point.
    """
    t = np.asarray(times)
    x = np.asarray(values, dtype=np.float64)
    r = rank(t, x)
    if max_points is not None and max_points < min(4, len(x)):
        raise ValueError("max_points must be at least 4 (the first, last, minimum and maximum points)")
    if max_error is not None and not (np.isfinite(max_error) and max_error >= 0):
        raise ValueError("max_error must be finite and non-negative")
    keep, rmse_met = r._select(error_per_point, max_points, max_error, noise_floor)
    idx = r._live[keep]
    out = x[idx].copy()
    rmse = worst = 0.0
    if len(r._tau):
        c = r._y[keep]
        if refit:
            fixed = np.zeros(r.n, np.bool_)
            fixed[r._required] = True
            c = refit_kernel(r._tau, r._y, keep, fixed)
            free = ~fixed[keep]
            out[free] = np.min(x[np.isfinite(x)]) + c[free] * r._span
        sq, worst = errors_kernel(r._tau, r._y, keep, c)
        rmse, worst = np.sqrt(sq) * r._span, worst * r._span
    max_error_met = max_error is None or worst <= max_error * (1 + 1e-12)
    max_points_met = max_points is None or len(idx) <= max_points
    if not max_points_met:
        warnings.warn(f"prune: {len(idx)} points kept, over max_points={max_points}: {r.gaps} of them are "
                      "there for gaps (non-finite values)", stacklevel=2)
    if max_error is not None and r._span > 0 and r.n >= NOISE_MIN_POINTS:
        sigma = r.noise
        if max_error < sigma:
            warnings.warn(f"prune: max_error={max_error:g} is below the noise (σ ≈ {sigma:.3g}): the max_error "
                          "pass keeps about half the points or more", stacklevel=2)
    return Pruned(t[idx], out, r.noise, float(rmse), float(worst), rmse_met, max_error_met, max_points_met)
