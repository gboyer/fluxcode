# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Kernels for the error of a kept subset and for the max_error pass (farthest reach).

The deviation of an input point is its distance, in value, from the straight line between the
kept points around it. Where kept points share a time (a vertical step), the line is the step,
and a point at that time deviates by its distance from the step's span of values. A segment with
a non-finite end is a forward hold with no input point inside it, so it has no error.
"""

import numpy as np
from numba import njit


@njit(cache=True, inline="always")
def _deviation(tau, y, a, b, ya, yb, j):
    """Distance of point j from the line (tau[a], ya) -> (tau[b], yb)."""
    du = tau[b] - tau[a]
    if du <= 0.0:
        lo, hi = min(ya, yb), max(ya, yb)
        return max(0.0, y[j] - hi, lo - y[j])
    return abs(y[j] - (ya + (yb - ya) * ((tau[j] - tau[a]) / du)))


@njit(cache=True)
def _worst(tau, y, a, b):
    """The largest deviation of an input point between kept points a and b from their chord."""
    best = 0.0
    for j in range(a + 1, b):
        best = max(best, _deviation(tau, y, a, b, y[a], y[b], j))
    return best


@njit(cache=True)
def errors_kernel(tau, y, keep, c):
    """(time-weighted mean squared error, largest deviation of any point) of the lines through the kept
    points at values c (y[keep], or refit ones) against the input's, tau running from 0 to 1. O(n)."""
    sq = 0.0
    worst = 0.0
    for s in range(len(keep) - 1):
        a, b = keep[s], keep[s + 1]
        ya, yb = c[s], c[s + 1]
        if not (np.isfinite(ya) and np.isfinite(yb)):
            continue
        du = tau[b] - tau[a]
        slope = (yb - ya) / du if du > 0.0 else 0.0
        d0 = 0.0
        for j in range(a, b + 1):
            d1 = y[j] - ya - slope * (tau[j] - tau[a]) if du > 0.0 else 0.0
            if j > a:
                sq += (tau[j] - tau[j - 1]) * (d0 * d0 + d0 * d1 + d1 * d1) / 3.0
            if a < j < b:
                worst = max(worst, _deviation(tau, y, a, b, ya, yb, j))
            d0 = d1
    return max(sq, 0.0), worst


@njit(cache=True)
def _reach(tau, y, a, b, tol, out, m):
    """Farthest reach between kept points a and b: from each anchor, scan on while some line from the
    anchor could still pass within tol of every point so far (the slope bounds each point sets, as in
    swinging door, still overlap), remember the last point whose own chord is such a line, and make it
    the next anchor. Writes the anchors after a to out[m:]; returns the new m. A point at the anchor's
    time (a vertical step) is always the next anchor."""
    anchor = a
    while True:
        lo, hi = -np.inf, np.inf
        last = anchor + 1  # the next point is always reachable: no point lies between
        j = anchor + 1
        while j <= b:
            du = tau[j] - tau[anchor]
            if du <= 0.0:  # only the next point can share the anchor's time
                break
            s = (y[j] - y[anchor]) / du
            if lo <= s <= hi:
                last = j
            lo = max(lo, (y[j] - tol - y[anchor]) / du)
            hi = min(hi, (y[j] + tol - y[anchor]) / du)
            if lo > hi:  # no line from the anchor passes every point so far: none reaches further
                break
            j += 1
        if last >= b:
            return m
        out[m] = last
        m += 1
        anchor = last


@njit(cache=True)
def max_error_kernel(tau, y, keep, tol):
    """The max_error pass: keep (sorted) with every segment that strays more than tol from the input
    split by farthest reach (_reach), so that no input point does. On slow process data it keeps 4%
    more points than the fewest that would (up to 20% on noisy signals; report/prune.html, "The
    max_error pass"), and 1-3x fewer than swinging door's rule, which ends each line at the first
    point it can't reach although a later one often can be."""
    out = np.empty(len(tau), np.int64)
    m = 0
    for s in range(len(keep) - 1):
        a, b = keep[s], keep[s + 1]
        out[m] = a
        m += 1
        if np.isfinite(y[a]) and np.isfinite(y[b]) and _worst(tau, y, a, b) > tol:
            m = _reach(tau, y, a, b, tol, out, m)
    out[m] = keep[-1]
    return out[: m + 1].copy()
