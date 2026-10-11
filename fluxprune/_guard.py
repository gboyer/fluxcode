# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Kernels for the error of a kept subset and for the max_error guard.

The deviation of an input point is its distance, in value, from the straight line between the
kept points around it. Where kept points share a time (a vertical step), the line is the step,
and a point at that time deviates by its distance from the step's span of values.
"""

import heapq

import numpy as np
from numba import njit


@njit(cache=True, inline="always")
def _deviation(tau, y, a, b, j):
    du = tau[b] - tau[a]
    if du <= 0.0:
        lo, hi = min(y[a], y[b]), max(y[a], y[b])
        return max(0.0, y[j] - hi, lo - y[j])
    return abs(y[j] - (y[a] + (y[b] - y[a]) * ((tau[j] - tau[a]) / du)))


@njit(cache=True)
def _worst(tau, y, a, b):
    """(deviation, index) of the input point between kept points a and b that strays furthest from
    their line; (0, -1) when there is none."""
    best, at = 0.0, -1
    for j in range(a + 1, b):
        d = _deviation(tau, y, a, b, j)
        if d > best:
            best, at = d, j
    return best, at


@njit(cache=True)
def errors_kernel(tau, y, keep):
    """(time-weighted mean squared error, largest deviation of any point) of the kept points' lines
    against the input's, tau running from 0 to 1. O(n)."""
    sq = 0.0
    worst = 0.0
    for s in range(len(keep) - 1):
        a, b = keep[s], keep[s + 1]
        du = tau[b] - tau[a]
        slope = (y[b] - y[a]) / du if du > 0.0 else 0.0
        d0 = 0.0
        for j in range(a, b + 1):
            d1 = y[j] - y[a] - slope * (tau[j] - tau[a]) if du > 0.0 else 0.0
            if j > a:
                sq += (tau[j] - tau[j - 1]) * (d0 * d0 + d0 * d1 + d1 * d1) / 3.0
            if a < j < b:
                worst = max(worst, _deviation(tau, y, a, b, j))
            d0 = d1
    return max(sq, 0.0), worst


@njit(cache=True)
def _door(tau, y, a, b, tol, out, m):
    """Swinging door between kept points a and b: from each anchor, the longest chord whose line stays
    within tol of every point it spans (slope bounds from the anchor, O(1) per point); the point before
    the first that breaks it becomes the next anchor. Writes the anchors after a to out[m:]; returns
    the new m. A point at the anchor's time ends the chord there."""
    anchor = a
    lo, hi = -np.inf, np.inf
    j = a + 1
    while j <= b:
        du = tau[j] - tau[anchor]
        if du <= 0.0:
            if j - 1 > anchor:
                out[m] = j - 1
                m += 1
            if j < b:
                out[m] = j
                m += 1
            anchor = j
            lo, hi = -np.inf, np.inf
            j += 1
            continue
        s = (y[j] - y[anchor]) / du
        if s < lo or s > hi:
            # j can't be reached from the anchor: the point before it becomes the anchor
            anchor = j - 1
            out[m] = anchor
            m += 1
            lo, hi = -np.inf, np.inf
            continue
        if j < b:
            lo = max(lo, (y[j] - tol - y[anchor]) / du)
            hi = min(hi, (y[j] + tol - y[anchor]) / du)
        j += 1
    return m


@njit(cache=True)
def fill_kernel(tau, y, keep, tol):
    """keep (sorted) with every segment that strays more than tol from the input filled by swinging door,
    so that no input point does: about the fewest points that bound the error of each segment."""
    out = np.empty(len(tau), np.int64)
    m = 0
    for s in range(len(keep) - 1):
        a, b = keep[s], keep[s + 1]
        out[m] = a
        m += 1
        if _worst(tau, y, a, b)[0] > tol:
            m = _door(tau, y, a, b, tol, out, m)
    out[m] = keep[-1]
    return out[: m + 1].copy()


@njit(cache=True)
def repair_kernel(tau, y, keep, tol, budget):
    """Add points to `keep` (sorted) until no input point deviates by more than tol from the kept
    lines, splitting the segment with the largest deviation at its worst point first, but adding at
    most `budget` points. Returns (the new sorted keep, whether every point is within tol)."""
    heap = [(0.0, 0, 0, 0)]  # (-deviation, worst point, a, b), typed by the first item
    heap.pop()
    for s in range(len(keep) - 1):
        d, j = _worst(tau, y, keep[s], keep[s + 1])
        if d > tol:
            heapq.heappush(heap, (-d, j, keep[s], keep[s + 1]))
    added = np.empty(min(budget, len(tau)), np.int64)
    count = 0
    while len(heap) and count < budget:
        _, j, a, b = heapq.heappop(heap)
        added[count] = j
        count += 1
        for p, q in ((a, j), (j, b)):
            d, m = _worst(tau, y, p, q)
            if d > tol:
                heapq.heappush(heap, (-d, m, p, q))
    out = np.concatenate((keep, added[:count]))
    out.sort()
    return out, len(heap) == 0
