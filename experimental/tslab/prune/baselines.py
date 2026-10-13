# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""What fluxprune's bottom-up ranking is compared with, each picking k of n points (sorted indices):

- top_down: greedy splitting, the reverse direction: start from the required points and add the
  point whose insertion lowers the time-weighted squared error most (O(n log n) on balanced splits,
  O(n k) at worst).
- optimal_with: the k points with the least error that include the required ones, by dynamic
  programming (O(n² k): small n only).
- sdt: swinging door with CompDev searched so it keeps k points (or the closest count above).
- lttb: Largest-Triangle-Three-Buckets (Steinarsson 2013), the plotting downsampler.
- minmax: the min and max of each of k/2 equal-time buckets (M4 without first/last), plus the ends.
- uniform: k points evenly spaced in index.
- visvalingam_whyatt: bottom-up by the area of the triangle each point makes with its neighbours (Visvalingam
  and Whyatt 1993, the cartographers' line simplification), in coordinates relative to the block's duration and range.

and `exact_rmse`: the relative time-weighted RMSE of any subset, by brute force.
"""

import heapq

import numpy as np
from fluxprune._guard import _worst
from fluxprune._rank import chord_errors
from numba import njit

from tslab.common.historian import swinging_door


def normalize(t, x):
    t = np.asarray(t)
    tau = (t - t[0]).astype(np.float64)
    tau /= tau[-1]
    lo, hi = x.min(), x.max()
    return tau, (x - lo) / (hi - lo if hi > lo else 1.0)


def exact_rmse(t, x, keep, max_err=False):
    """Time-weighted RMSE (and optionally the max abs error) of the straight lines through `keep`
    against those through every point, relative to max - min."""
    tau, y = normalize(t, x)
    g = np.interp(tau, tau[keep], y[keep])
    d = y - g
    h = np.diff(tau)
    rmse = np.sqrt(max(np.sum(h * (d[:-1] ** 2 + d[:-1] * d[1:] + d[1:] ** 2)) / 3, 0.0))
    return (rmse, np.abs(d).max()) if max_err else rmse


@njit(cache=True)
def _chord_errors_from(tau, y, a, b, forward):
    """Errors of the chords a -> m for every m in (a, b] (forward), or m -> b for m in [a, b) (backward),
    indexed by m - a: fluxprune's own kernel."""
    out = np.zeros(b - a + 1)
    chord_errors(tau, y, a, b, out, forward)
    return out


def _best_split(tau, y, a, b):
    left = _chord_errors_from(tau, y, a, b, True)
    right = _chord_errors_from(tau, y, a, b, False)
    whole = left[-1]
    split = left[1:-1] + right[1:-1]
    m = int(np.argmin(split))
    return whole - split[m], a + 1 + m


def top_down(t, x, k, required):
    tau, y = normalize(t, x)
    keep = sorted(set(int(i) for i in required))
    heap = []
    for a, b in zip(keep, keep[1:]):
        if b - a > 1:
            gain, m = _best_split(tau, y, a, b)
            heapq.heappush(heap, (-gain, a, b, m))
    keep = set(keep)
    while len(keep) < k and heap:
        _, a, b, m = heapq.heappop(heap)
        keep.add(m)
        for lo, hi in ((a, m), (m, b)):
            if hi - lo > 1:
                gain, mm = _best_split(tau, y, lo, hi)
                heapq.heappush(heap, (-gain, lo, hi, mm))
    return np.array(sorted(keep))


def sdt(t, x, k):
    """Swinging door (no exception filter, no CompMax) with the CompDev, found by bisection, that keeps
    the fewest points not below k."""
    tau = (np.asarray(t) - t[0]).astype(np.float64)
    span = x.max() - x.min() or 1.0
    lo, hi = 0.0, span
    best = np.arange(len(x))
    for _ in range(50):
        dev = (lo + hi) / 2
        idx = swinging_door(tau, x, dev * 1.0, np.inf)
        if len(idx) >= k:
            best, lo = idx, dev
        else:
            hi = dev
        if len(best) == k:
            break
    return best


def lttb(t, x, k):
    n = len(x)
    if k >= n or k < 3:
        return np.arange(n) if k >= n else np.array([0, n - 1])
    tau = (np.asarray(t) - t[0]).astype(np.float64)
    edges = np.linspace(1, n - 1, k - 1).astype(np.int64)
    out = [0]
    a = 0
    for j in range(k - 2):
        lo, hi = edges[j], edges[j + 1]
        nlo, nhi = edges[j + 1], (edges[j + 2] if j + 2 < k - 1 else n)
        ct, cx = tau[nlo:nhi].mean(), x[nlo:nhi].mean()
        area = np.abs((tau[a] - ct) * (x[lo:hi] - x[a]) - (tau[a] - tau[lo:hi]) * (cx - x[a]))
        a = lo + int(np.argmax(area))
        out.append(a)
    out.append(n - 1)
    return np.array(out)


def minmax(t, x, k):
    n = len(x)
    tau = (np.asarray(t) - t[0]).astype(np.float64)
    buckets = max((k - 2) // 2, 1)
    which = np.minimum((tau[1:-1] / tau[-1] * buckets).astype(np.int64), buckets - 1)
    keep = {0, n - 1}
    starts = np.searchsorted(which, np.arange(buckets))
    ends = np.append(starts[1:], len(which))
    for s, e in zip(starts, ends):
        if e > s:
            keep.add(1 + s + int(np.argmin(x[1 + s:1 + e])))
            keep.add(1 + s + int(np.argmax(x[1 + s:1 + e])))
    return np.array(sorted(keep))


def uniform(t, x, k):
    return np.unique(np.linspace(0, len(x) - 1, k).round().astype(np.int64))


def visvalingam_whyatt(t, x, k, required):
    """The k points left after removing, one at a time, the point whose triangle with its neighbours has the
    least area (each neighbour's area raised to at least the last one removed, as the original does); the
    required points stay."""
    tau, y = normalize(t, x)
    n = len(y)
    prv, nxt = list(range(-1, n - 1)), list(range(1, n + 1))
    alive = np.ones(n, bool)
    fixed = np.zeros(n, bool)
    fixed[required] = True

    def area(i):
        p, q = prv[i], nxt[i]
        return abs((tau[i] - tau[p]) * (y[q] - y[p]) - (tau[q] - tau[p]) * (y[i] - y[p])) / 2

    cur = {i: area(i) for i in range(1, n - 1) if not fixed[i]}
    heap = [(a, i) for i, a in cur.items()]
    heapq.heapify(heap)
    left = n
    while left > k and heap:
        a, i = heapq.heappop(heap)
        if not alive[i] or cur[i] != a:
            continue
        alive[i] = False
        left -= 1
        p, q = prv[i], nxt[i]
        nxt[p], prv[q] = q, p
        for j in (p, q):
            if j in cur:
                cur[j] = max(area(j), a)
                heapq.heappush(heap, (cur[j], j))
    return np.flatnonzero(alive)


def optimal_with(t, x, k, required):
    """The k points with the least error that include `required` (O(n² k))."""
    tau, y = normalize(t, x)
    req = np.zeros(len(x), np.bool_)
    req[required] = True
    return _optimal_with(tau, y, k, req)


@njit(cache=True)
def _optimal_with(tau, y, k, req):
    n = len(tau)
    cost = np.full((n, n), np.inf)
    for a in range(n - 1):
        e = _chord_errors_from(tau, y, a, n - 1, True)
        for b in range(a + 1, n):
            cost[a, b] = e[b - a]
            if req[b]:
                break  # no chord may skip a required point
    best = np.full((k, n), np.inf)
    arg = np.zeros((k, n), np.int64)
    best[0, 0] = 0.0
    for j in range(1, k):
        for b in range(1, n):
            for a in range(b):
                c = best[j - 1, a] + cost[a, b]
                if c < best[j, b]:
                    best[j, b] = c
                    arg[j, b] = a
    out = np.empty(k, np.int64)
    out[k - 1] = n - 1
    for j in range(k - 1, 0, -1):
        out[j - 1] = arg[j, out[j]]
    return out



# ---------------------------------------------------------------------------
# The max_error pass's alternatives, against fluxprune's farthest reach (fluxprune._guard.max_error_kernel)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _door_segment(tau, y, a, b, tol, out, m):
    """Swinging door's rule: each line ends at the first point the chord from the anchor can't reach."""
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
            anchor, lo, hi = j, -np.inf, np.inf
            j += 1
            continue
        s = (y[j] - y[anchor]) / du
        if s < lo or s > hi:
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
def _fewest_segment(tau, y, a, b, tol, out, m):
    """The fewest points between a and b whose lines stay within tol: BFS over every feasible chord, O(len²)."""
    L = b - a + 1
    dist = np.full(L, L + 1, np.int64)
    par = np.full(L, -1, np.int64)
    dist[0] = 0
    for i in range(L):
        p = a + i
        lo, hi = -np.inf, np.inf
        for q in range(p + 1, b + 1):
            du = tau[q] - tau[p]
            if du <= 0.0:
                if dist[i] + 1 < dist[q - a]:
                    dist[q - a], par[q - a] = dist[i] + 1, i
                break
            s = (y[q] - y[p]) / du
            if lo <= s <= hi and dist[i] + 1 < dist[q - a]:
                dist[q - a], par[q - a] = dist[i] + 1, i
            lo = max(lo, (y[q] - tol - y[p]) / du)
            hi = min(hi, (y[q] + tol - y[p]) / du)
            if lo > hi:
                break
    path = np.empty(L, np.int64)
    c = 0
    i = par[L - 1]
    while i > 0:
        path[c] = a + i
        c += 1
        i = par[i]
    for s in range(c - 1, -1, -1):
        out[m] = path[s]
        m += 1
    return m


@njit(cache=True)
def other_max_error_pass(tau, y, keep, tol, fewest):
    """keep with each segment that strays more than tol split by swinging door's rule, or (fewest) by the fewest
    points: what fluxprune's farthest reach is measured against."""
    out = np.empty(len(tau), np.int64)
    m = 0
    for s in range(len(keep) - 1):
        a, b = keep[s], keep[s + 1]
        out[m] = a
        m += 1
        if _worst(tau, y, a, b) > tol:
            m = _fewest_segment(tau, y, a, b, tol, out, m) if fewest else _door_segment(tau, y, a, b, tol, out, m)
    out[m] = keep[-1]
    return out[: m + 1].copy()
