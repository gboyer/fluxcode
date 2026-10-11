# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""The ranking kernel: bottom-up removal of the point that costs least, under time-weighted L2.

Both the input and any subset of it are read as straight lines between their points, so the
error of a pruned series is the integral of the squared difference of two piecewise-linear
functions, exact per input segment. Each current output segment keeps the moments of the input
over its span in coordinates local to its left end (u = t - t_left, w = y - y_left):

    L = ∫1, Su = ∫u, Sw = ∫w, Suu = ∫u², Sww = ∫w², Suw = ∫uw

so its error against the chord, E = Sww - 2 s Suw + s² Suu with s the chord's slope, and the
moments of two neighbours merged, are O(1) and lose no precision to the position in the block
(prefix sums over the whole block would: a short steep chord far from the start cancels them).
"""

import numpy as np
from numba import njit


@njit(cache=True, inline="always")
def _chord_error(L, Suu, Sww, Suw, du, dw):
    """∫(w - s u)² over a span whose chord rises dw over du; 0 for a span of no duration."""
    if du <= 0.0:
        return 0.0
    s = dw / du
    e = Sww - 2.0 * s * Suw + s * s * Suu
    return max(0.0, e)  # rounding: a straight span can come out a hair negative


@njit(cache=True)
def _sift_up(heap, pos, key, k):
    i = heap[k]
    while k > 0:
        parent = (k - 1) >> 1
        j = heap[parent]
        if key[j] < key[i] or (key[j] == key[i] and j < i):
            break
        heap[k] = j
        pos[j] = k
        k = parent
    heap[k] = i
    pos[i] = k


@njit(cache=True)
def _sift_down(heap, pos, key, k, size):
    i = heap[k]
    while True:
        c = 2 * k + 1
        if c >= size:
            break
        if c + 1 < size:
            a, b = heap[c], heap[c + 1]
            if key[b] < key[a] or (key[b] == key[a] and b < a):
                c += 1
        j = heap[c]
        if key[i] < key[j] or (key[i] == key[j] and i < j):
            break
        heap[k] = j
        pos[j] = k
        k = c
    heap[k] = i
    pos[i] = k


@njit(cache=True)
def rank_kernel(tau, y, removable):
    """Remove the removable points one at a time, always the one whose removal adds the least
    error (ties: the lower index). Returns the points in removal order and what each removal
    added to the integrated squared error (it can be negative: a chord can fit better than the
    two it replaces). tau must be non-decreasing; y finite wherever a removable point or its
    neighbours are."""
    n = len(tau)
    # moments of each current segment, indexed by its left end
    L = np.zeros(n)
    Su = np.zeros(n)
    Sw = np.zeros(n)
    Suu = np.zeros(n)
    Sww = np.zeros(n)
    Suw = np.zeros(n)
    err = np.zeros(n)
    prv = np.empty(n, np.int64)
    nxt = np.empty(n, np.int64)
    for j in range(n):
        prv[j] = j - 1
        nxt[j] = j + 1
    for j in range(n - 1):
        h = tau[j + 1] - tau[j]
        d = y[j + 1] - y[j]
        L[j] = h
        Su[j] = h * h / 2.0
        Sw[j] = h * d / 2.0
        Suu[j] = h * h * h / 3.0
        Sww[j] = h * d * d / 3.0
        Suw[j] = h * h * d / 3.0

    key = np.zeros(n)
    heap = np.empty(n, np.int64)
    pos = np.full(n, -1, np.int64)
    size = 0
    for i in range(1, n - 1):
        if removable[i]:
            p, q = i - 1, i + 1
            key[i] = _merged_error(tau, y, L, Su, Sw, Suu, Sww, Suw, p, i, q) - err[p] - err[i]
            heap[size] = i
            pos[i] = size
            size += 1
    for k in range(size // 2 - 1, -1, -1):
        _sift_down(heap, pos, key, k, size)

    order = np.empty(size, np.int64)
    delta = np.empty(size)
    for r in range(size):
        i = heap[0]
        pos[i] = -1
        size -= 1
        if size > 0:
            heap[0] = heap[size]
            pos[heap[0]] = 0
            _sift_down(heap, pos, key, 0, size)
        order[r] = i
        delta[r] = key[i]
        p, q = prv[i], nxt[i]
        # merge segment i into segment p
        a = tau[i] - tau[p]
        b = y[i] - y[p]
        Suu[p] += Suu[i] + 2.0 * a * Su[i] + a * a * L[i]
        Sww[p] += Sww[i] + 2.0 * b * Sw[i] + b * b * L[i]
        Suw[p] += Suw[i] + a * Sw[i] + b * Su[i] + a * b * L[i]
        Su[p] += Su[i] + a * L[i]
        Sw[p] += Sw[i] + b * L[i]
        L[p] += L[i]
        err[p] = _chord_error(L[p], Suu[p], Sww[p], Suw[p], tau[q] - tau[p], y[q] - y[p])
        nxt[p] = q
        prv[q] = p
        for j in (p, q):
            if pos[j] >= 0:
                old = key[j]
                key[j] = _merged_error(tau, y, L, Su, Sw, Suu, Sww, Suw, prv[j], j, nxt[j]) - err[prv[j]] - err[j]
                if key[j] < old:
                    _sift_up(heap, pos, key, pos[j])
                else:
                    _sift_down(heap, pos, key, pos[j], size)
    return order, delta


@njit(cache=True, inline="always")
def _merged_error(tau, y, L, Su, Sw, Suu, Sww, Suw, p, i, q):
    """Error of the chord p -> q over the segments p -> i and i -> q merged (without merging them)."""
    a = tau[i] - tau[p]
    b = y[i] - y[p]
    suu = Suu[p] + Suu[i] + 2.0 * a * Su[i] + a * a * L[i]
    sww = Sww[p] + Sww[i] + 2.0 * b * Sw[i] + b * b * L[i]
    suw = Suw[p] + Suw[i] + a * Sw[i] + b * Su[i] + a * b * L[i]
    return _chord_error(L[p] + L[i], suu, sww, suw, tau[q] - tau[p], y[q] - y[p])


@njit(cache=True)
def chord_errors(tau, y, a, b, out, forward):
    """out[m - a] = error of the chord a -> m for every m in (a, b] (forward), or of m -> b for every
    m in [a, b) (backward), accumulating the moments one input segment at a time."""
    L = Su = Sw = Suu = Sww = Suw = 0.0
    step = 1 if forward else -1
    anchor = a if forward else b
    m = a + 1 if forward else b - 1
    while a <= m <= b:
        last = m - step  # the segment last -> m, entered at the anchor's local coordinates
        h = abs(tau[m] - tau[last])
        u0 = abs(tau[last] - tau[anchor])
        w0 = y[last] - y[anchor]
        d = y[m] - y[last]
        L += h
        Su += h * u0 + h * h / 2.0
        Sw += h * w0 + h * d / 2.0
        Suu += h * (u0 * u0 + u0 * h + h * h / 3.0)
        Sww += h * (w0 * w0 + w0 * d + d * d / 3.0)
        Suw += h * (u0 * w0 + (u0 * d + w0 * h) / 2.0 + h * d / 3.0)
        out[m - a] = _chord_error(L, Suu, Sww, Suw, abs(tau[m] - tau[anchor]), y[m] - y[anchor])
        m += step


@njit(cache=True)
def refine_kernel(tau, y, keep, fixed, passes):
    """Move each kept point that isn't fixed to the position between its kept neighbours where the two
    chords fit best, in order, for up to `passes` passes (fewer if nothing moves). Each move can only
    lower the error. keep is sorted and modified in place; returns the number of passes made."""
    n = len(tau)
    left = np.empty(n)
    right = np.empty(n)
    for p in range(passes):
        moved = False
        for j in range(1, len(keep) - 1):
            i = keep[j]
            if fixed[i]:
                continue
            a, b = keep[j - 1], keep[j + 1]
            if b - a < 3:
                continue
            chord_errors(tau, y, a, b, left, True)
            chord_errors(tau, y, a, b, right, False)
            best = i
            best_err = left[i - a] + right[i - a]
            for m in range(a + 1, b):
                e = left[m - a] + right[m - a]
                if e < best_err:
                    best, best_err = m, e
            if best != i:
                keep[j] = best
                moved = True
        if not moved:
            return p + 1
    return passes
