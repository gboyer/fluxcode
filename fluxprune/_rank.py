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

A segment's moments and its error sit in one row (one cache line), and the candidates wait in a
4-ary heap that holds their keys inline: half the time of separate arrays and a binary heap of
indices at 3,600 points, two thirds at a million.
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


# the columns of a segment's row
_L, _SU, _SW, _SUU, _SWW, _SUW, _ERR = range(7)
_D = 4  # the heap's arity


@njit(cache=True, inline="always")
def _less(ka, ia, kb, ib):
    """(key, index) order: the lower key, then the lower index."""
    return ka < kb or (ka == kb and ia < ib)


@njit(cache=True, inline="always")
def _sift_up(hk, hi, pos, k, ck, ci):
    """Place the entry (ck, ci) at or above slot k."""
    while k > 0:
        par = (k - 1) // _D
        if not _less(ck, ci, hk[par], hi[par]):
            break
        hk[k] = hk[par]
        hi[k] = hi[par]
        pos[hi[k]] = k
        k = par
    hk[k] = ck
    hi[k] = ci
    pos[ci] = k


@njit(cache=True, inline="always")
def _sift_down(hk, hi, pos, k, ck, ci, size):
    """Place the entry (ck, ci) at or below slot k."""
    while True:
        c = _D * k + 1
        if c >= size:
            break
        best = c
        for e in range(c + 1, min(c + _D, size)):
            if _less(hk[e], hi[e], hk[best], hi[best]):
                best = e
        if not _less(hk[best], hi[best], ck, ci):
            break
        hk[k] = hk[best]
        hi[k] = hi[best]
        pos[hi[k]] = k
        k = best
    hk[k] = ck
    hi[k] = ci
    pos[ci] = k


@njit(cache=True, inline="always")
def _merged_error(tau, y, M, p, i, q):
    """Error of the chord p -> q over the segments p -> i and i -> q merged (without merging them)."""
    a = tau[i] - tau[p]
    b = y[i] - y[p]
    suu = M[p, _SUU] + M[i, _SUU] + 2.0 * a * M[i, _SU] + a * a * M[i, _L]
    sww = M[p, _SWW] + M[i, _SWW] + 2.0 * b * M[i, _SW] + b * b * M[i, _L]
    suw = M[p, _SUW] + M[i, _SUW] + a * M[i, _SW] + b * M[i, _SU] + a * b * M[i, _L]
    return _chord_error(M[p, _L] + M[i, _L], suu, sww, suw, tau[q] - tau[p], y[q] - y[p])


@njit(cache=True)
def rank_kernel(tau, y, removable):
    """Remove the removable points one at a time, always the one whose removal adds the least
    error (ties: the lower index). Returns the points in removal order and what each removal
    added to the integrated squared error (it can be negative: a chord can fit better than the
    two it replaces). tau must be non-decreasing; y finite wherever a removable point or its
    neighbours are."""
    n = len(tau)
    M = np.zeros((n, 8))  # each current segment's row, indexed by its left end (the last column pads)
    prv = np.empty(n, np.int64)
    nxt = np.empty(n, np.int64)
    for j in range(n):
        prv[j] = j - 1
        nxt[j] = j + 1
    for j in range(n - 1):
        h = tau[j + 1] - tau[j]
        d = y[j + 1] - y[j]
        M[j, _L] = h
        M[j, _SU] = h * h / 2.0
        M[j, _SW] = h * d / 2.0
        M[j, _SUU] = h * h * h / 3.0
        M[j, _SWW] = h * d * d / 3.0
        M[j, _SUW] = h * h * d / 3.0

    hk = np.empty(n)  # the heap: keys and indices side by side
    hi = np.empty(n, np.int64)
    pos = np.full(n, -1, np.int64)  # each candidate's slot in it
    size = 0
    for i in range(1, n - 1):
        if removable[i]:
            hk[size] = _merged_error(tau, y, M, i - 1, i, i + 1) - M[i - 1, _ERR] - M[i, _ERR]
            hi[size] = i
            pos[i] = size
            size += 1
    for k in range((size - 2) // _D, -1, -1):
        _sift_down(hk, hi, pos, k, hk[k], hi[k], size)

    order = np.empty(size, np.int64)
    delta = np.empty(size)
    for r in range(size):
        i = hi[0]
        order[r] = i
        delta[r] = hk[0]
        pos[i] = -1
        size -= 1
        if size > 0:
            _sift_down(hk, hi, pos, 0, hk[size], hi[size], size)
        p, q = prv[i], nxt[i]
        # merge segment i into segment p
        a = tau[i] - tau[p]
        b = y[i] - y[p]
        Li, Sui, Swi = M[i, _L], M[i, _SU], M[i, _SW]
        M[p, _SUU] += M[i, _SUU] + 2.0 * a * Sui + a * a * Li
        M[p, _SWW] += M[i, _SWW] + 2.0 * b * Swi + b * b * Li
        M[p, _SUW] += M[i, _SUW] + a * Swi + b * Sui + a * b * Li
        M[p, _SU] += Sui + a * Li
        M[p, _SW] += Swi + b * Li
        M[p, _L] += Li
        M[p, _ERR] = _chord_error(M[p, _L], M[p, _SUU], M[p, _SWW], M[p, _SUW], tau[q] - tau[p], y[q] - y[p])
        nxt[p] = q
        prv[q] = p
        for j in (p, q):
            k = pos[j]
            if k >= 0:
                key = _merged_error(tau, y, M, prv[j], j, nxt[j]) - M[prv[j], _ERR] - M[j, _ERR]
                if _less(key, j, hk[k], j):
                    _sift_up(hk, hi, pos, k, key, j)
                else:
                    _sift_down(hk, hi, pos, k, key, j, size)
    return order, delta


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
