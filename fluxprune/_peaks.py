# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Kernels for peaks(): local maxima, their prominence, the least distance between them, and the
alternation of peaks and troughs. Maxima only (troughs are the maxima of -y)."""

import numpy as np
from numba import njit


@njit(cache=True)
def maxima_kernel(y):
    """Mask of the local maxima: the first point of a top (flat or not) that falls on both sides."""
    n = len(y)
    out = np.zeros(n, np.bool_)
    i = 1
    while i < n - 1:
        if y[i] > y[i - 1]:
            k = i + 1
            while k < n - 1 and y[k] == y[i]:  # a flat top
                k += 1
            if y[k] < y[i]:
                out[i] = True
            i = k
        else:
            i += 1
    return out


@njit(cache=True)
def distance_kernel(tau, y, idx, half):
    """Mask over idx (sorted maxima): highest first, each kept one removes the lower ones less than
    `half` from it, so no two kept maxima are closer (ties: the earlier wins)."""
    m = len(idx)
    keep = np.ones(m, np.bool_)
    order = np.argsort(-y[idx], kind="mergesort")
    for s in range(m):
        i = order[s]
        if not keep[i]:
            continue
        j = i - 1
        while j >= 0 and tau[idx[i]] - tau[idx[j]] < half:
            keep[j] = False
            j -= 1
        j = i + 1
        while j < m and tau[idx[j]] - tau[idx[i]] < half:
            keep[j] = False
            j += 1
    return keep


@njit(cache=True)
def _bases(y, forward):
    """For each point, the lowest value between it and the nearest strictly higher point on one side
    (or the end), itself included, and whether there is such a point. A monotone stack of segment
    minima: O(n)."""
    n = len(y)
    base = np.empty(n)
    found = np.empty(n, np.bool_)
    stack = np.empty(n, np.int64)
    seg = np.empty(n)  # seg[i]: the lowest value from the stack entry below i (exclusive) to i
    top = 0
    for s in range(n):
        i = s if forward else n - 1 - s
        cur = y[i]
        while top > 0 and y[stack[top - 1]] <= y[i]:
            top -= 1
            cur = min(cur, seg[stack[top]])
        base[i] = cur
        found[i] = top > 0
        seg[i] = cur
        stack[top] = i
        top += 1
    return base, found


@njit(cache=True)
def prominence_kernel(y, idx):
    """Prominence of the points idx: how far each rises above the higher of the lowest points that
    separate it from a higher point on either side. A side with no higher point before the series' end
    doesn't count (a block's edge cuts what lies beyond); with neither, the deeper side counts."""
    left, lf = _bases(y, True)
    right, rf = _bases(y, False)
    out = np.empty(len(idx))
    for s in range(len(idx)):
        i = idx[s]
        if lf[i] and rf[i]:
            out[s] = y[i] - max(left[i], right[i])
        elif lf[i] or rf[i]:
            out[s] = y[i] - (left[i] if lf[i] else right[i])
        else:
            out[s] = y[i] - min(left[i], right[i])
    return out


@njit(cache=True)
def alternate_kernel(y, idx, is_peak):
    """idx (sorted extremes; is_peak marks peaks, the rest are troughs) with the lowest point between
    every two consecutive peaks and the highest between every two consecutive troughs inserted, so
    peaks and troughs alternate. Returns the sorted indices."""
    out = np.empty(2 * len(idx), np.int64)
    m = 0
    for s in range(len(idx)):
        if s > 0 and is_peak[s] == is_peak[s - 1] and idx[s] - idx[s - 1] > 1:
            a, b = idx[s - 1], idx[s]
            best = a + 1
            for j in range(a + 2, b):
                if (y[j] < y[best]) if is_peak[s] else (y[j] > y[best]):
                    best = j
            out[m] = best
            m += 1
        out[m] = idx[s]
        m += 1
    return out[:m]
