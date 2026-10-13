# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""The refit kernel: values at the kept times that minimise the time-weighted squared error.

The kept points' lines are a combination of hat functions, one per kept point, so the values that
fit the input best in time-weighted L2 solve a tridiagonal system (a Gram matrix of the hats, and
the input projected on them), exact per input segment since both are linear there. Fixed points
keep their sample; segments with a non-finite end (forward holds) don't count.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def refit_kernel(tau, y, keep, fixed):
    """Values at the kept points (sorted indices; fixed[i] pins point i to y[i]) whose lines fit the
    input best in time-weighted L2. O(n)."""
    k = len(keep)
    diag = np.zeros(k)
    off = np.zeros(k)  # off[s]: the coupling of s and s + 1
    rhs = np.zeros(k)
    for s in range(k - 1):
        a, b = keep[s], keep[s + 1]
        w = tau[b] - tau[a]
        if w <= 0.0 or not (np.isfinite(y[a]) and np.isfinite(y[b])):
            continue
        for j in range(a, b):
            h = tau[j + 1] - tau[j]
            r0 = (tau[j] - tau[a]) / w
            r1 = (tau[j + 1] - tau[a]) / w
            l0, l1 = 1.0 - r0, 1.0 - r1  # the left hat at both ends of the input segment
            diag[s] += h * (l0 * l0 + l0 * l1 + l1 * l1) / 3.0
            diag[s + 1] += h * (r0 * r0 + r0 * r1 + r1 * r1) / 3.0
            off[s] += h * (2.0 * l0 * r0 + l0 * r1 + l1 * r0 + 2.0 * l1 * r1) / 6.0
            rhs[s] += h * (2.0 * l0 * y[j] + l0 * y[j + 1] + l1 * y[j] + 2.0 * l1 * y[j + 1]) / 6.0
            rhs[s + 1] += h * (2.0 * r0 * y[j] + r0 * y[j + 1] + r1 * y[j] + 2.0 * r1 * y[j + 1]) / 6.0
    # a pinned point (or one no segment weighs) is an identity row; its neighbours see it as known
    lower = np.zeros(k)
    upper = np.zeros(k)
    for s in range(k):
        if fixed[keep[s]] or diag[s] <= 0.0:
            diag[s], rhs[s] = 1.0, y[keep[s]]
        else:
            if s > 0:
                lower[s] = off[s - 1]
            if s < k - 1:
                upper[s] = off[s]
    # Thomas: the free rows are a symmetric positive definite Gram matrix's, the pinned ones identity
    c = np.empty(k)
    for s in range(1, k):
        if lower[s] != 0.0:
            m = lower[s] / diag[s - 1]
            diag[s] -= m * upper[s - 1]
            rhs[s] -= m * rhs[s - 1]
    c[k - 1] = rhs[k - 1] / diag[k - 1]
    for s in range(k - 2, -1, -1):
        c[s] = (rhs[s] - (upper[s] * c[s + 1] if upper[s] != 0.0 else 0.0)) / diag[s]  # a gap's NaN stays put
    return c
