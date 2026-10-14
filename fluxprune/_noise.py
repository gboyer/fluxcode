# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""The noise kernel: a series' white-noise level, for the rmse pass's noise floor.

Each interior point's distance from the chord of its two neighbours is, for white noise σ on a
signal straight at that scale, normal with standard deviation σ·√(1 + w² + (1 - w)²), w the point's
position between its neighbours in time (√1.5 at regular times). Divided by that, its median
absolute value times 1.4826 estimates σ: at any spacing of times, and the median ignores the few
points on a spike or a step. A smooth signal's curvature between neighbours reads as a little noise;
correlated noise reads as less than it is.
"""

import numpy as np
from numba import njit

MAD_TO_SIGMA = 1.4826
"""The median absolute value of a normal variable is 0.6745σ."""


@njit(cache=True)
def noise_kernel(tau, y):
    """The white-noise σ of y (in y's units) from each interior point's distance to its neighbours'
    chord; points with a non-finite neighbour or none apart in time are skipped, 0 with none left.
    O(n)."""
    n = len(y)
    r = np.empty(max(n - 2, 0))
    m = 0
    for i in range(1, n - 1):
        a, b, c = y[i - 1], y[i], y[i + 1]
        du = tau[i + 1] - tau[i - 1]
        if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(c)) or du <= 0.0:
            continue
        w = (tau[i] - tau[i - 1]) / du
        r[m] = abs(b - (a + w * (c - a))) / np.sqrt(1.0 + w * w + (1.0 - w) * (1.0 - w))
        m += 1
    if m == 0:
        return 0.0
    return MAD_TO_SIGMA * np.median(r[:m])
