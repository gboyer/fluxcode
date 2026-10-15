# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""The noise kernel: a series' white-noise level, for the rmse pass's noise floor.

Each point's distance from the chord of its two neighbours is, for white noise σ on a signal straight
at that scale, normal with standard deviation σ·√(1 + w² + (1 - w)²), w the point's position between
its neighbours in time (√1.5 at regular times): the difference-based estimator of Gasser, Sroka and
Jennen-Steinmetz (1986) for uneven times. Divided by that, its median absolute value times 1.4826
estimates σ; the median ignores the few points on a spike or a step. A smooth signal's curvature
between neighbours reads as a little noise; correlated noise reads as less than it is.

It assumes the times were chosen independently of the values, which an archive's aren't: swinging
door or a deadband keeps a point because of its value, and its points look like noise to any test on
the values (report/prune.html). So, as fluxcode's own noise floor does, it only runs on a cadence:
at least 90% of the intervals one scan (fluxcode._noise.cadence, the one definition for both), and
only on triplets whose two intervals are one scan; otherwise σ is 0 and there is no floor.
"""

import numpy as np
from numba import njit

from fluxcode._noise import cadence
from fluxcode._time import CADENCE_SAMPLES

MAD_TO_SIGMA = 1.4826
"""The median absolute value of a normal variable is 0.6745σ."""

TICKS = float(1 << 52)
"""Times from 0 to 1 as int64 ticks for the cadence test: the ratios of intervals, which is all it
reads, to 2^-52."""


@njit(cache=True)
def noise_kernel(ticks, y, lower, upper):
    """The white-noise σ of y (in y's units) from each point's distance to its neighbours' chord, over
    the triplets whose two intervals are one scan (lower < interval < upper) and whose values are finite;
    0 with none. O(n)."""
    n = len(y)
    r = np.empty(max(n - 2, 0))
    m = 0
    for i in range(1, n - 1):
        d0, d1 = ticks[i] - ticks[i - 1], ticks[i + 1] - ticks[i]
        if not (lower < d0 < upper and lower < d1 < upper):
            continue
        a, b, c = y[i - 1], y[i], y[i + 1]
        if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(c)):
            continue
        w = d0 / (d0 + d1)
        r[m] = abs(b - (a + w * (c - a))) / np.sqrt(1.0 + w * w + (1.0 - w) * (1.0 - w))
        m += 1
    if m == 0:
        return 0.0
    return MAD_TO_SIGMA * np.median(r[:m])


def noise(tau, y):
    """σ of y at times tau (float offsets from 0, non-decreasing, the last positive), or 0 without a
    cadence."""
    ticks = np.round(tau * (TICKS / tau[-1])).astype(np.int64)
    lower, upper, _ = cadence(ticks, np.empty(CADENCE_SAMPLES, np.int64))
    if upper == 0:
        return 0.0
    return float(noise_kernel(ticks, y, lower, upper))
