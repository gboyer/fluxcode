# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Swinging-door trending, stored in the piecewise codecs' knot format."""

import numpy as np

from tslab.classic.piecewise import MAX_GAP, decode_knots, encode_knots


class SwingingDoor:
    """Swinging-door trending (Bristol 1990), the bounded-error piecewise-linear
    compression used by industrial historians. `dev` is the compression deviation
    as a fraction of full scale. Archived points are actual samples, so an
    impulse is archived at its own sample index."""

    variable_size = True

    def __init__(self, dev):
        self.dev = dev
        self.name = f"swingdoor-{dev * 100:g}%"
        self.label = f"Swinging-door trending, deviation {dev * 100:g}% of range"

    def knots(self, x):
        n = len(x)
        E = self.dev * (x.max() - x.min())
        idx = [0]
        a = 0
        smin, smax = -np.inf, np.inf
        t = 1
        while t < n:
            dt = t - a
            lo_s = (x[t] - E - x[a]) / dt
            hi_s = (x[t] + E - x[a]) / dt
            new_min, new_max = max(smin, lo_s), min(smax, hi_s)
            if new_min > new_max or dt > MAX_GAP:
                # Door closed (or offset field would overflow): archive previous
                # sample and restart the doors from it; re-examine sample t.
                a = t - 1
                idx.append(a)
                smin, smax = -np.inf, np.inf
                continue
            smin, smax = new_min, new_max
            t += 1
        if idx[-1] != n - 1:
            idx.append(n - 1)
        return idx

    def encode(self, x):
        idx = self.knots(x)
        return encode_knots(x, idx), {"knots": idx}

    def decode(self, data, n):
        idx, v = decode_knots(data, n)
        return np.interp(np.arange(n), idx, v)
