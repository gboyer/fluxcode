# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Prune a time series to the points that carry its shape, read as straight lines between them.

A prepass for storage and processing, not a codec: the output is a subset of the input points,
far fewer of them, chosen to keep the time-weighted RMSE of the linear interpolation low. The
first and last points (for stitching blocks back together) and the minimum and maximum are
always kept.

    import fluxprune

    r = fluxprune.rank(t, x)          # one O(n log n) pass ranks every point
    idx = r.keep(200)                 # the 200 most significant points, sorted
    r.rmse(200)                       # their time-weighted RMSE, relative to max - min

    # or in one call, refined: 200 points, within 1% RMSE, or a balance of the two
    idx = fluxprune.prune(t, x, points=200)
    idx = fluxprune.prune(t, x, rmse=0.01)
    idx = fluxprune.prune(t, x, points=200, rmse=0.01)

    # adaptive, one knob: stop where the RMSE left falls below 0.0015 per point added
    idx = fluxprune.prune(t, x, error_per_point=0.0015)

    # adaptive, two knobs: the balance, but never chasing the series' own noise
    idx = fluxprune.prune(t, x, points=200, rmse=0.01, noise_floor=True)
"""

from ._api import Ranking, prune, rank

__all__ = ["Ranking", "prune", "rank"]
