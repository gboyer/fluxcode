# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Prune a time series to the points that carry its shape, read as straight lines between them.

A prepass for storage and processing, not a codec: the output is a subset of the input points,
far fewer of them. By default the count adapts to the data: the time-weighted RMSE relative to
the block's range traded against the points spent at a fixed rate. Two optional guards bound it:
a hard cap on the points and a largest error for any point. The first and last points (for
stitching blocks back together) and the minimum and maximum are always kept.

    import fluxprune

    p = fluxprune.prune(t, x)                         # automatic: p.index, p.max_error, p.rmse
    p = fluxprune.prune(t, x, max_points=500, max_error=0.5)   # guarded: p.capped if the cap won
    p = fluxprune.prune(t, x, error_per_point=0.0015 / 12)     # 12 h blocks (fewer per hour)

    # periodic signals: keep each peak and trough, prune the rest
    pk = fluxprune.peaks(t, x, period=np.timedelta64(60, "s"), prominence=2.0)
    p = fluxprune.prune(t, x, keep=pk)

    r = fluxprune.rank(t, x)        # step by step: one O(n log n) pass ranks every point
    r.relative_rmse()               # the error of every count, relative to the block's range
    r.top(200)                      # the 200 most significant points, nested
"""

from ._api import ERROR_PER_POINT, Pruned, Ranking, peaks, prune, rank

__all__ = ["ERROR_PER_POINT", "Pruned", "Ranking", "peaks", "prune", "rank"]
