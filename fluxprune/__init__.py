# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Prune a time series to the points that carry its shape, read as straight lines between them.

A prepass for storage and processing, not a codec: the output is a subset of the input points,
far fewer of them. By default the count adapts to the data (the rmse pass): the time-weighted RMSE
relative to the block's range traded against the points spent at a fixed rate. Two optional guards
bound it: max_error (the max_error pass adds points until no input point strays further) and
max_points, a hard cap. The first and last points (for stitching blocks back together) and the
minimum and maximum are always kept.

    import fluxprune

    p = fluxprune.prune(t, x)                                  # p.times, p.values, p.rmse, p.max_error
    p = fluxprune.prune(t, x, max_points=500, max_error=0.5)   # p.rmse_met, p.max_error_met
    p = fluxprune.prune(t, x, error_per_point=0, max_points=36)        # the best 36 points
    p = fluxprune.prune(t, x, error_per_point=np.inf, max_error=0.5)   # the max_error pass alone
"""

from ._api import ERROR_PER_POINT, Pruned, prune

__all__ = ["ERROR_PER_POINT", "Pruned", "prune"]
