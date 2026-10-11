# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""How many points to keep, from a fluxprune.Ranking's error curve e[k] (relative RMSE of keeping k).

Every rule is a function of the curve only, so all of them run on the same ranking:

- count(K):         k = K (constant bit rate).
- rmse(eps):        the fewest points whose error, and that of every larger count, is at most eps.
- geomean(K, eps):  sqrt(K * k_eps): halfway between the two in log space.
- marginal(lam):    the k that minimises e² + lam * k: stop once a point lowers the mean squared
                    error by less than lam (best total squared error for the points spent).
- marginal_rmse(lam): the k that minimises e + lam * k: the allocation with the least mean RMSE over
                    blocks for a total number of points.
- amortized(theta): the first k where e[k] / (points added beyond the required) < theta.
- noisy(rule, ...):  the same rule with eps raised to the block's noise level, estimated from the curve:
                    keeping the best half of white noise's points leaves 0.29 of its sigma
                    (fluxprune.Ranking.noise).
- knee(alpha, eps): the first k where doubling the points improves the error by less than a factor
                    2**alpha (or the error is already below eps). Scale free: it does not depend on the
                    value range, the sampling rate or the block length, only on the curve's shape.
"""

import numpy as np


def _curve(r):
    e = r.rmse()
    e[: r.min_points] = np.inf
    return e


def count(r, K):
    return int(np.clip(K, r.min_points, r.n))


def rmse(r, eps):
    return r.count(rmse=eps)


def geomean(r, K, eps):
    return r.count(points=K, rmse=eps)


def noise(r):
    """The block's white-noise level relative to its range, from its own curve (fluxprune's estimate)."""
    return r.noise()


def rmse_noisy(r, eps):
    return r.count(rmse=eps, noise_floor=True)


def geomean_noisy(r, K, eps):
    return r.count(points=K, rmse=eps, noise_floor=True)


def marginal(r, lam):
    e = _curve(r)
    k = np.arange(len(e))
    return int(np.argmin(e ** 2 + lam * k))


def marginal_rmse(r, lam):
    e = _curve(r)
    return int(np.argmin(e + lam * np.arange(len(e))))


def amortized(r, theta):
    return r.count(error_per_point=theta)


def knee(r, alpha, eps=0.0):
    """Scan k upward; stop at the first k whose error is within eps, or where the best error with up to
    2k points is not 2**alpha times better than the best with up to k."""
    e = np.minimum.accumulate(_curve(r))  # best error with at most k points (the curve isn't monotone)
    n, m = r.n, r.min_points
    target = 2.0 ** -alpha
    for k in range(m, n + 1):
        if e[k] <= eps:
            return k
        if e[min(2 * k, n)] > target * e[k]:
            return k
    return n
