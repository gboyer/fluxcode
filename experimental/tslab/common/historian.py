# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""What a process historian that uses swinging-door compression (SDT) stores, simulated.

A historian scans each tag at a fixed rate, passes the scans through an exception (deadband)
filter, and then archives only the points that swinging-door trending keeps: actual samples, at
irregular times, such that the straight lines between them stay within CompDev of every snapshot.
Reading the archive back gives either the archived points themselves (irregular timestamps) or
a regular series resampled from them: interpolated (straight lines between the points) or held
(the last value, as a forward fill gives).

- TAGS / TAG_INFO: six process-tag archetypes, one day of 1 s scans each (tag_day).
- exception_filter, swinging_door, archive: the historian's two filters, PI style.
- tag_day(tag, seed): the scans and the archived points of one tag-day.
- interp / hold: a regular series read back from archived points (the trend, and a resampled export).

All seeds are fixed, so every run sees the same data.
"""

import numpy as np
from numba import njit

DAY_S = 86_400
DAY_START_NS = 1_790_035_200_000_000_000  # 2026-09-22 00:00 UTC, ns since 1970
SECOND_NS = 1_000_000_000
COMP_MAX_S = 8 * 3600  # PI's default CompMax: archive at least every 8 hours


# ---------------------------------------------------------------------------
# The historian's filters
# ---------------------------------------------------------------------------

@njit(cache=True)
def exception_filter(x, exc_dev):
    """Indices of the scans the interface reports: those more than exc_dev from the last reported
    value, plus the scan just before each one (so the start of a change keeps its time), plus the
    first and last scans."""
    n = len(x)
    keep = np.zeros(n, np.bool_)
    keep[0] = keep[n - 1] = True
    last = x[0]
    for k in range(1, n):
        if abs(x[k] - last) > exc_dev:
            keep[k] = keep[k - 1] = True
            last = x[k]
    return np.flatnonzero(keep)


@njit(cache=True)
def swinging_door(t, x, comp_dev, comp_max):
    """Indices of the snapshots (t, x) that swinging-door trending archives (Bristol 1990, as in
    PI): from the last archived point, the doors are the steepest and shallowest slopes that keep
    every later snapshot within comp_dev. When a snapshot closes the doors, or comes more than
    comp_max after the archived point, the snapshot before it is archived and the doors restart
    from there. The first and last snapshots are always archived."""
    n = len(t)
    out = np.empty(n, np.int64)
    out[0] = 0
    m = 1
    a = 0
    lo, hi = -np.inf, np.inf
    k = 1
    while k < n:
        dt = t[k] - t[a]
        new_lo = max(lo, (x[k] - comp_dev - x[a]) / dt)
        new_hi = min(hi, (x[k] + comp_dev - x[a]) / dt)
        if (new_lo > new_hi or dt > comp_max) and k - 1 > a:
            a = k - 1
            out[m] = a
            m += 1
            lo, hi = -np.inf, np.inf
            continue  # re-examine snapshot k against the new archived point
        lo, hi = new_lo, new_hi
        k += 1
    if out[m - 1] != n - 1:
        out[m] = n - 1
        m += 1
    return out[:m]


def archive(x, exc_dev, comp_dev, comp_max=COMP_MAX_S):
    """Indices of the scans (1 s apart) a historian archives: exception filter, then swinging door."""
    snap = exception_filter(x, exc_dev)
    return snap[swinging_door(snap.astype(np.float64), x[snap], comp_dev, float(comp_max))]


def interp(idx, values, n):
    """The interpolated read-back: straight lines between the archived points, at every index."""
    return np.interp(np.arange(n), idx, values)


def hold(idx, values, n):
    """The held read-back (a forward fill): each index takes the last archived value."""
    return values[np.searchsorted(idx, np.arange(n), side="right") - 1]


# ---------------------------------------------------------------------------
# Process tags: one day of 1 s scans
# ---------------------------------------------------------------------------

@njit(cache=True)
def _first_order(u, alpha, y0):
    """y[k] = y[k-1] + alpha * (u[k] - y[k-1]) from y[-1] = y0: a first-order lag (time constant
    ~1/alpha scans)."""
    y = np.empty_like(u)
    y[0] = y0 + alpha * (u[0] - y0)
    for k in range(1, len(u)):
        y[k] = y[k - 1] + alpha * (u[k] - y[k - 1])
    return y


def ou(rng, n, tau, sigma):
    """An Ornstein-Uhlenbeck process: noise with correlation time tau scans and std sigma."""
    alpha = 1 / tau
    return _first_order(rng.normal(0, sigma * np.sqrt(2 / alpha), n), alpha, rng.normal(0, sigma))


def setpoints(rng, n, per_day, lo, hi, quantum):
    """A piecewise-constant setpoint: Poisson(per_day) changes, each to a new level in [lo, hi]."""
    changes = np.sort(rng.integers(1, n, rng.poisson(per_day)))
    levels = np.round(rng.uniform(lo, hi, len(changes) + 1) / quantum) * quantum
    return levels[np.searchsorted(changes, np.arange(n), side="right")]


def ramps(rng, n, lo, hi):
    """A tank level: fill to near hi and drain to near lo at random rates (%/s), holding in between."""
    out = np.empty(n)
    level, k = rng.uniform(lo, hi), 0
    while k < n:
        hold_s = int(rng.integers(60, 1800))
        out[k:k + hold_s] = level
        k += hold_s
        target = rng.uniform(hi - 15, hi) if level < (lo + hi) / 2 else rng.uniform(lo, lo + 15)
        rate = rng.uniform(0.01, 0.05) * np.sign(target - level)
        steps = int(abs(target - level) / abs(rate))
        out[k:k + steps] = level + rate * np.arange(1, min(steps, n - k) + 1)
        k += steps
        level = target
    return out


def _decimal(x, places):
    return np.round(x, places)


def tag_scans(tag, seed):
    """One day (DAY_S scans, 1 s apart) of a tag's values, as the sensor and the PLC give them."""
    rng = np.random.default_rng(seed)
    n = DAY_S
    s = np.arange(n)
    if tag == "temperature":
        x = 60 + 6 * np.sin(2 * np.pi * s / DAY_S + rng.uniform(0, 2 * np.pi)) + ou(rng, n, 1800, 1.5)
        return _decimal(x + rng.normal(0, 0.03, n), 2)
    if tag == "flow":
        sp = setpoints(rng, n, 6, 80, 160, 5.0)
        x = _first_order(sp, 1 / 60, sp[0]) + ou(rng, n, 5, 1.5) + rng.normal(0, 0.8, n)
        return _decimal(x, 1)
    if tag == "tank-level":
        return _decimal(ramps(rng, n, 10, 90) + rng.normal(0, 0.02, n), 2)
    if tag == "valve":
        cmd = setpoints(rng, n, 30, 0, 100, 5.0)
        travel = np.empty(n)  # the actuator moves at 2 %/s towards the command
        travel[0] = cmd[0]
        for k in range(1, n):
            travel[k] = travel[k - 1] + np.clip(cmd[k] - travel[k - 1], -2.0, 2.0)
        return _decimal(travel, 1)
    if tag == "ph-float32":
        x = 7 + ou(rng, n, 600, 0.2) + rng.normal(0, 0.004, n)
        return x.astype(np.float32).astype(np.float64)
    if tag == "motor-current":
        sp = setpoints(rng, n, 24, 30, 60, 0.5)
        load = _first_order(sp, 1 / 10, sp[0])
        return _decimal(load + rng.normal(0, 1.0, n), 2)
    raise ValueError(tag)


TAGS = ["temperature", "flow", "tank-level", "valve", "ph-float32", "motor-current"]

# tag -> (description, ExcDev, CompDev, the window the report plots as (start hour, hours)). CompDev is
# in engineering units; ExcDev is half of it, the usual rule of thumb.
TAG_INFO = {
    "temperature": ("Daily cycle + slow drift (τ 30 min) + noise σ 0.03 °C, 0.01 °C resolution", 0.05, 0.1, (9, 1)),
    "flow": ("Setpoint steps (6/day) through a 60 s lag + process noise (σ 1.5, τ 5 s) + measurement noise σ 0.8, "
             "0.1 resolution: noise above CompDev", 0.5, 1.0, (9, 1 / 6)),
    "tank-level": ("Fill and drain ramps (0.01–0.05 %/s) between holds + noise σ 0.02 %, 0.01 % resolution",
                   0.1, 0.2, (0, 24)),
    "valve": ("Position commands in 5 % steps (30/day), 2 %/s travel, no noise, 0.1 % resolution", 0.25, 0.5, (0, 24)),
    "ph-float32": ("pH 7 + slow drift (τ 10 min) + noise σ 0.004, stored as float32 (not decimal)", 0.01, 0.02, (9, 1)),
    "motor-current": ("Load steps (24/day) through a 10 s lag + noise σ 1 A, 0.01 A resolution: CompDev far "
                      "below the noise, so most scans are archived", 0.1, 0.2, (9, 1 / 6)),
}

DAYS = 5  # tag-days per tag


def tag_seed(tag, day):
    return 7000 + 100 * TAGS.index(tag) + day


def tag_day(tag, day, comp_dev=None):
    """(scans, archived scan indices) for one tag-day; comp_dev overrides the tag's CompDev
    (ExcDev stays half of it)."""
    _, exc_dev, dev, _ = TAG_INFO[tag]
    if comp_dev is not None:
        exc_dev, dev = comp_dev / 2, comp_dev
    x = tag_scans(tag, tag_seed(tag, day))
    return x, archive(x, exc_dev, dev)


def times_ns(idx, tag, day, source_timestamps=False):
    """int64 ns timestamps of scan indices: whole seconds (scan-aligned, as a historian's scan
    classes give), or with source timestamps: each scan delayed by 0-200 ms, ms resolution, as
    OPC device timestamps look."""
    t = DAY_START_NS + idx.astype(np.int64) * SECOND_NS
    if source_timestamps:
        rng = np.random.default_rng(tag_seed(tag, day) + 50_000)
        t += rng.integers(0, 200, DAY_S)[idx] * 1_000_000
    return t
