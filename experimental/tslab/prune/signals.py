# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Slow series for the pruning experiments (report/prune.html): a day of 1 s scans per signal-day,
cut into storage blocks of fixed duration.

- The six process tags of tslab.common.historian (every scan, and what its swinging-door archive kept).
- A few more shapes a store sees: a totalizer, batch cycles, vibration with bursts, an on/off
  compressor, pure white noise, and a clean sine (smooth, no noise: the case with no knee).
- OFFSET: signals whose small variations matter on a large offset (a furnace at ~2115 °F, grid
  frequency), the same ambient temperature in °C (crossing zero) and in K, a tank level that sits
  still for hours, and the furnace with thermocouple dropouts: what the error should be relative to.
- NOISY: a pyrometer at a conveyor's exit: a noisy floor and narrow spikes as hot items pass, in
  production runs with quiet hours between (the noise floor; its noise-free curve from pyrometer()).
- GUARD: a max_error per signal (the historian tags' CompDev, otherwise about what an engineer would set).
- MAX_ERROR: a loose and a fine max_error per signal for the tuning examples: loose about what an operator
  would notice, fine about twice the sensor's noise (or the resolution that matters, if it has none).

All seeds are fixed.
"""

import numpy as np

from tslab.common.historian import (
    DAY_S,
    DAY_START_NS,
    SECOND_NS,
    TAG_INFO,
    TAGS,
    _first_order,
    ou,
    setpoints,
    tag_day,
    tag_scans,
    tag_seed,
)

EXTRA = ["totalizer", "batch", "vibration", "compressor", "white-noise", "sine"]
SIGNALS = TAGS + EXTRA
OFFSET = ["furnace", "furnace-dropout", "grid-frequency", "ambient-C", "ambient-K", "level-quiet"]
NOISY = ["pyrometer"]
GUARD = {
    **{tag: TAG_INFO[tag][2] for tag in TAGS},
    "totalizer": 1.0, "batch": 0.2, "vibration": 0.1, "compressor": 0.5, "white-noise": 1.0, "sine": 0.01,
    "furnace": 1.0, "furnace-dropout": 1.0, "grid-frequency": 0.005, "ambient-C": 0.2, "ambient-K": 0.2,
    "level-quiet": 0.1, "pyrometer": 20.0,
}
MAX_ERROR = {  # (loose, fine), in the signal's units
    "temperature": (0.5, 0.1), "flow": (5.0, 2.0), "tank-level": (1.0, 0.1), "valve": (2.0, 0.5),
    "ph-float32": (0.1, 0.02), "motor-current": (5.0, 2.0), "totalizer": (5.0, 1.0), "batch": (1.0, 0.1),
    "vibration": (0.2, 0.06), "compressor": (2.0, 0.25), "white-noise": (5.0, 2.0), "sine": (0.05, 0.005),
    "furnace": (1.0, 0.25), "furnace-dropout": (1.0, 0.25), "grid-frequency": (0.01, 0.003),
    "ambient-C": (0.5, 0.15), "ambient-K": (0.5, 0.15), "level-quiet": (0.05, 0.015), "pyrometer": (50.0, 5.0),
}
EXTRA_INFO = {
    "totalizer": "Flow totalizer: integral of a stepped flow, 0.1 resolution (piecewise straight)",
    "batch": "Batch reactor: heat ramp, soak, cool, ~2 h cycles + noise σ 0.05",
    "vibration": "Vibration RMS: slow drift + rare bursts (seconds to minutes) + noise",
    "compressor": "Compressor power: on/off cycles (5-15 min) with start-up overshoot + noise",
    "white-noise": "White noise only, σ 1",
    "sine": "One 20 min sine, nothing else (smooth: error falls as k⁻² with no floor)",
    "furnace": "Furnace temperature, °F: setpoints 2100-2130 with ramps, control cycling ±0.3-1.5 °F, 0.1 °F steps",
    "furnace-dropout": "The furnace with 6 thermocouple dropouts a day (reads 0 for 2-20 s)",
    "grid-frequency": "Grid frequency, Hz: 60 ± ~10 mHz, 4 generator trips a day (−40 to −150 mHz), 1 mHz steps",
    "ambient-C": "Outdoor temperature, °C: −3 to 9 over the day, weather drift, 0.1 °C steps (crosses zero)",
    "ambient-K": "The same ambient temperature in kelvin (+273.15)",
    "level-quiet": "Tank level, %: held at 62.4 (±1 step of 0.01 dither) but for 3 fill/draw cycles a day",
    "pyrometer": "Pyrometer at a furnace's exit conveyor, °C: a 45-55 floor with noise σ 2, and hot items passing "
                 "every 1.5-5 min in production runs of 1-4 h with 0.5-3 h quiet between: spikes of 500-900 above "
                 "the floor, a / (b + t²) with a half-width of 1.5-2.5 s, 0.1 °C steps",
}


def extra_scans(name, day):
    rng = np.random.default_rng(9000 + 100 * EXTRA.index(name) + day)
    n = DAY_S
    s = np.arange(n, dtype=np.float64)
    if name == "totalizer":
        changes = np.sort(rng.integers(1, n, 12))
        flow = rng.uniform(0, 3, 13)[np.searchsorted(changes, np.arange(n), side="right")]
        flow[rng.random(13)[np.searchsorted(changes, np.arange(n), side="right")] < 0.25] = 0  # some stops
        return np.round(np.cumsum(flow), 1)
    if name == "batch":
        out = np.empty(n)
        k = 0
        while k < n:
            heat, soak, cool = (int(rng.integers(a, b)) for a, b in ((900, 1800), (1800, 3600), (1200, 2400)))
            top = rng.uniform(70, 90)
            seg = np.concatenate([np.linspace(25, top, heat), np.full(soak, top),
                                  25 + (top - 25) * np.exp(-np.arange(cool) / 400)])
            out[k:k + len(seg)] = seg[: n - k]
            k += len(seg)
        return np.round(_first_order(out, 1 / 30, out[0]) + rng.normal(0, 0.05, n), 2)
    if name == "vibration":
        x = 2 + 0.3 * np.sin(2 * np.pi * s / DAY_S) + ou(rng, n, 3600, 0.1)
        for start in rng.integers(0, n - 600, 8):
            length = int(rng.integers(5, 300))
            x[start:start + length] += rng.uniform(2, 8) * np.exp(-np.arange(length) / (length / 3))
        return np.round(x + np.abs(rng.normal(0, 0.05, n)), 3)
    if name == "compressor":
        out = np.empty(n)
        k, on = 0, False
        while k < n:
            length = int(rng.integers(300, 900))
            seg = np.full(length, 0.2)
            if on:
                seg = 15 + 10 * np.exp(-np.arange(length) / 20)
            out[k:k + length] = seg[: n - k]
            k += length
            on = not on
        return np.round(out + rng.normal(0, 0.1, n), 2)
    if name == "white-noise":
        return rng.normal(0, 1, n)
    if name == "sine":
        return np.sin(2 * np.pi * s / 1200 + rng.uniform(0, 2 * np.pi))
    raise ValueError(name)


def offset_scans(name, day):
    if name == "ambient-K":
        return offset_scans("ambient-C", day) + 273.15
    if name == "furnace-dropout":
        x = offset_scans("furnace", day)
        rng = np.random.default_rng(9900 + day)
        for start in rng.integers(0, DAY_S - 20, 6):
            x[start:start + rng.integers(2, 20)] = 0.0
        return x
    rng = np.random.default_rng(9500 + 100 * OFFSET.index(name) + day)
    n = DAY_S
    s = np.arange(n, dtype=np.float64)
    if name == "furnace":
        sp = setpoints(rng, n, 6, 2100, 2130, 5.0)
        sp = _first_order(sp, 1 / 600, sp[0])  # the load follows a setpoint change over ~10 min
        amp = 0.3 + 1.2 * rng.random(n // 3600 + 1)[(s // 3600).astype(int)]  # cycling amplitude, per hour
        cyc = amp * np.sin(2 * np.pi * s / rng.uniform(480, 900) + rng.uniform(0, 2 * np.pi))
        return np.round(sp + cyc + ou(rng, n, 1200, 0.5) + rng.normal(0, 0.1, n), 1)
    if name == "grid-frequency":
        x = 60 + ou(rng, n, 30, 0.008) + 0.004 * np.sin(2 * np.pi * s / 3600)  # hourly schedule changes
        for start in rng.integers(0, n - 1200, 4):
            drop, recover = rng.uniform(0.04, 0.15), rng.uniform(60, 300)
            k = np.arange(n - start, dtype=np.float64)
            x[start:] -= drop * np.minimum(k / 5, 1) * np.exp(-np.maximum(k - 5, 0) / recover)
        return np.round(x, 3)
    if name == "ambient-C":
        return np.round(3 + 6 * np.sin(2 * np.pi * (s / DAY_S - 0.35)) + ou(rng, n, 3600, 1.0)
                        + rng.normal(0, 0.03, n), 1)
    if name == "level-quiet":
        x = np.full(n, 62.4)
        for start in rng.integers(0, n - 7200, 3):
            fill = np.concatenate([np.linspace(0, 1, 1800), np.ones(int(rng.integers(600, 2400))), np.linspace(1, 0, 2400)])
            x[start:start + len(fill)] += rng.uniform(5, 20) * fill
        return np.round(x + rng.normal(0, 0.006, n), 2)
    raise ValueError(name)


def pyrometer(day):
    """(a day of 1 s scans, the noise-free curve under them, the spikes' times in s)."""
    rng = np.random.default_rng(9800 + day)
    n = DAY_S
    s = np.arange(n, dtype=np.float64)
    truth = 50 + 5 * np.sin(2 * np.pi * s / DAY_S + rng.uniform(0, 2 * np.pi))
    centers = []
    k, run = float(rng.uniform(0, 3600)), False
    while k < n:
        length = rng.uniform(3600, 4 * 3600) if run else rng.uniform(1800, 3 * 3600)
        if run:
            c = k + rng.uniform(90, 300)
            while c < min(k + length, n - 1):
                centers.append(c)
                c += rng.uniform(90, 300)
        k += length
        run = not run
    for c in centers:
        lo, hi = max(int(c) - 600, 0), min(int(c) + 600, n)
        truth[lo:hi] += rng.uniform(500, 900) / (1 + ((s[lo:hi] - c) / rng.uniform(1.5, 2.5)) ** 2)
    return np.round(truth + rng.normal(0, 2.0, n), 1), truth, np.array(centers)


def scans(name, day):
    """A day of 1 s scans of a signal."""
    if name in TAGS:
        return tag_scans(name, tag_seed(name, day))
    if name in NOISY:
        return pyrometer(day)[0]
    return offset_scans(name, day) if name in OFFSET else extra_scans(name, day)


def blocks(name, day, hours, archive=False):
    """[(times ns, values)] of one signal-day cut into blocks of `hours`: every scan, or with archive=True the
    points the tag's swinging-door archive kept (tags only)."""
    if archive:
        x, idx = tag_day(name, day)
    else:
        x = scans(name, day)
        idx = np.arange(DAY_S)
    t = DAY_START_NS + idx * SECOND_NS
    out = []
    width = int(hours * 3600)
    for start in range(0, DAY_S, width):
        sel = (idx >= start) & (idx < start + width)
        if np.count_nonzero(sel) >= 2:
            out.append((t[sel], x[idx[sel]]))
    return out
