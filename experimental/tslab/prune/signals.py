# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Slow series for the pruning experiments (report/prune.html): a day of 1 s scans per signal-day,
cut into storage blocks of fixed duration.

- The six process tags of tslab.common.historian (every scan, and what its swinging-door archive kept).
- A few more shapes a store sees: a totalizer, batch cycles, vibration with bursts, an on/off
  compressor, pure white noise, and a clean sine (smooth, no noise: the case with no knee).

All seeds are fixed.
"""

import numpy as np

from tslab.common.historian import DAY_S, DAY_START_NS, SECOND_NS, TAGS, _first_order, ou, tag_day, tag_scans, tag_seed

EXTRA = ["totalizer", "batch", "vibration", "compressor", "white-noise", "sine"]
SIGNALS = TAGS + EXTRA
EXTRA_INFO = {
    "totalizer": "Flow totalizer: integral of a stepped flow, 0.1 resolution (piecewise straight)",
    "batch": "Batch reactor: heat ramp, soak, cool, ~2 h cycles + noise σ 0.05",
    "vibration": "Vibration RMS: slow drift + rare bursts (seconds to minutes) + noise",
    "compressor": "Compressor power: on/off cycles (5-15 min) with start-up overshoot + noise",
    "white-noise": "White noise only, σ 1",
    "sine": "One 20 min sine, nothing else (smooth: error falls as k⁻² with no floor)",
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


def scans(name, day):
    """A day of 1 s scans of a signal."""
    return tag_scans(name, tag_seed(name, day)) if name in TAGS else extra_scans(name, day)


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
