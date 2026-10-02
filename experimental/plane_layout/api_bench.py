# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Size and speed through the public API at each distinct Params.effort, against the encoder
before efforts (both layouts compressed, smaller kept, one block run, zstd 3), on the report's
signals (20 kinds x 4 minutes) and 400 random-family units. One thread; times are best of 3 over
the whole set. The numbers in docs/TUNING.md (Effort) come from this script.

    uv run python plane_layout/api_bench.py
"""

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fluxcode
from _signals import DISCRETE, KINDS, discrete_minute, minute
from corpus import configs, draw_signal
from fluxcode import Params, _compress, _unit

BEFORE = _compress.Effort("best", False, (3,))
"""The encoder before efforts: planes="best" without block flushes."""


def sets():
    std = [np.concatenate([minute(n, 800 + j) if n in KINDS else discrete_minute(n, 800 + j) for j in range(4)])
           for n in KINDS + [n for n, _, _ in DISCRETE]]
    rng = np.random.default_rng(11)
    rand = []
    for _ in range(400):
        _, q, x = draw_signal(rng)
        _, mx, nf = configs(rng)
        rand.append((x, Params(max_quantize_bits=mx, noise_floor_sigma=nf)))
    return {"report signals": [(x, Params()) for x in std], "random families": rand}


def run(items, effort):
    """(bytes, encode s, decode s) of every item encoded with effort's settings."""
    default = Params().effort
    saved = _compress.EFFORTS[default]
    _compress.EFFORTS[default] = effort
    try:
        def enc():
            return [fluxcode.encode(x, p)[0] for x, p in items]

        out = enc()
        t_enc = t_dec = float("inf")
        for _ in range(3):
            t0 = time.perf_counter()
            enc()
            t_enc = min(t_enc, time.perf_counter() - t0)
            t0 = time.perf_counter()
            for units in out:
                fluxcode.decode(units)
            t_dec = min(t_dec, time.perf_counter() - t0)
    finally:
        _compress.EFFORTS[default] = saved
    return sum(len(u) for units in out for u in units), t_enc, t_dec


if __name__ == "__main__":
    distinct = {}
    for effort, settings in sorted(_compress.EFFORTS.items()):
        distinct.setdefault(settings, []).append(effort)
    rows = [("before", BEFORE)] + [(f"{e[0]}–{e[-1]}" if len(e) > 1 else str(e[0]), s) for s, e in distinct.items()]
    for label, items in sets().items():
        n = sum(len(x) for x, _ in items)
        print(f"\n{label}: {n / 1e6:.1f} M samples")
        print(f"{'effort':8s} {'bits/sample':>12s} {'size':>8s} {'encode ns/sample':>24s} {'decode ns/sample':>24s}")
        base = None
        for name, settings in rows:
            b, te, td = run(items, settings)
            te, td = te / n * 1e9, td / n * 1e9
            base = base or (b, te, td)
            rel = lambda v, v0: f"{v:5.1f} ({v - v0:+5.1f}, {100 * (v / v0 - 1):+4.0f}%)"
            print(f"{name:8s} {8 * b / n:12.3f} {100 * (b / base[0] - 1):+7.2f}% {rel(te, base[1]):>24s} {rel(td, base[2]):>24s}",
                  flush=True)
