# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Size and speed through the public API for each Params.planes mode, on the report's signals
(20 kinds x 4 minutes) and 400 random-family units. Run it once on the baseline code and once on
the branch with flushing (PYTHONPATH picks which fluxcode is imported); modes a build lacks are
skipped. One thread; times are best of 3 over the whole set.

    PYTHONPATH=<repo> uv run python plane_layout/api_bench.py
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
from fluxcode import Params


def sets():
    std = [np.concatenate([minute(n, 800 + j) if n in KINDS else discrete_minute(n, 800 + j) for j in range(4)])
           for n in KINDS + [n for n, _, _ in DISCRETE]]
    rng = np.random.default_rng(11)
    rand = []
    for _ in range(400):
        _, q, x = draw_signal(rng)
        _, mx, nf = configs(rng)
        rand.append((x, Params(max_quantize_bits=mx, noise_floor_sigma=nf)))
    return std, rand


def run(items, planes):
    def enc():
        return [fluxcode.encode(x, Params(**{**(p.__dict__ if p else {}), "planes": planes}))[0] for x, p in items]

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
    return sum(len(u) for units in out for u in units), t_enc, t_dec


if __name__ == "__main__":
    print("fluxcode from", fluxcode.__file__)
    std, rand = sets()
    sets_ = {"report signals": [(x, None) for x in std], "random families": rand}
    for label, items in sets_.items():
        n = sum(len(x) for x, _ in items) / 1e6
        print(f"\n{label}: {n:.1f} M samples")
        print(f"{'planes':12s} {'bytes':>11s} {'bits/sample':>12s} {'encode s':>9s} {'ns/sample':>10s} {'decode s':>9s}")
        for mode in ("best", "heuristic", "bit", "byte"):
            try:
                Params(planes=mode)
            except ValueError:
                continue
            b, te, td = run(items, mode)
            print(f"{mode:12s} {b:11d} {8 * b / (n * 1e6):12.3f} {te:9.2f} {te / n * 1e3:10.0f} {td:9.2f}")
