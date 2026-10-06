# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""The layout rule and the efforts on the day-scale stress test's sensor mix (bench/stress.py:
analog, held, digital, vibration, counter, sensor-0.1, noisy-sine, random-walk, weighted as there),
which the study's corpus lacked.

1. Rules "byte planes iff fewer than t of the residuals reach 2^k", per-plane blocks, against the
   encoder before efforts, on the sensor mix, the random families (corpus.py, seed 1) and the
   report's signals; with and without recompressing frames under 16 KB with the other layout.
2. Size and single-thread encode time per block group of each distinct effort on the sensor mix.

    uv run python plane_layout/fleet.py             # ~5 min
"""

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tests"), str(ROOT / "bench"), str(Path(__file__).resolve().parent)]
import fluxcode
from _series import planes, group_rows
from _signals import DISCRETE, KINDS, discrete_minute, minute
from corpus import configs, draw_signal
from fluxcode import Params, _compress, _group
from stress import PRESETS, gen_minute

MIX = PRESETS["sensor-mix"]
BEFORE = _compress.Effort("best", False, (3,))


def row(x, params, weight):
    """(bit one run, bit flushed, byte one run, byte flushed, share >= 2^4 .. 2^8, weight)."""
    sizes = []
    for layout in ("bit", "byte"):
        for flush in (False, True):
            with planes(layout, flush=flush):
                sizes.append(len(fluxcode.encode(x, params)[0][0]))
    r = group_rows(fluxcode.encode(x, params)[0][0]).residuals.astype(np.int32)
    u = ((r << 1) ^ (r >> 15)) & 0xFFFF
    return sizes + [float(np.mean(u >= 2 ** k)) for k in range(4, 9)] + [weight]


def rule_sets():
    sets = {"sensor mix": [row(gen_minute(k, s), Params(), w) for k, w in MIX.items() for s in range(10)]}
    rng = np.random.default_rng(1)
    rows = []
    for _ in range(1000):
        _, _, x = draw_signal(rng)
        _, mx, nf = configs(rng)
        rows.append(row(x, Params(max_quantize_bits=mx, noise_floor_sigma=nf), 1))
    sets["random"] = rows
    names = KINDS + [n for n, _, _ in DISCRETE]
    sets["report"] = [row(minute(n, 500 + j) if n in KINDS else discrete_minute(n, 500 + j), Params(), 1)
                      for n in names for j in range(5)]
    return {name: np.array(rows) for name, rows in sets.items()}


def rules():
    sets = rule_sets()
    for name, a in sets.items():
        w = a[:, 9]
        before = (w * np.minimum(a[:, 0], a[:, 2])).sum()
        oracle = (w * np.minimum(a[:, 1], a[:, 3])).sum()
        print(f"{name}: per-group better flushed layout {100 * (oracle / before - 1):+.2f}% against before")
    for retry in (False, True):
        print(f"\nrule (byte planes iff share(u >= 2^k) < t), flushed{', other layout under 16 KB' if retry else ''}:")
        for k in (6, 7, 8):
            for t in (0.005, 0.01, 0.02, 0.05):
                cells = []
                for name, a in sets.items():
                    w, bit, byte = a[:, 9], a[:, 1], a[:, 3]
                    pick = np.where(a[:, 4 + k - 4] < t, byte, bit)
                    if retry:
                        pick = np.where(pick < 16_384, np.minimum(bit, byte), pick)
                    cells.append(f"{name} {100 * ((w * pick).sum() / (w * np.minimum(a[:, 0], a[:, 2])).sum() - 1):+6.2f}%")
                print(f"  k = {k}, t = {t:5}: " + "   ".join(cells))


def efforts():
    data = {kind: [gen_minute(kind, s) for s in range(8)] for kind in MIX}
    distinct = {}
    for effort, settings in sorted(_compress.EFFORTS.items()):
        distinct.setdefault(settings, []).append(effort)
    rows = [("before", BEFORE)] + [(f"{e[0]}–{e[-1]}" if len(e) > 1 else str(e[0]), s) for s, e in distinct.items()]
    saved = _compress.EFFORTS[4]
    print("\nsensor mix, per block group (weighted), one thread:")
    base = None
    for name, settings in rows:
        _compress.EFFORTS[4] = settings
        size = elapsed = 0.0
        for kind, w in MIX.items():
            xs = data[kind]
            size += w * sum(len(fluxcode.encode(x)[0][0]) for x in xs) / len(xs)
            best = np.inf
            for _ in range(5):
                t0 = time.perf_counter()
                for x in xs:
                    fluxcode.encode(x)
                best = min(best, time.perf_counter() - t0)
            elapsed += w * best / len(xs) * 1e6
        size, elapsed = size / sum(MIX.values()), elapsed / sum(MIX.values())
        base = base or (size, elapsed)
        print(f"  effort {name:5s} size {100 * (size / base[0] - 1):+6.2f}%   encode {elapsed:5.0f} µs/block group "
              f"({elapsed - base[1]:+5.0f} µs, {100 * (elapsed / base[1] - 1):+4.0f}%)")
    _compress.EFFORTS[4] = saved


if __name__ == "__main__":
    rules()
    efforts()
