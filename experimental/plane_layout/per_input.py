# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Bits/sample per input type at the encoder before efforts (both layouts, one block run: efforts
3-4), the heuristic in one run (effort 2) and both layouts flushed (effort 5), sorted by how much
effort 2 loses against the encoder before. Input types: the stress
test's fleet kinds, the report's signals (also under a 6-bit target) and the random families of
corpus.py, by family. The outlier table in docs/TUNING.md (Effort) comes from this script.

    uv run python plane_layout/per_input.py         # ~3 min
"""

import collections
import os
import sys
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tests"), str(ROOT / "bench"), str(Path(__file__).resolve().parent)]
import fluxcode
from _signals import DISCRETE, KINDS, discrete_minute, minute
from corpus import configs, draw_signal
from fluxcode import Params, _compress, _group
from stress import FLEET, gen_minute

SETTINGS = {"before": _compress.EFFORTS[4], "effort 2": _compress.EFFORTS[2], "effort 5": _compress.EFFORTS[5]}


def input_sets():
    out = collections.defaultdict(list)
    for kind in FLEET:
        out["fleet " + kind] = [(gen_minute(kind, s), Params()) for s in range(6)]
    for kind in KINDS:
        out[kind] = [(minute(kind, s), Params()) for s in range(6)]
        out[kind + " target 6"] = [(minute(kind, s), Params(target_bits_per_sample=6.0)) for s in range(6)]
    for kind, _, _ in DISCRETE:
        out[kind] = [(discrete_minute(kind, s), Params()) for s in range(6)]
    rng = np.random.default_rng(1)
    for _ in range(700):
        family, quantized, x = draw_signal(rng)
        _, mx, nf = configs(rng)
        out["random " + family + (" quantized" if quantized else "")].append(
            (x, Params(max_quantize_bits=mx, noise_floor_sigma=nf)))
    return out


def bits(items, settings):
    default = Params().effort
    saved = _compress.EFFORTS[default]
    _compress.EFFORTS[default] = settings
    try:
        return 8 * sum(len(fluxcode.encode(x, p)[0][0]) for x, p in items) / sum(len(x) for x, _ in items)
    finally:
        _compress.EFFORTS[default] = saved


if __name__ == "__main__":
    rows = [(name, len(items), *(bits(items, s) for s in SETTINGS.values())) for name, items in input_sets().items()]
    rows.sort(key=lambda r: r[3] - r[2], reverse=True)
    print(f"{'input':38s} {'groups':>5s} " + " ".join(f"{k:>9s}" for k in SETTINGS) + "   2 - before   5 - before")
    for name, n, before, e2, e5 in rows:
        print(f"{name:38s} {n:5d} {before:9.3f} {e2:9.3f} {e5:9.3f}   {e2 - before:+8.3f}   {e5 - before:+8.3f}")
