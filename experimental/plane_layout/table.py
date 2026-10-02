# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Per-unit table for combining the layout heuristic with block flushing: the real compressed
unit size (header included) of each layout (bit planes, byte planes) under each framing (one
block run as today, a flush after every plane / every 4 planes / every 8 planes), plus the
features a one-pass rule could use. Same signals as corpus.py (same seeds) and the report's.

    uv run python plane_layout/table.py     # writes table_fit.npz, table_test.npz, table_std.npz (~2 min)
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
import fluxcode
from corpus import configs, draw_signal, residuals
from flush_blocks import reflush
from fluxcode import Params, _format, _unit

EVERY = (1, 4, 8)


def row(x, mx=16, nf=0.25):
    units = {m: fluxcode.encode(x, Params(max_quantize_bits=mx, noise_floor_sigma=nf, planes=m))[0][0] for m in ("bit", "byte")}
    sizes = {m: [len(units[m])] + [len(reflush(units[m], e)) for e in EVERY] for m in units}
    parsed = _unit.decompress(units["bit"])
    start = _format.residual_start(parsed.header.num_blocks, parsed.has_time)
    g = int(parsed.layout.group_offsets[-1])
    planes = parsed.raw_body[start:start + 16 * g]
    u = residuals(units["byte"])
    return (sizes["bit"] + sizes["byte"], dict(hi_nz=float(np.mean(u > 255)), nonzero_bytes=int(np.count_nonzero(planes)),
            body=len(parsed.raw_body), raw_planes=16 * g))


def build(rows, name):
    sizes = np.array([r[0] for r in rows], float)
    feats = {k: np.array([r[1][k] for r in rows]) for k in rows[0][1]}
    np.savez(Path(__file__).with_name(name), sizes=sizes, **feats)
    print(name, len(rows), "units")


if __name__ == "__main__":
    for seed, n, name in ((1, 2000, "table_fit.npz"), (2, 1000, "table_test.npz")):
        rng = np.random.default_rng(seed)
        rows = []
        while len(rows) < n:
            fam, q, x = draw_signal(rng)
            _, mx, nf = configs(rng)
            rows.append(row(x, mx, nf))
        build(rows, name)
    from _signals import DISCRETE, KINDS, discrete_minute, minute
    names = KINDS + [n for n, _, _ in DISCRETE]
    build([row(minute(n, 500 + j) if n in KINDS else discrete_minute(n, 500 + j)) for n in names for j in range(8)], "table_std.npz")
