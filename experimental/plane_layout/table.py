# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Per-group table for combining the layout heuristic with block flushing: the real compressed
block group size (header included) of each layout (bit planes, byte planes) under each framing (one
block run as today, a flush after every plane / every 4 planes / every 8 planes, and the encoder's
rule: a flush after each plane with more than 1/16 of its bytes non-zero), plus the features a
one-pass rule could use. Same signals as corpus.py (same seeds) and the report's.

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
from _series import planes as forced_layout
from flush_blocks import reflush
from fluxcode import Params, _compress, _format, _group

EVERY = (1, 4, 8)


def pooled(group):
    """Block group size with the body in one block run (what the group-level encoder did before flushing)."""
    body = _group.decompress(group).raw_body
    return _format.HEADER_BYTES + len(_compress.zstd()[0].compress(body.data))


def dense(group):
    """The block group re-framed with the encoder's rule (no small-group retry)."""
    parsed = _group.decompress(group)
    cuts = _compress.flush_points(parsed.raw_body, parsed.header.num_blocks, parsed.layout, parsed.has_time, parsed.header.byte_planes)
    return group[:_format.HEADER_BYTES] + _compress.compress_body(parsed.raw_body, cuts, 3)


def row(x, mx=16, nf=0.25):
    groups = {}
    for m in ("bit", "byte"):
        with forced_layout(m, flush=False):
            groups[m] = fluxcode.encode(x, Params(max_quantize_bits=mx, noise_floor_sigma=nf))[0][0]
    sizes = {m: [pooled(groups[m])] + [len(reflush(groups[m], e)) for e in EVERY] + [len(dense(groups[m]))] for m in groups}
    parsed = _group.decompress(groups["bit"])
    start = _format.residual_start(parsed.header.num_blocks, parsed.has_time)
    g = int(parsed.layout.octet_offsets[-1])
    planes = parsed.raw_body[start:start + 16 * g]
    u = residuals(groups["byte"])
    return (sizes["bit"] + sizes["byte"], dict(hi_nz=float(np.mean(u > 255)), nonzero_bytes=int(np.count_nonzero(planes)),
            body=len(parsed.raw_body), raw_planes=16 * g))


def build(rows, name):
    sizes = np.array([r[0] for r in rows], float)
    feats = {k: np.array([r[1][k] for r in rows]) for k in rows[0][1]}
    np.savez(Path(__file__).with_name(name), sizes=sizes, **feats)
    print(name, len(rows), "block groups")


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
