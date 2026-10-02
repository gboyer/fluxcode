# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Is a higher zstd level ever larger per unit? Each unit (the report's signals, 5 minutes each, and
600 random-family units) at the default effort, at effort 6 (both layouts, zstd 3), and with zstd 7
or 9 alone instead of 3; then keeping the smaller of zstd 3 and the higher level (effort 9's rule).

    uv run python plane_layout/zstd_levels.py       # ~3 min
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fluxcode
from _series import planes
from _signals import DISCRETE, KINDS, discrete_minute, minute
from corpus import configs, draw_signal
from fluxcode import Params

COLUMNS = {"effort 5": None, "effort 6": (3,), "zstd 7 alone": (7,), "zstd 9 alone": (9,), "3 and 7": (3, 7), "3 and 9": (3, 9)}


def size(x, params, levels):
    if levels is None:
        return len(fluxcode.encode(x, params)[0][0])
    with planes("best", zstd_levels=levels):
        return len(fluxcode.encode(x, params)[0][0])


if __name__ == "__main__":
    items = [(minute(n, s) if n in KINDS else discrete_minute(n, s), Params())
             for n in KINDS + [n for n, _, _ in DISCRETE] for s in range(1, 6)]
    report = len(items)
    rng = np.random.default_rng(11)
    for _ in range(600):
        _, _, x = draw_signal(rng)
        _, mx, nf = configs(rng)
        items.append((x, Params(max_quantize_bits=mx, noise_floor_sigma=nf)))
    sz = np.array([[size(x, p, levels) for levels in COLUMNS.values()] for x, p in items], float)
    for label, part in (("report", sz[:report]), ("random", sz[report:])):
        ref = part[:, 1]
        print(f"\n{label} ({len(part)} units), against effort 6 (zstd 3):")
        for j, name in enumerate(COLUMNS):
            r = part[:, j] / ref - 1
            print(f"  {name:13s} total {100 * (part[:, j].sum() / ref.sum() - 1):+6.2f}%   larger on {np.mean(r > 0):4.0%} "
                  f"of units, worst {100 * r.max():+6.1f}%")
