# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Where the study's first heuristic layout rule ("byte planes iff fewer than 5% of residuals reach
256") misses, and two fixes tried. Per block group: the size with bit planes and with byte planes, block
flushed, and the share of residuals reaching 2^k. Against the better of the two layouts:

- a narrow-residual exception (keep bit planes when few residuals reach 2^k, k = 1..4), for the
  case the high-byte rule can't see (narrow noise, near-constant planes): worse in every variant;
- also compressing the other layout when the picked one's frame is under X bytes, and keeping
  the smaller: adopted at X = 16 KB at first, then dropped (see fleet.py: on a sensor mix of small
  block groups the extra zstd passes cost 14% of encode time for 0.02%, after the rule moved to 128).

"total" is the extra size over the better layout summed over block groups, "geo" the geometric mean of the
per-group ratio. Same random families and seeds as corpus.py, plus the report's signals.

    uv run python plane_layout/small_frames.py      # ~20 s
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fluxcode
from _series import planes, group_rows
from _signals import DISCRETE, KINDS, discrete_minute, minute
from corpus import configs, draw_signal
from fluxcode import Params

SHARE_BITS = range(9)  # columns: share of residuals >= 2^k


def row(x, params):
    sizes = []
    for layout in ("bit", "byte"):
        with planes(layout):  # flushed
            sizes.append(len(fluxcode.encode(x, params)[0][0]))
    r = group_rows(fluxcode.encode(x, params)[0][0]).residuals.astype(np.int32)
    u = ((r << 1) ^ (r >> 15)) & 0xFFFF
    return sizes + [float(np.mean(u >= 2 ** k)) for k in SHARE_BITS]


def table():
    out = {}
    for seed, n, name in ((1, 1500, "fit"), (2, 700, "held out")):
        rng = np.random.default_rng(seed)
        rows = []
        for _ in range(n):
            _, _, x = draw_signal(rng)
            _, mx, nf = configs(rng)
            rows.append(row(x, Params(max_quantize_bits=mx, noise_floor_sigma=nf)))
        out[name] = np.array(rows)
    names = KINDS + [n for n, _, _ in DISCRETE]
    out["report"] = np.array([row(minute(n, 500 + j) if n in KINDS else discrete_minute(n, 500 + j), Params())
                              for n in names for j in range(6)])
    return out


if __name__ == "__main__":
    for name, d in table().items():
        bit, byte, share = d[:, 0], d[:, 1], d[:, 2:]
        oracle = np.minimum(bit, byte)
        stats = lambda c: f"total {100 * (c.sum() / oracle.sum() - 1):+5.2f}%  geo {100 * np.mean(np.log(c / oracle)):+5.2f}%"
        picked_byte = share[:, 8] < 0.05  # the first rule
        picked = np.where(picked_byte, byte, bit)
        print(f"\n{name} ({len(d)} block groups). heuristic: {stats(picked)}")
        print("  narrow-residual exception, byte planes only if also share(u >= 2^k) >= t:")
        for k in (1, 2, 3, 4):
            print(f"    k = {k}: " + "   ".join(f"t {t}: {stats(np.where(picked_byte & (share[:, k] >= t), byte, bit))}"
                                             for t in (0.02, 0.1, 0.3)))
        print("  also the other layout when the picked frame is under X bytes:")
        for limit in (2000, 4000, 8000, 16384, 32768):
            small = picked < limit
            print(f"    X = {limit:5d}: {stats(np.where(small, oracle, picked))}  second compressions on {small.mean():.0%} "
                  f"of block groups ({picked[small].sum() / picked.sum():.1%} of bytes)")
