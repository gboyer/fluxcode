# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Can the small-frame retry (compressing a block-flushed frame again in one block run) be skipped
by a rule decided before compressing? From table.py's per-group sizes (heuristic layout, flushed
after dense planes vs one block run), against always retrying under 16 KB:

- flush only when the residual planes hold at least T non-zero bytes, else one run (one compression);
- retry only when the flushed frame is under X bytes.

    uv run python plane_layout/retry_rules.py       # needs table.py's table_*.npz
"""

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    for name in ("table_fit.npz", "table_test.npz", "table_std.npz"):
        d = np.load(HERE / name)
        s, byte = d["sizes"], d["hi_nz"] < 0.05
        # sizes columns: bit (one run, every 1/4/8 planes, dense), then the same for byte planes
        pooled, dense = np.where(byte, s[:, 5], s[:, 0]), np.where(byte, s[:, 9], s[:, 4])
        retried = np.minimum(pooled, dense)
        base = retried.sum()
        stats = lambda c: f"total {100 * (c.sum() / base - 1):+.3f}%  worst block group {100 * (c / retried - 1).max():+6.1f}%"
        print(f"\n{name} ({len(s)} block groups); never retry: {stats(dense)}")
        for limit in (0, 2000, 8000, 16000):
            print(f"  flush only from {limit:5d} non-zero plane bytes: "
                  f"{stats(np.where(d['nonzero_bytes'] >= limit, dense, pooled))}")
        for limit in (1000, 2000, 4000, 8000):
            small = dense < limit
            print(f"  retry only under {limit:4d} bytes: {stats(np.where(small, retried, dense))}  ({small.mean():.0%} retried)")
        for lo, hi in ((0, 1000), (1000, 4000), (4000, 16384), (16384, np.inf)):
            m = (dense >= lo) & (dense < hi)
            if m.any():
                print(f"  flushed frame {lo}-{hi}: {m.sum()} block groups, one run smaller on {np.mean(pooled[m] < dense[m]):.0%}, "
                      f"mean gain {100 * np.mean(1 - retried[m] / dense[m]):.2f}%")
