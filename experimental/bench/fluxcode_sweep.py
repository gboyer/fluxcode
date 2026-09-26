# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""How do fluxcode's options (delta orders, decimal detection) trade size against error, B = 9..16?

Method: the real fluxcode package in four versions (order 1 or orders 0-3, with or without decimal
detection) at B = 9..16 on continuous and discretized minutes, and the discretized signals at
B = 16 against the lossless ideal (the same pipeline told the quantum). Single thread; µs per
1000-sample block, best of 3. Sizes are whole units (header, per-block anchors, zstd frame). Errors
are % of the block's range.

Data: tslab.common.datasets.load() (7,200 continuous blocks) and load_discrete() (5,400 blocks of
signals on a fixed quantum, as real data often is).

    uv run python -m bench.fluxcode_sweep
"""

import os

os.environ.setdefault("NUMBA_NUM_THREADS", "1")

from tslab.common.datasets import DISCRETE
from tslab.flux.adapters import VERSIONS, sweep


def main():
    res, ideal = sweep()
    print("## Sweep: four versions, B = 9..16\n")
    print("| version | B | continuous bits/sample | continuous median RMSE | continuous worst max err | "
          "discretized bits/sample | discretized median RMSE | discretized bit-exact samples | encode µs (cont / disc) | decode µs |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for (label, B), (rc, rd) in res.items():
        print(f"| {label} | {B} | {rc['bps']:.2f} | {rc['rmse_med']:.5f}% | {rc['maxerr_max']:.4f}% | {rd['bps']:.2f} | "
              f"{rd['rmse_med']:.5f}% | {100 * rd['exact']:.1f}% | {rc['enc']:.1f} / {rd['enc']:.1f} | {rc['dec']:.1f} |")
    print("\nDiscretized signals at B = 16 (bits/sample):\n")
    print("| signal | ideal (lossless at the quantum) | " + " | ".join(l for l, _ in VERSIONS) + " |")
    print("|---|---|" + "---|" * len(VERSIONS))
    for name, _, _ in DISCRETE:
        print(f"| {name} | {ideal[name]:.2f} | " + " | ".join(f"{res[l, 16][1]['per_signal'][name]:.2f}" for l, _ in VERSIONS) + " |")


if __name__ == "__main__":
    main()
