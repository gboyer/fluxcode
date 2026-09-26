# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""How much does a decimal-detection near miss slow encode down?

A block on a decimal grid except for its last sample makes each candidate exponent scan the whole
block before failing, instead of stopping at its first off-grid sample (docs/SPEC.md §3.2).

Method: single thread, `encode_unit` on one minute (60 blocks) of the `random-walk` signal
(seed 7), best of 5 x 200 calls after a warm-up, reported in µs per block. The grids are that walk
rounded to 0.01, and the walk x100 rounded to integers; "off-grid" adds 0.3 of a grid step to the
last sample of every block. Prints a markdown table.

    uv run python bench/decimal_near_miss.py
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from _signals import minute

from fluxcode import Params, encode_unit

BLOCK = 1000
REPS, ROUNDS, WARMUP = 200, 5, 20
DEFAULTS = Params()
NO_DETECTION = Params(decimal_detection=False)


def off_grid(x, step):
    """x with the last sample of every block moved 0.3 of a grid step off the grid."""
    y = x.copy()
    y[BLOCK - 1 :: BLOCK] += 0.3 * step
    return y


def us_per_block(x, params):
    for _ in range(WARMUP):
        encode_unit(x, params)
    best = float("inf")
    for _ in range(ROUNDS):
        t0 = time.perf_counter()
        for _ in range(REPS):
            encode_unit(x, params)
        best = min(best, (time.perf_counter() - t0) / REPS)
    return best / (x.size // BLOCK) * 1e6


def main():
    walk = minute("random-walk", 7)
    cents = np.round(walk * 100) / 100
    integers = np.round(walk * 100)
    cases = [
        ("random walk", walk, DEFAULTS),
        ("random walk, `decimal_detection=False`", walk, NO_DETECTION),
        ("on a 0.01 grid", cents, DEFAULTS),
        ("integers", integers, DEFAULTS),
        ("integers, off-grid last sample", off_grid(integers, 1.0), DEFAULTS),
        ("0.01 grid, off-grid last sample", off_grid(cents, 0.01), DEFAULTS),
        ("the same, `decimal_detection=False`", off_grid(cents, 0.01), NO_DETECTION),
    ]
    print("| input | µs/block |")
    print("|---|---|")
    for name, x, params in cases:
        print(f"| {name} | {us_per_block(x, params):.2f} |")


if __name__ == "__main__":
    main()
