# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""The per-block-group target's size estimate (class entropy of the residual, _encoder.estimate_bits)
against the actual bit-shuffled zstd-3 size, per signal type, B = 16 (min = max = 16, noise
floor off), and what the target then delivers.

    uv run python bench/estimate.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from _series import decode_series, encode_series
from _signals import DISCRETE, KINDS, discrete_minute, minute

from fluxcode import Params, _compress, _encoder, _group

B16 = Params(min_quantize_bits=16, noise_floor_sigma=0)


def estimate_and_actual(x, params):
    sizes = np.full(x.size // 1000, 1000)
    rows, _ = _group.encode_rows(x, sizes, params)
    est = np.mean([_encoder.estimate_bits(r) for r in rows.residuals.reshape(-1, 1000)])
    return est, 8 * len(_compress.compress(rows, x.size, _compress.Effort("bit", False, (3,)))) / x.size


def main(minutes=5):
    sets = {k: [minute(k, 100 + j) for j in range(minutes)] for k in KINDS}
    sets.update({n: [discrete_minute(n, 200 + j) for j in range(minutes)] for n, _, _ in DISCRETE})
    targets = (6.0, 8.0, 10.0)
    print("| signal | estimate | actual | " + " | ".join(f"target {t:g}: bits/sample, worst max err % range"
                                                     for t in targets) + " |")
    print("|---|---|---|" + "---|" * len(targets))
    for name, xs in sets.items():
        ea = np.array([estimate_and_actual(x, B16) for x in xs]).mean(0)
        cells = []
        for t in targets:
            p = Params(noise_floor_sigma=0, target_bits_per_sample=t)
            bits, worst = 0, 0.0
            for x in xs:
                groups, lo, hi, _ = encode_series(x, p)
                bits += 8 * sum(map(len, groups))
                err = np.abs(decode_series(groups) - x).reshape(-1, 1000).max(1)
                worst = max(worst, (err / np.where(hi > lo, hi - lo, 1)).max())
            cells.append(f"{bits / sum(map(len, xs)):.2f}, {100 * worst:.3f}")
        print(f"| {name} | {ea[0]:.2f} | {ea[1]:.2f} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
