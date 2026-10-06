# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""What does the noise floor cost and save in the fluxcode prototype?

Method: fluxproto (orders 0-3, decimal detect, bit-shuffle, B = 16) with the noise floor off and at
f = 0.25 .. 2. The step becomes max(B-bit step, 2^floor(log2(f * sigma))) on blocks whose second
differences look like white noise (lag-1 autocorrelation < -0.6); sigma is a robust MAD
estimate. Scored against the input and against the clean signal (the same generator
without its noise), in units of the true noise sigma and % of range. Then every signal at f = 1
and 2, to check the clean ones are untouched, and the encode-time cost of the estimate.
Sizes are whole block groups (block minima included).

Data: tslab.common.datasets.noisy_sets(), load() and load_discrete().

    uv run python -m bench.fluxproto_noise_floor
"""

import os

os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import time

import numpy as np

from tslab.common.datasets import load, load_discrete, noisy_sets
from tslab.flux import proto as fp
from tslab.flux.adapters import noise_score

F_SWEEP = (0.25, 0.5, 1.0, 2.0)


def main():
    sets = noisy_sets()
    base = dict(step_detect=True, planes="bit")
    print("## Noisy signals: noise floor at step <= f * sigma (B = 16)\n")
    print("Errors: median RMS over blocks, in units of the true noise sigma. 'vs clean' >= 1 means the noise is still there.\n")
    print("| signal | f | bits/sample | RMS error vs input (σ) | RMS error vs clean (σ) | worst max err vs input (% range) |")
    print("|---|---|---|---|---|---|")
    for name, mins in sets.items():
        for f in (0.0,) + F_SWEEP:
            r = noise_score(fp.FluxProto(16, noise_f=f, **base), mins)
            print(f"| {name} | {'off' if f == 0 else f'{f:g}'} | {r['bps']:.2f} | {r['in']:.3f} | {r['clean']:.3f} | {r['max']:.3f}% |",
                  flush=True)
    # false positives and the mixed-set comparison against a lower fixed B
    allm = load() + [m[:4] for m in load_discrete()]
    print("\n## Every signal (7,200 continuous + 5,400 discretized blocks): does the noise floor change the clean ones?\n")
    print("| signal | off | f = 1 | f = 2 |\n|---|---|---|---|")
    tot = {}
    for kind in dict.fromkeys(m[0] for m in allm):
        sub = [m for m in allm if m[0] == kind]
        row = []
        for f in (0.0, 1.0, 2.0):
            c = fp.FluxProto(16, noise_f=f, **base)
            b = sum(8 * len(c.encode_group(m[1])[0]) for m in sub) / (60000 * len(sub))
            row.append(b)
            tot[f] = tot.get(f, 0) + b * len(sub)
        print(f"| {kind} | {row[0]:.2f} | {row[1]:.2f} | {row[2]:.2f} |")
    n = len(allm)
    print(f"| **all** | {tot[0.0] / n:.2f} | {tot[1.0] / n:.2f} | {tot[2.0] / n:.2f} |")
    # timing
    c0, c1 = fp.FluxProto(16, **base), fp.FluxProto(16, noise_f=1.0, **base)
    for c in (c0, c1):
        c.encode_group(allm[0][1])
    best = {}
    for _ in range(5):
        for k, c in (("off", c0), ("f = 1", c1)):
            t0 = time.perf_counter()
            for m in allm:
                c.encode_group(m[1])
            best[k] = min(best.get(k, np.inf), time.perf_counter() - t0)
    print("\nencode µs/block: " + ", ".join(f"{k} {1e6 * v / (60 * n):.2f}" for k, v in best.items()))


if __name__ == "__main__":
    main()
