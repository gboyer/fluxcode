# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Where do the value bits go, and how much would fitted (LPC) predictors save?

Per signal: today's size; the zeroth-order entropy of today's residuals per block (what an ideal
static coder with a free model would pay); and per block the best of the fixed orders and integer
LPC of orders P (charged 2 bytes per coefficient + 1), through the real back end.

    uv run python value/predictors.py [fast|slow|synth ...]
"""

import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import corpus
import harness as H

P_LIST = (2, 4, 8, 16, 32)


def analyze(item):
    set_name, name, x, block_len, bpg = item
    params = H.default_params()
    out = dict(set=set_name, name=name, n=len(x), base=0, h_cur=0.0, lpc=0, lpc_side=0, h_lpc=0.0,
               fixed_best=0, orders=np.zeros(4, int), lpc_orders={}, h_ideal_fixed=0.0)
    for part, sizes in H.groups_of(x, block_len, bpg):
        rows = H.encode_rows(part, sizes, params)
        out["base"] += H.size_with(rows, [rows.residuals[o:o + s] for o, s in zip(np.cumsum(sizes) - sizes, sizes)])
        qs = H.block_q(rows)
        cur, fixed_best, lpc_best = [], [], []
        for b, q in enumerate(qs):
            order = int(rows.block_flags[b]) & 3
            out["orders"][order] += 1
            r_cur = H.fixed_residual(q, order)
            h_cur = H.entropy_bits(r_cur)
            out["h_cur"] += h_cur
            cands = [(H.entropy_bits(H.fixed_residual(q, k)), H.fixed_residual(q, k), 0, f"d{k}") for k in range(4)]
            hf = min(cands, key=lambda c: c[0])
            out["h_ideal_fixed"] += hf[0]
            fixed_best.append(hf[1])
            best = (h_cur, r_cur, 0, f"d{order}")
            if len(q) >= 256:
                for p in P_LIST:
                    res = H.lpc_best(q, p)
                    if res is None:
                        continue
                    r, ci, shift = res
                    side = 8 * (2 * p + 1)
                    h = H.entropy_bits(r) + side
                    if h < best[0] + best[2]:
                        best = (h - side, r, side, f"p{p}")
            out["h_lpc"] += best[0] + best[2]
            out["lpc_side"] += best[2]
            out["lpc_orders"][best[3]] = out["lpc_orders"].get(best[3], 0) + 1
            lpc_best.append(best[1])
        out["fixed_best"] += H.size_with(rows, fixed_best)
        out["lpc"] += H.size_with(rows, lpc_best)
    return out


def main():
    sets = sys.argv[1:] or ["fast", "slow", "synth"]
    items = corpus.corpus(sets)
    with ProcessPoolExecutor(6) as ex:
        results = list(ex.map(analyze, items))
    hdr = ("set", "signal", "today", "H(today)", "best fixed", "LPC+side", "H(LPC)+side", "LPC side", "orders d0-3",
           "LPC picks")
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    tot = {}
    for r in results:
        n = r["n"]
        b = lambda v: 8 * v / n  # noqa: E731
        row = [r["set"], r["name"], f"{b(r['base']):.3f}", f"{r['h_cur'] / n:.3f}", f"{b(r['fixed_best']):.3f}",
               f"{(8 * r['lpc'] + r['lpc_side']) / n:.3f}", f"{r['h_lpc'] / n:.3f}", f"{r['lpc_side'] / n:.3f}",
               "/".join(map(str, r["orders"])),
               " ".join(f"{k}:{v}" for k, v in sorted(r["lpc_orders"].items()))]
        print("| " + " | ".join(row) + " |")
        t = tot.setdefault(r["set"], [0, 0, 0, 0])
        t[0] += n
        t[1] += 8 * r["base"]
        t[2] += 8 * r["lpc"] + r["lpc_side"]
        t[3] += 8 * r["fixed_best"]
    for s, (n, base, lpc, fb) in tot.items():
        print(f"{s}: today {base / n:.3f}  best-fixed {fb / n:.3f}  LPC {lpc / n:.3f}  ({100 * (lpc / base - 1):+.1f}%)")


if __name__ == "__main__":
    main()
