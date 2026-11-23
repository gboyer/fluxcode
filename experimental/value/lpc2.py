"""Per-block integer LPC with a fixed-order warm-up (the block's own best fixed order on the first p samples),
coefficients charged COEF_BITS each + 1 byte; p chosen per block by estimated bits (n/2 log2 var + side)."""
import sys, os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import corpus, harness as H
from pickwin2 import windows

COEF_BITS = int(os.environ.get("COEF_BITS", "14"))
P_LIST = (2, 4, 8, 16, 32)

def lpc_res(q, p, warm_order):
    fit = H.fit_lpc(q, p, COEF_BITS)
    if fit is None:
        return None
    ci, shift = fit
    r = H.lpc_residual(q, ci, shift, 0)
    r[:p] = H.fixed_residual(q[:p], warm_order)
    return H.wrap16(r)

def est_bits(r):
    v = np.var(r[32:].astype(float)) + 1 / 12
    return 0.5 * len(r) * np.log2(v)

def run(item):
    s, name, x, bl, bpg = item
    params = H.default_params()
    base = lpc = n = side = 0
    picks = {}
    for part, sizes in H.groups_of(x, bl, bpg):
        rows = H.encode_rows(part, sizes, params)
        qs = H.block_q(rows)
        b0, b1 = [], []
        for b, q in enumerate(qs):
            order = int(rows.block_flags[b]) & 3
            r0 = H.fixed_residual(q, order)
            b0.append(r0)
            best = (est_bits(r0), r0, 0, f"d{order}")
            if len(q) >= 256:
                for p in P_LIST:
                    r = lpc_res(q, p, order)
                    if r is None:
                        continue
                    sd = COEF_BITS * p + 8
                    e = est_bits(r) + sd
                    if e < best[0]:
                        best = (e, r, sd, f"p{p}")
            b1.append(best[1]); side += best[2]
            picks[best[3]] = picks.get(best[3], 0) + 1
        n += sizes.sum()
        base += 8 * H.size_with(rows, b0)
        lpc += 8 * H.size_with(rows, b1)
    return s, name, n, base, lpc + side, side, picks

if __name__ == "__main__":
    with ProcessPoolExecutor(6) as ex:
        res = list(ex.map(run, corpus.corpus(sys.argv[1:] or ["fast", "slow", "synth"])))
    agg = {}
    for s, name, n, base, lpc, side, picks in res:
        agg.setdefault(s, []).append(lpc / base)
        pk = " ".join(f"{k}:{v}" for k, v in sorted(picks.items(), key=lambda kv: -kv[1])[:4])
        print(f"| {s} | {name} | {base/n:.3f} | {lpc/n:.3f} | {100*(lpc/base-1):+.1f}% | {side/n:.3f} | {pk} |")
    for s, v in agg.items():
        print(s, f"{100*(np.exp(np.mean(np.log(v)))-1):+.1f}%")
