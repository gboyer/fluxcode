"""Fixed smoothing predictors (no side info): LS polynomial of degree d over the last m samples, extrapolated one
step, in Q12 fixed point. Per block the candidate of least residual variance; actual size through the back end.

    FLUXCODE_X_NOISE_ORDER=4 uv run python value/smoothing.py [sets]
"""
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import corpus
import harness as H

SHIFT = 12


def sg_pred(d, m):
    """Coefficients c[0..m-1] for q[n-1-j] predicting q[n]: LS fit of degree d over the last m samples."""
    t = -np.arange(1, m + 1, dtype=float)  # times of q[n-1], q[n-2], ...
    V = np.vander(t, d + 1, increasing=True)
    # predicted value at t=0 is the constant term of the LS polynomial: e0 . (V^T V)^-1 V^T
    c = np.linalg.solve(V.T @ V, V.T)[0]
    ci = np.round(c * 2 ** SHIFT).astype(np.int64)
    ci[0] += 2 ** SHIFT - ci.sum()  # exact for constants
    return ci


CANDS = {}
for d in range(4):
    CANDS[f"d{d}"] = None  # the shipped exact orders
    for mult in (2, 4, 8):
        m = (d + 1) * mult
        CANDS[f"d{d}m{m}"] = sg_pred(d, m)


def residuals(q, name):
    if CANDS[name] is None:
        return H.fixed_residual(q, int(name[1]))
    c = CANDS[name]
    warm = int(name[1])
    r = H.lpc_residual(q, c, SHIFT, min(warm, 1))
    # the warm-up (first m samples) as the exact order's residuals
    m = len(c)
    r[:m] = H.fixed_residual(q[:m], min(warm, 1))
    return H.wrap16(r)


def run(item):
    set_name, name, x, bl, bpg = item
    params = H.default_params()
    tot = dict(n=0, base=0, sm=0, picks={})
    for part, sizes in H.groups_of(x, bl, bpg):
        rows = H.encode_rows(part, sizes, params)
        qs = H.block_q(rows)
        base, sm = [], []
        for b, q in enumerate(qs):
            order = int(rows.block_flags[b]) & 3
            base.append(H.fixed_residual(q, order))
            best = None
            for c in CANDS:
                r = residuals(q, c)
                v = np.var(r[32:].astype(float)) if len(r) > 64 else np.var(r.astype(float))
                if best is None or v < best[0]:
                    best = (v, c, r)
            sm.append(best[2])
            tot["picks"][best[1]] = tot["picks"].get(best[1], 0) + 1
        tot["n"] += int(sizes.sum())
        tot["base"] += H.size_with(rows, base)
        tot["sm"] += H.size_with(rows, sm)
    return set_name, name, tot


if __name__ == "__main__":
    with ProcessPoolExecutor(6) as ex:
        res = list(ex.map(run, corpus.corpus(sys.argv[1:] or ["fast", "slow", "synth"])))
    print("| set | signal | today | smoothing | change | picks |")
    print("|---|---|---|---|---|---|")
    agg = {}
    for s, name, t in res:
        a, b = 8 * t["base"] / t["n"], 8 * t["sm"] / t["n"]
        agg.setdefault(s, []).append(t["sm"] / t["base"])
        picks = " ".join(f"{k}:{v}" for k, v in sorted(t["picks"].items(), key=lambda kv: -kv[1])[:4])
        print(f"| {s} | {name} | {a:.3f} | {b:.3f} | {100 * (b / a - 1):+.1f}% | {picks} |")
    for s, v in agg.items():
        print(s, f"{100 * (np.exp(np.mean(np.log(v))) - 1):+.1f}%")
