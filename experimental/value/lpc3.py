"""LPC on the k0-th difference (k0 = 1 or 2) with ridge, integer residuals; warm-up = fixed order."""
import sys, os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import corpus, harness as H

COEF_BITS = int(os.environ.get("COEF_BITS", "12"))
RIDGE = float(os.environ.get("RIDGE", "1e-3"))
P_LIST = (2, 4, 8, 16)

def fit(d, p):
    x = d.astype(float)
    A = np.lib.stride_tricks.sliding_window_view(x, p)[:-1][:, ::-1]
    y = x[p:]
    ata = A.T @ A
    ata += RIDGE * np.trace(ata) / p * np.eye(p) + 1e-9 * np.eye(p)
    c = np.linalg.solve(ata, A.T @ y)
    cmax = np.abs(c).max()
    if not np.isfinite(cmax) or cmax == 0:
        return None
    shift = max(0, min(30, int(np.floor(np.log2((2 ** (COEF_BITS - 1) - 1) / cmax)))))
    return np.round(c * 2.0 ** shift).astype(np.int64), shift

def residual(q, k0, p):
    d = np.diff(q, k0)  # d[i] = k0-th difference ending at q[i + k0]
    f = fit(d, p)
    if f is None:
        return None
    ci, shift = f
    rd = H.lpc_residual(d, ci, shift, 0)  # residual of d beyond its first p values
    r = H.fixed_residual(q, k0).copy()   # warm-up: the fixed order k0 residuals
    r[k0 + p:] = rd[p:]
    return H.wrap16(r)

def est_bits(r):
    return 0.5 * len(r) * np.log2(np.var(r[40:].astype(float)) + 1 / 12)

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
                for k0 in (1, 2):
                    for p in P_LIST:
                        r = residual(q, k0, p)
                        if r is None:
                            continue
                        sd = COEF_BITS * p + 8
                        e = est_bits(r) + sd
                        if e < best[0]:
                            best = (e, r, sd, f"k{k0}p{p}")
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
        pk = " ".join(f"{k}:{v}" for k, v in sorted(picks.items(), key=lambda kv: -kv[1])[:3])
        print(f"| {s} | {name} | {base/n:.3f} | {lpc/n:.3f} | {100*(lpc/base-1):+.1f}% | {side/n:.3f} | {pk} |")
    for s, v in agg.items():
        print(s, f"{100*(np.exp(np.mean(np.log(v)))-1):+.1f}%")
