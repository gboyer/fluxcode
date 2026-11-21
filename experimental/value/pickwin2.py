"""Order pick from W evenly spread windows of L samples (budget ~ W*L), variance vs robust statistics."""
import sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import corpus, harness as H

def windows(r, W, L):
    n = len(r)
    if W * L >= n - 3:
        return r[3:]
    starts = (np.arange(W) * (n - 3 - L) // max(W - 1, 1)) + 3
    return np.concatenate([r[s:s + L] for s in starts])

def stat_var(v):
    return np.var(v.astype(float))

def stat_clipvar(v):  # variance after clipping at 4 robust sd (MAD)
    v = v.astype(float); v = v - np.median(v)
    s = 1.2533 * np.mean(np.abs(v)) + 1e-12
    return np.var(np.clip(v, -4 * s, 4 * s))

def stat_mad(v):
    v = v.astype(float)
    return np.mean(np.abs(v - v.mean()))

POLICIES = {
    "first250 var": (1, 250, stat_var),
    "4x64 var": (4, 64, stat_var),
    "8x32 var": (8, 32, stat_var),
    "16x16 var": (16, 16, stat_var),
    "8x32 clipvar": (8, 32, stat_clipvar),
    "8x32 meanabsdev": (8, 32, stat_mad),
    "whole var": (1, 10**9, stat_var),
    "whole clipvar": (1, 10**9, stat_clipvar),
}

def run(item):
    s, name, x, bl, bpg = item
    params = H.default_params()
    tot = {p: 0 for p in POLICIES}
    n = 0
    for part, sizes in H.groups_of(x, bl, bpg):
        rows = H.encode_rows(part, sizes, params)
        qs = H.block_q(rows)
        rs = [[H.fixed_residual(q, k) for k in range(4)] for q in qs]
        n += sizes.sum()
        for p, (W, L, st) in POLICIES.items():
            chosen = []
            for b, q in enumerate(qs):
                if len(q) <= 8:
                    chosen.append(rs[b][int(rows.block_flags[b]) & 3]); continue
                chosen.append(min(rs[b], key=lambda r: st(windows(r, W, L))))
            tot[p] += H.size_with(rows, chosen)
    return s, name, n, tot

if __name__ == "__main__":
    with ProcessPoolExecutor(6) as ex:
        res = list(ex.map(run, corpus.corpus(sys.argv[1:] or ["fast", "slow", "synth"])))
    agg = {}
    worst = {}
    for s, name, n, tot in res:
        for p in POLICIES:
            r = tot[p] / tot["first250 var"]
            agg.setdefault((s, p), []).append(r)
            if r > worst.get(p, (0, ""))[0]:
                worst[p] = (r, name)
    for s in dict.fromkeys(r[0] for r in res):
        print(s, " | ".join(f"{p}: {100*(np.exp(np.mean(np.log(agg[(s,p)])))-1):+.2f}%" for p in POLICIES))
    print("worst per policy:", {p: f"{100*(w-1):+.1f}% {nm}" for p, (w, nm) in worst.items()})
