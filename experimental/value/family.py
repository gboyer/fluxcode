"""Which subset of smoothing predictors captures the gain? Picks by variance on the first 250 samples (as
pick_order does); sizes through the back end. Families are lists of candidate names from smoothing.CANDS."""
import sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import corpus, harness as H
from smoothing import CANDS, residuals

FAMILIES = {
    "today": ["d0", "d1", "d2", "d3"],
    "all": list(CANDS),
    "m≤8": [c for c in CANDS if c[2:] == "" or int(c.split("m")[1]) <= 8],
    "d0-2,x4": ["d0", "d1", "d2", "d3", "d0m4", "d1m8", "d2m12"],
    "x2,x4,x8 d≤2": ["d0", "d1", "d2", "d3", "d0m2", "d0m4", "d0m8", "d1m4", "d1m8", "d1m16", "d2m6", "d2m12", "d2m24"],
    "x4 all d": ["d0", "d1", "d2", "d3", "d0m4", "d1m8", "d2m12", "d3m16"],
    "x4,x8 all d": ["d0", "d1", "d2", "d3", "d0m4", "d1m8", "d2m12", "d3m16", "d0m8", "d1m16", "d2m24", "d3m32"],
}
PICK = 250

def run(item):
    set_name, name, x, bl, bpg = item
    params = H.default_params()
    tot = {f: 0 for f in FAMILIES}
    n = 0
    for part, sizes in H.groups_of(x, bl, bpg):
        rows = H.encode_rows(part, sizes, params)
        qs = H.block_q(rows)
        res = {c: [residuals(q, c) for q in qs] for c in CANDS}
        n += int(sizes.sum())
        for f, cands in FAMILIES.items():
            chosen = []
            for b, q in enumerate(qs):
                best = min(cands, key=lambda c: np.var(res[c][b][32:PICK].astype(float)) if len(q) > 64 else 0)
                chosen.append(res[best][b])
            tot[f] += H.size_with(rows, chosen)
    return set_name, name, n, tot

if __name__ == "__main__":
    with ProcessPoolExecutor(6) as ex:
        res = list(ex.map(run, corpus.corpus(sys.argv[1:] or ["fast", "slow", "synth"])))
    agg = {}
    for s, name, n, tot in res:
        for f in FAMILIES:
            agg.setdefault((s, f), []).append(tot[f] / tot["today"])
        print(f"{s:5s} {name:28s} " + " ".join(f"{f}:{100*(tot[f]/tot['today']-1):+.1f}%" for f in FAMILIES if f != "today"))
    for s in dict.fromkeys(r[0] for r in res):
        print(s, " ".join(f"{f}: {100*(np.exp(np.mean(np.log(agg[(s,f)])))-1):+.1f}%" for f in FAMILIES if f != "today"))
