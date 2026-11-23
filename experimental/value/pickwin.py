"""Fixed orders only: today's pick (first 250 samples) against variance over the whole block and against the
smallest-entropy order; actual sizes."""
import sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import corpus, harness as H

def run(item):
    s, name, x, bl, bpg = item
    params = H.default_params()
    out = np.zeros(4)
    for part, sizes in H.groups_of(x, bl, bpg):
        rows = H.encode_rows(part, sizes, params)
        qs = H.block_q(rows)
        today, whole, ent = [], [], []
        for b, q in enumerate(qs):
            rs = [H.fixed_residual(q, k) for k in range(4)]
            today.append(rs[int(rows.block_flags[b]) & 3])
            whole.append(min(rs, key=lambda r: np.var(r[3:].astype(float))))
            ent.append(min(rs, key=lambda r: H.entropy_bits(r)))
        out += [sizes.sum(), H.size_with(rows, today), H.size_with(rows, whole), H.size_with(rows, ent)]
    return s, name, out

if __name__ == "__main__":
    with ProcessPoolExecutor(6) as ex:
        res = list(ex.map(run, corpus.corpus(sys.argv[1:] or ["fast", "slow", "synth"])))
    agg = {}
    for s, name, (n, a, b, c) in res:
        agg.setdefault(s, []).append((b / a, c / a))
        if abs(b / a - 1) > 0.005 or abs(c / a - 1) > 0.005:
            print(f"{s:5s} {name:28s} today {8*a/n:.3f}  whole-variance {100*(b/a-1):+.1f}%  min-entropy {100*(c/a-1):+.1f}%")
    for s, v in agg.items():
        v = np.array(v); g = np.exp(np.log(v).mean(0)) - 1
        print(s, f"whole-variance {100*g[0]:+.2f}%  min-entropy {100*g[1]:+.2f}%")
