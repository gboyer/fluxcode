"""Order-0 blocks with residuals centered on the block (q - round(median q)): what would a signed-q format save?
Also centering the first residual of higher orders (r[0] = q[0] - c)."""
import sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import corpus, harness as H

def run(item):
    s, name, x, bl, bpg = item
    params = H.default_params()
    base = cen = n = 0
    n0 = 0
    for part, sizes in H.groups_of(x, bl, bpg):
        rows = H.encode_rows(part, sizes, params)
        qs = H.block_q(rows)
        b0, b1 = [], []
        for b, q in enumerate(qs):
            order = int(rows.block_flags[b]) & 3
            r = H.fixed_residual(q, order)
            b0.append(r)
            if len(q) == 0:
                b1.append(r); continue
            lo_c, hi_c = int(q.max()) - 32767, int(q.min()) + 32768  # every q - c must fit int16
            rc = r.copy()
            if order == 0:
                c = min(max(int(np.median(q)), lo_c), hi_c)
                rc = H.wrap16(q - c); n0 += 1
            else:
                c = min(max(int(q[0]), lo_c), hi_c)
                rc[0] = H.wrap16(q[0] - c)  # the first residual is q[0] - c: 0 unless the range is wide
            b1.append(rc)
        n += sizes.sum()
        base += H.size_with(rows, b0)
        cen += H.size_with(rows, b1)
    return s, name, n, base, cen, n0

if __name__ == "__main__":
    with ProcessPoolExecutor(6) as ex:
        res = list(ex.map(run, corpus.corpus(sys.argv[1:] or ["fast", "slow", "synth"])))
    agg = {}
    for s, name, n, base, cen, n0 in res:
        agg.setdefault(s, []).append(cen / base)
        if abs(cen / base - 1) > 0.003:
            print(f"{s:5s} {name:28s} {8*base/n:.3f} -> {8*cen/n:.3f} ({100*(cen/base-1):+.1f}%)  order-0 blocks {n0}")
    for s, v in agg.items():
        print(s, f"{100*(np.exp(np.mean(np.log(v)))-1):+.2f}%")
