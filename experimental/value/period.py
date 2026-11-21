"""Long-term (period) prediction: r = q[n] - interp(q, n - P - frac), optionally followed by a first difference.
P searched per block on the residual energy over integer periods, then a fractional offset in 1/8 steps.
Side info charged 3 bytes per block that uses it. Sizes through the real back end."""
import sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import corpus, harness as H

FRAC = 8  # fractional delay resolution

def period_residual(q, P, f8):
    """pred[n] = ((FRAC - f8) * q[n-P] + f8 * q[n-P-1] + FRAC/2) // FRAC for n >= P+1; warm-up: first difference."""
    q = q.astype(np.int64)
    n = len(q)
    r = H.fixed_residual(q, 1).astype(np.int64)
    if n > P + 1:
        pred = ((FRAC - f8) * q[1:n - P] + f8 * q[0:n - P - 1] + FRAC // 2) // FRAC
        r[P + 1:] = q[P + 1:] - pred
    return r

def best_period(q, pmin=8, pmax=None):
    q = q.astype(float)
    n = len(q)
    pmax = pmax or n // 3
    d = q - q.mean()
    # residual energy of integer periods (direct; prototype)
    best = None
    for P in range(pmin, pmax):
        e = np.mean((d[P:] - d[:-P]) ** 2)
        if best is None or e < best[0]:
            best = (e, P)
    return best[1]

def run(item):
    s, name, x, bl, bpg = item
    params = H.default_params()
    base = lt = n = 0
    used = 0
    for part, sizes in H.groups_of(x, bl, bpg):
        rows = H.encode_rows(part, sizes, params)
        qs = H.block_q(rows)
        b0, b1 = [], []
        side = 0
        for b, q in enumerate(qs):
            order = int(rows.block_flags[b]) & 3
            q = q.astype(np.int64)
            q = np.where(q >= 32768, q - 65536, q)  # signed (centered format)
            r0 = H.fixed_residual(q, order)
            b0.append(r0)
            best = (np.var(r0[8:].astype(float)), r0, 0)
            if len(q) >= 64:
                P = best_period(q)
                for Pb in (P - 1, P):
                    if Pb < 2 or len(q) < Pb + 20:
                        continue
                    # 2 taps at n-Pb, n-Pb-1, least squares, quantized to 1/64
                    y = q[Pb + 1:].astype(float); A = np.c_[q[1:len(q) - Pb], q[0:len(q) - Pb - 1]].astype(float)
                    c, *_ = np.linalg.lstsq(A, y, rcond=None)
                    ci = np.round(c * 64).astype(np.int64)
                    pred = (ci[0] * q[1:len(q) - Pb] + ci[1] * q[0:len(q) - Pb - 1] + 32) >> 6
                    r = H.fixed_residual(q, 1).astype(np.int64)
                    r[Pb + 1:] = q[Pb + 1:] - pred
                    for d1 in (False, True):
                        rr = np.r_[r[:1], np.diff(r)] if d1 else r
                        v = np.var(rr[Pb + 2:].astype(float))
                        if v < best[0] * 0.9:  # needs a clear gain to pay its side info
                            best = (v, H.wrap16(rr), 1)
            b1.append(best[1]); side += 24 * best[2]; used += best[2]
        n += sizes.sum()
        base += 8 * H.size_with(rows, b0)
        lt += 8 * H.size_with(rows, b1) + side
    return s, name, n, base, lt, used

if __name__ == "__main__":
    with ProcessPoolExecutor(6) as ex:
        res = list(ex.map(run, corpus.corpus(sys.argv[1:] or ["fast", "slow", "synth"])))
    agg = {}
    for s, name, n, base, lt, used in res:
        agg.setdefault(s, []).append(lt / base)
        print(f"{s:5s} {name:28s} {base/n:.3f} -> {lt/n:.3f} ({100*(lt/base-1):+.1f}%) blocks using it: {used}")
    for s, v in agg.items():
        print(s, f"{100*(np.exp(np.mean(np.log(v)))-1):+.2f}%")
