"""How far is planes+zstd from a static per-group symbol model on today's residuals?
H_group: empirical entropy of the zigzagged residuals over the whole group (model cost not charged)."""
import sys
import numpy as np
import corpus, harness as H
import fluxcode
from fluxcode import _compress

def run(item):
    set_name, name, x, bl, bpg = item
    params = fluxcode.Params()
    tot = dict(n=0, bytes=0, h=0.0, hb=0.0, bit=0, byte=0)
    for part, sizes in H.groups_of(x, bl, bpg):
        rows = H.encode_rows(part, sizes, params)
        n = int(sizes.sum())
        tot["n"] += n
        tot["bytes"] += len(_compress.compress(rows, n, _compress.EFFORTS[4], 0))
        r = rows.residuals.astype(np.int64)
        u = ((r << 1) ^ (r >> 63)) & 0xFFFF
        tot["h"] += H.entropy_bits(u)
        lo, hi = u & 0xFF, u >> 8
        tot["hb"] += H.entropy_bits(lo) + H.entropy_bits(hi)
    return set_name, name, tot

if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(6) as ex:
        res = list(ex.map(run, corpus.corpus(sys.argv[1:] or ["fast", "slow", "synth"])))
    print("| set | signal | actual | H(u) group | H(lo)+H(hi) | gap |")
    print("|---|---|---|---|---|---|")
    for s, name, t in res:
        a = 8 * t["bytes"] / t["n"]; h = t["h"] / t["n"]; hb = t["hb"] / t["n"]
        print(f"| {s} | {name} | {a:.3f} | {h:.3f} | {hb:.3f} | {100 * (a / h - 1) if h > 0.05 else 0:+.0f}% |")
