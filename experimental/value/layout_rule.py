"""Per block group: bit-plane vs byte-plane size (zstd 3) and features of the zigzagged residuals, to find a
better one-pass layout rule than 'byte planes iff < 1% of residuals reach 128'."""
import sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import corpus, harness as H
from fluxcode import _compress
sys.path.insert(0, str(H.ROOT / 'tests'))
from _series import planes

def run(item):
    s, name, x, bl, bpg = item
    out = []
    for part, sizes in H.groups_of(x, bl, bpg):
        rows = H.encode_rows(part, sizes, H.default_params())
        n = int(sizes.sum())
        with planes("bit", flush=False):
            b_bit = len(_compress.compress(rows, n, _compress.EFFORTS[4], 0))
        with planes("byte", flush=False):
            b_byte = len(_compress.compress(rows, n, _compress.EFFORTS[4], 0))
        r = rows.residuals.astype(np.int64); u = ((r << 1) ^ (r >> 63)) & 0xFFFF
        out.append(dict(name=name, set=s, bit=b_bit, byte=b_byte, wide128=np.mean(u >= 128), wide256=np.mean(u >= 256),
                        zero=np.mean(u == 0), mean_len=np.mean(np.log2(u + 1.0)), hi_nonzero=np.mean((u >> 8) != 0)))
    return out

if __name__ == "__main__":
    with ProcessPoolExecutor(6) as ex:
        rows = [r for res in ex.map(run, corpus.corpus(sys.argv[1:] or ["fast", "slow", "synth"])) for r in res]
    import json; json.dump(rows, open(sys.argv[0].replace(".py", ".json"), "w"))
    tot_best = sum(min(r["bit"], r["byte"]) for r in rows)
    def cost(rule):
        return sum(r["byte"] if rule(r) else r["bit"] for r in rows) / tot_best - 1
    print(f"groups {len(rows)}; byte planes win in {sum(r['byte'] < r['bit'] for r in rows)}")
    print(f"today (wide128 < 1%): +{100*cost(lambda r: r['wide128'] < 0.01):.2f}% over best")
    for t in (0.01, 0.05, 0.1, 0.2, 0.3, 0.5):
        print(f"wide256 < {t}: +{100*cost(lambda r, t=t: r['wide256'] < t):.2f}%   hi byte nonzero < {t}: +{100*cost(lambda r, t=t: r['hi_nonzero'] < t):.2f}%")
    for t in (2, 3, 4, 5, 6):
        print(f"mean log2 len between 1 and {t}: +{100*cost(lambda r, t=t: 1.0 < r['mean_len'] < t):.2f}%  >= : +{100*cost(lambda r, t=t: r['mean_len'] >= t):.2f}%")
    print(f"always bit +{100*cost(lambda r: False):.2f}%, always byte +{100*cost(lambda r: True):.2f}%")
