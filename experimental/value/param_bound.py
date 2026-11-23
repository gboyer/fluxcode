"""Per-block parametric code length (best of discretized Gaussian / Laplace fitted per block, +16 bits per
block) against the actual block group size: how much could a per-block adaptive entropy coder save?"""
import sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.stats import norm
import corpus, harness as H
from fluxcode import _compress

def bits_gauss(r):
    s = max(np.std(r), 0.3)
    p = norm.cdf((r + 0.5) / s) - norm.cdf((r - 0.5) / s)
    return -np.log2(np.maximum(p, 1e-300)).sum()

def bits_laplace(r):
    m = np.median(r); b = max(np.mean(np.abs(r - m)), 0.3)
    cdf = lambda v: np.where(v < 0, 0.5 * np.exp(v / b), 1 - 0.5 * np.exp(-v / b))
    p = cdf(r - m + 0.5) - cdf(r - m - 0.5)
    return -np.log2(np.maximum(p, 1e-300)).sum()

def run(item):
    s, name, x, bl, bpg = item
    actual = bound = n = 0
    for part, sizes in H.groups_of(x, bl, bpg):
        rows = H.encode_rows(part, sizes, H.default_params())
        actual += 8 * len(_compress.compress(rows, int(sizes.sum()), _compress.EFFORTS[4], 0))
        off = 0
        for size in sizes:
            r = rows.residuals[off:off + size].astype(np.int64); off += size
            bound += min(bits_gauss(r), bits_laplace(r)) + 16
        n += sizes.sum()
    return s, name, n, actual, bound

if __name__ == "__main__":
    with ProcessPoolExecutor(6) as ex:
        res = list(ex.map(run, corpus.corpus(sys.argv[1:] or ["fast", "slow"])))
    for s, name, n, a, b in res:
        print(f"{s:5s} {name:22s} actual {a/n:6.3f}  per-block parametric {b/n:6.3f}  ({100*(b/a-1):+.0f}%)")
