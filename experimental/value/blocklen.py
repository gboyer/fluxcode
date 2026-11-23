"""Bits/sample and RMSE of the corpus at several encode_group block lengths (groups of 60,000 samples).

    uv run python value/blocklen.py
"""
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import corpus
import harness as H  # noqa: F401
import fluxcode

LENS = (500, 1000, 2000, 4000)


def run(item):
    s, name, x, bl, bpg = item
    out = []
    for L in LENS:
        nbytes, sq = 0, 0.0
        for st in range(0, len(x), 60_000):
            part = x[st:st + 60_000]
            g = fluxcode.encode_group(part, block_len=L).group
            nbytes += len(g)
            y = fluxcode.decode_group(g).values
            m = np.isfinite(part)
            sq += float(((y[m] - part[m]) ** 2).sum())
        out.append((8 * nbytes / len(x), np.sqrt(sq / len(x))))
    return s, name, out


if __name__ == "__main__":
    with ProcessPoolExecutor(6) as ex:
        res = list(ex.map(run, corpus.corpus(["fast", "slow", "synth"])))
    print("| set | signal | " + " | ".join(str(L) for L in LENS) + " |")
    print("|---|---|" + "---|" * len(LENS))
    tot = np.zeros(len(LENS))
    for s, name, out in res:
        tot += [b for b, _ in out]
        base = out[1][1]
        print(f"| {s} | {name} | " + " | ".join(f"{b:.2f} (rmse ×{r / base if base else 1:.2f})" for b, r in out) + " |")
    print("| all | mean | " + " | ".join(f"{t / len(res):.3f}" for t in tot) + " |")
