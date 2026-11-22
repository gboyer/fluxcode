"""Bits/sample of every corpus signal with Params.max_error at a share of the signal's span (sensor
accuracy is usually quoted as % of span): 0, 0.01%, 0.1%, 1%.

    uv run python value/maxerr_span.py
"""
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import corpus
import harness as H
import fluxcode

SHARES = (0.0, 1e-4, 1e-3, 1e-2)


def run(item):
    set_name, name, x, block_len, bpg = item
    finite = x[np.isfinite(x)]
    span = float(np.ptp(finite)) if finite.size else 0.0
    out = []
    for share in SHARES:
        params = fluxcode.Params(max_error=share * span)
        nbytes, worst = 0, 0.0
        for part, _ in H.groups_of(x, block_len, bpg):
            g = fluxcode.encode_group(part, params, block_len=block_len).group
            nbytes += len(g)
            y = fluxcode.decode_group(g).values
            ok = np.isfinite(part)
            if ok.any():
                worst = max(worst, float(np.abs(y[ok] - part[ok]).max()))
        out.append((8 * nbytes / len(x), worst / span if span else 0.0))
    return set_name, name, out


if __name__ == "__main__":
    with ProcessPoolExecutor(6) as ex:
        res = list(ex.map(run, corpus.corpus(["fast", "slow", "synth"])))
    print("| set | signal | " + " | ".join(f"{100 * s:g}% span" for s in SHARES) + " |")
    print("|---|---|" + "---|" * len(SHARES))
    totals = np.zeros(len(SHARES))
    for set_name, name, out in res:
        totals += [b for b, _ in out]
        print(f"| {set_name} | {name} | " + " | ".join(f"{b:.2f} b/s (err {100 * w:.3f}%)" for b, w in out) + " |")
    print("| all | mean | " + " | ".join(f"{t / len(res):.2f}" for t in totals) + " |")
