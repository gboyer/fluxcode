"""Bits/sample and errors of every corpus signal under the current FLUXCODE_X_* settings; writes a JSON for diffs.

    FLUXCODE_X_NOISE_ORDER=6 uv run python value/sizes.py out.json [sets...]
"""
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import corpus
import harness as H  # noqa: F401  (sets the numba cache)
import fluxcode


def run(item):
    set_name, name, x, block_len, bpg = item
    import os
    params = fluxcode.Params(noise_floor_sigma=float(os.environ.get("FX_F", "0.25")))
    nbytes = 0
    sq = 0.0
    worst = 0.0
    t0 = time.perf_counter()
    for part, sizes in H.groups_of(x, block_len, bpg):
        g = fluxcode.encode_group(part, params, block_len=block_len).group
        nbytes += len(g)
        y = fluxcode.decode_group(g).values
        err = np.abs(y - part)
        sq += float((err ** 2).sum())
        for b in range(0, len(part), block_len):
            blk = part[b:b + block_len]
            r = np.ptp(blk)
            if r > 0:
                worst = max(worst, err[b:b + block_len].max() / r)
    return dict(set=set_name, name=name, n=len(x), bytes=nbytes, rmse=np.sqrt(sq / len(x)), worst=worst,
                secs=time.perf_counter() - t0)


if __name__ == "__main__":
    out = sys.argv[1]
    sets = sys.argv[2:] or ["fast", "slow", "synth"]
    with ProcessPoolExecutor(6) as ex:
        res = list(ex.map(run, corpus.corpus(sets)))
    json.dump(res, open(out, "w"), indent=1)
    for r in res:
        print(f"{r['set']:5s} {r['name']:28s} {8 * r['bytes'] / r['n']:7.3f} b/s  rmse {r['rmse']:.3g}  worst {100 * r['worst']:.4f}%")
