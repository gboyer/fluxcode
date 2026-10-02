# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""How much of encode() and decode() is zstd? Per signal and Params: wall time of the public call
vs zstd level 3 on the same bodies (compress, and decompress), single thread, best of reps.

    uv run python plane_coders/zstd_share.py [--minutes 2] [--reps 5]
"""

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
import numpy as np
import zstandard

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
import fluxcode
from _signals import KINDS, discrete_minute, minute
from fluxcode import Params, _unit


def best(f, reps):
    t = np.inf
    for _ in range(reps):
        t0 = time.perf_counter()
        f()
        t = min(t, time.perf_counter() - t0)
    return t * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=int, default=2)
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    zc = zstandard.ZstdCompressor(level=3, write_checksum=False, write_content_size=True)
    zd = zstandard.ZstdDecompressor()
    sigs = KINDS + ["sensor-0.1", "random-walk q0.01", "noisy-sine q0.1"]
    for name, params in (("planes=bit", Params(planes="bit")), ("planes=best (default)", Params())):
        print(f"\n{name}: ms for {a.minutes} min of each signal\n"
              f"{'signal':20s} {'encode':>7s} {'zstd c':>7s} {'share':>6s} | {'decode':>7s} {'zstd d':>7s} {'share':>6s}")
        tot = np.zeros(4)
        for sig in sigs:
            if sig in KINDS or sig == "sensor-0.1":
                x = np.concatenate([minute(sig, 7000 + j) for j in range(a.minutes)])
            else:
                x = np.concatenate([discrete_minute(sig, 7000 + j) for j in range(a.minutes)])
            units, *_ = fluxcode.encode(x, params)
            bodies = [bytes(_unit.decompress(u).raw_body) for u in units]
            frames = [u[8:] for u in units]
            enc = best(lambda: fluxcode.encode(x, params), a.reps)
            dec = best(lambda: fluxcode.decode(units), a.reps)
            zcomp = best(lambda: [zc.compress(b) for b in bodies], a.reps)
            zdec = best(lambda: [zd.decompress(f) for f in frames], a.reps)
            if params.planes == "best":  # encode also compresses the other layout's body
                units2, *_ = fluxcode.encode(x, Params(planes="byte" if units[0][1] & 1 == 0 else "bit"))
                bodies2 = [bytes(_unit.decompress(u).raw_body) for u in units2]
                zcomp += best(lambda: [zc.compress(b) for b in bodies2], a.reps)
            tot += [enc, zcomp, dec, zdec]
            print(f"{sig:20s} {enc:7.1f} {zcomp:7.1f} {zcomp / enc:6.0%} | {dec:7.1f} {zdec:7.1f} {zdec / dec:6.0%}")
        print(f"{'TOTAL':20s} {tot[0]:7.1f} {tot[1]:7.1f} {tot[1] / tot[0]:6.0%} | {tot[2]:7.1f} {tot[3]:7.1f} {tot[3] / tot[2]:6.0%}")


if __name__ == "__main__":
    main()
