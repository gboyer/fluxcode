# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""How fast and how small is delta0123-zstd, one compressed block group per minute (60 blocks)?

Method: verify delta0123-zstd (error <= step/2, exact min/max, delta orders equal to a plain numpy
reference), then time it at B = 8 and 10 with zstd / deflate at several levels, against the
quant8-delta1 block group references. Also lists the NEON vector operations in the compiled loops.

Scale: 200 signals x 86,400 blocks/day. Timings are per 1000-sample block,
best of 3, one thread. Sizes are whole block groups, including each block's min and max; the index
columns a store keeps next to the block groups (min/max/mean) are computed once per block regardless
of codec and timed separately.

Data: tslab.common.datasets.load() (7,200 continuous blocks).

    uv run python -m bench.delta_zstd
"""

import os

os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import re
import time

import numpy as np
from numba import njit

from tslab.common.datasets import load
from tslab.common.intcode import pick_order
from tslab.entropy import delta_zstd as dz
from tslab.entropy.native import Quant8Delta1

SIGNALS, BLOCKS_PER_DAY = 200, 86400


def best_of(f, reps=3):
    best = np.inf
    for _ in range(reps):
        t0 = time.perf_counter()
        out = f()
        best = min(best, time.perf_counter() - t0)
    return best, out


def run(codec, mins):
    for _, X, _, _ in mins[:2]:
        codec.decode_group(codec.encode_group(X)[0], *X.shape)  # compile / warm up
    te, blobs = best_of(lambda: [codec.encode_group(X)[0] for _, X, _, _ in mins])
    td, outs = best_of(lambda: [codec.decode_group(b, *X.shape) for b, (_, X, _, _) in zip(blobs, mins)])
    nblocks = sum(len(X) for _, X, _, _ in mins)
    rmse = []
    for (_, X, lo, hi), Y in zip(mins, outs):
        rmse.extend(np.sqrt(np.mean((Y - X) ** 2, axis=1)) / (hi - lo))
    bps = 8 * sum(map(len, blobs)) / (nblocks * 1000)
    return {"enc": 1e6 * te / nblocks, "dec": 1e6 * td / nblocks, "bps": bps,
            "rmse": 100 * np.median(rmse), "blobs": blobs, "outs": outs}


def verify(codec, mins):
    """Error bound per block from its B, min/max exact, orders match a plain numpy argmin of the variance."""
    agree = total = 0
    for _, X, lo, hi in mins:
        data, infos = codec.encode_group(X)
        Y = codec.decode_group(data, *X.shape)
        for b, info in enumerate(infos):
            B, order = info["bits"], info["order"]
            step = (hi[b] - lo[b]) / ((1 << B) - 1)
            assert np.abs(Y[b] - X[b]).max() <= step / 2 * (1 + 1e-9)
            q = np.floor((X[b] - lo[b]) * (((1 << B) - 1) / (hi[b] - lo[b])) + 0.5).astype(np.int64)
            agree += int(np.argmin([np.var(np.diff(q, k)) for k in range(4)])) == order
            total += 1
        assert np.all(Y.max(1) == hi) and np.all(Y.min(1) == lo)
    return agree, total


def vector_ops(fn, *args):
    f = njit(fn.py_func)
    f(*args)
    pat = re.compile(r"^\s*(\w+)\.(16b|8b|8h|4h|4s|2s|2d)\b")
    skip = {"movi", "mov", "dup", "ldr", "str", "ldp", "stp", "ld1", "st1", "ins", "fmov"}
    ops = set()
    for sig in f.signatures:
        for line in f.inspect_asm(sig).splitlines():
            m = pat.search(line)
            if m and m.group(1) not in skip:
                ops.add(f"{m.group(1)}.{m.group(2)}")
    return sorted(ops)


def main():
    mins = load()
    nblocks = sum(len(X) for _, X, _, _ in mins)
    ti, _ = best_of(lambda: [(X.min(1), X.max(1), X.mean(1)) for _, X, _, _ in mins])
    index_us = 1e6 * ti / nblocks

    variants = [dz.DeltaZstd(b, level, compressor=c) for b in (8, 10)
                for c, level in (("zstd", 1), ("zstd", 3), ("deflate", 1), ("deflate", 3), ("deflate", 4), ("deflate", 6))]
    refs = [Quant8Delta1("zstd", 3), Quant8Delta1("deflate", 3), Quant8Delta1("deflate", 6)]

    for codec in (variants[1], variants[7]):
        agree, total = verify(codec, mins)
        print(f"verified {codec.name}: error <= step/2 and exact min/max on {total} blocks; "
              f"order matches the numpy reference on {agree}/{total}")

    rows = []
    for codec in variants + refs:
        r = run(codec, mins)
        rows.append((codec.name, r))

    print(f"\nIndex stats (min/max/mean, numpy): {index_us:.2f} µs/block — same for every codec, not included below.")
    print(f"Sizes are whole block groups (each block's min and max included). "
          f"Scale: {SIGNALS} signals x {BLOCKS_PER_DAY:,} blocks = {SIGNALS * BLOCKS_PER_DAY / 1e6:.2f}M blocks/day.\n")
    print("| codec | bits/sample | median RMSE | encode µs/block | decode µs/block | "
          "200 signals x 1 day, encode core-s | 1:100 cores | 1:1000 cores |")
    print("|---|---|---|---|---|---|---|---|")
    for name, r in rows:
        core_s = r["enc"] * SIGNALS * BLOCKS_PER_DAY / 1e6
        print(f"| {name} | {r['bps']:.3f} | {r['rmse']:.3f}% | {r['enc']:.2f} | {r['dec']:.2f} | {core_s:.0f} | "
              f"{core_s / 864:.2f} | {core_s / 86.4:.1f} |")

    X = mins[0][1]
    q = np.zeros(1000, np.int32)
    u = np.zeros(1000, np.uint32)
    raw, _ = variants[1].raw_group(X[:2])
    print("\nVector (NEON) arithmetic in compiled loops:")
    for name, fn, args in [
        ("_quantize", dz._quantize, (X[0], 0.0, 1.0, 8, q)),
        ("pick_order", pick_order, (q, 15)),
        ("_residual_zigzag", dz._residual_zigzag, (q, 2, u)),
        ("_write_planes", dz._write_planes, (u, 998, 2, np.zeros(4000, np.uint8), 0)),
        ("_decode_group", dz._decode_group, (raw, np.empty((2, 1000)))),
    ]:
        ops = vector_ops(fn, *args)
        print(f"  {name:18s} {', '.join(ops) if ops else 'none (scalar)'}")


if __name__ == "__main__":
    main()
