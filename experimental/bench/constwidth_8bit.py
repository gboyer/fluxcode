# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Which ~8-bit format gives the lowest error: constant width, or entropy coded with a cap?

Method: every codec encodes the same 7,200 continuous blocks as one-minute block groups; reports size, error
percentiles, the largest block, and encode / decode time, then median error per signal type.

- quant8, cw8-delta1-linear, cw8-delta1-sqrt: constant width, 1 byte per sample
- cw8-delta1-bfp-e2 / -e4: constant width, 1 byte per sample + frame exponents
- cw4-delta1-bfp-e4 / cw6-delta1-bfp-e4: the same with 4- / 6-bit codes (for comparison at ~4 and ~6 bits)
- cw-delta1-tree-7 / -4: constant width, 7- / 4-bit leaves + 1 bit per tree node
- delta0123-zstd-16: 16-bit quantizer, order pick, zstd; the cap (estimated from the
  residual variance) lowers B only for blocks that would exceed 8 bits/sample
Constant-width block groups are the block encodings back to back; "+ zstd per block group" compresses that block group
with zstd-3. "each block its own block group" encodes every block as a one-block group, for comparison.
Sizes are everything the decoder reads, including each block's min and max.
Single thread; µs per 1000-sample block, best of 3.

Data: tslab.common.datasets.load() (7,200 continuous blocks).

    uv run python -m bench.constwidth_8bit
"""

import os

os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import time

import numpy as np
import zstandard

from tslab.common.datasets import load
from tslab.common.group import Concat
from tslab.constwidth import Delta1Bfp, Delta1Linear8, Delta1Sqrt8, Delta1Tree
from tslab.entropy import native
from tslab.entropy.delta_zstd import DeltaZstd


def best_of(f, reps=3):
    best, out = np.inf, None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = f()
        best = min(best, time.perf_counter() - t0)
    return best, out


class Quant8Native:
    """quant8 as a block group, via the numba port (same bytes as Concat(Quant8()))."""

    name = "quant8"

    def encode_group(self, X):
        return native.quant8_encode(X), [{}] * len(X)

    def decode_group(self, data, nb, n):
        return native.quant8_decode(data, nb, n)


class ZstdGroup:
    """<codec> + zstd per block group: the codec's block group, compressed with zstd-3."""

    def __init__(self, codec):
        self.c = codec
        self.name = codec.name + " + zstd per block group"
        self.zc = zstandard.ZstdCompressor(level=3, write_checksum=False)
        self.zd = zstandard.ZstdDecompressor()

    def encode_group(self, X):
        data, infos = self.c.encode_group(X)
        return self.zc.compress(data), infos

    def decode_group(self, data, nb, n):
        return self.c.decode_group(self.zd.decompress(data), nb, n)


class EachBlock:
    """<codec> with every block its own one-block group (a list of block groups per minute)."""

    def __init__(self, codec):
        self.c = codec
        self.name = codec.name + " (each block its own block group)"

    def encode_group(self, X):
        encs = [self.c.encode_group(X[b:b + 1]) for b in range(len(X))]
        return [d for d, _ in encs], [i for _, infos in encs for i in infos]

    def decode_group(self, data, nb, n):
        return np.concatenate([self.c.decode_group(d, 1, n) for d in data])


def run(codec, mins):
    X0 = mins[0][1]
    codec.decode_group(codec.encode_group(X0)[0], *X0.shape)  # compile / warm up
    te, encs = best_of(lambda: [codec.encode_group(X) for _, X, _, _ in mins])
    td, outs = best_of(lambda: [codec.decode_group(e, *X.shape) for (e, _), (_, X, _, _) in zip(encs, mins)])
    nblocks = sum(len(X) for _, X, _, _ in mins)
    rmse, maxerr, per_kind, bps_block = [], [], {}, []
    for (kind, X, lo, hi), Y, (e, _) in zip(mins, outs, encs):
        fs = hi - lo
        r = np.sqrt(np.mean((Y - X) ** 2, axis=1)) / fs
        rmse.extend(r)
        maxerr.extend(np.abs(Y - X).max(axis=1) / fs)
        per_kind.setdefault(kind, []).extend(r)
        if isinstance(e, list):
            bps_block.extend(8 * len(b) / 1000 for b in e)
        elif isinstance(codec, (Concat, Quant8Native)):  # every block the same size
            bps_block.append(8 * len(e) / X.size)
    nbytes = sum(sum(map(len, e)) if isinstance(e, list) else len(e) for e, _ in encs)
    out = {"bps": 8 * nbytes / (nblocks * 1000),
           "rmse_med": 100 * np.median(rmse), "rmse_p99": 100 * np.percentile(rmse, 99),
           "rmse_max": 100 * max(rmse), "maxerr_med": 100 * np.median(maxerr), "maxerr_max": 100 * max(maxerr),
           "enc": 1e6 * te / nblocks, "dec": 1e6 * td / nblocks,
           "per_kind": {k: 100 * np.median(v) for k, v in per_kind.items()},
           "bps_block_max": max(bps_block) if bps_block else None}
    if "bits" in encs[0][1][0]:
        out["B"] = np.array([i["bits"] for _, infos in encs for i in infos])
    return out


def main():
    mins = load()
    def bfp(exp_bits, code_bits=8):
        return Concat(Delta1Bfp(16, exp_bits, code_bits))

    codecs = [Quant8Native(), Concat(Delta1Linear8()), Concat(Delta1Sqrt8()),
              bfp(2), bfp(4), Concat(Delta1Tree(7, "aware")),
              ZstdGroup(bfp(4)), EachBlock(DeltaZstd(16)), DeltaZstd(16),
              bfp(4, 6), bfp(4, 4), Concat(Delta1Tree(4, "aware")),
              DeltaZstd(8)]
    res = {c.name: run(c, mins) for c in codecs}
    print("| codec | bits/sample | median RMSE | 99th pct RMSE | worst RMSE | median max err | worst max err | "
          "max bits/sample of any block | encode µs | decode µs |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for n, r in res.items():
        mb = f"{r['bps_block_max']:.2f}" if r["bps_block_max"] else "—"
        print(f"| {n} | {r['bps']:.2f} | {r['rmse_med']:.5f}% | {r['rmse_p99']:.4f}% | {r['rmse_max']:.4f}% | "
              f"{r['maxerr_med']:.4f}% | {r['maxerr_max']:.3f}% | {mb} | {r['enc']:.1f} | {r['dec']:.1f} |")
    for n, r in res.items():
        if "B" in r:
            vals, counts = np.unique(r["B"], return_counts=True)
            print(f"\n{n}: B chosen per block: " + ", ".join(f"B={v}: {100 * c / len(r['B']):.0f}%" for v, c in zip(vals, counts)))
    print("\nMedian RMSE by signal type (% of block range):")
    names = list(res)
    print("| type | " + " | ".join(names) + " |\n|---|" + "---|" * len(names))
    for k in res[names[0]]["per_kind"]:
        print(f"| {k} | " + " | ".join(f"{res[n]['per_kind'][k]:.5f}" for n in names) + " |")


if __name__ == "__main__":
    main()
