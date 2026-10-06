# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Does compressing one-minute chunks (60 blocks per block group) beat compressing each block alone?

Method: the same payloads (delta0123-zstd residual byte planes at B = 8 and 10, and quant8 delta
bytes) go through zstd and deflate at several levels, either per block or one block group per minute.
Reports size, encode / decode time, and the cost of reading one block. Single thread.

Storage model: per-block min/max/mean float64 live in Arrow columns (24 B/block,
0.19 bits/sample, reported separately). The payload is either compressed per
block, or the whole minute is one compressed block group: all 60 block headers (flags +
init varints) followed by all 60 blocks' residual byte planes.

Test signals are continuous one-minute versions of each dataset type, so the 60
blocks in a chunk are consecutive seconds of one channel — not repeats.

Data: tslab.common.datasets.minute(), 5 minutes of each of the 12 kinds.

    uv run python -m bench.chunking
"""

import os

os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import time
import zlib

import numpy as np
import zstandard

from tslab.common.datasets import BLOCK, KINDS, PER_CHUNK, minute
from tslab.entropy import native

MINUTES_PER_KIND = 5


def zstd(level):
    zc = zstandard.ZstdCompressor(level=level, write_content_size=False, write_checksum=False)
    zd = zstandard.ZstdDecompressor()
    return f"zstd-{level}", zc.compress, lambda b, size: zd.decompress(b, max_output_size=size)


def deflate(level):
    def c(b):
        o = zlib.compressobj(level, zlib.DEFLATED, -15)
        return o.compress(b) + o.flush()
    return f"deflate-{level}", c, lambda b, size: zlib.decompress(b, -15)


COMPRESSORS = [zstd(1), zstd(3), zstd(9), deflate(1), deflate(6), deflate(9)]


def delta_zstd_payloads(X, bits):
    """Per block: header bytes (flags + init varints) and residual byte planes."""
    lo, hi = X.min(1), X.max(1)
    planes = np.empty((len(X), 4 * BLOCK), np.uint8)
    meta = np.zeros((len(X), 6), np.int64)
    native._delta_zstd_front(X, lo, hi, bits, np.ones(4, np.bool_), planes, meta)
    heads, bodies = [], []
    for b in range(len(X)):
        o, w, size = int(meta[b, 0]), int(meta[b, 1]), int(meta[b, 2])
        heads.append(bytes([(bits - 1) << 4 | o << 2 | native.WCODE[w]])
                     + b"".join(native._varint_zz(int(meta[b, 3 + k])) for k in range(o)))
        bodies.append(planes[b, :size].tobytes())
    return heads, bodies


def quant8_delta_payloads(X):
    lo, hi = X.min(1), X.max(1)
    q = np.empty(X.shape, np.uint8)
    native._quantize_u8(X, lo, hi, q)
    d = np.empty_like(q)
    native._delta_u8(q, d)
    return [b""] * len(X), [d[b].tobytes() for b in range(len(X))]


def measure(chunks, comp):
    """chunks: list of (heads, bodies) per minute. Returns bits/sample, µs/block enc, dec, dec-one-block."""
    name, c, d = comp
    n_blocks = sum(len(b) for _, b in chunks)
    samples = n_blocks * BLOCK
    out = {}
    for mode in ("per-block", "chunk"):
        if mode == "per-block":
            groups = [h + b for heads, bodies in chunks for h, b in zip(heads, bodies)]
        else:
            groups = [b"".join(heads) + b"".join(bodies) for heads, bodies in chunks]
        te = min(timeit(lambda: [c(u) for u in groups]) for _ in range(3))
        packed = [c(u) for u in groups]
        td = min(timeit(lambda: [d(p, len(u)) for p, u in zip(packed, groups)]) for _ in range(3))
        assert all(d(p, len(u)) == u for p, u in zip(packed, groups))
        out[mode] = {"bps": 8 * sum(map(len, packed)) / samples, "enc": 1e6 * te / n_blocks,
                     "dec": 1e6 * td / n_blocks,
                     "one_block": 1e6 * td / len(groups)}  # cost to read one block = decode its block group
    return name, out


def timeit(f):
    t0 = time.perf_counter()
    f()
    return time.perf_counter() - t0


def main():
    mins = {k: [minute(k, 1000 * i + j) for j in range(MINUTES_PER_KIND)] for i, k in enumerate(KINDS)}
    native._delta_zstd_front(np.zeros((1, BLOCK)), np.zeros(1), np.ones(1), 8, np.ones(4, np.bool_),
                          np.empty((1, 4 * BLOCK), np.uint8), np.zeros((1, 6), np.int64))  # compile
    payloads = {
        "delta0123-zstd-8 residuals": lambda X: delta_zstd_payloads(X, 8),
        "delta0123-zstd-10 residuals": lambda X: delta_zstd_payloads(X, 10),
        "quant8-delta1 bytes": quant8_delta_payloads,
    }
    for pname, make in payloads.items():
        chunks = [make(m.reshape(PER_CHUNK, BLOCK)) for ms in mins.values() for m in ms]
        print(f"\n### {pname}  ({len(chunks)} minutes, {len(chunks) * PER_CHUNK} blocks, all 12 signal types)")
        print("| compressor | per-block bits/sample | chunk bits/sample | size change | per-block enc µs/block | "
              "chunk enc µs/block | per-block dec µs/block | chunk dec µs/block | read 1 block: per-block / chunk µs |")
        print("|---|---|---|---|---|---|---|---|---|")
        for comp in COMPRESSORS:
            name, r = measure(chunks, comp)
            a, b = r["per-block"], r["chunk"]
            print(f"| {name} | {a['bps']:.3f} | {b['bps']:.3f} | {100 * (b['bps'] / a['bps'] - 1):+.1f}% | "
                  f"{a['enc']:.1f} | {b['enc']:.1f} | {a['dec']:.1f} | {b['dec']:.1f} | "
                  f"{a['one_block']:.0f} / {b['one_block']:.0f} |")
        if pname == "delta0123-zstd-8 residuals":
            print("\nPer signal type, zstd-3 (bits/sample): per-block → chunk")
            for k, ms in mins.items():
                ch = [make(m.reshape(PER_CHUNK, BLOCK)) for m in ms]
                _, r = measure(ch, zstd(3))
                print(f"  {k:14s} {r['per-block']['bps']:.3f} → {r['chunk']['bps']:.3f}")
    print("\nIndex columns (min/max/mean float64 per block) add 0.192 bits/sample in either layout.")


if __name__ == "__main__":
    main()
