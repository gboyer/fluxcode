# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Two more RLE-style coders for the residual bit planes, against zstd level 3 and the alternating
zero-run coder of zero_runs.py (variant 0).

A token is a protobuf-style varint (high bit = more bytes follow) of `len << KB | kind`, then:

  variant 1 (KB = 2): kind 0 raw: `len` bytes follow; kind 1: `len` zero bytes; kind 2: `len`
            0xFF bytes (nothing follows the length). Runs of >= 3 bytes of 0x00 / 0xFF are split out.
  variant 2 (KB = 1): PackBits-like. kind 0 raw: `len` bytes follow; kind 1: `len` copies of the one
            byte that follows. Runs of >= 4 of any byte are split out.

    uv run python bench/rle_variants.py [--minutes 2] [--reps 10]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import zstandard
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

import zero_runs
from _signals import KINDS, discrete_minute, minute
from arith_planes import best, bit_planes
from zero_runs import copy, get_varint, put_varint

ONES = np.uint64(0x0101010101010101)
HIGH = np.uint64(0x8080808080808080)


@njit(cache=True)
def run_end(data, words, i, n):
    """End of the run of data[i] starting at i, 8 bytes per step."""
    b = data[i]
    pat = np.uint64(b) * ONES
    j = i + 1
    while j < n and j % 8 != 0 and data[j] == b:
        j += 1
    while j + 8 <= n and words[j >> 3] == pat:
        j += 8
    while j < n and data[j] == b:
        j += 1
    return j


@njit(cache=True)
def fill(dst, d, n, value):
    a = dst[d:d + n]
    for k in range(n):
        a[k] = value


@njit(cache=True)
def has_zero_byte(w):
    return ((w - ONES) & ~w & HIGH) != 0


@njit(cache=True)
def encode1(data, out):
    n = data.shape[0]
    words = data[:n - n % 8].view(np.uint64)
    pos = 0
    start = 0
    i = 0
    while i < n:
        if i % 8 == 0:  # skip words without a 0x00 or 0xFF byte
            while i + 8 <= n and not has_zero_byte(words[i >> 3]) and not has_zero_byte(~words[i >> 3]):
                i += 8
            if i >= n:
                break
        b = data[i]
        if b != 0 and b != 255:
            i += 1
            continue
        j = run_end(data, words, i, n)
        if j - i >= 3:
            if i > start:
                pos = put_varint(out, pos, (i - start) << 2)
                copy(out, pos, data, start, i - start)
                pos += i - start
            pos = put_varint(out, pos, ((j - i) << 2) | (1 if b == 0 else 2))
            start = j
        i = j
    if n > start:
        pos = put_varint(out, pos, (n - start) << 2)
        copy(out, pos, data, start, n - start)
        pos += n - start
    return pos


@njit(cache=True)
def decode1(stream, out):
    pos = 0
    o = 0
    n = out.shape[0]
    while o < n:
        v, pos = get_varint(stream, pos)
        ln = v >> 2
        kind = v & 3
        if kind == 0:
            copy(out, o, stream, pos, ln)
            pos += ln
        elif kind == 2:
            fill(out, o, ln, np.uint8(255))
        o += ln  # kind 1: out is pre-zeroed


@njit(cache=True)
def encode2(data, out):
    n = data.shape[0]
    words = data[:n - n % 8].view(np.uint64)
    pos = 0
    start = 0
    i = 0
    while i < n:
        if i % 8 == 0:  # skip words with no two equal neighbouring bytes
            while i + 8 <= n and not pair_in_word(words[i >> 3]):
                i += 8
            if i >= n:
                break
        j = run_end(data, words, i, n)
        if j - i >= 4:
            if i > start:
                pos = put_varint(out, pos, (i - start) << 1)
                copy(out, pos, data, start, i - start)
                pos += i - start
            pos = put_varint(out, pos, ((j - i) << 1) | 1)
            out[pos] = data[i]
            pos += 1
            start = j
        i = j
    if n > start:
        pos = put_varint(out, pos, (n - start) << 1)
        copy(out, pos, data, start, n - start)
        pos += n - start
    return pos


@njit(cache=True)
def pair_in_word(w):
    """Whether two neighbouring bytes among the 8 of w are equal (7 pairs)."""
    x = (w ^ (w >> np.uint64(8))) & np.uint64(0x00FFFFFFFFFFFFFF)  # byte k: b[k] ^ b[k+1], k < 7
    # a zero byte among the low 7 bytes; the top byte of x is masked to 0, so test only the low 7
    t = ((x - ONES) & ~x & HIGH) & np.uint64(0x0080808080808080)
    return t != 0


@njit(cache=True)
def decode2(stream, out):
    pos = 0
    o = 0
    n = out.shape[0]
    while o < n:
        v, pos = get_varint(stream, pos)
        ln = v >> 1
        if v & 1 == 0:
            copy(out, o, stream, pos, ln)
            pos += ln
        else:
            fill(out, o, ln, stream[pos])
            pos += 1
        o += ln


def wrap(enc, dec):
    def e(data):
        buf = np.empty(data.size + data.size // 8 + 64, np.uint8)
        return buf[:enc(data, buf)]

    def d(stream, n):
        out = np.zeros(n, np.uint8)
        dec(stream, out)
        return out

    return e, d


VARIANTS = {"alt-zero": (zero_runs.zr_encode, zero_runs.zr_decode),
            "zero/FF": wrap(encode1, decode1), "any-byte": wrap(encode2, decode2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=int, default=2)
    ap.add_argument("--reps", type=int, default=10)
    a = ap.parse_args()
    zc = zstandard.ZstdCompressor(level=3, write_checksum=False, write_content_size=True)
    zd = zstandard.ZstdDecompressor()
    sigs = KINDS + ["sensor-0.1", "random-walk q0.01", "noisy-sine q0.1"]
    names = list(VARIANTS)
    print("size / enc time / dec time, each relative to zstd 3 (below 1.00 = smaller / faster)")
    print(f"{'signal':20s} " + " | ".join(f"{n:^20s}" for n in names))
    tot = np.zeros((len(names), 3))
    zt = np.zeros(3)
    for sig in sigs:
        if sig in KINDS or sig == "sensor-0.1":
            x = np.concatenate([minute(sig, 7000 + j) for j in range(a.minutes)])
        else:
            x = np.concatenate([discrete_minute(sig, 7000 + j) for j in range(a.minutes)])
        row = np.zeros((len(names), 3))
        zrow = np.zeros(3)
        for _, planes in bit_planes(x):
            data = np.ascontiguousarray(planes.reshape(-1))
            raw = data.tobytes()
            z = zc.compress(raw)
            zrow += [len(z), best(lambda: zc.compress(raw), a.reps), best(lambda: zd.decompress(z), a.reps)]
            for k, (e, d) in enumerate(VARIANTS.values()):
                s = e(data)
                assert np.array_equal(d(s, data.size), data), (sig, names[k])
                row[k] += [len(s), best(lambda: e(data), a.reps), best(lambda: d(s, data.size), a.reps)]
        tot += row
        zt += zrow
        print(f"{sig:20s} " + " | ".join(f"{r[0] / zrow[0]:6.2f} {r[1] / zrow[1]:6.2f} {r[2] / zrow[2]:6.2f}" for r in row))
    print(f"{'TOTAL':20s} " + " | ".join(f"{r[0] / zt[0]:6.2f} {r[1] / zt[1]:6.2f} {r[2] / zt[2]:6.2f}" for r in tot))


if __name__ == "__main__":
    main()
