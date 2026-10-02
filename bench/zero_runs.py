# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Experiment: squash only runs of zero bytes. The stream alternates raw and zero runs (the first
is raw, possibly of length 0), each introduced by its length as a protobuf-style varint (high bit
= more bytes follow); raw runs are followed by their bytes. Compared with zstd level 3 on a unit's
16 residual bit planes (real units). A zero run is only split out of a raw run when it is at
least MIN_ZERO bytes (shorter ones cost more than they save), or when it ends the data.

    uv run python bench/zero_runs.py [--minutes 2] [--reps 10]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import zstandard
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from _signals import KINDS, discrete_minute, minute
from arith_planes import best, bit_planes

MIN_ZERO = 3


@njit(cache=True)
def copy(dst, d, src, p, n):
    # Loop over views: dst[d + k] = src[p + k] isn't vectorized (signed index arithmetic needs a
    # wraparound check), and slice assignment is ~45x slower than memcpy.
    a = dst[d:d + n]
    b = src[p:p + n]
    for k in range(n):
        a[k] = b[k]


@njit(cache=True)
def put_varint(out, pos, v):
    while v >= 128:
        out[pos] = np.uint8((v & 127) | 128)
        pos += 1
        v >>= 7
    out[pos] = np.uint8(v)
    return pos + 1


@njit(cache=True)
def encode(data, out):
    n = data.shape[0]
    words = data[:n - n % 8].view(np.uint64)
    pos = 0
    i = 0
    while True:
        start = i
        zstart = n  # where the zero run that ends this raw run starts (n: none)
        while i < n:
            if i % 8 == 0:  # skip 8 bytes at a time while none is zero
                w = words[i >> 3] if i + 8 <= n else np.uint64(0)
                while i + 8 <= n and ((w - np.uint64(0x0101010101010101)) & ~w & np.uint64(0x8080808080808080)) == 0:
                    i += 8
                    w = words[i >> 3] if i + 8 <= n else np.uint64(0)
                if i >= n:
                    break
            if data[i] != 0:
                i += 1
                continue
            j = i + 1
            while j < n and j % 8 != 0 and data[j] == 0:
                j += 1
            while j + 8 <= n and words[j >> 3] == 0:  # 8 zero bytes per step
                j += 8
            while j < n and data[j] == 0:
                j += 1
            if j - i >= MIN_ZERO or j == n:
                zstart = i
                i = j
                break
            i = j
        raw_end = zstart if zstart < n else n
        pos = put_varint(out, pos, raw_end - start)
        copy(out, pos, data, start, raw_end - start)
        pos += raw_end - start
        if raw_end == n:
            return pos
        pos = put_varint(out, pos, i - raw_end)
        if i == n:
            return pos


@njit(cache=True)
def get_varint(stream, pos):
    v = 0
    shift = 0
    while True:
        b = stream[pos]
        pos += 1
        v |= (np.int64(b) & 127) << shift
        if b < 128:
            return v, pos
        shift += 7


@njit(cache=True)
def decode(stream, out):
    """out must be zeroed and of the original length."""
    pos = 0
    o = 0
    n = out.shape[0]
    while o < n:
        ln, pos = get_varint(stream, pos)
        copy(out, o, stream, pos, ln)
        pos += ln
        o += ln
        if o >= n:
            break
        ln, pos = get_varint(stream, pos)
        o += ln


def zr_encode(data):
    buf = np.empty(data.size + data.size // 100 + 64, np.uint8)
    return buf[:encode(data, buf)]


def zr_decode(stream, n):
    out = np.zeros(n, np.uint8)
    decode(stream, out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=int, default=2)
    ap.add_argument("--reps", type=int, default=10)
    a = ap.parse_args()
    zc = zstandard.ZstdCompressor(level=3, write_checksum=False, write_content_size=True)
    zd = zstandard.ZstdDecompressor()
    sigs = KINDS + ["sensor-0.1", "random-walk q0.01", "noisy-sine q0.1"]
    print(f"{'signal':20s} {'planes B':>8s} | {'zstd B':>7s} {'enc us':>7s} {'dec us':>7s} | "
          f"{'zrun B':>7s} {'enc us':>7s} {'dec us':>7s} | {'size':>6s} {'enc x':>6s} {'dec x':>6s}")
    tot = np.zeros(7)
    for sig in sigs:
        if sig in KINDS or sig == "sensor-0.1":
            x = np.concatenate([minute(sig, 7000 + j) for j in range(a.minutes)])
        else:
            x = np.concatenate([discrete_minute(sig, 7000 + j) for j in range(a.minutes)])
        row = np.zeros(7)
        for _, planes in bit_planes(x):
            data = np.ascontiguousarray(planes.reshape(-1))
            raw = data.tobytes()
            z = zc.compress(raw)
            s = zr_encode(data)
            assert np.array_equal(zr_decode(s, data.size), data), sig
            row += [data.size, len(z), best(lambda: zc.compress(raw), a.reps) * 1e6,
                    best(lambda: zd.decompress(z), a.reps) * 1e6,
                    len(s), best(lambda: zr_encode(data), a.reps) * 1e6,
                    best(lambda: zr_decode(s, data.size), a.reps) * 1e6]
        tot += row
        r = row
        print(f"{sig:20s} {r[0]:8.0f} | {r[1]:7.0f} {r[2]:7.0f} {r[3]:7.0f} | {r[4]:7.0f} {r[5]:7.0f} {r[6]:7.0f} | "
              f"{r[4] / r[1]:6.2f} {r[5] / r[2]:6.2f} {r[6] / r[3]:6.2f}")
    r = tot
    print(f"{'TOTAL':20s} {r[0]:8.0f} | {r[1]:7.0f} {r[2]:7.0f} {r[3]:7.0f} | {r[4]:7.0f} {r[5]:7.0f} {r[6]:7.0f} | "
          f"{r[4] / r[1]:6.2f} {r[5] / r[2]:6.2f} {r[6] / r[3]:6.2f}")
    print("\nsize/enc x/dec x = zero-run / zstd (below 1 for time means zero-run is faster)")


if __name__ == "__main__":
    main()
