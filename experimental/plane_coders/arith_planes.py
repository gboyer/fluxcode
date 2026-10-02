# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Experiment: entropy-code the 16 residual bit planes with a static-density arithmetic coder
instead of zstd, and compare encode/decode speed and size.

Per plane: one density byte d = round(256 * ones / bits) (0 = all-zero plane, skipped; clipped to
1..255 otherwise). The plane's bytes are then coded as symbols of an i.i.d. Bernoulli(d/256)
byte: a 4-way interleaved 32-bit rANS (byte-wise renormalization, 16-bit probabilities). The
8-bit symbol's probability depends only on its popcount k, so the model is nine per-density
frequencies f_k (all symbols of popcount k share one), and a symbol's slot is
cum_k + rank_in_class * f_k. No per-symbol tables, no search in the decoder beyond nine compares.

    uv run python plane_coders/arith_planes.py [--minutes 2] [--reps 20]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import zstandard
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))

import fluxcode
from _signals import KINDS, MINUTE, discrete_minute, minute
from fluxcode import Params, _format, _unit

PB = 16  # probability bits
M = 1 << PB
LANES = 4
RANS_L = 1 << 23

POP = np.array([bin(i).count("1") for i in range(256)], np.int64)
# rank of each byte among the bytes of its popcount, and the bytes of each class in rank order
RANK = np.zeros(256, np.int64)
BY_CLASS = np.zeros(256, np.uint8)  # bytes sorted by (popcount, value)
_order = sorted(range(256), key=lambda b: (POP[b], b))
CLASS_START = np.zeros(10, np.int64)
for _i, _b in enumerate(_order):
    BY_CLASS[_i] = _b
for _k in range(9):
    CLASS_START[_k + 1] = CLASS_START[_k] + sum(1 for b in range(256) if POP[b] == _k)
for _i, _b in enumerate(_order):
    RANK[_b] = _i - CLASS_START[POP[_b]]


def build_model():
    """freq[d, k] and cum[d, k] (slot where class k starts) for densities d = 1..255."""
    freq = np.zeros((256, 9), np.int64)
    cum = np.zeros((256, 10), np.int64)
    sizes = CLASS_START[1:] - CLASS_START[:-1]
    for d in range(1, 256):
        p = d / 256
        f = np.maximum(1, np.floor(M * p ** np.arange(9) * (1 - p) ** (8 - np.arange(9))).astype(np.int64))
        big = int(np.argmax(f * sizes))
        f[big] += (M - int((f * sizes).sum())) // sizes[big]  # floor: also fixes an overshoot from the min of 1
        f[0 if d < 128 else 8] += M - int((f * sizes).sum())  # the rest (< sizes[big]) to a one-symbol class
        assert (f >= 1).all() and int((f * sizes).sum()) == M
        cum[d, 1:] = np.cumsum(f * sizes)
        freq[d] = f
    return freq, cum


FREQ, CUM = build_model()


@njit(cache=True)
def popcount64(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return (x * np.uint64(0x0101010101010101)) >> np.uint64(56)


@njit(cache=True)
def densities(planes):
    """planes: (16, G) uint8 -> density byte per plane."""
    out = np.zeros(planes.shape[0], np.uint8)
    for j in range(planes.shape[0]):
        ones = 0
        for i in range(planes.shape[1]):
            ones += POP_T[planes[j, i]]
        if ones > 0:
            d = (ones * 256 + planes.shape[1] * 4) // (planes.shape[1] * 8)
            out[j] = min(max(d, 1), 255)
    return out


POP_T = POP.astype(np.uint8)


@njit(cache=True)
def encode(planes, dens, freq, cum, rank, pop, out):
    """rANS-encode the planes with nonzero density; returns the number of bytes written to out.

    Symbols are processed in reverse; bytes are written backwards from the end of `out`, and the
    stream is returned as out[pos:]: the caller slices. Returns pos."""
    g = planes.shape[1]
    nplanes = planes.shape[0]
    state = np.empty(LANES, np.uint64)
    for s in range(LANES):
        state[s] = RANS_L
    pos = out.shape[0]
    total = 0
    for j in range(nplanes):
        if dens[j]:
            total += g
    idx = total
    for j in range(nplanes - 1, -1, -1):
        d = dens[j]
        if d == 0:
            continue
        fd = freq[d]
        cd = cum[d]
        for i in range(g - 1, -1, -1):
            idx -= 1
            lane = idx & (LANES - 1)
            b = planes[j, i]
            k = pop[b]
            f = np.uint64(fd[k])
            start = np.uint64(cd[k] + rank[b] * fd[k])
            x = state[lane]
            xmax = ((np.uint64(RANS_L) >> np.uint64(PB)) << np.uint64(8)) * f
            while x >= xmax:
                pos -= 1
                out[pos] = np.uint8(x & np.uint64(0xFF))
                x >>= np.uint64(8)
            state[lane] = ((x // f) << np.uint64(PB)) + (x % f) + start
    for s in range(LANES - 1, -1, -1):
        x = state[s]
        for _ in range(4):
            pos -= 1
            out[pos] = np.uint8(x & np.uint64(0xFF))
            x >>= np.uint64(8)
    return pos


@njit(cache=True)
def decode(stream, dens, freq, cum, bycls, cstart, out):
    """Inverse of encode: fills out (16, G) uint8 (zero planes untouched, assumed zeroed)."""
    g = out.shape[1]
    state = np.empty(LANES, np.uint64)
    p = 0
    for s in range(LANES):
        x = np.uint64(0)
        for _ in range(4):
            x = (x << np.uint64(8)) | np.uint64(stream[p])
            p += 1
        state[s] = x
    idx = 0
    mask = np.uint64(M - 1)
    for j in range(out.shape[0]):
        d = dens[j]
        if d == 0:
            continue
        fd = freq[d]
        cd = cum[d]
        for i in range(g):
            lane = idx & (LANES - 1)
            idx += 1
            x = state[lane]
            slot = np.int64(x & mask)
            k = 0
            while k < 8 and slot >= cd[k + 1]:
                k += 1
            f = fd[k]
            r = (slot - cd[k]) // f
            out[j, i] = bycls[cstart[k] + r]
            x = np.uint64(f) * (x >> np.uint64(PB)) + np.uint64(slot - cd[k] - r * f)
            while x < np.uint64(RANS_L):
                x = (x << np.uint64(8)) | np.uint64(stream[p])
                p += 1
            state[lane] = x


def arith_encode(planes):
    dens = densities(planes)
    buf = np.empty(planes.size + 64, np.uint8)
    pos = encode(planes, dens, FREQ, CUM, RANK, POP, buf)
    return dens, buf[pos:]


def arith_decode(dens, stream, shape):
    out = np.zeros(shape, np.uint8)
    decode(stream, dens, FREQ, CUM, BY_CLASS, CLASS_START, out)
    return out


def best(f, reps):
    t = np.inf
    for _ in range(reps):
        t0 = time.perf_counter()
        f()
        t = min(t, time.perf_counter() - t0)
    return t


def bit_planes(x):
    """(units' 16 x G planes) for a series, from real fluxcode units."""
    units, *_ = fluxcode.encode(x, Params(planes="bit"))
    for u in units:
        parsed = _unit.decompress(u)
        nb = parsed.header.num_blocks
        sizes = parsed.block_sizes
        g = int(((sizes + 7) // 8).sum())
        start = _format.residual_start(nb, parsed.has_time)
        yield parsed.raw_body, np.ascontiguousarray(parsed.raw_body[start:start + 16 * g].reshape(16, g))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=int, default=2)
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()
    zc = zstandard.ZstdCompressor(level=3, write_checksum=False, write_content_size=True)
    zd = zstandard.ZstdDecompressor()
    sigs = KINDS + ["sensor-0.1", "random-walk q0.01", "noisy-sine q0.1"]
    print(f"{'signal':20s} {'planes B':>8s} | {'zstd B':>7s} {'enc us':>7s} {'dec us':>7s} | "
          f"{'arith B':>7s} {'enc us':>7s} {'dec us':>7s} | {'size':>6s} {'enc x':>6s} {'dec x':>6s}")
    tot = np.zeros(7)
    for sig in sigs:
        if sig in KINDS or sig == "sensor-0.1":
            x = np.concatenate([minute(sig, 7000 + j) for j in range(a.minutes)])
        else:
            x = np.concatenate([discrete_minute(sig, 7000 + j) for j in range(a.minutes)])
        row = np.zeros(7)
        for body, planes in bit_planes(x):
            raw = planes.tobytes()
            z = zc.compress(raw)
            dens, stream = arith_encode(planes)
            assert np.array_equal(arith_decode(dens, stream, planes.shape), planes), sig
            row += [planes.size, len(z), best(lambda: zc.compress(raw), a.reps) * 1e6,
                    best(lambda: zd.decompress(z), a.reps) * 1e6,
                    len(stream) + 16, best(lambda: arith_encode(planes), a.reps) * 1e6,
                    best(lambda: arith_decode(dens, stream, planes.shape), a.reps) * 1e6]
        tot += row
        zb, ze, zdt, ab, ae, ad = row[1], row[2], row[3], row[4], row[5], row[6]
        print(f"{sig:20s} {row[0]:8.0f} | {zb:7.0f} {ze:7.0f} {zdt:7.0f} | {ab:7.0f} {ae:7.0f} {ad:7.0f} | "
              f"{ab / zb:6.2f} {ae / ze:6.2f} {ad / zdt:6.2f}")
    zb, ze, zdt, ab, ae, ad = tot[1:]
    print(f"{'TOTAL':20s} {tot[0]:8.0f} | {zb:7.0f} {ze:7.0f} {zdt:7.0f} | {ab:7.0f} {ae:7.0f} {ad:7.0f} | "
          f"{ab / zb:6.2f} {ae / ze:6.2f} {ad / zdt:6.2f}")
    print("\nsize/enc x/dec x = arithmetic / zstd (below 1 for time means arithmetic is faster)")


if __name__ == "__main__":
    main()
