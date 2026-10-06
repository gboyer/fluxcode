# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Does storing near-random bit planes raw (instead of through zstd) help the fluxcode prototype?

Method: bit-shuffled fluxproto block groups with near-random planes stored raw instead of through zstd.

A plane (all blocks of a minute, or one block's plane) is "near-random" if its density of
ones p satisfies |p - 0.5| < t. Kept planes (+ header + a 1-bit-per-plane mask) go through
zstd-3; raw planes are appended uncompressed. Orders 0-3, decimal detect, B = 16.
Sizes leave out the block minima (8 bytes per block, the same in every variant).
Single thread; µs per 1000-sample block, best of 5.

Data: tslab.common.datasets.load() and load_discrete().

    uv run python -m bench.fluxproto_raw_planes
"""

import os

os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import time

import numpy as np
import zstandard
from numba import njit

from tslab.common.datasets import load, load_discrete
from tslab.flux import proto as fp

POP = np.array([bin(i).count("1") for i in range(256)], np.int64)


@njit(cache=True)
def plane_ones(planes, pop):
    """planes (16, nb, g) uint8 -> ones (16, nb)."""
    P, nb, g = planes.shape
    out = np.zeros((P, nb), np.int64)
    for j in range(P):
        for b in range(nb):
            s = 0
            for i in range(g):
                s += pop[planes[j, b, i]]
            out[j, b] = s
    return out


def split(planes, ones, t, per_block):
    """Returns (mask bytes, kept bytes, raw bytes)."""
    P, nb, g = planes.shape
    nbits = 8 * g
    if per_block:
        raw = np.abs(ones / nbits - 0.5) < t  # (16, nb)
    else:
        raw = np.repeat((np.abs(ones.sum(1) / (nbits * nb) - 0.5) < t)[:, None], nb, axis=1)
    flat = planes.reshape(P * nb, g)
    r = raw.ravel()
    return np.packbits(r), flat[~r].ravel(), flat[r].ravel()


def main():
    signal_sets = [("continuous", load()), ("discretized", [m[:4] for m in load_discrete()])]
    codec = fp.FluxProto(16, step_detect=True, planes="bit")
    zc = zstandard.ZstdCompressor(level=3, write_checksum=False)
    zd = zstandard.ZstdDecompressor()
    groups = {g: [] for g, _ in signal_sets}
    for g, mins in signal_sets:
        for kind, X, lo, hi in mins:
            raw = codec.raw(X, lo, hi)
            nb = len(X)
            hdr, planes = raw[:9 * nb], raw[9 * nb:].reshape(16, nb, 125)
            groups[g].append((kind, hdr, planes, plane_ones(planes, POP)))
    configs = [("all planes through zstd (current)", None, False)]
    configs += [(f"minute-plane raw if |p-0.5|<{t}", t, False) for t in (0.01, 0.02, 0.05)]
    configs += [(f"block-plane raw if |p-0.5|<{t}", t, True) for t in (0.01, 0.02, 0.05, 0.1)]
    print("| layout | continuous bits/sample | discretized bits/sample | planes raw | encode µs (zstd + split) | decode µs (zstd) |")
    print("|---|---|---|---|---|---|")
    per_kind = {}
    for name, t, pb in configs:
        cells, nraw, nplanes, te, td = [], 0, 0, 0.0, 0.0
        for g, _ in signal_sets:
            size = 0
            nblocks = 0
            for kind, hdr, planes, ones in groups[g]:
                if t is None:
                    enc = lambda: (zc.compress(np.concatenate([hdr, planes.ravel()]).tobytes()), b"")
                else:
                    def enc(hdr=hdr, planes=planes, ones=ones):
                        m, kept, rawb = split(planes, ones, t, pb)
                        return zc.compress(np.concatenate([hdr, m, kept]).tobytes()), rawb.tobytes()
                best_e = np.inf
                for _ in range(5):
                    t0 = time.perf_counter()
                    z, rb = enc()
                    best_e = min(best_e, time.perf_counter() - t0)
                best_d = np.inf
                for _ in range(5):
                    t0 = time.perf_counter()
                    zd.decompress(z)
                    best_d = min(best_d, time.perf_counter() - t0)
                te += best_e
                td += best_d
                size += len(z) + len(rb)
                nblocks += planes.shape[1]
                nraw += len(rb) // 125
                nplanes += 16 * planes.shape[1]
                k = (kind, g)
                per_kind.setdefault(k, {})[name] = per_kind.get(k, {}).get(name, 0) + len(z) + len(rb)
            cells.append(8 * size / (nblocks * 1000))
        ntot = sum(16 * p.shape[1] for g, _ in signal_sets for _, _, p, _ in groups[g]) // 16
        print(f"| {name} | {cells[0]:.3f} | {cells[1]:.3f} | {100 * nraw / nplanes:.0f}% | "
              f"{1e6 * te / ntot:.2f} | {1e6 * td / ntot:.2f} |", flush=True)
    names = [c[0] for c in configs]
    print("\nPer signal, bits/sample: current vs block-plane raw at 0.02 and 0.05")
    for (kind, g), d in per_kind.items():
        nbk = 60 * sum(1 for k, *_ in groups[g] if k == kind)
        a, b, c = (8 * d[n] / (nbk * 1000) for n in (names[0], "block-plane raw if |p-0.5|<0.02", "block-plane raw if |p-0.5|<0.05"))
        print(f"  {kind} ({g}): {a:.2f} -> {b:.2f} / {c:.2f}")


if __name__ == "__main__":
    main()
