# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Does a relative-sign zigzag shrink the fluxcode prototype's units?

Relative-sign zigzag: negate each residual when the last nonzero residual was negative, then
zigzag, so the sign bit (and zigzag's sign-copied low bits) record "same/opposite sign as before".
Applied to the codec's own residuals (same order, grid, decimal and noise choices), bit-shuffled,
zstd-3. B = 16, orders 0-3, decimal detect (fluxproto); noise floor off and f = 1.
Sizes leave out the block minima (8 bytes per block, the same in every variant).

Data: tslab.common.datasets.load() and load_discrete().

    uv run python -m bench.fluxproto_relative_sign
"""

import os

os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import numpy as np
import zstandard
from numba import njit

from tslab.common.datasets import load, load_discrete
from tslab.flux import proto as fp

Z = zstandard.ZstdCompressor(level=3, write_checksum=False)


@njit(cache=True)
def relsign(U):
    """U: (nb, n) zigzag uint16 -> relative-sign zigzag. Sign state resets per block."""
    nb, n = U.shape
    out = np.empty_like(U)
    for b in range(nb):
        neg = False
        for i in range(n):
            z = np.int64(U[b, i])
            v = (z >> 1) ^ -(z & 1)
            w = -v if neg else v
            if w == 32768:  # -(-32768) wraps mod 2^16
                w = -32768
            out[b, i] = np.uint16(((w << 1) ^ (w >> 63)) & 0xFFFF)
            if v != 0:
                neg = v < 0
    return out


@njit(cache=True)
def unrelsign(R):
    nb, n = R.shape
    out = np.empty_like(R)
    for b in range(nb):
        neg = False
        for i in range(n):
            z = np.int64(R[b, i])
            w = (z >> 1) ^ -(z & 1)
            v = -w if neg else w
            if v == 32768:
                v = -32768
            out[b, i] = np.uint16(((v << 1) ^ (v >> 63)) & 0xFFFF)
            if v != 0:
                neg = v < 0
    return out


def unit(head_param, U):
    nb = U.shape[0]
    low, high = (U & 255).astype(np.uint8), (U >> 8).astype(np.uint8)
    planes = np.empty((16, nb, U.shape[1] // 8), np.uint8)
    fp._bitshuffle(np.ascontiguousarray(low).view(np.uint64), np.ascontiguousarray(high).view(np.uint64), planes)
    return len(Z.compress(np.concatenate([head_param, planes.ravel()]).tobytes()))


def main():
    groups = [("continuous", load()), ("discretized", [m[:4] for m in load_discrete()])]
    print("| signal | zigzag | relative-sign | change | zigzag, noise f=1 | relative-sign, noise f=1 | change | sign plane density (zigzag → relative) |")
    print("|---|---|---|---|---|---|---|---|")
    tot = {}
    for g, mins in groups:
        for kind in dict.fromkeys(m[0] for m in mins):
            row = []
            dens = None
            for f in (0.0, 1.0):
                c = fp.FluxProto(16, step_detect=True, noise_f=f)  # byte planes: easy to read U back
                a = b = 0
                d0 = d1 = 0.0
                for m in mins:
                    if m[0] != kind:
                        continue
                    raw = c.raw(*m[1:4])
                    nb = len(m[1])
                    U = (raw[9 * nb:9 * nb + nb * 1000].astype(np.uint16) | (raw[9 * nb + nb * 1000:].astype(np.uint16) << 8)).reshape(nb, 1000)
                    R = relsign(U)
                    assert np.array_equal(unrelsign(R), U)
                    a += unit(raw[:9 * nb], U)
                    b += unit(raw[:9 * nb], R)
                    d0 += np.mean(U & 1)
                    d1 += np.mean(R & 1)
                cnt = sum(1 for m in mins if m[0] == kind)
                nbk = 60000 * cnt
                row += [8 * a / nbk, 8 * b / nbk]
                tot[g, f] = tuple(np.add(tot.get((g, f), (0, 0)), (8 * a, 8 * b)))
                if f == 0.0:
                    dens = (d0 / cnt, d1 / cnt)
            ch = lambda x, y: f"{100 * (y - x) / x:+.1f}%" if x > 0.02 else "—"
            print(f"| {kind} ({g}) | {row[0]:.2f} | {row[1]:.2f} | {ch(row[0], row[1])} | {row[2]:.2f} | {row[3]:.2f} | "
                  f"{ch(row[2], row[3])} | {dens[0]:.2f} → {dens[1]:.2f} |", flush=True)
    for g, mins in groups:
        nbk = 60000 * len(mins)
        a0, b0 = tot[g, 0.0]
        a1, b1 = tot[g, 1.0]
        print(f"| **all {g}** | {a0 / nbk:.2f} | {b0 / nbk:.2f} | {100 * (b0 - a0) / a0:+.1f}% | {a1 / nbk:.2f} | {b1 / nbk:.2f} | "
              f"{100 * (b1 - a1) / a1:+.1f}% | |")


if __name__ == "__main__":
    main()
