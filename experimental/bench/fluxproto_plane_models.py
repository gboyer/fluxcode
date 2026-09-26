# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Could a per-bit-plane probability model beat bit-shuffle + zstd-3 on the fluxcode prototype's residuals?

Method: fluxproto (orders 0-3, decimal detect, B = 16) residual bit planes, costed under several
models against the real bit-shuffle + zstd-3 size.

Sizes are ideal arithmetic-coding costs (entropy under each model), plus the stored model
parameters and the zstd-compressed block headers; the zstd column is the real codec's compressed
payload. Every column leaves out the block minima (8 bytes per block, the same in every model).
No coder is implemented: this measures what the models could reach.

Data: tslab.common.datasets.load() and load_discrete().

    uv run python -m bench.fluxproto_plane_models
"""

import os

os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import math

import numpy as np
import zstandard
from numba import njit

from tslab.common.datasets import load, load_discrete
from tslab.flux import proto as fp


@njit(cache=True)
def ce_bits(ones, n, qbits):
    """Cost of n bits with `ones` ones under the empirical p quantized to qbits (ideal arithmetic coder)."""
    if ones == 0 or ones == n:
        return 0.0
    L = 1 << qbits
    pq = min(max(round(ones / n * L), 1), L - 1) / L
    return -(ones * math.log2(pq) + (n - ones) * math.log2(1 - pq))

@njit(cache=True)
def costs(U):
    """U: (nb, n) uint16 residuals of one minute. Returns bits for each model."""
    nb, n = U.shape
    static_blk = 0.0
    ctx_blk = 0.0
    nearhalf = 0
    coded = 0
    for b in range(nb):
        for j in range(16):
            ones = 0
            for i in range(n):
                ones += (U[b, i] >> j) & 1
            static_blk += 2 + ce_bits(ones, n, 8) + (8 if 0 < ones < n else 0)
            if 0 < ones < n:
                coded += 1
                if abs(ones / n - 0.5) < 0.05:
                    nearhalf += 1
            # context: has this sample a 1 in a higher plane (j+1..15)?
            n1 = o1 = n0 = o0 = 0
            for i in range(n):
                hi = (U[b, i] >> (j + 1)) != 0
                bit = (U[b, i] >> j) & 1
                if hi:
                    n1 += 1; o1 += bit
                else:
                    n0 += 1; o0 += bit
            for nn, oo in ((n0, o0), (n1, o1)):
                if nn > 0:
                    ctx_blk += 2 + ce_bits(oo, nn, 8) + (8 if 0 < oo < nn else 0)
    static_min = 0.0
    for j in range(16):
        ones = 0
        for b in range(nb):
            for i in range(n):
                ones += (U[b, i] >> j) & 1
        static_min += 2 + ce_bits(ones, nb * n, 12) + (12 if 0 < ones < nb * n else 0)
    adapt = np.zeros(2)
    for s, shift in enumerate((4, 5)):
        for j in range(16):
            p = 1024  # P(bit = 1) in 1/2048, LZMA-style
            for b in range(nb):
                for i in range(n):
                    bit = (U[b, i] >> j) & 1
                    pr = p / 2048.0
                    adapt[s] += -math.log2(pr if bit else 1 - pr)
                    if bit:
                        p += (2048 - p) >> shift
                    else:
                        p -= p >> shift
                    p = min(max(p, 31), 2017)
    return static_blk, static_min, adapt[0], adapt[1], ctx_blk, coded, nearhalf

bit = fp.FluxProto(16, step_detect=True, planes="bit")
byte = fp.FluxProto(16, step_detect=True)
z = zstandard.ZstdCompressor(level=3, write_checksum=False).compress
groups = [("continuous", load()), ("discretized", [m[:4] for m in load_discrete()])]
names = ["bit-shuffle + zstd-3", "static per block+plane", "static per minute+plane", "adaptive (shift 4)", "adaptive (shift 5)",
         "static per block+plane, magnitude context"]
tot = {g: np.zeros(len(names)) for g, _ in groups}
print("| signal | " + " | ".join(names) + " | coded planes/block | of which near p=0.5 |")
print("|---|" + "---|" * (len(names) + 2))
for g, mins in groups:
    for kind in dict.fromkeys(m[0] for m in mins):
        acc = np.zeros(len(names)); coded = half = 0; nbk = 0
        for m in [m for m in mins if m[0] == kind]:
            X, lo, hi = m[1:4]
            raw = byte.raw(X, lo, hi)
            nb = len(X)
            U = (raw[9 * nb:9 * nb + nb * 1000].astype(np.uint16) | (raw[9 * nb + nb * 1000:].astype(np.uint16) << 8)).reshape(nb, 1000)
            hdr = 8 * len(z(raw[:9 * nb].tobytes()))
            sb, sm, a4, a5, cx, cd, nh = costs(U)
            acc += [8 * len(bit._c(bit.raw(X, lo, hi))), sb + hdr, sm + hdr, a4 + hdr, a5 + hdr, cx + hdr]
            coded += cd; half += nh; nbk += nb
        bps = acc / (nbk * 1000)
        tot[g] += acc
        best = bps.min()
        print(f"| {kind} ({g}) | " + " | ".join(f"**{v:.2f}**" if v == best else f"{v:.2f}" for v in bps)
              + f" | {coded / nbk:.1f} | {half / nbk:.1f} |", flush=True)
for g, mins in groups:
    nbk = sum(len(m[1]) for m in mins)
    print(f"| **all {g}** | " + " | ".join(f"{v / (nbk * 1000):.2f}" for v in tot[g]) + " | | |")
