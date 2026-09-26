# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Is the noise floor better as a coarser block step or as low-bit truncation on the fine grid?

A   coarser step for the whole block (the current noise floor: e = e16 + k)
B   fine grid (e16), q rounded to multiples of 2^k: identical reconstruction to A,
    residual -> zigzag (as now) -> bit-shuffle -> zstd-3
B'  as B with sign-magnitude instead of zigzag (u = |v| << 1 | sign), so zeroed low bits stay zero
C   per-sample: samples near a second difference beyond 4 robust sigmas (spikes, edges; +/-2
    samples) keep full precision, the rest are truncated as in B' (and with zigzag)
Orders 0-3 picked per block on the final q; 60-block units, bit-shuffled, zstd-3 (the fluxcode
prototype's pipeline, tslab/flux/proto.py). Sizes leave out the block minima (8 bytes per block,
the same in every variant).

Data: tslab.common.datasets.noisy_sets() (noisy-sine, impulses, gauss-spikes).

    uv run python -m bench.fluxproto_truncation
"""

import os

os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import math

import numpy as np
import zstandard
from numba import njit

from tslab.common.datasets import noisy_sets
from tslab.common.intcode import pick_order
from tslab.flux import proto as fp

Z = zstandard.ZstdCompressor(level=3, write_checksum=False)


@njit(cache=True)
def signal_mask(x, width):
    """True where a second difference (centred on sample i+1) exceeds 4 robust sigmas, +/- width."""
    n = x.shape[0]
    m = n - 2
    d = np.empty(m)
    mean = 0.0
    for i in range(m):
        d[i] = x[i + 2] - 2.0 * x[i + 1] + x[i]
        mean += d[i]
    mean /= m
    mad = 0.0
    for i in range(m):
        d[i] -= mean
        mad += abs(d[i])
    mad /= m
    k = math.sqrt(math.pi / 2)
    for _ in range(2):
        c = 4.0 * k * mad
        s = 0.0
        cnt = 0
        for i in range(m):
            if abs(d[i]) <= c:
                s += abs(d[i])
                cnt += 1
        mad = s / cnt if cnt > 0 else 0.0
    c = 4.0 * k * mad
    out = np.zeros(n, np.bool_)
    for i in range(m):
        if abs(d[i]) > c:
            for j in range(max(0, i + 1 - width), min(n, i + 2 + width)):
                out[j] = True
    return out


@njit(cache=True)
def residual_sm(q, order, u):
    """As proto._residual_u16 but sign-magnitude: u = |v| << 1 | sign (v = -32768 wraps to 0x0001... clamp)."""
    n = q.shape[0]
    r = np.empty(n, np.int64)
    for i in range(n):
        r[i] = q[i]
    for _ in range(order):
        for i in range(n - 1, 0, -1):
            r[i] = r[i] - r[i - 1]
    for i in range(n):
        v = ((r[i] + 32768) & 0xFFFF) - 32768
        a = -v if v < 0 else v
        u[i] = np.uint16(((min(a, 32767) << 1) | (1 if v < 0 else 0)) & 0xFFFF)


def unit_bytes(Q, mapping):
    nb, n = Q.shape
    low, high = np.empty((nb, n), np.uint8), np.empty((nb, n), np.uint8)
    head = np.empty(nb, np.uint8)
    r, u = np.empty(n, np.int32), np.empty(n, np.uint16)
    for b in range(nb):
        q = Q[b].astype(np.int32)
        order, _ = pick_order(q)
        head[b] = order
        if mapping == "zigzag":
            fp._residual_u16(q, order, r, u)
        else:
            residual_sm(q, order, u)
        low[b], high[b] = u & 255, u >> 8
    planes = np.empty((16, nb, n // 8), np.uint8)
    fp._bitshuffle(low.view(np.uint64), high.view(np.uint64), planes)
    return len(Z.compress(np.concatenate([head, np.zeros(8 * nb, np.uint8), planes.ravel()]).tobytes()))


def main():
    sets = {k: v for k, v in noisy_sets().items() if k in ("noisy-sine", "impulses", "gauss-spikes")}
    print("| signal | f | A: coarser block step | B: fine grid, low bits zeroed (zigzag) | B': same, sign-magnitude | "
          "C: per-sample, spikes kept (sign-magnitude) | C with zigzag | C: RMS err noise samples (σ) | C: max err spike samples (σ) | A: max err spike samples (σ) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name, mins in sets.items():
        for f in (0.5, 1.0, 2.0):
            size = {"A": 0, "B": 0, "B'": 0, "C": 0, "Cz": 0}
            en, es, ea = [], [], []
            for X, lo, hi, C_, s in mins:
                nb = len(X)
                QA, QB, QC = (np.empty((nb, 1000), np.int64) for _ in range(3))
                for b in range(nb):
                    x = X[b]
                    e16 = fp._exponent(hi[b] - lo[b], 16)
                    sig, rho = fp._noise(x)
                    k = 0
                    if rho < -0.6 and sig > 0:
                        k = max(0, int(math.floor(math.log2(f * sig))) - e16)
                    q16 = np.floor((x - lo[b]) * 2.0 ** -e16 + 0.5).astype(np.int64)
                    qa = np.floor((x - lo[b]) * 2.0 ** -(e16 + k) + 0.5).astype(np.int64)
                    QA[b], QB[b] = qa, qa << k
                    keep = signal_mask(x, 2) if k > 0 else np.ones(1000, bool)
                    QC[b] = np.where(keep, q16, qa << k)
                    step = 2.0 ** e16
                    yA = lo[b] + QB[b] * step
                    yC = lo[b] + QC[b] * step
                    spikes = keep & (k > 0)
                    if (~keep).any():
                        en.append(np.sqrt(np.mean((yC[~keep] - x[~keep]) ** 2)) / s)
                    if spikes.any():
                        es.append(np.abs(yC[spikes] - x[spikes]).max() / s)
                        ea.append(np.abs(yA[spikes] - x[spikes]).max() / s)
                size["A"] += unit_bytes(QA, "zigzag")
                size["B"] += unit_bytes(QB, "zigzag")
                size["B'"] += unit_bytes(QB, "sm")
                size["C"] += unit_bytes(QC, "sm")
                size["Cz"] += unit_bytes(QC, "zigzag")
            nbk = 60 * len(mins)
            bps = {k: 8 * v / (nbk * 1000) for k, v in size.items()}
            bp = bps["B'"]
            print(f"| {name} | {f:g} | {bps['A']:.2f} | {bps['B']:.2f} | {bp:.2f} | {bps['C']:.2f} | {bps['Cz']:.2f} | "
                  f"{np.median(en):.3f} | {max(es) if es else float('nan'):.3f} | {max(ea) if ea else float('nan'):.3f} |",
                  flush=True)


if __name__ == "__main__":
    main()
