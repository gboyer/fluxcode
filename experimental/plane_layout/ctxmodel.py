# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""How far could a context-modelled bit-plane coder (EBCOT / adaptive Exp-Golomb style) go on the
residuals? Ideal adaptive code lengths, no coder implemented: each decision costs -log2 of an
adaptive count estimate, which practical arithmetic coders approach within a percent or two.

The binarization is the one JPEG2000-style coders and adaptive Golomb coders amount to: for
each sample, (1) its bit length L = 0..16 (which is where the significance of the planes is
decided, plane by plane from the top) as one symbol conditioned on the neighbouring samples'
bit lengths, then (2) the L - 1 bits below its leading one: the first `refine` of them adaptively
modelled given (L, the bits so far), the rest sent raw.

    uv run python plane_layout/ctxmodel.py
"""

import re
import sys
from pathlib import Path

import numpy as np
import zstandard
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model
from layouts import group_streams

HERE = Path(__file__).resolve().parent
ALPHA = 0.25


@njit(cache=True)
def bitlen(v):
    n = 0
    while v:
        n += 1
        v >>= 1
    return n


@njit(cache=True)
def cost(u, ctx_mode, refine):
    """Ideal adaptive bits to code the array u (uint16). ctx_mode: 0 none, 1 previous bit length,
    2 previous two, 3 previous and the mean of the two before."""
    nctx = 17 * 17
    counts = np.zeros((nctx, 17))
    tot = np.zeros(nctx)
    mant = np.zeros((17 * 1024, 2))
    bits = 0.0
    b1 = b2 = b3 = 0
    for i in range(u.shape[0]):
        v = np.int64(u[i])
        L = bitlen(v)
        if ctx_mode == 0:
            c = 0
        elif ctx_mode == 1:
            c = b1
        elif ctx_mode == 2:
            c = b1 * 17 + b2
        else:
            c = b1 * 17 + (b2 + b3 + 1) // 2
        bits -= np.log2((counts[c, L] + ALPHA) / (tot[c] + 17 * ALPHA))
        counts[c, L] += 1
        tot[c] += 1
        m = L - 1
        prefix = 1
        for k in range(m):
            bit = (v >> (m - 1 - k)) & 1
            if k < refine:
                ci = L * 1024 + prefix
                bits -= np.log2((mant[ci, bit] + ALPHA) / (mant[ci, 0] + mant[ci, 1] + 2 * ALPHA))
                mant[ci, bit] += 1
                prefix = prefix * 2 + bit
            else:
                bits += 1.0
        b3, b2, b1 = b2, b1, L
    return bits


def evaluate(name, sample=None):
    d = np.load(HERE / name)
    U = d["u"] if sample is None else d["u"][:: max(1, len(d["u"]) // sample)]
    zc = zstandard.ZstdCompressor(level=3, write_checksum=False, write_content_size=True)
    rows = []
    for u in U:
        sb, sy = model.streams(u)
        nib = group_streams(u, [4, 4, 4, 4])
        z_bit, z_byte = len(zc.compress(sb.tobytes())), len(zc.compress(sy.tobytes()))
        z_nib = sum(len(zc.compress(s.tobytes())) for s in nib)
        rows.append([z_bit, z_byte, z_nib] + [cost(u, m, r) / 8 for m, r in CONFIGS.values()])
    names = np.array([re.sub(r" P=.*", "", f) for f in d["fam"]])
    if sample is not None:
        names = names[:: max(1, len(d["fam"]) // sample)]
    return names, np.array(rows)


CONFIGS = {"order-0 bit length, raw mantissa": (0, 0), "ctx: previous L, raw mantissa": (1, 0), "ctx: previous 2 L, raw mantissa": (2, 0),
           "ctx: previous 2 L, 2 modelled mantissa bits": (2, 2), "ctx: L, mean of 2 before; 2 modelled bits": (3, 2)}


def report(label, names, r, by_family=False):
    n = len(r)
    bps = lambda a: a.sum() * 8 / (60_000 * n)
    pair = np.minimum(r[:, 0], r[:, 1])
    print(f"\n{label}: {n} units; plane bytes only, bits/sample (zstd rows are today's layouts and nibble-split)")
    rowsout = [("zstd 3, bit planes", r[:, 0]), ("zstd 3, byte planes", r[:, 1]), ("zstd 3, best of bit/byte (today)", pair),
               ("zstd 3, nibble planes split", r[:, 2])]
    rowsout += [(f"ideal: {k}", r[:, 3 + j]) for j, k in enumerate(CONFIGS)]
    best_ctx = r[:, 3:].min(axis=1)
    rowsout += [("oracle of best ideal config and best zstd", np.minimum(best_ctx, np.minimum(pair, r[:, 2])))]
    for k, a in rowsout:
        print(f"  {k:52s} {bps(a):7.3f} {a.sum() / pair.sum() - 1:+8.1%}")
    m = r[:, 6] < pair
    print(f"  the ideal context model (previous 2 L, 2 modelled mantissa bits) beats today's zstd on {m.mean():.0%} of units")
    if by_family:
        print(f"\n  by family: ideal (previous 2 L, 2 modelled bits) / today's best zstd")
        for f in np.unique(names):
            k = names == f
            print(f"   {f:18s} {r[k, 6].sum() / pair[k].sum():6.2f}   (ideal beats zstd on {np.mean(r[k, 6] < pair[k]):.0%})")


if __name__ == "__main__":
    for label, name, sample, fam in (("fit sample", "corpus.npz", 500, True), ("report signals", "standard.npz", None, False)):
        names, r = evaluate(name, sample)
        report(label, names, r, fam)
