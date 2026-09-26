# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Knot-based piecewise codecs: min/max knots per 16 samples, linear or PCHIP interpolation."""

import numpy as np
from scipy.interpolate import PchipInterpolator

from tslab.classic.quant import dequantize, quantize
from tslab.common.bitio import BitReader, BitWriter

# ---------------------------------------------------------------------------
# Piecewise (knot-based) codecs: shared knot encoding
#   header: max f64, min f64
#   per knot: 4-bit offset (0 -> next sample, 15 -> 16 ahead), 12-bit value
# The first offset is relative to index -1, and the stream ends at knot n-1.
# ---------------------------------------------------------------------------

MAX_GAP = 16


def fill_gaps(idx):
    """Insert knots so no two consecutive knots are more than MAX_GAP apart."""
    out = [idx[0]]
    for i in idx[1:]:
        while i - out[-1] > MAX_GAP:
            out.append(out[-1] + MAX_GAP)
        out.append(i)
    return out


def encode_knots(x, idx):
    lo, hi = x.min(), x.max()
    w = BitWriter()
    w.write_f64(hi)
    w.write_f64(lo)
    prev = -1
    for i, q in zip(idx, quantize(x[idx], lo, hi, 12)):
        assert 1 <= i - prev <= MAX_GAP
        w.write(i - prev - 1, 4)
        w.write(q, 12)
        prev = i
    return w.getvalue()


def decode_knots(data, n):
    r = BitReader(data)
    hi, lo = r.read_f64(), r.read_f64()
    idx, qs = [], []
    prev = -1
    while prev < n - 1:
        prev += r.read(4) + 1
        idx.append(prev)
        qs.append(r.read(12))
    return np.array(idx), dequantize(qs, lo, hi, 12)


def minmax_knots(x, group=16, consolidate=False):
    n = len(x)
    blocks = []
    for s in range(0, n, group):
        seg = x[s:s + group]
        blocks.append(sorted({s + int(np.argmin(seg)), s + int(np.argmax(seg))}))
    if consolidate:
        # If block i ends on an extreme at its last sample and block i+1 starts on
        # the opposite extreme at its first sample, and the series keeps moving in
        # the same direction across the boundary, the second point is redundant for
        # a monotone run: keep only the first.
        for i in range(len(blocks) - 1):
            s_next = (i + 1) * group
            a, b = blocks[i][-1], blocks[i + 1][0] if blocks[i + 1] else None
            if b is None or a != s_next - 1 or b != s_next or len(blocks[i + 1]) < 2:
                continue
            seg_a = x[i * group:s_next]
            a_is_max = x[a] == seg_a.max()
            b_is_min = x[b] == x[s_next:s_next + group].min()
            rising = a_is_max and b_is_min and x[b] >= x[a]
            falling = (not a_is_max) and (not b_is_min) and x[b] <= x[a]
            if rising or falling:
                blocks[i + 1] = blocks[i + 1][1:]
    idx = sorted(set(i for blk in blocks for i in blk) | {0, n - 1})
    return fill_gaps(idx)


class PiecewiseLinearMinMax:
    """pwlinear-minmax16: knots at the min and max of every 16-sample group (plus the first and
    last sample), linear interpolation. consolidate=True (-merged) drops a redundant knot where a
    monotone run crosses a group boundary."""

    family, interp = "pwlinear", "Piecewise linear"
    variable_size = True

    def __init__(self, consolidate=False):
        self.consolidate = consolidate
        self.name = f"{self.family}-minmax16" + ("-merged" if consolidate else "")
        self.label = f"{self.interp} through the min and max of every 16 samples" + (", boundary knots merged" if consolidate else "")

    def encode(self, x):
        idx = minmax_knots(x, consolidate=self.consolidate)
        return encode_knots(x, idx), {"knots": idx}

    def decode(self, data, n):
        idx, v = decode_knots(data, n)
        return np.interp(np.arange(n), idx, v)


class PchipMinMax(PiecewiseLinearMinMax):
    """pchip-minmax16: the same knots and bytes, decoded with monotone cubic (PCHIP) interpolation."""

    family, interp = "pchip", "PCHIP cubic"

    def decode(self, data, n):
        idx, v = decode_knots(data, n)
        return PchipInterpolator(idx, v)(np.arange(n))
