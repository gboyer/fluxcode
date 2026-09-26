# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""cw-delta1-tree-N: delta1 with hierarchical block floating point over the deltas.

    q      = 32-bit grid over [min, max];  d = diff(q)   (999 deltas)
    tree   = the binary tree of the bintree codec, over the deltas
             (bottom-up pairing; 1 bit per internal node, root included)
    scale  : root starts at S0; a node's bit is 0 (keep its parent's scale) or
             1 (halve it); a leaf uses its parent's final scale s
    leaf   : code c, decoded delta = c * s * dir, where dir is the direction of
             the last nonzero decoded delta (starts +1); a negative code flips dir.
             2-bit leaves: c in {-1, 0, 1, 2}; wider leaves: two's-complement range
             (4-bit: -8 .. 7, 7-bit: -64 .. 63). 7-bit leaves + 1-bit nodes
             amortize to ~8 bits/sample.
    encode : node bits greedy top-down from the original deltas (halve while the
             subtree's largest delta still fits at half scale); leaves closed loop
             with integer residuals against the decoder's reconstruction.

`fit` chooses what "fits" means: "flip" requires the delta to fit even in the
sign-flipping direction (range 1*s for 2-bit, 8*s for 4-bit); "same" only in the
current direction (2*s, 7*s), halving more often at the risk of clipping;
"aware" sizes each leaf for the direction it will actually have (continuing:
hi_code * s, reversing: -lo_code * s, judged from the original deltas' signs).

Layout: max f64 | min f64 | q0 u32 | S0 u64 | node bits (level order) | leaf codes.
"""

import struct

import numpy as np
from numba import njit

from tslab.classic.bintree import tree_levels
from tslab.common.unit import Concat

MASK32 = (1 << 32) - 1
HEADER = struct.calcsize(">ddIQ")  # 28 bytes


def _structure(n_deltas):
    """Per internal level (root first): parent index of each node in the level
    above, plus parent-of-leaf indices. Fixed for a given block length."""
    levels = tree_levels(n_deltas)  # root first, leaves last (spans over delta indices)
    parents = []
    for upper, lower in zip(levels[:-1], levels[1:]):
        parents.append(np.arange(len(lower)) // 2)  # child k of the level above -> parent k // 2
    return levels, parents


@njit(cache=True)
def _encode_leaves(q, scale, lo_code, hi_code, codes):
    recon = q[0]
    direction = 1
    for i in range(codes.shape[0]):
        s = scale[i]
        r = q[i + 1] - recon  # integer residual
        step = s * direction
        # round(r / step) for either sign of step
        c = (2 * r + step) // (2 * step) if step > 0 else (-2 * r - step) // (-2 * step)
        c = min(max(c, lo_code), hi_code)
        codes[i] = c
        recon += c * step
        if c < 0:
            direction = -direction


@njit(cache=True)
def _decode_leaves(codes, scale, q0, v):
    v[0] = q0
    direction = 1
    for i in range(codes.shape[0]):
        c = codes[i]
        v[i + 1] = v[i] + c * scale[i] * direction
        if c < 0:
            direction = -direction


class Delta1Tree:
    """cw-delta1-tree-<leaf_bits>: 1 halving bit per internal node, leaf_bits-wide sign-relative
    leaves (see the module docstring). The fit variant is a suffix unless it is "aware"."""

    def __init__(self, leaf_bits=2, fit="flip"):
        assert 2 <= leaf_bits <= 8 and fit in ("flip", "same", "aware")
        self.leaf_bits, self.fit = leaf_bits, fit
        self.lo_code, self.hi_code = (-1, 2) if leaf_bits == 2 else (-(1 << (leaf_bits - 1)), (1 << (leaf_bits - 1)) - 1)
        # capacity of a scale s: largest |delta| representable
        self.cap = -self.lo_code if fit == "flip" else self.hi_code
        self.name = f"cw-delta1-tree-{leaf_bits}" + {"flip": "-flip", "same": "-same", "aware": ""}[fit]
        how = {"flip": ", sign-flipping fit", "same": ", same-direction fit", "aware": ""}[fit]
        self.label = f"Delta tree: 1 halving bit per node, {leaf_bits}-bit sign-relative leaves{how}"
        self._cache = {}

    def _struct(self, m):
        if m not in self._cache:
            self._cache[m] = _structure(m)
        return self._cache[m]

    def _leaf_scales(self, bits_by_level, S0, levels, parents):
        """Top-down: scale of each node = parent's scale >> own bit; leaves inherit."""
        scale = np.array([S0], np.int64) >> bits_by_level[0]
        for lvl in range(1, len(levels) - 1):
            scale = scale[parents[lvl - 1]] >> bits_by_level[lvl]
        return scale[parents[-1]]

    def encode(self, x):
        lo, hi = float(x.min()), float(x.max())
        scale_f = MASK32 / (hi - lo) if hi > lo else 0.0
        q = np.rint((x - lo) * scale_f).astype(np.int64)
        dq = np.diff(q)
        d = np.abs(dq)
        levels, parents = self._struct(len(d))
        if self.fit == "aware":
            # direction each leaf will see = sign of the previous nonzero original delta (starts +1)
            sgn = np.sign(dq)
            idx = np.where(sgn != 0, np.arange(len(sgn)), -1)
            last = np.maximum.accumulate(idx)
            prev_dir = np.ones(len(dq), np.int64)
            prev_dir[1:] = np.where(last[:-1] >= 0, sgn[np.maximum(last[:-1], 0)], 1)
            same = (sgn == 0) | (sgn == prev_dir)
            cap_each = np.where(same, self.hi_code, -self.lo_code)
            # Required scale per leaf; capacity becomes 1 per unit scale. (A half-step margin for
            # carried-in error was tried: fewer clipped leaves but coarser scales, worse overall.)
            d = -(-d // cap_each)
            cap = 1
        else:
            cap = self.cap
        # subtree max (|delta| or required scale) per node, bottom-up
        maxes = [d]
        for lvl in range(len(levels) - 2, -1, -1):
            child = maxes[0]
            pad = np.append(child, 0) if len(child) % 2 else child
            maxes.insert(0, np.maximum(pad[0::2], pad[1::2]))
        maxes = maxes[:-1]  # internal levels only, root first
        S0 = max(int(-(-d.max() // cap)), 1)  # root scale covers the largest delta
        bits_by_level = []
        scale = np.array([S0], np.int64)
        for lvl, mx in enumerate(maxes):
            if lvl > 0:
                scale = scale[parents[lvl - 1]]
            b = ((mx <= cap * (scale >> 1)) & (scale > 1)).astype(np.int64)  # halve if it still fits
            bits_by_level.append(b)
            scale = scale >> b
        leaf_scale = scale[parents[-1]]
        codes = np.empty(len(d), np.int64)
        _encode_leaves(q, leaf_scale, self.lo_code, self.hi_code, codes)
        node_bits = np.packbits(np.concatenate(bits_by_level).astype(np.uint8))
        u = (codes - self.lo_code).astype(np.uint8)  # leaf_bits-wide unsigned codes, packed MSB first
        packed = np.packbits(np.unpackbits(u[:, None], axis=1)[:, 8 - self.leaf_bits:].ravel())
        info = {"halved": float(np.mean(np.concatenate(bits_by_level))),
                "clipped": float(np.mean((codes == self.lo_code) | (codes == self.hi_code)))}
        return struct.pack(">ddIQ", hi, lo, int(q[0]), S0) + node_bits.tobytes() + packed.tobytes(), info

    def decode(self, data, n):
        hi, lo, q0, S0 = struct.unpack(">ddIQ", data[:HEADER])
        levels, parents = self._struct(n - 1)
        n_nodes = sum(len(l) for l in levels[:-1])
        nb = -(-n_nodes // 8)
        flat = np.unpackbits(np.frombuffer(data, np.uint8, nb, HEADER))[:n_nodes].astype(np.int64)
        bits_by_level, pos = [], 0
        for l in levels[:-1]:
            bits_by_level.append(flat[pos:pos + len(l)])
            pos += len(l)
        leaf_scale = self._leaf_scales(bits_by_level, S0, levels, parents)
        bits = np.unpackbits(np.frombuffer(data, np.uint8, offset=HEADER + nb))[:(n - 1) * self.leaf_bits]
        u = bits.reshape(n - 1, self.leaf_bits) @ (1 << np.arange(self.leaf_bits - 1, -1, -1))
        codes = u.astype(np.int64) + self.lo_code
        v = np.empty(n, np.int64)
        _decode_leaves(codes, leaf_scale, np.int64(q0), v)
        return lo + v * ((hi - lo) / MASK32)


# The direction-aware halving test was best for every leaf width; the other two are kept for reference.
DELTA_TREE_CODECS = [Concat(Delta1Tree(bits, "aware")) for bits in (2, 4, 7)]
