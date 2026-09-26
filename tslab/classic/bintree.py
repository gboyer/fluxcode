# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""The 2-bit hierarchical narrowing tree from the original request."""

import numpy as np

from tslab.common.bitio import BitReader, BitWriter

# code -> (lower, upper) fraction of the parent's value range
NARROW = {0b00: (0.0, 1.0), 0b01: (0.25, 0.75), 0b10: (0.0, 0.5), 0b11: (0.5, 1.0)}


def tree_levels(n):
    """Levels of sample spans, root first, leaves last.

    Built bottom-up by pairing neighbours; an odd node out becomes a unary
    parent. For n=1000 the internal levels have
    1+2+4+8+16+32+63+125+250+500 = 1001 nodes, and 1000 leaves.
    """
    levels = [[(i, i + 1) for i in range(n)]]
    while len(levels[-1]) > 1:
        prev = levels[-1]
        levels.append([(prev[i][0], prev[min(i + 1, len(prev) - 1)][1])
                       for i in range(0, len(prev), 2)])
    return levels[::-1]


class BinaryTree:
    """Each internal node's 2-bit code narrows the value range it inherits (none, middle half,
    lower half, upper half); each leaf bit picks the lower or upper half of its final range.
    leaf="center" (bintree-center) decodes a leaf bit to the center of that half; leaf="edge"
    (bintree-edge) decodes to the literal bottom/top of the range (same bytes)."""

    def __init__(self, leaf="center"):
        self.leaf = leaf
        self.name = f"bintree-{leaf}"
        self.label = f"2-bit narrowing tree, 1-bit leaves decoded to the {leaf} of their range"

    def encode(self, x):
        n = len(x)
        lo, hi = x.min(), x.max()
        u = (x - lo) / (hi - lo) if hi > lo else np.zeros(n)
        levels = tree_levels(n)
        w = BitWriter()
        w.write_f64(hi)
        w.write_f64(lo)
        counts = {c: 0 for c in NARROW}
        eps = 1e-12
        ranges = [(0.0, 1.0)]  # range handed down to each node on the current level
        for level, next_level in zip(levels[:-1], levels[1:]):
            node_ranges = []
            for (s, e), (a, b) in zip(level, ranges):
                smin, smax = u[s:e].min(), u[s:e].max()
                mid = (smin + smax) / 2
                best, best_dist = 0b00, None
                # Narrowest choice that still contains every sample in the span;
                # if two halvings contain it, pick the one centred nearest the data.
                for code in (0b01, 0b10, 0b11):
                    f0, f1 = NARROW[code]
                    a2, b2 = a + (b - a) * f0, a + (b - a) * f1
                    if smin >= a2 - eps and smax <= b2 + eps:
                        dist = abs((a2 + b2) / 2 - mid)
                        if best_dist is None or dist < best_dist:
                            best, best_dist = code, dist
                f0, f1 = NARROW[best]
                node_ranges.append((a + (b - a) * f0, a + (b - a) * f1))
                counts[best] += 1
                w.write(best, 2)
            ranges = [node_ranges[k // 2] for k in range(len(next_level))]
        for i, (a, b) in enumerate(ranges):
            w.write(1 if u[i] >= (a + b) / 2 else 0, 1)
        internal = sum(len(l) for l in levels[:-1])
        return w.getvalue(), {"codes": counts, "internal_nodes": internal, "leaves": len(ranges)}

    def decode(self, data, n):
        r = BitReader(data)
        hi, lo = r.read_f64(), r.read_f64()
        levels = tree_levels(n)
        ranges = [(0.0, 1.0)]
        for level, next_level in zip(levels[:-1], levels[1:]):
            node_ranges = []
            for (a, b) in ranges:
                f0, f1 = NARROW[r.read(2)]
                node_ranges.append((a + (b - a) * f0, a + (b - a) * f1))
            ranges = [node_ranges[k // 2] for k in range(len(next_level))]
        lo_f, hi_f = (0.25, 0.75) if self.leaf == "center" else (0.0, 1.0)
        u = np.array([a + (b - a) * (hi_f if r.read(1) else lo_f) for (a, b) in ranges])
        return lo + u * (hi - lo)
