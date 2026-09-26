# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Integer coding helpers shared by the entropy-coded pipelines.

zigzag / unzigzag map signed residuals to unsigned codes; varint / read_varint are LEB128;
to_residual / from_residual take and undo the order-th difference, keeping the initial value of
each difference level; pack_init / unpack_init store those initial values as zigzag varints.
pick_order is the compiled order picker shared by delta0123-zstd and the fluxcode prototype.
"""

import numpy as np
from numba import njit

ORDERS = (0, 1, 2, 3)  # candidate predictor orders (fixed polynomial predictors)


def zigzag(r):
    return (r << 1) ^ (r >> 63)


def unzigzag(u):
    return (u >> 1) ^ -(u & 1)


def varint(v):
    out = bytearray()
    while True:
        b, v = v & 0x7F, v >> 7
        out.append(b | (0x80 if v else 0))
        if not v:
            return bytes(out)


def read_varint(buf, pos):
    v = shift = 0
    while True:
        b = buf[pos]
        pos += 1
        v |= (b & 0x7F) << shift
        shift += 7
        if not b & 0x80:
            return v, pos


def to_residual(q, order):
    """Initial values of each difference level, plus the order-th difference."""
    init = [int(np.diff(q, j)[0]) for j in range(order)]
    return init, np.diff(q, order)


def from_residual(init, r):
    cur = r
    for j in reversed(range(len(init))):
        cur = np.concatenate([[init[j]], init[j] + np.cumsum(cur)])
    return cur


def pack_init(init):
    return b"".join(varint(int(zigzag(np.int64(v)))) for v in init)


def unpack_init(buf, pos, order):
    init = []
    for _ in range(order):
        u, pos = read_varint(buf, pos)
        init.append(int(unzigzag(np.int64(u))))
    return init, pos


@njit(cache=True)
def pick_order(q, mask=0b1111):
    """(order, residual variance) minimizing the variance of diff(q, order) over the orders whose bit
    is set in mask (default: 0..3).

    Spread, not mean |r|: zstd is indifferent to a constant offset, and order 0's residuals (the raw
    codes) carry one, which hid order 0 from mean |r|. One pass computes all four sums of squares
    from direct formulas; the plain sums telescope. Exact integer sums, so no rounding ties.
    """
    n = q.shape[0]
    s0 = ss0 = ss1 = ss2 = ss3 = np.int64(0)
    for i in range(3):
        v = np.int64(q[i])
        s0 += v
        ss0 += v * v
    for i in range(1, 3):
        d = np.int64(q[i]) - q[i - 1]
        ss1 += d * d
    d = np.int64(q[2]) - 2 * np.int64(q[1]) + q[0]
    ss2 += d * d
    for i in range(3, n):
        a, b, c, e = np.int64(q[i]), np.int64(q[i - 1]), np.int64(q[i - 2]), np.int64(q[i - 3])
        s0 += a
        ss0 += a * a
        d1 = a - b
        ss1 += d1 * d1
        d2 = a - 2 * b + c
        ss2 += d2 * d2
        d3 = a - 3 * b + 3 * c - e
        ss3 += d3 * d3
    s1 = np.int64(q[n - 1]) - q[0]
    s2 = (np.int64(q[n - 1]) - q[n - 2]) - (np.int64(q[1]) - q[0])
    s3 = ((np.int64(q[n - 1]) - 2 * np.int64(q[n - 2]) + q[n - 3])
          - (np.int64(q[2]) - 2 * np.int64(q[1]) + q[0]))
    best, best_v = -1, 1e300
    for k, s, ss in ((0, s0, ss0), (1, s1, ss1), (2, s2, ss2), (3, s3, ss3)):
        if mask >> k & 1:
            m = n - k
            v = (m * float(ss) - float(s) * float(s)) / (m * m)
            if best < 0 or v < best_v:
                best, best_v = k, v
    return best, best_v
