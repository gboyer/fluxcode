# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""delta0123-zstd: one-shot quantize -> predict -> zstd encoder, no search, one compressor call per block group.

Per block (all compiled, no Python per block):
    step   = (max - min) / (2^B - 1)                       B-bit codes, min and max exact
    q      = round((x - min) / step)                        the only lossy step
    order  = argmin var(diff(q, k)) over the allowed k in 0..3   (exact integer sums, one fused pass)
    cap    : estimated bits = log2(std) + 2.05 (Gaussian entropy); if above CAP_BPS, B drops by the
             excess (rounded up) and the block is requantized
    planes = byte planes of zigzag(diff(q, order)), low byte first

Block group (nb blocks), headers interleaved by field, then bodies in block order, one compressor call:
    group   := compress( flags[nb] | min f64[nb] | max f64[nb] | init[0] .. init[nb-1] | planes[0] .. planes[nb-1] )
    flags  := u8 = (B-1) << 4 | order << 2 | width code (0,1,2 -> 1,2,4 bytes)
    init   := order x zigzag LEB128 varint (starting values diff(q, j)[0], j < order)
    planes := width byte planes of zigzag(diff(q, order)), n - order bytes each

Loops avoid cross-iteration state; LLVM vectorizes quantize and residual/zigzag (NEON), the order
picker and plane writes compile to scalar code.
"""

import zlib

import numpy as np
import zstandard
from numba import njit

from tslab.common.intcode import ORDERS, pick_order

CAP_BPS = 8.0
WIDTHS = np.array([1, 2, 4], np.int64)
MAX_HEADER = 1 + 16 + 3 * 10  # flags + min/max + three varints
DEFAULT_LEVEL = {"zstd": 3, "deflate": 9}


@njit(cache=True)
def _quantize(x, lo, hi, bits, q):
    top = (1 << bits) - 1
    rng = hi - lo
    scale = top / rng if rng > 0 else 0.0
    for i in range(x.shape[0]):
        q[i] = np.int32(np.floor((x[i] - lo) * scale + 0.5))


@njit(cache=True)
def _residual_zigzag(q, order, u):
    """u[i] = zigzag(diff(q, order)[i]); returns max(u)."""
    n = q.shape[0]
    m = n - order
    if order == 0:
        for i in range(m):
            r = q[i]
            u[i] = np.uint32((r << 1) ^ (r >> 31))
    elif order == 1:
        for i in range(m):
            r = q[i + 1] - q[i]
            u[i] = np.uint32((r << 1) ^ (r >> 31))
    elif order == 2:
        for i in range(m):
            r = q[i + 2] - 2 * q[i + 1] + q[i]
            u[i] = np.uint32((r << 1) ^ (r >> 31))
    else:
        for i in range(m):
            r = q[i + 3] - 3 * q[i + 2] + 3 * q[i + 1] - q[i]
            u[i] = np.uint32((r << 1) ^ (r >> 31))
    umax = np.uint32(0)
    for i in range(m):
        umax = max(umax, u[i])
    return umax


@njit(cache=True)
def _write_planes(u, m, w, out, pos):
    if w == 1:  # the common case
        for i in range(m):
            out[pos + i] = np.uint8(u[i])
        return pos + m
    for j in range(w):
        base = pos + j * m
        shift = 8 * j
        for i in range(m):
            out[base + i] = np.uint8((u[i] >> shift) & 255)
    return pos + w * m


@njit(cache=True)
def _put_varint_zz(out, pos, v):
    z = np.uint64((v << 1) ^ (v >> 63))
    while True:
        b = np.uint8(z & 0x7F)
        z >>= 7
        if z:
            out[pos] = b | 0x80
            pos += 1
        else:
            out[pos] = b
            return pos + 1


@njit(cache=True)
def _encode_group(X, bits, mask, cap_bps, out, planes, meta):
    """Fills out[] with the uncompressed block group; returns its length. meta[b] = (B, order)."""
    nb, n = X.shape
    q = np.empty(n, np.int32)
    u = np.empty(n, np.uint32)
    lohi = out[nb:17 * nb].view(np.float64)
    hpos = 17 * nb
    ppos = 0
    for b in range(nb):
        lo, hi = X[b].min(), X[b].max()
        lohi[b], lohi[nb + b] = lo, hi
        B = bits
        _quantize(X[b], lo, hi, B, q)
        order, var = pick_order(q, mask)
        if cap_bps > 0 and var > 0:
            est = 0.5 * np.log2(var) + 2.05
            if est > cap_bps:
                B = max(1, B - int(np.ceil(est - cap_bps)))
                _quantize(X[b], lo, hi, B, q)
                order, var = pick_order(q, mask)
        umax = _residual_zigzag(q, order, u)
        wcode = 0 if umax < 256 else (1 if umax < 65536 else 2)
        out[b] = np.uint8((B - 1) << 4 | order << 2 | wcode)
        meta[b, 0], meta[b, 1] = B, order
        if order >= 1:
            hpos = _put_varint_zz(out, hpos, np.int64(q[0]))
        if order >= 2:
            hpos = _put_varint_zz(out, hpos, np.int64(q[1]) - q[0])
        if order >= 3:
            hpos = _put_varint_zz(out, hpos, np.int64(q[2]) - 2 * np.int64(q[1]) + q[0])
        ppos = _write_planes(u, n - order, WIDTHS[wcode], planes, ppos)
    out[hpos:hpos + ppos] = planes[:ppos]
    return hpos + ppos


@njit(cache=True)
def _get_varint_zz(raw, pos):
    z = np.uint64(0)
    shift = 0
    while True:
        b = raw[pos]
        pos += 1
        z |= np.uint64(b & 0x7F) << np.uint64(shift)
        shift += 7
        if not b & 0x80:
            break
    return np.int64(z >> np.uint64(1)) ^ -np.int64(z & np.uint64(1)), pos


@njit(cache=True)
def _decode_group(raw, X):
    nb, n = X.shape
    lohi = raw[nb:17 * nb].copy().view(np.float64)
    init = np.zeros((nb, 3), np.int64)
    pos = 17 * nb
    for b in range(nb):
        for j in range((raw[b] >> 2) & 3):
            init[b, j], pos = _get_varint_zz(raw, pos)
    q = np.empty(n, np.int64)
    for b in range(nb):
        bits, o, w = (raw[b] >> 4) + 1, (raw[b] >> 2) & 3, WIDTHS[raw[b] & 3]
        m = n - o
        for i in range(m):
            z = np.int64(raw[pos + i])
            for j in range(1, w):
                z |= np.int64(raw[pos + j * m + i]) << (8 * j)
            q[i] = (z >> 1) ^ -(z & 1)
        pos += w * m
        for k in range(o - 1, -1, -1):  # integrate: prepend start value, prefix sum
            for i in range(m, 0, -1):
                q[i] = q[i - 1]
            q[0] = init[b, k]
            m += 1
            for i in range(1, m):
                q[i] += q[i - 1]
        lo, hi = lohi[b], lohi[nb + b]
        top = (1 << bits) - 1
        step = (hi - lo) / top
        for i in range(n):
            # Snap the top code to max: min + top * step can be off by an ulp.
            X[b, i] = hi if q[i] == top else lo + q[i] * step


class DeltaZstd:
    """delta<orders>-<compressor>-<bits>[-L<level>]: B-bit quantizer on each block's range -> the delta
    order (from `orders`) with the smallest residual variance -> zigzag byte planes -> one compressor
    call per block group (see the module docstring). The level is in the name only when it isn't the
    compressor's default (zstd 3, deflate 9)."""

    def __init__(self, bits, level=None, orders=ORDERS, compressor="zstd", cap_bps=CAP_BPS):
        assert 1 <= bits <= 16
        level = DEFAULT_LEVEL[compressor] if level is None else level
        self.bits, self.orders, self.cap_bps = bits, tuple(orders), cap_bps
        self.mask = sum(1 << k for k in self.orders)
        tag = "".join(map(str, self.orders))
        self.name = f"delta{tag}-{compressor}-{bits}" + ("" if level == DEFAULT_LEVEL[compressor] else f"-L{level}")
        self.label = f"{bits}-bit quantization, delta order {'/'.join(map(str, self.orders))} by variance, {compressor}"
        if compressor == "zstd":
            self._c = zstandard.ZstdCompressor(level=level, write_checksum=False).compress  # content size kept
            self._d = zstandard.ZstdDecompressor().decompress
        elif compressor == "deflate":
            def c(b, level=level):
                o = zlib.compressobj(level, zlib.DEFLATED, -15)
                return o.compress(b) + o.flush()
            self._c, self._d = c, lambda b: zlib.decompress(b, -15)
        else:
            raise ValueError(compressor)

    def raw_group(self, X):
        """The uncompressed block group and meta[b] = (B, order)."""
        nb, n = X.shape
        out = np.empty(nb * (MAX_HEADER + 4 * n), np.uint8)
        planes = np.empty(nb * 4 * n, np.uint8)
        meta = np.empty((nb, 2), np.int64)
        size = _encode_group(np.ascontiguousarray(X, np.float64), self.bits, self.mask, self.cap_bps, out, planes, meta)
        return out[:size], meta

    def encode_group(self, X):
        raw, meta = self.raw_group(X)
        return self._c(raw.tobytes()), [{"bits": int(B), "order": int(o)} for B, o in meta]

    def decode_group(self, data, nb, n):
        X = np.empty((nb, n))
        _decode_group(np.frombuffer(self._d(data), np.uint8), X)
        return X


DELTA_ZSTD_CODECS = [
    DeltaZstd(8),
    DeltaZstd(10),
    DeltaZstd(12),
    DeltaZstd(8, orders=(1, 2)),  # performance option: only the two most common orders
]
