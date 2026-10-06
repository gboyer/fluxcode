# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Single-threaded numba (native) versions of the fastest classic codecs, on whole block groups.

Each encoder is byte-compatible with its Python reference block group codec in tslab.classic (wrapped by
tslab.common.group), so the reference decoders can check it. Python work is limited to header packing
and one zstd/zlib call per block group; everything per-sample runs in compiled loops, no threads.
(delta0123-zstd is compiled itself: tslab.entropy.delta_zstd.) _delta_zstd_front is the per-block
delta0123-zstd front end used by bench/chunking.py.
"""

import zlib

import numpy as np
import zstandard
from numba import njit

WCODE = {1: 0, 2: 1, 4: 2}
DEFAULT_LEVEL = {"zstd": 3, "deflate": 9}


def _deflate(b, level=9):
    c = zlib.compressobj(level, zlib.DEFLATED, -15)
    return c.compress(b) + c.flush()


def _be_f64(v):
    return np.asarray(v, ">f8").tobytes()


def _varint_zz(v):
    v = (v << 1) ^ (v >> 63)
    out = bytearray()
    while True:
        byte, v = v & 0x7F, v >> 7
        out.append(byte | (0x80 if v else 0))
        if not v:
            return bytes(out)


# ---------------------------------------------------------------------------
# quant8: plain 8-bit quantization
# ---------------------------------------------------------------------------

@njit(cache=True)
def _quantize_u8(X, lo, hi, out):
    for b in range(X.shape[0]):
        rng = hi[b] - lo[b]
        scale = 255.0 / rng if rng > 0 else 0.0
        for i in range(X.shape[1]):
            out[b, i] = np.uint8(np.floor((X[b, i] - lo[b]) * scale + 0.5))


@njit(cache=True)
def _dequantize_u8(q, lo, hi, out):
    step = (hi - lo) / 255.0
    for i in range(q.shape[0]):
        out[i] = lo + q[i] * step


def quant8_encode(X):
    """Block group in Concat(Quant8) format: per block max f64 | min f64 | codes."""
    lo, hi = X.min(1), X.max(1)
    rows = np.empty((len(X), 16 + X.shape[1]), np.uint8)
    rows[:, :8] = np.frombuffer(_be_f64(hi), np.uint8).reshape(-1, 8)
    rows[:, 8:16] = np.frombuffer(_be_f64(lo), np.uint8).reshape(-1, 8)
    _quantize_u8(X, lo, hi, rows[:, 16:])
    return rows.tobytes()


def quant8_decode(data, nb, n):
    rows = np.frombuffer(data, np.uint8).reshape(nb, 16 + n)
    hi = rows[:, :8].copy().view(">f8").ravel()
    lo = rows[:, 8:16].copy().view(">f8").ravel()
    out = np.empty((nb, n))
    for b in range(nb):
        _dequantize_u8(rows[b, 16:], lo[b], hi[b], out[b])
    return out


# ---------------------------------------------------------------------------
# quant8-delta1-deflate: 8-bit quantization, delta mod 256, raw deflate level 9
# ---------------------------------------------------------------------------

@njit(cache=True)
def _delta_u8(q, out):
    for b in range(q.shape[0]):
        prev = np.uint8(0)
        for i in range(q.shape[1]):
            out[b, i] = q[b, i] - prev
            prev = q[b, i]


@njit(cache=True)
def _undelta_dequantize(d, lo, hi, out):
    step = (hi - lo) / 255.0
    acc = np.uint8(0)
    for i in range(out.shape[0]):
        acc = np.uint8(acc + d[i])
        out[i] = lo + acc * step


class Quant8Delta1:
    """quant8-delta1-<compressor>[-L<level>]: the block group format of SharedBackend(QuantDeltaDeflate(8)),
    compress(max f64 x nb | min f64 x nb | delta bytes), with a choice of compressor and level. The
    level is in the name only when it isn't the default (zstd 3, deflate 9)."""

    def __init__(self, compressor="deflate", level=None):
        level = DEFAULT_LEVEL[compressor] if level is None else level
        self.name = f"quant8-delta1-{compressor}" + ("" if level == DEFAULT_LEVEL[compressor] else f"-L{level}")
        if compressor == "zstd":
            self._c = zstandard.ZstdCompressor(level=level, write_checksum=False).compress
            self._d = zstandard.ZstdDecompressor().decompress
        elif compressor == "deflate":
            self._c, self._d = (lambda b: _deflate(b, level)), (lambda b: zlib.decompress(b, -15))
        else:
            raise ValueError(compressor)

    def encode_group(self, X):
        lo, hi = X.min(1), X.max(1)
        q = np.empty(X.shape, np.uint8)
        _quantize_u8(X, lo, hi, q)
        d = np.empty_like(q)
        _delta_u8(q, d)
        return self._c(_be_f64(hi) + _be_f64(lo) + d.tobytes()), [{}] * len(X)

    def decode_group(self, data, nb, n):
        raw = self._d(data)
        hi = np.frombuffer(raw, ">f8", nb)
        lo = np.frombuffer(raw, ">f8", nb, 8 * nb)
        d = np.frombuffer(raw, np.uint8, nb * n, 16 * nb).reshape(nb, n)
        out = np.empty((nb, n))
        for b in range(nb):
            _undelta_dequantize(d[b], lo[b], hi[b], out[b])
        return out


# ---------------------------------------------------------------------------
# Linear DPCM: closed loop, uniform levels D*k/K. With linear steps the nearest
# level is round(residual / step), so no search over levels is needed.
# ---------------------------------------------------------------------------

@njit(cache=True)
def _dpcm_encode(X, K, codes, D_out):
    for b in range(X.shape[0]):
        D = 0.0
        for i in range(1, X.shape[1]):
            D = max(D, abs(X[b, i] - X[b, i - 1]))
        if D == 0.0:
            D = 1.0
        D_out[b] = D
        step = D / K
        recon = X[b, 0]
        for i in range(1, X.shape[1]):
            k = np.floor((X[b, i] - recon) / step + 0.5)
            k = min(max(k, -K), K)
            codes[b, i - 1] = np.int8(k)
            recon += D * (k / K)  # same expression as the reference's level table


@njit(cache=True)
def _dpcm_decode(codes, x0, D, K, out):
    out[0] = recon = x0
    for i in range(codes.shape[0]):
        recon += D * (codes[i] / K)
        out[i + 1] = recon


def dpcm_encode(X, bits):
    """Block group in SharedBackend(CompandedDpcm, linear, byte packing) format:
    deflate(x0 f64 x nb | D f64 x nb | codes)."""
    K = (1 << (bits - 1)) - 1
    codes = np.empty((X.shape[0], X.shape[1] - 1), np.int8)
    D = np.empty(X.shape[0])
    _dpcm_encode(X, K, codes, D)
    return _deflate(_be_f64(X[:, 0]) + _be_f64(D) + codes.tobytes())


def dpcm_decode(data, bits, nb, n):
    raw = zlib.decompress(data, -15)
    x0 = np.frombuffer(raw, ">f8", nb)
    D = np.frombuffer(raw, ">f8", nb, 8 * nb)
    codes = np.frombuffer(raw, np.int8, nb * (n - 1), 16 * nb).reshape(nb, n - 1)
    out = np.empty((nb, n))
    for b in range(nb):
        _dpcm_decode(codes[b], x0[b], D[b], (1 << (bits - 1)) - 1, out[b])
    return out


# ---------------------------------------------------------------------------
# delta0123-zstd per-block front end (bench/chunking.py): fused quantize + order
# pick + residual + zigzag + byte split
# ---------------------------------------------------------------------------

@njit(cache=True)
def _delta_zstd_front(X, lo, hi, bits, allow, planes, meta):
    """meta[b] = order, width, payload length, init0, init1, init2."""
    nb, n = X.shape
    top = (1 << bits) - 1
    q = np.empty(n, np.int64)
    s = np.zeros(4)
    ss = np.zeros(4)
    for b in range(nb):
        rng = hi[b] - lo[b]
        scale = top / rng if rng > 0 else 0.0
        s[:] = 0.0
        ss[:] = 0.0
        d1p = 0
        d2p = 0
        for i in range(n):
            v = np.int64(np.floor((X[b, i] - lo[b]) * scale + 0.5))
            q[i] = v
            s[0] += v
            ss[0] += v * v
            if i >= 1:
                d1 = v - q[i - 1]
                s[1] += d1
                ss[1] += d1 * d1
                if i >= 2:
                    d2 = d1 - d1p
                    s[2] += d2
                    ss[2] += d2 * d2
                    if i >= 3:
                        d3 = d2 - d2p
                        s[3] += d3
                        ss[3] += d3 * d3
                    d2p = d2
                d1p = d1
        best, best_v = 0, 1e300
        for k in range(4):
            if allow[k]:
                m = n - k
                var = (m * ss[k] - s[k] * s[k]) / (m * m)
                if var < best_v:
                    best, best_v = k, var
        m = n
        for k in range(best):
            meta[b, 3 + k] = q[0]
            for i in range(m - 1):
                q[i] = q[i + 1] - q[i]
            m -= 1
        umax = 0
        for i in range(m):
            r = q[i]
            u = (r << 1) ^ (r >> 63)
            q[i] = u
            umax = max(umax, u)
        w = 1 if umax < 256 else (2 if umax < 65536 else 4)
        for j in range(w):
            for i in range(m):
                planes[b, j * m + i] = (q[i] >> (8 * j)) & 255
        meta[b, 0] = best
        meta[b, 1] = w
        meta[b, 2] = w * m
