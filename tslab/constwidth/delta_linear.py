# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""cw8-delta1-linear: closed-loop delta1 with uniform steps, one byte per sample.

It works on a 32-bit integer grid over the block's [min, max]
(q = round((x - min) / (max - min) * (2^32 - 1))), and is closed loop:
the encoder takes each residual against the decoder's own running
reconstruction, in integer arithmetic, so quantization error never accumulates.
min/max are written as a 16-byte header here; in a real store they come from the
per-block index. Layout: max f64 | min f64 | q0 u32 | ... | 999 codes.

cw8-delta1-linear   (+ D u32 in the header)
    D = max |q[i] - q[i-1]|. Code k in -127..127 decodes to the integer delta
    sign(k) * round(D * |k| / 127). Uniform steps of D/127: resolution adapts to
    the block's largest jump.
"""

import struct

import numpy as np
from numba import njit

from tslab.constwidth._grid import _dequantize32, _quantize32


@njit(cache=True)
def _linear_level(D, k):
    a = (D * abs(k) + 63) // 127
    return a if k >= 0 else -a


@njit(cache=True)
def _linear_encode(q, codes):
    n = q.shape[0]
    D = np.int64(0)
    for i in range(1, n):
        D = max(D, abs(q[i] - q[i - 1]))
    D = max(D, np.int64(1))
    recon = q[0]
    for i in range(1, n):
        r = q[i] - recon  # integer residual against the decoder's reconstruction
        k = (2 * 127 * r + D) // (2 * D)  # round(r * 127 / D), floor division for both signs
        k = min(max(k, -127), 127)
        codes[i - 1] = np.int8(k)
        recon += _linear_level(D, k)
    return D


@njit(cache=True)
def _linear_decode(codes, q0, D, v):
    v[0] = q0
    for i in range(codes.shape[0]):
        v[i + 1] = v[i] + _linear_level(D, np.int64(codes[i]))


class Delta1Linear8:
    """cw8-delta1-linear: closed-loop delta1 on the 32-bit grid, code k in -127..127 decoding to
    sign(k) * round(D * |k| / 127), D = the block's largest step (see the module docstring)."""

    name = "cw8-delta1-linear"
    label = "8-bit closed-loop delta, uniform steps of D/127"

    def encode(self, x):
        lo, hi, q = _quantize32(x)
        codes = np.empty(len(x) - 1, np.int8)
        D = _linear_encode(q, codes)
        return struct.pack(">ddII", hi, lo, int(q[0]), int(D)) + codes.tobytes(), {"D": int(D)}

    def decode(self, data, n):
        hi, lo, q0, D = struct.unpack(">ddII", data[:24])
        v = np.empty(n, np.int64)
        _linear_decode(np.frombuffer(data, np.int8, n - 1, 24), np.int64(q0), np.int64(D), v)
        out = np.empty(n)
        _dequantize32(v, lo, hi, out)
        return out
