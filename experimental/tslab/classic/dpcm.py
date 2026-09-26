# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Companded DPCM: closed-loop delta coding with a companded (S-curve) step table, then deflate."""

import struct

import numpy as np

from tslab.classic.quant import deflate, inflate

# ---------------------------------------------------------------------------
# Companded DPCM: signed codes for the residual against the decoder's own
# reconstruction (closed loop), so quantization error is corrected by later
# samples instead of accumulating. With K = 2^(bits-1) - 1, code k in -K..K
# decodes to D * g(k/K), where g is an odd "expander" that is flat near 0 (fine
# steps for small changes) and steep near ±1 (coarse steps for big jumps). The
# most negative code is unused so the curve stays symmetric and 0 is exact.
#   fields: x[0] f64, D f64; body: the codes, deflated (per unit, see
#   tslab.common.unit.SharedBackend). pack="byte" stores one signed byte per
#   code; pack="nibble" (bits=4) stores two codes per byte.
# ---------------------------------------------------------------------------

def cubic_expander(c):
    return lambda u: c * u ** 3 + (1 - c) * u


def mulaw_expander(mu):
    return lambda u: np.sign(u) * np.expm1(np.abs(u) * np.log1p(mu)) / mu


class CompandedDpcm:
    """dpcm<bits>-<curve>-deflate: closed-loop DPCM, code k decodes to D * g(k / K) (see above),
    D = d_scale * the block's largest step; codes deflated."""

    field_bytes = (8, 8)
    compress, decompress = staticmethod(deflate), staticmethod(inflate)

    def __init__(self, name, label, expander, d_scale=1.0, bits=6, pack="byte"):
        assert pack == "byte" or bits == 4
        self.name, self.label = name, label
        self.d_scale = d_scale  # D = d_scale * max |x[i] - x[i-1]|
        self.bits, self.pack = bits, pack
        self.K = (1 << (bits - 1)) - 1
        self.table = expander(np.arange(-self.K, self.K + 1) / self.K)

    def body_bytes(self, n):
        return n // 2 if self.pack == "nibble" else n - 1

    def split(self, x):
        D = self.d_scale * np.abs(np.diff(x)).max() or 1.0
        levels = D * self.table
        recon = x[0]
        codes = np.empty(len(x) - 1, dtype=np.int64)
        for i, v in enumerate(x[1:]):
            k = int(np.argmin(np.abs(levels - (v - recon))))
            recon += levels[k]
            codes[i] = k - self.K
        if self.pack == "nibble":
            nib = np.append(codes % 16, 0 if len(codes) % 2 else [])
            payload = (nib[0::2] << 4 | nib[1::2]).astype(np.uint8).tobytes()
        else:
            payload = (codes % 256).astype(np.uint8).tobytes()
        return (struct.pack(">d", x[0]), struct.pack(">d", D)), payload, {"codes": codes}

    def join(self, fields, body, n):
        (x0,), (D,) = struct.unpack(">d", fields[0]), struct.unpack(">d", fields[1])
        levels = D * self.table
        raw = np.frombuffer(body, dtype=np.uint8)
        if self.pack == "nibble":
            nib = np.stack([raw >> 4, raw & 15], axis=1).ravel().astype(np.int64)
            codes = np.where(nib >= 8, nib - 16, nib)
        else:
            codes = raw.astype(np.int8).astype(np.int64)
        out = np.empty(n)
        out[0] = recon = x0
        for i, k in enumerate(codes[:n - 1]):
            recon += levels[k + self.K]
            out[i + 1] = recon
        return out
