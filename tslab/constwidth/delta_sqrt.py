# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""cw8-delta1-sqrt: closed-loop delta1 whose resolution scales like sqrt(|delta|), one byte per sample.

It works on a 32-bit integer grid over the block's [min, max]
(q = round((x - min) / (max - min) * (2^32 - 1))), and is closed loop:
the encoder takes each residual against the decoder's own running
reconstruction, in integer arithmetic, so quantization error never accumulates.
min/max are written as a 16-byte header here; in a real store they come from the
per-block index. Layout: max f64 | min f64 | q0 u32 | ... | 999 codes.

cw8-delta1-sqrt ("semi-quadratic" delta)
    byte = [odd_shift: 1 bit][dampened: 7 bits, signed]; decode (32-bit, no table):
        d32   = dampened << 25            # bit 31 = sign
        s     = 0 - (d32 >> 31)           # 0 or 0xFFFFFFFF
        m'    = d32 ^ s
        c     = CLZ(m')                   # m' == 0 -> delta 0
        m     = m' >> (c - odd_shift)
        delta = m ^ s  (as int32);  next = uint32(prev + delta)
    encode (literal): s = d >> 31; m = d ^ s; lz = CLZ(m);
        m' = m << floor(lz / 2); byte = (lz & 1) << 7 | ((m' ^ s) >> 25)
    Delta resolution scales like sqrt(|delta|): ~7 bits of the range for the
    largest jumps, ~13 bits for small ones; deltas wrap mod 2^32.
    The closed-loop encoder evaluates the literal encoding of the residual and of
    the residual + 1/4, 1/2 and 1 step at that magnitude (step >= 2^19), and keeps
    whichever decodes closest; this matched a brute-force search over all 256
    codes on every test block (3,600 randomized blocks), at ~1/10 the cost.
    CLZ is the LLVM ctlz intrinsic (AArch64 `clz`, NEON `clz.4s`).
"""

import struct

import numpy as np
from numba import njit

from tslab.constwidth._grid import MASK32, _dequantize32, _quantize32, ctlz32


@njit(inline="always")
def sq_decode_delta(e):
    """One byte -> signed delta on the 32-bit grid (your decoder)."""
    odd = np.uint32(e >> 7)
    d32 = np.uint32(np.uint32(e & 0x7F) << np.uint32(25))
    s = np.uint32(np.uint32(0) - np.uint32(d32 >> np.uint32(31)))
    mp = np.uint32(d32 ^ s)
    c = ctlz32(mp)
    m = np.uint32(0) if mp == 0 else np.uint32(mp >> np.uint32(c - odd))
    return np.int64(np.int32(np.uint32(m ^ s)))


@njit(inline="always")
def sq_encode_literal(d):
    """Signed delta in [-2^31, 2^31) -> byte (your encoder, truncating)."""
    u = np.uint32(d & 0xFFFFFFFF)
    s = np.uint32(np.uint32(0) - np.uint32(u >> np.uint32(31)))
    m = np.uint32(u ^ s)
    lz = ctlz32(m)
    mp = np.uint32(m << np.uint32(lz >> np.uint32(1)))
    return np.uint8(np.uint32(np.uint32(lz & np.uint32(1)) << np.uint32(7))
                    | np.uint32(np.uint32(mp ^ s) >> np.uint32(25)))


@njit(cache=True)
def _sq_encode(q, out):
    full = np.int64(1) << 32
    half = np.int64(1) << 31
    recon = q[0]
    for i in range(1, q.shape[0]):
        t = q[i]
        want = ((t - recon + half) & (full - 1)) - half  # integer residual, wrapped
        u = np.uint32(want & 0xFFFFFFFF)
        m = np.uint32(u ^ np.uint32(np.uint32(0) - np.uint32(u >> np.uint32(31))))
        # Resolution at this magnitude, but never finer than the smallest nonzero level (2^19).
        step = np.int64(1) << max(25 - np.int64(ctlz32(m) >> np.uint32(1)), 19)
        best_err, best_c = full, np.uint8(0)
        # The literal encoding truncates in one direction, and the nearest level can
        # sit in a finer class, so also try the residual pushed up by 1/4, 1/2 and 1
        # step. This matched a brute-force search over all 256 codes on every test block.
        for cand in (want, want + (step >> 2), want + (step >> 1), want + step):
            cand = ((cand + half) & (full - 1)) - half
            e = sq_encode_literal(cand)
            err = abs(t - ((recon + sq_decode_delta(e)) & (full - 1)))
            if err < best_err:
                best_err, best_c = err, e
        out[i - 1] = best_c
        recon = (recon + sq_decode_delta(best_c)) & (full - 1)


@njit(cache=True)
def _sq_decode(codes, q0, delta, v):
    for i in range(codes.shape[0]):  # bytes -> deltas: independent per sample (vectorizable)
        delta[i] = sq_decode_delta(codes[i])
    acc = np.uint32(q0)
    v[0] = acc
    for i in range(codes.shape[0]):  # running sum mod 2^32: one add per sample
        acc = np.uint32(acc + np.uint32(delta[i] & 0xFFFFFFFF))
        v[i + 1] = acc


class Delta1Sqrt8:
    """cw8-delta1-sqrt: closed-loop delta1 on the 32-bit grid, one byte per delta, decoded with CLZ
    so the resolution scales like sqrt(|delta|) (see the module docstring)."""

    name = "cw8-delta1-sqrt"
    label = "8-bit closed-loop delta, resolution ∝ √|delta| (CLZ decode)"

    def encode(self, x):
        lo, hi, q = _quantize32(x)
        codes = np.empty(len(x) - 1, np.uint8)
        _sq_encode(q, codes)
        return struct.pack(">ddI", hi, lo, int(q[0])) + codes.tobytes(), {"codes": codes}

    def decode(self, data, n):
        hi, lo, q0 = struct.unpack(">ddI", data[:20])
        codes = np.frombuffer(data, np.uint8, n - 1, 20)
        v = np.empty(n, np.int64)
        _sq_decode(codes, np.int64(q0), np.empty(n - 1, np.int64), v)
        out = np.empty(n)
        _dequantize32(v, lo, hi, out)
        return out


# Reference (pure Python) versions of the byte transforms, for tests and docs.
def _clz32(v):
    return 32 - int(v).bit_length()


def sq_decode_byte(e):
    odd, v7 = e >> 7, e & 0x7F
    s = MASK32 if v7 & 0x40 else 0
    mp = ((v7 << 25) ^ s) & MASK32
    if mp == 0:
        return 0
    m = mp >> (_clz32(mp) - odd)
    return m if s == 0 else -m - 1


def sq_encode_byte(d):
    s = MASK32 if d < 0 else 0
    m = (d & MASK32) ^ s
    lz = _clz32(m)
    mp = (m << (lz // 2)) & MASK32
    return (lz % 2) << 7 | (((mp ^ s) & MASK32) >> 25)


SQ_TABLE = np.array([sq_decode_byte(e) for e in range(256)], np.int64)  # all decodable deltas
