# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""cw<bits>-delta1-bfp-e<e>: closed-loop delta1 codes with a block-floating-point step exponent per frame."""

import struct

import numpy as np
from numba import njit

from tslab.constwidth._grid import _pack_codes, _unpack_codes

# ---------------------------------------------------------------------------
# cw8-delta1-bfp-e2: 8-bit deltas with a 2-bit step exponent per frame (block floating point)
#
# Grid: [min, max] -> [2^23, 2^23 + 255 * 2^24] on uint32 (half a step of headroom on
# each side, so a reconstruction within half a step of min/max never wraps). Code k (int8, all 256 values)
# decodes to the delta k << (24 - e), added mod 2^32. At e = 0 the step is 2^24
# and wraparound reaches any value, so it is exactly the 8-bit (quant8) grid: max
# error <= range / 510, guaranteed. Frames of small deltas use e = 1..3 (steps
# 2^23..2^21) for 1-3 extra bits (with 4-bit exponents, e = 0..15: up to 15 extra
# bits). e per frame = finest step whose int8 range
# covers the frame's largest delta plus the error carried into the frame.
# Rollover only happens at e = 0, i.e. across the block's full quantization
# range (256 steps = 2^32); for e > 0 codes are clamped and never wrap.
# Layout: max f64 | min f64 | q0 u32 | exponents (2 or 4 bits each, packed LSB first) | codes.
# ---------------------------------------------------------------------------

FRAME = 16
GRID_TOP = 255 << 24
GRID_OFF = 1 << 23


@njit(cache=True)
def _bfp_encode(q, frame, emax, cbits, codes, exps):
    """cbits-wide signed codes; base (e = 0) step 2^(32 - cbits), i.e. 2^cbits grid levels."""
    n = q.shape[0]
    full = np.int64(1) << 32
    half = np.int64(1) << 31
    base = 32 - cbits
    hc = np.int64(1) << (cbits - 1)  # codes -hc .. hc - 1
    levels = (np.int64(1) << cbits) - 1
    recon = q[0]
    for f in range(exps.shape[0]):
        s0 = f * frame
        s1 = min(s0 + frame, n - 1)
        mx = np.int64(0)  # largest delta in the frame (independent per sample)
        for i in range(s0, s1):
            mx = max(mx, abs(q[i + 1] - q[i]))
        carried = abs(q[s0] - recon)  # error already in the reconstruction
        e = 0
        for c in range(emax, 0, -1):  # finest step whose +/-(hc - 0.5) steps cover the frame
            step = np.int64(1) << (base - c)
            if 2 * (mx + carried) + step <= levels * step:
                e = c
                break
        exps[f] = e
        sh = base - e
        for i in range(s0, s1):
            r = ((q[i + 1] - recon + half) & (full - 1)) - half  # integer residual, wrapped
            k = (r + (np.int64(1) << (sh - 1))) >> sh  # round(r / 2^sh)
            k = min(max(k, -hc), hc - 1) if e > 0 else ((k + hc) & (2 * hc - 1)) - hc  # e = 0: wraps mod 2^32
            codes[i] = np.int8(k)
            nxt = recon + (k << sh)
            if e > 0 and (nxt < 0 or nxt >= full):
                raise ValueError("bfp: reconstruction wrapped at e > 0")
            recon = nxt & (full - 1)


@njit(cache=True)
def _bfp_decode(codes, packed, q0, frame, ebits, cbits, lo, hi, d, out):
    """Per frame: codes -> deltas with one fixed shift (vectorizable); then one
    running uint32 sum, dequantized in the same pass."""
    n1 = codes.shape[0]
    base = 32 - cbits
    per = 8 // ebits
    for f in range((n1 + frame - 1) // frame):
        sh = base - ((packed[f // per] >> ((f % per) * ebits)) & ((1 << ebits) - 1))
        for i in range(f * frame, min(f * frame + frame, n1)):
            d[i] = np.uint32(np.int32(codes[i]) << sh)
    off = np.int64(1) << (base - 1)
    step = (hi - lo) / float(((np.int64(1) << cbits) - 1) << base)
    acc = np.uint32(q0)
    out[0] = lo + (np.int64(acc) - off) * step  # remove the half-step offset
    for i in range(n1):
        acc = np.uint32(acc + d[i])
        out[i + 1] = lo + (np.int64(acc) - off) * step


class Delta1Bfp:
    """cw<code_bits>-delta1-bfp-e<exp_bits>: closed-loop delta1, code_bits-wide codes (8, also 4
    and 6) with an exp_bits-wide step exponent per frame: step = 2^-e of the code_bits-bit step
    (block floating point; see the comment above)."""

    def __init__(self, frame=FRAME, exp_bits=2, code_bits=8):
        assert exp_bits in (2, 4) and 2 <= code_bits <= 8
        self.frame, self.exp_bits, self.code_bits = frame, exp_bits, code_bits
        self.name = f"cw{code_bits}-delta1-bfp-e{exp_bits}" + ("" if frame == FRAME else f"-f{frame}")
        self.label = f"{code_bits}-bit closed-loop delta, {exp_bits}-bit step exponent per {frame} samples"

    def _nframes(self, n):
        return -(-(n - 1) // self.frame)

    def encode(self, x):
        W = self.code_bits
        lo, hi = float(x.min()), float(x.max())
        top, off = ((1 << W) - 1) << (32 - W), 1 << (31 - W)
        scale = top / (hi - lo) if hi > lo else 0.0
        q = np.rint((x - lo) * scale).astype(np.int64) + off
        codes = np.empty(len(x) - 1, np.int8)
        exps = np.empty(self._nframes(len(x)), np.int64)
        _bfp_encode(q, self.frame, (1 << self.exp_bits) - 1, W, codes, exps)
        per = 8 // self.exp_bits
        e = np.zeros(-(-len(exps) // per) * per, np.uint8)
        e[:len(exps)] = exps
        packed = np.zeros(len(e) // per, np.uint8)
        for j in range(per):
            packed |= e[j::per] << (j * self.exp_bits)
        if W < 8:
            body = np.empty(-(-len(codes) * W // 8), np.uint8)
            _pack_codes(codes, W, body)
        else:
            body = codes
        return struct.pack(">ddI", hi, lo, int(q[0])) + packed.tobytes() + body.tobytes(), {"exps": exps}

    def decode(self, data, n):
        W = self.code_bits
        hi, lo, q0 = struct.unpack(">ddI", data[:20])
        nb = -(-self._nframes(n) // (8 // self.exp_bits))
        if W < 8:
            codes = np.empty(n - 1, np.int8)
            _unpack_codes(np.frombuffer(data, np.uint8, offset=20 + nb), W, codes)
        else:
            codes = np.frombuffer(data, np.int8, n - 1, 20 + nb)
        out = np.empty(n)
        _bfp_decode(codes, np.frombuffer(data, np.uint8, nb, 20), np.int64(q0), self.frame, self.exp_bits, W,
                       lo, hi, np.empty(n - 1, np.uint32), out)
        return out
