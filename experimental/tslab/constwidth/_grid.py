# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Shared pieces of the constant-width codecs: the 32-bit integer grid over [min, max], a
count-leading-zeros intrinsic, and LSB-first packing of narrow signed codes."""

import numpy as np
from llvmlite import ir
from numba import njit, types
from numba.core import cgutils
from numba.extending import intrinsic

MASK32 = (1 << 32) - 1


@intrinsic
def ctlz32(typingctx, x):
    """Count leading zeros of a uint32 (defined for 0 -> 32)."""
    sig = types.uint32(types.uint32)

    def codegen(context, builder, signature, args):
        fnty = ir.FunctionType(ir.IntType(32), [ir.IntType(32), ir.IntType(1)])
        fn = cgutils.get_or_insert_function(builder.module, fnty, "llvm.ctlz.i32")
        return builder.call(fn, [args[0], ir.Constant(ir.IntType(1), 0)])

    return sig, codegen


def _quantize32(x):
    lo, hi = float(x.min()), float(x.max())
    scale = MASK32 / (hi - lo) if hi > lo else 0.0
    return lo, hi, np.rint((x - lo) * scale).astype(np.int64)


@njit(cache=True)
def _dequantize32(v, lo, hi, out):
    step = (hi - lo) / 4294967295.0
    for i in range(v.shape[0]):
        out[i] = lo + v[i] * step


@njit(cache=True)
def _pack_codes(codes, cbits, out):
    """Signed cbits-wide codes -> LSB-first bitstream."""
    acc, nacc, j = np.uint64(0), 0, 0
    mask = (1 << cbits) - 1
    for i in range(codes.shape[0]):
        acc |= np.uint64(np.int64(codes[i]) & mask) << np.uint64(nacc)
        nacc += cbits
        while nacc >= 8:
            out[j] = np.uint8(acc & np.uint64(255))
            acc >>= np.uint64(8)
            nacc -= 8
            j += 1
    if nacc > 0:
        out[j] = np.uint8(acc & np.uint64(255))


@njit(cache=True)
def _unpack_codes(buf, cbits, codes):
    acc, nacc, j = np.uint64(0), 0, 0
    mask = np.uint64((1 << cbits) - 1)
    hc = 1 << (cbits - 1)
    for i in range(codes.shape[0]):
        while nacc < cbits:
            acc |= np.uint64(buf[j]) << np.uint64(nacc)
            nacc += 8
            j += 1
        u = np.int64(acc & mask)
        acc >>= np.uint64(cbits)
        nacc -= cbits
        codes[i] = np.int8(u - 2 * hc if u >= hc else u)
