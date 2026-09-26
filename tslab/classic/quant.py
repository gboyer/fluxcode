# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Uniform scalar quantization over the block's [min, max], with and without delta + deflate."""

import struct
import zlib

import numpy as np

from tslab.common.bitio import BitReader, BitWriter


def quantize(x, lo, hi, bits):
    top = (1 << bits) - 1
    if hi == lo:
        return np.zeros(len(x), dtype=np.int64)
    return np.clip(np.rint((np.asarray(x) - lo) / (hi - lo) * top), 0, top).astype(np.int64)


def dequantize(q, lo, hi, bits):
    return lo + np.asarray(q, dtype=np.float64) * (hi - lo) / ((1 << bits) - 1)


class Quant8:
    """quant8: q = round((x - min) / (max - min) * 255), one byte per sample.
    Layout: max f64 | min f64 | codes."""

    name = "quant8"
    label = "8-bit uniform quantization, one byte per sample"

    def encode(self, x):
        lo, hi = x.min(), x.max()
        w = BitWriter()
        w.write_f64(hi)
        w.write_f64(lo)
        for q in quantize(x, lo, hi, 8):
            w.write(q, 8)
        return w.getvalue(), {}

    def decode(self, data, n):
        r = BitReader(data)
        hi, lo = r.read_f64(), r.read_f64()
        return dequantize([r.read(8) for _ in range(n)], lo, hi, 8)


def deflate(b):
    c = zlib.compressobj(9, zlib.DEFLATED, -15)  # raw deflate, no zlib header
    return c.compress(b) + c.flush()


def inflate(b):
    return zlib.decompress(b, -15)


class QuantDeltaDeflate:
    """quant<bits>-delta1-deflate: uniform quantization to `bits`, one byte per sample holding
    the first difference of the codes mod 256, then raw deflate. Lossless on top of the
    quantization. Block parts for tslab.common.unit.SharedBackend: fields max f64, min f64;
    body = the n delta bytes."""

    field_bytes = (8, 8)

    def __init__(self, bits):
        assert bits <= 8
        self.bits = bits
        self.name = f"quant{bits}-delta1-deflate"
        self.label = f"{bits}-bit uniform quantization, first difference mod 256, deflate"

    compress, decompress = staticmethod(deflate), staticmethod(inflate)

    @staticmethod
    def body_bytes(n):
        return n

    def split(self, x):
        lo, hi = x.min(), x.max()
        q = quantize(x, lo, hi, self.bits)
        deltas = (np.diff(q, prepend=0) % 256).astype(np.uint8).tobytes()
        return (struct.pack(">d", hi), struct.pack(">d", lo)), deltas, {}

    def join(self, fields, body, n):
        (hi,), (lo,) = struct.unpack(">d", fields[0]), struct.unpack(">d", fields[1])
        q = np.cumsum(np.frombuffer(body, dtype=np.uint8).astype(np.int64)) % 256
        return dequantize(q[:n], lo, hi, self.bits)
