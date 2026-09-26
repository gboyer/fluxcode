# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Minimal MSB-first bit writer/reader so every codec produces real bytes."""

import struct


def f64_to_bits(x):
    return struct.unpack(">Q", struct.pack(">d", float(x)))[0]


def bits_to_f64(b):
    return struct.unpack(">d", struct.pack(">Q", b))[0]


class BitWriter:
    def __init__(self):
        self.acc = 0
        self.nbits = 0

    def write(self, value, nbits):
        value = int(value)
        assert 0 <= value < (1 << nbits) or (nbits == 0 and value == 0), (value, nbits)
        self.acc = (self.acc << nbits) | value
        self.nbits += nbits

    def write_f64(self, x):
        self.write(f64_to_bits(x), 64)

    def getvalue(self):
        pad = (-self.nbits) % 8
        return (self.acc << pad).to_bytes((self.nbits + pad) // 8, "big")


class BitReader:
    def __init__(self, data):
        self.acc = int.from_bytes(data, "big")
        self.total = len(data) * 8
        self.pos = 0

    def read(self, nbits):
        self.pos += nbits
        assert self.pos <= self.total, "read past end of stream"
        return (self.acc >> (self.total - self.pos)) & ((1 << nbits) - 1)

    def read_f64(self):
        return bits_to_f64(self.read(64))
