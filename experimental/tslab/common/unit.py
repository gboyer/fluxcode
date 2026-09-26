# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Unit codecs: every codec is measured on one-minute units (nb blocks of n samples, normally 60 x 1000).

    encode_unit(X: float64[nb, n]) -> (bytes, infos)     infos: one dict per block, for reporting
    decode_unit(data: bytes, nb: int, n: int) -> float64[nb, n]

decode_unit gets only the bytes and the shape, so the byte count is everything needed to decode the
unit. Range-dependent math (quantizer ranges, % of range thresholds) stays per block; only the framing
and the entropy-coding scope are per unit. Two adapters turn a block codec (encode(x) -> (bytes, info),
decode(bytes, n) -> x) into a unit codec:

- Concat: the block encodings back to back. A codec whose encoding size depends on the data sets
  `variable_size = True`, and then the unit starts with the nb block lengths (LEB128); otherwise the
  decoder splits the unit into nb equal parts.
- SharedBackend: for codecs whose last stage is a general-purpose compressor. The block codec exposes
  its parts before that stage, split(x) -> (fields, body, info) and join(fields, body, n) -> x, plus
  `field_bytes` (size of each header field) and body_bytes(n) (size of every body). The unit is one
  compressor call over the headers interleaved by field (field 0 of every block, then field 1, ...)
  followed by the bodies in block order.
"""

import numpy as np

from tslab.common.intcode import read_varint, varint


class Concat:
    """Block encodings back to back, with LEB128 lengths first only if the size is data-dependent."""

    def __init__(self, codec):
        self.codec = codec
        self.name, self.label = codec.name, codec.label
        self.variable = getattr(codec, "variable_size", False)

    def encode_unit(self, X):
        parts, infos = zip(*(self.codec.encode(x) for x in X))
        if self.variable:
            return b"".join(varint(len(p)) for p in parts) + b"".join(parts), list(infos)
        assert len(set(map(len, parts))) == 1, f"{self.name}: block size varies; set variable_size"
        return b"".join(parts), list(infos)

    def decode_unit(self, data, nb, n):
        if self.variable:
            sizes, pos = [], 0
            for _ in range(nb):
                s, pos = read_varint(data, pos)
                sizes.append(s)
        else:
            assert len(data) % nb == 0
            sizes, pos = [len(data) // nb] * nb, 0
        out = np.empty((nb, n))
        for b, s in enumerate(sizes):
            out[b] = self.codec.decode(data[pos:pos + s], n)
            pos += s
        return out


class SharedBackend:
    """One compressor call per unit over the block codec's headers (by field) and bodies (by block)."""

    def __init__(self, codec):
        self.codec = codec
        self.name, self.label = codec.name, codec.label

    def encode_unit(self, X):
        split = [self.codec.split(x) for x in X]
        heads = b"".join(f[i] for i in range(len(self.codec.field_bytes)) for f, _, _ in split)
        bodies = b"".join(body for _, body, _ in split)
        return self.codec.compress(heads + bodies), [info for _, _, info in split]

    def decode_unit(self, data, nb, n):
        raw = self.codec.decompress(data)
        fields, pos = [], 0
        for size in self.codec.field_bytes:
            fields.append([raw[pos + b * size:pos + (b + 1) * size] for b in range(nb)])
            pos += nb * size
        m = self.codec.body_bytes(n)
        assert len(raw) == pos + nb * m
        out = np.empty((nb, n))
        for b in range(nb):
            out[b] = self.codec.join([f[b] for f in fields], raw[pos + b * m:pos + (b + 1) * m], n)
        return out
