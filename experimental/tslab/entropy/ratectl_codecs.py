# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""The rate-controlled quantize -> predict -> entropy-code pipelines from
ratectl.py, wrapped as block group codecs for the codec matrix.

Block group format: step f64 (as a fraction of each block's max - min) | min f64 x nb |
max f64 x nb | the coder's block group (ratectl.*.encode_group). One step for the whole
block group, bisected so the block group lands at or just under target_bps. Max error is
step/2 (in units of each block's full scale).
"""

import struct

import numpy as np

from tslab.entropy.ratectl import FLAC, ZSTD


class RateCtlCodec:
    """ratectl-<coder>-<target>b: one quantizer step per block group, bisected so the block group lands at or just
    under target_bps, then the ratectl coder (delta0123-zstd, or libFLAC with its own LPC)."""

    def __init__(self, coder, target_bps):
        self.coder, self.target_bps = coder, target_bps
        self.name = f"ratectl-{coder.name}-{target_bps:g}b"
        self.label = f"Rate control to {target_bps:g} bits/sample, then {'FLAC' if coder is FLAC else coder.name}"

    def encode_step(self, X, step):
        lo, hi = X.min(1), X.max(1)
        U = (X - lo[:, None]) / np.where(hi > lo, hi - lo, 1.0)[:, None]
        head = struct.pack(f">d{len(X)}d{len(X)}d", step, *lo, *hi)
        return head + self.coder.encode_group([np.rint(u / step).astype(np.int64) for u in U])

    def encode_group(self, X):
        budget = self.target_bps * X.size / 8
        lo, hi = -30.0, 0.0  # log2(step) as a fraction of full scale
        for _ in range(30):
            mid = (lo + hi) / 2
            if len(self.encode_step(X, 2.0 ** mid)) <= budget:
                hi = mid
            else:
                lo = mid
        return self.encode_step(X, 2.0 ** hi), [{"step": 2.0 ** hi}] * len(X)

    def decode_group(self, data, nb, n):
        size = 8 * (1 + 2 * nb)
        vals = struct.unpack(f">d{nb}d{nb}d", data[:size])
        step, lo, hi = vals[0], np.array(vals[1:1 + nb]), np.array(vals[1 + nb:])
        Q = np.array(self.coder.decode_group(data[size:], nb, n))
        return lo[:, None] + Q * step * (hi - lo)[:, None]


RATECTL_CODECS = [RateCtlCodec(ZSTD, 4.0), RateCtlCodec(FLAC, 4.0)]
