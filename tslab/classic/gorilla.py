# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Gorilla XOR float compression (values only): lossless."""

import numpy as np

from tslab.common.bitio import BitReader, BitWriter, bits_to_f64, f64_to_bits

# ---------------------------------------------------------------------------
# Gorilla (Facebook/Meta, VLDB 2015) XOR float compression -- lossless.
# Values only: timestamps here are implicit (1 per ms). Gorilla's delta-of-delta
# timestamps would add ~1 bit/sample (~125 bytes) for a perfectly regular clock.
# ---------------------------------------------------------------------------

class Gorilla:
    """gorilla-xor: each value XORed with the previous one; the XOR's meaningful bits are stored,
    reusing the previous leading/trailing-zero window when it fits."""

    name = "gorilla-xor"
    variable_size = True
    label = "Gorilla XOR, lossless"

    def encode(self, x):
        w = BitWriter()
        prev = f64_to_bits(x[0])
        w.write(prev, 64)
        p_lead, p_trail = None, None
        stats = {"same": 0, "reuse_window": 0, "new_window": 0}
        for v in x[1:]:
            cur = f64_to_bits(v)
            xor = cur ^ prev
            prev = cur
            if xor == 0:
                w.write(0, 1)
                stats["same"] += 1
                continue
            w.write(1, 1)
            lead = min(64 - xor.bit_length(), 31)
            trail = (xor & -xor).bit_length() - 1
            if p_lead is not None and lead >= p_lead and trail >= p_trail:
                w.write(0, 1)
                w.write(xor >> p_trail, 64 - p_lead - p_trail)
                stats["reuse_window"] += 1
            else:
                sig = 64 - lead - trail
                w.write(1, 1)
                w.write(lead, 5)
                w.write(sig % 64, 6)  # 64 meaningful bits encoded as 0
                w.write(xor >> trail, sig)
                p_lead, p_trail = lead, trail
                stats["new_window"] += 1
        return w.getvalue(), stats

    def decode(self, data, n):
        r = BitReader(data)
        prev = r.read(64)
        out = [bits_to_f64(prev)]
        p_lead = p_trail = 0
        for _ in range(n - 1):
            if r.read(1):
                if r.read(1):
                    p_lead = r.read(5)
                    sig = r.read(6) or 64
                    p_trail = 64 - p_lead - sig
                prev ^= r.read(64 - p_lead - p_trail) << p_trail
            out.append(bits_to_f64(prev))
        return np.array(out)
