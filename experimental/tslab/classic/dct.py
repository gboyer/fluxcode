# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""DCT keeping the K largest coefficients: a frequency-domain contrast."""

import numpy as np
from scipy.fft import dct, idct

from tslab.common.bitio import BitReader, BitWriter

# ---------------------------------------------------------------------------
# Extra: DCT keeping the K largest coefficients.
#   header: max|coef| f64; per coef: 10-bit index, 16-bit signed-quantized value
# ---------------------------------------------------------------------------

class DctTopK:
    """dct-top<k>: orthonormal DCT-II of the block; the k largest coefficients are kept as a 10-bit
    index and a 16-bit value relative to the largest magnitude (header: max |coef| f64)."""

    def __init__(self, k):
        self.k = k
        self.name = f"dct-top{k}"
        self.label = f"DCT, {k} largest coefficients kept"

    def encode(self, x):
        c = dct(x, norm="ortho")
        keep = np.sort(np.argsort(-np.abs(c))[:self.k])
        amp = np.abs(c).max() or 1.0
        w = BitWriter()
        w.write_f64(amp)
        for i in keep:
            w.write(int(i), 10)
            w.write(int(np.rint(c[i] / amp * 32767)) + 32768, 16)
        return w.getvalue(), {}

    def decode(self, data, n):
        r = BitReader(data)
        amp = r.read_f64()
        c = np.zeros(n)
        for _ in range(self.k):
            i = r.read(10)
            c[i] = (r.read(16) - 32768) / 32767 * amp
        return idct(c, norm="ortho")
