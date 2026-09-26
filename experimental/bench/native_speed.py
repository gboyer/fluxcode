# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""How much faster are the numba ports than the Python reference codecs?

Method: encode and decode one-minute units with each Python reference unit codec (the block codecs
of tslab.classic behind tslab.common.unit's adapters) and its native (numba) port, single thread.
Timings are per 1000-sample block, best of 3; size and error are over every unit. Native units are
checked against the reference decoders (byte-format compatible). delta0123-zstd has no Python
reference (the codec is compiled); constant-width rows are the Python-glued numba block codecs.
Also reports the share of NEON vector instructions in each compiled kernel (unavailable when numba
loads the kernel from its cache).

Data: tslab.common.datasets.load() (120 one-minute units, 7,200 blocks).

    uv run python -m bench.native_speed
"""

import os

os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import re
import time

import numpy as np

from tslab.classic import CompandedDpcm, Quant8, QuantDeltaDeflate, cubic_expander
from tslab.common.datasets import load
from tslab.common.unit import Concat, SharedBackend
from tslab.constwidth import Delta1Bfp, Delta1Linear8, Delta1Sqrt8
from tslab.entropy import delta_zstd, native
from tslab.entropy.delta_zstd import DeltaZstd

UNITS = [X for _, X, _, _ in load()]


def best_of(f, reps=3):
    best = np.inf
    for _ in range(reps):
        t0 = time.perf_counter()
        out = f()
        best = min(best, time.perf_counter() - t0)
    return best, out


def row(name, kind, enc, dec, nunits, ref_decode=None):
    """enc(X) -> bytes, dec(bytes, nb, n) -> X, timed on the first nunits units; quality over all."""
    units = UNITS[:nunits]
    dec(enc(UNITS[0]), *UNITS[0].shape)  # warm up / compile
    te, blobs = best_of(lambda: [enc(X) for X in units])
    td, _ = best_of(lambda: [dec(b, *X.shape) for b, X in zip(blobs, units)])
    nbytes, rmse = 0, []
    for X in UNITS:
        b = enc(X)
        Y = dec(b, *X.shape)
        nbytes += len(b)
        rmse.extend(np.sqrt(np.mean((Y - X) ** 2, axis=1)) / (X.max(1) - X.min(1)))
        if ref_decode is not None:  # native bytes must decode with the Python reference decoder
            assert np.allclose(ref_decode(b, *X.shape), Y, rtol=0,
                               atol=1e-9 * (X.max() - X.min())), name
    nblocks = 60 * nunits
    total = sum(X.size for X in UNITS)
    return (name, kind, 1e6 * te / nblocks, 1e6 * td / nblocks, 8 * nbytes / total, 100 * np.median(rmse))


def ref(codec):
    return (lambda X: codec.encode_unit(X)[0]), codec.decode_unit


def simd_report():
    """Share of instructions in each compiled kernel that use NEON vector registers."""
    pat = re.compile(r"\bv\d+\.(16b|8b|8h|4h|4s|2s|2d)\b")
    out = []
    for mod, name in ((native, "_quantize_u8"), (native, "_delta_u8"), (native, "_dpcm_encode"),
                      (delta_zstd, "_encode_unit"), (delta_zstd, "_decode_unit"), (native, "_undelta_dequantize"),
                      (native, "_dpcm_decode")):
        f = getattr(mod, name)
        asm = "".join(f.inspect_asm(sig) for sig in f.signatures)
        instr = [l for l in asm.splitlines() if l.startswith("\t") and not l.lstrip().startswith(".")]
        vec = [l for l in instr if pat.search(l)]
        out.append((name, len(instr), len(vec)))
    return out


def main():
    rows = []
    q8, q8d = Concat(Quant8()), SharedBackend(QuantDeltaDeflate(8))
    rows.append(row(q8.name, "python", *ref(q8), 2))
    rows.append(row(q8.name, "native", native.quant8_encode, native.quant8_decode, 120, q8.decode_unit))
    rows.append(row(q8d.name, "python", *ref(q8d), 40))
    n8d = native.Quant8Delta1()
    rows.append(row(q8d.name, "native", *ref(n8d), 120, q8d.decode_unit))
    for bits in (6, 4):
        r = SharedBackend(CompandedDpcm(f"dpcm{bits}-linear-deflate", "", cubic_expander(0.0), bits=bits))
        rows.append(row(r.name, "python", *ref(r), 1))
        rows.append(row(r.name, "native", lambda X, b=bits: native.dpcm_encode(X, b),
                        lambda d, nb, n, b=bits: native.dpcm_decode(d, b, nb, n), 120, r.decode_unit))
    for bits, orders in ((8, (0, 1, 2, 3)), (10, (0, 1, 2, 3)), (12, (0, 1, 2, 3)), (8, (1, 2))):
        c = DeltaZstd(bits, orders=orders)
        rows.append(row(c.name, "native", *ref(c), 120))

    for codec in (Delta1Linear8(), Delta1Sqrt8(), Delta1Bfp()):
        c = Concat(codec)
        rows.append(row(c.name, "native", *ref(c), 120))

    print("| codec | impl | encode µs/block | decode µs/block | 1 day encode (s) | bits/sample | median RMSE |")
    print("|---|---|---|---|---|---|---|")
    for name, kind, te, td, bps, rmse in rows:
        print(f"| {name} | {kind} | {te:.1f} | {td:.1f} | {te * 86400 / 1e6:.2f} | {bps:.2f} | {rmse:.3f}% |")
    print("\nNEON vector instructions in compiled kernels:")
    for name, n, v in simd_report():
        if n == 0:  # numba loaded the kernel from its cache: no assembly to inspect
            print(f"  {name:22s} (cached; clear the numba cache to inspect)")
            continue
        print(f"  {name:22s} {v:4d} / {n:5d} instructions ({100 * v / n:.0f}%)")


if __name__ == "__main__":
    main()
