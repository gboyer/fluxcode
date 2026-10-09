# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""fluxcode (the package at the repo root) wrapped for the experiments: the block group codec adapter, the
B sweep over four versions (continuous and discretized data), and the noise-floor score used by the
report. Single thread; µs per 1000-sample block, best of 3. A block group is one fluxcode block group per minute and
decodes from its own bytes (header, per-block anchors, zstd frame); errors are % of the block's range.
"""

import time

import fluxcode
import numpy as np
import zstandard
from fluxcode import Params, _bitpacking, _encoder, _format, _group

from tslab.common.datasets import DISCRETE, load, load_discrete


class FluxCodec:
    """fluxcode-<bits>[-f<f>]: a minute (blocks X[nb, n]) as one real fluxcode block group, nothing added.
    Quantize on a power-of-two step (a decimal 10^p grid if the data sits on one), delta order
    0/1/2/3 by variance, residuals mod 2^16 -> zigzag -> 16 bit planes or 2 byte planes (whichever
    Params.effort decides) -> zstd; with f, the step
    is at most f * sigma on blocks that look like white noise. `params` overrides bits / noise_f."""

    def __init__(self, bits=16, noise_f=None, params=None):
        self.params = params or Params(max_quantize_bits=bits, noise_floor_sigma=noise_f or 0)
        self.name = f"fluxcode-{bits}" + (f"-f{noise_f:g}" if noise_f else "")
        self.label = (f"fluxcode, B = {bits}: power-of-two or decimal step, delta order 0/1/2/3, bit or byte planes, zstd"
                      + (f", noise floor f = {noise_f:g}" if noise_f else ""))

    def group(self, X):
        return fluxcode.encode_group(np.ascontiguousarray(X).ravel(), self.params).group

    def encode_group(self, X):
        group = self.group(X)
        parsed = _group.decompress(group)  # the package's own reader: no layout assumptions here
        nb = parsed.header.num_blocks
        params = [int(_format.get_int16(parsed.raw_body, _format.grid_params_start(nb), nb, b)) for b in range(nb)]
        infos = [{"order": int(f & _format.BLOCK_FLAG_ORDER), "decimal": bool(f & _format.BLOCK_FLAG_DECIMAL),
                  "param": p} for f, p in zip(parsed.block_flags, params)]
        return group, infos

    def decode_group(self, data, nb, n):
        return fluxcode.decode_group(data).values.reshape(nb, n)


FLUX_CODECS = [FluxCodec(b) for b in (10, 12, 16)] + [FluxCodec(16, f) for f in (0.01, 0.03, 0.1, 0.25, 0.5, 1.0)]


def best_of(f, reps=3):
    best, out = np.inf, None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = f()
        best = min(best, time.perf_counter() - t0)
    return best, out


def run(codec, mins):
    X0 = mins[0][1]
    codec.decode_group(codec.group(X0), *X0.shape)  # compile / warm up
    te, encs = best_of(lambda: [codec.group(m[1]) for m in mins])
    td, outs = best_of(lambda: [codec.decode_group(e, *m[1].shape) for e, m in zip(encs, mins)])
    nblocks = sum(len(m[1]) for m in mins)
    rmse, maxerr, same = [], [], 0
    for m, Y in zip(mins, outs):
        X, lo, hi = m[1], m[2], m[3]
        fs = np.where(hi > lo, hi - lo, 1.0)
        rmse.extend(np.sqrt(np.mean((Y - X) ** 2, axis=1)) / fs)
        maxerr.extend(np.abs(Y - X).max(axis=1) / fs)
        same += np.count_nonzero(Y == X)
    return {"bps": 8 * sum(map(len, encs)) / (nblocks * 1000), "rmse_med": 100 * np.median(rmse),
            "rmse_max": 100 * max(rmse), "maxerr_max": 100 * max(maxerr), "exact": same / (nblocks * 1000),
            "enc": 1e6 * te / nblocks, "dec": 1e6 * td / nblocks}


def ideal_quantum(mins):
    """Lossless reference for discretized data: the same pipeline (orders 0-3 over the whole block,
    bit planes, zstd-3, block minima as anchors) with step = the data's quantum, told in advance."""
    zc = zstandard.ZstdCompressor(level=3, write_checksum=False)
    total, nblocks = 0, 0
    for _, X, lo, hi, qt in mins:
        nb = len(X)
        head = np.empty(nb, np.uint8)
        resid = np.empty((nb, 1000), np.int16)
        for b in range(nb):
            q = np.rint((X[b] - lo[b]) / qt).astype(np.int32)
            head[b] = _encoder.pick_order(q, 1000, 0b1111)
            _encoder.residual(q, head[b], resid[b])
        anchors = np.ascontiguousarray(lo, np.float64).view(np.int64)  # where each block's grid starts
        sizes = np.full(nb, 1000)
        offsets = _format.layout(head, sizes)
        byte_body = _bitpacking.write_group(head, sizes, np.zeros(nb, np.int64), anchors, resid.ravel(),
                                            np.zeros(0, np.uint8), None, offsets)
        bit_body = _bitpacking.to_bit_planes(byte_body, nb, int(offsets.octet_offsets[-1]), False)
        total += len(zc.compress(bit_body))
        nblocks += nb
    return 8 * total / (nblocks * 1000)


# No noise floor here (it has its own sweep: noise_score / the report's noise section).
VERSIONS = [("order 1, power-of-two only", dict(diff_orders={1}, decimal_detection=False)),
            ("order 1, decimal detect", dict(diff_orders={1})),
            ("order 0-3, power-of-two only", dict(decimal_detection=False)),
            ("order 0-3, decimal detect", dict())]
SWEEP_BITS = range(9, 17)


def sweep(mins=None, dmins=None):
    """The four versions at B = 9..16 on continuous and discretized data."""
    mins = load() if mins is None else mins
    dmins = load_discrete() if dmins is None else dmins
    res = {}
    for label, kw in VERSIONS:
        for B in SWEEP_BITS:
            c = FluxCodec(params=Params(max_quantize_bits=B, noise_floor_sigma=0, **kw))
            rc, rd = run(c, mins), run(c, dmins)
            if B == 16:
                rd["per_signal"] = {name: run(c, [m for m in dmins if m[0] == name])["bps"] for name, _, _ in DISCRETE}
            res[label, B] = (rc, rd)
    ideal = {name: ideal_quantum([m for m in dmins if m[0] == name]) for name, _, _ in DISCRETE}
    return res, ideal


def noise_score(codec, mins):
    """Block group codec on noisy_sets() minutes: bits/sample and median RMS error against the input and
    the clean signal (groups of the true noise sigma), worst max error (% of range)."""
    bits = 0
    e_in, e_clean, e_max = [], [], []
    for X, lo, hi, C, s in mins:
        data = codec.encode_group(X)[0]
        Y = codec.decode_group(data, *X.shape)
        bits += 8 * len(data)
        e_in.append(np.sqrt(np.mean((Y - X) ** 2, axis=1)) / s)
        e_clean.append(np.sqrt(np.mean((Y - C) ** 2, axis=1)) / s)
        e_max.append(np.abs(Y - X).max(axis=1) / (hi - lo))
    nb = 60 * len(mins)
    return {"bps": bits / (nb * 1000), "in": np.median(np.concatenate(e_in)),
            "clean": np.median(np.concatenate(e_clean)), "max": 100 * np.max(np.concatenate(e_max))}
