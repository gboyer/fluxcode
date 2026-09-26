# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""How do the fluxcode prototype's options trade size against error and speed?

Method: fluxproto (tslab/flux/proto.py) on continuous and discretized minutes: a B sweep with a
re-encode idempotence check, field vs per-block layout, zstd level, order 1 vs orders 0-3 (and
against delta0123-zstd), an encode-time profile, the discretized signals against the
lossless ideal, four versions at B = 9..16, and byte / nibble / bit planes.
Single thread; µs per 1000-sample block, best of 3.

Data: tslab.common.datasets.load() (7,200 continuous blocks) and load_discrete() (5,400 blocks of
signals on a fixed quantum, as real data often is).
Sizes are whole units (block minima included; the lossless ideal stores them too). Errors are % of
the block's range.

    uv run python -m bench.fluxproto_sweep
"""

import os

os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import time

import numpy as np
import zstandard

from tslab.common.datasets import DISCRETE, load, load_discrete
from tslab.common.intcode import pick_order
from tslab.entropy.delta_zstd import DeltaZstd
from tslab.flux import proto as fp


def best_of(f, reps=3):
    best, out = np.inf, None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = f()
        best = min(best, time.perf_counter() - t0)
    return best, out


def run(codec, mins, idem=False):
    X0 = mins[0][1]
    codec.decode_unit(codec.encode_unit(X0)[0], *X0.shape)  # compile / warm up
    te, encs = best_of(lambda: [codec.encode_unit(m[1])[0] for m in mins])
    td, outs = best_of(lambda: [codec.decode_unit(e, *m[1].shape) for e, m in zip(encs, mins)])
    nblocks = sum(len(m[1]) for m in mins)
    rmse, maxerr, same = [], [], 0
    for m, Y in zip(mins, outs):
        X, lo, hi = m[1], m[2], m[3]
        fs = np.where(hi > lo, hi - lo, 1.0)
        rmse.extend(np.sqrt(np.mean((Y - X) ** 2, axis=1)) / fs)
        maxerr.extend(np.abs(Y - X).max(axis=1) / fs)
        if idem:  # re-encode the decoded minute (index recomputed from it): same bytes?
            same += np.array_equal(codec.raw(X, lo, hi), codec.raw(Y, Y.min(1), Y.max(1)))
    out = {"bps": 8 * sum(map(len, encs)) / (nblocks * 1000), "rmse_med": 100 * np.median(rmse),
           "rmse_max": 100 * max(rmse), "maxerr_max": 100 * max(maxerr),
           "enc": 1e6 * te / nblocks, "dec": 1e6 * td / nblocks}
    if idem:
        out["idem"] = same / len(mins)
    return out


def row(name, r):
    idem = f"{100 * r['idem']:.0f}%" if "idem" in r else "—"
    return (f"| {name} | {r['bps']:.2f} | {r['rmse_med']:.5f}% | {r['rmse_max']:.4f}% | {r['maxerr_max']:.4f}% | "
            f"{r['enc']:.1f} | {r['dec']:.1f} | {idem} |")


HEAD = ("| codec | bits/sample | median RMSE | worst RMSE | worst max err | encode µs | decode µs | re-encode identical |\n"
        "|---|---|---|---|---|---|---|---|")


def profile(mins):
    """Where fluxproto-16's encode time goes."""
    c = fp.FluxProto(16)
    nblocks = sum(len(m[1]) for m in mins)
    t_all, _ = best_of(lambda: [c.encode_unit(m[1]) for m in mins])
    t_raw, _ = best_of(lambda: [c.raw(m[1], m[2], m[3]) for m in mins])
    c1 = fp.FluxProto(16, orders=(1,))
    t_raw1, _ = best_of(lambda: [c1.raw(m[1], m[2], m[3]) for m in mins])
    q = np.empty(1000, np.int32)
    Q = []
    for m in mins:
        for b in range(len(m[1])):
            fp._quantize(m[1][b], m[2][b], fp._exponent(m[3][b] - m[2][b], 16), q)
            Q.append(q.copy())
    pick_order(Q[0])
    t_pick, _ = best_of(lambda: [pick_order(x) for x in Q])
    us = lambda t: 1e6 * t / nblocks
    print(f"\nfluxproto-16 encode profile (µs per block): total {us(t_all):.1f}; quantize + pick + residual + planes "
          f"{us(t_raw):.1f} (with order fixed at 1: {us(t_raw1):.1f}); order pick alone {us(t_pick):.2f}; "
          f"zstd-3 {us(t_all - t_raw):.1f}")


def ideal_quantum(mins, bitshuffle=True):
    """Lossless reference for discretized data: the same pipeline with step = the data's quantum
    (bit-shuffled planes by default, as in fluxcode's docs/SPEC.md)."""
    zc = zstandard.ZstdCompressor(level=3, write_checksum=False)
    total, nblocks = 0, 0
    r, u = np.empty(1000, np.int32), np.empty(1000, np.uint16)
    for _, X, lo, hi, qt in mins:
        nb = len(X)
        head = np.empty(nb, np.uint8)
        low, high = np.empty((nb, 1000), np.uint8), np.empty((nb, 1000), np.uint8)
        for b in range(nb):
            q = np.rint((X[b] - lo[b]) / qt).astype(np.int32)
            head[b], _ = pick_order(q)
            fp._residual_u16(q, head[b], r, u)
            low[b], high[b] = u & 255, u >> 8
        if bitshuffle:
            planes = np.empty((16, nb, 125), np.uint8)
            fp._bitshuffle(low.view(np.uint64), high.view(np.uint64), planes)
            body = planes.ravel()
        else:
            body = np.concatenate([low.ravel(), high.ravel()])
        anchor = np.ascontiguousarray(lo, "<f8").view(np.uint8).reshape(-1, 8).T.ravel()  # block minima, as the codec
        total += len(zc.compress(np.concatenate([head, body, anchor]).tobytes()))
        nblocks += nb
    return 8 * total / (nblocks * 1000)


VERSIONS = [("order 1, power-of-two only", dict(orders=(1,), planes="bit")),
            ("order 1, decimal detect", dict(orders=(1,), step_detect=True, planes="bit")),
            ("order 0-3, power-of-two only", dict(planes="bit")),
            ("order 0-3, decimal detect", dict(step_detect=True, planes="bit")),
            ("order 0-3, decimal detect, byte planes", dict(step_detect=True)),  # other plane layouts, for reference
            ("order 0-3, decimal detect, nibble planes", dict(step_detect=True, planes="nibble"))]
SWEEP_BITS = range(9, 17)


def exact_share(codec, mins):
    """Share of samples decoded to the identical float64."""
    same = total = 0
    for m in mins:
        same += np.count_nonzero(codec.decode_unit(codec.encode_unit(m[1])[0], *m[1].shape) == m[1])
        total += m[1].size
    return same / total


def sweep(mins=None, dmins=None):
    """The four versions at B = 9..16 on continuous and discretized data."""
    mins = load() if mins is None else mins
    dmins = load_discrete() if dmins is None else dmins
    res = {}
    for label, kw in VERSIONS:
        for B in SWEEP_BITS:
            c = fp.FluxProto(B, **kw)
            rc, rd = run(c, mins), run(c, dmins)
            rd["exact"] = exact_share(c, dmins)
            rd["per_signal"] = {name: run(c, [m for m in dmins if m[0] == name])["bps"] for name, _, _ in DISCRETE} \
                if B == 16 else None
            rc["per_signal_c"] = {k: run(c, [m for m in mins if m[0] == k])["bps"] for k in dict.fromkeys(m[0] for m in mins)} \
                if B == 16 else None
            res[label, B] = (rc, rd)
    ideal = {name: ideal_quantum([m for m in dmins if m[0] == name]) for name, _, _ in DISCRETE}
    return res, ideal


def print_sweep(res, ideal):
    print("\n## Sweep: four versions, B = 9..16\n")
    print("| version | B | continuous bits/sample | continuous median RMSE | continuous worst max err | "
          "discretized bits/sample | discretized median RMSE | discretized bit-exact samples | encode µs (cont / disc) | decode µs |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for (label, B), (rc, rd) in res.items():
        print(f"| {label} | {B} | {rc['bps']:.2f} | {rc['rmse_med']:.5f}% | {rc['maxerr_max']:.4f}% | {rd['bps']:.2f} | "
              f"{rd['rmse_med']:.5f}% | {100 * rd['exact']:.1f}% | {rc['enc']:.1f} / {rd['enc']:.1f} | {rc['dec']:.1f} |")
    print("\nDiscretized signals at B = 16 (bits/sample):\n")
    print("| signal | ideal (lossless at the quantum) | " + " | ".join(l for l, _ in VERSIONS) + " |")
    print("|---|---|" + "---|" * len(VERSIONS))
    for name, _, _ in DISCRETE:
        print(f"| {name} | {ideal[name]:.2f} | " + " | ".join(f"{res[l, 16][1]['per_signal'][name]:.2f}" for l, _ in VERSIONS) + " |")


def planes_compare(mins=None, dmins=None, rounds=5):
    """Byte planes vs bit-shuffle, orders 0-3 + decimal detect, B = 16. Configurations are timed in
    alternation (best of `rounds`) so machine drift hits them equally."""
    mins = load() if mins is None else mins
    dmins = [m[:4] for m in (load_discrete() if dmins is None else dmins)]
    both = mins + dmins
    configs = {f"{p} planes, zstd-{l}": fp.FluxProto(16, step_detect=True, planes=p, level=l)
               for l in (3, 1) for p in ("byte", "nibble", "bit")}
    nb = 60 * len(both)
    te = {k: np.inf for k in configs}
    td = dict(te)
    enc = {}
    for c in configs.values():
        c.decode_unit(c.encode_unit(both[0][1])[0], 60, 1000)
    for _ in range(rounds):
        for k, c in configs.items():
            t, enc[k] = best_of(lambda: [c.encode_unit(m[1])[0] for m in both], 1)
            te[k] = min(te[k], t)
            t, _ = best_of(lambda: [c.decode_unit(e, 60, 1000) for e in enc[k]], 1)
            td[k] = min(td[k], t)
    print("\n## Byte planes vs bit-shuffle (orders 0-3, decimal detect, B = 16)\n")
    print("| layout | continuous bits/sample | discretized bits/sample | encode µs | decode µs |\n|---|---|---|---|---|")
    nc = len(mins)
    for k in configs:
        bc = 8 * sum(map(len, enc[k][:nc])) / (60000 * nc)
        bd = 8 * sum(map(len, enc[k][nc:])) / (60000 * len(dmins))
        print(f"| {k} | {bc:.2f} | {bd:.2f} | {1e6 * te[k] / nb:.2f} | {1e6 * td[k] / nb:.2f} |")
    # the shuffle alone, one minute at a time as in the codec
    X, lo, hi = both[0][1:4]
    c = configs["byte planes, zstd-3"]
    raws = [c.raw(*m[1:4]) for m in both]
    lows = [r[540:60540].reshape(60, 1000) for r in raws]
    highs = [r[60540:].reshape(60, 1000) for r in raws]
    planes = np.empty((16, 60, 125), np.uint8)
    lo_, hi_ = np.empty((60, 1000), np.uint8), np.empty((60, 1000), np.uint8)
    fp._bitshuffle(lows[0].view(np.uint64), highs[0].view(np.uint64), planes)
    fp._bitunshuffle(planes, lo_.view(np.uint64), hi_.view(np.uint64))
    ts, _ = best_of(lambda: [fp._bitshuffle(a.view(np.uint64), b.view(np.uint64), planes) for a, b in zip(lows, highs)], 5)
    tu, _ = best_of(lambda: [fp._bitunshuffle(planes, lo_.view(np.uint64), hi_.view(np.uint64)) for _ in lows], 5)
    print(f"\nshuffle alone: {1e6 * ts / nb:.2f} µs/block, unshuffle {1e6 * tu / nb:.2f} µs/block")
    print("\nPer signal, zstd-3 (bits/sample):\n\n| signal | byte planes | bit-shuffle | change |\n|---|---|---|---|")
    for label, group in (("", mins), (" (discretized)", dmins)):
        for kind in dict.fromkeys(m[0] for m in group):
            idx = [i for i, m in enumerate(both) if m[0] == kind and (i >= nc) == (group is dmins)]
            a = 8 * sum(len(enc["byte planes, zstd-3"][i]) for i in idx) / (60000 * len(idx))
            b = 8 * sum(len(enc["bit planes, zstd-3"][i]) for i in idx) / (60000 * len(idx))
            print(f"| {kind}{label} | {a:.2f} | {b:.2f} | {100 * (b - a) / a:+.0f}% |")


def main():
    mins = load()
    print("## Continuous signals (7,200 blocks)\n")
    print(HEAD)
    for B in range(16, 8, -1):
        print(row(f"fluxproto-{B}", run(fp.FluxProto(B), mins, idem=True)), flush=True)
    for name, c in [("fluxproto-16, orders 1 only", fp.FluxProto(16, orders=(1,))),
                    ("fluxproto-16, per-block layout", fp.FluxProto(16, layout="block")),
                    ("fluxproto-16, zstd-1", fp.FluxProto(16, level=1)),
                    ("delta0123-zstd-16, no cap", DeltaZstd(16, cap_bps=0)),
                    ("delta0123-zstd-12, no cap", DeltaZstd(12, cap_bps=0)),
                    ("delta0123-zstd-16, cap 8", DeltaZstd(16))]:
        print(row(name, run(c, mins)), flush=True)
    profile(mins)

    dmins = load_discrete()
    print("\n## Discretized signals\n")
    print("| signal | quantum | ideal (lossless at the quantum) | " + " | ".join(f"fluxproto-{B}" for B in (16, 14, 12, 10))
          + " | fluxproto-16 lossless? |")
    print("|---|---|---|" + "---|" * 5)
    for name, _, qt in DISCRETE:
        sub = [m for m in dmins if m[0] == name]
        cells = []
        lossless = None
        for B in (16, 14, 12, 10):
            c = fp.FluxProto(B)
            r = run(c, sub)
            cells.append(f"{r['bps']:.2f} ({r['rmse_med']:.4f}%)")
            if B == 16:  # does rounding the decoded values to the quantum give back the input?
                lossless = all(np.array_equal(np.round(c.decode_unit(c.encode_unit(X)[0], *X.shape) / q) * q, X)
                               for _, X, lo, hi, q in sub)
        print(f"| {name} | {qt:.4g} | {ideal_quantum(sub):.2f} | " + " | ".join(cells) + f" | {'yes' if lossless else 'no'} |",
              flush=True)
    print_sweep(*sweep(mins, dmins))
    planes_compare(mins, dmins)


if __name__ == "__main__":
    main()
