# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Can the bit-vs-byte plane choice be predicted without compressing both ways?

Evaluates cheap rules against the real unit sizes in corpus.npz (the fitting set, seed 1),
test.npz (held out, seed 2) and standard.npz (the report's signals, default params), and prints
the i.i.d.-residual table that explains the main effect. Build the sets with corpus.py first.

    uv run python plane_layout/analyze.py [--iid-only]
"""

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import zstandard

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features
import model

HERE = Path(__file__).resolve().parent
RULES = {
    "always bit": lambda f: np.zeros(len(f["h_high"]), bool),
    "always byte": lambda f: np.ones(len(f["h_high"]), bool),
    "high byte ~constant: hi_nz < 5%": lambda f: f["hi_nz"] < 0.05,
    "high byte ~constant: H(high) < 0.25 bit": lambda f: f["h_high"] < 0.25,
    "greedy LZ estimate of both": lambda f: f["lz_ratio"] < 0,
    "H(high) < 0.25 and LZ estimate": lambda f: (f["h_high"] < 0.25) & (f["lz_ratio"] < 0.2),
}


def load(name):
    d = np.load(HERE / name)
    cache = HERE / name.replace(".npz", ".features.npz")
    if cache.exists():
        f = dict(np.load(cache))
    else:
        rows = [features.features(u) for u in d["u"]]
        f = {k: np.array([r[k] for r in rows]) for k in rows[0]}
        np.savez(cache, **f)
    return d, f


def regret(pred, bit, byte):
    """Total bytes of the chosen layouts over the oracle's, minus 1."""
    return np.where(pred, byte, bit).sum() / np.minimum(bit, byte).sum() - 1


def rules_table():
    print("Total size over the best-of-both oracle (0% = never wrong), and how often the pick is the smaller layout\n")
    sets = [(n, *load(n)) for n in ("corpus.npz", "test.npz", "standard.npz")]
    print(f"{'rule':44s}" + "".join(f"{label:>20s}" for label in ("fit (2000)", "held out (1000)", "report signals (160)")))
    for name, rule in RULES.items():
        cells = []
        for _, d, f in sets:
            bit, byte = d["bit"].astype(float), d["byte"].astype(float)
            p = rule(f)
            cells.append(f"{regret(p, bit, byte):+7.2%}  {np.mean(p == (byte < bit)):5.1%} ok")
        print(f"{name:44s}" + "".join(f"{c:>20s}" for c in cells))
    bit, byte = (sets[0][1][k].astype(float) for k in ("bit", "byte"))
    print(f"\nthe oracle itself is {1 - np.minimum(bit, byte).sum() / bit.sum():.1%} smaller than always-bit on the fit set")

    # by H(high) bin
    d, f = sets[0][1:]
    bit, byte = d["bit"].astype(float), d["byte"].astype(float)
    print("\nfit set by H(high byte), bits/sample: how often byte planes win, and what each layout loses to the oracle")
    print(f"{'H(high)':>12s} {'units':>6s} {'byte wins':>10s} {'geo-mean byte/bit':>18s} {'bit loses':>10s} {'byte loses':>11s}")
    edges = [0, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5]
    best = np.minimum(bit, byte)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (f["h_high"] >= lo) & (f["h_high"] < hi)
        if m.any():
            print(f"[{lo:5.2f},{hi:4.2f}) {m.sum():6d} {np.mean(byte[m] < bit[m]):10.0%} {np.exp(np.log(byte[m] / bit[m]).mean()):18.3f}"
                  f" {bit[m].sum() / best[m].sum() - 1:+10.1%} {byte[m].sum() / best[m].sum() - 1:+11.1%}")

    # standard signals: the winner per kind
    d, f = sets[2][1:]
    bit, byte = d["bit"].astype(float), d["byte"].astype(float)
    names = np.array([re.sub(r" P=.*", "", x) for x in d["fam"]])
    print("\nreport signals (8 minutes each): real byte/bit size, H(high), and the rules' picks (B = byte, b = bit)")
    print(f"{'signal':32s} {'byte/bit':>9s} {'H(high)':>8s}   H(high)<.25  LZ")
    for nm in dict.fromkeys(names):
        m = names == nm
        a = np.mean(f["h_high"][m] < 0.25) * 8
        z = np.mean(f["lz_ratio"][m] < 0) * 8
        print(f"{nm:32s} {byte[m].sum() / bit[m].sum():9.3f} {f['h_high'][m].mean():8.2f}   {int(round(a))}/8 byte    {int(round(z))}/8 byte")


def iid_table():
    zc = zstandard.ZstdCompressor(level=3)
    rng = np.random.default_rng(0)
    n = 60_000
    print("i.i.d. Gaussian residuals (zigzag), 60,000 samples; zstd 3 on the planes, bits/sample (excess over the entropy H(u))\n")
    print(f"{'sigma':>6s} {'H(u)':>6s} {'bit':>7s} {'byte':>7s}   {'bit excess':>10s} {'byte excess':>11s}  winner")
    for sig in (0.3, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096):
        x = np.clip(np.round(rng.normal(0, sig, n)), -32000, 32000).astype(np.int64)
        u = np.where(x >= 0, 2 * x, -2 * x - 1).astype(np.uint16)
        h = model.entropy_bits(np.bincount(u, minlength=65536)) / n
        sb, sy = model.streams(u)
        b = len(zc.compress(sb.tobytes())) * 8 / n
        y = len(zc.compress(sy.tobytes())) * 8 / n
        print(f"{sig:6g} {h:6.2f} {b:7.2f} {y:7.2f}   {b - h:+10.2f} {y - h:+11.2f}  {'byte' if y < b else 'bit'}")


def timings():
    d = np.load(HERE / "standard.npz")
    us = d["u"][::8][:20]
    zc = zstandard.ZstdCompressor(level=3, write_checksum=False, write_content_size=True)
    streams = [model.streams(u) for u in us]
    for s in streams:
        model.lz_estimate(s[0]), model.lz_estimate(s[1])

    def best(f):
        t = np.inf
        for _ in range(5):
            t0 = time.perf_counter()
            f()
            t = min(t, time.perf_counter() - t0)
        return t / len(us) * 1e6

    print("\nper unit, microseconds (20 report units):")
    print(f"  zstd 3 on both layouts' plane bytes : {best(lambda: [(zc.compress(s[0].tobytes()), zc.compress(s[1].tobytes())) for s in streams]):7.0f}")
    print(f"  greedy LZ estimate of both layouts  : {best(lambda: [(model.lz_estimate(s[0]), model.lz_estimate(s[1])) for s in streams]):7.0f}")
    print(f"  H(high) from a histogram            : {best(lambda: [model.entropy_bits(np.bincount(u >> 8, minlength=256)) for u in us]):7.0f}")
    print(f"  hi_nz = mean(u > 255)               : {best(lambda: [np.mean(u > 255) for u in us]):7.0f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iid-only", action="store_true")
    a = ap.parse_args()
    iid_table()
    if not a.iid_only:
        print()
        rules_table()
        timings()
