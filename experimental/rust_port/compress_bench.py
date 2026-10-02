# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""The optional Rust accelerator (rust/) against the Python compress path, on main's efforts.

1. Units from the two paths, every effort and signal: byte-identical (when both link the same libzstd).
2. Single-thread fluxcode.encode per signal at efforts 2, 4 (the default: best of both layouts, no block
   flush) and 5 (block flushes).
3. Thread scaling at the same efforts: one minute-series per task, MiB/s of float64 input.

Run on AC power with nothing else busy: a laptop on battery runs at about half clock and the thread
scaling depends on every core's clock. Both paths run back to back in one process, so a run's ratios hold
even where its absolute numbers don't.

    uv run python rust_port/compress_bench.py [--minutes 4] [--reps 5] [--threads 1,4,8]
"""

import argparse
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tests")]
import fluxcode_rs
import zstandard
from _signals import KINDS, discrete_minute, minute

import fluxcode
from fluxcode import Params, _unit

EFFORTS = (2, 4, 5)
SIGNALS = KINDS + ["sensor-0.1"]


def series(sig, minutes):
    make = discrete_minute if sig == "sensor-0.1" else minute
    return np.concatenate([make(sig, 10_000 + j) for j in range(minutes)])


def best(f, reps):
    t = np.inf
    for _ in range(reps):
        t0 = time.perf_counter()
        f()
        t = min(t, time.perf_counter() - t0)
    return t


def use(rust):
    _unit._compress_unit = fluxcode_rs.compress_unit if rust else None


def power():
    if platform.system() != "Darwin":
        return "unknown"
    return subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout.splitlines()[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=int, default=4)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--threads", default="1,4,8")
    a = ap.parse_args()
    data = {sig: series(sig, a.minutes) for sig in SIGNALS}
    print(f"power: {power()}; libzstd: rust {fluxcode_rs.zstd_version()}, python-zstandard {zstandard.ZSTD_VERSION}\n")

    print("## Units byte-identical, every effort 1-9 x every signal (1 minute each)\n")
    bad = 0
    for effort in range(1, 10):
        for x in data.values():
            use(False)
            ref = fluxcode.encode(x[:60_000], Params(effort=effort)).units
            use(True)
            bad += ref != fluxcode.encode(x[:60_000], Params(effort=effort)).units
    print("all identical\n" if not bad else f"{bad} MISMATCHES\n")

    for effort in EFFORTS:
        print(f"## fluxcode.encode, effort {effort}, single thread (best of {a.reps}; {a.minutes} min x 60k samples)\n")
        print("| signal | python | rust | speedup |")
        print("|---|---:|---:|---:|")
        for sig, x in data.items():
            res = []
            for rust in (False, True):
                use(rust)
                fluxcode.encode(x[:60_000], Params(effort=effort))
                res.append(best(lambda: fluxcode.encode(x, Params(effort=effort)), a.reps) / len(x) * 1e9)
            print(f"| {sig} | {res[0]:.2f} ns/sample | {res[1]:.2f} ns/sample | {res[0] / res[1]:.2f}x |")
        print()

    print(f"## Thread scaling, MiB/s of float64 input (best of {a.reps})\n")
    print("| effort | threads | python | rust | speedup |")
    print("|---|---:|---:|---:|---:|")
    sigs = ["sin-4.12hz", "random-walk", "gauss-spikes", "chirp"]
    tasks = [data[s][i * 60_000:(i + 1) * 60_000].copy() for s in sigs for i in range(a.minutes)] * 2
    mib = sum(t.nbytes for t in tasks) / 2**20
    for effort in EFFORTS:
        for nt in (int(t) for t in a.threads.split(",")):
            res = []
            for rust in (False, True):
                use(rust)
                with ThreadPoolExecutor(nt) as ex:
                    def run():
                        return list(ex.map(lambda t: fluxcode.encode(t, Params(effort=effort)), tasks))
                    run()
                    res.append(mib / best(run, a.reps))
            print(f"| {effort} | {nt} | {res[0]:.0f} | {res[1]:.0f} | {res[1] / res[0]:.2f}x |")
    print()


if __name__ == "__main__":
    main()
