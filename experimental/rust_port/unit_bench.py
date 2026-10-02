# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Microbenchmark of the Rust compress_unit alone: one default unit (a minute of 1000-sample blocks) per
signal, the best of many calls, in µs. Rust-only changes show here where compress_bench's run-to-run noise
(a few percent) hides them.

    uv run --extra rust python experimental/rust_port/unit_bench.py [--reps 200]
"""

import argparse
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[2] / "tests")]
import fluxcode_rs
from _signals import KINDS, discrete_minute, minute

import fluxcode
from fluxcode import Params, _unit

SIGNALS = ["sin-4.12hz", "random-walk", "linear", "noisy-sine", "sensor-0.1", "nonfinite"]


def rows_of(sig):
    if sig == "sensor-0.1":
        x = discrete_minute(sig, 1)
    else:
        x = minute("sin-4.12hz" if sig == "nonfinite" else sig, 1).copy()
    if sig == "nonfinite":
        x[np.random.default_rng(1).integers(0, len(x), 200)] = np.nan
    captured = []
    saved = _unit._compress_unit
    _unit._compress_unit = lambda *a: captured.append(a) or saved(*a)
    try:
        fluxcode.encode(x, Params(effort=4))
    finally:
        _unit._compress_unit = saved
    return captured[0][:6]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    a = ap.parse_args()
    cols = [("byte", False, [3]), ("bit", False, [3]), ("best", False, [3]), ("best", True, [3]), ("heuristic", True, [3]), ("best", True, [3, 9])]
    power = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout.splitlines()[:1] if platform.system() == "Darwin" else ["unknown"]
    print(f"power: {power[0] if power else 'unknown'}; libzstd {fluxcode_rs.zstd_version()}\n")
    print("| signal | " + " | ".join(f"{l}{'+flush' if f else ''}{'' if v == [3] else ' ' + str(v)}" for l, f, v in cols) + " |")
    print("|---|" + "---:|" * len(cols))
    for sig in SIGNALS:
        rows = rows_of(sig)
        out = []
        for layout, flush, levels in cols:
            t = np.inf
            for _ in range(a.reps):
                t0 = time.perf_counter()
                fluxcode_rs.compress_unit(*rows, layout, flush, levels)
                t = min(t, time.perf_counter() - t0)
            out.append(f"{t * 1e6:.0f}")
        print(f"| {sig} | " + " | ".join(out) + " |")


if __name__ == "__main__":
    main()
