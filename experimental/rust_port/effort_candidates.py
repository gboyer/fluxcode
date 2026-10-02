# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Which effort table would the Rust accelerator allow? Candidate settings (layout, block flushes,
zstd levels) on the day-scale stress test's sensor mix and on the report's signals, each with
python-zstandard and with the extension: size against today's default (best of both layouts, one
block run, zstd 3), and encode throughput (MiB/s of float64 input, one minute-series per task) at 1, 4
and 8 threads.

Run on AC power with nothing else busy; the paths of a candidate run back to back in one process, so
ratios hold even where absolute numbers don't. Stops if it finds the machine on battery (unless --allow-battery).

    uv run python rust_port/effort_candidates.py [--reps 7] [--threads 1,4,8]
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
sys.path[:0] = [str(ROOT / "tests"), str(ROOT / "bench")]
import fluxcode_rs
from _signals import KINDS, minute
from stress import PRESETS, gen_minute

import fluxcode
from fluxcode import Params, _unit

CANDIDATES = {
    "heuristic, one run (effort 2)": _unit.Effort("heuristic", False, (3,)),
    "best, one run (effort 4, default)": _unit.Effort("best", False, (3,)),
    "heuristic, flushed": _unit.Effort("heuristic", True, (3,)),
    "best, flushed (effort 5)": _unit.Effort("best", True, (3,)),
    "best, flushed, zstd 3 + 9 (effort 9)": _unit.Effort("best", True, (3, 9)),
}
BASE = "best, one run (effort 4, default)"


def power():
    if platform.system() != "Darwin":
        return "unknown"
    return subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout.splitlines()[0]


def use(rust):
    _unit._compress_unit = fluxcode_rs.compress_unit if rust else None


def best(f, reps):
    t = np.inf
    for _ in range(reps):
        t0 = time.perf_counter()
        f()
        t = min(t, time.perf_counter() - t0)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--threads", default="1,4,8")
    ap.add_argument("--allow-battery", action="store_true", help="smoke test only: the numbers mean little")
    a = ap.parse_args()
    threads = [int(t) for t in a.threads.split(",")]
    if "AC Power" not in power() and not a.allow_battery:
        sys.exit(f"not on AC power: {power()}")
    mix = PRESETS["sensor-mix"]
    sets = {
        "sensor mix (100 units, weighted as bench/stress.py)": [gen_minute(k, 100 + i) for k, w in mix.items() for i in range(w)],
        "report signals (15 kinds x 3)": [minute(k, 300 + i) for k in KINDS for i in range(3)],
    }
    print(f"{power()}\n")
    default_effort = Params().effort
    for label, tasks in sets.items():
        mib = sum(t.nbytes for t in tasks) / 2**20
        print(f"## {label}\n")
        print("| settings | size | " + " | ".join(f"{t} thr python / rust" for t in threads) + " |")
        print("|---|---:|" + "---:|" * len(threads))
        base_size = None
        rows = []
        for name, effort in CANDIDATES.items():
            _unit.EFFORTS[default_effort] = effort
            use(True)
            size = sum(len(fluxcode.encode(t)[0][0]) for t in tasks)
            if name == BASE:
                base_size = size
            cells = []
            for nt in threads:
                res = []
                for rust in (False, True):
                    use(rust)
                    with ThreadPoolExecutor(nt) as ex:
                        def run():
                            return list(ex.map(fluxcode.encode, tasks))
                        run()
                        res.append(mib / best(run, a.reps))
                cells.append(f"{res[0]:.0f} / {res[1]:.0f} MiB/s")
            rows.append((name, size, cells))
        for name, size, cells in rows:
            print(f"| {name} | {100 * (size / base_size - 1):+.2f}% | " + " | ".join(cells) + " |")
        print()
    print(f"power at the end: {power()}")


if __name__ == "__main__":
    main()
