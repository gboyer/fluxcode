# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""fluxprune speed: rank, prune (automatic, with max_error, with max_error and a cap) and peaks, per signal and
length, single thread, best of --reps. Writes bench/PRUNE_RESULTS.md.

    uv run python bench/prune.py [--reps 5]
"""

import argparse
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import _fresh_numba_cache  # noqa: F401  (before fluxprune: a cache keyed on its sources)
import numpy as np

import fluxprune

SIZES = (3_600, 86_400, 1_000_000)


def signals(n, rng):
    """name -> 1 s scans: a smooth trace with a little noise, a noisy one, steps, white noise, a vibration."""
    s = np.arange(n, dtype=np.float64)
    walk = np.cumsum(rng.normal(0, 0.01, n))
    period = 60 + 20 * np.sin(2 * np.pi * s / 7200)
    return {
        "smooth + noise": np.round(20 + 3 * np.sin(2 * np.pi * s / 86_400) + walk + rng.normal(0, 0.03, n), 2),
        "random walk + noise": np.cumsum(rng.normal(0, 0.1, n)) + rng.normal(0, 1, n),
        "steps": np.round(np.repeat(rng.uniform(0, 100, n // 600 + 1), 600)[:n] + rng.normal(0, 0.05, n), 1),
        "white noise": rng.normal(0, 1, n),
        "vibration": 6 * np.sin(np.cumsum(2 * np.pi / period)) + rng.normal(0, 0.3, n),
    }


def calls(t, x):
    n = len(x)
    tol = 0.01 * np.ptp(x)
    return {
        "rank": lambda: fluxprune.rank(t, x),
        "prune": lambda: fluxprune.prune(t, x),
        "max_error": lambda: fluxprune.prune(t, x, max_error=tol),
        "max_error + cap": lambda: fluxprune.prune(t, x, max_error=tol, max_points=n // 100),
        "peaks": lambda: fluxprune.peaks(t, x, period=np.timedelta64(60, "s"), prominence=0.05 * np.ptp(x)),
    }


def best(call, reps):
    call()
    out = np.inf
    for _ in range(reps):
        t0 = time.perf_counter()
        call()
        out = min(out, time.perf_counter() - t0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    start = time.time()
    rows = []
    for n in SIZES:
        t = (np.arange(n, dtype=np.int64) * 1_000_000_000).view("datetime64[ns]")
        for name, x in signals(n, rng).items():
            todo = calls(t, x)
            times = {key: best(call, args.reps) for key, call in todo.items()}
            kept = {key: len(todo[key]()) for key in ("prune", "max_error", "max_error + cap")}
            rows.append((n, name, times, kept))
            print(n, name, {k: f"{1e3 * v:.2f} ms" for k, v in times.items()}, flush=True)

    power = ""
    if sys.platform == "darwin":
        power = subprocess.run(["pmset", "-g", "ps"], capture_output=True, text=True, check=False).stdout
    source = "AC power" if "AC Power" in power else "power source unknown or battery"
    lines = [
        f"# fluxprune speed (bench/prune.py, {time.strftime('%Y-%m-%d')}, {source})",
        "",
        (f"{platform.machine()}, {platform.system()} {platform.release()}, Python {platform.python_version()}, "
         f"numpy {np.__version__}; one thread, best of {args.reps}, 1 s scans (datetime64[ns] times)."),
        "",
        ("rank: the O(n log n) ranking pass. prune: rank + the automatic count + 2 refinement passes. max_error: prune "
         "with max_error = 1% of the range (the swinging-door fill). + cap: the same with max_points = n / 100, which "
         "binds on the noisy signals (then a bisection over the automatic count, and the worst-first repair). peaks: a "
         "60 s period and a prominence of 5% of the range. Times added to rank are against rank; peaks is a separate "
         "call. Each call is timed on its own, so a difference below ~0.05 ms at 3,600 points is run-to-run noise."),
        "",
        f"(generated in {time.time() - start:.0f} s)",
        "",
        "| n | signal | rank | ns/point | prune | max_error | max_error + cap | peaks | points kept (prune / max_error / + cap) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for n, name, tm, kept in rows:
        r = tm["rank"]

        def added(key, r=r, tm=tm):
            return f"{1e3 * (tm[key] - r):+.2f} ms ({100 * (tm[key] - r) / r:+.0f}%)"

        lines.append(f"| {n:,} | {name} | {1e3 * r:.2f} ms | {1e9 * r / n:.0f} | {added('prune')} | {added('max_error')} | "
                     f"{added('max_error + cap')} | {1e3 * tm['peaks']:.2f} ms ({100 * tm['peaks'] / r:.0f}% of rank) | "
                     f"{kept['prune']:,} / {kept['max_error']:,} / {kept['max_error + cap']:,} |")
    (Path(__file__).parent / "PRUNE_RESULTS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
