# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Size, worst error and throughput per signal type: ~1 GiB of float64 through the public API,
single thread (plus optional thread scaling). Run on AC power with nothing else busy; for a
quick look, pass --gib 0.05.

Each signal type is one contiguous series of consecutive minutes (different seeds per minute;
minute boundaries are block boundaries), so encode() splits it into units itself. Timings are
best of --reps full passes; the index (min/max/mean) and zstd are included, data generation isn't.

    uv run python bench/bench_gb.py [--gib 1.0] [--reps 3] [--threads 1,4,8]
"""

import argparse
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "tests"))

from _series import decode_series, encode_series
from _signals import KINDS, MINUTE, discrete_minute, minute

import fluxcode
from fluxcode import Params

SIGNALS = KINDS + ["sensor-0.1", "random-walk q0.01", "noisy-sine q0.1", "noisy-sine q0.01 via float32"]
CONFIGS = {"default": Params(), "noise floor off": Params(noise_floor_sigma=0),
           "target 6": Params(target_bits_per_sample=6.0), "effort 1": Params(effort=1), "effort 5": Params(effort=5),
           "effort 9": Params(effort=9)}


def series(sig, minutes):
    if sig in KINDS:
        parts = [minute(sig, 10_000 + j) for j in range(minutes)]
    elif sig == "sensor-0.1":
        parts = [minute(sig, 20_000 + j) for j in range(minutes)]
    else:
        parts = [discrete_minute(sig, 30_000 + j) for j in range(minutes)]
    return np.concatenate(parts)


def best(f, reps):
    t, out = np.inf, None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = f()
        t = min(t, time.perf_counter() - t0)
    return t, out


def machine():
    try:
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=False).stdout.strip()
    except OSError:
        cpu = platform.processor()
    return f"{cpu}, {platform.system()} {platform.release()}, Python {platform.python_version()}, numpy {np.__version__}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gib", type=float, default=1.0)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--threads", default="1,4,8")
    a = ap.parse_args()
    minutes = max(1, round(a.gib * 2 ** 30 / 8 / MINUTE / len(SIGNALS)))
    n_sig = minutes * MINUTE
    total_bytes = 8 * n_sig * len(SIGNALS)
    print(f"# fluxcode gigabyte benchmark\n\n{machine()}\n\n{len(SIGNALS)} signal types x {minutes} minutes "
          f"= {total_bytes / 2 ** 30:.2f} GiB of float64 ({total_bytes // 8000:,} blocks); best of {a.reps}.\n")

    t0 = time.perf_counter()
    data = {s: series(s, minutes) for s in SIGNALS}
    print(f"(generated in {time.perf_counter() - t0:.0f} s)\n")
    warm = data[SIGNALS[0]][:MINUTE]
    for p in CONFIGS.values():
        decode_series(encode_series(warm, p)[0])

    for name, p in CONFIGS.items():
        print(f"## {name}\n")
        print("| signal | bits/sample | ratio | worst max err (% range) | encode MB/s | decode MB/s | "
              "encode µs/block | decode µs/block |")
        print("|---|---|---|---|---|---|---|---|")
        te_sum = td_sum = bits = 0.0
        for s, x in data.items():
            te, (units, lo, hi, _) = best(lambda x=x, p=p: encode_series(x, p), a.reps)
            td, ys = best(lambda units=units: fluxcode.decode(units), a.reps)  # one DecodedUnit per unit, as rows are read
            err = np.abs(np.concatenate([decoded.values for decoded in ys]) - x).reshape(-1, 1000).max(1)
            worst = 100 * (err / np.where(hi > lo, hi - lo, 1.0)).max()
            nb = len(lo)
            b = 8 * sum(map(len, units))
            te_sum, td_sum, bits = te_sum + te, td_sum + td, bits + b
            mb = 8 * len(x) / 1e6
            print(f"| {s} | {b / len(x):.2f} | {64 * len(x) / b:.1f} | {worst:.4f} | {mb / te:.0f} | {mb / td:.0f} | "
                  f"{1e6 * te / nb:.2f} | {1e6 * td / nb:.2f} |")
            del units, lo, ys
        nb = total_bytes // 8000
        print(f"| **all ({total_bytes / 2 ** 30:.2f} GiB)** | {bits / (total_bytes / 8):.2f} | "
              f"{64 * total_bytes / 8 / bits:.1f} | | {total_bytes / 1e6 / te_sum:.0f} | {total_bytes / 1e6 / td_sum:.0f} | "
              f"{1e6 * te_sum / nb:.2f} | {1e6 * td_sum / nb:.2f} |\n")

    threads = [int(t) for t in a.threads.split(",") if t]
    if threads and threads != [1]:
        print("## Threads (default params; one signal type per task; the kernels release the GIL)\n")
        print("| threads | encode MB/s | decode MB/s | encode speedup |")
        print("|---|---|---|---|")
        p = CONFIGS["default"]
        encoded = {s: encode_series(x, p)[0] for s, x in data.items()}
        base = None
        for t in threads:
            with ThreadPoolExecutor(t) as pool:
                te, _ = best(lambda: list(pool.map(lambda x: encode_series(x, p), data.values())), a.reps)
                td, _ = best(lambda: list(pool.map(lambda e: fluxcode.decode(e), encoded.values())), a.reps)
            base = base or te
            print(f"| {t} | {total_bytes / 1e6 / te:.0f} | {total_bytes / 1e6 / td:.0f} | {base / te:.1f}x |")


if __name__ == "__main__":
    main()
