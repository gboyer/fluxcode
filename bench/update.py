# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Update cost against encoding from scratch, single thread.

- `update` of one block of a 60 x 1000 unit (with and without times), and an append;
- `update_time_blocks` of a range straddling two blocks, on a unit of 60 one-second blocks of
  1000 samples and on one of 3,600 one-second blocks of 10 samples (an hour at 10 Hz);
- `update_time_blocks` upserts (no ranges) on the hour unit: one sample in one block, and one
  sample in each of the 3,600 blocks.

Timings are the median of --reps calls after a warm-up; each update is reported relative to
encoding the same unit from scratch.

    uv run python bench/update.py [--reps 300]
"""

import argparse
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from _signals import discrete_minute, minute

import fluxcode

START = np.datetime64("2026-01-01T00:00", "ms")
SECOND = np.timedelta64(1, "s")


def encode_seconds(x, t):
    """encode_time_blocks in one-second blocks from START."""
    return fluxcode.encode_time_blocks(x, t, start_time=START, block_duration=SECOND).unit


def update_seconds(unit, x, t, delete_range=None):
    """update_time_blocks of a unit made by encode_seconds."""
    return fluxcode.update_time_blocks(unit, x, t, start_time=START, block_duration=SECOND, delete_ranges=delete_range)


def median_us(call, reps):
    call()
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        call()
        times.append(time.perf_counter() - start)
    return float(np.median(times)) * 1e6


def relative(update, encode):
    return f"{update:.0f} µs | {update - encode:+.0f} µs ({100 * (update - encode) / encode:+.0f}%)"


def rows(name, x, reps):
    """Markdown rows for one signal of 60,000 samples."""
    t = START + np.arange(60_000) * np.timedelta64(1, "ms")  # 1 kHz: one-second blocks of 1000
    unit = fluxcode.encode_unit(x).unit
    timed = fluxcode.encode_unit(x, times=t).unit
    block = x[17_000:18_000][::-1].copy()
    encode = median_us(lambda: fluxcode.encode_unit(x), reps)
    encode_t = median_us(lambda: fluxcode.encode_unit(x, times=t), reps)
    one = median_us(lambda: fluxcode.update(unit, {17: block}), reps)
    one_t = median_us(lambda: fluxcode.update(timed, {17: block}, times={17: t[17_000:18_000]}), reps)
    append_t = median_us(lambda: fluxcode.update(timed, {60: block}, times={60: t[-1] + 1 + np.arange(1000)}), reps)
    tb = encode_seconds(x, t)
    lo, hi = START + np.timedelta64(30_500, "ms"), START + np.timedelta64(32_500, "ms")
    inside = (t >= lo) & (t < hi)
    encode_tb = median_us(lambda: encode_seconds(x, t), reps)
    straddle = median_us(
        lambda: update_seconds(tb, x[inside][::-1], t[inside], (lo, hi)), reps)
    # An hour at 10 Hz in one-second blocks: 3,600 blocks of 10 samples
    th = START + np.arange(36_000) * np.timedelta64(100, "ms")
    xh = np.resize(x, 36_000)
    hour = encode_seconds(xh, th)
    lo_h, hi_h = START + np.timedelta64(1_800_500, "ms"), START + np.timedelta64(1_802_500, "ms")
    inside_h = (th >= lo_h) & (th < hi_h)
    encode_h = median_us(lambda: encode_seconds(xh, th), max(reps // 3, 10))
    straddle_h = median_us(
        lambda: update_seconds(hour, xh[inside_h][::-1], th[inside_h], (lo_h, hi_h)), max(reps // 3, 10))
    # Upserts: existing times replaced, so the unit's shape stays the same
    one_t_h = median_us(lambda: update_seconds(hour, xh[5_000:5_001] + 1, th[5_000:5_001]), max(reps // 3, 10))
    every = np.arange(0, 36_000, 10)  # one sample from each block
    every_h = median_us(lambda: update_seconds(hour, xh[every] + 1, th[every]), max(reps // 3, 10))
    return [
        f"| {name} | `update`, 1 of 60 blocks | {encode:.0f} µs | {relative(one, encode)} |",
        f"| {name} | `update` with times, 1 of 60 | {encode_t:.0f} µs | {relative(one_t, encode_t)} |",
        f"| {name} | `update` with times, append 1 | {encode_t:.0f} µs | {relative(append_t, encode_t)} |",
        (f"| {name} | `update_time_blocks`, 2 s straddling 3 of 60 blocks | {encode_tb:.0f} µs | "
         f"{relative(straddle, encode_tb)} |"),
        (f"| {name} | `update_time_blocks`, 2 s straddling 3 of 3,600 blocks | {encode_h:.0f} µs | "
         f"{relative(straddle_h, encode_h)} |"),
        (f"| {name} | `update_time_blocks` upsert, 1 sample in 1 of 3,600 blocks | {encode_h:.0f} µs | "
         f"{relative(one_t_h, encode_h)} |"),
        (f"| {name} | `update_time_blocks` upsert, 1 sample in each of 3,600 blocks | {encode_h:.0f} µs | "
         f"{relative(every_h, encode_h)} |"),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=300)
    args = ap.parse_args()
    print(f"{platform.machine()} {platform.system()} {platform.release()}, Python {platform.python_version()}, "
          f"numpy {np.__version__}; median of {args.reps} calls (100 for the 3,600-block unit)\n")
    print("| signal | operation | encode from scratch | update | vs encode |")
    print("|---|---|---|---|---|")
    for name, x in [("random-walk", minute("random-walk", 1)), ("noisy-sine", minute("noisy-sine", 1)),
                    ("sensor-0.1", discrete_minute("sensor-0.1", 1))]:
        for row in rows(name, x, args.reps):
            print(row)


if __name__ == "__main__":
    main()
