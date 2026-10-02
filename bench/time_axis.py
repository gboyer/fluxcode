# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Cost of the time axis: bytes and encode/decode time a unit's timestamps add, per timestamp
pattern, on one-minute units (60 blocks of 1000) of a 4.12 Hz sine. Single thread.

    uv run python bench/time_axis.py [--reps 20]
"""

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from _signals import MINUTE, clock_minute, minute

import fluxcode
from fluxcode import _format, _unit

START_NS = 1_790_000_000_000_000_000  # 2026-09-21, in ns since 1970
MS = 1_000_000


def patterns(rng):
    """(name, int64 ns ticks) of MINUTE samples each."""
    index = np.arange(MINUTE)
    gaps = np.ones(MINUTE, np.int64)
    gaps[rng.choice(MINUTE, 20, replace=False)] = rng.integers(2, 500, 20)
    deadband = np.where(rng.random(MINUTE) < 0.05, rng.integers(1, 60_000, MINUTE), rng.integers(1, 20, MINUTE))
    return [
        ("grid: regular 1 kHz", clock_minute("grid", 1)),
        ("grid + a few gaps (2 per minute)", clock_minute("grid+gaps", 1)),
        ("noisy clock (σ=20 µs, µs resolution)", clock_minute("noisy", 1)),
        ("1 kHz, 20 gaps", START_NS + np.cumsum(gaps) * MS),
        ("1 kHz, 1% dropped", START_NS + np.sort(rng.choice(MINUTE * 101 // 100, MINUTE, replace=False)) * MS),
        ("drifting clock (0.99998 ms)", START_NS + np.round(index * 0.99998731 * MS).astype(np.int64)),
        ("jitter σ=10 µs, ns resolution", START_NS + np.round((index + rng.normal(0, 0.01, MINUTE)) * MS).astype(np.int64)),
        ("jitter σ=10 µs, µs resolution", START_NS + (index * 1000 + np.round(rng.normal(0, 10, MINUTE))).astype(np.int64) * 1000),
        ("Poisson events, mean 1 ms, µs", START_NS + np.cumsum(np.round(rng.exponential(1000, MINUTE))).astype(np.int64) * 1000),
        ("Poisson events, mean 1 ms, ns", START_NS + np.cumsum(np.round(rng.exponential(MS, MINUTE))).astype(np.int64)),
        ("deadband logging, ms grid", START_NS + np.cumsum(deadband) * MS),
        ("bursts 10 kHz / idle", START_NS + np.cumsum(np.where(index % 1000 < 900, 100_000, 5 * MS))),
    ]


def best(func, reps):
    fastest = np.inf
    for _ in range(reps):
        started = time.perf_counter()
        func()
        fastest = min(fastest, time.perf_counter() - started)
    return fastest * 1e6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=20)
    args = parser.parse_args()
    values = minute("sin-4.12hz", 1)
    plain_unit = fluxcode.encode_unit(values).unit
    plain_encode = best(lambda: fluxcode.encode_unit(values), args.reps)
    plain_decode = best(lambda: fluxcode.decode_unit(plain_unit), args.reps)
    print(f"values only: {len(plain_unit)} bytes, encode {plain_encode:.0f} µs, decode {plain_decode:.0f} µs per unit\n")
    print("| timestamps | irregular blocks | time bytes | time bits/sample | encode +µs | decode +µs |")
    print("|---|---|---|---|---|---|")
    for name, ticks in patterns(np.random.default_rng(0)):
        times = ticks.view("datetime64[ns]")
        unit = fluxcode.encode_unit(values, times=times).unit
        decoded = fluxcode.decode_unit(unit)
        assert decoded.times is not None and np.array_equal(decoded.times, times)
        # The values mustn't depend on the time axis (fails on a stale numba cache: see README)
        assert np.array_equal(decoded.values, fluxcode.decode_unit(plain_unit).values)
        parsed = _unit.decompress(unit)
        num_irregular = int(np.count_nonzero(parsed.block_flags & _format.BLOCK_FLAG_IRREGULAR_TIME))
        extra_bytes = len(unit) - len(plain_unit)
        encode_time = best(lambda times=times: fluxcode.encode_unit(values, times=times), args.reps) - plain_encode
        decode_time = best(lambda unit=unit: fluxcode.decode_unit(unit), args.reps) - plain_decode
        print(f"| {name} | {num_irregular}/60 | {extra_bytes} | {8 * extra_bytes / MINUTE:.3f} | "
              f"{encode_time:.0f} | {decode_time:.0f} |")


if __name__ == "__main__":
    main()
