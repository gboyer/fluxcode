# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Randomized block groups checked with fluxcode.verify: every guarantee of ENCODER §6 it checks,
and that each block encodes as it would alone.

Two fuzzers, each case drawing from its own generator seeded by (seed, case), so a failure
reproduces from those two numbers alone:

- values: 1 to 2,500 samples (one to three blocks) across the double range, through
  encode_group, with the fixed point (ENCODER §7.3) when there is no target;
- timed: the same values on clocks (regular, jittered, gappy, events, duplicate times) in every
  time unit, with a time error, through encode_group or encode_time_blocks, then one to three
  updates (update, update_time_blocks, replace_time_blocks), each checked against its input.

The tests run a small budget; for a long run:

    FLUXCODE_RUST=0 uv run python tests/test_fuzz.py --fuzzer values --seeds 0-39 --cases 1500
    FLUXCODE_RUST=0 uv run python tests/test_fuzz.py --fuzzer timed --seeds 0-39 --cases 500

which prints the failures by kind and params, with the (seed, case) of the first few (for values,
also how often the soft target moves decoded data on re-encode).
"""

import argparse
import collections
import sys
import traceback
from dataclasses import replace
from pathlib import Path
from typing import TypedDict, cast

if __name__ == "__main__":
    # A cache keyed on every source, as for the benchmarks (pytest's conftest uses a throwaway one)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
    import _fresh_numba_cache  # noqa: F401

import numpy as np
import pytest
from _series import group_rows

import fluxcode
from fluxcode import Params, TimeUnit, verify

PARAMS = [
    Params(),
    Params(noise_floor_sigma=0),
    Params(target_bits_per_sample=6.0),
    Params(min_quantize_bits=1, max_quantize_bits=4),
    Params(grids="binary"),
    Params(grids="decimal"),
]
"""Params drawn per case; with a target, the cap is soft and not checked for a fixed point (§7.3)."""

BLOCK_LEN = 1000
LENGTHS = [1, 2, 7, 9, 64, 300, 1000, 2500]
"""Short blocks (at most 8 samples skip the analysis), a few dozen, a full block and three blocks
(the last one short)."""


def make_case(seed: int, case: int) -> tuple[np.ndarray, int]:
    """One block's samples and the index of its params in PARAMS."""
    rng = np.random.default_rng([seed, case])
    n = int(rng.choice(LENGTHS))
    mag = 10.0 ** rng.uniform(-320, 307)
    k = np.cumsum(rng.integers(-50, 51, n)) + int(rng.integers(-(10**6), 10**6))
    num, den = int(rng.integers(1, 2000)), int(rng.integers(1, 2000))
    kind = int(rng.integers(0, 6))
    with np.errstate(all="ignore"):
        if kind == 0:  # noise at any magnitude
            x = rng.normal(size=n) * mag
        elif kind == 1:  # a scaled grid at any magnitude
            x = k * num / den * mag
        elif kind == 2:  # a rational step over a power of ten: decimal or scaled grids, or near misses
            x = (k * num) / (den * 10.0 ** int(rng.integers(0, 15)))
        elif kind == 3:  # a fine grid on a large offset
            x = rng.uniform(-1, 1) * mag + k * (num / den) * mag * 1e-6
        elif kind == 4:  # constant
            x = np.full(n, rng.normal() * mag)
        else:  # a power-of-two grid anywhere in the exponent range, subnormals included
            x = k * 2.0 ** int(rng.integers(-1074, 1000)) * num
        x = np.asarray(x, np.float64)
    if rng.random() < 0.2:
        x[rng.integers(0, n, max(1, n // 10))] = rng.choice([np.nan, np.inf, -np.inf])
    return x, int(rng.integers(0, len(PARAMS)))


def failure(violations: list[verify.Violation]) -> str | None:
    """The check of the first violation, or None."""
    return violations[0].check if violations else None


def check_case(x: np.ndarray, params: Params) -> str | None:
    """The first failure of a block group (a verify check, "crash" or "not alone"), or None."""
    try:
        with np.errstate(all="ignore"):
            result = fluxcode.encode_group(x, params, block_len=BLOCK_LEN)
            found = verify.encode_group(result, x, params, block_len=BLOCK_LEN) or verify.fixed_point(result.group, params)
            if found:
                return failure(found)
            if params.target_bits_per_sample is None and not encodes_alone(x, result.group, params):
                return "not alone"  # the target caps the whole group
    except Exception:  # noqa: BLE001 (any exception is the finding)
        return "crash"
    return None


def encodes_alone(x: np.ndarray, group: bytes, params: Params) -> bool:
    """Whether each block of a group has the grid, anchor and residuals it gets encoded alone.

    Today's encoder keeps no state between blocks (not a guarantee, ENCODER §6): this catches
    state leaking by accident, such as scratch read before it is written.
    """
    if x.size <= BLOCK_LEN:
        return True
    together = group_rows(group)
    for block, start in enumerate(range(0, x.size, BLOCK_LEN)):
        alone = group_rows(fluxcode.encode_group(x[start:start + BLOCK_LEN], params, block_len=BLOCK_LEN).group)
        for field in ("block_flags", "grid_params", "grid_scales", "value_anchors"):
            if getattr(alone, field)[0].tobytes() != getattr(together, field)[block].tobytes():
                return False
        if not np.array_equal(alone.residuals, together.residuals[start:start + alone.residuals.size]):
            return False
    return True


def target_fixed(x: np.ndarray, params: Params) -> bool:
    """Whether decoded data re-encode to themselves under a target (not guaranteed: §5.8)."""
    with np.errstate(all="ignore"):
        y = fluxcode.decode_group(fluxcode.encode_group(x, params).group).values
        return np.array_equal(fluxcode.decode_group(fluxcode.encode_group(y, params).group).values, y, equal_nan=True)


# Timed series and updates

TIME_ERRORS = [0.0, 0.0, 0.01, 0.05, 0.2, 0.5]
UNITS: list[TimeUnit] = ["s", "ms", "us", "ns"]
CLOCKS = ["regular", "jittered", "gaps", "events", "duplicates"]


class TimeBlocks(TypedDict):
    start_time: int
    block_duration: int
    time_unit: TimeUnit


def make_ticks(rng: np.random.Generator, n: int) -> tuple[np.ndarray, int]:
    """Non-decreasing ticks on a random clock, and the clock's period in ticks."""
    period = int(10 ** rng.uniform(0, 9))
    clock = CLOCKS[int(rng.integers(0, len(CLOCKS)))]
    if clock == "regular":
        steps = np.full(n, period)
    elif clock == "jittered":
        steps = np.full(n, period)
        offsets = np.round(rng.normal(0, rng.uniform(0.001, 0.05) * period, n)).astype(np.int64)
    elif clock == "gaps":
        steps = np.where(rng.random(n) < 0.02, period * rng.integers(2, 100, n), period)
    elif clock == "events":
        steps = np.round(rng.exponential(period, n)).astype(np.int64)
    else:
        steps = np.where(rng.random(n) < 0.3, 0, period)
    start = int(rng.integers(-(2**60), 2**60)) // max(period, 1)
    ticks = start + np.cumsum(steps.astype(np.int64)) - int(steps[0])
    if clock == "jittered":
        ticks = np.maximum.accumulate(ticks + offsets)
    return ticks.astype(np.int64), period


def check_timed_case(seed: int, case: int) -> tuple[str, str] | None:
    """The first failure of a timed case: (step, check), or None. Steps are "encode" and "update 1"..."""
    x, params_idx = make_case(seed, case)
    rng = np.random.default_rng([seed, case, 1])
    params = replace(PARAMS[params_idx], time_error=float(rng.choice(TIME_ERRORS)))
    ticks, period = make_ticks(rng, x.size)
    unit = UNITS[int(rng.integers(0, len(UNITS)))]
    step = "encode"
    try:
        with np.errstate(all="ignore"):
            if rng.random() < 0.5:
                block_len = int(rng.choice([1, 7, 64, BLOCK_LEN]))
                result = fluxcode.encode_group(x, params, block_len=block_len, times=ticks, time_unit=unit)
                found = verify.encode_group(result, x, params, block_len=block_len, times=ticks, time_unit=unit)
                if found:
                    return step, cast(str, failure(found))
                group = result.group
                for update_idx in range(int(rng.integers(1, 4))):
                    step = f"update {update_idx + 1}"
                    group, found = random_update(rng, group, params)
                    if found:
                        return step, cast(str, failure(found))
            else:
                span = int(ticks[-1]) - int(ticks[0])
                duration = max(period * int(rng.choice([1, 10, 100, 1000])), span // 60_000 + 1)
                hour: TimeBlocks = {"start_time": int(ticks[0]) - int(rng.integers(0, duration)),
                                    "block_duration": duration, "time_unit": unit}
                result = fluxcode.encode_time_blocks(x, ticks, params, **hour)
                found = verify.encode_time_blocks(result, x, ticks, params, **hour)
                if found:
                    return step, cast(str, failure(found))
                group = result.group
                for update_idx in range(int(rng.integers(1, 4))):
                    step = f"update {update_idx + 1}"
                    group, found = random_time_update(rng, group, x, ticks, params, hour)
                    if found:
                        return step, cast(str, failure(found))
    except Exception:  # noqa: BLE001 (any exception is the finding)
        return step, "crash: " + traceback.format_exc(limit=-1).strip().splitlines()[-1]
    return None


def random_update(rng: np.random.Generator, group: bytes, params: Params) -> tuple[bytes, list[verify.Violation]]:
    """update of a group with times: a block replaced by new values at its times, or one appended."""
    decoded = fluxcode.decode_group(group)
    assert decoded.times is not None
    sizes = decoded.block_sizes
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    nonempty = np.flatnonzero(sizes)
    ticks = decoded.times.view(np.int64)
    if rng.random() < 0.7 and nonempty.size:
        block = int(rng.choice(nonempty))
        block_ticks = ticks[offsets[block]:offsets[block + 1]]
        values = decoded.values[offsets[block]:offsets[block + 1]] * rng.uniform(0.5, 2) + rng.normal()
    else:
        block = sizes.size + int(rng.integers(0, 3))
        n = int(rng.integers(0, 20))
        last = int(ticks[-1]) if ticks.size else 0
        block_ticks = last + np.cumsum(rng.integers(0, 1000, n)).astype(np.int64)
        values = rng.normal(size=n) * 10.0 ** rng.uniform(-10, 10)
    blocks, times = {block: values}, {block: block_ticks.view(decoded.times.dtype)}
    result = fluxcode.update(group, blocks, params, times=times)
    return result.group, verify.update(result, group, blocks, params, times=times)


def random_time_update(
    rng: np.random.Generator, group: bytes, x: np.ndarray, ticks: np.ndarray, params: Params, hour: TimeBlocks,
) -> tuple[bytes, list[verify.Violation]]:
    """update_time_blocks (an upsert, a range deletion, or both) or replace_time_blocks from the source."""
    decoded = fluxcode.decode_group(group)
    assert decoded.times is not None
    stored = decoded.times.view(np.int64)
    start, duration = hour["start_time"], hour["block_duration"]
    if rng.random() < 0.6:
        exact = replace(params, time_error=0.0)  # update_time_blocks stores new times exactly
        n = int(rng.integers(0, 50))
        if stored.size and rng.random() < 0.5:  # replace stored times
            new_ticks = np.sort(rng.choice(stored, n))
        else:  # new times anywhere from the start to past the end
            last = int(stored[-1]) if stored.size else start
            new_ticks = np.sort(rng.integers(start, last + 3 * duration, n)).astype(np.int64)
        values = rng.normal(size=n) * (np.nanmax(np.abs(np.nan_to_num(x, posinf=0, neginf=0))) + 1)
        ranges = None
        if rng.random() < 0.5 and stored.size:
            a, b = np.sort(rng.choice(stored, 2))
            ranges = (int(a), int(b) + int(rng.integers(0, 2)))
            if rng.random() < 0.5:
                new_ticks, values = new_ticks[:0], values[:0]
        result = fluxcode.update_time_blocks(group, values, new_ticks, exact, delete_ranges=ranges, **hour)
        return result.group, verify.update_time_blocks(result, group, values, new_ticks, exact, delete_ranges=ranges,
                                                       **hour)
    # Whole blocks from the source data, perhaps scaled, perhaps one past the end
    ids = (ticks - start) // duration
    candidates = np.unique(ids)
    declared = sorted(set(rng.choice(candidates, int(rng.integers(1, 4))).tolist()))
    if rng.random() < 0.2:
        declared.append(int(decoded.block_sizes.size) + int(rng.integers(0, 3)))
    mask = np.isin(ids, declared)
    values = x[mask] * float(rng.choice([1.0, 1.0, 0.5, 3.0]))
    result = fluxcode.replace_time_blocks(group, declared, values, ticks[mask], params, **hour)
    return result.group, verify.replace_time_blocks(result, group, declared, values, ticks[mask], params, **hour)


@pytest.mark.parametrize("seed", range(4))
def test_fuzz(seed):
    failures = []
    for case in range(500):
        x, params_idx = make_case(seed, case)
        what = check_case(x, PARAMS[params_idx])
        if what:
            failures.append((what, seed, case, params_idx, x.size))
    assert not failures, failures[:5]


@pytest.mark.parametrize("seed", range(4))
def test_fuzz_timed(seed):
    failures = []
    for case in range(150):
        found = check_timed_case(seed, case)
        if found:
            failures.append((*found, seed, case))
    assert not failures, failures[:5]


# Cases the fuzz found, pinned: (seed, case) of make_case
REGRESSIONS = [
    (6, 119),  # a scaled grid on 2^-1016 / 512: D * 2^-e overflowed, and the anchor was written as -2^63
    (0, 600),  # 2 samples decoded from 2^-e, within a quarter step of a decimal grid: snapped again
    (1, 332),  # 7 samples (a short block), the same
    (8, 1189),  # three blocks at 4 bits: the same with few levels
]


@pytest.mark.parametrize("seed,case", REGRESSIONS)
def test_fuzz_regressions(seed, case):
    x, params_idx = make_case(seed, case)
    assert check_case(x, PARAMS[params_idx]) is None


# Timed cases the fuzz found, pinned: (seed, case) of check_timed_case
TIMED_REGRESSIONS = [
    pytest.param(28, 178, marks=pytest.mark.xfail(strict=True, reason=(
        "update_time_blocks: a stored time rounded up to its block's end (the next block's start) isn't "
        "replaced by a new sample at that time, which goes to the next block: both are kept"))),
]


@pytest.mark.parametrize("seed,case", TIMED_REGRESSIONS)
def test_fuzz_timed_regressions(seed, case):
    assert check_timed_case(seed, case) is None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fuzzer", choices=["values", "timed"], default="values")
    ap.add_argument("--seeds", default="0-9", help="a range like 0-9, or a single seed")
    ap.add_argument("--cases", type=int, default=1500, help="cases per seed")
    a = ap.parse_args()
    first, _, last = a.seeds.partition("-")
    failures, examples = collections.Counter(), collections.defaultdict(list)
    for seed in range(int(first), int(last or first) + 1):
        for case in range(a.cases):
            x, params_idx = make_case(seed, case)
            if a.fuzzer == "timed":
                found = check_timed_case(seed, case)
                what = None if found is None else f"{found[0]}: {found[1]}"
            else:
                params = PARAMS[params_idx]
                what = check_case(x, params)
                if what is None and params.target_bits_per_sample is not None and not target_fixed(x, params):
                    what = "target moved"  # the soft cap (ENCODER §5.8): counted, not a failure
            if what:
                failures[what, params_idx] += 1
                if len(examples[what, params_idx]) < 3:
                    examples[what, params_idx].append((seed, case, x.size))
    total = (int(last or first) - int(first) + 1) * a.cases
    moved = sum(count for (what, _), count in failures.items() if what == "target moved")
    print(f"{total} cases, {sum(failures.values()) - moved} failures, {moved} moved by the target")
    for (what, params_idx), count in sorted(failures.items()):
        print(f"  {what:18s} PARAMS[{params_idx}]: {count}, e.g. (seed, case, n) {examples[what, params_idx]}")


if __name__ == "__main__":
    main()
