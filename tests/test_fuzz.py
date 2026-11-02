# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Randomized block groups of 1 to 2,500 samples (one to three blocks) across the double range: no
crash, the error bound (ENCODER §7.1), each block encoded as it would be alone and, without a
target, the fixed point (ENCODER §7.3).

Each case draws from its own generator, seeded by (seed, case), so a failure reproduces
from those two numbers alone. The test runs a small budget; for a long run:

    FLUXCODE_RUST=0 uv run python tests/test_fuzz.py --seeds 0-39 --cases 1500

which prints the failures by kind and params, with the (seed, case) of the first few, and how
often the soft target moves decoded data on re-encode.
"""

import argparse
import collections
import sys
from pathlib import Path

if __name__ == "__main__":
    # A cache keyed on every source, as for the benchmarks (pytest's conftest uses a throwaway one)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
    import _fresh_numba_cache  # noqa: F401

import numpy as np
import pytest
from _series import group_rows

import fluxcode
from fluxcode import Params

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


def check_case(x: np.ndarray, params: Params) -> str | None:
    """The first failure of a block group ("crash", "non-finite", "bound", "not alone", "fixed point",
    "fixed point bytes"), or None."""
    try:
        with np.errstate(all="ignore"):
            first = fluxcode.encode_group(x, params, block_len=BLOCK_LEN).group
            y = fluxcode.decode_group(first).values
            finite = np.isfinite(x)
            if not np.array_equal(np.isfinite(y), finite) or not np.array_equal(y[~finite], x[~finite], equal_nan=True):
                return "non-finite"
            if finite.any():
                lo, hi = x[finite].min(), x[finite].max()
                ulps = 4 * np.spacing(np.abs(x[finite]).max())
                if np.abs(y[finite] - x[finite]).max() > (hi - lo) / (2**params.min_quantize_bits - 0.5) + ulps:
                    return "bound"
            if params.target_bits_per_sample is None:  # the target caps the whole group
                if not encodes_alone(x, first, params):
                    return "not alone"
                second = fluxcode.encode_group(y, params).group
                if not np.array_equal(fluxcode.decode_group(second).values, y, equal_nan=True):
                    return "fixed point"
                if fluxcode.encode_group(fluxcode.decode_group(second).values, params).group != second:
                    return "fixed point bytes"
    except Exception:  # noqa: BLE001 (any exception is the finding)
        return "crash"
    return None


def encodes_alone(x: np.ndarray, group: bytes, params: Params) -> bool:
    """Whether each block of a group has the grid, anchor and residuals it gets encoded alone."""
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


@pytest.mark.parametrize("seed", range(4))
def test_fuzz(seed):
    failures = []
    for case in range(500):
        x, params_idx = make_case(seed, case)
        what = check_case(x, PARAMS[params_idx])
        if what:
            failures.append((what, seed, case, params_idx, x.size))
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seeds", default="0-9", help="a range like 0-9, or a single seed")
    ap.add_argument("--cases", type=int, default=1500, help="cases per seed")
    a = ap.parse_args()
    first, _, last = a.seeds.partition("-")
    failures, examples = collections.Counter(), collections.defaultdict(list)
    for seed in range(int(first), int(last or first) + 1):
        for case in range(a.cases):
            x, params_idx = make_case(seed, case)
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
