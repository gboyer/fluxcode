# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""fluxcode.verify: every API call's result passes, and a result that breaks a guarantee is caught
by the check meant for it. The fuzz tests (test_fuzz.py) run it on random data."""

from typing import TypedDict

import numpy as np
import pytest

import fluxcode
from fluxcode import Params, TimeUnit, verify


class TimeBlocks(TypedDict):
    start_time: int
    block_duration: int
    time_unit: TimeUnit


HOUR: TimeBlocks = {"start_time": 0, "block_duration": 500_000_000, "time_unit": "ns"}
"""Time blocks of 500 ms, over the 5 s of ticks below."""


@pytest.fixture(scope="module")
def series():
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(size=5000))
    x[17] = np.nan
    x[4000] = -np.inf
    ticks = np.maximum.accumulate(np.arange(5000) * 1_000_000 + rng.integers(-20_000, 20_000, 5000)) + 100_000
    return x, ticks.astype(np.int64)


def checks(violations):
    return {violation.check for violation in violations}


def test_every_call_passes(series):
    x, t = series
    params = Params(time_error=0.05)
    result = fluxcode.encode_group(x)
    assert verify.encode_group(result, x) == []
    assert verify.fixed_point(result.group) == []
    result = fluxcode.encode_group(x, params, times=t, time_unit="ns")
    assert verify.encode_group(result, x, params, times=t, time_unit="ns") == []
    result = fluxcode.encode_blocks(x, [1000, 0, 7, 3993])
    assert verify.encode_blocks(result, x, [1000, 0, 7, 3993]) == []
    series_result = fluxcode.encode(x, block_len=100, blocks_per_group=7)
    assert verify.encode(series_result, x, block_len=100, blocks_per_group=7) == []
    timed = fluxcode.encode_time_blocks(x, t, params, **HOUR)
    assert verify.encode_time_blocks(timed, x, t, params, **HOUR) == []
    assert verify.fixed_point(timed.group, params) == []
    ranges = (int(t[100]), int(t[300]))
    upserted = fluxcode.update_time_blocks(timed.group, x[:10] + 1, t[:10] + 1, delete_ranges=ranges, **HOUR)
    assert verify.update_time_blocks(upserted, timed.group, x[:10] + 1, t[:10] + 1, delete_ranges=ranges, **HOUR) == []
    replaced = fluxcode.replace_time_blocks(upserted.group, [2, 3, 12], x[1000:1400], t[1000:1400], params, **HOUR)
    assert verify.replace_time_blocks(replaced, upserted.group, [2, 3, 12], x[1000:1400], t[1000:1400], params,
                                      **HOUR) == []
    group = fluxcode.encode_group(x).group
    blocks = {1: x[:300], 7: x[:5]}
    assert verify.update(fluxcode.update(group, blocks), group, blocks) == []


def test_wrong_statistics(series):
    x, _ = series
    result = fluxcode.encode_group(x)
    assert checks(verify.encode_group(result._replace(block_min=result.block_min - 1), x)) == {"stats"}
    assert checks(verify.encode_group(result._replace(block_mean=result.block_mean * (1 + 1e-9)), x)) == {"stats"}
    assert checks(verify.encode_group(result._replace(block_max=result.block_max[:2]), x)) == {"stats"}


def test_values_off_their_grid(series):
    x, _ = series
    moved = fluxcode.encode_group(x + 0.01)
    found = checks(verify.encode_group(moved._replace(block_min=moved.block_min - 0.01,
                                                      block_max=moved.block_max - 0.01,
                                                      block_mean=moved.block_mean - 0.01), x))
    assert found == {"step_bound"}  # 0.01 is within range / 63.5 (and the noise floor may coarsen these blocks)


def test_coarser_than_the_params(series):
    x, _ = series
    coarse = fluxcode.encode_group(x, Params(min_quantize_bits=1, max_quantize_bits=3))
    assert {"range_bound", "fine_bound"} <= checks(verify.encode_group(coarse, x, Params(noise_floor_sigma=0)))


def test_lost_non_finite_values(series):
    x, _ = series
    finite = np.nan_to_num(x, nan=0.0, neginf=0.0)
    result = fluxcode.encode_group(finite)
    assert "nonfinite" in checks(verify.encode_group(result, x))


def test_wrong_layout(series):
    x, _ = series
    result = fluxcode.encode_group(x, block_len=500)
    assert checks(verify.encode_group(result, x)) == {"layout"}
    assert checks(verify.encode_group(result._replace(group=b"junk"), x)) == {"decode"}


def test_times(series):
    x, t = series
    params = Params(time_error=0.05)
    shifted = fluxcode.encode_group(x, times=t + 1, time_unit="ns")
    assert checks(verify.encode_group(shifted, x, times=t, time_unit="ns")) == {"times"}
    # Rounded much further than time_error allows (to 0.5 of an interval)
    rough = fluxcode.encode_group(x, Params(time_error=0.5), times=t, time_unit="ns")
    assert checks(verify.encode_group(rough, x, Params(time_error=0.01), times=t, time_unit="ns")) == {"time_error"}
    # Time blocks a quarter as long: times decode outside the blocks they're checked against
    timed = fluxcode.encode_time_blocks(x, t, params, start_time=0, block_duration=125_000_000, time_unit="ns")
    found = verify.encode_time_blocks(timed, x, t, params, **HOUR)
    assert checks(found) == {"layout"}
    assert verify.encode_time_blocks(timed, x, t, params, start_time=0, block_duration=125_000_000,
                                     time_unit="ns") == []


def test_update_touching_other_blocks(series):
    x, _ = series
    group = fluxcode.encode_group(x).group
    blocks = {1: x[:300]}
    result = fluxcode.update(group, blocks)
    # Every block re-encoded coarser: the untouched ones changed
    decoded = fluxcode.decode_group(result.group)
    coarser = fluxcode.encode_blocks(decoded.values, decoded.block_sizes, Params(max_quantize_bits=8)).group
    assert "untouched" in checks(verify.update(result._replace(group=coarser), group, blocks))
    assert checks(verify.update(result._replace(indices=np.array([0, 1])), group, blocks)) == {"layout"}


def test_update_time_blocks_losing_a_sample(series):
    x, t = series
    timed = fluxcode.encode_time_blocks(x, t, **HOUR)
    new_values, new_ticks = x[:10] + 1, t[:10] + 1
    dropped = fluxcode.update_time_blocks(timed.group, new_values[:-1], new_ticks[:-1], **HOUR)
    assert "layout" in checks(verify.update_time_blocks(dropped, timed.group, new_values, new_ticks, **HOUR))


def test_kept_samples_moved(series):
    """A kept sample that moves on a grid holding its point is a step away: past the step bound."""
    x, t = series
    timed = fluxcode.encode_time_blocks(x, t, **HOUR)
    result = fluxcode.update_time_blocks(timed.group, x[:1] + 1, t[:1] + 1, **HOUR)
    decoded = fluxcode.decode_group(result.group)
    nudged = decoded.values.copy()
    nudged[500] += 0.01
    forged = fluxcode.encode_blocks(nudged, decoded.block_sizes, times=decoded.times)
    found = verify.update_time_blocks(result._replace(group=forged.group), timed.group, x[:1] + 1, t[:1] + 1, **HOUR)
    assert [(violation.check, violation.block) for violation in found] == [("step_bound", 0)]


def test_fixed_point_skips_a_target(series):
    x, _ = series
    params = Params(target_bits_per_sample=6.0)
    assert verify.fixed_point(fluxcode.encode_group(x, params).group, params) == []


def test_violation_reads(series):
    x, _ = series
    result = fluxcode.encode_group(x)
    (violation, *_) = verify.encode_group(result._replace(block_min=result.block_min - 1), x)
    assert str(violation).startswith("stats: block 0: block_min ")
