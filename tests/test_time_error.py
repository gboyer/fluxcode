# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Params.time_error: times rounded to a 1-2-5 quantum per block, within time_error of an interval,
in order, onto the clock's own grid, and through every encode and update path."""

import numpy as np
import pytest
from _series import group_rows

import fluxcode
from fluxcode import Params, _time
from fluxcode._format import BLOCK_FLAG_IRREGULAR_TIME

MS = 1_000_000  # ns
START = 1_790_000_000_000_000_000  # 2026-09-21, in ns: a multiple of every quantum below 1 s
OFF = Params(noise_floor_sigma=0)


def jittered(num_samples, period=MS, jitter=0.01, seed=0, start=START):
    """Non-decreasing ns ticks of a clock of the given period with Gaussian jitter (share of period)."""
    rng = np.random.default_rng(seed)
    ticks = start + np.arange(num_samples, dtype=np.int64) * period
    ticks += np.round(rng.normal(0, jitter * period, num_samples)).astype(np.int64)
    return np.maximum.accumulate(ticks)


def encoded_ticks(values, ticks, time_error, **kwargs):
    group = fluxcode.encode_group(values, Params(noise_floor_sigma=0, time_error=time_error),
                                  times=ticks.view("datetime64[ns]"), **kwargs).group
    return group, fluxcode.decode_group(group).times.view(np.int64)


def irregular_blocks(group):
    return int(np.count_nonzero(group_rows(group).block_flags & BLOCK_FLAG_IRREGULAR_TIME))


@pytest.mark.parametrize("time_error", [-0.01, 0.51, float("nan"), float("inf"), True, "0.1"])
def test_invalid_time_error_is_rejected(time_error):
    with pytest.raises(ValueError, match="time_error"):
        Params(time_error=time_error)


@pytest.mark.parametrize("interval,time_error,boundary,quantum", [
    (MS, 0.05, 0, 100_000),
    (MS, 0.1, 0, 200_000),
    (999_990, 0.1, 0, 200_000),  # the slack: a slightly fast clock keeps the round step
    (16_666_667, 0.02, 0, 500_000),
    (16_666_667, 0.02, 1_500_000, 500_000),
    (16_666_667, 0.02, 300_000, 100_000),  # must divide the boundaries
    (MS, 0.5, 0, MS),
    (10, 0.01, 0, 1),
    (1e18, 0.5, 0, 1_000_000_000_000_000_000),
])
def test_time_quantum(interval, time_error, boundary, quantum):
    assert _time.time_quantum(float(interval), time_error, boundary) == quantum


@pytest.mark.parametrize("jitter", [0.005, 0.02, 0.1])
@pytest.mark.parametrize("time_error", [0.01, 0.05, 0.2])
@pytest.mark.parametrize("period", [MS, 16_666_667])
def test_times_move_by_at_most_the_time_error(jitter, time_error, period):
    ticks = jittered(5000, period, jitter, seed=int(jitter * 1000))
    _, decoded = encoded_ticks(np.zeros(ticks.size), ticks, time_error)
    assert (np.diff(decoded) >= 0).all()
    # The interval estimate (the mean of one-scan intervals) is within a few tenths of a percent
    assert np.abs(decoded - ticks).max() <= time_error * _time.TIME_ERROR_SLACK * period * 1.01


def test_a_jittered_clock_rounds_onto_its_grid():
    """About 5 standard deviations of jitter within the time error: every block stores as regular."""
    ticks = jittered(60_000, jitter=0.01)
    group, decoded = encoded_ticks(np.zeros(ticks.size), ticks, 0.05)
    assert irregular_blocks(group) == 0
    assert (np.diff(decoded) == MS).all()
    exact, _ = encoded_ticks(np.zeros(ticks.size), ticks, 0)
    assert len(group) < len(exact) / 100


def test_a_regular_grid_is_unchanged():
    ticks = START + np.arange(5000, dtype=np.int64) * MS
    values = minute_values(5000)
    assert encoded_ticks(values, ticks, 0.1)[0] == encoded_ticks(values, ticks, 0)[0]


def minute_values(num_samples):
    return np.sin(np.arange(num_samples) / 50.0)


def test_rounding_counts_from_the_epoch_before_1970():
    ticks = jittered(3000, jitter=0.01, start=-START)
    _, decoded = encoded_ticks(np.zeros(ticks.size), ticks, 0.05)
    assert (decoded % 100_000 == 0).all() and np.abs(decoded - ticks).max() <= 51_000


def test_a_decrease_rounding_would_hide_is_rejected():
    ticks = START + np.arange(3000, dtype=np.int64) * MS
    ticks[1500] = ticks[1499] - 1
    with pytest.raises(ValueError, match="non-decreasing"):
        encoded_ticks(np.zeros(ticks.size), ticks, 0.05)
    with pytest.raises(ValueError, match="NaT"):
        encoded_ticks(np.zeros(3), np.array([np.iinfo(np.int64).min, 0, 1]), 0.05)


def rate_change():
    """A 1 kHz block (quantum 100 us at 0.05) whose last time rounds up past the start of a 10 kHz
    block (quantum 10 us), given 1 us later."""
    slow = START + np.arange(20, dtype=np.int64) * MS
    slow[-1] += 60_000  # rounds up to +100 us
    fast = slow[-1] + 1000 + np.arange(20, dtype=np.int64) * 100_000  # rounds down to +60 us
    return np.concatenate([slow, fast]), np.array([20, 20])


def test_blocks_with_different_quanta_stay_in_order():
    ticks, sizes = rate_change()
    params = Params(noise_floor_sigma=0, time_error=0.05)
    group = fluxcode.encode_blocks(np.zeros(ticks.size), sizes, params, times=ticks.view("datetime64[ns]")).group
    decoded = fluxcode.decode_group(group).times.view(np.int64)
    assert (np.diff(decoded) >= 0).all()
    assert decoded[20] == decoded[19] == START + 19 * MS + 100_000  # raised to the previous block's last
    assert np.abs(decoded - ticks).max() <= 50_000  # the larger quantum's half


def test_bulk_encode_keeps_block_groups_in_order():
    ticks, _ = rate_change()
    params = Params(noise_floor_sigma=0, time_error=0.05)
    groups = fluxcode.encode(np.zeros(ticks.size), params, block_len=20, blocks_per_group=1,
                             times=ticks.view("datetime64[ns]")).groups
    decoded = np.concatenate([fluxcode.decode_group(g).times.view(np.int64) for g in groups])
    assert (np.diff(decoded) >= 0).all() and np.abs(decoded - ticks).max() <= 50_000


def test_update_resending_a_block_is_clamped_to_its_stored_neighbour():
    ticks, sizes = rate_change()
    params = Params(noise_floor_sigma=0, time_error=0.05)
    values = np.arange(ticks.size, dtype=np.float64)
    group = fluxcode.encode_blocks(values, sizes, params, times=ticks.view("datetime64[ns]")).group
    fresh = fluxcode.decode_group(group)
    updated = fluxcode.update(group, {1: values[20:]}, params, times={1: ticks[20:].view("datetime64[ns]")}).group
    decoded = fluxcode.decode_group(updated)
    np.testing.assert_array_equal(decoded.times, fresh.times)
    np.testing.assert_array_equal(decoded.values, fresh.values)
    # And the other way: block 0 again, ending above the stored block 1's start
    updated = fluxcode.update(group, {0: values[:20]}, params, times={0: ticks[:20].view("datetime64[ns]")}).group
    decoded = fluxcode.decode_group(updated).times.view(np.int64)
    assert (np.diff(decoded) >= 0).all() and np.abs(decoded - ticks).max() <= 50_000


def test_update_still_rejects_a_real_overlap():
    ticks, sizes = rate_change()
    params = Params(noise_floor_sigma=0, time_error=0.05)
    group = fluxcode.encode_blocks(np.zeros(ticks.size), sizes, params, times=ticks.view("datetime64[ns]")).group
    earlier = ticks[20:] - 200_000  # 200 us before block 0's last time: more than either half quantum
    with pytest.raises(ValueError, match="non-decreasing"):
        fluxcode.update(group, {1: np.zeros(20)}, params, times={1: earlier.view("datetime64[ns]")})


SECOND = 1000 * MS


def at_or_after(ticks, start=START):
    """Jittered ticks moved to start where they fall before it."""
    return np.maximum(ticks, start)


def encode_seconds(values, ticks, time_error=0.05, start=START):
    return fluxcode.encode_time_blocks(values, ticks.view("datetime64[ns]"), Params(noise_floor_sigma=0, time_error=time_error),
                                       start_time=np.datetime64(start, "ns"), block_duration=np.timedelta64(SECOND, "ns"))


def update_seconds(group, values, ticks, time_error=0.05, start=START, **kwargs):
    return fluxcode.update_time_blocks(group, values, ticks.view("datetime64[ns]"),
                                       Params(noise_floor_sigma=0, time_error=time_error),
                                       start_time=np.datetime64(start, "ns"),
                                       block_duration=np.timedelta64(SECOND, "ns"), **kwargs)


def check_time_blocks(group, start=START):
    """Every block's times lie in its second."""
    decoded = fluxcode.decode_group(group)
    ticks = decoded.times.view(np.int64)
    blocks = np.repeat(np.arange(decoded.block_sizes.size), decoded.block_sizes)
    np.testing.assert_array_equal((ticks - start) // SECOND, blocks)
    return decoded, ticks


def test_time_blocks_round_onto_the_next_block():
    ticks = at_or_after(jittered(10_000, jitter=0.01, seed=4))
    group = encode_seconds(np.arange(ticks.size, dtype=np.float64), ticks).group
    decoded, decoded_ticks = check_time_blocks(group)
    assert irregular_blocks(group) == 0 and (decoded.block_sizes == 1000).all()
    # Samples given just before a second's start are stored at it, in the next block
    early = (ticks - START) % SECOND > SECOND - 50_000
    assert early.any()
    assert ((decoded_ticks[early] - START) % SECOND == 0).all()


def test_time_block_quanta_divide_the_boundaries():
    start = START + 30_000  # 30 us past a round second: quanta must divide 10 us
    ticks = at_or_after(jittered(5000, jitter=0.01, seed=5, start=start), start)
    group = encode_seconds(np.zeros(ticks.size), ticks, start=start).group
    _, decoded = check_time_blocks(group, start)
    assert (decoded % 10_000 == 0).all() and np.abs(decoded - ticks).max() <= 5000


def test_update_time_blocks_replaces_a_sample_at_its_rounded_time():
    ticks = at_or_after(jittered(5000, jitter=0.01, seed=6))
    values = np.arange(ticks.size, dtype=np.float64)
    group = encode_seconds(values, ticks).group
    before, before_ticks = check_time_blocks(group)
    # The same scans of block 2 stamped again with fresh jitter: they replace the stored ones
    resent = jittered(5000, jitter=0.01, seed=7)[2000:2010]
    updated, indices, *_ = update_seconds(group, -np.ones(10), resent)
    after, after_ticks = check_time_blocks(updated)
    assert indices.tolist() == [2]
    np.testing.assert_array_equal(after_ticks, before_ticks)
    assert (after.values[2000:2010] == -1).all()
    np.testing.assert_array_equal(np.delete(after.values, np.s_[2000:2010]), np.delete(before.values, np.s_[2000:2010]))


def test_update_time_blocks_moves_a_sample_onto_a_stored_next_block():
    """A new sample given just before block 3's start rounds onto it: it replaces block 3's first
    sample, block 3's other samples stay, and block 2 is left as it was."""
    ticks = START + np.arange(5000, dtype=np.int64) * MS
    values = np.arange(ticks.size, dtype=np.float64)
    group = encode_seconds(values, ticks).group
    updated, indices, *_ = update_seconds(group, np.array([-1.0]), np.array([START + 3 * SECOND - 3000]))
    after, after_ticks = check_time_blocks(updated)
    assert indices.tolist() == [3]
    np.testing.assert_array_equal(after_ticks, ticks)
    assert after.values[3000] == -1 and (np.delete(after.values, 3000) == np.delete(values, 3000)).all()


def test_update_time_blocks_keeps_stored_times():
    """Stored samples of a re-encoded block keep their times; only the new ones are rounded."""
    ticks = at_or_after(jittered(3000, jitter=0.01, seed=8))
    group = encode_seconds(np.zeros(ticks.size), ticks).group
    _, stored = check_time_blocks(group)
    new = np.array([START + SECOND + 500 * MS + 512_345])  # between two scans of block 1
    updated, indices, *_ = update_seconds(group, np.ones(1), new)
    _, after = check_time_blocks(updated)
    assert indices.tolist() == [1]
    np.testing.assert_array_equal(np.setdiff1d(after, stored), [START + SECOND + 500 * MS + 500_000])  # on 100 us
    assert np.isin(stored, after).all()
