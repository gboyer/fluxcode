# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Params.time_error: each block's times rounded once, from the values given, to the coarsest 1-2-5
step (or lattice) for which at least 95% of them move by at most time_error times their own interval
and none more than time_error mean intervals; in order, onto the clock's own grid, within each time
block's range; and whole blocks re-encoded by replace_time_blocks."""

from itertools import pairwise

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


def assert_keeps_to_the_time_error(given, decoded, block_sizes, time_error):
    """In every block of at least 3 times: none moved more than e mean intervals (2% more), and at most
    one in 20 more than e times its own interval (the smaller of the two either side)."""
    offsets = np.concatenate([[0], np.cumsum(block_sizes)])
    for first, end in zip(offsets[:-1].tolist(), offsets[1:].tolist()):
        raw, got = given[first:end], decoded[first:end]
        if raw.size < 3:
            np.testing.assert_array_equal(got, raw)
            continue
        moves = np.abs(got - raw)
        mean = (raw[-1] - raw[0]) / (raw.size - 1)
        assert moves.max() <= time_error * _time.TIME_ERROR_SLACK * mean + 1e-6, (first, moves.max(), mean)
        gaps = np.diff(raw).astype(np.float64)
        own = np.minimum(np.append(np.inf, gaps), np.append(gaps, np.inf))
        assert np.count_nonzero(moves > time_error * own) <= raw.size // 20, first


@pytest.mark.parametrize("time_error", [-0.01, 0.51, float("nan"), float("inf"), True, "0.1"])
def test_invalid_time_error_is_rejected(time_error):
    with pytest.raises(ValueError, match="time_error"):
        Params(time_error=time_error)


@pytest.mark.parametrize("interval,time_error,quantum", [
    (MS, 0.05, 100_000),
    (MS, 0.1, 200_000),
    (999_990, 0.1, 200_000),  # the slack: a slightly fast clock keeps the round step
    (16_666_667, 0.02, 500_000),
    (MS, 0.5, MS),
    (10, 0.01, 1),
    (1e18, 0.5, 1_000_000_000_000_000_000),
])
def test_time_quantum(interval, time_error, quantum):
    assert _time.time_quantum(float(interval), time_error) == quantum


@pytest.mark.parametrize("jitter", [0.005, 0.02, 0.1])
@pytest.mark.parametrize("time_error", [0.01, 0.05, 0.2])
@pytest.mark.parametrize("period", [MS, 16_666_667])
def test_times_move_by_at_most_the_time_error(jitter, time_error, period):
    ticks = jittered(5000, period, jitter, seed=int(jitter * 1000))
    group, decoded = encoded_ticks(np.zeros(ticks.size), ticks, time_error)
    assert (np.diff(decoded) >= 0).all()
    assert_keeps_to_the_time_error(ticks, decoded, fluxcode.decode_group(group).block_sizes, time_error)


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
    """A 1 kHz block (quantum 100 us at 0.05) whose last time rounds up past the start of a regular
    10 kHz block, given 1 us later. One late tick doesn't put the 1 kHz block on a lattice of its own:
    its period ignores the late interval."""
    slow = START + np.arange(200, dtype=np.int64) * MS
    slow[-1] += 60_000  # rounds up to +100 us
    fast = slow[-1] + 1000 + np.arange(200, dtype=np.int64) * 100_000
    return np.concatenate([slow, fast]), np.array([200, 200])


def test_blocks_with_different_quanta_stay_in_order():
    ticks, sizes = rate_change()
    params = Params(noise_floor_sigma=0, time_error=0.05)
    group = fluxcode.encode_blocks(np.zeros(ticks.size), sizes, params, times=ticks.view("datetime64[ns]")).group
    decoded = fluxcode.decode_group(group).times.view(np.int64)
    assert (np.diff(decoded) >= 0).all()
    # The fast block is regular, so it keeps its times; the slower block's last is lowered to it
    np.testing.assert_array_equal(decoded[200:], ticks[200:])
    assert decoded[199] == ticks[200] and np.abs(decoded - ticks)[:200].max() <= 50_000
    np.testing.assert_array_equal(decoded[:199], ticks[:199])


def test_bulk_encode_keeps_block_groups_in_order():
    ticks, _ = rate_change()
    params = Params(noise_floor_sigma=0, time_error=0.05)
    groups = fluxcode.encode(np.zeros(ticks.size), params, block_len=200, blocks_per_group=1,
                             times=ticks.view("datetime64[ns]")).groups
    decoded = np.concatenate([fluxcode.decode_group(g).times.view(np.int64) for g in groups])
    assert (np.diff(decoded) >= 0).all() and np.abs(decoded - ticks).max() <= 50_000


def test_update_resending_a_block_is_clamped_to_its_stored_neighbour():
    ticks, sizes = rate_change()
    params = Params(noise_floor_sigma=0, time_error=0.05)
    values = np.arange(ticks.size, dtype=np.float64)
    group = fluxcode.encode_blocks(values, sizes, params, times=ticks.view("datetime64[ns]")).group
    fresh = fluxcode.decode_group(group)
    updated = fluxcode.update(group, {1: values[200:]}, params, times={1: ticks[200:].view("datetime64[ns]")}).group
    decoded = fluxcode.decode_group(updated)
    np.testing.assert_array_equal(decoded.times, fresh.times)
    np.testing.assert_array_equal(decoded.values, fresh.values)
    # And the other way: block 0 again, ending above the stored block 1's start
    updated = fluxcode.update(group, {0: values[:200]}, params, times={0: ticks[:200].view("datetime64[ns]")}).group
    decoded = fluxcode.decode_group(updated).times.view(np.int64)
    assert (np.diff(decoded) >= 0).all() and np.abs(decoded - ticks).max() <= 50_000


def test_update_still_rejects_a_real_overlap():
    ticks, sizes = rate_change()
    params = Params(noise_floor_sigma=0, time_error=0.05)
    group = fluxcode.encode_blocks(np.zeros(ticks.size), sizes, params, times=ticks.view("datetime64[ns]")).group
    earlier = ticks[200:] - 200_000  # 200 us before block 0's last time: more than either block's move
    with pytest.raises(ValueError, match="non-decreasing"):
        fluxcode.update(group, {1: np.zeros(200)}, params, times={1: earlier.view("datetime64[ns]")})


SECOND = 1000 * MS


def at_or_after(ticks, start=START):
    """Jittered ticks moved to start where they fall before it."""
    return np.maximum(ticks, start)


def encode_seconds(values, ticks, time_error=0.05, start=START):
    return fluxcode.encode_time_blocks(values, ticks.view("datetime64[ns]"), Params(noise_floor_sigma=0, time_error=time_error),
                                       start_time=np.datetime64(start, "ns"), block_duration=np.timedelta64(SECOND, "ns"))


def replace_seconds(group, blocks, values, ticks, time_error=0.05, start=START):
    return fluxcode.replace_time_blocks(group, blocks, values, ticks.view("datetime64[ns]"),
                                        Params(noise_floor_sigma=0, time_error=time_error),
                                        start_time=np.datetime64(start, "ns"),
                                        block_duration=np.timedelta64(SECOND, "ns"))


def check_time_blocks(group, start=START):
    """Every block's times lie in its second, its end included (a time rounded up to it)."""
    decoded = fluxcode.decode_group(group)
    ticks = decoded.times.view(np.int64)
    lows = start + SECOND * np.repeat(np.arange(decoded.block_sizes.size), decoded.block_sizes)
    assert ((lows <= ticks) & (ticks <= lows + SECOND)).all()
    return decoded, ticks


def test_bulk_encode_rejects_a_decrease_between_block_groups_it_doesnt_round():
    """The order check between block groups runs whether or not any block is rounded."""
    ticks = START + np.arange(40, dtype=np.int64) * SECOND  # on a 1 s grid: nothing to round
    ticks[20:] -= 21 * SECOND  # the second block group starts before the first ends
    with pytest.raises(ValueError, match="non-decreasing"):
        fluxcode.encode(np.zeros(40), Params(time_error=0.05), block_len=20, blocks_per_group=1,
                        times=ticks.view("datetime64[ns]"))


def slow_then_fast():
    """A 1 Hz block on its grid but for its last scan, 4 ms early (it rounds up to the second),
    then a 1 kHz block starting 1 ms after that scan."""
    slow = START + np.arange(20, dtype=np.int64) * SECOND
    slow[-1] -= 4 * MS
    fast = slow[-1] + MS + np.arange(1000, dtype=np.int64) * MS
    return np.concatenate([slow, fast]), np.array([20, 1000])


def test_the_block_with_the_larger_quantum_gives_way():
    """Each time stays within half its own block's quantum: the slow block's last scan is lowered
    to the fast block's first time, rather than the fast block's first scans raised to it."""
    ticks, sizes = slow_then_fast()
    params = Params(noise_floor_sigma=0, time_error=0.05)
    group = fluxcode.encode_blocks(np.zeros(ticks.size), sizes, params, times=ticks.view("datetime64[ns]")).group
    decoded = fluxcode.decode_group(group).times.view(np.int64)
    assert (np.diff(decoded) >= 0).all()
    np.testing.assert_array_equal(decoded[20:], ticks[20:])  # the fast block is on its 100 us grid already
    assert decoded[19] == ticks[20] and abs(decoded[19] - ticks[19]) <= 50 * MS


def test_update_resends_a_fast_block_after_a_regular_stored_one():
    ticks, sizes = slow_then_fast()
    params = Params(noise_floor_sigma=0, time_error=0.05)
    values = np.arange(ticks.size, dtype=np.float64)
    group = fluxcode.encode_blocks(values, sizes, params, times=ticks.view("datetime64[ns]")).group
    for block, part in [(1, np.s_[20:]), (0, np.s_[:20])]:
        updated = fluxcode.update(group, {block: values[part]}, params, times={block: ticks[part].view("datetime64[ns]")})
        assert updated.group == group


def test_time_blocks_round_whatever_the_start():
    start = START + 123_457  # not a multiple of any quantum
    ticks = at_or_after(jittered(5000, jitter=0.01, seed=9, start=START), start)
    group = encode_seconds(np.zeros(ticks.size), ticks, start=start).group
    _, decoded = check_time_blocks(group, start)
    # On the 100 us grid, but for a time that would round below the start (raised to it) or past
    # its block's end (lowered to it): neither is on the grid
    at_edge = (decoded - start) % SECOND == 0
    assert decoded[0] == start and (decoded[~at_edge] % 100_000 == 0).all()
    assert np.abs(decoded - ticks).max() <= 51_000


@pytest.mark.parametrize("phase", [50_000, 25_000, 123_000])
def test_a_free_running_clock_rounds_onto_its_own_phase(phase):
    """Off the epoch's grid, at a candidate phase (50 us of q = 100 us) or between two (25 us):
    every block regular, all at one phase, so the block starts stay evenly spaced."""
    ticks = jittered(60_000, jitter=0.01, start=START + phase)
    group, decoded = encoded_ticks(np.zeros(ticks.size), ticks, 0.05)
    assert irregular_blocks(group) <= 1
    assert len(set((decoded % 100_000).tolist())) == 1
    assert np.abs(decoded - ticks).max() <= 51_000
    exact, _ = encoded_ticks(np.zeros(ticks.size), ticks, 0)
    assert len(group) < len(exact) / 100


def test_a_drifting_jittered_clock_follows_its_phase():
    rng = np.random.default_rng(12)
    index = np.arange(60_000, dtype=np.int64)
    ticks = np.maximum.accumulate(START + 37_000 + np.round(index * 0.99998731 * MS).astype(np.int64)
                                  + np.round(rng.normal(0, 10_000, index.size)).astype(np.int64))
    group, decoded = encoded_ticks(np.zeros(ticks.size), ticks, 0.05)
    assert np.abs(decoded - ticks).max() <= 51_000 and irregular_blocks(group) <= 5


def non_decimal_clock(num_samples, hz, jitter, seed=0):
    """ns ticks of a clock of hz (a period no 1-2-5 step divides) with Gaussian jitter (share of period)."""
    period = 1e9 / hz
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, jitter * period, num_samples) if jitter else 0
    return np.maximum.accumulate(START + 777_777 + np.round(np.arange(num_samples) * period + noise).astype(np.int64))


@pytest.mark.parametrize("hz,jitter,time_error", [(1024, 0, 0.05), (1024, 0.005, 0.05), (60, 0.005, 0.1),
                                                  (30, 0.005, 0.1), (1024, 0.005, 0.1)])
def test_a_non_decimal_clock_goes_onto_its_own_lattice(hz, jitter, time_error):
    """No 1-2-5 grid divides the period, so on one the intervals alternate; each block fits a lattice
    of its own (period and origin) instead, and stores as regular."""
    ticks = non_decimal_clock(30_000, hz, jitter)
    group, decoded = encoded_ticks(np.zeros(ticks.size), ticks, time_error)
    assert irregular_blocks(group) == 0
    assert (np.diff(decoded) >= 0).all()
    assert_keeps_to_the_time_error(ticks, decoded, fluxcode.decode_group(group).block_sizes, time_error)
    if jitter:
        exact, _ = encoded_ticks(np.zeros(ticks.size), ticks, 0)
        assert len(group) < len(exact) / 10


def test_fit_lattice_puts_a_clock_on_its_own_period():
    ticks = non_decimal_clock(1000, 1024, 0)
    max_move = 0.05 * 1e9 / 1024
    out = np.empty_like(ticks)
    assert _time.fit_lattice(ticks, 1e9 / 1024, np.int64(50_000), max_move, out)
    assert (np.diff(out) == np.diff(out)[0]).all() and _time.within_error(ticks, out, 0.05, max_move) >= 0
    # A tick further than the largest move fails the check
    late = ticks.copy()
    late[500] += int(2 * max_move)
    assert _time.fit_lattice(late, 1e9 / 1024, np.int64(50_000), max_move, out)
    assert _time.within_error(late, out, 0.05, max_move) == -1


def test_a_regular_block_keeps_its_times():
    ticks = START + 12_345 + np.arange(5000, dtype=np.int64) * MS  # regular, off every grid
    assert encoded_ticks(np.zeros(ticks.size), ticks, 0.1)[0] == encoded_ticks(np.zeros(ticks.size), ticks, 0)[0]




def test_update_keeps_the_stored_phase():
    """Re-sending a block of a clock between two candidate phases gives the block group back."""
    ticks = jittered(10_000, jitter=0.01, seed=13, start=START + 25_000)
    params = Params(noise_floor_sigma=0, time_error=0.05)
    values = np.arange(ticks.size, dtype=np.float64)
    group = fluxcode.encode_group(values, params, times=ticks.view("datetime64[ns]")).group
    for block in (1, 5, 9):
        part = np.s_[block * 1000:(block + 1) * 1000]
        updated = fluxcode.update(group, {block: values[part]}, params, times={block: ticks[part].view("datetime64[ns]")})
        assert updated.group == group


# Time blocks: a time stays in its block, at the block's end if it rounds up to (or past) it

def test_a_time_rounding_up_to_its_block_end_stays_in_its_block():
    ticks = at_or_after(jittered(10_000, jitter=0.01, seed=4))
    group = encode_seconds(np.arange(ticks.size, dtype=np.float64), ticks).group
    decoded, decoded_ticks = check_time_blocks(group)
    assert irregular_blocks(group) == 0
    # Blocks hold the samples given in their second; one given just before a second's start is
    # stored at that start, in the earlier block (its last time)
    np.testing.assert_array_equal(decoded.block_sizes, np.bincount((ticks - START) // SECOND))
    early = (ticks - START) % SECOND > SECOND - 50_000
    assert early.any()
    assert ((decoded_ticks[early] - START) % SECOND == 0).all()
    ends = np.cumsum(decoded.block_sizes) - 1
    assert np.isin(np.flatnonzero(early), ends).all()


def test_update_time_blocks_stores_new_times_exactly():
    ticks = at_or_after(jittered(3000, jitter=0.01, seed=8))
    lossy = Params(noise_floor_sigma=0, time_error=0.05)
    kwargs = {"start_time": np.datetime64(START, "ns"), "block_duration": np.timedelta64(SECOND, "ns")}
    group = encode_seconds(np.zeros(ticks.size), ticks).group
    _, stored = check_time_blocks(group)
    new = np.array([START + SECOND + 500 * MS + 512_345])
    with pytest.raises(ValueError, match="replace_time_blocks"):
        fluxcode.update_time_blocks(group, np.ones(1), new.view("datetime64[ns]"), lossy, **kwargs)
    # Exact params: the new time is kept as given, the stored ones stay
    updated = fluxcode.update_time_blocks(group, np.ones(1), new.view("datetime64[ns]"), Params(noise_floor_sigma=0), **kwargs)
    _, after = check_time_blocks(updated.group)
    np.testing.assert_array_equal(np.setdiff1d(after, stored), new)
    # A deletion alone takes any params
    deleted = fluxcode.update_time_blocks(group, [], np.zeros(0, "datetime64[ns]"), lossy,
                                          delete_ranges=(np.datetime64(START, "ns"), np.datetime64(START + SECOND, "ns")), **kwargs)
    assert deleted.indices.tolist() == [0] and fluxcode.decode_group(deleted.group).block_sizes[0] == 0


def test_replacing_blocks_with_their_source_data_gives_the_block_group_back():
    """Each replaced block continues the clock of the stored block before it, as in the encode."""
    ticks = at_or_after(jittered(10_000, jitter=0.01, seed=21, start=START + 25_000))
    values = np.arange(ticks.size, dtype=np.float64)
    group = encode_seconds(values, ticks).group
    blocks = (ticks - START) // SECOND
    for chosen in ([3], [0, 1], [2, 5, 9], range(10)):
        part = np.isin(blocks, list(chosen))
        assert replace_seconds(group, chosen, values[part], ticks[part]).group == group


def test_replace_time_blocks_replaces_only_the_declared_blocks():
    ticks = START + np.arange(5000, dtype=np.int64) * MS
    values = np.arange(ticks.size, dtype=np.float64)
    group = encode_seconds(values, ticks).group
    new_ticks = at_or_after(jittered(1000, jitter=0.01, seed=22, start=START + 2 * SECOND), START + 2 * SECOND)
    new_ticks = np.minimum(new_ticks, START + 3 * SECOND - 1)
    updated = replace_seconds(group, [2, 4], -np.ones(1000), new_ticks)
    assert updated.indices.tolist() == [2, 4]
    after, after_ticks = check_time_blocks(updated.group)
    assert after.block_sizes.tolist() == [1000, 1000, 1000, 1000, 0]  # block 4 got no samples: emptied
    assert (after.values[2000:3000] == -1).all() and (after.values[:2000] == values[:2000]).all()
    assert (after.values[3000:] == values[3000:4000]).all()
    np.testing.assert_array_equal(after_ticks[:2000], ticks[:2000])
    assert np.abs(after_ticks[2000:3000] - new_ticks).max() <= 51_000


def test_replace_time_blocks_appends_and_fills_a_gap():
    ticks = START + np.arange(2000, dtype=np.int64) * MS
    group = encode_seconds(np.zeros(ticks.size), ticks).group
    later = START + 4 * SECOND + np.arange(1000, dtype=np.int64) * MS
    updated = replace_seconds(group, [4], np.ones(1000), later)
    assert updated.indices.tolist() == [2, 3, 4]
    assert fluxcode.decode_group(updated.group).block_sizes.tolist() == [1000, 1000, 0, 0, 1000]


def test_replace_time_blocks_rejects_a_time_outside_the_declared_blocks():
    ticks = START + np.arange(3000, dtype=np.int64) * MS
    group = encode_seconds(np.zeros(ticks.size), ticks).group
    with pytest.raises(ValueError, match="block 2, which blocks doesn't declare"):
        replace_seconds(group, [1], np.zeros(1001), ticks[1000:2001])
    with pytest.raises(ValueError, match="doesn't declare"):
        replace_seconds(group, [], np.zeros(1), ticks[:1])
    with pytest.raises(ValueError, match="distinct"):
        replace_seconds(group, [1, 1], np.zeros(0), ticks[:0])
    with pytest.raises(ValueError, match="integers"):
        replace_seconds(group, [-1], np.zeros(0), ticks[:0])
    with pytest.raises(ValueError, match="non-decreasing"):
        replace_seconds(group, [1], np.zeros(2), ticks[[1001, 1000]])
    assert replace_seconds(group, [], np.zeros(0), ticks[:0]).group == group
    untimed = fluxcode.encode_group(np.zeros(10)).group
    with pytest.raises(ValueError, match="no time axis"):
        replace_seconds(untimed, [0], np.zeros(0), ticks[:0])


def test_replace_time_blocks_fills_an_empty_block_group():
    group = encode_seconds(np.zeros(0), np.zeros(0, np.int64)).group
    ticks = at_or_after(jittered(2500, jitter=0.01, seed=23))
    updated = replace_seconds(group, range(3), np.zeros(ticks.size), ticks)
    assert updated.group == encode_seconds(np.zeros(ticks.size), ticks).group


def test_a_minute_clock_streamed_by_replacing_the_open_block_equals_the_batch():
    """The source sends the open block whole, again and again as it fills, in us and in ns ticks."""
    rng = np.random.default_rng(0)
    n = 6000
    clock = (100 + np.arange(n) * 1000 + np.round(rng.normal(0, 5, n))).astype(np.int64)
    for unit, scale in [("us", 1), ("ns", 1000)]:
        kwargs = {"start_time": 0, "block_duration": 1_000_000 * scale, "time_unit": unit}
        ticks, values = clock * scale, np.arange(n, dtype=np.float64)
        batch = fluxcode.encode_time_blocks(values, ticks, E01, **kwargs).group
        group = fluxcode.encode_time_blocks(values[:0], ticks[:0], E01, **kwargs).group
        for end in range(250, n + 1, 250):
            block = int(ticks[end - 1] // (1_000_000 * scale))
            part = np.flatnonzero(ticks[:end] // (1_000_000 * scale) == block)
            group = fluxcode.replace_time_blocks(group, [block], values[part], ticks[part], E01, **kwargs).group
        assert group == batch and not irregular_blocks(group)


def test_a_block_continues_the_regular_clock_before_it():
    ticks = 1001 * np.arange(500)
    rng = np.random.default_rng(7)
    second = 1001 * (500 + np.arange(499)) + rng.integers(-20, 21, 499)
    kwargs = {"start_time": 0, "block_duration": 500_500, "time_unit": "us"}
    first_group = fluxcode.encode_time_blocks(np.zeros(500), ticks, E01, **kwargs).group
    both = np.concatenate([ticks, second])
    at_once = fluxcode.encode_time_blocks(np.zeros(both.size), both, E01, **kwargs).group
    replaced = fluxcode.replace_time_blocks(first_group, [1], np.zeros(499), second, E01, **kwargs).group
    assert at_once == replaced
    after = us_times(at_once)[500:]
    assert (np.diff(after) == 1001).all() and (after[0] - ticks[-1]) % 1001 == 0
    assert np.abs(after - second).max() <= 20


US = {"start_time": 0, "block_duration": 10**9, "time_unit": "us"}
E01 = Params(time_error=0.1, noise_floor_sigma=0)


def us_times(group):
    return fluxcode.decode_group(group).times.view(np.int64)


# Bugs of the earlier rounding (2026-10-12 review)

def us_group(blocks, time_error=0.05):
    ticks = np.concatenate(blocks).astype(np.int64)
    params = Params(noise_floor_sigma=0, time_error=time_error)
    group = fluxcode.encode_blocks(np.zeros(ticks.size), [len(b) for b in blocks], params,
                                   times=ticks.view("datetime64[us]")).group
    return ticks, us_times(group), group


def test_a_lattice_block_before_an_exact_one_is_put_in_order():
    """A block snapped onto the lattice before it ended 10 us past the next block's start: it was
    left out of order (every quantum 0), and the valid input rejected."""
    rng = np.random.default_rng(0)
    first = 1000 * np.arange(20)
    second = first[-1] + 1000 * np.arange(1, 21) + rng.integers(-15, 16, 20)
    second[-1] = first[-1] + 1000 * 20 - 20
    third = second[-1] + 10 + np.arange(20)  # exactly regular, 1 us
    ticks, decoded, _ = us_group([first, second, third])
    assert (np.diff(decoded) >= 0).all()
    np.testing.assert_array_equal(decoded[40:], third)  # the exact block keeps its times
    assert_keeps_to_the_time_error(ticks, decoded, [20, 20, 20], 0.05)


def test_an_exact_fast_block_after_a_lattice_block_keeps_its_times():
    rng = np.random.default_rng(1)
    first = 1000 * np.arange(20)
    second = first[-1] + 1000 * np.arange(1, 21) + rng.integers(-15, 16, 20)
    second[-1] = first[-1] + 1000 * 20 - 20
    third = second[-1] + 10 + 3 * np.arange(20)
    fourth = third[-1] + 1000 + 1000 * np.arange(20) + rng.integers(-15, 16, 20)
    ticks, decoded, _ = us_group([first, second, third, fourth])
    assert (np.diff(decoded) >= 0).all()
    np.testing.assert_array_equal(decoded[40:60], third)
    assert_keeps_to_the_time_error(ticks, decoded, [20] * 4, 0.05)


def test_a_burst_after_a_regular_block_is_not_put_on_its_lattice():
    """The burst's own interval is 1 us: the lattice of the 1 ms block before it isn't its clock."""
    first = 1000 * np.arange(20)
    burst = first[-1] + 1003 + np.arange(20)
    ticks, decoded, _ = us_group([first, burst])
    np.testing.assert_array_equal(decoded, ticks)


def test_a_slow_block_after_a_fast_regular_one_still_rounds():
    """Every second is on the lattice of a 1 s block: snapping onto it moved nothing but kept the
    60 s block from its own grid."""
    rng = np.random.default_rng(2)
    fast = np.arange(20)
    slow = 19 + 60 * np.arange(1, 21) + rng.integers(-2, 3, 20)
    ticks = np.concatenate([fast, slow])
    params = Params(noise_floor_sigma=0, time_error=0.05)
    group = fluxcode.encode_blocks(np.zeros(40), [20, 20], params, times=ticks.view("datetime64[s]")).group
    decoded = fluxcode.decode_group(group).times.view(np.int64)
    alone = fluxcode.encode_blocks(np.zeros(20), [20], params, times=slow.view("datetime64[s]")).group
    np.testing.assert_array_equal(decoded[20:], fluxcode.decode_group(alone).times.view(np.int64))
    assert not np.array_equal(decoded[20:], slow)


def test_clamp_boundaries_never_moves_a_tick_further():
    rng = np.random.default_rng(3)
    for _ in range(200):
        sizes = rng.integers(1, 6, 8)
        offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        ticks = np.sort(rng.integers(0, 100, offsets[-1])).astype(np.int64)
        rounded = ticks + rng.integers(-20, 21, ticks.size)
        for first, end in pairwise(offsets):
            rounded[first:end] = np.sort(rounded[first:end])
        moves = np.array([np.abs(rounded[f:e] - ticks[f:e]).max() for f, e in pairwise(offsets)])
        before = np.abs(rounded - ticks)
        out = rounded.copy()
        _time.clamp_boundaries(ticks, offsets, moves, out)
        assert (np.diff(out) >= 0).all() and (np.abs(out - ticks) <= before).all()


def multi_rate_blocks(rng, block_len):
    """Blocks of different clocks one after another: regular, jittered, Poisson, bursts, 1024 Hz."""
    parts, t = [], 10**6
    for _ in range(int(rng.integers(2, 6))):
        kind = rng.choice(["regular", "jitter", "poisson", "burst", "1024"])
        d = int(rng.choice([1, 3, 7, 10, 100, 977, 1000, 1001, 16667, 60000]))
        if kind == "regular":
            gaps = np.full(block_len, d)
        elif kind == "jitter":
            gaps = np.maximum(0, np.round(rng.normal(d, rng.choice([0.002, 0.01, 0.05]) * d, block_len)))
        elif kind == "poisson":
            gaps = rng.exponential(d, block_len)
        elif kind == "burst":
            gaps = rng.integers(0, 3, block_len)
        else:
            gaps = np.maximum(0, np.diff(np.round(np.arange(block_len + 1) * 976.5625 + rng.normal(0, 3, block_len + 1))))
        part = t + int(rng.integers(0, 2 * d)) + np.cumsum(gaps.astype(np.int64))
        parts.append(part)
        t = int(part[-1])
    return np.concatenate(parts).astype(np.int64)


@pytest.mark.parametrize("seed", range(4))
def test_multi_rate_blocks_keep_to_the_time_error(seed):
    """Valid input is never rejected, the blocks stay in order, and every block keeps to the time error."""
    rng = np.random.default_rng(seed)
    for _ in range(100):
        time_error = float(rng.choice([0.01, 0.05, 0.1, 0.3]))
        block_len = int(rng.choice([16, 20, 50]))
        ticks = multi_rate_blocks(rng, block_len)
        params = Params(noise_floor_sigma=0, time_error=time_error)
        group = fluxcode.encode_group(np.zeros(ticks.size), params, block_len=block_len, times=ticks.view("datetime64[us]")).group
        decoded = us_times(group)
        assert (np.diff(decoded) >= 0).all()
        assert_keeps_to_the_time_error(ticks, decoded, fluxcode.decode_group(group).block_sizes, time_error)


@pytest.mark.parametrize("seed", range(3))
def test_replaced_time_blocks_keep_to_the_time_error(seed):
    """Blocks of different clocks replaced again and again from the source: every time keeps to the
    error of its source time, in its block."""
    rng = np.random.default_rng(seed)
    duration = 50_000
    kwargs = {"start_time": 0, "block_duration": duration, "time_unit": "us"}
    periods = rng.choice([1000, 1001, 977, 250, 3000, 100], 12)
    time_error = float(rng.choice([0.05, 0.1, 0.3]))
    params = Params(noise_floor_sigma=0, time_error=time_error)

    def source_block(block):
        period = int(periods[block])
        ticks = block * duration + np.arange(rng.integers(0, period), duration, period)
        ticks = ticks + np.round(rng.normal(0, 0.01 * period, ticks.size)).astype(np.int64)
        return np.maximum.accumulate(np.clip(ticks, block * duration, (block + 1) * duration - 1))

    source = {block: source_block(block) for block in range(6)}
    group = fluxcode.encode_time_blocks(np.zeros(0), np.zeros(0, np.int64), params, **kwargs).group
    for _ in range(25):
        chosen = sorted(set(rng.integers(0, 12, int(rng.integers(1, 4))).tolist()))
        for block in chosen:
            if block in source and rng.random() < 0.3:  # some samples dropped at the source
                source[block] = source[block][rng.random(source[block].size) < 0.8]
            else:
                source[block] = source_block(block)
        ticks = np.concatenate([source[block] for block in chosen])
        group = fluxcode.replace_time_blocks(group, chosen, ticks.astype(np.float64), ticks, params, **kwargs).group
    all_ticks = np.concatenate([source[block] for block in sorted(source)])
    decoded = fluxcode.decode_group(group)
    np.testing.assert_array_equal(decoded.values.astype(np.int64), all_ticks)
    sizes = decoded.block_sizes
    assert_keeps_to_the_time_error(all_ticks, decoded.times.view(np.int64), sizes, time_error)
    # A replaced block continues the stored block before it as it was then: replaced in order,
    # the blocks round as one encode does
    batch = fluxcode.encode_time_blocks(all_ticks.astype(np.float64), all_ticks, params, **kwargs).group
    in_order = fluxcode.replace_time_blocks(group, sorted(source), all_ticks.astype(np.float64), all_ticks, params, **kwargs)
    assert in_order.group == batch
