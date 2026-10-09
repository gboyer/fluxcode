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
    # The fast block is regular, so it keeps its times; the slower block's last is lowered to it
    np.testing.assert_array_equal(decoded[20:], ticks[20:])
    assert decoded[19] == ticks[20] and np.abs(decoded - ticks)[:20].max() <= 50_000


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
    # On the 100 us grid, but for a time that would round below the start: raised to it
    assert decoded[0] == start and (decoded[1:] % 100_000 == 0).all()
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


def test_update_time_blocks_rounds_onto_the_stored_phase():
    ticks = at_or_after(jittered(3000, jitter=0.01, seed=14, start=START + 50_000))
    group = encode_seconds(np.zeros(ticks.size), ticks).group
    _, stored = check_time_blocks(group)
    phase = int(stored[1] % 100_000)
    new = np.array([START + SECOND + 500 * MS + 512_345])  # between two scans of block 1
    updated, *_ = update_seconds(group, np.ones(1), new)
    _, after = check_time_blocks(updated)
    added = np.setdiff1d(after, stored)
    assert added.size == 1 and added[0] % 100_000 == phase and abs(added[0] - new[0]) <= 50_000


def us_blocks(group_ticks, appended, time_error=0.1):
    """Encodes 1000 us ticks in 1 s blocks, then appends one tick; returns (group, update result)."""
    kwargs = {"start_time": 0, "block_duration": 10**6, "time_unit": "us"}
    params = Params(time_error=time_error)
    group = fluxcode.encode_time_blocks(np.zeros(group_ticks.size), group_ticks, params, **kwargs).group
    updated = fluxcode.update_time_blocks(group, [1.0], [appended], params, **kwargs)
    return group, updated[0]


def test_update_time_blocks_leaves_a_regular_block_as_it_is():
    """A new tick on the grid of a regular block (at a phase no 1-2-5 grid has) is not rounded."""
    ticks = np.arange(7, 7 + 1000 * 300, 1000)
    group, updated = us_blocks(ticks, ticks[-1] + 1000)
    assert not irregular_blocks(group)
    assert not irregular_blocks(updated)
    np.testing.assert_array_equal(fluxcode.decode_group(updated).times.view(np.int64)[-2:], ticks[-1] + [0, 1000])


def test_update_time_blocks_rounds_a_jittered_append_to_a_regular_round_block():
    ticks = np.arange(0, 1000 * 300, 1000)
    _, updated = us_blocks(ticks, ticks[-1] + 1003)
    assert fluxcode.decode_group(updated).times.view(np.int64)[-1] == ticks[-1] + 1000


# Every tick moves by at most the time error in all, however often its block is updated

US = {"start_time": 0, "block_duration": 10**9, "time_unit": "us"}
E01 = Params(time_error=0.1, noise_floor_sigma=0)


def us_encode(ticks, params=E01, **kwargs):
    return fluxcode.encode_time_blocks(np.zeros(ticks.size), ticks, params, **{**US, **kwargs}).group


def us_append(group, ticks, params=E01, **kwargs):
    return fluxcode.update_time_blocks(group, np.ones(len(ticks)), ticks, params, **{**US, **kwargs}).group


def us_times(group):
    return fluxcode.decode_group(group).times.view(np.int64)


def jittered_us(num, interval, seed, jitter=0.0):
    rng = np.random.default_rng(seed)
    return np.maximum.accumulate(
        np.arange(num) * interval + (rng.normal(0, jitter * interval, num) if jitter else 0)
    ).astype(np.int64)


def test_a_late_append_to_a_regular_block_goes_onto_its_lattice():
    ticks = 7 + 1000 * np.arange(300)
    group = us_encode(ticks)
    updated = us_append(group, [300_010])
    np.testing.assert_array_equal(us_times(updated), 7 + 1000 * np.arange(301))
    assert not irregular_blocks(updated)
    # on the lattice already: nothing moves
    assert us_times(us_append(group, [300_007]))[-1] == 300_007


def test_an_append_beyond_half_the_time_error_takes_a_grid_at_the_lattice_phase():
    """60 us late is past e / 2 * 1000 = 50: the stored ticks lie on 1000 us at phase 7, so the new one
    goes to 200 us at phase 7."""
    ticks = 7 + 1000 * np.arange(300)
    updated = us_append(us_encode(ticks), [300_067])
    after = us_times(updated)
    np.testing.assert_array_equal(after[:300], ticks)
    assert after[300] == 300_007 and abs(after[300] - 300_067) <= 100


def test_off_lattice_samples_in_a_1001_us_block_do_not_blow_it_up():
    """1001 us has no 1-2-5 grid: an append within e / 2 snaps to the lattice; a far one rounds the whole
    block, with a quantum of at most e x interval (100, not 200), so no tick moves more than e in all."""
    ticks = 5000 + 1001 * np.arange(300)
    group = us_encode(ticks)
    snapped = us_append(group, [5000 + 1001 * 300 + 30])
    given = np.append(ticks, 5000 + 1001 * 300 + 30)
    np.testing.assert_array_equal(us_times(snapped), np.append(ticks, 5000 + 1001 * 300))
    far = 5000 + 1001 * 301 + 500
    final = us_append(snapped, [far])
    given = np.append(given, far)
    after = us_times(final)
    assert np.abs(after - given).max() <= 0.1 * 1001 * 1.02
    assert (after % 100 == 5000 % 100).all() and (after % 200 != after[0] % 200).any()  # a 100 us grid
    assert len(final) < len(snapped) * 3 + 2000


@pytest.mark.parametrize("late", [60, 80, 99, -60, -80, -99])
@pytest.mark.parametrize("far", [433, 1700])
@pytest.mark.parametrize("offset", [0, 13, 27, 41, 58, 77])
def test_a_tick_snapped_to_a_lattice_and_rounded_again_stays_within_the_time_error(late, far, offset):
    """Snapping within e / 2 only: a tick e late would move e again when the block is rounded whole."""
    ticks = 5000 + offset + 1001 * np.arange(300)
    new = [5000 + offset + 1001 * 300 + late, 5000 + offset + 1001 * 301 + far]
    group = us_encode(ticks)
    after = us_times(us_append(us_append(group, new[:1]), new[1:]))
    assert np.abs(after - np.append(ticks, new)).max() <= 0.1 * 1001 * 1.02


def test_a_rounded_block_takes_new_ticks_on_a_finer_nested_grid():
    """Stored on 500 us; an influx at 250 us (q* = 50) rounds to 50 us, one at 1000 us (q* = 200) to 100 us
    (200 does not nest with 500). The stored ticks never move."""
    rng = np.random.default_rng(3)
    ticks = 10_000 + 2500 * np.arange(200) + rng.integers(-60, 60, 200)  # rounds to 500 us (q = 500 at e = 0.1)
    group = us_encode(ticks)
    stored = us_times(group)
    assert (stored % 500 == 0).all()
    for interval, quantum, count, jitter in [(250, 50, 3000, 12), (1000, 100, 1000, 120)]:
        new = np.sort(stored[-1] + 2500 + interval * np.arange(1, count + 1) + rng.integers(-jitter, jitter + 1, count))
        after = us_times(us_append(group, new))
        np.testing.assert_array_equal(after[:200], stored)
        added = after[200:]
        assert added.size == count and (added % quantum == 0).all() and not (added % (2 * quantum) == 0).all()
        assert np.abs(added - new).max() <= quantum // 2


def test_a_rounded_block_on_a_fine_grid_takes_new_ticks_on_its_phase():
    rng = np.random.default_rng(4)
    ticks = 10_000 + 100 * np.arange(300) + rng.integers(-5, 6, 300)  # q = 20
    group = us_encode(ticks)
    stored = us_times(group)
    assert (stored % 20 == 0).all()
    new = np.sort(stored[-1] + 120 * np.arange(1, 601) + rng.integers(-2, 3, 600))  # q* = 20: no budget to re-round
    after = us_times(us_append(group, new))
    np.testing.assert_array_equal(after[:300], stored)
    added = after[300:]
    assert ((added - stored[0]) % 20 == 0).all() and np.abs(added - new).max() <= 10


def test_a_much_coarser_merged_interval_rounds_the_block_whole_within_the_budget():
    rng = np.random.default_rng(4)
    ticks = 10_000 + 100 * np.arange(300) + rng.integers(-5, 6, 300)
    given = np.sort(np.concatenate([ticks, ticks[-1] + 2000 * np.arange(1, 601) + rng.integers(-15, 16, 600)]))
    after = us_times(us_append(us_encode(ticks), given[300:]))
    assert (after % 200 == after[0] % 200).all() and np.abs(after - given).max() <= 0.1 * 2000 * 1.02


def test_a_micro_second_clock_streamed_by_five_in_us_and_ns_ticks_equals_the_batch():
    rng = np.random.default_rng(0)
    n = 6000
    clock = (100 + np.arange(n) * 1000 + np.round(rng.normal(0, 5, n))).astype(np.int64)
    for unit, scale in [("us", 1), ("ns", 1000)]:
        kwargs = {"start_time": 0, "block_duration": 60_000_000 * scale, "time_unit": unit}
        t, x = clock * scale, np.zeros(n)
        batch = fluxcode.encode_time_blocks(x, t, E01, **kwargs).group
        group = fluxcode.encode_time_blocks(x[:5], t[:5], E01, **kwargs).group
        for i in range(5, n, 5):
            group = fluxcode.update_time_blocks(group, x[i:i + 5], t[i:i + 5], E01, **kwargs).group
        assert not irregular_blocks(batch) and not irregular_blocks(group)
        assert len(group) <= len(batch) + 2


def test_a_block_continues_the_regular_clock_before_it():
    ticks = 1001 * np.arange(500)
    rng = np.random.default_rng(7)
    second = 499_499 + 1001 * (100 + np.arange(500)) + rng.integers(-20, 21, 500)
    kwargs = {"start_time": 0, "block_duration": 500_000, "time_unit": "us"}
    first_group = fluxcode.encode_time_blocks(np.zeros(500), ticks, E01, **kwargs).group
    both = np.concatenate([ticks, second])
    at_once = fluxcode.encode_time_blocks(np.zeros(1000), both, E01, **kwargs).group
    stepwise = fluxcode.update_time_blocks(first_group, np.zeros(500), second, E01, **kwargs).group
    assert at_once == stepwise
    after = us_times(at_once)[500:]
    assert (np.diff(after) == 1001).all() and (after[0] - ticks[-1]) % 1001 == 0
    assert np.abs(after - second).max() <= 0.05 * 1001


def test_ten_one_minute_blocks_keep_one_phase():
    minute = 60_000_000
    kwargs = {"start_time": 0, "block_duration": minute, "time_unit": "us"}
    rng = np.random.default_rng(8)
    ticks = 30 + 1000 * np.arange(600_000) + rng.integers(-5, 6, 600_000)
    first = fluxcode.encode_time_blocks(np.zeros(60_000), ticks[:60_000], E01, **kwargs).group
    group = first
    for block in range(1, 10):
        part = ticks[block * 60_000:(block + 1) * 60_000]
        group = fluxcode.update_time_blocks(group, np.zeros(part.size), part, E01, **kwargs).group
    at_once = fluxcode.encode_time_blocks(np.zeros(600_000), ticks, E01, **kwargs).group
    assert group == at_once
    after = us_times(group)
    assert len(set((after % 200).tolist())) == 1 and not irregular_blocks(group)


def test_small_blocks_are_not_rounded_until_they_grow():
    kwargs = {"start_time": 0, "block_duration": 1_000_000, "time_unit": "us"}
    rng = np.random.default_rng(9)
    big = 1000 * np.arange(1000)
    small = 1_000_100 + 1000 * np.arange(10) + rng.integers(-30, 31, 10)
    group = fluxcode.encode_time_blocks(np.zeros(1010), np.concatenate([big, small]), E01, **kwargs).group
    np.testing.assert_array_equal(us_times(group)[1000:], small)
    more = 1_000_100 + 10_000 + 1000 * np.arange(20) + rng.integers(-30, 31, 20)
    after = us_times(fluxcode.update_time_blocks(group, np.zeros(20), more, E01, **kwargs).group)[1000:]
    given = np.sort(np.concatenate([small, more]))
    assert (after % 200 == after[0] % 200).all() and np.abs(after - given).max() <= 100 * 1.02


@pytest.mark.parametrize("kind,scale", [("clock", 1), ("clock", 1000), ("regular-1001", 1), ("regular-1001", 1000),
                                        ("poisson", 1)])
def test_repeated_updates_move_no_tick_further_than_the_time_error(kind, scale):
    """Every surviving sample stays within e * interval * 1.02 of the tick it was given, however often
    its blocks are updated."""
    rng = np.random.default_rng(11)
    interval = 1001 if kind == "regular-1001" else 1000
    bound = 0.1 * interval * 1.02 * scale
    kwargs = {"start_time": 0, "block_duration": 100_000 * scale, "time_unit": "us" if scale == 1 else "ns"}

    def gaps(num):
        if kind == "regular-1001":
            return np.full(num, 1001)
        if kind == "clock":
            return np.round(rng.normal(interval, 0.02 * interval, num)).astype(np.int64)
        return np.maximum(1, rng.exponential(interval, num).astype(np.int64))

    ticks = (1 + np.cumsum(gaps(400))) * scale
    given = dict(enumerate(ticks.tolist()))
    group = fluxcode.encode_time_blocks(np.arange(400.0), ticks, E01, **kwargs).group
    next_id = 400
    for _ in range(40):
        times = us_times(group)
        choice = rng.random()
        if choice < 0.35:  # a jittered re-send of some stored samples
            pick = np.sort(rng.choice(times.size, min(int(rng.integers(1, 30)), times.size), replace=False))
            new = np.maximum(times[pick] + rng.integers(-3, 4, pick.size) * scale, 0)
        elif choice < 0.55:  # samples off the lattice
            pick = np.sort(rng.choice(times.size, 5, replace=False))
            new = np.maximum(times[pick] + rng.integers(-interval // 2, interval // 2, 5) * scale, 0)
        else:  # an append
            new = times[-1] + np.cumsum(gaps(int(rng.integers(1, 80)))) * scale
        new = np.sort(new)
        ids = np.arange(next_id, next_id + new.size)
        next_id += new.size
        given.update(zip(ids.tolist(), new.tolist()))
        group = fluxcode.update_time_blocks(group, ids.astype(np.float64), new, E01, **kwargs).group
        decoded = fluxcode.decode_group(group)
        stored = decoded.times.view(np.int64)
        sample_ids = np.round(decoded.values).astype(np.int64)
        errors = np.abs(stored - np.array([given[i] for i in sample_ids.tolist()]))
        assert errors.max() <= bound, (errors.max(), bound)
        assert (np.diff(stored) >= 0).all()
