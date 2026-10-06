# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Blocks of any size: encode_blocks, empty and short blocks, and time-divided blocks
(encode_time_blocks and update_time_blocks)."""

import datetime

import numpy as np
import pytest
from _series import group_rows
from _signals import minute

import fluxcode
from fluxcode import Params, _compress, _encoder, _format, _group
from fluxcode._format import BLOCK_FLAG_NONFINITE, BLOCK_FLAG_ORDER, SHORT_BLOCK_LEN

SIZES = [1000, 0, 1, 2, 7, 8, 9, 0, 255, 4096, 0]


def times_of(group):
    """The decoded ticks of a block group with a time axis."""
    times = fluxcode.decode_group(group).times
    assert times is not None
    return times.view(np.int64)


def offsets(sizes):
    return np.concatenate([[0], np.cumsum(sizes)]).astype(int)


def test_encode_blocks_round_trip():
    x = minute("random-walk", 1)[:sum(SIZES)]
    group, lo, hi, mean = fluxcode.encode_blocks(x, SIZES)
    values, times, sizes = fluxcode.decode_group(group)
    assert times is None and sizes.tolist() == SIZES
    off = offsets(SIZES)
    for b, size in enumerate(SIZES):
        block = x[off[b]:off[b + 1]]
        if not size:
            assert np.isnan([lo[b], hi[b], mean[b]]).all()
            continue
        assert (lo[b], hi[b]) == (block.min(), block.max())
        assert mean[b] == pytest.approx(block.mean())
        assert np.abs(values[off[b]:off[b + 1]] - block).max() <= (block.max() - block.min()) / (2 ** 6 - 0.5)
    # Each block is encoded on its own: the same block encodes the same in any company
    rows = group_rows(group)
    alone = group_rows(fluxcode.encode_blocks(x[off[9]:off[10]], [4096]).group)
    np.testing.assert_array_equal(rows.residuals[off[9]:off[10]], alone.residuals)
    assert rows.grid_params[9] == alone.grid_params[0]


def test_empty_blocks_store_nothing():
    group = fluxcode.encode_blocks(np.arange(5.0), [0, 5, 0])
    rows = group_rows(group.group)
    assert rows.block_flags.tolist()[::2] == [0, 0] and rows.grid_params.tolist()[::2] == [0, 0]
    assert rows.value_anchors.tolist()[::2] == [0, 0]


@pytest.mark.parametrize("sizes", [[], [0], [0, 0, 0]])
def test_groups_without_samples(sizes):
    for times in (None, np.zeros(0, "datetime64[ms]")):
        group, lo, _, _ = fluxcode.encode_blocks(np.zeros(0), sizes, times=times)
        assert lo.shape == (len(sizes),) and np.isnan(lo).all()
        values, decoded_times, decoded_sizes = fluxcode.decode_group(group)
        assert values.shape == (0,) and decoded_sizes.tolist() == sizes
        assert (decoded_times is None) == (times is None)
        if times is not None:
            assert decoded_times is not None
            assert decoded_times.dtype == times.dtype and decoded_times.shape == (0,)


@pytest.mark.parametrize("size", range(1, SHORT_BLOCK_LEN + 1))
@pytest.mark.parametrize("decimal", [True, False])
def test_short_blocks_skip_the_analysis(size, decimal):
    """Blocks of up to 8 samples: order 0 at the finest step, whatever params say. Decimal
    detection still runs: decimal data that fits max_quantize_bits decodes bit-exact, and data
    that doesn't takes the power-of-two grid."""
    rng = np.random.default_rng(size)
    x = np.round(rng.normal(size=size) * 100, 2) if decimal else rng.normal(size=size) * 100
    x[size // 2] = np.nan
    params = Params(diff_orders={2}, noise_floor_sigma=1.0, target_bits_per_sample=6.0)
    group, lo, _, _ = fluxcode.encode_blocks(x, [size], params)
    rows = group_rows(group)
    flags = int(rows.block_flags[0])
    assert flags & BLOCK_FLAG_ORDER == 0 and flags & BLOCK_FLAG_NONFINITE
    finite = x[np.isfinite(x)]
    y = fluxcode.decode_group(group).values
    np.testing.assert_array_equal(np.isnan(y), np.isnan(x))
    on_grid = decimal and finite.size > 1 and finite.max() > finite.min()
    assert bool(flags & _format.BLOCK_FLAG_DECIMAL) == on_grid
    if on_grid:
        assert rows.grid_params[0] == -2
        np.testing.assert_array_equal(y[np.isfinite(x)], finite)  # bit-exact
        q = np.round(finite * 100).astype(np.int64) - rows.value_anchors[0]
    else:
        rng_x = finite.max() - finite.min() if finite.size else 0.0
        fine = _encoder.range_exponent(finite.min(), finite.max(), 16) if finite.size else 0
        assert rows.grid_params[0] == fine
        assert np.nanmax(np.abs(y - x), initial=0) <= rng_x / (2 ** 16 - 0.5)
        q = (np.rint(finite / 2.0 ** fine) - np.rint(lo[0] / 2.0 ** fine)).astype(np.int64)
    # Order 0: the residuals are the quantized values themselves
    np.testing.assert_array_equal(rows.residuals[np.isfinite(x)].astype(np.int64) & 0xFFFF, q & 0xFFFF)


def test_short_blocks_honour_decimal_detection_off():
    x = np.array([1.25, 3.5, 2.75])
    rows = group_rows(fluxcode.encode_blocks(x, [3], Params(decimal_detection=False)).group)
    assert not rows.block_flags[0] & _format.BLOCK_FLAG_DECIMAL
    rows = group_rows(fluxcode.encode_blocks(x, [3]).group)
    assert rows.block_flags[0] & _format.BLOCK_FLAG_DECIMAL and rows.grid_params[0] == -2


def test_target_weights_blocks_by_size():
    """The target budget covers the analyzed samples: short and empty blocks don't count."""
    x = minute("chirp", 2)
    base = fluxcode.encode_group(x, Params(noise_floor_sigma=0, target_bits_per_sample=6.0)).group
    sizes = [1000] * 60
    padded = np.concatenate([x, np.zeros(5)])
    mixed = fluxcode.encode_blocks(padded, sizes + [5, 0], Params(noise_floor_sigma=0, target_bits_per_sample=6.0))
    np.testing.assert_array_equal(group_rows(mixed.group).residuals[:60_000], group_rows(base).residuals)


@pytest.mark.parametrize("x,sizes,msg", [
    (np.zeros(10), [5, 4], "add up to 9"),
    (np.zeros(10), [5, 5.0], "integer"),
    (np.zeros(10), [[5, 5]], "1-D"),
    (np.zeros(10), [15, -5], "from 0 to 65535"),
    (np.zeros(70_000), [70_000], "from 0 to 65535"),
    (np.zeros(0), [0] * 65_536, "65535 blocks"),
])
def test_bad_block_sizes(x, sizes, msg):
    with pytest.raises(ValueError, match=msg):
        fluxcode.encode_blocks(x, sizes)


def test_times_in_tiny_and_empty_blocks():
    """Blocks of 0, 1 and 2 samples; leading empty blocks before negative ticks; a 2-sample block
    whose one delta exceeds int64 maximum."""
    sizes = [0, 0, 1, 2, 0, 3, 1, 2]
    ticks = np.array([-50, -40, -40, -10, 0, 7, 9, 10**18, 2**62 + 10**18], np.int64)
    group = fluxcode.encode_blocks(np.arange(9.0), sizes, times=ticks, time_unit="s").group
    np.testing.assert_array_equal(times_of(group), ticks)
    rows = group_rows(group).time_rows
    assert rows.steps.tolist() == [0, 0, 0, 0, 0, 1, 0, 2**62]
    huge = np.array([np.iinfo(np.int64).min + 1, np.iinfo(np.int64).max])
    group = fluxcode.encode_blocks(np.zeros(2), [2], times=huge, time_unit="ns").group
    np.testing.assert_array_equal(times_of(group), huge)
    assert group_rows(group).time_rows.steps[0] == 1 and group_rows(group).time_rows.refs[0] == 2**64 - 2


def test_single_sample_blocks_must_be_canonical():
    """A block of one sample has no deltas: the decoder accepts only a step and reference of 0,
    regular."""
    from test_time import corrupt_group

    ok = corrupt_group([False, False], [5, 9], [0, 0], [0, 0], np.zeros(2), sizes=[1, 1])
    assert times_of(ok).tolist() == [5, 9]
    for irregular, steps, refs in [([False, True], [0, 0], [0, 0]), ([False, False], [0, 3], [0, 0]),
                                   ([False, False], [0, 0], [0, 1])]:
        corrupt = corrupt_group(irregular, [5, 9], steps, refs, np.zeros(2), sizes=[1, 1])
        with pytest.raises(ValueError, match="block 1: a single sample"):
            fluxcode.decode_group(corrupt)
        # Every parse checks it, including an update that carries the block without expanding it
        with pytest.raises(ValueError, match="block 1: a single sample"):
            fluxcode.update(corrupt, {2: [1.0]}, times={2: [20]})


def test_noise_floor_needs_256_samples():
    """Below 256 samples the gate can't tell white noise from a random walk: blocks keep the
    finest step. From 256 white noise gates as before."""
    rng = np.random.default_rng(4)
    noise = rng.normal(size=2 * 255 + 256)
    group = fluxcode.encode_blocks(noise, [255, 255, 256], Params(decimal_detection=False))
    rows = group_rows(group.group)
    fine = [_encoder.range_exponent(lo, hi, 16) for lo, hi in zip(group.block_min, group.block_max)]
    assert (rows.grid_params[:2] == fine[:2]).all() and rows.grid_params[2] > fine[2]


def test_empty_block_time_columns_must_be_zero():
    """Writers store 0 in an empty block's time_start, time_step and time_ref; decoders check."""
    group = fluxcode.encode_blocks(np.zeros(16), [8, 0, 8], times=np.r_[np.arange(8), 100 + np.arange(8)],
                                  time_unit="ns").group
    body = _group.decompress(group).raw_body
    num_blocks = 3
    for field_start in (_format.time_start_start(num_blocks), _format.time_step_start(num_blocks),
                        _format.time_ref_start(num_blocks)):
        bad = body.copy()
        bad[field_start + 1] = 1  # byte 0 of block 1's value
        bad_group = group[:_format.HEADER_BYTES] + _compress.zstd()[0].compress(bad.tobytes())
        with pytest.raises(ValueError, match="block 1: empty block with nonzero time columns"):
            fluxcode.decode_group(bad_group)


# Time-divided blocks

START = np.datetime64("2026-03-01T12:00", "ms")
MINUTE = np.timedelta64(1, "m")


def encode_hour(x, t, params=Params()):
    """encode_time_blocks of an hour from START in one-minute blocks."""
    return fluxcode.encode_time_blocks(x, t, params, start_time=START, block_duration=MINUTE)


def update_hour(group, x, t, ranges, time_unit=None):
    """update_time_blocks of a block group made by encode_hour."""
    return fluxcode.update_time_blocks(group, x, t, start_time=START, block_duration=MINUTE, delete_ranges=ranges,
                                       time_unit=time_unit)


def hour_of_data(seed=0, gaps=((5, 20),)):
    """About 1 Hz of jittered ms times over an hour, minus the given minute ranges."""
    rng = np.random.default_rng(seed)
    t = START + np.sort(rng.integers(0, 3_600_000, 3600)).astype("timedelta64[ms]")
    for first, last in gaps:
        t = t[(t < START + first * MINUTE) | (t >= START + last * MINUTE)]
    x = np.round(100 * np.sin(np.arange(t.size) / 40), 2)  # decimals: lossless
    return x, t


def test_encode_time_blocks_divides_by_time():
    x, t = hour_of_data()
    group, lo, _, _ = encode_hour(x, t)
    values, times, sizes = fluxcode.decode_group(group)
    np.testing.assert_array_equal(values, x)
    np.testing.assert_array_equal(times, t)
    assert sizes.tolist() == np.bincount((t - START) // MINUTE).tolist()
    assert (sizes[5:20] == 0).all() and np.isnan(lo[5:20]).all()
    # The block group ends at the last block with a sample: no trailing empty blocks
    assert sizes[-1] > 0 and len(sizes) == int((t[-1] - START) // MINUTE) + 1


@pytest.mark.parametrize("start,duration,kwargs", [
    (START, MINUTE, {}),
    (datetime.datetime(2026, 3, 1, 12), datetime.timedelta(minutes=1), {}),  # noqa: DTZ001
    (START.astype("datetime64[s]"), np.timedelta64(60_000_000, "us"), {}),
    (START.astype("datetime64[m]"), np.timedelta64(1, "m"), {}),
    (int(START.astype(np.int64)), 60_000, {"time_unit": "ms", "int_times": True}),
    (START, MINUTE, {"time_unit": "ms", "int_times": True}),
])
def test_time_arguments_convert_exactly(start, duration, kwargs):
    x, t = hour_of_data()
    ref = encode_hour(x, t).group
    times = t.view(np.int64) if kwargs.pop("int_times", False) else t
    assert fluxcode.encode_time_blocks(x, times, start_time=start, block_duration=duration, **kwargs).group == ref


@pytest.mark.parametrize("start,duration,msg", [
    (START + 10 * MINUTE, MINUTE, "at or after start_time"),
    (START.astype("datetime64[us]") + np.timedelta64(1, "us"), MINUTE, "whole number of ms"),
    (START, np.timedelta64(1500, "us"), "whole number of ms"),
    (START, np.timedelta64(0, "ms"), "positive"),
    (START, -MINUTE, "positive"),
    (START, np.timedelta64(1, "ms"), "past the last block"),
    (np.datetime64("NaT", "ms"), MINUTE, "NaT"),
    (np.datetime64("2026-03", "M"), MINUTE, "fixed length"),
    (datetime.datetime(2026, 3, 1, 12, tzinfo=datetime.timezone.utc), MINUTE, "time zone"),
    (START, 1.5, "timedelta"),
    ("2026-03-01", MINUTE, "datetime"),
    (MINUTE, MINUTE, "datetime"),
])
def test_bad_time_arguments(start, duration, msg):
    x, t = hour_of_data()
    with pytest.raises(ValueError, match=msg):
        fluxcode.encode_time_blocks(x, t, start_time=start, block_duration=duration)


def test_encode_time_blocks_rejects_decreasing_times():
    x, t = hour_of_data()
    t = t.copy()
    t[100], t[101] = t[101], t[100] + np.timedelta64(1, "ms")
    t[2000] = t[10]  # a block id that goes back
    with pytest.raises(ValueError, match="non-decreasing"):
        encode_hour(x, t)


def test_block_too_large():
    t = START + np.zeros(65_536, "timedelta64[ms]")
    with pytest.raises(ValueError, match="65535 samples"):
        encode_hour(np.zeros(65_536), t)


def expected_after(x, t, x_new, t_new, ranges):
    """The series an update should leave: the old samples outside the ranges, plus the new ones."""
    inside = np.zeros(t.size, bool)
    for first, last in ranges:
        inside |= (t >= first) & (t < last)
    tt = np.concatenate([t[~inside], t_new])
    order = np.argsort(tt, kind="stable")
    return np.concatenate([x[~inside], x_new])[order], tt[order]


def minutes(first, last):
    return START + first * MINUTE, START + last * MINUTE


def new_data(ranges, step_ms=700, seed=1):
    t = np.concatenate([np.arange(a, b, np.timedelta64(step_ms, "ms")) for a, b in ranges])
    # A smooth signal on a 0.01 grid: decimal blocks, lossless however they are merged
    return np.round(50 * np.cos(np.arange(t.size) / 30 + seed), 2), t


@pytest.mark.parametrize("ranges", [
    [minutes(5, 20)],  # fills the gap exactly: whole empty blocks
    [minutes(30, 31)],  # one whole block
    [(START + np.timedelta64(30 * 60_000 + 15_000, "ms"), START + np.timedelta64(31 * 60_000 + 45_000, "ms"))],
    [minutes(2, 3), minutes(40, 45), (START + np.timedelta64(50 * 60_000 + 1, "ms"), START + 51 * MINUTE)],
    [minutes(58, 63)],  # past the end: appends
    [minutes(70, 71)],  # appends past a gap of empty blocks
])
def test_update_time_blocks_matches_from_scratch(ranges):
    x, t = hour_of_data()
    group = encode_hour(x, t).group
    x_new, t_new = new_data(ranges)
    group2, indices, lo, _, _ = update_hour(group, x_new, t_new, ranges)
    xx, tt = expected_after(x, t, x_new, t_new, ranges)
    values, times, _ = fluxcode.decode_group(group2)
    np.testing.assert_array_equal(times, tt)
    np.testing.assert_array_equal(values, xx)  # decimals: straddling blocks re-encode losslessly
    ref = encode_hour(xx, tt)
    assert group2 == ref.group
    np.testing.assert_array_equal(lo, ref.block_min[indices])
    # Exactly the blocks that meet a range and changed are re-encoded
    blocks = set()
    for first, last in ranges:
        blocks |= set(range(int((first - START) // MINUTE), -int(-(last - START) // MINUTE)))
    old_sizes = fluxcode.decode_group(group).block_sizes
    changed = {b for b in blocks if b >= len(old_sizes) or old_sizes[b] or ((t_new - START) // MINUTE == b).any()}
    num_new_blocks = len(ref.block_min)
    changed |= set(range(len(old_sizes), num_new_blocks))
    assert indices.tolist() == sorted(b for b in changed if b < num_new_blocks)


def test_update_time_blocks_carries_other_blocks_untouched(monkeypatch):
    """Blocks outside the ranges keep their rows; only straddling blocks are decoded."""
    x, t = hour_of_data()
    x = np.cumsum(np.random.default_rng(3).normal(size=x.size))  # lossy: re-encoding would show
    group = encode_hour(x, t).group
    ranges = [(START + np.timedelta64(30 * 60_000 + 15_000, "ms"), START + 33 * MINUTE)]  # straddles block 30
    decoded = []
    original = _group.decode_blocks
    monkeypatch.setattr(_group, "decode_blocks",
                        lambda parsed, ids, *rest: decoded.append(ids.tolist()) or original(parsed, ids, *rest))
    x_new, t_new = new_data(ranges)
    group2, indices, _, _, _ = update_hour(group, x_new, t_new, ranges)
    assert decoded == [[30]] and indices.tolist() == [30, 31, 32]
    before, after = group_rows(group), group_rows(group2)
    off_before, off_after = offsets(before.block_sizes), offsets(after.block_sizes)
    for b in set(range(len(before.block_sizes))) - {30, 31, 32}:
        assert before.block_flags[b] == after.block_flags[b]
        assert before.grid_params[b] == after.grid_params[b] and before.value_anchors[b] == after.value_anchors[b]
        np.testing.assert_array_equal(before.residuals[off_before[b]:off_before[b + 1]],
                                      after.residuals[off_after[b]:off_after[b + 1]])
        assert before.time_rows.starts[b] == after.time_rows.starts[b]
    # The straddling block keeps its samples before the range, re-encoded on its own grid
    old_times, new_times = fluxcode.decode_group(group).times, fluxcode.decode_group(group2).times
    assert old_times is not None and new_times is not None
    kept = (old_times >= START + 30 * MINUTE) & (old_times < ranges[0][0])
    np.testing.assert_array_equal(new_times[np.isin(new_times, old_times[kept])], old_times[kept])


def test_update_time_blocks_discards_only():
    x, t = hour_of_data()
    group = encode_hour(x, t).group
    ranges = [minutes(30, 32), minutes(31, 35)]  # overlapping
    group2, indices, lo, _, _ = update_hour(group, [], np.zeros(0, "datetime64[ms]"), ranges)
    assert indices.tolist() == [30, 31, 32, 33, 34] and np.isnan(lo).all()
    _, tt = expected_after(x, t, np.zeros(0), t[:0], ranges)
    np.testing.assert_array_equal(fluxcode.decode_group(group2).times, tt)
    # Ranges over empty or missing blocks change nothing
    nothing = update_hour(group2, [], np.zeros(0, np.int64), minutes(6, 9))
    assert nothing.group == group2 and nothing.indices.shape == (0,)
    nothing = update_hour(group2, [], np.zeros(0, np.int64), minutes(90, 99))
    assert nothing.group == group2


@pytest.mark.parametrize("ranges", [
    minutes(30, 31),
    [minutes(30, 31)],
    (minutes(30, 31),),
    np.array([minutes(30, 31)]),
    np.array([minutes(30, 31)]).astype(np.int64),
    [(datetime.datetime(2026, 3, 1, 12, 30), datetime.datetime(2026, 3, 1, 12, 31))],  # noqa: DTZ001
    [(datetime.datetime(2026, 3, 1, 12, 30), START + 31 * MINUTE)],  # noqa: DTZ001
])
def test_delete_range_forms(ranges):
    x, t = hour_of_data()
    group = encode_hour(x, t).group
    x_new, t_new = new_data([minutes(30, 31)])
    ref = update_hour(group, x_new, t_new, [minutes(30, 31)])
    assert update_hour(group, x_new, t_new, ranges).group == ref.group


def test_delete_ranges_from_any_iterable():
    x, t = hour_of_data()
    group = encode_hour(x, t).group
    x_new, t_new = new_data([minutes(30, 31)])
    ref = update_hour(group, x_new, t_new, [minutes(30, 31)]).group
    assert update_hour(group, x_new, t_new, (pair for pair in [minutes(30, 31)])).group == ref
    # Overlapping, touching, empty and unsorted ranges merge into the same union
    pieces = [minutes(31, 31), (START + 30 * MINUTE + np.timedelta64(20, "s"), START + 31 * MINUTE),
              minutes(30, 30) , (START + 30 * MINUTE, START + 30 * MINUTE + np.timedelta64(20, "s"))]
    assert update_hour(group, x_new, t_new, pieces).group == ref
    with pytest.raises(ValueError, match="pairs"):
        update_hour(group, x_new, t_new, 5)


def test_update_time_blocks_errors():
    x, t = hour_of_data()
    group = encode_hour(x, t).group
    x_new, t_new = new_data([minutes(30, 31)])
    with pytest.raises(ValueError, match="ends before it starts"):
        update_hour(group, x_new, t_new, minutes(31, 30))
    with pytest.raises(ValueError, match=r"\(start, end\) pairs"):
        update_hour(group, x_new, t_new, [(START, START, START)])
    with pytest.raises(ValueError, match="no time axis"):
        update_hour(fluxcode.encode_group(x).group, x_new, t_new, minutes(30, 31))
    with pytest.raises(ValueError, match="are in us but the block group stores ms"):
        update_hour(group, x_new, t_new.astype("datetime64[us]"), minutes(30, 31))
    with pytest.raises(ValueError, match="block group stores ms"):
        update_hour(group, x_new, t_new, minutes(30, 31), time_unit="us")
    with pytest.raises(ValueError, match="non-decreasing"):
        update_hour(group, x_new, t_new[::-1], minutes(30, 31))
    with pytest.raises(ValueError, match="one entry per sample"):
        update_hour(group, x_new[1:], t_new, minutes(30, 31))
    # Integer ticks are in the block group's time unit
    by_ticks = update_hour(group, x_new, t_new.view(np.int64), minutes(30, 31))
    assert by_ticks.group == update_hour(group, x_new, t_new, minutes(30, 31)).group
    # Empty ranges delete nothing: the new samples are upserted
    ref = update_hour(group, x_new, t_new, None).group
    for empty in ([], [minutes(30, 30)], np.zeros((0, 2), np.int64)):
        assert update_hour(group, x_new, t_new, empty).group == ref
    # 0-d arrays aren't pairs, as the ranges or as one of their items
    with pytest.raises(ValueError, match=r"\(start, end\) pairs"):
        update_hour(group, x_new, t_new, np.array(5))
    with pytest.raises(ValueError, match=r"\(start, end\) pairs"):
        update_hour(group, x_new, t_new, [np.array(5), np.array(6)])
    with pytest.raises(ValueError, match=r"\(start, end\) pairs"):
        update_hour(group, x_new, t_new, [(START, START + MINUTE), np.array(6)])


def test_update_time_blocks_upsert_replaces_equal_times():
    x, t = hour_of_data()
    group = encode_hour(x, t).group
    rng = np.random.default_rng(5)
    pick = np.sort(rng.choice(t.size, 40, replace=False))
    # Existing times get new values, and some new times land between them or past the end
    t_new = np.sort(np.concatenate([t[pick], t[pick[:10]] + np.timedelta64(1, "ms"), [START + 62 * MINUTE]]))
    x_new = np.round(np.linspace(-50, 50, t_new.size), 2)
    group2, indices, _, _, _ = update_hour(group, x_new, t_new, None)
    drop = np.isin(t, t_new)
    tt = np.concatenate([t[~drop], t_new])
    order = np.argsort(tt, kind="stable")
    xx = np.concatenate([x[~drop], x_new])[order]
    assert group2 == encode_hour(xx, tt[order]).group
    # Every time a new sample has replaced the old one: no extra samples at those times
    assert (np.isin(fluxcode.decode_group(group2).times, t[pick]).sum()) == len(pick)
    # The blocks the samples land in, and the empty ones appended to reach the last
    old_blocks = len(fluxcode.decode_group(group).block_sizes)
    hit = {int((v - START) // MINUTE) for v in t_new}
    assert indices.tolist() == sorted(hit | set(range(old_blocks, max(hit) + 1)))
    # Upserting the same samples again changes the bytes of no block's data
    assert update_hour(group2, x_new, t_new, None).group == group2


def test_update_time_blocks_upsert_keeps_duplicate_new_times():
    x, t = hour_of_data()
    group = encode_hour(x, t).group
    stamp = t[100]
    t_new = np.array([stamp, stamp, stamp, START + 59 * MINUTE + np.timedelta64(5, "ms")])
    x_new = np.array([1.0, 2.0, 3.0, 4.0])
    values, times, _ = fluxcode.decode_group(update_hour(group, x_new, t_new, None).group)
    assert values[times == stamp].tolist() == [1.0, 2.0, 3.0]  # the old sample at stamp is gone
    assert (times == stamp).sum() == 3
    # A time already duplicated in the block group is replaced entirely
    group = encode_hour(np.array([1.0, 2.0, 3.0]), np.array([START, START, START + MINUTE])).group
    values, times, _ = fluxcode.decode_group(update_hour(group, np.array([9.0]), np.array([START]), None).group)
    assert values.tolist() == [9.0, 3.0]


def test_update_time_blocks_delete_then_upsert_anywhere():
    """New samples may be timed outside the deleted ranges; ones inside a range are kept."""
    x, t = hour_of_data()
    group = encode_hour(x, t).group
    ranges = [minutes(30, 32)]
    x_new, t_new = new_data([minutes(31, 34)])  # half inside the range, half outside
    values, times, _ = fluxcode.decode_group(update_hour(group, x_new, t_new, ranges).group)
    inside = (t >= ranges[0][0]) & (t < ranges[0][1])
    drop = inside | np.isin(t, t_new)
    tt = np.concatenate([t[~drop], t_new])
    order = np.argsort(tt, kind="stable")
    np.testing.assert_array_equal(times, tt[order])
    np.testing.assert_array_equal(values, np.concatenate([x[~drop], x_new])[order])


def test_update_time_blocks_upsert_decodes_only_blocks_it_lands_in(monkeypatch):
    x, t = hour_of_data()
    group = encode_hour(x, t).group
    decoded = []
    original = _group.decode_blocks
    monkeypatch.setattr(_group, "decode_blocks",
                        lambda parsed, ids, *rest: decoded.append(ids.tolist()) or original(parsed, ids, *rest))
    result = update_hour(group, np.array([1.0, 2.0]), np.array([t[900], t[2000]]), None)
    assert decoded == [sorted({int((t[900] - START) // MINUTE), int((t[2000] - START) // MINUTE)})]
    assert result.indices.tolist() == decoded[0]


def test_update_time_blocks_nothing_to_do():
    x, t = hour_of_data()
    group = encode_hour(x, t).group
    result = update_hour(group, [], np.zeros(0, np.int64), None)
    assert result.group == group and result.indices.shape == (0,)


def test_update_time_blocks_ranges_past_int64_blocks():
    """With a duration of one tick, a range near int64 maximum is ~2^64 blocks past the start:
    far past the block group's blocks, so discarding it changes nothing."""
    int64 = np.iinfo(np.int64)
    group = fluxcode.encode_time_blocks(np.arange(3.0), np.arange(3) + int64.min + 1, start_time=int64.min + 1,
                                       block_duration=1, time_unit="ns").group
    result = fluxcode.update_time_blocks(group, [], np.zeros(0, np.int64), start_time=int64.min + 1, block_duration=1,
                                         delete_ranges=[(int64.max - 10, int64.max)])
    assert result.group == group and result.indices.shape == (0,)
