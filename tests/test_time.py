# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""docs/SPEC.md time axis: exact round trips, regular and irregular blocks, the layout, the
encoder's input checks, the decoder's corrupt-group checks, and update with times."""

import dataclasses

import numpy as np
import pytest
import zstandard
from _series import group_rows, planes
from _signals import minute

import fluxcode
from fluxcode import Params, _bitpacking, _format
from fluxcode._format import BLOCK_FLAG_IRREGULAR_TIME, BLOCK_FLAG_LONG_TIME

INT64_MAX = np.iinfo(np.int64).max
INT64_MIN = np.iinfo(np.int64).min
BLOCK_LEN = 1000
START_2026 = np.datetime64("2026-09-27T00:00:00", "ns")


def regular_times(num_samples, step_ns=1_000_000, start=START_2026):
    """num_samples timestamps from start, step_ns apart (1 kHz by default)."""
    return start + np.arange(num_samples) * np.timedelta64(step_ns, "ns")


def decoded_times(group):
    """The decoded times of a block group that has a time axis."""
    times = fluxcode.decode_group(group).times
    assert times is not None
    return times


def round_trip(values, times, params=Params(), block_len=BLOCK_LEN, **kwargs):
    """Encodes and decodes with times; the values must decode as they do without times at the
    same noise floor (the default one depends on whether there are times)."""
    group = fluxcode.encode_group(values, params, block_len=block_len, times=times, **kwargs).group
    decoded = fluxcode.decode_group(group)
    same_floor = dataclasses.replace(params, noise_floor_sigma=params.noise_factor(timed=True))
    plain = fluxcode.encode_group(values, same_floor, block_len=block_len).group
    np.testing.assert_array_equal(decoded.values, fluxcode.decode_group(plain).values)
    return group, decoded


def irregular_flags(group):
    return (group_rows(group).block_flags & BLOCK_FLAG_IRREGULAR_TIME) != 0


@pytest.mark.parametrize("unit_name", ["s", "ms", "us", "ns"])
def test_datetime64_round_trip_keeps_unit(unit_name):
    rng = np.random.default_rng(1)
    ticks = 1_790_000_000 * 10 ** {"s": 0, "ms": 3, "us": 6, "ns": 9}[unit_name] + np.cumsum(rng.integers(0, 50, 5500))
    times = ticks.view(f"datetime64[{unit_name}]")
    _, decoded = round_trip(minute("noisy-sine", 1)[:5500], times)
    assert decoded.times.dtype == times.dtype
    np.testing.assert_array_equal(decoded.times, times)


@pytest.mark.parametrize("unit_name", ["s", "ms", "us", "ns"])
def test_integer_ticks_round_trip_as_datetime64(unit_name):
    ticks = np.arange(3000, dtype=np.int64) * 7 - 12_345  # before 1970: negative ticks
    _, decoded = round_trip(np.zeros(3000), ticks, time_unit=unit_name)
    assert decoded.times.dtype == np.dtype(f"datetime64[{unit_name}]")
    np.testing.assert_array_equal(decoded.times.view(np.int64), ticks)


def test_group_without_times_decodes_times_none():
    decoded = fluxcode.decode_group(fluxcode.encode_group(np.arange(10.0)).group)
    assert decoded.times is None


def test_regular_times_store_no_planes():
    values = minute("sin-4.12hz", 2)
    group, decoded = round_trip(values, regular_times(values.size))
    np.testing.assert_array_equal(decoded.times, regular_times(values.size))
    assert not irregular_flags(group).any()
    rows = group_rows(group)
    np.testing.assert_array_equal(rows.time_rows.steps, 1_000_000)
    # 60 regular blocks cost only their start and step columns
    assert len(group) - len(fluxcode.encode_group(values).group) < 100


def test_gap_makes_only_its_block_irregular():
    values = minute("random-walk", 3)
    times = regular_times(values.size)
    times[5500:] += np.timedelta64(3, "s")  # inside block 5
    times[20_000:] += np.timedelta64(7, "s")  # exactly at the start of block 20: blocks stay regular
    group, decoded = round_trip(values, times)
    np.testing.assert_array_equal(decoded.times, times)
    np.testing.assert_array_equal(np.flatnonzero(irregular_flags(group)), [5])
    assert group_rows(group).time_rows.steps[5] == 1_000_000  # the GCD of 1 ms and 3.001 s


def test_irregular_block_step_is_gcd():
    rng = np.random.default_rng(4)
    ticks = np.cumsum(rng.integers(1, 40, 2000)) * 250  # every delta a multiple of 250
    group, decoded = round_trip(np.zeros(2000), ticks, time_unit="us")
    np.testing.assert_array_equal(decoded.times.view(np.int64), ticks)
    assert irregular_flags(group).all()
    assert (group_rows(group).time_rows.steps % 250 == 0).all()


@pytest.mark.parametrize("late_delta,step", [(1000, 1000), (1500, 500), (7, 1)])
def test_gcd_of_the_whole_block_when_the_first_deltas_share_a_larger_one(late_delta, step):
    """The encoder tries the GCD of a block's first deltas; a later delta can lower it."""
    deltas = np.full(999, 4000)
    deltas[1::2] = 2000
    deltas[500] = late_delta
    ticks = np.r_[0, np.cumsum(deltas)]
    group, decoded = round_trip(np.zeros(1000), ticks, Params(), time_unit="ns")
    np.testing.assert_array_equal(decoded.times.view(np.int64), ticks)
    assert group_rows(group).time_rows.steps[0] == step


def test_reference_is_rounded_mean_for_jitter_and_minimum_for_skewed_deltas():
    rng = np.random.default_rng(4)
    index = np.arange(3000)
    jitter = index * 1000 + np.round(rng.normal(0, 20, index.size)).astype(np.int64)
    events = np.cumsum(1 + np.round(rng.exponential(300, index.size))).astype(np.int64)
    gap = index.copy()
    gap[1500:] += 5000
    for name, ticks, expected in [
        ("jitter", jitter, lambda quotients: round(quotients.mean())),
        ("events", events, lambda quotients: quotients.min()),
        ("gap", gap, lambda quotients: quotients.min()),
    ]:
        group = fluxcode.encode_group(np.zeros(ticks.size), times=ticks, time_unit="us").group
        time_rows = group_rows(group).time_rows
        for block_idx in range(3):
            block = ticks[1000 * block_idx:1000 * (block_idx + 1)]
            quotients = np.diff(block) // time_rows.steps[block_idx]
            if (quotients == quotients[0]).all():
                assert time_rows.refs[block_idx] == 1, name
            else:
                assert time_rows.refs[block_idx] == expected(quotients), name
        np.testing.assert_array_equal(decoded_times(group).view(np.int64), ticks)


def test_equal_timestamps_are_allowed():
    ticks = np.repeat(np.arange(1500, dtype=np.int64), 2)  # every timestamp twice
    ticks[:1000] = 42  # block 0: all equal, a regular block with step 0
    group, decoded = round_trip(np.zeros(3000), ticks, time_unit="ms")
    np.testing.assert_array_equal(decoded.times.view(np.int64), ticks)
    rows = group_rows(group)
    assert not irregular_flags(group)[0] and rows.time_rows.steps[0] == 0
    assert irregular_flags(group)[1:].all()


@pytest.mark.parametrize("num_samples", [1, 2, 999, 1001, 2500])
def test_partial_last_block(num_samples):
    """Padding extends a regular last block's grid (keeping it regular) and is trimmed on decode."""
    times = regular_times(num_samples)
    group, decoded = round_trip(np.arange(float(num_samples)), times)
    np.testing.assert_array_equal(decoded.times, times)
    assert not irregular_flags(group).any()
    jittered = times + np.arange(num_samples) % 3 * np.timedelta64(1, "ns")
    _, decoded = round_trip(np.arange(float(num_samples)), jittered)
    np.testing.assert_array_equal(decoded.times, jittered)


def test_extreme_ticks():
    """Ticks at both ends of int64, and a block whose deltas span more than half of int64."""
    near_max = INT64_MAX - np.arange(1200, dtype=np.int64)[::-1] * 3
    _, decoded = round_trip(np.zeros(1200), near_max, time_unit="ns")
    np.testing.assert_array_equal(decoded.times.view(np.int64), near_max)
    near_min = INT64_MIN + 1 + np.arange(1200, dtype=np.int64)
    _, decoded = round_trip(np.zeros(1200), near_min, time_unit="ns")
    np.testing.assert_array_equal(decoded.times.view(np.int64), near_min)
    # One delta of 2^64 - 2: the GCD exceeds int64, so the block stores raw deltas with step 1
    huge_span = np.full(16, INT64_MAX, np.int64)
    huge_span[0] = INT64_MIN + 1
    group, decoded = round_trip(np.zeros(16), huge_span, block_len=8, time_unit="s")
    np.testing.assert_array_equal(decoded.times.view(np.int64), huge_span)
    assert group_rows(group).time_rows.steps[0] == 1


def test_bulk_encode_splits_times():
    values = np.tile(minute("chirp", 5), 3)[:130_000]
    times = regular_times(values.size)
    times[70_123:] += np.timedelta64(1, "s")
    groups = fluxcode.encode(values, times=times).groups
    decoded = fluxcode.decode(groups)
    assert len(decoded) == 3
    np.testing.assert_array_equal(np.concatenate([part.times for part in decoded]), times)


def test_bulk_encode_rejects_decrease_at_group_boundary():
    ticks = np.arange(40, dtype=np.int64)
    ticks[16:] -= 100
    with pytest.raises(ValueError, match="sample 16 is -84, below sample 15"):
        fluxcode.encode(np.zeros(40), block_len=8, blocks_per_group=2, times=ticks, time_unit="ns")


def test_byte_planes_and_nonfinite_with_times():
    """The time fields sit around the residual and nonfinite planes: all three coexist."""
    values = np.tile(100 * np.sin(2 * np.pi * np.arange(125) / 125), 480)
    values[[10, 3456, 12_000]] = [np.nan, np.inf, -np.inf]
    rng = np.random.default_rng(6)
    times = np.cumsum(rng.integers(1, 5, values.size)).view("datetime64[us]")
    with planes("byte"):
        group, decoded = round_trip(values, times, Params(max_quantize_bits=10))
    assert _format.unpack_header(group).byte_planes
    np.testing.assert_array_equal(decoded.times, times)
    np.testing.assert_array_equal(np.isnan(decoded.values), np.isnan(values))
    assert decoded.values[3456] == np.inf and decoded.values[12_000] == -np.inf


def test_worked_example_layout():
    """docs/SPEC.md §6's worked example: blocks of 8, 8 and 4 samples, block 1 irregular."""
    ticks = np.array([1000, 1010, 1020, 1030, 1040, 1050, 1060, 1070,
                      1080, 1090, 1100, 1130, 1140, 1150, 1160, 1170,
                      1180, 1190, 1200, 1210], np.int64)
    values = np.repeat([2.5, 2.25, 3.0], 8)[:20]
    with planes("bit"):
        group = fluxcode.encode_group(values, block_len=8, times=ticks, time_unit="ns").group
    header = group[:_format.HEADER_BYTES]
    assert header == bytes([1, 0x08, 3, 0, 20, 0, 0, 0])
    body = np.frombuffer(zstandard.ZstdDecompressor().decompress(group[_format.HEADER_BYTES:]), np.uint8)
    num_blocks = 3
    flags, sizes = body[:3], np.array([8, 8, 4])
    assert body.shape[0] == 191 == _format.group_size(num_blocks, _format.layout(flags, sizes), True)
    np.testing.assert_array_equal(flags & 0x10, [0, 0x10, 0])
    np.testing.assert_array_equal(body[3:9], [8, 8, 4, 0, 0, 0])
    assert not body[9:15].any()  # grid_params: exponent 0 for constant blocks
    value_anchor_planes = body[15:39].reshape(8, num_blocks)
    np.testing.assert_array_equal(value_anchor_planes[6], [0x04, 0x02, 0x08])
    np.testing.assert_array_equal(value_anchor_planes[7], [0x40, 0x40, 0x40])
    assert not value_anchor_planes[:6].any()
    time_start_planes = body[39:63].reshape(8, num_blocks)
    np.testing.assert_array_equal(time_start_planes[0], [0xE8, 0x50, 0x64])
    np.testing.assert_array_equal(time_start_planes[1], [0x03, 0x00, 0x00])
    assert not time_start_planes[2:].any()
    time_step_planes = body[63:87].reshape(8, num_blocks)
    np.testing.assert_array_equal(time_step_planes[0], [0x0A, 0x0A, 0x0A])
    assert not time_step_planes[1:].any()
    time_ref_planes = body[87:111].reshape(8, num_blocks)
    np.testing.assert_array_equal(time_ref_planes[0], [0x01, 0x01, 0x01])
    assert not time_ref_planes[1:].any()
    # Block 1's quotients [1, 1, 3, 1, 1, 1, 1] from sample 1: reference 1, zigzagged residuals
    # [0, 0, 0, 4, 0, 0, 0, 0], so only bit 2 of sample 3 is set
    time_residual_planes = body[159:191]
    assert time_residual_planes[2] == 0x08
    assert not np.delete(time_residual_planes, 2).any()
    np.testing.assert_array_equal(decoded_times(group).view(np.int64), ticks)


@pytest.mark.parametrize("times,kwargs,message", [
    (np.r_[np.arange(1500), np.arange(499, 999)], {"time_unit": "ns"}, "sample 1500 is 499, below sample 1499 at 1499"),
    (np.r_[np.arange(1000), np.arange(1000) - 5], {"time_unit": "ns"}, "sample 1000 is -5, below"),
    (np.r_[np.arange(10), np.arange(1990) + 5], {"time_unit": "ns"}, "sample 10 is 5"),
    (np.r_[[INT64_MIN], np.arange(1999)], {"time_unit": "ns"}, "NaT"),
    (np.r_[np.arange(1999), [INT64_MIN]].view("datetime64[ns]"), {}, "NaT"),
    (np.arange(2000.0), {"time_unit": "ns"}, "dtype float64"),
    (np.arange(2000).view("datetime64[D]"), {}, "datetime64 in s, ms, us or ns"),
    (np.arange(2000).view("datetime64[10ms]"), {}, "datetime64 in s, ms, us or ns"),
    (np.arange(2000).view("datetime64[ms]"), {"time_unit": "ms"}, "integer times only"),
    (np.arange(2000), {}, "need time_unit"),
    (np.arange(2000), {"time_unit": "minutes"}, "time_unit must be one of"),
    (np.arange(1999), {"time_unit": "ns"}, "one entry per sample"),
    (np.arange(2000, dtype=np.uint64) + np.uint64(INT64_MAX), {"time_unit": "ns"}, "fit in int64"),
    (np.array([object()] * 2000), {}, "time zone"),
])
def test_invalid_times_are_rejected(times, kwargs, message):
    with pytest.raises(ValueError, match=message):
        fluxcode.encode_group(np.zeros(2000), times=times, **kwargs)
    with pytest.raises(ValueError, match=message):
        fluxcode.encode(np.zeros(2000), times=times, **kwargs)


def test_time_unit_without_times_is_rejected():
    with pytest.raises(ValueError, match="time_unit needs times"):
        fluxcode.encode_group(np.zeros(10), time_unit="ns")


def zigzag(residuals):
    """Zigzagged int64 residuals, as the format stores them."""
    residuals = np.asarray(residuals, np.int64)
    return ((residuals << 1) ^ (residuals >> 63)).view(np.uint64)


def corrupt_group(block_flags_irregular, starts, steps, refs, residuals, block_len=8, long=None, sizes=None):
    """A block group with the given time rows (bypassing the encoder's checks) and zero values;
    long optionally flags blocks as long. Blocks hold block_len samples each, or sizes."""
    num_blocks = len(starts)
    sizes = np.full(num_blocks, block_len) if sizes is None else np.asarray(sizes)
    block_flags = np.where(block_flags_irregular, BLOCK_FLAG_IRREGULAR_TIME, 0).astype(np.uint8)
    if long is not None:
        block_flags |= np.where(long, BLOCK_FLAG_LONG_TIME, 0).astype(np.uint8)
    time_rows = _format.TimeRows(
        np.asarray(starts, np.int64), np.asarray(steps, np.int64), np.asarray(refs, np.uint64),
        zigzag(residuals),
    )
    body = _bitpacking.write_group(
        block_flags, sizes, np.zeros(num_blocks, np.int64), np.zeros(num_blocks, np.int64),
        np.zeros(sizes.sum(), np.int16), time_rows=time_rows,
    )
    header = _format.pack_header(num_blocks, int(sizes.sum()), False, int(_format.TimeUnitCode.NANOSECONDS))
    return header + zstandard.ZstdCompressor(level=3).compress(body.tobytes())


@pytest.mark.parametrize("irregular,starts,steps,refs,residuals,message", [
    ([False, False], [0, 100], [1, -1], [1, 1], np.zeros(16), "block 1: time step out of range"),
    ([True, False], [0, 100], [0, 1], [1, 1], np.r_[0, np.ones(7), np.zeros(8)], "block 0: time step out of range"),
    ([True, False], [0, 100], [1, 1], [1, 1], np.r_[1, np.ones(7), np.zeros(8)], "block 0: first time residual is not 0"),
    ([False, False], [0, INT64_MAX - 6], [1, 1], [1, 1], np.zeros(16), "block 1: times overflow"),
    ([False, False], [0, 100], [1, 1], [1, 2 ** 62], np.zeros(16), "block 1: times overflow"),  # the reference
    ([False, True], [0, INT64_MAX - 10], [1, 1], [1, 0], np.r_[np.zeros(8), 0, np.full(7, 2)], "block 1: times overflow"),
    ([False, True], [0, 100], [1, 1], [1, 0], np.r_[np.zeros(8), 0, np.full(7, -1)], "block 1: times overflow"),  # wraps
    ([False, False], [100, 50], [1, 1], [1, 1], np.zeros(16), "block 1: start time out of range"),  # a decreasing start
    ([False, False], [INT64_MIN, 0], [1, 1], [1, 1], np.zeros(16), "block 0: start time out of range"),
])
def test_corrupt_time_fields_are_rejected(irregular, starts, steps, refs, residuals, message):
    with pytest.raises(ValueError, match=message):
        fluxcode.decode_group(corrupt_group(irregular, starts, steps, refs, residuals))


def test_long_flag_without_irregular_or_needed_is_rejected():
    residuals = np.r_[np.zeros(8), 0, np.ones(7)]
    with pytest.raises(ValueError, match="block 0: block flags 0x20 sets reserved bits"):
        fluxcode.decode_group(corrupt_group([False, True], [0, 100], [1, 1], [1, 1], residuals, long=[True, False]))
    with pytest.raises(ValueError, match="block 1: long time residuals fit in 32 bits"):
        fluxcode.decode_group(corrupt_group([False, True], [0, 100], [1, 1], [1, 1], residuals, long=[False, True]))


def test_long_blocks_only_where_residuals_need_64_bits():
    """ns ticks with GCD 1: a gap of seconds needs residuals of 2^32 or more, in its block only."""
    rng = np.random.default_rng(8)
    ticks = np.sort(np.arange(4000) * 1_000_000 + rng.integers(0, 1000, 4000))
    ticks[2500:] += 5_000_000_000
    group, decoded = round_trip(np.zeros(4000), ticks, time_unit="ns")
    np.testing.assert_array_equal(decoded.times.view(np.int64), ticks)
    flags = group_rows(group).block_flags
    np.testing.assert_array_equal((flags & BLOCK_FLAG_LONG_TIME) != 0, [False, False, True, False])
    assert irregular_flags(group).all()


def test_corrupt_rows_decode_when_valid():
    """The corrupt_group helper itself builds decodable block groups, with any reference: a quotient
    is the reference plus its residual, mod 2^64."""
    quotients = np.arange(1, 8)
    group = corrupt_group([False, True, False], [0, 100, 200], [1, 5, 3], [1, 4, 2],
                        np.r_[np.zeros(8), 0, quotients - 4, np.zeros(8)])
    np.testing.assert_array_equal(
        decoded_times(group).view(np.int64),
        np.r_[np.arange(8), 100 + 5 * np.cumsum(np.r_[0, quotients]), 200 + 6 * np.arange(8)],
    )


def encoded_with_times():
    values = minute("noisy-sine", 8)
    times = regular_times(values.size)
    times[33_333:] += np.timedelta64(5, "ms")
    return values, times, fluxcode.encode_group(values, times=times).group


@pytest.mark.parametrize("indices", [[0], [33], [59], [3, 17, 42]])
def test_update_with_times_matches_reencode(indices):
    values, times, group = encoded_with_times()
    edited_values, edited_times = values.copy(), times.copy()
    new_blocks = minute("chirp", 9).reshape(60, BLOCK_LEN)[:len(indices)]
    new_times = []
    for block_idx in indices:
        # Jitter within the block's original span: stays in order with its neighbours
        block_times = times[block_idx * BLOCK_LEN:(block_idx + 1) * BLOCK_LEN] + (np.arange(BLOCK_LEN) % 2) * np.timedelta64(1, "ns")
        new_times.append(block_times)
    for position, block_idx in enumerate(indices):
        edited_values[block_idx * BLOCK_LEN:(block_idx + 1) * BLOCK_LEN] = new_blocks[position]
        edited_times[block_idx * BLOCK_LEN:(block_idx + 1) * BLOCK_LEN] = new_times[position]
    updated = fluxcode.update(group, dict(zip(indices, new_blocks)), times=dict(zip(indices, new_times))).group
    assert updated == fluxcode.encode_group(edited_values, times=edited_times).group
    np.testing.assert_array_equal(fluxcode.decode_group(updated).times, edited_times)


def test_update_appends_with_times():
    values, times, group = encoded_with_times()
    values, times = values[:30_000], times[:30_000]
    group = fluxcode.encode_group(values, times=times).group
    new_times = (times[-1] + np.arange(1, 2 * BLOCK_LEN + 1) * np.timedelta64(1, "ms")).reshape(2, BLOCK_LEN)
    new_blocks = np.ones((2, BLOCK_LEN))
    updated = fluxcode.update(group, {30: new_blocks[0], 31: new_blocks[1]}, times={30: new_times[0], 31: new_times[1]}).group
    decoded = fluxcode.decode_group(updated)
    np.testing.assert_array_equal(decoded.times, np.r_[times, new_times.reshape(-1)])
    # Integer ticks are read in the block group's time unit
    tick_blocks = new_times.view(np.int64)
    updated_from_ticks = fluxcode.update(group, {30: new_blocks[0], 31: new_blocks[1]},
                                         times={30: tick_blocks[0], 31: tick_blocks[1]}).group
    assert updated_from_ticks == updated


def test_update_time_errors():
    values, times, group = encoded_with_times()
    block = np.zeros(BLOCK_LEN)
    block_times = times[:BLOCK_LEN]
    with pytest.raises(ValueError, match="times are required"):
        fluxcode.update(group, {0: block})
    with pytest.raises(ValueError, match="no time axis"):
        fluxcode.update(fluxcode.encode_group(values).group, {0: block}, times={0: block_times})
    with pytest.raises(ValueError, match="are in us but the block group stores ns"):
        fluxcode.update(group, {0: block}, times={0: block_times.astype("datetime64[us]")})
    with pytest.raises(ValueError, match="shape of its samples"):
        fluxcode.update(group, {0: block}, times={0: block_times[:-1]})
    with pytest.raises(ValueError, match="same block indices"):
        fluxcode.update(group, {0: block}, times={1: block_times})
    # Block 1's new times start before block 0 ends
    with pytest.raises(ValueError, match="non-decreasing: block 1 starts"):
        fluxcode.update(group, {1: block}, times={1: block_times})
    # Block 0's new times end after block 1 starts
    with pytest.raises(ValueError, match="non-decreasing: block 1 starts"):
        fluxcode.update(group, {0: block}, times={0: times[BLOCK_LEN:2 * BLOCK_LEN] + np.timedelta64(1, "ns")})
    # Within a new block
    with pytest.raises(ValueError, match="non-decreasing: sample 1 "):
        fluxcode.update(group, {0: block}, times={0: block_times[::-1]})


def test_default_noise_floor_is_off_with_times():
    """noise_floor_sigma=None: DEFAULT_NOISE_FLOOR_SIGMA without times, off with them; an
    explicit floor applies either way."""
    values = minute("noisy-sine", 7)
    times = regular_times(values.shape[0])

    def decoded(params, **kwargs):
        return fluxcode.decode_group(fluxcode.encode_group(values, params, **kwargs).group).values

    floor = Params(noise_floor_sigma=fluxcode.DEFAULT_NOISE_FLOOR_SIGMA)
    off = Params(noise_floor_sigma=0)
    np.testing.assert_array_equal(decoded(Params()), decoded(floor))
    np.testing.assert_array_equal(decoded(Params(), times=times), decoded(off, times=times))
    np.testing.assert_array_equal(decoded(floor, times=times), decoded(floor))
    assert not np.array_equal(decoded(floor), decoded(off))  # the floor does gate this signal


def test_update_default_noise_floor_follows_the_group():
    """update re-encodes a timed block group's blocks with the floor off, an untimed one's with it on."""
    values = minute("noisy-sine", 8)[:5000]
    times = regular_times(values.shape[0])
    fresh = minute("noisy-sine", 9)[:BLOCK_LEN]
    off, floor = Params(noise_floor_sigma=0), Params(noise_floor_sigma=fluxcode.DEFAULT_NOISE_FLOOR_SIGMA)
    for kwargs, expected in (({"times": times}, off), ({}, floor)):
        group = fluxcode.encode_group(values, **kwargs).group
        new_times = {"times": {2: times[2 * BLOCK_LEN:3 * BLOCK_LEN]}} if kwargs else {}
        updated = fluxcode.update(group, {2: fresh}, **new_times).group
        reference = fluxcode.update(group, {2: fresh}, expected, **new_times).group
        np.testing.assert_array_equal(fluxcode.decode_group(updated).values, fluxcode.decode_group(reference).values)
