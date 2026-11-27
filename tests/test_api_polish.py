# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Tests for public API result dataclasses, required chunking args, and block start times."""

from dataclasses import FrozenInstanceError
from typing import Any

import numpy as np
import pytest

import fluxcode
from fluxcode import (
    DecodedGroup,
    EncodedGroup,
    EncodedSeries,
    Params,
    UpdatedGroup,
)


def test_result_types_are_frozen_dataclasses_not_tuples():
    x = np.linspace(0.0, 10.0, 25)
    t = np.datetime64("2026-01-01T00:00:00", "ms") + np.arange(25) * np.timedelta64(
        10, "ms"
    )
    enc = fluxcode.encode_group(x, block_len=10, times=t)
    series = fluxcode.encode(x, block_len=10, blocks_per_group=2, times=t)
    dec = fluxcode.decode_group(enc.payload)
    upd = fluxcode.update(enc.payload, {1: x[10:20] + 1.0}, times={1: t[10:20]})

    assert isinstance(enc, EncodedGroup) and not isinstance(enc, tuple)
    assert isinstance(series, EncodedSeries) and not isinstance(series, tuple)
    assert isinstance(dec, DecodedGroup) and not isinstance(dec, tuple)
    assert isinstance(upd, UpdatedGroup) and not isinstance(upd, tuple)

    enc_any: Any = enc
    dec_any: Any = dec
    with pytest.raises(TypeError):
        iter(enc_any)
    with pytest.raises(FrozenInstanceError):
        enc_any.payload = b""
    with pytest.raises(FrozenInstanceError):
        dec_any.values = np.zeros(0)


def test_block_len_and_blocks_per_group_are_required_keyword_only():
    x = np.arange(20.0)
    encode_group_fn: Any = fluxcode.encode_group
    encode_fn: Any = fluxcode.encode
    with pytest.raises(TypeError, match="block_len"):
        encode_group_fn(x)
    with pytest.raises(TypeError, match="block_len"):
        encode_fn(x, blocks_per_group=2)
    with pytest.raises(TypeError, match="blocks_per_group"):
        encode_fn(x, block_len=10)


def test_encoded_series_groups_and_flattening_properties():
    x = np.linspace(-5.0, 5.0, 55)
    t = np.datetime64("2026-01-01T00:00:00", "s") + np.arange(55) * np.timedelta64(
        1, "s"
    )
    series = fluxcode.encode(x, block_len=10, blocks_per_group=2, times=t)

    assert len(series.groups) == 3
    assert all(isinstance(g, EncodedGroup) for g in series.groups)
    assert series.payloads == [g.payload for g in series.groups]
    assert series.block_min.shape == (6,)
    assert series.block_max.shape == (6,)
    assert series.block_mean.shape == (6,)
    assert series.block_start_times is not None
    np.testing.assert_array_equal(series.block_start_times, t[[0, 10, 20, 30, 40, 50]])

    # decode() accepts both EncodedSeries and list[bytes] identically
    dec_from_series = fluxcode.decode(series)
    dec_from_payloads = fluxcode.decode(series.payloads)
    assert len(dec_from_series) == len(dec_from_payloads) == 3
    for a, b in zip(dec_from_series, dec_from_payloads, strict=True):
        np.testing.assert_array_equal(a.values, b.values)
        assert a.times is not None and b.times is not None
        np.testing.assert_array_equal(a.times, b.times)

    # Untimed series returns None for block_start_times
    untimed = fluxcode.encode(x, block_len=10, blocks_per_group=2)
    assert untimed.block_start_times is None
    assert all(g.block_start_times is None for g in untimed.groups)


def test_block_start_times_empty_blocks_and_updates():
    t = np.datetime64("2026-06-01T12:00:00", "ms") + np.arange(15) * np.timedelta64(
        100, "ms"
    )
    x = np.arange(15.0)
    enc = fluxcode.encode_blocks(x, [5, 0, 10], times=t)
    assert enc.block_start_times is not None
    assert enc.block_start_times.dtype == np.dtype("datetime64[ms]")
    assert enc.block_start_times[0] == t[0]
    assert np.isnat(enc.block_start_times[1])
    assert enc.block_start_times[2] == t[5]

    # Update replacing an empty block and appending past a gap
    t_b1 = t[4] + np.timedelta64(10, "ms") + np.arange(2) * np.timedelta64(10, "ms")
    t_b4 = t[-1] + np.timedelta64(500, "ms") + np.arange(3) * np.timedelta64(10, "ms")
    upd = fluxcode.update(
        enc.payload,
        {1: np.array([1.0, 2.0]), 4: np.array([3.0, 4.0, 5.0])},
        times={1: t_b1, 4: t_b4},
    )
    assert upd.indices.tolist() == [1, 3, 4]
    assert upd.block_start_times is not None
    assert upd.block_start_times[0] == t_b1[0]
    assert np.isnat(upd.block_start_times[1])  # gap block 3 is empty
    assert upd.block_start_times[2] == t_b4[0]

    # No-op update on timed group returns empty datetime64[ms] array
    noop = fluxcode.update(enc.payload, {}, times={})
    assert noop.block_start_times is not None
    assert noop.block_start_times.dtype == np.dtype("datetime64[ms]")
    assert noop.block_start_times.shape == (0,)

    # No-op update on untimed group returns None
    untimed = fluxcode.encode_group(x, block_len=5)
    assert untimed.block_start_times is None
    assert fluxcode.update(untimed.payload, {}).block_start_times is None


def test_block_start_times_reflect_time_error_rounding():
    rng = np.random.default_rng(7)
    base = 1_790_000_000_000_000_000
    ticks = np.maximum.accumulate(
        base
        + np.arange(200, dtype=np.int64) * 1_000_000
        + rng.integers(-5_000, 5_001, 200)
    )
    enc = fluxcode.encode_group(
        np.zeros(200),
        Params(time_error=0.05),
        block_len=100,
        times=ticks.view("datetime64[ns]"),
    )
    dec = fluxcode.decode_group(enc.payload)
    assert enc.block_start_times is not None and dec.times is not None
    np.testing.assert_array_equal(enc.block_start_times, dec.times[[0, 100]])
