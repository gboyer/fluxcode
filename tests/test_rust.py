# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""The optional Rust accelerator (rust/, the fluxcode[rust] extra): its units are the Python path's,
byte for byte while both link the same libzstd (otherwise they differ but decode alike), it is safe
across threads, and fluxcode works, with the same units, without it."""

from concurrent.futures import ThreadPoolExecutor

import platform

import numpy as np
import pytest
import zstandard
from _signals import KINDS, discrete_minute, minute

import fluxcode
from fluxcode import Params, _unit
from fluxcode._types import MAX_EFFORT, MIN_EFFORT

fluxcode_rs = pytest.importorskip("fluxcode_rs", reason="the fluxcode[rust] extra isn't installed")
pytestmark = pytest.mark.skipif(_unit._compress_unit is None, reason="FLUXCODE_RUST=0")

SAME_ZSTD = fluxcode_rs.zstd_version() == zstandard.ZSTD_VERSION


def inputs():
    rng = np.random.default_rng(3)
    nonfinite = minute("sin-4.12hz", 1)[:5000].copy()
    nonfinite[rng.integers(0, 5000, 300)] = np.nan
    nonfinite[rng.integers(0, 5000, 50)] = np.inf
    nonfinite[7] = -np.inf
    yield from ((kind, minute(kind, 1)[:20_000], None) for kind in KINDS)
    yield "sensor-0.1", discrete_minute("sensor-0.1", 1)[:20_000], None
    yield "nonfinite", nonfinite, None
    yield "constant", np.full(3000, 3.25), None
    yield "variable blocks", minute("random-walk", 2)[:2000], [7, 3, 1000, 0, 5, 300, 64, 621]
    yield "short blocks", minute("chirp", 2)[:40], [1, 8, 9, 2, 20]


def encode(x, sizes, params, rust):
    saved = _unit._compress_unit
    if not rust:
        _unit._compress_unit = None
    try:
        if sizes is None:
            return fluxcode.encode(x, params).units
        return [fluxcode.encode_blocks(x[:sum(sizes)], sizes, params).unit]
    finally:
        _unit._compress_unit = saved


@pytest.mark.parametrize("effort", range(MIN_EFFORT, MAX_EFFORT + 1))
def test_rust_units_match_python_units(effort):
    assert _unit._compress_unit is fluxcode_rs.compress_unit
    params = Params(effort=effort)
    for name, x, sizes in inputs():
        python, rust = encode(x, sizes, params, False), encode(x, sizes, params, True)
        if SAME_ZSTD:
            assert rust == python, name
        else:
            # another libzstd may code the same body differently: the units still decode alike
            for unit_rust, unit_python in zip(rust, python):
                a, b = fluxcode.decode_unit(unit_rust), fluxcode.decode_unit(unit_python)
                assert np.array_equal(a.values, b.values, equal_nan=True), name
                assert abs(len(unit_rust) - len(unit_python)) <= 0.02 * len(unit_python) + 16, name


def test_rust_units_decode_to_the_encoded_values():
    x = minute("noisy-sine", 1)
    for unit in encode(x, None, Params(effort=5), True):
        decoded = fluxcode.decode_unit(unit)
        assert decoded.values.shape[0] == sum(decoded.block_sizes)


def test_threads_give_the_serial_units():
    series = [minute("random-walk", 1 + i)[:30_000] for i in range(4)] * 4
    params = Params(effort=5)
    serial = [encode(x, None, params, True) for x in series]
    with ThreadPoolExecutor(8) as pool:
        assert list(pool.map(lambda x: encode(x, None, params, True), series)) == serial


def test_time_axis_units_use_the_python_path():
    x = minute("sin-4.12hz", 1)[:5000]
    times = np.arange(x.shape[0]).astype("datetime64[ms]")
    unit = fluxcode.encode_blocks(x, [5000], times=times).unit
    decoded = fluxcode.decode_unit(unit)
    assert decoded.times is not None and decoded.times.shape[0] == 5000


def test_mismatched_rows_are_rejected():
    flags = np.zeros(2, np.uint8)
    sizes = np.array([3, 3], np.int64)
    with pytest.raises(ValueError, match="rows"):
        fluxcode_rs.compress_unit(
            flags, sizes, np.zeros(2, np.int64), np.zeros(2, np.int64), np.zeros(5, np.int16), np.zeros(6, np.uint8),
            "heuristic", True, [3],
        )


def test_unknown_layout_is_rejected():
    flags = np.zeros(1, np.uint8)
    with pytest.raises(ValueError, match="layout"):
        fluxcode_rs.compress_unit(
            flags, np.array([3], np.int64), np.zeros(1, np.int64), np.zeros(1, np.int64), np.zeros(3, np.int16),
            np.zeros(3, np.uint8), "size", True, [3],
        )


def test_bit_planes_use_the_vector_path_of_the_architecture():
    expected = {"arm64": "neon", "aarch64": "neon", "x86_64": "sse2", "AMD64": "sse2"}
    assert fluxcode_rs.simd_path() == expected.get(platform.machine(), "scalar")
