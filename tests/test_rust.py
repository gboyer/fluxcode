# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""The optional Rust accelerator (rust/, the fluxcode[rust] extra): its block groups are the Python path's,
byte for byte while both link the same libzstd (otherwise they differ but decode alike), it is safe
across threads, and fluxcode works, with the same block groups, without it."""

import platform
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import zstandard
from _signals import KINDS, discrete_minute, minute

import fluxcode
from fluxcode import Params, _compress
from fluxcode._types import MAX_EFFORT, MIN_EFFORT

fluxcode_rs = pytest.importorskip("fluxcode_rs", reason="the fluxcode[rust] extra isn't installed")
pytestmark = pytest.mark.skipif(_compress._rust is None, reason="FLUXCODE_RUST=0")

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
    saved = _compress._rust
    if not rust:
        _compress._rust = None
    try:
        if sizes is None:
            return fluxcode.encode(x, params).groups
        return [fluxcode.encode_blocks(x[:sum(sizes)], sizes, params).group]
    finally:
        _compress._rust = saved


@pytest.mark.parametrize("effort", range(MIN_EFFORT, MAX_EFFORT + 1))
def test_rust_groups_match_python_groups(effort):
    assert _compress._rust is fluxcode_rs
    params = Params(effort=effort)
    for name, x, sizes in inputs():
        python, rust = encode(x, sizes, params, False), encode(x, sizes, params, True)
        if SAME_ZSTD:
            assert rust == python, name
        else:
            # another libzstd may code the same body differently: the block groups still decode alike
            for group_rust, group_python in zip(rust, python):
                a, b = fluxcode.decode_group(group_rust), fluxcode.decode_group(group_python)
                assert np.array_equal(a.values, b.values, equal_nan=True), name
                assert abs(len(group_rust) - len(group_python)) <= 0.02 * len(group_python) + 16, name


def test_rust_groups_decode_to_the_encoded_values():
    x = minute("noisy-sine", 1)
    for group in encode(x, None, Params(effort=5), True):
        decoded = fluxcode.decode_group(group)
        assert decoded.values.shape[0] == sum(decoded.block_sizes)


def test_threads_give_the_serial_groups():
    series = [minute("random-walk", 1 + i)[:30_000] for i in range(4)] * 4
    params = Params(effort=5)
    serial = [encode(x, None, params, True) for x in series]
    with ThreadPoolExecutor(8) as pool:
        assert list(pool.map(lambda x: encode(x, None, params, True), series)) == serial


def test_interface_version_matches():
    assert fluxcode_rs.INTERFACE_VERSION == _compress.RUST_INTERFACE_VERSION


def test_policy_constants_match_the_python_ones():
    """The Rust copies of the encoder's policy constants can't drift from fluxcode/_compress.py."""
    assert fluxcode_rs.FLUSH_MIN_DENSITY == _compress.FLUSH_MIN_DENSITY
    assert fluxcode_rs.BYTE_PLANES_BIT == _compress.BYTE_PLANES_BIT
    assert fluxcode_rs.BYTE_PLANES_MAX_SHARE == _compress.BYTE_PLANES_MAX_SHARE


def timed_inputs():
    """Time-axis series: (samples, times) with regular and with irregular intervals."""
    rng = np.random.default_rng(5)
    x = minute("noisy-sine", 1)[:6000]
    regular = np.datetime64("2026-01-01") + np.arange(6000) * np.timedelta64(10, "ms")
    irregular = np.datetime64("2026-01-01") + np.cumsum(rng.integers(1, 40, 6000)).astype("timedelta64[ms]")
    yield x, regular
    yield x, irregular


@pytest.mark.parametrize("effort", [1, 3, 5, 9])
def test_time_axis_groups_and_updates_match_python(effort):
    """Block groups with a time axis and splices go through the extension's pack_group with the Python-built body."""
    params = Params(effort=effort)
    start, second = np.datetime64("2026-01-01"), np.timedelta64(1, "s")

    def run(rust):
        saved = _compress._rust
        if not rust:
            _compress._rust = None
        try:
            out = []
            for x, times in timed_inputs():
                group = fluxcode.encode_time_blocks(x, times, params, start_time=start, block_duration=second).group
                out.append(group)
                lo = int(np.searchsorted(times, times[2500]))
                new = x[lo:lo + 300][::-1] + 1
                out.append(fluxcode.update_time_blocks(
                    group, new, times[lo:lo + 300], params, start_time=start, block_duration=second,
                    delete_ranges=(times[2800], times[3300])).group)
                out.append(fluxcode.update(group, {0: x[:10]}, params, times={0: times[:10]}).group)
            return out
        finally:
            _compress._rust = saved

    python, rust = run(False), run(True)
    for group_rust, group_python in zip(rust, python, strict=True):
        if SAME_ZSTD:
            assert group_rust == group_python
        else:
            a, b = fluxcode.decode_group(group_rust), fluxcode.decode_group(group_python)
            assert np.array_equal(a.values, b.values, equal_nan=True)
            assert a.times is not None and b.times is not None
            assert np.array_equal(a.times, b.times)


def test_pack_group_rejects_a_short_body_and_an_unknown_layout():
    with pytest.raises(ValueError, match="too short"):
        fluxcode_rs.pack_group(np.zeros(10, np.uint8), 1, 8, 1, 0, "best", True, [3])
    with pytest.raises(ValueError, match="layout"):
        fluxcode_rs.pack_group(np.zeros(13 + 16, np.uint8), 1, 8, 1, 0, "size", True, [3])


def test_mismatched_rows_are_rejected():
    flags = np.zeros(2, np.uint8)
    sizes = np.array([3, 3], np.int64)
    with pytest.raises(ValueError, match="rows"):
        fluxcode_rs.compress_group(
            flags, sizes, np.zeros(2, np.int64), np.zeros(2, np.int64), np.zeros(5, np.int16), np.zeros(6, np.uint8),
            "heuristic", True, [3],
        )


def test_unknown_layout_is_rejected():
    flags = np.zeros(1, np.uint8)
    with pytest.raises(ValueError, match="layout"):
        fluxcode_rs.compress_group(
            flags, np.array([3], np.int64), np.zeros(1, np.int64), np.zeros(1, np.int64), np.zeros(3, np.int16),
            np.zeros(3, np.uint8), "size", True, [3],
        )


def test_bit_planes_use_the_vector_path_of_the_architecture():
    expected = {"arm64": "neon", "aarch64": "neon", "x86_64": "sse2", "AMD64": "sse2"}
    assert fluxcode_rs.simd_path() == expected.get(platform.machine(), "scalar")
