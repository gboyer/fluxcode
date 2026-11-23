# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Params validation, input validation and the precedence of min/max bits, noise floor and target."""

import numpy as np
import pytest
from _series import decode_series, encode_series, encode_unchecked, group_rows
from _signals import KINDS, minute

import fluxcode
from fluxcode import Params, _grids, verify


@pytest.mark.parametrize("kw", [{"min_quantize_bits": 0}, {"max_quantize_bits": 17},
                                {"min_quantize_bits": 10, "max_quantize_bits": 9}, {"min_quantize_bits": 6.0},
                                {"diff_orders": frozenset()}, {"diff_orders": {4}}, {"diff_orders": "0"},
                                {"noise_floor_sigma": -1}, {"noise_floor_sigma": float("nan")}, {"noise_floor_sigma": None},
                                {"target_bits_per_sample": 0}, {"target_bits_per_sample": 5.99},
                                {"target_bits_per_sample": True}, {"effort": 0}, {"effort": 10}, {"effort": True},
                                {"effort": 5.0}, {"effort": None}, {"absolute_tolerance": -0.1},
                                {"absolute_tolerance": float("nan")}, {"absolute_tolerance": float("inf")},
                                {"absolute_tolerance": True}])
def test_invalid(kw):
    with pytest.raises(ValueError):
        Params(**kw)


def test_valid_and_normalized():
    p = Params(diff_orders=[1, 2], noise_floor_sigma=0, min_quantize_bits=16)
    assert p.diff_orders == frozenset({1, 2})
    hash(p)


@pytest.mark.parametrize("x,msg", [(np.array([]), "non-empty"), (np.zeros((2, 3)), "1-D")])
def test_bad_input(x, msg):
    with pytest.raises(ValueError, match=msg):
        encode_series(x)


def test_accepts_lists_and_float32():
    assert decode_series(encode_series([1.5, 2.5, 3.5])[0]).tolist() == [1.5, 2.5, 3.5]
    x = np.linspace(0, 1, 5000, dtype=np.float32)
    groups, _, _, _ = encode_series(x)
    assert np.abs(decode_series(groups) - x).max() < 1e-4


def exponents(x, params):
    groups, lo, hi, _ = encode_series(x, params)
    rows = group_rows(groups[0])
    flags, param = rows.block_flags, rows.grid_params
    return flags, param, hi - lo


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("target", [None, 6.0, 8.0])
def test_min_max_bits_never_violated(kind, target):
    p = Params(min_quantize_bits=8, max_quantize_bits=12, noise_floor_sigma=4.0, target_bits_per_sample=target,
               grids="binary")
    x = minute(kind, 3)
    _, e, rng = exponents(x, p)
    exact = (decode_series(encode_series(x, p)[0]) == x).reshape(len(e), -1).all(1)
    for b in range(len(e)):
        if rng[b] > 0:
            assert _grids.exponent(rng[b], 12) <= e[b]
            # Only an exact coarse grid (the index gcd: a +-100 square on 2^3) is coarser
            assert e[b] <= _grids.exponent(rng[b], 8) or exact[b]


def size(x, p):
    return 8 * sum(map(len, encode_series(x, p)[0])) / len(x)


@pytest.mark.parametrize("kind", ["chirp", "sin-50.3hz", "noisy-sine", "random-walk"])
def test_target_caps_expensive_signals(kind):
    x = minute(kind, 5)
    free = size(x, Params(noise_floor_sigma=0))
    capped = size(x, Params(noise_floor_sigma=0, target_bits_per_sample=6.0))
    assert capped < free
    assert capped < 6.0 + 1.0  # soft: the estimate is within about a bit


def test_target_leaves_cheap_groups_alone():
    x = minute("sin-4.12hz", 6)
    assert encode_series(x, Params(target_bits_per_sample=8.0))[0] == encode_series(x)[0]


def test_target_equalizes_expensive_blocks():
    """Mixed block group: the expensive blocks give up bits, the cheap ones aren't touched."""
    x = np.concatenate([minute("chirp", 7)[:30_000], minute("linear", 7)[:30_000]])
    base = exponents(x, Params())[1]
    capped = exponents(x, Params(target_bits_per_sample=6.0))[1]  # mean estimate ~6.5
    assert (capped[:30] > base[:30]).all()
    np.testing.assert_array_equal(capped[30:], base[30:])


def kernel_groups(x, target):
    """Targets below what Params allows (0: no target)."""
    return encode_unchecked(x, target_bits_per_sample=target or None)


def test_target_never_trades_a_decimal_grid_for_a_bigger_block():
    """Coarsening a 0.01-grid sine onto a power-of-two grid would grow it; the block keeps its grid."""
    x = np.round(minute("sin-4.12hz", 8), 2)
    assert kernel_groups(x, 2.0) == kernel_groups(x, 0.0)


def test_small_targets_can_grow_repeat_heavy_groups():
    """Known limitation, pinned (why Params requires a target >= 6): the estimate reads the quadratic at
    ~5.5 bits/sample where zstd gets ~0.2, so a target of 2 coarsens it into a larger block group. If this
    starts failing, the target got smarter; revisit MIN_TARGET_BITS and the TODO in encode_group."""
    x = minute("quadratic", 9)
    assert len(kernel_groups(x, 2.0)) > len(kernel_groups(x, 0.0))
    assert kernel_groups(x, 6.0) == kernel_groups(x, 0.0)  # under budget at the minimum target


@pytest.mark.parametrize("grids", ["binary", "decimal", "rational"])
def test_absolute_tolerance_coarsens_and_is_fixed_point(grids):
    """absolute_tolerance coarsens steps up to 2 * absolute_tolerance, including past min_quantize_bits
    on quiet blocks, and decoded data re-encodes as a fixed point."""
    rng = np.random.default_rng(17)
    # 1) Moderate range: 0..100, absolute_tolerance=0.3 (step <= 0.6)
    x = 50.0 + 50.0 * np.sin(np.arange(2000) / 150.0) + rng.normal(0, 0.01, 2000)
    params = Params(noise_floor_sigma=0, grids=grids, absolute_tolerance=0.3)
    res = fluxcode.encode_group(x, params)
    y = fluxcode.decode_group(res.group).values
    assert np.abs(y - x).max() <= 0.3 + 1e-12
    assert verify.encode_group(res, x, params) == []
    assert verify.fixed_point(res.group, params) == []

    # 2) Quiet block spanning 0.1 °C with absolute_tolerance=0.5: coarsens past min_quantize_bits (6)
    quiet = 20.0 + 0.1 * np.sin(np.arange(1000) / 80.0)
    p_quiet = Params(min_quantize_bits=6, noise_floor_sigma=0, grids=grids, absolute_tolerance=0.5)
    res_q = fluxcode.encode_group(quiet, p_quiet)
    y_q = fluxcode.decode_group(res_q.group).values
    assert np.abs(y_q - quiet).max() <= 0.5 + 1e-12
    assert np.unique(y_q).size <= 2  # NOT forced to 64 levels!
    assert verify.encode_group(res_q, quiet, p_quiet) == []
    assert verify.fixed_point(res_q.group, p_quiet) == []

    # 3) Quiet block straddling a midpoint [20.45, 20.55] with absolute_tolerance=0.6 (scaled under rational)
    straddle = np.linspace(20.45, 20.55, 1000)
    p_straddle = Params(min_quantize_bits=6, noise_floor_sigma=0, grids=grids, absolute_tolerance=0.6)
    res_s = fluxcode.encode_group(straddle, p_straddle)
    y_s = fluxcode.decode_group(res_s.group).values
    assert np.abs(y_s - straddle).max() <= 0.6 + 1e-12
    assert verify.encode_group(res_s, straddle, p_straddle) == []
    assert verify.fixed_point(res_s.group, p_straddle) == []

    # 4) Short block (<= 8 samples) honours absolute_tolerance
    short = np.array([10.0, 10.03, 10.07, 10.09])
    res_short = fluxcode.encode_blocks(short, [4], p_quiet)
    assert np.abs(fluxcode.decode_group(res_short.group).values - short).max() <= 0.5 + 1e-12
    assert verify.encode_blocks(res_short, short, [4], p_quiet) == []
    assert verify.fixed_point(res_short.group, p_quiet) == []


def test_absolute_tolerance_huge_value_is_safe():
    """Huge absolute_tolerance (near DBL_MAX) does not overflow and remains a fixed point."""
    x = np.linspace(1.0, 2.0, 1000)
    params = Params(noise_floor_sigma=0, absolute_tolerance=1e308)
    res = fluxcode.encode_group(x, params)
    assert verify.encode_group(res, x, params) == []
    assert verify.fixed_point(res.group, params) == []
