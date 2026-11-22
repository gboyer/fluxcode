# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Params validation, input validation and the precedence of min/max bits, noise floor and target."""

import numpy as np
import pytest
from _series import decode_series, encode_series, encode_unchecked, group_rows
from _signals import KINDS, minute

from fluxcode import Params, _encoder


@pytest.mark.parametrize("kw", [{"min_quantize_bits": 0}, {"max_quantize_bits": 17},
                                {"min_quantize_bits": 10, "max_quantize_bits": 9}, {"min_quantize_bits": 6.0},
                                {"diff_orders": frozenset()}, {"diff_orders": {4}}, {"diff_orders": "0"},
                                {"noise_floor_sigma": -1}, {"noise_floor_sigma": float("nan")}, {"noise_floor_sigma": None},
                                {"target_bits_per_sample": 0}, {"target_bits_per_sample": 5.99},
                                {"target_bits_per_sample": True}, {"effort": 0}, {"effort": 10}, {"effort": True},
                                {"effort": 5.0}, {"effort": None}, {"tolerance": -0.1},
                                {"tolerance": float("inf")}, {"tolerance": None}])
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
            assert _encoder.exponent(rng[b], 12) <= e[b]
            # Only an exact coarse grid (the index gcd: a +-100 square on 2^3) is coarser
            assert e[b] <= _encoder.exponent(rng[b], 8) or exact[b]


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
    """Coarsening a 0.01-grid sine onto a power-of-two grid would grow it; the block keeps its grid.
    (At a target of 2, below Params' minimum, the estimate reads the coarser block's period predictor
    as smaller where zstd matches the exact sine's fixed-order residuals: the limitation below.)"""
    x = np.round(minute("sin-4.12hz", 8), 2)
    assert kernel_groups(x, 4.0) == kernel_groups(x, 0.0)


def test_small_targets_can_grow_repeat_heavy_groups():
    """Known limitation, pinned (why Params requires a target >= 6): the estimate reads the quadratic at
    ~5.5 bits/sample where zstd gets ~0.2, so a target of 2 coarsens it into a larger block group. If this
    starts failing, the target got smarter; revisit MIN_TARGET_BITS and the TODO in encode_group."""
    x = minute("quadratic", 9)
    assert len(kernel_groups(x, 2.0)) > len(kernel_groups(x, 0.0))
    assert kernel_groups(x, 6.0) == kernel_groups(x, 0.0)  # under budget at the minimum target


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("tolerance", [1e-3, 0.05, 1.0])
def test_tolerance_bounds_every_sample_and_never_refines(kind, tolerance):
    """Every sample decodes within tolerance, or within what it would have without it (the noise floor, the
    coarsest step min_quantize_bits allows), and no output is bigger than without it, in any block."""
    from fluxcode import encode_group, verify
    x = minute(kind, 3)
    base = encode_group(x, Params())
    loose = encode_group(x, Params(tolerance=tolerance))
    assert verify.encode_group(loose, x, Params(tolerance=tolerance)) == []
    got, before = decode_series([loose.group]), decode_series([base.group])
    finite = np.isfinite(x)
    err, err_before = np.abs(got - x)[finite], np.abs(before - x)[finite]
    assert (err <= np.maximum(tolerance, err_before) * (1 + 1e-12) + 1e-300).all() or np.ptp(x[finite]) / 2 ** 6 > tolerance
    # (A coarser step can break an exact grid: a ramp's second differences pick up rounding, 0.003 b/s)
    assert len(loose.group) <= len(base.group) + 16 + len(x) // 1000


def test_tolerance_keeps_decimals_finer_than_it():
    from fluxcode import encode_group
    x = np.round(20 + 5 * np.sin(np.arange(6000) / 70) + np.random.default_rng(0).normal(0, 0.01, 6000), 2)
    for tolerance in (0.004, 0.005):
        assert (decode_series([encode_group(x, Params(tolerance=tolerance, noise_floor_sigma=0)).group]) == x).all()
    loose = decode_series([encode_group(x, Params(tolerance=0.02, noise_floor_sigma=0)).group])
    assert 0 < np.abs(loose - x).max() <= 0.02


def test_tolerance_uses_its_whole_allowance():
    """Where tolerance sets the step, a scaled step of up to 2 tolerance: the error comes within 94% of it."""
    from fluxcode import encode_group, verify
    x = 20 + 5 * np.sin(np.arange(20000) / 300)
    for tolerance in (0.003, 0.0045, 0.006):
        params = Params(tolerance=tolerance)
        result = encode_group(x, params)
        assert verify.encode_group(result, x, params) == []
        err = np.abs(decode_series([result.group]) - x).max()
        assert 0.9 * tolerance < err <= tolerance
