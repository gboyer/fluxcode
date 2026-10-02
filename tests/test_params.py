# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Params validation, input validation and the precedence of min/max bits, noise floor and target."""

import numpy as np
import pytest
from _series import decode_series, encode_series, encode_unchecked, unit_rows
from _signals import KINDS, minute

from fluxcode import Params, _encoder


@pytest.mark.parametrize("kw", [{"min_quantize_bits": 0}, {"max_quantize_bits": 17},
                                {"min_quantize_bits": 10, "max_quantize_bits": 9}, {"min_quantize_bits": 6.0},
                                {"diff_orders": frozenset()}, {"diff_orders": {4}}, {"diff_orders": "0"},
                                {"noise_floor_sigma": -1}, {"noise_floor_sigma": float("nan")},
                                {"target_bits_per_sample": 0}, {"target_bits_per_sample": 5.99},
                                {"target_bits_per_sample": True}, {"planes": "bits"}, {"planes": True},
                                {"planes": None}])
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
    units, _, _, _ = encode_series(x)
    assert np.abs(decode_series(units) - x).max() < 1e-4


def exponents(x, params):
    units, lo, hi, _ = encode_series(x, params)
    rows = unit_rows(units[0])
    flags, param = rows.block_flags, rows.grid_params
    return flags, param, hi - lo


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("target", [None, 6.0, 8.0])
def test_min_max_bits_never_violated(kind, target):
    p = Params(min_quantize_bits=8, max_quantize_bits=12, noise_floor_sigma=4.0, target_bits_per_sample=target,
               decimal_detection=False)
    _, e, rng = exponents(minute(kind, 3), p)
    for b in range(len(e)):
        if rng[b] > 0:
            assert _encoder.exponent(rng[b], 12) <= e[b] <= _encoder.exponent(rng[b], 8)


def size(x, p):
    return 8 * sum(map(len, encode_series(x, p)[0])) / len(x)


@pytest.mark.parametrize("kind", ["chirp", "sin-50.3hz", "noisy-sine", "random-walk"])
def test_target_caps_expensive_signals(kind):
    x = minute(kind, 5)
    free = size(x, Params(noise_floor_sigma=None))
    capped = size(x, Params(noise_floor_sigma=None, target_bits_per_sample=6.0))
    assert capped < free
    assert capped < 6.0 + 1.0  # soft: the estimate is within about a bit


def test_target_leaves_cheap_units_alone():
    x = minute("sin-4.12hz", 6)
    assert encode_series(x, Params(target_bits_per_sample=8.0))[0] == encode_series(x)[0]


def test_target_equalizes_expensive_blocks():
    """Mixed unit: the expensive blocks give up bits, the cheap ones aren't touched."""
    x = np.concatenate([minute("chirp", 7)[:30_000], minute("linear", 7)[:30_000]])
    base = exponents(x, Params())[1]
    capped = exponents(x, Params(target_bits_per_sample=6.0))[1]  # mean estimate ~6.5
    assert (capped[:30] > base[:30]).all()
    np.testing.assert_array_equal(capped[30:], base[30:])


def kernel_units(x, target):
    """Targets below what Params allows (0: no target)."""
    return encode_unchecked(x, target_bits_per_sample=target or None)


def test_target_never_trades_a_decimal_grid_for_a_bigger_block():
    """Coarsening a 0.01-grid sine onto a power-of-two grid would grow it; the block keeps its grid."""
    x = np.round(minute("sin-4.12hz", 8), 2)
    assert kernel_units(x, 2.0) == kernel_units(x, 0.0)


def test_small_targets_can_grow_repeat_heavy_units():
    """Known limitation, pinned (why Params requires a target >= 6): the estimate reads the quadratic at
    ~5.5 bits/sample where zstd gets ~0.2, so a target of 2 coarsens it into a larger unit. If this
    starts failing, the target got smarter; revisit MIN_TARGET_BITS and the TODO in encode_unit."""
    x = minute("quadratic", 9)
    assert len(kernel_units(x, 2.0)) > len(kernel_units(x, 0.0))
    assert kernel_units(x, 6.0) == kernel_units(x, 0.0)  # under budget at the minimum target
