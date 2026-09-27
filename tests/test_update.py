# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""update: untouched blocks decode identically; without a target, update == encoding the new data."""

import numpy as np
import pytest
import zstandard
from _series import encode_series, unit_rows
from _signals import minute

import fluxcode
from fluxcode import Params, _format

L = 1000


def encoded(kind="random-walk", seed=41, n=60_000, params=Params()):
    x = minute(kind, seed)[:n]
    units, lo, hi, mean = encode_series(x, params)
    return x, units[0], lo, hi, mean


def rebuilt(unit, edit):
    """unit with its body rows changed by edit(head, param, anchor, resid, codes, time_rows), same header."""
    rows = unit_rows(unit)
    edit(*rows)
    body = _format.write_unit(*rows[:5], time_rows=rows.time_rows)
    return unit[:_format.HEADER_BYTES] + zstandard.ZstdCompressor(level=3).compress(body.tobytes())


@pytest.mark.parametrize("indices", [[0], [59], [3, 17, 42], list(range(60)), [10, 5]])
def test_replace_matches_reencode(indices):
    x, unit, _, _, _ = encoded()
    blocks = minute("chirp", 42).reshape(60, L)[: len(indices)]
    unit2, lo2, hi2, mean2 = fluxcode.update(unit, indices, blocks)
    x2 = x.copy().reshape(60, L)
    x2[indices] = blocks
    ref_units, ref_lo, ref_hi, ref_mean = encode_series(x2.ravel())
    assert unit2 == ref_units[0]
    np.testing.assert_array_equal(lo2, ref_lo[indices])
    np.testing.assert_array_equal(hi2, ref_hi[indices])
    np.testing.assert_array_equal(mean2, ref_mean[indices])


def test_untouched_blocks_identical():
    _, unit, _, _, _ = encoded("noisy-sine")
    before = fluxcode.decode_unit(unit).values.reshape(-1, L)
    idx = [7, 30]
    unit2, _, _, _ = fluxcode.update(unit, idx, np.zeros((2, L)) + [[1.5], [-2.0]])
    after = fluxcode.decode_unit(unit2).values.reshape(-1, L)
    keep = np.setdiff1d(np.arange(60), idx)
    np.testing.assert_array_equal(after[keep], before[keep])
    np.testing.assert_array_equal(after[idx], [np.full(L, 1.5), np.full(L, -2.0)])


def test_untouched_blocks_identical_with_target():
    """With a target, re-encoding the whole unit would reallocate; update carries residuals over."""
    p = Params(noise_floor_sigma=None, target_bits_per_sample=6.0)
    _, unit, _, _, _ = encoded("noisy-sine", params=p)
    before = fluxcode.decode_unit(unit).values.reshape(-1, L)
    unit2, _, _, _ = fluxcode.update(unit, [0], minute("chirp", 1)[:L][None], p)
    np.testing.assert_array_equal(fluxcode.decode_unit(unit2).values.reshape(-1, L)[1:], before[1:])


def test_append():
    x, unit, lo, _, _ = encoded(n=20_000)
    new = minute("sin-4.12hz", 43)[:3 * L].reshape(3, L)
    unit2, lo2, hi2, _ = fluxcode.update(unit, [20, 21, 22], new)
    assert lo2.shape == hi2.shape == (3,)
    ref_units, ref_lo, _, _ = encode_series(np.concatenate([x, new.ravel()]))
    assert unit2 == ref_units[0]
    np.testing.assert_array_equal(np.concatenate([lo, lo2]), ref_lo)
    assert fluxcode.decode_unit(unit2).values.shape == (23 * L,)


def test_replace_and_append_together():
    x, unit, _, _, _ = encoded(n=10_000)
    new = minute("chirp", 44)[:2 * L].reshape(2, L)
    unit2, _, _, _ = fluxcode.update(unit, [10, 4], new)
    x2 = np.concatenate([x, new[0]]).reshape(11, L)
    x2[4] = new[1]
    assert unit2 == encode_series(x2.ravel())[0][0]


def test_partial_last_block():
    """The sample count follows the blocks: a partial last block stays partial until it's replaced."""
    x, unit, _, _, _ = encoded(n=10_500)
    new = np.ones((1, L))
    unit2, _, _, _ = fluxcode.update(unit, [3], new)  # an earlier block: still 10,500 samples
    assert fluxcode.decode_unit(unit2).values.shape == (10_500,)
    with pytest.raises(ValueError, match="partial"):
        fluxcode.update(unit, [11], new)  # appending after the partial block 10
    unit3, _, _, _ = fluxcode.update(unit, [10, 11], np.ones((2, L)))  # replace it, then append
    y = fluxcode.decode_unit(unit3).values
    assert y.shape == (12 * L,)
    np.testing.assert_array_equal(y[10 * L:], 1.0)
    ref = x.copy()
    ref = np.concatenate([ref[:10 * L], np.ones(2 * L)])
    assert unit3 == encode_series(ref)[0][0]


@pytest.mark.parametrize("indices,msg", [([61], "without gaps"), ([60, 60], "distinct"), ([-1], ">= 0"),
                                         ([], "non-empty"), ([1.0], "integer")])
def test_bad_indices(indices, msg):
    _, unit, _, _, _ = encoded()
    with pytest.raises(ValueError, match=msg):
        fluxcode.update(unit, np.array(indices), np.zeros((len(indices), L)))


def test_append_past_blocks_per_unit():
    _, unit, _, _, _ = encoded()
    with pytest.raises(ValueError, match="blocks_per_unit"):
        fluxcode.update(unit, [60], np.zeros((1, L)))


def test_wrong_shapes():
    _, unit, _, _, _ = encoded()
    with pytest.raises(ValueError, match="shape"):
        fluxcode.update(unit, [1, 2], np.zeros((2, 999)))
    with pytest.raises(ValueError, match="block_len"):
        fluxcode.update(unit, [1], np.zeros((1, 500)), Params(block_len=500))


def test_update_with_non_finite_blocks():
    """New blocks may hold NaN/inf; flagged blocks already in the unit keep their codes."""
    _, unit, _, _, _ = encoded(n=20_000)
    b = np.zeros((2, L))
    b[1, 5], b[1, 6] = np.nan, -np.inf
    unit2, _, _, _ = fluxcode.update(unit, [0, 1], b)
    y = fluxcode.decode_unit(unit2).values.reshape(-1, L)
    np.testing.assert_array_equal(y[:2], b)
    unit3, _, _, _ = fluxcode.update(unit2, [10, 20], np.ones((2, L)))  # replace one, append one
    y3 = fluxcode.decode_unit(unit3).values.reshape(-1, L)
    np.testing.assert_array_equal(y3[:2], b)
    np.testing.assert_array_equal(y3[2:10], y[2:10])
    np.testing.assert_array_equal(y3[[10, 20]], np.ones((2, L)))


@pytest.mark.parametrize("head_bits", [0x10, 0x20, 0x40, 0x80])
def test_update_refuses_units_it_cannot_read(head_bits):
    """A unit with reserved head bits (a future feature), or the irregular time bit (0x10) without a
    time axis, isn't rewritten: that could drop what they mean."""
    _, unit, _, _, _ = encoded()
    raw_body = _format.write_unit(*unit_rows(unit)[:5])
    raw_body[5] |= head_bits  # block_flags are the body's first bytes
    bad = unit[:_format.HEADER_BYTES] + zstandard.ZstdCompressor(level=3).compress(raw_body.tobytes())
    with pytest.raises(ValueError, match="block 5: head byte"):
        fluxcode.update(bad, [0], np.zeros((1, L)))
    with pytest.raises(ValueError, match="block 5: head byte"):
        fluxcode.decode_unit(bad)


def test_update_refuses_out_of_range_parameters():
    _, unit, _, _, _ = encoded()

    def bad_param(head, param, *_):
        param[2] = 5000

    with pytest.raises(ValueError, match="block 2: parameter"):
        fluxcode.update(rebuilt(unit, bad_param), [0], np.zeros((1, L)))


@pytest.mark.parametrize("anchor", [np.inf, -np.inf, np.nan])
def test_non_finite_float_anchors_are_rejected(anchor):
    _, unit, _, _, _ = encoded()

    def bad_anchor(head, param, anchors, *_):
        anchors.view(np.float64)[4] = anchor

    with pytest.raises(ValueError, match="block 4: anchor"):
        fluxcode.decode_unit(rebuilt(unit, bad_anchor))


def test_out_of_range_decimal_anchors_are_rejected():
    x = np.round(np.cumsum(np.random.default_rng(3).normal(size=3 * L)), 2)
    unit, _, _, _ = fluxcode.encode_unit(x, Params(noise_floor_sigma=None))
    assert (unit_rows(unit)[0] & _format.HEAD_DECIMAL).all()

    def bad_anchor(head, param, anchors, *_):
        anchors[1] = 1 << 52

    with pytest.raises(ValueError, match="block 1: anchor"):
        fluxcode.decode_unit(rebuilt(unit, bad_anchor))
