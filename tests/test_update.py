# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""update: untouched blocks decode identically; without a target, update == encoding the new data."""

import numpy as np
import pytest
import zstandard
from _series import encode_series, unit_rows
from _signals import minute

import fluxcode
from fluxcode import Params, _bitpacking, _format, _unit

L = 1000


def encoded(kind="random-walk", seed=41, n=60_000, params=Params()):
    x = minute(kind, seed)[:n]
    units, lo, hi, mean = encode_series(x, params)
    return x, units[0], lo, hi, mean


def rebuilt(unit, edit):
    """unit with its body rows changed by edit(flags, sizes, param, anchor, resid, codes, time_rows), same header."""
    rows = unit_rows(unit)
    edit(*rows)
    body = _bitpacking.write_unit(*rows[:6], time_rows=rows.time_rows)
    return unit[:_format.HEADER_BYTES] + zstandard.ZstdCompressor(level=3).compress(body.tobytes())


def by_index(indices, blocks):
    return dict(zip(indices, blocks))


@pytest.mark.parametrize("indices", [[0], [59], [3, 17, 42], list(range(60)), [10, 5]])
def test_replace_matches_reencode(indices):
    x, unit, _, _, _ = encoded()
    blocks = minute("chirp", 42).reshape(60, L)[: len(indices)]
    unit2, updated, lo2, hi2, mean2 = fluxcode.update(unit, by_index(indices, blocks))
    assert updated.tolist() == sorted(indices)
    x2 = x.copy().reshape(60, L)
    x2[indices] = blocks
    ref_units, ref_lo, ref_hi, ref_mean = encode_series(x2.ravel())
    assert unit2 == ref_units[0]
    np.testing.assert_array_equal(lo2, ref_lo[updated])
    np.testing.assert_array_equal(hi2, ref_hi[updated])
    np.testing.assert_array_equal(mean2, ref_mean[updated])


def test_untouched_blocks_identical():
    _, unit, _, _, _ = encoded("noisy-sine")
    before = fluxcode.decode_unit(unit).values.reshape(-1, L)
    idx = [7, 30]
    unit2 = fluxcode.update(unit, {7: np.full(L, 1.5), 30: np.full(L, -2.0)}).unit
    after = fluxcode.decode_unit(unit2).values.reshape(-1, L)
    keep = np.setdiff1d(np.arange(60), idx)
    np.testing.assert_array_equal(after[keep], before[keep])
    np.testing.assert_array_equal(after[idx], [np.full(L, 1.5), np.full(L, -2.0)])


def test_update_copies_carried_blocks_without_unpacking(monkeypatch):
    """Carried blocks' bytes are copied: update never unpacks every block's residuals, codes or
    time residuals."""
    t = np.datetime64("2026-01-01", "ms") + np.arange(60 * L) * np.timedelta64(100, "ms")
    t[5] += np.timedelta64(1, "ms")  # block 0 irregular: it has time residual planes
    x, _, _, _, _ = encoded()
    unit = fluxcode.encode_unit(x, times=t).unit
    expected = fluxcode.update(unit, {1: x[L:2 * L] + 1}, times={1: t[L:2 * L]}).unit

    def unpack_everything(*args, **kwargs):
        raise AssertionError("unpacked every block")

    monkeypatch.setattr(_bitpacking, "read_rows", unpack_everything)
    monkeypatch.setattr(_bitpacking, "unshuffle_block", unpack_everything)
    monkeypatch.setattr(_unit, "read_rows", unpack_everything)
    assert fluxcode.update(unit, {1: x[L:2 * L] + 1}, times={1: t[L:2 * L]}).unit == expected


def test_untouched_blocks_identical_with_target():
    """With a target, re-encoding the whole unit would reallocate; update carries residuals over."""
    p = Params(noise_floor_sigma=None, target_bits_per_sample=6.0)
    _, unit, _, _, _ = encoded("noisy-sine", params=p)
    before = fluxcode.decode_unit(unit).values.reshape(-1, L)
    unit2 = fluxcode.update(unit, {0: minute("chirp", 1)[:L]}, p).unit
    np.testing.assert_array_equal(fluxcode.decode_unit(unit2).values.reshape(-1, L)[1:], before[1:])


def test_append():
    x, unit, lo, _, _ = encoded(n=20_000)
    new = minute("sin-4.12hz", 43)[:3 * L].reshape(3, L)
    unit2, _, lo2, hi2, _ = fluxcode.update(unit, by_index([20, 21, 22], new))
    assert lo2.shape == hi2.shape == (3,)
    ref_units, ref_lo, _, _ = encode_series(np.concatenate([x, new.ravel()]))
    assert unit2 == ref_units[0]
    np.testing.assert_array_equal(np.concatenate([lo, lo2]), ref_lo)
    assert fluxcode.decode_unit(unit2).values.shape == (23 * L,)


def test_replace_and_append_together():
    x, unit, _, _, _ = encoded(n=10_000)
    new = minute("chirp", 44)[:2 * L].reshape(2, L)
    unit2 = fluxcode.update(unit, {10: new[0], 4: new[1]}).unit
    x2 = np.concatenate([x, new[0]]).reshape(11, L)
    x2[4] = new[1]
    assert unit2 == encode_series(x2.ravel())[0][0]


def test_blocks_change_size():
    """A block can be replaced by one of any size, emptied, or appended past a gap (filled with
    empty blocks); the result equals encoding the new blocks from scratch."""
    x, unit, _, _, _ = encoded(n=10_500)  # a short last block of 500
    sizes = [L] * 10 + [500]
    new = {3: np.arange(7.0), 5: np.zeros(0), 10: minute("chirp", 3)[:2 * L], 13: np.ones(9)}
    unit2, updated, lo, _, mean = fluxcode.update(unit, new)
    assert updated.tolist() == [3, 5, 10, 11, 12, 13]
    blocks = [x[b * L:(b + 1) * L] for b in range(11)]
    blocks += [np.zeros(0), np.zeros(0), np.zeros(0)]
    for idx, block in new.items():
        blocks[idx] = block
    ref = fluxcode.encode_blocks(np.concatenate(blocks), [len(b) for b in blocks])
    assert unit2 == ref.unit
    np.testing.assert_array_equal(lo, ref.block_min[updated])
    assert np.isnan(mean[[1, 3, 4]]).all()  # empty blocks: NaN statistics
    decoded = fluxcode.decode_unit(unit2)
    assert decoded.block_sizes.tolist() == [L, L, L, 7, L, 0, L, L, L, L, 2 * L, 0, 0, 9]
    assert sizes  # (the original sizes, for reference)


def test_empty_update_returns_the_unit():
    _, unit, _, _, _ = encoded(n=3 * L)
    updated = fluxcode.update(unit, {})
    assert updated.unit == unit and updated.indices.shape == (0,)


@pytest.mark.parametrize("blocks,msg", [({-1: np.zeros(L)}, ">= 0"), ({1.0: np.zeros(L)}, "integers"),
                                        ([np.zeros(L)], "map block indices"), ({1: np.zeros((2, L))}, "1-D"),
                                        ({1: np.zeros(65_536)}, "0 to 65535")])
def test_bad_blocks(blocks, msg):
    _, unit, _, _, _ = encoded()
    with pytest.raises(ValueError, match=msg):
        fluxcode.update(unit, blocks)


@pytest.mark.parametrize("index", [65_535, 10**10, 2**63, 2**80])
def test_append_past_block_limit(index):
    """Rejected before anything is sized by the index (10^10 used to exhaust memory)."""
    _, unit, _, _, _ = encoded(n=3 * L)
    with pytest.raises(ValueError, match="65535 blocks"):
        fluxcode.update(unit, {index: np.zeros(1)})


def test_update_with_non_finite_blocks():
    """New blocks may hold NaN/inf; flagged blocks already in the unit keep their codes."""
    _, unit, _, _, _ = encoded(n=20_000)
    b = np.zeros((2, L))
    b[1, 5], b[1, 6] = np.nan, -np.inf
    unit2 = fluxcode.update(unit, by_index([0, 1], b)).unit
    y = fluxcode.decode_unit(unit2).values.reshape(-1, L)
    np.testing.assert_array_equal(y[:2], b)
    unit3 = fluxcode.update(unit2, {10: np.ones(L), 20: np.ones(L)}).unit  # replace one, append one
    y3 = fluxcode.decode_unit(unit3).values.reshape(-1, L)
    np.testing.assert_array_equal(y3[:2], b)
    np.testing.assert_array_equal(y3[2:10], y[2:10])
    np.testing.assert_array_equal(y3[[10, 20]], np.ones((2, L)))


@pytest.mark.parametrize("head_bits", [0x10, 0x20, 0x40, 0x80])
def test_update_refuses_units_it_cannot_read(head_bits):
    """A unit with reserved flags bits (a future feature), or the irregular time bit (0x10) without a
    time axis, isn't rewritten: that could drop what they mean."""
    _, unit, _, _, _ = encoded()
    raw_body = _bitpacking.write_unit(*unit_rows(unit)[:6])
    raw_body[5] |= head_bits  # block_flags are the body's first bytes
    bad = unit[:_format.HEADER_BYTES] + zstandard.ZstdCompressor(level=3).compress(raw_body.tobytes())
    with pytest.raises(ValueError, match="block 5: block flags"):
        fluxcode.update(bad, {0: np.zeros(L)})
    with pytest.raises(ValueError, match="block 5: block flags"):
        fluxcode.decode_unit(bad)


def test_update_refuses_out_of_range_parameters():
    _, unit, _, _, _ = encoded()

    def bad_param(flags, sizes, param, *_):
        param[2] = 5000

    with pytest.raises(ValueError, match="block 2: parameter"):
        fluxcode.update(rebuilt(unit, bad_param), {0: np.zeros(L)})


@pytest.mark.parametrize("anchor", [np.inf, -np.inf, np.nan])
def test_non_finite_float_anchors_are_rejected(anchor):
    _, unit, _, _, _ = encoded()

    def bad_anchor(flags, sizes, param, anchors, *_):
        anchors.view(np.float64)[4] = anchor

    with pytest.raises(ValueError, match="block 4: anchor"):
        fluxcode.decode_unit(rebuilt(unit, bad_anchor))


def test_out_of_range_decimal_anchors_are_rejected():
    x = np.round(np.cumsum(np.random.default_rng(3).normal(size=3 * L)), 2)
    unit, _, _, _ = fluxcode.encode_unit(x, Params(noise_floor_sigma=None))
    assert (unit_rows(unit).block_flags & _format.BLOCK_FLAG_DECIMAL).all()

    def bad_anchor(flags, sizes, param, anchors, *_):
        anchors[1] = 1 << 52

    with pytest.raises(ValueError, match="block 1: anchor"):
        fluxcode.decode_unit(rebuilt(unit, bad_anchor))
