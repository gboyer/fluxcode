# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""update: untouched blocks decode identically; without a target, update == encoding the new data."""

import _oracle
import numpy as np
import pytest
import zstandard
from _series import encode_series, group_rows
from _signals import minute

import fluxcode
from fluxcode import Params, _bitpacking, _format

L = 1000


def encoded(kind="random-walk", seed=41, n=60_000, params=Params()):
    x = minute(kind, seed)[:n]
    groups, lo, hi, mean = encode_series(x, params)
    return x, groups[0], lo, hi, mean


def rebuilt(group, edit):
    """block group with its body rows changed by edit(flags, sizes, param, anchor, resid, codes, time_rows), same header."""
    rows = group_rows(group)
    edit(*rows)
    body = _oracle.write_group(*rows[:6], time_rows=rows.time_rows)
    return group[:_format.HEADER_BYTES] + zstandard.ZstdCompressor(level=3).compress(body.tobytes())


def by_index(indices, blocks):
    return dict(zip(indices, blocks))


@pytest.mark.parametrize("indices", [[0], [59], [3, 17, 42], list(range(60)), [10, 5]])
def test_replace_matches_reencode(indices):
    x, group, _, _, _ = encoded()
    blocks = minute("chirp", 42).reshape(60, L)[: len(indices)]
    group2, updated, lo2, hi2, mean2 = fluxcode.update(group, by_index(indices, blocks))
    assert updated.tolist() == sorted(indices)
    x2 = x.copy().reshape(60, L)
    x2[indices] = blocks
    ref_groups, ref_lo, ref_hi, ref_mean = encode_series(x2.ravel())
    assert group2 == ref_groups[0]
    np.testing.assert_array_equal(lo2, ref_lo[updated])
    np.testing.assert_array_equal(hi2, ref_hi[updated])
    np.testing.assert_array_equal(mean2, ref_mean[updated])


def test_untouched_blocks_identical():
    _, group, _, _, _ = encoded("noisy-sine")
    before = fluxcode.decode_group(group).values.reshape(-1, L)
    idx = [7, 30]
    group2 = fluxcode.update(group, {7: np.full(L, 1.5), 30: np.full(L, -2.0)}).group
    after = fluxcode.decode_group(group2).values.reshape(-1, L)
    keep = np.setdiff1d(np.arange(60), idx)
    np.testing.assert_array_equal(after[keep], before[keep])
    np.testing.assert_array_equal(after[idx], [np.full(L, 1.5), np.full(L, -2.0)])


def test_update_copies_carried_blocks_without_unpacking(monkeypatch):
    """Carried blocks' bytes are copied: update never unpacks every block's residuals, codes or
    time residuals."""
    t = np.datetime64("2026-01-01", "ms") + np.arange(60 * L) * np.timedelta64(100, "ms")
    t[5] += np.timedelta64(1, "ms")  # block 0 irregular: it has time residual planes
    x, _, _, _, _ = encoded()
    group = fluxcode.encode_group(x, times=t).group
    expected = fluxcode.update(group, {1: x[L:2 * L] + 1}, times={1: t[L:2 * L]}).group

    def unpack_everything(*args, **kwargs):
        raise AssertionError("unpacked every block")

    monkeypatch.setattr(_bitpacking, "unshuffle_block", unpack_everything)
    assert fluxcode.update(group, {1: x[L:2 * L] + 1}, times={1: t[L:2 * L]}).group == expected


def test_untouched_blocks_identical_with_target():
    """With a target, re-encoding the whole block group would reallocate; update carries residuals over."""
    p = Params(noise_floor_sigma=0, target_bits_per_sample=6.0)
    _, group, _, _, _ = encoded("noisy-sine", params=p)
    before = fluxcode.decode_group(group).values.reshape(-1, L)
    group2 = fluxcode.update(group, {0: minute("chirp", 1)[:L]}, p).group
    np.testing.assert_array_equal(fluxcode.decode_group(group2).values.reshape(-1, L)[1:], before[1:])


def test_append():
    x, group, lo, _, _ = encoded(n=20_000)
    new = minute("sin-4.12hz", 43)[:3 * L].reshape(3, L)
    group2, _, lo2, hi2, _ = fluxcode.update(group, by_index([20, 21, 22], new))
    assert lo2.shape == hi2.shape == (3,)
    ref_groups, ref_lo, _, _ = encode_series(np.concatenate([x, new.ravel()]))
    assert group2 == ref_groups[0]
    np.testing.assert_array_equal(np.concatenate([lo, lo2]), ref_lo)
    assert fluxcode.decode_group(group2).values.shape == (23 * L,)


def test_replace_and_append_together():
    x, group, _, _, _ = encoded(n=10_000)
    new = minute("chirp", 44)[:2 * L].reshape(2, L)
    group2 = fluxcode.update(group, {10: new[0], 4: new[1]}).group
    x2 = np.concatenate([x, new[0]]).reshape(11, L)
    x2[4] = new[1]
    assert group2 == encode_series(x2.ravel())[0][0]


def test_blocks_change_size():
    """A block can be replaced by one of any size, emptied, or appended past a gap (filled with
    empty blocks); the result equals encoding the new blocks from scratch."""
    x, group, _, _, _ = encoded(n=10_500)  # a short last block of 500
    sizes = [L] * 10 + [500]
    new = {3: np.arange(7.0), 5: np.zeros(0), 10: minute("chirp", 3)[:2 * L], 13: np.ones(9)}
    group2, updated, lo, _, mean = fluxcode.update(group, new)
    assert updated.tolist() == [3, 5, 10, 11, 12, 13]
    blocks = [x[b * L:(b + 1) * L] for b in range(11)]
    blocks += [np.zeros(0), np.zeros(0), np.zeros(0)]
    for idx, block in new.items():
        blocks[idx] = block
    ref = fluxcode.encode_blocks(np.concatenate(blocks), [len(b) for b in blocks])
    assert group2 == ref.group
    np.testing.assert_array_equal(lo, ref.block_min[updated])
    assert np.isnan(mean[[1, 3, 4]]).all()  # empty blocks: NaN statistics
    decoded = fluxcode.decode_group(group2)
    assert decoded.block_sizes.tolist() == [L, L, L, 7, L, 0, L, L, L, L, 2 * L, 0, 0, 9]
    assert sizes  # (the original sizes, for reference)


def test_empty_update_returns_the_group():
    _, group, _, _, _ = encoded(n=3 * L)
    updated = fluxcode.update(group, {})
    assert updated.group == group and updated.indices.shape == (0,)


@pytest.mark.parametrize("blocks,msg", [({-1: np.zeros(L)}, ">= 0"), ({1.0: np.zeros(L)}, "integers"),
                                        ([np.zeros(L)], "map block indices"), ({1: np.zeros((2, L))}, "1-D"),
                                        ({1: np.zeros(65_536)}, "0 to 65535")])
def test_bad_blocks(blocks, msg):
    _, group, _, _, _ = encoded()
    with pytest.raises(ValueError, match=msg):
        fluxcode.update(group, blocks)


@pytest.mark.parametrize("index", [65_535, 10**10, 2**63, 2**80])
def test_append_past_block_limit(index):
    """Rejected before anything is sized by the index (10^10 used to exhaust memory)."""
    _, group, _, _, _ = encoded(n=3 * L)
    with pytest.raises(ValueError, match="65535 blocks"):
        fluxcode.update(group, {index: np.zeros(1)})


def test_update_with_non_finite_blocks():
    """New blocks may hold NaN/inf; flagged blocks already in the block group keep their codes."""
    _, group, _, _, _ = encoded(n=20_000)
    b = np.zeros((2, L))
    b[1, 5], b[1, 6] = np.nan, -np.inf
    group2 = fluxcode.update(group, by_index([0, 1], b)).group
    y = fluxcode.decode_group(group2).values.reshape(-1, L)
    np.testing.assert_array_equal(y[:2], b)
    group3 = fluxcode.update(group2, {10: np.ones(L), 20: np.ones(L)}).group  # replace one, append one
    y3 = fluxcode.decode_group(group3).values.reshape(-1, L)
    np.testing.assert_array_equal(y3[:2], b)
    np.testing.assert_array_equal(y3[2:10], y[2:10])
    np.testing.assert_array_equal(y3[[10, 20]], np.ones((2, L)))


@pytest.mark.parametrize("head_bits", [0x10, 0x20, 0x40, 0x80])
def test_update_refuses_groups_it_cannot_read(head_bits):
    """A block group with reserved flags bits (a future feature), or the irregular time bit (0x10) without a
    time axis, isn't rewritten: that could drop what they mean."""
    _, group, _, _, _ = encoded()
    raw_body = _oracle.write_group(*group_rows(group)[:6])
    raw_body[5] |= head_bits  # block_flags are the body's first bytes
    bad = group[:_format.HEADER_BYTES] + zstandard.ZstdCompressor(level=3).compress(raw_body.tobytes())
    with pytest.raises(ValueError, match="block 5: block flags"):
        fluxcode.update(bad, {0: np.zeros(L)})
    with pytest.raises(ValueError, match="block 5: block flags"):
        fluxcode.decode_group(bad)


def test_update_refuses_out_of_range_parameters():
    _, group, _, _, _ = encoded()

    def bad_param(flags, sizes, param, *_):
        param[2] = 5000

    with pytest.raises(ValueError, match="block 2: parameter"):
        fluxcode.update(rebuilt(group, bad_param), {0: np.zeros(L)})


@pytest.mark.parametrize("anchor", [np.inf, -np.inf, np.nan])
def test_non_finite_float_anchors_are_rejected(anchor):
    _, group, _, _, _ = encoded()

    def bad_anchor(flags, sizes, param, anchors, *_):
        anchors.view(np.float64)[4] = anchor

    with pytest.raises(ValueError, match="block 4: anchor"):
        fluxcode.decode_group(rebuilt(group, bad_anchor))


def test_out_of_range_decimal_anchors_are_rejected():
    x = np.round(np.cumsum(np.random.default_rng(3).normal(size=3 * L)), 2)
    group, _, _, _ = fluxcode.encode_group(x, Params(noise_floor_sigma=0))
    assert (group_rows(group).block_flags & _format.BLOCK_FLAG_DECIMAL).all()

    def bad_anchor(flags, sizes, param, anchors, *_):
        anchors[1] = 1 << 52

    with pytest.raises(ValueError, match="block 1: anchor"):
        fluxcode.decode_group(rebuilt(group, bad_anchor))


def test_update_checks_time_unit_if_given():
    t = np.datetime64("2026-01-01") + np.arange(20) * np.timedelta64(1, "ms")
    group = fluxcode.encode_group(np.arange(20.0), times=t, block_len=10).group
    new = {1: np.zeros(10)}
    new_times = {1: t[10:]}
    assert fluxcode.update(group, new, times=new_times, time_unit="ms").group == fluxcode.update(
        group, new, times=new_times).group
    with pytest.raises(ValueError, match="time_unit is us but the block group stores ms"):
        fluxcode.update(group, new, times=new_times, time_unit="us")
    with pytest.raises(ValueError, match="time_unit must be one of"):
        fluxcode.update(group, new, times=new_times, time_unit="h")


def test_update_time_unit_on_a_group_without_times():
    group = fluxcode.encode_group(np.arange(20.0), block_len=10).group
    with pytest.raises(ValueError, match="no time axis"):
        fluxcode.update(group, {1: np.zeros(10)}, time_unit="ms")
