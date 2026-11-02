# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Scaled grids (SPEC §4-§6, ENCODER §5.4): (K0 + N q) * B^p / D, found by the index gcd (D = 1) and
the rational search; their decoding (one rounding), validation, fixed points and updates."""

import math
from fractions import Fraction

import _oracle
import numpy as np
import pytest
import zstandard
from _series import group_rows

import fluxcode
from fluxcode import Params, _format, _rational
from fluxcode._format import BLOCK_FLAG_DECIMAL, BLOCK_FLAG_SCALED, SCALE_SHIFT

L = 1000
RNG = np.random.default_rng(11)


def walk(seed, n=10 * L, sigma=1.0):
    return np.cumsum(np.random.default_rng(seed).normal(0, sigma, n))


def counts(seed, lo, hi, n=10 * L):
    """Slowly varying ADC counts in [lo, hi]."""
    t = np.arange(n) / L
    mid, amp = (lo + hi) / 2, (hi - lo) / 2
    v = mid + 0.8 * amp * np.sin(2 * np.pi * 0.37 * t + seed) + np.cumsum(np.random.default_rng(seed).normal(0, amp / 3000, n))
    return np.round(np.clip(v, lo, hi))


def fahrenheit_text(seed):
    c = np.round((20 + 5 * np.sin(np.arange(10 * L) / 6000) + walk(seed, sigma=0.02)) * 10) / 10
    return np.round((32 + 1.8 * c) * 100) / 100


# name -> (values, exact): exact data is the double nearest each rational, as decoding gives
GRIDS = {
    "thirds": (lambda: np.round(walk(1) * 3) / 3, True),
    "0.05": (lambda: np.round(walk(2) * 20) / 20, True),
    "F from 0.1 C": (lambda: fahrenheit_text(3), True),
    "ADC 10 V / 4096": (lambda: counts(4, 0, 4095) * (10 / 4096), True),
    "ADC 10 V / 4095": (lambda: counts(5, 0, 4095) * 10 / 4095, True),
    "ADC 10 V / 4095, float step": (lambda: counts(5, 0, 4095) * (10 / 4095), False),
    "calibrated gain": (lambda: (-32100 + 137 * counts(6, 100, 600)) / 10000, True),
    "calibrated gain, float": (lambda: -3.21 + counts(6, 100, 600) * 0.0137, False),
    "4-20 mA": (lambda: (counts(7, 6553, 32767) - 6553) * 250 / 26214, True),
    "odd integers": (lambda: 2 * np.round(walk(8) * 3) + 1, True),
    "100 + 300 k": (lambda: 100 + 300 * np.round(walk(9)), True),
}


def decode(group):
    return fluxcode.decode_group(group).values


def step_of(rows, b):
    """Block b's grid step."""
    flags, param, scale = int(rows.block_flags[b]), int(rows.grid_params[b]), int(rows.grid_scales[b])
    step = 10.0 ** param if flags & BLOCK_FLAG_DECIMAL else 2.0 ** param
    if flags & BLOCK_FLAG_SCALED:
        step *= (scale & 0xFFFF) / (scale >> SCALE_SHIFT)
    return step


@pytest.mark.parametrize("name", GRIDS)
def test_grids_are_found_and_decode(name):
    """Every block goes on a scaled grid (or a coarse exact power of two), exactly for exact data and
    within a quarter of the power-of-two step otherwise; without the rational search only the
    index gcd's grids (D = 1) remain."""
    make, exact = GRIDS[name]
    x = make()
    group = fluxcode.encode_group(x, block_len=L).group
    rows = group_rows(group)
    plain = fluxcode.encode_group(x, Params(grids="decimal"), block_len=L).group
    if name != "ADC 10 V / 4096":  # an exact binary grid: 5 x 2^-11, scaled by the index gcd
        assert (rows.block_flags & BLOCK_FLAG_SCALED).all()
    y = decode(group)
    if exact:
        np.testing.assert_array_equal(y, x)
    else:
        err = np.abs(y - x).reshape(-1, L).max(1)
        fine = [2.0 ** fluxcode._encoder.range_exponent(b.min(), b.max(), 16) for b in x.reshape(-1, L)]
        assert (err <= np.array(fine) / 4).all()
    dens = group_rows(plain).grid_scales >> SCALE_SHIFT
    assert (dens <= 1).all()  # no rational search: D = 1 (or unscaled)
    if (rows.grid_scales >> SCALE_SHIFT > 1).any():
        assert len(group) < len(plain)


@pytest.mark.parametrize("name", GRIDS)
def test_scaled_grids_are_fixed_points(name):
    """decode(encode(y)) = y for decoded y, and the bytes are identical from the second encode on."""
    x = GRIDS[name][0]()
    y = decode(fluxcode.encode_group(x, block_len=L).group)
    again = fluxcode.encode_group(y, block_len=L).group
    np.testing.assert_array_equal(decode(again), y)
    assert fluxcode.encode_group(decode(again), block_len=L).group == again


@pytest.mark.parametrize("name", GRIDS)
def test_each_block_encodes_as_it_would_alone(name):
    """Nothing carries over from the block before: a block's grid and residuals are those it gets
    alone (so an update re-encodes a block as a fresh encode would)."""
    x = GRIDS[name][0]()
    together = group_rows(fluxcode.encode_group(x, block_len=L).group)
    for b in range(x.size // L):
        alone = group_rows(fluxcode.encode_group(x[b * L:(b + 1) * L], block_len=L).group)
        for field in ("block_flags", "grid_params", "grid_scales", "value_anchors"):
            assert getattr(alone, field)[0] == getattr(together, field)[b], (b, field)
        np.testing.assert_array_equal(alone.residuals, together.residuals[b * L:(b + 1) * L])


def test_an_unrationalizable_step_stays_binary():
    """A gain no fraction of 16-bit N and D reaches: power-of-two grid, alone or in a group."""
    x = -3.21 + counts(12, 100, 600) * 0.013712345678
    together = group_rows(fluxcode.encode_group(x, block_len=L).group)
    assert not (together.block_flags & BLOCK_FLAG_SCALED).any()
    for b in range(x.size // L):
        alone = group_rows(fluxcode.encode_group(x[b * L:(b + 1) * L], block_len=L).group)
        assert alone.grid_params[0] == together.grid_params[b]


@pytest.mark.parametrize("target", [6.0, 8.0])
def test_the_target_coarsens_scaled_grids_within_half_a_step(target):
    """A wide thirds walk (about 11 bits a sample) capped: coarser blocks, half a step of error."""
    x = np.round(walk(13, sigma=300) * 3) / 3
    exact = fluxcode.encode_group(x, block_len=L).group
    capped = fluxcode.encode_group(x, Params(target_bits_per_sample=target), block_len=L).group
    assert len(capped) < len(exact)
    rows = group_rows(capped)
    err = np.abs(decode(capped) - x).reshape(-1, L).max(1)
    for b in range(x.size // L):
        assert err[b] <= step_of(rows, b) / 2 * (1 + 1e-12)


@pytest.mark.parametrize(("name", "make"), [
    ("0.25 walk", lambda: np.round(walk(21, sigma=40) * 4) / 4),
    ("thirds walk", lambda: np.round(walk(22, sigma=40) * 3) / 3),
])
@pytest.mark.parametrize("target", [6.0, 7.0])
def test_the_target_caps_exact_grids_and_keeps_fixed_points(name, make, target):
    """An exact coarse grid (the index gcd's, a rational one) is capped like any block: the budget
    counts it at its planned exponent. And decoded data, back on the coarsened grid, re-encodes to
    itself: the cap doesn't coarsen it again."""
    x = make()
    params = Params(target_bits_per_sample=target)
    free = len(fluxcode.encode_group(x, block_len=L).group)
    capped = fluxcode.encode_group(x, params, block_len=L).group
    assert len(capped) < free
    assert 8 * len(capped) / x.size < target + 1.0  # soft: the estimate is within about a bit
    y = decode(capped)
    again = fluxcode.encode_group(y, params, block_len=L).group
    np.testing.assert_array_equal(decode(again), y)
    assert fluxcode.encode_group(decode(again), params, block_len=L).group == again


@pytest.mark.parametrize(("grids", "allowed"), [
    ("binary", 0), ("decimal", BLOCK_FLAG_DECIMAL), ("rational", BLOCK_FLAG_DECIMAL | BLOCK_FLAG_SCALED),
])
@pytest.mark.parametrize("name", ["thirds", "0.05", "F from 0.1 C", "odd integers"])
def test_grid_levels_nest(grids, allowed, name):
    """Each level of Params.grids adds grids to the one before: binary stores no decimal or scaled
    block, decimal no scaled one; every level decodes within its bounds."""
    x = GRIDS[name][0]()
    group = fluxcode.encode_group(x, Params(grids=grids), block_len=L).group
    flags = group_rows(group).block_flags
    forbidden = (BLOCK_FLAG_DECIMAL | BLOCK_FLAG_SCALED) & ~allowed
    assert not (flags & forbidden).any()
    if grids == "rational":
        np.testing.assert_array_equal(decode(group), x)


def test_unknown_grids_level_is_rejected():
    with pytest.raises(ValueError, match="grids must be one of"):
        Params(grids="rationals")


@pytest.mark.parametrize("step", [3e15, 3e19, 3e22])
def test_large_decimal_steps_fold_or_stay_decimal(step):
    """A decimal grid of p > 4 can't fold into N (10^5 > 2^16): it stays a plain decimal grid
    (10^19 overflowed int64 once, giving a scale the decoder rejected)."""
    x = np.array([0.0, step, 2 * step, step] * 8)
    np.testing.assert_array_equal(decode(fluxcode.encode_group(x).group), x)


# --- Decoding: one rounding (SPEC §5) ------------------------------------------------------------

def scaled_group(flags, param, num, den, k0, q):
    """A one-block group of the given scaled grid and indices (order 0)."""
    n = q.size
    rows = _format.GroupRows(
        np.array([flags], np.uint8), np.array([n], np.int64), np.array([param], np.int64),
        np.array([num | den << SCALE_SHIFT], np.int64), np.array([k0], np.int64),
        ((q.astype(np.int64) + 2 ** 15) % 2 ** 16 - 2 ** 15).astype(np.int16), np.zeros(n, np.uint8), None,
    )
    body = _oracle.write_group(*rows[:7])
    return _format.pack_header(1, n, False, 0) + zstandard.ZstdCompressor().compress(body.tobytes())


@pytest.mark.parametrize("decimal", [True, False])
def test_decoding_rounds_once(decimal):
    """Each sample decodes to the double nearest (K0 + N q) B^p / D (Python's float(Fraction) is
    correctly rounded), down to the subnormals and up to the largest exponent."""
    rng = np.random.default_rng(14)
    params = rng.integers(-22, 1, 40) if decimal else np.r_[rng.integers(-1022, 971, 37), -1022, 970, -1000]
    for param in params:
        num, den = int(rng.integers(1, 2 ** 16)), int(rng.integers(1, 2 ** 16))
        if decimal and not 5 ** -int(param) < (1 << 53) // den:
            den = 1
        k0 = int(rng.integers(-2 ** 51, 2 ** 51))
        q = rng.integers(0, 2 ** 16, 64)
        y = decode(scaled_group(BLOCK_FLAG_SCALED | (BLOCK_FLAG_DECIMAL if decimal else 0), int(param), num, den, k0, q))
        base = Fraction(10) if decimal else Fraction(2)
        want = [float((k0 + num * int(v)) * base ** int(param) / den) for v in q]
        np.testing.assert_array_equal(y, want)


@pytest.mark.parametrize(("flags", "param", "num", "den", "message"), [
    (BLOCK_FLAG_SCALED, 0, 0, 1, "grid scale"),                          # N = 0
    (BLOCK_FLAG_SCALED, 0, 1, 0, "grid scale"),                          # D = 0
    (BLOCK_FLAG_SCALED | BLOCK_FLAG_DECIMAL, 1, 3, 1, "grid scale"),     # decimal p > 0
    (BLOCK_FLAG_SCALED | BLOCK_FLAG_DECIMAL, -23, 3, 1, "grid scale"),   # p < -22
    (BLOCK_FLAG_SCALED | BLOCK_FLAG_DECIMAL, -22, 3, 4, "grid scale"),   # 4 x 5^22 >= 2^53: D 10^22 inexact
    (BLOCK_FLAG_SCALED, 971, 3, 1, "grid scale"),                        # could overflow
    (BLOCK_FLAG_SCALED, -1023, 3, 1, "grid scale"),                      # would scale into the subnormals
    (0, 0, 3, 1, "grid scale"),                                          # a scale on an unscaled block
])
def test_invalid_scales_are_rejected(flags, param, num, den, message):
    with pytest.raises(ValueError, match=f"block 0: {message}"):
        decode(scaled_group(flags, param, num, den, 5, np.arange(8)))


def test_scaled_anchor_bound():
    decode(scaled_group(BLOCK_FLAG_SCALED, 0, 3, 1, 2 ** 52 - 1, np.arange(8)))
    with pytest.raises(ValueError, match="block 0: anchor"):
        decode(scaled_group(BLOCK_FLAG_SCALED, 0, 3, 1, 2 ** 52, np.arange(8)))


def test_largest_valid_decimal_denominators_decode():
    """D 10^22 with 3 x 5^22 < 2^53 is exact: it decodes like any other."""
    y = decode(scaled_group(BLOCK_FLAG_SCALED | BLOCK_FLAG_DECIMAL, -22, 1, 3, 10 ** 15, np.arange(8)))
    np.testing.assert_array_equal(y, [float(Fraction(10 ** 15 + v, 3 * 10 ** 22)) for v in range(8)])


# --- The fraction search ------------------------------------------------------------------------

def test_simplest_between_matches_brute_force():
    """The smallest-denominator fraction in an interval, against a search over denominators."""
    rng = np.random.default_rng(15)
    for _ in range(300):
        center = rng.uniform(1, 10)
        width = 10.0 ** rng.uniform(-9, -1)
        lo, hi = center - width, center + width
        num, den = _rational.simplest_between(lo, hi, 5000)
        best = next(((math.ceil(lo * d), d) for d in range(1, 5001) if math.ceil(lo * d) <= hi * d), (0, 0))
        assert (num, den) == best
        if den:
            assert lo <= num / den <= hi


def test_the_step_uncertainty_picks_the_true_fraction():
    """0.0137 from float-computed counts (offset -3.21): 137/100, not a nearer fraction of larger D."""
    x = -3.21 + counts(16, 100, 600, L) * 0.0137
    rows = group_rows(fluxcode.encode_group(x, block_len=L).group)
    assert rows.block_flags[0] & BLOCK_FLAG_SCALED and rows.block_flags[0] & BLOCK_FLAG_DECIMAL
    assert (int(rows.grid_params[0]), int(rows.grid_scales[0]) & 0xFFFF, int(rows.grid_scales[0]) >> 16) == (-2, 137, 100)


# --- Updates (ENCODER §6) -----------------------------------------------------------------------

START = np.datetime64("2026-01-01T00:00:00", "ns")
SECOND = np.timedelta64(1, "s")


def test_updates_from_the_same_sensor_keep_samples_exact():
    """Repeated straddling updates with new samples on the same thirds grid (min and max moving):
    the kept samples come back bit for bit every time."""
    rng = np.random.default_rng(17)
    t = START + np.arange(1000) * np.timedelta64(10, "ms")
    x = np.round(walk(18, 1000) * 3) / 3
    group = fluxcode.encode_time_blocks(x, t, start_time=START, block_duration=SECOND).group
    lo, hi = START + 5 * SECOND + np.timedelta64(500, "ms"), START + 6 * SECOND
    kept = (t >= START + 5 * SECOND) & (t < lo)
    replaced = (t >= lo) & (t < hi)
    for _ in range(100):
        new = np.round((x[kept].mean() + rng.uniform(-3, 3) * rng.normal(0, 3, replaced.sum())) * 3) / 3
        group = fluxcode.update_time_blocks(group, new, t[replaced], delete_ranges=[(lo, hi)],
                                           start_time=START, block_duration=SECOND).group
        assert group_rows(group).block_flags[5] & BLOCK_FLAG_SCALED
        np.testing.assert_array_equal(decode(group)[kept], x[kept])


def test_mixed_updates_keep_a_bounded_error():
    """Updates whose new samples come from the same thirds grid, another grid (cents) or a noisy
    source, moving the block's range 1-4x: a kept sample is exact while its block stays on its
    fine grid, rounds once when it leaves it, and its error stays under the coarsest step the
    block has used."""
    rng = np.random.default_rng(19)
    t = START + np.arange(1000) * np.timedelta64(10, "ms")
    x = np.round(walk(20, 1000) * 3) / 3
    group = fluxcode.encode_time_blocks(x, t, start_time=START, block_duration=SECOND).group
    lo, hi = START + 5 * SECOND + np.timedelta64(500, "ms"), START + 6 * SECOND
    kept = (t >= START + 5 * SECOND) & (t < lo)
    replaced = (t >= lo) & (t < hi)
    original = x[kept]
    steps, kinds = [], set()
    for _ in range(300):
        center, spread = original.mean(), np.ptp(original)
        new = center + rng.uniform(-2, 2) * spread * rng.uniform(0.5, 2, replaced.sum())
        kind = rng.choice(["same", "cents", "noisy"])
        if kind == "same":
            new = np.round(new * 3) / 3
        elif kind == "cents":
            new = np.round(new * 100) / 100
        group = fluxcode.update_time_blocks(group, new, t[replaced], delete_ranges=[(lo, hi)],
                                           start_time=START, block_duration=SECOND).group
        rows = group_rows(group)
        steps.append(step_of(rows, 5))
        kinds.add((kind, int(rows.block_flags[5]) & (BLOCK_FLAG_SCALED | BLOCK_FLAG_DECIMAL)))
        err = np.abs(decode(group)[kept] - original).max()
        assert err < max(steps)
    assert len(kinds) >= 4  # the block moved between grids
