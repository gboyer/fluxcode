# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""NaN and +-inf: exact round trips, held values, index columns, and unchanged bytes for blocks
without them."""

import numpy as np
import pytest
from _series import decode_series, encode_series, gated, unit_rows
from _signals import KINDS, minute

import fluxcode
from fluxcode import Params
from fluxcode._format import BLOCK_FLAG_NONFINITE

L = 1000
OFF = Params(noise_floor_sigma=None)


def same(y, x):
    """Equal as values, NaN where NaN (the only NaN encoded is the canonical quiet NaN)."""
    np.testing.assert_array_equal(np.isnan(y), np.isnan(x))
    fin = ~np.isnan(x)
    return y[fin], x[fin]


def check_round_trip(x, params=Params()):
    unit, lo, hi, mean = fluxcode.encode_unit(x, params)
    y = fluxcode.decode_unit(unit).values
    yv, xv = same(y, x)
    inf = np.isinf(xv)
    np.testing.assert_array_equal(yv[inf], xv[inf])
    X = np.full(len(lo) * L, np.nan)
    X[:len(x)] = x
    X = X.reshape(-1, L)
    for b in range(len(lo)):
        fin = np.isfinite(X[b])
        if not fin.any():
            assert np.isnan([lo[b], hi[b], mean[b]]).all()
            continue
        assert (lo[b], hi[b]) == (X[b, fin].min(), X[b, fin].max())
        assert np.isclose(mean[b], X[b, fin].mean())
        err = np.abs(y[b * L:(b + 1) * L][fin[:len(y) - b * L]] - X[b, fin][:len(y) - b * L]).max(initial=0)
        assert err <= (hi[b] - lo[b]) / (2 ** params.min_quantize_bits - 0.5) + 4 * np.spacing(np.abs(X[b, fin]).max())
    return unit, lo


@pytest.mark.parametrize("where", ["start", "middle", "end", "scattered", "whole", "alternating"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, "mixed"])
@pytest.mark.parametrize("params", [Params(), OFF, Params(target_bits_per_sample=6.0)], ids=["default", "off", "target"])
def test_round_trip(where, value, params):
    x = minute("noisy-sine", 1)[:5 * L].copy()
    r = np.random.default_rng(2)
    idx = {"start": np.arange(0, 37), "middle": np.arange(1400, 1900), "end": np.arange(2950, 3000),
           "scattered": r.choice(5 * L, 300, replace=False), "whole": np.arange(2 * L, 3 * L),
           "alternating": np.arange(3 * L, 4 * L, 2)}[where]
    vals = r.choice([np.nan, np.inf, -np.inf], len(idx)) if value == "mixed" else value
    x[idx] = vals
    check_round_trip(x, params)


@pytest.mark.parametrize("kind", KINDS)
def test_every_signal_kind_with_gaps(kind):
    x = minute(kind, 3)[:10 * L].copy()
    x[123:456] = np.nan
    x[5000] = np.inf
    x[9999] = -np.inf
    check_round_trip(x)


def test_all_non_finite_unit_and_padded_last_block():
    check_round_trip(np.full(2500, np.nan))
    x = np.r_[np.arange(2000.0), np.nan, np.inf]  # the padding repeats the last (non-finite) sample
    unit, _ = check_round_trip(x)
    assert fluxcode.decode_unit(unit).values[-1] == np.inf


@pytest.mark.parametrize("tail,mean", [([1e308] * 500, 1e308), ([np.nan, 1.0, 2.0, np.inf, np.nan], 1.5),
                                       ([np.nan] * 7, np.nan), ([-0.0] * 3, 0.0)])
def test_padded_last_block_mean(tail, mean):
    """The last block's mean covers its real, finite samples, without overflow."""
    _, _, _, means = fluxcode.encode_unit(np.r_[np.zeros(L), tail])
    np.testing.assert_allclose(means, [0.0, mean], rtol=1e-12)
    assert not np.signbit(means).any()


def test_finite_blocks_unchanged():
    """Blocks without NaN/inf produce the same bytes, and non-finite blocks cost only their planes."""
    x = minute("random-walk", 4)
    ref, _, _, _ = fluxcode.encode_unit(x)
    y = x.copy()
    y[30_500] = np.nan  # block 30
    unit, _, _, _ = fluxcode.encode_unit(y)
    ha, _, pa, _, ra, _, _ = unit_rows(ref)
    hb, _, pb, _, rb, codes, _ = unit_rows(unit)
    ra, rb, codes = ra.reshape(60, L), rb.reshape(60, L), codes.reshape(60, L)
    keep = np.arange(60) != 30
    np.testing.assert_array_equal(ha[keep], hb[keep])
    np.testing.assert_array_equal(pa[keep], pb[keep])
    np.testing.assert_array_equal(ra[keep], rb[keep])
    assert hb[30] & BLOCK_FLAG_NONFINITE and not (hb[keep] & BLOCK_FLAG_NONFINITE).any()
    assert codes[30].tolist() == [1 if i == 500 else 0 for i in range(L)]
    assert len(unit) - len(ref) < 200  # 250 bytes of mostly-zero planes, compressed


def test_held_values():
    """The encoder quantizes each non-finite sample as the previous finite value (the first finite
    one for a leading run), so a gap costs little."""
    x = np.arange(3000.0)
    x[1000:1500] = np.nan
    x[2000:2010] = np.inf
    gaps, _, _, _ = fluxcode.encode_unit(x)
    held = np.arange(3000.0)
    held[1000:1500] = 999.0
    held[2000:2010] = 1999.0
    ref, _, _, _ = fluxcode.encode_unit(held)
    assert len(gaps) - len(ref) < 100


@pytest.mark.parametrize("kind", ["noisy-sine", "impulses", "gauss-spikes"])
def test_occasional_dropouts_keep_the_noise_floor(kind):
    """Flagged blocks estimate noise from their finite samples, so one NaN per block costs
    almost nothing instead of the noise floor (2.3-2.5x the size)."""
    x = minute(kind, 5)
    y = x.copy()
    y[500::1000] = np.nan
    # Bit planes isolate the noise floor: byte planes gain 5% on clean impulses, which the
    # held dropouts break up (the best of both is then still below bit planes on clean data)
    size = lambda v: sum(map(len, fluxcode.encode(v, Params(planes="bit"))[0]))
    assert size(y) < 1.05 * size(x)
    assert (gated(y) != gated(x)).sum() <= 2  # the same blocks gate, bar ones near the threshold


def test_mostly_non_finite_blocks_skip_the_noise_floor():
    """Below half finite, the estimate is unreliable (a random walk starts passing the gate), so
    the block keeps full precision."""
    x = np.random.default_rng(5).normal(size=2 * L)
    x[L + 100:L + 700] = np.nan  # block 1: 40% finite
    g = gated(x)
    assert g[0] and not g[1]


GAPS = ["one", "run", "every-2nd", "every-3rd", "every-4th", "every-5th", "every-7th", "pairs-of-3", "random-10%",
        "random-30%", "bursts"]


def with_gaps(x, gap, value=np.nan):
    x = x.copy()
    blocks = x.reshape(-1, L)
    rng = np.random.default_rng(11)
    if gap == "one":
        blocks[:, 321] = value
    elif gap == "run":
        blocks[:, 100:450] = value
    elif gap.startswith("every-"):
        blocks[:, ::int(gap[6])] = value
    elif gap == "pairs-of-3":  # 2 missing, 1 present
        blocks[:, 0::3] = value
        blocks[:, 1::3] = value
    elif gap.startswith("random-"):
        x[rng.random(len(x)) < int(gap[7:-1]) / 100] = value
    else:  # bursts of 1-20 samples
        i = 0
        while i < len(x):
            i += rng.integers(5, 60)
            x[i:i + rng.integers(1, 20)] = value
            i += 20
    return x


@pytest.mark.parametrize("kind", ["random-walk", "sin-4.12hz", "sin-50.3hz", "chirp", "square-2.24hz", "linear"])
@pytest.mark.parametrize("gap", GAPS)
def test_gaps_never_gate_clean_signals(kind, gap):
    """Regular gap patterns mustn't make a clean signal look like white noise (joining the finite
    samples of an every-3rd-missing sine alternates its time steps: rho ~ -1)."""
    assert not gated(with_gaps(minute(kind, 6), gap)).any()


@pytest.mark.parametrize("gap", GAPS)
def test_gaps_keep_clean_precision(gap):
    """A clean 4 Hz sine with every third sample missing kept 31x worse RMSE."""
    x = minute("sin-4.12hz", 12)
    y = with_gaps(x, gap)
    fin = np.isfinite(y)
    rmse = lambda v: np.sqrt(np.mean((decode_series(encode_series(v)[0])[fin] - x[fin]) ** 2))
    assert rmse(y) <= 1.5 * rmse(x)


@pytest.mark.parametrize("kind", ["noisy-sine", "impulses"])
@pytest.mark.parametrize("gap", ["one", "random-10%", "bursts"])
def test_noisy_signals_with_sparse_gaps_still_gate(kind, gap):
    x = minute(kind, 13)
    assert gated(with_gaps(x, gap)).mean() >= 0.9 * gated(x).mean()


def test_only_canonical_nan():
    """NaN payloads and signs are not kept: every NaN decodes to the canonical quiet NaN."""
    odd = np.array([0x7FF0000000000001, 0xFFF8000000001234], np.uint64).view(np.float64)
    x = np.r_[odd, np.zeros(998)]
    unit, _, _, _ = fluxcode.encode_unit(x)
    y = fluxcode.decode_unit(unit).values
    assert np.isnan(y[:2]).all()
    assert (y[:2].view(np.uint64) == np.array(np.nan).view(np.uint64)).all()


WHOLE = {"nan": np.nan, "+inf": np.inf, "-inf": -np.inf}
PARAMS = {"default": Params(), "off": OFF, "target": Params(target_bits_per_sample=6.0),
          "4bit": Params(min_quantize_bits=1, max_quantize_bits=4), "orders1": Params(diff_orders={1}),
          "nodecimal": Params(decimal_detection=False)}


@pytest.mark.parametrize("pname", PARAMS)
@pytest.mark.parametrize("value", WHOLE)
@pytest.mark.parametrize("n", [1, 999, 1000, 2500, 60_000])
def test_whole_unit_of_one_non_finite_value(n, value, pname):
    """A unit that is nothing but NaN (or +inf, or -inf), including a padded partial last block."""
    x = np.full(n, WHOLE[value])
    unit, lo, hi, mean = fluxcode.encode_unit(x, PARAMS[pname])
    assert np.isnan(lo).all() and np.isnan(hi).all() and np.isnan(mean).all()
    y = fluxcode.decode_unit(unit).values
    np.testing.assert_array_equal(y, x)
    assert len(unit) < 100 + n // 400  # all-zero residuals, one code per sample: compresses to almost nothing


@pytest.mark.parametrize("pname", PARAMS)
def test_whole_blocks_between_finite_ones(pname):
    """Whole blocks of NaN, +inf and -inf (and a mixed one) between finite blocks of every kind."""
    rng = np.random.default_rng(8)
    blocks = [minute("noisy-sine", 1)[:L], np.full(L, np.nan), np.round(minute("random-walk", 2)[:L], 2),
              np.full(L, np.inf), minute("chirp", 3)[:L], np.full(L, -np.inf),
              rng.choice([np.nan, np.inf, -np.inf], L), np.zeros(L), np.full(L, np.nan)]
    x = np.concatenate(blocks)
    check_round_trip(x, PARAMS[pname])
    unit, _, _, _ = fluxcode.encode_unit(x, PARAMS[pname])
    head, _, _, _, _, codes, _ = unit_rows(unit)
    codes = codes.reshape(-1, L)
    assert [bool(h & BLOCK_FLAG_NONFINITE) for h in head] == [False, True, False, True, False, True, True, False, True]
    assert (codes[1] == 1).all() and (codes[3] == 2).all() and (codes[5] == 3).all()


def test_update_whole_non_finite_blocks():
    """Replace a finite block with all -inf, append an all-NaN block, then replace that with data."""
    x = minute("random-walk", 9)[:5 * L]
    unit, _, _, _ = fluxcode.encode_unit(x)
    unit, _, mins, maxs, means = fluxcode.update(unit, {2: np.full(L, -np.inf), 5: np.full(L, np.nan)})
    assert np.isnan(mins).all() and np.isnan(maxs).all() and np.isnan(means).all()
    y = fluxcode.decode_unit(unit).values.reshape(-1, L)
    assert (y[2] == -np.inf).all() and np.isnan(y[5]).all()
    np.testing.assert_array_equal(y[[0, 1, 3, 4]], fluxcode.decode_unit(fluxcode.encode_unit(x)[0]).values.reshape(-1, L)[[0, 1, 3, 4]])
    new = minute("sin-4.12hz", 9)[:L]
    unit = fluxcode.update(unit, {5: new}).unit
    y2 = fluxcode.decode_unit(unit).values.reshape(-1, L)
    np.testing.assert_array_equal(y2[:5], y[:5])
    assert np.abs(y2[5] - new).max() <= (new.max() - new.min()) / (2 ** 16 - 0.5)


def test_bulk_encode_rows_round_trip():
    """encode() returns what each row would store, and decode() takes it back directly."""
    x = minute("noisy-sine", 10)
    x = np.r_[x, x[:12_345]]
    x[70_000:71_000] = np.nan
    units, mins, maxs, means = fluxcode.encode(x)
    assert [len(m) for m in mins] == [60, 13] and len(units) == len(maxs) == len(means) == 2
    ys = fluxcode.decode(units)
    y = np.concatenate([decoded.values for decoded in ys])
    np.testing.assert_array_equal(np.isnan(y), np.isnan(x))
    assert np.nanmax(np.abs(y - x)) < 1.0


def test_negative_zero_index_normalized():
    """The index reports +0.0 like the decoded data (one zero)."""
    _, lo, hi, mean = fluxcode.encode_unit(np.full(1000, -0.0))
    assert not np.signbit(lo).any() and not np.signbit(hi).any() and not np.signbit(mean).any()


def test_block_len_limit():
    fluxcode.decode_unit(fluxcode.encode_unit(np.arange(70_000.0), block_len=65_535)[0])
    with pytest.raises(ValueError, match="65535"):
        fluxcode.encode_unit(np.arange(10.0), block_len=65_536)
