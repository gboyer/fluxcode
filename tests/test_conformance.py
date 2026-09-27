# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""docs/SPEC.md §7 conformance tests, through the public API."""

import numpy as np
import pytest
from _series import decode_series, encode_series, gated, heads_params, unit_rows
from _signals import CLEAN, DISCRETE, KINDS, NOISY, discrete_minute, minute

import fluxcode
from fluxcode import Params, _encoder, _format
from fluxcode._format import HEAD_DECIMAL, HEAD_ORDER

OFF = Params(noise_floor_sigma=None)


def rows(unit):
    """(head, param, residual) of a one-unit encoding."""
    head, param, _, residual, _, _ = unit_rows(unit)
    return head, param, residual


def blockwise(x, L=1000):
    return x.reshape(-1, L)


ALL = [(k, minute(k, s)) for k in KINDS for s in (11, 12)] + [(n, discrete_minute(n, 13)) for n, _, _ in DISCRETE]
IDS = [a[0] for a in ALL]


@pytest.mark.parametrize("params", [Params(), OFF, Params(min_quantize_bits=16, noise_floor_sigma=None),
                                    Params(max_quantize_bits=10), Params(noise_floor_sigma=None, target_bits_per_sample=6.0)],
                         ids=["default", "no-noise", "16bit", "max10", "target6"])
@pytest.mark.parametrize("name,x", ALL, ids=IDS)
def test_round_trip_error_bound(name, x, params):
    """§7.1: max error <= half the step used; <= range / (2^min_bits - 1/2) always; min exact in
    power-of-two mode."""
    units, lo, hi, mean = encode_series(x, params)
    y = decode_series(units)
    X, Y = blockwise(x), blockwise(y)
    np.testing.assert_array_equal(lo, X.min(1))
    np.testing.assert_array_equal(hi, X.max(1))
    np.testing.assert_allclose(mean, X.mean(1), rtol=1e-12, atol=1e-12 * np.abs(X).max())
    head, param = heads_params(units)
    err = np.abs(Y - X).max(1)
    rng = hi - lo
    dec = (head & HEAD_DECIMAL) != 0
    for b in range(len(lo)):
        ulps = 4 * np.spacing(np.abs(X[b]).max())  # x - lo and lo + q * step each round once
        if dec[b]:
            assert err[b] <= 10.0 ** param[b] / 4 + ulps  # every sample was within 2^e / 4 < 10^p / 4
        else:
            assert err[b] <= 2.0 ** (param[b] - 1) + ulps
            assert Y[b].min() == lo[b]
        assert err[b] <= rng[b] / (2 ** params.min_quantize_bits - 0.5) + ulps


@pytest.mark.parametrize("d", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("f32", [False, True])
def test_decimal_bit_exact(d, f32):
    """§7.2: K / 10^d with a range under 2^16 steps decode bit-identical. Their float32 roundings
    decode to the decimals themselves (spec §6: float32 rounding artifacts are intentionally lost),
    and casting back to float32 recovers the input."""
    rng = np.random.default_rng(d)
    step = 300 if f32 else 3  # float32 rounding must sit within the block's 2^e / 4 tolerance
    K = np.cumsum(rng.integers(-step, step + 1, 5000)) + 12345
    decimals = K / 10.0 ** d
    x = decimals.astype(np.float32).astype(np.float64) if f32 else decimals
    units, _, _, _ = encode_series(x)
    head, _ = heads_params(units)
    assert (head & HEAD_DECIMAL).all()
    y = decode_series(units)
    np.testing.assert_array_equal(y, decimals)
    np.testing.assert_array_equal(y.astype(np.float32), x.astype(np.float32))


@pytest.mark.parametrize("params", [Params(), OFF], ids=["default", "no-noise"])
@pytest.mark.parametrize("name,x", ALL, ids=IDS)
def test_fixed_point(name, x, params):
    """§7.3: y = decode(encode(x)) is a fixed point, and bytes are identical from the second encode."""
    u1, _, _, _ = encode_series(x, params)
    y = decode_series(u1)
    u2, _, _, _ = encode_series(y, params)
    y2 = decode_series(u2)
    np.testing.assert_array_equal(y2, y)
    u3, _, _, _ = encode_series(y2, params)
    assert u3 == u2


def test_bit_order_vector():
    """§7.4: u[3] = 0x0020, u[6] = 0x0400 -> plane 5 byte 0 = 0b00001000, plane 10 byte 0 = 0b01000000."""
    v = np.zeros((1, 8), np.int16)
    v[0, 3] = 0x0010  # zigzag(16) = 32 = 0x0020
    v[0, 6] = 0x0200  # zigzag(512) = 1024 = 0x0400
    raw = _format.write_unit(np.zeros(1, np.uint8), np.zeros(1, np.int64), np.zeros(1, np.int64), v)
    planes = raw[17:].reshape(16, 1)  # after head (1), param (8) and anchor (8)
    expect = np.zeros(16, np.uint8)
    expect[5], expect[10] = 0b00001000, 0b01000000
    np.testing.assert_array_equal(planes[:, 0], expect)


@pytest.mark.parametrize("orders", [{0}, {1}, {2}, {3}, {0, 1}, {2, 3}, {0, 1, 2, 3}])
@pytest.mark.parametrize("name", ["random-walk", "chirp", "quadratic", "noisy-sine"])
def test_order_independence(name, orders):
    """§7.5: any orders setting decodes with the same decoder, to the same values (the grid doesn't
    depend on the order), and every block uses an allowed order."""
    x = minute(name, 21)
    ref_units, _, _, _ = encode_series(x)
    units, _, _, _ = encode_series(x, Params(diff_orders=orders))
    np.testing.assert_array_equal(decode_series(units), decode_series(ref_units))
    head, _ = heads_params(units)
    assert set(np.unique(head & HEAD_ORDER).tolist()) <= orders


def test_constant_block():
    x = np.full(3000, -7.25)
    units, _, _, _ = encode_series(x)
    _, param, resid = rows(units[0])
    assert not resid.any() and not param.any()
    np.testing.assert_array_equal(decode_series(units), x)


def test_range_just_below_power_of_two():
    """The exponent bump: a range that would round up to 2^16 steps takes the next exponent."""
    rng = 1.0 - 2.0 ** -18  # rng * 2^16 rounds to 2^16
    x = np.linspace(0.0, rng, 1000)
    x[-1] = rng
    assert _encoder.exponent(rng, 16) == -15
    units, _, _, _ = encode_series(x, Params(decimal_detection=False, noise_floor_sigma=None))
    _, param, _ = rows(units[0])
    assert param[0] == -15
    assert np.abs(decode_series(units) - x).max() <= 2.0 ** -16


def test_decimal_too_wide_falls_back():
    """More than 2^16 decimal steps across the block: the power-of-two grid."""
    x = np.round(np.linspace(0, 100_000, 1000) + 0.1 * np.arange(1000) % 1, 3)  # 0.001 grid, 1e8 steps
    units, _, _, _ = encode_series(x, OFF)
    head, _ = heads_params(units)
    assert not (head & HEAD_DECIMAL).any()


@pytest.mark.parametrize("n", [1, 7, 999, 1000, 1001, 59_999, 60_000, 60_001, 150_500])
def test_lengths_and_short_units(n):
    """N < 60, partial last block (padded), several units."""
    x = np.cumsum(np.random.default_rng(n).normal(size=n))
    units, lo, hi, mean = encode_series(x)
    assert len(units) == -(-n // 60_000)
    assert len(lo) == -(-n // 1000)
    y = decode_series(units)
    assert y.shape == (n,)
    assert np.abs(y - x).max() <= (hi - lo).max() / (2 ** 16 - 0.5)
    last = x[(len(lo) - 1) * 1000:]
    assert lo[-1] == last.min() and hi[-1] == last.max()
    assert np.isclose(mean[-1], last.mean())
    assert fluxcode.decode_unit(units[-1]).values.shape == (n - (len(units) - 1) * 60_000,)


@pytest.mark.parametrize("seed", range(3))
def test_noise_gate_fires_on_white_noise(seed):
    rng = np.random.default_rng(seed)
    t = np.arange(60_000)
    assert gated(rng.normal(0, 1, 60_000)).all()
    assert gated(50 * np.sin(t / 3000) + rng.normal(0, 1, 60_000)).all()


@pytest.mark.parametrize("name", ["random-walk", "sin-4.12hz", "sin-9.87hz", "sin-50.3hz", "chirp", "linear",
                                  "square-2.24hz"])
def test_noise_gate_never_fires_on_clean_signals(name):
    assert not gated(minute(name, 31)).any()


@pytest.mark.parametrize("name", CLEAN)
@pytest.mark.parametrize("f", [0.01, 0.25, 1.0])
def test_noise_floor_no_effect_on_clean_signals(name, f):
    x = minute(name, 32)
    assert encode_series(x, Params(noise_floor_sigma=f))[0] == encode_series(x, OFF)[0]


@pytest.mark.parametrize("name", NOISY)
def test_noise_floor_shrinks_noisy_signals(name):
    x = minute(name, 33)
    size = lambda p: sum(map(len, encode_series(x, p)[0]))
    assert size(Params(noise_floor_sigma=0.25)) < 0.6 * size(OFF)


@pytest.mark.parametrize("scale", [1e-300, 1e-150, 1e150, 1e300])
def test_extreme_magnitudes(scale):
    """Noise gating and the error bound are scale-free: white noise gates at 1e+-300 as at 1."""
    x = np.random.default_rng(1).normal(size=6000)
    x = x / np.abs(x).max() * scale
    assert gated(x).all()
    for p in (Params(), OFF):
        units, lo, hi, _ = encode_series(x, p)
        err = np.abs(decode_series(units) - x).reshape(-1, 1000).max(1)
        assert (err <= (hi - lo) / (2 ** p.min_quantize_bits - 0.5)).all()
    smooth = scale * np.sin(np.arange(6000) / 100)
    assert not gated(smooth).any()


def test_extreme_offsets():
    """A block around 1e300 with a small range, and one spanning nearly the full double range."""
    x = 1e300 + np.random.default_rng(2).normal(size=1000) * 1e290
    units, lo, hi, _ = encode_series(x)
    assert np.abs(decode_series(units) - x).max() <= (hi - lo)[0] / 63.5
    y = np.linspace(-4e307, 4e307, 1000)
    units, lo, hi, _ = encode_series(y, OFF)
    assert np.abs(decode_series(units) - y).max() <= (hi - lo)[0] / (2 ** 16 - 0.5)


DBL_MAX = np.finfo(np.float64).max


@pytest.mark.parametrize("make,lossless", [
    (lambda r: np.linspace(0, 1e-305, 1000), False),
    (lambda r: r.integers(-500, 500, 3000) * 5e-324, True),  # subnormal integers: under 2^16 steps of 2^-1074
    (lambda r: 1e-300 + r.integers(0, 2 ** 20, 3000) * 5e-324, False),
    (lambda r: np.cumsum(r.normal(size=3000)) * 1e-310, False),
], ids=["linspace-1e-305", "subnormal-ints", "1e-300-offset", "walk-1e-310"])
@pytest.mark.parametrize("params", [Params(), OFF, Params(min_quantize_bits=1, max_quantize_bits=4)],
                         ids=["default", "no-noise", "4bit"])
def test_subnormal_ranges(make, lossless, params):
    """Ranges down to the smallest subnormal encode within the bound, and exactly once the
    16-bit step reaches 2^-1074."""
    x = make(np.random.default_rng(3))
    units, lo, hi, _ = encode_series(x, params)
    y = decode_series(units)
    err = np.abs(y - x).reshape(-1, 1000).max(1)
    assert (err <= (hi - lo) / (2 ** params.min_quantize_bits - 0.5)).all()
    _, param = heads_params(units)
    assert (err[param == -1074] == 0).all()
    if lossless and params.max_quantize_bits == 16 and params.noise_floor_sigma is None:  # noise may gate
        assert (param == -1074).all()
        np.testing.assert_array_equal(y, x)


def half_range(lo, hi):
    return 0.5 * hi - 0.5 * lo  # (hi - lo) / 2 without overflow


@pytest.mark.parametrize("make", [
    lambda r: r.uniform(-1, 1, 3000) * DBL_MAX,
    lambda r: np.linspace(-1, 1, 3000) * DBL_MAX,
    lambda r: np.r_[-DBL_MAX, DBL_MAX, r.normal(size=998) * 1e307],
    lambda r: np.r_[0.0, DBL_MAX, r.uniform(0, 1, 998) * DBL_MAX],
    lambda r: np.cumsum(r.normal(size=3000)) * 1e305,
], ids=["uniform-full", "linspace-full", "extremes-plus-noise", "0-to-max", "walk-1e305"])
@pytest.mark.parametrize("params", [Params(), OFF, Params(min_quantize_bits=1, max_quantize_bits=4),
                                    Params(target_bits_per_sample=6.0)], ids=["default", "no-noise", "4bit", "target"])
def test_huge_ranges(make, params):
    """Blocks whose range reaches or exceeds 2^1023 (hi - lo overflows): within the bound, finite,
    min exact, max never past the largest double."""
    x = make(np.random.default_rng(4))
    units, lo, hi, _ = encode_series(x, params)
    y = decode_series(units)
    assert np.isfinite(y).all()
    X, Y = x.reshape(-1, 1000), y.reshape(-1, 1000)
    np.testing.assert_array_equal(Y.min(1), lo)
    with np.errstate(over="ignore"):
        half_err = np.abs(0.5 * Y - 0.5 * X).max(1)  # |y - x| / 2, which can't overflow
    bound = half_range(lo, hi) / (2 ** params.min_quantize_bits - 0.5)
    assert (half_err <= bound * (1 + 1e-15)).all()


def test_decoder_clamps_to_largest_double():
    """A step that rounds DBL_MAX up to 2^1024 would decode to inf; the decoder clamps it."""
    x = np.r_[0.0, DBL_MAX, np.zeros(998)]
    units, _, _, _ = encode_series(x, Params(min_quantize_bits=1, max_quantize_bits=1, noise_floor_sigma=None))
    y = decode_series(units)
    assert y[1] == DBL_MAX and np.isfinite(y).all()


def test_huge_range_noise_gates():
    """White noise spanning more than 2^1023 is gated like the same noise at scale 1."""
    base = np.random.default_rng(5).normal(size=6000)
    for scale in (1.0, DBL_MAX):
        x = base / np.abs(base).max() * scale
        assert gated(x).all()


def test_negative_zero_decodes_as_positive_zero():
    """Documented exception to bit-exactness (spec §6): -0.0 comes back as +0.0 (equal, other sign bit)."""
    x = np.full(1000, -0.0)
    y = decode_series(encode_series(x)[0])
    assert (y == x).all() and not np.signbit(y).any()


@pytest.mark.parametrize("block_len", [8, 64, 1000, 4096])
@pytest.mark.parametrize("n_blocks", [1, 7, 60])
@pytest.mark.parametrize("flagged", [0, 1, "all"])
def test_units_are_self_describing(block_len, n_blocks, flagged):
    """The header records n and the sample count: decode_unit needs only the unit."""
    x = np.cumsum(np.random.default_rng(block_len).normal(size=block_len * n_blocks - block_len // 2))
    if flagged == "all":
        x[::block_len] = np.nan
    elif flagged:
        x[0] = np.inf
    unit, _, _, _ = fluxcode.encode_unit(x, Params(block_len=block_len, blocks_per_unit=60))
    assert _format.unpack_header(unit) == (block_len, len(x), n_blocks, False, 0)
    y = fluxcode.decode_unit(unit).values
    assert y.shape == x.shape
    np.testing.assert_array_equal(np.isnan(y), np.isnan(x))


def _with_header(unit, **fields):
    """unit with some header fields replaced (version, flags, reserved, block_len, num_samples)."""
    values = dict(zip(["version", "flags", "reserved", "block_len", "num_samples"], _format.UNIT_HEADER.unpack_from(unit)))
    values.update(fields)
    return _format.UNIT_HEADER.pack(*values.values()) + unit[_format.HEADER_BYTES:]


@pytest.mark.parametrize("fields,msg", [
    ({"version": 2}, "version 2"),
    ({"flags": 0x10}, "reserved"),
    ({"flags": 0x80}, "reserved"),
    ({"flags": 5 << 1}, "time unit 5"),
    ({"flags": 7 << 1}, "time unit 7"),
    ({"flags": 4 << 1}, "doesn't fit"),  # a time axis, but the body has no time fields
    ({"reserved": 1}, "reserved"),
    ({"num_samples": 2500 | (1 << 40)}, "reserved"),  # the top 3 bytes of the sample count
    ({"num_samples": 2500 | (1 << 63)}, "reserved"),
    ({"block_len": 1001}, "block length"),
    ({"block_len": 0}, "block length"),
    ({"num_samples": 0}, "sample count"),
    ({"num_samples": _format.MAX_UNIT_SAMPLES + 1}, "sample count"),
    ({"num_samples": 1000}, "doesn't fit"),  # 1 block, but the body holds 3
    ({"block_len": 2000}, "doesn't fit"),
])
def test_bad_headers_are_rejected(fields, msg):
    unit, _, _, _ = fluxcode.encode_unit(np.arange(2500.0))
    with pytest.raises(ValueError, match=msg):
        fluxcode.decode_unit(_with_header(unit, **fields))
    with pytest.raises(ValueError, match="shorter"):
        fluxcode.decode_unit(unit[:10])


def _periodic(n=60_000):
    """A clean sine with an exact 125-sample period: its residuals repeat across the unit."""
    return 100 * np.sin(2 * np.pi * np.arange(n) / 125)


@pytest.mark.parametrize("kind", [*KINDS, "periodic"])
def test_try_byte_planes_never_bigger_and_decodes_the_same(kind):
    x = _periodic() if kind == "periodic" else minute(kind, 7)
    x[123] = np.nan  # a flagged block too
    for bits in (10, 16):
        bit_unit = fluxcode.encode_unit(x, Params(max_quantize_bits=bits)).unit
        best_unit = fluxcode.encode_unit(x, Params(max_quantize_bits=bits, try_byte_planes=True)).unit
        assert len(best_unit) <= len(bit_unit)
        np.testing.assert_array_equal(fluxcode.decode_unit(best_unit).values, fluxcode.decode_unit(bit_unit).values)
        if not _format.unpack_header(best_unit).byte_planes:
            assert best_unit == bit_unit


def test_try_byte_planes_picks_byte_planes_on_repeating_cycles():
    params = Params(max_quantize_bits=10, try_byte_planes=True)
    unit = fluxcode.encode_unit(_periodic(), params).unit
    assert _format.unpack_header(unit).byte_planes
    assert len(unit) < 0.8 * len(fluxcode.encode_unit(_periodic(), Params(max_quantize_bits=10)).unit)


def test_update_byte_plane_unit():
    params = Params(max_quantize_bits=10, try_byte_planes=True)
    x = _periodic()
    unit = fluxcode.encode_unit(x, params).unit
    new = 50 * np.cos(2 * np.pi * np.arange(1000) / 125)
    x2 = x.copy()
    x2[5000:6000] = new
    # Same params: byte-identical to encoding the edited series from scratch
    assert fluxcode.update(unit, [5], new, params).unit == fluxcode.encode_unit(x2, params).unit
    # Default params re-encode with bit planes; untouched blocks still decode identically
    unit2 = fluxcode.update(unit, [5], new, Params(max_quantize_bits=10)).unit
    assert not _format.unpack_header(unit2).byte_planes
    np.testing.assert_array_equal(fluxcode.decode_unit(unit2).values, fluxcode.decode_unit(fluxcode.encode_unit(x2, params).unit).values)


def test_partial_last_block_is_trimmed():
    x = np.arange(2500.0)
    np.testing.assert_array_equal(fluxcode.decode_unit(fluxcode.encode_unit(x)[0]).values, x)


def test_decode_many_independent_units():
    """Units are separate rows of different sizes (e.g. a stream's partial hour, then the full one)."""
    xs = [np.arange(1500.0), np.arange(60_000.0) % 7, np.cumsum(np.ones(12_345))]
    encs = [fluxcode.encode_unit(x) for x in xs]
    ys = fluxcode.decode([e[0] for e in encs])
    for x, y in zip(xs, ys):
        np.testing.assert_array_equal(y.values, x)
        assert y.times is None


def test_encode_unit_limit():
    with pytest.raises(ValueError, match="blocks_per_unit"):
        fluxcode.encode_unit(np.zeros(60_001))
