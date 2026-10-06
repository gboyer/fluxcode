# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""docs/SPEC.md §8 and docs/ENCODER.md §7 conformance tests, through the public API."""

import numpy as np
import pytest
import zstandard
from _series import decode_series, encode_series, flags_params, gated, planes, unit_rows
from _signals import CLEAN, DISCRETE, KINDS, NOISY, discrete_minute, minute

import fluxcode
from fluxcode import Params, _bitpacking, _encoder, _format, _unit
from fluxcode._format import BLOCK_FLAG_DECIMAL, BLOCK_FLAG_ORDER

OFF = Params(noise_floor_sigma=0)


def rows(unit):
    """(flags, param, residual) of a one-unit encoding."""
    rows = unit_rows(unit)
    return rows.block_flags, rows.grid_params, rows.residuals


def blockwise(x, L=1000):
    return x.reshape(-1, L)


ALL = [(k, minute(k, s)) for k in KINDS for s in (11, 12)] + [(n, discrete_minute(n, 13)) for n, _, _ in DISCRETE]
IDS = [a[0] for a in ALL]


@pytest.mark.parametrize("params", [Params(), OFF, Params(min_quantize_bits=16, noise_floor_sigma=0),
                                    Params(max_quantize_bits=10), Params(noise_floor_sigma=0, target_bits_per_sample=6.0)],
                         ids=["default", "no-noise", "16bit", "max10", "target6"])
@pytest.mark.parametrize("name,x", ALL, ids=IDS)
def test_round_trip_error_bound(name, x, params):
    """ENCODER §7.1: max error <= half the step used; <= range / (2^min_bits - 1/2) always. The decoded
    minimum is the grid point nearest the minimum, so it's within half a step too."""
    units, lo, hi, mean = encode_series(x, params)
    y = decode_series(units)
    X, Y = blockwise(x), blockwise(y)
    np.testing.assert_array_equal(lo, X.min(1))
    np.testing.assert_array_equal(hi, X.max(1))
    np.testing.assert_allclose(mean, X.mean(1), rtol=1e-12, atol=1e-12 * np.abs(X).max())
    flags, param = flags_params(units)
    err = np.abs(Y - X).max(1)
    rng = hi - lo
    dec = (flags & BLOCK_FLAG_DECIMAL) != 0
    for b in range(len(lo)):
        ulps = 4 * np.spacing(np.abs(X[b]).max())  # x - lo and lo + q * step each round once
        if dec[b]:
            assert err[b] <= 10.0 ** param[b] / 4 + ulps  # every sample was within 2^e / 4 < 10^p / 4
        else:
            assert err[b] <= 2.0 ** (param[b] - 1) + ulps
            assert abs(Y[b].min() - lo[b]) <= 2.0 ** (param[b] - 1) + ulps
        assert err[b] <= rng[b] / (2 ** params.min_quantize_bits - 0.5) + ulps


@pytest.mark.parametrize("d", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("f32", [False, True])
def test_decimal_bit_exact(d, f32):
    """ENCODER §7.2: K / 10^d with a range under 2^16 steps decode bit-identical. Their float32 roundings
    decode to the decimals themselves (ENCODER §6: float32 rounding artifacts are intentionally lost),
    and casting back to float32 recovers the input."""
    rng = np.random.default_rng(d)
    step = 300 if f32 else 3  # float32 rounding must sit within the block's 2^e / 4 tolerance
    K = np.cumsum(rng.integers(-step, step + 1, 5000)) + 12345
    decimals = K / 10.0 ** d
    x = decimals.astype(np.float32).astype(np.float64) if f32 else decimals
    units, _, _, _ = encode_series(x)
    flags, _ = flags_params(units)
    assert (flags & BLOCK_FLAG_DECIMAL).all()
    y = decode_series(units)
    np.testing.assert_array_equal(y, decimals)
    np.testing.assert_array_equal(y.astype(np.float32), x.astype(np.float32))


@pytest.mark.parametrize("params", [Params(), OFF], ids=["default", "no-noise"])
@pytest.mark.parametrize("name,x", ALL, ids=IDS)
def test_fixed_point(name, x, params):
    """ENCODER §7.3: y = decode(encode(x)) is a fixed point, and bytes are identical from the second encode."""
    u1, _, _, _ = encode_series(x, params)
    y = decode_series(u1)
    u2, _, _, _ = encode_series(y, params)
    y2 = decode_series(u2)
    np.testing.assert_array_equal(y2, y)
    u3, _, _, _ = encode_series(y2, params)
    assert u3 == u2


@pytest.mark.parametrize("orders", [{0}, {1}, {2}, {3}, {0, 1}, {2, 3}, {0, 1, 2, 3}])
@pytest.mark.parametrize("name", ["random-walk", "chirp", "quadratic", "noisy-sine"])
def test_order_independence(name, orders):
    """SPEC §8.2: any orders setting decodes with the same decoder, to the same values (the grid doesn't
    depend on the order), and every block uses an allowed order."""
    x = minute(name, 21)
    ref_units, _, _, _ = encode_series(x)
    units, _, _, _ = encode_series(x, Params(diff_orders=orders))
    np.testing.assert_array_equal(decode_series(units), decode_series(ref_units))
    flags, _ = flags_params(units)
    assert set(np.unique(flags & BLOCK_FLAG_ORDER).tolist()) <= orders


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
    units, _, _, _ = encode_series(x, Params(decimal_detection=False, noise_floor_sigma=0))
    _, param, _ = rows(units[0])
    assert param[0] == -15
    assert np.abs(decode_series(units) - x).max() <= 2.0 ** -16


def test_decimal_too_wide_falls_back():
    """More than 2^16 decimal steps across the block: the power-of-two grid."""
    x = np.round(np.linspace(0, 100_000, 1000) + 0.1 * np.arange(1000) % 1, 3)  # 0.001 grid, 1e8 steps
    units, _, _, _ = encode_series(x, OFF)
    flags, _ = flags_params(units)
    assert not (flags & BLOCK_FLAG_DECIMAL).any()


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
    _, param = flags_params(units)
    assert (err[param == -1074] == 0).all()
    if lossless and params.max_quantize_bits == 16 and params.noise_factor(False) == 0:  # noise may gate
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
    max never past the largest double."""
    x = make(np.random.default_rng(4))
    units, lo, hi, _ = encode_series(x, params)
    y = decode_series(units)
    assert np.isfinite(y).all()
    X, Y = x.reshape(-1, 1000), y.reshape(-1, 1000)
    with np.errstate(over="ignore"):
        half_err = np.abs(0.5 * Y - 0.5 * X).max(1)  # |y - x| / 2, which can't overflow
    bound = half_range(lo, hi) / (2 ** params.min_quantize_bits - 0.5)
    assert (half_err <= bound * (1 + 1e-15)).all()


def test_decoder_clamps_to_largest_double():
    """A step that rounds DBL_MAX up to 2^1024 would decode to inf; the decoder clamps it."""
    x = np.r_[0.0, DBL_MAX, np.zeros(998)]
    units, _, _, _ = encode_series(x, Params(min_quantize_bits=1, max_quantize_bits=1, noise_floor_sigma=0))
    y = decode_series(units)
    assert y[1] == DBL_MAX and np.isfinite(y).all()


def test_huge_range_noise_gates():
    """White noise spanning more than 2^1023 is gated like the same noise at scale 1."""
    base = np.random.default_rng(5).normal(size=6000)
    for scale in (1.0, DBL_MAX):
        x = base / np.abs(base).max() * scale
        assert gated(x).all()


def test_negative_zero_decodes_as_positive_zero():
    """Documented exception to bit-exactness (SPEC §7): -0.0 comes back as +0.0 (equal, other sign bit)."""
    x = np.full(1000, -0.0)
    y = decode_series(encode_series(x)[0])
    assert (y == x).all() and not np.signbit(y).any()


@pytest.mark.parametrize("block_len", [5, 8, 13, 64, 1000, 1001, 4096])
@pytest.mark.parametrize("n_blocks", [1, 7, 60])
@pytest.mark.parametrize("flagged", [0, 1, "all"])
def test_units_are_self_describing(block_len, n_blocks, flagged):
    """The header records the block and sample counts and each block its size: decode_unit
    needs only the unit."""
    x = np.cumsum(np.random.default_rng(block_len).normal(size=block_len * n_blocks - block_len // 2))
    if flagged == "all":
        x[::block_len] = np.nan
    elif flagged:
        x[0] = np.inf
    unit, _, _, _ = fluxcode.encode_unit(x, block_len=block_len)
    header = _format.unpack_header(unit)  # byte_planes: the effort picks either
    assert (header.num_blocks, header.num_samples, header.time_unit) == (n_blocks, len(x), 0)
    y, _, sizes = fluxcode.decode_unit(unit)
    assert y.shape == x.shape
    assert sizes.tolist() == [block_len] * (n_blocks - 1) + [block_len - block_len // 2]
    np.testing.assert_array_equal(np.isnan(y), np.isnan(x))


def _plane_fields(body, num_blocks):
    """The plane fields of a parsed unit body with a time axis, as (planes, layout) pairs."""
    _, lay = _format.read_layout(body, num_blocks)
    octets, code_octets = int(lay.octet_offsets[-1]), int(lay.code_offsets[-1])
    short, long = _bitpacking.time_planes_views(body, num_blocks, octets, code_octets, int(lay.short_offsets[-1]),
                                            int(lay.long_offsets[-1]))
    return [(_bitpacking.planes_view(body, num_blocks, octets, True), lay.octet_offsets),
            (_bitpacking.code_planes_view(body, num_blocks, octets, code_octets, True), lay.code_offsets),
            (short, lay.short_offsets), (long, lay.long_offsets)]


@pytest.mark.parametrize("sizes", [[4] * 7, [5] * 7, [13] * 7, [1001] * 7, [13, 1, 7, 0, 21, 16, 3, 9, 1001, 2]])
def test_block_sizes_not_a_multiple_of_8(sizes):
    """Every plane field pads each block to whole bytes with zero bits (byte planes with zero
    bytes); values, non-finite codes and short and long irregular times round-trip exactly."""
    sizes = np.array(sizes)
    num_blocks, n = len(sizes), int(sizes.sum())
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    rng = np.random.default_rng(int(sizes[0]))
    x = rng.integers(-1000, 1000, n).astype(float)  # integers: decimal grid, lossless
    x[[1, n // 2, n - 1]] = [np.nan, np.inf, -np.inf]
    ticks = np.cumsum(rng.integers(1, 5, n))
    ticks[offsets[4] + 2:] += 2 ** 40  # inside block 4, which then needs long time residuals
    params = Params(noise_floor_sigma=0)
    unit = fluxcode.encode_blocks(x, sizes, params, times=ticks, time_unit="ns").unit
    rows = unit_rows(unit)
    assert np.count_nonzero(rows.block_flags & _format.BLOCK_FLAG_NONFINITE) == len(set(np.searchsorted(offsets, [1, n // 2, n - 1], "right")))
    assert np.count_nonzero(rows.block_flags & _format.BLOCK_FLAG_LONG_TIME) == 1
    body = _unit.decompress(unit).raw_body
    for field, field_offsets in _plane_fields(body, num_blocks):
        assert field.shape[1]
        for block_idx in range(num_blocks):
            block_bytes = field[:, field_offsets[block_idx]:field_offsets[block_idx + 1]]
            if block_bytes.shape[1]:
                bits = np.unpackbits(block_bytes, axis=1, bitorder="little")
                assert not bits[:, sizes[block_idx]:].any()
    # Decoders ignore the padding: setting every padding bit changes nothing
    dirty_body = body.copy()
    for field, field_offsets in _plane_fields(dirty_body, num_blocks):
        for block_idx in range(num_blocks):
            if field_offsets[block_idx + 1] > field_offsets[block_idx] and sizes[block_idx] % 8:
                field[:, field_offsets[block_idx + 1] - 1] |= np.uint8(0xFF << (sizes[block_idx] % 8) & 0xFF)
    if (sizes % 8).any():
        assert (dirty_body != body).any()
    dirty_unit = unit[:_format.HEADER_BYTES] + zstandard.ZstdCompressor().compress(dirty_body.tobytes())
    assert unit_rows(dirty_unit).block_flags.tolist() == rows.block_flags.tolist()
    decoded = fluxcode.decode_unit(dirty_unit)
    np.testing.assert_array_equal(decoded.values, x)
    assert decoded.times is not None
    np.testing.assert_array_equal(decoded.times.view(np.int64), ticks)
    # The same rows as byte planes: each block's bytes padded with zero bytes
    byte_body = _bitpacking.write_unit(*rows[:6], byte_planes=True, time_rows=rows.time_rows)
    lay = _format.layout(rows.block_flags, sizes)
    byte_planes = _bitpacking.byte_planes_view(byte_body, num_blocks, int(lay.octet_offsets[-1]), True)
    for block_idx in range(num_blocks):
        assert not byte_planes[:, 8 * lay.octet_offsets[block_idx] + sizes[block_idx]:8 * lay.octet_offsets[block_idx + 1]].any()
    byte_unit = (_format.pack_header(num_blocks, n, True, int(_format.TimeUnitCode.NANOSECONDS))
                 + zstandard.ZstdCompressor().compress(byte_body.tobytes()))
    for encoded in (unit, byte_unit):
        decoded = fluxcode.decode_unit(encoded)
        np.testing.assert_array_equal(decoded.values, x)
        assert decoded.times is not None
    np.testing.assert_array_equal(decoded.times.view(np.int64), ticks)
    # update: replace the last block with a longer one, then append a block; equals encoding the whole series
    extra = rng.integers(-1000, 1000, 9).astype(float)
    extra_ticks = ticks[-1] + np.arange(1, 10)
    new_sizes = np.concatenate([sizes[:-1], [sizes[-1] + 4, 5]])
    updated = fluxcode.update(
        unit, {num_blocks - 1: np.concatenate([x[offsets[-2]:], extra[:4]]), num_blocks: extra[4:]}, params,
        times={num_blocks - 1: np.concatenate([ticks[offsets[-2]:], extra_ticks[:4]]), num_blocks: extra_ticks[4:]},
    ).unit
    assert updated == fluxcode.encode_blocks(np.concatenate([x, extra]), new_sizes, params,
                                             times=np.concatenate([ticks, extra_ticks]), time_unit="ns").unit


def _with_header(unit, **fields):
    """unit with some header fields replaced (version, flags, num_blocks, num_samples)."""
    values = dict(zip(["version", "flags", "num_blocks", "num_samples"], _format.UNIT_HEADER.unpack_from(unit)))
    values.update(fields)
    return _format.UNIT_HEADER.pack(*values.values()) + unit[_format.HEADER_BYTES:]


@pytest.mark.parametrize("fields,msg", [
    ({"version": 2}, "version 2"),
    ({"flags": 0x10}, "reserved"),
    ({"flags": 0x80}, "reserved"),
    ({"flags": 5 << 1}, "time unit 5"),
    ({"flags": 7 << 1}, "time unit 7"),
    ({"flags": 4 << 1}, "doesn't fit"),  # a time axis, but the body has no time fields
    ({"num_blocks": 2}, "add up"),  # the body holds 3: its sizes and flags read wrong
    ({"num_blocks": 4}, "doesn't fit"),
    ({"num_blocks": 0}, "sample count"),  # no blocks can't hold samples
    ({"num_samples": 0}, "doesn't fit"),
    ({"num_samples": 2499}, "add up"),  # the body's block sizes add up to 2500
    ({"num_samples": 3001}, "doesn't fit"),
    ({"num_samples": 1000}, "doesn't fit"),
    ({"num_blocks": 2000, "num_samples": _format.MAX_UNIT_SAMPLES + 1}, "sample count"),
])
def test_bad_headers_are_rejected(fields, msg):
    unit, _, _, _ = fluxcode.encode_unit(np.arange(2500.0))
    with pytest.raises(ValueError, match=msg):
        fluxcode.decode_unit(_with_header(unit, **fields))
    with pytest.raises(ValueError, match="shorter"):
        fluxcode.decode_unit(unit[:7])


def test_bad_block_sizes_are_rejected():
    """Block sizes that don't add up to the header's sample count, or an empty block with
    nonzero columns, are rejected."""
    unit = fluxcode.encode_blocks(np.arange(30.0), [10, 0, 20]).unit
    body = _unit.decompress(unit).raw_body.copy()

    def rebuilt(edit):
        b = body.copy()
        edit(b)
        return unit[:_format.HEADER_BYTES] + zstandard.ZstdCompressor(write_content_size=True).compress(b.tobytes())

    def resize(b):
        b[3], b[5] = 11, 19  # block 0: 11 samples, block 2: 19 (same total, same plane bytes)
    assert fluxcode.decode_unit(rebuilt(resize)).block_sizes.tolist() == [11, 0, 19]

    def overflow(b):
        b[3] = 12
    with pytest.raises(ValueError, match="add up"):
        fluxcode.decode_unit(rebuilt(overflow))

    def flags(b):
        b[1] = 1  # the empty block's flags
    with pytest.raises(ValueError, match="block 1: block flags"):
        fluxcode.decode_unit(rebuilt(flags))

    def param(b):
        b[3 * 3 + 1] = 1  # byte 0 of the empty block's grid parameter
    with pytest.raises(ValueError, match="block 1: parameter"):
        fluxcode.decode_unit(rebuilt(param))

    def anchor(b):
        b[11 * 3 + 1] = 1  # byte 0 of its anchor
    with pytest.raises(ValueError, match="block 1: anchor"):
        fluxcode.decode_unit(rebuilt(anchor))


def _periodic(n=60_000):
    """A clean sine with an exact 125-sample period: its residuals repeat across the unit."""
    return 100 * np.sin(2 * np.pi * np.arange(n) / 125)


@pytest.mark.parametrize("kind", [*KINDS, "periodic"])
def test_best_planes_is_the_smaller_and_decodes_the_same(kind):
    """Effort 7 ("best" layout) is the smaller of bit and byte planes, ties to byte planes; all
    three decode identically."""
    x = _periodic() if kind == "periodic" else minute(kind, 7)
    x[123] = np.nan  # a flagged block too
    for bits in (10, 16):
        with planes("bit"):
            bit_unit = fluxcode.encode_unit(x, Params(max_quantize_bits=bits)).unit
        with planes("byte"):
            byte_unit = fluxcode.encode_unit(x, Params(max_quantize_bits=bits)).unit
        best_unit = fluxcode.encode_unit(x, Params(max_quantize_bits=bits, effort=7)).unit
        assert not _format.unpack_header(bit_unit).byte_planes and _format.unpack_header(byte_unit).byte_planes
        assert best_unit == (byte_unit if len(byte_unit) <= len(bit_unit) else bit_unit)
        for unit in (bit_unit, byte_unit):
            np.testing.assert_array_equal(fluxcode.decode_unit(unit).values, fluxcode.decode_unit(best_unit).values)


@pytest.mark.parametrize("effort", [1, 5, 9])
def test_byte_planes_on_repeating_cycles(effort):
    unit = fluxcode.encode_unit(_periodic(), Params(max_quantize_bits=10, effort=effort)).unit
    assert _format.unpack_header(unit).byte_planes
    with planes("bit"):
        assert len(unit) < 0.8 * len(fluxcode.encode_unit(_periodic(), Params(max_quantize_bits=10)).unit)


def test_update_byte_plane_unit():
    params = Params(max_quantize_bits=10)
    x = _periodic()
    unit = fluxcode.encode_unit(x, params).unit
    new = 50 * np.cos(2 * np.pi * np.arange(1000) / 125)
    x2 = x.copy()
    x2[5000:6000] = new
    # Same params: byte-identical to encoding the edited series from scratch
    assert fluxcode.update(unit, {5: new}, params).unit == fluxcode.encode_unit(x2, params).unit
    # Bit planes: untouched blocks are converted, and still decode identically
    with planes("bit"):
        unit2 = fluxcode.update(unit, {5: new}, params).unit
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


def test_encode_blocks_per_unit_beyond_the_block_limit():
    """blocks_per_unit is a maximum: a larger one than a unit can hold is fine for a shorter series."""
    units = fluxcode.encode(np.arange(5000.0), blocks_per_unit=100_000).units
    assert len(units) == 1 and fluxcode.decode_unit(units[0]).block_sizes.tolist() == [1000] * 5


def test_encode_unit_limit():
    with pytest.raises(ValueError, match="65535 blocks"):
        fluxcode.encode_unit(np.zeros(65_536), block_len=1)
    with pytest.raises(ValueError, match="block_len"):
        fluxcode.encode_unit(np.zeros(10), block_len=0)
    with pytest.raises(ValueError, match="block_len"):
        fluxcode.encode_unit(np.zeros(10), block_len=65_536)
    with pytest.raises(ValueError, match="2\\^26|67108864"):
        fluxcode.encode_unit(np.zeros(_format.MAX_UNIT_SAMPLES + 1), block_len=65_535)
