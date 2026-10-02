# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Params.effort: block flushing inside the unit's one zstd frame, the heuristic layout and the
zstd level. The frame stays an ordinary one (content size recorded, one-shot decodable), the
blocks end where the planes do, the heuristic picks a layout from the residuals' high byte, and
no effort changes the decoded values."""

import numpy as np
import pytest
import zstandard
from _series import planes, unit_rows
from _signals import minute

import fluxcode
from fluxcode import Params, _bitpacking, _compress, _format, _unit
from fluxcode._types import MAX_EFFORT, MIN_EFFORT

EFFORTS = range(MIN_EFFORT, MAX_EFFORT + 1)


def body_of(unit):
    return _unit.decompress(unit).raw_body


def test_every_effort_has_settings():
    assert sorted(_compress.EFFORTS) == list(EFFORTS)
    for bad in (0, MAX_EFFORT + 1, 5.0, True, "5"):
        with pytest.raises(ValueError, match="effort"):
            Params(effort=bad)


@pytest.mark.parametrize("kind", ["random-walk", "sin-9.87hz", "linear", "noisy-sine"])
def test_efforts_decode_the_same(kind):
    x = minute(kind, 1)
    x[123] = np.nan  # a flagged block too
    reference = None
    for effort in EFFORTS:
        (unit,), *_ = fluxcode.encode(x, Params(effort=effort))
        frame = unit[_format.HEADER_BYTES:]
        assert zstandard.frame_content_size(frame) == len(body_of(unit))
        assert zstandard.ZstdDecompressor().decompress(frame) == bytes(body_of(unit))  # one shot, no section lengths
        values = fluxcode.decode_unit(unit).values
        if reference is None:
            reference = values
        np.testing.assert_array_equal(values, reference)


def test_large_unit_is_smaller_flushed():
    with planes("bit"):
        (unit,), *_ = fluxcode.encode(minute("random-walk", 3))
    assert len(unit) - _format.HEADER_BYTES < len(zstandard.ZstdCompressor(level=3).compress(bytes(body_of(unit))))


@pytest.mark.parametrize("byte_planes", [False, True])
def test_flush_points(byte_planes):
    with planes("byte" if byte_planes else "bit"):
        (unit,), *_ = fluxcode.encode(minute("sin-9.87hz", 4))
    parsed = _unit.decompress(unit)
    nb = parsed.header.num_blocks
    cuts = _compress.flush_points(parsed.raw_body, nb, parsed.layout, parsed.has_time, byte_planes)
    start = _format.residual_start(nb, parsed.has_time)
    assert cuts == sorted(set(cuts)) and cuts[0] == start and cuts[-1] <= len(parsed.raw_body)
    assert len(cuts) <= (3 if byte_planes else 17)
    groups = int(parsed.layout.group_offsets[-1])
    plane_bytes = 8 * groups if byte_planes else groups
    for cut in cuts[1:]:  # every cut after the columns ends a plane that holds data
        assert cut % plane_bytes == start % plane_bytes
        assert np.count_nonzero(parsed.raw_body[cut - plane_bytes:cut]) * _compress.FLUSH_MIN_DENSITY > plane_bytes


def test_flush_points_skip_empty_planes():
    with planes("bit"):
        (unit,), *_ = fluxcode.encode(minute("linear", 4))  # near-constant residuals
    parsed = _unit.decompress(unit)
    cuts = _compress.flush_points(parsed.raw_body, parsed.header.num_blocks, parsed.layout, parsed.has_time, False)
    assert len(cuts) < 17


def test_compress_body_cuts():
    body = np.random.default_rng(0).integers(0, 4, 5000, dtype=np.uint8)
    for cuts in ([], [0], [5000], [10, 4000], [1, 2, 3, 4999]):
        frame = _compress.compress_body(body, cuts, 3)
        assert zstandard.ZstdDecompressor().decompress(frame) == body.tobytes()
        assert zstandard.frame_content_size(frame) == 5000


def test_compress_body_gives_up_over_the_limit():
    body = np.random.default_rng(0).integers(0, 256, 20_000, dtype=np.uint8)  # incompressible: a block ~ its size
    cuts = [5000, 10_000, 15_000]
    frame = _compress.compress_body(body, cuts, 3)
    assert frame is not None
    assert _compress.compress_body(body, cuts, 3, limit=len(frame)) == frame
    assert _compress.compress_body(body, cuts, 3, limit=4000) is None  # dropped after the first block


def test_heuristic_choice():
    for kind, byte in (("sin-4.12hz", True), ("random-walk", False), ("chirp", False), ("noisy-sine", True)):
        (unit,), *_ = fluxcode.encode(minute(kind, 5), Params(effort=2))
        assert _format.unpack_header(unit).byte_planes is byte, kind


@pytest.mark.parametrize("kind", ["sin-9.87hz", "random-walk", "noisy-sine", "chirp", "gauss-spikes", "quadratic"])
def test_higher_efforts_are_never_larger(kind):
    """Effort 9 tries every candidate of effort 5 (and zstd 9), so no unit grows from 5 to 9
    (noisy-sine seed 1 was larger at zstd 9 alone)."""
    for seed in (1, 6):
        x = minute(kind, seed)
        sizes = [len(fluxcode.encode(x, Params(effort=effort))[0][0]) for effort in (9, 5)]
        assert sizes == sorted(sizes), seed


def test_efforts_trade_size():
    kinds = ("sin-9.87hz", "random-walk", "noisy-sine", "chirp", "gauss-spikes")
    size = lambda effort: sum(len(fluxcode.encode(minute(kind, 6), Params(effort=effort))[0][0]) for kind in kinds)
    assert size(5) <= size(4) <= 1.03 * size(5)  # flushed against one block run: smaller in total, not per unit
    assert size(4) <= size(2) <= size(1)


@pytest.mark.parametrize("kind", ["sin-4.12hz", "random-walk", "noisy-sine"])
def test_wide_share_is_the_same_in_both_layouts(kind):
    x = minute(kind, 3)
    bodies = []
    for layout in ("bit", "byte"):
        with planes(layout):
            (unit,), *_ = fluxcode.encode(x, Params(max_quantize_bits=12))
        bodies.append(_unit.decompress(unit))
    rows = unit_rows(fluxcode.encode(x, Params(max_quantize_bits=12))[0][0]).residuals.astype(np.int32)
    zigzag = ((rows << 1) ^ (rows >> 15)) & 0xFFFF
    for bit_idx in (0, 3, 7, 8, 12, 15):
        shares = [_bitpacking.wide_share(p.raw_body, p.header.num_blocks, int(p.layout.group_offsets[-1]), p.has_time,
                                         p.header.byte_planes, bit_idx) for p in bodies]
        slots = 8 * int(bodies[0].layout.group_offsets[-1])
        assert shares[0] == shares[1] == np.count_nonzero(zigzag >> bit_idx) / slots, bit_idx
