# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Block flushing inside the unit's one zstd frame, and planes="heuristic": the frame stays an
ordinary one (content size recorded, one-shot decodable), the blocks end where the planes do, and
the heuristic picks a layout from the residuals' high byte."""

import numpy as np
import pytest
import zstandard
from _signals import minute

import fluxcode
from fluxcode import Params, _format, _unit


def body_of(unit):
    return _unit.decompress(unit).raw_body


def test_frame_is_ordinary():
    x = minute("random-walk", 1)
    for mode in ("bit", "byte", "best", "heuristic"):
        (unit,), *_ = fluxcode.encode(x, Params(planes=mode))
        frame = unit[_format.HEADER_BYTES:]
        assert zstandard.frame_content_size(frame) == len(body_of(unit))
        assert zstandard.ZstdDecompressor().decompress(frame) == bytes(body_of(unit))  # one shot, no section lengths


@pytest.mark.parametrize("kind", ["random-walk", "sin-9.87hz", "linear"])
def test_flushed_frame_decodes_like_a_single_run(kind):
    (unit,), *_ = fluxcode.encode(minute(kind, 2), Params(planes="bit"))
    body = body_of(unit)
    single = unit[:_format.HEADER_BYTES] + zstandard.ZstdCompressor(level=3, write_content_size=True).compress(bytes(body))
    assert np.array_equal(fluxcode.decode([unit])[0].values, fluxcode.decode([single])[0].values)
    if len(single) - _format.HEADER_BYTES < _unit.FLUSH_RETRY_BYTES:  # small: also compressed in one run, the smaller kept
        assert len(unit) <= len(single)


def test_large_unit_is_smaller_flushed():
    (unit,), *_ = fluxcode.encode(minute("random-walk", 3), Params(planes="bit"))
    body = bytes(body_of(unit))
    assert len(unit) - _format.HEADER_BYTES < len(zstandard.ZstdCompressor(level=3).compress(body))


@pytest.mark.parametrize("byte_planes", [False, True])
def test_flush_points(byte_planes):
    (unit,), *_ = fluxcode.encode(minute("sin-9.87hz", 4), Params(planes="byte" if byte_planes else "bit"))
    parsed = _unit.decompress(unit)
    nb = parsed.header.num_blocks
    cuts = _format.flush_points(parsed.raw_body, nb, parsed.layout, parsed.has_time, byte_planes)
    start = _format.residual_start(nb, parsed.has_time)
    assert cuts == sorted(set(cuts)) and cuts[0] == start and cuts[-1] <= len(parsed.raw_body)
    assert len(cuts) <= (3 if byte_planes else 17)
    groups = int(parsed.layout.group_offsets[-1])
    plane_bytes = 8 * groups if byte_planes else groups
    for b in cuts[1:]:  # every cut after the columns ends a plane that holds data
        assert b % plane_bytes == start % plane_bytes
        assert np.count_nonzero(parsed.raw_body[b - plane_bytes:b]) * _format.FLUSH_MIN_DENSITY > plane_bytes


def test_flush_points_skip_empty_planes():
    (unit,), *_ = fluxcode.encode(minute("linear", 4), Params(planes="bit"))  # near-constant residuals
    parsed = _unit.decompress(unit)
    cuts = _format.flush_points(parsed.raw_body, parsed.header.num_blocks, parsed.layout, parsed.has_time, False)
    assert len(cuts) < 17


def test_compress_body_cuts():
    body = np.frombuffer(np.random.default_rng(0).integers(0, 4, 5000, dtype=np.uint8).tobytes(), np.uint8)
    for cuts in ([], [0], [5000], [10, 10, 4000], [1, 2, 3, 4999]):
        frame = _unit.compress_body(body, sorted(set(cuts)))
        assert zstandard.ZstdDecompressor().decompress(frame) == body.tobytes()
        assert zstandard.frame_content_size(frame) == 5000


def test_heuristic_choice():
    for kind, byte in (("sin-4.12hz", True), ("random-walk", False), ("chirp", False), ("noisy-sine", True)):
        (unit,), *_ = fluxcode.encode(minute(kind, 5), Params(planes="heuristic"))
        assert _format.unpack_header(unit).byte_planes is byte, kind


def test_heuristic_close_to_best():
    sizes = {m: sum(len(fluxcode.encode(minute(k, 6), Params(planes=m))[0][0])
                    for k in ("sin-9.87hz", "random-walk", "noisy-sine", "chirp", "gauss-spikes"))
             for m in ("best", "heuristic", "bit")}
    assert sizes["best"] <= sizes["heuristic"] <= 1.03 * sizes["best"]
    assert sizes["heuristic"] <= sizes["bit"]
