# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Compression policy: how a unit's body becomes its zstd frame.

Which residual layout to use, where zstd blocks end, which zstd levels to try and which candidate
frame to keep are encoder choices, not format: any frame of the body decodes. The optional Rust
extension (rust/src/compress.rs) makes the same choices without holding the GIL; the constants
here are checked against its in tests/test_rust.py.
"""

import importlib
import os
import threading
from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import zstandard

from . import _bitpacking, _format


def _load_rust() -> Callable[..., bytes] | None:
    """The extension's compress_unit, or None if it isn't installed or FLUXCODE_RUST=0."""
    if os.environ.get("FLUXCODE_RUST") == "0":
        return None
    try:
        return vars(importlib.import_module("fluxcode_rs"))["compress_unit"]  # type: ignore[no-any-return]
    except ImportError:
        return None


_compress_unit: Callable[..., bytes] | None = _load_rust()
"""The optional Rust accelerator's compress_unit (the fluxcode[rust] extra): builds a unit's bytes without
holding the GIL, which python-zstandard does inside a block flush. None if it isn't installed or
FLUXCODE_RUST=0; units are the same either way (rust/README.md)."""


class Effort(NamedTuple):
    """How a unit is compressed at one Params.effort.

    Attributes:
        layout: "heuristic" (byte planes when few residuals reach 128, else bit planes, one
            compression), "best" (both, the smaller kept), "bit" or "byte" (tests only).
        flush: Whether a zstd block ends after the columns and each dense residual plane.
        zstd_levels: The zstd compression levels tried, the smallest frame kept.
    """

    layout: str
    flush: bool
    zstd_levels: tuple[int, ...]


EFFORTS: dict[int, Effort] = {
    1: Effort("heuristic", False, (1,)),
    2: Effort("heuristic", False, (3,)),
    **dict.fromkeys(range(3, 5), Effort("best", False, (3,))),
    # Block flushes gain 1.5-2% but hold the GIL inside python-zstandard's flush(): 4 threads
    # encode 40% slower, so they start above the default (TUNING.md, Effort)
    **dict.fromkeys(range(5, 9), Effort("best", True, (3,))),
    # zstd 9 alone is larger than zstd 3 on about a fifth of units (up to 9%): keep both
    9: Effort("best", True, (3, 9)),
}
"""Params.effort to Effort (SPEC.md §1; the measured size and speed of each are in TUNING.md)."""

BYTE_PLANES_BIT: int = 7
BYTE_PLANES_MAX_SHARE: float = 0.01
"""The heuristic layout picks byte planes when fewer than BYTE_PLANES_MAX_SHARE of the (zigzagged)
residuals reach 2^BYTE_PLANES_BIT: narrow residuals, whose high byte is constant and whose low byte
byte-wise literals model well. Wider ones, even within a byte, compress better as bit planes with
a zstd block (and Huffman table) per plane (experimental/plane_layout, What was adopted)."""

FLUSH_MIN_DENSITY: int = 16
"""A residual plane ends a zstd block only if more than 1 / FLUSH_MIN_DENSITY of its bytes are
non-zero: a block costs tens of bytes of header and table, which a plane that is mostly zero
(the high planes of small residuals) doesn't repay, and which also slows encoding."""

_local = threading.local()


def zstd(level: int = 3) -> tuple[zstandard.ZstdCompressor, zstandard.ZstdDecompressor]:
    """Retrieves thread-local zstandard compressor (at level) and decompressor instances.

    Returns:
        A tuple of (compressor, decompressor) dedicated to the current thread.
    """
    codecs = getattr(_local, "z", None)
    if codecs is None:
        # Separate compressors and decompressor per thread for thread-safety
        codecs = _local.z = {"decompressor": zstandard.ZstdDecompressor()}
    if level not in codecs:
        codecs[level] = zstandard.ZstdCompressor(level=level, write_checksum=False, write_content_size=True)
    return codecs[level], codecs["decompressor"]


def flush_points(
    raw_unit: np.ndarray, num_blocks: int, offsets: _format.Layout, has_time: bool, byte_planes: bool
) -> list[int]:
    """Body offsets where the encoder ends a zstd block: after the per-block columns, and after
    each residual plane (16 bit planes or 2 byte planes) that is dense enough to have statistics
    of its own. Every zstd block carries its own literal Huffman table, so such a plane is coded
    with the statistics of its own bytes (about 3% smaller than one block run for typical
    units); the result is still one zstd frame, which any decoder reads unchanged.

    Args:
        raw_unit: 1D uint8 array of the uncompressed body.
        num_blocks: Number of blocks.
        offsets: The blocks' Layout.
        has_time: Whether the unit has a time axis.
        byte_planes: Whether the residuals are stored as byte planes.

    Returns:
        Strictly increasing offsets, empty for a unit with no residual plane bytes.
    """
    start = _format.residual_start(num_blocks, has_time)
    groups = int(offsets.group_offsets[-1])
    plane_bytes, planes = (8 * groups, 2) if byte_planes else (groups, 16)
    if not groups:
        return []
    points = [start]
    for plane_idx in range(planes):
        plane_start = start + plane_idx * plane_bytes
        if np.count_nonzero(raw_unit[plane_start:plane_start + plane_bytes]) * FLUSH_MIN_DENSITY > plane_bytes:
            points.append(plane_start + plane_bytes)
    return points


def compress_body(body: np.ndarray, cuts: list[int], zstd_level: int, limit: int | None = None) -> bytes | None:
    """Compresses a unit body into one zstd frame, ending a block at each cut.

    Every zstd block has its own literal Huffman table, so cutting the body between its
    columns and planes codes each with its own statistics. The frame is an ordinary one that
    records the content size.

    Args:
        body: 1D uint8 array of the uncompressed body.
        cuts: Strictly increasing body offsets where a block ends.
        zstd_level: The zstd compression level.
        limit: Gives up (None) as soon as the blocks written so far are longer than this: the
            frame can only grow.

    Returns:
        The frame, or None if it would be longer than limit.
    """
    compressor = zstd(zstd_level)[0].compressobj(size=body.shape[0])
    view = body.data
    parts = []
    written = 0
    previous = 0
    for cut in cuts:
        if previous < cut < body.shape[0]:
            parts.append(compressor.compress(view[previous:cut]))
            parts.append(compressor.flush(zstandard.COMPRESSOBJ_FLUSH_BLOCK))
            previous = cut
            written += len(parts[-2]) + len(parts[-1])
            if limit is not None and written > limit:
                return None
    parts.append(compressor.compress(view[previous:]))
    parts.append(compressor.flush())
    return b"".join(parts)


def _smallest_unit(
    candidates: list[tuple[bool, np.ndarray]], num_blocks: int, num_samples: int, offsets: _format.Layout,
    time_unit: int, effort: Effort,
) -> bytes:
    """The unit of the smallest frame of the candidate bodies (byte planes or not) over effort's zstd
    levels. Ties go to byte planes (they decode faster: no bit transpose), then to the earlier level,
    whatever the order the candidates are tried in; with flushes, a candidate that can't beat the best
    so far is dropped as soon as its blocks show it. All candidates are tried at a level before the
    next level, so the cheap level's frames set the limit for the expensive ones."""
    has_time = time_unit != 0
    levels = effort.zstd_levels
    cuts = {
        byte_planes: flush_points(body, num_blocks, offsets, has_time, byte_planes) if effort.flush else []
        for byte_planes, body in candidates
    }
    best: tuple[bytes, int, bool] | None = None
    for level_idx, level in enumerate(levels):
        for byte_planes, body in candidates:
            rank = level_idx if byte_planes else len(levels) + level_idx
            frame: bytes | None
            if effort.flush:
                # A frame of this length or less wins (a tie only against a worse rank)
                limit = None if best is None else len(best[0]) - (0 if rank < best[1] else 1)
                frame = compress_body(body, cuts[byte_planes], level, limit)
            else:
                frame = zstd(level)[0].compress(body.data)
            if frame is not None and (best is None or len(frame) < len(best[0])
                                      or (len(frame) == len(best[0]) and rank < best[1])):
                best = (frame, rank, byte_planes)
    assert best is not None
    return _format.pack_header(num_blocks, num_samples, best[2], time_unit) + best[0]


def pack(
    build_body: Callable[[], tuple[np.ndarray, _format.Layout]],
    num_blocks: int,
    num_samples: int,
    effort: Effort,
    time_unit: int,
) -> bytes:
    """The unit of the byte-plane body build_body() returns (with its Layout), compressed as
    effort says. The byte-plane body is the cheaper one to write: the bit-plane body is derived
    from it by one transpose of the residual region, and only if it is wanted."""
    has_time = time_unit != 0
    body, offsets = build_body()
    num_groups = int(offsets.group_offsets[-1])

    def byte_planes_predicted() -> bool:
        share = _bitpacking.wide_share(body, num_blocks, num_groups, has_time, True, BYTE_PLANES_BIT)
        return bool(share < BYTE_PLANES_MAX_SHARE)

    def bit_planes() -> np.ndarray:
        return _bitpacking.to_bit_planes(body, num_blocks, num_groups, has_time)

    if effort.layout == "byte" or (effort.layout == "heuristic" and byte_planes_predicted()):
        candidates = [(True, body)]
    elif effort.layout in ("bit", "heuristic"):
        candidates = [(False, bit_planes())]
    else:
        # With flushes the heuristic's pick goes first, since the other is often dropped partway;
        # without them nothing can be dropped and the order is free (so the statistic isn't computed)
        bit_first = effort.flush and not byte_planes_predicted()
        candidates = [(False, bit_planes()), (True, body)] if bit_first else [(True, body), (False, bit_planes())]
    return _smallest_unit(candidates, num_blocks, num_samples, offsets, time_unit, effort)


def compress(rows: _format.UnitRows, num_samples: int, effort: Effort, time_unit: int = 0) -> bytes:
    """Serializes unit rows and builds the unit: header plus zstd frame of the body.

    Args:
        rows: The unit's rows (time_rows None for a unit without a time axis).
        num_samples: Sample count recorded in the header (the sum of the block sizes).
        effort: How to compress (Params.effort).
        time_unit: Time unit code recorded in the header (0 without a time axis).

    Returns:
        The unit bytes.
    """
    if _compress_unit is not None and rows.time_rows is None:
        return _compress_unit(
            rows.block_flags, rows.block_sizes, rows.grid_params, rows.value_anchors, rows.residuals, rows.codes,
            effort.layout, effort.flush, list(effort.zstd_levels),
        )
    offsets = _format.layout(rows.block_flags, rows.block_sizes)

    def build_body() -> tuple[np.ndarray, _format.Layout]:
        # Separate from the encode kernel; fusing measured no gain (PERFORMANCE.md).
        body = _bitpacking.write_unit(
            *(rows.block_flags, rows.block_sizes, rows.grid_params, rows.value_anchors, rows.residuals, rows.codes),
            byte_planes=True, time_rows=rows.time_rows, offsets=offsets,
        )
        return body, offsets

    return pack(build_body, rows.block_flags.shape[0], num_samples, effort, time_unit)
