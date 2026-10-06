# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Compression policy: how a block group's body becomes its zstd frame.

Which residual layout to use, where zstd blocks end, which zstd levels to try and which
candidate frame to keep are encoder choices, not format: any frame of the body decodes.
The optional Rust extension (rust/src/compress.rs) makes the same choices without
holding the GIL; the constants here are checked against its in tests/test_rust.py.
"""

import importlib
import os
import threading
import warnings
from types import ModuleType
from typing import NamedTuple

import numpy as np
import zstandard

from . import _bitpacking, _format

RUST_INTERFACE_VERSION: int = 2
"""Version of the extension interface this code calls (compress_group, pack_group and the
policy constants).

rust/src/lib.rs exports the same INTERFACE_VERSION; the two are raised together when the
interface changes.
"""


def _load_rust() -> ModuleType | None:
    """Loads the optional Rust accelerator extension module if available.

    Returns:
        The loaded fluxcode_rs module, or None if unavailable or disabled.
    """
    if os.environ.get("FLUXCODE_RUST") == "0":
        return None
    try:
        module = importlib.import_module("fluxcode_rs")
    except ImportError:
        return None
    # An extension built before this interface (or after it changed) falls back to Python
    if getattr(module, "INTERFACE_VERSION", None) != RUST_INTERFACE_VERSION:
        warnings.warn(
            "the installed fluxcode_rs doesn't match this fluxcode (interface version "
            f"{getattr(module, 'INTERFACE_VERSION', None)}, expected {RUST_INTERFACE_VERSION}): using the Python "
            "path; rebuild it (uv sync --extra rust --reinstall-package fluxcode-rs)",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    return module


_rust: ModuleType | None = _load_rust()
"""The optional Rust accelerator (the fluxcode[rust] extra), or None.

It builds a block group's bytes from its rows or from a body without holding the GIL, which
python-zstandard does inside a block flush. It is None if the extension isn't installed
or FLUXCODE_RUST=0; block groups are the same either way (rust/README.md).
"""


class Effort(NamedTuple):
    """How a block group is compressed at one Params.effort.

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
    # zstd 9 alone is larger than zstd 3 on about a fifth of block groups (up to 9%): keep both
    9: Effort("best", True, (3, 9)),
}
"""Params.effort to Effort (ENCODER.md §1; the measured size and speed of each are in TUNING.md)."""

BYTE_PLANES_BIT: int = 7
BYTE_PLANES_MAX_SHARE: float = 0.01
"""Threshold of the heuristic layout.

It picks byte planes when fewer than BYTE_PLANES_MAX_SHARE of the (zigzagged) residuals
reach 2^BYTE_PLANES_BIT: narrow residuals, whose high byte is constant and whose low byte
byte-wise literals model well. Wider ones, even within a byte, compress better as bit
planes with a zstd block (and Huffman table) per plane (experimental/plane_layout, What
was adopted).
"""

FLUSH_MIN_DENSITY: int = 16
"""Density threshold for ending a zstd block after a residual plane.

A plane ends a block only if more than 1 / FLUSH_MIN_DENSITY of its bytes are non-zero: a
block costs tens of bytes of header and table, which a plane that is mostly zero (the
high planes of small residuals) doesn't repay, and which also slows encoding.
"""

_local = threading.local()


def zstd(level: int = 3) -> tuple[zstandard.ZstdCompressor, zstandard.ZstdDecompressor]:
    """Retrieves thread-local zstandard compressor and decompressor instances.

    Args:
        level: Zstandard compression level for the compressor.

    Returns:
        A tuple of (compressor, decompressor) dedicated to the current thread.
    """
    if not hasattr(_local, "decompressor"):
        # Separate compressors and decompressor per thread for thread-safety
        _local.decompressor = zstandard.ZstdDecompressor()
        _local.compressors = {}
    if level not in _local.compressors:
        _local.compressors[level] = zstandard.ZstdCompressor(level=level, write_checksum=False, write_content_size=True)
    return _local.compressors[level], _local.decompressor


def flush_points(
    raw_group: np.ndarray, num_blocks: int, offsets: _format.Layout, has_time: bool, byte_planes: bool
) -> list[int]:
    """Finds the body offsets where the encoder ends a zstd block.

    A block ends after the per-block columns, and after each residual plane (16 bit planes
    or 2 byte planes) that is dense enough to have statistics of its own. Every zstd block
    carries its own literal Huffman table, so such a plane is coded with the statistics of
    its own bytes (about 3% smaller than one block run for typical block groups); the result is
    still one zstd frame, which any decoder reads unchanged.

    Args:
        raw_group: 1D uint8 array of the uncompressed body.
        num_blocks: Number of blocks.
        offsets: The blocks' Layout.
        has_time: Whether the block group has a time axis.
        byte_planes: Whether the residuals are stored as byte planes.

    Returns:
        Strictly increasing offsets, empty for a block group with no residual plane bytes.
    """
    start = _format.residual_start(num_blocks, has_time)
    octets = int(offsets.octet_offsets[-1])
    plane_bytes, planes = (8 * octets, 2) if byte_planes else (octets, 16)
    if not octets:
        return []
    points = [start]
    for plane_idx in range(planes):
        plane_start = start + plane_idx * plane_bytes
        if np.count_nonzero(raw_group[plane_start:plane_start + plane_bytes]) * FLUSH_MIN_DENSITY > plane_bytes:
            points.append(plane_start + plane_bytes)
    return points


def compress_body(body: np.ndarray, cuts: list[int], zstd_level: int, limit: int | None = None) -> bytes | None:
    """Compresses a block group body into one zstd frame, ending a block at each cut.

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


def _smallest_group(
    candidates: list[tuple[bool, np.ndarray]],
    num_blocks: int,
    num_samples: int,
    offsets: _format.Layout,
    time_unit: int,
    effort: Effort,
) -> bytes:
    """Compresses each candidate body at each zstd level and builds the smallest block group.

    Ties favor byte planes (which decode faster without bit transposing), and
    then lower compression levels.

    Args:
        candidates: List of (byte_planes, body) candidate tuples to evaluate.
        num_blocks: Total number of blocks in the block group.
        num_samples: Total number of samples across all blocks.
        offsets: Layout offsets for the block group.
        time_unit: Time unit code (0 for no time axis).
        effort: Effort policy specifying compression levels and flush behavior.

    Returns:
        Serialized block group bytes containing header and compressed zstd frame.
    """
    has_time = time_unit != 0
    levels = effort.zstd_levels
    # Precompute flush points for candidates if block flushes are enabled
    cuts = {
        byte_planes: flush_points(body, num_blocks, offsets, has_time, byte_planes) if effort.flush else []
        for byte_planes, body in candidates
    }
    best: tuple[bytes, int, bool] | None = None
    # Try cheap compression levels first to establish early-termination limits
    for level_idx, level in enumerate(levels):
        for byte_planes, body in candidates:
            rank = level_idx if byte_planes else len(levels) + level_idx
            frame: bytes | None
            if effort.flush:
                # Early cutoff: abort if partial compressed output exceeds best frame so far
                limit = None if best is None else len(best[0]) - (0 if rank < best[1] else 1)
                frame = compress_body(body, cuts[byte_planes], level, limit)
            else:
                frame = zstd(level)[0].compress(body.data)
            # Track best frame by size, breaking ties with candidate rank
            if frame is not None and (best is None or len(frame) < len(best[0])
                                      or (len(frame) == len(best[0]) and rank < best[1])):
                best = (frame, rank, byte_planes)
    assert best is not None
    # Prepend 8-byte block group header to the winning compressed zstd frame
    return _format.pack_header(num_blocks, num_samples, best[2], time_unit) + best[0]


def pack(
    body: np.ndarray,
    offsets: _format.Layout,
    num_blocks: int,
    num_samples: int,
    effort: Effort,
    time_unit: int,
) -> bytes:
    """Compresses an uncompressed byte-plane body into complete block group bytes.

    Args:
        body: 1D uint8 array containing the uncompressed body, residuals as byte planes.
        offsets: The block group's Layout.
        num_blocks: Total number of blocks.
        num_samples: Total number of samples across all blocks.
        effort: Effort policy specifying layout strategy, flushes, and levels.
        time_unit: Time unit code (0 for no time axis).

    Returns:
        Serialized block group bytes containing header and compressed zstd frame.
    """
    has_time = time_unit != 0
    num_octets = int(offsets.octet_offsets[-1])
    # Fast path: delegate to Rust extension if available
    if _rust is not None:
        return _rust.pack_group(  # type: ignore[no-any-return]
            body, num_blocks, num_samples, num_octets, time_unit, effort.layout, effort.flush,
            list(effort.zstd_levels),
        )

    def byte_planes_predicted() -> bool:
        """Predicts whether byte planes compress better than bit planes."""
        share = _bitpacking.wide_share(body, num_blocks, num_octets, has_time, True, BYTE_PLANES_BIT)
        return bool(share < BYTE_PLANES_MAX_SHARE)

    def bit_planes() -> np.ndarray:
        """Returns the body with its residual region as bit planes."""
        return _bitpacking.to_bit_planes(body, num_blocks, num_octets, has_time)

    # Build candidate layouts according to effort policy
    if effort.layout == "byte" or (effort.layout == "heuristic" and byte_planes_predicted()):
        candidates = [(True, body)]
    elif effort.layout in ("bit", "heuristic"):
        candidates = [(False, bit_planes())]
    else:
        # Order candidates so the predicted winner runs first when flushes allow early cutoff
        bit_first = effort.flush and not byte_planes_predicted()
        candidates = [(False, bit_planes()), (True, body)] if bit_first else [(True, body), (False, bit_planes())]
    return _smallest_group(candidates, num_blocks, num_samples, offsets, time_unit, effort)


def compress(rows: _format.GroupRows, num_samples: int, effort: Effort, time_unit: int = 0) -> bytes:
    """Serializes block group rows and builds the block group: header plus zstd frame of the body.

    Args:
        rows: The block group's rows (time_rows None for a block group without a time axis).
        num_samples: Sample count recorded in the header (the sum of the block sizes).
        effort: How to compress (Params.effort).
        time_unit: Time unit code recorded in the header (0 without a time axis).

    Returns:
        The block group bytes.
    """
    if _rust is not None and rows.time_rows is None:
        return _rust.compress_group(  # type: ignore[no-any-return]
            rows.block_flags, rows.block_sizes, rows.grid_params, rows.value_anchors, rows.residuals, rows.codes,
            effort.layout, effort.flush, list(effort.zstd_levels),
        )
    offsets = _format.layout(rows.block_flags, rows.block_sizes)

    # Separate from the encode kernel; fusing measured no gain (PERFORMANCE.md).
    body = _bitpacking.write_group(
        *(rows.block_flags, rows.block_sizes, rows.grid_params, rows.value_anchors, rows.residuals, rows.codes),
        byte_planes=True, time_rows=rows.time_rows, offsets=offsets,
    )
    return pack(body, offsets, rows.block_flags.shape[0], num_samples, effort, time_unit)
