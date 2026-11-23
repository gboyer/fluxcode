# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Compression policy: turning an uncompressed block group body into a zstd-framed group.

Selects between byte-plane and bit-plane residual layouts, optionally inserts zstd block
flushes between metadata columns and dense residual planes so each plane gets its own
Huffman table, tries configured zstd compression levels, and keeps the smallest valid frame.
When installed, the optional Rust extension (``fluxcode_rs``) runs the same policy without
holding the Python GIL.
"""

import importlib
import os
import threading
import warnings
from types import ModuleType
from typing import NamedTuple

import numpy as np
import zstandard

from . import _bitpacking, _body, _escapes, _format

RUST_INTERFACE_VERSION: int = 7
"""Lockstep version guard matching ``fluxcode_rs.INTERFACE_VERSION``.

``fluxcode`` and ``fluxcode_rs`` are always developed and deployed in lockstep (no cross-version
ABI compatibility is intended). Bump this freely whenever the Python<->Rust boundary changes so
stale local builds fall back cleanly with a warning.
"""


def _load_rust() -> ModuleType | None:
    """Loads the optional ``fluxcode_rs`` accelerator if installed and interface-compatible.

    Returns:
        The imported ``fluxcode_rs`` module, or None if disabled (``FLUXCODE_RUST=0``),
        not installed, or built against a different interface version.
    """
    if os.environ.get("FLUXCODE_RUST") == "0":
        return None
    try:
        module = importlib.import_module("fluxcode_rs")
    except ImportError:
        return None
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
"""Optional Rust accelerator module (``fluxcode[rust]``), or None if unavailable."""


class Effort(NamedTuple):
    """Compression policy for a single ``Params.effort`` level.

    Attributes:
        layout: ``"heuristic"`` (choose byte vs. bit planes from residual width and compress once)
            or ``"best"`` (compress both layouts and keep the smaller frame).
        flush: Whether to end a zstd block after the metadata columns and each dense residual plane.
        zstd_levels: Tuple of zstd compression levels to try, keeping the smallest output.
    """

    layout: str
    flush: bool
    zstd_levels: tuple[int, ...]


EFFORTS: dict[int, Effort] = {
    1: Effort("heuristic", False, (1,)),
    2: Effort("heuristic", False, (3,)),
    **dict.fromkeys(range(3, 5), Effort("best", False, (3,))),
    # Zstd block flushes hold the GIL in python-zstandard, so they begin above the default effort (4).
    **dict.fromkeys(range(5, 9), Effort("best", True, (3,))),
    # Level 9 occasionally expands relative to level 3; try both and keep the smaller frame.
    9: Effort("best", True, (3, 9)),
}
"""Mapping from ``Params.effort`` (1..9) to its ``Effort`` policy."""

BYTE_PLANES_BIT: int = 7
BYTE_PLANES_MAX_SHARE: float = 0.01
"""Heuristic layout threshold: pick byte planes when fewer than 1% of zigzagged residuals reach 2^7."""

FLUSH_MIN_DENSITY: int = 16
"""Minimum non-zero byte fraction (``> 1 / 16``) for a residual plane to earn its own zstd block."""

_local = threading.local()


def get_zstd_codecs(level: int = 3) -> tuple[zstandard.ZstdCompressor, zstandard.ZstdDecompressor]:
    """Returns thread-local ``(compressor, decompressor)`` instances for ``level``.

    Args:
        level: Zstandard compression level for the returned compressor.

    Returns:
        Tuple ``(ZstdCompressor, ZstdDecompressor)`` private to the calling thread.
    """
    if not hasattr(_local, "decompressor"):
        _local.decompressor = zstandard.ZstdDecompressor()
        _local.compressors = {}
    if level not in _local.compressors:
        _local.compressors[level] = zstandard.ZstdCompressor(
            level=level, write_checksum=False, write_content_size=True
        )
    return _local.compressors[level], _local.decompressor


def find_flush_points(
    body: np.ndarray, num_blocks: int, offsets: _format.BodyLayout, has_time: bool, byte_planes: bool
) -> list[int]:
    """Computes uncompressed body offsets where zstd blocks should be flushed.

    Flushing after the metadata columns and after each dense residual plane gives each plane
    its own Huffman table inside a single standard zstd frame.

    Args:
        body: 1D uint8 array of uncompressed body bytes.
        num_blocks: Number of blocks in the group.
        offsets: The group's ``BodyLayout``.
        has_time: Whether the group has a time axis.
        byte_planes: True for 2 byte planes, False for 16 bit planes.

    Returns:
        Strictly increasing list of byte offsets, or an empty list if there are no residuals.
    """
    # Always cut at the boundary between metadata columns and residual planes, then cut after
    # each residual plane whose non-zero byte density exceeds 1 / FLUSH_MIN_DENSITY.
    start = _format.residual_planes_start(num_blocks, has_time)
    num_plane_bytes = int(offsets.plane_byte_offsets[-1])
    if not num_plane_bytes:
        return []
    plane_len, num_planes = (
        (_format.SAMPLES_PER_PLANE_BYTE * num_plane_bytes, 2) if byte_planes else (num_plane_bytes, 16)
    )
    points = [start]
    for plane_idx in range(num_planes):
        plane_start = start + plane_idx * plane_len
        if np.count_nonzero(body[plane_start : plane_start + plane_len]) * FLUSH_MIN_DENSITY > plane_len:
            points.append(plane_start + plane_len)
    return points


def compress_body_with_flushes(
    body: np.ndarray, cuts: list[int], zstd_level: int, limit: int | None = None
) -> bytes | None:
    """Compresses ``body`` into a single zstd frame, flushing a zstd block at each cut offset.

    Args:
        body: 1D uint8 array of uncompressed body bytes.
        cuts: Strictly increasing byte offsets at which to flush zstd blocks.
        zstd_level: Zstandard compression level.
        limit: Optional byte budget; aborts early and returns None once emitted bytes exceed this.

    Returns:
        Compressed zstd frame bytes, or None if the output exceeds ``limit`` before completion.
    """
    # Stream slices between consecutive cut offsets into a single zstd frame with explicit
    # block flushes, checking the running size against `limit` to prune losing candidates early.
    compressor = get_zstd_codecs(zstd_level)[0].compressobj(size=body.shape[0])
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


def select_smallest_group(
    candidates: list[tuple[bool, np.ndarray]],
    num_blocks: int,
    num_samples: int,
    offsets: _format.BodyLayout,
    time_unit: int,
    effort: Effort,
    group_time_step: bool = False,
) -> bytes:
    """Compresses all candidate body layouts across ``effort.zstd_levels`` and returns the smallest group.

    Ties prefer byte planes (faster to decode without bit transposing), then lower zstd levels.

    Args:
        candidates: List of ``(is_byte_planes, body)`` pairs to evaluate.
        num_blocks: Number of blocks in the group.
        num_samples: Total sample count in the group.
        offsets: The group's ``BodyLayout``.
        time_unit: ``TimeUnitCode`` integer (0 for no time axis).
        effort: Active ``Effort`` compression policy.
        group_time_step: Whether a ``group_time_step`` trailer is appended to each candidate body.

    Returns:
        Complete compressed block group bytes (8-byte header + winning zstd frame).
    """
    # Precompute flush cuts if enabled, then evaluate cheaper zstd levels first so any later
    # flushed candidate can abort early as soon as it exceeds the best frame found so far.
    has_time = time_unit != 0
    levels = effort.zstd_levels
    cuts = {
        byte_planes: find_flush_points(body, num_blocks, offsets, has_time, byte_planes) if effort.flush else []
        for byte_planes, body in candidates
    }
    best: tuple[bytes, int, bool] | None = None
    for level_idx, level in enumerate(levels):
        for byte_planes, body in candidates:
            # Lower rank wins ties: all byte-plane levels outrank all bit-plane levels.
            rank = level_idx if byte_planes else len(levels) + level_idx
            frame: bytes | None
            if effort.flush:
                limit = None if best is None else len(best[0]) - (0 if rank < best[1] else 1)
                frame = compress_body_with_flushes(body, cuts[byte_planes], level, limit)
            else:
                frame = get_zstd_codecs(level)[0].compress(body.data)
            if frame is not None and (
                best is None or len(frame) < len(best[0]) or (len(frame) == len(best[0]) and rank < best[1])
            ):
                best = (frame, rank, byte_planes)
    assert best is not None
    return _format.pack_header(num_blocks, num_samples, best[2], time_unit, group_time_step) + best[0]


def pack_body(
    body: np.ndarray,
    offsets: _format.BodyLayout,
    num_blocks: int,
    num_samples: int,
    effort: Effort,
    time_unit: int,
) -> bytes:
    """Factors out any common ``group_time_step`` and compresses a byte-plane body into a block group.

    Args:
        body: 1D uint8 array of an uncompressed byte-plane body (including escaped-value trailer).
        offsets: The group's ``BodyLayout``.
        num_blocks: Number of blocks in the group.
        num_samples: Total sample count across all blocks.
        effort: Active ``Effort`` compression policy.
        time_unit: ``TimeUnitCode`` integer (0 for no time axis).

    Returns:
        Complete compressed block group bytes (8-byte header + zstd frame).
    """
    # If timed and all time_start increases and time_steps share a GCD > 1, divide them in a
    # copy of the body and append the 8-byte uint64 group_time_step trailer. Then dispatch to
    # Rust if available, or run the Python layout selection and zstd compression.
    has_time = time_unit != 0
    has_group_time_step = False
    if has_time:
        step = _format.group_time_step(body, offsets.sample_offsets)
        if step > 1:
            body = body.copy()
            _format.scale_time_columns(body, offsets.sample_offsets, step, True)
            body = np.concatenate([body, np.array([step], np.uint64).view(np.uint8)])
            has_group_time_step = True
    num_plane_bytes = int(offsets.plane_byte_offsets[-1])
    if _rust is not None:
        return _rust.pack_group(  # type: ignore[no-any-return]
            body,
            num_blocks,
            num_samples,
            num_plane_bytes,
            time_unit,
            effort.layout,
            effort.flush,
            list(effort.zstd_levels),
            has_group_time_step,
        )

    def predict_byte_planes() -> bool:
        """Predicts whether byte planes will compress smaller than bit planes."""
        share = _bitpacking.wide_residual_fraction(body, num_blocks, num_plane_bytes, has_time, BYTE_PLANES_BIT)
        return bool(share < BYTE_PLANES_MAX_SHARE)

    def as_bit_planes() -> np.ndarray:
        """Returns a copy of the body with its residual region transposed to 16 bit planes."""
        return _bitpacking.to_bit_planes(body, num_blocks, num_plane_bytes, has_time)

    if effort.layout == "heuristic":
        candidates = [(True, body)] if predict_byte_planes() else [(False, as_bit_planes())]
    else:
        # Run the predicted winner first when block flushes are enabled so early cutoff triggers sooner.
        bit_first = effort.flush and not predict_byte_planes()
        candidates = [(False, as_bit_planes()), (True, body)] if bit_first else [(True, body), (False, as_bit_planes())]
    return select_smallest_group(candidates, num_blocks, num_samples, offsets, time_unit, effort, has_group_time_step)


def compress_columns(columns: _format.EncodedBlockColumns, num_samples: int, effort: Effort, time_unit: int) -> bytes:
    """Serializes ``EncodedBlockColumns`` into a body and compresses it into a block group.

    Args:
        columns: In-memory encoded block columns (``time_columns`` is None when untimed).
        num_samples: Total sample count recorded in the header.
        effort: Active ``Effort`` compression policy.
        time_unit: ``TimeUnitCode`` integer (0 for no time axis).

    Returns:
        Complete compressed block group bytes (8-byte header + zstd frame).
    """
    # Untimed groups can take the zero-copy Rust fast path directly from columnar arrays.
    if _rust is not None and columns.time_columns is None:
        return _rust.compress_group(  # type: ignore[no-any-return]
            columns.block_flags,
            columns.block_sizes,
            columns.grid_exponents,
            columns.grid_scales,
            columns.block_anchors,
            columns.residuals,
            columns.sample_codes,
            columns.escaped_values,
            effort.layout,
            effort.flush,
            list(effort.zstd_levels),
        )
    offsets = _format.compute_body_layout(columns.block_flags, columns.block_sizes)
    body = _body.write_body(
        columns.block_flags,
        columns.block_sizes,
        columns.grid_exponents,
        columns.grid_scales,
        columns.block_anchors,
        columns.residuals,
        columns.sample_codes,
        columns.time_columns,
        offsets,
    )
    if columns.escaped_values.shape[0]:
        body = np.concatenate([body, _escapes.pack_escaped_values(columns.escaped_values)])
    return pack_body(body, offsets, columns.block_flags.shape[0], num_samples, effort, time_unit)
