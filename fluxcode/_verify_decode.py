# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Wire-format decoding and reference assembly helpers for ``fluxcode.verify``."""

from typing import NamedTuple

import numpy as np

from . import _bitpacking, _format, _group
from ._verify_common import Violation


class Decoded(NamedTuple):
    """Decoded block group with its wire-format grid metadata and escape masks."""

    unit: int
    sizes: np.ndarray
    offsets: np.ndarray
    flags: np.ndarray
    exponents: np.ndarray
    scales: np.ndarray
    values: np.ndarray
    ticks: np.ndarray | None
    escaped: np.ndarray
    escaped_samples: np.ndarray

    @property
    def num_blocks(self) -> int:
        return int(self.sizes.shape[0])


def decode(group: bytes) -> Decoded:
    """Decompresses and decodes ``group``, extracting per-block grid metadata and
    escaped sample masks.
    """
    parsed = _group.decompress(group)
    num_blocks = parsed.header.num_blocks
    values, ticks = _group.decode_blocks(parsed, np.arange(num_blocks, dtype=np.int64))
    raw = parsed.body.astype(np.int64)
    # Unpack byte-planed int16 grid_exponents and sign-extend bit 15.
    start = _format.grid_exponents_start(num_blocks)
    grid_exponents = raw[start : start + num_blocks] | (
        raw[start + num_blocks : start + 2 * num_blocks] << 8
    )
    grid_exponents -= (grid_exponents & 0x8000) << 1
    # Unpack byte-planed uint32 grid_scales.
    start = _format.grid_scales_start(num_blocks)
    scales = np.zeros(num_blocks, np.int64)
    for byte_idx in range(4):
        scales |= raw[
            start + byte_idx * num_blocks : start + (byte_idx + 1) * num_blocks
        ] << (8 * byte_idx)
    sizes = parsed.block_sizes.astype(np.int64)
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    escaped = np.diff(parsed.escape_offsets) > 0
    escaped_samples = np.zeros(parsed.header.num_samples, np.bool_)
    if escaped.any():
        layout = parsed.layout
        code_planes = _bitpacking.sample_code_planes_view(
            parsed.body,
            num_blocks,
            int(layout.plane_byte_offsets[-1]),
            int(layout.code_plane_offsets[-1]),
            parsed.has_time,
        )
        for block_idx in np.flatnonzero(escaped).tolist():
            # Plane 1 bit is 1 iff sample code is CODE_ESCAPE (10b) or
            # CODE_REPEAT_ESCAPE (11b).
            c0, c1 = (
                int(layout.code_plane_offsets[block_idx]),
                int(layout.code_plane_offsets[block_idx + 1]),
            )
            s0, s1 = int(offsets[block_idx]), int(offsets[block_idx + 1])
            escaped_samples[s0:s1] = np.unpackbits(
                code_planes[1, c0:c1], bitorder="little"
            )[: s1 - s0].astype(np.bool_)
    return Decoded(
        parsed.header.time_unit,
        sizes,
        offsets,
        parsed.block_flags.astype(np.int64),
        grid_exponents,
        scales,
        values,
        ticks,
        escaped,
        escaped_samples,
    )


def decode_or_report(group: bytes, found: list[Violation]) -> Decoded | None:
    """Decodes ``group``, appending a ``"decode"`` violation and returning None on any
    error.
    """
    try:
        return decode(group)
    except Exception as exc:
        found.append(Violation("decode", -1, f"{type(exc).__name__}: {exc}"))
        return None


def block_positions(
    offsets: np.ndarray, which: np.ndarray, sizes: np.ndarray
) -> np.ndarray:
    """Returns flat sample indices of the blocks selected by boolean mask ``which``, in
    block order.
    """
    lengths = np.where(which, sizes, 0)
    return np.asarray(
        np.arange(int(lengths.sum()))
        + np.repeat(offsets[:-1] - (np.cumsum(lengths) - lengths), lengths),
        dtype=np.int64,
    )


def assemble(
    sizes: np.ndarray,
    sources: np.ndarray,
    values: list[np.ndarray],
    ticks: list[np.ndarray] | None,
    positions: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray | None]:
    """Interleaves blocks from multiple source arrays according to per-block ``sources``
    indices.
    """
    out_values = np.empty(int(sizes.sum()))
    out_ticks = None if ticks is None else np.empty(out_values.shape[0], np.int64)
    sample_sources = np.repeat(sources, sizes)
    for source_idx, source_positions in enumerate(positions):
        target = sample_sources == source_idx
        out_values[target] = values[source_idx][source_positions]
        if out_ticks is not None and ticks is not None:
            out_ticks[target] = ticks[source_idx][source_positions]
    return out_values, out_ticks
