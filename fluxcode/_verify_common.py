# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Shared types, constants, and reporting helpers for ``fluxcode.verify``."""

from typing import NamedTuple

import numpy as np

TIME_SLACK: float = 1.02
"""Maximum relative slack above ``time_error * mean_interval`` allowed for rounded timestamps."""

NOISE_FLOOR_MIN_SAMPLES: int = 128
"""Blocks with fewer samples than this never have the noise floor applied."""

MAX_REPORTED: int = 5
"""Maximum number of failing blocks reported per check."""


class Violation(NamedTuple):
    """A single guarantee violation detected during verification.

    Attributes:
        check: Name of the violated guarantee (see ``fluxcode.verify`` module docstring).
        block: Zero-based block index, or ``-1`` for group-level violations.
        message: Human-readable explanation of the observed mismatch and bound.
    """

    check: str
    block: int
    message: str

    def __str__(self) -> str:
        where = "block group" if self.block < 0 else f"block {self.block}"
        return f"{self.check}: {where}: {self.message}"


def same_doubles_mask(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise equality treating NaN == NaN and -0.0 == 0.0."""
    with np.errstate(invalid="ignore"):
        return np.asarray((a == b) | (np.isnan(a) & np.isnan(b)), dtype=np.bool_)


def same_doubles(a: np.ndarray, b: np.ndarray) -> bool:
    """Returns True if ``a`` and ``b`` have the same shape and match element-wise (NaN == NaN)."""
    return a.shape == b.shape and bool(same_doubles_mask(a, b).all())


def block_of(offsets: np.ndarray, sample_idx: int) -> int:
    """Returns the block index containing flat sample index ``sample_idx``."""
    return int(np.searchsorted(offsets, sample_idx, side="right")) - 1


def report_blocks(
    found: list[Violation],
    check: str,
    which: np.ndarray,
    counts: np.ndarray,
    firsts: np.ndarray,
    what: str,
) -> None:
    """Appends up to ``MAX_REPORTED`` block-level violations from precomputed per-block counts."""
    for block_idx in np.flatnonzero(which & (counts > 0))[:MAX_REPORTED].tolist():
        found.append(Violation(check, block_idx, f"{what}: {counts[block_idx]}, first sample {firsts[block_idx]}"))


def report_samples(found: list[Violation], check: str, bad: np.ndarray, offsets: np.ndarray, what: str) -> None:
    """Appends up to ``MAX_REPORTED`` block-level violations from a per-sample boolean failure mask."""
    if not bad.any():
        return
    samples = np.flatnonzero(bad)
    blocks = np.searchsorted(offsets, samples, side="right") - 1
    unique_blocks, first = np.unique(blocks, return_index=True)
    counts = np.bincount(blocks - blocks.min())[unique_blocks - blocks.min()]
    for block_idx, sample_idx, count in list(
        zip(unique_blocks.tolist(), samples[first].tolist(), counts.tolist(), strict=True)
    )[:MAX_REPORTED]:
        found.append(Violation(check, block_idx, f"{what}: {count}, first sample {sample_idx}"))


def summary(array: np.ndarray, limit: int = 8) -> str:
    """Formats an array as a compact list string, truncating past ``limit`` entries."""
    items = np.asarray(array).tolist()
    return str(items) if len(items) <= limit else f"{items[:limit]}... ({len(items)})"
