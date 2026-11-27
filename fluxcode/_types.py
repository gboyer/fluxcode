# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Public types of the fluxcode API: encoder parameters and result dataclasses."""

from collections.abc import Iterable, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from typing import Literal

import numpy as np

MIN_TARGET_BITS: float = 6.0
"""Minimum allowed per-block-group bit target per sample."""

MIN_EFFORT: int = 1
MAX_EFFORT: int = 9
"""The range of Params.effort."""

DEFAULT_NOISE_FLOOR_SIGMA: float = 0.25
"""The default noise_floor_sigma."""

MAX_TIME_ERROR: float = 0.5
"""Largest Params.time_error: at 0.5 a block's ticks round to its own interval."""

TimeUnit = Literal["s", "ms", "us", "ns"]
"""Resolution of integer timestamps: seconds, milliseconds, microseconds or ns."""


@dataclass(frozen=True)
class EncodedGroup:
    """The compressed payload and per-block summary statistics of a single block group.

    Attributes:
        payload: Self-describing block group bytes (header and zstd frame).
        block_min: 1D float64 array of minimum finite values per block.
        block_max: 1D float64 array of maximum finite values per block.
        block_mean: 1D float64 array of finite sample means per block.
        block_start_times: 1D datetime64 array of the first timestamp of each block
            (NaT for empty blocks), in the unit the times were encoded with; None for
            a block group encoded without times.

    The statistics are of the samples given, not of the decoded values: those can be up
    to half a step away (min included), so allow that much if using them as hard bounds.
    """

    payload: bytes
    block_min: np.ndarray
    block_max: np.ndarray
    block_mean: np.ndarray
    block_start_times: np.ndarray | None = None


@dataclass(frozen=True)
class EncodedSeries:
    """A sequence of compressed block groups for a series, with flattening views.

    The flattening properties (``payloads``, ``block_min``, ``block_max``,
    ``block_mean``, ``block_start_times``) allocate and concatenate on each access;
    bind the result to a local variable if you index into it repeatedly.

    Attributes:
        groups: List of ``EncodedGroup`` results, one per block group.
    """

    groups: list[EncodedGroup]

    @property
    def payloads(self) -> list[bytes]:
        """Compressed block group byte payloads, one per group."""
        return [group.payload for group in self.groups]

    @property
    def block_min(self) -> np.ndarray:
        """Flattened 1D float64 array of per-block minima across all groups."""
        if not self.groups:
            return np.empty(0, np.float64)
        return np.concatenate([group.block_min for group in self.groups])

    @property
    def block_max(self) -> np.ndarray:
        """Flattened 1D float64 array of per-block maxima across all groups."""
        if not self.groups:
            return np.empty(0, np.float64)
        return np.concatenate([group.block_max for group in self.groups])

    @property
    def block_mean(self) -> np.ndarray:
        """Flattened 1D float64 array of per-block means across all groups."""
        if not self.groups:
            return np.empty(0, np.float64)
        return np.concatenate([group.block_mean for group in self.groups])

    @property
    def block_start_times(self) -> np.ndarray | None:
        """Flattened 1D datetime64 array of block start times, or None if untimed."""
        if not self.groups:
            return None
        starts = [group.block_start_times for group in self.groups]
        if all(s is None for s in starts):
            return None
        if any(s is None for s in starts):
            raise ValueError(
                "EncodedSeries mixes timed and untimed groups; cannot flatten "
                "block_start_times"
            )
        return np.concatenate([s for s in starts if s is not None])


@dataclass(frozen=True)
class DecodedGroup:
    """The samples, timestamps and block sizes of a decoded block group.

    Attributes:
        values: 1D float64 array of the block group's samples.
        times: 1D datetime64 array of the samples' timestamps, in the unit they were
            encoded with (s, ms, us or ns); None for a block group encoded without
            times.
        block_sizes: 1D int64 array of the sample count of each block: block b holds
            values[offsets[b]:offsets[b + 1]], with offsets the cumulative sum from 0.
    """

    values: np.ndarray
    times: np.ndarray | None
    block_sizes: np.ndarray


@dataclass(frozen=True)
class UpdatedGroup:
    """The updated block group payload and summary statistics of re-encoded blocks.

    Attributes:
        payload: New compressed block group bytes.
        indices: 1D int64 array of the re-encoded blocks (replaced, emptied or
            appended), in increasing order. Every other block is unchanged.
        block_min: 1D float64 array of the minima of those blocks.
        block_max: 1D float64 array of the maxima of those blocks.
        block_mean: 1D float64 array of the means of those blocks.
        block_start_times: 1D datetime64 array of the first timestamp of each
            re-encoded block (NaT for empty blocks), in the group's time unit; None
            for a block group without a time axis.
    """

    payload: bytes
    indices: np.ndarray
    block_min: np.ndarray
    block_max: np.ndarray
    block_mean: np.ndarray
    block_start_times: np.ndarray | None = None


GRIDS: tuple[str, ...] = ("binary", "decimal", "rational")
"""Params.grids levels, each adding grids to the one before (ENCODER.md §1)."""


@dataclass(frozen=True)
class Params:
    """Encoder configuration parameters.

    Controls quantization step limits, difference predictor orders, noise floor gating
    and per-block-group soft bit targets. How a series is divided into blocks and block
    groups is an argument of each encode function, not a parameter. Decoders do not
    require these parameters because block groups are self-describing.

    Attributes:
        min_quantize_bits: Hard lower bound on precision (coarsest step across
            the range, 1..16).
        max_quantize_bits: Hard upper bound on precision (finest step across
            the range, min..16).
        diff_orders: Non-empty subset of {0, 1, 2, 3} specifying predictor
            difference orders evaluated by the encoder.
        noise_floor_sigma: Noise floor multiplier f: steps on white-noise blocks coarsen
            up to f * sigma; 0 turns it off. A block with irregular times gets it only
            if at least 90% of its intervals are one scan (a jittered clock, or one with
            gaps), with sigma from consecutive scans: none for a sparse archive (a
            historian's swinging-door archive looks like white noise without being
            noise) or events. Without times the samples are taken as evenly spaced.
        target_bits_per_sample: Soft per-block-group cap on compressed bits per sample
            (must be >= 6.0 if set, or None to disable).
        grids: Which grids a block may take, each level adding to the one before:
            "binary" (power-of-two steps 2^e only), "decimal" (exact decimal steps 10^p
            too), "rational" (the default: scaled grids too, a step N / D times a power
            of ten or two with N and D up to 65,535: 0.05, thirds, ADC counts of
            10 V / 4095, a calibration's gain and offset, stored exactly).
        effort: Encoder effort, MIN_EFFORT (fastest) to MAX_EFFORT (smallest). It
            changes how the block group is compressed (the residual layout, zstd block
            boundaries and level), never the decoded values; ENCODER.md lists what each
            effort does.
        absolute_tolerance: An absolute error acceptable for every value (in the values'
            units), 0.0 (the default) for none. Each block's step may coarsen up to
            2 * absolute_tolerance, even past min_quantize_bits (so quiet blocks
            spanning less than 2 * absolute_tolerance are not forced to finer levels),
            while the noise floor may still coarsen further where noise dominates. With
            grids "rational" a step set by absolute_tolerance uses 94-100% of the
            allowance (a scaled grid); with "binary" or "decimal", 50-100% (a power of
            two).
        time_error: How far timestamps may move, as a share of an interval (0 to
            MAX_TIME_ERROR); 0 (the default) stores times exactly. Each block's times
            are rounded to one 1-2-5 step: the largest of at most 2 * time_error mean
            intervals (the block's span over its intervals, 2% more at most) if at least
            95% of its times move by no more than time_error times their own interval
            (the smaller of the two either side), else the coarsest that keeps to that
            from the step of a short interval down (the 3rd smallest of 15 evenly spaced
            ones; for events, whose intervals vary widely). No time moves more than half
            the step, so the other 5% (the unusually close times) are off by no more
            than the rest. The grid is at the clock's own phase: a jittered clock rounds
            back onto its own grid once time_error is about 5 times its jitter's
            standard deviation over its interval, and then stores as regular times. A
            block may instead take the lattice of a regular block before it, or, for a
            clock whose period no 1-2-5 step divides (1024 Hz, 60 Hz), a lattice of its
            own, on the same condition. Regular blocks are left as they are. In time
            blocks (encode_time_blocks) a time that would round past its block's end is
            set to the end itself. Times are rounded once, from the values given:
            update_time_blocks stores new times exactly, and replace_time_blocks
            re-encodes whole blocks.
    """

    min_quantize_bits: int = 6
    max_quantize_bits: int = 16
    # Any set or sequence of orders is accepted; __post_init__ stores a frozenset.
    diff_orders: AbstractSet[int] | Sequence[int] = frozenset({0, 1, 2, 3})
    noise_floor_sigma: float = DEFAULT_NOISE_FLOOR_SIGMA
    target_bits_per_sample: float | None = None
    grids: str = "rational"
    effort: int = 4
    time_error: float = 0.0
    absolute_tolerance: float = 0.0

    def __post_init__(self) -> None:
        """Validates parameter types, domains, and structural constraints.

        Raises:
            ValueError: If any parameter falls outside its defined domain.
        """
        min_bits = self.min_quantize_bits
        max_bits = self.max_quantize_bits
        if not (
            is_int(min_bits) and is_int(max_bits) and 1 <= min_bits <= max_bits <= 16
        ):
            raise ValueError(
                f"need 1 <= min_quantize_bits <= max_quantize_bits <= 16, "
                f"got {min_bits!r}, {max_bits!r}"
            )
        orders = self.diff_orders
        if not isinstance(orders, Iterable) or isinstance(orders, (str, bytes)):
            raise ValueError(f"diff_orders must be a set of orders, got {orders!r}")
        frozen_orders = frozenset(orders)
        if not frozen_orders or not all(
            is_int(order) and 0 <= order <= 3 for order in frozen_orders
        ):
            raise ValueError(
                f"diff_orders must be a non-empty subset of {{0, 1, 2, 3}}, "
                f"got {set(frozen_orders)!r}"
            )
        object.__setattr__(
            self, "diff_orders", frozenset(int(order) for order in frozen_orders)
        )
        noise_factor = self.noise_floor_sigma
        if not (
            is_real(noise_factor) and noise_factor >= 0 and np.isfinite(noise_factor)
        ):
            raise ValueError(
                f"noise_floor_sigma must be a finite number >= 0, got {noise_factor!r}"
            )
        target_bits = self.target_bits_per_sample
        if target_bits is not None and not (
            is_real(target_bits)
            and target_bits >= MIN_TARGET_BITS
            and np.isfinite(target_bits)
        ):
            raise ValueError(
                f"target_bits_per_sample must be None or a finite number "
                f">= {MIN_TARGET_BITS:g}, got {target_bits!r}"
            )
        abs_tol = self.absolute_tolerance
        if not (is_real(abs_tol) and abs_tol >= 0 and np.isfinite(abs_tol)):
            raise ValueError(
                f"absolute_tolerance must be a finite number >= 0, got {abs_tol!r}"
            )
        if self.grids not in GRIDS:
            raise ValueError(
                f"grids must be one of {', '.join(map(repr, GRIDS))}, "
                f"got {self.grids!r}"
            )
        if not (is_int(self.effort) and MIN_EFFORT <= self.effort <= MAX_EFFORT):
            raise ValueError(
                f"effort must be an integer from {MIN_EFFORT} to {MAX_EFFORT}, "
                f"got {self.effort!r}"
            )
        time_error = self.time_error
        if not (is_real(time_error) and 0 <= time_error <= MAX_TIME_ERROR):
            raise ValueError(
                f"time_error must be a number from 0 to {MAX_TIME_ERROR:g}, "
                f"got {time_error!r}"
            )


def is_int(val: object) -> bool:
    """Checks whether a value is an integer type (excluding booleans).

    Args:
        val: Value to inspect.

    Returns:
        True if val is an integer and not a bool, False otherwise.
    """
    return isinstance(val, (int, np.integer)) and not isinstance(val, (bool, np.bool_))


def is_real(val: object) -> bool:
    """Checks whether a value is a real numeric type (excluding booleans).

    Args:
        val: Value to inspect.

    Returns:
        True if val is an int or float and not a bool, False otherwise.
    """
    return isinstance(val, (int, float, np.integer, np.floating)) and not isinstance(
        val, (bool, np.bool_)
    )


DEFAULT_PARAMS: Params = Params()
"""Default encoder parameters (immutable, shared as the default argument)."""
