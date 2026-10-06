# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Public types of the fluxcode API: encoder parameters and result tuples."""

from collections.abc import Iterable, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from typing import Literal, NamedTuple

import numpy as np

MIN_TARGET_BITS: float = 6.0
"""Minimum allowed per-unit bit target per sample."""

MIN_EFFORT: int = 1
MAX_EFFORT: int = 9
"""The range of Params.effort."""

DEFAULT_NOISE_FLOOR_SIGMA: float = 0.25
"""The noise floor that noise_floor_sigma=None gives a unit without times."""

TimeUnit = Literal["s", "ms", "us", "ns"]
"""Resolution of integer timestamps: seconds, milliseconds, microseconds or nanoseconds."""


class EncodedUnit(NamedTuple):
    """The compressed bytes and per-block summary statistics of a single unit.

    Attributes:
        unit: Self-describing unit bytes (header and zstd frame).
        block_min: 1D float64 array of minimum finite values per block.
        block_max: 1D float64 array of maximum finite values per block.
        block_mean: 1D float64 array of finite sample means per block.

    The statistics are of the samples given, not of the decoded values: those can be up to half
    a step away (min included), so allow that much if using them as hard bounds.
    """

    unit: bytes
    block_min: np.ndarray
    block_max: np.ndarray
    block_mean: np.ndarray


class EncodedSeries(NamedTuple):
    """The sequence of compressed units and per-block summary statistics for a series.

    Attributes:
        units: List of compressed unit byte strings.
        block_mins: List of 1D float64 arrays of block minima.
        block_maxs: List of 1D float64 arrays of block maxima.
        block_means: List of 1D float64 arrays of block means.
    """

    units: list[bytes]
    block_mins: list[np.ndarray]
    block_maxs: list[np.ndarray]
    block_means: list[np.ndarray]


class DecodedUnit(NamedTuple):
    """The samples, timestamps and block sizes of a decoded unit.

    Attributes:
        values: 1D float64 array of the unit's samples.
        times: 1D datetime64 array of the samples' timestamps, in the unit they were
            encoded with (s, ms, us or ns); None for a unit encoded without times.
        block_sizes: 1D int64 array of the sample count of each block: block b holds
            values[offsets[b]:offsets[b + 1]], with offsets the cumulative sum from 0.
    """

    values: np.ndarray
    times: np.ndarray | None
    block_sizes: np.ndarray


class UpdatedUnit(NamedTuple):
    """The updated unit bytes and summary statistics of the blocks the update re-encoded.

    Attributes:
        unit: New compressed unit bytes.
        indices: 1D int64 array of the re-encoded blocks (replaced, emptied or appended), in
            increasing order. Every other block is unchanged.
        block_min: 1D float64 array of the minima of those blocks.
        block_max: 1D float64 array of the maxima of those blocks.
        block_mean: 1D float64 array of the means of those blocks.
    """

    unit: bytes
    indices: np.ndarray
    block_min: np.ndarray
    block_max: np.ndarray
    block_mean: np.ndarray


@dataclass(frozen=True)
class Params:
    """Encoder configuration parameters.

    Controls quantization step limits, difference predictor orders, noise floor gating and
    per-unit soft bit targets. How a series is divided into blocks and units is an argument
    of each encode function, not a parameter. Decoders do not require these parameters
    because units are self-describing.

    Attributes:
        min_quantize_bits: Hard lower bound on precision (coarsest step across
            the range, 1..16).
        max_quantize_bits: Hard upper bound on precision (finest step across
            the range, min..16).
        diff_orders: Non-empty subset of {0, 1, 2, 3} specifying predictor
            difference orders evaluated by the encoder.
        noise_floor_sigma: Noise floor multiplier f: steps on white-noise blocks coarsen up
            to f * sigma; 0 turns it off. None (the default) is DEFAULT_NOISE_FLOOR_SIGMA for
            a unit without times and off for a unit with times, which are often irregular (a
            historian's swinging-door archive, events): their samples don't oversample the
            signal, so they look like white noise without being noise.
        target_bits_per_sample: Soft per-unit cap on compressed bits per sample
            (must be >= 6.0 if set, or None to disable).
        decimal_detection: Whether to test for exact decimal grids (10^p) before
            falling back to power-of-two grids.
        effort: Encoder effort, MIN_EFFORT (fastest) to MAX_EFFORT (smallest). It changes how
            the unit is compressed (the residual layout, zstd block boundaries and level),
            never the decoded values; ENCODER.md lists what each effort does.
    """

    min_quantize_bits: int = 6
    max_quantize_bits: int = 16
    # Any set or sequence of orders is accepted; __post_init__ stores it as a frozenset.
    diff_orders: AbstractSet[int] | Sequence[int] = frozenset({0, 1, 2, 3})
    noise_floor_sigma: float | None = None
    target_bits_per_sample: float | None = None
    decimal_detection: bool = True
    effort: int = 4

    def noise_factor(self, timed: bool) -> float:
        """The noise floor multiplier for a unit with or without times (0.0 when off).

        Args:
            timed: Whether the unit stores timestamps.

        Returns:
            noise_floor_sigma, or for None DEFAULT_NOISE_FLOOR_SIGMA without times and 0.0
            with them.
        """
        if self.noise_floor_sigma is None:
            return 0.0 if timed else DEFAULT_NOISE_FLOOR_SIGMA
        return float(self.noise_floor_sigma)

    def __post_init__(self) -> None:
        """Validates parameter types, domains, and structural constraints.

        Raises:
            ValueError: If any parameter falls outside its defined domain.
        """
        min_bits = self.min_quantize_bits
        max_bits = self.max_quantize_bits
        # Verify quantization bit bounds: 1 <= min <= max <= 16
        if not (is_int(min_bits) and is_int(max_bits) and 1 <= min_bits <= max_bits <= 16):
            raise ValueError(
                f"need 1 <= min_quantize_bits <= max_quantize_bits <= 16, got {min_bits!r}, {max_bits!r}"
            )
        orders = self.diff_orders
        if not isinstance(orders, Iterable) or isinstance(orders, (str, bytes)):
            raise ValueError(f"diff_orders must be a set of orders, got {orders!r}")  # noqa: TRY004
        frozen_orders = frozenset(orders)
        # Ensure diff_orders is a non-empty subset of {0, 1, 2, 3}
        if not frozen_orders or not all(is_int(order) and 0 <= order <= 3 for order in frozen_orders):
            raise ValueError(f"diff_orders must be a non-empty subset of {{0, 1, 2, 3}}, got {set(frozen_orders)!r}")
        object.__setattr__(self, "diff_orders", frozenset(int(order) for order in frozen_orders))
        noise_factor = self.noise_floor_sigma
        # Validate noise floor multiplier
        if noise_factor is not None and not (is_real(noise_factor) and noise_factor >= 0 and np.isfinite(noise_factor)):
            raise ValueError(f"noise_floor_sigma must be None or a finite number >= 0, got {noise_factor!r}")
        target_bits = self.target_bits_per_sample
        # Validate soft target bit rate
        if target_bits is not None and not (is_real(target_bits) and MIN_TARGET_BITS <= target_bits and np.isfinite(target_bits)):
            raise ValueError(
                f"target_bits_per_sample must be None or a finite number >= {MIN_TARGET_BITS:g}, got {target_bits!r}"
            )
        if not (is_int(self.effort) and MIN_EFFORT <= self.effort <= MAX_EFFORT):
            raise ValueError(f"effort must be an integer from {MIN_EFFORT} to {MAX_EFFORT}, got {self.effort!r}")


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
    return isinstance(val, (int, float, np.integer, np.floating)) and not isinstance(val, (bool, np.bool_))


DEFAULT_PARAMS: Params = Params()
"""Default encoder parameters (immutable, shared as the default argument)."""
