# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Time axis kernels: per-block regularity and GCD analysis, and exact reconstruction.

Timestamps are int64 ticks in the unit's time unit (seconds to nanoseconds), naive
(no time zone). Within a block they must be non-decreasing. A block whose time deltas
are all equal is regular: time[i] = start + i * step, with no per-sample data. Any
other block is irregular: its step is the GCD of its deltas, and each sample stores
(time[i] - time[i-1]) / step as a uint64 delta quotient (0 for sample 0).
"""

import enum
from collections.abc import Callable
from typing import cast

import numpy as np
from numba import njit
from numba.cpython.unsafe.numbers import trailing_zeros as _trailing_zeros_intrinsic

from ._format import HEAD_IRREGULAR_TIME

_trailing_zeros = cast("Callable[[int | np.integer], int]", _trailing_zeros_intrinsic)
"""Numba intrinsic counting trailing zero bits, typed as its jitted call signature."""

INT64_MAX: int = 0x7FFFFFFFFFFFFFFF
"""Largest int64 tick."""

INT64_MIN: int = -0x8000000000000000
"""Smallest int64, which datetime64 reads as NaT: never a valid tick."""


class TimeStatus(enum.IntEnum):
    """Outcome of time axis analysis (encoding) or expansion (decoding)."""

    OK = 0
    DECREASING = 1
    BAD_STEP = 2
    BAD_FIRST_DELTA = 3
    OVERFLOW = 4


OK: int = int(TimeStatus.OK)
"""The times are valid."""

DECREASING: int = int(TimeStatus.DECREASING)
"""Encoding: a time is below the previous time in its block."""

BAD_STEP: int = int(TimeStatus.BAD_STEP)
"""Decoding: a time step is negative, or zero on an irregular block."""

BAD_FIRST_DELTA: int = int(TimeStatus.BAD_FIRST_DELTA)
"""Decoding: an irregular block's first delta quotient is not 0."""

OVERFLOW: int = int(TimeStatus.OVERFLOW)
"""Decoding: a time exceeds int64 maximum."""


@njit(inline="always")
def _gcd(first: np.uint64, second: np.uint64) -> np.uint64:
    """Greatest common divisor of two uint64 values (gcd(0, x) = x)."""
    while second != 0:
        first, second = second, first % second
    return first


UINT64_MAX: int = 0xFFFFFFFFFFFFFFFF
"""Largest uint64."""


@njit(inline="always")
def _odd_inverse(odd: np.uint64) -> np.uint64:
    """Multiplicative inverse of an odd number mod 2^64, by Newton's iteration.

    odd * odd = 1 mod 8, so odd is its own inverse to 3 bits; each step doubles the correct
    bits (3, 6, 12, 24, 48, 96).
    """
    inverse = odd
    for _ in range(5):
        inverse *= np.uint64(2) - odd * inverse
    return inverse


@njit(nogil=True, cache=True)
def encode_times(
    block_times: np.ndarray,
    out_block_flags: np.ndarray,
    out_time_starts: np.ndarray,
    out_time_steps: np.ndarray,
    out_time_deltas: np.ndarray,
) -> tuple[int, int]:
    """Analyzes each block's times into a start, a step and (if irregular) delta quotients.

    Sets HEAD_IRREGULAR_TIME in out_block_flags for irregular blocks; other bits are kept.
    Only checks order within blocks: callers check the order across blocks.

    Args:
        block_times: 2D int64 array of shape (num_blocks, block_len) of ticks, padded.
        out_block_flags: In-out 1D uint8 array of block flags.
        out_time_starts: Output 1D int64 array receiving block start times.
        out_time_steps: Output 1D int64 array receiving block time steps.
        out_time_deltas: Output 2D uint64 array of shape (num_blocks, block_len) receiving
            the delta quotients of irregular blocks (rows of regular blocks are not written).

    Returns:
        A tuple of (status, sample_idx): OK, or DECREASING and the flat index
        (block_idx * block_len + i) of the first time below its predecessor.
    """
    num_blocks, block_len = block_times.shape
    int64_max = np.uint64(INT64_MAX)
    for block_idx in range(num_blocks):
        times = block_times[block_idx]
        out_time_starts[block_idx] = times[0]
        # Read-only regularity and order check, branch-free so that it vectorizes: a decrease
        # is located afterwards. Deltas as uint64: a non-decreasing pair's difference fits.
        first_delta = np.uint64(times[1]) - np.uint64(times[0])
        differing_bits = np.uint64(0)
        decreased = False
        for sample_idx in range(1, block_len):
            differing_bits |= (np.uint64(times[sample_idx]) - np.uint64(times[sample_idx - 1])) ^ first_delta
            decreased |= times[sample_idx] < times[sample_idx - 1]
        if decreased:
            for sample_idx in range(1, block_len):
                if times[sample_idx] < times[sample_idx - 1]:
                    return DECREASING, block_idx * block_len + sample_idx
        if differing_bits == 0:
            # block_len - 1 >= 7 equal deltas span less than 2^64, so the step fits int64
            out_time_steps[block_idx] = np.int64(first_delta)
            out_block_flags[block_idx] &= ~HEAD_IRREGULAR_TIME
            continue
        deltas = out_time_deltas[block_idx]
        deltas[0] = 0
        for sample_idx in range(1, block_len):
            deltas[sample_idx] = np.uint64(times[sample_idx]) - np.uint64(times[sample_idx - 1])
        # GCD of the deltas. Most deltas are multiples of the GCD so far, confirmed without a
        # 64-bit division: 2^shift * odd divides a delta exactly when the delta's low shift
        # bits are 0 and (delta >> shift) * odd^-1 mod 2^64 <= UINT64_MAX // odd (Granlund and
        # Montgomery, 1994). That product is then the quotient, which the division below uses.
        step_gcd = np.uint64(0)
        low_mask = np.uint64(0)
        shift = np.uint64(0)
        inverse = np.uint64(0)
        quotient_limit = np.uint64(0)
        for sample_idx in range(1, block_len):
            delta = deltas[sample_idx]
            if step_gcd != 0 and (delta & low_mask) == 0 and (delta >> shift) * inverse <= quotient_limit:
                continue
            step_gcd = _gcd(step_gcd, delta)
            if step_gcd <= 1:
                # 0 while every delta so far is 0; 1 is final
                if step_gcd == 1:
                    break
                continue
            shift = np.uint64(_trailing_zeros(step_gcd))
            low_mask = (np.uint64(1) << shift) - np.uint64(1)
            odd_part = step_gcd >> shift
            inverse = _odd_inverse(odd_part)
            quotient_limit = np.uint64(UINT64_MAX) // odd_part
        if step_gcd > int64_max:
            # Only when one delta spans more than half the int64 range: store raw deltas
            step_gcd = np.uint64(1)
        elif step_gcd > 1:
            # Exact division: shift out the power of two, multiply by the odd part's inverse
            for sample_idx in range(1, block_len):
                deltas[sample_idx] = (deltas[sample_idx] >> shift) * inverse
        out_time_steps[block_idx] = np.int64(step_gcd)
        out_block_flags[block_idx] |= HEAD_IRREGULAR_TIME
    return OK, 0


@njit(nogil=True, cache=True)
def expand_times(
    block_flags: np.ndarray,
    time_starts: np.ndarray,
    time_steps: np.ndarray,
    time_deltas: np.ndarray,
    out_times: np.ndarray,
) -> tuple[int, int]:
    """Reconstructs every block's ticks from its start, step and delta quotients.

    Args:
        block_flags: 1D uint8 array of block flags (HEAD_IRREGULAR_TIME selects the deltas).
        time_starts: 1D int64 array of block start times.
        time_steps: 1D int64 array of block time steps.
        time_deltas: 2D uint64 array of shape (num_blocks, block_len) of delta quotients.
        out_times: Output 2D int64 array of shape (num_blocks, block_len) receiving ticks.

    Returns:
        A tuple of (status, block_idx): OK, or BAD_STEP, BAD_FIRST_DELTA or OVERFLOW and
        the first failing block.
    """
    num_blocks, block_len = out_times.shape
    int64_max = np.uint64(INT64_MAX)
    for block_idx in range(num_blocks):
        start = time_starts[block_idx]
        step = time_steps[block_idx]
        irregular = block_flags[block_idx] & HEAD_IRREGULAR_TIME
        if step < 0 or (irregular and step == 0):
            return BAD_STEP, block_idx
        step_unsigned = np.uint64(step)
        times = out_times[block_idx]
        times[0] = start
        if not irregular:
            # The last time start + (block_len - 1) * step must not exceed int64 maximum
            if step > 0 and np.uint64(block_len - 1) > (int64_max - np.uint64(start)) // step_unsigned:
                return OVERFLOW, block_idx
            for sample_idx in range(1, block_len):
                times[sample_idx] = np.int64(np.uint64(start) + np.uint64(sample_idx) * step_unsigned)
            continue
        deltas = time_deltas[block_idx]
        if deltas[0] != 0:
            return BAD_FIRST_DELTA, block_idx
        current = np.uint64(start)
        for sample_idx in range(1, block_len):
            # headroom = int64 maximum - current time, exact in uint64 arithmetic
            headroom = int64_max - current
            if deltas[sample_idx] > headroom // step_unsigned:
                return OVERFLOW, block_idx
            current += deltas[sample_idx] * step_unsigned
            times[sample_idx] = np.int64(current)
    return OK, 0
