# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Lossy timestamp rounding kernels (``Params.time_error > 0``).

Rounds irregular blocks' timestamps onto the coarsest regular grid or clock lattice that
preserves every block's time-error guarantee:

- No tick moves by more than ``time_error * TIME_ERROR_SLACK * mean_interval``.
- At most ``1 / TIME_ERROR_OUTLIERS`` (5%) of ticks in a block move by more than
  ``time_error`` times their own local interval (the smaller of the two adjacent intervals).

For each non-regular block of at least 3 ticks, ``round_blocks`` tries in order:
1. Continuing the previous block's regular lattice if both share the same clock period.
2. A 1-2-5 x 10^k grid with a phase chosen from ``PHASE_STEPS`` candidates, starting at the
   coarsest step for the block's mean interval and stepping finer (using interval pruning
   to skip steps that cannot satisfy the error bound).
3. A custom clock lattice ``origin + k * period`` for clocks whose period no 1-2-5 grid
   divides (such as 1024 Hz or 60 Hz).

After independent per-block rounding, ``clamp_block_boundaries`` restores cross-block
monotonicity wherever two adjacent blocks rounded past each other.
"""

import numpy as np
from numba import njit

from ._time import (
    CADENCE_SAMPLES,
    CADENCE_SHARE,
    DECREASING,
    INT64_MAX,
    INT64_MIN,
    OK,
    detect_scan_interval,
    sampled_intervals,
)

TIME_ERROR_SLACK: float = 1.02
"""2% slack on ``time_error * mean_interval`` so slight mean-interval drift across blocks
does not flip the chosen 1-2-5 quantum back and forth."""

SHORT_INTERVAL_RANK: int = 2
"""Rank (0-based among positive ``CADENCE_SAMPLES`` intervals, ~14th percentile) jumped to
when the mean-interval quantum fails, skipping intermediate quanta that rarely pass on
event-driven or deadband-logged timestamps."""

PHASE_STEPS: int = 10
"""Number of candidate grid phases (multiples of ``quantum / PHASE_STEPS``) evaluated per block."""

PHASE_SAMPLES: int = 64
"""Number of evenly spaced ticks sampled from a block to score candidate grid phases."""

PHASE_QUANTUM_LIMIT: int = 1 << 56
"""Maximum quantum for phase selection; ensures ``PHASE_SAMPLES`` distances fit in int64
without overflow (coarser quanta stay on the epoch grid, phase 0)."""

CHECK_CHUNK: int = 64
"""Chunk size for branch-free vectorized loops that exit early when a candidate fails."""

TIME_ERROR_OUTLIERS: int = 20
"""Denominator of the allowed per-tick outlier fraction (at most 1 in 20, i.e. 5%)."""

FLOAT_ROUNDING_SPAN: int = 1 << 48
"""Maximum tick distance from origin (and maximum step) for fast float64 lattice rounding.
Below 2^48, ``offset / step`` errs by < 1/16 tick so adding 1/8 tick before floor matches
exact integer tie-up rounding."""

GRID_TEST_CHUNK: int = 64
"""Chunk size for interval-based grid pruning in ``grid_cannot_meet_error``."""

GRID_TEST_GIVE_UP: int = 256
"""Number of intervals after which ``grid_cannot_meet_error`` stops early if forced-outlier
pace is too low to rule the quantum out."""

FLOAT_EXACT_INTERVAL: float = 2.0**52
"""Upper bound on intervals representable exactly in IEEE-754 float64."""


@njit(inline="always")
def choose_125_step(interval: float, time_error: float) -> int:
    """Returns the largest ``{1, 2, 5} * 10^k`` step not exceeding ``2 * time_error * TIME_ERROR_SLACK * interval``.

    Args:
        interval: Reference interval in ticks.
        time_error: Positive relative time error tolerance.

    Returns:
        Largest fitting 1-2-5 step, or 1 if no step above 1 fits.
    """
    # Rounding to step q moves each tick by at most q / 2, so q <= 2 * e * slack * interval
    # respects the max-move cap while dividing standard decimal clock periods.
    limit = 2.0 * time_error * TIME_ERROR_SLACK * interval
    best = 1
    decade = 1
    while decade <= limit:
        for mantissa in (1, 2, 5):
            step = mantissa * decade
            if step <= limit:
                best = step
        if decade > INT64_MAX // 50:
            break
        decade *= 10
    return best


@njit(inline="always")
def estimate_clock_period(ticks: np.ndarray, scratch_pivots: np.ndarray, tolerance: float) -> float:
    """Estimates a clock block's true period as a trimmed mean of its one-scan intervals.

    Args:
        ticks: 1D int64 array of block timestamps (at least 2 samples).
        scratch_pivots: Preallocated 1D int64 scratch buffer of length ``>= CADENCE_SAMPLES``.
        tolerance: Maximum deviation from the median scan interval for an interval to be included.

    Returns:
        Estimated clock period in ticks, or 0.0 if the block does not exhibit a regular cadence.
    """
    # Require at least CADENCE_SHARE of intervals to be one scan so event streams are excluded.
    # Accumulate integer offsets from the median scan in fixed order to trim isolated glitches
    # without biasing symmetric clock jitter.
    scan, _, _, num_one_scan = detect_scan_interval(ticks, scratch_pivots)
    if scan == 0 or num_one_scan < CADENCE_SHARE * (ticks.shape[0] - 1):
        return 0.0
    num_intervals = ticks.shape[0] - 1
    ticks_next, ticks_prev = ticks[1:], ticks[:num_intervals]
    total, count = 0, 0
    for interval_idx in range(num_intervals):
        offset = ticks_next[interval_idx] - ticks_prev[interval_idx] - scan
        near = abs(offset) <= tolerance
        total += offset * near
        count += near
    return scan + total / count if count else 0.0


@njit(inline="always")
def _round_tick(tick: np.int64, quantum: np.int64, phase: np.int64) -> np.int64:
    """Rounds a single tick to the nearest ``phase + k * quantum`` (ties up), preserving int64 bounds.

    Args:
        tick: Input int64 timestamp tick.
        quantum: Grid step (``> 1``).
        phase: Grid phase offset (``0 <= phase < quantum``).

    Returns:
        Rounded int64 tick, or ``tick`` unchanged if rounding would overflow int64.
    """
    int64_min, int64_max = np.int64(INT64_MIN), np.int64(INT64_MAX)
    if tick < int64_min + quantum:
        return tick
    shifted = tick - phase
    # Numba uses Python's floored modulo, so 0 <= remainder < quantum.
    remainder = shifted % quantum
    down = tick - remainder
    if remainder < quantum - remainder:
        return down
    if down > int64_max - quantum:
        return tick
    return down + quantum


@njit(inline="always")
def sample_phase_ticks(times: np.ndarray, out: np.ndarray) -> np.ndarray:
    """Gathers up to ``PHASE_SAMPLES`` evenly spaced ticks from ``times`` for phase scoring.

    Args:
        times: 1D int64 array of block timestamps (at least 1 sample).
        out: Preallocated 1D int64 buffer of length ``>= PHASE_SAMPLES``.

    Returns:
        Leading slice of ``out`` containing the sampled ticks.
    """
    block_len = times.shape[0]
    num_sampled = min(PHASE_SAMPLES, block_len)
    for sampled_idx in range(num_sampled):
        out[sampled_idx] = times[sampled_idx * (block_len - 1) // max(num_sampled - 1, 1)]
    return out[:num_sampled]


@njit(inline="always")
def choose_grid_phase(
    sampled_ticks: np.ndarray, quantum: np.int64, previous: np.int64, scratch: np.ndarray
) -> np.int64:
    """Selects the grid phase ``0 <= phase < quantum`` minimizing mean rounding distance.

    Evaluates ``PHASE_STEPS`` evenly spaced candidate phases (multiples of ``quantum // PHASE_STEPS``)
    with hysteresis toward ``previous``'s phase (or phase 0 when there is no previous block) so
    noisy clocks do not oscillate between adjacent phases across consecutive blocks.

    Args:
        sampled_ticks: 1D int64 array of representative block ticks (from ``sample_phase_ticks``).
        quantum: Grid step (``> 1``).
        previous: Previous block's last rounded tick, or ``INT64_MIN`` if none.
        scratch: Preallocated 1D int64 buffer of length ``>= len(sampled_ticks)``.

    Returns:
        Chosen phase offset in ``[0, quantum)`` (always 0 if ``quantum >= PHASE_QUANTUM_LIMIT``).
    """
    # Compute each sampled tick's residue mod quantum once (via vectorized float64 when within
    # FLOAT_ROUNDING_SPAN), then score every candidate phase by circular L1 distance. Retain
    # the previous block's phase unless another beats it by more than half a phase step per tick.
    if quantum >= PHASE_QUANTUM_LIMIT:
        return np.int64(0)
    step = max(quantum // PHASE_STEPS, np.int64(1))
    num_candidates = quantum // step
    num_sampled = sampled_ticks.shape[0]
    sampled = scratch[:num_sampled]
    first = sampled_ticks[0]
    first_residue = first % quantum
    if _float_offset(sampled_ticks[num_sampled - 1], first) < FLOAT_ROUNDING_SPAN and quantum < FLOAT_ROUNDING_SPAN:
        quantum_float = np.float64(quantum)
        inverse = 1.0 / quantum_float
        for sampled_idx in range(num_sampled):
            offset = np.float64(sampled_ticks[sampled_idx] - first + first_residue)
            residue = offset - np.floor(offset * inverse) * quantum_float
            # Guard against 1-ULP boundary error near an exact multiple of quantum.
            residue += quantum_float if residue < 0.0 else 0.0
            residue -= quantum_float if residue >= quantum_float else 0.0
            sampled[sampled_idx] = np.int64(residue)
    else:
        for sampled_idx in range(num_sampled):
            sampled[sampled_idx] = sampled_ticks[sampled_idx] % quantum
    # Carry forward the previous block's phase if it aligns with a candidate (epoch phase 0 if none).
    previous_phase = np.int64(0) if previous == INT64_MIN else np.int64(-1)
    if previous != INT64_MIN and (previous % quantum) % step == 0:
        previous_phase = previous % quantum
    best_cost, best_phase, previous_cost = np.int64(INT64_MAX), np.int64(0), np.int64(INT64_MAX)
    for candidate_idx in range(num_candidates):
        phase = candidate_idx * step
        cost = np.int64(0)
        for sampled_idx in range(num_sampled):
            # Circular distance mod quantum, branch-free so LLVM vectorizes it.
            distance = abs(sampled[sampled_idx] - phase)
            cost += min(distance, quantum - distance)
        if cost < best_cost:
            best_cost, best_phase = cost, phase
        if phase == previous_phase:
            previous_cost = cost
    if previous_cost <= best_cost + num_sampled * step // 2:
        return previous_phase
    return best_phase


@njit(inline="always")
def snap_to_grid_exact(times: np.ndarray, quantum: np.int64, phase: np.int64, out: np.ndarray) -> None:
    """Snaps every tick in ``times`` to the nearest ``phase + k * quantum`` using exact integer math.

    Used as the fallback when ``snap_and_check_lattice`` cannot round in float64.

    Args:
        times: 1D int64 array of block timestamps (no NaT).
        quantum: Grid step (``> 1``).
        phase: Grid phase offset (``0 <= phase < quantum``).
        out: Preallocated 1D int64 output array.
    """
    for sample_idx in range(times.shape[0]):
        out[sample_idx] = _round_tick(times[sample_idx], quantum, phase)


@njit(inline="always")
def snap_to_lattice_exact(
    times: np.ndarray, origin: np.int64, step: np.int64, min_multiple: np.int64, out: np.ndarray
) -> bool:
    """Snaps every tick in ``times`` to ``origin + k * step`` (``k >= min_multiple``, ties up) in int64.

    Args:
        times: 1D int64 array of block timestamps.
        origin: Anchor tick on the lattice.
        step: Positive lattice step.
        min_multiple: Lower bound on ``k`` (``INT64_MIN`` for unbounded).
        out: Preallocated 1D int64 output array.

    Returns:
        True if ticks were snapped, or False (leaving ``out`` untouched) when ``step`` is
        outside ``(0, 2^61)``.
    """
    if not 0 < step < (1 << 61):
        return False
    for sample_idx in range(times.shape[0]):
        offset = times[sample_idx] - origin
        multiple = offset // step
        if 2 * (offset - multiple * step) >= step:
            multiple += 1
        out[sample_idx] = origin + max(multiple, min_multiple) * step
    return True


@njit(inline="always")
def estimate_lattice_origin(times: np.ndarray, period: np.int64, quantum: np.int64) -> np.int64:
    """Estimates a clock lattice origin near ``times[0]``, rounded to a clean sub-quantum step.

    Args:
        times: 1D int64 array of block timestamps (at least 1 sample).
        period: Candidate clock lattice period in ticks.
        quantum: Reference 1-2-5 quantum (used to round the origin to ``quantum // PHASE_STEPS``).

    Returns:
        Estimated int64 lattice origin.
    """
    # Average phase residuals of up to PHASE_SAMPLES ticks within a quarter period of times[0]
    # so jitter on times[0] cancels out while isolated glitches are ignored.
    first = times[0]
    period_float = np.float64(period)
    block_len = times.shape[0]
    num_sampled = min(PHASE_SAMPLES, block_len)
    total, count = 0.0, 0
    for sampled_idx in range(num_sampled):
        sample_idx = sampled_idx * (block_len - 1) // max(num_sampled - 1, 1)
        offset = np.float64(times[sample_idx] - first)
        residual = offset - np.round(offset / period_float) * period_float
        near = abs(residual) < 0.25 * period_float
        total += residual * near
        count += int(near)
    origin = first + np.int64(np.round(total / max(count, 1)))
    step = max(quantum // PHASE_STEPS, np.int64(1))
    return _round_tick(origin, step, np.int64(0)) if step > 1 else origin


@njit(inline="always")
def fit_clock_lattice(
    times: np.ndarray, interval: float, quantum: np.int64, max_move: float
) -> tuple[np.int64, np.int64]:
    """Fits a custom ``(origin, period)`` lattice for a clock whose period no 1-2-5 grid divides.

    Args:
        times: 1D int64 array of block timestamps (at least 2 samples).
        interval: Estimated clock period in ticks (from ``estimate_clock_period``).
        quantum: Coarsest 1-2-5 quantum for the block (``>= 1``).
        max_move: Maximum allowed tick displacement.

    Returns:
        Tuple ``(origin, period)`` defining the lattice ``origin + k * period``.
    """
    # Round interval to the coarsest 1-2-5 step <= quantum whose cumulative drift over the block
    # stays within max_move / 4 (leaving the remaining budget for jitter), then fit the origin.
    num_intervals = times.shape[0] - 1
    step = quantum
    period = np.int64(max(np.round(interval / step), 1.0) * step)
    while step > 1 and abs(interval - period) * num_intervals > 0.25 * max_move:
        step = _finer_step(step)
        period = np.int64(max(np.round(interval / step), 1.0) * step)
    return estimate_lattice_origin(times, period, quantum), period


@njit(inline="always")
def one_scan_intervals_equal(times: np.ndarray, interval: float) -> bool:
    """Checks whether every one-scan interval in ``times`` (within ``(0.5, 1.5) * interval``) is identical.

    Args:
        times: 1D int64 array of block timestamps (at least 2 samples).
        interval: Estimated single-scan interval in ticks.

    Returns:
        True if all intervals strictly between ``0.5 * interval`` and ``1.5 * interval`` are equal.
    """
    # Find the first one-scan interval, then XOR-accumulate differences over all one-scan
    # intervals without branching so LLVM vectorizes the scan.
    lower, upper = 0.5 * interval, 1.5 * interval
    num_intervals = times.shape[0] - 1
    ticks_next, ticks_prev = times[1:], times[:num_intervals]
    scan = np.int64(-1)
    for interval_idx in range(num_intervals):
        delta = ticks_next[interval_idx] - ticks_prev[interval_idx]
        if lower < delta < upper:
            scan = delta
            break
    differing_bits = np.int64(0)
    for interval_idx in range(num_intervals):
        delta = ticks_next[interval_idx] - ticks_prev[interval_idx]
        is_scan = (delta > lower) & (delta < upper)
        differing_bits |= (delta ^ scan) * is_scan
    return differing_bits == 0


@njit(inline="always")
def _finer_step(step: np.int64) -> np.int64:
    """Returns the next smaller ``{1, 2, 5} * 10^k`` step below ``step`` (for ``step > 1``)."""
    decade = np.int64(1)
    while step >= 10 * decade:
        decade *= 10
    mantissa = step // decade
    if mantissa == 1:
        return decade // 2 if decade >= 10 else np.int64(1)
    return decade if mantissa == 2 else 2 * decade


@njit(inline="always")
def uniform_interval(times: np.ndarray) -> int:
    """Returns the common positive interval of ``times`` if all consecutive intervals are equal, else 0.

    Args:
        times: 1D int64 array of block timestamps (at least 2 samples).

    Returns:
        The uniform positive delta, or 0 if deltas differ or are non-positive.
    """
    # Scan in branch-free chunks over zero-based slices so LLVM vectorizes the loop while still
    # bailing out quickly on irregular blocks.
    first = times[1] - times[0]
    if first <= 0:
        return 0
    num_intervals = times.shape[0] - 1
    ticks_next, ticks_prev = times[1:], times[:num_intervals]
    differing_bits = 0
    for chunk_start in range(0, num_intervals, CHECK_CHUNK):
        chunk_end = min(chunk_start + CHECK_CHUNK, num_intervals)
        chunk_next, chunk_prev = ticks_next[chunk_start:chunk_end], ticks_prev[chunk_start:chunk_end]
        for chunk_idx in range(chunk_end - chunk_start):
            differing_bits |= (chunk_next[chunk_idx] - chunk_prev[chunk_idx]) ^ first
        if differing_bits != 0:
            return 0
    return first


@njit(inline="always")
def compute_tick_allowances(times: np.ndarray, time_error: float, out_allowed: np.ndarray) -> None:
    """Computes each tick's local move budget: ``time_error`` times its smaller adjacent interval.

    Args:
        times: 1D int64 array of block timestamps (at least 2 samples).
        time_error: Positive relative time error tolerance.
        out_allowed: Preallocated 1D float64 array of length ``len(times)``.
    """
    block_len = times.shape[0]
    out_allowed[0] = time_error * np.float64(times[1] - times[0])
    out_allowed[block_len - 1] = time_error * np.float64(times[block_len - 1] - times[block_len - 2])
    # Zero-based offset slices let LLVM vectorize the interior loop.
    num_inner = block_len - 2
    ticks_prev, ticks_mid, ticks_next = times[:num_inner], times[1 : num_inner + 1], times[2:]
    inner = out_allowed[1 : num_inner + 1]
    for inner_idx in range(num_inner):
        own = min(ticks_mid[inner_idx] - ticks_prev[inner_idx], ticks_next[inner_idx] - ticks_mid[inner_idx])
        inner[inner_idx] = time_error * np.float64(own)


@njit(inline="always")
def max_move_within_error(times: np.ndarray, out: np.ndarray, allowed: np.ndarray, max_move: float) -> int:
    """Checks rounded ticks against precomputed per-tick allowances and the block-wide ``max_move`` cap.

    Args:
        times: 1D int64 array of original block timestamps.
        out: 1D int64 array of candidate rounded timestamps.
        allowed: 1D float64 array of per-tick allowances (from ``compute_tick_allowances``).
        max_move: Maximum allowed displacement for any tick in the block.

    Returns:
        Largest tick displacement if the candidate passes, else ``-1``.
    """
    # Check in branch-free chunks over chunk-local slices so LLVM vectorizes each chunk while
    # aborting early on the first failing chunk.
    block_len = times.shape[0]
    max_outliers = block_len // TIME_ERROR_OUTLIERS
    outliers, largest, too_far = 0, 0, 0
    for chunk_start in range(0, block_len, CHECK_CHUNK):
        chunk_end = min(chunk_start + CHECK_CHUNK, block_len)
        given, rounded = times[chunk_start:chunk_end], out[chunk_start:chunk_end]
        chunk_allowed = allowed[chunk_start:chunk_end]
        for chunk_idx in range(chunk_end - chunk_start):
            move = abs(rounded[chunk_idx] - given[chunk_idx])
            move_float = np.float64(move)
            too_far += int(move_float > max_move)
            outliers += int(move_float > chunk_allowed[chunk_idx])
            largest = max(largest, move)
        if too_far or outliers > max_outliers:
            return -1
    return largest


@njit(inline="always")
def check_within_time_error(times: np.ndarray, out: np.ndarray, time_error: float, max_move: float) -> int:
    """Checks whether candidate rounded timestamps satisfy the block's time-error guarantee.

    Args:
        times: 1D int64 array of original block timestamps (at least 2 samples).
        out: 1D int64 array of candidate rounded timestamps.
        time_error: Positive relative time error tolerance.
        max_move: Maximum allowed displacement for any tick in the block.

    Returns:
        Largest tick displacement if valid, else ``-1``.
    """
    allowed = np.empty(times.shape[0], np.float64)
    compute_tick_allowances(times, time_error, allowed)
    return max_move_within_error(times, out, allowed, max_move)


@njit(inline="always")
def _float_offset(bound: np.int64, base: np.int64) -> float:
    """Computes ``bound - base`` as float64 without signed overflow, clamped to ``[-2^60, 2^60]``."""
    limit = np.float64(1 << 60)
    if bound >= base:
        return min(np.float64(np.uint64(bound) - np.uint64(base)), limit)
    return -min(np.float64(np.uint64(base) - np.uint64(bound)), limit)


@njit(inline="always")
def snap_and_check_lattice(
    times: np.ndarray,
    origin: np.int64,
    step: np.int64,
    min_multiple: np.int64,
    low: np.int64,
    high: np.int64,
    allowed: np.ndarray,
    max_move: float,
    out: np.ndarray,
) -> int:
    """Snaps ``times`` to ``origin + k * step`` (clamped to ``[low, high]``) and checks the error bound in one pass.

    Args:
        times: 1D int64 array of original block timestamps (at least 1 sample).
        origin: Anchor tick on the lattice.
        step: Positive lattice step.
        min_multiple: Minimum allowed multiple ``k`` (``INT64_MIN`` for unbounded).
        low: Minimum allowed rounded tick.
        high: Maximum allowed rounded tick.
        allowed: 1D float64 array of per-tick allowances (from ``compute_tick_allowances``).
        max_move: Maximum allowed displacement for any tick.
        out: Preallocated 1D int64 array receiving snapped ticks.

    Returns:
        Largest displacement (>= 0) if the lattice passes (``out`` holds the rounded ticks),
        ``-1`` if it violates the error bound (``out`` partially written), or ``-2`` if float64
        cannot guarantee exact tie-up rounding (``out`` untouched; caller must use exact int64).
    """
    # When all offsets from origin and the step stay below FLOAT_ROUNDING_SPAN (2^48), float64
    # rounding with +0.5 + 0.125/step matches exact integer ties-up rounding and fuses snapping,
    # clamping, and error checking into a single vectorized chunked pass.
    block_len = times.shape[0]
    first, last = times[0], times[block_len - 1]
    first_offset, last_offset = _float_offset(first, origin), _float_offset(last, origin)
    if not (
        np.int64(INT64_MIN) + step <= first
        and last <= np.int64(INT64_MAX) - step
        and step < FLOAT_ROUNDING_SPAN
        and -FLOAT_ROUNDING_SPAN < first_offset
        and last_offset < FLOAT_ROUNDING_SPAN
    ):
        return -2
    step_float = np.float64(step)
    inverse = 1.0 / step_float
    half_up = 0.5 + 0.125 * inverse
    min_offset = max(np.float64(min_multiple), -2.0 * FLOAT_ROUNDING_SPAN * inverse) * step_float
    low_offset, high_offset = _float_offset(low, origin), _float_offset(high, origin)
    floored = min_multiple != np.int64(INT64_MIN)
    # Skip per-tick clamping when [low, high] cannot intersect any rounded tick; LLVM hoists these flags.
    clamps = (
        low_offset > min(first_offset - step_float, min_offset)
        or high_offset < max(last_offset, min_offset) + step_float
    )
    max_outliers = block_len // TIME_ERROR_OUTLIERS
    outliers, largest, too_far = 0, 0, 0
    for chunk_start in range(0, block_len, CHECK_CHUNK):
        chunk_end = min(chunk_start + CHECK_CHUNK, block_len)
        given, chunk_allowed = times[chunk_start:chunk_end], allowed[chunk_start:chunk_end]
        rounded_ticks = out[chunk_start:chunk_end]
        for chunk_idx in range(chunk_end - chunk_start):
            offset = np.float64(given[chunk_idx] - origin)
            rounded = np.floor(offset * inverse + half_up) * step_float
            if floored:
                rounded = max(rounded, min_offset)
            if clamps:
                rounded = min(max(rounded, low_offset), high_offset)
            rounded_ticks[chunk_idx] = origin + np.int64(rounded)
            move = abs(offset - rounded)
            too_far += move > max_move
            outliers += move > chunk_allowed[chunk_idx]
            # Track max as int64 because float max inhibits SIMD vectorization due to NaN semantics.
            largest = max(largest, int(move))
        if too_far or outliers > max_outliers:
            return -1
    return largest


@njit(inline="always")
def _clamp(out: np.ndarray, low: np.int64, high: np.int64) -> None:
    """Clamps a non-decreasing array of rounded ticks in place to ``[low, high]``."""
    block_len = out.shape[0]
    sample_idx = 0
    while sample_idx < block_len and out[sample_idx] < low:
        out[sample_idx] = low
        sample_idx += 1
    sample_idx = block_len - 1
    while sample_idx >= 0 and out[sample_idx] > high:
        out[sample_idx] = high
        sample_idx -= 1


@njit(inline="always")
def _clamped_error(
    times: np.ndarray, out: np.ndarray, low: np.int64, high: np.int64, allowed: np.ndarray, max_move: float
) -> int:
    """Clamps ``out`` to ``[low, high]`` and returns ``max_move_within_error``."""
    _clamp(out, low, high)
    return max_move_within_error(times, out, allowed, max_move)


@njit(inline="always")
def _snap_and_check_lattice_any(
    times: np.ndarray,
    origin: np.int64,
    step: np.int64,
    min_multiple: np.int64,
    low: np.int64,
    high: np.int64,
    allowed: np.ndarray,
    max_move: float,
    out: np.ndarray,
) -> int:
    """Runs ``snap_and_check_lattice``, falling back to exact int64 snapping when float64 is out of range."""
    largest_move = snap_and_check_lattice(times, origin, step, min_multiple, low, high, allowed, max_move, out)
    if largest_move != -2:
        return largest_move
    if not snap_to_lattice_exact(times, origin, step, min_multiple, out):
        return -1
    return _clamped_error(times, out, low, high, allowed, max_move)


@njit(inline="always")
def grid_cannot_meet_error(times: np.ndarray, block_len: int, quantum: np.int64, time_error: float) -> bool:
    """Conservative phase-independent filter that rules out a grid ``quantum`` from intervals alone.

    Caller must pass only interior ticks far enough from block bounds and int64 limits that
    clamping cannot occur. Never produces a false positive (returns True only when every phase
    is guaranteed to exceed the 5% outlier budget).

    Args:
        times: 1D int64 array of unclamped block timestamps (at least 2 samples).
        block_len: Full block sample count ``n``.
        quantum: Candidate grid step (``> 1``).
        time_error: Positive relative time error tolerance.

    Returns:
        True if no phase of ``quantum`` can satisfy the error bound; False if one might.
    """
    # Any two adjacent ticks separated by interval I must round to grid points a multiple of
    # quantum apart, forcing at least one tick to move by >= d / 2 where d is I's distance to
    # the nearest multiple of quantum. If d > 2 * e * I, at least one endpoint is an outlier.
    # Since each tick borders at most 2 intervals, > 2 * (n // 20) forced intervals guarantee > n // 20 outliers.
    num_intervals = times.shape[0] - 1
    limit = 2 * (block_len // TIME_ERROR_OUTLIERS)
    quantum_float = np.float64(quantum)
    inverse = 1.0 / quantum_float
    # Slightly below 2 * time_error so float rounding never falsely counts a borderline interval.
    threshold = 2.0 * time_error * (1.0 - 1e-9)
    ticks_next, ticks_prev = times[1:], times[:num_intervals]
    forced = 0
    for chunk_start in range(0, num_intervals, GRID_TEST_CHUNK):
        chunk_end = min(chunk_start + GRID_TEST_CHUNK, num_intervals)
        chunk_next, chunk_prev = ticks_next[chunk_start:chunk_end], ticks_prev[chunk_start:chunk_end]
        for chunk_idx in range(chunk_end - chunk_start):
            interval = np.float64(chunk_next[chunk_idx] - chunk_prev[chunk_idx])
            distance = abs(interval - np.round(interval * inverse) * quantum_float)
            # Handle half-way ties where np.round may pick either multiple.
            distance = min(distance, quantum_float - distance)
            forced += int((distance > threshold * interval) & (interval < FLOAT_EXACT_INTERVAL))
        if forced > limit:
            return True
        if chunk_end >= GRID_TEST_GIVE_UP and forced * num_intervals < limit * chunk_end:
            return False
    return False


@njit(inline="always")
def _try_grid_rounding(
    times: np.ndarray,
    largest_quantum: np.int64,
    time_error: float,
    low: np.int64,
    high: np.int64,
    previous_last: np.int64,
    allowed: np.ndarray,
    max_move: float,
    sampled_ticks: np.ndarray,
    phase_scratch: np.ndarray,
    pivots: np.ndarray,
    out: np.ndarray,
) -> int:
    """Searches for the coarsest 1-2-5 grid step (starting from ``largest_quantum``) that meets the error bound.

    Args:
        times: 1D int64 array of block timestamps.
        largest_quantum: Coarsest candidate 1-2-5 step (from ``choose_125_step``).
        time_error: Positive relative time error tolerance.
        low: Minimum allowed rounded tick for this block.
        high: Maximum allowed rounded tick for this block.
        previous_last: Previous block's last rounded tick, or ``INT64_MIN`` if none.
        allowed: 1D float64 array of per-tick allowances.
        max_move: Maximum allowed tick displacement.
        sampled_ticks: Representative ticks for phase scoring (from ``sample_phase_ticks``).
        phase_scratch: Scratch buffer for ``choose_grid_phase``.
        pivots: Scratch buffer for ``sampled_intervals``.
        out: Output 1D int64 array receiving the rounded ticks on success.

    Returns:
        Largest tick displacement (>= 0) if a grid step succeeded, else ``-1``.
    """
    # Precompute the interior slice [safe_first, safe_last] immune to boundary/int64 clamping so
    # grid_cannot_meet_error can cheaply prune steps once the coarsest quantum fails. After that
    # first failure, jump down to the quantum of a short sampled interval (SHORT_INTERVAL_RANK).
    block_len = times.shape[0]
    safe_first, safe_last = 0, -1
    if largest_quantum < (1 << 60):
        safe_low = max(low, np.int64(INT64_MIN) + 2 * largest_quantum) + largest_quantum
        safe_high = min(high, np.int64(INT64_MAX) - 2 * largest_quantum) - largest_quantum
        safe_first, safe_last = 0, block_len - 1
        while safe_first < block_len and times[safe_first] < safe_low:
            safe_first += 1
        while safe_last >= 0 and times[safe_last] > safe_high:
            safe_last -= 1
    largest_move = -1
    quantum = largest_quantum
    jump = largest_quantum
    grid_failed = False
    while largest_move < 0 and quantum > 1:
        if not (
            grid_failed
            and safe_last - safe_first >= 1
            and grid_cannot_meet_error(times[safe_first : safe_last + 1], block_len, quantum, time_error)
        ):
            phase = choose_grid_phase(sampled_ticks, quantum, previous_last, phase_scratch)
            base = _round_tick(times[0], quantum, phase)
            largest_move = snap_and_check_lattice(
                times, base, quantum, np.int64(INT64_MIN), low, high, allowed, max_move, out
            )
            if largest_move == -2:
                snap_to_grid_exact(times, quantum, phase, out)
                largest_move = _clamped_error(times, out, low, high, allowed, max_move)
            if largest_move < 0 and not grid_failed:
                # Mean-interval step failed: jump to the step of a short positive interval.
                sampled = sampled_intervals(times, pivots)
                positive = 0
                while positive < CADENCE_SAMPLES - 1 and sampled[positive] <= 0:
                    positive += 1
                short = sampled[min(positive + SHORT_INTERVAL_RANK, CADENCE_SAMPLES - 1)]
                jump = np.int64(choose_125_step(max(short, 1), time_error))
            grid_failed = True
        quantum = min(_finer_step(quantum), jump)
    return largest_move


@njit(inline="always")
def _try_clock_lattice(
    times: np.ndarray,
    largest_quantum: np.int64,
    low: np.int64,
    high: np.int64,
    allowed: np.ndarray,
    max_move: float,
    pivots: np.ndarray,
    trial: np.ndarray,
    out: np.ndarray,
    current_move: int,
) -> tuple[int, np.int64]:
    """Attempts to fit a custom clock lattice when grid rounding left a clock block irregular.

    Args:
        times: 1D int64 array of original block timestamps.
        largest_quantum: Coarsest 1-2-5 quantum for the block.
        low: Minimum allowed rounded tick.
        high: Maximum allowed rounded tick.
        allowed: 1D float64 array of per-tick allowances.
        max_move: Maximum allowed tick displacement.
        pivots: Scratch buffer for ``estimate_clock_period``.
        trial: Scratch 1D int64 array for testing the lattice without clobbering ``out``.
        out: In-out 1D int64 array holding the best rounded ticks so far; overwritten if the
            custom lattice succeeds.
        current_move: Largest displacement of the current contents of ``out``.

    Returns:
        Tuple ``(largest_move, out_step)`` with the updated largest displacement and uniform
        interval of ``out`` (0 if still irregular).
    """
    # Only try a custom lattice if out is not already regular, the original ticks show a clock
    # cadence, and the grid left one-scan intervals unequal (e.g. 1024 Hz or 60 Hz clocks).
    out_step = np.int64(uniform_interval(out))
    if out_step != 0:
        return current_move, out_step
    interval = estimate_clock_period(times, pivots, max_move)
    if interval > 0.0 and not one_scan_intervals_equal(out, interval):
        origin, period = fit_clock_lattice(times, interval, largest_quantum, max_move)
        lattice_move = _snap_and_check_lattice_any(
            times, origin, period, np.int64(INT64_MIN), low, high, allowed, max_move, trial
        )
        if lattice_move >= 0:
            out[:] = trial
            return lattice_move, np.int64(uniform_interval(out))
    return current_move, out_step


@njit(nogil=True, cache=True)
def round_blocks(
    ticks: np.ndarray,
    sample_offsets: np.ndarray,
    time_error: float,
    lows: np.ndarray,
    highs: np.ndarray,
    seed_ticks: np.ndarray,
    seed_steps: np.ndarray,
    out_ticks: np.ndarray,
    out_moves: np.ndarray,
) -> tuple[int, int]:
    """Rounds each irregular block's timestamps onto the first grid or lattice meeting ``time_error``.

    Exactly regular blocks and blocks of fewer than 3 samples are kept verbatim. Cross-block
    ordering between independently rounded blocks is reconciled afterwards by
    ``clamp_block_boundaries``.

    Args:
        ticks: 1D int64 array of all samples' timestamps (no NaT).
        sample_offsets: 1D int64 array of cumulative per-block sample offsets (``num_blocks + 1``).
        time_error: Positive relative time error tolerance.
        lows: 1D int64 array of per-block lower rounding bounds (``INT64_MIN`` for none).
        highs: 1D int64 array of per-block upper rounding bounds (``INT64_MAX`` for none).
        seed_ticks: 1D int64 array giving a preceding stored block's last tick, or ``INT64_MIN``
            to continue from the preceding block in this call.
        seed_steps: 1D int64 array giving that stored block's uniform interval (0 if irregular).
        out_ticks: Output 1D int64 array receiving rounded timestamps (fully populated only when
            at least one tick moved, i.e. ``out_moves.any()``).
        out_moves: Output 1D int64 array receiving each block's largest tick displacement.

    Returns:
        Tuple ``(OK, 0)`` on success, or ``(DECREASING, sample_idx)`` for the first decreasing tick.
    """
    # Validate global non-decreasing order first, then process each block through three stages:
    # (1) continue previous block's regular clock if matching, (2) coarsest valid 1-2-5 grid,
    # (3) custom clock lattice if still irregular. Unchanged blocks are deferred and copied
    # into out_ticks at the end only if at least one block actually moved.
    num_samples = ticks.shape[0]
    decreased = False
    for sample_idx in range(1, num_samples):
        decreased |= ticks[sample_idx] < ticks[sample_idx - 1]
    if decreased:
        for sample_idx in range(1, num_samples):
            if ticks[sample_idx] < ticks[sample_idx - 1]:
                return DECREASING, sample_idx
    pivots = np.empty(CADENCE_SAMPLES, np.int64)
    phase_scratch = np.empty(PHASE_SAMPLES, np.int64)
    phase_ticks = np.empty(PHASE_SAMPLES, np.int64)
    trial_ticks = np.empty(num_samples, np.int64)
    allowances = np.empty(num_samples, np.float64)
    kept = np.zeros(sample_offsets.shape[0] - 1, np.bool_)
    any_moved = False
    previous_last = np.int64(INT64_MIN)
    previous_step = np.int64(0)
    for block_idx in range(sample_offsets.shape[0] - 1):
        first_sample = sample_offsets[block_idx]
        block_len = sample_offsets[block_idx + 1] - first_sample
        out_moves[block_idx] = 0
        if seed_ticks[block_idx] != INT64_MIN:
            previous_last = seed_ticks[block_idx]
            previous_step = seed_steps[block_idx]
        if block_len == 0:
            continue
        times = ticks[first_sample : first_sample + block_len]
        step = np.int64(uniform_interval(times)) if block_len >= 2 else np.int64(0)
        mean = (times[block_len - 1] - times[0]) / (block_len - 1) if block_len >= 3 else 0.0
        # Keep exactly regular blocks and blocks of fewer than 3 samples unchanged.
        if step != 0 or mean <= 0.0:
            kept[block_idx] = True
            previous_last = times[block_len - 1]
            previous_step = step
            continue
        out = out_ticks[first_sample : first_sample + block_len]
        trial = trial_ticks[first_sample : first_sample + block_len]
        allowed = allowances[first_sample : first_sample + block_len]
        compute_tick_allowances(times, time_error, allowed)
        low, high = lows[block_idx], highs[block_idx]
        max_move = time_error * TIME_ERROR_SLACK * mean
        largest_move = -1
        # Stage 1: continue the preceding regular clock if the mean interval matches its period.
        same_clock = previous_step > 0 and times[0] >= previous_last and np.round(mean / previous_step) == 1.0
        if same_clock:
            largest_move = _snap_and_check_lattice_any(
                times, previous_last, previous_step, np.int64(1), low, high, allowed, max_move, out
            )
        # Stage 2: search for the coarsest valid 1-2-5 grid step.
        largest_quantum = np.int64(choose_125_step(mean, time_error))
        if largest_move < 0:
            sampled_ticks = sample_phase_ticks(times, phase_ticks)
            largest_move = _try_grid_rounding(
                times,
                largest_quantum,
                time_error,
                low,
                high,
                previous_last,
                allowed,
                max_move,
                sampled_ticks,
                phase_scratch,
                pivots,
                out,
            )
        if largest_move < 0:
            out[:] = times
            largest_move = 0
        # Stage 3: fit a custom clock lattice if a clock block remained irregular after grid snapping.
        largest_move, out_step = _try_clock_lattice(
            times, largest_quantum, low, high, allowed, max_move, pivots, trial, out, largest_move
        )
        out_moves[block_idx] = largest_move
        any_moved |= largest_move > 0
        previous_last = out[block_len - 1]
        previous_step = out_step
    if any_moved:
        for block_idx in range(sample_offsets.shape[0] - 1):
            if kept[block_idx]:
                first_sample, end_sample = sample_offsets[block_idx], sample_offsets[block_idx + 1]
                out_ticks[first_sample:end_sample] = ticks[first_sample:end_sample]
    return OK, 0


@njit(nogil=True, cache=True)
def clamp_block_boundaries(
    ticks: np.ndarray, sample_offsets: np.ndarray, moves: np.ndarray, out_ticks: np.ndarray
) -> None:
    """Restores monotonic ordering across block boundaries after independent per-block rounding.

    Args:
        ticks: 1D int64 array of original non-decreasing timestamps.
        sample_offsets: 1D int64 array of cumulative per-block sample offsets (``num_blocks + 1``).
        moves: 1D int64 array of each block's largest displacement (from ``round_blocks``).
        out_ticks: In-out 1D int64 array of rounded timestamps, clamped in place at overlaps.
    """
    # At any boundary where the previous block's last rounded tick exceeds the next block's first,
    # pick a meeting point clamped within [given_prev_last, given_next_first] (letting the block
    # with the larger move give way). Clamping earlier ticks down and later ticks up only moves
    # ticks back toward their original values, preserving every block's error bound.
    previous_block, previous_end = -1, 0
    for block_idx in range(sample_offsets.shape[0] - 1):
        first_sample, end = sample_offsets[block_idx], sample_offsets[block_idx + 1]
        if end == first_sample:
            continue
        if previous_block >= 0 and out_ticks[previous_end - 1] > out_ticks[first_sample]:
            low, high = ticks[previous_end - 1], ticks[first_sample]
            if moves[previous_block] >= moves[block_idx]:
                boundary = min(max(out_ticks[first_sample], low), high)
            else:
                boundary = min(max(out_ticks[previous_end - 1], low), high)
            sample_idx = previous_end - 1
            while sample_idx >= 0 and out_ticks[sample_idx] > boundary:
                out_ticks[sample_idx] = boundary
                sample_idx -= 1
            sample_idx = first_sample
            while sample_idx < end and out_ticks[sample_idx] < boundary:
                out_ticks[sample_idx] = boundary
                sample_idx += 1
        previous_block, previous_end = block_idx, end
