# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Unified 2-bit sample exception codes and escaped-value handling.

Every block flagged with ``BLOCK_FLAG_SAMPLE_CODES`` carries two bit planes assigning
each sample one of four 2-bit codes:

- ``00`` (``CODE_ON_GRID``): quantized on the block's grid;
- ``01`` (``CODE_NAN``): canonical quiet NaN;
- ``10`` (``CODE_ESCAPE``): next float64 in the group's byte-planed ``escaped_values``
  trailer (``+/-inf`` or a finite outlier stored separately so the bulk retains
  precision);
- ``11`` (``CODE_REPEAT_ESCAPE``): repeats the block's previous escaped float64 value.

Provides non-finite value replacement, outlier escaping (when it meaningfully refines
the bulk step), escape collection, byte-plane packing/unpacking, group splicing, and
decode-time exception restoration.
"""

import math

import numpy as np
from numba import njit

from . import _extreme_magnitudes as xm
from ._basic_stats import sum_squares
from ._bitpacking import sample_code_planes_view, unpack_sample_code
from ._format import (
    BLOCK_FLAG_SAMPLE_CODES,
    BYTES_PER_ESCAPED_VALUE,
    CODE_ESCAPE,
    CODE_NAN,
    CODE_ON_GRID,
    CODE_REPEAT_ESCAPE,
)
from ._grids import (
    GRIDS_DECIMAL,
    NO_DECIMAL,
    choose_decimal_or_binary_grid,
    snapped_exponent_for_range,
)
from ._noise import MAD_TO_SD

ESCAPE_MIN_LEN: int = 16
"""Minimum block length eligible for finite-outlier escaping."""

ESCAPE_TRIGGER: float = 0.05
"""Mean proximity to either range endpoint (as a fraction of span) that triggers outlier
inspection.
"""

ESCAPE_RANGE_SPREAD: float = 16.0
"""Range-to-standard-deviation ratio that also triggers outlier inspection (two-sided
outliers).
"""

ESCAPE_DEVIATIONS: float = 16.0
"""Cutoff in robust standard deviations from the trimmed center beyond which samples are
outliers.
"""

ESCAPE_MAX_SHARE: int = 16
"""Maximum fraction denominator: at most ``1 / ESCAPE_MAX_SHARE`` of a block may be
escaped.
"""

ESCAPE_MIN_GAIN: int = 2
"""Minimum bit-precision gain (``2^ESCAPE_MIN_GAIN`` finer bulk step) required to escape
outliers.
"""


@njit(nogil=True, cache=True)
def replace_nonfinite(
    samples: np.ndarray, out_held_samples: np.ndarray, out_codes: np.ndarray
) -> tuple[float, float, float, int]:
    """Classifies non-finite samples and replaces them via sample-and-hold.

    Each non-finite sample is substituted in ``out_held_samples`` by the preceding
    finite sample (or the block's first finite sample for leading runs; ``0.0`` if
    none are finite).

    Args:
        samples: 1D float64 array of raw block samples.
        out_held_samples: Output 1D float64 array receiving cleaned samples.
        out_codes: Output 1D uint8 array receiving ``CODE_ON_GRID``, ``CODE_NAN``, or
            ``CODE_ESCAPE``.

    Returns:
        Tuple ``(min_val, max_val, mean_val, finite_count)`` computed over finite
        samples only (all NaN with count 0 when no sample is finite).
    """
    # Find the first finite sample to seed leading non-finite runs, then make one pass
    # to classify codes, hold values, and accumulate finite summary statistics.
    num_samples = samples.shape[0]
    first_finite_idx = -1
    for sample_idx in range(num_samples):
        if math.isfinite(samples[sample_idx]):
            first_finite_idx = sample_idx
            break
    prev_finite_val = samples[first_finite_idx] if first_finite_idx >= 0 else 0.0
    min_val = max_val = prev_finite_val
    accum_sum = 0.0
    finite_count = 0
    for sample_idx in range(num_samples):
        val = samples[sample_idx]
        if math.isfinite(val):
            prev_finite_val = val
            min_val = min(min_val, val)
            max_val = max(max_val, val)
            accum_sum += val
            finite_count += 1
            out_codes[sample_idx] = CODE_ON_GRID
        else:
            out_codes[sample_idx] = CODE_NAN if math.isnan(val) else CODE_ESCAPE
        out_held_samples[sample_idx] = prev_finite_val
    if finite_count == 0:
        return math.nan, math.nan, math.nan, 0
    mean_val = accum_sum / finite_count
    if not math.isfinite(mean_val):
        mean_val = xm.mean_by_division(samples, finite_count)
    return min_val, max_val, mean_val, finite_count


@njit(nogil=True, cache=True)
def hold_exception_samples(
    samples: np.ndarray, codes: np.ndarray, out_held: np.ndarray
) -> None:
    """Replaces every non-on-grid sample with the nearest preceding on-grid sample.

    Leading exception runs take the block's first on-grid sample (or ``0.0`` if none
    exist).

    Args:
        samples: 1D float64 array of block samples.
        codes: 1D uint8 array of per-sample exception codes.
        out_held: Output 1D float64 array receiving sample-and-held values.
    """
    num_samples = samples.shape[0]
    previous = 0.0
    for sample_idx in range(num_samples):
        if codes[sample_idx] == CODE_ON_GRID:
            previous = samples[sample_idx]
            break
    for sample_idx in range(num_samples):
        if codes[sample_idx] == CODE_ON_GRID:
            previous = samples[sample_idx]
        out_held[sample_idx] = previous


@njit(inline="always")
def _on_decimal_grid(
    samples: np.ndarray, codes: np.ndarray, has_codes: bool, decimal_exp: int
) -> bool:
    """Checks whether every on-grid sample equals the float64 nearest ``k *
    10^decimal_exp``.

    Args:
        samples: 1D float64 array of block samples.
        codes: 1D uint8 array of sample codes (consulted only when ``has_codes`` is
            True).
        has_codes: Whether ``codes`` contains active exception markers.
        decimal_exp: Candidate decimal exponent ``p``.

    Returns:
        True if every on-grid sample reconstructs identically on ``10^decimal_exp``.
    """
    for sample_idx in range(samples.shape[0]):
        if has_codes and codes[sample_idx] != CODE_ON_GRID:
            continue
        value = samples[sample_idx]
        if decimal_exp < 0:
            divisor = 10.0**-decimal_exp
            if float(round(value * divisor)) / divisor != value:
                return False
        else:
            multiplier = 10.0**decimal_exp
            if float(round(value / multiplier)) * multiplier != value:
                return False
    return True


@njit(inline="always")
def _on_binary_grid(
    samples: np.ndarray, codes: np.ndarray, has_codes: bool, quant_exp: int
) -> bool:
    """Checks whether every on-grid sample is an exact multiple of ``2^quant_exp``.

    Args:
        samples: 1D float64 array of block samples.
        codes: 1D uint8 array of sample codes (consulted only when ``has_codes`` is
            True).
        has_codes: Whether ``codes`` contains active exception markers.
        quant_exp: Candidate power-of-two exponent.

    Returns:
        True if every on-grid sample lies exactly on ``k * 2^quant_exp``.
    """
    for sample_idx in range(samples.shape[0]):
        if has_codes and codes[sample_idx] != CODE_ON_GRID:
            continue
        scaled = math.ldexp(samples[sample_idx], -quant_exp)
        if scaled != math.floor(scaled):
            return False
    return True


@njit(inline="always")
def _triggers_outlier_check(
    samples: np.ndarray,
    codes: np.ndarray,
    has_codes: bool,
    mean: float,
    lower_bound: float,
    upper_bound: float,
    span: float,
) -> bool:
    """Checks whether a block's mean skew or range-to-SD ratio warrants full outlier
    inspection.

    Args:
        samples: 1D float64 array of block samples.
        codes: 1D uint8 array of sample codes.
        has_codes: Whether ``codes`` contains active exception markers.
        mean: Arithmetic mean of finite samples.
        lower_bound: Minimum finite sample value.
        upper_bound: Maximum finite sample value.
        span: Positive finite range ``upper_bound - lower_bound``.

    Returns:
        True if the block should proceed to trimmed-center outlier detection.
    """
    # Fast O(1) trigger: one-sided outliers pull the mean within ESCAPE_TRIGGER of one
    # end.
    if (
        mean - lower_bound < ESCAPE_TRIGGER * span
        or upper_bound - mean < ESCAPE_TRIGGER * span
    ):
        return True
    # Secondary trigger for symmetric/two-sided outliers: range exceeds
    # ESCAPE_RANGE_SPREAD * SD.
    squares = 0.0
    count = samples.shape[0]
    if has_codes:
        count = 0
        for sample_idx in range(samples.shape[0]):
            if codes[sample_idx] == CODE_ON_GRID:
                dev = samples[sample_idx] - mean
                squares += dev * dev
                count += 1
    else:
        squares = sum_squares(samples, mean)
    return bool(
        span * span
        > ESCAPE_RANGE_SPREAD * ESCAPE_RANGE_SPREAD * squares / max(count, 1)
    )


@njit(inline="always")
def _trimmed_center_and_mad(
    samples: np.ndarray, codes: np.ndarray, has_codes: bool, initial_center: float
) -> tuple[bool, float, float]:
    """Estimates a robust center and MAD of on-grid samples via three iterative trimming
    passes.

    Args:
        samples: 1D float64 array of block samples.
        codes: 1D uint8 array of sample codes.
        has_codes: Whether ``codes`` contains active exception markers.
        initial_center: Starting center estimate (block mean).

    Returns:
        Tuple ``(ok, center, mad)`` where ``ok`` is False if no on-grid samples remain.
    """
    num_samples = samples.shape[0]
    center_val = initial_center
    deviation = 0.0
    for _ in range(3):
        total = 0.0
        count = 0
        for sample_idx in range(num_samples):
            if not has_codes or codes[sample_idx] == CODE_ON_GRID:
                total += abs(samples[sample_idx] - center_val)
                count += 1
        if count == 0:
            return False, 0.0, 0.0
        limit = 4.0 * MAD_TO_SD * total / count
        kept_sum = 0.0
        kept = 0
        deviation = 0.0
        for sample_idx in range(num_samples):
            if not has_codes or codes[sample_idx] == CODE_ON_GRID:
                dev = abs(samples[sample_idx] - center_val)
                if dev <= limit:
                    kept_sum += samples[sample_idx]
                    deviation += dev
                    kept += 1
        if kept == 0:
            return False, 0.0, 0.0
        center_val = kept_sum / kept
        deviation /= kept
    return True, center_val, deviation


@njit(nogil=True, cache=True)
def escape_outliers(
    samples: np.ndarray,
    held: np.ndarray,
    codes: np.ndarray,
    has_codes: bool,
    mean: float,
    lower_bound: float,
    upper_bound: float,
    floor_exp: int,
    max_bits: int,
    grids: int,
    scratch_quantized: np.ndarray,
) -> tuple[bool, float, float]:
    """Escapes finite outliers in a block when doing so meaningfully refines the bulk
    step.

    Escaped samples are tagged ``CODE_ESCAPE`` in ``codes`` and sample-and-held into
    ``held`` so the block can be quantized over the narrower bulk range ``[bulk_min,
    bulk_max]``.

    Args:
        samples: 1D float64 array of block samples.
        held: Output 1D float64 array receiving sample-and-held values when escaping
            triggers.
        codes: In-out 1D uint8 array of sample exception codes.
        has_codes: Whether ``codes`` is already initialized with non-finite markers.
        mean: Arithmetic mean of finite samples.
        lower_bound: Minimum finite sample value.
        upper_bound: Maximum finite sample value.
        floor_exp: Coarsest active floor exponent (noise floor or absolute tolerance).
        max_bits: Maximum quantization bit budget.
        grids: Active grid mode (``GRIDS_BINARY``, ``GRIDS_DECIMAL``, or
            ``GRIDS_RATIONAL``).
        scratch_quantized: Preallocated 1D int32 scratch buffer for decimal-grid
            checking.

    Returns:
        Tuple ``(escaped, bulk_min, bulk_max)``. When ``escaped`` is False, returns
        ``(False, lower_bound, upper_bound)`` and leaves ``codes`` and ``held``
        unchanged.
    """
    # 1. Check cheap prerequisites (length, finite positive span, floor headroom,
    #    skew/spread trigger).
    # 2. Estimate trimmed center and MAD, count samples beyond ESCAPE_DEVIATIONS robust
    #    sigmas, and verify both outlier share <= 1 / ESCAPE_MAX_SHARE and >=
    #    ESCAPE_MIN_GAIN bit refinement.
    # 3. Skip blocks already exact on their full-range binary or decimal grid; otherwise
    #    mark outliers as CODE_ESCAPE and re-hold all coded samples strictly inside the
    #    bulk range.
    num_samples = samples.shape[0]
    span = upper_bound - lower_bound
    if num_samples < ESCAPE_MIN_LEN or not span > 0 or not math.isfinite(span):
        return False, lower_bound, upper_bound
    old_exp = snapped_exponent_for_range(lower_bound, upper_bound, max_bits)
    if old_exp - floor_exp < ESCAPE_MIN_GAIN:
        return False, lower_bound, upper_bound
    if not _triggers_outlier_check(
        samples, codes, has_codes, mean, lower_bound, upper_bound, span
    ):
        return False, lower_bound, upper_bound
    ok, center_val, deviation = _trimmed_center_and_mad(samples, codes, has_codes, mean)
    if not ok:
        return False, lower_bound, upper_bound
    threshold = ESCAPE_DEVIATIONS * MAD_TO_SD * deviation
    if not math.isfinite(threshold) or threshold < 0.0:
        return False, lower_bound, upper_bound
    # Measure outlier count and bulk bounds without mutating codes yet.
    num_escaped = 0
    bulk_min, bulk_max = math.inf, -math.inf
    for sample_idx in range(num_samples):
        if has_codes and codes[sample_idx] != CODE_ON_GRID:
            continue
        value = samples[sample_idx]
        if abs(value - center_val) > threshold:
            num_escaped += 1
        else:
            bulk_min = min(bulk_min, value)
            bulk_max = max(bulk_max, value)
    if (
        num_escaped == 0
        or num_escaped * ESCAPE_MAX_SHARE > num_samples
        or not bulk_max >= bulk_min
    ):
        return False, lower_bound, upper_bound
    new_exp = (
        snapped_exponent_for_range(bulk_min, bulk_max, max_bits)
        if bulk_max > bulk_min
        else old_exp - ESCAPE_MIN_GAIN
    )
    if old_exp - max(new_exp, floor_exp) < ESCAPE_MIN_GAIN:
        return False, lower_bound, upper_bound
    # Do not escape samples when the full-range grid already represents the entire block
    # exactly.
    if _on_binary_grid(samples, codes, has_codes, old_exp):
        return False, lower_bound, upper_bound
    if grids >= GRIDS_DECIMAL and span < xm.WIDE_RANGE:
        decimal_exp, _, _ = choose_decimal_or_binary_grid(
            samples, lower_bound, upper_bound, old_exp, scratch_quantized
        )
        if decimal_exp != NO_DECIMAL and _on_decimal_grid(
            samples, codes, has_codes, decimal_exp
        ):
            return False, lower_bound, upper_bound
    # Commit outlier codes and re-hold every exception sample from the bulk so leading
    # outliers and non-finite runs after an outlier stay within [bulk_min, bulk_max].
    if not has_codes:
        codes[:] = CODE_ON_GRID
    for sample_idx in range(num_samples):
        if (
            codes[sample_idx] == CODE_ON_GRID
            and abs(samples[sample_idx] - center_val) > threshold
        ):
            codes[sample_idx] = CODE_ESCAPE
    hold_exception_samples(samples, codes, held)
    return True, bulk_min, bulk_max


def collect_escaped_values(
    samples: np.ndarray,
    block_sizes: np.ndarray,
    block_flags: np.ndarray,
    in_out_codes: np.ndarray,
    block_min: np.ndarray,
    block_max: np.ndarray,
    max_bits: int,
) -> np.ndarray:
    """Collects escaped float64 values in block/sample order and collapses consecutive
    repeats.

    Values exact in float32 (such as ``+/-inf`` and common sentinel codes) are stored
    verbatim; other finite values are rounded to the block's full-range ``max_bits``
    step. Consecutive identical escaped values within the same block are rewritten to
    ``CODE_REPEAT_ESCAPE`` in ``in_out_codes``.

    Args:
        samples: 1D float64 array of all raw samples in the block group.
        block_sizes: 1D int64 array of per-block sample counts.
        block_flags: 1D uint8 array of per-block flags.
        in_out_codes: In-out 1D uint8 array of per-sample exception codes.
        block_min: 1D float64 array of per-block finite minima.
        block_max: 1D float64 array of per-block finite maxima.
        max_bits: Maximum quantization bit budget used for rounding non-float32
            outliers.

    Returns:
        1D float64 array of escaped values to store in the group trailer.
    """
    if not (block_flags & BLOCK_FLAG_SAMPLE_CODES).any():
        return np.zeros(0)
    offsets = np.zeros(block_sizes.shape[0] + 1, np.int64)
    np.cumsum(block_sizes, out=offsets[1:])
    flagged = np.repeat((block_flags & BLOCK_FLAG_SAMPLE_CODES) != 0, block_sizes)
    positions = np.flatnonzero(flagged & (in_out_codes == CODE_ESCAPE))
    if not positions.shape[0]:
        return np.zeros(0)
    blocks = np.searchsorted(offsets, positions, side="right") - 1
    values: list[float] = []
    previous_block, previous_value = -1, math.nan
    for position, block in zip(positions.tolist(), blocks.tolist(), strict=True):
        value = float(samples[position])
        # Round non-float32 finite outliers to the full-range max_bits grid so low
        # mantissa bytes zero out.
        if math.isfinite(value) and float(np.float32(value)) != value:
            exp = snapped_exponent_for_range(
                float(block_min[block]), float(block_max[block]), max_bits
            )
            rounded = math.ldexp(round(math.ldexp(value, -exp)), exp)
            value = rounded if math.isfinite(rounded) else value
        if block == previous_block and value == previous_value:
            in_out_codes[position] = CODE_REPEAT_ESCAPE
            continue
        values.append(value)
        previous_block, previous_value = block, value
    return np.array(values, np.float64)


def pack_escaped_values(escaped_values: np.ndarray) -> np.ndarray:
    """Serializes escaped float64 values into 8 byte planes (all byte 0s, then byte 1s,
    ...).

    Args:
        escaped_values: 1D float64 array of escaped sample values.

    Returns:
        1D uint8 array of length ``8 * len(escaped_values)``.
    """
    return np.ascontiguousarray(
        np.asarray(escaped_values, np.float64)
        .view(np.uint8)
        .reshape(-1, BYTES_PER_ESCAPED_VALUE)
        .T
    ).ravel()


def unpack_escaped_values(body: np.ndarray, start: int, num_escapes: int) -> np.ndarray:
    """Deserializes ``num_escapes`` byte-planed float64 values from an uncompressed
    body.

    Args:
        body: 1D uint8 array of uncompressed body bytes.
        start: Starting byte offset of the ``escaped_values`` trailer in ``body``.
        num_escapes: Number of escaped float64 values to read.

    Returns:
        1D float64 array of length ``num_escapes``.
    """
    if not num_escapes:
        return np.zeros(0)
    end = start + BYTES_PER_ESCAPED_VALUE * num_escapes
    return (
        np.ascontiguousarray(
            body[start:end].reshape(BYTES_PER_ESCAPED_VALUE, num_escapes).T
        )
        .view(np.float64)
        .ravel()
    )


def splice_escaped_values(
    old_escapes: np.ndarray,
    old_escape_offsets: np.ndarray,
    num_old_blocks: int,
    num_blocks: int,
    indices: np.ndarray,
    new_flags: np.ndarray,
    new_codes: np.ndarray,
    new_escapes: np.ndarray,
    new_sample_offsets: np.ndarray,
) -> np.ndarray:
    """Merges carried-over old block escapes with newly encoded block escapes in block
    order.

    Args:
        old_escapes: 1D float64 array of escaped values from the existing group.
        old_escape_offsets: 1D int64 cumulative escape counts for the existing blocks.
        num_old_blocks: Number of blocks in the existing group.
        num_blocks: Number of blocks in the spliced group.
        indices: 1D int64 array of target block indices being replaced or appended.
        new_flags: 1D uint8 array of block flags for the newly encoded blocks.
        new_codes: 1D uint8 array of sample codes for the newly encoded blocks.
        new_escapes: 1D float64 array of escaped values for the newly encoded blocks.
        new_sample_offsets: 1D int64 cumulative sample offsets for the newly encoded
            blocks.

    Returns:
        1D float64 array of merged escaped values in final block order.
    """
    # Count CODE_ESCAPE entries per newly encoded block to slice new_escapes, then
    # assemble each block's slice from either new_escapes (for updated indices) or
    # old_escapes.
    if not old_escapes.shape[0] and not new_escapes.shape[0]:
        return np.zeros(0)
    new_counts = np.zeros(indices.shape[0], np.int64)
    for rank in range(indices.shape[0]):
        if new_flags[rank] & BLOCK_FLAG_SAMPLE_CODES:
            first, end = (
                int(new_sample_offsets[rank]),
                int(new_sample_offsets[rank + 1]),
            )
            new_counts[rank] = int(
                np.count_nonzero(new_codes[first:end] == CODE_ESCAPE)
            )
    new_esc_offsets = np.zeros(indices.shape[0] + 1, np.int64)
    np.cumsum(new_counts, out=new_esc_offsets[1:])
    rank_of = np.full(num_blocks, -1, np.int64)
    rank_of[indices] = np.arange(indices.shape[0], dtype=np.int64)
    parts: list[np.ndarray] = []
    for block_idx in range(num_blocks):
        rank = int(rank_of[block_idx])
        if rank >= 0:
            part = new_escapes[new_esc_offsets[rank] : new_esc_offsets[rank + 1]]
        elif block_idx < num_old_blocks:
            part = old_escapes[
                old_escape_offsets[block_idx] : old_escape_offsets[block_idx + 1]
            ]
        else:
            continue
        if part.shape[0]:
            parts.append(part)
    return np.concatenate(parts) if parts else np.zeros(0)


@njit(nogil=True, cache=True)
def count_block_escapes(
    body: np.ndarray,
    sample_offsets: np.ndarray,
    plane_byte_offsets: np.ndarray,
    code_plane_offsets: np.ndarray,
    has_time: bool,
    out_offsets: np.ndarray,
) -> bool:
    """Computes each block's starting offset in the ``escaped_values`` trailer and
    validates repeats.

    Args:
        body: 1D uint8 array of uncompressed body bytes.
        sample_offsets: 1D int64 cumulative sample counts (length ``num_blocks + 1``).
        plane_byte_offsets: 1D int64 cumulative residual plane-byte counts.
        code_plane_offsets: 1D int64 cumulative sample-code plane-byte counts.
        has_time: Whether the body includes a time axis.
        out_offsets: Output 1D int64 array of length ``num_blocks + 1`` receiving
            cumulative escape counts.

    Returns:
        True if all sample code planes are valid, or False if any block uses
        ``CODE_REPEAT_ESCAPE`` before its first ``CODE_ESCAPE``.
    """
    num_blocks = sample_offsets.shape[0] - 1
    num_plane_bytes = int(plane_byte_offsets[num_blocks])
    code_planes = sample_code_planes_view(
        body, num_blocks, num_plane_bytes, int(code_plane_offsets[num_blocks]), has_time
    )
    out_offsets[0] = 0
    for block_idx in range(num_blocks):
        count = 0
        if code_plane_offsets[block_idx + 1] > code_plane_offsets[block_idx]:
            seen = False
            block_len = sample_offsets[block_idx + 1] - sample_offsets[block_idx]
            for byte_idx in range(
                code_plane_offsets[block_idx + 1] - code_plane_offsets[block_idx]
            ):
                plane0_byte = code_planes[0, code_plane_offsets[block_idx] + byte_idx]
                plane1_byte = code_planes[1, code_plane_offsets[block_idx] + byte_idx]
                # Skip 8-sample groups whose codes are all CODE_ON_GRID (00b).
                if not plane0_byte | plane1_byte:
                    continue
                for bit_idx in range(min(8, block_len - 8 * byte_idx)):
                    code = unpack_sample_code(plane0_byte, plane1_byte, bit_idx)
                    if code == CODE_ESCAPE:
                        count += 1
                        seen = True
                    elif code == CODE_REPEAT_ESCAPE and not seen:
                        return False
        out_offsets[block_idx + 1] = out_offsets[block_idx] + count
    return True


@njit(nogil=True, cache=True)
def restore_exception_samples(
    code_planes: np.ndarray,
    plane_byte_offset: int,
    escaped_values: np.ndarray,
    in_out_samples: np.ndarray,
) -> None:
    """Overwrites a flagged block's decoded samples with NaNs and escaped float64
    values.

    Args:
        code_planes: 2D uint8 view of shape ``(2, total_code_plane_bytes)``.
        plane_byte_offset: Starting plane-byte column for this block in ``code_planes``.
        escaped_values: 1D float64 slice of escaped values belonging to this block.
        in_out_samples: In-out 1D float64 array of decoded block samples.
    """
    num_samples = in_out_samples.shape[0]
    next_escape = 0
    for byte_idx in range((num_samples + 7) // 8):
        plane0_byte = code_planes[0, plane_byte_offset + byte_idx]
        plane1_byte = code_planes[1, plane_byte_offset + byte_idx]
        # Skip 8-sample groups with no exception codes.
        if not plane0_byte | plane1_byte:
            continue
        for bit_idx in range(min(8, num_samples - 8 * byte_idx)):
            code = unpack_sample_code(plane0_byte, plane1_byte, bit_idx)
            sample_offset = 8 * byte_idx + bit_idx
            if code == CODE_NAN:
                in_out_samples[sample_offset] = np.nan
            elif code == CODE_ESCAPE:
                in_out_samples[sample_offset] = escaped_values[next_escape]
                next_escape += 1
            elif code == CODE_REPEAT_ESCAPE:
                in_out_samples[sample_offset] = escaped_values[next_escape - 1]
