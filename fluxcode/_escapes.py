# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Unified 2-bit code plane and escaped-value handling (non-finite samples and outliers).

Every block with `BLOCK_FLAG_CODED` (bit 3) carries two bit planes encoding per-sample codes:
  - 00 (`CODE_FINITE`): quantized on the block's grid;
  - 01 (`CODE_NAN`): canonical quiet NaN;
  - 10 (`CODE_ESCAPE`): next float64 in the block group's byte-planed escapes trailer
    (±inf or an outlier stored apart so the bulk keeps its precision);
  - 11 (`CODE_REPEAT`): repeats the block's previous escaped value (runs of an error code
    store the value once).

Consolidates imputation (`fill_nonfinite`, `hold_coded`), benefit-gated outlier detection
(`escape_outliers`), escape collection/serialization (`collect_escapes`, `escape_bytes`,
`read_escapes`, `escape_offsets`, `splice_escapes`), and decode restoration (`restore_coded`).
"""

import math

import numpy as np
from numba import njit

from . import _extreme_magnitudes as xm
from ._bitpacking import code_planes_view, get_code
from ._format import (
    BLOCK_FLAG_CODED,
    BYTES_PER_ESCAPE,
    CODE_ESCAPE,
    CODE_FINITE,
    CODE_NAN,
    CODE_REPEAT,
)
from ._grids import (
    GRIDS_DECIMAL,
    NO_DECIMAL,
    decimal_grid,
    range_exponent,
)
from ._noise import MAD_TO_SD

ESCAPE_MIN_LEN: int = 16
"""Smallest block whose finite outliers may be escaped."""

ESCAPE_TRIGGER: float = 0.05
"""A block is inspected for outliers when its mean lies within this share of its range from one end."""

ESCAPE_RANGE_SPREAD: float = 16.0
"""... or when its range exceeds this many standard deviations (outliers on both sides)."""

ESCAPE_DEVIATIONS: float = 16.0
"""Outliers lie beyond this many robust standard deviations from the trimmed center."""

ESCAPE_MAX_SHARE: int = 16
"""At most 1 / ESCAPE_MAX_SHARE of a block's samples may be escaped as finite outliers."""

ESCAPE_MIN_GAIN: int = 2
"""Finite outliers are escaped only when the bulk's step is at least 2^ESCAPE_MIN_GAIN times finer."""


@njit(nogil=True, cache=True)
def fill_nonfinite(
    samples: np.ndarray, out_held_samples: np.ndarray, out_codes: np.ndarray
) -> tuple[float, float, float, int]:
    """Imputes non-finite samples with held values and classifies sample categories.

    Replaces each non-finite sample in samples with the preceding finite value
    (or the first finite value in the block for leading non-finite runs). Records
    2-bit classification codes (`CODE_FINITE`, `CODE_NAN`, or `CODE_ESCAPE` for ±inf)
    and calculates block statistics (min, max, mean, count) over finite samples only.
    """
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
            out_codes[sample_idx] = CODE_FINITE
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
def hold_coded(samples: np.ndarray, codes: np.ndarray, out_held: np.ndarray) -> None:
    """Sample-and-holds every coded (non-finite or escaped) sample from the previous uncoded one
    (the first uncoded one for a leading run; 0.0 if none)."""
    num_samples = samples.shape[0]
    previous = 0.0
    for sample_idx in range(num_samples):
        if codes[sample_idx] == CODE_FINITE:
            previous = samples[sample_idx]
            break
    for sample_idx in range(num_samples):
        if codes[sample_idx] == CODE_FINITE:
            previous = samples[sample_idx]
        out_held[sample_idx] = previous


@njit(nogil=True, cache=True, fastmath=True)
def _sum_squares(samples: np.ndarray, mean: float) -> float:
    """Sum of squared deviations from mean (vectorized trigger check)."""
    total = 0.0
    for sample_idx in range(samples.shape[0]):
        dev = samples[sample_idx] - mean
        total += dev * dev
    return total


@njit(inline="always")
def _on_decimal_grid(samples: np.ndarray, codes: np.ndarray, coded: bool, decimal_exp: int) -> bool:
    """Whether every uncoded sample is the double nearest k * 10^p (as the decoder makes it)."""
    for sample_idx in range(samples.shape[0]):
        if coded and codes[sample_idx] != CODE_FINITE:
            continue
        value = samples[sample_idx]
        if decimal_exp < 0:
            divisor = 10.0 ** -decimal_exp
            if float(round(value * divisor)) / divisor != value:
                return False
        else:
            multiplier = 10.0 ** decimal_exp
            if float(round(value / multiplier)) * multiplier != value:
                return False
    return True


@njit(inline="always")
def _on_binary_grid(samples: np.ndarray, codes: np.ndarray, coded: bool, exponent: int) -> bool:
    """Whether every uncoded sample is a multiple of 2^exponent."""
    for sample_idx in range(samples.shape[0]):
        if coded and codes[sample_idx] != CODE_FINITE:
            continue
        scaled = math.ldexp(samples[sample_idx], -exponent)
        if scaled != math.floor(scaled):
            return False
    return True


@njit(nogil=True, cache=True)
def escape_outliers(
    samples: np.ndarray,
    held: np.ndarray,
    codes: np.ndarray,
    coded: bool,
    mean: float,
    lower_bound: float,
    upper_bound: float,
    floor_exp: int,
    max_bits: int,
    grids: int,
    scratch_quantized: np.ndarray,
) -> tuple[bool, float, float]:
    """Escapes a block's finite outliers when that refines the rest of the block (ENCODER §5.1.1).

    Escaped samples receive code `CODE_ESCAPE` (10b) and are sample-and-held into `held`;
    the block is then quantized on the bulk's narrower range. Taken only when:
    - the mean is within `ESCAPE_TRIGGER` of one end OR range > `ESCAPE_RANGE_SPREAD` * std;
    - at most `1 / ESCAPE_MAX_SHARE` of samples lie beyond `ESCAPE_DEVIATIONS` robust sigmas;
    - the bulk step is at least `2^ESCAPE_MIN_GAIN` times finer than the full-range step and
      the noise/tolerance floor; and
    - the block is not already exact on its full-range binary or decimal grid.

    Returns:
        (escaped, bulk_min, bulk_max).
    """
    num_samples = samples.shape[0]
    span = upper_bound - lower_bound
    if num_samples < ESCAPE_MIN_LEN or not span > 0 or not math.isfinite(span):
        return False, lower_bound, upper_bound
    old_exp = range_exponent(lower_bound, upper_bound, max_bits)
    if old_exp - floor_exp < ESCAPE_MIN_GAIN:
        return False, lower_bound, upper_bound
    if not (mean - lower_bound < ESCAPE_TRIGGER * span or upper_bound - mean < ESCAPE_TRIGGER * span):
        squares = 0.0
        count = num_samples
        if coded:
            count = 0
            for sample_idx in range(num_samples):
                if codes[sample_idx] == CODE_FINITE:
                    dev = samples[sample_idx] - mean
                    squares += dev * dev
                    count += 1
        else:
            squares = _sum_squares(samples, mean)
        if not span * span > ESCAPE_RANGE_SPREAD * ESCAPE_RANGE_SPREAD * squares / max(count, 1):
            return False, lower_bound, upper_bound
    # Trimmed center and MAD of uncoded samples
    center_val = mean
    deviation = 0.0
    for _ in range(3):
        total = 0.0
        count = 0
        for sample_idx in range(num_samples):
            if not coded or codes[sample_idx] == CODE_FINITE:
                total += abs(samples[sample_idx] - center_val)
                count += 1
        if count == 0:
            return False, lower_bound, upper_bound
        limit = 4.0 * MAD_TO_SD * total / count
        kept_sum = 0.0
        kept = 0
        deviation = 0.0
        for sample_idx in range(num_samples):
            if not coded or codes[sample_idx] == CODE_FINITE:
                dev = abs(samples[sample_idx] - center_val)
                if dev <= limit:
                    kept_sum += samples[sample_idx]
                    deviation += dev
                    kept += 1
        if kept == 0:
            return False, lower_bound, upper_bound
        center_val = kept_sum / kept
        deviation /= kept
    threshold = ESCAPE_DEVIATIONS * MAD_TO_SD * deviation
    if not threshold > 0:
        return False, lower_bound, upper_bound
    num_escaped = 0
    bulk_min, bulk_max = math.inf, -math.inf
    for sample_idx in range(num_samples):
        if coded and codes[sample_idx] != CODE_FINITE:
            continue
        value = samples[sample_idx]
        if abs(value - center_val) > threshold:
            num_escaped += 1
        else:
            bulk_min = min(bulk_min, value)
            bulk_max = max(bulk_max, value)
    if num_escaped == 0 or num_escaped * ESCAPE_MAX_SHARE > num_samples or not bulk_max >= bulk_min:
        return False, lower_bound, upper_bound
    new_exp = range_exponent(bulk_min, bulk_max, max_bits) if bulk_max > bulk_min else old_exp - ESCAPE_MIN_GAIN
    if old_exp - max(new_exp, floor_exp) < ESCAPE_MIN_GAIN:
        return False, lower_bound, upper_bound
    if _on_binary_grid(samples, codes, coded, old_exp):
        return False, lower_bound, upper_bound
    if grids >= GRIDS_DECIMAL and span < xm.WIDE_RANGE:
        decimal_exp, _, _ = decimal_grid(samples, lower_bound, upper_bound, old_exp, scratch_quantized)
        if decimal_exp != NO_DECIMAL and _on_decimal_grid(samples, codes, coded, decimal_exp):
            return False, lower_bound, upper_bound
    # Hold escaped samples (and re-hold any prior non-finite samples so they don't hold a glitch)
    held[:] = samples
    previous = center_val
    for sample_idx in range(num_samples):
        uncoded = not coded or codes[sample_idx] == CODE_FINITE
        if uncoded and abs(samples[sample_idx] - center_val) <= threshold:
            previous = samples[sample_idx]
        else:
            held[sample_idx] = previous
    if not coded:
        codes[:] = CODE_FINITE
    for sample_idx in range(num_samples):
        if codes[sample_idx] == CODE_FINITE and abs(samples[sample_idx] - center_val) > threshold:
            codes[sample_idx] = CODE_ESCAPE
    return True, bulk_min, bulk_max


def collect_escapes(
    samples: np.ndarray,
    block_sizes: np.ndarray,
    block_flags: np.ndarray,
    in_out_codes: np.ndarray,
    block_min: np.ndarray,
    block_max: np.ndarray,
    max_bits: int,
) -> np.ndarray:
    """Collects escaped values for samples coded `CODE_ESCAPE` in block/sample order.

    A value equal to its block's previous escaped value becomes `CODE_REPEAT` (11b).
    Values representable exactly in float32 (error codes like -9999, 32767, 1e6; ±inf)
    are stored verbatim; others are rounded to the block's full-range `max_bits` step so
    low mantissa bytes compress away while preserving the full-range bound.
    """
    if not (block_flags & BLOCK_FLAG_CODED).any():
        return np.zeros(0)
    offsets = np.zeros(block_sizes.shape[0] + 1, np.int64)
    np.cumsum(block_sizes, out=offsets[1:])
    flagged = np.repeat((block_flags & BLOCK_FLAG_CODED) != 0, block_sizes)
    positions = np.flatnonzero(flagged & (in_out_codes == CODE_ESCAPE))
    if not positions.shape[0]:
        return np.zeros(0)
    blocks = np.searchsorted(offsets, positions, side="right") - 1
    values: list[float] = []
    previous_block, previous_value = -1, math.nan
    for position, block in zip(positions.tolist(), blocks.tolist()):
        value = float(samples[position])
        if math.isfinite(value) and float(np.float32(value)) != value:
            exp = range_exponent(float(block_min[block]), float(block_max[block]), max_bits)
            rounded = math.ldexp(round(math.ldexp(value, -exp)), exp)
            value = rounded if math.isfinite(rounded) else value
        if block == previous_block and value == previous_value:
            in_out_codes[position] = CODE_REPEAT
            continue
        values.append(value)
        previous_block, previous_value = block, value
    return np.array(values, np.float64)


def escape_bytes(escapes: np.ndarray) -> np.ndarray:
    """Serializes escaped float64 values as 8 byte planes (all byte 0s, all byte 1s, ...)."""
    return np.ascontiguousarray(np.asarray(escapes, np.float64).view(np.uint8).reshape(-1, BYTES_PER_ESCAPE).T).ravel()


def read_escapes(raw_body: np.ndarray, start: int, num_escapes: int) -> np.ndarray:
    """Deserializes `num_escapes` byte-planed float64 values starting at `start` in `raw_body`."""
    if not num_escapes:
        return np.zeros(0)
    end = start + BYTES_PER_ESCAPE * num_escapes
    return np.ascontiguousarray(raw_body[start:end].reshape(BYTES_PER_ESCAPE, num_escapes).T).view(np.float64).ravel()


def splice_escapes(
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
    """Merges carried-over old block escapes and newly encoded block escapes in block order."""
    if not old_escapes.shape[0] and not new_escapes.shape[0]:
        return np.zeros(0)
    new_counts = np.zeros(indices.shape[0], np.int64)
    for rank in range(indices.shape[0]):
        if new_flags[rank] & BLOCK_FLAG_CODED:
            first, end = int(new_sample_offsets[rank]), int(new_sample_offsets[rank + 1])
            new_counts[rank] = int(np.count_nonzero(new_codes[first:end] == CODE_ESCAPE))
    new_esc_offsets = np.zeros(indices.shape[0] + 1, np.int64)
    np.cumsum(new_counts, out=new_esc_offsets[1:])
    rank_of = np.full(num_blocks, -1, np.int64)
    rank_of[indices] = np.arange(indices.shape[0], dtype=np.int64)
    parts: list[np.ndarray] = []
    for block_idx in range(num_blocks):
        rank = int(rank_of[block_idx])
        if rank >= 0:
            part = new_escapes[new_esc_offsets[rank]:new_esc_offsets[rank + 1]]
        elif block_idx < num_old_blocks:
            part = old_escapes[old_escape_offsets[block_idx]:old_escape_offsets[block_idx + 1]]
        else:
            continue
        if part.shape[0]:
            parts.append(part)
    return np.concatenate(parts) if parts else np.zeros(0)


@njit(nogil=True, cache=True)
def escape_offsets(
    raw_group: np.ndarray,
    sample_offsets: np.ndarray,
    octet_offsets: np.ndarray,
    code_offsets: np.ndarray,
    has_time: bool,
    out_offsets: np.ndarray,
) -> bool:
    """Computes each block's starting index in the escapes array (length num_blocks + 1).

    Returns False if a block contains `CODE_REPEAT` before any `CODE_ESCAPE` (corrupt).
    """
    num_blocks = sample_offsets.shape[0] - 1
    num_octets = int(octet_offsets[num_blocks])
    code_planes = code_planes_view(raw_group, num_blocks, num_octets, int(code_offsets[num_blocks]), has_time)
    out_offsets[0] = 0
    for block_idx in range(num_blocks):
        count = 0
        if code_offsets[block_idx + 1] > code_offsets[block_idx]:
            seen = False
            block_len = sample_offsets[block_idx + 1] - sample_offsets[block_idx]
            for octet_idx in range(code_offsets[block_idx + 1] - code_offsets[block_idx]):
                plane0_byte = code_planes[0, code_offsets[block_idx] + octet_idx]
                plane1_byte = code_planes[1, code_offsets[block_idx] + octet_idx]
                if not plane0_byte | plane1_byte:
                    continue
                for bit_idx in range(min(8, block_len - 8 * octet_idx)):
                    code = get_code(plane0_byte, plane1_byte, bit_idx)
                    if code == CODE_ESCAPE:
                        count += 1
                        seen = True
                    elif code == CODE_REPEAT and not seen:
                        return False
        out_offsets[block_idx + 1] = out_offsets[block_idx] + count
    return True


@njit(nogil=True, cache=True)
def restore_coded(
    code_planes: np.ndarray, plane_byte_offset: int, escapes: np.ndarray, in_out_samples: np.ndarray
) -> None:
    """Overwrites a flagged block's coded samples: NaN, next escaped value, or previous repeat."""
    num_samples = in_out_samples.shape[0]
    next_escape = 0
    for octet_idx in range((num_samples + 7) // 8):
        plane0_byte = code_planes[0, plane_byte_offset + octet_idx]
        plane1_byte = code_planes[1, plane_byte_offset + octet_idx]
        if not plane0_byte | plane1_byte:
            continue
        for bit_idx in range(min(8, num_samples - 8 * octet_idx)):
            code = get_code(plane0_byte, plane1_byte, bit_idx)
            sample_offset = 8 * octet_idx + bit_idx
            if code == CODE_NAN:
                in_out_samples[sample_offset] = np.nan
            elif code == CODE_ESCAPE:
                in_out_samples[sample_offset] = escapes[next_escape]
                next_escape += 1
            elif code == CODE_REPEAT:
                in_out_samples[sample_offset] = escapes[next_escape - 1]
