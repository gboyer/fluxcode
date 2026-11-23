# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Independent post-hoc verifier for fluxcode's encoding and update guarantees (ENCODER.md §6).

Each public function takes a fluxcode call's return value followed by that call's own arguments
and returns a list of ``Violation`` objects (empty when every guarantee holds):

    result = fluxcode.encode_group(x, params, times=t)
    for violation in fluxcode.verify.encode_group(result, x, params, times=t):
        log.error("fluxcode: %s", violation)

Checks use only the decoder and wire-format metadata columns (never the encoder, except in
``fixed_point``), so an encoder bug cannot mask itself.

Violation ``check`` names:
    - ``"decode"``: compressed output failed to decode.
    - ``"layout"``: time unit, block count, block sizes, or re-encoded block indices mismatch.
    - ``"stats"``: returned ``block_min``, ``block_max``, or ``block_mean`` mismatch.
    - ``"nonfinite"``: NaN/±inf did not round-trip, or a finite sample decoded non-finite.
    - ``"range_bound"``: sample error exceeded ``range / (2^min_quantize_bits - 1/2)`` (or tolerance).
    - ``"fine_bound"``: uncoarsenable block error exceeded ``range / (2^max_quantize_bits - 1/2)``.
    - ``"step_bound"``: sample error exceeded its stored grid's half-step (or escaped sample bound).
    - ``"untouched"``: carried-over block in an update decoded differently.
    - ``"times"``: exact timestamp changed, or decoded timestamps are not non-decreasing.
    - ``"time_error"``: rounded timestamp exceeded max displacement or 5% local-interval outlier cap.
    - ``"time_block"``: timestamp decoded outside its time block ``[start, end]``.
    - ``"fixed_point"``: re-encoding decoded data changed values or diverged on the third encode.
    - ``"decimal"``: guaranteed-exact decimal block changed a finite sample.
"""

from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
from numba import njit

from . import _api, _args, _bitpacking, _format, _group
from ._args import DurationLike, RangesLike, TimeLike
from ._format import BLOCK_FLAG_DECIMAL, BLOCK_FLAG_SCALED
from ._types import (
    DEFAULT_PARAMS,
    EncodedGroup,
    EncodedSeries,
    Params,
    TimeUnit,
    UpdatedGroup,
)

TIME_SLACK: float = 1.02
"""Maximum relative slack above ``time_error * mean_interval`` allowed for rounded timestamps."""

NOISE_FLOOR_MIN_SAMPLES: int = 128
"""Blocks with fewer samples than this never have the noise floor applied."""


class Violation(NamedTuple):
    """A single guarantee violation detected during verification.

    Attributes:
        check: Name of the violated guarantee (see module docstring).
        block: Zero-based block index, or ``-1`` for group-level violations.
        message: Human-readable explanation of the observed mismatch and bound.
    """

    check: str
    block: int
    message: str

    def __str__(self) -> str:
        where = "block group" if self.block < 0 else f"block {self.block}"
        return f"{self.check}: {where}: {self.message}"


# Encoding verifiers


def encode_group(
    result: EncodedGroup,
    samples: npt.ArrayLike,
    params: Params = DEFAULT_PARAMS,
    *,
    block_len: int = _api.DEFAULT_BLOCK_LEN,
    times: npt.ArrayLike | None = None,
    time_unit: TimeUnit | None = None,
    decimals: int | None = None,
) -> list[Violation]:
    """Verifies the result of ``fluxcode.encode_group(samples, params, ...)``.

    Args:
        result: ``EncodedGroup`` returned by ``fluxcode.encode_group``.
        samples: Original input samples passed to ``encode_group``.
        params: Encoder parameters passed to ``encode_group``.
        block_len: Block length passed to ``encode_group``.
        times: Optional timestamps passed to ``encode_group``.
        time_unit: Optional time unit passed to ``encode_group``.
        decimals: Optional decimal places to verify bit-exact decimal preservation.

    Returns:
        List of detected ``Violation`` records (empty if all guarantees hold).
    """
    series = _args.as_series(samples)
    ticks, unit = _args.series_ticks(times, time_unit, series.shape[0])
    return _check_encode(
        result, series, _args.fixed_sizes(series.shape[0], block_len), ticks, unit, params, decimals=decimals
    )


def encode_blocks(
    result: EncodedGroup,
    samples: npt.ArrayLike,
    block_sizes: npt.ArrayLike,
    params: Params = DEFAULT_PARAMS,
    *,
    times: npt.ArrayLike | None = None,
    time_unit: TimeUnit | None = None,
    decimals: int | None = None,
) -> list[Violation]:
    """Verifies the result of ``fluxcode.encode_blocks(samples, block_sizes, params, ...)``.

    Args:
        result: ``EncodedGroup`` returned by ``fluxcode.encode_blocks``.
        samples: Original input samples passed to ``encode_blocks``.
        block_sizes: Per-block sample counts passed to ``encode_blocks``.
        params: Encoder parameters passed to ``encode_blocks``.
        times: Optional timestamps passed to ``encode_blocks``.
        time_unit: Optional time unit passed to ``encode_blocks``.
        decimals: Optional decimal places to verify bit-exact decimal preservation.

    Returns:
        List of detected ``Violation`` records (empty if all guarantees hold).
    """
    series = _args.as_series(samples, allow_empty=True)
    ticks, unit = _args.series_ticks(times, time_unit, series.shape[0])
    return _check_encode(result, series, _args.as_block_sizes(block_sizes), ticks, unit, params, decimals=decimals)


def encode_time_blocks(
    result: EncodedGroup,
    samples: npt.ArrayLike,
    times: npt.ArrayLike,
    params: Params = DEFAULT_PARAMS,
    *,
    start_time: TimeLike,
    block_duration: DurationLike,
    time_unit: TimeUnit | None = None,
    decimals: int | None = None,
) -> list[Violation]:
    """Verifies the result of ``fluxcode.encode_time_blocks(samples, times, params, ...)``.

    Args:
        result: ``EncodedGroup`` returned by ``fluxcode.encode_time_blocks``.
        samples: Original input samples passed to ``encode_time_blocks``.
        times: Original timestamps passed to ``encode_time_blocks``.
        params: Encoder parameters passed to ``encode_time_blocks``.
        start_time: Block 0 anchor time passed to ``encode_time_blocks``.
        block_duration: Fixed block duration passed to ``encode_time_blocks``.
        time_unit: Optional time unit passed to ``encode_time_blocks``.
        decimals: Optional decimal places to verify bit-exact decimal preservation.

    Returns:
        List of detected ``Violation`` records (empty if all guarantees hold).
    """
    series = _args.as_series(samples, allow_empty=True)
    ticks, unit = _args.series_ticks(times, time_unit, series.shape[0])
    assert ticks is not None
    start = _args.to_ticks(start_time, unit, "start_time")
    duration = _args.to_ticks(block_duration, unit, "block_duration", duration=True)
    ids = _time_block_ids(ticks, start, duration)
    sizes = np.bincount(ids, minlength=0).astype(np.int64) if ids.shape[0] else np.zeros(0, np.int64)
    return _check_encode(
        result, series, sizes, ticks, unit, params, time_blocks=(start, duration), decimals=decimals
    )


def encode(
    result: EncodedSeries,
    samples: npt.ArrayLike,
    params: Params = DEFAULT_PARAMS,
    *,
    block_len: int = _api.DEFAULT_BLOCK_LEN,
    blocks_per_group: int = _api.DEFAULT_BLOCKS_PER_GROUP,
    times: npt.ArrayLike | None = None,
    time_unit: TimeUnit | None = None,
    decimals: int | None = None,
) -> list[Violation]:
    """Verifies the result of ``fluxcode.encode(samples, params, ...)`` across all block groups.

    Args:
        result: ``EncodedSeries`` returned by ``fluxcode.encode``.
        samples: Original input samples passed to ``encode``.
        params: Encoder parameters passed to ``encode``.
        block_len: Block length passed to ``encode``.
        blocks_per_group: Blocks per group passed to ``encode``.
        times: Optional timestamps passed to ``encode``.
        time_unit: Optional time unit passed to ``encode``.
        decimals: Optional decimal places to verify bit-exact decimal preservation.

    Returns:
        List of detected ``Violation`` records, with each message prefixed by its group index.
    """
    series = _args.as_series(samples)
    ticks, unit = _args.series_ticks(times, time_unit, series.shape[0])
    per_group = block_len * blocks_per_group
    num_groups = -(-series.shape[0] // per_group)
    found = []
    if len(result.groups) != num_groups:
        return [Violation("layout", -1, f"{len(result.groups)} block groups, not {num_groups}")]
    for group_idx in range(num_groups):
        part = slice(group_idx * per_group, (group_idx + 1) * per_group)
        encoded = EncodedGroup(
            result.groups[group_idx],
            result.block_mins[group_idx],
            result.block_maxs[group_idx],
            result.block_means[group_idx],
        )
        chunk = series[part]
        for violation in _check_encode(
            encoded,
            chunk,
            _args.fixed_sizes(chunk.shape[0], block_len),
            None if ticks is None else ticks[part],
            unit,
            params,
            decimals=decimals,
        ):
            found.append(violation._replace(message=f"block group {group_idx}: {violation.message}"))
    return found


# Update verifiers


def update(
    result: UpdatedGroup,
    group: bytes,
    blocks: Mapping[int, npt.ArrayLike],
    params: Params = DEFAULT_PARAMS,
    *,
    times: Mapping[int, npt.ArrayLike] | None = None,
    time_unit: TimeUnit | None = None,
    decimals: int | None = None,
) -> list[Violation]:
    """Verifies the result of ``fluxcode.update(group, blocks, params, ...)``.

    Args:
        result: ``UpdatedGroup`` returned by ``fluxcode.update``.
        group: Original compressed block group bytes before the update.
        blocks: Replacement/appended block mapping passed to ``update``.
        params: Encoder parameters passed to ``update``.
        times: Optional timestamp mapping passed to ``update``.
        time_unit: Optional time unit passed to ``update``.
        decimals: Optional decimal places to verify bit-exact decimal preservation.

    Returns:
        List of detected ``Violation`` records (empty if all guarantees hold).
    """
    # Reconstruct the expected series by combining decoded untouched blocks from `group` with
    # the new blocks, then verify untouched blocks bit-for-bit and new blocks against bounds.
    old = _decode(group)
    if time_unit is not None:
        _args.check_stored_time_unit(time_unit, old.unit)
    indices, new_values, new_sizes, new_ticks = _args.update_blocks(blocks, times, old.unit)
    num_blocks = max(old.num_blocks, int(indices[-1]) + 1 if indices.shape[0] else 0)
    sizes = np.zeros(num_blocks, np.int64)
    sizes[: old.num_blocks] = old.sizes
    sizes[indices] = new_sizes
    encoded = np.zeros(num_blocks, np.bool_)
    encoded[indices] = True
    encoded[old.num_blocks :] = True
    sources = np.where(encoded, 1, 0)
    values, ticks = _assemble(
        sizes,
        sources,
        [old.values, new_values],
        None if old.ticks is None or new_ticks is None else [old.ticks, new_ticks],
        [_block_positions(old.offsets, ~encoded[: old.num_blocks], old.sizes), np.arange(new_values.shape[0])],
    )
    allowance = None
    if old.ticks is not None and params.time_error > 0:
        allowance = _neighbour_intervals(old, encoded, sizes)
    return _check_result(
        result,
        values,
        ticks,
        sizes,
        encoded,
        old.unit,
        params,
        params.time_error,
        allowance=allowance,
        decimals=decimals,
    )


def update_time_blocks(
    result: UpdatedGroup,
    group: bytes,
    samples: npt.ArrayLike,
    times: npt.ArrayLike,
    params: Params = DEFAULT_PARAMS,
    *,
    start_time: TimeLike,
    block_duration: DurationLike,
    delete_ranges: RangesLike | None = None,
    time_unit: TimeUnit | None = None,
    decimals: int | None = None,
) -> list[Violation]:
    """Verifies the result of ``fluxcode.update_time_blocks(group, samples, times, params, ...)``.

    Args:
        result: ``UpdatedGroup`` returned by ``fluxcode.update_time_blocks``.
        group: Original compressed block group bytes before the update.
        samples: Incoming samples passed to ``update_time_blocks``.
        times: Incoming timestamps passed to ``update_time_blocks``.
        params: Encoder parameters passed to ``update_time_blocks``.
        start_time: Block 0 anchor time.
        block_duration: Fixed block duration.
        delete_ranges: Optional deletion ranges passed to ``update_time_blocks``.
        time_unit: Optional time unit passed to ``update_time_blocks``.
        decimals: Optional decimal places to verify bit-exact decimal preservation.

    Returns:
        List of detected ``Violation`` records (empty if all guarantees hold).
    """
    # Compute the reference post-update series directly by filtering old samples against
    # delete_ranges and incoming timestamps, then stable-sorting survivors with new samples.
    old = _decode(group)
    if old.ticks is None:
        raise ValueError("the block group has no time axis")
    new_values, new_ticks, start, duration, ranges = _args.time_blocks_update(
        samples, times, start_time, block_duration, delete_ranges, time_unit, old.unit
    )
    new_ids = _time_block_ids(new_ticks, start, duration)
    old_ids = np.repeat(np.arange(old.num_blocks), old.sizes)
    deleted = _in_ranges(old.ticks, ranges) | np.isin(old.ticks, new_ticks)
    kept = ~deleted
    num_blocks = max(old.num_blocks, int(new_ids.max()) + 1 if new_ids.shape[0] else 0)
    changed = np.zeros(num_blocks, np.bool_)
    changed[old_ids[deleted]] = True
    changed[new_ids] = True
    changed[old.num_blocks :] = True
    merged_ids = np.concatenate([old_ids[kept], new_ids])
    merged_ticks = np.concatenate([old.ticks[kept], new_ticks])
    order = np.lexsort((merged_ticks, merged_ids))
    merged_values = np.concatenate([old.values[kept], new_values])[order]
    sizes = np.bincount(merged_ids, minlength=num_blocks).astype(np.int64)
    met = _met_blocks(ranges, start, duration, old.num_blocks, num_blocks)
    # Re-encoded blocks must cover all changed blocks and stay within changed | met.
    reported = np.asarray(result.indices, np.int64)
    if reported.shape[0] and (reported.min() < 0 or reported.max() >= num_blocks):
        return [Violation("layout", -1, f"re-encoded indices {_summary(reported)} outside the {num_blocks} blocks")]
    encoded = np.zeros(num_blocks, np.bool_)
    encoded[reported] = True
    if not (changed <= encoded).all() or not (encoded <= changed | met).all():
        return [
            Violation(
                "layout",
                -1,
                f"re-encoded blocks {_summary(reported)}: must include "
                f"{_summary(np.flatnonzero(changed))} and lie within {_summary(np.flatnonzero(changed | met))}",
            )
        ]
    return _check_result(
        result,
        merged_values,
        merged_ticks[order],
        sizes,
        encoded,
        old.unit,
        params,
        0.0,
        time_blocks=(start, duration),
        decimals=decimals,
    )


def replace_time_blocks(
    result: UpdatedGroup,
    group: bytes,
    blocks: Iterable[int],
    samples: npt.ArrayLike,
    times: npt.ArrayLike,
    params: Params = DEFAULT_PARAMS,
    *,
    start_time: TimeLike,
    block_duration: DurationLike,
    time_unit: TimeUnit | None = None,
    decimals: int | None = None,
) -> list[Violation]:
    """Verifies the result of ``fluxcode.replace_time_blocks(group, blocks, samples, times, params, ...)``.

    Args:
        result: ``UpdatedGroup`` returned by ``fluxcode.replace_time_blocks``.
        group: Original compressed block group bytes before replacement.
        blocks: Declared block indices passed to ``replace_time_blocks``.
        samples: Replacement samples passed to ``replace_time_blocks``.
        times: Replacement timestamps passed to ``replace_time_blocks``.
        params: Encoder parameters passed to ``replace_time_blocks``.
        start_time: Block 0 anchor time.
        block_duration: Fixed block duration.
        time_unit: Optional time unit passed to ``replace_time_blocks``.
        decimals: Optional decimal places to verify bit-exact decimal preservation.

    Returns:
        List of detected ``Violation`` records (empty if all guarantees hold).
    """
    old = _decode(group)
    if old.ticks is None:
        raise ValueError("the block group has no time axis")
    declared = _args.block_list(blocks)
    new_values, new_ticks, start, duration, _ = _args.time_blocks_update(
        samples, times, start_time, block_duration, None, time_unit, old.unit
    )
    new_ids = _time_block_ids(new_ticks, start, duration)
    if not declared.shape[0]:
        if result.group != group or np.asarray(result.indices).shape[0]:
            return [Violation("layout", -1, "no blocks declared, but the block group changed")]
        return []
    num_blocks = max(old.num_blocks, int(declared[-1]) + 1)
    encoded = np.zeros(num_blocks, np.bool_)
    encoded[declared] = True
    encoded[old.num_blocks :] = True
    sizes = np.zeros(num_blocks, np.int64)
    sizes[: old.num_blocks] = old.sizes
    sizes[encoded] = 0
    sizes += np.bincount(new_ids, minlength=num_blocks)[:num_blocks]
    sources = np.where(encoded, 1, 0)
    values, ticks = _assemble(
        sizes,
        sources,
        [old.values, new_values],
        [old.ticks, new_ticks],
        [_block_positions(old.offsets, ~encoded[: old.num_blocks], old.sizes), np.arange(new_values.shape[0])],
    )
    return _check_result(
        result,
        values,
        ticks,
        sizes,
        encoded,
        old.unit,
        params,
        params.time_error,
        time_blocks=(start, duration),
        decimals=decimals,
    )


def fixed_point(group: bytes, params: Params = DEFAULT_PARAMS) -> list[Violation]:
    """Verifies that re-encoding a decoded block group is a fixed point (ENCODER.md §6).

    Skipped (returns empty) when ``params.target_bits_per_sample`` is set.

    Args:
        group: Compressed block group bytes to test.
        params: Encoder parameters used to produce ``group``.

    Returns:
        List of detected ``Violation`` records (empty if values and third-encode bytes match).
    """
    # Decode once, re-encode with exact times, verify decoded values are unchanged, then encode a
    # third time and verify identical compressed bytes.
    if params.target_bits_per_sample is not None:
        return []
    exact = replace(params, time_error=0.0)
    first = _api.decode_group(group)
    second = _api.encode_blocks(first.values, first.block_sizes, exact, times=first.times).group
    again = _api.decode_group(second).values
    if not _same_doubles(again, first.values):
        moved = np.flatnonzero(again.view(np.int64) != first.values.view(np.int64))
        block = int(np.searchsorted(np.cumsum(first.block_sizes), moved[0], side="right"))
        return [
            Violation(
                "fixed_point",
                block,
                f"{moved.shape[0]} decoded samples changed on re-encode, "
                f"first sample {int(moved[0])}: {first.values[moved[0]]:.17g} -> {again[moved[0]]:.17g}",
            )
        ]
    third = _api.encode_blocks(again, first.block_sizes, exact, times=first.times).group
    if third != second:
        return [Violation("fixed_point", -1, "the third encode's bytes differ from the second's")]
    return []


# Decoding and reference-series assembly


class _Decoded(NamedTuple):
    """Decoded block group paired with its wire-format grid metadata and escape masks."""

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
        return self.sizes.shape[0]


def _decode(group: bytes) -> _Decoded:
    """Decompresses and decodes ``group``, extracting per-block grid metadata and escaped sample masks."""
    parsed = _group.decompress(group)
    num_blocks = parsed.header.num_blocks
    values, ticks = _group.decode_blocks(parsed, np.arange(num_blocks, dtype=np.int64))
    raw = parsed.body.astype(np.int64)
    # Unpack byte-planed int16 grid_exponents and sign-extend bit 15.
    start = _format.grid_exponents_start(num_blocks)
    grid_exponents = raw[start : start + num_blocks] | (raw[start + num_blocks : start + 2 * num_blocks] << 8)
    grid_exponents -= (grid_exponents & 0x8000) << 1
    # Unpack byte-planed uint32 grid_scales.
    start = _format.grid_scales_start(num_blocks)
    scales = np.zeros(num_blocks, np.int64)
    for byte_idx in range(4):
        scales |= raw[start + byte_idx * num_blocks : start + (byte_idx + 1) * num_blocks] << (8 * byte_idx)
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
            # Plane 1 bit is 1 iff sample code is CODE_ESCAPE (10b) or CODE_REPEAT_ESCAPE (11b).
            c0, c1 = int(layout.code_plane_offsets[block_idx]), int(layout.code_plane_offsets[block_idx + 1])
            s0, s1 = int(offsets[block_idx]), int(offsets[block_idx + 1])
            escaped_samples[s0:s1] = np.unpackbits(code_planes[1, c0:c1], bitorder="little")[: s1 - s0].astype(np.bool_)
    return _Decoded(
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


def _decode_or_report(group: bytes, found: list[Violation]) -> _Decoded | None:
    """Decodes ``group``, appending a ``"decode"`` violation and returning None on any error."""
    try:
        return _decode(group)
    except Exception as exc:  # noqa: BLE001 (any failure to decode is the finding)
        found.append(Violation("decode", -1, f"{type(exc).__name__}: {exc}"))
        return None


def _block_positions(offsets: np.ndarray, which: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """Returns flat sample indices of the blocks selected by boolean mask ``which``, in block order."""
    lengths = np.where(which, sizes, 0)
    return np.arange(int(lengths.sum())) + np.repeat(offsets[:-1] - (np.cumsum(lengths) - lengths), lengths)


def _assemble(
    sizes: np.ndarray,
    sources: np.ndarray,
    values: list[np.ndarray],
    ticks: list[np.ndarray] | None,
    positions: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray | None]:
    """Interleaves blocks from multiple source arrays according to per-block ``sources`` indices."""
    out_values = np.empty(int(sizes.sum()))
    out_ticks = None if ticks is None else np.empty(out_values.shape[0], np.int64)
    sample_sources = np.repeat(sources, sizes)
    for source_idx, source_positions in enumerate(positions):
        target = sample_sources == source_idx
        out_values[target] = values[source_idx][source_positions]
        if out_ticks is not None and ticks is not None:
            out_ticks[target] = ticks[source_idx][source_positions]
    return out_values, out_ticks


def _time_block_ids(ticks: np.ndarray, start: int, duration: int) -> np.ndarray:
    """Computes ``floor((tick - start) / duration)`` in unsigned 64-bit arithmetic."""
    if ticks.shape[0] and int(ticks.min()) < start:
        raise ValueError("times must be at or after start_time")
    return ((ticks.view(np.uint64) - np.uint64(start % 2**64)) // np.uint64(duration)).astype(np.int64)


def _in_ranges(ticks: np.ndarray, ranges: np.ndarray) -> np.ndarray:
    """Returns a boolean mask of ticks contained in any sorted, disjoint ``[start, end)`` range."""
    if not ranges.shape[0]:
        return np.zeros(ticks.shape[0], np.bool_)
    idx = np.searchsorted(ranges[:, 0], ticks, side="right") - 1
    return (idx >= 0) & (ticks < ranges[np.maximum(idx, 0), 1])


def _met_blocks(ranges: np.ndarray, start: int, duration: int, num_old: int, num_blocks: int) -> np.ndarray:
    """Returns a boolean mask of existing blocks overlapped by any deletion range."""
    met = np.zeros(num_blocks, np.bool_)
    for range_start, range_end in ranges.tolist():
        if range_end <= start:
            continue
        first = max(0, (range_start - start) // duration)
        last = min(num_old - 1, (range_end - 1 - start) // duration)
        met[first : last + 1] = True
    return met


def _neighbour_intervals(old: _Decoded, encoded: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """Computes per-block maximum mean interval of adjacent carried-over stored blocks."""
    assert old.ticks is not None
    num_blocks = sizes.shape[0]
    means = np.zeros(num_blocks)
    for block_idx in range(old.num_blocks):
        if not encoded[block_idx] and old.sizes[block_idx] > 1:
            first, last = int(old.ticks[old.offsets[block_idx]]), int(old.ticks[old.offsets[block_idx + 1] - 1])
            means[block_idx] = (last - first) / (int(old.sizes[block_idx]) - 1)
    allowance = np.zeros(num_blocks)
    stored = np.flatnonzero(~encoded & (sizes > 0))
    for block_idx in np.flatnonzero(encoded).tolist():
        before = stored[stored < block_idx]
        after = stored[stored > block_idx]
        for neighbour in ([before[-1]] if before.shape[0] else []) + ([after[0]] if after.shape[0] else []):
            allowance[block_idx] = max(allowance[block_idx], means[neighbour])
    return allowance


# Core guarantee checks


def _check_encode(
    result: EncodedGroup,
    series: np.ndarray,
    sizes: np.ndarray,
    ticks: np.ndarray | None,
    unit: int,
    params: Params,
    time_blocks: tuple[int, int] | None = None,
    decimals: int | None = None,
) -> list[Violation]:
    """Verifies a fresh encode where every block was encoded from ``series``."""
    return _check_result(
        result,
        series,
        ticks,
        sizes,
        np.ones(sizes.shape[0], np.bool_),
        unit,
        params,
        params.time_error,
        time_blocks=time_blocks,
        decimals=decimals,
    )


def _check_result(
    result: EncodedGroup | UpdatedGroup,
    values: np.ndarray,
    ticks: np.ndarray | None,
    sizes: np.ndarray,
    encoded: np.ndarray,
    unit: int,
    params: Params,
    time_error: float,
    *,
    allowance: np.ndarray | None = None,
    time_blocks: tuple[int, int] | None = None,
    decimals: int | None = None,
) -> list[Violation]:
    """Decodes ``result.group`` and verifies layout, untouched blocks, value bounds, decimals, and times."""
    found: list[Violation] = []
    new = _decode_or_report(result.group, found)
    if new is None:
        return found
    if new.unit != unit:
        return [Violation("layout", -1, f"time unit code {new.unit}, not {unit}")]
    if new.num_blocks != sizes.shape[0] or not np.array_equal(new.sizes, sizes):
        return [Violation("layout", -1, f"block sizes {_summary(new.sizes)}, not {_summary(sizes)}")]
    if isinstance(result, UpdatedGroup) and not np.array_equal(result.indices, np.flatnonzero(encoded)):
        return [
            Violation(
                "layout",
                -1,
                f"re-encoded indices {_summary(result.indices)}, not {_summary(np.flatnonzero(encoded))}",
            )
        ]
    if not encoded.all():
        changed = ~np.repeat(encoded, sizes) & ~_same_doubles_mask(new.values, values)
        _report_samples(found, "untouched", changed, new.offsets, "samples of a block not re-encoded changed")
    _check_values(found, values, new, sizes, encoded, params, result)
    if decimals is not None:
        _check_decimals(found, values, new, sizes, encoded, params, decimals)
    if ticks is not None and new.ticks is not None:
        _check_times(found, ticks, new.ticks, sizes, new.offsets, encoded, time_error, allowance, time_blocks)
    return found


def _check_values(
    found: list[Violation],
    given: np.ndarray,
    new: _Decoded,
    sizes: np.ndarray,
    encoded: np.ndarray,
    params: Params,
    result: EncodedGroup | UpdatedGroup,
) -> None:
    """Checks non-finite preservation, range/fine/step error bounds, and per-block statistics."""
    # Evaluate all errors and ranges at half scale (x / 2) so blocks spanning [-DBL_MAX, +DBL_MAX]
    # never overflow during subtraction.
    half_error, lo, hi, magnitude, count, mean, nonfinite_bad, nonfinite_first = _value_stats(
        given, new.values, new.offsets
    )
    _report_blocks(
        found, "nonfinite", encoded, nonfinite_bad, nonfinite_first, "non-finite samples didn't decode as themselves"
    )
    with np.errstate(all="ignore"):
        half_range = np.where(count > 0, hi / 2 - lo / 2, 0.0)
        # Allow 4 ULPs (2 at half scale) for quantization + reconstruction rounding.
        half_ulps = 2 * np.spacing(magnitude)
        grid_half = _grid_bounds(new) / 2
        bulk_error, escaped_error = _split_errors(given, new, half_error)
        bounds = [
            (
                "range_bound",
                np.maximum(half_range / (2.0**params.min_quantize_bits - 0.5), params.absolute_tolerance / 2),
                "range / (2^min_quantize_bits - 1/2), or absolute_tolerance",
                half_error,
            ),
            ("step_bound", grid_half, "the stored grid's", bulk_error),
            (
                "step_bound",
                np.maximum(grid_half, half_range / (2.0**params.max_quantize_bits - 1) * 2),
                "an escaped sample's: the full range's at max_quantize_bits",
                escaped_error,
            ),
        ]
        if params.target_bits_per_sample is None:
            fine = (params.noise_floor_sigma == 0) | (sizes < NOISE_FLOOR_MIN_SAMPLES)
            fine_bound = np.where(
                fine,
                np.maximum(half_range / (2.0**params.max_quantize_bits - 0.5), params.absolute_tolerance / 2),
                np.inf,
            )
            bounds.append(
                ("fine_bound", fine_bound, "range / (2^max_quantize_bits - 1/2), or absolute_tolerance", half_error)
            )
        for check, half_bound, what, block_error in bounds:
            over = encoded & (count > 0) & ~(block_error <= half_bound + half_ulps)
            for block_idx in np.flatnonzero(over)[:_MAX_REPORTED].tolist():
                found.append(
                    Violation(
                        check,
                        block_idx,
                        f"max error {2 * block_error[block_idx]:.17g} over {what} {2 * half_bound[block_idx]:.17g}",
                    )
                )
    _check_stats(found, result, encoded, count, lo, hi, mean, magnitude)


def _check_decimals(
    found: list[Violation],
    given: np.ndarray,
    new: _Decoded,
    sizes: np.ndarray,
    encoded: np.ndarray,
    params: Params,
    decimals: int,
) -> None:
    """Checks bit-exact decimal preservation on blocks where ENCODER.md §6 guarantees it."""
    # A block is guaranteed bit-exact when all finite samples are exact k / 10^decimals with
    # |k| < 2^52 spanning < 2^(max_quantize_bits - 1) steps, decimal grids are enabled,
    # absolute_tolerance does not exceed half a decimal step, and neither target nor noise floor applies.
    if not -22 <= decimals <= 22:
        raise ValueError(f"decimals must be from -22 to 22, got {decimals}")
    if params.target_bits_per_sample is not None or params.grids == "binary" or not given.shape[0]:
        return
    if params.absolute_tolerance > 10.0**-decimals / 2:
        return
    finite = np.isfinite(given)
    with np.errstate(all="ignore"):
        k = np.round(given * 10.0**decimals) if decimals >= 0 else np.round(given / 10.0**-decimals)
        nearest = k / 10.0**decimals if decimals >= 0 else k * 10.0**-decimals
    off = finite & ~((nearest == given) & (np.abs(k) < 2.0**52))
    block_ids = np.repeat(np.arange(sizes.shape[0]), sizes)
    k = np.where(finite, k, np.nan)
    k_lo = np.full(sizes.shape[0], np.inf)
    k_hi = np.full(sizes.shape[0], -np.inf)
    np.fmin.at(k_lo, block_ids, k)
    np.fmax.at(k_hi, block_ids, k)
    checked = (
        encoded
        & (np.bincount(block_ids, off, sizes.shape[0]) == 0)
        & (k_hi - k_lo < 2.0 ** (params.max_quantize_bits - 1))
        & ((params.noise_floor_sigma == 0) | (sizes < NOISE_FLOOR_MIN_SAMPLES))
    )
    bad = np.repeat(checked, sizes) & finite & (new.values != given)
    _report_samples(found, "decimal", bad, new.offsets, "decimal samples of a block guaranteed exact changed")


def _split_errors(given: np.ndarray, new: _Decoded, half_error: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Splits each block's half-scale max error into bulk-sample error and escaped-sample error."""
    bulk = half_error.copy()
    escaped = np.zeros_like(half_error)
    for block_idx in np.flatnonzero(new.escaped).tolist():
        first, end = int(new.offsets[block_idx]), int(new.offsets[block_idx + 1])
        with np.errstate(all="ignore"):
            error = np.abs(given[first:end] / 2 - new.values[first:end] / 2)
        error = np.where(np.isfinite(given[first:end]) & np.isfinite(new.values[first:end]), error, 0.0)
        which = new.escaped_samples[first:end]
        bulk[block_idx] = error[~which].max(initial=0.0)
        escaped[block_idx] = error[which].max(initial=0.0)
    return bulk, escaped


@njit(nogil=True, cache=True)
def _value_stats(given: np.ndarray, got: np.ndarray, offsets: np.ndarray) -> tuple:
    """Computes per-block half-scale max error, finite min/max/magnitude/count/mean, and non-finite mismatches."""
    # Run a fast branch-free pass first; fall back to _careful only on blocks containing non-finite
    # values or overflowing sums.
    num_blocks = offsets.shape[0] - 1
    half_error = np.zeros(num_blocks)
    lo = np.full(num_blocks, np.inf)
    hi = np.full(num_blocks, -np.inf)
    magnitude = np.zeros(num_blocks)
    count = np.zeros(num_blocks, np.int64)
    mean = np.zeros(num_blocks)
    bad = np.zeros(num_blocks, np.int64)
    first_bad = np.full(num_blocks, -1, np.int64)
    for block_idx in range(num_blocks):
        first, end = offsets[block_idx], offsets[block_idx + 1]
        if first == end:
            continue
        _all_finite(given, got, first, end, block_idx, half_error, lo, hi, mean)
        if np.isfinite(half_error[block_idx]) and np.isfinite(mean[block_idx]):
            count[block_idx] = end - first
        else:
            _careful(given, got, first, end, block_idx, half_error, lo, hi, count, mean, bad, first_bad)
        if count[block_idx]:
            magnitude[block_idx] = max(-lo[block_idx], hi[block_idx])
    return half_error, lo, hi, magnitude, count, mean, bad, first_bad


@njit(nogil=True, cache=True, fastmath={"reassoc", "nsz"})
def _all_finite(given, got, first, end, block_idx, half_error, lo, hi, mean):
    """Fast branch-free pass assuming all samples are finite; poisons ``half_error`` with NaN if not."""
    error, low, high, total, poison = 0.0, np.inf, -np.inf, 0.0, 0.0
    for sample_idx in range(first, end):
        x, y = given[sample_idx], got[sample_idx]
        error = max(error, abs(0.5 * y - 0.5 * x))
        low = min(low, x)
        high = max(high, x)
        total += x
        poison += (x - x) + (y - y)
    half_error[block_idx] = error + poison
    lo[block_idx], hi[block_idx] = low, high
    mean[block_idx] = total / (end - first)


@njit(nogil=True, cache=True)
def _careful(given, got, first, end, block_idx, half_error, lo, hi, count, mean, bad, first_bad):
    """Fallback statistics pass handling non-finite samples and overflowing finite sums."""
    half_error[block_idx], lo[block_idx], hi[block_idx], count[block_idx] = 0.0, np.inf, -np.inf, 0
    total = 0.0
    for sample_idx in range(first, end):
        x, y = given[sample_idx], got[sample_idx]
        if np.isfinite(x):
            if np.isfinite(y):
                half_error[block_idx] = max(half_error[block_idx], abs(0.5 * y - 0.5 * x))
            else:
                bad[block_idx] += 1
                if first_bad[block_idx] < 0:
                    first_bad[block_idx] = sample_idx
            lo[block_idx] = min(lo[block_idx], x)
            hi[block_idx] = max(hi[block_idx], x)
            count[block_idx] += 1
            total += x
        elif not (x == y or (np.isnan(x) and np.isnan(y))):
            bad[block_idx] += 1
            if first_bad[block_idx] < 0:
                first_bad[block_idx] = sample_idx
    mean[block_idx] = total / max(count[block_idx], 1)
    if not np.isfinite(mean[block_idx]):
        # Avoid overflow by dividing each finite sample before summing.
        mean[block_idx] = 0.0
        for sample_idx in range(first, end):
            if np.isfinite(given[sample_idx]):
                mean[block_idx] += given[sample_idx] / count[block_idx]


def _check_stats(
    found: list[Violation],
    result: EncodedGroup | UpdatedGroup,
    encoded: np.ndarray,
    count: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    mean: np.ndarray,
    magnitude: np.ndarray,
) -> None:
    """Checks returned ``block_min``, ``block_max``, and ``block_mean`` for all encoded blocks."""
    blocks = np.flatnonzero(encoded)
    stats = [np.asarray(result.block_min), np.asarray(result.block_max), np.asarray(result.block_mean)]
    if any(stat.shape != blocks.shape for stat in stats):
        found.append(Violation("stats", -1, f"{[stat.shape for stat in stats]} statistics for {blocks.shape[0]} blocks"))
        return
    some = count[blocks] > 0
    expected = [np.where(some, lo[blocks], np.nan), np.where(some, hi[blocks], np.nan)]
    with np.errstate(invalid="ignore"):
        # Fastmath summation may reorder terms, causing up to 2 * count ULPs of rounding error.
        mean_tolerance = 2.0 * count[blocks] * np.spacing(magnitude[blocks])
        mean_ok = np.where(some, np.abs(stats[2] - mean[blocks]) <= mean_tolerance, np.isnan(stats[2]))
    for name, stat, want in zip(("block_min", "block_max"), stats, expected):
        for pos in np.flatnonzero(~((stat == want) | (np.isnan(stat) & np.isnan(want))))[:_MAX_REPORTED].tolist():
            found.append(Violation("stats", int(blocks[pos]), f"{name} {stat[pos]:.17g}, not {want[pos]:.17g}"))
    for pos in np.flatnonzero(~mean_ok)[:_MAX_REPORTED].tolist():
        want_mean = mean[blocks[pos]] if some[pos] else np.nan
        found.append(Violation("stats", int(blocks[pos]), f"block_mean {stats[2][pos]:.17g}, not {want_mean:.17g}"))


def _grid_bounds(new: _Decoded) -> np.ndarray:
    """Computes each block's maximum allowed sample error implied by its stored grid metadata."""
    # Power-of-two and scaled grids allow half a grid step; unscaled decimal grids are only
    # accepted within a quarter step of the planned binary step (which is finer than 10^p).
    decimal = (new.flags & BLOCK_FLAG_DECIMAL) != 0
    scaled = (new.flags & BLOCK_FLAG_SCALED) != 0
    with np.errstate(over="ignore"):
        base = np.where(
            decimal, 10.0 ** np.clip(new.exponents, -22, 22), np.ldexp(1.0, np.clip(new.exponents, -1074, 1023))
        )
    numerators, denominators = new.scales & 0xFFFF, new.scales >> 16
    step = np.where(scaled, base * numerators / np.maximum(denominators, 1), base)
    return np.where(decimal & ~scaled, step / 4, step / 2)


def _check_times(
    found: list[Violation],
    given: np.ndarray,
    got: np.ndarray,
    sizes: np.ndarray,
    offsets: np.ndarray,
    encoded: np.ndarray,
    time_error: float,
    allowance: np.ndarray | None,
    time_blocks: tuple[int, int] | None,
) -> None:
    """Checks monotonicity, exact-time fidelity, ``time_error`` bounds, and time-block membership."""
    steps = np.diff(got)
    if (steps < 0).any():
        first = int(np.flatnonzero(steps < 0)[0]) + 1
        found.append(Violation("times", _block_of(offsets, first), f"time of sample {first} decreases"))
    rounded, wrong, first_wrong, largest, far, mean_interval = _time_stats(given, got, offsets, encoded, time_error)
    _report_blocks(found, "times", ~rounded, wrong, first_wrong, "times stored exactly decoded differently")
    if allowance is not None:
        mean_interval = np.maximum(mean_interval, allowance)
    cap = time_error * TIME_SLACK * mean_interval * (1 + 1e-12) + 1e-6
    for block_idx in np.flatnonzero(rounded & (largest > cap))[:_MAX_REPORTED].tolist():
        found.append(
            Violation(
                "time_error",
                block_idx,
                f"a time moved {largest[block_idx]:g} ticks, over "
                f"{TIME_SLACK} x {time_error} x the mean interval {mean_interval[block_idx]:g}",
            )
        )
    for block_idx in np.flatnonzero(rounded & (far > sizes // 20))[:_MAX_REPORTED].tolist():
        found.append(
            Violation(
                "time_error",
                block_idx,
                f"{far[block_idx]} of {sizes[block_idx]} times moved more "
                f"than {time_error} x their own interval (at most 1 in 20)",
            )
        )
    if time_blocks is not None:
        start, duration = time_blocks
        block = np.repeat(np.arange(sizes.shape[0]), sizes)
        offset = got.view(np.uint64) - np.uint64(start % 2**64)
        ids = (offset // np.uint64(duration)).astype(np.int64)
        at_end = (ids == block + 1) & (offset % np.uint64(duration) == 0)
        _report_samples(
            found,
            "time_block",
            (got < start) | ~((ids == block) | at_end),
            offsets,
            "times decoded outside their time block",
        )


@njit(nogil=True, cache=True)
def _time_stats(
    given: np.ndarray, got: np.ndarray, offsets: np.ndarray, encoded: np.ndarray, time_error: float
) -> tuple:
    """Computes per-block timestamp rounding status, exact mismatches, max displacement, and outlier counts."""
    # A block is eligible for rounding iff it was encoded with time_error > 0, has >= 3 ticks,
    # and was not already uniform. Unsigned subtraction keeps all non-decreasing gaps exact.
    num_blocks = offsets.shape[0] - 1
    rounded = np.zeros(num_blocks, np.bool_)
    wrong = np.zeros(num_blocks, np.int64)
    first_wrong = np.full(num_blocks, -1, np.int64)
    largest = np.zeros(num_blocks)
    far = np.zeros(num_blocks, np.int64)
    mean_interval = np.zeros(num_blocks)
    for block_idx in range(num_blocks):
        first, end = offsets[block_idx], offsets[block_idx + 1]
        if encoded[block_idx] and time_error > 0 and end - first >= 3:
            gap = np.uint64(given[first + 1] - given[first])
            for sample_idx in range(first + 2, end):
                if np.uint64(given[sample_idx] - given[sample_idx - 1]) != gap:
                    rounded[block_idx] = True
                    break
        if not rounded[block_idx]:
            for sample_idx in range(first, end):
                if got[sample_idx] != given[sample_idx]:
                    wrong[block_idx] += 1
                    if first_wrong[block_idx] < 0:
                        first_wrong[block_idx] = sample_idx
            continue
        mean_interval[block_idx] = float(np.uint64(given[end - 1] - given[first])) / (end - first - 1)
        for sample_idx in range(first, end):
            move = abs(float(got[sample_idx] - given[sample_idx]))
            largest[block_idx] = max(largest[block_idx], move)
            own = np.inf
            if sample_idx > first:
                own = float(np.uint64(given[sample_idx] - given[sample_idx - 1]))
            if sample_idx + 1 < end:
                own = min(own, float(np.uint64(given[sample_idx + 1] - given[sample_idx])))
            if move > time_error * own:
                far[block_idx] += 1
    return rounded, wrong, first_wrong, largest, far, mean_interval


# Formatting and reporting helpers

_MAX_REPORTED = 5
"""Maximum number of failing blocks reported per check."""


def _same_doubles_mask(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise equality treating NaN == NaN and -0.0 == 0.0."""
    with np.errstate(invalid="ignore"):
        return (a == b) | (np.isnan(a) & np.isnan(b))


def _same_doubles(a: np.ndarray, b: np.ndarray) -> bool:
    """Returns True if ``a`` and ``b`` have the same shape and match element-wise (NaN == NaN)."""
    return a.shape == b.shape and bool(_same_doubles_mask(a, b).all())


def _block_of(offsets: np.ndarray, sample_idx: int) -> int:
    """Returns the block index containing flat sample index ``sample_idx``."""
    return int(np.searchsorted(offsets, sample_idx, side="right")) - 1


def _report_blocks(
    found: list[Violation], check: str, which: np.ndarray, counts: np.ndarray, firsts: np.ndarray, what: str
) -> None:
    """Appends up to ``_MAX_REPORTED`` block-level violations from precomputed per-block counts."""
    for block_idx in np.flatnonzero(which & (counts > 0))[:_MAX_REPORTED].tolist():
        found.append(Violation(check, block_idx, f"{what}: {counts[block_idx]}, first sample {firsts[block_idx]}"))


def _report_samples(found: list[Violation], check: str, bad: np.ndarray, offsets: np.ndarray, what: str) -> None:
    """Appends up to ``_MAX_REPORTED`` block-level violations from a per-sample boolean failure mask."""
    if not bad.any():
        return
    samples = np.flatnonzero(bad)
    blocks = np.searchsorted(offsets, samples, side="right") - 1
    unique_blocks, first = np.unique(blocks, return_index=True)
    counts = np.bincount(blocks - blocks.min())[unique_blocks - blocks.min()]
    for block_idx, sample_idx, count in list(zip(unique_blocks.tolist(), samples[first].tolist(), counts.tolist()))[
        :_MAX_REPORTED
    ]:
        found.append(Violation(check, block_idx, f"{what}: {count}, first sample {sample_idx}"))


def _summary(array: np.ndarray, limit: int = 8) -> str:
    """Formats an array as a compact list string, truncating past ``limit`` entries."""
    items = np.asarray(array).tolist()
    return str(items) if len(items) <= limit else f"{items[:limit]}... ({len(items)})"
