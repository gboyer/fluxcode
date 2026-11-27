# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Independent post-hoc verifier for fluxcode's encoding and update guarantees
(ENCODER.md §6).

Each public function takes a fluxcode call's return value followed by that call's own
arguments and returns a list of ``Violation`` objects (empty when every guarantee
holds):

    result = fluxcode.encode_group(x, params, times=t)
    for violation in fluxcode.verify.encode_group(result, x, params, times=t):
        log.error("fluxcode: %s", violation)

Checks use only the decoder and wire-format metadata columns (never the encoder, except
in ``fixed_point``), so an encoder bug cannot mask itself.

Violation ``check`` names:
    - ``"decode"``: compressed output failed to decode.
    - ``"layout"``: time unit, block count, block sizes, or re-encoded block indices
      mismatch.
    - ``"stats"``: returned ``block_min``, ``block_max``, or ``block_mean`` mismatch.
    - ``"nonfinite"``: NaN/±inf did not round-trip, or a finite sample decoded
      non-finite.
    - ``"range_bound"``: sample error exceeded ``range / (2^min_quantize_bits - 1/2)``
      (or tolerance).
    - ``"fine_bound"``: uncoarsenable block error exceeded ``range /
      (2^max_quantize_bits - 1/2)``.
    - ``"step_bound"``: sample error exceeded its stored grid's half-step (or escaped
      sample bound).
    - ``"untouched"``: carried-over block in an update decoded differently.
    - ``"times"``: exact timestamp changed, or decoded timestamps are not
      non-decreasing.
    - ``"time_error"``: rounded timestamp exceeded max displacement or 5% local-interval
      outlier cap.
    - ``"time_block"``: timestamp decoded outside its time block ``[start, end]``.
    - ``"fixed_point"``: re-encoding decoded data changed values or diverged on the
      third encode.
    - ``"decimal"``: guaranteed-exact decimal block changed a finite sample.
"""

from collections.abc import Iterable, Mapping
from dataclasses import replace

import numpy as np
import numpy.typing as npt

from . import _api, _args, _verify_common, _verify_decode, _verify_times, _verify_values
from ._args import DurationLike, RangesLike, TimeLike
from ._types import (
    DEFAULT_PARAMS,
    EncodedGroup,
    EncodedSeries,
    Params,
    TimeUnit,
    UpdatedGroup,
)
from ._verify_common import NOISE_FLOOR_MIN_SAMPLES, TIME_SLACK, Violation

__all__ = [
    "NOISE_FLOOR_MIN_SAMPLES",
    "TIME_SLACK",
    "Violation",
    "encode",
    "encode_blocks",
    "encode_group",
    "encode_time_blocks",
    "fixed_point",
    "replace_time_blocks",
    "update",
    "update_time_blocks",
]


# ======================================================================================
# Encoding verifiers
# ======================================================================================


def encode_group(
    result: EncodedGroup,
    samples: npt.ArrayLike,
    params: Params = DEFAULT_PARAMS,
    *,
    block_len: int,
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
        result,
        series,
        _args.fixed_sizes(series.shape[0], block_len),
        ticks,
        unit,
        params,
        decimals=decimals,
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
    """Verifies the result of ``fluxcode.encode_blocks(samples, block_sizes, params,
    ...)``.

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
    return _check_encode(
        result,
        series,
        _args.as_block_sizes(block_sizes),
        ticks,
        unit,
        params,
        decimals=decimals,
    )


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
    """Verifies the result of ``fluxcode.encode_time_blocks(samples, times, params,
    ...)``.

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
    ids = _verify_times.time_block_ids(ticks, start, duration)
    sizes = (
        np.bincount(ids, minlength=0).astype(np.int64)
        if ids.shape[0]
        else np.zeros(0, np.int64)
    )
    return _check_encode(
        result,
        series,
        sizes,
        ticks,
        unit,
        params,
        time_blocks=(start, duration),
        decimals=decimals,
    )


def encode(
    result: EncodedSeries,
    samples: npt.ArrayLike,
    params: Params = DEFAULT_PARAMS,
    *,
    block_len: int,
    blocks_per_group: int,
    times: npt.ArrayLike | None = None,
    time_unit: TimeUnit | None = None,
    decimals: int | None = None,
) -> list[Violation]:
    """Verifies the result of ``fluxcode.encode(samples, params, ...)`` across all block
    groups.

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
        List of detected ``Violation`` records, with each message prefixed by its group
        index.
    """
    series = _args.as_series(samples)
    ticks, unit = _args.series_ticks(times, time_unit, series.shape[0])
    per_group = block_len * blocks_per_group
    num_groups = -(-series.shape[0] // per_group)
    found = []
    if len(result.groups) != num_groups:
        return [
            Violation(
                "layout", -1, f"{len(result.groups)} block groups, not {num_groups}"
            )
        ]
    for group_idx, encoded in enumerate(result.groups):
        part = slice(group_idx * per_group, (group_idx + 1) * per_group)
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
            found.append(
                violation._replace(
                    message=f"block group {group_idx}: {violation.message}"
                )
            )
    return found


# ======================================================================================
# Update verifiers
# ======================================================================================


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
    # Reconstruct the expected series by combining decoded untouched blocks from `group`
    # with the new blocks, then verify untouched blocks bit-for-bit and new blocks
    # against bounds.
    old = _verify_decode.decode(group)
    if time_unit is not None:
        _args.check_stored_time_unit(time_unit, old.unit)
    indices, new_values, new_sizes, new_ticks = _args.update_blocks(
        blocks, times, old.unit
    )
    num_blocks = max(old.num_blocks, int(indices[-1]) + 1 if indices.shape[0] else 0)
    sizes = np.zeros(num_blocks, np.int64)
    sizes[: old.num_blocks] = old.sizes
    sizes[indices] = new_sizes
    encoded = np.zeros(num_blocks, np.bool_)
    encoded[indices] = True
    encoded[old.num_blocks :] = True
    sources = np.where(encoded, 1, 0)
    values, ticks = _verify_decode.assemble(
        sizes,
        sources,
        [old.values, new_values],
        None if old.ticks is None or new_ticks is None else [old.ticks, new_ticks],
        [
            _verify_decode.block_positions(
                old.offsets, ~encoded[: old.num_blocks], old.sizes
            ),
            np.arange(new_values.shape[0]),
        ],
    )
    allowance = None
    if old.ticks is not None and params.time_error > 0:
        allowance = _verify_times.neighbour_intervals(old, encoded, sizes)
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
    """Verifies the result of ``fluxcode.update_time_blocks(group, samples, times,
    params, ...)``.

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
    # delete_ranges and incoming timestamps, then stable-sorting survivors with new
    # samples.
    old = _verify_decode.decode(group)
    if old.ticks is None:
        raise ValueError("the block group has no time axis")
    new_values, new_ticks, start, duration, ranges = _args.time_blocks_update(
        samples, times, start_time, block_duration, delete_ranges, time_unit, old.unit
    )
    new_ids = _verify_times.time_block_ids(new_ticks, start, duration)
    old_ids = np.repeat(np.arange(old.num_blocks), old.sizes)
    deleted = _verify_times.in_ranges(old.ticks, ranges) | np.isin(old.ticks, new_ticks)
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
    met = _verify_times.met_blocks(ranges, start, duration, old.num_blocks, num_blocks)
    # Re-encoded blocks must cover all changed blocks and stay within changed | met.
    reported = np.asarray(result.indices, np.int64)
    if reported.shape[0] and (reported.min() < 0 or reported.max() >= num_blocks):
        return [
            Violation(
                "layout",
                -1,
                f"re-encoded indices {_verify_common.summary(reported)} outside the "
                f"{num_blocks} blocks",
            )
        ]
    encoded = np.zeros(num_blocks, np.bool_)
    encoded[reported] = True
    if not (changed <= encoded).all() or not (encoded <= changed | met).all():
        return [
            Violation(
                "layout",
                -1,
                f"re-encoded blocks {_verify_common.summary(reported)}: must include "
                f"{_verify_common.summary(np.flatnonzero(changed))} and lie within "
                f"{_verify_common.summary(np.flatnonzero(changed | met))}",
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
    """Verifies the result of ``fluxcode.replace_time_blocks(group, blocks, samples,
    times, params, ...)``.

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
    old = _verify_decode.decode(group)
    if old.ticks is None:
        raise ValueError("the block group has no time axis")
    declared = _args.block_list(blocks)
    new_values, new_ticks, start, duration, _ = _args.time_blocks_update(
        samples, times, start_time, block_duration, None, time_unit, old.unit
    )
    new_ids = _verify_times.time_block_ids(new_ticks, start, duration)
    if not declared.shape[0]:
        if result.payload != group or np.asarray(result.indices).shape[0]:
            return [
                Violation(
                    "layout", -1, "no blocks declared, but the block group changed"
                )
            ]
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
    values, ticks = _verify_decode.assemble(
        sizes,
        sources,
        [old.values, new_values],
        [old.ticks, new_ticks],
        [
            _verify_decode.block_positions(
                old.offsets, ~encoded[: old.num_blocks], old.sizes
            ),
            np.arange(new_values.shape[0]),
        ],
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
        List of detected ``Violation`` records (empty if values and third-encode bytes
        match).
    """
    # Decode once, re-encode with exact times, verify decoded values are unchanged, then
    # encode a third time and verify identical compressed bytes.
    if params.target_bits_per_sample is not None:
        return []
    exact = replace(params, time_error=0.0)
    first = _api.decode_group(group)
    second = _api.encode_blocks(
        first.values, first.block_sizes, exact, times=first.times
    ).payload
    again = _api.decode_group(second).values
    if not _verify_common.same_doubles(again, first.values):
        moved = np.flatnonzero(again.view(np.int64) != first.values.view(np.int64))
        block = int(
            np.searchsorted(np.cumsum(first.block_sizes), moved[0], side="right")
        )
        return [
            Violation(
                "fixed_point",
                block,
                f"{moved.shape[0]} decoded samples changed on re-encode, first sample "
                f"{int(moved[0])}: {first.values[moved[0]]:.17g} -> "
                f"{again[moved[0]]:.17g}",
            )
        ]
    third = _api.encode_blocks(
        again, first.block_sizes, exact, times=first.times
    ).payload
    if third != second:
        return [
            Violation(
                "fixed_point", -1, "the third encode's bytes differ from the second's"
            )
        ]
    return []


# ======================================================================================
# Core check orchestration
# ======================================================================================


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
    """Decodes ``result.payload`` and verifies layout, untouched blocks, value bounds,
    decimals, and times.
    """
    found: list[Violation] = []
    new = _verify_decode.decode_or_report(result.payload, found)
    if new is None:
        return found
    if new.unit != unit:
        return [Violation("layout", -1, f"time unit code {new.unit}, not {unit}")]
    if new.num_blocks != sizes.shape[0] or not np.array_equal(new.sizes, sizes):
        return [
            Violation(
                "layout",
                -1,
                f"block sizes {_verify_common.summary(new.sizes)}, not "
                f"{_verify_common.summary(sizes)}",
            )
        ]
    if isinstance(result, UpdatedGroup) and not np.array_equal(
        result.indices, np.flatnonzero(encoded)
    ):
        return [
            Violation(
                "layout",
                -1,
                f"re-encoded indices {_verify_common.summary(result.indices)}, "
                f"not {_verify_common.summary(np.flatnonzero(encoded))}",
            )
        ]
    if not encoded.all():
        changed = ~np.repeat(encoded, sizes) & ~_verify_common.same_doubles_mask(
            new.values, values
        )
        _verify_common.report_samples(
            found,
            "untouched",
            changed,
            new.offsets,
            "samples of a block not re-encoded changed",
        )
    _verify_values.check_values(found, values, new, sizes, encoded, params, result)
    if decimals is not None:
        _verify_values.check_decimals(
            found, values, new, sizes, encoded, params, decimals
        )
    if ticks is not None and new.ticks is not None:
        _verify_times.check_times(
            found,
            ticks,
            new.ticks,
            sizes,
            new.offsets,
            encoded,
            time_error,
            allowance,
            time_blocks,
        )
    _verify_times.check_block_start_times(
        found, result.block_start_times, new.ticks, sizes, new.offsets, encoded, unit
    )
    return found
