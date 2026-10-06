# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Building and parsing units: the layer between the public API and the kernels.

A unit's blocks are handled as flat arrays (every sample of the unit, block after block)
with per-block sizes, so blocks of any size share one code path. The public entry points
in `_api` divide their input into blocks and call `encode`, `decode` or `splice` here.
"""

from collections.abc import Mapping
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import zstandard

from . import _args, _bitpacking, _compress, _decoder, _encoder, _format, _time
from ._types import DecodedUnit, EncodedSeries, EncodedUnit, Params, UpdatedUnit


def sample_offsets(block_sizes: np.ndarray) -> np.ndarray:
    """Computes sample offsets from block sizes.

    Args:
        block_sizes: 1D int64 array of sample counts per block.

    Returns:
        1D int64 array of cumulative sample offsets starting with 0.
    """
    offsets = np.zeros(block_sizes.shape[0] + 1, np.int64)
    np.cumsum(block_sizes, out=offsets[1:])
    return offsets


class BlockStats(NamedTuple):
    """Per-block summary statistics returned by encoding.

    Attributes:
        block_min: 1D float64 array of minimum finite values per block.
        block_max: 1D float64 array of maximum finite values per block.
        block_mean: 1D float64 array of finite means per block.
    """

    block_min: np.ndarray
    block_max: np.ndarray
    block_mean: np.ndarray


def kernel_args(params: Params, timed: bool) -> tuple[int, int, int, float, float, bool, int]:
    """Converts encoder parameters into the arguments of the encoding kernel.

    Args:
        params: Encoder configuration parameters.
        timed: Whether the unit stores timestamps (it decides the default noise floor).

    Returns:
        A tuple of (min_bits, max_bits, orders_mask, noise_factor, target_bits, decimal,
        pick_len): orders_mask has bit k set for each enabled order, 0.0 for noise_factor
        or target_bits turns that feature off, and pick_len is the encoder's PICK_LEN.
    """
    return (
        params.min_quantize_bits,
        params.max_quantize_bits,
        sum(1 << order for order in params.diff_orders),
        params.noise_factor(timed),
        float(params.target_bits_per_sample or 0.0),
        bool(params.decimal_detection),
        _encoder.PICK_LEN,
    )


def encode_rows(
    samples: np.ndarray, block_sizes: np.ndarray, params: Params, timed: bool
) -> tuple[_format.UnitRows, BlockStats]:
    """Encodes blocks into their rows (without a time axis) and summary statistics.

    Args:
        samples: 1D float64 array of every sample, block after block.
        block_sizes: 1D int64 array of block sizes adding up to the sample count.
        params: Encoder parameters.
        timed: Whether the unit stores timestamps.

    Returns:
        A tuple of (rows, stats): the blocks' UnitRows (without time rows) and their
        BlockStats.
    """
    num_blocks = block_sizes.shape[0]
    num_samples = samples.shape[0]
    rows = _format.UnitRows(
        np.empty(num_blocks, np.uint8),
        block_sizes,
        np.empty(num_blocks, np.int64),
        np.empty(num_blocks, np.int64),
        np.empty(num_samples, np.int16),
        np.empty(num_samples, np.uint8),
        None,
    )
    stats = BlockStats(np.empty(num_blocks), np.empty(num_blocks), np.empty(num_blocks))
    # Invoke numba kernel to process all blocks of the unit
    _encoder.encode_unit(
        samples,
        sample_offsets(block_sizes),
        *kernel_args(params, timed),
        rows.block_flags,
        rows.grid_params,
        rows.value_anchors,
        rows.residuals,
        rows.codes,
        *stats,
    )
    return rows, stats


def encode_time_rows(ticks: np.ndarray, block_sizes: np.ndarray, in_out_block_flags: np.ndarray) -> _format.TimeRows:
    """Analyzes the blocks' ticks into time rows, checking that they never decrease.

    Args:
        ticks: 1D int64 array of every sample's tick.
        block_sizes: 1D int64 array of block sizes.
        in_out_block_flags: 1D uint8 array of block flags (the time bits are set or cleared).

    Returns:
        The time rows.

    Raises:
        ValueError: If the times contain NaT or decrease anywhere.
    """
    # NaT is int64 minimum: past the first time it is a decrease
    if ticks.shape[0] and ticks[0] == _time.INT64_MIN:
        raise ValueError("times contain NaT")
    time_rows = _format.allocate_time_rows(block_sizes.shape[0], ticks.shape[0])
    status, sample_idx = _time.encode_times(ticks, sample_offsets(block_sizes), in_out_block_flags, *time_rows)
    if status != _time.OK:
        raise _args.decrease_error(ticks, sample_idx)
    return time_rows


def encode(
    samples: np.ndarray, block_sizes: np.ndarray, params: Params, ticks: np.ndarray | None = None, time_unit: int = 0
) -> EncodedUnit:
    """Encodes blocks of samples (and their ticks) into a unit.

    Args:
        samples: 1D float64 array of every sample, block after block.
        block_sizes: 1D int64 array of block sizes (0 to 65,535) adding up to the sample count.
        params: Encoder parameters.
        ticks: 1D int64 array of the samples' timestamps, or None for no time axis.
        time_unit: Time unit code of ticks (0 without a time axis).

    Returns:
        EncodedUnit holding compressed unit bytes and block min/max/mean.

    Raises:
        ValueError: If the unit would be too large, or the times contain NaT or decrease
            anywhere.
    """
    _args.check_unit_counts(block_sizes.shape[0], samples.shape[0], block_sizes)
    rows, stats = encode_rows(samples, block_sizes, params, ticks is not None)
    if ticks is not None:
        rows = rows._replace(time_rows=encode_time_rows(ticks, block_sizes, rows.block_flags))
    return EncodedUnit(_compress.compress(rows, samples.shape[0], _compress.EFFORTS[params.effort], time_unit), *stats)


def encode_series(
    samples: np.ndarray, block_len: int, blocks_per_unit: int, ticks: np.ndarray | None, time_unit: int, params: Params
) -> EncodedSeries:
    """Encodes a series as a sequence of fixed-size compressed units.

    Divides the input series into chunks of blocks_per_unit blocks of block_len samples,
    encoding each chunk independently via encode.

    Args:
        samples: 1D float64 array of samples.
        block_len: Samples per block.
        blocks_per_unit: Blocks per unit.
        ticks: Optional 1D int64 array of timestamps in time_unit ticks.
        time_unit: Time unit code (0 without a time axis).
        params: Encoder configuration parameters.

    Returns:
        EncodedSeries containing lists of unit byte strings and block summary statistics.

    Raises:
        ValueError: If a unit would be too large, or the times decrease anywhere.
    """
    samples_per_unit = blocks_per_unit * block_len
    num_blocks = -(-min(samples_per_unit, samples.shape[0]) // block_len)
    _args.check_unit_counts(num_blocks, min(samples_per_unit, samples.shape[0]))
    if ticks is not None:
        # Verify chronological order across unit boundaries
        unit_starts = np.arange(samples_per_unit, ticks.shape[0], samples_per_unit)
        decreases = unit_starts[ticks[unit_starts] < ticks[unit_starts - 1]]
        if decreases.shape[0]:
            raise _args.decrease_error(ticks, int(decreases[0]))
    parts = []
    # Encode each chunk as an independent unit
    for idx in range(0, samples.shape[0], samples_per_unit):
        chunk = samples[idx:idx + samples_per_unit]
        parts.append(encode(
            chunk,
            _args.fixed_sizes(chunk.shape[0], block_len),
            params,
            None if ticks is None else ticks[idx:idx + samples_per_unit],
            time_unit,
        ))
    units, mins, maxs, means = (list(col) for col in zip(*parts))
    return EncodedSeries(units, mins, maxs, means)


class ParsedUnit(NamedTuple):
    """A decompressed and validated unit.

    Attributes:
        raw_body: 1D uint8 array of the uncompressed body.
        header: The parsed header.
        block_flags: 1D uint8 array of block flags (a view of the body).
        block_sizes: 1D int64 array of block sizes.
        layout: The blocks' Layout.
    """

    raw_body: np.ndarray
    header: _format.UnitHeader
    block_flags: np.ndarray
    block_sizes: np.ndarray
    layout: _format.Layout

    @property
    def has_time(self) -> bool:
        """Whether the unit has a time axis."""
        return self.header.time_unit != 0


def decompress(unit: bytes) -> ParsedUnit:
    """Parses a unit's header, decompresses its body and validates the layout.

    Args:
        unit: Unit bytes (header and zstd frame).

    Returns:
        The parsed unit.

    Raises:
        ValueError: If the header is invalid, the frame's content size is missing
            or doesn't fit the header, the block sizes don't add up to the sample count, or
            block validation fails.
        zstandard.ZstdError: If the zstd frame is corrupt.
    """
    header = _format.unpack_header(unit)
    num_blocks, num_samples = header.num_blocks, header.num_samples
    has_time = header.time_unit != 0
    frame = memoryview(unit)[_format.HEADER_BYTES:]
    # Check the recorded body size against the header before allocating it
    content_size = zstandard.frame_content_size(frame)
    if content_size < 0:
        raise ValueError("unit's zstd frame doesn't record its content size")
    smallest, largest = _format.unit_size_bounds(num_blocks, num_samples, has_time)
    if not smallest <= content_size <= largest:
        raise ValueError(f"unit body of {content_size} bytes doesn't fit {num_blocks} blocks of {num_samples} samples")
    raw_body = np.frombuffer(_compress.zstd()[1].decompress(frame), np.uint8)
    block_flags = raw_body[:num_blocks]
    block_sizes, offsets = _format.read_layout(raw_body, num_blocks)
    if int(offsets.sample_offsets[-1]) != num_samples:
        raise ValueError(f"unit's block sizes add up to {int(offsets.sample_offsets[-1])}, not its {num_samples} samples")
    # The exact size depends on which blocks carry non-finite code planes and time residual planes
    if raw_body.shape[0] != _format.unit_size(num_blocks, offsets, has_time):
        raise ValueError(
            f"unit body of {raw_body.shape[0]} bytes doesn't match its blocks' sizes and flags"
        )
    validation_status, failing_block_idx = _decoder.check_unit(raw_body, offsets.sample_offsets, has_time)
    if validation_status == _decoder.BAD_FLAGS:
        raise ValueError(
            f"block {failing_block_idx}: block flags {raw_body[failing_block_idx]:#04x} sets reserved bits "
            "(not supported by this version)"
        )
    if validation_status == _decoder.BAD_PARAM:
        raise ValueError(f"block {failing_block_idx}: parameter out of range (corrupt unit)")
    if validation_status == _decoder.BAD_ANCHOR:
        raise ValueError(f"block {failing_block_idx}: anchor out of range (corrupt unit)")
    return ParsedUnit(raw_body, header, block_flags, block_sizes, offsets)


def read_time_rows(parsed: ParsedUnit, block_ids: np.ndarray | None = None) -> _format.TimeRows:
    """Reads a unit's time rows, validating the time columns.

    Args:
        parsed: A unit with a time axis.
        block_ids: The blocks whose time residuals to unpack (every block if None; an empty
            array reads the columns only). Other blocks' residuals are left unwritten.

    Returns:
        The time rows.

    Raises:
        ValueError: If a block start time is int64 minimum or overflows int64, a long
            block's residuals fit in 32 bits, an empty block has nonzero time columns, or a
            one-sample block isn't regular with step and reference 0.
    """
    time_rows = _format.allocate_time_rows(parsed.header.num_blocks, parsed.header.num_samples)
    _bitpacking.check_time_rows_status(
        *_bitpacking.read_time_rows(parsed.raw_body, *parsed.layout, *time_rows, block_ids=block_ids)
    )
    return time_rows


def expand_times(
    block_flags: np.ndarray,
    offsets: np.ndarray,
    time_rows: _format.TimeRows,
    block_ids: np.ndarray,
    out_ticks: np.ndarray,
) -> None:
    """Reconstructs the ticks of the given blocks from their time rows into out_ticks.

    Args:
        block_flags: 1D uint8 array of block flags.
        offsets: 1D int64 array of sample offsets for each block.
        time_rows: Time rows containing start times, step, reference, and residuals.
        block_ids: 1D int64 array of block indices to expand.
        out_ticks: 1D int64 destination array where reconstructed ticks are written.

    Raises:
        ValueError: If a time step is negative (or zero on an irregular block), an
            irregular block's first residual isn't 0, or a time overflows int64.
    """
    status, block_idx = _time.expand_times(block_flags, offsets, *time_rows, block_ids, out_ticks)
    if status == _time.BAD_STEP:
        raise ValueError(f"block {block_idx}: time step out of range (corrupt unit)")
    if status == _time.BAD_FIRST_RESIDUAL:
        raise ValueError(f"block {block_idx}: first time residual is not 0 (corrupt unit)")
    if status == _time.OVERFLOW:
        raise ValueError(f"block {block_idx}: times overflow int64 (corrupt unit)")


def decode_blocks(
    parsed: ParsedUnit, block_ids: np.ndarray, time_rows: _format.TimeRows | None = None
) -> tuple[np.ndarray, np.ndarray | None]:
    """Decodes the given blocks of a parsed unit.

    Args:
        parsed: The unit.
        block_ids: 1D int64 array of the blocks to decode.
        time_rows: The unit's time rows if already read (with at least the given blocks'
            residuals).

    Returns:
        A tuple of (values, ticks): 1D arrays of every sample of the unit (float64 values
        and int64 ticks, None without a time axis) in which only the given blocks' samples
        are written.
    """
    offsets = parsed.layout
    values = np.empty(parsed.header.num_samples)
    _decoder.decode_unit(
        parsed.raw_body, offsets.sample_offsets, offsets.octet_offsets, offsets.code_offsets, block_ids, values,
        parsed.header.byte_planes, parsed.has_time,
    )
    ticks = None
    if parsed.has_time:
        ticks = np.empty(parsed.header.num_samples, np.int64)
        if time_rows is None:
            time_rows = read_time_rows(parsed, block_ids)
        expand_times(parsed.block_flags, offsets.sample_offsets, time_rows, block_ids, ticks)
    return values, ticks


def time_dtype(time_unit: int) -> np.dtype:
    """Returns the datetime64 dtype of a time unit code.

    Args:
        time_unit: Integer code representing the time unit.

    Returns:
        NumPy datetime64 dtype matching the specified unit.
    """
    return np.dtype(f"datetime64[{_format.TIME_UNIT_NAMES[time_unit]}]")


def decode(unit: bytes) -> DecodedUnit:
    """Decodes a unit into its samples, timestamps and block sizes (see decode_unit).

    Args:
        unit: Compressed byte string of the encoded unit.

    Returns:
        DecodedUnit namedtuple containing values, optional times, and block sizes.
    """
    parsed = decompress(unit)
    values, ticks = decode_blocks(parsed, np.arange(parsed.header.num_blocks, dtype=np.int64))
    times = None if ticks is None else ticks.view(time_dtype(parsed.header.time_unit))
    return DecodedUnit(values, times, parsed.block_sizes)


def read_rows(parsed: ParsedUnit, time_rows: _format.TimeRows | None = None) -> _format.UnitRows:
    """Reads every block's rows (residuals and codes, without dequantizing) and time rows.

    Args:
        parsed: Parsed unit container.
        time_rows: Optional pre-read time rows; if None and the unit has time,
            time rows are read from the unit.

    Returns:
        UnitRows namedtuple containing block flags, sizes, grid params, value anchors,
        residuals, codes, and optional time rows.
    """
    num_blocks, num_samples = parsed.header.num_blocks, parsed.header.num_samples
    offsets = parsed.layout
    rows = _format.UnitRows(
        parsed.block_flags.copy(),
        parsed.block_sizes,
        np.empty(num_blocks, np.int64),
        np.empty(num_blocks, np.int64),
        np.empty(num_samples, np.int16),
        np.zeros(num_samples, np.uint8),
        (time_rows if time_rows is not None else read_time_rows(parsed)) if parsed.has_time else None,
    )
    _bitpacking.read_rows(
        parsed.raw_body, parsed.header.byte_planes, parsed.has_time, offsets.sample_offsets, offsets.octet_offsets,
        offsets.code_offsets, rows.grid_params, rows.value_anchors, rows.residuals, rows.codes,
    )
    return rows


def _last_ticks(parsed: ParsedUnit, time_rows: _format.TimeRows, block_ids: np.ndarray) -> np.ndarray:
    """Returns the last ticks of the given non-empty blocks of a parsed unit.

    Only those blocks' time residuals are unpacked and expanded.

    Args:
        parsed: The unit.
        time_rows: Its time rows: the columns at least (the blocks' residuals are unpacked
            into them).
        block_ids: 1D int64 array of the blocks.

    Returns:
        1D int64 array containing the last tick value for each specified non-empty block.
    """
    if not block_ids.shape[0]:
        return np.zeros(0, np.int64)
    offsets = parsed.layout.sample_offsets
    _bitpacking.read_time_residuals(parsed.raw_body, *parsed.layout, block_ids, time_rows.residuals)
    out_ticks = np.empty(parsed.header.num_samples, np.int64)
    expand_times(parsed.block_flags, offsets, time_rows, block_ids, out_ticks)
    return out_ticks[offsets[block_ids + 1] - 1]


def _check_spliced_order(
    parsed: ParsedUnit, time_rows: _format.TimeRows, block_sizes: np.ndarray, block_starts: np.ndarray,
    is_new: np.ndarray, new_last_ticks: np.ndarray,
) -> None:
    """Checks that the times never decrease where a new block meets its non-empty neighbours.

    Args:
        parsed: The existing unit.
        time_rows: Its time rows (the columns at least).
        block_sizes: 1D int64 array of the spliced unit's block sizes.
        block_starts: 1D int64 array of its block start times.
        is_new: 1D bool array marking the new blocks.
        new_last_ticks: 1D int64 array of the last tick of each non-empty new block (by block).

    Raises:
        ValueError: If a block starts before the previous non-empty block's last time.
    """
    filled = np.flatnonzero(block_sizes)
    # Only pairs of consecutive non-empty blocks where one is new: the others were in order
    pairs = np.flatnonzero(is_new[filled[:-1]] | is_new[filled[1:]])
    previous, following = filled[pairs], filled[pairs + 1]
    last = new_last_ticks[previous]
    carried = ~is_new[previous]
    last[carried] = _last_ticks(parsed, time_rows, previous[carried])
    decreasing = np.flatnonzero(block_starts[following] < last)
    if decreasing.shape[0]:
        pair_idx = int(decreasing[0])
        raise ValueError(
            f"times must be non-decreasing: block {int(following[pair_idx])} starts at "
            f"{int(block_starts[following[pair_idx]])}, below block {int(previous[pair_idx])}'s last time "
            f"{int(last[pair_idx])}"
        )


def splice(
    parsed: ParsedUnit,
    indices: np.ndarray,
    samples: np.ndarray,
    block_sizes: np.ndarray,
    ticks: np.ndarray | None,
    params: Params,
    old_time_rows: _format.TimeRows | None = None,
) -> UpdatedUnit:
    """Replaces or appends blocks of a unit, carrying every other block over untouched.

    Untouched blocks keep their exact bytes in every field (_bitpacking.splice_body): they
    are neither unpacked, dequantized nor re-encoded. Blocks past the unit's end that
    indices skip are appended empty.

    Args:
        parsed: The existing unit.
        indices: 1D int64 array of distinct, increasing, non-negative block indices.
        samples: 1D float64 array of the new blocks' samples, in indices order.
        block_sizes: 1D int64 array of the new blocks' sizes.
        ticks: 1D int64 array of the new blocks' ticks (in the unit's time unit), required
            exactly when the unit has a time axis.
        params: Encoder parameters.
        old_time_rows: The unit's time rows if already read (the columns at least).

    Returns:
        UpdatedUnit with the new unit and the statistics of every re-encoded block
        (including empty ones appended to fill a gap).

    Raises:
        ValueError: If the unit would be too large or its times would decrease.
    """
    num_old_blocks = parsed.header.num_blocks
    # Determine the total number of blocks after applying updates
    num_blocks = max(num_old_blocks, int(indices[-1]) + 1) if indices.shape[0] else num_old_blocks
    _args.check_unit_counts(num_blocks, 0)
    # Appended blocks that indices skip become empty blocks; they hold no samples, so the
    # samples (and ticks) keep their order
    gaps = np.setdiff1d(np.arange(num_old_blocks, num_blocks), indices) if num_blocks > num_old_blocks else indices[:0]
    if gaps.shape[0]:
        merged = np.union1d(indices, gaps)
        merged_sizes = np.zeros(merged.shape[0], np.int64)
        merged_sizes[np.searchsorted(merged, indices)] = block_sizes
        indices, block_sizes = merged, merged_sizes

    # Assemble updated block sizes across the entire unit
    sizes = np.zeros(num_blocks, np.int64)
    sizes[:num_old_blocks] = parsed.block_sizes
    sizes[indices] = block_sizes
    num_samples = int(sizes.sum())
    _args.check_unit_counts(num_blocks, num_samples)

    # Encode value residuals and codes for the replacement blocks
    new_rows, stats = encode_rows(samples, block_sizes, params, ticks is not None)
    new_offsets = sample_offsets(block_sizes)
    old_starts = None
    if ticks is not None:
        # Encode timestamp rows for the replacement blocks
        new_time_rows = encode_time_rows(ticks, block_sizes, new_rows.block_flags)
        new_rows = new_rows._replace(time_rows=new_time_rows)
        if old_time_rows is None:
            old_time_rows = read_time_rows(parsed, np.zeros(0, np.int64))
        old_starts = old_time_rows.starts
        is_new = np.zeros(num_blocks, bool)
        is_new[indices] = True
        starts = np.zeros(num_blocks, np.int64)
        starts[:num_old_blocks] = old_starts
        starts[indices] = new_time_rows.starts
        new_last_ticks = np.zeros(num_blocks, np.int64)
        filled = block_sizes > 0
        new_last_ticks[indices[filled]] = ticks[new_offsets[1:][filled] - 1]
        # Verify monotonic timestamp ordering across block boundaries
        _check_spliced_order(parsed, old_time_rows, sizes, starts, is_new, new_last_ticks)

    has_time = parsed.has_time
    old_body = parsed.raw_body
    # Splicing copies byte planes, so convert a bit-plane body first
    if not parsed.header.byte_planes:
        old_body = _bitpacking.to_byte_planes(
            old_body, num_old_blocks, int(parsed.layout.octet_offsets[-1]), has_time
        )
    # Splice modified block rows directly into the binary body
    body, offsets = _bitpacking.splice_body(old_body, parsed.layout, old_starts, indices, new_rows, new_offsets)
    # Repack and compress the spliced unit
    unit = _compress.pack(
        body, offsets, num_blocks, num_samples, _compress.EFFORTS[params.effort], parsed.header.time_unit,
    )
    return UpdatedUnit(unit, indices, *stats)


def update(
    unit: bytes,
    blocks: Mapping[int, npt.ArrayLike],
    params: Params,
    times: Mapping[int, npt.ArrayLike] | None,
    time_unit: str | None = None,
) -> UpdatedUnit:
    """Validates update's arguments and splices the new blocks into the unit (see _api.update).

    Args:
        unit: Compressed byte string of the original encoded unit.
        blocks: Mapping from block index to new sample values.
        params: Encoder parameters.
        times: Optional mapping from block index to new timestamps.
        time_unit: Optional time unit string specifying the expected timestamp resolution.

    Returns:
        UpdatedUnit containing the spliced unit bytes, updated indices, and compression statistics.

    Raises:
        ValueError: If the unit is corrupt, an index is invalid, or the times don't match the
            unit's time axis or the blocks.
        zstandard.ZstdError: If the zstd frame is corrupt.
    """
    parsed = decompress(unit)
    _args.check_stored_time_unit(time_unit, parsed.header.time_unit)
    indices, samples, sizes, ticks = _args.update_blocks(blocks, times, parsed.header.time_unit)
    if not indices.shape[0]:
        return UpdatedUnit(unit, np.zeros(0, np.int64), np.zeros(0), np.zeros(0), np.zeros(0))
    return splice(parsed, indices, samples, sizes, ticks, params)
