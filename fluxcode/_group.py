# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Building and parsing block groups: the layer between the public API and the kernels.

A block group's blocks are handled as flat arrays (every sample of the block group, block after block)
with per-block sizes, so blocks of any size share one code path. The public entry points
in `_api` divide their input into blocks and call `encode`, `encode_series`, `decode` and `update` here (`splice` is reached
through `update` and `_time_blocks`).
"""

from collections.abc import Mapping
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import zstandard

from . import _args, _bitpacking, _compress, _decoder, _encoder, _format, _time
from ._types import DecodedGroup, EncodedGroup, EncodedSeries, Params, UpdatedGroup


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


def kernel_args(params: Params) -> tuple[int, int, int, float, float, bool]:
    """Converts encoder parameters into the arguments of the encoding kernel.

    Args:
        params: Encoder configuration parameters.

    Returns:
        A tuple of (min_bits, max_bits, orders_mask, noise_factor, target_bits, decimal):
        orders_mask has bit k set for each enabled order, and 0.0 for noise_factor or
        target_bits turns that feature off.
    """
    return (
        params.min_quantize_bits,
        params.max_quantize_bits,
        sum(1 << order for order in params.diff_orders),
        float(params.noise_floor_sigma),
        float(params.target_bits_per_sample or 0.0),
        bool(params.decimal_detection),
    )


def encode_rows(
    samples: np.ndarray, block_sizes: np.ndarray, params: Params, ticks: np.ndarray | None = None
) -> tuple[_format.GroupRows, BlockStats]:
    """Encodes blocks into their rows (and time rows, given ticks) and summary statistics.

    Args:
        samples: 1D float64 array of every sample, block after block.
        block_sizes: 1D int64 array of block sizes adding up to the sample count.
        params: Encoder parameters.
        ticks: 1D int64 array of the samples' ticks, or None without a time axis. They are
            checked and encoded first: irregular blocks' ticks then steer the noise floor.

    Returns:
        A tuple of (rows, stats): the blocks' GroupRows (time rows None without ticks) and
        their BlockStats.

    Raises:
        ValueError: If the times contain NaT or decrease anywhere.
    """
    num_blocks = block_sizes.shape[0]
    num_samples = samples.shape[0]
    block_flags = np.zeros(num_blocks, np.uint8)
    time_rows = None if ticks is None else encode_time_rows(ticks, block_sizes, block_flags)
    rows = _format.GroupRows(
        np.empty(num_blocks, np.uint8),
        block_sizes,
        np.empty(num_blocks, np.int64),
        np.empty(num_blocks, np.int64),
        np.empty(num_samples, np.int16),
        np.empty(num_samples, np.uint8),
        time_rows,
    )
    stats = BlockStats(np.empty(num_blocks), np.empty(num_blocks), np.empty(num_blocks))
    # Invoke numba kernel to process all blocks of the block group
    _encoder.encode_group(
        samples,
        sample_offsets(block_sizes),
        np.zeros(0, np.int64) if ticks is None else ticks,
        block_flags[:0] if ticks is None else block_flags,
        *kernel_args(params),
        rows.block_flags,
        rows.grid_params,
        rows.value_anchors,
        rows.residuals,
        rows.codes,
        *stats,
    )
    # The kernel writes the value flags; the time rows set the time flags
    rows.block_flags[:] |= block_flags
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


def snap_ticks(
    ticks: np.ndarray, block_sizes: np.ndarray, time_error: float, seed_ticks: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Rounds each block's ticks to its grid (_time.time_quanta, time_phases, snap_times).

    The order of the ticks is checked only if a block is rounded (rounding could hide a
    decrease); otherwise they are returned as given, for encode_times to check.

    Args:
        ticks: 1D int64 array of every sample's tick.
        block_sizes: 1D int64 array of block sizes adding up to the tick count.
        time_error: The time error (Params.time_error); 0 keeps the ticks.
        seed_ticks: Optional 1D int64 array per block: the last tick of a stored block before
            it (whose phase it keeps if that fits), or INT64_MIN.

    Returns:
        A tuple of (ticks, quanta): the rounded ticks (a new array if any block was rounded,
        else the given one) and each block's quantum (0 where its ticks were kept).

    Raises:
        ValueError: If the times contain NaT, or decrease anywhere when a block is rounded.
    """
    num_blocks = block_sizes.shape[0]
    quanta = np.zeros(num_blocks, np.int64)
    if time_error <= 0 or not ticks.shape[0]:
        return ticks, quanta
    # NaT is int64 minimum: past the first time it is a decrease
    if ticks[0] == _time.INT64_MIN:
        raise ValueError("times contain NaT")
    offsets = sample_offsets(block_sizes)
    regular = np.zeros(num_blocks, np.bool_)
    _time.time_quanta(ticks, offsets, float(time_error), np.empty(_time.CADENCE_SAMPLES, np.int64), quanta, regular)
    # A regular block stores as cheaply as it can already: rounding could only move it
    quanta[regular] = 0
    if not quanta.any():
        return ticks, quanta
    if seed_ticks is None:
        seed_ticks = np.full(num_blocks, _time.INT64_MIN, np.int64)
    phases = np.zeros(num_blocks, np.int64)
    _time.time_phases(ticks, offsets, quanta, seed_ticks, phases)
    snapped = np.empty_like(ticks)
    status, sample_idx = _time.snap_times(ticks, offsets, quanta, phases, snapped)
    if status != _time.OK:
        raise _args.decrease_error(ticks, sample_idx)
    return snapped, quanta


def encode(
    samples: np.ndarray,
    block_sizes: np.ndarray,
    params: Params,
    ticks: np.ndarray | None = None,
    time_unit: int = 0,
    *,
    snap: bool = True,
) -> EncodedGroup:
    """Encodes blocks of samples (and their ticks) into a block group.

    Args:
        samples: 1D float64 array of every sample, block after block.
        block_sizes: 1D int64 array of block sizes (0 to 65,535) adding up to the sample count.
        params: Encoder parameters.
        ticks: 1D int64 array of the samples' timestamps, or None for no time axis.
        time_unit: Time unit code of ticks (0 without a time axis).
        snap: Whether to round the ticks to params.time_error (False: the caller has).

    Returns:
        EncodedGroup holding compressed block group bytes and block min/max/mean.

    Raises:
        ValueError: If the block group would be too large, or the times contain NaT or decrease
            anywhere.
    """
    _args.check_group_counts(block_sizes.shape[0], samples.shape[0], block_sizes)
    if ticks is not None and snap:
        ticks, _ = snap_ticks(ticks, block_sizes, params.time_error)
    rows, stats = encode_rows(samples, block_sizes, params, ticks)
    return EncodedGroup(_compress.compress(rows, samples.shape[0], _compress.EFFORTS[params.effort], time_unit), *stats)


def encode_series(
    samples: np.ndarray, block_len: int, blocks_per_group: int, ticks: np.ndarray | None, time_unit: int, params: Params
) -> EncodedSeries:
    """Encodes a series as a sequence of fixed-size compressed block groups.

    Divides the input series into chunks of blocks_per_group blocks of block_len samples,
    encoding each chunk independently via encode.

    Args:
        samples: 1D float64 array of samples.
        block_len: Samples per block.
        blocks_per_group: Blocks per block group.
        ticks: Optional 1D int64 array of timestamps in time_unit ticks.
        time_unit: Time unit code (0 without a time axis).
        params: Encoder configuration parameters.

    Returns:
        EncodedSeries containing lists of block group byte strings and block summary statistics.

    Raises:
        ValueError: If a block group would be too large, or the times decrease anywhere.
    """
    samples_per_group = blocks_per_group * block_len
    num_blocks = -(-min(samples_per_group, samples.shape[0]) // block_len)
    _args.check_group_counts(num_blocks, min(samples_per_group, samples.shape[0]))
    if ticks is not None:
        # Verify chronological order across block group boundaries
        group_starts = np.arange(samples_per_group, ticks.shape[0], samples_per_group)
        decreases = group_starts[ticks[group_starts] < ticks[group_starts - 1]]
        if decreases.shape[0]:
            raise _args.decrease_error(ticks, int(decreases[0]))
        # Round the whole series at once (its blocks are the block groups' blocks), so that
        # block groups stay in order where neighbouring blocks' quanta differ
        ticks, _ = snap_ticks(ticks, _args.fixed_sizes(ticks.shape[0], block_len), params.time_error)
    parts = []
    # Encode each chunk as an independent block group
    for idx in range(0, samples.shape[0], samples_per_group):
        chunk = samples[idx:idx + samples_per_group]
        parts.append(encode(
            chunk,
            _args.fixed_sizes(chunk.shape[0], block_len),
            params,
            None if ticks is None else ticks[idx:idx + samples_per_group],
            time_unit,
            snap=False,
        ))
    groups, mins, maxs, means = (list(col) for col in zip(*parts))
    return EncodedSeries(groups, mins, maxs, means)


class ParsedGroup(NamedTuple):
    """A decompressed and validated block group.

    Attributes:
        raw_body: 1D uint8 array of the uncompressed body.
        header: The parsed header.
        block_flags: 1D uint8 array of block flags (a view of the body).
        block_sizes: 1D int64 array of block sizes.
        layout: The blocks' Layout.
    """

    raw_body: np.ndarray
    header: _format.GroupHeader
    block_flags: np.ndarray
    block_sizes: np.ndarray
    layout: _format.Layout

    @property
    def has_time(self) -> bool:
        """Whether the block group has a time axis."""
        return self.header.time_unit != 0


def decompress(group: bytes) -> ParsedGroup:
    """Parses a block group's header, decompresses its body and validates the layout.

    Args:
        group: Block group bytes (header and zstd frame).

    Returns:
        The parsed block group.

    Raises:
        ValueError: If the header is invalid, the frame's content size is missing
            or doesn't fit the header, the block sizes don't add up to the sample count, or
            block validation fails.
        zstandard.ZstdError: If the zstd frame is corrupt.
    """
    header = _format.unpack_header(group)
    num_blocks, num_samples = header.num_blocks, header.num_samples
    has_time = header.time_unit != 0
    frame = memoryview(group)[_format.HEADER_BYTES:]
    # Check the recorded body size against the header before allocating it
    content_size = zstandard.frame_content_size(frame)
    if content_size < 0:
        raise ValueError("block group's zstd frame doesn't record its content size")
    smallest, largest = _format.group_size_bounds(num_blocks, num_samples, has_time)
    if not smallest <= content_size <= largest:
        raise ValueError(f"block group body of {content_size} bytes doesn't fit {num_blocks} blocks of {num_samples} samples")
    raw_body = np.frombuffer(_compress.zstd()[1].decompress(frame), np.uint8)
    block_flags = raw_body[:num_blocks]
    block_sizes, offsets = _format.read_layout(raw_body, num_blocks)
    if int(offsets.sample_offsets[-1]) != num_samples:
        raise ValueError(f"block group's block sizes add up to {int(offsets.sample_offsets[-1])}, not its {num_samples} samples")
    # The exact size depends on which blocks carry non-finite code planes and time residual planes
    if raw_body.shape[0] != _format.group_size(num_blocks, offsets, has_time):
        raise ValueError(
            f"block group body of {raw_body.shape[0]} bytes doesn't match its blocks' sizes and flags"
        )
    validation_status, failing_block_idx = _decoder.check_group(raw_body, offsets.sample_offsets, has_time)
    if validation_status == _decoder.BAD_FLAGS:
        raise ValueError(
            f"block {failing_block_idx}: block flags {raw_body[failing_block_idx]:#04x} sets reserved bits "
            "(not supported by this version)"
        )
    if validation_status == _decoder.BAD_PARAM:
        raise ValueError(f"block {failing_block_idx}: parameter out of range (corrupt block group)")
    if validation_status == _decoder.BAD_ANCHOR:
        raise ValueError(f"block {failing_block_idx}: anchor out of range (corrupt block group)")
    return ParsedGroup(raw_body, header, block_flags, block_sizes, offsets)


def read_time_rows(parsed: ParsedGroup, block_ids: np.ndarray) -> _format.TimeRows:
    """Reads a block group's time rows, validating the time columns.

    Args:
        parsed: A block group with a time axis.
        block_ids: 1D int64 array of the blocks whose time residuals to unpack (an empty
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
        raise ValueError(f"block {block_idx}: time step out of range (corrupt block group)")
    if status == _time.BAD_FIRST_RESIDUAL:
        raise ValueError(f"block {block_idx}: first time residual is not 0 (corrupt block group)")
    if status == _time.OVERFLOW:
        raise ValueError(f"block {block_idx}: times overflow int64 (corrupt block group)")


def decode_blocks(
    parsed: ParsedGroup, block_ids: np.ndarray, time_rows: _format.TimeRows | None = None
) -> tuple[np.ndarray, np.ndarray | None]:
    """Decodes the given blocks of a parsed block group.

    Args:
        parsed: The block group.
        block_ids: 1D int64 array of the blocks to decode.
        time_rows: The block group's time rows if already read (with at least the given blocks'
            residuals).

    Returns:
        A tuple of (values, ticks): 1D arrays of every sample of the block group (float64 values
        and int64 ticks, None without a time axis) in which only the given blocks' samples
        are written.
    """
    offsets = parsed.layout
    values = np.empty(parsed.header.num_samples)
    _decoder.decode_group(
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


def decode(group: bytes) -> DecodedGroup:
    """Decodes a block group into its samples, timestamps and block sizes (see decode_group).

    Args:
        group: Compressed byte string of the encoded block group.

    Returns:
        DecodedGroup namedtuple containing values, optional times, and block sizes.
    """
    parsed = decompress(group)
    values, ticks = decode_blocks(parsed, np.arange(parsed.header.num_blocks, dtype=np.int64))
    times = None if ticks is None else ticks.view(time_dtype(parsed.header.time_unit))
    return DecodedGroup(values, times, parsed.block_sizes)


def _block_ticks(parsed: ParsedGroup, time_rows: _format.TimeRows, block_ids: np.ndarray) -> np.ndarray:
    """Expands the ticks of the given blocks of a parsed block group.

    Only those blocks' time residuals are unpacked and expanded.

    Args:
        parsed: The block group.
        time_rows: Its time rows: the columns at least (the blocks' residuals are unpacked
            into them).
        block_ids: 1D int64 array of the blocks.

    Returns:
        1D int64 array of every sample of the block group, written at the given blocks' samples.
    """
    _bitpacking.read_time_residuals(parsed.raw_body, *parsed.layout, block_ids, time_rows.residuals)
    out_ticks = np.empty(parsed.header.num_samples, np.int64)
    expand_times(parsed.block_flags, parsed.layout.sample_offsets, time_rows, block_ids, out_ticks)
    return out_ticks


def _last_ticks(parsed: ParsedGroup, time_rows: _format.TimeRows, block_ids: np.ndarray) -> np.ndarray:
    """Returns the last ticks of the given non-empty blocks of a parsed block group.

    Args:
        parsed: The block group.
        time_rows: Its time rows: the columns at least (the blocks' residuals are unpacked
            into them).
        block_ids: 1D int64 array of the blocks.

    Returns:
        1D int64 array containing the last tick value for each specified non-empty block.
    """
    if not block_ids.shape[0]:
        return np.zeros(0, np.int64)
    out_ticks = _block_ticks(parsed, time_rows, block_ids)
    return out_ticks[parsed.layout.sample_offsets[block_ids + 1] - 1]


def _check_spliced_order(
    parsed: ParsedGroup, time_rows: _format.TimeRows, block_sizes: np.ndarray, block_starts: np.ndarray,
    is_new: np.ndarray, new_last_ticks: np.ndarray,
) -> None:
    """Checks that the times never decrease where a new block meets its non-empty neighbours.

    Args:
        parsed: The existing block group.
        time_rows: Its time rows (the columns at least).
        block_sizes: 1D int64 array of the spliced block group's block sizes.
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


def _stored_quantum(parsed: ParsedGroup, time_rows: _format.TimeRows, block_idx: int, time_error: float) -> int:
    """The quantum the time error gives a stored block's times.

    A regular block's interval is in its columns (step times reference); an irregular one's
    times are expanded.

    Args:
        parsed: The block group.
        time_rows: Its time rows (the columns at least; an irregular block's residuals are
            unpacked).
        block_idx: A non-empty stored block.
        time_error: The time error (> 0).

    Returns:
        The quantum, 0 if there is none.
    """
    if not parsed.block_flags[block_idx] & _format.BLOCK_FLAG_IRREGULAR_TIME:
        interval = float(time_rows.steps[block_idx]) * float(time_rows.refs[block_idx])
        quantum = _time.time_quantum(interval, time_error)
        return quantum if quantum > 1 else 0
    offsets = parsed.layout.sample_offsets
    ticks = _block_ticks(parsed, time_rows, np.array([block_idx], np.int64))[offsets[block_idx]:offsets[block_idx + 1]]
    quanta = np.zeros(1, np.int64)
    _time.time_quanta(
        ticks, np.array([0, ticks.shape[0]], np.int64), time_error, np.empty(_time.CADENCE_SAMPLES, np.int64), quanta,
        np.zeros(1, np.bool_),
    )
    return int(quanta[0])


class StoredNeighbours(NamedTuple):
    """The stored non-empty blocks around a splice's new non-empty blocks.

    Attributes:
        rows: 1D int64 array of the new non-empty blocks, as positions among the new blocks.
        previous: 1D int64 array of each one's previous non-empty block (-1 if none).
        following: 1D int64 array of each one's next non-empty block (-1 if none).
        stored_previous: 1D bool array: the previous one is stored (not new).
        stored_following: 1D bool array: the next one is stored.
        last_ticks: 1D int64 array of the stored previous blocks' last ticks (0 elsewhere).
    """

    rows: np.ndarray
    previous: np.ndarray
    following: np.ndarray
    stored_previous: np.ndarray
    stored_following: np.ndarray
    last_ticks: np.ndarray


def _stored_neighbours(
    parsed: ParsedGroup, time_rows: _format.TimeRows, sizes: np.ndarray, indices: np.ndarray
) -> StoredNeighbours:
    """Finds the stored non-empty blocks around the new non-empty blocks of a splice.

    Args:
        parsed: The existing block group.
        time_rows: Its time rows (the columns at least; the stored previous blocks' residuals
            are unpacked).
        sizes: 1D int64 array of the spliced block group's block sizes.
        indices: 1D int64 array of the new blocks, increasing.

    Returns:
        The StoredNeighbours.
    """
    num_old_blocks = parsed.header.num_blocks
    is_new = np.zeros(sizes.shape[0], bool)
    is_new[indices] = True
    filled = np.flatnonzero(sizes)
    rows = np.flatnonzero(sizes[indices] > 0)
    positions = np.searchsorted(filled, indices[rows])
    previous = np.where(positions > 0, filled[np.maximum(positions - 1, 0)], -1)
    following = np.where(positions + 1 < filled.shape[0], filled[np.minimum(positions + 1, filled.shape[0] - 1)], -1)
    stored_previous = (previous >= 0) & ~is_new[np.maximum(previous, 0)]
    stored_following = (following >= 0) & (following < num_old_blocks) & ~is_new[np.maximum(following, 0)]
    last_ticks = np.zeros(rows.shape[0], np.int64)
    last_ticks[stored_previous] = _last_ticks(parsed, time_rows, previous[stored_previous])
    return StoredNeighbours(rows, previous, following, stored_previous, stored_following, last_ticks)


def _clamp_to_neighbours(
    parsed: ParsedGroup,
    time_rows: _format.TimeRows,
    neighbours: StoredNeighbours,
    new_offsets: np.ndarray,
    quanta: np.ndarray,
    time_error: float,
    raw_ticks: np.ndarray,
    in_out_ticks: np.ndarray,
) -> None:
    """Keeps rounded new blocks in order with the stored blocks around them.

    A stored block's times were rounded too, by up to half its quantum, so a new block whose
    times are in order with the samples a stored neighbour was made from can still round past
    it. A stored block can't give way, so a new block that starts below the previous stored
    block's last time is raised to it if its first time as given is within half the larger of
    the two blocks' quanta of it; one that ends above the next stored block's start is lowered
    to it likewise. Every time then stays within half that quantum of the time given. Larger
    overlaps are left for _check_spliced_order to reject.

    Args:
        parsed: The existing block group.
        time_rows: Its time rows (the columns at least).
        neighbours: The new blocks' stored neighbours (_stored_neighbours).
        new_offsets: 1D int64 array of the new blocks' sample offsets.
        quanta: 1D int64 array of the new blocks' quanta (0: not rounded).
        time_error: The time error (> 0).
        raw_ticks: 1D int64 array of the new blocks' ticks as given.
        in_out_ticks: 1D int64 array of the new blocks' rounded ticks, clamped in place.
    """
    for row, new_idx in enumerate(neighbours.rows.tolist()):
        first, end = int(new_offsets[new_idx]), int(new_offsets[new_idx + 1])
        block = in_out_ticks[first:end]
        if neighbours.stored_previous[row] and block[0] < neighbours.last_ticks[row]:
            floor = int(neighbours.last_ticks[row])
            stored = _stored_quantum(parsed, time_rows, int(neighbours.previous[row]), time_error)
            if raw_ticks[first] >= floor - max(int(quanta[new_idx]), stored) // 2:
                np.maximum(block, floor, out=block)
        following = int(neighbours.following[row])
        if neighbours.stored_following[row] and block[-1] > time_rows.starts[following]:
            ceiling = int(time_rows.starts[following])
            stored = _stored_quantum(parsed, time_rows, following, time_error)
            if raw_ticks[end - 1] <= ceiling + max(int(quanta[new_idx]), stored) // 2:
                np.minimum(block, ceiling, out=block)


def splice(
    parsed: ParsedGroup,
    indices: np.ndarray,
    samples: np.ndarray,
    block_sizes: np.ndarray,
    ticks: np.ndarray | None,
    params: Params,
    old_time_rows: _format.TimeRows | None = None,
    *,
    snap: bool = True,
) -> UpdatedGroup:
    """Replaces or appends blocks of a block group, carrying every other block over untouched.

    Untouched blocks keep their exact bytes in every field (_bitpacking.splice_body): they
    are neither unpacked, dequantized nor re-encoded. Blocks past the block group's end that
    indices skip are appended empty.

    Args:
        parsed: The existing block group.
        indices: 1D int64 array of distinct, increasing, non-negative block indices.
        samples: 1D float64 array of the new blocks' samples, in indices order.
        block_sizes: 1D int64 array of the new blocks' sizes.
        ticks: 1D int64 array of the new blocks' ticks (in the block group's time unit), required
            exactly when the block group has a time axis.
        params: Encoder parameters.
        old_time_rows: The block group's time rows if already read (the columns at least).
        snap: Whether to round the new blocks' ticks to params.time_error (False: the caller
            has). New blocks that overlap a stored neighbour by up to half the larger of their
            quanta are clamped to it (_clamp_to_neighbours).

    Returns:
        UpdatedGroup with the new block group and the statistics of every re-encoded block
        (including empty ones appended to fill a gap).

    Raises:
        ValueError: If the block group would be too large or its times would decrease.
    """
    num_old_blocks = parsed.header.num_blocks
    # Determine the total number of blocks after applying updates
    num_blocks = max(num_old_blocks, int(indices[-1]) + 1) if indices.shape[0] else num_old_blocks
    _args.check_group_counts(num_blocks, 0)
    # Appended blocks that indices skip become empty blocks; they hold no samples, so the
    # samples (and ticks) keep their order
    gaps = np.setdiff1d(np.arange(num_old_blocks, num_blocks), indices) if num_blocks > num_old_blocks else indices[:0]
    if gaps.shape[0]:
        merged = np.union1d(indices, gaps)
        merged_sizes = np.zeros(merged.shape[0], np.int64)
        merged_sizes[np.searchsorted(merged, indices)] = block_sizes
        indices, block_sizes = merged, merged_sizes

    # Assemble updated block sizes across the entire block group
    sizes = np.zeros(num_blocks, np.int64)
    sizes[:num_old_blocks] = parsed.block_sizes
    sizes[indices] = block_sizes
    num_samples = int(sizes.sum())
    _args.check_group_counts(num_blocks, num_samples)
    new_offsets = sample_offsets(block_sizes)
    if ticks is not None and snap and params.time_error > 0:
        if old_time_rows is None:
            old_time_rows = read_time_rows(parsed, np.zeros(0, np.int64))
        neighbours = _stored_neighbours(parsed, old_time_rows, sizes, indices)
        # A new block after a stored one keeps its phase if that fits
        seed_ticks = np.full(indices.shape[0], _time.INT64_MIN, np.int64)
        seeded = neighbours.rows[neighbours.stored_previous]
        seed_ticks[seeded] = neighbours.last_ticks[neighbours.stored_previous]
        raw_ticks = ticks
        ticks, quanta = snap_ticks(ticks, block_sizes, params.time_error, seed_ticks)
        if ticks is raw_ticks:
            ticks = ticks.copy()
        _clamp_to_neighbours(parsed, old_time_rows, neighbours, new_offsets, quanta, params.time_error, raw_ticks, ticks)

    # Encode value residuals and codes for the replacement blocks
    new_rows, stats = encode_rows(samples, block_sizes, params, ticks)
    old_starts = None
    if ticks is not None:
        # encode_rows encoded the replacement blocks' timestamp rows
        new_time_rows = new_rows.time_rows
        assert new_time_rows is not None
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
    # Repack and compress the spliced block group
    group = _compress.pack(
        body, offsets, num_blocks, num_samples, _compress.EFFORTS[params.effort], parsed.header.time_unit,
    )
    return UpdatedGroup(group, indices, *stats)


def update(
    group: bytes,
    blocks: Mapping[int, npt.ArrayLike],
    params: Params,
    times: Mapping[int, npt.ArrayLike] | None,
    time_unit: str | None = None,
) -> UpdatedGroup:
    """Validates update's arguments and splices the new blocks into the block group (see _api.update).

    Args:
        group: Compressed byte string of the original encoded block group.
        blocks: Mapping from block index to new sample values.
        params: Encoder parameters.
        times: Optional mapping from block index to new timestamps.
        time_unit: Optional time unit string specifying the expected timestamp resolution.

    Returns:
        UpdatedGroup containing the spliced block group bytes, updated indices, and compression statistics.

    Raises:
        ValueError: If the block group is corrupt, an index is invalid, or the times don't match the
            block group's time axis or the blocks.
        zstandard.ZstdError: If the zstd frame is corrupt.
    """
    parsed = decompress(group)
    _args.check_stored_time_unit(time_unit, parsed.header.time_unit)
    indices, samples, sizes, ticks = _args.update_blocks(blocks, times, parsed.header.time_unit)
    if not indices.shape[0]:
        return UpdatedGroup(group, np.zeros(0, np.int64), np.zeros(0), np.zeros(0), np.zeros(0))
    return splice(parsed, indices, samples, sizes, ticks, params)
