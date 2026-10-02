# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Building and parsing units: the layer between the public API and the kernels.

A unit's blocks are handled as flat arrays (every sample of the unit, block after block)
with per-block sizes, so blocks of any size share one code path. The public entry points
in `_api` divide their input into blocks and call `encode`, `decode` or `splice` here.
"""

import threading
from collections.abc import Callable, Mapping
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import zstandard

from . import _bitpacking, _decoder, _encoder, _format, _time
from ._types import DecodedUnit, EncodedUnit, Params, UpdatedUnit, is_int


class Effort(NamedTuple):
    """How a unit is compressed at one Params.level.

    Attributes:
        layout: "heuristic" (byte planes when few residuals reach 256, else bit planes, one
            compression), "best" (both, the smaller kept), "bit" or "byte" (tests only).
        flush: Whether a zstd block ends after the columns and each dense residual plane.
        zstd_level: The zstd compression level.
    """

    layout: str
    flush: bool
    zstd_level: int


EFFORTS: dict[int, Effort] = {
    1: Effort("heuristic", False, 1),
    2: Effort("heuristic", False, 3),
    **dict.fromkeys(range(3, 6), Effort("heuristic", True, 3)),
    **dict.fromkeys(range(6, 8), Effort("best", True, 3)),
    8: Effort("best", True, 7),
    9: Effort("best", True, 9),
}
"""Params.level to Effort (SPEC.md §1; the measured size and speed of each are in TUNING.md)."""

BYTE_PLANES_MAX_HIGH_SHARE: float = 0.05
"""The heuristic layout picks byte planes when fewer than this share of the residuals reach 256:
the high byte is then nearly constant, and byte-wise literals model the low byte well."""

SMALL_FRAME_BYTES: int = 16_384
"""A frame smaller than this is cheap to compress again, and is where the one-pass choices miss
most (per unit, not in total bytes): a block-flushed frame is also compressed in one block run, and
under the heuristic layout a unit is also compressed with the other layout; the smaller is kept."""

_local = threading.local()


def zstd(level: int = 3) -> tuple[zstandard.ZstdCompressor, zstandard.ZstdDecompressor]:
    """Retrieves thread-local zstandard compressor (at level) and decompressor instances.

    Returns:
        A tuple of (compressor, decompressor) dedicated to the current thread.
    """
    codecs = getattr(_local, "z", None)
    if codecs is None:
        # Separate compressors and decompressor per thread for thread-safety
        codecs = _local.z = {"decompressor": zstandard.ZstdDecompressor()}
    if level not in codecs:
        codecs[level] = zstandard.ZstdCompressor(level=level, write_checksum=False, write_content_size=True)
    return codecs[level], codecs["decompressor"]


def as_series(series_input: npt.ArrayLike, allow_empty: bool = False) -> np.ndarray:
    """Validates and converts input into a contiguous 1D float64 array.

    Args:
        series_input: Array-like input.
        allow_empty: Whether an empty array is accepted.

    Returns:
        Contiguous 1D float64 numpy array.

    Raises:
        ValueError: If series_input is not 1D, or is empty and allow_empty is False.
    """
    series_array = np.ascontiguousarray(series_input, dtype=np.float64)
    if series_array.ndim != 1 or (series_array.shape[0] == 0 and not allow_empty):
        raise ValueError(f"x must be a {'' if allow_empty else 'non-empty '}1-D array, got shape {series_array.shape}")
    return series_array


def as_ticks(times: npt.ArrayLike, time_unit: str | None, stored_unit: int = 0) -> tuple[np.ndarray, int]:
    """Converts timestamps into contiguous int64 ticks and their time unit code.

    Args:
        times: datetime64[s|ms|us|ns] array (the unit is taken from the dtype), or an
            integer array of ticks in time_unit.
        time_unit: Unit of integer ticks ('s', 'ms', 'us' or 'ns'); None for datetime64.
        stored_unit: For update: the unit's time unit code, which datetime64 times must
            match and integer ticks are in (time_unit, if given, must match it too). 0 when
            encoding.

    Returns:
        A tuple of (ticks, time_unit_code): a contiguous int64 array with the input's shape
        and the TimeUnit code.

    Raises:
        ValueError: If the dtype isn't datetime64[s|ms|us|ns] or integer, time_unit is
            missing for integer ticks or given for datetime64, the unit doesn't match
            stored_unit, or an unsigned tick exceeds int64.
    """
    times_array = np.asarray(times)
    if times_array.dtype.kind == "M":
        if time_unit is not None:
            raise ValueError("time_unit applies to integer times only: datetime64 times carry their own unit")
        dtype_unit, unit_count = np.datetime_data(times_array.dtype)
        if unit_count != 1 or dtype_unit not in _format.TIME_UNIT_CODES:
            raise ValueError(f"times must be datetime64 in s, ms, us or ns, got {times_array.dtype}")
        unit_code = _format.TIME_UNIT_CODES[dtype_unit]
        ticks = times_array.view(np.int64)
    elif times_array.dtype.kind in "iu" or (times_array.size == 0 and times_array.dtype.kind == "f"):
        # An empty list converts to float64: take it as integer ticks
        unit_code = time_unit_code(time_unit, stored_unit)
        if times_array.dtype.kind == "u" and times_array.size and int(times_array.max()) > _time.INT64_MAX:
            raise ValueError("integer times must fit in int64")
        ticks = times_array.astype(np.int64, copy=False)
    else:
        raise ValueError(
            f"times must be datetime64 (s, ms, us or ns) or integer ticks with time_unit, got dtype {times_array.dtype}"
            + (" (time zone aware times aren't supported: convert to naive UTC)" if times_array.dtype == object else "")
        )
    if stored_unit and unit_code != stored_unit:
        raise ValueError(
            f"times are in {_format.TIME_UNIT_NAMES[unit_code]} but the unit stores "
            f"{_format.TIME_UNIT_NAMES[stored_unit]}"
        )
    return np.ascontiguousarray(ticks), unit_code


def time_unit_code(time_unit: str | None, stored_unit: int = 0) -> int:
    """The TimeUnit code of integer ticks: time_unit's, or the stored unit's when it is None.

    Raises:
        ValueError: If time_unit is invalid, or None without a stored unit.
    """
    unit_names = "', '".join(_format.TIME_UNIT_CODES)
    if time_unit is None:
        if not stored_unit:
            raise ValueError(f"integer times need time_unit ('{unit_names}')")
        return stored_unit
    if time_unit not in _format.TIME_UNIT_CODES:
        raise ValueError(f"time_unit must be one of '{unit_names}', got {time_unit!r}")
    return _format.TIME_UNIT_CODES[time_unit]


def series_ticks(
    times: npt.ArrayLike | None, time_unit: str | None, num_samples: int
) -> tuple[np.ndarray | None, int]:
    """Converts encode's times argument into 1D int64 ticks (None without times).

    Raises:
        ValueError: If the times are invalid or don't have num_samples entries.
    """
    if times is None:
        if time_unit is not None:
            raise ValueError("time_unit needs times")
        return None, 0
    ticks, unit_code = as_ticks(times, time_unit)
    if ticks.shape != (num_samples,):
        raise ValueError(f"times must be 1-D with one entry per sample ({num_samples}), got shape {ticks.shape}")
    return ticks, unit_code


def as_block_sizes(block_sizes: npt.ArrayLike) -> np.ndarray:
    """Validates block sizes: a 1-D integer array of 0 to 65,535 each.

    Returns:
        Contiguous 1D int64 array.

    Raises:
        ValueError: If block_sizes isn't a 1-D integer array of sizes in range.
    """
    sizes = np.asarray(block_sizes)
    if sizes.ndim != 1 or not (np.issubdtype(sizes.dtype, np.integer) or sizes.shape[0] == 0):
        raise ValueError(f"block_sizes must be a 1-D integer array, got {sizes.dtype} of shape {sizes.shape}")
    sizes = np.ascontiguousarray(sizes, np.int64)
    if sizes.shape[0] and not (0 <= sizes.min() and sizes.max() <= _format.MAX_BLOCK_LEN):
        raise ValueError(f"block sizes must be from 0 to {_format.MAX_BLOCK_LEN}")
    return sizes


def check_unit_counts(num_blocks: int, num_samples: int, block_sizes: np.ndarray | None = None) -> None:
    """Checks that a unit's block and sample counts (and block sizes) are within the format's bounds.

    Raises:
        ValueError: If there are over 65,535 blocks or 2^26 samples, or a block holds over
            65,535 samples.
    """
    if block_sizes is not None and block_sizes.shape[0] and int(block_sizes.max()) > _format.MAX_BLOCK_LEN:
        raise ValueError(f"a block holds at most {_format.MAX_BLOCK_LEN} samples, got {int(block_sizes.max())}")
    if num_blocks > _format.MAX_BLOCKS:
        raise ValueError(f"a unit holds at most {_format.MAX_BLOCKS} blocks, got {num_blocks}")
    if num_samples > _format.MAX_UNIT_SAMPLES:
        raise ValueError(f"a unit holds at most {_format.MAX_UNIT_SAMPLES} samples, got {num_samples}")


def sample_offsets(block_sizes: np.ndarray) -> np.ndarray:
    """The blocks' sample offsets: 0, then the cumulative block sizes."""
    offsets = np.zeros(block_sizes.shape[0] + 1, np.int64)
    np.cumsum(block_sizes, out=offsets[1:])
    return offsets


class BlockStats(NamedTuple):
    """Per-block summary statistics returned by encoding."""

    block_min: np.ndarray
    block_max: np.ndarray
    block_mean: np.ndarray


def kernel_args(params: Params) -> tuple[int, int, int, float, float, bool, int]:
    """Packs encoder parameters into _encoder.encode_unit's arguments.

    Returns:
        A tuple of (min_bits, max_bits, orders_mask, noise_f, target, decimal, pick_len):
        orders_mask has bit k set for each enabled order, and 0.0 turns the noise floor or
        the target off.
    """
    return (
        params.min_quantize_bits,
        params.max_quantize_bits,
        sum(1 << order for order in params.diff_orders),
        float(params.noise_floor_sigma or 0.0),
        float(params.target_bits_per_sample or 0.0),
        bool(params.decimal_detection),
        _encoder.PICK_LEN,
    )


def encode_rows(samples: np.ndarray, block_sizes: np.ndarray, params: Params) -> tuple[_format.UnitRows, BlockStats]:
    """Encodes blocks into their rows (without a time axis) and summary statistics.

    Args:
        samples: 1D float64 array of every sample, block after block.
        block_sizes: 1D int64 array of block sizes adding up to the sample count.
        params: Encoder parameters.

    Returns:
        A tuple of (rows, stats).
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
        *kernel_args(params),
        rows.block_flags,
        rows.grid_params,
        rows.value_anchors,
        rows.residuals,
        rows.codes,
        *stats,
    )
    return rows, stats


def decrease_error(ticks: np.ndarray, sample_idx: int) -> ValueError:
    """The error for times that decrease at sample_idx (or contain NaT)."""
    if (ticks == _time.INT64_MIN).any():
        return ValueError("times contain NaT")
    return ValueError(
        f"times must be non-decreasing: sample {sample_idx} is {int(ticks[sample_idx])}, "
        f"below sample {sample_idx - 1} at {int(ticks[sample_idx - 1])}"
    )


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
        raise decrease_error(ticks, sample_idx)
    return time_rows


def compress(rows: _format.UnitRows, num_samples: int, effort: Effort, time_unit: int = 0) -> bytes:
    """Serializes unit rows and builds the unit: header plus zstd frame of the body.

    Args:
        rows: The unit's rows (time_rows None for a unit without a time axis).
        num_samples: Sample count recorded in the header (the sum of the block sizes).
        effort: How to compress (Params.level).
        time_unit: Time unit code recorded in the header (0 without a time axis).

    Returns:
        The unit bytes.
    """
    # Separate from the encode kernel; fusing measured no gain (PERFORMANCE.md).
    fields = (rows.block_flags, rows.block_sizes, rows.grid_params, rows.value_anchors, rows.residuals, rows.codes)
    return _pack(
        lambda byte_planes: _bitpacking.write_unit(*fields, byte_planes=byte_planes, time_rows=rows.time_rows),
        rows.block_flags.shape[0], num_samples, effort, time_unit,
    )


def compress_body(body: np.ndarray, cuts: list[int], zstd_level: int) -> bytes:
    """Compresses a unit body into one zstd frame, ending a block at each cut.

    Every zstd block has its own literal Huffman table, so cutting the body between its
    columns and planes codes each with its own statistics. The frame is an ordinary one that
    records the content size.

    Args:
        body: 1D uint8 array of the uncompressed body.
        cuts: Strictly increasing body offsets where a block ends.
        zstd_level: The zstd compression level.

    Returns:
        The frame.
    """
    compressor = zstd(zstd_level)[0].compressobj(size=body.shape[0])
    view = body.data
    parts = []
    previous = 0
    for cut in cuts:
        if previous < cut < body.shape[0]:
            parts.append(compressor.compress(view[previous:cut]))
            parts.append(compressor.flush(zstandard.COMPRESSOBJ_FLUSH_BLOCK))
            previous = cut
    parts.append(compressor.compress(view[previous:]))
    parts.append(compressor.flush())
    return b"".join(parts)


def _frame(body: np.ndarray, num_blocks: int, byte_planes: bool, has_time: bool, effort: Effort) -> bytes:
    """The zstd frame of a body: in one block run, or block-flushed at the columns and planes
    and then also in one block run when the flushed frame comes out small (SMALL_FRAME_BYTES)."""
    single = lambda: zstd(effort.zstd_level)[0].compress(body.data)
    if not effort.flush:
        return single()
    _, offsets = _format.read_layout(body, num_blocks)
    flushed = compress_body(body, _format.flush_points(body, num_blocks, offsets, has_time, byte_planes), effort.zstd_level)
    if len(flushed) < SMALL_FRAME_BYTES:
        one_run = single()
        return one_run if len(one_run) < len(flushed) else flushed
    return flushed


def _pack(
    write_body: Callable[[bool], np.ndarray], num_blocks: int, num_samples: int, effort: Effort, time_unit: int
) -> bytes:
    """The unit of the body write_body(byte_planes) builds, compressed as effort says."""
    has_time = time_unit != 0

    def build(byte_planes: bool, body: np.ndarray | None = None) -> bytes:
        header = _format.pack_header(num_blocks, num_samples, byte_planes, time_unit)
        body = write_body(byte_planes) if body is None else body
        return header + _frame(body, num_blocks, byte_planes, has_time, effort)

    if effort.layout == "heuristic":
        bit_body = write_body(False)
        _, offsets = _format.read_layout(bit_body, num_blocks)
        share = _bitpacking.high_byte_share(bit_body, num_blocks, int(offsets.group_offsets[-1]), has_time)
        byte_planes = share < BYTE_PLANES_MAX_HIGH_SHARE
        unit = build(True) if byte_planes else build(False, bit_body)
        if len(unit) - _format.HEADER_BYTES >= SMALL_FRAME_BYTES:
            return unit
        other = build(False, bit_body) if byte_planes else build(True)
        bit_unit, byte_unit = (other, unit) if byte_planes else (unit, other)
        return byte_unit if len(byte_unit) <= len(bit_unit) else bit_unit
    if effort.layout != "best":
        return build(effort.layout == "byte")
    bit_unit, byte_unit = build(False), build(True)
    # Ties go to byte planes (they decode faster: no bit transpose), so the choice is deterministic
    return byte_unit if len(byte_unit) <= len(bit_unit) else bit_unit


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
    check_unit_counts(block_sizes.shape[0], samples.shape[0], block_sizes)
    rows, stats = encode_rows(samples, block_sizes, params)
    if ticks is not None:
        rows = rows._replace(time_rows=encode_time_rows(ticks, block_sizes, rows.block_flags))
    return EncodedUnit(compress(rows, samples.shape[0], EFFORTS[params.level], time_unit), *stats)


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
    raw_body = np.frombuffer(zstd()[1].decompress(frame), np.uint8)
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
        parsed.raw_body, offsets.sample_offsets, offsets.group_offsets, offsets.code_offsets, block_ids, values,
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
    """The datetime64 dtype of a time unit code."""
    return np.dtype(f"datetime64[{_format.TIME_UNIT_NAMES[time_unit]}]")


def decode(unit: bytes) -> DecodedUnit:
    """Decodes a unit into its samples, timestamps and block sizes (see decode_unit)."""
    parsed = decompress(unit)
    values, ticks = decode_blocks(parsed, np.arange(parsed.header.num_blocks, dtype=np.int64))
    times = None if ticks is None else ticks.view(time_dtype(parsed.header.time_unit))
    return DecodedUnit(values, times, parsed.block_sizes)


def read_rows(parsed: ParsedUnit, time_rows: _format.TimeRows | None = None) -> _format.UnitRows:
    """Reads every block's rows (residuals and codes, without dequantizing) and time rows
    (unless already read: time_rows)."""
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
        parsed.raw_body, parsed.header.byte_planes, parsed.has_time, offsets.sample_offsets, offsets.group_offsets,
        offsets.code_offsets, rows.grid_params, rows.value_anchors, rows.residuals, rows.codes,
    )
    return rows


def _last_tick(parsed: ParsedUnit, time_rows: _format.TimeRows, block_idx: int) -> int:
    """The last tick of a non-empty block of a parsed unit, unpacking and expanding that block only.

    Args:
        parsed: The unit.
        time_rows: Its time rows: the columns at least (the block's residuals are unpacked
            into them).
        block_idx: The block.
    """
    offsets = parsed.layout.sample_offsets
    first, end = int(offsets[block_idx]), int(offsets[block_idx + 1])
    block_ids = np.array([block_idx], np.int64)
    _bitpacking.read_time_residuals(parsed.raw_body, *parsed.layout, block_ids, time_rows.residuals)
    block_rows = _format.TimeRows(
        time_rows.starts[block_idx:block_idx + 1], time_rows.steps[block_idx:block_idx + 1],
        time_rows.refs[block_idx:block_idx + 1], time_rows.residuals[first:end],
    )
    out_ticks = np.empty(end - first, np.int64)
    expand_times(parsed.block_flags[block_idx:block_idx + 1], np.array([0, end - first], np.int64), block_rows,
                 np.zeros(1, np.int64), out_ticks)
    return int(out_ticks[-1])


def _check_spliced_order(
    block_sizes: np.ndarray, block_starts: np.ndarray, is_new: np.ndarray, new_last_ticks: dict[int, int],
    old_last_tick: Callable[[int], int],
) -> None:
    """Checks that the times never decrease where a new block meets its non-empty neighbours.

    Args:
        block_sizes: 1D int64 array of the spliced unit's block sizes.
        block_starts: 1D int64 array of its block start times.
        is_new: 1D bool array marking the new blocks.
        new_last_ticks: Last tick of each non-empty new block.
        old_last_tick: The last tick of a carried block, by index.

    Raises:
        ValueError: If a block starts before the previous non-empty block's last time.
    """
    filled = np.flatnonzero(block_sizes)
    # Only pairs of consecutive non-empty blocks where one is new: the others were in order
    pairs = np.flatnonzero(is_new[filled[:-1]] | is_new[filled[1:]])
    for previous_idx, block_idx in zip(filled[pairs].tolist(), filled[pairs + 1].tolist()):
        last = new_last_ticks[previous_idx] if is_new[previous_idx] else old_last_tick(previous_idx)
        first = int(block_starts[block_idx])
        if first < last:
            raise ValueError(
                f"times must be non-decreasing: block {block_idx} starts at {first}, "
                f"below block {previous_idx}'s last time {last}"
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
    num_blocks = max(num_old_blocks, int(indices[-1]) + 1) if indices.shape[0] else num_old_blocks
    check_unit_counts(num_blocks, 0)
    # Appended blocks that indices skip become empty blocks; they hold no samples, so the
    # samples (and ticks) keep their order
    gaps = np.setdiff1d(np.arange(num_old_blocks, num_blocks), indices) if num_blocks > num_old_blocks else indices[:0]
    if gaps.shape[0]:
        merged = np.union1d(indices, gaps)
        merged_sizes = np.zeros(merged.shape[0], np.int64)
        merged_sizes[np.searchsorted(merged, indices)] = block_sizes
        indices, block_sizes = merged, merged_sizes
    sizes = np.zeros(num_blocks, np.int64)
    sizes[:num_old_blocks] = parsed.block_sizes
    sizes[indices] = block_sizes
    num_samples = int(sizes.sum())
    check_unit_counts(num_blocks, num_samples)
    new_rows, stats = encode_rows(samples, block_sizes, params)
    old_starts = None
    if ticks is not None:
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
        new_offsets = sample_offsets(block_sizes)
        new_last_ticks = {
            int(b): int(ticks[new_offsets[rank + 1] - 1]) for rank, b in enumerate(indices) if block_sizes[rank]
        }
        old_rows = old_time_rows
        _check_spliced_order(sizes, starts, is_new, new_last_ticks,
                             lambda block_idx: _last_tick(parsed, old_rows, block_idx))
    unit = _pack(
        lambda byte_planes: _bitpacking.splice_body(
            parsed.raw_body, parsed.layout, parsed.header.byte_planes, old_starts, indices, new_rows, byte_planes
        ),
        num_blocks, num_samples, EFFORTS[params.level], parsed.header.time_unit,
    )
    return UpdatedUnit(unit, indices, *stats)


def update(
    unit: bytes,
    blocks: Mapping[int, npt.ArrayLike],
    params: Params,
    times: Mapping[int, npt.ArrayLike] | None,
) -> UpdatedUnit:
    """Validates update's arguments and splices the new blocks into the unit (see _api.update).

    Raises:
        ValueError: If the unit is corrupt, an index is invalid, or the times don't match the
            unit's time axis or the blocks.
        zstandard.ZstdError: If the zstd frame is corrupt.
    """
    parsed = decompress(unit)
    if not isinstance(blocks, Mapping):
        raise ValueError(f"blocks must map block indices to samples, got {type(blocks).__name__}")  # noqa: TRY004
    if not all(is_int(idx) and idx >= 0 for idx in blocks):
        raise ValueError(f"block indices must be integers >= 0, got {sorted(map(repr, blocks))}")
    indices = sorted(int(idx) for idx in blocks)
    # Before anything is sized by the indices
    if indices and indices[-1] >= _format.MAX_BLOCKS:
        raise ValueError(f"a unit holds at most {_format.MAX_BLOCKS} blocks, got block index {indices[-1]}")
    if parsed.has_time:
        if times is None:
            raise ValueError("the unit has a time axis: times are required")
        if sorted(int(idx) for idx in times) != indices:
            raise ValueError("times must have the same block indices as blocks")
    elif times is not None:
        raise ValueError("the unit has no time axis: times must be None")
    by_index = {int(idx): block for idx, block in blocks.items()}
    samples = [as_series(by_index[idx], allow_empty=True) for idx in indices]
    sizes = as_block_sizes([block.shape[0] for block in samples]) if samples else np.zeros(0, np.int64)
    ticks = None
    if times is not None:
        times_by_index = {int(idx): block_times for idx, block_times in times.items()}
        block_ticks = [as_ticks(times_by_index[idx], None, parsed.header.time_unit)[0] for idx in indices]
        for idx, block, tick_block in zip(indices, samples, block_ticks):
            if tick_block.shape != block.shape:
                raise ValueError(f"block {idx}: times must have the shape of its samples {block.shape}, got {tick_block.shape}")
        ticks = np.concatenate(block_ticks) if block_ticks else np.zeros(0, np.int64)
    if not indices:
        return UpdatedUnit(unit, np.zeros(0, np.int64), np.zeros(0), np.zeros(0), np.zeros(0))
    return splice(
        parsed, np.array(indices, np.int64), np.concatenate(samples), sizes, ticks, params
    )
