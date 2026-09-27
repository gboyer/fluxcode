# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Public API for fluxcode encoding, decoding, and updating time series.

Provides entry points for compressing float64 arrays, optionally with their
timestamps, into independent, self-describing units (storage rows) and
decompressing them back. Encoding also returns per-block summary statistics
(min, max, and mean), which decoding doesn't need.
"""

import threading
from collections.abc import Iterable, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from typing import Literal, NamedTuple

import numpy as np
import numpy.typing as npt
import zstandard

from . import _decoder, _encoder, _format, _time

ZSTD_LEVEL: int = 3
"""Zstandard compression level used for encoding units."""

MIN_TARGET_BITS: float = 6.0
"""Minimum allowed per-unit bit target per sample."""

TimeUnit = Literal["s", "ms", "us", "ns"]
"""Resolution of integer timestamps: seconds, milliseconds, microseconds or nanoseconds."""


class EncodedUnit(NamedTuple):
    """The compressed bytes and per-block summary statistics of a single unit.

    Attributes:
        unit: Self-describing unit bytes (header and zstd frame).
        block_min: 1D float64 array of minimum finite values per block.
        block_max: 1D float64 array of maximum finite values per block.
        block_mean: 1D float64 array of finite sample means per block.
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
    """The samples and timestamps of a decoded unit.

    Attributes:
        values: 1D float64 array of the unit's samples.
        times: 1D datetime64 array of the samples' timestamps, in the unit they were
            encoded with (s, ms, us or ns); None for a unit encoded without times.
    """

    values: np.ndarray
    times: np.ndarray | None


class UpdatedUnit(NamedTuple):
    """The updated unit bytes and summary statistics for modified or appended blocks.

    Attributes:
        unit: New compressed unit bytes.
        block_min: 1D float64 array of minima for updated blocks in indices order.
        block_max: 1D float64 array of maxima for updated blocks in indices order.
        block_mean: 1D float64 array of means for updated blocks in indices order.
    """

    unit: bytes
    block_min: np.ndarray
    block_max: np.ndarray
    block_mean: np.ndarray


@dataclass(frozen=True)
class Params:
    """Encoder configuration parameters.

    Controls quantization step limits, difference predictor orders, noise
    floor gating, per-unit soft bit targets, and block geometry. Decoders do not
    require these parameters because units are self-describing.

    Attributes:
        min_quantize_bits: Hard lower bound on precision (coarsest step across
            the range, 1..16).
        max_quantize_bits: Hard upper bound on precision (finest step across
            the range, min..16).
        diff_orders: Non-empty subset of {0, 1, 2, 3} specifying predictor
            difference orders evaluated by the encoder.
        noise_floor_sigma: Noise floor multiplier f (or None to disable). When
            active, steps on white-noise blocks coarsen up to f * sigma.
        target_bits_per_sample: Soft per-unit cap on compressed bits per sample
            (must be >= 6.0 if set, or None to disable).
        decimal_detection: Whether to test for exact decimal grids (10^p) before
            falling back to power-of-two grids.
        block_len: Number of samples per block (must be a multiple of 8, at most
            65536).
        blocks_per_unit: Maximum number of blocks in a single compressed unit row
            (must be >= 1).
        try_byte_planes: Also compress each unit with byte planes instead of bit
            planes and keep the smaller. Doubles the zstd work of encoding; pays
            off on clean periodic signals whose cycles repeat across a unit.
    """

    min_quantize_bits: int = 6
    max_quantize_bits: int = 16
    # Any set or sequence of orders is accepted; __post_init__ stores it as a frozenset.
    diff_orders: AbstractSet[int] | Sequence[int] = frozenset({0, 1, 2, 3})
    noise_floor_sigma: float | None = 0.25
    target_bits_per_sample: float | None = None
    decimal_detection: bool = True
    block_len: int = 1000
    blocks_per_unit: int = 60
    try_byte_planes: bool = False

    def __post_init__(self) -> None:
        """Validates parameter types, domains, and structural constraints.

        Raises:
            ValueError: If any parameter falls outside its defined domain.
        """
        min_bits = self.min_quantize_bits
        max_bits = self.max_quantize_bits
        # Verify quantization bit bounds: 1 <= min <= max <= 16
        if not (_is_int(min_bits) and _is_int(max_bits) and 1 <= min_bits <= max_bits <= 16):
            raise ValueError(
                f"need 1 <= min_quantize_bits <= max_quantize_bits <= 16, got {min_bits!r}, {max_bits!r}"
            )
        orders = self.diff_orders
        if not isinstance(orders, Iterable) or isinstance(orders, (str, bytes)):
            raise ValueError(f"diff_orders must be a set of orders, got {orders!r}")  # noqa: TRY004
        frozen_orders = frozenset(orders)
        # Ensure diff_orders is a non-empty subset of {0, 1, 2, 3}
        if not frozen_orders or not all(_is_int(order) and 0 <= order <= 3 for order in frozen_orders):
            raise ValueError(f"diff_orders must be a non-empty subset of {{0, 1, 2, 3}}, got {set(frozen_orders)!r}")
        object.__setattr__(self, "diff_orders", frozenset(int(order) for order in frozen_orders))
        noise_factor = self.noise_floor_sigma
        # Validate noise floor multiplier
        if noise_factor is not None and not (_is_real(noise_factor) and noise_factor >= 0 and np.isfinite(noise_factor)):
            raise ValueError(f"noise_floor_sigma must be None or a finite number >= 0, got {noise_factor!r}")
        target_bits = self.target_bits_per_sample
        # Validate soft target bit rate
        if target_bits is not None and not (_is_real(target_bits) and MIN_TARGET_BITS <= target_bits and np.isfinite(target_bits)):
            raise ValueError(
                f"target_bits_per_sample must be None or a finite number >= {MIN_TARGET_BITS:g}, got {target_bits!r}"
            )
        # Block length must be a multiple of 8 within allowable format limits
        if not (_is_int(self.block_len) and 0 < self.block_len <= _format.MAX_BLOCK_LEN and self.block_len % 8 == 0):
            raise ValueError(
                f"block_len must be a multiple of 8 from 8 to {_format.MAX_BLOCK_LEN}, got {self.block_len!r}"
            )
        # Blocks per unit must be at least 1
        if not (_is_int(self.blocks_per_unit) and self.blocks_per_unit >= 1):
            raise ValueError(f"blocks_per_unit must be >= 1, got {self.blocks_per_unit!r}")
        if not isinstance(self.try_byte_planes, (bool, np.bool_)):
            raise ValueError(f"try_byte_planes must be a bool, got {self.try_byte_planes!r}")  # noqa: TRY004
        # A full unit must fit the decoder's sanity bound on sample counts
        if self.blocks_per_unit * self.block_len > _format.MAX_UNIT_SAMPLES:
            raise ValueError(
                f"blocks_per_unit * block_len must be at most {_format.MAX_UNIT_SAMPLES}, "
                f"got {self.blocks_per_unit} * {self.block_len}"
            )

    def _kernel_args(self) -> tuple[int, int, int, float, float, bool, int]:
        """Packs encoder configuration into kernel arguments.

        Returns:
            A tuple of (min_bits, max_bits, orders_mask, noise_f, target,
            decimal, pick_len) suitable for numba kernels.
        """
        # Convert diff_orders set to a bitmask (bit k set if order k enabled)
        orders_mask = sum(1 << order for order in self.diff_orders)
        return (
            self.min_quantize_bits,
            self.max_quantize_bits,
            orders_mask,
            float(self.noise_floor_sigma or 0.0),
            float(self.target_bits_per_sample or 0.0),
            bool(self.decimal_detection),
            _encoder.PICK_LEN,
        )


def _is_int(val: object) -> bool:
    """Checks whether a value is an integer type (excluding booleans).

    Args:
        val: Value to inspect.

    Returns:
        True if val is an integer and not a bool, False otherwise.
    """
    return isinstance(val, (int, np.integer)) and not isinstance(val, bool)


def _is_real(val: object) -> bool:
    """Checks whether a value is a real numeric type (excluding booleans).

    Args:
        val: Value to inspect.

    Returns:
        True if val is an int or float and not a bool, False otherwise.
    """
    return isinstance(val, (int, float, np.integer, np.floating)) and not isinstance(val, bool)


_local = threading.local()


DEFAULT_PARAMS: Params = Params()
"""Default encoder parameters (immutable, shared as the default argument)."""


def _zstd() -> tuple[zstandard.ZstdCompressor, zstandard.ZstdDecompressor]:
    """Retrieves thread-local zstandard compressor and decompressor instances.

    Returns:
        A tuple of (compressor, decompressor) dedicated to the current thread.
    """
    cached_codec = getattr(_local, "z", None)
    if cached_codec is None:
        # Create separate compressor and decompressor per thread for thread-safety
        cached_codec = _local.z = (
            zstandard.ZstdCompressor(level=ZSTD_LEVEL, write_checksum=False, write_content_size=True),
            zstandard.ZstdDecompressor(),
        )
    return cached_codec


def _encode_blocks(
    block_matrix: np.ndarray, params: Params, num_real_samples_last_block: int | None = None
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Encodes a 2D array of blocks into raw unit arrays and summary statistics.

    Args:
        block_matrix: 2D float64 array of shape (num_blocks, block_len).
        params: Encoder parameters.
        num_real_samples_last_block: Real sample count in the terminal block (all if None).

    Returns:
        A tuple of (block_flags, grid_params, value_anchors, residuals, codes, minima,
        maxima, means).
    """
    num_blocks, block_len = block_matrix.shape
    out_block_flags = np.empty(num_blocks, np.uint8)
    out_grid_params = np.empty(num_blocks, np.int64)
    out_value_anchors = np.empty(num_blocks, np.int64)
    out_residuals = np.empty((num_blocks, block_len), np.int16)
    out_codes = np.empty((num_blocks, block_len), np.uint8)
    out_minima = np.empty(num_blocks)
    out_maxima = np.empty(num_blocks)
    out_means = np.empty(num_blocks)
    # Invoke numba kernel to process all blocks of the unit
    _encoder.encode_unit(
        block_matrix,
        block_len if num_real_samples_last_block is None else num_real_samples_last_block,
        *params._kernel_args(),
        out_block_flags,
        out_grid_params,
        out_value_anchors,
        out_residuals,
        out_codes,
        out_minima,
        out_maxima,
        out_means,
    )
    return (
        out_block_flags,
        out_grid_params,
        out_value_anchors,
        out_residuals,
        out_codes,
        out_minima,
        out_maxima,
        out_means,
    )


def _compress(
    block_flags: np.ndarray,
    grid_params: np.ndarray,
    value_anchors: np.ndarray,
    residuals: np.ndarray,
    codes: np.ndarray,
    num_samples: int,
    try_byte_planes: bool = False,
    time_rows: _format.TimeRows | None = None,
    time_unit: int = 0,
) -> bytes:
    """Serializes unit fields and builds the unit: header plus zstd frame of the body.

    Args:
        block_flags: 1D uint8 array of block flags.
        grid_params: 1D int64 array of block grid parameters.
        value_anchors: 1D int64 array of block value anchors.
        residuals: 2D int16 array of block residuals, shape (num_blocks, block_len).
        codes: 2D uint8 array of sample codes.
        num_samples: Real sample count recorded in the header.
        try_byte_planes: Also build the unit with byte planes and return the smaller.
        time_rows: The time axis rows, or None for a unit without a time axis.
        time_unit: Time unit code recorded in the header (0 without a time axis).

    Returns:
        The unit bytes.
    """
    # Separate from the encode kernel (update shares it); fusing measured no gain (PERFORMANCE.md).
    block_len = residuals.shape[1]
    body = _format.write_unit(block_flags, grid_params, value_anchors, residuals, codes, time_rows=time_rows)
    unit = _format.pack_header(block_len, num_samples, False, time_unit) + _zstd()[0].compress(body.data)
    if try_byte_planes:
        body = _format.write_unit(
            block_flags, grid_params, value_anchors, residuals, codes, byte_planes=True, time_rows=time_rows
        )
        byte_unit = _format.pack_header(block_len, num_samples, True, time_unit) + _zstd()[0].compress(body.data)
        # Ties keep bit planes, so the choice is deterministic
        if len(byte_unit) < len(unit):
            return byte_unit
    return unit


def _as_series(series_input: npt.ArrayLike) -> np.ndarray:
    """Validates and converts input into a contiguous 1D float64 array.

    Args:
        series_input: Array-like input.

    Returns:
        Contiguous 1D float64 numpy array.

    Raises:
        ValueError: If series_input is not 1D or is empty.
    """
    series_array = np.ascontiguousarray(series_input, dtype=np.float64)
    if series_array.ndim != 1 or series_array.shape[0] == 0:
        raise ValueError(f"x must be a non-empty 1-D array, got shape {series_array.shape}")
    return series_array


def _as_ticks(times: npt.ArrayLike, time_unit: str | None, stored_unit: int = 0) -> tuple[np.ndarray, int]:
    """Converts timestamps into contiguous int64 ticks and their time unit code.

    Args:
        times: datetime64[s|ms|us|ns] array (the unit is taken from the dtype), or an
            integer array of ticks in time_unit.
        time_unit: Unit of integer ticks ('s', 'ms', 'us' or 'ns'); None for datetime64.
        stored_unit: For update: the unit's time unit code, which datetime64 times must
            match and integer ticks are in (time_unit must then be None). 0 when encoding.

    Returns:
        A tuple of (ticks, time_unit_code): a contiguous int64 array with the input's shape
        and the TimeUnit code.

    Raises:
        ValueError: If the dtype isn't datetime64[s|ms|us|ns] or integer, time_unit is
            missing for integer ticks or given for datetime64, the unit doesn't match
            stored_unit, or an unsigned tick exceeds int64.
    """
    times_array = np.asarray(times)
    unit_names = "', '".join(_format.TIME_UNIT_CODES)
    if times_array.dtype.kind == "M":
        if time_unit is not None:
            raise ValueError("time_unit applies to integer times only: datetime64 times carry their own unit")
        dtype_unit, unit_count = np.datetime_data(times_array.dtype)
        if unit_count != 1 or dtype_unit not in _format.TIME_UNIT_CODES:
            raise ValueError(f"times must be datetime64 in s, ms, us or ns, got {times_array.dtype}")
        unit_code = _format.TIME_UNIT_CODES[dtype_unit]
        ticks = times_array.view(np.int64)
    elif times_array.dtype.kind in "iu":
        if stored_unit:
            unit_code = stored_unit
        elif time_unit is None:
            raise ValueError(f"integer times need time_unit ('{unit_names}')")
        elif time_unit not in _format.TIME_UNIT_CODES:
            raise ValueError(f"time_unit must be one of '{unit_names}', got {time_unit!r}")
        else:
            unit_code = _format.TIME_UNIT_CODES[time_unit]
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


def _pad_ticks(ticks: np.ndarray, num_blocks: int, block_len: int) -> np.ndarray:
    """Pads ticks to whole blocks: extending a regular last block's grid, else repeating.

    Args:
        ticks: 1D int64 array of ticks, at most num_blocks * block_len.
        num_blocks: Number of blocks.
        block_len: Samples per block.

    Returns:
        2D int64 array of shape (num_blocks, block_len).
    """
    num_ticks = ticks.shape[0]
    padded_len = num_blocks * block_len
    if num_ticks == padded_len:
        return ticks.reshape(num_blocks, block_len)
    block_times = np.empty(padded_len, np.int64)
    block_times[:num_ticks] = ticks
    last_block = ticks[(num_blocks - 1) * block_len:]
    last_tick = int(last_block[-1])
    step = 0
    if last_block.shape[0] >= 2:
        steps = np.diff(last_block)
        # Extend the grid only if the real samples are regular and the extension fits int64
        if (steps == steps[0]).all() and steps[0] > 0 and last_tick + int(steps[0]) * (padded_len - num_ticks) <= _time.INT64_MAX:
            step = int(steps[0])
    block_times[num_ticks:] = last_tick + step * np.arange(1, padded_len - num_ticks + 1, dtype=np.int64)
    return block_times.reshape(num_blocks, block_len)


def _decrease_error(ticks: np.ndarray, sample_idx: int) -> ValueError:
    """The error for times that decrease at sample_idx (or contain NaT)."""
    if (ticks == _time.INT64_MIN).any():
        return ValueError("times contain NaT")
    return ValueError(
        f"times must be non-decreasing: sample {sample_idx} is {int(ticks[sample_idx])}, "
        f"below sample {sample_idx - 1} at {int(ticks[sample_idx - 1])}"
    )


def _encode_time_rows(block_times: np.ndarray, out_block_flags: np.ndarray) -> _format.TimeRows:
    """Analyzes padded ticks into time rows, checking that they never decrease.

    Args:
        block_times: 2D int64 array of shape (num_blocks, block_len) of padded ticks.
        out_block_flags: In-out 1D uint8 array of block flags (the irregular time bit is set).

    Returns:
        The time rows.

    Raises:
        ValueError: If the times contain NaT or decrease anywhere.
    """
    num_blocks, block_len = block_times.shape
    flat_ticks = block_times.reshape(-1)
    if flat_ticks[0] == _time.INT64_MIN:
        raise ValueError("times contain NaT")
    # Across blocks: each block starts at or after the previous block's last time
    across = np.flatnonzero(block_times[1:, 0] < block_times[:-1, -1])
    if across.shape[0]:
        raise _decrease_error(flat_ticks, int(across[0] + 1) * block_len)
    # Zeroed: the kernel leaves regular blocks' delta rows untouched
    time_rows = _format.TimeRows(
        np.empty(num_blocks, np.int64), np.empty(num_blocks, np.int64), np.zeros((num_blocks, block_len), np.uint64)
    )
    status, sample_idx = _time.encode_times(
        block_times, out_block_flags, time_rows.starts, time_rows.steps, time_rows.deltas
    )
    if status != _time.OK:
        raise _decrease_error(flat_ticks, sample_idx)
    return time_rows


def _encode_rows(
    series_samples: np.ndarray, params: Params, ticks: np.ndarray | None = None, time_unit: int = 0
) -> EncodedUnit:
    """Encodes samples fitting into a single unit, padding short terminal blocks.

    Args:
        series_samples: 1D float64 array of samples (at most blocks_per_unit * block_len).
        params: Encoder parameters.
        ticks: 1D int64 array of the samples' timestamps, or None for no time axis.
        time_unit: Time unit code of ticks (0 without a time axis).

    Returns:
        EncodedUnit holding compressed unit bytes and block min/max/mean.

    Raises:
        ValueError: If the times contain NaT or decrease anywhere.
    """
    block_len = params.block_len
    total_samples = series_samples.shape[0]
    # Ceiling division: compute required block count
    num_blocks = -(-total_samples // block_len)
    if total_samples == num_blocks * block_len:
        # Reshape directly if input aligns with block boundaries
        block_matrix = series_samples.reshape(num_blocks, block_len)
    else:
        # Pad short terminal block by repeating the final sample
        block_matrix = np.empty(num_blocks * block_len)
        block_matrix[:total_samples] = series_samples
        block_matrix[total_samples:] = series_samples[-1]
        block_matrix = block_matrix.reshape(num_blocks, block_len)
    # Number of valid samples in the padded terminal block
    num_real_samples_last_block = total_samples - (num_blocks - 1) * block_len
    (
        block_flags,
        grid_params,
        value_anchors,
        residuals,
        codes,
        block_min,
        block_max,
        block_mean,
    ) = _encode_blocks(block_matrix, params, num_real_samples_last_block)
    time_rows = None
    if ticks is not None:
        time_rows = _encode_time_rows(_pad_ticks(ticks, num_blocks, block_len), block_flags)
    unit = _compress(
        block_flags,
        grid_params,
        value_anchors,
        residuals,
        codes,
        total_samples,
        params.try_byte_planes,
        time_rows,
        time_unit,
    )
    return EncodedUnit(unit, block_min, block_max, block_mean)


def encode_unit(
    x: npt.ArrayLike,
    params: Params = DEFAULT_PARAMS,
    *,
    times: npt.ArrayLike | None = None,
    time_unit: TimeUnit | None = None,
) -> EncodedUnit:
    """Encodes a time series into a single compressed unit (one storage row).

    Encodes up to `params.blocks_per_unit` blocks of `params.block_len` samples.
    Pads a partial final block by repeating its last sample. The unit records its
    sample count and block length, so decode_unit needs nothing else. Non-finite
    values (NaN, +inf, -inf) are encoded exactly (canonical quiet NaN).

    With times, the unit also stores the samples' timestamps exactly: naive (no time
    zone; UTC is recommended, since local time can repeat or skip), non-decreasing, and
    without NaT. Equal consecutive timestamps are allowed.

    Args:
        x: 1D array-like of float64 samples.
        params: Encoder configuration parameters.
        times: Optional timestamps, one per sample: a datetime64[s|ms|us|ns] array (the
            unit is taken from the dtype), or integer ticks with time_unit.
        time_unit: Unit of integer times ('s', 'ms', 'us' or 'ns'); must be None for
            datetime64 times.

    Returns:
        EncodedUnit tuple (unit, block_min, block_max, block_mean) containing:
            unit: Self-describing unit bytes.
            block_min: 1D float64 array of minimum finite values per block.
            block_max: 1D float64 array of maximum finite values per block.
            block_mean: 1D float64 array of finite means per block (terminal
                block mean covers only real samples).
            Blocks containing only non-finite samples receive NaN for min, max,
            and mean. These summary statistics come free with encoding; decoding
            doesn't use them.

    Raises:
        ValueError: If x requires more blocks than params.blocks_per_unit or is
            empty, or the times are invalid (see above) or don't match x in length.
    """
    series_arr = _as_series(x)
    # Calculate required blocks
    required_blocks = -(-series_arr.shape[0] // params.block_len)
    if required_blocks > params.blocks_per_unit:
        raise ValueError(
            f"x needs {required_blocks} blocks, over blocks_per_unit={params.blocks_per_unit}; use encode()"
        )
    ticks, time_unit_code = _series_ticks(times, time_unit, series_arr.shape[0])
    return _encode_rows(series_arr, params, ticks, time_unit_code)


def _series_ticks(
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
    ticks, time_unit_code = _as_ticks(times, time_unit)
    if ticks.shape != (num_samples,):
        raise ValueError(f"times must be 1-D with one entry per sample ({num_samples}), got shape {ticks.shape}")
    return ticks, time_unit_code


def encode(
    x: npt.ArrayLike,
    params: Params = DEFAULT_PARAMS,
    *,
    times: npt.ArrayLike | None = None,
    time_unit: TimeUnit | None = None,
) -> EncodedSeries:
    """Encodes an arbitrary-length series into a sequence of compressed units.

    Partitions the input series (and its times, if given) into chunks of
    `params.blocks_per_unit` blocks and encodes each chunk as an independent unit
    via `encode_unit`.

    Args:
        x: 1D array-like of float64 samples.
        params: Encoder configuration parameters.
        times: Optional timestamps, one per sample (see encode_unit).
        time_unit: Unit of integer times (see encode_unit).

    Returns:
        EncodedSeries tuple (units, block_mins, block_maxs, block_means):
            units: List of compressed unit byte strings.
            block_mins: List of 1D float64 arrays of block minima.
            block_maxs: List of 1D float64 arrays of block maxima.
            block_means: List of 1D float64 arrays of block means.

    Raises:
        ValueError: If x is empty, or the times are invalid or don't match x in length.
    """
    series_arr = _as_series(x)
    ticks, time_unit_code = _series_ticks(times, time_unit, series_arr.shape[0])
    # Samples per full unit
    samples_per_unit = params.blocks_per_unit * params.block_len
    parts = [
        _encode_rows(
            series_arr[idx:idx + samples_per_unit],
            params,
            None if ticks is None else ticks[idx:idx + samples_per_unit],
            time_unit_code,
        )
        for idx in range(0, series_arr.shape[0], samples_per_unit)
    ]
    units, mins, maxs, means = (list(col) for col in zip(*parts))
    return EncodedSeries(units, mins, maxs, means)


def _decompress(unit: bytes) -> tuple[np.ndarray, _format.UnitHeader]:
    """Parses a unit's header, decompresses its body and validates the layout.

    Args:
        unit: Unit bytes (header and zstd frame).

    Returns:
        A tuple of (raw_body, header).

    Raises:
        ValueError: If the header is invalid, the frame's content size is missing
            or doesn't fit the header, or block validation fails.
        zstandard.ZstdError: If the zstd frame is corrupt.
    """
    header = _format.unpack_header(unit)
    num_blocks, block_len = header.num_blocks, header.block_len
    has_time = header.time_unit != 0
    frame = memoryview(unit)[_format.HEADER_BYTES:]
    # Check the recorded body size against the header before allocating it
    content_size = zstandard.frame_content_size(frame)
    if content_size < 0:
        raise ValueError("unit's zstd frame doesn't record its content size")
    smallest = _format.unit_size(num_blocks, block_len, 0, has_time, 0)
    largest = _format.unit_size(num_blocks, block_len, num_blocks, has_time, num_blocks if has_time else 0)
    if not smallest <= content_size <= largest:
        raise ValueError(f"unit body of {content_size} bytes doesn't fit {num_blocks} blocks of {block_len}")
    raw_body = np.frombuffer(_zstd()[1].decompress(frame), np.uint8)
    # The exact size depends on how many blocks carry non-finite code planes and time delta planes
    num_flagged = _format.count_flagged(raw_body, num_blocks)
    num_irregular = _format.count_irregular(raw_body, num_blocks)
    if raw_body.shape[0] != _format.unit_size(num_blocks, block_len, num_flagged, has_time, num_irregular):
        raise ValueError(
            f"unit body of {raw_body.shape[0]} bytes doesn't match its {num_flagged} flagged blocks "
            f"and {num_irregular} irregular time blocks"
        )
    _check_unit(raw_body, num_blocks, has_time)
    return raw_body, header


def _check_unit(raw_body: np.ndarray, num_blocks: int, has_time: bool) -> None:
    """Verifies that block flags, grid parameters and value anchors conform to format bounds.

    Args:
        raw_body: 1D uint8 array of uncompressed body bytes.
        num_blocks: Number of blocks in the unit.
        has_time: Whether the unit has a time axis.

    Raises:
        ValueError: If any block's flags set reserved bits (or the irregular time bit
            without a time axis) or it has an out-of-range grid parameter or value anchor.
    """
    validation_status, failing_block_idx = _decoder.check_unit(raw_body, num_blocks, has_time)
    if validation_status == _decoder.BAD_HEAD:
        raise ValueError(
            f"block {failing_block_idx}: head byte {raw_body[failing_block_idx]:#04x} sets reserved bits "
            "(not supported by this version)"
        )
    if validation_status == _decoder.BAD_PARAM:
        raise ValueError(f"block {failing_block_idx}: parameter out of range (corrupt unit)")
    if validation_status == _decoder.BAD_ANCHOR:
        raise ValueError(f"block {failing_block_idx}: anchor out of range (corrupt unit)")


def _read_time_rows(raw_body: np.ndarray, header: _format.UnitHeader) -> _format.TimeRows:
    """Reads a unit's time rows, validating the start times.

    Raises:
        ValueError: If a block start time is int64 minimum or overflows int64.
    """
    num_blocks, block_len = header.num_blocks, header.block_len
    time_rows = _format.TimeRows(
        np.empty(num_blocks, np.int64), np.empty(num_blocks, np.int64), np.empty((num_blocks, block_len), np.uint64)
    )
    status, block_idx = _format.read_time_rows(
        raw_body, _format.count_flagged(raw_body, num_blocks), time_rows.starts, time_rows.steps, time_rows.deltas
    )
    if status != _format.TIME_ROWS_OK:
        raise ValueError(f"block {block_idx}: start time out of range (corrupt unit)")
    return time_rows


def _expand_times(block_flags: np.ndarray, time_rows: _format.TimeRows) -> np.ndarray:
    """Reconstructs the ticks of every block (including padding) from its time rows.

    Returns:
        2D int64 array of shape (num_blocks, block_len).

    Raises:
        ValueError: If a time step is negative (or zero on an irregular block), an
            irregular block's first delta isn't 0, or a time overflows int64.
    """
    block_times = np.empty(time_rows.deltas.shape, np.int64)
    status, block_idx = _time.expand_times(block_flags, time_rows.starts, time_rows.steps, time_rows.deltas, block_times)
    if status == _time.BAD_STEP:
        raise ValueError(f"block {block_idx}: time step out of range (corrupt unit)")
    if status == _time.BAD_FIRST_DELTA:
        raise ValueError(f"block {block_idx}: first time delta is not 0 (corrupt unit)")
    if status == _time.OVERFLOW:
        raise ValueError(f"block {block_idx}: times overflow int64 (corrupt unit)")
    return block_times


def decode_unit(unit: bytes) -> DecodedUnit:
    """Decodes a single self-describing unit into its samples and timestamps.

    Args:
        unit: Unit bytes, as returned by encode_unit, encode or update.

    Returns:
        DecodedUnit tuple (values, times):
            values: Reconstructed 1D float64 array of the unit's sample count.
            times: 1D datetime64 array of the same length, in the unit the times were
                encoded with; None if the unit was encoded without times.

    Raises:
        ValueError: If the header or body is invalid or out of range.
        zstandard.ZstdError: If the zstd frame is corrupt.
    """
    raw_body, header = _decompress(unit)
    num_blocks, block_len, num_samples = header.num_blocks, header.block_len, header.num_samples
    has_time = header.time_unit != 0
    reconstructed_samples = np.empty(num_blocks * block_len)
    # Run fused dequantization kernel
    _decoder.decode_unit(raw_body, reconstructed_samples.reshape(num_blocks, block_len), header.byte_planes, has_time)
    times = None
    if has_time:
        block_times = _expand_times(raw_body[:num_blocks], _read_time_rows(raw_body, header))
        times = block_times.reshape(-1)[:num_samples].view(f"datetime64[{_format.TIME_UNIT_NAMES[header.time_unit]}]")
    # Slice off the padding of a partial last block
    return DecodedUnit(reconstructed_samples[:num_samples], times)


def decode(units: Sequence[bytes]) -> list[DecodedUnit]:
    """Decodes multiple independent units into their samples and timestamps.

    Args:
        units: Sequence of unit bytes.

    Returns:
        List of DecodedUnit tuples (values, times), one per unit.

    Raises:
        ValueError: If any unit's header or body is invalid (see decode_unit).
        zstandard.ZstdError: If any unit's zstd frame is corrupt.
    """
    return [decode_unit(unit) for unit in units]


def update(
    unit: bytes,
    indices: npt.ArrayLike,
    blocks: npt.ArrayLike,
    params: Params = DEFAULT_PARAMS,
    *,
    times: npt.ArrayLike | None = None,
) -> UpdatedUnit:
    """Replaces or appends whole blocks within an existing unit.

    Untouched blocks retain their exact residuals, grid parameters, value anchors,
    non-finite codes and times: they are carried over without being decoded or
    re-quantized. A partial last block can be replaced by a full one; blocks can be
    appended only after a full last block.

    Args:
        unit: Existing unit bytes.
        indices: 1D integer array-like specifying block positions to replace
            (0..num_blocks-1) or append (num_blocks, num_blocks+1, ...). Appended
            indices must extend the unit contiguously without gaps.
        blocks: 2D float64 array of shape (k, block_len) holding new samples, or
            1D array if k == 1.
        params: Encoder configuration parameters.
        times: The new blocks' timestamps, shaped like blocks: required if and only if
            the unit has a time axis. datetime64 in the unit's time unit, or integer
            ticks in it. The updated unit's times must be non-decreasing throughout.

    Returns:
        UpdatedUnit tuple (unit, block_min, block_max, block_mean) containing:
            unit: New unit bytes.
            block_min: 1D float64 array of minima for updated blocks in
                indices order.
            block_max: 1D float64 array of maxima for updated blocks in
                indices order.
            block_mean: 1D float64 array of means for updated blocks in
                indices order.

    Raises:
        ValueError: If unit is corrupt, block lengths mismatch, indices are
            invalid or non-contiguous, blocks would be appended after a partial
            last block, total blocks exceed blocks_per_unit, or times are missing,
            unexpected, mis-shaped or out of order.
        zstandard.ZstdError: If the zstd frame is corrupt.
    """
    block_len = params.block_len
    raw_body, header = _decompress(unit)
    num_existing_blocks, num_samples = header.num_blocks, header.num_samples
    has_time = header.time_unit != 0
    if header.block_len != block_len:
        raise ValueError(f"unit has blocks of {header.block_len} samples but params.block_len is {block_len}")
    indices_arr = np.asarray(indices)
    if indices_arr.ndim != 1 or indices_arr.shape[0] == 0 or not np.issubdtype(indices_arr.dtype, np.integer):
        raise ValueError("indices must be a non-empty 1-D integer array")
    indices_arr = indices_arr.astype(np.int64)
    # Ensure all update indices are unique and non-negative
    if np.unique(indices_arr).shape[0] != indices_arr.shape[0]:
        raise ValueError("indices must be distinct")
    if (indices_arr < 0).any():
        raise ValueError("indices must be >= 0")
    # Verify that appended indices start at num_existing_blocks and have no gaps
    appended_indices = np.sort(indices_arr[indices_arr >= num_existing_blocks])
    if appended_indices.shape[0] and not np.array_equal(
        appended_indices, np.arange(num_existing_blocks, num_existing_blocks + appended_indices.shape[0])
    ):
        raise ValueError(f"appended indices must be {num_existing_blocks}, {num_existing_blocks + 1}, ... without gaps")
    num_total_blocks = num_existing_blocks + appended_indices.shape[0]
    # Check unit block capacity
    if num_total_blocks > params.blocks_per_unit:
        raise ValueError(
            f"update would make {num_total_blocks} blocks, over blocks_per_unit={params.blocks_per_unit}; start a new unit"
        )
    # A partial last block becomes full only when it is replaced
    last_partial = num_samples < num_existing_blocks * block_len
    last_replaced = bool(np.any(indices_arr == num_existing_blocks - 1))
    if last_partial and appended_indices.shape[0] and not last_replaced:
        raise ValueError(
            f"the unit's last block is partial ({num_samples - (num_existing_blocks - 1) * block_len} of "
            f"{block_len} samples); replace it with a full block to append after it"
        )
    new_num_samples = num_samples
    if appended_indices.shape[0] or (last_partial and last_replaced):
        new_num_samples = num_total_blocks * block_len
    new_blocks_arr = np.ascontiguousarray(blocks, dtype=np.float64)
    if new_blocks_arr.ndim == 1 and indices_arr.shape[0] == 1:
        new_blocks_arr = new_blocks_arr.reshape(1, -1)
    if new_blocks_arr.shape != (indices_arr.shape[0], block_len):
        raise ValueError(f"blocks must have shape ({indices_arr.shape[0]}, {block_len}), got {new_blocks_arr.shape}")
    new_block_times = None
    if has_time:
        if times is None:
            raise ValueError("the unit has a time axis: times are required")
        new_block_times, _ = _as_ticks(times, None, header.time_unit)
        if new_block_times.ndim == 1 and indices_arr.shape[0] == 1:
            new_block_times = new_block_times.reshape(1, -1)
        if new_block_times.shape != new_blocks_arr.shape:
            raise ValueError(f"times must have the shape of blocks {new_blocks_arr.shape}, got {new_block_times.shape}")
    elif times is not None:
        raise ValueError("the unit has no time axis: times must be None")

    block_flags = np.empty(num_total_blocks, np.uint8)
    grid_params = np.empty(num_total_blocks, np.int64)
    value_anchors = np.empty(num_total_blocks, np.int64)
    residuals = np.empty((num_total_blocks, block_len), np.int16)
    codes = np.zeros((num_total_blocks, block_len), np.uint8)
    # Unpack existing unit blocks into preallocated buffers
    _format.read_rows(
        raw_body,
        header.byte_planes,
        has_time,
        block_flags[:num_existing_blocks],
        grid_params[:num_existing_blocks],
        value_anchors[:num_existing_blocks],
        residuals[:num_existing_blocks],
        codes[:num_existing_blocks],
    )
    time_rows = None
    if new_block_times is not None:
        # Existing times decode exactly; re-analyzing them reproduces their rows byte for byte
        block_times = np.empty((num_total_blocks, block_len), np.int64)
        block_times[:num_existing_blocks] = _expand_times(
            block_flags[:num_existing_blocks], _read_time_rows(raw_body, header)
        )
        block_times[indices_arr] = new_block_times
    # Encode the new/replacement blocks
    (
        new_block_flags,
        new_grid_params,
        new_value_anchors,
        new_residuals,
        new_codes,
        new_minima,
        new_maxima,
        new_means,
    ) = _encode_blocks(new_blocks_arr, params)
    # Splice updated blocks into existing arrays
    block_flags[indices_arr] = new_block_flags
    grid_params[indices_arr] = new_grid_params
    value_anchors[indices_arr] = new_value_anchors
    residuals[indices_arr] = new_residuals
    codes[indices_arr] = new_codes
    if new_block_times is not None:
        # Sets or clears every block's irregular time bit
        time_rows = _encode_time_rows(block_times, block_flags)
    # Re-serialize and compress
    new_unit = _compress(
        block_flags,
        grid_params,
        value_anchors,
        residuals,
        codes,
        new_num_samples,
        params.try_byte_planes,
        time_rows,
        header.time_unit,
    )
    return UpdatedUnit(new_unit, new_minima, new_maxima, new_means)
