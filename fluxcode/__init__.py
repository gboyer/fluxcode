# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Bounded-error and decimal-exact compression of float64 time series.

Provides lossy (bounded-error) and decimal-exact compression for 1D float64
arrays. Data is organized into self-describing block groups of blocks, each block holding
0 to 65,535 samples: a block group decodes on its own. Encoding also returns per-block summary
statistics (minimum, maximum, and mean of finite samples) for the caller to store if
useful.

Typical usage:

    import fluxcode

    # Encode a dense series into a single block group, in blocks of 1000 samples:
    group, block_min, block_max, block_mean = fluxcode.encode_group(x)

    # Decode the block group (lossy: within the error bound, exact for decimal data):
    x_rec, _, block_sizes = fluxcode.decode_group(group)

    # With timestamps (datetime64, or integer ticks with time_unit), stored exactly (or within
    # Params(time_error=...) of an interval):
    group, *_ = fluxcode.encode_group(x, times=t)
    x_rec, t_rec, _ = fluxcode.decode_group(group)

    # Explicit blocks of any size (a flat array and each block's size):
    group, *_ = fluxcode.encode_blocks(x, block_sizes=[1000, 998, 0, 1003])

    # Replace block 3 and append a block:
    group, indices, mins, maxs, means = fluxcode.update(group, {3: block3, 4: block4})

    # An hour in one-minute blocks, then replace the data of a few minutes. New samples
    # replace existing ones with the same times; delete_ranges are deleted first (with
    # only samples it is an upsert, with only ranges a deletion):
    hour = dict(start_time=np.datetime64("2026-01-01T10:00"), block_duration=np.timedelta64(1, "m"))
    group, *_ = fluxcode.encode_time_blocks(x, t, **hour)
    group, *_ = fluxcode.update_time_blocks(group, x_new, t_new, delete_ranges=(t0, t1), **hour)
"""

from ._api import (
    decode,
    decode_group,
    encode,
    encode_blocks,
    encode_group,
    encode_time_blocks,
    update,
    update_time_blocks,
)
from ._types import (
    DEFAULT_NOISE_FLOOR_SIGMA,
    DecodedGroup,
    EncodedGroup,
    EncodedSeries,
    Params,
    TimeUnit,
    UpdatedGroup,
)

__all__ = [
    "DEFAULT_NOISE_FLOOR_SIGMA",
    "DecodedGroup",
    "EncodedGroup",
    "EncodedSeries",
    "Params",
    "TimeUnit",
    "UpdatedGroup",
    "decode",
    "decode_group",
    "encode",
    "encode_blocks",
    "encode_group",
    "encode_time_blocks",
    "update",
    "update_time_blocks",
]

__version__: str = "0.1.0"
"""Package version string."""

