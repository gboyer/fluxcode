# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Bounded-error and decimal-exact compression of float64 time series.

Provides lossy (bounded-error) and decimal-exact compression for 1D float64
arrays. Data is organized into self-describing units of blocks, each block holding
0 to 65,535 samples: a unit decodes on its own. Encoding also returns per-block summary
statistics (minimum, maximum, and mean of finite samples) for the caller to store if
useful.

Typical usage:

    import fluxcode

    # Encode a dense series into a single storage unit, in blocks of 1000 samples:
    unit, block_min, block_max, block_mean = fluxcode.encode_unit(x)

    # Decode the unit (lossy: within the error bound, exact for decimal data):
    x_rec, _, block_sizes = fluxcode.decode_unit(unit)

    # With timestamps (datetime64, or integer ticks with time_unit), stored exactly:
    unit, *_ = fluxcode.encode_unit(x, times=t)
    x_rec, t_rec, _ = fluxcode.decode_unit(unit)

    # Explicit blocks of any size (a flat array and each block's size):
    unit, *_ = fluxcode.encode_blocks(x, block_sizes=[1000, 998, 0, 1003])

    # Replace block 3 and append a block:
    unit, indices, mins, maxs, means = fluxcode.update(unit, {3: block3, 4: block4})

    # An hour in one-minute blocks, then replace the data of a few minutes:
    hour = dict(start_time=np.datetime64("2026-01-01T10:00"), block_duration=np.timedelta64(1, "m"))
    unit, *_ = fluxcode.encode_time_blocks(x, t, **hour)
    unit, *_ = fluxcode.update_time_blocks(unit, x_new, t_new, update_ranges=(t0, t1), **hour)
"""

from ._api import (
    decode,
    decode_unit,
    encode,
    encode_blocks,
    encode_time_blocks,
    encode_unit,
    update,
    update_time_blocks,
)
from ._types import (
    DecodedUnit,
    EncodedSeries,
    EncodedUnit,
    Params,
    TimeUnit,
    UpdatedUnit,
)

__all__ = [
    "DecodedUnit",
    "EncodedSeries",
    "EncodedUnit",
    "Params",
    "TimeUnit",
    "UpdatedUnit",
    "decode",
    "decode_unit",
    "encode",
    "encode_blocks",
    "encode_time_blocks",
    "encode_unit",
    "update",
    "update_time_blocks",
]

__version__: str = "0.1.0"
"""Package version string."""

