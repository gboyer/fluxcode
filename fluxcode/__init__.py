# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Bounded-error compression of float64 time series with exact grid detection.

Provides bounded-error lossy compression for 1D float64 arrays, while automatically
detecting and preserving values that lie on exact binary, decimal, or rational-step
grids (such as 0.01, 0.05, thirds, or ADC / calibration steps). Data is organized into
self-describing block groups of blocks, each block holding 0 to 65,535 samples: a block
group decodes on its own. Encoding also returns per-block summary statistics (minimum,
maximum, mean of finite samples, and block start timestamps when timed) for the caller
to store if useful.

Typical usage:

    import fluxcode

    # Encode a dense series into a single block group, in blocks of 1000 samples:
    encoded = fluxcode.encode_group(x, block_len=1000)

    # Decode the block group (within the error bound, exact on detected grids):
    decoded = fluxcode.decode_group(encoded.payload)

    # With timestamps (datetime64, or integer ticks with time_unit), stored exactly
    # (or within Params(time_error=...) of an interval):
    encoded = fluxcode.encode_group(x, block_len=1000, times=t)
    decoded = fluxcode.decode_group(encoded.payload)

    # Explicit blocks of any size (a flat array and each block's size):
    encoded = fluxcode.encode_blocks(x, block_sizes=[1000, 998, 0, 1003])

    # Replace block 3 and append a block:
    updated = fluxcode.update(encoded.payload, {3: block3, 4: block4})

    # An hour in one-minute blocks, then replace the data of a few minutes. New samples
    # replace existing ones with the same times; delete_ranges are deleted first (with
    # only samples it is an upsert, with only ranges a deletion):
    hour = dict(
        start_time=np.datetime64("2026-01-01T10:00"),
        block_duration=np.timedelta64(1, "m"),
    )
    encoded = fluxcode.encode_time_blocks(x, t, **hour)
    updated = fluxcode.update_time_blocks(
        encoded.payload, x_new, t_new, delete_ranges=(t0, t1), **hour
    )

    # With Params(time_error=...), replace whole minutes from the source data instead:
    updated = fluxcode.replace_time_blocks(
        encoded.payload, range(12, 14), x_new, t_new, params, **hour
    )
"""

from ._api import (
    decode,
    decode_group,
    encode,
    encode_blocks,
    encode_group,
    encode_time_blocks,
    replace_time_blocks,
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
    "replace_time_blocks",
    "update",
    "update_time_blocks",
]

__version__: str = "0.1.0"
"""Package version string."""
