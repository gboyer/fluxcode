# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Bounded-error and decimal-exact compression of float64 time series.

Provides lossy (bounded-error) and decimal-exact compression for 1D float64
arrays. Data is organized into self-describing units containing up to
`blocks_per_unit` fixed-size blocks: a unit decodes on its own. Encoding also
returns per-block summary statistics (minimum, maximum, and mean of finite
samples) for the caller to store if useful. Short final blocks are padded to
full block length upon encoding and trimmed upon decoding.

Typical usage:

    import fluxcode

    # Encode a time series into a single storage unit:
    unit, block_min, block_max, block_mean = fluxcode.encode_unit(x)

    # Decode the unit (lossy: within the error bound, exact for decimal data):
    x_rec, _ = fluxcode.decode_unit(unit)

    # With timestamps (datetime64, or integer ticks with time_unit), stored exactly:
    unit, *_ = fluxcode.encode_unit(x, times=t)
    x_rec, t_rec = fluxcode.decode_unit(unit)

    # Replace block 3 and append a block (appending needs a full last block):
    unit, mins, maxs, means = fluxcode.update(
        unit, indices=[3, len(block_min)], blocks=new_blocks
    )
"""

from ._api import (
    DecodedUnit,
    EncodedSeries,
    EncodedUnit,
    Params,
    TimeUnit,
    UpdatedUnit,
    decode,
    decode_unit,
    encode,
    encode_unit,
    update,
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
    "encode_unit",
    "update",
]

__version__: str = "0.1.0"
"""Package version string."""

