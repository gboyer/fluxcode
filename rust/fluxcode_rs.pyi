# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Type stubs for the optional ``fluxcode_rs`` Rust accelerator extension."""

from collections.abc import Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt

INTERFACE_VERSION: int
FLUSH_MIN_DENSITY: int
BYTE_PLANES_BIT: int
BYTE_PLANES_MAX_SHARE: float

def compress_group(
    block_flags: npt.NDArray[np.uint8],
    block_sizes: npt.NDArray[np.int64],
    grid_exponents: npt.NDArray[np.int64],
    grid_scales: npt.NDArray[np.int64],
    block_anchors: npt.NDArray[np.int64],
    residuals: npt.NDArray[np.int16],
    sample_codes: npt.NDArray[np.uint8],
    escaped_values: npt.NDArray[np.float64],
    layout: str,
    flush: bool,
    zstd_levels: Sequence[int],
) -> bytes: ...
def pack_group(
    body: npt.NDArray[np.uint8],
    num_blocks: int,
    num_samples: int,
    num_plane_bytes: int,
    time_unit: int,
    layout: str,
    flush: bool,
    zstd_levels: Sequence[int],
    group_time_step: bool = False,
) -> bytes: ...
def simd_path() -> Literal["neon", "sse2", "scalar"]: ...
def zstd_version() -> tuple[int, int, int]: ...
