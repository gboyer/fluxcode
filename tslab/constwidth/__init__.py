# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Constant-width formats: a fixed number of bits per sample, no entropy coding.

Each codec has encode(x) -> (bytes, info) and decode(bytes, n) -> np.ndarray, with the same size
for every block, so a block group is the block encodings back to back (tslab.common.group.Concat). All of
them are closed loop on an integer grid over the block's [min, max], so quantization error never
accumulates; min and max are in each block's header.
"""

from tslab.common.group import Concat
from tslab.constwidth.delta_bfp import Delta1Bfp
from tslab.constwidth.delta_linear import Delta1Linear8
from tslab.constwidth.delta_sqrt import Delta1Sqrt8
from tslab.constwidth.delta_tree import DELTA_TREE_CODECS, Delta1Tree

__all__ = ["CONSTWIDTH_CODECS", "DELTA_TREE_CODECS", "Delta1Bfp", "Delta1Linear8", "Delta1Sqrt8", "Delta1Tree"]

CONSTWIDTH_CODECS = [Concat(c) for c in (Delta1Linear8(), Delta1Sqrt8(), Delta1Bfp(), Delta1Bfp(16, exp_bits=4),
                                          Delta1Bfp(16, exp_bits=4, code_bits=4), Delta1Bfp(16, exp_bits=4, code_bits=6))]
