# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Classic codecs: the ones from the original request plus a few baselines.

Each is a block codec: encode(x) -> (bytes, info) and decode(bytes, n) -> np.ndarray, or, for the
deflate-backed ones, split / join around the compressor. CLASSIC_CODECS wraps them as unit codecs
(tslab.common.unit): Concat for the ones without an entropy coder, SharedBackend for deflate.
`info` carries extras for reporting (tree code frequencies, knots, ...).
"""

from tslab.classic.bintree import BinaryTree
from tslab.classic.dct import DctTopK
from tslab.classic.dpcm import CompandedDpcm, cubic_expander, mulaw_expander
from tslab.classic.gorilla import Gorilla
from tslab.classic.piecewise import PchipMinMax, PiecewiseLinearMinMax
from tslab.classic.quant import Quant8, QuantDeltaDeflate
from tslab.classic.swinging_door import SwingingDoor
from tslab.common.unit import Concat, SharedBackend

__all__ = ["CLASSIC_CODECS", "BinaryTree", "CompandedDpcm", "DctTopK", "Gorilla", "PchipMinMax",
           "PiecewiseLinearMinMax", "Quant8", "QuantDeltaDeflate", "SwingingDoor", "cubic_expander", "mulaw_expander"]

CLASSIC_CODECS = [
    Concat(Quant8()),
    SharedBackend(QuantDeltaDeflate(8)),
    SharedBackend(QuantDeltaDeflate(6)),
    SharedBackend(CompandedDpcm("dpcm6-linear-deflate", "6-bit DPCM, linear steps, deflate", cubic_expander(0.0))),
    SharedBackend(CompandedDpcm("dpcm6-cubic-deflate", "6-bit DPCM, cubic steps 0.8u³ + 0.2u, deflate", cubic_expander(0.8))),
    SharedBackend(CompandedDpcm("dpcm6-mulaw-deflate", "6-bit DPCM, μ-law steps (μ = 7), deflate", mulaw_expander(7))),
    SharedBackend(CompandedDpcm("dpcm4-linear-deflate", "4-bit DPCM, linear steps, deflate", cubic_expander(0.0), bits=4)),
    SharedBackend(CompandedDpcm("dpcm4-cubic-deflate", "4-bit DPCM, cubic steps 0.8u³ + 0.2u, deflate", cubic_expander(0.8), bits=4)),
    Concat(BinaryTree()),
    Concat(BinaryTree(leaf="edge")),
    Concat(PiecewiseLinearMinMax()),
    Concat(PiecewiseLinearMinMax(consolidate=True)),
    Concat(PchipMinMax()),
    Concat(SwingingDoor(0.005)),
    Concat(SwingingDoor(0.02)),
    Concat(Gorilla()),
    Concat(DctTopK(64)),
]
