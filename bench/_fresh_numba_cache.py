# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Import this before fluxcode in a benchmark or report script: it points numba's cache at a
directory named by a hash of the fluxcode sources.

numba's cache=True checks only the timestamp of the file defining a kernel, not the helpers and
constants it inlines from other modules, so editing `_noise.py` leaves `_encoder.py`'s cached
kernel running the old code. Keyed on every source file, an edit anywhere starts a fresh cache
(one compile, about 5 s), and unchanged code keeps a warm one. The tests use a throwaway cache
instead (tests/conftest.py). An explicit NUMBA_CACHE_DIR is respected.
"""

import hashlib
import os
from pathlib import Path

if "NUMBA_CACHE_DIR" not in os.environ:
    _sources = sorted((Path(__file__).resolve().parents[1] / "fluxcode").glob("*.py"))
    _digest = hashlib.sha256(b"".join(p.name.encode() + p.read_bytes() for p in _sources)).hexdigest()[:16]
    _cache = Path(os.environ.get("TMPDIR", "/tmp")) / f"fluxcode-numba-{_digest}"
    os.environ["NUMBA_CACHE_DIR"] = str(_cache)
