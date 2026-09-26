# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

# Compile every kernel from source on each run. numba's cache=True checks only the timestamp of
# the file defining a function, not the constants and helpers it uses from other modules, so a
# warm cache can keep testing stale code (it hid an IntEnum constant that doesn't compile). Must
# be set before numba is imported. Costs ~5 s of compilation per run.
_cache_dir = tempfile.mkdtemp(prefix="fluxcode-numba-")
os.environ["NUMBA_CACHE_DIR"] = _cache_dir
atexit.register(shutil.rmtree, _cache_dir, ignore_errors=True)

sys.path.insert(0, str(Path(__file__).parent))
