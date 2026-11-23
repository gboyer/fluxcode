# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Compatibility shim for scripts that put bench/ on sys.path; canonical module is tests._fresh_numba_cache."""

import tests._fresh_numba_cache  # noqa: F401
