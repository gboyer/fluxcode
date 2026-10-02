# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Loading the optional Rust extension."""

import sys
import types

import pytest

from fluxcode import _compress


def test_out_of_date_extension_falls_back_to_python(monkeypatch):
    """A fluxcode_rs without the current interface (built before pack_unit, say) isn't used."""
    old = types.ModuleType("fluxcode_rs")
    old.compress_unit = lambda *args: b""
    monkeypatch.setitem(sys.modules, "fluxcode_rs", old)
    monkeypatch.delenv("FLUXCODE_RUST", raising=False)
    with pytest.warns(RuntimeWarning, match="rebuild"):
        assert _compress._load_rust() is None
    old.INTERFACE_VERSION = _compress.RUST_INTERFACE_VERSION
    assert _compress._load_rust() is old
