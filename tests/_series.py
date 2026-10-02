# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Helpers for tests and benchmarks that check per-block properties across a long series:
encode() with its per-unit index lists joined into flat per-block arrays, and back."""

from contextlib import contextmanager

import numpy as np

import fluxcode
from fluxcode import Params, _api, _encoder, _unit

PER_UNIT = _api.DEFAULT_BLOCKS_PER_UNIT  # the helpers assume the default unit size


def encode_series(x, params=Params()):
    units, mins, maxs, means = fluxcode.encode(x, params)
    return units, np.concatenate(mins), np.concatenate(maxs), np.concatenate(means)


def decode_series(units):
    """decode() of encode_series' units, joined back into one series."""
    return np.concatenate([decoded.values for decoded in fluxcode.decode(units)])


@contextmanager
def planes(layout, flush=True, zstd_levels=(3,)):
    """Every Params.effort compresses with layout ("bit", "byte", "heuristic" or "best") while
    inside: the effort picks the layout, and some tests need a particular one."""
    saved = dict(_unit.EFFORTS)
    _unit.EFFORTS.update(dict.fromkeys(saved, _unit.Effort(layout, flush, zstd_levels)))
    try:
        yield
    finally:
        _unit.EFFORTS.update(saved)


def unit_rows(unit):
    """UnitRows (block_flags, block_sizes, grid_params, value_anchors, residuals, codes, time_rows) of a unit."""
    return _unit.read_rows(_unit.decompress(unit))


def flags_params(units):
    hp = [(unit_rows(u).block_flags, unit_rows(u).grid_params) for u in units]
    return np.concatenate([h for h, _ in hp]), np.concatenate([p for _, p in hp])


def gated(x, f=0.25):
    """Per block: did the noise floor coarsen the step? (Decimal detection off: p isn't an exponent.)"""
    units, lo, hi, _ = encode_series(x, Params(noise_floor_sigma=f, decimal_detection=False))
    _, param = flags_params(units)
    fine = np.array([_encoder.range_exponent(a, b, 16) for a, b in zip(lo, hi)])
    return param > fine


def unchecked_params(**kw):
    """Params with values validation rejects (e.g. targets below 6), for probing the encoder."""
    p = Params()
    for k, v in kw.items():
        object.__setattr__(p, k, v)
    return p


def encode_unchecked(x, **kw):
    """encode_unit's bytes for one unit under unchecked_params(**kw)."""
    x = np.asarray(x, np.float64)
    return _unit.encode(x, _api._fixed_sizes(x.shape[0], _api.DEFAULT_BLOCK_LEN), unchecked_params(**kw)).unit
