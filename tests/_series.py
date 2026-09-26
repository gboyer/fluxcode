# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Helpers for tests and benchmarks that check per-block properties across a long series:
encode() with its per-unit index lists joined into flat per-block arrays, and back."""

import numpy as np

import fluxcode
from fluxcode import Params, _api, _encoder, _format

PER_UNIT = Params().blocks_per_unit  # the helpers assume the default unit size


def encode_series(x, params=Params()):
    assert params.blocks_per_unit == PER_UNIT
    units, mins, maxs, means = fluxcode.encode(x, params)
    return units, np.concatenate(mins), np.concatenate(maxs), np.concatenate(means)


def decode_series(units):
    """decode() of encode_series' units, joined back into one series."""
    return np.concatenate(fluxcode.decode(units))


def unit_rows(unit):
    """(head, param, anchor, residual, codes) of a unit."""
    raw_body, num_blocks, block_len, _ = _api._decompress(unit)
    return _format.read_unit(raw_body, num_blocks, block_len)


def heads_params(units):
    hp = [unit_rows(u)[:2] for u in units]
    return np.concatenate([h for h, _ in hp]), np.concatenate([p for _, p in hp])


def gated(x, f=0.25):
    """Per block: did the noise floor coarsen the step? (Decimal detection off: p isn't an exponent.)"""
    units, lo, hi, _ = encode_series(x, Params(noise_floor_sigma=f, decimal_detection=False))
    _, param = heads_params(units)
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
    return _api._encode_rows(np.asarray(x, np.float64), unchecked_params(**kw))[0]
