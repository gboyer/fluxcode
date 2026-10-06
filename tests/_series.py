# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Helpers for tests and benchmarks that check per-block properties across a long series:
encode() with its per-block-group index lists joined into flat per-block arrays, and back."""

from contextlib import contextmanager

import numpy as np

import fluxcode
from fluxcode import Params, _api, _args, _compress, _encoder, _group

PER_GROUP = _api.DEFAULT_BLOCKS_PER_GROUP  # the helpers assume the default block group size


def encode_series(x, params=Params()):
    groups, mins, maxs, means = fluxcode.encode(x, params)
    return groups, np.concatenate(mins), np.concatenate(maxs), np.concatenate(means)


def decode_series(groups):
    """decode() of encode_series' block groups, joined back into one series."""
    return np.concatenate([decoded.values for decoded in fluxcode.decode(groups)])


@contextmanager
def planes(layout, flush=True, zstd_levels=(3,)):
    """Every Params.effort compresses with layout ("bit", "byte", "heuristic" or "best") while
    inside: the effort picks the layout, and some tests need a particular one."""
    saved = dict(_compress.EFFORTS)
    _compress.EFFORTS.update(dict.fromkeys(saved, _compress.Effort(layout, flush, zstd_levels)))
    try:
        yield
    finally:
        _compress.EFFORTS.update(saved)


def group_rows(group):
    """GroupRows (block_flags, block_sizes, grid_params, value_anchors, residuals, codes, time_rows) of a block group."""
    return _group.read_rows(_group.decompress(group))


def flags_params(groups):
    hp = [(group_rows(u).block_flags, group_rows(u).grid_params) for u in groups]
    return np.concatenate([h for h, _ in hp]), np.concatenate([p for _, p in hp])


def gated(x, f=0.25):
    """Per block: did the noise floor coarsen the step? (Decimal detection off: p isn't an exponent.)"""
    groups, lo, hi, _ = encode_series(x, Params(noise_floor_sigma=f, decimal_detection=False))
    _, param = flags_params(groups)
    fine = np.array([_encoder.range_exponent(a, b, 16) for a, b in zip(lo, hi)])
    return param > fine


def unchecked_params(**kw):
    """Params with values validation rejects (e.g. targets below 6), for probing the encoder."""
    p = Params()
    for k, v in kw.items():
        object.__setattr__(p, k, v)
    return p


def encode_unchecked(x, **kw):
    """encode_group's bytes for one block group under unchecked_params(**kw)."""
    x = np.asarray(x, np.float64)
    return _group.encode(x, _args.fixed_sizes(x.shape[0], _api.DEFAULT_BLOCK_LEN), unchecked_params(**kw)).group
