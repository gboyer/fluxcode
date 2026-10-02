# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Interpretable per-unit features of a residual array u (uint16, flat), for plane_layout/analyze.py."""

import numpy as np

import model

def features(u):
    """A dict of features of the residual array."""
    n = len(u)
    low, high = u & 0xFF, u >> 8
    h_low = model.entropy_bits(np.bincount(low, minlength=256)) / n
    h_high = model.entropy_bits(np.bincount(high, minlength=256)) / n
    h_u = model.entropy_bits(np.bincount(u, minlength=65536)) / n
    sb, sy = model.streams(u)
    est_bit = model.lz_estimate(sb)[0]
    est_byte = model.lz_estimate(sy)[0]
    return dict(h_u=h_u, h_low=h_low, h_high=h_high, hi_nz=float(np.mean(u > 255)),
                est_bit=est_bit, est_byte=est_byte, lz_ratio=float(np.log(est_byte / est_bit)))
