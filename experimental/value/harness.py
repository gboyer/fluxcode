# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Shared helpers: encode a signal into block groups, get each block's grid indices q, swap in other
residual streams and measure them through fluxcode's own back end (planes + zstd)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "bench")]
import os  # noqa: E402
if "NUMBA_CACHE_DIR" not in os.environ:
    import hashlib
    _src = b"".join(p.read_bytes() for p in sorted((ROOT / "fluxcode").glob("*.py")))
    _env = "".join(f"{k}={v};" for k, v in sorted(os.environ.items()) if k.startswith("FLUXCODE_X_"))
    os.environ["NUMBA_CACHE_DIR"] = f"{os.environ.get('TMPDIR', '/tmp')}/fluxcode-x-{hashlib.sha256(_src + _env.encode()).hexdigest()[:16]}"
import _fresh_numba_cache  # noqa: E402,F401

import numpy as np  # noqa: E402
from numba import njit  # noqa: E402

import fluxcode  # noqa: E402
from fluxcode import _compress, _decoder, _group  # noqa: E402


def groups_of(x, block_len, blocks_per_group):
    per = block_len * blocks_per_group
    for s in range(0, len(x), per):
        part = x[s:s + per]
        sizes = np.full(len(part) // block_len, block_len, np.int64)
        rem = len(part) - sizes.sum()
        if rem:
            sizes = np.append(sizes, rem)
        yield part, sizes


def encode_rows(part, sizes, params):
    rows, _ = _group.encode_rows(np.ascontiguousarray(part), sizes, params)
    return rows


def block_q(rows):
    """[int64 q per block] from the rows' residuals and orders (empty blocks give empty arrays)."""
    out = []
    off = 0
    for b, n in enumerate(rows.block_sizes):
        r = rows.residuals[off:off + n].astype(np.int64) & 0xFFFF
        r = np.where(r >= 32768, r - 65536, r).astype(np.int64)
        order = int(rows.block_flags[b]) & 3
        q = r.copy()
        for _ in range(order):
            q = np.cumsum(q) & 0xFFFF
        out.append(q & 0xFFFF)
        off += n
    return out


def size_with(rows, residual_blocks, effort=4):
    """Bytes of the block group with each block's residuals replaced (int arrays, wrapped to int16)."""
    res = np.concatenate([((np.asarray(r, np.int64) + 32768) & 0xFFFF) - 32768 for r in residual_blocks])
    new = rows._replace(residuals=res.astype(np.int16))
    return len(_compress.compress(new, int(rows.block_sizes.sum()), _compress.EFFORTS[effort], 0))


def wrap16(r):
    return ((np.asarray(r, np.int64) + 32768) & 0xFFFF) - 32768


def entropy_bits(values):
    """Empirical zeroth-order entropy, in total bits."""
    if len(values) == 0:
        return 0.0
    _, c = np.unique(values, return_counts=True)
    p = c / c.sum()
    return float(-(c * np.log2(p)).sum())


def fixed_residual(q, order):
    """fluxcode's order-k residuals (zeros prepended)."""
    r = np.concatenate([np.zeros(order, np.int64), q])
    for _ in range(order):
        r = np.diff(r)
    return wrap16(r)


@njit(cache=True)
def lpc_residual(q, coefs, shift, warm_order):
    """Integer LPC: pred = (sum c_j q[i-1-j] + half) >> shift; the first len(coefs) samples use the fixed
    order `warm_order` difference (zeros prepended) instead."""
    n = q.shape[0]
    p = coefs.shape[0]
    out = np.empty(n, np.int64)
    half = (np.int64(1) << shift) >> 1 if shift > 0 else np.int64(0)
    for i in range(min(p, n)):
        if warm_order == 0 or i == 0:
            out[i] = q[i] if warm_order == 0 or i == 0 else 0
        else:
            out[i] = q[i] - q[i - 1]
    for i in range(p, n):
        acc = np.int64(0)
        for j in range(p):
            acc += coefs[j] * q[i - 1 - j]
        out[i] = q[i] - ((acc + half) >> shift)
    return out


def fit_lpc(q, p, precision=14, ridge=1e-9):
    """Least-squares (covariance method) predictor of order p on q, quantized to `precision`-bit signed
    coefficients with a shared shift. Returns (coefs int64, shift)."""
    n = len(q)
    if n <= 2 * p + 4:
        return None
    x = q.astype(np.float64)
    x = x - x.mean()
    # columns: q[i-1], ..., q[i-p] for i in p..n-1
    A = np.lib.stride_tricks.sliding_window_view(x, p)[:-1][:, ::-1]
    y = x[p:]
    ata = A.T @ A
    ata += ridge * np.trace(ata) / p * np.eye(p)
    try:
        c = np.linalg.solve(ata, A.T @ y)
    except np.linalg.LinAlgError:
        return None
    cmax = np.abs(c).max()
    if not np.isfinite(cmax) or cmax == 0:
        return None
    # shift so that max |c| * 2^shift < 2^(precision-1)
    shift = int(np.floor(np.log2((2 ** (precision - 1) - 1) / cmax)))
    shift = max(0, min(shift, 30))
    ci = np.round(c * 2.0 ** shift).astype(np.int64)
    return ci, shift


def lpc_best(q, p, precision=14):
    fit = fit_lpc(q, p, precision)
    if fit is None:
        return None
    ci, shift = fit
    best = None
    for warm in (0, 1):
        r = wrap16(lpc_residual(q, ci, shift, warm))
        h = entropy_bits(r)
        if best is None or h < best[0]:
            best = (h, r)
    return best[1], ci, shift


def default_params(**kw):
    return fluxcode.Params(**kw)


def decimate_note():
    pass


def _check():
    q = np.arange(100, dtype=np.int64) * 3 % 1000
    assert (fixed_residual(q, 0) == q).all()
    print("ok")


if __name__ == "__main__":
    _check()


def _self_check():
    """Fail loudly if the compiled noise kernel doesn't match the FLUXCODE_X_NOISE_ORDER in effect (stale cache):
    its rho on white noise must be the k-th differences' own."""
    import math

    from fluxcode import _noise

    x = np.random.default_rng(1).normal(0, 1, 1000)
    k = _noise.NOISE_ORDER
    _, rho_share = _noise.noise(x, math.frexp(np.ptp(x))[1], np.zeros(1000))
    d = np.diff(x, k)
    d = d - d.mean()
    rho = np.dot(d[1:], d[:-1]) / np.dot(d, d)
    if abs(rho_share * _noise.WHITE_RHO - rho) > 0.02:
        raise RuntimeError(f"stale numba cache: noise kernel isn't order {k} ({os.environ['NUMBA_CACHE_DIR']})")


_self_check()
