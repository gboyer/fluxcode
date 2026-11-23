"""Sinusoidal baseline vs the period predictor (2026-11-22). Per block: fit c + sum_k a_k cos(k w n) + b_k sin(k w n)
(k = 1..K, optionally amplitudes linear in time) to the block's indices q by least squares, w refined around the
period predictor's crossing estimate; residuals e = q - round(fit), order 0, side info charged (32 bits for w,
24 per coefficient). A block takes the baseline only if its entropy estimate (plus side info) beats its fixed
order's. Sizes through fluxcode's back end (planes + zstd, effort 4) for: fixed orders only, the shipped period
predictor, and the baseline. Offline only: a decoder would need deterministic integer evaluation of the fit."""
import sys
import numpy as np
import corpus, harness as H
from fluxcode import _encoder, _compress, _group
import fluxcode

def design(n, w, K, trend):
    t = np.arange(n)
    cols = [np.ones(n)] + ([t / n] if trend else [])
    for k in range(1, K + 1):
        c, s = np.cos(k * w * t), np.sin(k * w * t)
        cols += [c, s] + ([c * t / n, s * t / n] if trend else [])
    return np.stack(cols, 1)

def fit(q, w, K, trend):
    A = design(len(q), w, K, trend)
    coef, *_ = np.linalg.lstsq(A, q, rcond=None)
    return A @ coef, A.shape[1]

def baseline(q, K, trend):
    """Best (residuals, side bits) over w near the crossing estimate, or None."""
    d = q - q.mean()
    up = np.nonzero((d[:-1] < 0) & (d[1:] >= 0))[0]
    if len(up) < 3:
        return None
    w0 = 2 * np.pi * (len(up) - 1) / (up[-1] - up[0])
    best = None
    lo, hi = w0 * 0.97, w0 * 1.03
    for _ in range(3):  # coarse-to-fine grid on w
        for w in np.linspace(lo, hi, 15):
            f, ncoef = fit(q.astype(float), w, K, trend)
            e = q - np.round(f).astype(np.int64)
            v = float(np.var(e))
            if best is None or v < best[0]:
                best = (v, e, w, ncoef)
        step = (hi - lo) / 14
        lo, hi = best[2] - step, best[2] + step
    return best[1], 32 + 24 * best[3]

def gaussian_bits(r):
    v = max(float(np.var(r)), 1e-9)
    return 0.5 * len(r) * np.log2(2 * np.pi * np.e * v) if v > 1 / 12 else 0.0

def group_bytes(rows):
    return len(_compress.compress(rows, int(rows.block_sizes.sum()), _compress.EFFORTS[4], 0))

def run(name, x, bl, bpg, variants):
    fixed = per = 0
    base = {v: 0.0 for v in variants}
    used = {v: 0 for v in variants}
    params = fluxcode.Params()
    for part, sizes in H.groups_of(x, bl, bpg):
        _encoder.PERIOD_EFFORT = 99
        rows_off = H.encode_rows(part, sizes, params)
        _encoder.PERIOD_EFFORT = 0
        rows_on = H.encode_rows(part, sizes, params)
        fixed += group_bytes(rows_off)
        per += group_bytes(rows_on)
        qs = [np.where(q >= 32768, q - 65536, q).astype(np.int64) for q in H.block_q(rows_off)]
        r_off, offs = rows_off.residuals.astype(np.int64), np.concatenate([[0], np.cumsum(rows_off.block_sizes)])
        for v in variants:
            K, trend = v
            blocks, side = [], 0
            for b, q in enumerate(qs):
                r0 = r_off[offs[b]:offs[b + 1]]
                got = baseline(q, K, trend) if len(q) >= 64 else None
                if got is not None and gaussian_bits(got[0]) + got[1] < gaussian_bits(r0):
                    blocks.append(got[0]); side += got[1]; used[v] += 1
                else:
                    blocks.append(r0)
            base[v] += H.size_with(rows_off, blocks) + side / 8
    n = len(x)
    out = f"{name:18s} fixed {8*fixed/n:5.2f} b/s  period {8*per/n:5.2f} ({per/fixed-1:+.1%})"
    for v in variants:
        out += f"  K{v[0]}{'t' if v[1] else ''} {8*base[v]/n:5.2f} ({base[v]/fixed-1:+.1%}, {used[v]} blk)"
    print(out, flush=True)

if __name__ == "__main__":
    variants = [(1, False), (3, False), (7, False), (7, True)]
    names = sys.argv[1:] or ["motor-current", "line-voltage", "sin-50.3hz", "noisy-sine", "accel-gearbox"]
    for s, name, x, bl, bpg in corpus.corpus(["fast", "synth"]):
        if name in names:
            run(name, x, bl, bpg, variants)
