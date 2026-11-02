# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Lossy steps: does a denser step ladder than powers of two pay on noise-floored blocks?

On each block the noise floor gates (ENCODER.md §5.2), the target is T = f·σ (f = 0.25, the
default) and each family takes its largest step ≤ T, on an absolute grid:

- pow2:   2^e (today);
- d125:   10^p·{1, 2, 5};
- d1225:  10^p·{1, 2, 2.5, 5};
- d1235:  10^p·{1, 2, 10/3, 5};
- exact:  T itself (what an N/D or float step could hold): the continuous reference.

Every family keeps max error ≤ T/2. Bytes are real (encoder kernels and compress, effort 4); RMSE
is in units of σ. Ungated blocks keep the encoder's own grid.

    cd experimental && uv run python grids/lossy.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bench"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))

import _fresh_numba_cache  # noqa: F401,E402  (before fluxcode)
import numpy as np  # noqa: E402
from _signals import minute  # noqa: E402
from fluxcode import Params  # noqa: E402
from fluxcode._compress import compress  # noqa: E402
from fluxcode._encoder import NOISE_MIN_LEN, NOISE_RHO, pick_order, range_scale, residual  # noqa: E402
from fluxcode._format import BLOCK_FLAG_DECIMAL, BLOCK_FLAG_ORDER  # noqa: E402
from fluxcode._group import encode_rows  # noqa: E402
from fluxcode._noise import noise  # noqa: E402

from grids import BLOCK, EFFORT, MINUTE, SEEDS, SIZES  # noqa: E402

F = Params().noise_floor_sigma


def ladder_step(mults, t):
    best = 0.0
    p = math.floor(math.log10(t))
    for pp in (p - 1, p, p + 1):
        for num, den in mults:
            step = num / den * 10.0 ** pp
            if step <= t * (1 + 1e-12):
                best = max(best, step)
    return best


FAMILIES = {
    "pow2": lambda t: math.ldexp(1.0, math.floor(math.log2(t))),
    "d125": lambda t: ladder_step([(1, 1), (2, 1), (5, 1)], t),
    "d1225": lambda t: ladder_step([(1, 1), (2, 1), (5, 2), (5, 1)], t),
    "d1235": lambda t: ladder_step([(1, 1), (2, 1), (10, 3), (5, 1)], t),
    "exact": lambda t: t,
}


def varying_noise(seed):
    """A 100-amplitude sine plus white noise whose σ is log-uniform in [0.3, 30] per block."""
    rng = np.random.default_rng(seed)
    x = minute("sin-9.87hz", seed)
    sig = np.repeat(np.exp(rng.uniform(np.log(0.3), np.log(30), MINUTE // BLOCK)), BLOCK)
    return x + rng.normal(0, 1, MINUTE) * sig, sig


def dataset(name, seed):
    if name == "sine + varying σ":
        return varying_noise(seed)
    x = minute(name, seed)
    sig = {"noisy-sine": 5.0, "gauss-spikes": 20 / math.sqrt(12), "impulses": 20 / math.sqrt(12)}[name]
    return x, np.full(MINUTE, sig)


def run(x, sig, family):
    rows, _ = encode_rows(x, SIZES, Params(grids="decimal"))
    rows = rows._replace(block_flags=rows.block_flags.copy(), grid_params=rows.grid_params.copy(),
                         value_anchors=rows.value_anchors.copy(), residuals=rows.residuals.copy())
    sq, n, gated = 0.0, 0, 0
    scratch = np.empty(BLOCK)
    for b in range(SIZES.size):
        sl = slice(b * BLOCK, (b + 1) * BLOCK)
        xb = x[sl]
        lo, hi = float(xb.min()), float(xb.max())
        sigma, rho = noise(xb, range_scale(lo, hi), scratch)
        if not (BLOCK >= NOISE_MIN_LEN and rho < NOISE_RHO and sigma > 0):
            continue
        t = F * sigma
        step = FAMILIES[family](t)
        k = np.round(xb / step)
        q = (k - k.min()).astype(np.int32)
        if q.max() > 65535:
            continue
        recon = (k.min() + q) * step
        order = pick_order(q, min(250, BLOCK), 15)
        residual(q, order, rows.residuals[sl])
        rows.block_flags[b] = (int(rows.block_flags[b]) & ~BLOCK_FLAG_ORDER & 0xFF) | order | (
            0 if family == "pow2" else BLOCK_FLAG_DECIMAL)
        rows.grid_params[b] = round(math.log10(step)) if family != "pow2" else round(math.log2(step))
        rows.value_anchors[b] = int(k.min())
        sq += float(np.sum(((recon - xb) / sig[sl]) ** 2))
        n += BLOCK
        gated += 1
    return len(compress(rows, x.size, EFFORT, 0)), math.sqrt(sq / max(n, 1)), gated


def main():
    names = ["sine + varying σ", "noisy-sine", "gauss-spikes", "impulses"]
    print("Each cell: bytes per block (vs pow2), RMSE/σ, and the bits/sample difference at equal RMSE "
          "(Δbits + log2(RMSE / RMSE_pow2)).\n")
    print("| dataset | gated blocks | " + " | ".join(FAMILIES) + " |")
    print("|---|---:|" + "---:|" * len(FAMILIES))
    for name in names:
        res = {f: [] for f in FAMILIES}
        for seed in SEEDS:
            x, sig = dataset(name, seed)
            for f in FAMILIES:
                res[f].append(run(x, sig, f))
        base = np.mean([r[0] for r in res["pow2"]])
        base_rmse = np.mean([r[1] for r in res["pow2"]])
        cells = []
        for f in FAMILIES:
            size = np.mean([r[0] for r in res[f]])
            rmse = np.mean([r[1] for r in res[f]])
            eq = 8 * (size - base) / x.size + math.log2(rmse / base_rmse)
            cells.append(f"{size / SIZES.size:.0f} B ({100 * (size / base - 1):+.1f}%), {rmse:.3f}, {eq:+.3f} b")
        gated = sum(r[2] for r in res["pow2"])
        print(f"| {name} | {gated} | " + " | ".join(cells) + " |", flush=True)


if __name__ == "__main__":
    main()
