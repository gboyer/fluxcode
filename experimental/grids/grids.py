# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Richer value grids: which steps are worth a format change?

Today a block is on a power-of-two grid a + q·2^e, or on an exact decimal grid (K0 + q)·10^p
when every sample sits on one coarser than the power-of-two step (ENCODER.md §5.3). This tries:

- d125:   10^p·{1, 2, 5}, an absolute grid (2 bits of multiplier: the reserved flag bits);
- d1235:  10^p·{1, 2, 10/3, 5} (also 2 bits);
- nd255:  (K0 + N·q)·B^p / D, B = 10 or 2, D ≤ 255, found by an approximate GCD of the
          block's values; K0 sits on the fine grid B^p/D, so offset grids (°F) fit too;
- nd4095: the same with D ≤ 4095 (ADC full scales like 10 V / 4095);
- float:  a + q·S with S any float64 (reference: the most general linear grid, never bit-exact).

Method: each dataset is encoded with the real encoder (defaults). A family re-grids a block only
when it finds a step coarser than the block's current one that every sample fits within a
quarter of the power-of-two step, the rule decimal detection uses today; then order and residuals
are recomputed with the encoder's own kernels and the block group is compressed with the real
compress (effort 4). The extra per-block column (N, D or S) is zstd'd on its own and added.

    cd experimental && uv run python grids/grids.py
"""

import math
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
sys.path.insert(0, str(ROOT / "tests"))

import _fresh_numba_cache  # noqa: F401,E402  (before fluxcode)
import numpy as np  # noqa: E402
import zstandard  # noqa: E402
from _signals import DISCRETE, KINDS, MINUTE, discrete_minute, minute  # noqa: E402
from fluxcode import Params, decode_group  # noqa: E402
from fluxcode._compress import EFFORTS, compress  # noqa: E402
from fluxcode._encoder import pick_order, residual  # noqa: E402
from fluxcode._format import BLOCK_FLAG_DECIMAL, BLOCK_FLAG_ORDER  # noqa: E402
from fluxcode._group import encode_rows  # noqa: E402

BLOCK = 1000
SEEDS = (1, 2, 3)
EFFORT = EFFORTS[Params().effort]
SIZES = np.full(MINUTE // BLOCK, BLOCK, np.int64)


# --- New datasets: a sensor's grid the writer doesn't know ---------------------------------------

def _text(x, den):
    """x rounded to 1/den as parsing its decimal text gives (den a power of ten)."""
    return np.round(x * den) / den


def _temp_c(seed):
    rng = np.random.default_rng(seed)
    t = np.arange(MINUTE) / 1000
    return 20 + 5 * np.sin(2 * np.pi * t / 37 + rng.uniform(0, 6)) + np.cumsum(rng.normal(0, 0.02, MINUTE))


def _volts(seed, lo, hi):
    """A 0.2-1 Hz wobble plus drift, spanning about [lo, hi]."""
    rng = np.random.default_rng(seed)
    t = np.arange(MINUTE) / 1000
    mid, amp = (lo + hi) / 2, (hi - lo) / 2
    v = mid + 0.8 * amp * np.sin(2 * np.pi * 0.37 * t + rng.uniform(0, 6)) + np.cumsum(rng.normal(0, amp / 3000, MINUTE))
    return np.clip(v, lo, hi)


def new_dataset(name, seed):
    rw = minute("random-walk", seed)
    ns = minute("noisy-sine", seed)
    if name == "rw q0.2":
        return np.round(rw * 5) / 5
    if name == "rw q0.05":
        return np.round(rw * 20) / 20
    if name == "rw q0.02":
        return np.round(rw * 50) / 50
    if name == "rw q0.5 (binary)":
        return np.round(rw * 2) / 2
    if name == "rw q0.25 (binary)":
        return np.round(rw * 4) / 4
    if name == "rw k·0.2 float":  # counts times a float step: not the double nearest each decimal
        return np.round(rw * 5) * 0.2
    if name == "rw q0.2 via float32":
        return (np.round(rw * 5) / 5).astype(np.float32).astype(np.float64)
    if name == "rw q1/3":
        return np.round(rw * 3) / 3
    if name == "noisy-sine q0.2 (σ=5)":
        return np.round(ns * 5) / 5
    if name == "°F text from 0.1 °C":
        return _text(32 + 1.8 * _text(_temp_c(seed), 10), 100)
    if name == "°F float from 0.1 °C":
        return 32 + 1.8 * _text(_temp_c(seed), 10)
    if name == "0.5 °C sensor, slow":
        return np.round(_temp_c(seed) * 2) / 2
    if name == "ADC12 10V/4096":
        return np.round(_volts(seed, 0, 10) / 10 * 4096) * (10 / 4096)
    if name == "ADC12 10V/4095":
        return np.round(_volts(seed, 0, 10) / 10 * 4095) * (10 / 4095)
    if name == "ADC16 ±10V, ±2V used":
        return np.round(_volts(seed, -2, 2) / 20 * 65536) * (20 / 65536)
    if name == "4-20 mA, 0-250 kPa":  # counts 6553..32767 of 0..32767 scaled to 0..250
        c = np.round(6553 + (_volts(seed, 40, 210) / 250) * (32767 - 6553))
        return (c - 6553) * (250 / (32767 - 6553))
    if name == "calibrated gain":
        c = np.round(_volts(seed, 1, 9) / 0.0137)
        return -3.21 + c * 0.0137
    raise ValueError(name)


NEW = ["rw q0.2", "rw q0.05", "rw q0.02", "rw q0.5 (binary)", "rw q0.25 (binary)", "rw k·0.2 float",
       "rw q0.2 via float32", "rw q1/3", "noisy-sine q0.2 (σ=5)", "°F text from 0.1 °C", "°F float from 0.1 °C",
       "0.5 °C sensor, slow", "ADC12 10V/4096", "ADC12 10V/4095", "ADC16 ±10V, ±2V used", "4-20 mA, 0-250 kPa",
       "calibrated gain"]


def dataset(group, name, seed):
    if group == "continuous":
        return minute(name, seed)
    if group == "discrete":
        return discrete_minute(name, seed)
    return new_dataset(name, seed)


# --- Grids --------------------------------------------------------------------------------------

class Grid:
    """y = (K0 + N·q) · base^p / D (rational), or anchor + q·step (float)."""

    def __init__(self, q, step, recon, extra, anchor):
        self.q, self.step, self.recon, self.extra = q, step, recon, extra
        self.anchor = anchor  # the anchor column's int64: K0, or a float anchor's bits
        self.p = 0  # the grid parameter column


def _scale(base, p, den):
    if base == 10:
        return den * 10.0 ** -p if p <= 0 else den / 10.0 ** p
    return den * math.ldexp(1.0, -p)


def rational_grid(x, base, p, num, den, tol, absolute):
    """The block on the grid of step num/den · base^p, every sample within tol, or None."""
    s = _scale(base, p, den)
    y = x * s
    k = np.floor(y + 0.5)
    if np.max(np.abs(y - k)) > tol * s or np.max(np.abs(k)) >= 2.0 ** 52:
        return None
    k = k.astype(np.int64)
    k0 = int(k.min())
    off = k if absolute else k - k0
    if np.any(off % num):
        return None
    q = (k - k0) // num
    if q.max() > 65535:
        return None
    numer = (k0 + num * q).astype(np.float64)
    if base == 10:
        recon = numer / (den * 10.0 ** -p) if p <= 0 else numer * 10.0 ** p / den
    else:
        recon = numer / (den * math.ldexp(1.0, -p)) if p <= 0 else numer * math.ldexp(1.0, p) / den
    g = Grid(q.astype(np.int32), num / den * float(base) ** p, recon, (num, den), k0)
    g.p = p
    return g


def search_multipliers(x, lo, hi, ps, mults):
    """The coarsest absolute grid mult · 10^p coarser than ps; mults as (num, den), coarsest first."""
    p = min(math.floor(math.log10(hi - lo)), 22)
    while p >= -22:
        for num, den in mults:
            if num / den * 10.0 ** p <= ps:
                return None
            g = rational_grid(x, 10, p, num, den, ps / 4, absolute=True)
            if g is not None:
                return g
        p -= 1
    return None


def approx_gcd(x, lo, tol, ps):
    """The coarsest step S > ps with every sample within tol of lo + k·S, or None."""
    u = np.unique(x) - lo
    gaps = np.diff(u)
    gaps = gaps[gaps > 2 * tol]
    if gaps.size == 0:
        return None
    g0 = gaps.min()
    for m in range(1, 17):
        step = g0 / m
        if step <= ps:
            return None
        ones = gaps[np.abs(gaps - step) <= 2 * tol]  # gaps of one step: their mean averages the errors out
        if ones.size:
            step = float(ones.mean())
        lim, ok = 4 * step, True
        while ok:
            sub = u[u <= lim]
            k = np.round(sub / step)
            ok = np.max(np.abs(sub - k * step)) <= tol
            if ok and k.any():
                step = float(k @ sub / (k @ k))  # least squares through the minimum
            if lim >= u[-1]:
                break
            lim *= 4
        if ok and np.max(np.abs(u - np.round(u / step) * step)) <= tol:
            return step
    return None


def search_rational(x, lo, hi, ps, max_den):
    step = approx_gcd(x, lo, ps / 4, ps)
    if step is None:
        return None
    for base in (10, 2):
        p = math.floor(math.log(step, base) + 1e-6)
        if not -1074 <= p <= 1023 or (base == 10 and not -22 <= p <= 22):
            continue
        r = Fraction(step / float(base) ** p).limit_denominator(max_den)
        if r.numerator == 0 or r.numerator >= 4096:
            continue
        g = rational_grid(x, base, p, r.numerator, r.denominator, ps / 4, absolute=False)
        if g is not None:
            return g
    return None


def search_float(x, lo, hi, ps):
    step = approx_gcd(x, lo, ps / 4, ps)
    if step is None:
        return None
    q = np.round((x - lo) / step)
    recon = lo + q * step
    if np.max(np.abs(recon - x)) > ps / 4 or q.max() > 65535:
        return None
    return Grid(q.astype(np.int32), step, recon, step, int(np.float64(lo).view(np.int64)))


FAMILIES = {
    "today, fixed": lambda *a: fixed_today(*a),
    "d125": lambda x, lo, hi, ps: search_multipliers(x, lo, hi, ps, [(5, 1), (2, 1), (1, 1)]),
    "d1235": lambda x, lo, hi, ps: search_multipliers(x, lo, hi, ps, [(5, 1), (10, 3), (2, 1), (1, 1)]),
    "nd255": lambda x, lo, hi, ps: search_rational(x, lo, hi, ps, 255),
    "nd4095": lambda x, lo, hi, ps: search_rational(x, lo, hi, ps, 4095),
    "float": search_float,
}


# --- Measuring ---------------------------------------------------------------------------------

_ZSTD = zstandard.ZstdCompressor(level=3)
_EMPTY = len(_ZSTD.compress(b"\0"))


def column_bytes(rows, values, dtype):
    """What a byte-planed per-block column adds to the zstd of the block group's real columns."""
    from fluxcode import _bitpacking, _format
    offsets = _format.layout(rows.block_flags, rows.block_sizes)
    body = _bitpacking.write_group(rows.block_flags, rows.block_sizes, rows.grid_params, rows.value_anchors,
                                   rows.residuals, rows.codes, rows.time_rows, offsets)
    cols = body[: _format.residual_start(rows.block_flags.shape[0], False)].tobytes()
    raw = np.ascontiguousarray(np.asarray(values, dtype).view(np.uint8).reshape(len(values), -1).T).tobytes()
    return len(_ZSTD.compress(cols + raw)) - len(_ZSTD.compress(cols))


def baseline(x):
    rows, _ = encode_rows(x, SIZES, Params())
    pow2, _ = encode_rows(x, SIZES, Params(decimal_detection=False))
    return rows, pow2


def block_step(rows, pow2, b):
    """(current step, power-of-two step ps) of block b, or None for a constant block."""
    e = int(pow2.grid_params[b])
    ps = math.ldexp(1.0, e)
    cur = 10.0 ** int(rows.grid_params[b]) if rows.block_flags[b] & BLOCK_FLAG_DECIMAL else ps
    return cur, ps


def pow2_effective(x, step):
    """step times the power of two dividing every index on the absolute grid of step: the zero
    low bits of the indices cost almost nothing in bit planes, odd factors do."""
    k = np.round(x / step)
    g = int(np.gcd.reduce((k - k.min()).astype(np.int64)))
    return step * ((g & -g) if g else 1)


def pow2_grid(x, e):
    k = np.rint(x * math.ldexp(1.0, -e))
    q = (k - k.min()).astype(np.int32)
    step = math.ldexp(1.0, e)
    g = Grid(q, step, (k.min() + q) * step, None, int(np.float64(k.min() * step).view(np.int64)))
    g.p = e
    return g


def fixed_today(x, lo, hi, ps):
    """Today's grids, but a decimal grid only when coarser than the binary grid's effective step."""
    g = search_multipliers(x, lo, hi, ps, [(1, 1)])
    if g is not None and g.step <= pow2_effective(x, ps) * (1 + 1e-9):
        return pow2_grid(x, round(math.log2(ps)))
    return g


def evaluate(x, family):
    rows, pow2 = baseline(x)
    base_bytes = compress(rows, x.size, EFFORT, 0)
    decoded = decode_group(base_bytes).values
    if family is None:
        return len(base_bytes), 0, _exact(decoded, x), 0
    rows = rows._replace(block_flags=rows.block_flags.copy(), grid_params=rows.grid_params.copy(),
                         value_anchors=rows.value_anchors.copy(), residuals=rows.residuals.copy())
    recon, extras, changed = decoded.copy(), [], 0
    for b in range(SIZES.size):
        sl = slice(b * BLOCK, (b + 1) * BLOCK)
        xb = x[sl]
        lo, hi = float(xb.min()), float(xb.max())
        extras.append((1, 1) if family != "float" else 0.0)
        if not hi > lo:
            continue
        cur, ps = block_step(rows, pow2, b)
        eff = pow2_effective(xb, cur)
        g = FAMILIES[family](xb, lo, hi, ps)
        if g is None:
            continue
        gq = int(np.gcd.reduce(g.q.astype(np.int64)))
        g_eff = g.step * ((gq & -gq) if gq else 1)
        # better: a coarser effective grid, or the same one held more compactly
        if not (g_eff > eff * (1 + 1e-9) or (g_eff >= eff * (1 - 1e-9) and g.step > cur * (1 + 1e-9))):
            continue
        order = pick_order(g.q, min(250, BLOCK), 15)
        residual(g.q, order, rows.residuals[sl])
        rows.block_flags[b] = (int(rows.block_flags[b]) & ~BLOCK_FLAG_ORDER & 0xFF) | order | BLOCK_FLAG_DECIMAL
        rows.grid_params[b] = g.p
        rows.value_anchors[b] = g.anchor
        recon[sl] = g.recon
        extras[-1] = g.extra
        changed += 1
    size = len(compress(rows, x.size, EFFORT, 0))
    if family == "float":
        extra = column_bytes(rows, extras, np.float64)
    elif family in ("d125", "d1235", "today, fixed"):
        extra = 0  # 2 bits of multiplier in the reserved flag bits
    else:
        extra = column_bytes(rows, [n * 4096 + d - 1 for n, d in extras], np.uint32)  # N, D: 12 bits each
    return size + extra, changed, _exact(recon, x), np.max(np.abs(recon - x))


def _exact(y, x):
    return float(np.mean(y.view(np.int64) == x.view(np.int64)))


def main():
    groups = [("continuous", KINDS), ("discrete", [d[0] for d in DISCRETE]), ("new", NEW)]
    fams = [None, *FAMILIES]
    print("| dataset | " + " | ".join(["baseline B/blk", *FAMILIES]) + " | exact % (base → nd255) |")
    print("|---|" + "---:|" * (len(fams) + 1))
    for group, names in groups:
        for name in names:
            res = {f: [] for f in fams}
            for seed in SEEDS:
                x = dataset(group, name, seed)
                for f in fams:
                    res[f].append(evaluate(x, f))
            base = np.mean([r[0] for r in res[None]])
            cells = [f"{base / SIZES.size:.0f}"]
            for f in FAMILIES:
                s = np.mean([r[0] for r in res[f]])
                ch = sum(r[1] for r in res[f])
                cells.append(f"{100 * (s / base - 1):+.1f}%" + (f" ({ch})" if ch else ""))
            ex0 = 100 * np.mean([r[2] for r in res[None]])
            ex1 = 100 * np.mean([r[2] for r in res["nd255"]])
            print(f"| {name} | " + " | ".join(cells) + f" | {ex0:.0f} → {ex1:.0f} |", flush=True)


if __name__ == "__main__":
    main()
