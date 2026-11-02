# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""What would the multiplier searches cost an encoder? numba kernels, timed against encode_group.

- index gcd: the gcd N of a block's indices on its own grid (decimal or power of two), stopping
  when it reaches 1; N > 1 then divides them (q // N). Runs on every block.
- rational: an approximate GCD of the values for a step N/D · B^p (D ≤ 4095), for blocks with no
  decimal grid and index gcd 1. It rejects on a 64-sample prefix, tracks k from consecutive
  differences (robust to a slightly-off step), fits the step and offset by least squares,
  verifies, then rationalizes (continued fractions) and checks the exact rational grid.

Single thread, best of 5 rounds after a warm-up, µs per block of 1000 (60 blocks per group).
The kernels run as separate passes after the encoder: an integrated version would also skip the
first residual pass on re-gridded blocks, so this is an upper bound. On AC power only.

    cd experimental && uv run python grids/timing.py
"""

import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bench"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))

import _fresh_numba_cache  # noqa: F401,E402  (before fluxcode)
import numpy as np  # noqa: E402
from fluxcode import Params, encode_group  # noqa: E402
from fluxcode._format import BLOCK_FLAG_DECIMAL  # noqa: E402
from fluxcode._group import encode_rows  # noqa: E402
from numba import njit  # noqa: E402

from grids import BLOCK, SIZES, dataset  # noqa: E402

PREFIX = 64
MAX_DEN = 4095
ROUNDS, WARMUP = 5, 3


@njit(cache=True)
def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


@njit(cache=True)
def index_gcd(q):
    n = np.int64(0)
    for i in range(q.shape[0]):
        v = np.int64(q[i])
        if n == 0:
            n = v
        elif v % n:
            n = _gcd(n, v)
            if n == 1:
                return 1
    return n


@njit(cache=True)
def index_gcd_fast(q):
    """index_gcd, with the divisibility test as multiplies: |q| < 2^16, so q·(1/n) rounds to
    exactly q / n when n divides q. Chunks of 64 without an exit inside vectorize; the exact
    gcd (Euclid) runs only on a chunk with a remainder."""
    n = np.int64(0)
    num = q.shape[0]
    i = 0
    while i < num and n == 0:
        n = abs(np.int64(q[i]))
        i += 1
    while i < num:
        end = min(i + 64, num)
        inv = 1.0 / n
        bad = 0.0
        for j in range(i, end):
            v = float(q[j])
            bad += abs(v - n * math.floor(v * inv + 0.5))
        if bad:
            for j in range(i, end):
                v = np.int64(q[j])
                if v % n:
                    n = _gcd(n, abs(v))
                    if n == 1:
                        return 1
        i = end
    return n


@njit(cache=True)
def compact(q, n):
    inv = 1.0 / n
    for i in range(q.shape[0]):
        q[i] = np.int32(math.floor(q[i] * inv + 0.5))


@njit(cache=True)
def best_rational(r, max_den):
    p0, q0, p1, q1 = 0, 1, 1, 0
    x = r
    for _ in range(40):
        a = math.floor(x)
        p2, q2 = a * p1 + p0, a * q1 + q0
        if q2 > max_den:
            break
        p0, q0, p1, q1 = p1, q1, p2, q2
        frac = x - a
        if frac < 1e-12:
            break
        x = 1.0 / frac
    return p1, q1


@njit(cache=True)
def _rational_grid(x, lo, base, p, num, den, tol, out_q):
    if base == 10:
        s = den * 10.0 ** -p if p <= 0 else den / 10.0 ** p
    else:
        s = den * math.ldexp(1.0, -p)
    k0 = math.floor(lo * s + 0.5)
    if abs(k0) >= 2.0 ** 52:
        return False
    inv_num = 1.0 / num
    for i in range(x.shape[0]):
        y = x[i] * s
        k = math.floor(y + 0.5)
        if abs(y - k) > tol * s:
            return False
        off = k - k0
        q = math.floor(off * inv_num + 0.5)
        if q * num != off or q > 65535:  # off < 2^17 · num here, so q·num is exact
            return False
        out_q[i] = np.int32(q)
    return True


@njit(cache=True)
def rational(x, lo, ps, out_q):
    """(found, base, p, num, den) for a step num/den · base^p coarser than ps, or found = False."""
    n = x.shape[0]
    tol = ps / 4
    g0 = math.inf
    for i in range(1, n):
        d = abs(x[i] - x[i - 1])
        if 2 * tol < d < g0:
            g0 = d
        if i == PREFIX - 1 and g0 < math.inf:
            break  # the prefix has a nonzero step: seed from it
    if g0 == math.inf or g0 <= ps:
        return False, 0, 0, 0, 0
    step = 0.0
    for m in range(1, 17):
        s = g0 / m
        if s <= ps:
            return False, 0, 0, 0, 0
        inv = 1.0 / s
        # the prefix, k tracked from consecutive differences
        k, ok = 0.0, True
        for i in range(1, min(PREFIX, n)):
            k += math.floor((x[i] - x[i - 1]) * inv + 0.5)
            if abs(x[i] - x[0] - k * s) > 2 * tol:
                ok = False
                break
        if not ok:
            continue
        # the block: least squares of u = x - x[0] on k, with an offset
        k, sk, skk, su, sku = 0.0, 0.0, 0.0, 0.0, 0.0
        for i in range(n):
            if i:
                k += math.floor((x[i] - x[i - 1]) * inv + 0.5)
            u = x[i] - x[0]
            sk += k
            skk += k * k
            su += u
            sku += k * u
        det = n * skk - sk * sk
        if det <= 0:
            continue
        slope = (n * sku - sk * su) / det
        icpt = (su - slope * sk) / n
        k = 0.0
        for i in range(n):
            if i:
                k += math.floor((x[i] - x[i - 1]) * inv + 0.5)
            if abs(x[i] - x[0] - icpt - k * slope) > tol:
                ok = False
                break
        if ok:
            step = slope
            break
    if step <= ps:
        return False, 0, 0, 0, 0
    return _rationalize(x, lo, ps, step, out_q)


@njit(cache=True)
def _rationalize(x, lo, ps, step, out_q):
    tol = ps / 4
    for base in (10, 2):
        p = math.floor(math.log(step) / math.log(base) + 1e-6)
        if base == 10 and not -22 <= p <= 22:
            continue
        r = step / float(base) ** p
        num, den = best_rational(r, MAX_DEN)
        if num == 0 or num >= 4096 or abs(num / den - r) > 1e-6 * r:
            continue
        if _rational_grid(x, lo, base, p, num, den, tol, out_q):
            return True, base, p, num, den
    return False, 0, 0, 0, -1  # den -1: a step was fitted, but isn't rational


@njit(cache=True)
def run_gcd(qs, offsets, do_compact, fast):
    found = 0
    for b in range(offsets.shape[0] - 1):
        q = qs[offsets[b]:offsets[b + 1]]
        nn = index_gcd_fast(q) if fast else index_gcd(q)
        if nn > 1:
            found += 1
            if do_compact:
                if fast:
                    compact(q, nn)
                else:
                    for i in range(q.shape[0]):
                        q[i] //= nn
    return found


@njit(cache=True)
def _prefix_fits(x, step, tol):
    inv = 1.0 / step
    k = 0.0
    for i in range(1, min(PREFIX, x.shape[0])):
        k += math.floor((x[i] - x[i - 1]) * inv + 0.5)
        if abs(x[i] - x[0] - k * step) > 2 * tol:
            return False
    return True


@njit(cache=True)
def run_rational(x, offsets, lo, hi, ps, gate, hints, out_q):
    """hints: try the previous block's grid first; skip a block whose prefix fits a step that
    didn't rationalize in the previous block."""
    found = 0
    prev_base, prev_p, prev_num, prev_den = 0, 0, 0, 0
    failed_step = 0.0
    for b in range(offsets.shape[0] - 1):
        if not gate[b] or not hi[b] > lo[b]:
            continue
        xb = x[offsets[b]:offsets[b + 1]]
        qb = out_q[offsets[b]:offsets[b + 1]]
        if hints and prev_num and _rational_grid(xb, lo[b], prev_base, prev_p, prev_num, prev_den, ps[b] / 4, qb):
            found += 1
            continue
        if hints and failed_step > ps[b] and _prefix_fits(xb, failed_step, ps[b] / 4):
            continue
        ok, base, p, num, den = rational(xb, lo[b], ps[b], qb)
        if ok:
            found += 1
            prev_base, prev_p, prev_num, prev_den = base, p, num, den
        elif den == -1:
            failed_step = _fitted_step(xb, lo[b], ps[b])
    return found


@njit(cache=True)
def _fitted_step(x, lo, ps):
    """The step rational() fitted before failing to rationalize it (recomputed: a rare path)."""
    n = x.shape[0]
    tol = ps / 4
    g0 = math.inf
    for i in range(1, n):
        d = abs(x[i] - x[i - 1])
        if 2 * tol < d < g0:
            g0 = d
        if i == PREFIX - 1 and g0 < math.inf:
            break
    for m in range(1, 17):
        s = g0 / m
        if s <= ps:
            return 0.0
        if _prefix_fits(x, s, tol):
            return s
    return 0.0


def indices(x, rows):
    """Each block's indices on the grid the encoder chose (exception blocks: zeros)."""
    q = np.zeros(x.size, np.int32)
    for b in range(SIZES.size):
        sl = slice(b * BLOCK, (b + 1) * BLOCK)
        param = int(rows.grid_params[b])
        if rows.block_flags[b] & BLOCK_FLAG_DECIMAL:
            s = 10.0 ** -param if param <= 0 else 1 / 10.0 ** param
            q[sl] = np.floor(x[sl] * s + 0.5).astype(np.int64) - int(rows.value_anchors[b])
        else:
            anchor = float(rows.value_anchors[b:b + 1].view(np.float64)[0])
            base = np.rint(anchor * math.ldexp(1.0, -param))
            if base * math.ldexp(1.0, param) == anchor:
                q[sl] = np.rint(x[sl] * math.ldexp(1.0, -param)) - base
    return q


def best(fn, reps):
    for _ in range(WARMUP):
        fn()
    t = math.inf
    for _ in range(ROUNDS):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        t = min(t, (time.perf_counter() - t0) / reps)
    return t / SIZES.size * 1e6


CASES = [("continuous", "random-walk"), ("continuous", "sin-9.87hz"), ("continuous", "noisy-sine"),
         ("continuous", "gauss-spikes"), ("continuous", "linear"),
         ("discrete", "random-walk q0.01"), ("discrete", "sensor-0.1"), ("discrete", "noisy-sine adc12"),
         ("discrete", "noisy-sine q0.01 via float32"),
         ("new", "rw q0.05"), ("new", "°F text from 0.1 °C"), ("new", "ADC12 10V/4096"),
         ("new", "rw q1/3"), ("new", "ADC12 10V/4095"), ("new", "calibrated gain"), ("new", "4-20 mA, 0-250 kPa")]


def main():
    offsets = np.arange(0, SIZES.size + 1, dtype=np.int64) * BLOCK
    print("| data | encode µs/block | index gcd µs | multiplies | found | rational µs (gated) | with hints | found | "
          "rational µs, every block, hints |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for group, name in CASES:
        x = dataset(group, name, 1)
        rows, _ = encode_rows(x, SIZES, Params(grids="decimal"))
        pow2, _ = encode_rows(x, SIZES, Params(grids="binary"))
        ps = np.ldexp(1.0, pow2.grid_params.astype(np.int64))
        q = indices(x, rows)
        out_q = np.empty(x.size, np.int32)
        enc = best(lambda: encode_group(x, Params(grids="decimal")), 20)
        t_gcd = best(lambda: run_gcd(q.copy(), offsets, True, False), 200)
        t_fast = best(lambda: run_gcd(q.copy(), offsets, True, True), 200)
        assert run_gcd(q.copy(), offsets, False, True) == run_gcd(q.copy(), offsets, False, False)
        t_copy = best(lambda: q.copy(), 200)
        found_gcd = run_gcd(q.copy(), offsets, False, True)
        gcds = np.array([index_gcd(q[o:o + BLOCK]) for o in offsets[:-1]])
        gate = ~(rows.block_flags & BLOCK_FLAG_DECIMAL).astype(bool) & (gcds <= 1)
        everywhere = np.ones(SIZES.size, bool)
        lo = np.array([x[o:o + BLOCK].min() for o in offsets[:-1]])
        hi = np.array([x[o:o + BLOCK].max() for o in offsets[:-1]])
        t_rat = best(lambda: run_rational(x, offsets, lo, hi, ps, gate, False, out_q), 200)
        t_hint = best(lambda: run_rational(x, offsets, lo, hi, ps, gate, True, out_q), 200)
        t_all = best(lambda: run_rational(x, offsets, lo, hi, ps, everywhere, True, out_q), 200)
        found_rat = run_rational(x, offsets, lo, hi, ps, gate, True, out_q)
        print(f"| {name} | {enc:.2f} | {t_gcd - t_copy:.2f} | {t_fast - t_copy:.2f} | {found_gcd} | {t_rat:.2f} | {t_hint:.2f} | {found_rat} "
              f"| {t_all:.2f} |", flush=True)


if __name__ == "__main__":
    main()
