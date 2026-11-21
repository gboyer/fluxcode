import math, time
import numpy as np
import harness as H
from numba import njit
from fluxcode import _encoder, _noise
import sys; sys.path.insert(0, str(H.ROOT / "tests"))
from _signals import minute

@njit
def loop_center(q, qq, reps, e, a, order=0):
    s = 0.0
    for _ in range(reps):
        qq[:] = q
        g, an = _encoder.center(qq, order, 0, e, np.int64(0), a, np.int64(0))
        s += an
    return s

@njit
def loop_copy(q, qq, reps):
    s = 0
    for _ in range(reps):
        qq[:] = q
        s += qq[5]
    return s

@njit
def loop_noise(x, sc, buf, reps, ms):
    s = 0.0
    for _ in range(reps):
        s += _noise.noise(x, sc, buf, ms)[0]
    return s

@njit
def loop_pick(q, reps):
    s = 0
    for _ in range(reps):
        s += _encoder.pick_order(q, 15)
    return s

def best(f, reps=5000):
    f(10); b = 1e9
    for _ in range(5):
        t = time.perf_counter(); f(reps); b = min(b, (time.perf_counter() - t) / reps)
    return b * 1e6

for kind in ("sin-4.12hz", "noisy-sine"):
    x = minute(kind, 1)[:1000].copy()
    lo, hi = x.min(), x.max()
    e = _encoder.range_exponent(lo, hi, 16)
    q = np.empty(1000, np.int32); _encoder.quantize(x, lo, e, q)
    qq = q.copy(); buf = np.empty(1000); sc = _encoder.range_scale(lo, hi); a = math.ldexp(3.0, e); ms16 = math.ldexp(1.0, e) / 0.25
    c = best(lambda r: loop_center(q, qq, r, e, a)) - best(lambda r: loop_copy(q, qq, r))
    print(kind, f"center {c:.3f}", f"noise {best(lambda r: loop_noise(x, sc, buf, r, 0.0)):.2f} at-16-bits {best(lambda r: loop_noise(x, sc, buf, r, ms16)):.2f} base {best(lambda r: loop_noise(x, sc, buf, r, np.inf)):.2f}", f"pick {best(lambda r: loop_pick(q, r)):.3f} µs")
