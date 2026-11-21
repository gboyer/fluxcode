import time, math
import numpy as np
import harness as H, corpus
from numba import njit
from fluxcode import _encoder
import fluxcode
x = corpus.fast("motor-current")[:60000]
rows = H.encode_rows(x, np.full(60, 1000, np.int64), fluxcode.Params())
q = H.block_q(rows)[5].astype(np.int64); q = np.where(q >= 32768, q - 65536, q).astype(np.int32)
r0 = rows.residuals[5000:6000].copy()
order = int(rows.block_flags[5]) & 3

@njit
def loop_try(q, r0, reps, order):
    s = 0
    r = r0.copy(); scratch = np.empty(q.shape[0], np.int32)
    for _ in range(reps):
        r[:] = r0
        use, o, b = _encoder.try_period(q, r, scratch, order)
        s += o
    return s
@njit
def loop_est(r0, reps):
    s = 0.0
    for _ in range(reps):
        s += _encoder.estimate_bits(r0)
    return s
@njit
def loop_err(q, reps):
    sc = np.empty(q.shape[0], np.int32)
    for _ in range(reps):
        _encoder.period_errors(q, 83, 21, sc)
    return sc[5]
def best(f, reps=2000):
    f(10); b = 1e9
    for _ in range(5):
        t = time.perf_counter(); f(reps); b = min(b, (time.perf_counter() - t) / reps)
    return b * 1e6
print("try_period", best(lambda r: loop_try(q, r0, r, order)), "estimate_bits", best(lambda r: loop_est(r0, r)), "period_errors", best(lambda r: loop_err(q, r)))

@njit
def crossings(quantized, reps):
    out = 0
    for _ in range(reps):
        num_samples = quantized.shape[0]
        total = np.int64(0)
        for i in range(num_samples):
            total += quantized[i]
        first_cross = last_cross = -1
        num_cross = 0
        interval_sq = 0.0
        previous_below = quantized[0] * np.int64(num_samples) < total
        for i in range(1, num_samples):
            below = quantized[i] * np.int64(num_samples) < total
            if previous_below and not below:
                if first_cross < 0:
                    first_cross = i
                else:
                    interval = float(i - last_cross)
                    interval_sq += interval * interval
                last_cross = i
                num_cross += 1
            previous_below = below
        out += num_cross
    return out
@njit
def energies(quantized, reps, guess):
    out = np.int64(0)
    for _ in range(reps):
        span = 512
        for period in range(guess - 3, guess + 4):
            energy = np.int64(0)
            later, earlier = quantized[period:span], quantized[:span - period]
            for i in range(span - period):
                d = np.int64(later[i] - earlier[i])
                energy += d * d
            out += energy
    return out
print("crossings", best(lambda r: crossings(q, r)), "energies", best(lambda r: energies(q, r, 83)))

@njit
def lsvar(quantized, reps, period):
    out = 0.0
    sc = np.empty(quantized.shape[0], np.int32)
    num_samples = quantized.shape[0]
    for _ in range(reps):
        cross = squares = np.int64(0)
        current, near, far = quantized[period + 1:], quantized[1:num_samples - period], quantized[:num_samples - period - 1]
        for i in range(num_samples - period - 1):
            y = np.int64(current[i] - near[i])
            z = np.int64(far[i] - near[i])
            cross += y * z
            squares += z * z
        out += cross / max(squares, 1)
    return out
@njit
def alloc(reps, n):
    s = 0
    for _ in range(reps):
        c = np.empty(n, np.int16)
        c[0] = 1
        s += c[0]
    return s
print("ls", best(lambda r: lsvar(q, r, 83)), "alloc", best(lambda r: alloc(r, 1000)))
