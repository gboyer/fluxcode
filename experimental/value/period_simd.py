"""Variants of the period decode loop: which compile to vector instructions, and their speed (fresh NUMBA_CACHE_DIR to inspect)."""
import re, time
import numpy as np
from numba import njit

@njit(nogil=True)
def runs_int32(x, period_len, fraction):
    n = x.shape[0]
    near_w = np.int32(64 - fraction); far_w = np.int32(fraction)
    start = min(n, period_len + 1)
    while start < n:
        end = min(start + period_len, n)
        cur = x[start:end]; near = x[start - period_len:end - period_len]; far = x[start - period_len - 1:end - period_len - 1]
        for i in range(end - start):
            p = (near_w * near[i] + far_w * far[i] + np.int32(32)) >> np.int32(6)
            cur[i] = np.int32(np.int16(cur[i] + p))
        start = end

@njit(nogil=True)
def runs_scratch(x, period_len, fraction, scratch):
    n = x.shape[0]
    near_w = np.int32(64 - fraction); far_w = np.int32(fraction)
    start = min(n, period_len + 1)
    while start < n:
        end = min(start + period_len, n)
        m = end - start
        near = x[start - period_len:end - period_len]; far = x[start - period_len - 1:end - period_len - 1]
        pred = scratch[:m]
        for i in range(m):
            pred[i] = (near_w * near[i] + far_w * far[i] + np.int32(32)) >> np.int32(6)
        cur = x[start:end]
        for i in range(m):
            cur[i] = np.int32(np.int16(cur[i] + pred[i]))
        start = end

def report(fn, *extra):
    x = np.zeros(1000, np.int32); fn(x, 83, 21, *extra)
    ir = list(fn.inspect_llvm().values())[0]
    print(fn.__name__, "vector types:", sorted(set(re.findall(r"<(\d+) x (i\d+)>", ir))) or "none")

report(runs_int32)
report(runs_scratch, np.empty(1000, np.int32))

# correctness against the shipped decoder and timing
from fluxcode import _decoder
rng = np.random.default_rng(0)
base = rng.integers(-200, 200, (240, 1000)).astype(np.int32)
scratch = np.empty(1000, np.int32)
@njit
def run_all(fn_id, blocks, scratch, reps):
    for _ in range(reps):
        for b in range(blocks.shape[0]):
            if fn_id == 0: _decoder.unpredict_period(blocks[b], 83, 21)
            elif fn_id == 1: runs_int32(blocks[b], 83, 21)
            else: runs_scratch(blocks[b], 83, 21, scratch)
outs = []
for fid in range(3):
    y = base.copy(); run_all(fid, y, scratch, 1); outs.append(y)
print("same results:", all(np.array_equal(outs[0], o) for o in outs))
best = [1e9] * 3
for _ in range(15):
    for fid in range(3):
        y = base.copy(); t0 = time.perf_counter(); run_all(fid, y, scratch, 20); best[fid] = min(best[fid], time.perf_counter() - t0)
for name, t in zip(("shipped runs", "runs int32", "runs + scratch"), best):
    print(f"{name}: {t / 4800 * 1e6:.2f} us per 1000-sample block ({t / best[0] - 1:+.0%} vs shipped runs)")
