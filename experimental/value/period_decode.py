# Run from experimental/value: uv run --with scipy python period_decode.py
"""The period decode loop: scalar (as before) vs one run of period_len at a time (now), and whole-group decode."""
import sys, time
sys.path.insert(0, ".")
import numpy as np
from numba import njit
import corpus, harness as H
import fluxcode
from fluxcode import _decoder, _encoder

@njit(nogil=True, cache=True)
def scalar(errors_in_out, period, weight):
    num_samples = errors_in_out.shape[0]
    head = min(num_samples, period + 1)
    near_weight = np.int32(64 - weight)
    far_weight = np.int32(weight)
    for sample_idx in range(head, num_samples):
        prediction = (near_weight * errors_in_out[sample_idx - period] + far_weight * errors_in_out[sample_idx - period - 1] + 32) >> 6
        errors_in_out[sample_idx] = ((errors_in_out[sample_idx] + prediction + 32768) & 0xFFFF) - 32768

@njit(nogil=True)
def loop(fn, blocks, period, weight, reps):
    for _ in range(reps):
        for b in range(blocks.shape[0]):
            fn(blocks[b], period, weight)

rng = np.random.default_rng(0)
for period in (83, 128, 20):
    base = rng.integers(-200, 200, (240, 1000)).astype(np.int32)
    a, b = base.copy(), base.copy()
    scalar(a[0], period, 21); _decoder.unpredict_period(b[0], period, 21)
    a, b = base.copy(), base.copy()
    loop(scalar, a, period, 21, 1); loop(_decoder.unpredict_period, b, period, 21, 1)
    assert np.array_equal(a, b), "results differ"
    ts, tv = [], []
    for _ in range(15):
        a = base.copy(); t0 = time.perf_counter(); loop(scalar, a, period, 21, 20); ts.append(time.perf_counter() - t0)
        b = base.copy(); t0 = time.perf_counter(); loop(_decoder.unpredict_period, b, period, 21, 20); tv.append(time.perf_counter() - t0)
    s, v = min(ts) / (240 * 20) * 1e6, min(tv) / (240 * 20) * 1e6
    print(f"period length {period}: loop per 1000-sample block scalar {s:.2f} us -> runs {v:.2f} us ({v/s-1:+.0%})")

# Whole decode of motor-current groups at effort 6, against the same groups without the predictor
x = next(x for s, n, x, bl, bpg in corpus.corpus(["fast"]) if n == "motor-current")
_encoder.PERIOD_EFFORT = 6
on = [fluxcode.encode_group(p, fluxcode.Params(effort=6)).group for p, _ in H.groups_of(x, 1000, 60)]
_encoder.PERIOD_EFFORT = 99
off = [fluxcode.encode_group(p, fluxcode.Params(effort=6)).group for p, _ in H.groups_of(x, 1000, 60)]
for g in on + off: fluxcode.decode_group(g)
t_on, t_off = [], []
for _ in range(15):
    t0 = time.perf_counter(); [fluxcode.decode_group(g) for g in on for _ in range(5)]; t_on.append(time.perf_counter() - t0)
    t0 = time.perf_counter(); [fluxcode.decode_group(g) for g in off for _ in range(5)]; t_off.append(time.perf_counter() - t0)
nb = 5 * 60 * len(on)
print(f"motor-current decode per block: without the predictor {min(t_off)/nb*1e6:.2f} us, with it {min(t_on)/nb*1e6:.2f} us ({min(t_on)/min(t_off)-1:+.0%})")
