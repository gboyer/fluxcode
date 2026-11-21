import math, time
import numpy as np
import harness as H
import corpus
from fluxcode import _noise
import fluxcode
x = corpus.fast("flow-turbulent")[:120_000]
blocks = x.reshape(-1, 1000)
buf = np.empty(1000)
exps = [math.frexp(np.ptp(b))[1] for b in blocks]
for b, e in zip(blocks, exps):
    _noise.noise(b, e, buf)
best = 1e9
for _ in range(20):
    t = time.perf_counter()
    for b, e in zip(blocks, exps):
        _noise.noise(b, e, buf)
    best = min(best, (time.perf_counter() - t) / len(blocks))
g = fluxcode.encode_group(x[:60000])
best_enc = 1e9
for _ in range(20):
    t = time.perf_counter(); fluxcode.encode_group(x[:60000]); best_enc = min(best_enc, (time.perf_counter() - t) / 60)
print(f"k={_noise.NOISE_ORDER}: noise {best*1e6:.2f} µs/block, encode {best_enc*1e6:.2f} µs/block")
