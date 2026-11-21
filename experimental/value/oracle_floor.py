"""Upper bound on what a better white-noise estimate could save: quantize each FAST signal to the power-of-two
step <= 0.25 sigma_w (its known white noise) and encode with the noise floor off."""
import math
import numpy as np
import corpus, harness as H
import fluxcode

SIGMA = {"accel-gearbox": 0.03, "motor-current": 0.05, "line-voltage": 0.03, "pressure-pump": 0.004,
         "loadcell": 0.02, "servo-position": 0.002, "flow-turbulent": 0.3, "tach-rpm": 0.4, "thermocouple-10hz": 0.015}

def size(x, params):
    return sum(len(fluxcode.encode_group(p, params, block_len=1000).group) for p, _ in H.groups_of(x, 1000, 60))

print("| signal | today | floor off | oracle floor (0.25σ) | change |")
print("|---|---|---|---|---|")
for name in corpus.FAST_NAMES:
    x = corpus.fast(name)
    n = len(x)
    base = size(x, fluxcode.Params())
    off = size(x, fluxcode.Params(noise_floor_sigma=0))
    step = 2.0 ** math.floor(math.log2(0.25 * SIGMA[name]))
    # don't go finer than the data's own grid: use the encoder's step if it's coarser
    xq = np.round(x / step) * step
    orc = min(size(xq, fluxcode.Params(noise_floor_sigma=0)), base)
    print(f"| {name} | {8*base/n:.2f} | {8*off/n:.2f} | {8*orc/n:.2f} | {100*(orc/base-1):+.0f}% |")
