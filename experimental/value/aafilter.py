"""Band-limited noise (a DAQ's anti-alias / sigma-delta decimation filter): does the floor still find it?"""
import math
import numpy as np
from scipy.signal import firwin, lfilter
import harness as H  # noqa: F401
import fluxcode
from fluxcode import _noise, _encoder

rng = np.random.default_rng(2)
n = 60_000
print("| cutoff (of fs) | floor σ̂ / true σ (median) | gated blocks | bits/sample | vs unfiltered white |")
print("|---|---|---|---|---|")
white = rng.normal(0, 1, n + 400)
base = len(fluxcode.encode_group(white[:n] + 100 * np.sin(np.arange(n) / 300)).group)
for cut in (0.5, 0.48, 0.45, 0.42, 0.4, 0.35, 0.3, 0.25):
    if cut >= 0.5:
        y = white[:n].copy()
    else:
        h = firwin(255, cut * 2)  # linear-phase, sharp
        y = lfilter(h, 1, white)[400:400 + n]
    true = np.std(y)
    x = y + 100 * np.sin(np.arange(n) / 300)  # a slow signal on top
    est, gated = [], 0
    for b in range(0, n, 1000):
        blk = x[b:b + 1000]
        s, r = _noise.noise(blk, _encoder.range_scale(blk.min(), blk.max()), np.empty(1000))
        est.append(s / true)
        gated += r > _encoder.NOISE_GATE and s > 0
    g = len(fluxcode.encode_group(x).group)
    print(f"| {cut} | {np.median(est):.2f} | {gated}/60 | {8*g/n:.2f} | {100*(g/base-1):+.0f}% |")
