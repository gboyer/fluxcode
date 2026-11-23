"""Long-lag decay test: white noise through a kernel of span s is uncorrelated beyond s lags; a tone isn't."""
import math, sys, io, contextlib
import numpy as np
with contextlib.redirect_stdout(io.StringIO()):
    import bands as B
import corpus
LAGS = (20, 21, 22, 23)
def acf_long(x, h):
    y = np.convolve(x, h[::-1], "valid"); y = y - y.mean()
    s = 1.2533 * np.mean(np.abs(y))
    for _ in range(2):
        k = np.abs(y) <= 4 * s; s = 1.2533 * np.mean(np.abs(y[k]))
    yc = np.clip(y, -4 * s, 4 * s); d = np.dot(yc, yc)
    return max(abs(np.dot(yc[L:], yc[:-L]) / d) for L in LAGS)
cases = dict(B.cases)
sd = corpus.fast("accel-sigma-delta"); gb = corpus.fast("accel-gearbox")
rng = np.random.default_rng(1)
cases["accel-sigma-delta"] = lambda: (lambda s: sd[s:s + 1000])(int(rng.integers(0, len(sd) - 1000)))
cases["accel-gearbox"] = lambda: (lambda s: gb[s:s + 1000])(int(rng.integers(0, len(gb) - 1000)))
cases["tone 0.44 clean"] = lambda: np.sin(2 * np.pi * 0.44 * np.arange(1000) + rng.uniform(0, 6))
cases["tone 0.44 + white 1%"] = lambda: np.sin(2 * np.pi * 0.44 * np.arange(1000) + rng.uniform(0, 6)) + rng.normal(0, 0.01, 1000)
print("max |rho| at lags 20-23 (white: ~0.03-0.06 at n=1000), median of 30 blocks; σ̂ and the band gate")
for name, mk in cases.items():
    cells = []
    for f in B.BANDS:
        res = []
        for _ in range(30):
            x = mk(); s, r1, r2 = B.est(x, B.H[f]); res.append((acf_long(x, B.H[f]), s, B.gate(f, r1, r2)))
        cells.append(f"{f}: {np.median([r[0] for r in res]):.2f} σ̂{np.median([r[1] for r in res]):.3g} g{np.mean([r[2] for r in res]):.0%}")
    print(f"{name:22s} " + " | ".join(cells))
