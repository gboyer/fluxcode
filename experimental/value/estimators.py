"""Per-block white-noise estimators against the known sigma: median over blocks of estimate / truth."""
import numpy as np
from math import comb
import corpus

TRUE = {"accel-gearbox": 0.03, "motor-current": 0.05, "line-voltage": 0.03, "pressure-pump": 0.004,
        "loadcell": 0.02, "servo-position": 0.002, "flow-turbulent": 0.3, "tach-rpm": 0.4, "thermocouple-10hz": 0.015,
        "noisy-sine": 5.0, "gauss-spikes": 20 / np.sqrt(12), "impulses": 20 / np.sqrt(12), "random-walk": 0.0,
        "sin-50.3hz": 0.0, "chirp": 0.0, "sin-4.12hz": 0.0, "quadratic": 0.0, "square-2.24hz": 0.0}

def acov(y, k):
    y = y - y.mean()
    return np.dot(y[k:], y[:len(y) - k]) / (len(y) - k)

def ests(b):
    d1, d2 = np.diff(b), np.diff(b, 2)
    g = [acov(d2, k) for k in range(4)]
    out = {}
    rho = g[1] / g[0] if g[0] > 0 else 0
    out["cur(gated)"] = np.sqrt(g[0] / 6) if rho < -0.6 else 0.0
    out["Δ2"] = np.sqrt(g[0] / 6)
    out["-γ1(Δ)"] = np.sqrt(max(-acov(d1, 1), 0))
    out["γ2(Δ2)"] = np.sqrt(max(g[2], 0))
    for k in (3, 4, 6):
        out[f"Δ{k}"] = np.sqrt(np.var(np.diff(b, k)) / comb(2 * k, k))
    return out

names = corpus.FAST_NAMES + ["noisy-sine", "gauss-spikes", "impulses", "random-walk", "sin-50.3hz", "chirp", "sin-4.12hz", "quadratic", "square-2.24hz"]
rows = []
for name in names:
    x = corpus.fast(name) if name in corpus.FAST_NAMES else corpus.synth(name, minutes=1)
    blocks = x[: len(x) // 1000 * 1000].reshape(-1, 1000)[:60]
    es = [ests(b) for b in blocks]
    keys = list(es[0])
    t = TRUE[name]
    rng_med = np.median(np.ptp(blocks, axis=1))
    cells = []
    for k in keys:
        v = np.median([e[k] for e in es])
        cells.append(f"{v / t:.2f}" if t else f"{v / rng_med * 65536:.1f}st")
    rows.append((name, cells))
print("| signal | " + " | ".join(keys) + " |"); print("|---" * (len(keys) + 1) + "|")
for n, c in rows:
    print(f"| {n} | " + " | ".join(c) + " |")
print("(ratio to the true σ; for noise-free signals, the estimate in 16-bit steps of the block range)")
