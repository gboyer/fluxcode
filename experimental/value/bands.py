"""Modulated 12th-difference kernels: the noise level in a band centered at f0 (0.5, 0.375, 0.25 of fs). Robust
std (MAD) of the filtered block over the kernel's norm, and its lag-1/lag-2 autocorrelation against white noise's
through the same kernel."""
import math
import numpy as np
from scipy.signal import firwin, lfilter
import corpus

K = 12
C = np.array([(-1) ** (K - j) * math.comb(K, j) for j in range(K + 1)], float)
def kernel(f0):
    h = C * np.cos(2 * np.pi * (0.5 - f0) * np.arange(K + 1))
    if f0 < 0.5:
        h = np.convolve(h, [1, -4, 6, -4, 1])  # 4 more zeros at DC: slow signals stay out
    return h
def white_acf(h, lag):
    return np.dot(h[lag:], h[:len(h) - lag]) / np.dot(h, h)
def est(x, h):
    y = np.convolve(x, h[::-1], "valid"); y = y - y.mean()
    s = 1.2533 * np.mean(np.abs(y))
    for _ in range(2):
        k = np.abs(y) <= 4 * s; s = 1.2533 * np.mean(np.abs(y[k]))
    yc = np.clip(y, -4 * s, 4 * s)
    r1 = np.dot(yc[1:], yc[:-1]) / np.dot(yc, yc); r2 = np.dot(yc[2:], yc[:-2]) / np.dot(yc, yc)
    return s / np.linalg.norm(h), r1, r2
BANDS = (0.5, 0.375, 0.25, 0.125)
H = {f: kernel(f) for f in BANDS}
W = {f: (white_acf(H[f], 1), white_acf(H[f], 2)) for f in BANDS}
print("white acf:", {f: tuple(round(v, 3) for v in W[f]) for f in BANDS})
def gate(f, r1, r2):
    w1, w2 = W[f]
    return abs(r1 - w1) < 0.1 * max(abs(w1), 0.3) + 0.03 and abs(r2 - w2) < 0.2 * max(abs(w2), 0.3) + 0.03

rng = np.random.default_rng(4); n = 1000
white = rng.normal(0, 1, 20000)
cases = {"white": lambda: rng.normal(0, 1, n)}
for cut in (0.45, 0.4, 0.35, 0.25):
    yy = lfilter(firwin(255, cut * 2), 1, white)
    cases[f"band-limited {cut}"] = (lambda yy=yy: (lambda s: yy[s:s + n] / np.std(yy))(int(rng.integers(400, 18000))))
cases["random walk"] = lambda: np.cumsum(rng.normal(0, 1, n))
cases["tone 0.3 clean"] = lambda: np.sin(2 * np.pi * 0.3 * np.arange(n) + rng.uniform(0, 6))
cases["tone 0.37 clean"] = lambda: np.sin(2 * np.pi * 0.37 * np.arange(n) + rng.uniform(0, 6))
cases["tone 0.2 + 40dB"] = lambda: 100 * np.sin(2 * np.pi * 0.2 * np.arange(n) + rng.uniform(0, 6)) + rng.normal(0, 1, n)
cases["slow sine + bl0.4"] = lambda: 100 * np.sin(np.arange(n) / 50) + (lambda s: lfilter(firwin(255, 0.8), 1, white)[s:s + n])(int(rng.integers(400, 18000))) * 1.0
print("| case | " + " | ".join(f"f0={f}: σ̂ gate" for f in BANDS) + " |")
for name, mk in cases.items():
    cells = []
    for f in BANDS:
        res = [est(mk(), H[f]) for _ in range(30)]
        s = np.median([r[0] for r in res]); g = np.mean([gate(f, r[1], r[2]) for r in res])
        cells.append(f"{s:.3g} {g:.0%}")
    print(f"| {name} | " + " | ".join(cells) + " |")

print()
print("plateau rule: top band gated -> its σ; else two adjacent lower bands gated and within 25% -> min")
def floor(x):
    e = {f: est(x, H[f]) for f in BANDS}
    if gate(0.5, *e[0.5][1:]):
        return e[0.5][0], "top"
    for a, b in ((0.375, 0.25), (0.25, 0.125)):
        if gate(a, *e[a][1:]) and gate(b, *e[b][1:]) and 0.8 < e[a][0] / e[b][0] < 1.25:
            return min(e[a][0], e[b][0]), f"{a}/{b}"
    return 0.0, "none"
for name, mk in cases.items():
    res = [floor(mk()) for _ in range(30)]
    used = {}
    for _, u in res:
        used[u] = used.get(u, 0) + 1
    print(f"{name:22s} σ̂ median {np.median([r[0] for r in res]):.3g}  via {used}")
