"""High-pass filters for the noise estimate: sigma ratio to truth, and gate pass rate (rho < 0.9 rho_white)."""
import numpy as np
import corpus
from estimators import TRUE

def poly(*fs):
    h = np.array([1.0])
    for f in fs:
        h = np.convolve(h, f)
    return h
D = [1, -1]
FILTERS = {
    "Δ2": poly(D, D),
    "Δ4": poly(D, D, D, D),
    "Δ6": poly(*[D] * 6),
    "Δ2·Q·T": poly(D, D, [1, 0, 1], [1, 1, 1]),        # zeros at 0,0,1/4,1/3
    "Δ2·Q·T·H": poly(D, D, [1, 0, 1], [1, 1, 1], [1, 1, 1]),  # double zero at 1/3
    "Δ2·Q·T·S": poly(D, D, [1, 0, 1], [1, 1, 1], [1, 2 * np.cos(2 * np.pi * 0.375) , 1]),  # +zero at 3/8 (non-integer)
    "Δ2·Q·Q·T": poly(D, D, [1, 0, 1], [1, 0, 1], [1, 1, 1]),
}

def white_rho(h):
    return np.dot(h[1:], h[:-1]) / np.dot(h, h)

def est(b, h):
    y = np.convolve(b, h, "valid")
    y = y - y.mean()
    s = np.sqrt(np.mean(y * y)) / np.linalg.norm(h)
    rho = np.dot(y[1:], y[:-1]) / np.dot(y, y)
    return s, rho

names = corpus.FAST_NAMES + ["noisy-sine", "gauss-spikes", "random-walk", "sin-50.3hz", "chirp", "sin-4.12hz", "square-2.24hz"]
print("white rho:", {k: round(white_rho(h), 3) for k, h in FILTERS.items()})
print("| signal | " + " | ".join(FILTERS) + " |"); print("|---" * (len(FILTERS) + 1) + "|")
for name in names:
    x = corpus.fast(name) if name in corpus.FAST_NAMES else corpus.synth(name, minutes=1)
    blocks = x[: len(x) // 1000 * 1000].reshape(-1, 1000)[:60]
    t = TRUE[name]
    cells = []
    for k, h in FILTERS.items():
        e = [est(b, h) for b in blocks]
        g = [s for s, r in e if r < 0.9 * white_rho(h)]
        frac = len(g) / len(e)
        v = np.median([s for s, _ in e])
        if t:
            cells.append(f"{v / t:.2f} {frac:.0%}")
        else:
            cells.append(f"{v / np.median(np.ptp(blocks, axis=1)) * 65536:.1f}st {frac:.0%}")
    print(f"| {name} | " + " | ".join(cells) + " |")
