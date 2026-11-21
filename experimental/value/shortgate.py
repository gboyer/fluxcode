"""Short blocks under the k=12 gate: pass rate and sigma estimate spread on white noise, false passes on clean
shapes (with 1e-6 relative noise), by block length."""
import math
import numpy as np
import harness as H  # noqa: F401
from fluxcode import _noise, _encoder

rng = np.random.default_rng(9)
def stats(make, n, trials=500):
    buf = np.empty(n)
    passes, sig = 0, []
    for _ in range(trials):
        x = make(n)
        s, r = _noise.noise(x, math.frexp(np.ptp(x))[1], buf)
        if r > _encoder.NOISE_GATE and s > 0:
            passes += 1
            sig.append(s)
    return passes / trials, (np.percentile(sig, [5, 50, 95]) if sig else [0, 0, 0])

t = lambda n: np.arange(n)
cases = {
    "white": lambda n: rng.normal(0, 1, n),
    "ramp+white 0.01": lambda n: t(n) * 0.5 + rng.normal(0, 0.01, n),
    "step (clean)": lambda n: np.where(t(n) > n // 2, 10.0, 0.0) + rng.normal(0, 1e-9, n),
    "slow sine": lambda n: np.sin(2 * np.pi * t(n) / 40 + rng.uniform(0, 7)),
    "tone 0.3fs": lambda n: np.sin(2 * np.pi * 0.3 * t(n) + rng.uniform(0, 7)),
    "exp decay": lambda n: np.exp(-t(n) / rng.uniform(2, 20)),
    "spike (clean)": lambda n: np.where(t(n) == n // 3, 5.0, 0.0) + 1e-9 * rng.normal(size=n),
    "square period 20": lambda n: np.where((t(n) + rng.integers(20)) % 20 < 10, 1.0, 0.0),
    "staircase every 10": lambda n: np.floor((t(n) + rng.integers(10)) / 10),
    "sawtooth period 16": lambda n: ((t(n) + rng.integers(16)) % 16) / 16.0,
    "decimal walk 0.1": lambda n: np.round(np.cumsum(rng.normal(0, 0.05, n)), 1),
}
print("| case | n=64 | n=128 | n=256 |"); print("|---|---|---|---|")
for name, mk in cases.items():
    cells = []
    for n in (64, 128, 256):
        p, s = stats(mk, n)
        cells.append(f"{p:.0%} σ̂ {s[0]:.2g}/{s[1]:.2g}/{s[2]:.2g}" if p else "0%")
    print(f"| {name} | " + " | ".join(cells) + " |")
