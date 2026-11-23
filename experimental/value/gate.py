"""Gate pass rates of the production estimator (FLUXCODE_X_NOISE_ORDER) by block length: white noise should pass,
clean tones and tone + weak noise shouldn't (a pass on a tone is a false floor)."""
import math
import numpy as np
import harness as H  # noqa: F401
from fluxcode import _noise, _encoder

rng = np.random.default_rng(5)
def rate(make, n, trials=400):
    buf = np.empty(n)
    p = 0
    for _ in range(trials):
        x = make(n)
        s, rho = _noise.noise(x, math.frexp(np.ptp(x))[1], buf)
        p += rho > _encoder.NOISE_GATE and s > 0
    return p / trials

def tone(f, snr_noise):
    return lambda n: 100 * np.sin(2 * np.pi * f * np.arange(n) + rng.uniform(0, 7)) + rng.normal(0, snr_noise, n)

cases = {"white": lambda n: rng.normal(0, 1, n), "random walk": lambda n: np.cumsum(rng.normal(0, 1, n))}
for f in (0.05, 0.2, 0.3, 0.35, 0.38, 0.4, 0.45):
    cases[f"tone {f}fs + 1e-3 noise"] = tone(f, 1e-3)
    cases[f"tone {f}fs + 1 noise (40 dB)"] = tone(f, 1.0)
print(f"k = {_noise.NOISE_ORDER}, gate share > {_encoder.NOISE_GATE}")
print("| case | n=256 | n=1000 | n=3600 |"); print("|---|---|---|---|")
for name, mk in cases.items():
    print(f"| {name} | " + " | ".join(f"{rate(mk, n):.0%}" for n in (256, 1000, 3600)) + " |")
