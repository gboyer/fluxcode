"""Noisy square / PWM-like signals at off-grid levels: max error as a share of the amplitude."""
import numpy as np
import harness as H  # noqa: F401
import fluxcode
rng = np.random.default_rng(3)
print("| period | noise/amp | bits/sample | max err / amp | max err / noise σ |"); print("|---|---|---|---|---|")
for period in (6, 20, 100, 1000):
    for rel in (1e-4, 1e-3, 1e-2):
        n = 60_000
        A = np.pi
        x = np.where((np.arange(n) // (period // 2)) % 2 == 0, 0.0, A) + 1.7 + rng.normal(0, rel * A, n)
        g = fluxcode.encode_group(x).group
        y = fluxcode.decode_group(g).values
        e = np.abs(y - x).max()
        print(f"| {period} | {rel:g} | {8*len(g)/n:.2f} | {e/A:.2e} | {e/(rel*A):.2f} |")
