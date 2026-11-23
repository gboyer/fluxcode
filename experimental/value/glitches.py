"""Single-sample glitches (error codes, overrange) in a noisy 1 kHz signal: the error on the other samples of
a glitched block, against a clean block's."""
import numpy as np
import harness as H  # noqa: F401
import fluxcode
rng = np.random.default_rng(8)
n = 240_000
x = 5 + 0.3 * np.sin(np.arange(n) / 3000) + rng.normal(0, 0.002, n)
glitch = rng.choice(n, 40, replace=False)
for code in (-9999.0, 1e6, 0.0, 10.0):
    y = x.copy(); y[glitch] = code
    groups = [fluxcode.encode_group(y[s:s + 60000]).group for s in range(0, n, 60000)]
    d = np.concatenate([fluxcode.decode_group(g).values for g in groups])
    err = np.abs(d - y); err[glitch] = 0
    blocks = err.reshape(-1, 1000).max(1)
    hit = np.unique(glitch // 1000)
    clean = np.setdiff1d(np.arange(n // 1000), hit)
    print(f"glitch to {code:>8}: bits/sample {8*sum(map(len, groups))/n:.2f}, max error on the other samples: "
          f"glitched blocks {np.median(blocks[hit]):.2g} (worst {blocks[hit].max():.2g}), clean blocks {np.median(blocks[clean]):.2g}"
          f"  (noise σ 0.002)")
