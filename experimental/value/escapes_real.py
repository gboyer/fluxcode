"""The production escapes on the glitch scenarios: other-sample error in glitched blocks, sizes, and codes exact."""
import numpy as np
import harness as H  # noqa: F401
import fluxcode
rng = np.random.default_rng(3); n = 120_000
base = 5 + 0.3 * np.sin(np.arange(n) / 3000) + rng.normal(0, 0.002, n)
cases = {}
y = base.copy(); idx = rng.choice(n, 40, replace=False); y[idx] = -9999.0; cases["-9999 codes"] = (y, idx)
y = base.copy(); idx = rng.choice(n, 40, replace=False); y[idx] = 1e6; cases["1e6 codes"] = (y, idx)
y = base.copy(); idx = rng.choice(n, 200, replace=False); y[idx[:100]] = -9999.0; y[idx[100:]] = 1e6; cases["both sides"] = (y, idx)
y = base.copy(); st = rng.choice(n - 20, 6, replace=False); idx = np.concatenate([np.arange(s, s + 20) for s in st]); y[idx] = -9999.0; cases["dropout runs"] = (y, idx)
dec = np.round(base * 100) / 100; y = dec.copy(); idx = rng.choice(n, 40, replace=False); y[idx] = -99999.0; cases["0.01 decimals + code"] = (y, idx)
noisy = 5 + rng.normal(0, 2.0, n); y = noisy.copy(); idx = rng.choice(n, 40, replace=False); y[idx] = 500.0; cases["noisy + spikes"] = (y, idx)
for name, (y, idx) in cases.items():
    groups = [fluxcode.encode_group(y[s:s + 60000]).group for s in range(0, n, 60000)]
    d = np.concatenate([fluxcode.decode_group(g).values for g in groups])
    mask = np.ones(n, bool); mask[idx] = False
    err = np.where(mask, np.abs(d - y), 0).reshape(-1, 1000)
    hit = np.unique(idx // 1000)
    print(f"{name:22s} bytes {sum(map(len, groups)):7d}  other-sample error in hit blocks {err[hit].max():.3g}  "
          f"glitch values exact: {np.array_equal(d[idx], y[idx])}")
