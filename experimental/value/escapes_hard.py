import numpy as np, fluxcode
from escapes import encode_with_escapes
rng = np.random.default_rng(3); n = 120_000
base = 5 + 0.3 * np.sin(np.arange(n) / 3000) + rng.normal(0, 0.002, n)
cases = {}
y = base.copy(); idx = rng.choice(n, 200, replace=False); y[idx[:100]] = -9999.0; y[idx[100:]] = 1e6; cases["both sides, ~1.7/block"] = (y, idx)
y = base.copy(); starts = rng.choice(n - 20, 6, replace=False); idx = np.concatenate([np.arange(s, s + 20) for s in starts]); y[idx] = -9999.0; cases["20-sample dropout runs"] = (y, idx)
dec = np.round(base * 100) / 100; y = dec.copy(); idx = rng.choice(n, 40, replace=False); y[idx] = -99999.0; cases["0.01 decimals + code"] = (y, idx)
noisy = 5 + rng.normal(0, 2.0, n); y = noisy.copy(); idx = rng.choice(n, 40, replace=False); y[idx] = 500.0; cases["noisy (σ 2) + spikes to 500"] = (y, idx)
for name, (y, idx) in cases.items():
    g0 = [fluxcode.encode_group(y[s:s + 60000]).group for s in range(0, n, 60000)]
    d0 = np.concatenate([fluxcode.decode_group(g).values for g in g0])
    size, d1, k = encode_with_escapes(y)
    mask = np.ones(n, bool); mask[idx] = False
    hit = np.unique(idx // 1000)
    e0 = np.abs(d0 - y)[mask].reshape(-1)  # other samples
    blocks0 = np.abs(d0 - y).reshape(-1, 1000); blocks1 = np.abs(d1 - y).reshape(-1, 1000)
    blocks0[~mask.reshape(-1, 1000)] = 0; blocks1[~mask.reshape(-1, 1000)] = 0
    print(f"{name:30s} escaped {k:4d}  bytes {sum(map(len, g0))} -> {size}  worst other-sample error in hit blocks "
          f"{blocks0[hit].max():.3g} -> {blocks1[hit].max():.3g}")
