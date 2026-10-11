# fluxprune speed (bench/prune.py, 2026-10-10, AC power)

arm64, Darwin 25.6.0, Python 3.11.13, numpy 2.4.6; one thread, best of 5, 1 s scans (datetime64[ns] times).

rank: the O(n log n) ranking pass. prune: rank + the automatic count + 2 refinement passes. max_error: prune with max_error = 1% of the range (the swinging-door fill). + cap: the same with max_points = n / 100, which binds on the noisy signals (then a bisection over the automatic count, and the worst-first repair). peaks: a 60 s period and a prominence of 5% of the range. Times added to rank are against rank; peaks is a separate call. Each call is timed on its own, so a difference below ~0.05 ms at 3,600 points is run-to-run noise.

(generated in 77 s)

| n | signal | rank | ns/point | prune | max_error | max_error + cap | peaks | points kept (prune / max_error / + cap) |
|---|---|---|---|---|---|---|---|---|
| 3,600 | smooth + noise | 0.62 ms | 173 | +0.11 ms (+17%) | +0.11 ms (+17%) | +0.14 ms (+23%) | 0.14 ms (23% of rank) | 38 / 3,072 / 36 |
| 3,600 | random walk + noise | 0.63 ms | 174 | +0.12 ms (+19%) | +0.12 ms (+20%) | +0.17 ms (+28%) | 0.14 ms (23% of rank) | 47 / 3,224 / 36 |
| 3,600 | steps | 0.61 ms | 169 | -0.02 ms (-4%) | -0.04 ms (-7%) | -0.05 ms (-8%) | 0.10 ms (17% of rank) | 14 / 14 / 14 |
| 3,600 | white noise | 0.59 ms | 163 | +0.08 ms (+14%) | +0.10 ms (+16%) | +0.13 ms (+23%) | 0.15 ms (25% of rank) | 90 / 3,445 / 36 |
| 3,600 | vibration | 0.57 ms | 158 | +0.10 ms (+17%) | +0.10 ms (+18%) | +0.15 ms (+27%) | 0.11 ms (19% of rank) | 96 / 2,696 / 36 |
| 86,400 | smooth + noise | 23.96 ms | 277 | +2.88 ms (+12%) | +3.19 ms (+13%) | +4.78 ms (+20%) | 3.91 ms (16% of rank) | 21 / 8,855 / 864 |
| 86,400 | random walk + noise | 24.49 ms | 283 | +2.80 ms (+11%) | +3.53 ms (+14%) | +5.37 ms (+22%) | 5.07 ms (21% of rank) | 32 / 62,684 / 864 |
| 86,400 | steps | 21.66 ms | 251 | +2.89 ms (+13%) | +2.70 ms (+12%) | +3.08 ms (+14%) | 3.02 ms (14% of rank) | 87 / 303 / 303 |
| 86,400 | white noise | 24.57 ms | 284 | +4.61 ms (+19%) | +3.88 ms (+16%) | +6.14 ms (+25%) | 7.96 ms (32% of rank) | 74 / 81,458 / 864 |
| 86,400 | vibration | 24.61 ms | 285 | +2.14 ms (+9%) | +3.03 ms (+12%) | +7.00 ms (+28%) | 3.59 ms (15% of rank) | 258 / 64,004 / 864 |
| 1,000,000 | smooth + noise | 507.02 ms | 507 | +34.07 ms (+7%) | +28.72 ms (+6%) | +61.06 ms (+12%) | 45.36 ms (9% of rank) | 35 / 13,275 / 10,000 |
| 1,000,000 | random walk + noise | 533.15 ms | 533 | +31.95 ms (+6%) | +24.91 ms (+5%) | +57.75 ms (+11%) | 44.98 ms (8% of rank) | 26 / 203,127 / 10,000 |
| 1,000,000 | steps | 391.18 ms | 391 | +31.02 ms (+8%) | +20.30 ms (+5%) | +21.78 ms (+6%) | 37.47 ms (10% of rank) | 197 / 3,419 / 3,419 |
| 1,000,000 | white noise | 520.39 ms | 520 | +10.51 ms (+2%) | +29.45 ms (+6%) | +66.60 ms (+13%) | 106.63 ms (20% of rank) | 61 / 937,037 / 10,000 |
| 1,000,000 | vibration | 522.16 ms | 522 | +2.10 ms (+0%) | +9.04 ms (+2%) | +81.45 ms (+16%) | 45.67 ms (9% of rank) | 242 / 730,970 / 10,000 |
