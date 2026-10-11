# fluxprune speed (bench/prune.py, 2026-10-10, AC power)

arm64, Darwin 25.6.0, Python 3.11.13, numpy 2.4.6; one thread, best of 5, 1 s scans (datetime64[ns] times).

rank: the O(n log n) ranking pass. prune: rank + the automatic count + 2 refinement passes. max_error: prune with max_error = 1% of the range (the swinging-door fill). + cap: the same with max_points = n / 100, which binds on the noisy signals (then max_error is relaxed to the tightest bound that fits: a bisection over the fill). peaks: a 60 s period and a prominence of 5% of the range. Times added to rank are against rank; peaks is a separate call. Each call is timed on its own, so a difference below ~0.05 ms at 3,600 points is run-to-run noise.

(generated in 77 s)

| n | signal | rank | ns/point | prune | max_error | max_error + cap | peaks | points kept (prune / max_error / + cap) |
|---|---|---|---|---|---|---|---|---|
| 3,600 | smooth + noise | 0.58 ms | 162 | +0.11 ms (+19%) | +0.21 ms (+35%) | +0.18 ms (+31%) | 0.13 ms (22% of rank) | 38 / 3,072 / 36 |
| 3,600 | random walk + noise | 0.68 ms | 189 | +0.11 ms (+16%) | +0.09 ms (+13%) | +0.06 ms (+8%) | 0.14 ms (21% of rank) | 47 / 3,224 / 36 |
| 3,600 | steps | 0.56 ms | 157 | +0.05 ms (+8%) | +0.06 ms (+11%) | +0.05 ms (+9%) | 0.10 ms (18% of rank) | 14 / 14 / 14 |
| 3,600 | white noise | 0.62 ms | 171 | +0.11 ms (+17%) | +0.14 ms (+22%) | +0.12 ms (+20%) | 0.15 ms (24% of rank) | 90 / 3,445 / 35 |
| 3,600 | vibration | 0.61 ms | 170 | +0.08 ms (+13%) | +0.11 ms (+19%) | +0.11 ms (+17%) | 0.11 ms (18% of rank) | 96 / 2,696 / 35 |
| 86,400 | smooth + noise | 24.60 ms | 285 | +2.53 ms (+10%) | +3.25 ms (+13%) | +3.51 ms (+14%) | 3.84 ms (16% of rank) | 21 / 8,855 / 863 |
| 86,400 | random walk + noise | 25.01 ms | 289 | +2.80 ms (+11%) | +4.06 ms (+16%) | +3.75 ms (+15%) | 5.28 ms (21% of rank) | 32 / 62,684 / 864 |
| 86,400 | steps | 21.51 ms | 249 | +2.90 ms (+13%) | +3.38 ms (+16%) | +3.44 ms (+16%) | 3.31 ms (15% of rank) | 87 / 303 / 303 |
| 86,400 | white noise | 24.45 ms | 283 | +2.55 ms (+10%) | +3.37 ms (+14%) | +4.61 ms (+19%) | 8.29 ms (34% of rank) | 74 / 81,458 / 864 |
| 86,400 | vibration | 23.59 ms | 273 | +2.29 ms (+10%) | +3.62 ms (+15%) | +4.13 ms (+17%) | 3.72 ms (16% of rank) | 258 / 64,004 / 863 |
| 1,000,000 | smooth + noise | 483.40 ms | 483 | +53.70 ms (+11%) | +61.20 ms (+13%) | +62.43 ms (+13%) | 47.71 ms (10% of rank) | 35 / 13,275 / 9,983 |
| 1,000,000 | random walk + noise | 520.71 ms | 521 | +43.65 ms (+8%) | +56.83 ms (+11%) | +55.04 ms (+11%) | 47.47 ms (9% of rank) | 26 / 203,127 / 9,974 |
| 1,000,000 | steps | 388.35 ms | 388 | +32.49 ms (+8%) | +38.33 ms (+10%) | +38.04 ms (+10%) | 40.02 ms (10% of rank) | 197 / 3,419 / 3,419 |
| 1,000,000 | white noise | 515.89 ms | 516 | +33.93 ms (+7%) | +45.92 ms (+9%) | +61.08 ms (+12%) | 109.84 ms (21% of rank) | 61 / 937,037 / 9,996 |
| 1,000,000 | vibration | 508.65 ms | 509 | +44.85 ms (+9%) | +55.78 ms (+11%) | +71.50 ms (+14%) | 48.32 ms (9% of rank) | 242 / 730,970 / 9,943 |
