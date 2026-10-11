# fluxprune speed (bench/prune.py, 2026-10-10, AC power)

arm64, Darwin 25.6.0, Python 3.11.13, numpy 2.4.6; one thread, best of 5, 1 s scans (datetime64[ns] times).

rank: the O(n log n) ranking pass. prune: rank + the automatic count + 2 refinement passes. max_error: prune with max_error = 1% of the range (the swinging-door fill). + cap: the same with max_points = n / 100, which binds on the noisy signals (then max_error is relaxed to the tightest bound that fits: a bisection over the fill). peaks: a 60 s period and a prominence of 5% of the range. Times added to rank are against rank; peaks is a separate call. Each call is timed on its own, so a difference below ~0.05 ms at 3,600 points is run-to-run noise.

(generated in 77 s)

| n | signal | rank | ns/point | prune | max_error | max_error + cap | peaks | points kept (prune / max_error / + cap) |
|---|---|---|---|---|---|---|---|---|
| 3,600 | smooth + noise | 0.65 ms | 179 | +0.10 ms (+16%) | +0.16 ms (+25%) | +0.13 ms (+20%) | 0.13 ms (20% of rank) | 38 / 3,028 / 35 |
| 3,600 | random walk + noise | 0.65 ms | 182 | +0.11 ms (+17%) | +0.19 ms (+29%) | +0.14 ms (+21%) | 0.14 ms (22% of rank) | 47 / 3,219 / 36 |
| 3,600 | steps | 0.59 ms | 165 | +0.03 ms (+4%) | +0.03 ms (+5%) | +0.03 ms (+6%) | 0.10 ms (17% of rank) | 14 / 14 / 14 |
| 3,600 | white noise | 0.62 ms | 172 | +0.09 ms (+15%) | +0.18 ms (+29%) | +0.21 ms (+34%) | 0.15 ms (24% of rank) | 90 / 3,445 / 26 |
| 3,600 | vibration | 0.60 ms | 167 | +0.10 ms (+17%) | +0.19 ms (+32%) | +0.18 ms (+30%) | 0.11 ms (18% of rank) | 96 / 2,640 / 36 |
| 86,400 | smooth + noise | 24.31 ms | 281 | +2.49 ms (+10%) | +3.24 ms (+13%) | +3.72 ms (+15%) | 3.51 ms (14% of rank) | 21 / 2,849 / 864 |
| 86,400 | random walk + noise | 24.61 ms | 285 | +2.81 ms (+11%) | +5.67 ms (+23%) | +5.37 ms (+22%) | 5.27 ms (21% of rank) | 32 / 61,152 / 850 |
| 86,400 | steps | 21.94 ms | 254 | +2.99 ms (+14%) | +3.54 ms (+16%) | +3.52 ms (+16%) | 3.27 ms (15% of rank) | 87 / 300 / 300 |
| 86,400 | white noise | 24.93 ms | 289 | +2.44 ms (+10%) | +4.80 ms (+19%) | +5.66 ms (+23%) | 8.20 ms (33% of rank) | 74 / 81,395 / 864 |
| 86,400 | vibration | 23.94 ms | 277 | +2.36 ms (+10%) | +5.21 ms (+22%) | +5.56 ms (+23%) | 3.72 ms (16% of rank) | 258 / 62,547 / 825 |
| 1,000,000 | smooth + noise | 503.60 ms | 504 | +37.80 ms (+8%) | +40.48 ms (+8%) | +44.60 ms (+9%) | 48.06 ms (10% of rank) | 35 / 5,652 / 5,652 |
| 1,000,000 | random walk + noise | 527.83 ms | 528 | +20.31 ms (+4%) | +39.54 ms (+7%) | +45.14 ms (+9%) | 43.44 ms (8% of rank) | 26 / 74,494 / 9,917 |
| 1,000,000 | steps | 385.68 ms | 386 | +38.21 ms (+10%) | +53.07 ms (+14%) | +50.54 ms (+13%) | 40.32 ms (10% of rank) | 197 / 3,360 / 3,360 |
| 1,000,000 | white noise | 514.90 ms | 515 | +19.56 ms (+4%) | +57.48 ms (+11%) | +66.26 ms (+13%) | 108.95 ms (21% of rank) | 61 / 936,091 / 9,906 |
| 1,000,000 | vibration | 517.11 ms | 517 | +22.22 ms (+4%) | +64.51 ms (+12%) | +71.10 ms (+14%) | 48.20 ms (9% of rank) | 242 / 713,169 / 9,861 |
