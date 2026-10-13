# fluxprune speed (bench/prune.py, 2026-10-12, AC power)

arm64, Darwin 25.6.0, Python 3.11.13, numpy 2.4.6; one thread, best of 5, 1 s scans (datetime64[ns] times).

rank: the O(n log n) ranking pass. prune: the rmse pass (rank, the count, 2 refinement passes). max_error: prune with max_error = 1% of the range (the max_error pass, farthest reach). + cap: the same with max_points = n / 100, which binds on the noisy signals (then max_error is relaxed to the tightest bound that fits: a bisection over the max_error pass). refit: prune with refit=True (the rmse pass, then least-squares values at the kept times). Times are added to rank's. Each call is timed on its own, so a difference below ~0.05 ms at 3,600 points is run-to-run noise.

(generated in 57 s)

| n | signal | rank | ns/point | prune | max_error | max_error + cap | refit | points kept (prune / max_error / + cap) |
|---|---|---|---|---|---|---|---|---|
| 3,600 | smooth + noise | 0.35 ms | 98 | +0.09 ms (+26%) | +0.18 ms (+52%) | +0.12 ms (+35%) | +0.09 ms (+26%) | 38 / 3,028 / 35 |
| 3,600 | random walk + noise | 0.36 ms | 99 | +0.06 ms (+18%) | +0.18 ms (+52%) | +0.16 ms (+44%) | +0.07 ms (+19%) | 47 / 3,219 / 36 |
| 3,600 | steps | 0.29 ms | 79 | +0.05 ms (+16%) | +0.08 ms (+29%) | +0.07 ms (+23%) | +0.06 ms (+20%) | 14 / 14 / 14 |
| 3,600 | white noise | 0.40 ms | 110 | +0.11 ms (+29%) | +0.18 ms (+46%) | +0.13 ms (+33%) | +0.07 ms (+17%) | 90 / 3,445 / 26 |
| 3,600 | vibration | 0.38 ms | 106 | +0.12 ms (+32%) | +0.15 ms (+38%) | +0.11 ms (+30%) | +0.06 ms (+17%) | 96 / 2,640 / 36 |
| 86,400 | smooth + noise | 13.90 ms | 161 | +2.67 ms (+19%) | +3.32 ms (+24%) | +3.79 ms (+27%) | +3.13 ms (+23%) | 21 / 2,849 / 864 |
| 86,400 | random walk + noise | 14.35 ms | 166 | +2.82 ms (+20%) | +5.64 ms (+39%) | +5.18 ms (+36%) | +3.33 ms (+23%) | 32 / 61,152 / 850 |
| 86,400 | steps | 12.99 ms | 150 | +2.73 ms (+21%) | +3.02 ms (+23%) | +3.03 ms (+23%) | +2.99 ms (+23%) | 87 / 300 / 300 |
| 86,400 | white noise | 14.13 ms | 164 | +2.87 ms (+20%) | +5.24 ms (+37%) | +5.78 ms (+41%) | +3.34 ms (+24%) | 74 / 81,395 / 864 |
| 86,400 | vibration | 14.07 ms | 163 | +2.31 ms (+16%) | +5.04 ms (+36%) | +5.22 ms (+37%) | +2.74 ms (+19%) | 258 / 62,547 / 825 |
| 1,000,000 | smooth + noise | 294.53 ms | 295 | +33.80 ms (+11%) | +43.40 ms (+15%) | +43.56 ms (+15%) | +43.49 ms (+15%) | 35 / 5,652 / 5,652 |
| 1,000,000 | random walk + noise | 314.74 ms | 315 | +30.30 ms (+10%) | +39.73 ms (+13%) | +43.03 ms (+14%) | +38.34 ms (+12%) | 26 / 74,494 / 9,917 |
| 1,000,000 | steps | 222.02 ms | 222 | +35.29 ms (+16%) | +40.52 ms (+18%) | +40.60 ms (+18%) | +39.91 ms (+18%) | 197 / 3,360 / 3,360 |
| 1,000,000 | white noise | 317.86 ms | 318 | +25.78 ms (+8%) | +54.30 ms (+17%) | +58.51 ms (+18%) | +29.87 ms (+9%) | 61 / 936,091 / 9,906 |
| 1,000,000 | vibration | 315.78 ms | 316 | +21.81 ms (+7%) | +57.21 ms (+18%) | +59.00 ms (+19%) | +25.46 ms (+8%) | 242 / 713,169 / 9,861 |
