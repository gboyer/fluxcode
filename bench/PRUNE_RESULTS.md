# fluxprune speed (bench/prune.py, 2026-10-14, AC power)

arm64, Darwin 25.6.0, Python 3.11.13, numpy 2.4.6; one thread, best of 5, 1 s scans (datetime64[ns] times).

rank: the O(n log n) ranking pass. prune: the rmse pass (rank, the count, 2 refinement passes). max_error: prune with max_error = 1% of the range (the max_error pass, farthest reach). + cap: the same with max_points = n / 100, which binds on the noisy signals (then max_error is relaxed to the tightest bound that fits: a bisection over the max_error pass). refit: prune with refit=True (the rmse pass, then least-squares values at the kept times). Times are added to rank's. Each call is timed on its own, so a difference below ~0.05 ms at 3,600 points is run-to-run noise.

(generated in 64 s)

| n | signal | rank | ns/point | prune | max_error | max_error + cap | refit | points kept (prune / max_error / + cap) |
|---|---|---|---|---|---|---|---|---|
| 3,600 | smooth + noise | 0.38 ms | 106 | +0.10 ms (+26%) | +0.16 ms (+43%) | +0.12 ms (+32%) | +0.08 ms (+22%) | 207 / 3,056 / 35 |
| 3,600 | random walk + noise | 0.43 ms | 118 | +0.13 ms (+29%) | +0.17 ms (+40%) | +0.13 ms (+31%) | +0.05 ms (+11%) | 145 / 3,222 / 36 |
| 3,600 | steps | 0.39 ms | 108 | +0.11 ms (+29%) | +0.13 ms (+35%) | +0.13 ms (+34%) | +0.13 ms (+35%) | 17 / 17 / 17 |
| 3,600 | white noise | 0.34 ms | 94 | +0.10 ms (+30%) | +0.19 ms (+57%) | +0.21 ms (+62%) | +0.11 ms (+32%) | 9 / 3,441 / 26 |
| 3,600 | vibration | 0.40 ms | 110 | +0.13 ms (+34%) | +0.18 ms (+45%) | +0.13 ms (+32%) | +0.06 ms (+15%) | 307 / 2,678 / 36 |
| 86,400 | smooth + noise | 14.27 ms | 165 | +3.19 ms (+22%) | +3.80 ms (+27%) | +3.87 ms (+27%) | +3.81 ms (+27%) | 5,182 / 5,817 / 864 |
| 86,400 | random walk + noise | 14.82 ms | 171 | +3.36 ms (+23%) | +6.04 ms (+41%) | +4.97 ms (+34%) | +3.36 ms (+23%) | 2,369 / 62,058 / 850 |
| 86,400 | steps | 13.21 ms | 153 | +3.17 ms (+24%) | +3.50 ms (+27%) | +3.59 ms (+27%) | +3.71 ms (+28%) | 317 / 317 / 317 |
| 86,400 | white noise | 14.95 ms | 173 | +1.62 ms (+11%) | +4.87 ms (+33%) | +4.84 ms (+32%) | +2.28 ms (+15%) | 8 / 81,392 / 864 |
| 86,400 | vibration | 14.45 ms | 167 | +3.32 ms (+23%) | +5.60 ms (+39%) | +5.60 ms (+39%) | +3.66 ms (+25%) | 8,978 / 63,908 / 825 |
| 1,000,000 | smooth + noise | 317.45 ms | 317 | +33.31 ms (+10%) | +42.81 ms (+13%) | +486.11 ms (+153%) | +50.58 ms (+16%) | 54,126 / 54,131 / 10,000 |
| 1,000,000 | random walk + noise | 334.63 ms | 335 | +36.99 ms (+11%) | +45.95 ms (+14%) | +43.15 ms (+13%) | +39.31 ms (+12%) | 30,080 / 87,843 / 9,917 |
| 1,000,000 | steps | 234.63 ms | 235 | +35.50 ms (+15%) | +38.93 ms (+17%) | +38.68 ms (+16%) | +39.84 ms (+17%) | 3,568 / 3,572 / 3,572 |
| 1,000,000 | white noise | 338.43 ms | 338 | +21.22 ms (+6%) | +49.71 ms (+15%) | +61.75 ms (+18%) | +25.41 ms (+8%) | 8 / 936,086 / 9,906 |
| 1,000,000 | vibration | 333.84 ms | 334 | +40.62 ms (+12%) | +71.05 ms (+21%) | +63.17 ms (+19%) | +49.35 ms (+15%) | 104,076 / 730,521 / 9,861 |
