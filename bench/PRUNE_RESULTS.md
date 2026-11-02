# fluxprune speed (bench/prune.py, 2026-11-01, AC power)

arm64, Darwin 25.6.0, Python 3.11.13, numpy 2.4.6; one thread, best of 5, 1 s scans (datetime64[ns] times).

rank: the O(n log n) ranking pass. prune: the rmse pass (rank, the count, 2 refinement passes). max_error: prune with max_error = 1% of the range (the max_error pass, farthest reach). + cap: the same with max_points = n / 100, which binds on the noisy signals (then max_error is relaxed to the tightest bound that fits: a bisection over the max_error pass). refit: prune with refit=True (the rmse pass, then least-squares values at the kept times). Times are added to rank's. Each call is timed on its own, so a difference below ~0.05 ms at 3,600 points is run-to-run noise.

(generated in 65 s)

| n | signal | rank | ns/point | prune | max_error | max_error + cap | refit | points kept (prune / max_error / + cap) |
|---|---|---|---|---|---|---|---|---|
| 3,600 | smooth + noise | 0.40 ms | 110 | +0.14 ms (+35%) | +0.25 ms (+63%) | +0.16 ms (+41%) | +0.11 ms (+29%) | 207 / 3,056 / 35 |
| 3,600 | random walk + noise | 0.42 ms | 115 | +0.12 ms (+28%) | +0.22 ms (+52%) | +0.20 ms (+49%) | +0.13 ms (+32%) | 145 / 3,222 / 36 |
| 3,600 | steps | 0.32 ms | 88 | +0.09 ms (+30%) | +0.11 ms (+34%) | +0.12 ms (+39%) | +0.10 ms (+33%) | 17 / 17 / 17 |
| 3,600 | white noise | 0.43 ms | 118 | +0.10 ms (+25%) | +0.14 ms (+32%) | +0.13 ms (+30%) | +0.06 ms (+13%) | 9 / 3,441 / 26 |
| 3,600 | vibration | 0.40 ms | 110 | +0.13 ms (+33%) | +0.17 ms (+43%) | +0.12 ms (+31%) | +0.06 ms (+16%) | 307 / 2,678 / 36 |
| 86,400 | smooth + noise | 14.04 ms | 162 | +3.29 ms (+23%) | +3.77 ms (+27%) | +3.72 ms (+26%) | +3.82 ms (+27%) | 5,182 / 5,817 / 864 |
| 86,400 | random walk + noise | 14.53 ms | 168 | +3.01 ms (+21%) | +5.69 ms (+39%) | +5.15 ms (+35%) | +3.52 ms (+24%) | 2,369 / 62,058 / 850 |
| 86,400 | steps | 13.05 ms | 151 | +2.97 ms (+23%) | +3.43 ms (+26%) | +3.32 ms (+25%) | +3.39 ms (+26%) | 317 / 317 / 317 |
| 86,400 | white noise | 14.56 ms | 168 | +1.97 ms (+14%) | +4.28 ms (+29%) | +5.43 ms (+37%) | +2.49 ms (+17%) | 8 / 81,392 / 864 |
| 86,400 | vibration | 14.20 ms | 164 | +3.23 ms (+23%) | +5.83 ms (+41%) | +5.16 ms (+36%) | +3.86 ms (+27%) | 8,978 / 63,908 / 825 |
| 1,000,000 | smooth + noise | 311.36 ms | 311 | +38.26 ms (+12%) | +46.25 ms (+15%) | +486.40 ms (+156%) | +48.43 ms (+16%) | 54,126 / 54,131 / 10,000 |
| 1,000,000 | random walk + noise | 328.62 ms | 329 | +36.58 ms (+11%) | +47.16 ms (+14%) | +48.65 ms (+15%) | +47.26 ms (+14%) | 30,080 / 87,843 / 9,917 |
| 1,000,000 | steps | 233.59 ms | 234 | +35.80 ms (+15%) | +40.07 ms (+17%) | +42.76 ms (+18%) | +43.07 ms (+18%) | 3,568 / 3,572 / 3,572 |
| 1,000,000 | white noise | 339.17 ms | 339 | +23.05 ms (+7%) | +52.99 ms (+16%) | +61.71 ms (+18%) | +26.16 ms (+8%) | 8 / 936,086 / 9,906 |
| 1,000,000 | vibration | 337.91 ms | 338 | +37.89 ms (+11%) | +69.58 ms (+21%) | +60.58 ms (+18%) | +46.35 ms (+14%) | 104,076 / 730,521 / 9,861 |
