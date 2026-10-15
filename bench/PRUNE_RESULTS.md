# fluxprune speed (bench/prune.py, 2026-10-14, AC power)

arm64, Darwin 25.6.0, Python 3.11.13, numpy 2.4.6; one thread, best of 5, 1 s scans (datetime64[ns] times).

rank: the O(n log n) ranking pass. prune: the rmse pass (rank, the count, 2 refinement passes). max_error: prune with max_error = 1% of the range (the max_error pass, farthest reach). + cap: the same with max_points = n / 100, which binds on the noisy signals (then max_error is relaxed to the tightest bound that fits: a bisection over the max_error pass). refit: prune with refit=True (the rmse pass, then least-squares values at the kept times). Times are added to rank's. Each call is timed on its own, so a difference below ~0.05 ms at 3,600 points is run-to-run noise.

(generated in 64 s)

| n | signal | rank | ns/point | prune | max_error | max_error + cap | refit | points kept (prune / max_error / + cap) |
|---|---|---|---|---|---|---|---|---|
| 3,600 | smooth + noise | 0.36 ms | 100 | +0.13 ms (+35%) | +0.22 ms (+60%) | +0.18 ms (+50%) | +0.12 ms (+33%) | 207 / 3,056 / 35 |
| 3,600 | random walk + noise | 0.43 ms | 120 | +0.12 ms (+28%) | +0.15 ms (+35%) | +0.12 ms (+27%) | +0.08 ms (+18%) | 145 / 3,222 / 36 |
| 3,600 | steps | 0.39 ms | 108 | +0.12 ms (+29%) | +0.14 ms (+35%) | +0.13 ms (+34%) | +0.14 ms (+35%) | 17 / 17 / 17 |
| 3,600 | white noise | 0.35 ms | 97 | +0.11 ms (+31%) | +0.23 ms (+67%) | +0.21 ms (+60%) | +0.11 ms (+32%) | 9 / 3,441 / 26 |
| 3,600 | vibration | 0.40 ms | 111 | +0.13 ms (+33%) | +0.17 ms (+42%) | +0.16 ms (+40%) | +0.09 ms (+24%) | 307 / 2,678 / 36 |
| 86,400 | smooth + noise | 14.52 ms | 168 | +3.34 ms (+23%) | +4.56 ms (+31%) | +4.74 ms (+33%) | +4.91 ms (+34%) | 5,182 / 5,817 / 864 |
| 86,400 | random walk + noise | 16.00 ms | 185 | +3.15 ms (+20%) | +5.92 ms (+37%) | +5.33 ms (+33%) | +3.69 ms (+23%) | 2,369 / 62,058 / 850 |
| 86,400 | steps | 14.41 ms | 167 | +3.02 ms (+21%) | +3.43 ms (+24%) | +3.41 ms (+24%) | +3.48 ms (+24%) | 317 / 317 / 317 |
| 86,400 | white noise | 16.09 ms | 186 | +0.90 ms (+6%) | +4.47 ms (+28%) | +5.60 ms (+35%) | +2.36 ms (+15%) | 8 / 81,392 / 864 |
| 86,400 | vibration | 15.52 ms | 180 | +3.28 ms (+21%) | +6.03 ms (+39%) | +5.36 ms (+35%) | +3.86 ms (+25%) | 8,978 / 63,908 / 825 |
| 1,000,000 | smooth + noise | 320.51 ms | 321 | +38.12 ms (+12%) | +42.67 ms (+13%) | +483.90 ms (+151%) | +44.57 ms (+14%) | 54,126 / 54,131 / 10,000 |
| 1,000,000 | random walk + noise | 332.35 ms | 332 | +35.96 ms (+11%) | +44.55 ms (+13%) | +44.59 ms (+13%) | +41.44 ms (+12%) | 30,080 / 87,843 / 9,917 |
| 1,000,000 | steps | 236.05 ms | 236 | +35.79 ms (+15%) | +39.78 ms (+17%) | +40.21 ms (+17%) | +40.66 ms (+17%) | 3,568 / 3,572 / 3,572 |
| 1,000,000 | white noise | 334.39 ms | 334 | +22.81 ms (+7%) | +52.44 ms (+16%) | +62.41 ms (+19%) | +26.68 ms (+8%) | 8 / 936,086 / 9,906 |
| 1,000,000 | vibration | 333.41 ms | 333 | +38.29 ms (+11%) | +69.89 ms (+21%) | +62.02 ms (+19%) | +47.19 ms (+14%) | 104,076 / 730,521 / 9,861 |
