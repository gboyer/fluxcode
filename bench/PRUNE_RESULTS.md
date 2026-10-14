# fluxprune speed (bench/prune.py, 2026-10-14, AC power)

arm64, Darwin 25.6.0, Python 3.11.13, numpy 2.4.6; one thread, best of 5, 1 s scans (datetime64[ns] times).

rank: the O(n log n) ranking pass. prune: the rmse pass (rank, the count, 2 refinement passes). max_error: prune with max_error = 1% of the range (the max_error pass, farthest reach). + cap: the same with max_points = n / 100, which binds on the noisy signals (then max_error is relaxed to the tightest bound that fits: a bisection over the max_error pass). refit: prune with refit=True (the rmse pass, then least-squares values at the kept times). Times are added to rank's. Each call is timed on its own, so a difference below ~0.05 ms at 3,600 points is run-to-run noise.

(generated in 61 s)

| n | signal | rank | ns/point | prune | max_error | max_error + cap | refit | points kept (prune / max_error / + cap) |
|---|---|---|---|---|---|---|---|---|
| 3,600 | smooth + noise | 0.34 ms | 94 | +0.09 ms (+26%) | +0.18 ms (+55%) | +0.16 ms (+48%) | +0.10 ms (+28%) | 33 / 3,031 / 35 |
| 3,600 | random walk + noise | 0.41 ms | 115 | +0.11 ms (+27%) | +0.13 ms (+32%) | +0.10 ms (+25%) | +0.03 ms (+6%) | 26 / 3,217 / 36 |
| 3,600 | steps | 0.38 ms | 106 | +0.05 ms (+14%) | +0.07 ms (+19%) | +0.07 ms (+19%) | +0.08 ms (+22%) | 14 / 14 / 14 |
| 3,600 | white noise | 0.32 ms | 90 | +0.10 ms (+31%) | +0.19 ms (+60%) | +0.21 ms (+63%) | +0.11 ms (+33%) | 9 / 3,441 / 26 |
| 3,600 | vibration | 0.39 ms | 108 | +0.12 ms (+31%) | +0.15 ms (+39%) | +0.11 ms (+28%) | +0.05 ms (+12%) | 96 / 2,640 / 36 |
| 86,400 | smooth + noise | 13.90 ms | 161 | +2.61 ms (+19%) | +3.31 ms (+24%) | +3.63 ms (+26%) | +3.17 ms (+23%) | 21 / 2,849 / 864 |
| 86,400 | random walk + noise | 14.63 ms | 169 | +2.90 ms (+20%) | +5.63 ms (+39%) | +5.24 ms (+36%) | +3.34 ms (+23%) | 30 / 61,151 / 850 |
| 86,400 | steps | 13.31 ms | 154 | +2.95 ms (+22%) | +3.00 ms (+23%) | +3.12 ms (+23%) | +3.14 ms (+24%) | 87 / 300 / 300 |
| 86,400 | white noise | 14.65 ms | 170 | +1.97 ms (+13%) | +4.40 ms (+30%) | +5.52 ms (+38%) | +2.55 ms (+17%) | 8 / 81,392 / 864 |
| 86,400 | vibration | 14.33 ms | 166 | +2.38 ms (+17%) | +4.93 ms (+34%) | +5.40 ms (+38%) | +3.14 ms (+22%) | 258 / 62,547 / 825 |
| 1,000,000 | smooth + noise | 311.71 ms | 312 | +33.08 ms (+11%) | +42.47 ms (+14%) | +51.89 ms (+17%) | +43.07 ms (+14%) | 35 / 5,652 / 5,652 |
| 1,000,000 | random walk + noise | 330.00 ms | 330 | +32.88 ms (+10%) | +42.41 ms (+13%) | +48.21 ms (+15%) | +40.61 ms (+12%) | 26 / 74,494 / 9,917 |
| 1,000,000 | steps | 232.89 ms | 233 | +34.51 ms (+15%) | +43.64 ms (+19%) | +42.41 ms (+18%) | +50.32 ms (+22%) | 197 / 3,360 / 3,360 |
| 1,000,000 | white noise | 388.08 ms | 388 | +3.30 ms (+1%) | +14.64 ms (+4%) | +11.80 ms (+3%) | -22.36 ms (-6%) | 8 / 936,086 / 9,906 |
| 1,000,000 | vibration | 336.21 ms | 336 | +24.21 ms (+7%) | +57.54 ms (+17%) | +68.03 ms (+20%) | +32.15 ms (+10%) | 242 / 713,169 / 9,861 |
