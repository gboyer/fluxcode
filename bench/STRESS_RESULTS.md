# fluxcode day-scale stress test (bench/stress.py, 2026-10-01, 4 threads, AC power)

One run of each, at 5117ff8. Load average about 1.6 before the runs. Since the previous results
(2026-09-25): variable block sizes, snapped power-of-two grids, and `planes="best"` as the
default, which compresses every unit twice (bit and byte planes) and keeps the smaller. That
second zstd pass is most of the encode change: the day compresses 2.2% smaller (40.78 → 39.89
GB) and encodes in 105 s instead of 71 s (+33 s, +47%); decode is unchanged (39.3 s). `bench/bench_gb.py`
separates the two: with bit planes only, encode is within 4% of the previous format.

## Full target: `uv run python bench/stress.py --breakdown`

Apple M3 Mac15,13, Darwin 25.6.0, Python 3.11.13, numpy 2.4.6

data `sensor-mix`: 1000 tags x 1440 units x 60 x 1000 = 86.40e9 samples (691.2 GB float64); 4 threads; pool 32 units/kind
NaN: 3 min/tag/day in 2 runs -> 5,011 units touched (0.35%), 0.208% of samples
tags per kind: analog 350, held 250, digital 100, vibration 100, counter 50, sensor-0.1 50, noisy-sine 50, random-walk 50
(pool generated, roundtrip-checked (worst error 1.41e-02 of block range) and warmed in 2 s)

### Single thread, one unit, µs (best of 5 x 200)

| kind | bits/sample | kernels | write_unit | zstd | encode_unit | unzstd | decode kernel | decode_unit |
|---|---|---|---|---|---|---|---|---|
| analog | 5.90 | 100 | 32 | 60 | 256 | 15 | 62 | 91 |
| held | 0.27 | 97 | 31 | 24 | 182 | 25 | 70 | 105 |
| digital | 0.01 | 46 | 31 | 9 | 106 | 30 | 50 | 93 |
| vibration | 6.15 | 101 | 31 | 53 | 251 | 39 | 83 | 128 |
| counter | 1.58 | 119 | 31 | 24 | 277 | 44 | 83 | 133 |
| sensor-0.1 | 1.77 | 117 | 31 | 61 | 342 | 22 | 84 | 112 |
| noisy-sine | 5.46 | 100 | 32 | 57 | 251 | 52 | 82 | 138 |
| random-walk | 12.67 | 101 | 32 | 88 | 317 | 10 | 83 | 100 |

### Result

| phase | wall | float64 GB/s | samples/s | µs/block/worker | in-call % |
|---|---|---|---|---|---|
| encode | 104.7 s | 6.60 GB/s | 826 M | 4.85 | 100% |
| decode | 39.3 s | 17.59 GB/s | 2199 M | 1.82 | 99% |

compressed 39.89 GB (3.69 bits/sample, ratio 17.3); real time is 86,400 s per day, so encode runs at 826x real time for 1000 tags.

encode GB/s per tenth of the phase: 7.1 7.1 6.5 6.8 6.8 6.4 6.7 5.9 6.4 6.3
decode GB/s per tenth of the phase: 17.6 18.1 17.3 17.6 18.0 17.4 17.6 17.6 17.2 17.4

## Adversarial variants (`--scale 0.25`: 250 tags x 1 day; per-block cost doesn't depend on scale)

| variant | encode µs/block/worker | encode GB/s | decode GB/s | bits/sample | full day (1000 tags) enc / dec |
|---|---|---|---|---|---|
| sensor-mix, 3 NaN min/tag/day (the full run above) | 4.85 | 6.60 | 17.59 | 3.69 | 105 s / 39 s |
| `--data random-walk` (worst entropy) | 7.17 | 4.46 | 14.91 | 12.57 | 155 s / 46 s |
| `--nan-minutes 120 --nan-runs 20` (8% NaN, in runs) | 5.35 | 5.98 | 16.57 | 3.38 | 116 s / 42 s |
| `--sprinkle 0.0005` (~40% of blocks flagged) | 6.25 | 5.12 | 15.02 | 3.69 | 135 s / 46 s |
| `--sprinkle 0.01` (every block flagged) | 8.59 | 3.73 | 12.92 | 3.85 | 186 s / 54 s |
| `--data held --sprinkle 0.01` | 6.48 | 4.93 | 15.56 | 0.29 | 140 s / 44 s |

## Decimal detection near miss: `uv run python bench/decimal_near_miss.py`

A block on a decimal grid except for its last sample makes each candidate exponent scan the whole
block before failing. Single thread, µs/block; the method is in the script's docstring. With the
second zstd pass in every block's cost, the near miss is now within run-to-run noise (it cost
+0.6–0.8 µs/block before, against 3.5).

| input | µs/block |
|---|---|
| random walk | 5.34 |
| random walk, `decimal_detection=False` | 5.29 |
| on a 0.01 grid | 5.94 |
| integers | 5.86 |
| integers, off-grid last sample | 5.79 |
| 0.01 grid, off-grid last sample | 5.95 |
| the same, `decimal_detection=False` | 5.20 |

## Timestamps: `uv run python bench/stress.py --times clock-mix`

Apple M3 Mac15,13, Darwin 25.6.0, Python 3.11.13, numpy 2.4.6

data `sensor-mix`: 1000 tags x 1440 units x 60 x 1000 = 86.40e9 samples (691.2 GB float64); 4 threads; pool 32 units/kind
NaN: 3 min/tag/day in 2 runs -> 5,011 units touched (0.35%), 0.208% of samples
tags per kind: analog 350, held 250, digital 100, vibration 100, counter 50, sensor-0.1 50, noisy-sine 50, random-walk 50
timestamps (datetime64[ns], exact), tags per clock: grid 600, grid+gaps 300, noisy 100
(pool generated, roundtrip-checked (worst error 1.41e-02 of block range) and warmed in 3 s)

### Result

| phase | wall | float64 GB/s | samples/s | µs/block/worker | in-call % |
|---|---|---|---|---|---|
| encode | 148.6 s | 4.65 GB/s | 582 M | 6.88 | 100% |
| decode | 55.6 s | 12.44 GB/s | 1555 M | 2.57 | 100% |

compressed 47.68 GB (4.42 bits/sample, ratio 14.5); real time is 86,400 s per day, so encode runs at 582x real time for 1000 tags.

encode GB/s per tenth of the phase: 4.8 4.7 4.6 4.6 4.7 4.6 4.7 4.7 4.5 4.7
decode GB/s per tenth of the phase: 12.6 12.3 12.4 11.9 12.8 12.4 12.8 12.7 11.9 12.4

### Per clock kind: `--scale 0.25 --times <clock>` (same session)

| clock | phase | wall | float64 GB/s | samples/s | µs/block/worker | in-call % |
|---|---|---|---|---|---|---|
| none | encode | 27.3 s | 6.34 GB/s | 792 M | 5.05 | 99% |
| none | decode | 9.9 s | 17.39 GB/s | 2174 M | 1.84 | 99% |
| grid | encode | 32.4 s | 5.34 GB/s | 667 M | 6.00 | 99% |
| grid | decode | 13.2 s | 13.08 GB/s | 1635 M | 2.45 | 99% |
| grid+gaps | encode | 35.5 s | 4.87 GB/s | 609 M | 6.57 | 99% |
| grid+gaps | decode | 13.7 s | 12.57 GB/s | 1572 M | 2.55 | 99% |
| noisy | encode | 69.9 s | 2.47 GB/s | 309 M | 12.95 | 99% |
| noisy | decode | 26.4 s | 6.54 GB/s | 818 M | 4.89 | 100% |

- none compressed 9.91 GB (3.67 bits/sample, ratio 17.4);
- grid compressed 9.92 GB (3.68 bits/sample, ratio 17.4);
- grid+gaps compressed 9.95 GB (3.68 bits/sample, ratio 17.4);
- noisy compressed 29.16 GB (10.80 bits/sample, ratio 5.9);
