# fluxcode day-scale stress test (bench/stress.py, 2026-10-01, 4 threads, AC power)

One run of each, at 8bfc6c6 (later commits change only docs and tests), after the int16
`grid_params` column. Load average 1.6–1.7 before the runs (2.3–2.7 for the last few, with a
system agent busy on one core). Against the previous runs the same day (5117ff8): the day is the
same size (39.89 → 39.88 GB) and as fast (encode 104.7 → 104.5 s, decode 39.3 → 38.1 s). With
timestamps encode is 9% faster (148.6 → 135.6 s); `bench/time_axis.py` shows the same
13–16 µs per unit against main on one thread, with fresh numba caches on both, though the cause
isn't pinned down. The adversarial variants are 10–15% faster than before (random walk 155 →
138 s, every block flagged 186 → 158 s): this time each run started after a 45 s pause, so the
fanless machine started cooler; the earlier runs went back to back.

## Full target: `uv run python bench/stress.py --breakdown`

Apple M3 Mac15,13, Darwin 25.6.0, Python 3.11.13, numpy 2.4.6

data `sensor-mix`: 1000 tags x 1440 units x 60 x 1000 = 86.40e9 samples (691.2 GB float64); 4 threads; pool 32 units/kind
NaN: 3 min/tag/day in 2 runs -> 5,011 units touched (0.35%), 0.208% of samples
tags per kind: analog 350, held 250, digital 100, vibration 100, counter 50, sensor-0.1 50, noisy-sine 50, random-walk 50
(pool generated, roundtrip-checked (worst error 1.41e-02 of block range) and warmed in 2 s)

## Single thread, one unit, µs (best of 5 x 200)

| kind | bits/sample | kernels | write_unit | zstd | encode_unit | unzstd | decode kernel | decode_unit |
|---|---|---|---|---|---|---|---|---|
| analog | 5.90 | 95 | 31 | 59 | 252 | 14 | 57 | 78 |
| held | 0.26 | 96 | 31 | 22 | 175 | 19 | 65 | 90 |
| digital | 0.01 | 44 | 31 | 8 | 102 | 21 | 48 | 74 |
| vibration | 6.15 | 98 | 32 | 53 | 247 | 36 | 79 | 121 |
| counter | 1.58 | 116 | 31 | 23 | 272 | 27 | 79 | 116 |
| sensor-0.1 | 1.77 | 115 | 31 | 60 | 335 | 21 | 80 | 108 |
| noisy-sine | 5.46 | 98 | 31 | 53 | 249 | 41 | 79 | 129 |
| random-walk | 12.67 | 99 | 32 | 87 | 314 | 8 | 82 | 98 |

## Result

| phase | wall | float64 GB/s | samples/s | µs/block/worker | in-call % |
|---|---|---|---|---|---|
| encode | 104.5 s | 6.62 GB/s | 827 M | 4.84 | 100% |
| decode | 38.1 s | 18.16 GB/s | 2270 M | 1.76 | 99% |

compressed 39.88 GB (3.69 bits/sample, ratio 17.3); real time is 86,400 s per day, so encode runs at 827x real time for 1000 tags.

encode GB/s per tenth of the phase: 7.0 7.1 6.3 6.5 6.7 6.6 6.5 6.3 6.7 6.6
decode GB/s per tenth of the phase: 18.0 18.6 17.8 17.9 18.6 18.0 18.2 18.0 18.3 18.3

## Adversarial variants (`--scale 0.25`: 250 tags x 1 day; per-block cost doesn't depend on scale)

| variant | encode µs/block/worker | encode GB/s | decode GB/s | bits/sample | full day (1000 tags) enc / dec |
|---|---|---|---|---|---|
| sensor-mix, 3 NaN min/tag/day (the full run above) | 4.84 | 6.62 | 18.16 | 3.69 | 104 s / 38 s |
| `--data random-walk` (worst entropy) | 6.40 | 5.00 | 16.16 | 12.57 | 138 s / 43 s |
| `--nan-minutes 120 --nan-runs 20` (8% NaN, in runs) | 4.73 | 6.77 | 18.18 | 3.38 | 102 s / 38 s |
| `--sprinkle 0.0005` (~40% of blocks flagged) | 5.35 | 5.98 | 16.59 | 3.69 | 116 s / 42 s |
| `--sprinkle 0.01` (every block flagged) | 7.31 | 4.38 | 13.64 | 3.84 | 158 s / 51 s |
| `--data held --sprinkle 0.01` | 5.71 | 5.60 | 16.61 | 0.29 | 123 s / 42 s |

## Decimal detection near miss: `uv run python bench/decimal_near_miss.py`

A block on a decimal grid except for its last sample makes each candidate exponent scan the whole
block before failing. Single thread, µs/block; the method is in the script's docstring. With the
second zstd pass in every block's cost, the near miss is within run-to-run noise.

| input | µs/block |
|---|---|
| random walk | 5.23 |
| random walk, `decimal_detection=False` | 5.18 |
| on a 0.01 grid | 5.80 |
| integers | 5.80 |
| integers, off-grid last sample | 5.79 |
| 0.01 grid, off-grid last sample | 5.98 |
| the same, `decimal_detection=False` | 5.14 |

## Timestamps: `uv run python bench/stress.py --times clock-mix`

Apple M3 Mac15,13, Darwin 25.6.0, Python 3.11.13, numpy 2.4.6

data `sensor-mix`: 1000 tags x 1440 units x 60 x 1000 = 86.40e9 samples (691.2 GB float64); 4 threads; pool 32 units/kind
NaN: 3 min/tag/day in 2 runs -> 5,011 units touched (0.35%), 0.208% of samples
tags per kind: analog 350, held 250, digital 100, vibration 100, counter 50, sensor-0.1 50, noisy-sine 50, random-walk 50
timestamps (datetime64[ns], exact), tags per clock: grid 600, grid+gaps 300, noisy 100
(pool generated, roundtrip-checked (worst error 1.41e-02 of block range) and warmed in 6 s)

## Result

| phase | wall | float64 GB/s | samples/s | µs/block/worker | in-call % |
|---|---|---|---|---|---|
| encode | 135.6 s | 5.10 GB/s | 637 M | 6.28 | 100% |
| decode | 54.0 s | 12.79 GB/s | 1599 M | 2.50 | 100% |

compressed 47.68 GB (4.41 bits/sample, ratio 14.5); real time is 86,400 s per day, so encode runs at 637x real time for 1000 tags.

encode GB/s per tenth of the phase: 5.5 5.4 5.0 5.1 5.3 4.8 5.0 5.0 4.9 4.9
decode GB/s per tenth of the phase: 12.7 12.6 12.9 12.5 12.9 12.9 12.9 13.2 12.5 12.8

### Per clock kind: `--scale 0.25 --times <clock>` (same session)

| clock | phase | wall | float64 GB/s | samples/s | µs/block/worker | in-call % |
|---|---|---|---|---|---|---|
| none | encode | 25.9 s | 6.68 GB/s | 835 M | 4.79 | 99% |
| none | decode | 9.4 s | 18.33 GB/s | 2291 M | 1.75 | 99% |
| grid | encode | 29.0 s | 5.96 GB/s | 745 M | 5.37 | 99% |
| grid | decode | 12.1 s | 14.22 GB/s | 1778 M | 2.25 | 99% |
| grid+gaps | encode | 30.4 s | 5.68 GB/s | 710 M | 5.64 | 99% |
| grid+gaps | decode | 12.8 s | 13.50 GB/s | 1687 M | 2.37 | 99% |
| noisy | encode | 60.7 s | 2.85 GB/s | 356 M | 11.23 | 99% |
| noisy | decode | 24.9 s | 6.95 GB/s | 868 M | 4.61 | 100% |

- none 9.91 GB (3.67 bits/sample, ratio 17.4);
- grid 9.92 GB (3.68 bits/sample, ratio 17.4);
- grid+gaps 9.95 GB (3.68 bits/sample, ratio 17.4);
- noisy 29.16 GB (10.80 bits/sample, ratio 5.9);
