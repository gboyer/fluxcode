# fluxcode day-scale stress test (bench/stress.py, 2026-09-25, 4 threads, AC power, idle, quiet rerun)

Sizes are current (format version 1, with the unit header and per-block anchors). Every timing is
the fastest of 5 complete runs in one session; the 5 agree within 1.5% for every phase, except
decode with `--sprinkle 0.0005` (4.6%). Load snapshots before each run showed `top`,
WindowServer and the screen saver at 5-17% CPU, and occasional Spotlight bursts.

## Full target: `uv run python bench/stress.py --breakdown`

Apple M3 Mac15,13, Darwin 25.6.0, Python 3.11.13, numpy 2.4.6

data `sensor-mix`: 1000 tags x 1440 units x 60 x 1000 = 86.40e9 samples (691.2 GB float64); 4 threads; pool 32 units/kind
NaN: 3 min/tag/day in 2 runs -> 5,011 units touched (0.35%), 0.208% of samples
tags per kind: analog 350, held 250, digital 100, vibration 100, counter 50, sensor-0.1 50, noisy-sine 50, random-walk 50
(pool generated, roundtrip-checked (worst error 1.41e-02 of block range) and warmed in 1 s)

## Single thread, one unit, µs (best of 5 x 200)

| kind | bits/sample | kernels | write_unit | zstd | encode_unit | unzstd | decode kernel | decode_unit |
|---|---|---|---|---|---|---|---|---|
| analog | 5.91 | 89 | 28 | 55 | 174 | 20 | 56 | 81 |
| held | 0.27 | 86 | 28 | 22 | 138 | 24 | 64 | 94 |
| digital | 0.01 | 39 | 28 | 8 | 77 | 19 | 47 | 70 |
| vibration | 6.17 | 91 | 28 | 50 | 170 | 36 | 76 | 116 |
| counter | 1.58 | 111 | 28 | 22 | 159 | 44 | 78 | 129 |
| sensor-0.1 | 1.77 | 108 | 28 | 57 | 190 | 19 | 78 | 100 |
| noisy-sine | 5.53 | 93 | 28 | 53 | 176 | 51 | 76 | 130 |
| random-walk | 12.70 | 95 | 29 | 87 | 213 | 9 | 77 | 88 |

## Result

| phase | wall | float64 GB/s | samples/s | µs/block/worker | in-call % |
|---|---|---|---|---|---|
| encode | 71.4 s | 9.67 GB/s | 1209 M | 3.31 | 100% |
| decode | 39.3 s | 17.59 GB/s | 2199 M | 1.82 | 99% |

compressed 40.78 GB (3.78 bits/sample, ratio 17.0); real time is 86,400 s per day, so encode runs at 1209x real time for 1000 tags.

encode GB/s per tenth of the phase: 10.1 10.3 9.6 9.8 9.8 9.4 9.2 9.5 9.4 9.8
decode GB/s per tenth of the phase: 17.3 18.2 18.1 17.1 17.9 17.4 17.2 17.3 17.5 17.8

## Adversarial variants (`--scale 0.25`: 250 tags x 1 day; per-block cost doesn't depend on scale)

| variant | encode µs/block/worker | encode GB/s | decode GB/s | bits/sample | full day (1000 tags) enc / dec |
|---|---|---|---|---|---|
| sensor-mix, 3 NaN min/tag/day (the full run above) | 3.31 | 9.67 | 17.59 | 3.78 | 71 s / 39 s |
| `--data random-walk` (worst entropy) | 4.14 | 7.73 | 17.52 | 12.61 | 89 s / 39 s |
| `--nan-minutes 120 --nan-runs 20` (8% NaN, in runs) | 3.36 | 9.51 | 17.83 | 3.46 | 73 s / 39 s |
| `--sprinkle 0.0005` (~40% of blocks flagged) | 3.90 | 8.21 | 16.60 | 3.77 | 84 s / 42 s |
| `--sprinkle 0.01` (every block flagged) | 4.98 | 6.42 | 14.23 | 3.92 | 108 s / 49 s |
| `--data held --sprinkle 0.01` | 4.41 | 7.26 | 14.70 | 0.44 | 95 s / 47 s |

## Decimal detection near miss: `uv run python bench/decimal_near_miss.py`

A block on a decimal grid except for its last sample makes each candidate exponent scan the whole
block before failing. Single thread, µs/block; the method is in the script's docstring.

| input | µs/block |
|---|---|
| random walk | 3.53 |
| random walk, `decimal_detection=False` | 3.48 |
| on a 0.01 grid | 3.52 |
| integers | 3.51 |
| integers, off-grid last sample | 4.11 |
| 0.01 grid, off-grid last sample | 4.28 |
| the same, `decimal_detection=False` | 3.48 |

An earlier figure of 2.96 µs/block with detection off had no recorded script and isn't reproduced
here.

## Timestamps: `uv run python bench/stress.py --times clock-mix` (2026-09-27, AC power, one run; per-block time references, 32 residual planes)

Apple M3 Mac15,13, Darwin 25.6.0, Python 3.11.13, numpy 2.4.6

data `sensor-mix`: 1000 tags x 1440 units x 60 x 1000 = 86.40e9 samples (691.2 GB float64); 4 threads; pool 32 units/kind
NaN: 3 min/tag/day in 2 runs -> 5,011 units touched (0.35%), 0.208% of samples
tags per kind: analog 350, held 250, digital 100, vibration 100, counter 50, sensor-0.1 50, noisy-sine 50, random-walk 50
timestamps (datetime64[ns], exact), tags per clock: grid 600, grid+gaps 300, noisy 100
(pool generated, roundtrip-checked (worst error 1.41e-02 of block range) and warmed in 2 s)

## Result

| phase | wall | float64 GB/s | samples/s | µs/block/worker | in-call % |
|---|---|---|---|---|---|
| encode | 92.5 s | 7.47 GB/s | 934 M | 4.28 | 99% |
| decode | 53.9 s | 12.82 GB/s | 1602 M | 2.50 | 100% |

compressed 48.51 GB (4.49 bits/sample, ratio 14.2); real time is 86,400 s per day, so encode runs at 934x real time for 1000 tags.

encode GB/s per tenth of the phase: 8.0 7.8 7.4 7.5 7.5 7.4 7.4 7.3 7.1 7.2
decode GB/s per tenth of the phase: 12.6 12.7 12.8 12.6 13.2 13.0 13.3 12.9 12.3 12.8

### Per clock kind: `--scale 0.25 --times <clock>` (same session)

| clock | phase | wall | float64 GB/s | samples/s | µs/block/worker | in-call % |
|---|---|---|---|---|---|---|
| none | encode | 19.5 s | 8.88 GB/s | 1110 M | 3.60 | 99% |
| none | decode | 10.2 s | 16.99 GB/s | 2124 M | 1.88 | 99% |
| grid | encode | 22.3 s | 7.75 GB/s | 968 M | 4.13 | 99% |
| grid | decode | 12.2 s | 14.14 GB/s | 1768 M | 2.26 | 99% |
| grid+gaps | encode | 23.0 s | 7.50 GB/s | 937 M | 4.27 | 99% |
| grid+gaps | decode | 12.9 s | 13.40 GB/s | 1675 M | 2.39 | 99% |
| noisy | encode | 42.5 s | 4.07 GB/s | 508 M | 7.87 | 99% |
| noisy | decode | 25.4 s | 6.81 GB/s | 851 M | 4.70 | 99% |

- none compressed 10.13 GB (3.75 bits/sample, ratio 17.1);
- grid compressed 10.14 GB (3.76 bits/sample, ratio 17.0);
- grid+gaps compressed 10.16 GB (3.76 bits/sample, ratio 17.0);
- noisy compressed 29.31 GB (10.85 bits/sample, ratio 5.9);
