# fluxcode day-scale stress test (bench/stress.py, 2026-10-02, 4 threads, AC power)

One run of each at 5ee16fa, on AC power with a 45 s pause before each step. The default (effort 4) is
the encoder as it was at 8bfc6c6, byte for byte: the day compresses to the same 39.88 GB, and the speeds are
within run-to-run spread of that run's (encode 104.5 → 106.6 s, +2%; decode 38.1 → 38.5 s, +1%; with
timestamps 135.6 → 141.5 s, +4%; the adversarial variants +1 to +4%). The machine showed a load average
of 1.2–3.5 before the steps (other apps), higher than the 1.6–1.7 of that run. These are the Python path (no
Rust extension); `Params.effort` and the extension's effect on 4 threads are in
`experimental/rust_port/results/stress_effort.md`: at effort 5 (block flushes) 3.62 GB/s without it and
5.98 with it, against 6.71 and 7.10 at the default.

## Full target: `uv run python bench/stress.py --breakdown`

Apple M3 Mac15,13, Darwin 25.6.0, Python 3.11.13, numpy 2.4.6

data `sensor-mix`: 1000 tags x 1440 units x 60 x 1000 = 86.40e9 samples (691.2 GB float64); 4 threads; pool 32 units/kind
NaN: 3 min/tag/day in 2 runs -> 5,011 units touched (0.35%), 0.208% of samples
tags per kind: analog 350, held 250, digital 100, vibration 100, counter 50, sensor-0.1 50, noisy-sine 50, random-walk 50
(pool generated, roundtrip-checked (worst error 1.41e-02 of block range) and warmed in 2 s)

## Single thread, one unit, µs (best of 5 x 200)

| kind | bits/sample | kernels | write_unit | zstd | encode_unit | unzstd | decode kernel | decode_unit |
|---|---|---|---|---|---|---|---|---|
| analog | 5.90 | 97 | 31 | 60 | 251 | 14 | 60 | 81 |
| held | 0.26 | 96 | 31 | 23 | 179 | 19 | 68 | 94 |
| digital | 0.01 | 45 | 31 | 8 | 106 | 18 | 50 | 76 |
| vibration | 6.15 | 98 | 31 | 53 | 248 | 38 | 82 | 126 |
| counter | 1.58 | 116 | 31 | 23 | 273 | 29 | 83 | 119 |
| sensor-0.1 | 1.77 | 116 | 31 | 61 | 342 | 21 | 85 | 112 |
| noisy-sine | 5.46 | 98 | 31 | 57 | 251 | 48 | 82 | 146 |
| random-walk | 12.67 | 99 | 32 | 87 | 316 | 9 | 83 | 100 |

## Result

| phase | wall | float64 GB/s | samples/s | µs/block/worker | in-call % |
|---|---|---|---|---|---|
| encode | 106.6 s | 6.49 GB/s | 811 M | 4.93 | 100% |
| decode | 38.5 s | 17.95 GB/s | 2244 M | 1.78 | 99% |

compressed 39.88 GB (3.69 bits/sample, ratio 17.3); real time is 86,400 s per day, so encode runs at 811x real time for 1000 tags.

encode GB/s per tenth of the phase: 7.1 6.9 6.1 6.6 6.6 6.3 6.4 6.0 6.4 6.4
decode GB/s per tenth of the phase: 18.0 19.0 17.8 18.0 18.5 17.1 17.9 17.6 17.3 18.4

## Adversarial variants (`--scale 0.25`: 250 tags x 1 day; per-block cost doesn't depend on scale)

| variant | encode µs/block/worker | encode GB/s | decode GB/s | bits/sample | full day (1000 tags) enc / dec |
|---|---|---|---|---|---|
| sensor-mix, 3 NaN min/tag/day (the full run above) | 4.93 | 6.49 | 17.95 | 3.69 | 107 s / 38 s |
| `--data random-walk` (worst entropy) | 6.61 | 4.84 | 15.77 | 12.57 | 143 s / 44 s |
| `--nan-minutes 120 --nan-runs 20` (8% NaN, in runs) | 4.87 | 6.57 | 18.26 | 3.38 | 105 s / 38 s |
| `--sprinkle 0.0005` (~40% of blocks flagged) | 5.50 | 5.82 | 16.34 | 3.69 | 119 s / 42 s |
| `--sprinkle 0.01` (every block flagged) | 7.49 | 4.27 | 13.42 | 3.84 | 162 s / 52 s |
| `--data held --sprinkle 0.01` | 5.84 | 5.48 | 16.45 | 0.29 | 126 s / 42 s |

## Decimal detection near miss: `uv run python bench/decimal_near_miss.py`

A block on a decimal grid except for its last sample makes each candidate exponent scan the whole
block before failing. Single thread, µs/block; the method is in the script's docstring. With the
second zstd pass in every block's cost, the near miss is within run-to-run noise.

| input | µs/block |
|---|---|
| random walk | 5.26 |
| random walk, `decimal_detection=False` | 5.20 |
| on a 0.01 grid | 5.82 |
| integers | 5.84 |
| integers, off-grid last sample | 5.81 |
| 0.01 grid, off-grid last sample | 5.98 |
| the same, `decimal_detection=False` | 5.13 |

## Timestamps: `uv run python bench/stress.py --times clock-mix`

Apple M3 Mac15,13, Darwin 25.6.0, Python 3.11.13, numpy 2.4.6

data `sensor-mix`: 1000 tags x 1440 units x 60 x 1000 = 86.40e9 samples (691.2 GB float64); 4 threads; pool 32 units/kind
NaN: 3 min/tag/day in 2 runs -> 5,011 units touched (0.35%), 0.208% of samples
tags per kind: analog 350, held 250, digital 100, vibration 100, counter 50, sensor-0.1 50, noisy-sine 50, random-walk 50
timestamps (datetime64[ns], exact), tags per clock: grid 600, grid+gaps 300, noisy 100
(pool generated, roundtrip-checked (worst error 1.41e-02 of block range) and warmed in 3 s)

## Result

| phase | wall | float64 GB/s | samples/s | µs/block/worker | in-call % |
|---|---|---|---|---|---|
| encode | 141.5 s | 4.89 GB/s | 611 M | 6.55 | 100% |
| decode | 54.7 s | 12.64 GB/s | 1580 M | 2.53 | 100% |

compressed 47.68 GB (4.41 bits/sample, ratio 14.5); real time is 86,400 s per day, so encode runs at 611x real time for 1000 tags.

encode GB/s per tenth of the phase: 5.4 5.2 4.7 5.1 5.0 4.5 4.9 4.6 4.6 4.7
decode GB/s per tenth of the phase: 12.7 12.6 12.6 12.2 13.0 12.8 12.9 13.0 12.2 12.4

### Per clock kind: `--scale 0.25 --times <clock>` (same session)

| clock | phase | wall | float64 GB/s | samples/s | µs/block/worker | in-call % |
|---|---|---|---|---|---|---|
| none | encode | 26.6 s | 6.50 GB/s | 813 M | 4.92 | 99% |
| none | decode | 9.6 s | 18.05 GB/s | 2256 M | 1.77 | 99% |
| grid | encode | 29.7 s | 5.82 GB/s | 727 M | 5.50 | 99% |
| grid | decode | 12.1 s | 14.25 GB/s | 1781 M | 2.25 | 99% |
| grid+gaps | encode | 31.2 s | 5.54 GB/s | 692 M | 5.78 | 99% |
| grid+gaps | decode | 12.8 s | 13.49 GB/s | 1686 M | 2.37 | 99% |
| noisy | encode | 61.9 s | 2.79 GB/s | 349 M | 11.46 | 99% |
| noisy | decode | 25.6 s | 6.75 GB/s | 844 M | 4.74 | 100% |

- none 9.91 GB (3.67 bits/sample, ratio 17.4);
- grid 9.92 GB (3.68 bits/sample, ratio 17.4);
- grid+gaps 9.95 GB (3.68 bits/sample, ratio 17.4);
- noisy 29.16 GB (10.80 bits/sample, ratio 5.9);
