# fluxcode day-scale stress test (bench/stress.py, 2026-10-07, 4 threads, AC power)

One run of each at 4e89772, on AC power with a 45 s pause before each step, replacing the tables at 0581c29. The noise
floor's cadence on irregular blocks is now the median of 15 intervals, all or nothing (a block gets the floor if 90% of
its intervals are one scan), its estimate on all-valid triplets takes the plain estimator, and σ is the windows' mean
unless the quietest is below 0.75 of it. The block groups are the same size: the day compresses to 39.88 GB (39.89 at
0581c29, 39.88 before the windows), with timestamps to the same 47.68 GB. Against 0581c29, encode µs/block/worker: the
full day 4.78 → 4.62 (−3%); with timestamps 6.45 → 6.16 (−4%), noisy clocks alone 12.24 → 11.74 (−4%), regular grids
5.31 → 5.12 (−4%), grids with gaps 5.58 → 5.37 (−4%); the random walk 6.40 → 6.19, sprinkled NaN 7.16 → 7.13 and with
held data 5.59 → 5.59. Decode is unchanged code (within 1–2%). The load average was 1.9–3.9 from other apps before the
steps. These are the Python path (no Rust extension); the extension's effect on 4 threads is in
`experimental/rust_port/results/stress_effort.md` and, for timestamps at effort 5, in the last table here.

## Full target: `uv run python bench/stress.py --breakdown`

Apple M3 Mac15,13, Darwin 25.6.0, Python 3.11.13, numpy 2.4.6

data `sensor-mix`: 1000 tags x 1440 block groups x 60 x 1000 = 86.40e9 samples (691.2 GB float64); 4 threads; pool 32 block groups/kind
NaN: 3 min/tag/day in 2 runs -> 5,011 block groups touched (0.35%), 0.208% of samples
tags per kind: analog 350, held 250, digital 100, vibration 100, counter 50, sensor-0.1 50, noisy-sine 50, random-walk 50

## Single thread, one block group, µs (best of 5 x 200)

| kind | bits/sample | kernels | write_group | to_bit_planes | zstd (both layouts) | encode_group | unzstd | decode kernel | decode_group |
|---|---|---|---|---|---|---|---|---|---|
| analog | 5.90 | 105 | 5 | 23 | 110 | 250 | 20 | 57 | 84 |
| held | 0.26 | 83 | 5 | 23 | 37 | 158 | 24 | 65 | 97 |
| digital | 0.01 | 45 | 5 | 22 | 17 | 96 | 30 | 48 | 91 |
| vibration | 6.15 | 106 | 5 | 23 | 105 | 246 | 36 | 79 | 122 |
| counter | 1.58 | 124 | 5 | 22 | 113 | 273 | 49 | 78 | 133 |
| sensor-0.1 | 1.77 | 123 | 5 | 23 | 180 | 337 | 21 | 81 | 109 |
| noisy-sine | 5.46 | 105 | 5 | 23 | 106 | 248 | 50 | 79 | 135 |
| random-walk | 12.67 | 106 | 5 | 23 | 168 | 311 | 9 | 81 | 97 |

## Result

| phase | wall | float64 GB/s | samples/s | µs/block/worker | in-call % |
|---|---|---|---|---|---|
| encode | 99.8 s | 6.92 GB/s | 866 M | 4.62 | 100% |
| decode | 38.2 s | 18.09 GB/s | 2262 M | 1.77 | 99% |

compressed 39.88 GB (3.69 bits/sample, ratio 17.3); real time is 86,400 s per day, so encode runs at 866x real time for 1000 tags.

encode GB/s per tenth of the phase: 7.3 7.5 6.6 7.1 7.1 6.8 6.6 6.7 6.8 6.9
decode GB/s per tenth of the phase: 18.1 18.6 17.7 17.9 18.5 17.5 18.1 17.9 18.1 18.5

## Adversarial variants (`--scale 0.25`: 250 tags x 1 day; per-block cost doesn't depend on scale)

| variant | encode µs/block/worker | encode GB/s | decode GB/s | bits/sample | full day (1000 tags) enc / dec |
|---|---|---|---|---|---|
| sensor-mix, 3 NaN min/tag/day (the full run above) | 4.62 | 6.92 | 18.09 | 3.69 | 100 s / 38 s |
| `--data random-walk` (worst entropy) | 6.19 | 5.17 | 16.01 | 12.57 | 134 s / 43 s |
| `--nan-minutes 120 --nan-runs 20` (8% NaN, in runs) | 4.53 | 7.06 | 18.57 | 3.38 | 98 s / 37 s |
| `--sprinkle 0.0005` (~40% of blocks flagged) | 5.15 | 6.21 | 16.55 | 3.69 | 111 s / 42 s |
| `--sprinkle 0.01` (every block flagged) | 7.13 | 4.49 | 13.43 | 3.84 | 154 s / 52 s |
| `--data held --sprinkle 0.01` | 5.59 | 5.73 | 16.58 | 0.29 | 121 s / 42 s |

## Decimal detection near miss: `uv run python bench/decimal_near_miss.py`

A block on a decimal grid except for its last sample makes each candidate exponent scan the whole
block before failing. Single thread, µs/block; the method is in the script's docstring. With the
second zstd pass in every block's cost, the near miss is within run-to-run noise.

| input | µs/block |
|---|---|
| random walk | 5.18 |
| random walk, `decimal_detection=False` | 5.14 |
| on a 0.01 grid | 5.76 |
| integers | 5.77 |
| integers, off-grid last sample | 5.72 |
| 0.01 grid, off-grid last sample | 5.90 |
| the same, `decimal_detection=False` | 5.06 |

## Timestamps: `uv run python bench/stress.py --times clock-mix`

Apple M3 Mac15,13, Darwin 25.6.0, Python 3.11.13, numpy 2.4.6

data `sensor-mix`: 1000 tags x 1440 block groups x 60 x 1000 = 86.40e9 samples (691.2 GB float64); 4 threads; pool 32 block groups/kind
NaN: 3 min/tag/day in 2 runs -> 5,011 block groups touched (0.35%), 0.208% of samples
tags per kind: analog 350, held 250, digital 100, vibration 100, counter 50, sensor-0.1 50, noisy-sine 50, random-walk 50
timestamps (datetime64[ns], exact), tags per clock: grid 600, grid+gaps 300, noisy 100

## Result

| phase | wall | float64 GB/s | samples/s | µs/block/worker | in-call % |
|---|---|---|---|---|---|
| encode | 133.0 s | 5.20 GB/s | 650 M | 6.16 | 100% |
| decode | 54.2 s | 12.75 GB/s | 1593 M | 2.51 | 100% |

compressed 47.68 GB (4.41 bits/sample, ratio 14.5); real time is 86,400 s per day, so encode runs at 650x real time for 1000 tags.

encode GB/s per tenth of the phase: 5.7 5.5 5.1 5.2 5.4 5.0 5.1 5.0 5.0 5.0
decode GB/s per tenth of the phase: 12.7 12.8 12.6 12.4 13.0 12.9 12.9 13.1 12.4 12.7

### Per clock kind: `--scale 0.25 --times <clock>` (same session)

| clock | phase | wall | float64 GB/s | samples/s | µs/block/worker | in-call % |
|---|---|---|---|---|---|---|
| none | encode | 24.8 s | 6.96 GB/s | 871 M | 4.59 | 99% |
| none | decode | 9.4 s | 18.42 GB/s | 2302 M | 1.74 | 99% |
| grid | encode | 27.7 s | 6.24 GB/s | 781 M | 5.12 | 99% |
| grid | decode | 12.0 s | 14.44 GB/s | 1805 M | 2.22 | 99% |
| grid+gaps | encode | 29.0 s | 5.96 GB/s | 745 M | 5.37 | 99% |
| grid+gaps | decode | 12.8 s | 13.46 GB/s | 1683 M | 2.38 | 99% |
| noisy | encode | 63.4 s | 2.73 GB/s | 341 M | 11.74 | 99% |
| noisy | decode | 24.7 s | 6.98 GB/s | 873 M | 4.58 | 100% |

- none 9.91 GB (3.67 bits/sample, ratio 17.4);
- grid 9.92 GB (3.68 bits/sample, ratio 17.4);
- grid+gaps 9.95 GB (3.68 bits/sample, ratio 17.4);
- noisy 29.16 GB (10.80 bits/sample, ratio 5.9);

### Time-axis block groups at effort 5 (block flushes), with and without the Rust extension

`--scale 0.25 --times clock-mix --effort 5`, 4 threads, same session. Block groups with a time axis go through
the extension's `pack_group` (the body is built in Python, the layout choice, flush points and zstd frame
without the GIL), so the flushes scale: the block groups are the same bytes (11.65 GB, 4.32 bits/sample).

| path | encode | float64 GB/s | µs/block/worker | decode |
|---|---|---|---|---|
| Python | 51.8 s | 3.34 GB/s | 9.59 | 13.5 s |
| Rust extension | 33.8 s | 5.11 GB/s | 6.26 | 13.3 s |
