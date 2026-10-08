# fluxcode day-scale stress test (bench/stress.py, 2026-10-07, 4 threads, AC power)

One run of each at 0581c29, on AC power with a 45 s pause before each step, replacing the tables at d039350. The noise
floor's estimate now takes the quietest of 256-difference windows, and runs on block groups with times too (on
irregular blocks from consecutive scans only, scaled by their share; it used to be off with times). The block groups
are the same size here: the day compresses to 39.89 GB (39.88 at d039350), with timestamps to the same 47.68 GB. Against
d039350, encode µs/block/worker: the full day 4.74 → 4.78 (+1%); with timestamps 6.05 → 6.45 (+7%), noisy clocks alone
10.49 → 12.24 (+17%: on jittered times every block measures its cadence and estimates the noise on one-scan
triplets), regular grids 5.16 → 5.31 (+3%: the estimate that used to be skipped with times); sprinkled NaN 6.77 →
7.16 (+6%) and with held data 5.17 → 5.59 (+8%), where every block takes the weighted estimate; the
other variants +1 to +4%. Decode is unchanged code (within 1%). The load average was 2.2–3.6 from other apps
before the steps (about 2 at d039350). These are the Python path (no Rust extension); the extension's effect on
4 threads is in `experimental/rust_port/results/stress_effort.md` and, for timestamps at effort 5, in the last table here.

## Full target: `uv run python bench/stress.py --breakdown`

Apple M3 Mac15,13, Darwin 25.6.0, Python 3.11.13, numpy 2.4.6

data `sensor-mix`: 1000 tags x 1440 block groups x 60 x 1000 = 86.40e9 samples (691.2 GB float64); 4 threads; pool 32 block groups/kind
NaN: 3 min/tag/day in 2 runs -> 5,011 block groups touched (0.35%), 0.208% of samples
tags per kind: analog 350, held 250, digital 100, vibration 100, counter 50, sensor-0.1 50, noisy-sine 50, random-walk 50

## Single thread, one block group, µs (best of 5 x 200)

| kind | bits/sample | kernels | write_group | to_bit_planes | zstd (both layouts) | encode_group | unzstd | decode kernel | decode_group |
|---|---|---|---|---|---|---|---|---|---|
| analog | 5.90 | 104 | 5 | 23 | 110 | 249 | 18 | 61 | 88 |
| held | 0.26 | 84 | 5 | 23 | 40 | 159 | 20 | 69 | 101 |
| digital | 0.01 | 44 | 5 | 23 | 17 | 97 | 30 | 50 | 91 |
| vibration | 6.15 | 104 | 5 | 23 | 105 | 244 | 39 | 82 | 127 |
| counter | 1.58 | 122 | 5 | 23 | 113 | 270 | 43 | 82 | 134 |
| sensor-0.1 | 1.77 | 121 | 5 | 23 | 177 | 331 | 21 | 84 | 112 |
| noisy-sine | 5.46 | 104 | 5 | 24 | 107 | 246 | 48 | 81 | 140 |
| random-walk | 12.67 | 105 | 5 | 24 | 168 | 308 | 9 | 83 | 99 |

## Result

| phase | wall | float64 GB/s | samples/s | µs/block/worker | in-call % |
|---|---|---|---|---|---|
| encode | 103.2 s | 6.70 GB/s | 837 M | 4.78 | 100% |
| decode | 38.6 s | 17.92 GB/s | 2240 M | 1.79 | 99% |

compressed 39.89 GB (3.69 bits/sample, ratio 17.3); real time is 86,400 s per day, so encode runs at 837x real time for 1000 tags.

encode GB/s per tenth of the phase: 7.4 7.2 6.5 6.8 6.8 6.4 6.6 6.1 6.6 6.6
decode GB/s per tenth of the phase: 17.9 18.5 17.6 17.7 18.5 17.5 17.7 17.7 17.8 18.4

## Adversarial variants (`--scale 0.25`: 250 tags x 1 day; per-block cost doesn't depend on scale)

| variant | encode µs/block/worker | encode GB/s | decode GB/s | bits/sample | full day (1000 tags) enc / dec |
|---|---|---|---|---|---|
| sensor-mix, 3 NaN min/tag/day (the full run above) | 4.78 | 6.70 | 17.92 | 3.69 | 103 s / 39 s |
| `--data random-walk` (worst entropy) | 6.40 | 5.00 | 15.75 | 12.57 | 138 s / 44 s |
| `--nan-minutes 120 --nan-runs 20` (8% NaN, in runs) | 4.61 | 6.95 | 18.40 | 3.38 | 100 s / 38 s |
| `--sprinkle 0.0005` (~40% of blocks flagged) | 5.25 | 6.09 | 16.37 | 3.69 | 114 s / 42 s |
| `--sprinkle 0.01` (every block flagged) | 7.16 | 4.47 | 13.46 | 3.84 | 155 s / 51 s |
| `--data held --sprinkle 0.01` | 5.59 | 5.73 | 16.54 | 0.29 | 121 s / 42 s |

## Decimal detection near miss: `uv run python bench/decimal_near_miss.py`

A block on a decimal grid except for its last sample makes each candidate exponent scan the whole
block before failing. Single thread, µs/block; the method is in the script's docstring. With the
second zstd pass in every block's cost, the near miss is within run-to-run noise.

| input | µs/block |
|---|---|
| random walk | 5.18 |
| random walk, `decimal_detection=False` | 5.13 |
| on a 0.01 grid | 5.75 |
| integers | 5.76 |
| integers, off-grid last sample | 5.72 |
| 0.01 grid, off-grid last sample | 5.90 |
| the same, `decimal_detection=False` | 5.05 |

## Timestamps: `uv run python bench/stress.py --times clock-mix`

Apple M3 Mac15,13, Darwin 25.6.0, Python 3.11.13, numpy 2.4.6

data `sensor-mix`: 1000 tags x 1440 block groups x 60 x 1000 = 86.40e9 samples (691.2 GB float64); 4 threads; pool 32 block groups/kind
NaN: 3 min/tag/day in 2 runs -> 5,011 block groups touched (0.35%), 0.208% of samples
tags per kind: analog 350, held 250, digital 100, vibration 100, counter 50, sensor-0.1 50, noisy-sine 50, random-walk 50
timestamps (datetime64[ns], exact), tags per clock: grid 600, grid+gaps 300, noisy 100

## Result

| phase | wall | float64 GB/s | samples/s | µs/block/worker | in-call % |
|---|---|---|---|---|---|
| encode | 139.4 s | 4.96 GB/s | 620 M | 6.45 | 100% |
| decode | 54.8 s | 12.62 GB/s | 1578 M | 2.54 | 100% |

compressed 47.68 GB (4.42 bits/sample, ratio 14.5); real time is 86,400 s per day, so encode runs at 620x real time for 1000 tags.

encode GB/s per tenth of the phase: 5.5 5.3 4.9 5.2 5.1 4.7 4.9 4.7 4.7 4.8
decode GB/s per tenth of the phase: 12.6 12.6 12.6 12.1 12.9 12.7 12.6 12.9 12.2 12.7

### Per clock kind: `--scale 0.25 --times <clock>` (same session)

| clock | phase | wall | float64 GB/s | samples/s | µs/block/worker | in-call % |
|---|---|---|---|---|---|---|
| none | encode | 25.1 s | 6.87 GB/s | 859 M | 4.66 | 99% |
| none | decode | 9.4 s | 18.31 GB/s | 2289 M | 1.75 | 99% |
| grid | encode | 28.7 s | 6.02 GB/s | 753 M | 5.31 | 99% |
| grid | decode | 12.1 s | 14.23 GB/s | 1778 M | 2.25 | 99% |
| grid+gaps | encode | 30.2 s | 5.73 GB/s | 716 M | 5.58 | 99% |
| grid+gaps | decode | 12.8 s | 13.52 GB/s | 1690 M | 2.37 | 99% |
| noisy | encode | 66.1 s | 2.61 GB/s | 327 M | 12.24 | 99% |
| noisy | decode | 25.1 s | 6.88 GB/s | 860 M | 4.65 | 100% |

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
| Python | 52.1 s | 3.32 GB/s | 9.65 | 13.5 s |
| Rust extension | 35.2 s | 4.91 GB/s | 6.51 | 13.6 s |
