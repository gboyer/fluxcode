# fluxcode day-scale stress test (bench/stress.py, 2026-10-13, 4 threads, AC power)

One run of each at 7fa0991, on AC power with a 45 s pause before each step, replacing the tables at 4e89772. Since then
the encoder gained timestamp rounding (off by default, so not in these runs) and lost the explicit layouts and test-only
code. The block groups are the same size: the day compresses to 39.88 GB, with timestamps to 47.68 GB. Against
4e89772, encode µs/block/worker: the full day 4.62 → 4.52 (−2%); with timestamps 6.16 → 5.87 (−5%), noisy clocks
alone 11.74 → 11.57 (−1%), regular grids 5.12 → 5.05, grids with gaps 5.37 → 5.26 (−2%); the random walk 6.19 → 6.05,
sprinkled NaN 7.13 → 6.90 and with held data 5.59 → 5.43. Decode is 1–5% faster (the full day 1.77 → 1.69). The load
average was 1.4–3.1 before the steps. These are the Python path (no Rust extension); the extension's effect on 4
threads is in `experimental/rust_port/results/stress_effort.md` and, for timestamps at effort 5, in the last table here.

## Full target: `uv run python bench/stress.py --breakdown`

Apple M3 Mac15,13, Darwin 25.6.0, Python 3.11.13, numpy 2.4.6

data `sensor-mix`: 1000 tags x 1440 block groups x 60 x 1000 = 86.40e9 samples (691.2 GB float64); 4 threads; pool 32 block groups/kind
NaN: 3 min/tag/day in 2 runs -> 5,011 block groups touched (0.35%), 0.208% of samples
tags per kind: analog 350, held 250, digital 100, vibration 100, counter 50, sensor-0.1 50, noisy-sine 50, random-walk 50

## Single thread, one block group, µs (best of 5 x 200)

| kind | bits/sample | kernels | write_group | to_bit_planes | zstd (both layouts) | encode_group | unzstd | decode kernel | decode_group |
|---|---|---|---|---|---|---|---|---|---|
| analog | 5.90 | 102 | 4 | 22 | 106 | 243 | 20 | 56 | 84 |
| held | 0.26 | 80 | 4 | 21 | 37 | 151 | 24 | 65 | 97 |
| digital | 0.01 | 43 | 4 | 21 | 17 | 92 | 30 | 48 | 91 |
| vibration | 6.15 | 103 | 4 | 22 | 103 | 241 | 36 | 77 | 118 |
| counter | 1.58 | 122 | 4 | 22 | 110 | 265 | 47 | 78 | 133 |
| sensor-0.1 | 1.77 | 120 | 4 | 22 | 171 | 326 | 19 | 78 | 104 |
| noisy-sine | 5.46 | 104 | 4 | 23 | 105 | 243 | 50 | 76 | 129 |
| random-walk | 12.67 | 104 | 4 | 23 | 168 | 308 | 8 | 77 | 93 |

## Result

| phase | wall | float64 GB/s | samples/s | µs/block/worker | in-call % |
|---|---|---|---|---|---|
| encode | 97.7 s | 7.08 GB/s | 885 M | 4.52 | 100% |
| decode | 36.6 s | 18.88 GB/s | 2360 M | 1.69 | 99% |

compressed 39.88 GB (3.69 bits/sample, ratio 17.3); real time is 86,400 s per day, so encode runs at 885x real time for 1000 tags.

encode GB/s per tenth of the phase: 7.4 7.6 6.8 7.1 7.2 6.8 6.9 6.8 6.9 7.0
decode GB/s per tenth of the phase: 18.2 19.2 18.6 18.3 19.2 18.9 18.8 19.3 18.7 19.4

## Adversarial variants (`--scale 0.25`: 250 tags x 1 day; per-block cost doesn't depend on scale)

| variant | encode µs/block/worker | encode GB/s | decode GB/s | bits/sample | full day (1000 tags) enc / dec |
|---|---|---|---|---|---|
| sensor-mix, 3 NaN min/tag/day (the full run above) | 4.52 | 7.08 | 18.88 | 3.69 | 98 s / 37 s |
| `--data random-walk` (worst entropy) | 6.05 | 5.29 | 16.64 | 12.57 | 131 s / 42 s |
| `--nan-minutes 120 --nan-runs 20` (8% NaN, in runs) | 4.36 | 7.34 | 18.95 | 3.38 | 94 s / 36 s |
| `--sprinkle 0.0005` (~40% of blocks flagged) | 4.99 | 6.41 | 16.61 | 3.69 | 108 s / 42 s |
| `--sprinkle 0.01` (every block flagged) | 6.90 | 4.64 | 13.93 | 3.84 | 149 s / 50 s |
| `--data held --sprinkle 0.01` | 5.43 | 5.89 | 16.55 | 0.29 | 117 s / 42 s |

## Decimal detection near miss: `uv run python bench/decimal_near_miss.py`

A block on a decimal grid except for its last sample makes each candidate exponent scan the whole
block before failing. Single thread, µs/block; the method is in the script's docstring. With the
second zstd pass in every block's cost, the near miss is within run-to-run noise.

| input | µs/block |
|---|---|
| random walk | 5.03 |
| random walk, `decimal_detection=False` | 5.09 |
| on a 0.01 grid | 5.75 |
| integers | 5.74 |
| integers, off-grid last sample | 5.71 |
| 0.01 grid, off-grid last sample | 5.88 |
| the same, `decimal_detection=False` | 5.04 |

## Timestamps: `uv run python bench/stress.py --times clock-mix`

Apple M3 Mac15,13, Darwin 25.6.0, Python 3.11.13, numpy 2.4.6

data `sensor-mix`: 1000 tags x 1440 block groups x 60 x 1000 = 86.40e9 samples (691.2 GB float64); 4 threads; pool 32 block groups/kind
NaN: 3 min/tag/day in 2 runs -> 5,011 block groups touched (0.35%), 0.208% of samples
tags per kind: analog 350, held 250, digital 100, vibration 100, counter 50, sensor-0.1 50, noisy-sine 50, random-walk 50
timestamps (datetime64[ns], exact), tags per clock: grid 600, grid+gaps 300, noisy 100

## Result

| phase | wall | float64 GB/s | samples/s | µs/block/worker | in-call % |
|---|---|---|---|---|---|
| encode | 126.8 s | 5.45 GB/s | 681 M | 5.87 | 100% |
| decode | 51.9 s | 13.32 GB/s | 1665 M | 2.40 | 100% |

compressed 47.68 GB (4.41 bits/sample, ratio 14.5); real time is 86,400 s per day, so encode runs at 681x real time for 1000 tags.

encode GB/s per tenth of the phase: 5.8 5.6 5.3 5.4 5.6 5.3 5.3 5.4 5.2 5.3
decode GB/s per tenth of the phase: 13.2 13.3 13.1 12.8 13.7 13.5 13.5 13.7 12.9 13.5

### Per clock kind: `--scale 0.25 --times <clock>` (same session)

| clock | phase | wall | float64 GB/s | samples/s | µs/block/worker | in-call % |
|---|---|---|---|---|---|---|
| none | encode | 23.8 s | 7.25 GB/s | 906 M | 4.42 | 99% |
| none | decode | 9.1 s | 18.96 GB/s | 2370 M | 1.69 | 99% |
| grid | encode | 27.3 s | 6.33 GB/s | 792 M | 5.05 | 99% |
| grid | decode | 11.8 s | 14.60 GB/s | 1825 M | 2.19 | 99% |
| grid+gaps | encode | 28.4 s | 6.08 GB/s | 760 M | 5.26 | 99% |
| grid+gaps | decode | 12.6 s | 13.75 GB/s | 1719 M | 2.33 | 99% |
| noisy | encode | 62.5 s | 2.76 GB/s | 346 M | 11.57 | 99% |
| noisy | decode | 24.4 s | 7.07 GB/s | 884 M | 4.53 | 100% |

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
| Python | 50.3 s | 3.44 GB/s | 9.31 | 12.9 s |
| Rust extension | 32.6 s | 5.30 GB/s | 6.04 | 12.8 s |
