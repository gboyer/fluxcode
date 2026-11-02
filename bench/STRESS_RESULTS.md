# fluxcode day-scale stress test (bench/stress.py, 2026-11-01, 4 threads, AC power)

One run of each with the scaled grids (`grids="rational"`, the default), on AC power with a 45 s pause before each step
and the load average below 2.0, and of the two full targets on `main` at b48c7a3 (the encoder before them) in the same
session. The day compresses to 39.91 GB against 39.88 (+0.08%: the 4-byte grid scale per block), with timestamps to
47.71 GB against 47.68. Encode µs/block/worker against b48c7a3: the full day 4.59 → 4.74 (+0.15 µs, +3%), with
timestamps 5.99 → 6.19 (+0.20 µs, +3%); decode 1.75 → 1.76 and 2.46 → 2.48. Held data pay most in the kernels
(84 → 112 µs per block group: their few levels go through the index gcd), and their zstd and decode got faster. Two new
variants are on grids a writer doesn't know. These are the Python path (no Rust extension); the extension's effect on
4 threads is in `experimental/rust_port/results/stress_effort.md` and, for timestamps at effort 5, in the last table
here.

## Full target: `uv run python bench/stress.py --breakdown`

Apple M3 Mac15,13, Darwin 25.6.0, Python 3.11.13, numpy 2.4.6

data `sensor-mix`: 1000 tags x 1440 block groups x 60 x 1000 = 86.40e9 samples (691.2 GB float64); 4 threads; pool 32 block groups/kind
NaN: 3 min/tag/day in 2 runs -> 5,011 block groups touched (0.35%), 0.208% of samples
tags per kind: analog 350, held 250, digital 100, vibration 100, counter 50, sensor-0.1 50, noisy-sine 50, random-walk 50
(pool generated, roundtrip-checked (worst error 1.41e-02 of block range) and warmed in 2 s)

## Single thread, one block group, µs (best of 5 x 200)

| kind | bits/sample | kernels | write_group | to_bit_planes | zstd (both layouts) | encode_group | unzstd | decode kernel | decode_group |
|---|---|---|---|---|---|---|---|---|---|
| analog | 5.90 | 107 | 5 | 23 | 109 | 252 | 20 | 57 | 83 |
| held | 0.25 | 112 | 4 | 22 | 37 | 184 | 11 | 66 | 82 |
| digital | 0.01 | 45 | 5 | 22 | 17 | 96 | 18 | 48 | 74 |
| vibration | 6.15 | 109 | 5 | 23 | 105 | 250 | 36 | 77 | 120 |
| counter | 1.58 | 125 | 4 | 22 | 112 | 273 | 43 | 78 | 130 |
| sensor-0.1 | 1.77 | 124 | 5 | 22 | 177 | 335 | 20 | 79 | 106 |
| noisy-sine | 5.46 | 108 | 5 | 23 | 107 | 250 | 49 | 78 | 131 |
| random-walk | 12.67 | 112 | 5 | 24 | 168 | 316 | 9 | 80 | 96 |

## Result

| phase | wall | float64 GB/s | samples/s | µs/block/worker | in-call % |
|---|---|---|---|---|---|
| encode | 102.4 s | 6.75 GB/s | 844 M | 4.74 | 100% |
| decode | 37.9 s | 18.22 GB/s | 2277 M | 1.76 | 99% |

compressed 39.91 GB (3.70 bits/sample, ratio 17.3); real time is 86,400 s per day, so encode runs at 844x real time for 1000 tags.

encode GB/s per tenth of the phase: 7.2 7.2 6.5 7.0 6.8 6.5 6.6 6.4 6.6 6.7
decode GB/s per tenth of the phase: 18.2 18.6 18.0 18.0 18.2 18.1 18.2 18.3 18.3 18.2

## Adversarial variants (`--scale 0.25`: 250 tags x 1 day; per-block cost doesn't depend on scale)

| variant | encode µs/block/worker | encode GB/s | decode GB/s | bits/sample | full day (1000 tags) enc / dec |
|---|---|---|---|---|---|
| sensor-mix, 3 NaN min/tag/day (the full run above) | 4.74 | 6.75 | 18.22 | 3.70 | 102 s / 38 s |
| `--data random-walk` (worst entropy) | 6.21 | 5.15 | 16.23 | 12.57 | 134 s / 42 s |
| `--nan-minutes 120 --nan-runs 20` (8% NaN, in runs) | 4.57 | 7.01 | 18.83 | 3.38 | 99 s / 37 s |
| `--sprinkle 0.0005` (~40% of blocks flagged) | 5.21 | 6.14 | 16.68 | 3.69 | 112 s / 42 s |
| `--sprinkle 0.01` (every block flagged) | 7.16 | 4.47 | 13.93 | 3.85 | 155 s / 50 s |
| `--data held --sprinkle 0.01` | 5.89 | 5.44 | 16.59 | 0.30 | 127 s / 42 s |
| `--data "random-walk q0.05"` (a scaled grid, N = 5 on 0.01) | 6.36 | 5.03 | 15.46 | 6.51 | 137 s / 45 s |
| `--data "random-walk adc 10/4095"` (the rational search) | 13.05 | 2.45 | 16.11 | 11.32 | 282 s / 43 s |

## Grid detection near miss: `uv run python bench/decimal_near_miss.py`

A block on a grid except for its last sample makes each candidate grid scan the block before failing, and then the
rational search runs. Single thread, µs/block; the method is in the script's docstring. The integers with one
off-grid sample cost +3.0 µs (+51%) over integers, the 0.01 grid +1.7 µs (+29%): each tries the scaled grids before
storing the block on a power-of-two grid. `grids="binary"` runs no search.

| input | µs/block |
|---|---|
| random walk | 5.11 |
| random walk, `grids="binary"` | 5.12 |
| on a 0.01 grid | 5.80 |
| integers | 5.82 |
| integers, off-grid last sample | 8.80 |
| 0.01 grid, off-grid last sample | 7.50 |
| the same, `grids="binary"` | 5.09 |

## Timestamps: `uv run python bench/stress.py --times clock-mix`

Apple M3 Mac15,13, Darwin 25.6.0, Python 3.11.13, numpy 2.4.6

data `sensor-mix`: 1000 tags x 1440 block groups x 60 x 1000 = 86.40e9 samples (691.2 GB float64); 4 threads; pool 32 block groups/kind
NaN: 3 min/tag/day in 2 runs -> 5,011 block groups touched (0.35%), 0.208% of samples
tags per kind: analog 350, held 250, digital 100, vibration 100, counter 50, sensor-0.1 50, noisy-sine 50, random-walk 50
timestamps (datetime64[ns], exact), tags per clock: grid 600, grid+gaps 300, noisy 100
(pool generated, roundtrip-checked (worst error 1.41e-02 of block range) and warmed in 6 s)

## Result

| phase | wall | float64 GB/s | samples/s | µs/block/worker | in-call % |
|---|---|---|---|---|---|
| encode | 133.7 s | 5.17 GB/s | 646 M | 6.19 | 99% |
| decode | 53.6 s | 12.88 GB/s | 1610 M | 2.48 | 100% |

compressed 47.71 GB (4.42 bits/sample, ratio 14.5); real time is 86,400 s per day, so encode runs at 646x real time for 1000 tags.

encode GB/s per tenth of the phase: 5.5 5.5 5.1 5.2 5.1 5.1 5.1 5.1 4.9 5.0
decode GB/s per tenth of the phase: 12.8 12.9 12.8 12.6 13.1 13.0 13.0 13.3 12.5 13.0

### Per clock kind: `--scale 0.25 --times <clock>` (same session)

| clock | phase | wall | float64 GB/s | samples/s | µs/block/worker | in-call % |
|---|---|---|---|---|---|---|
| none | encode | 24.7 s | 6.99 GB/s | 874 M | 4.58 | 99% |
| none | decode | 9.2 s | 18.86 GB/s | 2358 M | 1.70 | 99% |
| grid | encode | 28.0 s | 6.16 GB/s | 770 M | 5.19 | 99% |
| grid | decode | 11.9 s | 14.52 GB/s | 1815 M | 2.20 | 99% |
| grid+gaps | encode | 29.0 s | 5.97 GB/s | 746 M | 5.36 | 99% |
| grid+gaps | decode | 12.6 s | 13.71 GB/s | 1714 M | 2.33 | 99% |
| noisy | encode | 62.3 s | 2.77 GB/s | 347 M | 11.53 | 99% |
| noisy | decode | 23.9 s | 7.22 GB/s | 902 M | 4.43 | 100% |

- none 9.91 GB (3.67 bits/sample, ratio 17.4);
- grid 9.93 GB (3.68 bits/sample, ratio 17.4);
- grid+gaps 9.95 GB (3.69 bits/sample, ratio 17.4);
- noisy 29.17 GB (10.80 bits/sample, ratio 5.9);

### Time-axis block groups at effort 5 (block flushes), with and without the Rust extension

`--scale 0.25 --times clock-mix --effort 5`, 4 threads, same session. Block groups with a time axis go through
the extension's `pack_group` (the body is built in Python, the layout choice, flush points and zstd frame
without the GIL), so the flushes scale: the block groups are the same bytes (11.66 GB, 4.32 bits/sample).

| path | encode | float64 GB/s | µs/block/worker | decode |
|---|---|---|---|---|
| Python | 50.5 s | 3.42 GB/s | 9.35 | 13.0 s |
| Rust extension | 33.3 s | 5.19 GB/s | 6.17 | 13.0 s |
