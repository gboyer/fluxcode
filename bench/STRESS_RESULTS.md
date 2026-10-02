# fluxcode day-scale stress test (bench/stress.py, 2026-10-02, 4 threads, AC power)

One run of each at d039350, on AC power with a 45 s pause before each step. The default (effort 4) produces
the same units as at 5ee16fa, byte for byte: the day compresses to the same 39.88 GB, and the speeds are a little
better (encode 106.6 → 102.4 s, -4%; decode 38.5 → 38.1 s, -1%; with timestamps 141.5 → 130.7 s, -8%; the
adversarial variants -2 to -7%). The Python path now writes the byte-plane body once and derives the
bit planes with one transpose (a unit's `write_unit` is 5 µs
instead of 31, plus a 23 µs transpose for the second layout). The machine showed a load average of about 2 from other
apps before the steps. These are the Python path (no Rust extension); the extension's effect on 4 threads is in
`experimental/rust_port/results/stress_effort.md` and, for timestamps at effort 5, in the last table here.

## Full target: `uv run python bench/stress.py --breakdown`

Apple M3 Mac15,13, Darwin 25.6.0, Python 3.11.13, numpy 2.4.6

data `sensor-mix`: 1000 tags x 1440 units x 60 x 1000 = 86.40e9 samples (691.2 GB float64); 4 threads; pool 32 units/kind
NaN: 3 min/tag/day in 2 runs -> 5,011 units touched (0.35%), 0.208% of samples
tags per kind: analog 350, held 250, digital 100, vibration 100, counter 50, sensor-0.1 50, noisy-sine 50, random-walk 50
(pool generated, roundtrip-checked (worst error 1.41e-02 of block range) and warmed in 2 s)

## Single thread, one unit, µs (best of 5 x 200)

| kind | bits/sample | kernels | write_unit | to_bit_planes | zstd (both layouts) | encode_unit | unzstd | decode kernel | decode_unit |
|---|---|---|---|---|---|---|---|---|---|
| analog | 5.90 | 97 | 5 | 22 | 107 | 247 | 20 | 60 | 88 |
| held | 0.26 | 96 | 5 | 23 | 39 | 172 | 25 | 67 | 91 |
| digital | 0.01 | 44 | 5 | 22 | 17 | 97 | 20 | 49 | 83 |
| vibration | 6.15 | 100 | 5 | 24 | 106 | 242 | 37 | 82 | 126 |
| counter | 1.58 | 118 | 5 | 23 | 112 | 267 | 43 | 79 | 130 |
| sensor-0.1 | 1.77 | 117 | 5 | 23 | 180 | 336 | 21 | 85 | 112 |
| noisy-sine | 5.46 | 100 | 5 | 24 | 107 | 242 | 49 | 82 | 139 |
| random-walk | 12.67 | 101 | 5 | 24 | 169 | 306 | 9 | 83 | 95 |

## Result

| phase | wall | float64 GB/s | samples/s | µs/block/worker | in-call % |
|---|---|---|---|---|---|
| encode | 102.4 s | 6.75 GB/s | 844 M | 4.74 | 100% |
| decode | 38.1 s | 18.12 GB/s | 2265 M | 1.77 | 99% |

compressed 39.88 GB (3.69 bits/sample, ratio 17.3); real time is 86,400 s per day, so encode runs at 844x real time for 1000 tags.

encode GB/s per tenth of the phase: 7.2 7.2 6.4 7.0 6.8 6.4 6.6 6.5 6.6 6.8
decode GB/s per tenth of the phase: 18.1 18.7 17.8 17.9 18.7 17.8 17.8 17.9 18.0 18.6

## Adversarial variants (`--scale 0.25`: 250 tags x 1 day; per-block cost doesn't depend on scale)

| variant | encode µs/block/worker | encode GB/s | decode GB/s | bits/sample | full day (1000 tags) enc / dec |
|---|---|---|---|---|---|
| sensor-mix, 3 NaN min/tag/day (the full run above) | 4.74 | 6.75 | 18.12 | 3.69 | 102 s / 38 s |
| `--data random-walk` (worst entropy) | 6.18 | 5.17 | 16.05 | 12.57 | 134 s / 43 s |
| `--nan-minutes 120 --nan-runs 20` (8% NaN, in runs) | 4.54 | 7.04 | 18.65 | 3.38 | 98 s / 37 s |
| `--sprinkle 0.0005` (~40% of blocks flagged) | 5.05 | 6.34 | 16.58 | 3.69 | 109 s / 42 s |
| `--sprinkle 0.01` (every block flagged) | 6.77 | 4.72 | 13.68 | 3.84 | 146 s / 50 s |
| `--data held --sprinkle 0.01` | 5.17 | 6.18 | 16.53 | 0.29 | 112 s / 42 s |

## Decimal detection near miss: `uv run python bench/decimal_near_miss.py`

A block on a decimal grid except for its last sample makes each candidate exponent scan the whole
block before failing. Single thread, µs/block; the method is in the script's docstring. With the
second zstd pass in every block's cost, the near miss is within run-to-run noise.

| input | µs/block |
|---|---|
| random walk | 5.08 |
| random walk, `decimal_detection=False` | 5.04 |
| on a 0.01 grid | 5.66 |
| integers | 5.66 |
| integers, off-grid last sample | 5.62 |
| 0.01 grid, off-grid last sample | 5.79 |
| the same, `decimal_detection=False` | 4.96 |

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
| encode | 130.7 s | 5.29 GB/s | 661 M | 6.05 | 100% |
| decode | 54.2 s | 12.76 GB/s | 1595 M | 2.51 | 100% |

compressed 47.68 GB (4.41 bits/sample, ratio 14.5); real time is 86,400 s per day, so encode runs at 661x real time for 1000 tags.

encode GB/s per tenth of the phase: 5.7 5.6 5.2 5.3 5.4 5.0 5.2 5.1 5.1 5.1
decode GB/s per tenth of the phase: 12.6 12.5 12.9 12.5 12.9 12.9 12.9 13.1 12.4 12.9

### Per clock kind: `--scale 0.25 --times <clock>` (same session)

| clock | phase | wall | float64 GB/s | samples/s | µs/block/worker | in-call % |
|---|---|---|---|---|---|---|
| none | encode | 24.9 s | 6.93 GB/s | 866 M | 4.62 | 99% |
| none | decode | 9.4 s | 18.43 GB/s | 2304 M | 1.74 | 99% |
| grid | encode | 27.9 s | 6.20 GB/s | 774 M | 5.16 | 99% |
| grid | decode | 12.1 s | 14.28 GB/s | 1785 M | 2.24 | 99% |
| grid+gaps | encode | 28.9 s | 5.98 GB/s | 747 M | 5.35 | 99% |
| grid+gaps | decode | 12.7 s | 13.57 GB/s | 1696 M | 2.36 | 99% |
| noisy | encode | 56.7 s | 3.05 GB/s | 381 M | 10.49 | 99% |
| noisy | decode | 24.7 s | 7.00 GB/s | 875 M | 4.57 | 100% |

- none 9.91 GB (3.67 bits/sample, ratio 17.4);
- grid 9.92 GB (3.68 bits/sample, ratio 17.4);
- grid+gaps 9.95 GB (3.68 bits/sample, ratio 17.4);
- noisy 29.16 GB (10.80 bits/sample, ratio 5.9);

### Time-axis units at effort 5 (block flushes), with and without the Rust extension

`--scale 0.25 --times clock-mix --effort 5`, 4 threads, same session. Units with a time axis now go through
the extension's `pack_unit` (the body is built in Python, the layout choice, flush points and zstd frame
without the GIL), so the flushes scale: the units are the same bytes (11.65 GB, 4.32 bits/sample).

| path | encode | float64 GB/s | µs/block/worker | decode |
|---|---|---|---|---|
| Python | 51.3 s | 3.37 GB/s | 9.50 | 13.4 s |
| Rust extension | 33.4 s | 5.17 GB/s | 6.19 | 13.4 s |
