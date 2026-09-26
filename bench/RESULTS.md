# fluxcode gigabyte benchmark (bench/bench_gb.py, 2026-09-25, AC power, idle, quiet rerun; decode is per unit, as rows are read)

Apple M3, Darwin 25.6.0, Python 3.11.13, numpy 2.4.6

Fastest of 5 complete runs (their overall speed agrees within 3%, single cells within 15%). Before the run: load average 1.3, no other process above 1% CPU.

15 signal types x 149 minutes = 1.00 GiB of float64 (134,100 blocks); best of 3.

(generated in 5 s)

## default

| signal | bits/sample | ratio | worst max err (% range) | encode MB/s | decode MB/s | encode µs/block | decode µs/block |
|---|---|---|---|---|---|---|---|
| linear | 0.02 | 2981.4 | 0.0000 | 3538 | 5287 | 2.26 | 1.51 |
| quadratic | 0.25 | 258.8 | 0.0015 | 3368 | 5042 | 2.38 | 1.59 |
| sin-4.12hz | 2.78 | 23.0 | 0.0010 | 2716 | 4185 | 2.95 | 1.91 |
| sin-9.87hz | 3.69 | 17.4 | 0.0010 | 2356 | 3628 | 3.40 | 2.20 |
| sin-50.3hz | 6.73 | 9.5 | 0.0010 | 2315 | 4079 | 3.46 | 1.96 |
| gauss-spikes | 6.68 | 9.6 | 1.4667 | 2315 | 4759 | 3.46 | 1.68 |
| impulses | 5.10 | 12.6 | 1.2602 | 2884 | 5020 | 2.77 | 1.59 |
| square-2.24hz | 0.05 | 1201.5 | 0.0000 | 3367 | 4985 | 2.38 | 1.60 |
| random-walk | 12.66 | 5.1 | 0.0015 | 2157 | 4813 | 3.71 | 1.66 |
| chirp | 7.84 | 8.2 | 0.0010 | 1726 | 4295 | 4.64 | 1.86 |
| noisy-sine | 5.49 | 11.7 | 0.2333 | 2604 | 3672 | 3.07 | 2.18 |
| sensor-0.1 | 1.78 | 36.0 | 0.0000 | 2301 | 4021 | 3.48 | 1.99 |
| random-walk q0.01 | 9.28 | 6.9 | 0.0000 | 2245 | 4585 | 3.56 | 1.74 |
| noisy-sine q0.1 | 5.46 | 11.7 | 0.2341 | 2552 | 3670 | 3.14 | 2.18 |
| noisy-sine q0.01 via float32 | 5.46 | 11.7 | 0.2341 | 2689 | 3673 | 2.98 | 2.18 |
| **all (1.00 GiB)** | 4.88 | 13.1 | | 2521 | 4308 | 3.17 | 1.86 |

## noise floor off

| signal | bits/sample | ratio | worst max err (% range) | encode MB/s | decode MB/s | encode µs/block | decode µs/block |
|---|---|---|---|---|---|---|---|
| linear | 0.02 | 2981.4 | 0.0000 | 4908 | 5384 | 1.63 | 1.49 |
| quadratic | 0.25 | 258.8 | 0.0015 | 5244 | 5055 | 1.53 | 1.58 |
| sin-4.12hz | 2.78 | 23.0 | 0.0010 | 3808 | 4204 | 2.10 | 1.90 |
| sin-9.87hz | 3.69 | 17.4 | 0.0010 | 3118 | 3613 | 2.57 | 2.21 |
| sin-50.3hz | 6.73 | 9.5 | 0.0010 | 3051 | 4070 | 2.62 | 1.97 |
| gauss-spikes | 12.10 | 5.3 | 0.0015 | 2629 | 5278 | 3.04 | 1.52 |
| impulses | 11.72 | 5.5 | 0.0015 | 3011 | 6905 | 2.66 | 1.16 |
| square-2.24hz | 0.05 | 1201.5 | 0.0000 | 4523 | 4882 | 1.77 | 1.64 |
| random-walk | 12.66 | 5.1 | 0.0015 | 2728 | 4796 | 2.93 | 1.67 |
| chirp | 7.84 | 8.2 | 0.0010 | 2061 | 4293 | 3.88 | 1.86 |
| noisy-sine | 13.76 | 4.7 | 0.0009 | 2776 | 4453 | 2.88 | 1.80 |
| sensor-0.1 | 1.78 | 36.0 | 0.0000 | 2994 | 3995 | 2.67 | 2.00 |
| random-walk q0.01 | 9.28 | 6.9 | 0.0000 | 2814 | 4612 | 2.84 | 1.73 |
| noisy-sine q0.1 | 8.89 | 7.2 | 0.0000 | 3001 | 3416 | 2.67 | 2.34 |
| noisy-sine q0.01 via float32 | 12.35 | 5.2 | 0.0000 | 2635 | 4830 | 3.04 | 1.66 |
| **all (1.00 GiB)** | 6.93 | 9.2 | | 3091 | 4524 | 2.59 | 1.77 |

## target 6

| signal | bits/sample | ratio | worst max err (% range) | encode MB/s | decode MB/s | encode µs/block | decode µs/block |
|---|---|---|---|---|---|---|---|
| linear | 0.02 | 2981.4 | 0.0000 | 2640 | 5301 | 3.03 | 1.51 |
| quadratic | 0.25 | 258.8 | 0.0015 | 2567 | 5034 | 3.12 | 1.59 |
| sin-4.12hz | 2.78 | 23.0 | 0.0010 | 2185 | 4172 | 3.66 | 1.92 |
| sin-9.87hz | 3.69 | 17.4 | 0.0010 | 1932 | 3609 | 4.14 | 2.22 |
| sin-50.3hz | 4.09 | 15.6 | 0.0625 | 1450 | 3830 | 5.52 | 2.09 |
| gauss-spikes | 5.35 | 12.0 | 1.4667 | 1429 | 3998 | 5.60 | 2.00 |
| impulses | 5.10 | 12.6 | 1.2602 | 2282 | 4994 | 3.51 | 1.60 |
| square-2.24hz | 0.05 | 1201.5 | 0.0000 | 2515 | 4929 | 3.18 | 1.62 |
| random-walk | 5.54 | 11.5 | 0.1952 | 1461 | 3646 | 5.48 | 2.19 |
| chirp | 4.72 | 13.6 | 0.0625 | 1380 | 3670 | 5.80 | 2.18 |
| noisy-sine | 5.49 | 11.7 | 0.2333 | 2124 | 3688 | 3.77 | 2.17 |
| sensor-0.1 | 1.78 | 36.0 | 0.0000 | 1917 | 4023 | 4.17 | 1.99 |
| random-walk q0.01 | 6.50 | 9.8 | 0.1750 | 1430 | 3586 | 5.59 | 2.23 |
| noisy-sine q0.1 | 5.46 | 11.7 | 0.2341 | 2118 | 3673 | 3.78 | 2.18 |
| noisy-sine q0.01 via float32 | 5.46 | 11.7 | 0.2341 | 2117 | 3658 | 3.78 | 2.19 |
| **all (1.00 GiB)** | 3.75 | 17.1 | | 1872 | 4044 | 4.27 | 1.98 |

## Threads (default params; one signal type per task; the kernels release the GIL)

| threads | encode MB/s | decode MB/s | encode speedup |
|---|---|---|---|
| 1 | 2458 | 4261 | 1.0x |
| 4 | 8425 | 13339 | 3.4x |
| 8 | 10043 | 15923 | 4.1x |
