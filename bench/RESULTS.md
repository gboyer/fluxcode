# fluxcode gigabyte benchmark (bench/bench_gb.py, 2026-10-07, AC power; decode is per block group, as rows are read)

One complete run at 4e89772 (the noise floor's cadence from the median of 15 intervals, all or nothing; σ the
windows' mean unless the quietest is below 0.75 of it), replacing the tables at 0581c29. The data have no times, so
only the window rule could change anything, and the block groups are the same size (the default 4.76 bits/sample,
ratio 13.4; Gaussian spikes still 6.60). Encode µs/block against 0581c29: default 5.03 → 5.01, effort 1 3.41 → 3.40,
effort 5 5.78 → 5.77, effort 9 22.67 → 22.68, noise floor off 4.59 → 4.60, all within run-to-run noise; decode
1.81 → 1.76 µs/block for the default (the load average was 1.9–3.9 from other apps). Threads: 4 threads scale 3.3x,
8 threads 3.4x. Python path only (FLUXCODE_RUST=0).

Apple M3, Darwin 25.6.0, Python 3.11.13, numpy 2.4.6

15 signal types x 149 minutes = 1.00 GiB of float64 (134,100 blocks); best of 3.

(generated in 5 s)

## default

| signal | bits/sample | ratio | worst max err (% range) | encode MB/s | decode MB/s | encode µs/block | decode µs/block |
|---|---|---|---|---|---|---|---|
| linear | 0.02 | 3038.0 | 0.0000 | 3370 | 5818 | 2.37 | 1.38 |
| quadratic | 0.25 | 257.1 | 0.0015 | 2075 | 4716 | 3.85 | 1.70 |
| sin-4.12hz | 2.40 | 26.7 | 0.0010 | 1318 | 3801 | 6.07 | 2.10 |
| sin-9.87hz | 3.42 | 18.7 | 0.0010 | 1069 | 3540 | 7.48 | 2.26 |
| sin-50.3hz | 6.74 | 9.5 | 0.0010 | 1350 | 3735 | 5.93 | 2.14 |
| gauss-spikes | 6.60 | 9.7 | 1.4661 | 1605 | 4277 | 4.98 | 1.87 |
| impulses | 4.76 | 13.5 | 1.2592 | 1982 | 6803 | 4.04 | 1.18 |
| square-2.24hz | 0.04 | 1438.2 | 0.0000 | 3181 | 7033 | 2.51 | 1.14 |
| random-walk | 12.62 | 5.1 | 0.0015 | 1456 | 4386 | 5.50 | 1.82 |
| chirp | 7.81 | 8.2 | 0.0010 | 1080 | 3917 | 7.41 | 2.04 |
| noisy-sine | 5.24 | 12.2 | 0.2334 | 1911 | 5010 | 4.19 | 1.60 |
| sensor-0.1 | 1.78 | 36.0 | 0.0000 | 1292 | 3754 | 6.19 | 2.13 |
| random-walk q0.01 | 9.28 | 6.9 | 0.0000 | 1273 | 4260 | 6.29 | 1.88 |
| noisy-sine q0.1 | 5.24 | 12.2 | 0.2341 | 1910 | 5037 | 4.19 | 1.59 |
| noisy-sine q0.01 via float32 | 5.24 | 12.2 | 0.2341 | 1910 | 5000 | 4.19 | 1.60 |
| **all (1.00 GiB)** | 4.76 | 13.4 | | 1596 | 4542 | 5.01 | 1.76 |

## noise floor off

| signal | bits/sample | ratio | worst max err (% range) | encode MB/s | decode MB/s | encode µs/block | decode µs/block |
|---|---|---|---|---|---|---|---|
| linear | 0.02 | 3038.0 | 0.0000 | 4041 | 5981 | 1.98 | 1.34 |
| quadratic | 0.25 | 257.1 | 0.0015 | 2751 | 4671 | 2.91 | 1.71 |
| sin-4.12hz | 2.40 | 26.7 | 0.0010 | 1573 | 3737 | 5.09 | 2.14 |
| sin-9.87hz | 3.42 | 18.7 | 0.0010 | 1225 | 3490 | 6.53 | 2.29 |
| sin-50.3hz | 6.74 | 9.5 | 0.0010 | 1606 | 3714 | 4.98 | 2.15 |
| gauss-spikes | 12.06 | 5.3 | 0.0015 | 1505 | 4801 | 5.32 | 1.67 |
| impulses | 11.69 | 5.5 | 0.0015 | 1865 | 6216 | 4.29 | 1.29 |
| square-2.24hz | 0.04 | 1438.2 | 0.0000 | 3726 | 6946 | 2.15 | 1.15 |
| random-walk | 12.62 | 5.1 | 0.0015 | 1754 | 4356 | 4.56 | 1.84 |
| chirp | 7.81 | 8.2 | 0.0010 | 1244 | 3897 | 6.43 | 2.05 |
| noisy-sine | 13.72 | 4.7 | 0.0009 | 1813 | 4184 | 4.41 | 1.91 |
| sensor-0.1 | 1.78 | 36.0 | 0.0000 | 1528 | 3629 | 5.23 | 2.20 |
| random-walk q0.01 | 9.28 | 6.9 | 0.0000 | 1501 | 4190 | 5.33 | 1.91 |
| noisy-sine q0.1 | 8.89 | 7.2 | 0.0000 | 1574 | 3173 | 5.08 | 2.52 |
| noisy-sine q0.01 via float32 | 12.32 | 5.2 | 0.0000 | 1722 | 4387 | 4.65 | 1.82 |
| **all (1.00 GiB)** | 6.87 | 9.3 | | 1741 | 4285 | 4.60 | 1.87 |

## target 6

| signal | bits/sample | ratio | worst max err (% range) | encode MB/s | decode MB/s | encode µs/block | decode µs/block |
|---|---|---|---|---|---|---|---|
| linear | 0.02 | 3038.0 | 0.0000 | 2569 | 5973 | 3.11 | 1.34 |
| quadratic | 0.25 | 257.1 | 0.0015 | 1740 | 4615 | 4.60 | 1.73 |
| sin-4.12hz | 2.40 | 26.7 | 0.0010 | 1196 | 3715 | 6.69 | 2.15 |
| sin-9.87hz | 3.42 | 18.7 | 0.0010 | 982 | 3497 | 8.15 | 2.29 |
| sin-50.3hz | 0.92 | 69.7 | 0.0625 | 1204 | 5349 | 6.65 | 1.50 |
| gauss-spikes | 5.28 | 12.1 | 1.4661 | 996 | 3687 | 8.03 | 2.17 |
| impulses | 4.76 | 13.5 | 1.2592 | 1702 | 6603 | 4.70 | 1.21 |
| square-2.24hz | 0.04 | 1438.2 | 0.0000 | 2453 | 6928 | 3.26 | 1.15 |
| random-walk | 5.39 | 11.9 | 0.1951 | 1218 | 4831 | 6.57 | 1.66 |
| chirp | 4.68 | 13.7 | 0.0625 | 882 | 3299 | 9.07 | 2.42 |
| noisy-sine | 5.24 | 12.2 | 0.2334 | 1651 | 4907 | 4.84 | 1.63 |
| sensor-0.1 | 1.78 | 36.0 | 0.0000 | 1167 | 3656 | 6.85 | 2.19 |
| random-walk q0.01 | 6.21 | 10.3 | 0.1750 | 1165 | 4433 | 6.87 | 1.80 |
| noisy-sine q0.1 | 5.24 | 12.2 | 0.2341 | 1649 | 4913 | 4.85 | 1.63 |
| noisy-sine q0.01 via float32 | 5.24 | 12.2 | 0.2341 | 1653 | 4893 | 4.84 | 1.63 |
| **all (1.00 GiB)** | 3.39 | 18.9 | | 1347 | 4526 | 5.94 | 1.77 |

## effort 1

| signal | bits/sample | ratio | worst max err (% range) | encode MB/s | decode MB/s | encode µs/block | decode µs/block |
|---|---|---|---|---|---|---|---|
| linear | 0.05 | 1297.3 | 0.0000 | 4089 | 5803 | 1.96 | 1.38 |
| quadratic | 0.25 | 252.6 | 0.0015 | 2974 | 4621 | 2.69 | 1.73 |
| sin-4.12hz | 2.50 | 25.6 | 0.0010 | 1774 | 3508 | 4.51 | 2.28 |
| sin-9.87hz | 3.65 | 17.6 | 0.0010 | 1843 | 3869 | 4.34 | 2.07 |
| sin-50.3hz | 6.88 | 9.3 | 0.0010 | 2076 | 3720 | 3.85 | 2.15 |
| gauss-spikes | 6.59 | 9.7 | 1.4661 | 2232 | 3944 | 3.58 | 2.03 |
| impulses | 4.74 | 13.5 | 1.2592 | 2768 | 6192 | 2.89 | 1.29 |
| square-2.24hz | 0.06 | 999.0 | 0.0000 | 3982 | 5661 | 2.01 | 1.41 |
| random-walk | 12.66 | 5.1 | 0.0015 | 2030 | 4375 | 3.94 | 1.83 |
| chirp | 7.90 | 8.1 | 0.0010 | 1874 | 4035 | 4.27 | 1.98 |
| noisy-sine | 5.24 | 12.2 | 0.2334 | 2838 | 4549 | 2.82 | 1.76 |
| sensor-0.1 | 2.18 | 29.4 | 0.0000 | 1747 | 3108 | 4.58 | 2.57 |
| random-walk q0.01 | 9.30 | 6.9 | 0.0000 | 2021 | 4087 | 3.96 | 1.96 |
| noisy-sine q0.1 | 5.24 | 12.2 | 0.2341 | 2848 | 4534 | 2.81 | 1.76 |
| noisy-sine q0.01 via float32 | 5.25 | 12.2 | 0.2341 | 2837 | 4528 | 2.82 | 1.77 |
| **all (1.00 GiB)** | 4.83 | 13.2 | | 2352 | 4290 | 3.40 | 1.86 |

## effort 5

| signal | bits/sample | ratio | worst max err (% range) | encode MB/s | decode MB/s | encode µs/block | decode µs/block |
|---|---|---|---|---|---|---|---|
| linear | 0.02 | 2981.4 | 0.0000 | 2799 | 7356 | 2.86 | 1.09 |
| quadratic | 0.27 | 240.7 | 0.0015 | 1845 | 4575 | 4.34 | 1.75 |
| sin-4.12hz | 2.38 | 26.9 | 0.0010 | 1253 | 3715 | 6.38 | 2.15 |
| sin-9.87hz | 3.40 | 18.8 | 0.0010 | 942 | 3488 | 8.49 | 2.29 |
| sin-50.3hz | 6.22 | 10.3 | 0.0010 | 1108 | 3037 | 7.22 | 2.63 |
| gauss-spikes | 6.46 | 9.9 | 1.4661 | 1325 | 3874 | 6.04 | 2.07 |
| impulses | 4.72 | 13.5 | 1.2592 | 1802 | 6576 | 4.44 | 1.22 |
| square-2.24hz | 0.05 | 1366.4 | 0.0000 | 2768 | 6524 | 2.89 | 1.23 |
| random-walk | 12.29 | 5.2 | 0.0015 | 1054 | 3885 | 7.59 | 2.06 |
| chirp | 7.76 | 8.2 | 0.0010 | 969 | 3533 | 8.26 | 2.26 |
| noisy-sine | 5.22 | 12.3 | 0.2334 | 1632 | 4856 | 4.90 | 1.65 |
| sensor-0.1 | 1.76 | 36.3 | 0.0000 | 1178 | 3517 | 6.79 | 2.27 |
| random-walk q0.01 | 8.91 | 7.2 | 0.0000 | 1228 | 3774 | 6.51 | 2.12 |
| noisy-sine q0.1 | 5.22 | 12.3 | 0.2341 | 1633 | 4821 | 4.90 | 1.66 |
| noisy-sine q0.01 via float32 | 5.22 | 12.3 | 0.2341 | 1633 | 4846 | 4.90 | 1.65 |
| **all (1.00 GiB)** | 4.66 | 13.7 | | 1387 | 4271 | 5.77 | 1.87 |

## effort 9

| signal | bits/sample | ratio | worst max err (% range) | encode MB/s | decode MB/s | encode µs/block | decode µs/block |
|---|---|---|---|---|---|---|---|
| linear | 0.02 | 3038.0 | 0.0000 | 2000 | 7313 | 4.00 | 1.09 |
| quadratic | 0.25 | 261.2 | 0.0015 | 559 | 3798 | 14.31 | 2.11 |
| sin-4.12hz | 2.24 | 28.5 | 0.0010 | 241 | 3674 | 33.22 | 2.18 |
| sin-9.87hz | 3.26 | 19.7 | 0.0010 | 217 | 3331 | 36.87 | 2.40 |
| sin-50.3hz | 6.09 | 10.5 | 0.0010 | 294 | 3154 | 27.17 | 2.54 |
| gauss-spikes | 6.41 | 10.0 | 1.4661 | 435 | 3949 | 18.38 | 2.03 |
| impulses | 4.72 | 13.5 | 1.2592 | 514 | 6609 | 15.57 | 1.21 |
| square-2.24hz | 0.04 | 1685.4 | 0.0000 | 1681 | 4829 | 4.76 | 1.66 |
| random-walk | 12.26 | 5.2 | 0.0015 | 276 | 3871 | 28.94 | 2.07 |
| chirp | 7.69 | 8.3 | 0.0010 | 276 | 3566 | 28.96 | 2.24 |
| noisy-sine | 5.22 | 12.3 | 0.2334 | 466 | 4837 | 17.15 | 1.65 |
| sensor-0.1 | 1.73 | 37.0 | 0.0000 | 206 | 3643 | 38.75 | 2.20 |
| random-walk q0.01 | 8.89 | 7.2 | 0.0000 | 213 | 3813 | 37.64 | 2.10 |
| noisy-sine q0.1 | 5.22 | 12.3 | 0.2341 | 466 | 4836 | 17.17 | 1.65 |
| noisy-sine q0.01 via float32 | 5.22 | 12.3 | 0.2341 | 463 | 4845 | 17.27 | 1.65 |
| **all (1.00 GiB)** | 4.62 | 13.9 | | 353 | 4171 | 22.68 | 1.92 |

## Threads (default params; one signal type per task; the kernels release the GIL)

| threads | encode MB/s | decode MB/s | encode speedup |
|---|---|---|---|
| 1 | 1590 | 3686 | 1.0x |
| 4 | 5287 | 14568 | 3.3x |
| 8 | 5424 | 12767 | 3.4x |
