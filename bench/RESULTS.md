# fluxcode gigabyte benchmark (bench/bench_gb.py, 2026-10-02, AC power; decode is per unit, as rows are read)

One complete run at 5ee16fa, replacing the 2026-10-01 tables (which predate `Params.effort`). The default
(effort 4) is the encoder as it was at 8bfc6c6: same bytes (4.76 bits/sample, ratio 13.4), encode 5.13 µs/block
(5.13 before), decode 1.79 (1.80). Sections `effort 1`, `effort 5` and `effort 9` replace "bit planes only":
effort 1 (heuristic layout, zstd level 1) encodes 32% faster (3.47 µs/block, -1.66 µs) for 1.5% more bytes
(4.83 bits/sample); effort 5 (both layouts, block flushes) is 2.1% smaller (4.66) at +19% encode time (6.09,
+0.96 µs); effort 9 (also zstd 9) 3.0% smaller (4.62) at +383% (24.80, +19.67 µs). Decode is the same
within noise at every effort. Threads: 4 threads scale 3.3x (the kernels release the GIL), 8 threads 3.4x.
Python path only; effort 5 and up scale better across threads with the optional Rust extension
(experimental/rust_port/results/).

Apple M3, Darwin 25.6.0, Python 3.11.13, numpy 2.4.6

15 signal types x 149 minutes = 1.00 GiB of float64 (134,100 blocks); best of 3.

(generated in 5 s)

## default

| signal | bits/sample | ratio | worst max err (% range) | encode MB/s | decode MB/s | encode µs/block | decode µs/block |
|---|---|---|---|---|---|---|---|
| linear | 0.02 | 3038.0 | 0.0000 | 2872 | 5870 | 2.79 | 1.36 |
| quadratic | 0.25 | 257.1 | 0.0015 | 2035 | 4619 | 3.93 | 1.73 |
| sin-4.12hz | 2.40 | 26.7 | 0.0010 | 1315 | 3752 | 6.08 | 2.13 |
| sin-9.87hz | 3.42 | 18.7 | 0.0010 | 1060 | 3510 | 7.55 | 2.28 |
| sin-50.3hz | 6.74 | 9.5 | 0.0010 | 1330 | 3735 | 6.01 | 2.14 |
| gauss-spikes | 6.63 | 9.7 | 1.4661 | 1577 | 4264 | 5.07 | 1.88 |
| impulses | 4.76 | 13.5 | 1.2592 | 1951 | 6641 | 4.10 | 1.20 |
| square-2.24hz | 0.04 | 1438.2 | 0.0000 | 2728 | 6794 | 2.93 | 1.18 |
| random-walk | 12.62 | 5.1 | 0.0015 | 1430 | 4379 | 5.59 | 1.83 |
| chirp | 7.81 | 8.2 | 0.0010 | 1071 | 3925 | 7.47 | 2.04 |
| noisy-sine | 5.24 | 12.2 | 0.2334 | 1880 | 4955 | 4.26 | 1.61 |
| sensor-0.1 | 1.78 | 36.0 | 0.0000 | 1275 | 3539 | 6.27 | 2.26 |
| random-walk q0.01 | 9.28 | 6.9 | 0.0000 | 1253 | 4166 | 6.38 | 1.92 |
| noisy-sine q0.1 | 5.24 | 12.2 | 0.2341 | 1878 | 4934 | 4.26 | 1.62 |
| noisy-sine q0.01 via float32 | 5.24 | 12.2 | 0.2341 | 1878 | 4923 | 4.26 | 1.62 |
| **all (1.00 GiB)** | 4.76 | 13.4 | | 1559 | 4475 | 5.13 | 1.79 |

## noise floor off

| signal | bits/sample | ratio | worst max err (% range) | encode MB/s | decode MB/s | encode µs/block | decode µs/block |
|---|---|---|---|---|---|---|---|
| linear | 0.02 | 3038.0 | 0.0000 | 3745 | 5812 | 2.14 | 1.38 |
| quadratic | 0.25 | 257.1 | 0.0015 | 2608 | 4627 | 3.07 | 1.73 |
| sin-4.12hz | 2.40 | 26.7 | 0.0010 | 1532 | 3748 | 5.22 | 2.13 |
| sin-9.87hz | 3.42 | 18.7 | 0.0010 | 1197 | 3519 | 6.68 | 2.27 |
| sin-50.3hz | 6.74 | 9.5 | 0.0010 | 1550 | 3726 | 5.16 | 2.15 |
| gauss-spikes | 12.06 | 5.3 | 0.0015 | 1456 | 4847 | 5.49 | 1.65 |
| impulses | 11.69 | 5.5 | 0.0015 | 1794 | 6237 | 4.46 | 1.28 |
| square-2.24hz | 0.04 | 1438.2 | 0.0000 | 3480 | 6825 | 2.30 | 1.17 |
| random-walk | 12.62 | 5.1 | 0.0015 | 1689 | 4382 | 4.74 | 1.83 |
| chirp | 7.81 | 8.2 | 0.0010 | 1208 | 3924 | 6.62 | 2.04 |
| noisy-sine | 13.72 | 4.7 | 0.0009 | 1742 | 4158 | 4.59 | 1.92 |
| sensor-0.1 | 1.78 | 36.0 | 0.0000 | 1479 | 3536 | 5.41 | 2.26 |
| random-walk q0.01 | 9.28 | 6.9 | 0.0000 | 1454 | 4205 | 5.50 | 1.90 |
| noisy-sine q0.1 | 8.89 | 7.2 | 0.0000 | 1518 | 3137 | 5.27 | 2.55 |
| noisy-sine q0.01 via float32 | 12.32 | 5.2 | 0.0000 | 1661 | 4411 | 4.82 | 1.81 |
| **all (1.00 GiB)** | 6.87 | 9.3 | | 1679 | 4273 | 4.76 | 1.87 |

## target 6

| signal | bits/sample | ratio | worst max err (% range) | encode MB/s | decode MB/s | encode µs/block | decode µs/block |
|---|---|---|---|---|---|---|---|
| linear | 0.02 | 3038.0 | 0.0000 | 2272 | 5758 | 3.52 | 1.39 |
| quadratic | 0.25 | 257.1 | 0.0015 | 1715 | 4619 | 4.66 | 1.73 |
| sin-4.12hz | 2.40 | 26.7 | 0.0010 | 1185 | 3754 | 6.75 | 2.13 |
| sin-9.87hz | 3.42 | 18.7 | 0.0010 | 974 | 3512 | 8.21 | 2.28 |
| sin-50.3hz | 0.92 | 69.7 | 0.0625 | 1190 | 5407 | 6.72 | 1.48 |
| gauss-spikes | 5.27 | 12.1 | 1.4661 | 978 | 3671 | 8.18 | 2.18 |
| impulses | 4.76 | 13.5 | 1.2592 | 1676 | 6654 | 4.77 | 1.20 |
| square-2.24hz | 0.04 | 1438.2 | 0.0000 | 2169 | 6831 | 3.69 | 1.17 |
| random-walk | 5.39 | 11.9 | 0.1951 | 1202 | 4726 | 6.66 | 1.69 |
| chirp | 4.68 | 13.7 | 0.0625 | 872 | 3297 | 9.18 | 2.43 |
| noisy-sine | 5.24 | 12.2 | 0.2334 | 1628 | 4929 | 4.91 | 1.62 |
| sensor-0.1 | 1.78 | 36.0 | 0.0000 | 1156 | 3524 | 6.92 | 2.27 |
| random-walk q0.01 | 6.21 | 10.3 | 0.1750 | 1150 | 4449 | 6.96 | 1.80 |
| noisy-sine q0.1 | 5.24 | 12.2 | 0.2341 | 1628 | 4941 | 4.91 | 1.62 |
| noisy-sine q0.01 via float32 | 5.24 | 12.2 | 0.2341 | 1628 | 4946 | 4.91 | 1.62 |
| **all (1.00 GiB)** | 3.39 | 18.9 | | 1319 | 4510 | 6.06 | 1.77 |

## effort 1

| signal | bits/sample | ratio | worst max err (% range) | encode MB/s | decode MB/s | encode µs/block | decode µs/block |
|---|---|---|---|---|---|---|---|
| linear | 0.05 | 1297.3 | 0.0000 | 3566 | 5169 | 2.24 | 1.55 |
| quadratic | 0.25 | 252.6 | 0.0015 | 2865 | 4620 | 2.79 | 1.73 |
| sin-4.12hz | 2.50 | 25.6 | 0.0010 | 1792 | 3757 | 4.46 | 2.13 |
| sin-9.87hz | 3.65 | 17.6 | 0.0010 | 1864 | 3891 | 4.29 | 2.06 |
| sin-50.3hz | 6.88 | 9.3 | 0.0010 | 2023 | 3723 | 3.96 | 2.15 |
| gauss-spikes | 6.61 | 9.7 | 1.4661 | 2163 | 3884 | 3.70 | 2.06 |
| impulses | 4.74 | 13.5 | 1.2592 | 2819 | 6159 | 2.84 | 1.30 |
| square-2.24hz | 0.06 | 999.0 | 0.0000 | 3443 | 5438 | 2.32 | 1.47 |
| random-walk | 12.66 | 5.1 | 0.0015 | 1967 | 4349 | 4.07 | 1.84 |
| chirp | 7.90 | 8.1 | 0.0010 | 1824 | 4028 | 4.39 | 1.99 |
| noisy-sine | 5.24 | 12.2 | 0.2334 | 2881 | 4537 | 2.78 | 1.76 |
| sensor-0.1 | 2.18 | 29.4 | 0.0000 | 1764 | 3062 | 4.54 | 2.61 |
| random-walk q0.01 | 9.30 | 6.9 | 0.0000 | 1963 | 4062 | 4.08 | 1.97 |
| noisy-sine q0.1 | 5.24 | 12.2 | 0.2341 | 2880 | 4542 | 2.78 | 1.76 |
| noisy-sine q0.01 via float32 | 5.25 | 12.2 | 0.2341 | 2874 | 4507 | 2.78 | 1.77 |
| **all (1.00 GiB)** | 4.83 | 13.2 | | 2307 | 4263 | 3.47 | 1.88 |

## effort 5

| signal | bits/sample | ratio | worst max err (% range) | encode MB/s | decode MB/s | encode µs/block | decode µs/block |
|---|---|---|---|---|---|---|---|
| linear | 0.02 | 2981.4 | 0.0000 | 2449 | 7340 | 3.27 | 1.09 |
| quadratic | 0.27 | 240.7 | 0.0015 | 1749 | 4598 | 4.57 | 1.74 |
| sin-4.12hz | 2.38 | 26.9 | 0.0010 | 1168 | 3733 | 6.85 | 2.14 |
| sin-9.87hz | 3.40 | 18.8 | 0.0010 | 935 | 3513 | 8.56 | 2.28 |
| sin-50.3hz | 6.22 | 10.3 | 0.0010 | 1017 | 3068 | 7.87 | 2.61 |
| gauss-spikes | 6.48 | 9.9 | 1.4661 | 1260 | 3942 | 6.35 | 2.03 |
| impulses | 4.72 | 13.5 | 1.2592 | 1664 | 6636 | 4.81 | 1.21 |
| square-2.24hz | 0.05 | 1366.4 | 0.0000 | 2439 | 6520 | 3.28 | 1.23 |
| random-walk | 12.29 | 5.2 | 0.0015 | 1047 | 3865 | 7.64 | 2.07 |
| chirp | 7.76 | 8.2 | 0.0010 | 859 | 3550 | 9.31 | 2.25 |
| noisy-sine | 5.22 | 12.3 | 0.2334 | 1559 | 4887 | 5.13 | 1.64 |
| sensor-0.1 | 1.76 | 36.3 | 0.0000 | 1168 | 3492 | 6.85 | 2.29 |
| random-walk q0.01 | 8.91 | 7.2 | 0.0000 | 1215 | 3825 | 6.58 | 2.09 |
| noisy-sine q0.1 | 5.22 | 12.3 | 0.2341 | 1554 | 4851 | 5.15 | 1.65 |
| noisy-sine q0.01 via float32 | 5.22 | 12.3 | 0.2341 | 1556 | 4862 | 5.14 | 1.65 |
| **all (1.00 GiB)** | 4.66 | 13.7 | | 1314 | 4292 | 6.09 | 1.86 |

## effort 9

| signal | bits/sample | ratio | worst max err (% range) | encode MB/s | decode MB/s | encode µs/block | decode µs/block |
|---|---|---|---|---|---|---|---|
| linear | 0.02 | 3038.0 | 0.0000 | 1833 | 7400 | 4.36 | 1.08 |
| quadratic | 0.25 | 261.2 | 0.0015 | 536 | 3774 | 14.92 | 2.12 |
| sin-4.12hz | 2.24 | 28.5 | 0.0010 | 206 | 3689 | 38.75 | 2.17 |
| sin-9.87hz | 3.26 | 19.7 | 0.0010 | 197 | 3337 | 40.64 | 2.40 |
| sin-50.3hz | 6.09 | 10.5 | 0.0010 | 248 | 3190 | 32.30 | 2.51 |
| gauss-spikes | 6.43 | 10.0 | 1.4661 | 393 | 3996 | 20.35 | 2.00 |
| impulses | 4.72 | 13.5 | 1.2592 | 405 | 6652 | 19.77 | 1.20 |
| square-2.24hz | 0.04 | 1685.4 | 0.0000 | 1559 | 4938 | 5.13 | 1.62 |
| random-walk | 12.26 | 5.2 | 0.0015 | 277 | 3881 | 28.87 | 2.06 |
| chirp | 7.69 | 8.3 | 0.0010 | 227 | 3582 | 35.20 | 2.23 |
| noisy-sine | 5.22 | 12.3 | 0.2334 | 438 | 4891 | 18.27 | 1.64 |
| sensor-0.1 | 1.73 | 37.0 | 0.0000 | 205 | 3406 | 39.06 | 2.35 |
| random-walk q0.01 | 8.89 | 7.2 | 0.0000 | 212 | 3814 | 37.67 | 2.10 |
| noisy-sine q0.1 | 5.22 | 12.3 | 0.2341 | 438 | 4860 | 18.27 | 1.65 |
| noisy-sine q0.01 via float32 | 5.22 | 12.3 | 0.2341 | 435 | 4860 | 18.40 | 1.65 |
| **all (1.00 GiB)** | 4.62 | 13.9 | | 323 | 4171 | 24.80 | 1.92 |

## Threads (default params; one signal type per task; the kernels release the GIL)

| threads | encode MB/s | decode MB/s | encode speedup |
|---|---|---|---|
| 1 | 1555 | 4291 | 1.0x |
| 4 | 5130 | 14959 | 3.3x |
| 8 | 5238 | 12571 | 3.4x |
