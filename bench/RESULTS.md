# fluxcode gigabyte benchmark (bench/bench_gb.py, 2026-10-02, AC power; decode is per unit, as rows are read)

One complete run at d039350, replacing the tables at 5ee16fa. The units are byte for byte the same (the default
still 4.76 bits/sample, ratio 13.4; every section's size is unchanged), and encoding is a little faster because
the Python path now writes the byte-plane body once and derives the bit planes with one transpose of the
residual region, and drops a candidate frame as soon as its flushed blocks show it can't win (as the Rust
path does). Encode µs/block against 5ee16fa: default 5.13 → 4.98 (-3%), effort 1 3.47 → 3.35 (-3%), effort 5
6.09 → 5.76 (-5%), effort 9 24.80 → 22.67 (-9%); the other sections -3 to -5%. Decode is the same within
noise at every effort (1.79 → 1.76 for the default). Threads: 4 threads scale 3.3x (the kernels release the
GIL), 8 threads 3.4x. Python path only (FLUXCODE_RUST=0); effort 5 and up scale better across threads with
the optional Rust extension (experimental/rust_port/results/).

Apple M3, Darwin 25.6.0, Python 3.11.13, numpy 2.4.6

15 signal types x 149 minutes = 1.00 GiB of float64 (134,100 blocks); best of 3.

(generated in 5 s)

## default

| signal | bits/sample | ratio | worst max err (% range) | encode MB/s | decode MB/s | encode µs/block | decode µs/block |
|---|---|---|---|---|---|---|---|
| linear | 0.02 | 3038.0 | 0.0000 | 3069 | 6038 | 2.61 | 1.32 |
| quadratic | 0.25 | 257.1 | 0.0015 | 2133 | 4665 | 3.75 | 1.71 |
| sin-4.12hz | 2.40 | 26.7 | 0.0010 | 1353 | 3767 | 5.91 | 2.12 |
| sin-9.87hz | 3.42 | 18.7 | 0.0010 | 1088 | 3681 | 7.35 | 2.17 |
| sin-50.3hz | 6.74 | 9.5 | 0.0010 | 1367 | 3698 | 5.85 | 2.16 |
| gauss-spikes | 6.63 | 9.7 | 1.4661 | 1621 | 4378 | 4.94 | 1.83 |
| impulses | 4.76 | 13.5 | 1.2592 | 2018 | 6618 | 3.96 | 1.21 |
| square-2.24hz | 0.04 | 1438.2 | 0.0000 | 2874 | 7005 | 2.78 | 1.14 |
| random-walk | 12.62 | 5.1 | 0.0015 | 1472 | 4382 | 5.43 | 1.83 |
| chirp | 7.81 | 8.2 | 0.0010 | 1087 | 3844 | 7.36 | 2.08 |
| noisy-sine | 5.24 | 12.2 | 0.2334 | 1946 | 5035 | 4.11 | 1.59 |
| sensor-0.1 | 1.78 | 36.0 | 0.0000 | 1288 | 3747 | 6.21 | 2.13 |
| random-walk q0.01 | 9.28 | 6.9 | 0.0000 | 1285 | 4256 | 6.23 | 1.88 |
| noisy-sine q0.1 | 5.24 | 12.2 | 0.2341 | 1946 | 5184 | 4.11 | 1.54 |
| noisy-sine q0.01 via float32 | 5.24 | 12.2 | 0.2341 | 1942 | 5011 | 4.12 | 1.60 |
| **all (1.00 GiB)** | 4.76 | 13.4 | | 1606 | 4558 | 4.98 | 1.76 |

## noise floor off

| signal | bits/sample | ratio | worst max err (% range) | encode MB/s | decode MB/s | encode µs/block | decode µs/block |
|---|---|---|---|---|---|---|---|
| linear | 0.02 | 3038.0 | 0.0000 | 4045 | 6708 | 1.98 | 1.19 |
| quadratic | 0.25 | 257.1 | 0.0015 | 2760 | 4725 | 2.90 | 1.69 |
| sin-4.12hz | 2.40 | 26.7 | 0.0010 | 1589 | 4062 | 5.03 | 1.97 |
| sin-9.87hz | 3.42 | 18.7 | 0.0010 | 1203 | 3798 | 6.65 | 2.11 |
| sin-50.3hz | 6.74 | 9.5 | 0.0010 | 1595 | 3831 | 5.02 | 2.09 |
| gauss-spikes | 12.06 | 5.3 | 0.0015 | 1511 | 4828 | 5.30 | 1.66 |
| impulses | 11.69 | 5.5 | 0.0015 | 1872 | 6250 | 4.27 | 1.28 |
| square-2.24hz | 0.04 | 1438.2 | 0.0000 | 3758 | 7002 | 2.13 | 1.14 |
| random-walk | 12.62 | 5.1 | 0.0015 | 1757 | 4374 | 4.55 | 1.83 |
| chirp | 7.81 | 8.2 | 0.0010 | 1233 | 3880 | 6.49 | 2.06 |
| noisy-sine | 13.72 | 4.7 | 0.0009 | 1817 | 4171 | 4.40 | 1.92 |
| sensor-0.1 | 1.78 | 36.0 | 0.0000 | 1529 | 3752 | 5.23 | 2.13 |
| random-walk q0.01 | 9.28 | 6.9 | 0.0000 | 1500 | 4236 | 5.33 | 1.89 |
| noisy-sine q0.1 | 8.89 | 7.2 | 0.0000 | 1575 | 3155 | 5.08 | 2.54 |
| noisy-sine q0.01 via float32 | 12.32 | 5.2 | 0.0000 | 1718 | 4408 | 4.66 | 1.81 |
| **all (1.00 GiB)** | 6.87 | 9.3 | | 1739 | 4394 | 4.60 | 1.82 |

## target 6

| signal | bits/sample | ratio | worst max err (% range) | encode MB/s | decode MB/s | encode µs/block | decode µs/block |
|---|---|---|---|---|---|---|---|
| linear | 0.02 | 3038.0 | 0.0000 | 2393 | 6654 | 3.34 | 1.20 |
| quadratic | 0.25 | 257.1 | 0.0015 | 1779 | 4642 | 4.50 | 1.72 |
| sin-4.12hz | 2.40 | 26.7 | 0.0010 | 1193 | 4046 | 6.70 | 1.98 |
| sin-9.87hz | 3.42 | 18.7 | 0.0010 | 978 | 3796 | 8.18 | 2.11 |
| sin-50.3hz | 0.92 | 69.7 | 0.0625 | 1227 | 5454 | 6.52 | 1.47 |
| gauss-spikes | 5.27 | 12.1 | 1.4661 | 1001 | 3687 | 7.99 | 2.17 |
| impulses | 4.76 | 13.5 | 1.2592 | 1742 | 6613 | 4.59 | 1.21 |
| square-2.24hz | 0.04 | 1438.2 | 0.0000 | 2279 | 6955 | 3.51 | 1.15 |
| random-walk | 5.39 | 11.9 | 0.1951 | 1237 | 4789 | 6.46 | 1.67 |
| chirp | 4.68 | 13.7 | 0.0625 | 895 | 3383 | 8.94 | 2.37 |
| noisy-sine | 5.24 | 12.2 | 0.2334 | 1685 | 4930 | 4.75 | 1.62 |
| sensor-0.1 | 1.78 | 36.0 | 0.0000 | 1185 | 3648 | 6.75 | 2.19 |
| random-walk q0.01 | 6.21 | 10.3 | 0.1750 | 1183 | 4441 | 6.76 | 1.80 |
| noisy-sine q0.1 | 5.24 | 12.2 | 0.2341 | 1683 | 4955 | 4.75 | 1.61 |
| noisy-sine q0.01 via float32 | 5.24 | 12.2 | 0.2341 | 1687 | 4944 | 4.74 | 1.62 |
| **all (1.00 GiB)** | 3.39 | 18.9 | | 1356 | 4635 | 5.90 | 1.73 |

## effort 1

| signal | bits/sample | ratio | worst max err (% range) | encode MB/s | decode MB/s | encode µs/block | decode µs/block |
|---|---|---|---|---|---|---|---|
| linear | 0.05 | 1297.3 | 0.0000 | 3655 | 5723 | 2.19 | 1.40 |
| quadratic | 0.25 | 252.6 | 0.0015 | 3103 | 4616 | 2.58 | 1.73 |
| sin-4.12hz | 2.50 | 25.6 | 0.0010 | 1824 | 3759 | 4.39 | 2.13 |
| sin-9.87hz | 3.65 | 17.6 | 0.0010 | 1893 | 4014 | 4.23 | 1.99 |
| sin-50.3hz | 6.88 | 9.3 | 0.0010 | 2140 | 3721 | 3.74 | 2.15 |
| gauss-spikes | 6.61 | 9.7 | 1.4661 | 2281 | 3965 | 3.51 | 2.02 |
| impulses | 4.74 | 13.5 | 1.2592 | 2859 | 6149 | 2.80 | 1.30 |
| square-2.24hz | 0.06 | 999.0 | 0.0000 | 3531 | 5894 | 2.27 | 1.36 |
| random-walk | 12.66 | 5.1 | 0.0015 | 2073 | 4340 | 3.86 | 1.84 |
| chirp | 7.90 | 8.1 | 0.0010 | 1918 | 4016 | 4.17 | 1.99 |
| noisy-sine | 5.24 | 12.2 | 0.2334 | 2942 | 4541 | 2.72 | 1.76 |
| sensor-0.1 | 2.18 | 29.4 | 0.0000 | 1794 | 3454 | 4.46 | 2.32 |
| random-walk q0.01 | 9.30 | 6.9 | 0.0000 | 2069 | 4108 | 3.87 | 1.95 |
| noisy-sine q0.1 | 5.24 | 12.2 | 0.2341 | 2940 | 4544 | 2.72 | 1.76 |
| noisy-sine q0.01 via float32 | 5.25 | 12.2 | 0.2341 | 2930 | 4507 | 2.73 | 1.78 |
| **all (1.00 GiB)** | 4.83 | 13.2 | | 2390 | 4368 | 3.35 | 1.83 |

## effort 5

| signal | bits/sample | ratio | worst max err (% range) | encode MB/s | decode MB/s | encode µs/block | decode µs/block |
|---|---|---|---|---|---|---|---|
| linear | 0.02 | 2981.4 | 0.0000 | 2570 | 7275 | 3.11 | 1.10 |
| quadratic | 0.27 | 240.7 | 0.0015 | 1874 | 4592 | 4.27 | 1.74 |
| sin-4.12hz | 2.38 | 26.9 | 0.0010 | 1277 | 4098 | 6.26 | 1.95 |
| sin-9.87hz | 3.40 | 18.8 | 0.0010 | 959 | 3817 | 8.34 | 2.10 |
| sin-50.3hz | 6.22 | 10.3 | 0.0010 | 1112 | 3037 | 7.19 | 2.63 |
| gauss-spikes | 6.48 | 9.9 | 1.4661 | 1321 | 3919 | 6.05 | 2.04 |
| impulses | 4.72 | 13.5 | 1.2592 | 1826 | 6567 | 4.38 | 1.22 |
| square-2.24hz | 0.05 | 1366.4 | 0.0000 | 2526 | 6504 | 3.17 | 1.23 |
| random-walk | 12.29 | 5.2 | 0.0015 | 1051 | 3840 | 7.61 | 2.08 |
| chirp | 7.76 | 8.2 | 0.0010 | 972 | 3523 | 8.23 | 2.27 |
| noisy-sine | 5.22 | 12.3 | 0.2334 | 1658 | 4811 | 4.82 | 1.66 |
| sensor-0.1 | 1.76 | 36.3 | 0.0000 | 1174 | 3584 | 6.82 | 2.23 |
| random-walk q0.01 | 8.91 | 7.2 | 0.0000 | 1233 | 3791 | 6.49 | 2.11 |
| noisy-sine q0.1 | 5.22 | 12.3 | 0.2341 | 1653 | 4828 | 4.84 | 1.66 |
| noisy-sine q0.01 via float32 | 5.22 | 12.3 | 0.2341 | 1652 | 4839 | 4.84 | 1.65 |
| **all (1.00 GiB)** | 4.66 | 13.7 | | 1388 | 4335 | 5.76 | 1.85 |

## effort 9

| signal | bits/sample | ratio | worst max err (% range) | encode MB/s | decode MB/s | encode µs/block | decode µs/block |
|---|---|---|---|---|---|---|---|
| linear | 0.02 | 3038.0 | 0.0000 | 1881 | 7329 | 4.25 | 1.09 |
| quadratic | 0.25 | 261.2 | 0.0015 | 561 | 4114 | 14.26 | 1.94 |
| sin-4.12hz | 2.24 | 28.5 | 0.0010 | 241 | 4072 | 33.13 | 1.96 |
| sin-9.87hz | 3.26 | 19.7 | 0.0010 | 218 | 3667 | 36.76 | 2.18 |
| sin-50.3hz | 6.09 | 10.5 | 0.0010 | 295 | 3220 | 27.11 | 2.48 |
| gauss-spikes | 6.43 | 10.0 | 1.4661 | 433 | 3941 | 18.46 | 2.03 |
| impulses | 4.72 | 13.5 | 1.2592 | 517 | 6608 | 15.47 | 1.21 |
| square-2.24hz | 0.04 | 1685.4 | 0.0000 | 1592 | 5473 | 5.02 | 1.46 |
| random-walk | 12.26 | 5.2 | 0.0015 | 277 | 3855 | 28.90 | 2.08 |
| chirp | 7.69 | 8.3 | 0.0010 | 276 | 3567 | 29.01 | 2.24 |
| noisy-sine | 5.22 | 12.3 | 0.2334 | 467 | 4850 | 17.14 | 1.65 |
| sensor-0.1 | 1.73 | 37.0 | 0.0000 | 207 | 3672 | 38.65 | 2.18 |
| random-walk q0.01 | 8.89 | 7.2 | 0.0000 | 213 | 3858 | 37.59 | 2.07 |
| noisy-sine q0.1 | 5.22 | 12.3 | 0.2341 | 468 | 4866 | 17.10 | 1.64 |
| noisy-sine q0.01 via float32 | 5.22 | 12.3 | 0.2341 | 465 | 4864 | 17.22 | 1.64 |
| **all (1.00 GiB)** | 4.62 | 13.9 | | 353 | 4305 | 22.67 | 1.86 |

## Threads (default params; one signal type per task; the kernels release the GIL)

| threads | encode MB/s | decode MB/s | encode speedup |
|---|---|---|---|
| 1 | 1612 | 4212 | 1.0x |
| 4 | 5315 | 14776 | 3.3x |
| 8 | 5507 | 12386 | 3.4x |
