# fluxcode gigabyte benchmark (bench/bench_gb.py, 2026-10-07, AC power; decode is per block group, as rows are read)

One complete run at 0581c29 (the noise floor from the quietest of 256-difference windows, and on timed data from
consecutive scans), replacing the tables at d039350. The data have no times, so only the windows change anything: the
block groups are the same size (the default 4.76 bits/sample, ratio 13.4) except Gaussian spikes (6.63 → 6.60
bits/sample: one block in 60 now gates). Encode µs/block against d039350: default 4.98 → 5.03 (+1%), effort 1
3.35 → 3.41 (+2%), effort 5 5.76 → 5.78, effort 9 22.67 → 22.67; with the noise floor off 4.60 → 4.59, so the
default's +1% is the windowed estimate. Decode is unchanged code; 1.76 → 1.81 µs/block for the default is run-to-run
noise (the load average was 2.4–2.6 from other apps, against about 2 then). Threads: 4 threads scale 3.3x, 8 threads
3.4x. Python path only (FLUXCODE_RUST=0).

Apple M3, Darwin 25.6.0, Python 3.11.13, numpy 2.4.6

15 signal types x 149 minutes = 1.00 GiB of float64 (134,100 blocks); best of 3.

(generated in 5 s)

## default

| signal | bits/sample | ratio | worst max err (% range) | encode MB/s | decode MB/s | encode µs/block | decode µs/block |
|---|---|---|---|---|---|---|---|
| linear | 0.02 | 3038.0 | 0.0000 | 3344 | 5632 | 2.39 | 1.42 |
| quadratic | 0.25 | 257.1 | 0.0015 | 2059 | 4493 | 3.88 | 1.78 |
| sin-4.12hz | 2.40 | 26.7 | 0.0010 | 1329 | 3728 | 6.02 | 2.15 |
| sin-9.87hz | 3.42 | 18.7 | 0.0010 | 1071 | 3486 | 7.47 | 2.29 |
| sin-50.3hz | 6.74 | 9.5 | 0.0010 | 1349 | 3659 | 5.93 | 2.19 |
| gauss-spikes | 6.60 | 9.7 | 1.4661 | 1602 | 4211 | 4.99 | 1.90 |
| impulses | 4.76 | 13.5 | 1.2592 | 1972 | 6596 | 4.06 | 1.21 |
| square-2.24hz | 0.04 | 1438.2 | 0.0000 | 3151 | 6744 | 2.54 | 1.19 |
| random-walk | 12.62 | 5.1 | 0.0015 | 1447 | 4269 | 5.53 | 1.87 |
| chirp | 7.81 | 8.2 | 0.0010 | 1080 | 3870 | 7.41 | 2.07 |
| noisy-sine | 5.24 | 12.2 | 0.2334 | 1903 | 4903 | 4.20 | 1.63 |
| sensor-0.1 | 1.78 | 36.0 | 0.0000 | 1284 | 3497 | 6.23 | 2.29 |
| random-walk q0.01 | 9.28 | 6.9 | 0.0000 | 1271 | 4152 | 6.29 | 1.93 |
| noisy-sine q0.1 | 5.24 | 12.2 | 0.2341 | 1897 | 4887 | 4.22 | 1.64 |
| noisy-sine q0.01 via float32 | 5.24 | 12.2 | 0.2341 | 1900 | 4927 | 4.21 | 1.62 |
| **all (1.00 GiB)** | 4.76 | 13.4 | | 1592 | 4416 | 5.03 | 1.81 |

## noise floor off

| signal | bits/sample | ratio | worst max err (% range) | encode MB/s | decode MB/s | encode µs/block | decode µs/block |
|---|---|---|---|---|---|---|---|
| linear | 0.02 | 3038.0 | 0.0000 | 4035 | 5806 | 1.98 | 1.38 |
| quadratic | 0.25 | 257.1 | 0.0015 | 2746 | 4584 | 2.91 | 1.75 |
| sin-4.12hz | 2.40 | 26.7 | 0.0010 | 1584 | 3738 | 5.05 | 2.14 |
| sin-9.87hz | 3.42 | 18.7 | 0.0010 | 1229 | 3501 | 6.51 | 2.29 |
| sin-50.3hz | 6.74 | 9.5 | 0.0010 | 1614 | 3696 | 4.96 | 2.16 |
| gauss-spikes | 12.06 | 5.3 | 0.0015 | 1503 | 4766 | 5.32 | 1.68 |
| impulses | 11.69 | 5.5 | 0.0015 | 1865 | 6176 | 4.29 | 1.30 |
| square-2.24hz | 0.04 | 1438.2 | 0.0000 | 3724 | 6803 | 2.15 | 1.18 |
| random-walk | 12.62 | 5.1 | 0.0015 | 1753 | 4350 | 4.56 | 1.84 |
| chirp | 7.81 | 8.2 | 0.0010 | 1246 | 3871 | 6.42 | 2.07 |
| noisy-sine | 13.72 | 4.7 | 0.0009 | 1813 | 4110 | 4.41 | 1.95 |
| sensor-0.1 | 1.78 | 36.0 | 0.0000 | 1526 | 3513 | 5.24 | 2.28 |
| random-walk q0.01 | 9.28 | 6.9 | 0.0000 | 1498 | 4163 | 5.34 | 1.92 |
| noisy-sine q0.1 | 8.89 | 7.2 | 0.0000 | 1569 | 3098 | 5.10 | 2.58 |
| noisy-sine q0.01 via float32 | 12.32 | 5.2 | 0.0000 | 1722 | 4370 | 4.65 | 1.83 |
| **all (1.00 GiB)** | 6.87 | 9.3 | | 1742 | 4236 | 4.59 | 1.89 |

## target 6

| signal | bits/sample | ratio | worst max err (% range) | encode MB/s | decode MB/s | encode µs/block | decode µs/block |
|---|---|---|---|---|---|---|---|
| linear | 0.02 | 3038.0 | 0.0000 | 2544 | 5779 | 3.14 | 1.38 |
| quadratic | 0.25 | 257.1 | 0.0015 | 1729 | 4593 | 4.63 | 1.74 |
| sin-4.12hz | 2.40 | 26.7 | 0.0010 | 1197 | 3725 | 6.69 | 2.15 |
| sin-9.87hz | 3.42 | 18.7 | 0.0010 | 982 | 3503 | 8.15 | 2.28 |
| sin-50.3hz | 0.92 | 69.7 | 0.0625 | 1206 | 5321 | 6.64 | 1.50 |
| gauss-spikes | 5.28 | 12.1 | 1.4661 | 994 | 3627 | 8.05 | 2.21 |
| impulses | 4.76 | 13.5 | 1.2592 | 1689 | 6625 | 4.74 | 1.21 |
| square-2.24hz | 0.04 | 1438.2 | 0.0000 | 2427 | 6701 | 3.30 | 1.19 |
| random-walk | 5.39 | 11.9 | 0.1951 | 1214 | 4719 | 6.59 | 1.70 |
| chirp | 4.68 | 13.7 | 0.0625 | 882 | 3274 | 9.08 | 2.44 |
| noisy-sine | 5.24 | 12.2 | 0.2334 | 1643 | 4924 | 4.87 | 1.62 |
| sensor-0.1 | 1.78 | 36.0 | 0.0000 | 1166 | 3510 | 6.86 | 2.28 |
| random-walk q0.01 | 6.21 | 10.3 | 0.1750 | 1162 | 4429 | 6.89 | 1.81 |
| noisy-sine q0.1 | 5.24 | 12.2 | 0.2341 | 1642 | 4926 | 4.87 | 1.62 |
| noisy-sine q0.01 via float32 | 5.24 | 12.2 | 0.2341 | 1644 | 4900 | 4.87 | 1.63 |
| **all (1.00 GiB)** | 3.39 | 18.9 | | 1343 | 4482 | 5.96 | 1.78 |

## effort 1

| signal | bits/sample | ratio | worst max err (% range) | encode MB/s | decode MB/s | encode µs/block | decode µs/block |
|---|---|---|---|---|---|---|---|
| linear | 0.05 | 1297.3 | 0.0000 | 4058 | 5128 | 1.97 | 1.56 |
| quadratic | 0.25 | 252.6 | 0.0015 | 2956 | 4607 | 2.71 | 1.74 |
| sin-4.12hz | 2.50 | 25.6 | 0.0010 | 1781 | 3511 | 4.49 | 2.28 |
| sin-9.87hz | 3.65 | 17.6 | 0.0010 | 1844 | 3878 | 4.34 | 2.06 |
| sin-50.3hz | 6.88 | 9.3 | 0.0010 | 2080 | 3708 | 3.85 | 2.16 |
| gauss-spikes | 6.59 | 9.7 | 1.4661 | 2218 | 3927 | 3.61 | 2.04 |
| impulses | 4.74 | 13.5 | 1.2592 | 2755 | 6156 | 2.90 | 1.30 |
| square-2.24hz | 0.06 | 999.0 | 0.0000 | 3929 | 5613 | 2.04 | 1.43 |
| random-walk | 12.66 | 5.1 | 0.0015 | 2015 | 4328 | 3.97 | 1.85 |
| chirp | 7.90 | 8.1 | 0.0010 | 1876 | 3933 | 4.26 | 2.03 |
| noisy-sine | 5.24 | 12.2 | 0.2334 | 2824 | 4540 | 2.83 | 1.76 |
| sensor-0.1 | 2.18 | 29.4 | 0.0000 | 1745 | 3068 | 4.58 | 2.61 |
| random-walk q0.01 | 9.30 | 6.9 | 0.0000 | 2007 | 4058 | 3.99 | 1.97 |
| noisy-sine q0.1 | 5.24 | 12.2 | 0.2341 | 2831 | 4541 | 2.83 | 1.76 |
| noisy-sine q0.01 via float32 | 5.25 | 12.2 | 0.2341 | 2824 | 4532 | 2.83 | 1.77 |
| **all (1.00 GiB)** | 4.83 | 13.2 | | 2344 | 4239 | 3.41 | 1.89 |

## effort 5

| signal | bits/sample | ratio | worst max err (% range) | encode MB/s | decode MB/s | encode µs/block | decode µs/block |
|---|---|---|---|---|---|---|---|
| linear | 0.02 | 2981.4 | 0.0000 | 2785 | 7298 | 2.87 | 1.10 |
| quadratic | 0.27 | 240.7 | 0.0015 | 1839 | 4575 | 4.35 | 1.75 |
| sin-4.12hz | 2.38 | 26.9 | 0.0010 | 1254 | 3708 | 6.38 | 2.16 |
| sin-9.87hz | 3.40 | 18.8 | 0.0010 | 942 | 3499 | 8.49 | 2.29 |
| sin-50.3hz | 6.22 | 10.3 | 0.0010 | 1107 | 3041 | 7.23 | 2.63 |
| gauss-spikes | 6.46 | 9.9 | 1.4661 | 1316 | 3880 | 6.08 | 2.06 |
| impulses | 4.72 | 13.5 | 1.2592 | 1792 | 6623 | 4.46 | 1.21 |
| square-2.24hz | 0.05 | 1366.4 | 0.0000 | 2740 | 6471 | 2.92 | 1.24 |
| random-walk | 12.29 | 5.2 | 0.0015 | 1052 | 3841 | 7.60 | 2.08 |
| chirp | 7.76 | 8.2 | 0.0010 | 970 | 3530 | 8.25 | 2.27 |
| noisy-sine | 5.22 | 12.3 | 0.2334 | 1628 | 4866 | 4.92 | 1.64 |
| sensor-0.1 | 1.76 | 36.3 | 0.0000 | 1176 | 3454 | 6.80 | 2.32 |
| random-walk q0.01 | 8.91 | 7.2 | 0.0000 | 1223 | 3769 | 6.54 | 2.12 |
| noisy-sine q0.1 | 5.22 | 12.3 | 0.2341 | 1625 | 4800 | 4.92 | 1.67 |
| noisy-sine q0.01 via float32 | 5.22 | 12.2 | 0.2341 | 1626 | 4845 | 4.92 | 1.65 |
| **all (1.00 GiB)** | 4.66 | 13.7 | | 1383 | 4259 | 5.78 | 1.88 |

## effort 9

| signal | bits/sample | ratio | worst max err (% range) | encode MB/s | decode MB/s | encode µs/block | decode µs/block |
|---|---|---|---|---|---|---|---|
| linear | 0.02 | 3038.0 | 0.0000 | 1990 | 7318 | 4.02 | 1.09 |
| quadratic | 0.25 | 261.2 | 0.0015 | 558 | 3722 | 14.34 | 2.15 |
| sin-4.12hz | 2.24 | 28.5 | 0.0010 | 241 | 3676 | 33.13 | 2.18 |
| sin-9.87hz | 3.26 | 19.7 | 0.0010 | 218 | 3338 | 36.75 | 2.40 |
| sin-50.3hz | 6.09 | 10.5 | 0.0010 | 295 | 3147 | 27.16 | 2.54 |
| gauss-spikes | 6.41 | 10.0 | 1.4661 | 435 | 3931 | 18.38 | 2.04 |
| impulses | 4.72 | 13.5 | 1.2592 | 514 | 6600 | 15.56 | 1.21 |
| square-2.24hz | 0.04 | 1685.4 | 0.0000 | 1672 | 4891 | 4.78 | 1.64 |
| random-walk | 12.26 | 5.2 | 0.0015 | 277 | 3836 | 28.88 | 2.09 |
| chirp | 7.69 | 8.3 | 0.0010 | 276 | 3541 | 28.93 | 2.26 |
| noisy-sine | 5.22 | 12.3 | 0.2334 | 466 | 4847 | 17.15 | 1.65 |
| sensor-0.1 | 1.73 | 37.0 | 0.0000 | 207 | 3386 | 38.72 | 2.36 |
| random-walk q0.01 | 8.89 | 7.2 | 0.0000 | 212 | 3749 | 37.68 | 2.13 |
| noisy-sine q0.1 | 5.22 | 12.3 | 0.2341 | 465 | 4603 | 17.21 | 1.74 |
| noisy-sine q0.01 via float32 | 5.22 | 12.3 | 0.2341 | 463 | 4826 | 17.29 | 1.66 |
| **all (1.00 GiB)** | 4.62 | 13.9 | | 353 | 4120 | 22.67 | 1.94 |

## Threads (default params; one signal type per task; the kernels release the GIL)

| threads | encode MB/s | decode MB/s | encode speedup |
|---|---|---|---|
| 1 | 1587 | 3766 | 1.0x |
| 4 | 5184 | 14763 | 3.3x |
| 8 | 5440 | 12761 | 3.4x |
