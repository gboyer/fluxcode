# fluxcode gigabyte benchmark (bench/bench_gb.py, 2026-10-01, AC power; decode is per unit, as rows are read)

One complete run at 5117ff8 (8b7bb76, which only speeds up update, writes the same units).
Before the run: load average 1.6. Since the previous table (2026-09-26): variable
block sizes, snapped power-of-two grids, and `planes="best"` as the default, which compresses
every unit with bit and with byte planes and keeps the smaller ("bit planes only" is the old
default layout).

Apple M3, Darwin 25.6.0, Python 3.11.13, numpy 2.4.6

15 signal types x 149 minutes = 1.00 GiB of float64 (134,100 blocks); best of 3.

(generated in 5 s)

## default

| signal | bits/sample | ratio | worst max err (% range) | encode MB/s | decode MB/s | encode µs/block | decode µs/block |
|---|---|---|---|---|---|---|---|
| linear | 0.02 | 3178.8 | 0.0000 | 2906 | 6164 | 2.75 | 1.30 |
| quadratic | 0.25 | 254.0 | 0.0015 | 2058 | 4826 | 3.89 | 1.66 |
| sin-4.12hz | 2.40 | 26.7 | 0.0010 | 1331 | 3831 | 6.01 | 2.09 |
| sin-9.87hz | 3.42 | 18.7 | 0.0010 | 1069 | 3571 | 7.49 | 2.24 |
| sin-50.3hz | 6.74 | 9.5 | 0.0010 | 1346 | 3793 | 5.94 | 2.11 |
| gauss-spikes | 6.63 | 9.7 | 1.4661 | 1588 | 4411 | 5.04 | 1.81 |
| impulses | 4.76 | 13.5 | 1.2592 | 1961 | 6869 | 4.08 | 1.16 |
| square-2.24hz | 0.04 | 1439.8 | 0.0000 | 2746 | 7089 | 2.91 | 1.13 |
| random-walk | 12.62 | 5.1 | 0.0015 | 1444 | 4487 | 5.54 | 1.78 |
| chirp | 7.81 | 8.2 | 0.0010 | 1080 | 3982 | 7.41 | 2.01 |
| noisy-sine | 5.24 | 12.2 | 0.2334 | 1903 | 5141 | 4.20 | 1.56 |
| sensor-0.1 | 1.78 | 36.0 | 0.0000 | 1287 | 3723 | 6.21 | 2.15 |
| random-walk q0.01 | 9.28 | 6.9 | 0.0000 | 1268 | 4354 | 6.31 | 1.84 |
| noisy-sine q0.1 | 5.24 | 12.2 | 0.2341 | 1897 | 5179 | 4.22 | 1.54 |
| noisy-sine q0.01 via float32 | 5.24 | 12.2 | 0.2341 | 1900 | 5131 | 4.21 | 1.56 |
| **all (1.00 GiB)** | 4.76 | 13.4 | | 1574 | 4626 | 5.08 | 1.73 |

## noise floor off

| signal | bits/sample | ratio | worst max err (% range) | encode MB/s | decode MB/s | encode µs/block | decode µs/block |
|---|---|---|---|---|---|---|---|
| linear | 0.02 | 3178.8 | 0.0000 | 3833 | 6132 | 2.09 | 1.30 |
| quadratic | 0.25 | 254.0 | 0.0015 | 2644 | 4806 | 3.03 | 1.66 |
| sin-4.12hz | 2.40 | 26.7 | 0.0010 | 1552 | 3819 | 5.16 | 2.09 |
| sin-9.87hz | 3.42 | 18.7 | 0.0010 | 1206 | 3567 | 6.63 | 2.24 |
| sin-50.3hz | 6.74 | 9.5 | 0.0010 | 1573 | 3755 | 5.09 | 2.13 |
| gauss-spikes | 12.06 | 5.3 | 0.0015 | 1470 | 4902 | 5.44 | 1.63 |
| impulses | 11.69 | 5.5 | 0.0015 | 1809 | 6372 | 4.42 | 1.26 |
| square-2.24hz | 0.04 | 1439.8 | 0.0000 | 3552 | 7098 | 2.25 | 1.13 |
| random-walk | 12.62 | 5.1 | 0.0015 | 1704 | 4502 | 4.70 | 1.78 |
| chirp | 7.81 | 8.2 | 0.0010 | 1222 | 3955 | 6.55 | 2.02 |
| noisy-sine | 13.72 | 4.7 | 0.0009 | 1759 | 4286 | 4.55 | 1.87 |
| sensor-0.1 | 1.78 | 36.0 | 0.0000 | 1493 | 3791 | 5.36 | 2.11 |
| random-walk q0.01 | 9.28 | 6.9 | 0.0000 | 1465 | 4318 | 5.46 | 1.85 |
| noisy-sine q0.1 | 8.89 | 7.2 | 0.0000 | 1530 | 3266 | 5.23 | 2.45 |
| noisy-sine q0.01 via float32 | 12.35 | 5.2 | 0.0000 | 1669 | 4538 | 4.79 | 1.76 |
| **all (1.00 GiB)** | 6.87 | 9.3 | | 1697 | 4397 | 4.72 | 1.82 |

## target 6

| signal | bits/sample | ratio | worst max err (% range) | encode MB/s | decode MB/s | encode µs/block | decode µs/block |
|---|---|---|---|---|---|---|---|
| linear | 0.02 | 3178.8 | 0.0000 | 2286 | 6266 | 3.50 | 1.28 |
| quadratic | 0.25 | 254.0 | 0.0015 | 1732 | 4779 | 4.62 | 1.67 |
| sin-4.12hz | 2.40 | 26.7 | 0.0010 | 1198 | 3802 | 6.68 | 2.10 |
| sin-9.87hz | 3.42 | 18.7 | 0.0010 | 982 | 3567 | 8.14 | 2.24 |
| sin-50.3hz | 0.92 | 69.7 | 0.0625 | 1208 | 5481 | 6.62 | 1.46 |
| gauss-spikes | 5.27 | 12.1 | 1.4661 | 985 | 3765 | 8.12 | 2.12 |
| impulses | 4.76 | 13.5 | 1.2592 | 1687 | 6797 | 4.74 | 1.18 |
| square-2.24hz | 0.04 | 1439.8 | 0.0000 | 2186 | 7074 | 3.66 | 1.13 |
| random-walk | 5.39 | 11.9 | 0.1951 | 1214 | 4972 | 6.59 | 1.61 |
| chirp | 4.68 | 13.7 | 0.0625 | 883 | 3384 | 9.06 | 2.36 |
| noisy-sine | 5.24 | 12.2 | 0.2334 | 1643 | 5158 | 4.87 | 1.55 |
| sensor-0.1 | 1.78 | 36.0 | 0.0000 | 1163 | 3717 | 6.88 | 2.15 |
| random-walk q0.01 | 6.21 | 10.3 | 0.1750 | 1162 | 4610 | 6.88 | 1.74 |
| noisy-sine q0.1 | 5.24 | 12.2 | 0.2341 | 1643 | 5147 | 4.87 | 1.55 |
| noisy-sine q0.01 via float32 | 5.24 | 12.2 | 0.2341 | 1642 | 5145 | 4.87 | 1.55 |
| **all (1.00 GiB)** | 3.39 | 18.9 | | 1332 | 4667 | 6.01 | 1.71 |

## bit planes only

| signal | bits/sample | ratio | worst max err (% range) | encode MB/s | decode MB/s | encode µs/block | decode µs/block |
|---|---|---|---|---|---|---|---|
| linear | 0.02 | 3018.9 | 0.0000 | 3230 | 5045 | 2.48 | 1.59 |
| quadratic | 0.25 | 254.0 | 0.0015 | 3115 | 4745 | 2.57 | 1.69 |
| sin-4.12hz | 2.73 | 23.4 | 0.0010 | 2576 | 3915 | 3.11 | 2.04 |
| sin-9.87hz | 3.64 | 17.6 | 0.0010 | 2256 | 3400 | 3.55 | 2.35 |
| sin-50.3hz | 6.74 | 9.5 | 0.0010 | 2187 | 3776 | 3.66 | 2.12 |
| gauss-spikes | 6.63 | 9.7 | 1.4661 | 2184 | 4352 | 3.66 | 1.84 |
| impulses | 5.05 | 12.7 | 1.2592 | 2697 | 4673 | 2.97 | 1.71 |
| square-2.24hz | 0.05 | 1204.4 | 0.0000 | 3092 | 4597 | 2.59 | 1.74 |
| random-walk | 12.62 | 5.1 | 0.0015 | 2049 | 4520 | 3.90 | 1.77 |
| chirp | 7.81 | 8.2 | 0.0010 | 1657 | 3940 | 4.83 | 2.03 |
| noisy-sine | 5.44 | 11.8 | 0.2334 | 2470 | 3388 | 3.24 | 2.36 |
| sensor-0.1 | 1.78 | 36.0 | 0.0000 | 2178 | 3666 | 3.67 | 2.18 |
| random-walk q0.01 | 9.28 | 6.9 | 0.0000 | 2097 | 4235 | 3.82 | 1.89 |
| noisy-sine q0.1 | 5.44 | 11.8 | 0.2341 | 2451 | 3358 | 3.26 | 2.38 |
| noisy-sine q0.01 via float32 | 5.45 | 11.8 | 0.2341 | 2458 | 3341 | 3.25 | 2.39 |
| **all (1.00 GiB)** | 4.86 | 13.2 | | 2374 | 3988 | 3.37 | 2.01 |

## Threads (default params; one signal type per task; the kernels release the GIL)

| threads | encode MB/s | decode MB/s | encode speedup |
|---|---|---|---|
| 1 | 1571 | 4419 | 1.0x |
| 4 | 5200 | 14888 | 3.3x |
| 8 | 5388 | 13242 | 3.4x |
