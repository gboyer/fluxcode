# fluxcode gigabyte benchmark (bench/bench_gb.py, 2026-11-21, AC power; decode is per block group, as rows are read)

One complete run of the value-exp encoder: the noise floor on 12th differences (the top-band level, with lower
bands under anti-alias filters, a decay test against tones and a first-difference cap), from 128 samples; the
order picked on 8 spread windows; signed indices around a centered anchor. Against the previous run (b48c7a3 →
d0c80b1, 2026-11-01): all 15 signals 4.76 → 4.01 bits/sample (−16%), almost all of it the random walks, which
the floor now treats like the noise of their top band (12.62 → 5.63 bits/sample, the 0.01-decimal walk 9.28 →
5.63, lossy now: 0.35% of range at most); default encode 5.12 → 6.01 µs/block (+0.89 µs, +17%: the 12th
differences and the floor's checks), decode 1.79 → 1.69 µs/block (−0.10 µs, −6%: centered order-0 blocks
need no integration). Python path only (FLUXCODE_RUST=0); load average about 2 (an idle desktop).
On the realistic corpus of experimental/value/ the same encoder is 28% smaller on high-rate DAQ-like signals
and 13% on 1 s historian tags (docs/TUNING.md).

Apple M3, Darwin 25.6.0, Python 3.11.13, numpy 2.4.6

15 signal types x 149 minutes = 1.00 GiB of float64 (134,100 blocks); best of 3.

(generated in 5 s)

## default

| signal | bits/sample | ratio | worst max err (% range) | encode MB/s | decode MB/s | encode µs/block | decode µs/block |
|---|---|---|---|---|---|---|---|
| linear | 0.02 | 3137.3 | 0.0000 | 2615 | 6690 | 3.06 | 1.20 |
| quadratic | 0.32 | 201.2 | 0.0015 | 1817 | 5034 | 4.40 | 1.59 |
| sin-4.12hz | 2.40 | 26.7 | 0.0010 | 1132 | 4069 | 7.07 | 1.97 |
| sin-9.87hz | 3.42 | 18.7 | 0.0010 | 932 | 3796 | 8.58 | 2.11 |
| sin-50.3hz | 6.73 | 9.5 | 0.0010 | 1142 | 3969 | 7.01 | 2.02 |
| gauss-spikes | 6.04 | 10.6 | 1.4661 | 1341 | 4179 | 5.97 | 1.91 |
| impulses | 4.78 | 13.4 | 1.2592 | 1414 | 6718 | 5.66 | 1.19 |
| square-2.24hz | 0.04 | 1742.3 | 0.0000 | 1818 | 6150 | 4.40 | 1.30 |
| random-walk | 5.63 | 11.4 | 0.3303 | 1397 | 4768 | 5.73 | 1.68 |
| chirp | 7.80 | 8.2 | 0.0010 | 966 | 4228 | 8.28 | 1.89 |
| noisy-sine | 5.22 | 12.3 | 0.2334 | 1435 | 5090 | 5.58 | 1.57 |
| sensor-0.1 | 1.71 | 37.3 | 0.0000 | 1055 | 3665 | 7.59 | 2.18 |
| random-walk q0.01 | 5.63 | 11.4 | 0.3501 | 1393 | 4910 | 5.74 | 1.63 |
| noisy-sine q0.1 | 5.22 | 12.3 | 0.2341 | 1434 | 5137 | 5.58 | 1.56 |
| noisy-sine q0.01 via float32 | 5.22 | 12.3 | 0.2341 | 1437 | 5123 | 5.57 | 1.56 |
| **all (1.00 GiB)** | 4.01 | 16.0 | | 1330 | 4733 | 6.01 | 1.69 |
| random-walk q0.05 | 5.62 | 11.4 | 0.2915 | 1387 | 4777 | 5.77 | 1.67 |
| random-walk q1/3 | 3.92 | 16.3 | 0.0000 | 662 | 3599 | 12.09 | 2.22 |
| random-walk adc 10/4095 | 5.63 | 11.4 | 0.3634 | 1395 | 4747 | 5.73 | 1.69 |

## noise floor off

| signal | bits/sample | ratio | worst max err (% range) | encode MB/s | decode MB/s | encode µs/block | decode µs/block |
|---|---|---|---|---|---|---|---|
| linear | 0.02 | 3137.3 | 0.0000 | 3611 | 6763 | 2.22 | 1.18 |
| quadratic | 0.32 | 201.2 | 0.0015 | 2672 | 5097 | 2.99 | 1.57 |
| sin-4.12hz | 2.40 | 26.7 | 0.0010 | 1440 | 4062 | 5.56 | 1.97 |
| sin-9.87hz | 3.42 | 18.7 | 0.0010 | 1139 | 3765 | 7.02 | 2.12 |
| sin-50.3hz | 6.73 | 9.5 | 0.0010 | 1438 | 3968 | 5.56 | 2.02 |
| gauss-spikes | 12.22 | 5.2 | 0.0015 | 1336 | 4914 | 5.99 | 1.63 |
| impulses | 11.72 | 5.5 | 0.0015 | 1242 | 6259 | 6.44 | 1.28 |
| square-2.24hz | 0.04 | 1742.3 | 0.0000 | 2889 | 6156 | 2.77 | 1.30 |
| random-walk | 12.60 | 5.1 | 0.0015 | 1623 | 4974 | 4.93 | 1.61 |
| chirp | 7.80 | 8.2 | 0.0010 | 1180 | 4239 | 6.78 | 1.89 |
| noisy-sine | 13.71 | 4.7 | 0.0009 | 1674 | 4677 | 4.78 | 1.71 |
| sensor-0.1 | 1.71 | 37.3 | 0.0000 | 1473 | 3758 | 5.43 | 2.13 |
| random-walk q0.01 | 9.23 | 6.9 | 0.0000 | 1442 | 4507 | 5.55 | 1.78 |
| noisy-sine q0.1 | 8.86 | 7.2 | 0.0000 | 1498 | 3414 | 5.34 | 2.34 |
| noisy-sine q0.01 via float32 | 12.28 | 5.2 | 0.0000 | 1622 | 4828 | 4.93 | 1.66 |
| **all (1.00 GiB)** | 6.87 | 9.3 | | 1573 | 4584 | 5.09 | 1.75 |
| random-walk q0.05 | 6.49 | 9.9 | 0.0000 | 1600 | 4483 | 5.00 | 1.78 |
| random-walk q1/3 | 3.92 | 16.3 | 0.0000 | 803 | 3629 | 9.97 | 2.20 |
| random-walk adc 10/4095 | 11.29 | 5.7 | 0.0015 | 720 | 4625 | 11.12 | 1.73 |

## target 6

| signal | bits/sample | ratio | worst max err (% range) | encode MB/s | decode MB/s | encode µs/block | decode µs/block |
|---|---|---|---|---|---|---|---|
| linear | 0.02 | 3137.3 | 0.0000 | 1086 | 7168 | 7.37 | 1.12 |
| quadratic | 0.32 | 201.2 | 0.0015 | 1553 | 5085 | 5.15 | 1.57 |
| sin-4.12hz | 2.40 | 26.7 | 0.0010 | 1018 | 4065 | 7.86 | 1.97 |
| sin-9.87hz | 3.42 | 18.7 | 0.0010 | 868 | 3788 | 9.22 | 2.11 |
| sin-50.3hz | 0.92 | 69.6 | 0.0625 | 1026 | 5709 | 7.79 | 1.40 |
| gauss-spikes | 6.02 | 10.6 | 1.4661 | 1206 | 4208 | 6.63 | 1.90 |
| impulses | 4.78 | 13.4 | 1.2592 | 1269 | 6794 | 6.31 | 1.18 |
| square-2.24hz | 0.04 | 1742.3 | 0.0000 | 1338 | 6182 | 5.98 | 1.29 |
| random-walk | 5.63 | 11.4 | 0.3303 | 1255 | 4814 | 6.38 | 1.66 |
| chirp | 4.64 | 13.8 | 0.0625 | 792 | 3390 | 10.10 | 2.36 |
| noisy-sine | 5.22 | 12.3 | 0.2334 | 1286 | 5029 | 6.22 | 1.59 |
| sensor-0.1 | 1.71 | 37.3 | 0.0000 | 968 | 3675 | 8.26 | 2.18 |
| random-walk q0.01 | 5.63 | 11.4 | 0.3501 | 1252 | 4782 | 6.39 | 1.67 |
| noisy-sine q0.1 | 5.22 | 12.3 | 0.2341 | 1285 | 5145 | 6.23 | 1.55 |
| noisy-sine q0.01 via float32 | 5.22 | 12.3 | 0.2341 | 1285 | 5172 | 6.22 | 1.55 |
| **all (1.00 GiB)** | 3.41 | 18.8 | | 1131 | 4780 | 7.07 | 1.67 |
| random-walk q0.05 | 5.62 | 11.4 | 0.2915 | 1240 | 4758 | 6.45 | 1.68 |
| random-walk q1/3 | 3.92 | 16.3 | 0.0000 | 621 | 3539 | 12.88 | 2.26 |
| random-walk adc 10/4095 | 5.63 | 11.4 | 0.3634 | 1249 | 4738 | 6.41 | 1.69 |

## effort 1

| signal | bits/sample | ratio | worst max err (% range) | encode MB/s | decode MB/s | encode µs/block | decode µs/block |
|---|---|---|---|---|---|---|---|
| linear | 0.05 | 1300.8 | 0.0000 | 3012 | 6164 | 2.66 | 1.30 |
| quadratic | 0.65 | 97.9 | 0.0015 | 2139 | 5808 | 3.74 | 1.38 |
| sin-4.12hz | 2.51 | 25.5 | 0.0010 | 1437 | 3810 | 5.57 | 2.10 |
| sin-9.87hz | 3.65 | 17.5 | 0.0010 | 1478 | 4303 | 5.41 | 1.86 |
| sin-50.3hz | 6.90 | 9.3 | 0.0010 | 1683 | 4023 | 4.75 | 1.99 |
| gauss-spikes | 6.05 | 10.6 | 1.4661 | 1729 | 4323 | 4.63 | 1.85 |
| impulses | 4.77 | 13.4 | 1.2592 | 1773 | 6628 | 4.51 | 1.21 |
| square-2.24hz | 0.06 | 1032.4 | 0.0000 | 2031 | 6344 | 3.94 | 1.26 |
| random-walk | 5.63 | 11.4 | 0.3303 | 1881 | 4941 | 4.25 | 1.62 |
| chirp | 7.89 | 8.1 | 0.0010 | 1569 | 4367 | 5.10 | 1.83 |
| noisy-sine | 5.21 | 12.3 | 0.2334 | 1902 | 5281 | 4.21 | 1.51 |
| sensor-0.1 | 2.16 | 29.6 | 0.0000 | 1340 | 3327 | 5.97 | 2.40 |
| random-walk q0.01 | 5.63 | 11.4 | 0.3501 | 1872 | 4908 | 4.27 | 1.63 |
| noisy-sine q0.1 | 5.21 | 12.3 | 0.2341 | 1889 | 5241 | 4.24 | 1.53 |
| noisy-sine q0.01 via float32 | 5.21 | 12.3 | 0.2341 | 1885 | 5251 | 4.24 | 1.52 |
| **all (1.00 GiB)** | 4.11 | 15.6 | | 1778 | 4802 | 4.50 | 1.67 |
| random-walk q0.05 | 5.62 | 11.4 | 0.2915 | 1862 | 4860 | 4.30 | 1.65 |
| random-walk q1/3 | 3.76 | 17.0 | 0.0000 | 925 | 5042 | 8.65 | 1.59 |
| random-walk adc 10/4095 | 5.63 | 11.4 | 0.3634 | 1873 | 4900 | 4.27 | 1.63 |

## effort 5

| signal | bits/sample | ratio | worst max err (% range) | encode MB/s | decode MB/s | encode µs/block | decode µs/block |
|---|---|---|---|---|---|---|---|
| linear | 0.02 | 3000.0 | 0.0000 | 2240 | 9152 | 3.57 | 0.87 |
| quadratic | 0.33 | 192.3 | 0.0015 | 1595 | 4975 | 5.02 | 1.61 |
| sin-4.12hz | 2.39 | 26.8 | 0.0010 | 1039 | 3988 | 7.70 | 2.01 |
| sin-9.87hz | 3.40 | 18.8 | 0.0010 | 812 | 3720 | 9.85 | 2.15 |
| sin-50.3hz | 6.23 | 10.3 | 0.0010 | 972 | 3216 | 8.23 | 2.49 |
| gauss-spikes | 5.90 | 10.9 | 1.4661 | 1162 | 3960 | 6.88 | 2.02 |
| impulses | 4.77 | 13.4 | 1.2592 | 1300 | 6740 | 6.15 | 1.19 |
| square-2.24hz | 0.04 | 1467.9 | 0.0000 | 1645 | 6085 | 4.86 | 1.31 |
| random-walk | 5.59 | 11.4 | 0.3303 | 1245 | 5671 | 6.43 | 1.41 |
| chirp | 7.76 | 8.3 | 0.0010 | 868 | 3801 | 9.21 | 2.10 |
| noisy-sine | 5.18 | 12.4 | 0.2334 | 1265 | 5955 | 6.33 | 1.34 |
| sensor-0.1 | 1.71 | 37.3 | 0.0000 | 967 | 4396 | 8.27 | 1.82 |
| random-walk q0.01 | 5.59 | 11.4 | 0.3501 | 1237 | 5660 | 6.47 | 1.41 |
| noisy-sine q0.1 | 5.18 | 12.4 | 0.2341 | 1261 | 5948 | 6.35 | 1.35 |
| noisy-sine q0.01 via float32 | 5.18 | 12.4 | 0.2341 | 1258 | 5945 | 6.36 | 1.35 |
| **all (1.00 GiB)** | 3.95 | 16.2 | | 1180 | 4912 | 6.78 | 1.63 |
| random-walk q0.05 | 5.59 | 11.5 | 0.2915 | 1244 | 5653 | 6.43 | 1.42 |
| random-walk q1/3 | 3.83 | 16.7 | 0.0000 | 617 | 4355 | 12.97 | 1.84 |
| random-walk adc 10/4095 | 5.59 | 11.4 | 0.3634 | 1246 | 5691 | 6.42 | 1.41 |

## effort 9

| signal | bits/sample | ratio | worst max err (% range) | encode MB/s | decode MB/s | encode µs/block | decode µs/block |
|---|---|---|---|---|---|---|---|
| linear | 0.02 | 3038.0 | 0.0000 | 1698 | 9161 | 4.71 | 0.87 |
| quadratic | 0.30 | 212.9 | 0.0015 | 661 | 4909 | 12.11 | 1.63 |
| sin-4.12hz | 2.25 | 28.5 | 0.0010 | 232 | 4098 | 34.45 | 1.95 |
| sin-9.87hz | 3.26 | 19.6 | 0.0010 | 210 | 3591 | 38.03 | 2.23 |
| sin-50.3hz | 6.09 | 10.5 | 0.0010 | 286 | 3335 | 27.92 | 2.40 |
| gauss-spikes | 5.87 | 10.9 | 1.4661 | 487 | 3710 | 16.43 | 2.16 |
| impulses | 4.77 | 13.4 | 1.2592 | 440 | 6812 | 18.18 | 1.17 |
| square-2.24hz | 0.03 | 2427.1 | 0.0000 | 1339 | 5494 | 5.98 | 1.46 |
| random-walk | 5.59 | 11.4 | 0.3303 | 491 | 5698 | 16.30 | 1.40 |
| chirp | 7.69 | 8.3 | 0.0010 | 273 | 3835 | 29.33 | 2.09 |
| noisy-sine | 5.18 | 12.4 | 0.2334 | 430 | 5991 | 18.60 | 1.34 |
| sensor-0.1 | 1.70 | 37.7 | 0.0000 | 203 | 4467 | 39.48 | 1.79 |
| random-walk q0.01 | 5.59 | 11.4 | 0.3501 | 481 | 5665 | 16.64 | 1.41 |
| noisy-sine q0.1 | 5.18 | 12.4 | 0.2341 | 431 | 5944 | 18.58 | 1.35 |
| noisy-sine q0.01 via float32 | 5.18 | 12.4 | 0.2341 | 427 | 5990 | 18.75 | 1.34 |
| **all (1.00 GiB)** | 3.91 | 16.4 | | 380 | 4882 | 21.03 | 1.64 |
| random-walk q0.05 | 5.59 | 11.5 | 0.2915 | 477 | 5681 | 16.78 | 1.41 |
| random-walk q1/3 | 3.83 | 16.7 | 0.0000 | 237 | 4570 | 33.71 | 1.75 |
| random-walk adc 10/4095 | 5.59 | 11.4 | 0.3634 | 486 | 5656 | 16.47 | 1.41 |

## Threads (default params; one signal type per task; the kernels release the GIL)

| threads | encode MB/s | decode MB/s | encode speedup |
|---|---|---|---|
| 1 | 1332 | 4080 | 1.0x |
| 4 | 4579 | 15729 | 3.4x |
| 8 | 4991 | 13370 | 3.7x |
