# fluxcode gigabyte benchmark (bench/bench_gb.py, 2026-11-01, AC power; decode is per block group, as rows are read)

One complete run of the scaled grids (`grids="rational"`, the default) and, in the same session, of `main` at b48c7a3
(the encoder before them). The 15 signals are the same size, except under `target 6`: the random walk on 0.01 is
5.46 bits/sample against 6.21 on b48c7a3, whose cap stopped short on exact grids (ENCODER §5.8). Each block stores a
4-byte grid scale now; the block groups here compress it to nothing. Encode µs/block against b48c7a3: default
4.99 → 5.12 (+0.13 µs, +3%), noise floor off 4.58 → 4.77 (+0.19 µs, +4%), effort 1 3.38 → 3.53 (+0.15 µs, +4%),
effort 5 5.75 → 5.96 (+0.21 µs, +4%), effort 9 22.69 → 22.87 (+0.18 µs, +1%), target 6 5.91 → 6.44 (+0.53 µs, +9%:
the 0..999 ramp `linear` costs 7 bits/sample at its planned exponent, so the cap re-encodes it and keeps the
original, +3.4 µs on that signal). Decode 1.75 → 1.79 µs/block (+0.04 µs, +2%); the square wave's 1.13 → 1.57 is a
layout choice: on its N = 200 grid the residuals are ±1, and bit planes are 2.5% smaller there than byte planes
(they decode at 1.00 µs/block). Python path only (FLUXCODE_RUST=0); the load average was 1.6–2.0 before each step.

The three rows after each total are on grids a writer doesn't know (`_signals.HIDDEN_GRIDS`), outside the total. On
b48c7a3 at default params (`bench` script in the same session, same series): 0.05 8.09 bits/sample at 5.95 µs/block,
thirds 7.68 at 11.32 (lossy: the noise floor's grid), ADC counts of 10 V / 4095 12.61 at 5.35. So the scaled grids are
19%, 48% and 10% smaller, lossless where b48c7a3 rounded (the ADC in all but one of its 8,940 blocks, the widest at
53,464 counts), at −0.13, −0.50 and +6.63 µs/block (−2%, −4%, +124%). The rational search for the ADC grid
2000 · 10⁻³ / 819 is about 4 µs of that.

Apple M3, Darwin 25.6.0, Python 3.11.13, numpy 2.4.6

15 signal types x 149 minutes = 1.00 GiB of float64 (134,100 blocks); best of 3.

(generated in 5 s)

## default

| signal | bits/sample | ratio | worst max err (% range) | encode MB/s | decode MB/s | encode µs/block | decode µs/block |
|---|---|---|---|---|---|---|---|
| linear | 0.02 | 3137.3 | 0.0000 | 3337 | 5895 | 2.40 | 1.36 |
| quadratic | 0.25 | 257.1 | 0.0015 | 1958 | 4709 | 4.09 | 1.70 |
| sin-4.12hz | 2.40 | 26.7 | 0.0010 | 1311 | 3782 | 6.10 | 2.12 |
| sin-9.87hz | 3.42 | 18.7 | 0.0010 | 1055 | 3532 | 7.58 | 2.26 |
| sin-50.3hz | 6.74 | 9.5 | 0.0010 | 1319 | 3696 | 6.06 | 2.16 |
| gauss-spikes | 6.60 | 9.7 | 1.4661 | 1581 | 4311 | 5.06 | 1.86 |
| impulses | 4.76 | 13.5 | 1.2592 | 1943 | 6813 | 4.12 | 1.17 |
| square-2.24hz | 0.04 | 1505.8 | 0.0000 | 2705 | 5112 | 2.96 | 1.57 |
| random-walk | 12.62 | 5.1 | 0.0015 | 1422 | 4395 | 5.63 | 1.82 |
| chirp | 7.81 | 8.2 | 0.0010 | 1069 | 3894 | 7.48 | 2.05 |
| noisy-sine | 5.24 | 12.2 | 0.2334 | 1877 | 5075 | 4.26 | 1.58 |
| sensor-0.1 | 1.78 | 36.0 | 0.0000 | 1284 | 3655 | 6.23 | 2.19 |
| random-walk q0.01 | 9.28 | 6.9 | 0.0000 | 1265 | 4254 | 6.32 | 1.88 |
| noisy-sine q0.1 | 5.24 | 12.2 | 0.2341 | 1873 | 5082 | 4.27 | 1.57 |
| noisy-sine q0.01 via float32 | 5.24 | 12.2 | 0.2341 | 1877 | 5045 | 4.26 | 1.59 |
| **all (1.00 GiB)** | 4.76 | 13.4 | | 1562 | 4465 | 5.12 | 1.79 |
| random-walk q0.05 | 6.52 | 9.8 | 0.0000 | 1375 | 4035 | 5.82 | 1.98 |
| random-walk q1/3 | 3.97 | 16.1 | 0.0000 | 739 | 3392 | 10.82 | 2.36 |
| random-walk adc 10/4095 | 11.34 | 5.6 | 0.0015 | 668 | 4203 | 11.98 | 1.90 |

## noise floor off

| signal | bits/sample | ratio | worst max err (% range) | encode MB/s | decode MB/s | encode µs/block | decode µs/block |
|---|---|---|---|---|---|---|---|
| linear | 0.02 | 3137.3 | 0.0000 | 3966 | 5838 | 2.02 | 1.37 |
| quadratic | 0.25 | 257.1 | 0.0015 | 2528 | 4587 | 3.16 | 1.74 |
| sin-4.12hz | 2.40 | 26.7 | 0.0010 | 1509 | 3724 | 5.30 | 2.15 |
| sin-9.87hz | 3.42 | 18.7 | 0.0010 | 1177 | 3491 | 6.79 | 2.29 |
| sin-50.3hz | 6.74 | 9.5 | 0.0010 | 1551 | 3681 | 5.16 | 2.17 |
| gauss-spikes | 12.06 | 5.3 | 0.0015 | 1458 | 4743 | 5.49 | 1.69 |
| impulses | 11.69 | 5.5 | 0.0015 | 1804 | 6132 | 4.43 | 1.30 |
| square-2.24hz | 0.04 | 1505.8 | 0.0000 | 3076 | 4925 | 2.60 | 1.62 |
| random-walk | 12.62 | 5.1 | 0.0015 | 1694 | 4314 | 4.72 | 1.85 |
| chirp | 7.81 | 8.2 | 0.0010 | 1210 | 3861 | 6.61 | 2.07 |
| noisy-sine | 13.72 | 4.7 | 0.0009 | 1747 | 4117 | 4.58 | 1.94 |
| sensor-0.1 | 1.78 | 36.0 | 0.0000 | 1484 | 3554 | 5.39 | 2.25 |
| random-walk q0.01 | 9.28 | 6.9 | 0.0000 | 1475 | 4169 | 5.42 | 1.92 |
| noisy-sine q0.1 | 8.89 | 7.2 | 0.0000 | 1542 | 3100 | 5.19 | 2.58 |
| noisy-sine q0.01 via float32 | 12.31 | 5.2 | 0.0000 | 1693 | 4341 | 4.73 | 1.84 |
| **all (1.00 GiB)** | 6.87 | 9.3 | | 1676 | 4166 | 4.77 | 1.92 |
| random-walk q0.05 | 6.52 | 9.8 | 0.0000 | 1631 | 4011 | 4.90 | 1.99 |
| random-walk q1/3 | 3.97 | 16.1 | 0.0000 | 807 | 3364 | 9.91 | 2.38 |
| random-walk adc 10/4095 | 11.34 | 5.6 | 0.0015 | 726 | 4212 | 11.03 | 1.90 |

## target 6

| signal | bits/sample | ratio | worst max err (% range) | encode MB/s | decode MB/s | encode µs/block | decode µs/block |
|---|---|---|---|---|---|---|---|
| linear | 0.02 | 3137.3 | 0.0000 | 1222 | 5838 | 6.55 | 1.37 |
| quadratic | 0.25 | 257.1 | 0.0015 | 1647 | 4577 | 4.86 | 1.75 |
| sin-4.12hz | 2.40 | 26.7 | 0.0010 | 1156 | 3720 | 6.92 | 2.15 |
| sin-9.87hz | 3.42 | 18.7 | 0.0010 | 953 | 3481 | 8.40 | 2.30 |
| sin-50.3hz | 0.92 | 69.7 | 0.0625 | 1158 | 5342 | 6.91 | 1.50 |
| gauss-spikes | 5.28 | 12.1 | 1.4661 | 967 | 3618 | 8.27 | 2.21 |
| impulses | 4.76 | 13.5 | 1.2592 | 1662 | 6572 | 4.81 | 1.22 |
| square-2.24hz | 0.04 | 1505.8 | 0.0000 | 1753 | 4926 | 4.56 | 1.62 |
| random-walk | 5.39 | 11.9 | 0.1951 | 1173 | 4698 | 6.82 | 1.70 |
| chirp | 4.68 | 13.7 | 0.0625 | 859 | 3268 | 9.31 | 2.45 |
| noisy-sine | 5.24 | 12.2 | 0.2334 | 1613 | 4909 | 4.96 | 1.63 |
| sensor-0.1 | 1.78 | 36.0 | 0.0000 | 1121 | 3499 | 7.14 | 2.29 |
| random-walk q0.01 | 5.46 | 11.7 | 0.1915 | 1119 | 4662 | 7.15 | 1.72 |
| noisy-sine q0.1 | 5.24 | 12.2 | 0.2341 | 1612 | 4890 | 4.96 | 1.64 |
| noisy-sine q0.01 via float32 | 5.24 | 12.2 | 0.2341 | 1616 | 4905 | 4.95 | 1.63 |
| **all (1.00 GiB)** | 3.34 | 19.2 | | 1243 | 4417 | 6.44 | 1.81 |
| random-walk q0.05 | 5.97 | 10.7 | 0.1914 | 912 | 4393 | 8.77 | 1.82 |
| random-walk q1/3 | 3.97 | 16.1 | 0.0000 | 440 | 3367 | 18.20 | 2.38 |
| random-walk adc 10/4095 | 5.42 | 11.8 | 0.1950 | 708 | 4668 | 11.30 | 1.71 |

## effort 1

| signal | bits/sample | ratio | worst max err (% range) | encode MB/s | decode MB/s | encode µs/block | decode µs/block |
|---|---|---|---|---|---|---|---|
| linear | 0.05 | 1300.8 | 0.0000 | 3981 | 5251 | 2.01 | 1.52 |
| quadratic | 0.25 | 252.5 | 0.0015 | 2727 | 4568 | 2.93 | 1.75 |
| sin-4.12hz | 2.50 | 25.6 | 0.0010 | 1734 | 3499 | 4.61 | 2.29 |
| sin-9.87hz | 3.65 | 17.6 | 0.0010 | 1796 | 3856 | 4.45 | 2.07 |
| sin-50.3hz | 6.88 | 9.3 | 0.0010 | 2003 | 3688 | 3.99 | 2.17 |
| gauss-spikes | 6.59 | 9.7 | 1.4661 | 2172 | 3929 | 3.68 | 2.04 |
| impulses | 4.74 | 13.5 | 1.2592 | 2682 | 6159 | 2.98 | 1.30 |
| square-2.24hz | 0.07 | 982.6 | 0.0000 | 3220 | 5229 | 2.48 | 1.53 |
| random-walk | 12.66 | 5.1 | 0.0015 | 1956 | 4307 | 4.09 | 1.86 |
| chirp | 7.90 | 8.1 | 0.0010 | 1828 | 3970 | 4.38 | 2.02 |
| noisy-sine | 5.24 | 12.2 | 0.2334 | 2747 | 4515 | 2.91 | 1.77 |
| sensor-0.1 | 2.18 | 29.4 | 0.0000 | 1727 | 3029 | 4.63 | 2.64 |
| random-walk q0.01 | 9.30 | 6.9 | 0.0000 | 1988 | 4031 | 4.02 | 1.98 |
| noisy-sine q0.1 | 5.24 | 12.2 | 0.2341 | 2751 | 4523 | 2.91 | 1.77 |
| noisy-sine q0.01 via float32 | 5.25 | 12.2 | 0.2341 | 2745 | 4495 | 2.91 | 1.78 |
| **all (1.00 GiB)** | 4.83 | 13.2 | | 2264 | 4212 | 3.53 | 1.90 |
| random-walk q0.05 | 6.53 | 9.8 | 0.0000 | 1804 | 3833 | 4.44 | 2.09 |
| random-walk q1/3 | 3.79 | 16.9 | 0.0000 | 1089 | 4630 | 7.35 | 1.73 |
| random-walk adc 10/4095 | 11.36 | 5.6 | 0.0015 | 959 | 4248 | 8.34 | 1.88 |

## effort 5

| signal | bits/sample | ratio | worst max err (% range) | encode MB/s | decode MB/s | encode µs/block | decode µs/block |
|---|---|---|---|---|---|---|---|
| linear | 0.02 | 3000.0 | 0.0000 | 2741 | 7323 | 2.92 | 1.09 |
| quadratic | 0.27 | 240.7 | 0.0015 | 1738 | 4559 | 4.60 | 1.75 |
| sin-4.12hz | 2.38 | 26.9 | 0.0010 | 1207 | 3711 | 6.63 | 2.16 |
| sin-9.87hz | 3.40 | 18.8 | 0.0010 | 913 | 3484 | 8.76 | 2.30 |
| sin-50.3hz | 6.22 | 10.3 | 0.0010 | 1078 | 3024 | 7.42 | 2.65 |
| gauss-spikes | 6.46 | 9.9 | 1.4661 | 1293 | 3876 | 6.19 | 2.06 |
| impulses | 4.72 | 13.5 | 1.2592 | 1753 | 6563 | 4.56 | 1.22 |
| square-2.24hz | 0.04 | 1434.4 | 0.0000 | 2389 | 4826 | 3.35 | 1.66 |
| random-walk | 12.29 | 5.2 | 0.0015 | 1023 | 3822 | 7.82 | 2.09 |
| chirp | 7.76 | 8.2 | 0.0010 | 947 | 3510 | 8.45 | 2.28 |
| noisy-sine | 5.22 | 12.3 | 0.2334 | 1597 | 4825 | 5.01 | 1.66 |
| sensor-0.1 | 1.76 | 36.3 | 0.0000 | 1151 | 3444 | 6.95 | 2.32 |
| random-walk q0.01 | 8.91 | 7.2 | 0.0000 | 1206 | 3766 | 6.63 | 2.12 |
| noisy-sine q0.1 | 5.22 | 12.3 | 0.2341 | 1589 | 4774 | 5.04 | 1.68 |
| noisy-sine q0.01 via float32 | 5.22 | 12.3 | 0.2341 | 1591 | 4828 | 5.03 | 1.66 |
| **all (1.00 GiB)** | 4.66 | 13.7 | | 1343 | 4182 | 5.96 | 1.91 |
| random-walk q0.05 | 6.49 | 9.9 | 0.0000 | 1239 | 4081 | 6.46 | 1.96 |
| random-walk q1/3 | 3.87 | 16.6 | 0.0000 | 680 | 3584 | 11.76 | 2.23 |
| random-walk adc 10/4095 | 10.96 | 5.8 | 0.0015 | 590 | 3743 | 13.56 | 2.14 |

## effort 9

| signal | bits/sample | ratio | worst max err (% range) | encode MB/s | decode MB/s | encode µs/block | decode µs/block |
|---|---|---|---|---|---|---|---|
| linear | 0.02 | 3038.0 | 0.0000 | 1967 | 7304 | 4.07 | 1.10 |
| quadratic | 0.25 | 261.2 | 0.0015 | 548 | 3703 | 14.61 | 2.16 |
| sin-4.12hz | 2.24 | 28.5 | 0.0010 | 239 | 3663 | 33.46 | 2.18 |
| sin-9.87hz | 3.26 | 19.6 | 0.0010 | 215 | 3311 | 37.14 | 2.42 |
| sin-50.3hz | 6.09 | 10.5 | 0.0010 | 292 | 3156 | 27.40 | 2.54 |
| gauss-spikes | 6.41 | 10.0 | 1.4661 | 432 | 3927 | 18.53 | 2.04 |
| impulses | 4.72 | 13.5 | 1.2592 | 510 | 6567 | 15.69 | 1.22 |
| square-2.24hz | 0.04 | 1679.1 | 0.0000 | 1516 | 4413 | 5.28 | 1.81 |
| random-walk | 12.26 | 5.2 | 0.0015 | 274 | 3837 | 29.17 | 2.08 |
| chirp | 7.69 | 8.3 | 0.0010 | 274 | 3539 | 29.15 | 2.26 |
| noisy-sine | 5.22 | 12.3 | 0.2334 | 462 | 4816 | 17.32 | 1.66 |
| sensor-0.1 | 1.73 | 37.0 | 0.0000 | 206 | 3365 | 38.91 | 2.38 |
| random-walk q0.01 | 8.89 | 7.2 | 0.0000 | 213 | 3731 | 37.64 | 2.14 |
| noisy-sine q0.1 | 5.22 | 12.3 | 0.2341 | 463 | 4788 | 17.27 | 1.67 |
| noisy-sine q0.01 via float32 | 5.22 | 12.3 | 0.2341 | 460 | 4824 | 17.39 | 1.66 |
| **all (1.00 GiB)** | 4.62 | 13.9 | | 350 | 4093 | 22.87 | 1.95 |
| random-walk q0.05 | 6.49 | 9.9 | 0.0000 | 690 | 3973 | 11.60 | 2.01 |
| random-walk q1/3 | 3.85 | 16.6 | 0.0000 | 241 | 3434 | 33.13 | 2.33 |
| random-walk adc 10/4095 | 10.94 | 5.9 | 0.0015 | 176 | 3786 | 45.50 | 2.11 |

## Threads (default params; one signal type per task; the kernels release the GIL)

| threads | encode MB/s | decode MB/s | encode speedup |
|---|---|---|---|
| 1 | 1556 | 4000 | 1.0x |
| 4 | 5246 | 14744 | 3.4x |
| 8 | 5510 | 12857 | 3.5x |
