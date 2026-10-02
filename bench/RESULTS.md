# fluxcode gigabyte benchmark (bench/bench_gb.py, 2026-10-01, AC power; decode is per unit, as rows are read)

One complete run at 8bfc6c6 (later commits change only docs and tests), after the int16
`grid_params` column (13 column bytes per block instead of 19). Before the run: load average 1.7.
Against the previous run the same day (5117ff8): every size is within 0.01 bits/sample, and
times are within run-to-run noise except linear decode (1.30 → 1.67 µs/block): its byte-plane
unit grew from 151 to 158 bytes, a tie with bit planes, and ties keep bit planes, which decode
slower. That is zstd's block boundaries moving with the shorter columns, not the decoder
(decoding each layout takes the same time as before).

Since then ties go to byte planes (2026-10-01, after this run): linear's default unit now
decodes from byte planes again, about 25% faster than from bit planes in an A/B on battery.
Only that unit changed among the golden cases; the tables below predate it and weren't rerun
(the machine was on battery).

Apple M3, Darwin 25.6.0, Python 3.11.13, numpy 2.4.6

15 signal types x 149 minutes = 1.00 GiB of float64 (134,100 blocks); best of 3.

(generated in 5 s)

## default

| signal | bits/sample | ratio | worst max err (% range) | encode MB/s | decode MB/s | encode µs/block | decode µs/block |
|---|---|---|---|---|---|---|---|
| linear | 0.02 | 3038.0 | 0.0000 | 2916 | 4799 | 2.74 | 1.67 |
| quadratic | 0.25 | 257.1 | 0.0015 | 2040 | 4578 | 3.92 | 1.75 |
| sin-4.12hz | 2.40 | 26.7 | 0.0010 | 1295 | 3725 | 6.18 | 2.15 |
| sin-9.87hz | 3.42 | 18.7 | 0.0010 | 1055 | 3850 | 7.58 | 2.08 |
| sin-50.3hz | 6.74 | 9.5 | 0.0010 | 1347 | 3709 | 5.94 | 2.16 |
| gauss-spikes | 6.63 | 9.7 | 1.4661 | 1579 | 4233 | 5.07 | 1.89 |
| impulses | 4.76 | 13.5 | 1.2592 | 1952 | 6573 | 4.10 | 1.22 |
| square-2.24hz | 0.04 | 1438.2 | 0.0000 | 2737 | 6767 | 2.92 | 1.18 |
| random-walk | 12.62 | 5.1 | 0.0015 | 1431 | 4312 | 5.59 | 1.86 |
| chirp | 7.81 | 8.2 | 0.0010 | 1065 | 3924 | 7.51 | 2.04 |
| noisy-sine | 5.24 | 12.2 | 0.2334 | 1893 | 4906 | 4.23 | 1.63 |
| sensor-0.1 | 1.78 | 36.0 | 0.0000 | 1263 | 3582 | 6.33 | 2.23 |
| random-walk q0.01 | 9.28 | 6.9 | 0.0000 | 1259 | 4166 | 6.36 | 1.92 |
| noisy-sine q0.1 | 5.24 | 12.2 | 0.2341 | 1897 | 5166 | 4.22 | 1.55 |
| noisy-sine q0.01 via float32 | 5.24 | 12.2 | 0.2341 | 1888 | 4907 | 4.24 | 1.63 |
| **all (1.00 GiB)** | 4.76 | 13.4 | | 1560 | 4454 | 5.13 | 1.80 |

## noise floor off

| signal | bits/sample | ratio | worst max err (% range) | encode MB/s | decode MB/s | encode µs/block | decode µs/block |
|---|---|---|---|---|---|---|---|
| linear | 0.02 | 3038.0 | 0.0000 | 3805 | 4855 | 2.10 | 1.65 |
| quadratic | 0.25 | 257.1 | 0.0015 | 2618 | 4602 | 3.06 | 1.74 |
| sin-4.12hz | 2.40 | 26.7 | 0.0010 | 1508 | 3732 | 5.30 | 2.14 |
| sin-9.87hz | 3.42 | 18.7 | 0.0010 | 1204 | 3788 | 6.64 | 2.11 |
| sin-50.3hz | 6.74 | 9.5 | 0.0010 | 1567 | 3724 | 5.10 | 2.15 |
| gauss-spikes | 12.06 | 5.3 | 0.0015 | 1452 | 4787 | 5.51 | 1.67 |
| impulses | 11.69 | 5.5 | 0.0015 | 1804 | 6243 | 4.44 | 1.28 |
| square-2.24hz | 0.04 | 1438.2 | 0.0000 | 3545 | 6912 | 2.26 | 1.16 |
| random-walk | 12.62 | 5.1 | 0.0015 | 1709 | 4426 | 4.68 | 1.81 |
| chirp | 7.81 | 8.2 | 0.0010 | 1215 | 3916 | 6.58 | 2.04 |
| noisy-sine | 13.72 | 4.7 | 0.0009 | 1746 | 4165 | 4.58 | 1.92 |
| sensor-0.1 | 1.78 | 36.0 | 0.0000 | 1460 | 3511 | 5.48 | 2.28 |
| random-walk q0.01 | 9.28 | 6.9 | 0.0000 | 1450 | 4174 | 5.52 | 1.92 |
| noisy-sine q0.1 | 8.89 | 7.2 | 0.0000 | 1521 | 3105 | 5.26 | 2.58 |
| noisy-sine q0.01 via float32 | 12.32 | 5.2 | 0.0000 | 1663 | 4363 | 4.81 | 1.83 |
| **all (1.00 GiB)** | 6.87 | 9.3 | | 1683 | 4244 | 4.75 | 1.89 |

## target 6

| signal | bits/sample | ratio | worst max err (% range) | encode MB/s | decode MB/s | encode µs/block | decode µs/block |
|---|---|---|---|---|---|---|---|
| linear | 0.02 | 3038.0 | 0.0000 | 2286 | 4837 | 3.50 | 1.65 |
| quadratic | 0.25 | 257.1 | 0.0015 | 1720 | 4596 | 4.65 | 1.74 |
| sin-4.12hz | 2.40 | 26.7 | 0.0010 | 1171 | 4054 | 6.83 | 1.97 |
| sin-9.87hz | 3.42 | 18.7 | 0.0010 | 974 | 3830 | 8.21 | 2.09 |
| sin-50.3hz | 0.92 | 69.7 | 0.0625 | 1206 | 5444 | 6.64 | 1.47 |
| gauss-spikes | 5.27 | 12.1 | 1.4661 | 985 | 3659 | 8.12 | 2.19 |
| impulses | 4.76 | 13.5 | 1.2592 | 1683 | 6646 | 4.75 | 1.20 |
| square-2.24hz | 0.04 | 1438.2 | 0.0000 | 2192 | 6823 | 3.65 | 1.17 |
| random-walk | 5.39 | 11.9 | 0.1951 | 1209 | 4790 | 6.62 | 1.67 |
| chirp | 4.68 | 13.7 | 0.0625 | 880 | 3292 | 9.09 | 2.43 |
| noisy-sine | 5.24 | 12.2 | 0.2334 | 1640 | 4950 | 4.88 | 1.62 |
| sensor-0.1 | 1.78 | 36.0 | 0.0000 | 1161 | 3552 | 6.89 | 2.25 |
| random-walk q0.01 | 6.21 | 10.3 | 0.1750 | 1163 | 4470 | 6.88 | 1.79 |
| noisy-sine q0.1 | 5.24 | 12.2 | 0.2341 | 1629 | 4907 | 4.91 | 1.63 |
| noisy-sine q0.01 via float32 | 5.24 | 12.2 | 0.2341 | 1629 | 4907 | 4.91 | 1.63 |
| **all (1.00 GiB)** | 3.39 | 18.9 | | 1326 | 4527 | 6.04 | 1.77 |

## bit planes only

| signal | bits/sample | ratio | worst max err (% range) | encode MB/s | decode MB/s | encode µs/block | decode µs/block |
|---|---|---|---|---|---|---|---|
| linear | 0.02 | 3038.0 | 0.0000 | 3239 | 4847 | 2.47 | 1.65 |
| quadratic | 0.25 | 257.1 | 0.0015 | 3092 | 4598 | 2.59 | 1.74 |
| sin-4.12hz | 2.73 | 23.4 | 0.0010 | 2552 | 3803 | 3.14 | 2.10 |
| sin-9.87hz | 3.64 | 17.6 | 0.0010 | 2242 | 3325 | 3.57 | 2.41 |
| sin-50.3hz | 6.74 | 9.5 | 0.0010 | 2182 | 3735 | 3.67 | 2.14 |
| gauss-spikes | 6.63 | 9.7 | 1.4661 | 2183 | 4267 | 3.66 | 1.87 |
| impulses | 5.05 | 12.7 | 1.2592 | 2690 | 4524 | 2.97 | 1.77 |
| square-2.24hz | 0.05 | 1219.3 | 0.0000 | 3113 | 4402 | 2.57 | 1.82 |
| random-walk | 12.62 | 5.1 | 0.0015 | 2047 | 4428 | 3.91 | 1.81 |
| chirp | 7.81 | 8.2 | 0.0010 | 1651 | 3965 | 4.85 | 2.02 |
| noisy-sine | 5.44 | 11.8 | 0.2334 | 2456 | 3253 | 3.26 | 2.46 |
| sensor-0.1 | 1.78 | 36.0 | 0.0000 | 2157 | 3516 | 3.71 | 2.28 |
| random-walk q0.01 | 9.28 | 6.9 | 0.0000 | 2078 | 4168 | 3.85 | 1.92 |
| noisy-sine q0.1 | 5.44 | 11.8 | 0.2341 | 2451 | 3244 | 3.26 | 2.47 |
| noisy-sine q0.01 via float32 | 5.45 | 11.8 | 0.2341 | 2455 | 3358 | 3.26 | 2.38 |
| **all (1.00 GiB)** | 4.86 | 13.2 | | 2365 | 3892 | 3.38 | 2.06 |

## Threads (default params; one signal type per task; the kernels release the GIL)

| threads | encode MB/s | decode MB/s | encode speedup |
|---|---|---|---|
| 1 | 1569 | 4220 | 1.0x |
| 4 | 5166 | 14799 | 3.3x |
| 8 | 5334 | 12765 | 3.4x |
