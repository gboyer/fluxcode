power: Now drawing from 'AC Power'; libzstd: rust (1, 5, 7), python-zstandard (1, 5, 7)

## Block groups byte-identical, every effort 1-9 x every signal (1 minute each)

all identical

## fluxcode.encode, effort 2, single thread (best of 5; 4 min x 60k samples)

| signal | python | rust | speedup |
|---|---:|---:|---:|
| linear | 2.11 ns/sample | 2.10 ns/sample | 1.01x |
| quadratic | 2.81 ns/sample | 2.56 ns/sample | 1.10x |
| sin-4.12hz | 4.53 ns/sample | 4.55 ns/sample | 1.00x |
| sin-9.87hz | 5.42 ns/sample | 5.36 ns/sample | 1.01x |
| sin-50.3hz | 3.63 ns/sample | 3.34 ns/sample | 1.09x |
| gauss-spikes | 3.75 ns/sample | 3.51 ns/sample | 1.07x |
| impulses | 2.88 ns/sample | 2.82 ns/sample | 1.02x |
| square-2.24hz | 2.25 ns/sample | 2.17 ns/sample | 1.03x |
| random-walk | 4.02 ns/sample | 3.80 ns/sample | 1.06x |
| chirp | 5.00 ns/sample | 4.66 ns/sample | 1.07x |
| noisy-sine | 2.78 ns/sample | 2.73 ns/sample | 1.02x |
| sensor-0.1 | 4.40 ns/sample | 4.42 ns/sample | 1.00x |

## fluxcode.encode, effort 4, single thread (best of 5; 4 min x 60k samples)

| signal | python | rust | speedup |
|---|---:|---:|---:|
| linear | 2.75 ns/sample | 2.51 ns/sample | 1.10x |
| quadratic | 3.90 ns/sample | 3.72 ns/sample | 1.05x |
| sin-4.12hz | 5.76 ns/sample | 5.64 ns/sample | 1.02x |
| sin-9.87hz | 7.04 ns/sample | 6.97 ns/sample | 1.01x |
| sin-50.3hz | 5.04 ns/sample | 5.06 ns/sample | 1.00x |
| gauss-spikes | 4.85 ns/sample | 4.66 ns/sample | 1.04x |
| impulses | 3.98 ns/sample | 3.81 ns/sample | 1.04x |
| square-2.24hz | 2.91 ns/sample | 2.68 ns/sample | 1.09x |
| random-walk | 5.33 ns/sample | 5.11 ns/sample | 1.04x |
| chirp | 6.96 ns/sample | 6.98 ns/sample | 1.00x |
| noisy-sine | 4.15 ns/sample | 3.93 ns/sample | 1.06x |
| sensor-0.1 | 6.00 ns/sample | 5.71 ns/sample | 1.05x |

## fluxcode.encode, effort 5, single thread (best of 5; 4 min x 60k samples)

| signal | python | rust | speedup |
|---|---:|---:|---:|
| linear | 3.23 ns/sample | 2.94 ns/sample | 1.10x |
| quadratic | 4.50 ns/sample | 4.14 ns/sample | 1.09x |
| sin-4.12hz | 6.51 ns/sample | 5.94 ns/sample | 1.09x |
| sin-9.87hz | 8.10 ns/sample | 7.88 ns/sample | 1.03x |
| sin-50.3hz | 6.89 ns/sample | 6.28 ns/sample | 1.10x |
| gauss-spikes | 6.10 ns/sample | 5.57 ns/sample | 1.09x |
| impulses | 4.67 ns/sample | 4.13 ns/sample | 1.13x |
| square-2.24hz | 3.24 ns/sample | 2.95 ns/sample | 1.10x |
| random-walk | 7.49 ns/sample | 7.16 ns/sample | 1.05x |
| chirp | 8.84 ns/sample | 7.44 ns/sample | 1.19x |
| noisy-sine | 5.02 ns/sample | 4.56 ns/sample | 1.10x |
| sensor-0.1 | 6.57 ns/sample | 6.19 ns/sample | 1.06x |

## Thread scaling, MiB/s of float64 input (best of 5)

| effort | threads | python | rust | speedup |
|---|---:|---:|---:|---:|
| 2 | 1 | 1672 | 1717 | 1.03x |
| 2 | 4 | 5619 | 5939 | 1.06x |
| 2 | 8 | 5005 | 6547 | 1.31x |
| 4 | 1 | 1237 | 1292 | 1.04x |
| 4 | 4 | 4300 | 4526 | 1.05x |
| 4 | 8 | 4121 | 5469 | 1.33x |
| 5 | 1 | 1001 | 1108 | 1.11x |
| 5 | 4 | 2306 | 3929 | 1.70x |
| 5 | 8 | 1858 | 5063 | 2.72x |

