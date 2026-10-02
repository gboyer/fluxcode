power: Now drawing from 'AC Power'; libzstd: rust (1, 5, 7), python-zstandard (1, 5, 7)

## Units byte-identical, every effort 1-9 x every signal (1 minute each)

all identical

## fluxcode.encode, effort 2, single thread (best of 5; 4 min x 60k samples)

| signal | python | rust | speedup |
|---|---:|---:|---:|
| linear | 2.19 ns/sample | 2.15 ns/sample | 1.02x |
| quadratic | 2.86 ns/sample | 2.93 ns/sample | 0.98x |
| sin-4.12hz | 4.69 ns/sample | 4.73 ns/sample | 0.99x |
| sin-9.87hz | 5.54 ns/sample | 5.68 ns/sample | 0.98x |
| sin-50.3hz | 3.67 ns/sample | 3.75 ns/sample | 0.98x |
| gauss-spikes | 3.82 ns/sample | 3.97 ns/sample | 0.96x |
| impulses | 2.92 ns/sample | 2.91 ns/sample | 1.00x |
| square-2.24hz | 2.25 ns/sample | 2.22 ns/sample | 1.01x |
| random-walk | 4.09 ns/sample | 4.17 ns/sample | 0.98x |
| chirp | 5.01 ns/sample | 5.27 ns/sample | 0.95x |
| noisy-sine | 2.82 ns/sample | 2.80 ns/sample | 1.01x |
| sensor-0.1 | 4.42 ns/sample | 4.47 ns/sample | 0.99x |

## fluxcode.encode, effort 4, single thread (best of 5; 4 min x 60k samples)

| signal | python | rust | speedup |
|---|---:|---:|---:|
| linear | 2.75 ns/sample | 2.72 ns/sample | 1.01x |
| quadratic | 3.93 ns/sample | 3.93 ns/sample | 1.00x |
| sin-4.12hz | 5.79 ns/sample | 5.89 ns/sample | 0.98x |
| sin-9.87hz | 7.10 ns/sample | 7.22 ns/sample | 0.98x |
| sin-50.3hz | 5.09 ns/sample | 5.27 ns/sample | 0.96x |
| gauss-spikes | 4.94 ns/sample | 4.91 ns/sample | 1.01x |
| impulses | 4.03 ns/sample | 4.00 ns/sample | 1.01x |
| square-2.24hz | 2.93 ns/sample | 2.89 ns/sample | 1.02x |
| random-walk | 5.38 ns/sample | 5.39 ns/sample | 1.00x |
| chirp | 7.09 ns/sample | 7.32 ns/sample | 0.97x |
| noisy-sine | 4.21 ns/sample | 4.16 ns/sample | 1.01x |
| sensor-0.1 | 6.14 ns/sample | 6.16 ns/sample | 1.00x |

## fluxcode.encode, effort 5, single thread (best of 5; 4 min x 60k samples)

| signal | python | rust | speedup |
|---|---:|---:|---:|
| linear | 3.25 ns/sample | 3.14 ns/sample | 1.04x |
| quadratic | 4.57 ns/sample | 4.49 ns/sample | 1.02x |
| sin-4.12hz | 6.71 ns/sample | 6.63 ns/sample | 1.01x |
| sin-9.87hz | 8.21 ns/sample | 8.22 ns/sample | 1.00x |
| sin-50.3hz | 6.99 ns/sample | 7.05 ns/sample | 0.99x |
| gauss-spikes | 6.21 ns/sample | 5.94 ns/sample | 1.04x |
| impulses | 4.74 ns/sample | 4.57 ns/sample | 1.04x |
| square-2.24hz | 3.25 ns/sample | 3.16 ns/sample | 1.03x |
| random-walk | 7.63 ns/sample | 7.53 ns/sample | 1.01x |
| chirp | 8.99 ns/sample | 8.60 ns/sample | 1.05x |
| noisy-sine | 5.10 ns/sample | 4.93 ns/sample | 1.04x |
| sensor-0.1 | 6.74 ns/sample | 6.65 ns/sample | 1.01x |

## Thread scaling, MiB/s of float64 input (best of 5)

| effort | threads | python | rust | speedup |
|---|---:|---:|---:|---:|
| 2 | 1 | 1652 | 1638 | 0.99x |
| 2 | 4 | 5506 | 5727 | 1.04x |
| 2 | 8 | 4814 | 5639 | 1.17x |
| 4 | 1 | 1221 | 1252 | 1.02x |
| 4 | 4 | 4222 | 4409 | 1.04x |
| 4 | 8 | 3939 | 5120 | 1.30x |
| 5 | 1 | 996 | 1013 | 1.02x |
| 5 | 4 | 2276 | 3611 | 1.59x |
| 5 | 8 | 1643 | 4407 | 2.68x |

