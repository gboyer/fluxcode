power: Now drawing from 'AC Power'; libzstd: rust (1, 5, 7), python-zstandard (1, 5, 7)

## Units byte-identical, every effort 1-9 x every signal (1 minute each)

all identical

## fluxcode.encode, effort 2, single thread (best of 5; 4 min x 60k samples)

| signal | python | rust | speedup |
|---|---:|---:|---:|
| linear | 2.18 ns/sample | 2.14 ns/sample | 1.02x |
| quadratic | 2.81 ns/sample | 2.58 ns/sample | 1.09x |
| sin-4.12hz | 4.57 ns/sample | 4.64 ns/sample | 0.98x |
| sin-9.87hz | 5.44 ns/sample | 5.41 ns/sample | 1.01x |
| sin-50.3hz | 3.64 ns/sample | 3.33 ns/sample | 1.09x |
| gauss-spikes | 3.77 ns/sample | 3.52 ns/sample | 1.07x |
| impulses | 2.87 ns/sample | 2.85 ns/sample | 1.01x |
| square-2.24hz | 2.26 ns/sample | 2.20 ns/sample | 1.03x |
| random-walk | 4.03 ns/sample | 3.78 ns/sample | 1.07x |
| chirp | 4.94 ns/sample | 4.64 ns/sample | 1.06x |
| noisy-sine | 2.78 ns/sample | 2.75 ns/sample | 1.01x |
| sensor-0.1 | 4.47 ns/sample | 4.49 ns/sample | 0.99x |

## fluxcode.encode, effort 4, single thread (best of 5; 4 min x 60k samples)

| signal | python | rust | speedup |
|---|---:|---:|---:|
| linear | 2.78 ns/sample | 2.40 ns/sample | 1.16x |
| quadratic | 3.92 ns/sample | 3.61 ns/sample | 1.09x |
| sin-4.12hz | 5.78 ns/sample | 5.52 ns/sample | 1.05x |
| sin-9.87hz | 7.05 ns/sample | 6.84 ns/sample | 1.03x |
| sin-50.3hz | 5.06 ns/sample | 4.91 ns/sample | 1.03x |
| gauss-spikes | 4.89 ns/sample | 4.54 ns/sample | 1.08x |
| impulses | 3.99 ns/sample | 3.67 ns/sample | 1.09x |
| square-2.24hz | 2.93 ns/sample | 2.58 ns/sample | 1.13x |
| random-walk | 5.32 ns/sample | 4.97 ns/sample | 1.07x |
| chirp | 7.18 ns/sample | 6.62 ns/sample | 1.09x |
| noisy-sine | 4.15 ns/sample | 3.89 ns/sample | 1.07x |
| sensor-0.1 | 5.98 ns/sample | 5.74 ns/sample | 1.04x |

## fluxcode.encode, effort 5, single thread (best of 5; 4 min x 60k samples)

| signal | python | rust | speedup |
|---|---:|---:|---:|
| linear | 3.25 ns/sample | 2.84 ns/sample | 1.14x |
| quadratic | 4.51 ns/sample | 4.11 ns/sample | 1.10x |
| sin-4.12hz | 6.50 ns/sample | 6.14 ns/sample | 1.06x |
| sin-9.87hz | 8.12 ns/sample | 7.69 ns/sample | 1.06x |
| sin-50.3hz | 6.89 ns/sample | 6.72 ns/sample | 1.02x |
| gauss-spikes | 6.10 ns/sample | 5.56 ns/sample | 1.10x |
| impulses | 4.69 ns/sample | 4.22 ns/sample | 1.11x |
| square-2.24hz | 3.25 ns/sample | 2.86 ns/sample | 1.14x |
| random-walk | 7.56 ns/sample | 6.92 ns/sample | 1.09x |
| chirp | 8.91 ns/sample | 8.13 ns/sample | 1.10x |
| noisy-sine | 5.04 ns/sample | 4.52 ns/sample | 1.11x |
| sensor-0.1 | 6.62 ns/sample | 6.03 ns/sample | 1.10x |

## Thread scaling, MiB/s of float64 input (best of 5)

| effort | threads | python | rust | speedup |
|---|---:|---:|---:|---:|
| 2 | 1 | 1661 | 1716 | 1.03x |
| 2 | 4 | 5786 | 5879 | 1.02x |
| 2 | 8 | 4991 | 6158 | 1.23x |
| 4 | 1 | 1244 | 1315 | 1.06x |
| 4 | 4 | 4222 | 4643 | 1.10x |
| 4 | 8 | 4038 | 5410 | 1.34x |
| 5 | 1 | 997 | 1075 | 1.08x |
| 5 | 4 | 2406 | 3808 | 1.58x |
| 5 | 8 | 1871 | 4819 | 2.58x |

