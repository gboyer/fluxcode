# Rust encoder prototype: results (compress stage re-measured)

> **Measured on battery power** (the laptop ran at about half clock; python's encode was ~25% slower than in the AC run in
> `compress_bench.md`). Absolute ns/sample and MiB/s are not comparable to AC runs; ratios within a table are (both paths
> ran back to back in one process).

**Caveat: this run was on battery power (laptop at roughly half clock speed), so absolute ns/sample and
MiB/s are lower than the earlier plugged-in runs and not comparable to them.** The python/rust columns of
one run are measured back to back in one process, so their ratios are meaningful; re-run on AC power
before quoting absolute numbers or the thread scaling (which depends on the clock of every core).

Changes since the previous run: bit-plane packing zigzags into low/high byte scratch and transposes u64
loads (as numba does), the flush density count is a vectorizable sum, and the unit is copied once into
its bytes object. Single-thread encode at effort 4 is now 0.96-1.04x of python on finite data (it was
0.85-0.97x) for the compress stage; the kernel is unchanged.

## Byte-identical units, every effort 1-9 x every signal (1 minute each), full Rust vs python

all identical

## Kernel only (_encoder.encode_unit: analysis, quantize, residuals), single thread, best of 7

| signal | numba | rust | speedup |
|---|---:|---:|---:|
| linear | 2.86 ns/sample | 2.73 ns/sample | 1.05x |
| quadratic | 2.78 ns/sample | 3.04 ns/sample | 0.92x |
| sin-4.12hz | 2.83 ns/sample | 3.08 ns/sample | 0.92x |
| sin-9.87hz | 2.83 ns/sample | 3.09 ns/sample | 0.92x |
| sin-50.3hz | 2.83 ns/sample | 3.08 ns/sample | 0.92x |
| gauss-spikes | 2.78 ns/sample | 3.04 ns/sample | 0.91x |
| impulses | 2.73 ns/sample | 2.98 ns/sample | 0.92x |
| square-2.24hz | 2.88 ns/sample | 2.75 ns/sample | 1.05x |
| random-walk | 2.77 ns/sample | 3.01 ns/sample | 0.92x |
| chirp | 2.83 ns/sample | 3.08 ns/sample | 0.92x |
| noisy-sine | 2.74 ns/sample | 3.00 ns/sample | 0.91x |
| sensor-0.1 | 3.25 ns/sample | 3.28 ns/sample | 0.99x |
| with-nonfinite | 4.18 ns/sample | 6.09 ns/sample | 0.69x |

## fluxcode.encode, effort 2, single thread (best of 7; 4 min x 60k samples), ns/sample

| signal | python | compress only | kernel only | full rust | full speedup |
|---|---:|---:|---:|---:|---:|
| linear | 3.79 | 3.71 | 3.65 | 3.59 | 1.05x |
| quadratic | 4.89 | 5.01 | 5.14 | 5.28 | 0.93x |
| sin-4.12hz | 7.78 | 7.98 | 7.93 | 7.95 | 0.98x |
| sin-9.87hz | 9.58 | 9.28 | 9.80 | 9.22 | 1.04x |
| sin-50.3hz | 6.10 | 6.34 | 6.35 | 6.55 | 0.93x |
| gauss-spikes | 6.53 | 6.68 | 6.79 | 6.93 | 0.94x |
| impulses | 4.97 | 4.95 | 5.22 | 5.21 | 0.96x |
| square-2.24hz | 3.89 | 3.84 | 3.77 | 3.73 | 1.04x |
| random-walk | 6.99 | 7.16 | 7.23 | 7.42 | 0.94x |
| chirp | 8.43 | 8.59 | 8.73 | 8.83 | 0.95x |
| noisy-sine | 4.84 | 4.82 | 5.11 | 5.09 | 0.95x |
| sensor-0.1 | 7.57 | 7.72 | 7.54 | 7.60 | 1.00x |
| with-nonfinite | 12.83 | 12.24 | 14.74 | 14.40 | 0.89x |

## fluxcode.encode, effort 4, single thread (best of 7; 4 min x 60k samples), ns/sample

| signal | python | compress only | kernel only | full rust | full speedup |
|---|---:|---:|---:|---:|---:|
| linear | 4.10 | 4.04 | 3.98 | 3.93 | 1.04x |
| quadratic | 5.52 | 5.49 | 5.74 | 5.76 | 0.96x |
| sin-4.12hz | 8.17 | 8.27 | 8.32 | 8.37 | 0.98x |
| sin-9.87hz | 9.71 | 9.86 | 9.80 | 9.63 | 1.01x |
| sin-50.3hz | 8.94 | 8.61 | 8.94 | 8.84 | 1.01x |
| gauss-spikes | 8.15 | 8.06 | 8.40 | 8.30 | 0.98x |
| impulses | 5.31 | 5.24 | 5.56 | 5.52 | 0.96x |
| square-2.24hz | 4.10 | 4.06 | 3.99 | 3.94 | 1.04x |
| random-walk | 8.03 | 7.93 | 8.28 | 8.18 | 0.98x |
| chirp | 10.80 | 10.78 | 11.02 | 10.88 | 0.99x |
| noisy-sine | 5.24 | 5.18 | 5.52 | 5.44 | 0.96x |
| sensor-0.1 | 7.96 | 8.10 | 7.93 | 7.95 | 1.00x |
| with-nonfinite | 15.23 | 14.69 | 17.11 | 16.71 | 0.91x |

## Thread scaling: one minute-series per task, MiB/s of float64 input (best of 7)

| effort | threads | python | compress only | kernel only | full rust | full speedup |
|---|---:|---:|---:|---:|---:|---:|
| 2 | 1 | 968 | 961 | 926 | 928 | 0.96x |
| 2 | 8 | 3216 | 4430 | 3191 | 4384 | 1.36x |
| 4 | 1 | 823 | 835 | 797 | 812 | 0.99x |
| 4 | 8 | 1517 | 4109 | 1523 | 3945 | 2.60x |

