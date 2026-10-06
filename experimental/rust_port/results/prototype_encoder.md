# Rust encoder prototype: results

> **Measured on battery power** (the laptop ran at about half clock; python's encode was ~25% slower than in the AC run in
> `compress_bench.md`). Absolute ns/sample and MiB/s are not comparable to AC runs; ratios within a table are (both paths
> ran back to back in one process).

Apple Silicon (arm64, 8 logical CPUs), `uv run python bench/rust_encoder.py`. Three Rust pieces, switched
together by `FLUXCODE_ENCODER=rust` (the benchmark toggles them in-process):

- compress: `compress_group` (layout choice, body packing, flush points and the flushed zstd frame in one
  GIL-free call, thread-local zstd context) for block groups without a time axis;
- kernel: `encode_group` (block statistics, non-finite imputation, noise floor, decimal detection,
  quantization, order choice, residuals, bit target), same arguments and outputs as the numba kernel;
- full: both. The time-axis encoding (`_time`) and splice/update stay Python/numba.

Equality: the block groups are byte-identical to the numba path at every effort 1-9 on every signal and on all 27
golden cases, and flags, params, anchors, residuals, codes, min and max match exactly in a 530-case
differential test (edge cases included: subnormal, near +-DBL_MAX, non-finite, constant, decimal). Only
the block means (and so the golden hash) differ in the last bits: numba sums with fastmath in an order LLVM
picks; here the sums are fixed-lane (`wide::f64x4`) sums, deterministic on every platform.

The kernel is at parity with numba (0.9-1.04x) on finite data and 0.68x on blocks with non-finite values
(`noise_finite` and `fill_nonfinite` are scalar here). Profiling found LLVM did not vectorize array-lane
float reductions on aarch64: explicit `wide` SIMD took the noise estimate from 2.7 to 2.2 ns/sample
(numba 1.7).

## Byte-identical block groups, every effort 1-9 x every signal (1 minute each), full Rust vs python

all identical

## Kernel only (_encoder.encode_group: analysis, quantize, residuals), single thread, best of 5

| signal | numba | rust | speedup |
|---|---:|---:|---:|
| linear | 2.86 ns/sample | 2.75 ns/sample | 1.04x |
| quadratic | 2.79 ns/sample | 3.05 ns/sample | 0.91x |
| sin-4.12hz | 2.83 ns/sample | 3.10 ns/sample | 0.92x |
| sin-9.87hz | 2.84 ns/sample | 3.09 ns/sample | 0.92x |
| sin-50.3hz | 2.84 ns/sample | 3.09 ns/sample | 0.92x |
| gauss-spikes | 2.78 ns/sample | 3.05 ns/sample | 0.91x |
| impulses | 2.74 ns/sample | 2.99 ns/sample | 0.92x |
| square-2.24hz | 2.88 ns/sample | 2.76 ns/sample | 1.04x |
| random-walk | 2.78 ns/sample | 3.02 ns/sample | 0.92x |
| chirp | 2.84 ns/sample | 3.09 ns/sample | 0.92x |
| noisy-sine | 2.75 ns/sample | 3.01 ns/sample | 0.91x |
| sensor-0.1 | 3.25 ns/sample | 3.29 ns/sample | 0.99x |
| with-nonfinite | 4.19 ns/sample | 6.17 ns/sample | 0.68x |

## fluxcode.encode, effort 2, single thread (best of 5; 4 min x 60k samples), ns/sample

| signal | python | compress only | kernel only | full rust | full speedup |
|---|---:|---:|---:|---:|---:|
| linear | 3.82 | 3.74 | 3.68 | 3.63 | 1.05x |
| quadratic | 4.92 | 5.55 | 5.18 | 5.82 | 0.85x |
| sin-4.12hz | 8.08 | 8.13 | 8.17 | 8.34 | 0.97x |
| sin-9.87hz | 9.67 | 9.63 | 9.73 | 9.75 | 0.99x |
| sin-50.3hz | 6.21 | 6.89 | 6.39 | 7.10 | 0.87x |
| gauss-spikes | 6.58 | 7.24 | 6.85 | 7.52 | 0.88x |
| impulses | 5.02 | 5.01 | 5.26 | 5.25 | 0.96x |
| square-2.24hz | 3.92 | 3.87 | 3.80 | 3.75 | 1.04x |
| random-walk | 7.04 | 7.75 | 7.27 | 7.96 | 0.88x |
| chirp | 8.56 | 9.34 | 8.77 | 9.45 | 0.91x |
| noisy-sine | 4.87 | 4.86 | 5.14 | 5.13 | 0.95x |
| sensor-0.1 | 7.77 | 7.87 | 7.73 | 7.79 | 1.00x |
| with-nonfinite | 13.15 | 12.70 | 15.08 | 15.16 | 0.87x |

## fluxcode.encode, effort 4, single thread (best of 5; 4 min x 60k samples), ns/sample

| signal | python | compress only | kernel only | full rust | full speedup |
|---|---:|---:|---:|---:|---:|
| linear | 4.15 | 4.31 | 4.02 | 4.21 | 0.99x |
| quadratic | 5.55 | 6.28 | 5.80 | 6.52 | 0.85x |
| sin-4.12hz | 8.37 | 8.71 | 8.53 | 8.89 | 0.94x |
| sin-9.87hz | 10.05 | 10.23 | 10.34 | 10.35 | 0.97x |
| sin-50.3hz | 8.72 | 9.64 | 8.91 | 9.67 | 0.90x |
| gauss-spikes | 8.16 | 8.83 | 8.40 | 9.10 | 0.90x |
| impulses | 5.32 | 5.52 | 5.58 | 5.76 | 0.92x |
| square-2.24hz | 4.11 | 4.31 | 4.00 | 4.19 | 0.98x |
| random-walk | 8.08 | 8.68 | 8.27 | 8.91 | 0.91x |
| chirp | 10.85 | 11.68 | 11.04 | 11.69 | 0.93x |
| noisy-sine | 5.27 | 5.44 | 5.53 | 5.71 | 0.92x |
| sensor-0.1 | 8.03 | 8.36 | 7.95 | 8.22 | 0.98x |
| with-nonfinite | 15.35 | 14.94 | 17.65 | 17.45 | 0.88x |

## Thread scaling: one minute-series per task, MiB/s of float64 input (best of 5)

| effort | threads | python | compress only | kernel only | full rust | full speedup |
|---|---:|---:|---:|---:|---:|---:|
| 2 | 1 | 962 | 899 | 944 | 880 | 0.91x |
| 2 | 4 | 3308 | 3298 | 3410 | 3322 | 1.00x |
| 2 | 8 | 3304 | 4107 | 2876 | 3455 | 1.05x |
| 4 | 1 | 802 | 772 | 804 | 753 | 0.94x |
| 4 | 4 | 1940 | 2810 | 1913 | 2818 | 1.45x |
| 4 | 8 | 1515 | 3757 | 1506 | 3614 | 2.39x |

