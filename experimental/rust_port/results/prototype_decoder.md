# Rust decoder prototype: results

> **Measured on battery power** (the laptop ran at about half clock; python's encode was ~25% slower than in the AC run in
> `compress_bench.md`). Absolute ns/sample and MiB/s are not comparable to AC runs; ratios within a table are (both paths
> ran back to back in one process).

Apple Silicon (arm64), single thread, `uv run python bench/rust_decoder.py`. Rust: rust/ (PyO3,
release, fat LTO), selected with `FLUXCODE_DECODER=rust`: `decode_unit` and `check_unit` kernels, and
`decompress_body` (zstd frame, layout, size checks and validation in one GIL-free call, used by
`_unit.decompress`).

Where the time goes (decompression only, ns/sample): sin-4.12hz python-zstandard 1.90 vs
`decompress_body` (zstd + layout + check) 1.97; random-walk 0.78 vs 0.84; linear 0.07 vs 0.12. The
python-to-zstd seam costs nothing measurable on decode: zstd's own decompression is the cost, and
the Rust call is the same speed (slightly slower: it zeroes the body buffer and parses the layout).

| signal | planes | numba | rust | speedup | bit-identical |
|---|---|---:|---:|---:|---|
| linear | byte | 1.40 ns/sample | 1.28 ns/sample | 1.09x | yes |
| quadratic | bit | 2.38 ns/sample | 2.19 ns/sample | 1.09x | yes |
| sin-4.12hz | byte | 1.48 ns/sample | 1.35 ns/sample | 1.10x | yes |
| sin-9.87hz | byte | 1.46 ns/sample | 1.35 ns/sample | 1.08x | yes |
| sin-50.3hz | bit | 2.45 ns/sample | 2.22 ns/sample | 1.11x | yes |
| gauss-spikes | bit | 2.12 ns/sample | 1.89 ns/sample | 1.12x | yes |
| impulses | byte | 0.60 ns/sample | 0.49 ns/sample | 1.23x | yes |
| square-2.24hz | byte | 1.40 ns/sample | 1.28 ns/sample | 1.09x | yes |
| random-walk | bit | 2.45 ns/sample | 2.20 ns/sample | 1.11x | yes |
| chirp | bit | 2.45 ns/sample | 2.22 ns/sample | 1.11x | yes |
| noisy-sine | byte | 1.44 ns/sample | 1.33 ns/sample | 1.09x | yes |
| sensor-0.1 | byte | 1.53 ns/sample | 1.41 ns/sample | 1.09x | yes |
| with-nonfinite | bit | 3.31 ns/sample | 2.46 ns/sample | 1.35x | yes |

Total: numba 5.9 ms, rust 5.2 ms (1.13x)

## End to end fluxcode.decode_unit (zstd included), effort 4

| signal | numba | rust | speedup |
|---|---:|---:|---:|
| linear | 1.61 ns/sample | 1.49 ns/sample | 1.08x |
| quadratic | 2.76 ns/sample | 2.56 ns/sample | 1.08x |
| sin-4.12hz | 3.50 ns/sample | 3.41 ns/sample | 1.03x |
| sin-9.87hz | 3.67 ns/sample | 3.55 ns/sample | 1.03x |
| sin-50.3hz | 4.09 ns/sample | 3.84 ns/sample | 1.07x |
| gauss-spikes | 3.19 ns/sample | 2.96 ns/sample | 1.08x |
| impulses | 1.73 ns/sample | 1.59 ns/sample | 1.09x |
| square-2.24hz | 1.68 ns/sample | 1.54 ns/sample | 1.09x |
| random-walk | 3.37 ns/sample | 3.14 ns/sample | 1.07x |
| chirp | 3.48 ns/sample | 3.26 ns/sample | 1.07x |
| noisy-sine | 2.60 ns/sample | 2.46 ns/sample | 1.06x |
| sensor-0.1 | 3.47 ns/sample | 3.34 ns/sample | 1.04x |
| with-nonfinite | 6.75 ns/sample | 5.88 ns/sample | 1.15x |

