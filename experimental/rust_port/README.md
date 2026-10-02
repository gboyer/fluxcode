# A Rust port of the encoder and decoder: what it buys

An experiment to see whether porting fluxcode's numba kernels to Rust (PyO3) would make it faster, and
which parts are worth keeping. **Conclusion: only the zstd frame build, as an optional extension**
([`rust/`](../../rust/), the `fluxcode[rust]` extra). The decoder is a few percent faster, the encoder's
analysis kernels are no faster than numba, and the frame build is where Rust changes the picture: it
doesn't hold the GIL, and python-zstandard does inside a block flush.

- **Synthetic data only**, like the rest of this directory; one laptop (Apple Silicon, 8 logical CPUs).
- **Power state matters.** Everything under `results/prototype_*` ran on battery (about half clock, and the
  thread scaling depends on every core's clock). `results/compress_bench.md` ran on AC power. The two
  paths of a run are measured back to back in one process, so ratios hold; absolute numbers don't carry
  over between runs.
- The prototype (all of the below, plus the decoder and kernel ports) is on the `rust-decoder` branch:
  `8c384d2` decode kernels, `c23e297` Rust zstd decompress, `569cd0b` compress stage, `acb411c` encode
  kernels, `f504916` compress-stage tuning. The decoder and kernel benchmarks (`bench/rust_decoder.py`,
  `bench/rust_encoder.py`) live there; here are the compress stage's.

## What was ported, and what it measured

| part | how | result |
|---|---|---|
| decoder kernels (`decode_unit`, `check_unit`) | same signatures behind `FLUXCODE_DECODER=rust` | kernel 1.08-1.12x (1.24x `impulses`, 1.35x non-finite); end to end `decode_unit` 1.03-1.15x. [results](results/prototype_decoder.md) |
| zstd decompress + layout + validation in one call | `decompress_body` | no change end to end: python-zstandard's decompress is already a thin C call and zstd's own time dominates (sin-4.12hz: 1.90 vs 1.97 ns/sample) |
| encode analysis kernels (`encode_unit`) | stats, non-finite fill, noise floor, decimal detection, quantize, order pick, residuals, bit target | 0.91-1.04x of numba on finite data, 0.68x with non-finite values; units byte-identical, block means differ in the last bits. [results](results/prototype_encoder.md) |
| compress stage (`compress_unit`) | layout choice, body packing, flush points, zstd frame, no GIL | 0.94-1.11x single thread; effort 5 (block flushes) at 8 threads 2.7-3.1x. [results](results/compress_bench.md) |

### Why the frame build is the one that matters

python-zstandard holds the GIL inside `flush(FLUSH_BLOCK)`. `compress` only buffers until a 128 KB zstd
block fills, so the compression of each cut segment happens inside the flush. Same call count, 16 KB
chunks, MiB/s at 1 / 4 / 8 threads (`zstd_flush_gil.py`, on battery):

| calls | 1 | 4 | 8 |
|---|---:|---:|---:|
| `compress` only | 4019 | 13146 | 3574 |
| `compress` + `flush(BLOCK)` | 4360 | 2428 | 1444 |

Four threads of the flushed path are slower than one. `stream_writer.flush`, `chunker.flush` and the cffi
backend behave the same (cffi is also slower: its wrapper does Python work per call), and the switch
interval doesn't help. With the frame built in Rust, the whole pack, flush and zstd sequence runs without
the GIL. Block flushes are the encoder's roughly 1.5-2% size gain; `Params.effort` starts them at 5 on main
because of this (docs/TUNING.md).

### Why the kernels don't win

numba's `fastmath` vectorizes the noise estimate and statistics well already. A straight Rust port was 35%
slower; profiling showed LLVM didn't vectorize array-lane float reductions on aarch64, and explicit SIMD
(the `wide` crate) got it to about parity (noise estimate 2.2 vs numba 1.7 ns/sample). Gains would need
more hand-tuned SIMD for a second implementation to maintain.

`fastmath` also means numba's sums (block sums and means, the noise estimate, `estimate_bits`) have an
order LLVM picks. The Rust sums are fixed-lane and deterministic on every platform, but not bit-identical:
the block means differ in the last bits (so the golden hash changes), and the units, flags, parameters,
anchors, residuals and codes matched exactly in a 530-case differential test and on all 27 golden cases.

## What was merged

- `rust/`: only `compress_unit` (plus `zstd_version`), for units without a time axis. `splice`/`update` and
  time-axis units keep the Python path. Depends on pyo3, numpy and zstd-safe (the libzstd it bundles is
  1.5.7, as python-zstandard 0.25's).
- `fluxcode._unit`: uses the extension if `fluxcode_rs` imports and `FLUXCODE_RUST` isn't `0`; otherwise
  the existing path. **Same units either way** while both link the same libzstd (`tests/test_rust.py`
  checks byte equality across every effort when the versions match, and a round trip when they don't).
- CI builds the extension and runs the suite with it and with `FLUXCODE_RUST=0`.
- `Params.effort` is **unchanged**, so output doesn't depend on whether the extension is installed.

## Inputs for the effort question

With block flushes no longer holding the GIL, efforts below 5 could flush again when the extension is
present. This experiment measured throughput only (sizes are in docs/TUNING.md). On the prototype's old
effort table (flush from effort 3, heuristic layout; battery power, 8 threads) flush plus heuristic encoded
at 3757 MiB/s with Rust against 1515 with Python, and unflushed effort 2 encoded at 3185 with Python. Two
things to settle first: output would depend on whether the extension is installed unless the table
changes for everyone, and main's default (effort 4, best of both layouts, no flush) costs differently from
the heuristic's single compression.

## Reproduce

```sh
cd rust && uv run --with maturin maturin develop --release   # builds fluxcode_rs into the environment
cd ../experimental
uv run python rust_port/compress_bench.py                   # ~3 min, on AC power
uv run python rust_port/zstd_flush_gil.py                   # the flush scaling table above
```

## Revisit if

- Cold start matters: numba's JIT warmup and import (`import fluxcode` is about 180 ms, mostly numba) were
  not measured against a Rust-only build, which a full port would remove. Not tried here.
- numba stops being a dependency, or a platform can't run it.
- Time-axis units or `splice`/`update` become a throughput bottleneck: the extension doesn't cover them yet.
