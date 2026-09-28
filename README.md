# fluxcode

Compression for float64 time series, geared towards scientific and engineering usage.

Guarantees a maximum error using a quantization, and is exact for decimal series up
to 4 significant figures, or more when the range is narrow.

High level properties:

* **Fast**: A single MacBook Air M3 core decodes at 4 GiB/s, and encodes at 2 GiB/s.
* **Blocks**: Uses fixed-sized blocks of regularly sampled data, usually
  1000 samples/block.
* **Min-Max Quantization**: Each block is quantized up to 2^16 steps between its min and
  max value. Blocks with tight range preserve accuracy better than wide ranges.
* **Decimals**: If the block's samples fall close to a decimal grid, it is stored as exact
  decimals if the bit range allows. For example, 14.32 will be compressed as 1432e-2.
* **Multi-Order Delta Encoding**: Stores raw samples or their first, second, or third
  difference, whichever has the lowest variance.
* **Unit-Compressed zstd**: Blocks are assembled into units; for example, a minute might
  be a single unit with 60 blocks, each with 1000 samples (the defaults; both are
  parameters, and the unit records them). Their bit planes are
  interleaved and compressed with zstd to separate high and low entropy signals.
* **Full Range**: Supports the full dynamic range of IEEE 754 64-bit floats, including
  subnormal ranges, +/- Infinity, and NaN.
* **Timestamps**: Optionally stores exact timestamps (s, ms, µs or ns) for irregular sampling.
  Regular intervals store very cheaply, but noisy timestamps dramatically
  increase compressed size.

Encoding features to balance size, accuracy, and performance depending on data
type:

* **Noise floor**: For data with high frequency noise, you can optionally specify the
  sigma multiplier that you want to preserve, and the quantization grid is reduced
  accordingly. This helps compression for noisy data, but
  preserves clean periodic data (unless the frequency approaches the sample rate).
  By default, 0.25 sigmas.
* **Target bit rate**: Uses an entropy estimator to determine if the above would likely
  exceed your bit rate after compression. By default, off, to preserve quality.
* **Byte planes**: `try_byte_planes=True` sees if organizing high/low bytes
  works better than individual bit planes. Helpful for noise-free periodic
  signals, but off by default for performance.

**Status:** version 0.1, alpha. The unit format is version 1 and specified in
[docs/SPEC.md](docs/SPEC.md); decoders reject other versions. The API and format may
still change before 1.0.

## Install

```sh
pip install git+https://github.com/gboyer/fluxcode
# or
uv add git+https://github.com/gboyer/fluxcode
```

Python 3.10 or later. Dependencies: numpy, numba, zstandard.

## Quickstart

```python
import numpy as np
import fluxcode

rng = np.random.default_rng(0)
t = np.arange(30_000) / 1000                                # 30 s at 1 kHz: 30 blocks of 1000
x = 100 * np.sin(2 * np.pi * 3.3 * t) + rng.normal(0, 1, t.size)

# One unit = one storage row (up to blocks_per_unit blocks, 60 by default). Units are self-describing.
unit, block_min, block_max, block_mean = fluxcode.encode_unit(x)
y, _ = fluxcode.decode_unit(unit)                           # (values, times); times is None here
print(f"{8 * len(unit) / x.size:.2f} bits/sample, max error {np.abs(y - x).max():.3f}")

# Decimal data (here a price rounded to cents) decodes to the identical float64.
price = np.round(20 + np.cumsum(rng.normal(0, 0.05, 5_000)), 2)
assert np.array_equal(fluxcode.decode_unit(fluxcode.encode_unit(price).unit).values, price)

# Timestamps (datetime64 in s/ms/us/ns, or integer ticks with time_unit) are stored exactly.
stamps = np.datetime64("2026-09-27T00:00", "ns") + np.arange(x.size) * np.timedelta64(1, "ms")
stamps[12_345:] += np.timedelta64(2, "s")                  # a gap: only its block pays for it
values, times = fluxcode.decode_unit(fluxcode.encode_unit(x, times=stamps).unit)
assert np.array_equal(times, stamps)

# Replace block 3 and append block 30 as more of the stream arrives.
new_blocks = 100 * np.sin(2 * np.pi * 3.3 * (np.arange(2000).reshape(2, 1000) / 1000 + 30))
unit, mins, maxs, means = fluxcode.update(unit, indices=[3, 30], blocks=new_blocks)
assert fluxcode.decode_unit(unit).values.size == 31_000

# Long series: encode() splits into units; decode() returns one (values, times) per unit.
units, mins, maxs, means = fluxcode.encode(np.tile(x, 5))
assert sum(len(decoded.values) for decoded in fluxcode.decode(units)) == 5 * x.size
```

This prints `5.81 bits/sample, max error 0.125`: the noise (σ = 1) triggered the noise floor,
which set the step to 0.25σ.

- `encode_unit(x, params, *, times=None, time_unit=None)`: one unit (a 16-byte header and a zstd
  frame) of up to `blocks_per_unit` blocks of `block_len` samples. It also returns per-block min,
  max and mean (over finite samples) as summary statistics: they come free with encoding, and
  decoding doesn't need them. A short last block is padded, and trimmed again on decode.
  `times` optionally stores one timestamp per sample, exactly: `datetime64[s|ms|us|ns]`, or
  integer ticks with `time_unit="s" | "ms" | "us" | "ns"`. They must be naive (store UTC) and
  non-decreasing; equal timestamps are fine. A regular grid costs about 45 bytes per unit.
- `decode_unit(unit)`: returns `DecodedUnit(values, times)`. `times` is `datetime64` in the
  encoded unit, or `None` for a unit encoded without times. The unit records everything needed.
- `update(unit, indices, blocks, params, *, times=None)`: replace or append whole blocks;
  untouched blocks decode to identical values. `times` (shaped like `blocks`) is required exactly
  when the unit has a time axis. Returns the new unit and min/max/mean of the updated blocks, in
  `indices` order. A partial last block must be replaced by a full one before appending after it.
- `encode(x, params, *, times=None, time_unit=None)` / `decode(units)`: bulk versions. `encode`
  splits a long series (and its times, non-decreasing across unit boundaries too) into full units and returns
  `(units, block_mins, block_maxs, block_means)`, one entry per unit; `decode` returns one
  `DecodedUnit` per unit.
- `Params`: `min_quantize_bits=6`, `max_quantize_bits=16`, `diff_orders={0,1,2,3}`,
  `noise_floor_sigma=0.25` (`None` turns the noise floor off), `target_bits_per_sample=None`
  (≥ 6 when set), `decimal_detection=True`, `block_len=1000` (a multiple of 8, at most 65,536),
  `blocks_per_unit=60` (at most 2^26 = 67,108,864 samples per unit, the bound decoders accept),
  `try_byte_planes=False`. [docs/TUNING.md](docs/TUNING.md) has the measurements behind the defaults.

## Numbers

1 GiB of synthetic 1 kHz signals through the public API with default parameters, one thread, on an
Apple M3 (MacBook Air), Python 3.11, numpy 2.4. Error is the worst block's max error as a
percentage of that block's range. Full tables (noise floor off, a 6-bit target, thread scaling) are
in [bench/RESULTS.md](bench/RESULTS.md); the signal generators are in
[tests/_signals.py](tests/_signals.py).

| signal | bits/sample | worst max error (% of range) | encode µs/block | decode µs/block |
|---|---|---|---|---|
| linear ramp | 0.02 | 0.0000% | 2.32 | 1.53 |
| square wave | 0.05 | 0.0000% | 2.45 | 1.68 |
| sine, 4.12 Hz | 2.78 | 0.0010% | 3.02 | 1.93 |
| sine, 50.3 Hz | 6.73 | 0.0010% | 3.54 | 2.04 |
| chirp | 7.84 | 0.0010% | 4.71 | 1.91 |
| random walk | 12.66 | 0.0015% | 3.80 | 1.72 |
| random walk rounded to 0.01 | 9.28 | 0% (exact) | 3.73 | 1.76 |
| sensor drift rounded to 0.1 | 1.78 | 0% (exact) | 3.56 | 2.09 |
| noisy sine (σ = 5) | 5.49 | 0.2333% | 3.13 | 2.18 |
| Gaussian spikes on uniform noise | 6.68 | 1.4667% | 3.55 | 1.70 |
| **all 15 signals** | **4.88** | | **3.25** | **1.91** |

Clean signals keep 16 bits of their range (error ≤ 0.0015%). The noisy ones have larger errors
relative to the range because the noise floor sets their step to 0.25σ of the noise, which bounds
the error at 0.125σ; `min_quantize_bits` keeps even those under 1.6% of the range.
With `try_byte_planes=True` the whole set is 4.80 bits/sample (sines 4.12 Hz 2.44, 9.87 Hz 3.44)
at 4.94 µs/block to encode; the gain is much larger at lower `max_quantize_bits` (see the
[research report](experimental/report/index.html#scatter-kinds)).

Decoding runs at about 4 GB/s per core, and the kernels release the GIL:
8 threads encode about 10 GB/s (note: 4 of those are "efficiency cores").
[docs/PERFORMANCE.md](docs/PERFORMANCE.md) has a sample industrial-scale run
(1000 channels × 1 day at 1 kHz) and where the time goes.
The implementation is fully in numba, SIMD optimized, and tuned for ARM NEON.

## Guarantees

From [docs/SPEC.md §6](docs/SPEC.md#6-guarantees), which states them exactly:

- **Bounded error.** Every sample decodes within half the block's step, and never further than
  range / (2^`min_quantize_bits` − ½) (1.6% of the block's range at the default of 6), whatever
  the noise floor or the size target do. Blocks at full precision are within about
  2^−`max_quantize_bits` of their range (0.0015% at 16); noise-floor blocks within f·σ/2.
- **Min is exact** on the power-of-two grid; max is within half a step.
- **Decimal data is lossless.** Values that are decimals of p places (as parsed from text) decode
  to the identical float64, as long as the block spans fewer than 2^16 decimal steps. Decimals
  stored as float32 upstream decode as the decimal itself.
- **Any float64 encodes.** NaN, +inf and −inf round-trip exactly (every NaN as the canonical quiet
  NaN; −0.0 decodes as +0.0), as do subnormal ranges and blocks spanning ±DBL_MAX.
- **Stable under edits and re-encoding.** `update` leaves the other blocks' bytes unchanged, so
  they decode identically; without a size target, the updated unit is byte-identical to encoding
  the new series from scratch. Decoded data is a fixed point: re-encoding it gives the same bytes.
- **Timestamps decode exactly.**

## How it works

Per block of 1000 samples (details and pseudocode in [docs/SPEC.md](docs/SPEC.md)):

- **[Quantization](https://en.wikipedia.org/wiki/Quantization_(signal_processing))** to a
  power-of-two step 2^e with at most 2^16 steps across the block's range, or to the coarsest exact
  decimal step 10^p every sample sits on. Power-of-two steps keep the grid stable when an edit
  changes the range.
- **Noise floor:** a robust (clipped [mean absolute deviation](https://en.wikipedia.org/wiki/Average_absolute_deviation))
  estimate of the noise in the second differences, plus their lag-1 autocorrelation (−2/3 for
  white noise). Blocks that look like white noise get a step of at most 0.25σ.
- **Fixed polynomial predictors** of order 0–3, picked per block by residual variance, as in
  [Shorten](https://en.wikipedia.org/wiki/Shorten_(file_format)) and
  [FLAC's fixed predictors](https://www.rfc-editor.org/rfc/rfc9639#name-fixed-predictor-subframe).
  Residuals wrap mod 2^16, so they always fit in 16 bits.
- **[Zigzag](https://protobuf.dev/programming-guides/encoding/#signed-ints)** maps signed
  residuals to unsigned, so small magnitudes have leading zero bits.
- **[Bit-shuffle](https://github.com/kiyo-masui/bitshuffle)** ([paper](https://arxiv.org/abs/1503.00638))
  across the whole unit: bit plane j of every residual is stored together, so the high planes are
  long runs of zeros.
- **[Zstandard](https://www.rfc-editor.org/rfc/rfc8878)** level 3 compresses the whole unit as one
  frame.

## Documentation

- [docs/SPEC.md](docs/SPEC.md): the format, encoding algorithm, guarantees and conformance tests.
- [docs/PERFORMANCE.md](docs/PERFORMANCE.md): speed, the day-scale stress run, implementation notes.
- [docs/TUNING.md](docs/TUNING.md): measurements behind the parameter defaults and what to check on
  your data.
- [docs/design/api-plan.md](docs/design/api-plan.md): the original API design plan (history).
- [bench/](bench/): `bench_gb.py` (size, error and speed per signal), `estimate.py` (the size
  target's estimate against actual size), `decimal_near_miss.py` (the encode cost of a
  decimal-detection near miss), `stress.py` (1000 channels × 1 day, multiple workers), and their
  results.

## Related work

- [Gorilla](https://www.vldb.org/pvldb/vol8/p1816-teller.pdf) (VLDB 2015): lossless XOR of
  consecutive float64 values, the classic time-series float codec.
- [Chimp](https://www.vldb.org/pvldb/vol15/p3058-liakos.pdf) (VLDB 2022): Gorilla-style XOR
  encoding that also exploits trailing zeros and earlier values.
- [ALP](https://doi.org/10.1145/3626717) ([code](https://github.com/cwida/ALP)) (SIGMOD 2024):
  lossless floats by detecting decimals and encoding them as integers, the closest relative of
  fluxcode's decimal grids.
- [Bitshuffle](https://github.com/kiyo-masui/bitshuffle) and [Blosc](https://www.blosc.org/): bit-
  and byte-shuffle filters in front of a general-purpose compressor, the layout fluxcode uses.
- [FLAC](https://xiph.org/flac/format.html): lossless audio with fixed and LPC predictors and Rice
  coding; fluxcode's predictor orders are its fixed predictors.

## Research

[experimental/](experimental/) holds the investigation that led to this design: dozens of codec
variants (classic DPCM and piecewise schemes, constant-width formats, rate-controlled
predict-and-entropy-code pipelines, and the fluxcode prototype), their benchmarks, and a report
comparing them. It's kept for reference and isn't maintained or covered by CI.

## Development

```sh
uv sync
uv run pytest
uv run ruff check
uv run mypy fluxcode
uv run ty check
uv run python bench/bench_gb.py --gib 0.05   # a quick benchmark; the full run is 1 GiB
```

The kernels are compiled by numba with `cache=True`, which checks only the timestamp of the file
defining each kernel, not the constants and helpers it inlines from other modules. After editing
`fluxcode/_format.py` (for example), a benchmark can run stale compiled code: clear the cache with
`find fluxcode -name '*.nb[ic]' -delete`. The tests always compile from source (`tests/conftest.py`).

## License

[MIT](LICENSE). Copyright (c) 2026 Garry Boyer.
