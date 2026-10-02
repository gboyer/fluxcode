# fluxcode

Compression for float64 time series, geared towards scientific and engineering usage.

Guarantees a maximum error using a quantization, and is exact for decimal series up
to 4 significant figures, or more when the range is narrow.

High level properties:

* **Fast**: A single MacBook Air M3 core decodes at 4 GiB/s, and encodes at 1.5 GiB/s at the
  default effort (about a third faster at effort 1).
* **Blocks**: Each block holds 0 to 65,535 samples: fixed-size blocks of regularly sampled
  data (usually 1000 samples/block), explicit sizes, or blocks of a fixed duration of time
  for data that arrives late or with gaps.
* **Min-Max Quantization**: Each block is quantized up to 2^16 steps between its min and
  max value. Blocks with tight range preserve accuracy better than wide ranges.
* **Decimals**: If the block's samples fall close to a decimal grid, it is stored as exact
  decimals if the bit range allows. For example, 14.32 will be compressed as 1432e-2.
* **Multi-Order Delta Encoding**: Stores raw samples or their first, second, or third
  difference, whichever has the lowest variance.
* **Unit-Compressed zstd**: Blocks are assembled into units; for example, a minute might
  be a single unit with 60 blocks, each with 1000 samples (the defaults of `encode`; the
  unit records every block's size). Their bit planes are
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
  By default, 0.25 sigmas. It only applies to blocks of at least 256 samples, so a fixed
  `block_len` under 256 never gets one.
* **Target bit rate**: Uses an entropy estimator to determine if the above would likely
  exceed your bit rate after compression. By default, off, to preserve quality.
* **Effort**: `effort=1` (fastest) to `9` (smallest), default 4, trades encode time for size
  without changing the decoded values. It picks bit or byte planes for the residuals (by a
  one-pass heuristic at efforts 1–2, or by compressing both from 3), from effort 5 gives each dense
  plane its own zstd block, and at effort 9 also tries zstd 9. Against the default, effort 1 encodes
  about a third faster for 1.5–2.5% more bytes; effort 5 is about 2% smaller for about 18% more
  time; effort 9 is about 3% smaller for about 5× the time ([docs/TUNING.md](docs/TUNING.md#effort)).
  Efforts 5 and up scale poorly across threads with python-zstandard (it holds the GIL while ending
  a block); the optional Rust extension ([rust/](rust/): `uv sync --extra rust` from a checkout; not on PyPI
  yet) fixes that and gives the same units.

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

# One unit = one storage row, here 30 blocks of 1000 samples. Units are self-describing.
unit, block_min, block_max, block_mean = fluxcode.encode_unit(x)
y, _, sizes = fluxcode.decode_unit(unit)                    # (values, times, block_sizes); no times here
print(f"{8 * len(unit) / x.size:.2f} bits/sample, max error {np.abs(y - x).max():.3f}")

# Decimal data (here a price rounded to cents) decodes to the identical float64.
price = np.round(20 + np.cumsum(rng.normal(0, 0.05, 5_000)), 2)
assert np.array_equal(fluxcode.decode_unit(fluxcode.encode_unit(price).unit).values, price)

# Timestamps (datetime64 in s/ms/us/ns, or integer ticks with time_unit) are stored exactly.
stamps = np.datetime64("2026-09-27T00:00", "ns") + np.arange(x.size) * np.timedelta64(1, "ms")
stamps[12_345:] += np.timedelta64(2, "s")                  # a gap: only its block pays for it
values, times, _ = fluxcode.decode_unit(fluxcode.encode_unit(x, times=stamps).unit)
assert np.array_equal(times, stamps)

# Replace block 3 and append block 30 as more of the stream arrives: blocks by index, any size.
new_blocks = 100 * np.sin(2 * np.pi * 3.3 * (np.arange(2000).reshape(2, 1000) / 1000 + 30))
unit, indices, mins, maxs, means = fluxcode.update(unit, {3: new_blocks[0], 30: new_blocks[1][:700]})
assert fluxcode.decode_unit(unit).values.size == 30_700

# Blocks of a fixed duration (here one second): blocks with no data are empty, and the unit
# grows as data arrives. update_time_blocks upserts samples and deletes time ranges.
ms = np.datetime64("2026-09-27T00:00", "ms") + np.arange(x.size)
hour = {"start_time": ms[0], "block_duration": np.timedelta64(1, "s")}
first, late = slice(0, 10_000), slice(10_000, 20_000)      # 20 s of 30; seconds 10-19 come later
unit, *_ = fluxcode.encode_time_blocks(x[first], ms[first], **hour)
gap = (ms[10_000], ms[20_000])                              # [start, end)
unit, *_ = fluxcode.update_time_blocks(unit, x[late], ms[late], delete_ranges=gap, **hour)
assert fluxcode.decode_unit(unit).block_sizes.tolist() == [1000] * 20

# Long series: encode() splits into units; decode() returns one (values, times) per unit.
units, mins, maxs, means = fluxcode.encode(np.tile(x, 5))
assert sum(len(decoded.values) for decoded in fluxcode.decode(units)) == 5 * x.size
```

This prints `5.77 bits/sample, max error 0.125`: the noise (σ = 1) triggered the noise floor,
which set the step to 0.25σ.

- `encode_unit(x, params, *, block_len=1000, times=None, time_unit=None)`: one unit (an 8-byte
  header and a zstd frame) of blocks of `block_len` samples (1 to 65,535; the last block holds the
  rest). It also returns per-block min, max and mean (over finite samples) as summary statistics:
  they come free with encoding, and decoding doesn't need them. `times` optionally stores one
  timestamp per sample, exactly: `datetime64[s|ms|us|ns]`, or integer ticks with
  `time_unit="s" | "ms" | "us" | "ns"`. They must be naive (store UTC) and non-decreasing; equal
  timestamps are fine. A regular grid costs about 65 bytes per unit.
- `encode_blocks(x, block_sizes, params, *, times=None, time_unit=None)`: one unit of blocks of the
  given sizes (0 to 65,535 each), as one flat array and each block's size, like Arrow list arrays.
  An empty block stores nothing (NaN statistics). Blocks of at most 8 samples skip the analysis:
  they are stored at the finest step, without differences (decimal data still stays exact).
- `encode_time_blocks(x, times, params, *, start_time, block_duration, time_unit=None)`: one unit
  in which block b holds the samples timed in `[start_time + b·block_duration, start_time +
  (b+1)·block_duration)`; blocks without samples are empty. `start_time` (`datetime64`, naive
  `datetime` or ticks) and `block_duration` (`timedelta64`, `timedelta` or ticks) aren't stored:
  keep them with the unit, e.g. the start in its storage key.
- `encode(x, params, *, block_len=1000, blocks_per_unit=60, times=None, time_unit=None)` /
  `decode(units)`: bulk versions. `encode` splits a long series (and its times, non-decreasing
  across unit boundaries too) into units of `blocks_per_unit` blocks and returns
  `(units, block_mins, block_maxs, block_means)`, one entry per unit; `decode` returns one
  `DecodedUnit` per unit. A unit holds at most 65,535 blocks and 2^26 = 67,108,864 samples (the
  bound decoders accept).
- `decode_unit(unit)`: returns `DecodedUnit(values, times, block_sizes)`. `times` is `datetime64`
  in the encoded unit, or `None` for a unit encoded without times. The unit records everything
  needed.
- `update(unit, blocks, params, *, times=None)`: replace or append whole blocks, given as a dict
  `{index: samples}` of any sizes (skipped indices past the end are appended empty); untouched
  blocks decode to identical values. `times` (a dict with the same keys) is required exactly when
  the unit has a time axis. The update functions return `UpdatedUnit(unit, indices, block_min,
  block_max, block_mean)`: the new unit and the statistics of the blocks they re-encoded.
- `update_time_blocks(unit, x, times, params, *, start_time, block_duration, delete_ranges=None,
  time_unit=None)`: an upsert plus a range deletion. It first deletes the unit's samples in
  `delete_ranges` (`[start, end)` pairs: one pair, a list, or a `(k, 2)` array), then adds the new
  samples, which may be timed anywhere: an existing sample with the same timestamp as a new one is
  replaced, and new samples sharing a timestamp are all kept. Samples only is an upsert, ranges
  only a deletion, both a range replacement. Blocks that no range meets and no new sample falls in
  are carried over without being decoded; the others are decoded, merged and re-encoded: their kept samples are grid points, so they come back bit for
  bit unless the block's step coarsens, and stay within one step of the coarsest grid it has
  used however often it is updated.
- `Params`: `min_quantize_bits=6`, `max_quantize_bits=16`, `diff_orders={0,1,2,3}`,
  `noise_floor_sigma=0.25` (`None` turns the noise floor off), `target_bits_per_sample=None`
  (≥ 6 when set), `decimal_detection=True`, `effort=4` (1–9). [docs/TUNING.md](docs/TUNING.md) has the measurements behind the defaults.

## Numbers

1 GiB of synthetic 1 kHz signals through the public API with default parameters, one thread, on an
Apple M3 (MacBook Air), Python 3.11, numpy 2.4. Error is the worst block's max error as a
percentage of that block's range. Full tables (noise floor off, a 6-bit target, thread scaling) are
in [bench/RESULTS.md](bench/RESULTS.md); the signal generators are in
[tests/_signals.py](tests/_signals.py).

| signal | bits/sample | worst max error (% of range) | encode µs/block | decode µs/block |
|---|---|---|---|---|
| linear ramp | 0.02 |0.0000%| 2.61 | 1.32 |
| square wave | 0.04 |0.0000%| 2.78 | 1.14 |
| sine, 4.12 Hz | 2.40 |0.0010%| 5.91 | 2.12 |
| sine, 50.3 Hz | 6.74 |0.0010%| 5.85 | 2.16 |
| chirp | 7.81 |0.0010%| 7.36 | 2.08 |
| random walk | 12.62 |0.0015%| 5.43 | 1.83 |
| random walk rounded to 0.01 | 9.28 |0% (exact)| 6.23 | 1.88 |
| sensor drift rounded to 0.1 | 1.78 |0% (exact)| 6.21 | 2.13 |
| noisy sine (σ = 5) | 5.24 |0.2334%| 4.11 | 1.59 |
| Gaussian spikes on uniform noise | 6.63 |1.4661%| 4.94 | 1.83 |
| **all 15 signals** | **4.76** | | **4.98** | **1.76** |

Clean signals keep 16 bits of their range (error ≤ 0.0015%). The noisy ones have larger errors
relative to the range because the noise floor sets their step to 0.25σ of the noise, which bounds
the error at 0.125σ; `min_quantize_bits` keeps even those under 1.6% of the range.
The default effort compresses each unit with bit and with byte planes and keeps the smaller. The
whole set at effort 1 is 4.83 bits/sample at 3.35 µs/block to encode, and at effort 5 4.66 at 5.76
([bench/RESULTS.md](bench/RESULTS.md)); byte planes gain much more at lower `max_quantize_bits`
(see the [research report](experimental/report/index.html#scatter-kinds)).

Decoding runs at about 4.5 GB/s per core, and the kernels release the GIL:
4 threads encode about 5 GB/s and decode about 15 GB/s (8 threads add little: 4 of the 8 cores
are "efficiency cores").
[docs/PERFORMANCE.md](docs/PERFORMANCE.md) has a sample industrial-scale run
(1000 channels × 1 day at 1 kHz) and where the time goes.
The implementation is fully in numba, SIMD optimized, and tuned for ARM NEON.

## Guarantees

From [docs/SPEC.md §8](docs/SPEC.md#8-guarantees), which states them exactly:

- **Bounded error.** Every sample decodes within half the block's step, and never further than
  range / (2^`min_quantize_bits` − ½) (1.6% of the block's range at the default of 6), whatever
  the noise floor or the size target do. Blocks at full precision are within about
  2^−`max_quantize_bits` of their range (0.0015% at 16); noise-floor blocks within f·σ/2.
- **Min and max decode within half a step.** The power-of-two grid is absolute (multiples of the
  step), so the decoded minimum is the grid point nearest the minimum. The `block_min` and
  `block_max` that encode returns are the samples' own: allow half a step either way if you use
  them as hard bounds on decoded values.
- **Updates don't drift.** Decoded values are grid points, so re-encoding them on the same or a
  finer step returns them bit for bit. Samples an `update_time_blocks` keeps in a block it re-encodes stay within
  one step of the coarsest grid their block has used, however many times it runs.
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
  white noise). Blocks that look like white noise get a step of at most 0.25σ. Blocks under 256
  samples keep their finest step: the autocorrelation estimate is too noisy there to tell white
  noise from a random walk.
- **Fixed polynomial predictors** of order 0–3, picked per block by residual variance, as in
  [Shorten](https://en.wikipedia.org/wiki/Shorten_(file_format)) and
  [FLAC's fixed predictors](https://www.rfc-editor.org/rfc/rfc9639#name-fixed-predictor-subframe).
  Residuals wrap mod 2^16, so they always fit in 16 bits.
- **[Zigzag](https://protobuf.dev/programming-guides/encoding/#signed-ints)** maps signed
  residuals to unsigned, so small magnitudes have leading zero bits.
- **[Bit-shuffle](https://github.com/kiyo-masui/bitshuffle)** ([paper](https://arxiv.org/abs/1503.00638))
  across the whole unit: bit plane j of every residual is stored together, so the high planes are
  long runs of zeros.
- **[Zstandard](https://www.rfc-editor.org/rfc/rfc8878)** compresses the whole unit as one frame
  (zstd level 3, or 1 at `effort=1`; at `effort=9` the smaller of levels 3 and 9), from effort 5 with a
  zstd block, and so its own Huffman table, per dense plane.

## Documentation

- [docs/SPEC.md](docs/SPEC.md): the format, encoding algorithm, guarantees and conformance tests.
- [docs/PERFORMANCE.md](docs/PERFORMANCE.md): speed, the day-scale stress run, implementation notes.
- [docs/TUNING.md](docs/TUNING.md): measurements behind the parameter defaults and the time axis
  design, and what to check on your data.
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

Code layout (`fluxcode/`), from the public API down:

- `_api.py`: the public functions (encode, decode, update); `_types.py`: `Params` and the result
  tuples.
- `_unit.py`: building and parsing units from flat sample arrays and block sizes;
  `_time_blocks.py`: dividing timed samples into blocks of a fixed duration.
- `_compress.py`: the compression policy (efforts, layout choice, flush points, framing, candidate
  choice); `rust/src/compress.rs` is its port.
- `_encoder.py`, `_decoder.py`: the per-block kernels (SPEC §5, §6); `_time.py`: the time axis
  kernels (§4); `_noise.py`, `_nonfinite.py` and `_extreme_magnitudes.py`: the noise estimate,
  NaN/inf blocks and extreme ranges, each off the common path.
- `_format.py`: the header, field offsets and row types (§7); `_bitpacking.py`: writing and
  reading the body's fields, and splicing bodies for `update`. `rust/src/` has the same-named
  `format.rs` and `bitpacking.rs`.

The kernels are compiled by numba with `cache=True`, which checks only the timestamp of the file
defining each kernel, not the constants and helpers it inlines from other modules. After editing
`fluxcode/_format.py` (for example), a benchmark can run stale compiled code: clear the cache with
`find fluxcode -name '*.nb[ic]' -delete`. The tests always compile from source (`tests/conftest.py`).

## License

[MIT](LICENSE). Copyright (c) 2026 Garry Boyer.
