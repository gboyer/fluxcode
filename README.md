# fluxcode

Bounded-error compression for float64 time series, built for scientific and engineering data.

Fluxcode compresses float64 streams into self-describing block groups with a hard per-sample error bound, while automatically detecting and preserving values on exact binary, decimal, or rational-step grids (such as `0.01`, `0.05`, thirds, or calibrated ADC counts) whenever the bit range allows.

**Live benchmarks & documentation:** [gboyer.github.io/fluxcode/report/](https://gboyer.github.io/fluxcode/report/) — interactive results across synthetic signals ([Codec Comparison](https://gboyer.github.io/fluxcode/report/codecs.html)) and 14 real-world public dataset families ([Public Datasets](https://gboyer.github.io/fluxcode/report/public.html)), plus the [Format Specification](https://gboyer.github.io/fluxcode/report/spec.html), [Timestamps](https://gboyer.github.io/fluxcode/report/time.html), [Historian](https://gboyer.github.io/fluxcode/report/sdt.html), [Fluxprune](https://gboyer.github.io/fluxcode/report/prune.html), and [Rate Control](https://gboyer.github.io/fluxcode/report/rate.html).

## Key properties

- **Fast**: On a single Apple M3 core, fluxcode decodes at ~4.4 GiB/s and encodes at ~1.5 GiB/s at default effort (~2.2 GiB/s at `effort=1`). Kernels are compiled with Numba and release the GIL across threads.
- **Self-describing block groups**: Each block holds 0 to 65,535 samples (fixed-size blocks, explicit sizes, or fixed time durations for late/gapped arrivals). Blocks are grouped into a single zstd frame with bit- or byte-shuffled residuals; decoding needs only the payload bytes.
- **Per-block quantization & exact grids**: Each block quantizes to up to 2¹⁶ steps across its range. Blocks on exact decimal grids (`10^p`) or rational-step grids (`N / D` times a power of 10 or 2, `N, D ≤ 65,535`, with an offset — e.g. `0.05`, `0.18`, thirds, `10 V / 4095`) are detected automatically and round-trip bit for bit.
- **Multi-order delta prediction**: Picks raw samples or 1st, 2nd, or 3rd differences per block by residual variance.
- **Full IEEE 754 float64 range**: Round-trips `NaN` (canonical quiet NaN), `±inf`, subnormals, and extreme magnitudes; isolated outliers/error codes are escaped so they do not coarsen neighboring samples.
- **Optional timestamps**: Stores exact `s`, `ms`, `us`, or `ns` timestamps (~65 bytes per regular block group), or lossy timestamps within a configurable fraction of the sample interval (`time_error`) to collapse clock jitter back onto a regular grid.

### Encoding controls

- **Noise floor** (`noise_floor_sigma=0.25`, `0` to disable): Estimates each block's high-frequency noise σ and avoids keeping detail finer than `f · σ` (bounding error at `f · σ / 2`, ~23 dB below the noise spectrum), while leaving clean periodic signals, short blocks (< 128 samples), and irregular sparse archives untouched.
- **Absolute tolerance** (`absolute_tolerance=0.0`): Optional per-series absolute error allowance in the data's own units (e.g. sensor accuracy), letting quiet blocks coarsen up to `2 × absolute_tolerance`.
- **Time error** (`time_error=0.0`, up to `0.5`): Rounds jittered timestamps to a 1-2-5 step or fitted clock lattice at the clock's own phase so 95% of timestamps move by at most `time_error` of their own interval. Update groups with `time_error` via `replace_time_blocks` from source data.
- **Target bit rate** (`target_bits_per_sample=None`, `≥ 6.0` when set): Soft per-group size cap guided by an entropy estimator.
- **Effort** (`effort=4`, range `1–9`): Trades encode time for size without changing decoded values — choosing bit vs. byte planes, per-plane zstd blocks (from effort 5), and zstd level 9 (at effort 9). An optional Rust extension ([rust/](rust/), `uv sync --extra rust`) avoids Python's GIL during zstd block flushes at efforts 5+.

**Status:** `0.1.0` (alpha). The wire format is version 1 ([docs/SPEC.md](docs/SPEC.md)); decoders reject unknown versions. Both API and format may change before 1.0.

## Install

```sh
pip install git+https://github.com/gboyer/fluxcode
# or
uv add git+https://github.com/gboyer/fluxcode
```

Requires Python 3.10+ (`numpy`, `numba`, `zstandard`).

## Quickstart

```python
import numpy as np
import fluxcode

rng = np.random.default_rng(0)
t = np.arange(30_000) / 1000  # 30 s at 1 kHz: 30 blocks of 1000
x = 100 * np.sin(2 * np.pi * 3.3 * t) + rng.normal(0, 1, t.size)

# One block group = one storage row, self-describing on decode.
encoded = fluxcode.encode_group(x, block_len=1000)
decoded = fluxcode.decode_group(
    encoded.payload
)  # DecodedGroup(values, times, block_sizes)
print(
    f"{8 * len(encoded.payload) / x.size:.2f} bits/sample, "
    f"max error {np.abs(decoded.values - x).max():.3f}"
)

# Decimal / rational-step data (here prices rounded to cents) decodes bit for bit.
price = np.round(20 + np.cumsum(rng.normal(0, 0.05, 5_000)), 2)
assert np.array_equal(
    fluxcode.decode_group(fluxcode.encode_group(price, block_len=1000).payload).values,
    price,
)

# Timestamps (datetime64 in s/ms/us/ns, or integer ticks with time_unit) are exact by default.
stamps = np.datetime64("2026-09-27T00:00", "ns") + np.arange(x.size) * np.timedelta64(
    1, "ms"
)
stamps[12_345:] += np.timedelta64(2, "s")  # a gap: only its block pays for it
timed = fluxcode.encode_group(x, block_len=1000, times=stamps)
assert np.array_equal(fluxcode.decode_group(timed.payload).times, stamps)
assert timed.block_start_times.shape == timed.block_min.shape

# Replace block 3 and append block 30 without re-encoding untouched blocks.
new_blocks = 100 * np.sin(
    2 * np.pi * 3.3 * (np.arange(2000).reshape(2, 1000) / 1000 + 30)
)
updated = fluxcode.update(encoded.payload, {3: new_blocks[0], 30: new_blocks[1][:700]})
assert fluxcode.decode_group(updated.payload).values.size == 30_700

# Fixed-duration time blocks (1 s each) with upserts and range deletion.
ms = np.datetime64("2026-09-27T00:00", "ms") + np.arange(x.size)
hour = {"start_time": ms[0], "block_duration": np.timedelta64(1, "s")}
first, late = slice(0, 10_000), slice(10_000, 20_000)
enc_tb = fluxcode.encode_time_blocks(x[first], ms[first], **hour)
gap = (ms[10_000], ms[20_000])  # [start, end)
upd_tb = fluxcode.update_time_blocks(
    enc_tb.payload, x[late], ms[late], delete_ranges=gap, **hour
)
assert fluxcode.decode_group(upd_tb.payload).block_sizes.tolist() == [1000] * 20

# Long series: encode() splits into block groups; decode() accepts EncodedSeries or payloads.
series = fluxcode.encode(np.tile(x, 5), block_len=1000, blocks_per_group=60)
assert sum(len(d.values) for d in fluxcode.decode(series)) == 5 * x.size
```

This prints `5.77 bits/sample, max error 0.125`: the σ = 1 noise triggers the noise floor, setting the step to 0.25σ.

## API overview

All encode functions return per-block summary statistics alongside the compressed payload; decoding does not need them. A block group holds up to 65,535 blocks and 2²⁶ (67,108,864) samples.

- `encode_group(samples, params=Params(), *, block_len, times=None, time_unit=None) -> EncodedGroup`
  Encodes a 1D series into fixed-size blocks (`1..65,535`; final block may be short). Returns `EncodedGroup(payload, block_min, block_max, block_mean, block_start_times)`.
- `encode_blocks(samples, block_sizes, params=Params(), *, times=None, time_unit=None) -> EncodedGroup`
  Encodes explicit block sizes (`0..65,535` each) from a flat sample array (like Arrow list arrays). Empty blocks get `NaN` stats and `NaT` start times; blocks of ≤ 8 samples skip analysis and use the finest step.
- `encode_time_blocks(samples, times, params=Params(), *, start_time, block_duration, time_unit=None) -> EncodedGroup`
  Partitions timed samples into fixed-duration buckets `[start_time + b·block_duration, start_time + (b+1)·block_duration)`. `start_time` and `block_duration` are not stored in the payload; keep them alongside the group (e.g. in its storage key).
- `encode(samples, params=Params(), *, block_len, blocks_per_group, times=None, time_unit=None) -> EncodedSeries`
  Splits a long series into multiple block groups, returning `EncodedSeries(groups)` with flattening properties `payloads`, `block_min`, `block_max`, `block_mean`, and `block_start_times`.
- `decode_group(payload) -> DecodedGroup` / `decode(groups) -> list[DecodedGroup]`
  Returns `DecodedGroup(values, times, block_sizes)`. `times` is a `datetime64` array in the encoded unit (`s`, `ms`, `us`, or `ns`), or `None` if untimed.
- `update(payload, blocks, params=Params(), *, times=None, time_unit=None) -> UpdatedGroup`
  Replaces or appends whole blocks by index (`{index: samples}`) without decoding untouched blocks. Returns `UpdatedGroup(payload, indices, block_min, block_max, block_mean, block_start_times)` for the re-encoded blocks.
- `update_time_blocks(payload, samples, times, params=Params(), *, start_time, block_duration, delete_ranges=None, time_unit=None) -> UpdatedGroup`
  Deletes samples in `delete_ranges` (`[start, end)` pairs) and upserts new samples (exact timestamps only; untouched blocks are carried over directly).
- `replace_time_blocks(payload, blocks, samples, times, params=Params(), *, start_time, block_duration, time_unit=None) -> UpdatedGroup`
  Re-encodes the specified block indices from source samples; use this to update time-block groups when `time_error > 0`.
- `Params`:
  `min_quantize_bits=6`, `max_quantize_bits=16`, `diff_orders=frozenset({0, 1, 2, 3})`, `noise_floor_sigma=0.25`, `target_bits_per_sample=None`, `grids="rational"` (`"binary" | "decimal" | "rational"`), `effort=4` (`1–9`), `time_error=0.0` (`0–0.5`), `absolute_tolerance=0.0`. See [docs/TUNING.md](docs/TUNING.md) for the rationale behind the defaults.

Timestamps must be naive (store UTC) and non-decreasing; duplicate timestamps are allowed. Pass `datetime64[s|ms|us|ns]` directly, or integer ticks with `time_unit`.

## Numbers

### Synthetic 1 GiB benchmark

1.00 GiB of synthetic 1 kHz float64 signals (15 signal types, 134,100 blocks) with default parameters, single-threaded on an Apple M3 (Python 3.11, NumPy 2.4). Error is the worst block's max error as a percentage of that block's range. Full tables (noise floor off, 6-bit target, efforts 1/5/9, thread scaling) are in [bench/RESULTS.md](bench/RESULTS.md); signal generators are in [tests/_signals.py](tests/_signals.py).

| signal | bits/sample | worst max error (% of range) | encode µs/block | decode µs/block |
|---|---:|---:|---:|---:|
| linear ramp | 0.02 | 0.0000% | 2.40 | 1.36 |
| square wave | 0.04 | 0.0000% | 2.96 | 1.57 |
| sine, 4.12 Hz | 2.40 | 0.0010% | 6.10 | 2.12 |
| sine, 50.3 Hz | 6.74 | 0.0010% | 6.06 | 2.16 |
| chirp | 7.81 | 0.0010% | 7.48 | 2.05 |
| random walk | 12.62 | 0.0015% | 5.63 | 1.82 |
| random walk rounded to 0.01 | 9.28 | 0% (exact) | 6.32 | 1.88 |
| sensor drift rounded to 0.1 | 1.78 | 0% (exact) | 6.23 | 2.19 |
| noisy sine (σ = 5) | 5.24 | 0.2334% | 4.26 | 1.58 |
| Gaussian spikes on uniform noise | 6.60 | 1.4661% | 5.06 | 1.86 |
| **all 15 signals (1.00 GiB)** | **4.76** | | **5.12** | **1.79** |
| random walk on thirds (not in the 15) | 3.97 | 0% (exact) | 10.82 | 2.36 |

Clean signals keep 16-bit range precision (≤ 0.0015% error); exact decimal and rational-step signals decode bit for bit; noisy blocks coarsen to 0.25σ (bounded by `min_quantize_bits` to < 1.6% of range). Against the default (`4.76` bits/sample at `5.12 µs/block`), `effort=1` gives `4.83` at `3.53 µs/block`, `effort=5` gives `4.66` at `5.96 µs/block`, and `effort=9` gives `4.62` at `22.87 µs/block`. Byte planes gain more at lower `max_quantize_bits` (see [Codec Comparison](https://gboyer.github.io/fluxcode/report/codecs.html#scatter-kinds)). Across 4 threads on M3, throughput reaches ~5.2 GB/s encode and ~14.7 GB/s decode (see [docs/PERFORMANCE.md](docs/PERFORMANCE.md) for a 1,000-channel × 1-day stress run).

### Real-world public datasets

Evaluated across **94 series** (**29.2M float64 samples**, 233.8 MB uncompressed) spanning **14 public dataset families** — CWRU bearing vibration, MIT-BIH ECG, IRIS seismic, BIDMC clinical monitors, UCI household power, hydraulics, SKAB industrial, air quality, appliances, NOAA buoys, ERA5 weather, Binance trades, smartphone IMU, and the Elf corpus. Across these families, default fluxcode averages **3.03–9.41 bits/sample** (6.8×–21.2× compression vs. float64), beating both raw zstd and byte-shuffled zstd on every family, with exact decimal/rational grid detection saving up to 57% over binary-only quantization (e.g. MIT-BIH ECG at 4.14 bits/sample, exact). See the [Public Datasets report](https://gboyer.github.io/fluxcode/report/public.html) and [experimental/public_datasets/RESULTS.md](experimental/public_datasets/RESULTS.md) for full per-series tables and timestamp-axis results.

## Guarantees

Stated formally in [docs/ENCODER.md §6](docs/ENCODER.md#6-guarantees):

- **Bounded error**: Every sample decodes within half its block's step, and never further than `range / (2^min_quantize_bits − ½)` (~1.6% at the default of 6), regardless of noise floor or rate target. Full-precision blocks stay within ~2⁻¹⁶ of range (0.0015%); noise-floor blocks within `f · σ / 2`.
- **Summary stats**: `block_min` and `block_max` are computed from the input samples; decoded values can sit up to half a step outside them.
- **No update drift**: Untouched blocks in `update` carry over byte-for-byte. Decoded values are grid points, so re-encoding on the same or finer step returns identical bytes, and repeated `update_time_blocks` merges stay within one step of the coarsest grid the block has used.
- **Exact decimal & rational grids**: Values on decimal steps `10^p` or rational steps (`k / 3`, `10 · c / 4095`) decode bit for bit when the block fits in the bit budget, has no `target_bits_per_sample`, and is not floored by white noise (`noise_floor_sigma=0` keeps noisy decimals exact too).
- **Outlier isolation**: Isolated glitches or error codes (e.g. `-9999`, `1e6`) are escaped separately so the rest of the block keeps its fine step.
- **Full float64 support**: `NaN` (as canonical quiet NaN), `±inf`, subnormals, and `±DBL_MAX` ranges all encode cleanly (`-0.0` decodes as `+0.0`).
- **Timestamps**: Exact by default. With `time_error=e`, timestamps stay ordered and inside their time blocks, none moves more than `e` times the block's mean interval (+2% margin), at most 5% move more than `e` times their own interval, and no timestamp is rounded twice.

To verify these invariants in production or tests, `fluxcode.verify` checks any call's result against its inputs and returns a list of violations (empty when all guarantees hold):

```python
violations = fluxcode.verify.encode_group(encoded, x, params, times=t)
```

## How it works

Per block (details and pseudocode in [docs/ENCODER.md](docs/ENCODER.md)):

1. **Grid selection & quantization**: Picks a power-of-two step `2^e` (up to 2¹⁶ steps across the block's range), the coarsest exact decimal step `10^p`, or a scaled rational step `N / D · 10^p` (or `2^e`) found via approximate GCD of sample differences. Outliers are escaped separately.
2. **Noise floor**: Estimates top-band noise via clipped MAD of 12th differences with lag-1/lag-2 autocorrelation whiteness checks (plus lower-subband checks for band-limited DAQs and a first-difference cap). Skipped on blocks < 128 samples and on irregular/sparse timestamp cadences.
3. **Fixed polynomial predictors**: Selects difference order 0–3 per block by residual variance (as in Shorten and FLAC), wrapping residuals mod 2¹⁶.
4. **Zigzag & plane shuffle**: Zigzag-encodes signed residuals so small magnitudes have leading zero bits, then transposes across the block group into bit planes (or byte planes) so high planes become long zero runs.
5. **Zstandard framing**: Compresses the entire block group in one zstd frame (with per-plane zstd blocks at `effort >= 5`).

## Documentation

- [Live report & spec site](https://gboyer.github.io/fluxcode/report/): interactive benchmarks, public dataset results, and rendered spec.
- [docs/SPEC.md](docs/SPEC.md): wire format v1 layout, decoding rules, and conformance tests.
- [docs/ENCODER.md](docs/ENCODER.md): reference encoder algorithm, parameters, guarantees, and verification.
- [docs/PERFORMANCE.md](docs/PERFORMANCE.md): throughput, day-scale stress run, and Numba/SIMD notes.
- [docs/TUNING.md](docs/TUNING.md): measurements behind the parameter defaults and timestamp design.
- [bench/](bench/): benchmark scripts (`bench_gb.py`, `estimate.py`, `decimal_near_miss.py`, `stress.py`, `time_axis.py`, `update.py`, `prune.py`) and saved results.
- [fluxprune/](fluxprune/README.md): optional lossy prepass that retains only the shape-carrying points of a series.
- [experimental/](experimental/): historical codec exploration and public dataset benchmark scripts (reference only; not covered by main package guarantees).

## Related work

- [Gorilla](https://www.vldb.org/pvldb/vol8/p1816-teller.pdf) (VLDB 2015) and [Chimp](https://www.vldb.org/pvldb/vol15/p3058-liakos.pdf) (VLDB 2022): lossless XOR float64 time-series codecs.
- [ALP](https://doi.org/10.1145/3626717) ([code](https://github.com/cwida/ALP)) (SIGMOD 2024): lossless float compression via decimal detection, the closest relative to fluxcode's decimal grids.
- [Bitshuffle](https://github.com/kiyo-masui/bitshuffle) ([paper](https://arxiv.org/abs/1503.00638)) and [Blosc](https://www.blosc.org/): bit- and byte-shuffle filters in front of general-purpose compressors.
- [FLAC](https://www.rfc-editor.org/rfc/rfc9639#name-fixed-predictor-subframe) and [Shorten](https://en.wikipedia.org/wiki/Shorten_(file_format)): fixed polynomial predictors for integer residual coding.

## Development

```sh
make init          # install locked root and experimental environments
make check         # CI checks: ruff check/format, experimental smoke, mypy --strict, ty, offline pytest
make check-rust    # rustfmt, clippy, and SIMD bit-plane tests
make test          # offline unit test suite
make test-public   # opt-in tests against downloaded public datasets
make datasets      # download & verify public real-world datasets (honors FLUXCODE_DATA_DIR)
make public-bench  # benchmark fluxcode on public datasets
make public-report # render public dataset HTML/SVG report
uv run python bench/bench_gb.py --gib 0.05  # quick synthetic benchmark (full run is 1 GiB)
```

### Code layout (`fluxcode/`)

- `_api.py`, `_types.py`, `_args.py`: public functions, `Params` / result dataclasses, argument validation.
- `_group.py`, `_time_blocks.py`, `_basic_stats.py`: block-group assembly/parsing, fixed-duration time bucketing, summary statistics.
- `_compress.py`: compression policy (efforts, plane layout, flush points, framing); `rust/src/compress.rs` is its Rust port.
- `_encoder.py`, `_decoder.py`: per-block Numba kernels (`ENCODER.md §5`, `SPEC.md §5`).
- `_time.py`, `_time_rounding.py`: exact timestamp coding, cadence detection, and `time_error` rounding / lattice fitting.
- `_grids.py`, `_rational.py`, `_noise.py`, `_escapes.py`, `_extreme_magnitudes.py`: quantization grids, noise-floor estimation, outlier escapes, extreme magnitudes.
- `verify.py` (with `_verify_common.py`, `_verify_decode.py`, `_verify_times.py`, `_verify_values.py`): guarantee verifier (`ENCODER.md §6`); not imported on the encode path.
- `_format.py`, `_body.py`, `_bitpacking.py`: wire-format constants/layout, body serialization & `update` splicing, bit/byte-plane transposition (`rust/src/format.rs`, `bitpacking.rs`, `planes.rs`).

> **Numba cache note:** Kernels use `cache=True`, which tracks only the defining file's mtime, not inlined helpers or constants from other modules. Tests always compile fresh (`tests/conftest.py`), and benchmark scripts import `bench/_fresh_numba_cache.py` to key the cache on a hash of `fluxcode/` sources. If you run standalone scripts after editing internals, clear stale caches with `find fluxcode -name '*.nb[ic]' -delete`.

## License

[MIT](LICENSE). Copyright (c) 2026 Garry Boyer.
