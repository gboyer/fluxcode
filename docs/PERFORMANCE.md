# fluxcode performance report

2026-09-24; sizes updated 2026-09-25 for the current format (unit header and per-block anchors).
Timings are from a quiet rerun on 2026-09-25 (AC power, idle, fastest of 5 runs). How fast fluxcode encodes and decodes a day of data from a sensor fleet (a synthetic
day-scale stress test), where the time goes, and which inputs slow it down. Raw output: [bench/STRESS_RESULTS.md](../bench/STRESS_RESULTS.md);
harness: [bench/stress.py](../bench/stress.py).

## Summary

The target workload is 1000 tags at 1 kHz, stored as 1-minute units of 60 one-second blocks,
for one day: 1000 x 1440 x 60 x 1000 = 86.4e9 samples, 691 GB of float64. On 4 performance cores
of an Apple M3 (MacBook Air, Mac15,13):

| phase | wall time | throughput | per block, per worker |
|---|---|---|---|
| encode | **71.4 s** | 9.7 GB/s (1.21e9 samples/s) | 3.31 µs |
| decode | **39.3 s** | 17.6 GB/s | 1.82 µs |

The day compresses to 40.8 GB (3.78 bits/sample, 17x). Encode runs at about 1210x real time, so
keeping up with the live stream takes about 0.3% of one core; speed only matters for
backfill. The worst input found (a NaN scattered through every block) takes encode to 108 s per
day, 1.5x the baseline. Performance is more than sufficient for this use case.

## Method

- **Workload.** `bench/stress.py` runs the full tag x unit grid through the public
  `encode_unit` / `decode_unit`, one call per unit, with the summary statistics and zstd included.
  Workers claim whole tags (a day each) from a shared counter. Threads (default) and processes
  are both supported.
- **Data.** A preset sensor-fleet mix (`sensor-mix`): 35% analog process values (16-bit ADC), 25% held /
  deadband values on a 0.01 grid, 10% digital 0/1, 10% vibration stored as float32, 5% totalizers
  on a 0.001 grid, and 5% each of a 0.1-rounded sensor, a noisy sine and a random walk. The
  weights are a guess; `--data` takes any weighted list of the 25 kinds.
- **Data reuse.** Generating 691 GB would take far longer than coding it, so each kind has a pool
  of 32 distinct one-minute units (15 MB per kind, larger than the caches), cycled across tags.
  fluxcode keeps no state between units, so the per-unit work is what distinct data would cost.
- **NaN.** By default each tag gets 3 NaN minutes per day in 2 runs at random sample offsets, not
  aligned to blocks or units (0.35% of units touched). `--sprinkle` scatters isolated NaNs
  through every unit.
- **Checks.** Every pool unit is decoded and verified before timing: non-finite positions exact,
  and every finite error within the quantization bound.
- **Not measured.** Parsing the ingest format, storage I/O, and cold-cache reads of compressed
  data. The decode phase reads its compressed input from the pool, which is small and cache-hot.

## Results

### Scaling and overhead

- **Python overhead is negligible.** `encode_unit` costs the sum of its stages to within a few
  µs per unit (for example 174 µs total against 89 + 28 + 55 for analog).
- **The GIL is not a bottleneck.** Threads and processes give the same throughput, and 99-100%
  of worker wall time is spent inside codec calls.
- **Load balance.** An earlier static split of tags across workers left some idle for the second
  half of a run, because kinds differ in cost (77-213 µs per unit). Claiming tags dynamically
  fixed it.
- **Thermals.** The machine is fanless. Encode drifted from 10.1-10.3 to about 9.5 GB/s over the 71 s
  phase. Runs longer than about 2 minutes, such as a multi-day backfill, weren't measured and
  would likely slow further.

### Where the time goes (one unit, single thread, µs)

| kind | bits/sample | kernels | write_unit | zstd | encode_unit | decode_unit |
|---|---|---|---|---|---|---|
| analog | 5.91 | 89 | 28 | 55 | 174 | 81 |
| held | 0.27 | 86 | 28 | 22 | 138 | 94 |
| digital | 0.01 | 39 | 28 | 8 | 77 | 70 |
| sensor-0.1 | 1.77 | 108 | 28 | 57 | 190 | 100 |
| random-walk | 12.70 | 95 | 29 | 87 | 213 | 88 |

Encode is about half numba kernels (quantization, order pick, residual), a fixed ~28 µs for
`write_unit` (the bit-plane shuffle), and 8-87 µs of zstd depending on entropy.

### Adversarial inputs (full-day times, extrapolated from 250-tag runs)

| input | encode | decode | vs baseline encode |
|---|---|---|---|
| baseline mix, 3 NaN min/tag/day | 71 s | 39 s | 1.00x |
| every tag a random walk (12.6 bits/sample) | 89 s | 39 s | 1.25x |
| 120 NaN min/tag/day in 20 runs (8% of samples) | 73 s | 39 s | 1.02x |
| isolated NaNs, 0.05% of samples (~40% of blocks) | 84 s | 42 s | 1.18x |
| isolated NaNs, 1% of samples (every block) | 108 s | 49 s | 1.51x |

- **NaN runs cost nothing measurable.** Blocks that are entirely NaN are cheap; only the two
  partial blocks at the ends of each run take the non-finite path.
- **Scattered NaNs are the worst case.** A tag with intermittent bad-quality samples sends every
  affected block down the non-finite path. Before the `noise_finite` rewrite (branch-free
  loops, scratch buffers allocated once per unit), the 1%-scattered case took 137 s; it now takes
  108 s.
- **Decimal detection near misses.** A block on a decimal grid except for one late sample makes
  each candidate exponent scan the whole block before failing: 4.28 µs/block against 3.53 for
  plain data (single thread; [bench/decimal_near_miss.py](../bench/decimal_near_miss.py), results in
  STRESS_RESULTS.md). `decimal_detection=False`
  avoids it (3.48 µs/block). The note is
  in the spec, §3.2.

### Timestamps (`--times`, 2026-09-27)

The same day with an exact timestamp per sample (datetime64[ns]; docs/SPEC.md §2a). Each tag gets
one clock kind, from a pool like the values: `clock-mix` is 60% a perfect 1 kHz grid, 30% a grid
with a few gaps (Poisson, mean 2 per minute, each 5 ms to 3 s) and 10% a noisy host clock
(σ = 20 µs jitter at µs resolution). Full day, 4 threads, AC power:

| run | encode | decode | compressed | bits/sample |
|---|---|---|---|---|
| no timestamps | 81.4 s | 41.9 s | 40.78 GB | 3.78 |
| `--times clock-mix` | 92.5 s (+14%) | 53.9 s (+29%) | 48.51 GB (+19%) | 4.49 |

Per clock kind, every tag on that clock (250 tags, the same session; percentages against no timestamps):

| clock | encode µs/block/worker | decode µs/block/worker | bits/sample |
|---|---|---|---|
| none | 3.60 | 1.88 | 3.75 |
| perfect grid | 4.13 (+15%) | 2.26 (+20%) | 3.76 |
| grid with a few gaps | 4.27 (+19%) | 2.39 (+27%) | 3.76 |
| noisy clock | 7.87 (+119%) | 4.70 (+150%) | 10.85 |

- **Grids cost bandwidth, not bits.** A regular block stores 24 bytes before compression, but
  encode reads and decode writes 8 bytes of ticks per sample, as many as the values. Single
  threaded that is +16 µs (+9%) to encode and +12 µs (+11%) to decode a unit
  (`bench/time_axis.py`); on 4 threads, which share memory bandwidth, it is the 15-20% above.
- **Noisy clocks dominate the mix.** The 10% of tags on a noisy clock add 7.1 bits/sample of real
  jitter entropy, most of the day's extra 7.7 GB, and each irregular block goes through the GCD,
  the reference, 32 residual planes and zstd.
- **32 residual planes, 64 only for long blocks** (a residual of 2^32 or more: in ns ticks, a gap
  of seconds). zstd scans zero bytes at about 9 GB/s, so the 32 dead planes per block cost 26-38 µs
  of encode per irregular unit (the day: 101.1 s before, 92.5 s after), at the same size.
- **The per-block reference** (the rounded mean or the minimum of the quotients) cut the day's
  timestamps from 9.14 to 7.73 GB (the noisy clocks from 12.16 to 10.86 bits/sample) at no
  measurable encode cost and about 2% on decode (alternating quarter-day runs, 3 each).
- **No regression without timestamps.** Alternating quarter-day runs of this version and the
  version before the time axis (3 each) differ by about 1.5% (3.74 against 3.70 µs/block encode,
  1.92 against 1.89 decode), within the run-to-run spread of a fanless machine; sizes are
  byte-identical. `bench_gb.py` shows the same (within 1-2% either way).

### Single thread, per signal type (`bench/bench_gb.py`)

A separate benchmark from the day-scale stress run above: 1 GiB of float64 (134,100 blocks, 15 signal
types) through the public API on one thread, idle machine on AC power. Full tables:
[bench/RESULTS.md](../bench/RESULTS.md). Per-block times are lower than the stress run's 3.31 µs
because there is one worker and no thermal drift, and the signal mix differs.

| | bits/sample | encode µs/block (MB/s) | decode µs/block (MB/s) |
|---|---|---|---|
| default (noise floor 0.25) | 4.88 | 3.17 (2521) | 1.86 (4308) |
| noise floor off | 6.93 | 2.59 (3091) | 1.77 (4524) |
| target 6 bits/sample | 3.75 | 4.27 (1872) | 1.98 (4044) |

Default encode ranges from 2.3 µs/block (linear) to 4.6 (chirp). Timings on battery with
other applications running were 1.5–1.8× slower, so benchmark on AC power with the machine idle.

## Implementation notes

None of these change the format.

**Structure**
- **Compiled calls per unit, never per block.** The encoder stages for all blocks of a unit run
  in one `@njit` call (`_encoder.encode_unit`), and the serializer (`_format.write_unit`) in a
  second one, writing every field straight into one output buffer. Allocations are per unit and
  small (the 120 KB unit buffer costs 0.2 µs). zstd contexts are reused, one per thread.
- **Rare inputs stay off the common path.** Blocks holding NaN or ±inf are handled in
  `_nonfinite.py`. Subnormal ranges, ranges at or past 2^1023 and overflowing sums are handled in
  `_extreme_magnitudes.py`. Each is reached through one branch per block, so the clean path in
  `_encoder.py` and `_decoder.py` reads as the plain algorithm. The noise estimator and its
  constants live in `_noise.py`, shared by the clean and non-finite variants.
- **Parallelism:** none inside the codec. Units are independent, and the kernels release the
  GIL, so threads and processes scale the same.

**Encode**
- **Order selection on the first 250 samples** agrees with a full-block pick on 97% of blocks,
  costs 0.4% in size on continuous data (none on discretized), and cuts the pick from ~2.7 to
  ~0.15 µs. One fused pass computes all four sums of squares.
- **Single-pass residual per order:** the first `order` samples separately, then one
  branch-free loop per order (difference, then mod 2^16). LLVM vectorizes it.
- **Quantize** (`floor(x·2^-e + 0.5)`) is a straight vectorizable loop, separate from detection.
- **Noise estimate** is compiled with `fastmath=True`; its float sums otherwise run serially at
  FP-add latency (7.2 → 1.5 µs/block). That's safe because only the encoder evaluates it. The
  format-defining arithmetic (quantize, reconstruct) is compiled without fastmath. Blocks holding
  NaN or inf use a separate weighted copy (`_nonfinite.noise_finite`), so the weights cost clean blocks nothing.
- **Decimal detection** exits on the first off-grid sample, typical for continuous data (~0.4 µs
  per block). The successful candidate's quantized values are kept instead of quantizing again.
- **Bit-shuffle** uses an 8×8 bit transpose per 8 bytes: three mask-and-shift steps on a uint64
  holding 8 consecutive low (or high) residual bytes (Hacker's Delight `transpose8`), its own
  inverse.

**Decode**
- **Fused per block:** unshuffle (the same transpose), un-zigzag, integrate, dequantize.
  Integration is a serial prefix sum per order (mod 2^16 in int32 with a mask); it stays scalar.
  Dequantize is vectorizable.

**Leave alone**
- **zstd** is the largest single stage of encode for noisy data (table above). zstd levels ≤ 1
  cost 1–2 bits/sample, and stronger codecs (zstd-19, bzip2, lzma) cost 50–100× the encode time.
  A background recompression of cold data with zstd-19 is the only sensible variation, since
  decode speed barely changes.

**numba pitfalls** (each blocks LLVM's vectorizer; times per 1000-sample block):
- **`q[i - k]` in a loop** keeps numba's negative-index wraparound check, and the loop runs
  scalar: residual orders 2 and 3 took 1.4–1.7 µs. Index offset views from 0 (`a, b = q[k:], q[:n-k]`
  and `a[i] − b[i]`): 0.2 µs.
- **int64 products** have no NEON multiply. Compute differences in int32 (they fit: |d3| < 2^19)
  and widen for the square: the order pick went from 0.47 to 0.15 µs.
- **Float min/max reductions** don't vectorize even with fastmath: use four independent chains
  (0.87 → 0.40 µs). Keep the sum out of fastmath's no-NaN assumption (`reassoc` only), so the
  non-finite check it feeds survives.
- **Zigzag in int16 arithmetic** (0.06 µs), not int32 (0.42 µs); store bit planes through a 2-D
  (16, N·n/8) view (0.72 µs vs 1.04 µs through computed flat offsets).
- **Histograms** (the target's estimate): four interleaved sub-histograms, so consecutive
  increments of the same bin don't wait on each other (1.4 → 1.25 µs).
- **fastmath reassociates across inlined helpers:** `x * s1 * s2` can become `x * (s1 * s2)`,
  which overflows. The first multiply of a two-step scale lives in a function compiled without
  fastmath (`_extreme_magnitudes._scaled`).
- **Small code changes can move a hot loop's codegen:** an overflow fallback in `_nonfinite.fill_nonfinite`
  that tested `isfinite(x[i])` instead of the codes slowed the whole function from 0.7 to
  1.15 µs/block, even though the fallback never ran. Time the stage before and after a refactor.
- Benchmark stages with every output used: LLVM removes unused reductions, which makes them look free.

## Optimizations considered and not taken

**Fusing `write_unit` into the encode kernel.** Rejected. It would couple the encoder to the
format writer that `update` shares, and a measurement showed it would save almost nothing:

| part of `write_unit` (one unit, data in cache) | µs |
|---|---|
| whole `write_unit` | 27-29 |
| allocating the 120 KB buffer | 0.2 |
| zigzag | 2 |
| 8x8 bit transposes | ~13 |
| byte stores into the 16 bit planes | ~12 |

The cost is the transpose and the stores, which a fused encoder would still do; there's no
separate memory pass to remove. An earlier estimate of ~15% of encode for fusion was wrong.

**A faster shuffle.** Two variants that produce identical bytes were tried: storing one plane at a
time (27.3 to 24.2 µs per unit, about 1.5% of encode), and also skipping the high-byte planes of
blocks whose residuals are all small (21.5 µs on smooth data, but 28.3 to 32.4 µs on a random
walk). At most ~3% of encode, with a regression on noisy data. Not worth the code.

## Reproducing

    uv run python bench/stress.py --breakdown                   # full target, ~2 minutes
    uv run python bench/stress.py --scale 0.25 --sprinkle 0.01  # an adversarial variant
    uv run python bench/stress.py --times clock-mix             # with exact timestamps, ~3 minutes
    uv run python bench/time_axis.py                            # timestamp cost per clock shape, one unit
    uv run python bench/stress.py --list                        # signal kinds and presets
