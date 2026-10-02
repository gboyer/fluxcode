# fluxcode performance report

2026-09-24; every table rerun on 2026-10-01 (AC power, one run each, load average about 1.6),
last after the int16 `grid_params` column. How fast fluxcode encodes and decodes a day of data
from a sensor fleet (a synthetic day-scale stress test), where the time goes, and which inputs
slow it down. Raw output: [bench/STRESS_RESULTS.md](../bench/STRESS_RESULTS.md);
harness: [bench/stress.py](../bench/stress.py).

**Since these tables, `Params.planes` became `Params.effort`** (TUNING.md, Effort). `planes="best"`
below is the old default, which is now effort 3–4 (the default, 4): the same units, byte for byte.
`planes="bit"` is gone; effort 1 is the fast option (about a third faster, 1.5–2.5% larger). The
tables below are from 2026-10-01; the 2026-10-02 reruns in `bench/` (gigabyte, day-scale, timestamps,
updates, with the default effort) agree within 1–4%, with the same sizes.

Most of the change from the previous tables is `planes="best"`: every unit is compressed twice,
with bit planes and with byte planes, and the smaller kept. That makes the day 2.2% smaller and
encode about 1.5x slower; `planes="bit"` skips the second pass. Variable block sizes alone (a
per-block `block_sizes` column, an 8-byte header, flat per-block offsets in every kernel) cost
encode +1–3% (+2–6 µs per unit) and decode +2–4% (+2–5 µs), and units are 1–7 bytes smaller.
Snapping the grid costs nothing measurable, nor does the int16 `grid_params` column, except on
units of thousands of tiny blocks whose body drops below 256 KB, where libzstd switches
parameters (updates below).

## Summary

The target workload is 1000 tags at 1 kHz, stored as 1-minute units of 60 one-second blocks,
for one day: 1000 x 1440 x 60 x 1000 = 86.4e9 samples, 691 GB of float64. On 4 performance cores
of an Apple M3 (MacBook Air, Mac15,13):

| phase | wall time | throughput | per block, per worker |
|---|---|---|---|
| encode | **104.5 s** | 6.6 GB/s (0.83e9 samples/s) | 4.84 µs |
| decode | **38.1 s** | 18.2 GB/s | 1.76 µs |

The day compresses to 39.9 GB (3.69 bits/sample, 17x). Encode runs at about 830x real time, so
keeping up with the live stream takes about 0.5% of one core; speed only matters for
backfill. The worst input found (a NaN scattered through every block) takes encode to 158 s per
day, 1.5x the baseline. Performance is more than sufficient for this use case. With
`planes="bit"` (the layout before 2026-10-01) the day encoded in 71 s at 40.8 GB.

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

- **Python overhead is negligible.** `encode_unit` costs about the sum of its stages (for
  example 252 µs for analog: 95 for the kernels, 31 + 59 for the bit-plane body and its zstd
  pass, and the rest for the byte-plane body and its pass).
- **The GIL is not a bottleneck.** Threads and processes give the same throughput, and 99-100%
  of worker wall time is spent inside codec calls.
- **Load balance.** An earlier static split of tags across workers left some idle for the second
  half of a run, because kinds differ in cost (77-213 µs per unit). Claiming tags dynamically
  fixed it.
- **Thermals.** The machine is fanless. Encode drifted from 7.1 to about 6.5 GB/s over the 105 s
  phase, and back-to-back runs slow each other: the adversarial runs below started after a 45 s
  pause each and came out 10–15% faster than an earlier back-to-back set. Runs longer than about 2 minutes, such as a multi-day backfill, weren't measured and
  would likely slow further.

### Where the time goes (one unit, single thread, µs)

| kind | bits/sample | kernels | write_unit | zstd | encode_unit | decode_unit |
|---|---|---|---|---|---|---|
| analog | 5.90 | 95 | 31 | 59 | 252 | 78 |
| held | 0.26 | 96 | 31 | 22 | 175 | 90 |
| digital | 0.01 | 44 | 31 | 8 | 102 | 74 |
| sensor-0.1 | 1.77 | 115 | 31 | 60 | 335 | 108 |
| random-walk | 12.67 | 99 | 32 | 87 | 314 | 98 |

`write_unit` and `zstd` are one bit-plane pass; `encode_unit` (default `planes="best"`) does a
bit-plane and a byte-plane pass. Encode is numba kernels (quantization, order pick, residual),
about 31 µs for the bit-plane shuffle, and 8-87 µs of zstd per pass depending on entropy.

### Adversarial inputs (full-day times, extrapolated from 250-tag runs)

| input | encode | decode | vs baseline encode |
|---|---|---|---|
| baseline mix, 3 NaN min/tag/day | 104 s | 38 s | 1.00x |
| every tag a random walk (12.6 bits/sample) | 138 s | 43 s | 1.32x |
| 120 NaN min/tag/day in 20 runs (8% of samples) | 102 s | 38 s | 0.98x |
| isolated NaNs, 0.05% of samples (~40% of blocks) | 116 s | 42 s | 1.11x |
| isolated NaNs, 1% of samples (every block) | 158 s | 51 s | 1.51x |

- **NaN runs cost little.** Blocks that are entirely NaN are cheap; only the two
  partial blocks at the ends of each run take the non-finite path.
- **Scattered NaNs are the worst case.** A tag with intermittent bad-quality samples sends every
  affected block down the non-finite path. Before the `noise_finite` rewrite (branch-free
  loops, scratch buffers allocated once per unit), the 1%-scattered case took 137 s; after it,
  108 s with bit planes only (2026-09-25), and 158 s with both plane passes now.
- **Decimal detection near misses.** A block on a decimal grid except for one late sample makes
  each candidate exponent scan the whole block before failing. With bit planes only that cost
  4.28 µs/block against 3.53 for plain data (2026-09-25); with both plane passes it is within
  run-to-run noise (5.79-5.98 against 5.23-5.80; single thread,
  [bench/decimal_near_miss.py](../bench/decimal_near_miss.py), results in STRESS_RESULTS.md).
  `decimal_detection=False` avoids it. The note is in the spec, §5.3.

### Timestamps (`--times`)

The same day with an exact timestamp per sample (datetime64[ns]; docs/SPEC.md §4). Each tag gets
one clock kind, from a pool like the values: `clock-mix` is 60% a perfect 1 kHz grid, 30% a grid
with a few gaps (Poisson, mean 2 per minute, each 5 ms to 3 s) and 10% a noisy host clock
(σ = 20 µs jitter at µs resolution). Full day, 4 threads, AC power:

| run | encode | decode | compressed | bits/sample |
|---|---|---|---|---|
| no timestamps | 104.5 s | 38.1 s | 39.88 GB | 3.69 |
| `--times clock-mix` | 135.6 s (+30%) | 54.0 s (+42%) | 47.68 GB (+20%) | 4.41 |

Per clock kind, every tag on that clock (250 tags, the same session; percentages against no timestamps):

| clock | encode µs/block/worker | decode µs/block/worker | bits/sample |
|---|---|---|---|
| none | 4.79 | 1.75 | 3.67 |
| perfect grid | 5.37 (+12%) | 2.25 (+29%) | 3.68 |
| grid with a few gaps | 5.64 (+18%) | 2.37 (+35%) | 3.68 |
| noisy clock | 11.23 (+134%) | 4.61 (+163%) | 10.80 |

- **Both plane passes carry the time fields.** Each body holds the time columns and time residual
  planes, so `planes="best"` compresses them twice too: the time axis costs more encode time
  than it did with bit planes only (+14% for the day, 2026-09-27), at the same size.

- **Grids cost bandwidth, not bits.** A regular block stores 24 bytes before compression, but
  encode reads and decode writes 8 bytes of ticks per sample, as many as the values. Single
  threaded that is +17 µs (+5%) to encode and +14 µs (+12%) to decode a unit
  (`bench/time_axis.py`); on 4 threads, which share memory bandwidth, it is the 12-35% above.
- **Noisy clocks dominate the mix.** The 10% of tags on a noisy clock add 7.1 bits/sample of real
  jitter entropy, most of the day's extra 7.8 GB, and each irregular block goes through the GCD,
  the reference, 32 residual planes and zstd.
- **32 residual planes, 64 only for long blocks** (a residual of 2^32 or more: in ns ticks, a gap
  of seconds). zstd scans zero bytes at about 9 GB/s, so the 32 dead planes per block cost 26-38 µs
  of encode per irregular unit (the day: 101.1 s before, 92.5 s after, 2026-09-27), at the same size.
- **The per-block reference** (the rounded mean or the minimum of the quotients) cut the day's
  timestamps from 9.14 to 7.73 GB (the noisy clocks from 12.16 to 10.86 bits/sample) at no
  measurable encode cost and about 2% on decode (alternating quarter-day runs, 3 each; 2026-09-27).
- **No regression without timestamps** (2026-09-27). Alternating quarter-day runs of this version and the
  version before the time axis (3 each) differ by about 1.5% (3.74 against 3.70 µs/block encode,
  1.92 against 1.89 decode), within the run-to-run spread of a fanless machine; sizes are
  byte-identical. `bench_gb.py` shows the same (within 1-2% either way).

### Single thread, per signal type (`bench/bench_gb.py`)

A separate benchmark from the day-scale stress run above: 1 GiB of float64 (134,100 blocks, 15 signal
types) through the public API on one thread, on AC power. Full tables:
[bench/RESULTS.md](../bench/RESULTS.md). Per-block times differ from the stress run's 4.84 µs
because there is one worker and no thermal drift, and the signal mix differs.

| | bits/sample | encode µs/block (MB/s) | decode µs/block (MB/s) |
|---|---|---|---|
| default (noise floor 0.25, `planes="best"`) | 4.76 | 5.13 (1560) | 1.80 (4454) |
| noise floor off | 6.87 | 4.75 (1683) | 1.89 (4244) |
| target 6 bits/sample | 3.39 | 6.04 (1326) | 1.77 (4527) |
| `planes="bit"` | 4.86 | 3.38 (2365) | 2.06 (3892) |

`planes="best"` costs +1.75 µs/block (+52%) to encode for 2.1% smaller output, and decodes 13%
faster: the units that pick byte planes skip the bit transpose. Default encode ranges from
2.8 µs/block (linear) to 7.5 (sin-9.87hz). Timings on battery with
other applications running were 1.5–1.8× slower, so benchmark on AC power with the machine idle.

### Updates (`bench/update.py`)

Single thread, default params, each update against encoding the same series from scratch (full
table: [bench/UPDATE_RESULTS.md](../bench/UPDATE_RESULTS.md)):

| operation | vs encode |
|---|---|
| `update` of 1 of 60 blocks (with or without times) | −13% to −16% (−33 to −52 µs) |
| `update` appending a 61st block | −2% to −6% |
| `update_time_blocks`, 2 s straddling 3 of 60 one-second blocks | −25% to −26% |
| `update_time_blocks`, 2 s straddling 3 of 3,600 blocks of 10 samples | −25% to −34% (−370 to −400 µs) |

An update decompresses the old unit, encodes only the new blocks, and copies every carried block's
bytes into the new body (`_bitpacking.splice_body`), so it skips the kernels and the shuffle for
the rest. Its floor is zstd: with `planes="best"` it compresses twice, as encode does, and builds
the body in the other plane mode by converting the carried blocks one by one. With `planes="bit"`
a one-block update was −40% against encode (on battery).
The 3,600-block rows depend on zstd: their bit-plane body is 248 KB, just under 256 KB, where
libzstd picks the level-3 parameters for smaller inputs, which compress it in 348 µs instead of
148 (full details in UPDATE_RESULTS.md).

## Implementation notes

None of these change the format.

**Structure**
- **Compiled calls per unit, never per block.** The encoder stages for all blocks of a unit run
  in one `@njit` call (`_encoder.encode_unit`), and the serializer (`_bitpacking.write_unit`) in a
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
- **numba slice assignment between arrays is slow** (`dst[a:b] = src[c:d]`, and 2-D slices worse):
  it doesn't vectorize, and in `splice_body` cost 6-10x an explicit loop. Indexing with a runtime
  signed offset (`dst[offset + i]`) also blocks vectorization, through the negative-index
  wraparound check. The fast form slices both rows first and indexes them from 0
  (`_bitpacking._copy_columns` / `_copy_bytes`): 96 → 14 µs for a 60-block splice, and 100 → 40 µs
  when converting plane modes.

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
    uv run python bench/update.py                               # update / update_time_blocks vs encode
    uv run python bench/stress.py --list                        # signal kinds and presets
