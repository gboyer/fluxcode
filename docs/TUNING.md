# fluxcode tuning notes

Measurements behind the parameter defaults, and what to check on your data before settling
them. The format is in [SPEC.md](SPEC.md), the encoder's algorithm in [ENCODER.md](ENCODER.md).

## Effort

**`effort`** (1–9, default 4) sets how hard the encoder works on compression: the residual layout,
block flushes inside the zstd frame and the zstd level (the table is in ENCODER.md §1). It never
changes the decoded values. Efforts 3 and 4 are the encoder as it was before efforts existed (both
layouts compressed, the smaller kept, one block run, zstd 3), and the table below is against it.
Measured 2026-10-02 through the public API, one thread, AC power
(`experimental/plane_layout/api_bench.py`; times are best of 3 over the whole set). *Report*: the
report's 20 signals, 4 minutes each (4.8 M samples); *random*: 400 one-minute block groups of 14
random signal families with random `max_quantize_bits` and noise floor (24 M samples).

Baseline (efforts 3–4): 5.174 and 4.009 bits/sample, encode 5.4 and 4.8 ns/sample, decode 1.7.

| effort | size, report | size, random | encode, report | encode, random | decode |
|---|---|---|---|---|---|
| 1 | +1.4% | +2.5% | −1.9 ns/sample (−36%) | −1.7 ns/sample (−35%) | same |
| 2 | +0.3% | +1.3% | −1.7 ns/sample (−31%) | −1.5 ns/sample (−31%) | same |
| 3–4 (default 4) | 0 | 0 | 0 | 0 | same |
| 5–8 | −2.1% | −2.0% | +1.0 ns/sample (+19%) | +0.8 ns/sample (+17%) | same |
| 9 | −2.9% | −3.2% | +21.1 ns/sample (+391%) | +18.0 ns/sample (+374%) | same |

"Same" decode is within +0.1 ns/sample (up to +6%) of the baseline, about the run-to-run spread.
These are one thread; with several threads efforts 5 and up depend on the Rust extension (below).

Where the gains come from (`experimental/plane_layout/README.md` has the full study):

- **Block flushes** (efforts 5–9). zstd codes the literals it doesn't match with one Huffman table
  per block, and one table for every plane fits none of them: the low byte is nearly uniform where
  the high one is peaked, and bit planes differ in density. Ending a block after each dense plane
  gives each its own table, 1.5–2% smaller, in the same frame (no format change). A plane with at
  most 1/16 of its bytes non-zero doesn't repay a block's header and table, so it shares one. **They
  cost threaded throughput**: python-zstandard holds the GIL inside the call that ends a block
  (`flush(FLUSH_BLOCK)`; a one-shot compress releases it). On the day-scale stress test with 4
  threads, flushing made encoding 40% slower (146 s against 104 s for 1000 tags × a day, in the same
  size), and the 8-thread gigabyte encode 32% slower; one thread is unaffected, and 4 processes
  scale normally. Calling libzstd's streaming API directly through cffi still scaled only 1.8× on 4
  threads, so the fix is a call that compresses a whole block group with its cuts without the GIL:
  the optional Rust extension (`rust/`; `uv sync --extra rust` from a checkout, not on PyPI yet)
  does, and gives the same block groups. With it, effort 5 on the day-scale test (4 threads) encodes
  at 5.98 GB/s against 3.62 without (the default, effort 4: 7.10 against 6.71); block groups with a
  time axis and `update` use it for the compression of a body that Python built (`pack_group`).
  Without the extension, efforts 5 and up suit one thread per process, or throughput that doesn't
  matter.
- **The heuristic layout** (efforts 1–2) compresses once instead of twice: byte planes when the
  residuals are narrow (the high byte is a constant and byte-wise literals model the low byte) and
  on exactly repeating structure; bit planes from about 7 bits of residual up. "Fewer than 1% of
  residuals reach 128" is within 0.1% of picking the better layout per block group on the stress
  test's sensor mix and the report's signals (the plane_layout study's first rule, "fewer than 5%
  reach 256", fit its synthetic corpus but picked byte planes for analog ADC noise, 11% too large).
  It can't see repeats, so it misses on low-entropy repeating or quantized signals. Per input type
  (`experimental/plane_layout/per_input.py`), effort 2 against the encoder before efforts: within
  ±0.1 bits/sample on most, and these are the outliers, which both layouts bring back to the old
  size or better:

  | input | before | effort 2 |
  |---|---|---|
  | sensor drift rounded to 0.1 | 1.78 | 2.13 |
  | held values (report by exception, 0.01 grid) | 0.15 | 0.32 |
  | quantized triangle and square waves | 0.15–2.41 | 0.35–2.73 |
  | integer-period sines | 0.01–0.02 | 0.17–0.21 |
  | random chirps | 4.50 | 4.69 |

  High-entropy signals are not hurt, and are 0.1–0.5 bits/sample smaller with flushing (random
  walks, AR(1), sin-50.3hz, analog ADC data). **Effort 3–4 or higher for tags of held, counter or
  decimal-quantized values**, where the heuristic misses.
- **No retries.** Compressing small frames again (in one block run, or with the other layout) guards
  single small block groups, but zstd's time follows the 120 KB input, not the small output: on the
  sensor mix, retries cost +32 µs per block group (+14%) for 0.02% with both layouts compressed. A
  flushed frame can therefore be larger than one run on tiny block groups (up to +90% under 1 KB,
  tens of bytes), which is why 5 against 4 is a smaller total, not a smaller block group every time.
- **zstd 9 as well as 3** (effort 9). zstd 9 alone is larger than zstd 3 on 20–23% of block groups
  (up to +9%; zstd 7 up to +37%), which cancels most of its gain: −0.5% and −0.7% against effort 5
  alone, −0.8% and −1.2% keeping the smaller per block group. zstd 7 alone gained only 0.1–0.25% and
  was dropped.

Tried and left out: nibble planes (4 × 4 bits, a format change; best of three layouts adds 0.6
points over best of two for a third compression), separate zstd frames (worse than blocks),
predicting the layout from order-0/1 entropies, repeat searches or an LZ size estimate (no
better, or slower than compressing twice), and narrow-residual exceptions to the heuristic
(worse in every variant tried).

## Bit planes and byte planes

The header records the residual layout: 16 bit planes or 2 byte planes. On clean periodic signals
whose cycles repeat across the block group, zstd finds long matches in the byte planes that the
8-sample octets of the bit planes break up: in the prototype on synthetic minute block groups, sines
at B = 10 went from 2.39 to 1.12 bits/sample (sin-50.3hz 4.09 to 0.99).

Measured on 2026-10-01 (default params, B = 16, zstd level 3, 8 one-minute block groups per signal),
the winner is consistent per signal kind: it wins on all 8 block groups, by a similar margin.

| byte planes smaller | vs bit planes | bit planes smaller | byte planes vs bit |
|---|---|---|---|
| square-2.24hz | −18% | quadratic | +358% |
| sin-4.12hz rounded to 0.01 | −17% | sin-50.3hz | +20% |
| sin-4.12hz | −12% | sensor-0.1 | +20% |
| sin-9.87hz, impulses | −6% | random-walk on a 2^−4 grid | +19% |
| linear, random-walk q0.1 | −5% | chirp | +14% |
| noisy sines (4 variants) | −4% | random-walk, q0.001 | +9% |
| | | gauss-spikes | +7% |

Byte planes win on 11 of these 20 signals and bit planes on 9; over the whole set (dominated by the
large random-walk block groups) byte planes alone are 5.3% bigger and the best of both 2.3% smaller.
Bit planes win where residuals are small, so the high planes are almost all zero, or where nothing
repeats (more exactly: see Effort above). These measurements are from before block flushes, which
favour bit planes.

With a single layout for everything, as the prototype compared them, bit planes were smaller over
its continuous and discretized signal sets (6.08 against 6.71 and 8.14 against 8.66
bits/sample) for B ≥ 13, and byte planes by 0.1–0.2 bits/sample at B ≤ 11, where the residuals
fit in about a byte. Nibble planes (four 4-bit planes) fell between the two and never beat bit
planes overall ([REPORT.md §11](../experimental/REPORT.md#11-power-of-two-quantization-the-fluxcode-design)
in `experimental/`).

**zstd level** is 3 (bar efforts 1 and 9, below), measured on the 2026-10-01 set above (sizes against level 3 with bit planes;
compression time against level 3):

| level | bit planes | best of both | compress time | decompress time |
|---|---|---|---|---|
| −1 | +3.0% | +1.9% | 0.49× | 0.63× |
| 1 | +0.3% | −1.7% | 0.89× | 1.14× |
| 3 | 0 | −2.3% | 1× | 1× |
| 6 | −1.1% | −2.7% | 3.2× | 1.14× |
| 9 | −1.3% | −3.1% | 4.5× | 1.14× |
| 12 | −1.8% | −3.7% | 13× | 1.11× |
| 19 | −3.2% | −5.5% | 94× | 1.36× |

Level 3 is the knee (zstd 1 is used only at effort 1, zstd 9 only at effort 9, alongside 3). Above
it, compression time grows much faster than size falls, and the best of both planes at level 3 saves
more than bit planes up to level 12. Level 1 saves only 10% of the zstd time and doubles the tiny
linear and square block groups. The high levels pay off only on blocky or periodic signals (square
−50%, sin-50.3hz −16% at 19); noisy and random-walk signals gain 1–3% even at 19.

## Noise floor: measured behaviour

B = 16, orders 0–3, decimal detection, bit-shuffle, minute block groups. Noisy test signals only:
clean signals are byte-identical at every f. Errors are median RMS over blocks, in units of the
true noise σ, against the input and against the same signal generated without noise
(the noise-floor bench in [experimental/](../experimental/) and the generated report's "Noise floor sweep").

| f | noisy-sine bits/sample | impulses | gauss-spikes | added error vs input | error vs clean signal |
|---|---|---|---|---|---|
| 0 (off) | 13.69 | 11.66 | 12.16 | 0 | 1.000σ |
| 0.01 | 10.65 | 10.08 | 11.13 | 0.002σ | 1.000σ |
| 0.03 | 8.58 | 8.13 | 9.40 | 0.006σ | 1.000σ |
| 0.1 | 7.05 | 6.01 | 7.55 | 0.015–0.025σ | 1.000σ |
| 0.25 | 5.42 | 5.01 | 6.62 | 0.05–0.06σ | 1.002σ |
| 0.5 | 4.38 | 4.00 | 5.69 | 0.10–0.12σ | 1.004–1.009σ |
| 1 | 3.37 | 3.04 | 4.83 | 0.20–0.23σ | 1.025–1.038σ |

- **Each halving of f costs about one bit per sample.** The noise floor exchanges noise entropy
  for storage one-for-one, which is the intended behaviour.
- **Against the clean signal, f ≤ 0.25 is indistinguishable from storing the noise exactly**
  (within 0.2%). f = 0.5 costs under 1%, f = 1 costs 2.5–4%. Recommended range 0.1–0.5, default
  0.25 (indistinguishable from exact against the clean signal), to be tuned on your data.
- **Irregular times: only blocks with a cadence, on consecutive scans** (ENCODER §5.2). A block
  gets the floor only if at least 90% of its intervals are one scan. A swinging-door archive keeps
  only the points a straight line can't predict, so its sparse points look like white noise to the
  gate: measured on them, σ came out 4× the sensor's. On the simulated archives of
  [sdt.html](https://gboyer.github.io/fluxcode/report/sdt.html):
  - at each tag's own CompDev no archive gets the floor: temperature, flow and pH keep under 40%
    of intervals one scan and their decimals decode exactly, and motor-current keeps 80–84%;
  - motor-current, whose noise (1 A) is far above its CompDev (0.2 A), gets it at half its CompDev
    (89–92% one scan, 99 of 120 blocks): 33% smaller, and at a quarter (94–96%, every block): 41%
    smaller, at a max error of 0.125 A.
  - Archives that kept 80–90% of scans got part of the floor from an earlier rule (a ramp from
    50% to 90%): 36% smaller for motor-current at its own CompDev. The all-or-nothing rule gives
    that up so that unanticipated spreads of intervals get no floor rather than a wrong one.
  - Consecutive scans in an archive read about 1.3× the sensor's σ (swinging door kept them
    because they broke the line), so the step there is up to f·1.3σ.
  - Jittered clocks keep the floor: up to ±20% uniform, sd 15% Gaussian or a 10% mean receive
    delay on every block, with gaps or near-duplicate times on up to 10% of intervals.
  - Regular times, or none, give the same steps as before. Without times the samples are taken as
    evenly spaced: store an archive with its times, or with the noise floor off.
- **Deterministic fast signals are deliberately not gated.** The ~50 Hz sine and the chirp keep
  full precision and full cost (6.6 and 7.8 bits/sample at B = 16) at every f, because they are
  signal. Bit-shuffled zstd handles such signals only moderately well. Very clean
  high-frequency periodicity is expected to be rare in industrial data.
- **Discretized noisy data loses decimal exactness** when the noise is coarser than the decimal
  step: the noisy sine on a 0.1 grid goes from exact at 8.88 bits/sample to 4.39 at f = 0.5, the
  same as its continuous version.

## Validating on your data

What to measure per tag before settling `noise_floor_sigma` (and `target_bits_per_sample`, if used):
1. **Gate rate and σ:** the fraction of blocks gated, and the distribution of σ against the
   tag's engineering resolution. A gate rate that swings between 0% and 100% across minutes
   suggests mixed blocks (below).
2. **Mixed blocks:** a block that is part clean, part noisy is gated all or nothing. Look for
   blocks with ρ near the −0.6 threshold (−0.55 to −0.65), and inspect them.
3. **Fast signal plus noise:** high-frequency real content under moderate noise inflates σ
   while pulling ρ toward the threshold. It's the case most likely to coarsen real signal.
   Compare the spectrum above ~50 Hz before and after encoding.
4. **Size against f:** bits/sample at f = 0.1, 0.25, 0.5 per tag class. The synthetic data
   predicts about one bit per halving of f.
5. **Downstream consumers:** check anything that uses high-frequency content (vibration
   analysis, spectral features, anomaly models) with the noise floor on and off. Those tags
   should run with `noise_floor_sigma` off.
6. **Decimal tags:** confirm which tags lose decimal exactness to the noise floor. That's
   expected where noise exceeds the decimal step, and should be acceptable.

## Time axis

The format is in [SPEC.md §3](SPEC.md#3-time-axis). Measured on one-minute block groups
(`bench/time_axis.py`, Apple M3, AC power, single thread); the first three rows are the common
shapes, and [experimental/report/time.html](../experimental/report/time.html) shows them in detail:

| timestamps | irregular blocks | bytes | bits/sample | encode time added | decode time added |
|---|---|---|---|---|---|
| grid: regular 1 kHz | 0/60 | 65 | 0.009 | +20 µs (+6%) | +4 µs (+3%) |
| grid + a few gaps (2 per minute) | 2/60 | 177 | 0.024 | +6 µs (+2%) | +9 µs (+8%) |
| noisy clock (σ=20 µs, µs resolution) | 60/60 | 53422 | 7.123 | +355 µs (+110%) | +121 µs (+102%) |
| 1 kHz, 20 gaps | 19/60 | 1094 | 0.146 | +102 µs (+32%) | +44 µs (+37%) |
| 1 kHz, 1% dropped | 60/60 | 1461 | 0.195 | +279 µs (+86%) | +106 µs (+89%) |
| drifting clock (0.99998 ms) | 60/60 | 339 | 0.045 | +224 µs (+69%) | +80 µs (+67%) |
| jitter σ=10 µs, ns resolution | 60/60 | 120712 | 16.095 | +445 µs (+138%) | +159 µs (+134%) |
| jitter σ=10 µs, µs resolution | 60/60 | 45903 | 6.120 | +358 µs (+111%) | +123 µs (+103%) |
| Poisson events, mean 1 ms, µs | 60/60 | 87889 | 11.719 | +360 µs (+111%) | +158 µs (+133%) |
| Poisson events, mean 1 ms, ns | 60/60 | 164661 | 21.955 | +403 µs (+125%) | +182 µs (+153%) |
| deadband logging, ms grid | 60/60 | 55498 | 7.400 | +572 µs (+177%) | +220 µs (+185%) |
| bursts 10 kHz / idle | 60/60 | 102 | 0.014 | +132 µs (+41%) | +84 µs (+71%) |

For scale, the same block group's values take 17,976 bytes, 323 µs to encode (default effort, with
both layouts compressed, which compresses the time fields twice too) and 119 µs to decode; the
percentages are of those times (bench/time_axis.py, Apple M3, AC power, single thread, 2026-10-13,
at 7fa0991; against the 2026-10-07 run at 4e89772 the encode times are within run-to-run noise,
and decoding irregular times adds 10–30% less). The noise floor runs on block groups with times:
the encode time includes, on regular blocks, the noise estimate (about 7 µs for 60 blocks), and on
irregular ones the cadence check (the median of 15 intervals and one counting pass), then, where 90%
of the intervals are one scan, the estimate on consecutive scans. Against the previous cadence (a 1st-percentile scan interval with a
histogram fallback, at 0581c29) the added encode time is 6–9% lower on jittered and noisy clocks
(the noisy clock 399 → 364 µs), 23% lower on Poisson events (468 → 359 µs, which took the histogram)
and half on bursts (267 → 130 µs). The values come out the same with and without times here.
Irregular timestamps can cost more than the values: their entropy is what it is. Even a perfect grid
adds 60,000 int64 ticks to read on encode and write on decode, as many bytes as the values: on 4
threads, where memory bandwidth is shared, that costs 12–29% ([PERFORMANCE.md](PERFORMANCE.md)).

**Design notes** (measured while designing, on the timestamps above):
- The GCD matters: without it, µs or ms data stored in ns ticks costs 1.4–2.6× more (zstd alone
  doesn't find the grid). With it, the block group's resolution doesn't change the size.
- Plain deltas beat delta-of-deltas and residuals from the nominal grid overall: the grid
  residual saves about 1 bit/sample on jitter around a fixed grid but loses on gaps, drift and
  events. Delta-of-deltas measured worse again with the reference below (for example 2,682 bytes
  against 1,489 for 1% dropped samples).
- The per-block reference: bit planes of raw quotients around a center like 1000 waste about a bit
  per sample, because a spread of ±60 flips planes 3–10 together (and crossing 1024 flips more), and
  each plane is coded on its own. Residuals from the reference cut jitter by 4–16%, dropped samples
  by half and a drifting clock by 80%, and cost about 15 bytes per block group with gaps (the
  `time_ref` column). The mean alone lost 4–17% on skewed deltas, and the median matched the
  mean-or-minimum choice at the cost of a selection per block; the variance rule matched a per-block
  best of both on every shape measured.
- 32 planes, with 64 for long blocks only: zstd scans a run of zero bytes at about 9 GB/s, so 64
  planes per block spent 50–70 µs per irregular block group compressing the dead planes 32–63. Fewer
  planes lose size instead: this libzstd (1.5.7) compresses these bodies 1–3% better above 256 KB
  (splitting blocks where the statistics change), and a group-wide plane count or per-block widths
  drop most bodies below it. At 32 planes a block group of 60 blocks stays at about 360 KB and its
  size is unchanged; at 24 it drops to 300 KB and some shapes grew 1.6–1.8%.
- Bit planes and byte planes for the residuals are within a few percent either way; bit planes
  match the value residuals.
- Block starts are stored as unsigned increases (monotonic by requirement), which costs 45
  bytes per regular block group against 280 raw. The value anchor is not transformed: XOR with the
  previous anchor (+0.1% overall) and zigzagged integer deltas (−0.03%) were both noise.

### Time error

`time_error` = e rounds each block's times once, from the values given (ENCODER §4): to the
coarsest 1-2-5 step of at most 2e mean intervals (the block's span over its intervals, +2%) for
which at least 95% of its times move by at most e times their own interval (the smaller of the
two either side) and none by more than e mean intervals, at the clock's own phase. A block may
instead continue the lattice of a regular block before it (if it is the same clock), or, if no
1-2-5 step divides its period (1024 Hz, 60 Hz), take a lattice of its own, on the same check.
Regular blocks aren't rounded. Updates never round a stored time again: `update_time_blocks`
stores new times exactly, `replace_time_blocks` re-encodes whole blocks (2026-10-12; before, an
update could round a block's stored ticks again, and lattices were limited to e/2 to leave it a
budget). Sizes measured with `bench/time_axis.py --time-error 0.02,0.05,0.1 --reps 1` (one-minute
block groups of 60 × 1000, 2026-10-12): bytes the times add, irregular blocks of 60, and the
largest move in median intervals; "(was …)" is the rule before 2026-10-12, where it differs.

| timestamps | exact | e = 0.02 | e = 0.05 | e = 0.1 |
|---|---|---|---|---|
| grid: regular 1 kHz | 65 B, 0 | 65 B, 0, 0.000 | 65 B, 0, 0.000 | 65 B, 0, 0.000 |
| grid + a few gaps (2 per minute) | 177 B, 2 | 177 B, 2, 0.000 | 177 B, 2, 0.000 | 177 B, 2, 0.000 |
| noisy clock (σ=20 µs, µs resolution) | 53,422 B, 60 | 21,124 B, 60, 0.010 | 3,010 B, 60, 0.050 | 65 B, 0, 0.088 |
| 1 kHz, 20 gaps | 1,094 B, 19 | 1,094 B, 19, 0.000 | 1,094 B, 19, 0.000 | 1,094 B, 19, 0.000 |
| 1 kHz, 1% dropped | 1,461 B, 60 | 1,461 B, 60, 0.000 | 1,461 B, 60, 0.000 | 1,461 B, 60, 0.000 |
| drifting clock (0.99998 ms) | 339 B, 60 | 87 B, 0, 0.020 (was 88) | 93 B, 0, 0.049 (was 89) | 93 B, 0, 0.098 (was 98) |
| jitter σ=10 µs, ns resolution | 120,712 B, 60 | 14,726 B, 60, 0.010 | 63 B, 0, 0.046 | 63 B, 0, 0.046 |
| jitter σ=10 µs, µs resolution | 45,903 B, 60 | 14,702 B, 60, 0.010 | 63 B, 0, 0.042 | 63 B, 0, 0.042 |
| Poisson events, mean 1 ms, µs | 87,889 B, 60 | 83,245 B, 60, 0.001 (was 57,359) | 73,202 B, 60, 0.003 (was 47,498) | 66,009 B, 60, 0.007 (was 39,487) |
| Poisson events, mean 1 ms, ns | 164,661 B, 60 | 83,222 B, 60, 0.001 (was 57,152) | 74,009 B, 60, 0.004 (was 47,567) | 66,078 B, 60, 0.007 (was 39,778) |
| deadband logging, ms grid | 55,498 B, 60 | 55,498 B, 60, 0.000 | 55,498 B, 60, 0.000 | 55,498 B, 60, 0.000 (was 51,270) |
| bursts 10 kHz / idle | 102 B, 60 | 102 B, 60, 0.000 | 102 B, 60, 0.000 | 102 B, 60, 0.000 |
| jitter σ=10 µs, ns, phase 25 µs | 120,653 B, 60 | 14,712 B, 60, 0.010 | 66 B, 0, 0.045 | 103 B, 0, 0.049 |
| jitter σ=10 µs, ns, phase 50 µs | 120,653 B, 60 | 14,667 B, 60, 0.010 | 66 B, 0, 0.044 | 66 B, 0, 0.054 |
| jitter σ=10 µs, ns, phase 123 µs | 120,653 B, 60 | 14,702 B, 60, 0.010 | 74 B, 0, 0.047 | 74 B, 0, 0.047 |
| drift + jitter σ=10 µs, phase 37 µs | 120,648 B, 60 | 14,661 B, 60, 0.010 | 210 B, 1, 0.050 (was 258) | 111 B, 0, 0.100 (was 102) |
| 1024 Hz, jitter 0.5% | 112,889 B, 60 | 9,372 B, 51, 0.020 (was 10,618) | 154 B, 0, 0.050 (was 530) | 100 B, 0, 0.102 (was 151) |
| 60 Hz, jitter 0.5% | 144,082 B, 60 | 6,239 B, 52, 0.020 (was 6,678) | 267 B, 0, 0.051 (was 1,180) | 169 B, 0, 0.100 (was 184) |

- **Clocks keep their sizes:** jittered clocks on round times or free-running, drift, gaps and
  grids are unchanged; drift with jitter improves (258 → 210 B at e = 0.05).
- **Non-decimal clocks gain most:** with the full e for their own lattice, 1024 Hz at e = 0.05
  stores 154 B (was 530) and 60 Hz 267 B (was 1,180).
- **Events are rounded less, by design:** the old rule rounded Poisson events with one step from
  the median interval, moving about 22% of them further than e of their own interval; the 95%
  rule keeps them there, at +40–55% bytes. Deadband logging no longer gains at e = 0.1 (the
  ms grid's GCD is already coarse).
- **The noisy clock** (σ = 20 µs at e = 0.05) stays at 3,010 B: rounding to the nearest grid
  point never moves a time more than half a step, so jitter-tail times can't go back to their
  slots; the times taking both neighbouring intervals as their own also makes jittered clocks
  near the limit drop a step (σ = 5% at e = 0.05: 5.6% of times over e of their own interval at
  q = 100 µs, against 2.9% with the previous interval alone; 1M-sample test, +14% bytes).
- **Encode time** (table below): at e = 0.05, against exact times in the same run, as a share of the
  whole encode with exact times. Rounding pays for itself where it makes blocks regular: jittered
  clocks encode 39–46% faster, 1024 and 60 Hz 39%, drift with jitter 43%, the drifting clock 22%.
  Elsewhere it costs: +5% on a perfect grid (its regular blocks aren't rounded, but they are
  checked), +7% with a few gaps, +14%, +15% and +22% with 20 gaps, 1% dropped and bursts (already on
  their grid), +12% on the noisy clock (still irregular), +25–34% on Poisson events and +9% on
  deadband logging, which gains nothing at these e. Poisson events take about five tries per block:
  four fail, and the one that passes reads every tick.
- **Speedups** (2026-10-13, same bytes): the steps below a failed one are first tested on the
  intervals alone (ENCODER §4), the clamp touches only the runs at either end, and blocks left as
  they are aren't copied. Against the first version of the 95% rule (the same AC benchmark, an hour
  apart): perfect grid +30% → +13%, deadband logging +57% → +35%, the drifting clock +20% → +3%,
  gaps, dropped samples and bursts 6–9 points less, Poisson events +77% → +72% (µs) and +65% → +65%
  (ns). Measured and dropped: testing every step in one pass over the intervals (6 µs per block, as
  there are 14–24 steps from the cap down to 2 ticks); going down a decade at a
  time and back up through the skipped steps (exact too, but slower: Poisson 5.9 → 7.5 µs per block,
  deadband 3.0 → 4.2, since a decade that passes usually overshoots and adds a second full pass).
- **Vectorized** (2026-10-13, same bytes): every candidate (the lattice of the block before, each
  grid, a lattice of its own) is rounded, clamped and checked in one pass (`lattice_error`; a grid
  is the lattice through its rounded first tick) against each tick's allowed move, worked out once
  per block (`tick_allowances`), in float64 chunks of 64 ticks that stop at the chunk a failing
  candidate fails in; `regular_step`, `within_error` and `grid_fails` run in the same branch-free
  chunks. LLVM vectorizes such a chunk only as a loop over slices of its own (an inner loop over a
  range of the whole arrays stays scalar), and the largest move is an integer maximum. Float64
  rounds exactly within 2^48 ticks of the lattice's origin, ties up (an eighth of a tick added
  before flooring); beyond that, or near the int64 limits, the integer path rounds. Clamping, and
  the lattice's smallest multiple, are tested once per pass and taken out of the loop where they
  can't bite (0.92 → 0.76 µs per pass without bounds). The phase's 64 sampled ticks are gathered
  once per block and their residues taken in float64 from the first one's (0.20 → 0.16 µs per
  try). Per block of 1000 ticks: a failing try no longer rounds every tick first and stops within
  64 ticks of where it fails, `regular_step` on irregular ticks 0.54 → 0.02 µs (three calls per
  block), the lattice of the block before 1.22 → 0.63 µs (was `snap_lattice`'s integer
  division and a separate check). `snap_ticks` per block at e = 0.05, against 7908862: Poisson
  7.8 → 3.5 µs, deadband 4.9 → 2.3, the noisy clock 5.0 → 2.1, jittered clocks 3.6 → 1.2, 1024 Hz
  4.2 → 1.6, a perfect grid 0.73 → 0.31.
- **Measured and dropped** (2026-10-13): a stronger interval test (`grid_fails` with each tick's
  own allowed move, and outliers counted as a cover of the forced intervals: at least the larger of
  half of them and those less their adjacent pairs) cut Poisson from 4.5 to 3 tries per block and
  deadband from 2 to 1, but was slower overall (Poisson 3.9 → 4.5 µs per block, clocks +0.3 to
  +0.6 µs): ruling out a step near the passing one reads most of the block (0.4 µs), more than a
  failing try, and its intervals and bounds cost 0.23 µs per block. The order of the steps can't
  save tries: the coarsest step that passes is the answer only once every coarser one has failed
  (by a try or the interval test), so going down from the cap tries each step once and passes
  once, while a decade search adds passing tries, each reading every tick. The phase's 10 × 64
  cost loop (0.12 µs) already vectorizes; in float64 it was slower (its sum doesn't vectorize).

Encode/decode time the times add, per block group, against the values-only 312/110 µs (Apple M3,
AC power, single thread, best of 20, `bench/time_axis.py --time-error 0.02,0.05,0.1`, 2026-10-13,
vectorized):

| timestamps | exact | e = 0.02 | e = 0.05 | e = 0.1 |
|---|---|---|---|---|
| grid: regular 1 kHz | +17/+13 µs | +40/+20 µs | +32/+20 µs | +56/+20 µs |
| grid + a few gaps (2 per minute) | +45/+25 µs | +69/+26 µs | +69/+26 µs | +69/+26 µs |
| noisy clock (σ=20 µs, µs resolution) | +354/+126 µs | +476/+120 µs | +434/+110 µs | +117/+17 µs |
| 1 kHz, 20 gaps | +94/+55 µs | +153/+50 µs | +149/+60 µs | +147/+60 µs |
| 1 kHz, 1% dropped | +293/+116 µs | +382/+119 µs | +385/+118 µs | +429/+119 µs |
| drifting clock (0.99998 ms) | +218/+100 µs | +122/+20 µs | +100/+20 µs | +121/+20 µs |
| jitter σ=10 µs, ns resolution | +422/+164 µs | +470/+111 µs | +88/+20 µs | +117/+20 µs |
| jitter σ=10 µs, µs resolution | +338/+116 µs | +440/+131 µs | +87/+20 µs | +116/+20 µs |
| Poisson events, mean 1 ms, µs | +335/+161 µs | +589/+118 µs | +554/+156 µs | +550/+177 µs |
| Poisson events, mean 1 ms, ns | +373/+176 µs | +517/+139 µs | +545/+157 µs | +542/+135 µs |
| deadband logging, ms grid | +604/+212 µs | +747/+211 µs | +684/+194 µs | +729/+215 µs |
| bursts 10 kHz / idle | +124/+107 µs | +224/+93 µs | +221/+107 µs | +258/+108 µs |
| jitter σ=10 µs, ns, phase 25 µs | +423/+156 µs | +429/+134 µs | +117/+20 µs | +117/+22 µs |
| jitter σ=10 µs, ns, phase 50 µs | +473/+172 µs | +461/+126 µs | +116/+20 µs | +116/+20 µs |
| jitter σ=10 µs, ns, phase 123 µs | +471/+172 µs | +411/+117 µs | +117/+20 µs | +88/+11 µs |
| drift + jitter σ=10 µs, phase 37 µs | +472/+172 µs | +415/+140 µs | +135/+23 µs | +97/+20 µs |
| 1024 Hz, jitter 0.5% | +435/+172 µs | +407/+108 µs | +144/+21 µs | +109/+20 µs |
| 60 Hz, jitter 0.5% | +468/+187 µs | +387/+93 µs | +160/+20 µs | +119/+20 µs |

Earlier rule (2026-10-10), with timings: bytes the times add, encode/decode time they add (and
its share of the values-only 322/111 µs), irregular blocks of 60, and the largest move in median
intervals (Apple M3, AC power, single thread, best of three runs).

| timestamps | exact | e = 0.02 | e = 0.05 | e = 0.1 |
|---|---|---|---|---|
| jitter σ=10 µs, ns | 120,712 B, +417/+170 µs (+130%/+153%), 60 | 14,726 B, +483/+122 µs (+150%/+110%), 60, 0.010 | 63 B, +165/+11 µs (+51%/+10%), 0, 0.046 | 63 B, +136/+9 µs (+42%/+8%), 0, 0.046 |
| jitter σ=10 µs, µs | 45,903 B, +310/+122 µs (+96%/+110%), 60 | 14,702 B, +483/+124 µs (+150%/+112%), 60, 0.010 | 63 B, +170/+12 µs (+53%/+11%), 0, 0.042 | 63 B, +134/+10 µs (+42%/+9%), 0, 0.042 |
| the same (ns), free-running: 25 µs off the epoch's grid | 120,653 B, +424/+165 µs (+132%/+149%), 60 | 14,712 B, +497/+132 µs (+154%/+119%), 60, 0.010 | 66 B, +177/+2 µs (+55%/+2%), 0, 0.045 | 103 B, +142/+2 µs (+44%/+2%), 0, 0.049 |
| 50 µs off | 120,653 B, +425/+161 µs (+132%/+145%), 60 | 14,667 B, +490/+118 µs (+152%/+106%), 60, 0.010 | 66 B, +186/+1 µs (+58%/+1%), 0, 0.044 | 66 B, +144/+1 µs (+45%/+1%), 0, 0.054 |
| 123 µs off | 120,653 B, +442/+161 µs (+137%/+145%), 60 | 14,702 B, +500/+129 µs (+155%/+116%), 60, 0.010 | 74 B, +180/+1 µs (+56%/+1%), 0, 0.047 | 74 B, +143/+0 µs (+44%/+0%), 0, 0.047 |
| noisy clock (σ=20 µs, µs) | 53,422 B, +317/+128 µs (+98%/+115%), 60 | 21,124 B, +499/+122 µs (+155%/+110%), 60, 0.010 | 3,010 B, +495/+105 µs (+154%/+95%), 60, 0.050 | 65 B, +166/+9 µs (+52%/+8%), 0, 0.088 |
| drifting clock (0.99998 ms) | 339 B, +192/+86 µs (+60%/+77%), 60 | 88 B, +198/+11 µs (+61%/+10%), 0, 0.007 | 89 B, +168/+10 µs (+52%/+9%), 0, 0.023 | 98 B, +157/+10 µs (+49%/+9%), 0, 0.050 |
| drift and jitter σ=10 µs, 37 µs off | 120,648 B, +443/+158 µs (+138%/+142%), 60 | 14,661 B, +482/+131 µs (+150%/+118%), 60, 0.010 | 258 B, +181/+14 µs (+56%/+13%), 2, 0.049 | 102 B, +209/+1 µs (+65%/+1%), 0, 0.057 |
| 1024 Hz, jitter 0.5% | 112,889 B, +392/+150 µs (+122%/+135%), 60 | 10,618 B, +551/+130 µs (+171%/+117%), 60, 0.010 | 530 B, +283/+6 µs (+88%/+5%), 2, 0.026 | 151 B, +222/+1 µs (+69%/+1%), 0, 0.048 |
| 60 Hz, jitter 0.5% | 144,082 B, +403/+175 µs (+125%/+158%), 60 | 6,678 B, +525/+125 µs (+163%/+113%), 60, 0.015 | 1,180 B, +363/+15 µs (+113%/+14%), 5, 0.030 | 184 B, +262/+10 µs (+81%/+9%), 0, 0.050 |
| Poisson events, mean 1 ms, ns | 164,661 B, +388/+194 µs (+120%/+175%), 60 | 57,152 B, +469/+142 µs (+146%/+128%), 60, 0.036 | 47,567 B, +442/+140 µs (+137%/+126%), 60, 0.072 | 39,778 B, +421/+121 µs (+131%/+109%), 60, 0.145 |
| deadband logging, ms grid | 55,498 B, +530/+203 µs (+165%/+183%), 60 | 55,498 B, +699/+218 µs (+217%/+196%), 60, 0.000 | 55,498 B, +698/+224 µs (+217%/+202%), 60, 0.000 | 51,270 B, +654/+212 µs (+203%/+191%), 60, 0.100 |

- **About 5× the jitter makes a clock regular.** A time stays on its grid point while its jitter
  is under q/2 = e intervals: at 5σ, 6 in 10 million fall outside. At 2.5σ about 1% do, which
  costs a few KB a minute instead of 63 B (2% jitter at e = 0.05: 2,530 B instead of 127 KB, the
  noisy clock's 3,010 B). Below that the size falls in steps, one per 1-2-5 step (for a 1 ms
  clock at e = …, 0.005, 0.01, 0.025, 0.05, 0.1; chart in the time report), and each pays off on
  its own: there's no threshold below which rounding stops helping.
- **Free-running clocks** round onto their own phase, a multiple of q/10: 66–108 B against 63 on
  round times (a phase between two candidates is up to q/20 off, which takes a little of the
  jitter margin; the 103 B at 25 µs and e = 0.1 is zstd compressing the sine's values differently
  around other start bytes: on zero values both phases cost 47 B). With the grid fixed at the
  epoch's phase they cost 1.2–13.6 KB at e = 0.05 and up to 2.5 KB at e = 0.1.
- **A drifting clock** follows its phase block by block: 88–98 B against 339 exact (on the
  epoch's grid it was 543–594 B at e ≥ 0.05, as times crossed to the next grid point).
- **Clocks no 1-2-5 step divides** (1024 Hz: 976.5625 µs; 60 Hz) can't land on a grid: the
  rounded intervals alternate, and on main before the lattice fit 1024 Hz with 0.5% jitter stored
  3,863 B at e = 0.05 and 2,546 B at e = 0.1 (60 Hz: 2,444 and 275 B). On lattices of their own:
  530 and 151 B (60 Hz: 1,180 and 184), and encoding is faster too, by 159 and 196 µs (−53%, −65%
  of the values-only encode), as the blocks become regular. A lattice needed every time within e/2
  then, so about 10σ of jitter; since 2026-10-12 it takes the same check as the grid (154 and
  100 B, 60 Hz 267 and 169, in the table above). The attempt is
  only made on clocks the grid leaves with unequal one-scan intervals (not on gaps alone), from
  64 sampled ticks, and stops at the first tick that doesn't fit; on noisy clocks it fails it
  costs about +20–90 µs per block group (+6–28%) against main, measured alternately with it.
- **Choosing the phase** costs about 0.25 µs per block (64 sampled ticks, ten candidates, integer
  sums so it vectorizes and doesn't depend on the machine). Per-block phases without the
  hysteresis flipped between two candidates around a clock's phase (135 B instead of 47 on zero
  values); one phase for the whole block group couldn't follow drift (4.6 KB for drift and
  jitter); hysteresis relative to the score (5%) was too tight, since 64 noisy ticks put one
  candidate up to 24% ahead of an equally good neighbour.
- **Regular clocks are faster to encode and decode once rounded**: the regular time path, the
  plain noise estimate, the regular decode loop.
- **Times already on their grid** (the grid with gaps, 1% dropped, bursts) don't change; checking
  them cost about +80 µs per block group (+26%) on a regular grid (regular blocks aren't
  rounded) and +87 µs (+29%) with a few gaps under the earlier rule (time report, 2026-10-10);
  +15 and +24 µs under the 95% rule, vectorized (2026-10-13).
- **Events and deadband logging** have no grid to land on. Under the earlier rule events got
  1.5–4× smaller, with times moved by up to 1.45 e of the overall median; under the 95% rule,
  1.1–1.3× in µs ticks and 2–2.5× in ns (above).

**Design notes** (measured with prototypes on the patterns above, 2026-10-08):
- **1-2-5 steps, not powers of two:** a power-of-two step doesn't divide a 1 ms period, so the
  rounded times never come back to a grid (jitter σ=10 µs at a step near the median/8: 6,250 B
  against 343 B; the noisy clock 10,932 against 5,652). 1-2-10 has a 5× gap that drops a 5 ms clock
  to a 200 µs step (5,524 B against 43 at e = 0.05).
- **The interval estimate** was the mean of one-scan intervals where there was a cadence (on a
  nice period and a nice e, the median of 15 would flip the step between blocks). Since
  2026-10-12 the step's cap is the span over the intervals and the check decides: gaps inside a
  block only loosen the cap, and clocks take one or two tries. A lattice's period is still a
  cadence estimate: the mean of the intervals near the scan interval. Rounding per sample (not
  fitting a period) keeps the error bounded through drift and gaps.
- **Values are not interpolated onto the rounded times.** Against the physical signal, under
  sampling jitter (each sample taken at its stamp) interpolating is close to ideal and plain
  rounding is off by slope × jitter; under stamp jitter (samples on the grid, stamps late or early)
  it is the other way round. Interpolation also costs value bytes: decimals and integers stop being
  exact (+14% on integer ADC counts to +350% on clean or decimal signals, though those were
  generated on the sample index, so part of it is the jitter put back; free only where the noise
  floor already rounds). The values are kept as measured.
- **Resampling onto a grid** (values interpolated at every grid point) was dominated by rounding
  at the same step: as small on jittered clocks with 3× the error against the input, and 6–150×
  the samples on bursts and deadband logging, 2× when a 1-2-5 grid falls under a clock running
  slightly fast.
- **The lattice limit was e/2** (2026-10-10 to 2026-10-12), so that `update_time_blocks` could
  still round a lattice block whole within e. Updates no longer round stored times, so lattices
  take the full check now (60 Hz at e = 0.05: 267 B, was 1,180).
- **A lattice's period ignores outlying intervals:** with the mean of all one-scan intervals, one
  tick 60 µs late on a clean 1 ms clock of 200 ticks put the whole block on a 1.0003 ms lattice,
  moving the 199 punctual ticks by up to 30 µs to save one residual.
- **Alternatives measured and dropped** (prototypes, 2026-10-10, same block groups):
  - An octave (bit-length) histogram of the intervals with the 5th percentile as the interval:
    on a jittered clock the 5th percentile is below the period, so the step falls a rung (σ=10 µs
    at e = 0.05: 3,076 B against 63), and it costs a full pass.
  - A ladder of coarser 1-2-5 steps, each kept if 95% of times stay within e of their own
    interval and all within 2e: noisy clocks become regular (σ=20 µs at e = 0.05: 3,010 → 65 B)
    only by moving their jitter tails beyond e, it moves genuine off-clock stamps onto the clock
    (up to 0.42 intervals without the 2e cap), misses non-decimal clocks, and costs +15–38% encode
    on blocks it doesn't help. The 95% rule adopted on 2026-10-12 caps every move at e mean
    intervals instead of 2e, so it doesn't get this gain.
  - A one-sided cadence test (90% of intervals above c/2, else the smallest of the 15 sampled as
    the interval) kept 95% of event times within e of their own interval but made Poisson events
    60–80% larger. The 95% rule adopted on 2026-10-12 has the same effect (+40–55%).
  - The epoch's grid or the first tick as the phase instead of the phase search: 594–4,916 B on
    free-running and drifting clocks against 74–258 (epoch), 215–222 B against 63–74 (first
    tick); the search costs about 15 µs per block group.
- **The cadence stays the median of 15:** a geometric mean of the intervals took 2.33 µs per block
  against 0.33, and counts 1×/2× mixes with 42% or more two-scan intervals as one scan (a 50% mix
  scored 100%).
