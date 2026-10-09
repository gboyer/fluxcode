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
| grid: regular 1 kHz | 0/60 | 65 | 0.009 | +19 µs (+6%) | +13 µs (+11%) |
| grid + a few gaps (2 per minute) | 2/60 | 177 | 0.024 | +21 µs (+6%) | +19 µs (+16%) |
| noisy clock (σ=20 µs, µs resolution) | 60/60 | 53422 | 7.123 | +364 µs (+111%) | +140 µs (+119%) |
| 1 kHz, 20 gaps | 19/60 | 1094 | 0.146 | +99 µs (+30%) | +66 µs (+56%) |
| 1 kHz, 1% dropped | 60/60 | 1461 | 0.195 | +278 µs (+85%) | +120 µs (+102%) |
| drifting clock (0.99998 ms) | 60/60 | 339 | 0.045 | +232 µs (+71%) | +96 µs (+81%) |
| jitter σ=10 µs, ns resolution | 60/60 | 120712 | 16.095 | +442 µs (+135%) | +192 µs (+163%) |
| jitter σ=10 µs, µs resolution | 60/60 | 45903 | 6.120 | +356 µs (+109%) | +133 µs (+113%) |
| Poisson events, mean 1 ms, µs | 60/60 | 87889 | 11.719 | +359 µs (+110%) | +182 µs (+154%) |
| Poisson events, mean 1 ms, ns | 60/60 | 164661 | 21.955 | +399 µs (+122%) | +210 µs (+178%) |
| deadband logging, ms grid | 60/60 | 55498 | 7.400 | +566 µs (+173%) | +241 µs (+204%) |
| bursts 10 kHz / idle | 60/60 | 102 | 0.014 | +130 µs (+40%) | +100 µs (+85%) |

For scale, the same block group's values take 17,976 bytes, 327 µs to encode (default effort, with
both layouts compressed, which compresses the time fields twice too) and 118 µs to decode; the
percentages are of those times (bench/time_axis.py, Apple M3, AC power, single thread, 2026-10-07,
at 4e89772). The noise floor runs on block groups with times: the encode time includes, on regular
blocks, the noise estimate (about 7 µs for 60 blocks), and on irregular ones the cadence check (the
median of 15 intervals and one counting pass), then, where 90% of the intervals are one scan, the
estimate on consecutive scans. Against the previous cadence (a 1st-percentile scan interval with a
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

`time_error` = e lets each time move by up to e times its block's interval (ENCODER §4): times
round to a 1-2-5 step of at most 2e intervals, from the epoch. Measured with
`bench/time_axis.py --time-error 0.02,0.05,0.1` (one-minute block groups of 60 × 1000, Apple M3, AC
power, single thread, Python without the Rust extension, 2026-10-08): bytes the times add, encode
and decode time they add, irregular blocks of 60, and the largest move in median intervals.

| timestamps | exact | e = 0.02 | e = 0.05 | e = 0.1 |
|---|---|---|---|---|
| jitter σ=10 µs, ns | 120,712 B, +465/+179 µs, 60 | 14,678 B, +367/+139 µs, 60, 0.010 | 63 B, +76/+4 µs, 0, 0.046 | 63 B, +75/+11 µs, 0, 0.046 |
| jitter σ=10 µs, µs | 45,903 B, +369/+130 µs, 60 | 14,625 B, +362/+109 µs, 60, 0.010 | 63 B, +67/+13 µs, 0, 0.042 | 63 B, +66/+13 µs, 0, 0.042 |
| noisy clock (σ=20 µs, µs) | 53,422 B, +366/+132 µs, 60 | 21,143 B, +379/+139 µs, 60, 0.010 | 3,010 B, +375/+98 µs, 60, 0.050 | 65 B, +65/+11 µs, 0, 0.088 |
| drifting clock (0.99998 ms) | 339 B, +233/+96 µs, 60 | 272 B, +272/+67 µs, 38, 0.010 | 594 B, +118/+33 µs, 8, 0.050 | 543 B, +90/+27 µs, 4, 0.100 |
| Poisson events, mean 1 ms, ns | 164,661 B, +404/+187 µs, 60 | 57,146 B, +381/+143 µs, 60, 0.036 | 47,544 B, +371/+138 µs, 60, 0.072 | 39,711 B, +332/+129 µs, 60, 0.145 |
| deadband logging, ms grid | 55,498 B, +567/+226 µs, 60 | unchanged, +613 µs | unchanged, +610 µs | 51,259 B, +575/+223 µs, 60, 0.100 |

- **About 5× the jitter makes a clock regular**, if it ticks on round times (multiples of q from
  the epoch; every clock above does). A time stays on its grid point while its jitter is under
  q/2 = e intervals: at 5σ, 6 in 10 million fall outside. At 2.5σ about 1% do, which costs a few
  KB a minute instead of 63 B (2% jitter at e = 0.05: 2,530 B instead of 127 KB). Below that the
  size falls smoothly with e: there's no threshold where it jumps.
- **A clock at another phase** (free-running acquisition) rounds each tick to whichever of two
  grid points is nearer and stays irregular: 1 kHz with 10 µs jitter at e = 0.05: 43 B a minute on a round phase, 1.7 KB 25 µs off it, 13.6 KB 50 µs off (exact: 120 KB). At e = 0.1
  (q = 200 µs) the 25 and 50 µs phases come back to 43 B and 123 µs costs 2.5 KB. Anchoring each
  block's grid to the clock's own phase would remove this; it isn't done yet.
- **Regular clocks are faster to encode and decode once rounded**: the regular time path, the
  plain noise estimate, the regular decode loop.
- **Times already on their grid** (the grid with gaps, 1% dropped, bursts) don't change; finding
  each block's quantum and rounding costs +22 µs per block group on a regular grid (a regular
  block on its grid isn't rounded) and +50–65 µs on irregular ones.
- **A drifting clock** stays regular for a few hundred samples at a time, then a time crosses to
  the next grid point: a few hundred bytes, as exact.
- **Events and deadband logging** have no grid to land on. Events get 1.5–4× smaller (more in ns than in µs ticks); times on a
  1 ms grid already have a coarse GCD and gain only from e = 0.1. Without a cadence the interval
  is the median of 15, a rough estimate: times moved by up to 1.45 e of the overall median.

**Design notes** (measured with prototypes on the patterns above, 2026-10-08):
- **1-2-5 steps, not powers of two:** a power-of-two step doesn't divide a 1 ms period, so the
  rounded times never come back to a grid (jitter σ=10 µs at a step near the median/8: 6,250 B
  against 343 B; the noisy clock 10,932 against 5,652). 1-2-10 has a 5× gap that drops a 5 ms clock
  to a 200 µs step (5,524 B against 43 at e = 0.05).
- **The interval estimate** is the mean of one-scan intervals where there is a cadence: on a
  nice period and a nice e, the median of 15 would flip the step between blocks. Rounding per
  sample (not fitting a period) keeps the error bounded through drift and gaps.
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
- **The cadence stays the median of 15:** a geometric mean of the intervals took 2.33 µs per block
  against 0.33, and counts 1×/2× mixes with 42% or more two-scan intervals as one scan (a 50% mix
  scored 100%).
