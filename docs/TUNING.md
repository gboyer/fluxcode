# fluxcode tuning notes

Measurements behind the parameter defaults, and what to check on your data before settling
them. The format and algorithm are in [SPEC.md](SPEC.md).

## Levels

**`level`** (1–9, default 5) sets how hard the encoder works on compression: the residual layout,
block flushes inside the zstd frame and the zstd level (the table is in SPEC.md §1). It never
changes the decoded values. Measured 2026-10-01 through the public API, one thread, against the
encoder before levels existed (both layouts compressed, smaller kept, one block run, zstd 3: 5.174
and 4.009 bits/sample at 9.4 and 8.5 ns/sample). *Report*: the report's 20 signals, 4 minutes each
(4.8 M samples); *random*: 400 one-minute units of 14 random signal families with random
`max_quantize_bits` and noise floor (24 M samples; `experimental/plane_layout/`).

| level | size, report | size, random | encode time, report / random | decode |
|---|---|---|---|---|
| 1 | +1.0% | +2.3% | −2.3 ns/sample (−24%) / −1.8 (−21%) | same |
| 2 | 0.0% | +1.3% | −1.9 ns/sample (−20%) / −1.4 (−16%) | same |
| 3–5 | −2.1% | −0.3% | 0.0 ns/sample (0%) / +0.4 (+5%) | same |
| 6–7 | −2.1% | −2.1% | +2.7 ns/sample (+29%) / +2.2 (+26%) | same |
| 8 | −2.3% | −2.3% | +24 ns/sample (+256%) / +23 (+267%) | up to +0.4 ns/sample (+13%) |
| 9 | −2.6% | −2.8% | +36 ns/sample (+387%) / +33 (+389%) | up to +0.5 ns/sample (+16%) |

Where the gains come from (`experimental/plane_layout/README.md` has the full study):

- **Block flushes** (levels 3–9). zstd codes the literals it doesn't match with one Huffman table
  per block, and one table for every plane fits none of them: the low byte is nearly uniform where
  the high one is peaked, and bit planes differ in density. Ending a block after each dense plane
  gives each its own table, 1–2% smaller, in the same frame (no format change). A plane with at
  most 1/16 of its bytes non-zero doesn't repay a block's header and table, so it shares one.
- **The heuristic layout** (levels 1–5). Byte planes win when the residuals fit in the low byte
  (the high byte is a constant) and on exactly repeating structure; bit planes win when the
  residuals are very narrow (mostly zero: 8 samples per byte escape Huffman's 1-bit floor) or wide
  (the two bytes' statistics pooled in one table fit neither). "Fewer than 5% of residuals reach
  256" captures the wide case; the threshold scan is flat from 1% to 5%. It misses narrow noise and
  near-constant planes, but those units are small: the retry with the other layout under 16 KB
  brings the per-unit miss from 8% to 1.3% (geometric mean against the better layout) for a
  second compression on a quarter to a third of units, all cheap to compress.
- **The small-frame retry in one block run.** Under 1 KB, per-plane tables cost more than they
  save on 75% of units (up to +90%); compressing those again in one run and keeping the smaller
  makes flushing never worse than 3% on any unit. It moves the total by only 0.05%: it guards
  single units, not the aggregate.
- **zstd 7 and 9** (levels 8–9): see the zstd level table below; only worth it with flushes and
  both layouts already in place.

Tried and left out: nibble planes (4 × 4 bits, a format change; best of three layouts adds 0.6
points over best of two for a third compression), separate zstd frames (worse than blocks),
predicting the layout from order-0/1 entropies, repeat searches or an LZ size estimate (no
better, or slower than compressing twice), and narrow-residual exceptions to the heuristic
(worse in every variant tried).

## Bit planes and byte planes

The header records the residual layout: 16 bit planes or 2 byte planes. On clean periodic signals whose cycles repeat across the unit, zstd finds long
matches in the byte planes that the 8-sample bit groups break up: in the prototype on synthetic
minute units, sines at B = 10 went from 2.39 to 1.12 bits/sample (sin-50.3hz 4.09 to 0.99).

Measured on 2026-10-01 (default params, B = 16, zstd level 3, 8 one-minute units per signal), the
winner is consistent per signal kind: it wins on all 8 units, by a similar margin.

| byte planes smaller | vs bit planes | bit planes smaller | byte planes vs bit |
|---|---|---|---|
| square-2.24hz | −18% | quadratic | +358% |
| sin-4.12hz rounded to 0.01 | −17% | sin-50.3hz | +20% |
| sin-4.12hz | −12% | sensor-0.1 | +20% |
| sin-9.87hz, impulses | −6% | random-walk on a 2^−4 grid | +19% |
| linear, random-walk q0.1 | −5% | chirp | +14% |
| noisy sines (4 variants) | −4% | random-walk, q0.001 | +9% |
| | | gauss-spikes | +7% |

Byte planes win on 11 of these 20 signals and bit planes on 9; over the whole set (dominated by
the large random-walk units) byte planes alone are 5.3% bigger and the best of both 2.3% smaller.
Bit planes win where residuals are small, so the high planes are almost all zero, or where nothing
repeats (more exactly: see Levels above). These measurements are from before block flushes,
which favour bit planes.

With a single layout for everything, as the prototype compared them, bit planes were smaller over
its continuous and discretized signal sets (6.08 against 6.71 and 8.14 against 8.66
bits/sample) for B ≥ 13, and byte planes by 0.1–0.2 bits/sample at B ≤ 11, where the residuals
fit in about a byte. Nibble planes (four 4-bit planes) fell between the two and never beat bit
planes overall ([REPORT.md §11](../experimental/REPORT.md#11-power-of-two-quantization-the-fluxcode-design)
in `experimental/`).

**zstd level** is 3 up to level 7, measured on the 2026-10-01 set above (sizes against level 3 with bit planes;
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

Level 3 is the knee (level 1 is used only at level 1, levels 8 and 9 use zstd 7 and 9). Above it, compression time grows much faster than size falls, and the best
of both planes at level 3 saves more than bit planes up to level 12. Level 1 saves only 10% of the
zstd time and doubles the tiny linear and square units. The high levels pay off only on blocky or
periodic signals (square −50%, sin-50.3hz −16% at 19); noisy and random-walk signals gain 1–3%
even at 19.

## Noise floor: measured behaviour

B = 16, orders 0–3, decimal detection, bit-shuffle, minute units. Noisy test signals only:
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

The format is in [SPEC.md §4](SPEC.md#4-time-axis). Measured on one-minute units
(`bench/time_axis.py`, Apple M3, AC power, single thread); the first three rows are the common
shapes, and [experimental/report/time.html](../experimental/report/time.html) shows them in detail:

| timestamps | irregular blocks | bytes | bits/sample | encode time added | decode time added |
|---|---|---|---|---|---|
| grid: regular 1 kHz | 0/60 | 65 | 0.009 | +17 µs (+5%) | +14 µs (+12%) |
| grid + a few gaps (2 per minute) | 2/60 | 177 | 0.024 | +20 µs (+6%) | +17 µs (+14%) |
| noisy clock (σ=20 µs, µs resolution) | 60/60 | 53,422 | 7.12 | +328 µs (+99%) | +152 µs (+129%) |
| 1 kHz, 20 gaps | 19/60 | 1,094 | 0.146 | +88 µs (+27%) | +55 µs (+47%) |
| 1 kHz, 1% dropped | 60/60 | 1,461 | 0.195 | +229 µs (+69%) | +123 µs (+104%) |
| drifting clock (0.99998 ms) | 60/60 | 339 | 0.045 | +194 µs (+59%) | +92 µs (+78%) |
| jitter σ=10 µs, ns resolution | 60/60 | 120,712 | 16.1 | +452 µs (+137%) | +195 µs (+165%) |
| jitter σ=10 µs, µs resolution | 60/60 | 45,903 | 6.12 | +317 µs (+96%) | +139 µs (+118%) |
| Poisson events, mean 1 ms, µs | 60/60 | 87,889 | 11.7 | +452 µs (+137%) | +184 µs (+156%) |
| Poisson events, mean 1 ms, ns | 60/60 | 164,661 | 22 | +524 µs (+158%) | +206 µs (+175%) |
| deadband logging, ms grid | 60/60 | 55,498 | 7.4 | +684 µs (+207%) | +245 µs (+208%) |
| bursts 10 kHz / idle | 60/60 | 102 | 0.014 | +199 µs (+60%) | +101 µs (+86%) |

For scale, the same unit's values take 17,976 bytes, 331 µs to encode (before `level`, with both
layouts compressed, which compresses the time fields twice too) and 118 µs to decode; the percentages are of those
times.
Irregular timestamps can cost more than the values: their entropy is what it is. Even a perfect
grid adds 60,000 int64 ticks to read on encode and write on decode, as many bytes as the values:
on 4 threads, where memory bandwidth is shared, that costs 12–29% ([PERFORMANCE.md](PERFORMANCE.md)).

**Design notes** (measured while designing, on the timestamps above):
- The GCD matters: without it, µs or ms data stored in ns ticks costs 1.4–2.6× more (zstd alone
  doesn't find the grid). With it, the unit's resolution doesn't change the size.
- Plain deltas beat delta-of-deltas and residuals from the nominal grid overall: the grid
  residual saves about 1 bit/sample on jitter around a fixed grid but loses on gaps, drift and
  events. Delta-of-deltas measured worse again with the reference below (for example 2,682 bytes
  against 1,489 for 1% dropped samples).
- The per-block reference: bit planes of raw quotients around a center like 1000 waste about a
  bit per sample, because a spread of ±60 flips planes 3–10 together (and crossing 1024 flips
  more), and each plane is coded on its own. Residuals from the reference cut jitter by 4–16%,
  dropped samples by half and a drifting clock by 80%, and cost about 15 bytes per unit with gaps
  (the `time_ref` column). The mean alone lost 4–17% on skewed deltas, and the median matched
  the mean-or-minimum choice at the cost of a selection per block; the variance rule matched a
  per-block best of both on every shape measured.
- 32 planes, with 64 for long blocks only: zstd scans a run of zero bytes at about 9 GB/s, so
  64 planes per block spent 50–70 µs per irregular unit compressing the dead planes 32–63. Fewer
  planes lose size instead: this libzstd (1.5.7) compresses these bodies 1–3% better above 256 KB
  (splitting blocks where the statistics change), and a unit-wide plane count or per-block widths
  drop most bodies below it. At 32 planes a 60-block unit stays at about 360 KB and its size is
  unchanged; at 24 it drops to 300 KB and some shapes grew 1.6–1.8%.
- Bit planes and byte planes for the residuals are within a few percent either way; bit planes
  match the value residuals.
- Block starts are stored as unsigned increases (monotonic by requirement), which costs 45
  bytes per regular unit against 280 raw. The value anchor is not transformed: XOR with the
  previous anchor (+0.1% overall) and zigzagged integer deltas (−0.03%) were both noise.
