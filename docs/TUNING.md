# fluxcode tuning notes

Measurements behind the parameter defaults, and what to check on your data before settling
them. The format and algorithm are in [SPEC.md](SPEC.md).

## Bit planes and byte planes

**`planes`** chooses the residual layout: `"bit"` (16 bit planes), `"byte"` (2 byte planes), or
`"best"` (the default), which compresses each unit both ways and keeps the smaller (the header
records which). On clean periodic signals whose cycles repeat across the unit, zstd finds long
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
repeats. That split is why `"best"` is the default: it costs a second zstd pass (about +45% on
encode) and never loses. `"bit"` or `"byte"` skips the second pass when a tag's kind is known. A
cheap size heuristic instead of encoding twice was tried and misjudged decimal sensor data by 20%.

With a single layout for everything, as the prototype compared them, bit planes were smaller over
its continuous and discretized signal sets (6.08 against 6.71 and 8.14 against 8.66
bits/sample) for B ≥ 13, and byte planes by 0.1–0.2 bits/sample at B ≤ 11, where the residuals
fit in about a byte. Nibble planes (four 4-bit planes) fell between the two and never beat bit
planes overall ([REPORT.md §11](../experimental/REPORT.md#11-power-of-two-quantization-the-fluxcode-design)
in `experimental/`).

**zstd level** stays 3, measured on the 2026-10-01 set above (sizes against level 3 with bit planes;
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

Level 3 is the knee. Above it, compression time grows much faster than size falls, and the best
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
