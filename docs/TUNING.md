# fluxcode tuning notes

Measurements behind the parameter defaults, and what to check on your data before settling
them. The format and algorithm are in [SPEC.md](SPEC.md).

## Bit-shuffle against byte planes

Bit-shuffle is smaller overall (continuous data 6.08 vs 6.71 bits/sample, discretized 8.14 vs
8.66) and faster end to end. Above all it bounds the cost of the expensive inputs best: noisy,
wide signals get 4–18% smaller. It loses 5–21% (at most 0.45 bits/sample) on smooth periodic
signals, which were already cheap.

The size advantage holds for B ≥ 13. At B ≤ 11 byte planes are 0.1–0.2 bits/sample smaller: the
residuals fit in about a byte, and the low/high byte split already separates the random bits
from the mostly-zero ones. Bit-shuffle stays mandatory for a single format; if a deployment runs
at B ≤ 11, measure both layouts on its data. Nibble planes (four 4-bit planes) fall between the
two layouts at B ≥ 13 and never beat bit-shuffle overall ([REPORT.md §11](../experimental/REPORT.md#11-power-of-two-quantization-the-fluxcode-design) in
`experimental/`).

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
