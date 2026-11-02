# Richer value grids (2026-11-01)

Can the value grid do better than `a + q·2^e` and exact `(K0 + q)·10^p` on data quantized by a
sensor whose grid the writer doesn't know? Scripts: `grids.py` (detecting an existing grid,
`detect.md`) and `lossy.py` (choosing a lossy step, `lossy.md`). Sizes are real block groups:
the encoder's kernels for order and residuals, `compress` at effort 4, 3 seeds × 60 blocks of 1000.

## Families

| family | grid | header |
|---|---|---|
| today, fixed | today's two grids; a decimal grid only when coarser than the binary grid's *effective* step | none |
| d125 | 10^p·{1, 2, 5}, absolute | 2 bits (reserved flag bits) |
| d1235 | 10^p·{1, 2, 10/3, 5}, absolute | 2 bits |
| nd255 | (K0 + N·q)·B^p / D, B ∈ {10, 2}, D ≤ 255, K0 on the fine grid B^p/D | N, D column |
| nd4095 | the same with D ≤ 4095 | N, D column |
| float | a + q·S, any float64 S (reference) | 8-byte column |

nd* find the step with an approximate GCD of the block's distinct values (smallest gap / m,
seeded by the mean of the one-step gaps, refined by least squares), then `limit_denominator`.
A family re-grids a block only if every sample is within ¼ power-of-two step, today's rule.

## Findings

1. **Today's decimal detection loses on coarse binary grids.** It takes 10^p as soon as that
   is coarser than the 16-bit step, even when the binary indices share zero low bits and are
   effectively coarser: data on 0.25 goes onto 0.01 (indices × 25): **+34%** (44.4 vs 33.1 KB),
   0.5 onto 0.1: +7%. Fix in the current format: skip a decimal grid that isn't coarser than
   the binary grid's effective step (2^e × the power of two dividing all indices).
2. **Zero low bits are nearly free; odd factors are not.** 0.2 data on the 0.1 grid costs 0.1%,
   0.02 on 0.01 2.7%, 0.05 on 0.01 **19%**, 1/3 on 2^e **49%**. Bit planes absorb factors of 2.
3. **d125 helps only 0.05-type grids** (−19%); 0.2 and 0.02 were already almost free. d1235
   adds the 1/3 grids (−49%, and bit-exact: 33% → 100%), a rare case.
4. **nd255 covers everything d125/d1235 do, plus**: °F from 0.1 °C (step 0.18 offset 32) −28%,
   ADC 10 V/4096 (5·2^−11) −19%, 16-bit ADC on part of its range −28%, a calibrated gain
   0.0137 with offset −35%. D ≤ 4095 adds 10 V/4095 ADCs (−23%). Only float catches
   4-20 mA scalings (D = 26214, −29%). No regressions on continuous data beyond the always-
   present N/D column (+8 B per group: a flag bit to omit it would make that 0).
5. **Lossy steps: the ladder doesn't matter at equal error.** On noise-floored blocks, 1-2-5,
   1-2-2.5-5 and 1-2-3⅓-5 are within ±0.03 bits/sample of powers of two at equal RMSE; they
   only change which error a given f gives (pow2 averages 0.72·f·σ). Decimal lossy steps would
   be for readable decoded values, not size.
6. **Unexplained:** on near-constant data (3 levels per block, 2.7 KB groups) binary indices
   with zero low bits beat the compact 0.5 grid by ~100 B per group.

## Follow-up (2026-11-01): the decimal fix, index gcd, and encode cost

- **Finding 1 is fixed on main** (`b48c7a3`): a block exactly on a power-of-two grid coarser
  than its decimal one takes the coarsest such grid. The tables above now run against it.
- **Index gcd** (family `index gcd`): N = gcd of the indices the encoder already has, on its own
  grid (decimal or binary); store q / N with multiplier N. No search. It alone gets every D = 1
  case at full size (0.05, °F text/float, ADC 4096, 16-bit ADC, binary compaction −4%).
- **The rational search is needed only for D > 1**: 1/3, 10 V/4095, calibrated gains, and with
  16-bit N and D (`nd16`, scratch run) also 4-20 mA scalings (−29%): no float step needed.

`timing.py` → `timing.md` (AC power, single thread, kernels as separate passes after encode):

| | index gcd | rational search (gated: no decimal grid, index gcd 1) |
|---|---:|---:|
| continuous, decimal and noisy data | 0.01–0.03 µs (≤ 1%) | 0.08–0.13 µs (2–3%) |
| blocks it re-grids | ~1.0 µs (+15–20%) | ~0.95 µs with the previous block's grid tried first (+10–20%); 2.9 µs without |
| a fitted step that isn't rational (4-20 mA at D ≤ 4095) | – | 0.09 µs with the previous block's failed step remembered; 2.4 µs without |

Each block encodes in 2.3–10.5 µs. A full pass costs ~1 ns/sample, and float divisibility tests
didn't beat integer `%` much (0.92 vs 1.07 µs). Fusing the gcd into the quantize loop would be
the next saving; not measured. Decode cost is nil: N·2^e and D·10^-p are exact doubles.
