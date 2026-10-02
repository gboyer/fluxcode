# Can we predict bit planes vs byte planes without compressing both?

**Question.** `planes="best"` compresses every unit twice (bit planes, byte planes) and keeps the
smaller, because nobody had a coherent account of which wins. Is there a cheap statistic of the
residuals that predicts the winner, so a unit is compressed once?

**Short answer.** Partly. One statistic (how much information the residuals' high byte carries)
explains most of the effect and costs almost nothing to compute. Using it alone gives units about
1.2–1.5% larger than the encode-both oracle, against 2.3–4.6% for always using bit planes. The rest
is exactly-repeating structure, and finding that costs more than zstd does.

## The explanation

Residuals are 16-bit zigzag values `u` per sample. zstd sees either 16 bit-plane streams (8 samples
per byte) or 2 byte-plane streams, and builds **one Huffman table** for all the literals it
doesn't cover with matches. Three effects, from isolating the layouts on i.i.d. Gaussian residuals
(`analyze.py`, first table: bits/sample of each layout above the entropy H(u)):

| residual width (σ) | H(u) | bit planes | byte planes | winner and why |
|---|---|---|---|---|
| ≤ 2 | ≤ 3 | +0.1…0.25 | +0.5…0.65 | **bit.** Skewed byte symbols (the value 0 is most of the stream) can't be Huffman-coded below 1 bit each; the 8-samples-per-byte packing of bit planes escapes that |
| 4–32 | 4–7 | +0.3…0.4 | +0.07…0.13 | **byte.** The whole residual fits in the low byte, and byte-level Huffman captures the joint distribution of its 8 bits, where bit planes pay for treating planes as independent and pooling their statistics. The high byte is a constant, so it costs nothing |
| ≥ 64 | ≥ 8 | +0.5…0.9 | +0.6…1.8 | **bit.** Both bytes carry information. The nearly uniform low byte and the peaked high byte share one Huffman table, which fits neither (cost grows with the width, +1.8 bits at σ = 512) |

So the width of the residuals decides, and the "does the high byte carry information" feature
splits the last row from the middle one. The first row is the one the cheap features don't capture.

Beyond i.i.d. data, **exact repeats** matter. On a signal whose residual sequence repeats (a
periodic signal sampled so the cycle recurs), zstd finds long matches in the byte layout at any lag,
and in the bit layout only when the lag is a multiple of 8 samples. That is why byte planes win by
2–10× on those units, and why they win on clean sines at all.

## What was measured

`corpus.py` draws 3,000 one-minute units from 14 signal families with random parameters (sines at
random and integer periods, noisy sines at 0.3–3,000 SNR, random walks, AR(1) noise, square /
sawtooth / triangle, steps, spikes, ramps, chirps, 40% quantized to decimal or binary quanta, and
`max_quantize_bits` 8–16), encodes each with `planes="bit"` and `"byte"`, and keeps the real unit
sizes and residuals. 2,000 units (seed 1) fit the thresholds, 1,000 (seed 2) are held out, and
`standard.npz` is the report's own signals (`tests/_signals.py`, 8 minutes each, default params).
Synthetic data only.

Excess size over the encode-both oracle (0% = never wrong; "ok" = how often the pick was the
smaller layout):

| rule | fit (2,000) | held out (1,000) | report signals (160) |
|---|---|---|---|
| always bit | +4.62% (45% ok) | +3.69% (47%) | +2.30% (50%) |
| always byte | +5.65% (55%) | +5.58% (53%) | +7.47% (50%) |
| byte iff `mean(u > 255)` < 5% | +1.52% (73%) | +1.40% (73%) | +1.21% (85%) |
| byte iff H(high byte) < 0.25 bit/sample | +1.47% (73%) | +1.32% (74%) | +1.21% (85%) |
| byte iff a greedy-LZ size estimate of both layouts says so | +1.07% (80%) | +0.86% (80%) | +0.98% (76%) |
| H(high) < 0.25 and the LZ estimate | +1.03% (80%) | +0.94% (79%) | +0.00% (95%) |

The oracle is 4.4% smaller than always-bit on the fit set.

By the entropy of the high byte (fit set): below 0.25 bit/sample, byte planes win 63–75% of the
units, by 3–43% in the geometric mean, and bit planes lose 6–12% against the oracle; above it,
byte planes win 11–38% and lose 4–10% when they're picked, and bit planes lose under 1%. The
failures of the cheap rule are in the first group: units where byte planes are *not* better, either
because the residuals are narrow noise (sensor-0.1: bit planes win by 20%, `random-walk q2^-4` by
17%) or because the planes are almost entirely constant, where bit planes cost nothing and byte
planes don't (quadratic: byte planes are 4.6× larger).

## What didn't work

- **An order-0 or order-1 statistic of the low byte** (its entropy, or the entropy given the previous
  byte) to separate narrow noise from repeating structure: worse than no rule (3–5% regret).
  Smooth deterministic signals have low order-0 entropy too, and they are the ones byte planes
  win on.
- **Searching for exact repeats** (best repeat fraction over lags 2–1500): a depth-3 tree on it
  gained 0.1 point over the single threshold, and it costs 90 M compares per unit.
- **The greedy LZ estimate** is the best predictor, but it is 7× slower than zstd on both layouts
  (about 1,300 µs against 170 µs per unit): no use as a cheap predictor, only as a diagnosis.
- Cheap features are cheap: `mean(u > 255)` is 28 µs per unit in numpy (about 10 µs in a numba
  loop over the residuals the encoder has already computed). A second zstd pass is about 85 µs.

## Takeaways for the format and the tuning notes

- A one-pass `planes` choice from the share of residuals above 255 is a defensible fast option
  (about +1.2–1.5% against `"best"` here), and roughly doubles encode speed on units where zstd is
  half of encode. `"best"` stays the right default for size.
- The `docs/TUNING.md` statement "bit planes win where residuals are small or aperiodic" is only half
  right: the data say *narrow* and *wide* residuals favour bit planes (for different reasons) and
  byte planes win in between, and on repeating structure at any width.
- Nibble planes (not tested here; earlier work put them halfway between the two) would sit between
  these effects too: the narrow-residual Huffman penalty shrinks with a 4-bit alphabet, but pooling
  and the loss of cross-nibble structure return.

## Reproduce

```sh
cd experimental
uv run python plane_layout/corpus.py --n 2000 --seed 1 --out corpus.npz   # ~25 s
uv run python plane_layout/corpus.py --n 1000 --seed 2 --out test.npz
uv run python plane_layout/corpus.py --standard --out standard.npz
uv run python plane_layout/analyze.py     # all the tables (the first run extracts features, ~2 min)
```

The `.npz` files (about 110 MB) are not committed.
