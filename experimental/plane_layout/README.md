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

Checked directly for the wide row: compressing the two byte planes *separately* and adding the
sizes, instead of in one stream, saves 0.01 bit/sample at σ ≤ 64, 0.21 at σ = 128, 1.26 at σ = 512
and 1.34 at σ = 2048, so the shared-table cost is real. At σ = 64 (5% of high bytes non-zero) the
byte layout is already worse than bit planes without any pooling cost (8.61 against 8.53): splitting
`u` into two bytes throws away the dependence between them.

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
units, by 7–43% in the geometric mean, and bit planes lose 6–12% against the oracle; above it,
byte planes win 11–38% and lose 4–10% when they're picked, and bit planes lose under 1%. The
failures of the cheap rule are in the first group: units where byte planes are *not* better, either
because the residuals are narrow noise (sensor-0.1: byte planes are 20% larger, `random-walk q2^-4` 17%) or because the planes are almost entirely constant, where bit planes cost nothing and byte
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

## Other layouts and other compressors (`layouts.py`)

Plane bytes only, zstd 3, sizes against today's `"best"` (the best of bit and byte planes, one
shared Huffman table), on the fit, held-out and report sets. A layout is a list of group widths:
bit planes `[1]*16`, byte planes `[8, 8]`, nibble planes `[4]*4`, and mixes such as the low byte
as a byte plane with the high byte as 8 bit planes. **pooled** = one zstd frame, as today;
**split** = each group's stream in its own frame (own Huffman table); **halves** = one frame for
the low byte's groups and one for the high byte's.

| layout | fit | held out | report |
|---|---|---|---|
| bit, pooled (today's bit) | +4.7% | +3.8% | +2.4% |
| byte, pooled (today's byte) | +5.6% | +5.5% | +7.5% |
| byte, split (2 frames) | +2.9% | +2.7% | +3.6% |
| **nibble, split (4 frames)** | **+1.0%** | **+0.7%** | +2.1% |
| nibble-low + byte-high, split | +1.6% | +1.4% | +2.0% |
| bit-low + byte-high, halves | n/a | n/a | +1.1% |
| oracle(bit, byte), split | −0.9% | −1.0% | −1.1% |
| oracle over all 20+ layouts tried | −3.2% | −3.0% | −3.5% |

- **Sharing one Huffman table is a real cost.** Compressing the planes as separate frames helps
  every layout: byte planes 2.5% smaller, and the best-of-two oracle 1–1.1% smaller (frame headers
  and tables included).
- **Nibble planes in separate frames are the best single fixed layout**: 0.7–1.0% above today's
  best-of-two on the random sets, with no choice, no second pass and about the same time. That
  beats the high-byte rule (+1.2–1.5%) on the fit and held-out sets and loses to it on the report
  signals (+2.1% against +1.2%). Earlier work put nibbles halfway between bit and byte planes; that
  was with one pooled frame, where nibbles are 8.3% worse than the oracle.
- **No fixed mix beats the oracle of the pair**, but the headroom is real: choosing among all
  layouts per unit would save a further 3–3.5%. That needs an oracle over about 20 layouts, so only
  a predictor could capture it; none of this has been tried.
- **Other compressors** (200 fit units, bits/sample, both layouts, ms per unit for both):

| codec | bit | byte | byte/bit | oracle | ms |
|---|---|---|---|---|---|
| zstd 3 | 4.084 | 4.086 | 1.000 | 3.888 | 0.1 |
| zstd 19 | 3.948 | 3.792 | 0.961 | 3.723 | 9.4 |
| bzip2 9 | 4.112 | 3.779 | 0.919 | 3.767 | 16.5 |
| xz 6 | 3.911 | 3.726 | 0.953 | 3.686 | 38.8 |
| xz 6, lc=0 lp=0 pb=0 | 3.881 | 3.692 | 0.951 | 3.657 | 39.0 |
| xz 6, lc=0 lp=3 pb=3 | 3.943 | 3.789 | 0.961 | 3.730 | 39.2 |

  Every stronger compressor prefers byte planes, including LZMA, whose literal coder is bit-wise
  with the previous bits of the byte as context. zstd 3 is where the two layouts tie.
  Level 3's Huffman-only literal coding hurts byte planes most. xz is 300× slower for 9% less.
- The order-0 entropy of the residuals averages 4.40 bits/sample on the fit set while the best of
  bit and byte planes under zstd 3 averages 3.92: zstd's gain over a memoryless model comes from
  matches and structure, though in 47% of units a memoryless model would already beat it.

## Separate frames and the rest of the body (`sections.py`)

The body is the per-block columns, the 16 residual planes, the non-finite code planes (flagged
blocks only) and the time residual planes (irregular blocks only), one zstd frame today. Mean
compressed bytes per unit over 18 units per scenario (6 signals × 3 seeds), `today` = best of bit
and byte planes as one frame; **A** = residuals as 4 nibble frames plus one frame for the columns,
code planes and time residuals; **B** = a frame each for the columns, the code planes and the time
residuals; **C** = all of those pooled into the first nibble frame:

| scenario | columns | residuals | code planes | time residuals | today | A | B | C |
|---|---|---|---|---|---|---|---|---|
| plain | 126 | 46,922 | | | 47,041 | +2.8% | +2.8% | +3.1% |
| regular clock | 173 | 46,922 | | | 47,094 | +2.7% | +2.7% | +2.9% |
| clock with gaps | 196 | 46,922 | | 53 | 47,222 | +2.6% | +2.6% | +2.8% |
| noisy clock | 316 | 46,922 | | 54,009 | 100,820 | +1.7% | +1.7% | +3.4% |
| NaN/inf blocks | 151 | 50,721 | 452 | | 51,360 | −0.4% | −0.4% | −0.1% |
| NaN/inf + noisy clock | 342 | 50,721 | 452 | 54,009 | 104,134 | +1.2% | +1.2% | +2.8% |

(The first four columns are each section compressed alone.) The columns and code planes are tiny
(0.1–0.5 KB per unit): putting them in one extra frame (A) or one each (B) makes no difference, and
pooling them into a nibble frame (C) is slightly worse. So the extras don't constrain the idea; they
go in one more frame. A format using several frames needs the frame lengths (zstd frames don't
record their compressed size), a few bytes in the header.

A **noisy clock's time residuals** (µs jitter on a ns clock, 32 planes) cost 54 KB per unit, more than
the values here: the same bit-vs-byte-vs-nibble question applies to them, and was not studied.

## How far could a context-modelled bit-plane coder go? (`ctxmodel.py`)

An EBCOT-style coder was *not* built; this estimates its ideal code length. Each sample is coded as
its bit length (the plane-by-plane significance decision) conditioned on its neighbours' bit lengths,
then the first mantissa bits adaptively and the rest raw; every decision costs −log2 of an adaptive
count estimate. Calibrated on i.i.d. Gaussian residuals against the true entropy, the model is
within 0.02–0.07 bits/sample of it, at every width (σ = 1…2048).

| i.i.d. Gaussian, bits/sample over the entropy | ideal context model | best of zstd bit / byte / nibble-split |
|---|---|---|
| σ ≤ 32 and σ = 512 | +0.02…0.07 | +0.05…0.12 |
| σ = 64, 128, 2048 | +0.06…0.07 | +0.32, +0.40, +0.39 |

So on noise, zstd with the right layout (nibble-split is the best fixed one) is within about 0.05–0.1
bit of the limit most of the time, and loses 0.3–0.4 bit (4–5%) at a few widths where no layout
fits. On the corpus the picture reverses (500 fit units, 160 report units, plane bytes only,
bits/sample against today's best of bit and byte planes):

| | fit | report |
|---|---|---|
| zstd 3, best of bit/byte (today) | 3.78 | 5.15 |
| zstd 3, nibble split | +1.6% | +2.1% |
| ideal: bit length from the previous 2 samples, 2 modelled mantissa bits | +25.7% | +14.6% |
| best of the ideal model and zstd, per unit | −2.8% | −3.3% |

- The ideal model beats zstd on 34% (fit) and 46% (report) of units, but only wins by a few
  percent where it does (noisy sine, random walk, AR(1): 1.00–1.05× of zstd's size), and loses by
  1.4–160× on periodic and stepped signals (sawtooth 2.6×, sine 1.7×, steps 1.4×, integer-period
  sine 160×) where zstd's matches find exact repeats the model can't.
- **Conclusion:** the plane coder's limit is the noise entropy, and zstd with a good layout is
  already near it on noise; the remaining headroom is in structure, which needs prediction or
  matching, not a better bit-plane model. A hybrid choosing per unit would gain 3%, no more.
- A per-sample binary arithmetic coder would also cost about 1 M decisions per unit: not viable as
  a speed play.

## Takeaways for the format and the tuning notes

- A one-pass `planes` choice from the share of residuals above 255 is a defensible fast option
  (about +1.2–1.5% against `"best"` here), and drops the second zstd pass, about a quarter of
  encode time (zstd is 48% of encode with `"best"`, see `../plane_coders/`). `"best"` stays the right default for size.
- The `docs/TUNING.md` statement "bit planes win where residuals are small or aperiodic" is only half
  right: the data say *narrow* and *wide* residuals favour bit planes (for different reasons) and
  byte planes win in between, and on repeating structure at any width.
- Nibble planes in separate frames are the best fixed layout tested (see above), which is a better
  baseline for the heuristic to beat than always-bit.

## Reproduce

```sh
cd experimental
uv run python plane_layout/corpus.py --n 2000 --seed 1 --out corpus.npz   # ~25 s
uv run python plane_layout/corpus.py --n 1000 --seed 2 --out test.npz
uv run python plane_layout/corpus.py --standard --out standard.npz
uv run python plane_layout/analyze.py     # all the tables (the first run extracts features, ~2 min)
uv run python plane_layout/sections.py    # separate frames with timestamps / non-finite codes (~1 min)
uv run python plane_layout/ctxmodel.py    # ideal cost of a context-modelled bit-plane coder
uv run python plane_layout/layouts.py     # bit/byte/nibble/mixed layouts, pooled vs split (~3 min); --codecs for xz, bzip2, zstd 19
```

The `.npz` files (about 110 MB) are not committed.
