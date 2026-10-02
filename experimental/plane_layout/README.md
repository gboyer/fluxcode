# Can we predict bit planes vs byte planes without compressing both?

**Question.** `planes="best"` compresses every unit twice (bit planes, byte planes) and keeps the
smaller, because nobody had a coherent account of which wins. Is there a cheap statistic of the
residuals that predicts the winner, so a unit is compressed once?

**Short answer.** Partly. One statistic (how much information the residuals' high byte carries)
explains most of the effect and costs almost nothing to compute. Using it alone gives units about
1.2–1.5% larger than the encode-both oracle, against 2.3–4.6% for always using bit planes. The rest
is exactly-repeating structure, and finding that costs more than zstd does.

**Outcome.** `Params.planes` became `Params.effort` (1–9): the revised heuristic at efforts 1–2,
both layouts (the old default, byte for byte) at 3–4, per-plane zstd blocks from 5, see
[What was adopted](#what-was-adopted-paramseffort). The rest of this page is the study as it ran,
against the encoder of its day; `planes="..."` below refers to that API.

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

## Best of 2 / best of 3, and three ways to frame (`combos.py`, `flush_blocks.py`)

A zstd *frame* is a complete, independent compressed stream (header, content size, blocks of at most
128 KiB, no state shared with other frames); separate frames are separate compress calls. Inside one
frame each *block* has its own literal Huffman table, and the streaming API can end a block early
(`FLUSH_BLOCK`), so one frame can still give every plane its own table, with the window shared and
no frame headers. Plane bytes only, bits/sample and size against today's best of pooled bit/byte
planes (3.902 / 4.048 / 5.153 bits/sample on the three sets):

| | pooled (today) | frames | blocks (one frame, flush per group) |
|---|---|---|---|
| bit, fit / held / report | +4.7 / +3.7 / +2.4% | +5.7 / +4.8 / +5.9% | +1.8 / +1.2 / −0.5% |
| byte | +5.6 / +5.5 / +7.5% | +2.9 / +2.7 / +3.6% | +2.9 / +2.7 / +3.6% |
| nibble | +8.3 / +7.7 / +8.1% | +1.0 / +0.7 / +2.1% | **+1.0 / +0.6 / +2.0%** |
| best of bit/byte | 0 | −0.9 / −1.0 / −1.1% | **−1.9 / −1.7 / −2.0%** |
| best of bit/nibble | +2.0 / +1.4 / +1.1% | −0.5 / −0.9 / 0.0% | −1.7 / −1.9 / −2.1% |
| best of byte/nibble | +2.5 / +2.4 / +2.4% | −1.0 / −1.0 / −1.1% | −1.0 / −0.9 / −1.0% |
| best of bit/byte/nibble | −0.8 / −0.8 / −0.5% | −2.1 / −2.1 / −1.9% | **−2.8 / −2.7 / −2.8%** |

- Best of three pooled saves only 0.5–0.8% over best of two (the nibble pooled frame is bad).
  With separate tables it is worth 2–3%, and best of three adds about 0.9 point over best of two.
- Separate *frames* are worse than *blocks* for bit planes (16 frames pay 16 headers and tables, and
  lose matches across planes): bit planes become the best layout in blocks, not in frames.
- Trying every layout and framing (9 combinations) gains nothing over best of three layouts in
  pooled/blocks (−2.9% either way).
- **Blocks need no format change.** The body bytes and the one-frame container are identical, and a
  standard decoder reads the frame as before. `flush_blocks.py` re-frames real units with a flush
  after the columns and after each plane: `fluxcode.decode` returns identical values for every unit,
  and the sizes (best of bit/byte per unit, header included, 3 minutes × 20 signals) are:

| flush after | total size vs today |
|---|---|
| every plane | −1.7% |
| every 4 planes | −1.4% |
| every 8 planes (halves) | −1.2% |

  Per signal it is −3.8% to +9.3%: random walks and noisy units gain 2–4%, while tiny units (a few
  hundred bytes: linear, square wave, quadratic) lose 4–9% from the extra block headers and tables
  (20–70 bytes) and the matches no longer crossing planes. An encoder could flush only when the body
  is large, or compare both ways (a second compress).

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

## The heuristic and block flushing together (`table.py`, `combine.py`, `api_bench.py`)

Nibble planes are left out: they need a format change, and best of 3 needs a third encode.

From a per-unit table of real unit sizes (header included, `table.py`; summarized by `combine.py`)
for bit and byte planes under five framings, sizes against today's `"best"` (two compressions, one
block run). Framings: a block ends after the columns and after **every plane**, every 4th or 8th
plane, or after each **dense plane**: one with more than 1/16 of its bytes non-zero (the rule the
encoder prototype uses). **+ retry**: a flushed frame under 16 KB is also compressed in one block
run and the smaller kept.

| policy | compressions | fit | held out | report |
|---|---|---|---|---|
| today: best of bit/byte | 2 | 0 | 0 | 0 |
| heuristic (`mean(u > 255)` < 5% → byte), one block run | 1 | +1.52% | +1.40% | +1.21% |
| heuristic, flush every plane | 1 | −0.05% | −0.06% | −0.84% |
| heuristic, flush every 4 planes | 1 | +0.26% | +0.23% | −0.48% |
| heuristic, flush every 8 planes | 1 | +0.57% | +0.49% | +0.16% |
| heuristic, flush dense planes | 1 | −0.12% | −0.12% | −0.98% |
| **heuristic, flush dense planes + retry (the encoder prototype)** | 1–2 | **−0.16%** | **−0.17%** | **−0.98%** |
| always bit planes, flush dense planes + retry | 1–2 | +1.23% | +0.67% | −1.12% |
| best of bit/byte, flush every plane | 2 | −1.93% | −1.77% | −2.01% |
| best of bit/byte, flush dense planes | 2 | −2.17% | −1.97% | −2.17% |
| best of bit/byte, flush dense planes + retry | 2–3 | −2.19% | −2.00% | −2.19% |
| oracle over all 10 layout × framing combinations | 10 | −2.22% | −2.03% | −2.19% |

- Flush and the heuristic add up: **the heuristic plus flushing matches today's size with one
  compression instead of two**. Flushing helps the heuristic more than it helps best-of-two,
  because with a table per plane bit planes stop being the poor layout.
- Cutting only after dense planes beats cutting after every plane (−0.12% against −0.05% for the
  heuristic, −2.17% against −1.93% for best-of-two), and it is cheaper: fewer blocks.
- The heuristic's threshold is the same with this framing: byte iff fewer than 5% of residuals
  reach 256 (the scan is flat from 1% to 5%, and worse from 7.5%).
- Flushing helps big units and hurts tiny ones (fit set, heuristic layout, total bytes saved against
  one block run; 'worse' = share of units that grow):

| one-run unit size | units | every plane | dense planes | dense + retry |
|---|---|---|---|---|
| under 1 KB | 349 | 8.5% bigger (86% worse) | 5.2% bigger (75% worse) | +0.6% saved (0% worse) |
| 1–3 KB | 97 | 0.5% bigger | +0.9% saved | +3.3% saved |
| 3–10 KB | 148 | 0.5% bigger | 0.3% bigger | +0.9% saved |
| 10–30 KB | 427 | +0.7% saved | +0.8% saved | +0.8% saved |
| 30–60 KB | 736 | +1.2% saved | +1.2% saved | +1.2% saved |
| over 60 KB | 243 | +2.6% saved | +2.7% saved | +2.7% saved |
| worst unit | | +136% | +91% | +2.9% |
| mean per-unit change | | +1.70% | +0.57% | −1.23% |

  Without the retry the tiny units are still worse (their frames are a few hundred bytes, so every
  block's header and table shows); the retry fixes them at a cost of one extra, cheap compression on
  the 25–35% of units whose frame is small.

### Integration in the encoder (prototype, commit 93fe638)

Everything goes through `_unit._pack`, the one place a body becomes a unit (used by `encode` and
by `update`). Changes, about 90 lines and no format change:

- `_unit.compress_body(body, cuts)`: one zstd frame via the streaming API, ending a block at each
  cut (`COMPRESSOBJ_FLUSH_BLOCK`), content size recorded. `_unit._frame` calls it, with the
  small-unit retry.
- `_format.flush_points(raw_unit, ...)`: the cut offsets, from the layout the body already
  carries (after the columns, after each dense residual plane). Time residual and code planes stay in
  the last block.
- `PlaneMode` gains `"heuristic"`; `_pack` writes the bit body once, computes the share of
  residuals reaching 256 from planes 8–15 (`_bitpacking.high_byte_share`, one pass over 8 plane
  bytes per group) and rewrites the body as byte planes only if it is below 5%.
- Decoders are untouched; `tests/golden.json` is regenerated (the bytes change), a new
  `tests/test_flush.py` checks that the frame is ordinary (one-shot decodable, content size), that
  cuts follow the planes, the retry guarantee and the heuristic's choices. Flushing applies to every
  plane mode.

Through the public API (one thread, 4.8 M samples of the report's signals, 24 M of random
families; times are best of 3 over the whole set, so coarse):

| planes | report: bits/sample | ns/sample | random: bits/sample | ns/sample |
|---|---|---|---|---|
| today, `best` | 5.174 | 9 | 4.009 | 8 |
| today, `bit` | 5.291 | 6 | 4.203 | 5 |
| flush, `heuristic` | 5.128 (−0.9%) | 9 | 4.015 (+0.1%) | 8 |
| flush, `bit` | 5.119 (−1.1%) | 7 | 4.071 (+1.6%) | 7 |
| flush, `best` | 5.065 (−2.1%) | 12 | 3.927 (−2.0%) | 11 |

Flushing is not free: about +65 µs per unit for the cuts and +20 µs for the small-unit retry on a
320 µs encode, so `heuristic` plus flushing matches `best` in size **and in encode time** (the
second compression it saves roughly equals the flush overhead). Decode time is unchanged. Without
flushing, `heuristic` is about 30% faster than `best` and 1.5% larger. The flush overhead is mostly
per-block work inside zstd (one table per block) plus about 2 µs of Python call overhead per cut; it
would shrink in a C implementation.

## What was adopted (`Params.effort`)

The prototype went into the encoder with these changes, each from a follow-up measurement here
(scripts below):

- **One knob, `effort` 1–9**, in place of `planes`: layout (heuristic or both), block flushes and
  zstd level move together, because they interact (with a table per plane, bit planes stop being
  the poor layout; stronger zstd levels prefer byte planes). The table is in docs/SPEC.md §1,
  measured size and speed in docs/TUNING.md (Effort).
- **A different layout rule** (`fleet.py`). The rule above, "fewer than 5% of residuals reach 256",
  fit this study's synthetic corpus, but on the day-scale stress test's sensor mix (analog ADC
  data, held values, counters: not in the corpus) it picked byte planes for analog noise of about
  33 steps, 11% larger than bit planes with per-plane blocks; the mix came out 4.5% larger than
  the encoder before efforts. "Fewer than 1% reach 128" is within 0.1% of the per-unit better
  layout on the sensor mix and the report's signals, and better than the old rule on every set.
- **No retries.** The prototype recompressed flushed frames under 16 KB in one block run, and a
  first port also tried the other layout there (`small_frames.py`: per-unit misses 8% → 1.3%).
  On the sensor mix, with many small units, those passes cost 14% of encode time for 0.02%:
  zstd's time follows the 120 KB input, not the small output. Both were dropped; `retry_rules.py`
  shows no cheaper rule replacing the one-run retry.
- **Block flushes cost threaded throughput, so they are not the default** (the stress test: 4
  threads, 1000 tags × a day). Flushing gives 1.5–2% smaller units with the
  heuristic at the old encode speed on one thread, which is why the first port made it the default
  (effort 4: sensor mix +0.02%, −8% encode time). But python-zstandard holds the GIL in
  `flush(FLUSH_BLOCK)`: the day took 146 s instead of 104 s with threads (same size; 7.6 against
  4.7 GB/s on 4 processes versus 4 threads), and the 8-thread gigabyte encode was 32% slower.
  Every streaming API measured behaved the same (`compressobj`, `stream_writer`, `chunker`: 1.3–1.7×
  on 4 threads against 3.7× for a one-shot compress), and libzstd's `ZSTD_compressStream2` called
  directly through cffi reached 1.8×. A fix needs a call that compresses a unit with its cuts
  without the GIL (a C extension or numba wrapper); until then efforts 3–4 are the old encoder
  and flushing starts at effort 5.
- **The heuristic's misses** are low-entropy repeating or quantized signals (held values, sensor
  drift on a 0.1 grid, quantized periodic waves: up to +0.35 bits/sample, `per_input.py`), which
  both layouts fix; high-entropy signals are not hurt.
- **zstd 9 only alongside zstd 3** (`zstd_levels.py`). zstd 9 alone is larger than zstd 3 on
  22–30% of units (up to +9%), zstd 7 on 26–37% (up to +37%), which cancels most of their gain.
  Keeping the smaller of zstd 3 and 9 never grows a unit; that is effort 9. zstd 7 was dropped.
- **The heuristic reads whichever body is written first** (byte planes on encode, the unit's own
  layout on update), 64 bits at a time: padding is zero, so the share is identical in both
  layouts, and update doesn't convert carried blocks just to measure it.

## Takeaways for the format and the tuning notes

- A one-pass layout choice from the share of residuals above 255 is a defensible fast option
  (about +1.2–1.5% against `"best"` here), and drops the second zstd pass, about a quarter of
  encode time (zstd is 48% of encode with `"best"`, see `../plane_coders/`). With block flushes
  and a revised threshold it became efforts 1–2 (above).
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
uv run python plane_layout/table.py       # per-unit table for the combinations (~3 min), then combine.py
uv run python plane_layout/combine.py
uv run python plane_layout/api_bench.py    # public-API size and speed of each effort (docs/TUNING.md)
uv run python plane_layout/fleet.py        # layout rules and efforts on the stress test's sensor mix (~5 min)
uv run python plane_layout/per_input.py    # bits/sample per input type at efforts 2, 4, 5 (~3 min)
uv run python plane_layout/small_frames.py # the first rule's misses: narrow-residual rules, other layout on small frames
uv run python plane_layout/retry_rules.py  # rules to skip the one-run retry (needs table.py's output)
uv run python plane_layout/zstd_levels.py  # zstd 7 / 9 alone vs alongside zstd 3, per unit (~3 min)
uv run python plane_layout/combos.py      # best of 2 / 3 layouts x three framings (~5 min)
uv run python plane_layout/flush_blocks.py  # one frame with a block per plane: real units, decoder unchanged
uv run python plane_layout/sections.py    # separate frames with timestamps / non-finite codes (~1 min)
uv run python plane_layout/ctxmodel.py    # ideal cost of a context-modelled bit-plane coder
uv run python plane_layout/layouts.py     # bit/byte/nibble/mixed layouts, pooled vs split (~3 min); --codecs for xz, bzip2, zstd 19
```

The `.npz` files (about 110 MB) are not committed. The scripts run against today's encoder, forcing
a layout with `tests/_series.planes` (one block run, as the encoder of the study); the numbers in the
sections before "What was adopted" came from the encoder at 93fe638, so a rerun at HEAD can
differ slightly where block boundaries moved.
