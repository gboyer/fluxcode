# Lossy codecs for 1 kHz time series — investigation report

**Question.** How do we store 1000-sample (1 s at 1 kHz) blocks of float64 sensor data in far fewer
than 8 bytes per sample, and what does the loss look like?

**Answer.** [fluxcode](../README.md), the package at the root of this repository. Per 1000-sample
block it [quantizes](https://en.wikipedia.org/wiki/Quantization_(signal_processing)) to a
power-of-two step with up to 2¹⁶ steps across the block's range, or to an exact decimal grid when
every sample sits on one; coarsens the step to a fraction of the noise on blocks of white
measurement noise; picks a fixed polynomial predictor (order 0–3); and zigzags, bit-shuffles and
[zstd](https://www.rfc-editor.org/rfc/rfc8878)-compresses one-minute units of 60 blocks. Units are
self-describing: no index column is needed to decode them. §11 is the investigation behind it;
its format is specified in [docs/SPEC.md](../docs/SPEC.md).

In this report's matrix (12 synthetic signal kinds, one-minute units, §1):

| codec | bits/sample, median over kinds | error, % of block range | notes |
|---|---|---|---|
| fluxcode-16, noise floor f = 0.25 (the default) | 4.39 | RMSE 0.00056% on clean kinds, max ≤ 0.0015%; ≤ f·σ/2 on white-noise blocks | decimal data (sensor-0.1) exact |
| fluxcode-16, noise floor off | 5.23 | RMSE 0.00056%, max ≤ 0.0015% | |
| delta0123-zstd-10, the step before (§7, §9) | 2.07 | RMSE 0.028%; max ≤ 0.049% on the 88% of blocks at B = 10, 0.39% where the 8 bits/sample cap lowers B | range-scaled step; no decimal or noise handling |

On 1 GiB of float64 (15 signal types), fluxcode's defaults average 4.88 bits/sample and take about
3.2 µs to encode and 1.9 µs to decode a block on one core of an Apple M3
([bench/RESULTS.md](../bench/RESULTS.md)).

delta0123-zstd is smaller in the median because of the synthetic sines, whose byte planes
repeat across a minute (§3, §9, §12). Elsewhere the two are close at B = 10: fluxcode-10 is 4–25%
smaller on the spiky kinds, the random walk and the chirp, and equal on the noisy sine, at 1.3× the
RMSE (its power-of-two step is up to 2× coarser). fluxcode is also lossless on decimal data and
stable under edits (§11).

**If a fixed size per sample is required** (seek to any sample without an index, size known in
advance): `cw8-delta1-bfp-e4`, 8-bit deltas with a 4-bit step exponent per 16-sample frame. It's
8.41 bits/sample with the block's min and max, about 16× lower typical error than plain 8-bit
quantization, and never worse than it (max error ≤ 0.196% of range, guaranteed). See
[§10](#10-constant-width-8-bit-formats).

Everything here is on **synthetic data**. The most important next step is to rerun on real signals
(see [Open items](#13-open-items)).

---

## Contents

1. [Setup and method](#1-setup-and-method)
2. [The original codecs](#2-the-original-codecs)
3. [A data artifact: integer frequencies](#3-a-data-artifact-integer-frequencies)
4. [Nonlinear quantization and DPCM](#4-nonlinear-quantization-and-dpcm)
5. [How to score size against error](#5-how-to-score-size-against-error)
6. [Quantize → predict → entropy code, with rate control](#6-quantize--predict--entropy-code-with-rate-control)
7. [The one-shot encoder](#7-the-one-shot-encoder)
8. [Performance: native code and SIMD](#8-performance-native-code-and-simd)
9. [One-minute chunks](#9-one-minute-chunks)
10. [Constant-width ~8-bit formats](#10-constant-width-8-bit-formats)
11. [Power-of-two quantization (the fluxcode design)](#11-power-of-two-quantization-the-fluxcode-design)
12. [Recommendation](#12-recommendation)
13. [Open items](#13-open-items)
14. [Files and how to run](#14-files-and-how-to-run)
- [References](#references)

---

## 1. Setup and method

**Data.** Twelve synthetic signal kinds at 1 kHz (`tslab/common/datasets.py`). Each kind is
generated as continuous one-minute signals (60,000 samples), so consecutive blocks are consecutive
seconds of the same channel rather than repeats. The report's matrix uses 5 one-minute signals
(seeds) per kind, 3,600 blocks in all: a subset of the 10 per kind that the benchmarks draw from.

| name | description |
|---|---|
| linear, quadratic | a ramp of 1 per sample; t²/1000 |
| sin-4.12hz, sin-9.87hz, sin-50.3hz | 100·sin at √17, π², 16π Hz (irrational, see §3) |
| gauss-spikes | uniform noise ±10 + Gaussian spikes (σ = 10 samples, peak 1000), about 3 per second |
| impulses | uniform noise ±10 + single-sample +500 impulses, about 3 per second |
| square-2.24hz | ±100 square wave, hard edges |
| random-walk | cumulative Gaussian steps |
| chirp | 100·sin, a linear sweep √2 → 32π Hz repeating every second |
| noisy-sine | π² Hz sine + Gaussian noise σ = 5 |
| sensor-0.1 | slow drift + √2 Hz wobble, rounded to 0.1 |

**Evaluation unit.** Every codec encodes **one-minute units**: 60 blocks of 1000 samples. A unit's
size is every byte needed to decode it from those bytes alone. Nothing is charged for an external
index unless the decoder reads one, and nothing it needs is left out. Anything that depends on a
region's range stays per 1-second block, because dynamic range can change from second to second:
quantizer ranges, constant-width grids, deviation as a % of range, DCT scaling. Only the framing
and the scope of the entropy coder are per minute. §9 shows why the minute is the right unit: a
back end such as zstd pays a fixed cost per call, and 60 blocks spread it out.

The investigation first measured each 1-second block as its own compressed unit, on 12 single
blocks. Where the sections below keep those figures for their historical point, they're labeled
"per-block (original evaluation)"; current comparisons and the recommendation use per-minute
figures. Timings are Apple M3, one thread.

**Metrics.** Every error is computed per block, relative to that block's full scale (max − min):

- RMSE of the samples
- error of the block mean
- error of the block max and min (negative = peak clipped)
- max |error|
- where relevant, how far a peak moved (in samples)

Each is summarized over a unit's blocks and over the seeds as the **median** and the **worst**
block. Size is reported in bits per sample (raw float64 is 64). Plots show a single second: one
block chosen per signal kind to show its feature (the largest spike, an edge), decoded from its
unit.

**Rigor.** Every codec writes real bytes and decodes each unit from only those bytes, so the byte
counts are real. Every lossless stage is verified to round-trip exactly. Every quantizer is checked
against its |error| ≤ step/2 bound, per block.

---

## 2. The original codecs

Bits/sample are whole one-minute units, median over the 12 kinds (range in brackets where it
varies); RMSE is the median over each kind's blocks, as a range over kinds. The full matrix is in
`report/index.html`.

| codec | bits/sample | typical RMSE | what we learned |
|---|---|---|---|
| **8-bit quantize** (quant8) | 8.13 | 0.11% | Predictable, uniform error; 7.9× smaller than float64. |
| **Binary tree** (bintree-center / bintree-edge: 2-bit narrowing codes, 1-bit leaves) | 3.14 | 0.06–5% | Works as specified: 1001 internal nodes, 1000 leaves, 3002 bits per block. Error grows quickly with frequency (sin-50.3hz: 4.8%). Decoding a leaf to the centre of its half-range clips impulses by 25% but has lower typical error (median 1.1% against 1.8% at the literal edge); decoding to the edge keeps peaks. Code usage is fairly even on most data; 00 (no narrowing) dominates only on noisy and high-frequency signals. |
| **[Piecewise linear](https://en.wikipedia.org/wiki/Piecewise_linear_function), min/max per 16** (pwlinear-minmax16, -merged) | 2.17 (1.23–2.31) | 0.005–23% | Keeps peak *values* exactly, but a single-sample impulse becomes a triangle several samples wide (10% RMSE on impulses), and the ~50 Hz sine and chirp alias badly (8%, 23%). Spec gap: two knots per 16 can sit ~30 samples apart, past the 4-bit offset's 16, so filler knots are needed. Consolidating monotone runs (-merged) saves bits mainly on ramps and slow signals. |
| **[PCHIP](https://en.wikipedia.org/wiki/Monotone_cubic_interpolation)** (pchip-minmax16, same knots) | 2.17 | 0.01–23% | Same bits as piecewise linear, much better on smooth data (sin-9.87hz: 0.45% vs 2.1%); same failure on impulses (11%). |
| **[Swinging door](https://patents.google.com/patent/US4669097A)** (swingdoor-0.5%, swingdoor-2%) | 5.83 / 1.58 (1.17–13.5) | 0–1.7% | The standard historian algorithm. Impulse peaks land on exactly the right sample. Size is data-dependent and unbounded: 13.5 bits/sample on the ~50 Hz sine at 0.5%. |
| **[Gorilla](https://www.vldb.org/pvldb/vol8/p1816-teller.pdf)** (gorilla-xor: XOR floats, lossless) | 65.0 (1.10–66.0) | 0 | No compression (slightly over 64 bits/sample) on arbitrary floats; good only on repeats or rounded values (square wave 1.10, sensor-0.1 24.9). |
| **[DCT](https://en.wikipedia.org/wiki/Discrete_cosine_transform), top 64 coefficients** (dct-top64, extra) | 1.73 | 0.09–22% | Strong on smooth periodic data, poor on spikes (impulses: peaks clipped by 70%). |
| **quant8 + [delta](https://en.wikipedia.org/wiki/Delta_encoding) + [deflate](https://en.wikipedia.org/wiki/Deflate)** (quant8-delta1-deflate, extra) | 1.48 (0.05–5.73) | 0.11% | Same samples as quant8, 5.5× smaller in the median: the first sign that entropy coding does most of the work. |

**Takeaway.** The fixed-format "clever" codecs (tree, piecewise, swinging door) can't adapt their
bit spending to the data. Quantize + entropy code was already the most efficient family, and in
the original per-block evaluation too (quant8-delta1-deflate: 423 bytes per block on average,
2.4× smaller than quant8).

---

## 3. A data artifact: integer frequencies

The first datasets used 4, 10 and 50 Hz. At 1 kHz those repeat exactly every 250, 100 and 20
samples, and deflate compressed the repeats (sine ~50 Hz in 93 bytes). Switching every periodic
signal to an irrational frequency made results realistic (per-block (original evaluation); the
integer-frequency data is no longer generated):

| dataset | quant8-delta1-deflate, integer Hz | irrational Hz |
|---|---|---|
| sin ~4 Hz | 100 B | 234 B |
| sin ~10 Hz | 107 B | 340 B |
| sin ~50 Hz | 93 B | 597 B |

**Lesson:** synthetic benchmarks can flatter dictionary coders. The same concern applies to the
one-minute units (§9): even at irrational frequencies, a minute of a clean sine gives zstd and
deflate near-repeats across its blocks. quant8-delta1-deflate needs 0.56 bits/sample on the ~50 Hz
sine per minute, against 4.8 per block.

---

## 4. Nonlinear quantization and DPCM

**How a signed S-curve works with deltas.** The curve is applied to the residual (next sample
minus the decoder's own running reconstruction), not to the value. This is closed-loop [DPCM](https://en.wikipedia.org/wiki/Differential_pulse-code_modulation):
because the encoder tracks what the decoder will rebuild, quantization error is corrected by later
samples instead of accumulating. The curve maps each code to a delta. The cubic c·u³ + (1−c)·u is a
valid shape for that (fine steps near zero); a true sigmoid is its inverse and blows up at ±1, so
[μ-law](https://en.wikipedia.org/wiki/%CE%9C-law_algorithm) was used as the standard signed S-curve (as in [companding](https://en.wikipedia.org/wiki/Companding)).

**Findings** (6-bit and 4-bit codes, deflated per minute; median over the 12 kinds):

- **The step scale D must be ≥ the largest step in the data.** With D smaller, smooth signals fall
  behind and can't catch up ("slope overload"): at half the largest step, median RMSE is 6.9%
  and up to 29% per kind (the DPCM sweep in `report/index.html`).
- **Curves:** linear steps have the best typical error. S-curves (cubic c ≈ 0.8, μ-law μ ≈ 7) cut the
  worst case on impulses (0.88% → 0.25–0.33% at 6 bits) but lose precision elsewhere. Strong curves
  (μ = 255, c = 1) are worse everywhere.
- **6-bit DPCM** (dpcm6-linear-deflate) is 1.76 bits/sample at 0.049% median RMSE: larger than
  quant8-delta1-deflate (1.48) but 2.3× more accurate. Codes that flip between neighbouring levels
  deflate poorly: more levels resolve more noise, and resolved noise is entropy.
- **4-bit DPCM** (dpcm4-linear-deflate) is 1.01 bits/sample, close to 6-bit quantization + deflate
  (0.89) in the median, but not per kind: on the sines 2.5–3.5× the bits for 3–8× less error;
  on the chirp, impulses and noisy sine fewer bits and more error. Packing two codes per byte
  helps a little per minute (1.38 against 1.48 bits/sample on average), though per block it
  didn't: deflate finds repeats on byte boundaries.
- **A side effect:** on impulses, 4-bit linear DPCM acts as a noise gate (0.45 bits/sample). Its
  smallest step exceeds the noise, so it only fires on the spikes.
- **Structural cost:** the closed loop is inherently sequential (§8). With uniform steps, quantizing
  first and then predicting on integers gives essentially the same result with no loop, which made
  DPCM redundant.

---

## 5. How to score size against error

The standard framework is [rate-distortion](https://en.wikipedia.org/wiki/Rate%E2%80%93distortion_theory): with uniform quantization, each extra bit per sample
halves RMSE (the "6 dB per bit" rule). A single-number score consistent with that:

    RMSE × 256^(bytes/1000)  =  RMSE × 2^(bits per sample)          (lower is better)

or, additively (higher is better; plain 8-bit quantization scores ≈ 0):

    bit gain = effective bits − bits spent
             = log2(FS / (√12 · RMSE)) − 8 · bytes / 1000

The originally proposed 8^(bytes/1000) under-penalizes size: it lets a byte per sample buy a
factor of only 8 in error instead of 256. Median bit gain over the 12 kinds (per minute; RMSE is
each kind's median):

| codec | median bit gain |
|---|---|
| delta0123-zstd-10 / -8 (§7) | 7.9 / 6.2 |
| dpcm6-linear-deflate | 7.2 |
| quant8-delta1-deflate | 6.6 |
| dpcm4-linear-deflate, quant6-delta1-deflate | 5.1–5.4 |
| swingdoor-2%, dct-top64 | 2.7–3.0 |
| pchip-minmax16, bintree-center, pwlinear-minmax16 | 1.5–1.8 |
| swingdoor-0.5% | 0.6 |
| quant8 | −0.1 |

**Caveats:**
- The score assumes the 6 dB/bit slope, which piecewise and tree codecs don't follow.
- Lossless results score infinity.
- RMSE hides spikes.

Comparing rate-distortion curves (and the BD-rate average between them) is the robust method, and
is what the size-vs-error chart in `report/index.html` shows.

---

## 6. Quantize → predict → entropy code, with rate control

**Pipeline** (`tslab/entropy/ratectl.py`; the coders appear as `delta0123-rice`, `delta0123-zstd`,
`delta0123-deflate` and `flac` in `report/rate.html`):

1. `q = round(x / Δ)` — the only lossy step, so |error| ≤ Δ/2 always.
2. Residual = `diff(q, k)` for k ∈ {0, 1, 2, 3}. This is lossless and needs no closed loop:
   decoding is repeated cumulative sums, and nothing drifts.
3. Entropy code the residual. Coders tested:
   - [Rice](https://en.wikipedia.org/wiki/Golomb_coding) (partitioned, FLAC-style)
   - zstd / deflate ([zigzag](https://protobuf.dev/programming-guides/encoding/#signed-ints), byte planes)
   - real [libFLAC](https://xiph.org/flac/format.html) (fitted [linear predictors](https://en.wikipedia.org/wiki/Linear_predictive_coding) on the same integers)

**Rate control.** Bits per sample ≈ [entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory))(residual) − log₂Δ, so Δ is the only knob; the entropy
coder just measures the result. Two policies at a 4 bits/sample budget per unit, on "regime"
minutes: 5 consecutive seconds of each of the 12 kinds, every block rescaled to [0, 1], standing in
for one sensor passing through different regimes (5 units, `make_rate_report.py`):

| coder | shared Δ for the unit (cap 8 bits/sample): median / worst RMSE | fixed 4 bits/sample per block: median / worst RMSE |
|---|---|---|
| delta0123-rice | 0.057% / 0.21% | 0.063% / 3.6% |
| delta0123-zstd | 0.019% / 0.24% | 0.040% / 2.7% |
| delta0123-deflate | 0.021% / 0.23% | 0.042% / 3.3% |
| flac | 0.040% / 0.21% | 0.019% / 3.9% |
| best: rice\|zstd\|flac (best of the three per block) | **0.0078%** / 0.21% | 0.0012% / 2.7% |

(RMSE per block, % of the block's range; median and worst over the units' blocks.)

- **A shared step equalizes error.** Giving every block the same Δ minimizes total squared error for
  a total budget, and it caps the worst block: 0.2% against 2.7–3.9% with each block forced to the
  same bit rate. Fixed-per-block spends 4 bits/sample making smooth blocks absurdly precise and
  starves noisy ones. The medians are closer, and for FLAC and the best-of coder the fixed policy's
  median is lower, because most blocks are smooth.
- **Each coder has a blind spot:**
  - Rice and FLAC spend ≥ 1 bit on every sample even when all residuals are 0. In the matrix
    (ratectl-flac-4b against ratectl-delta0123-zstd-4b), the linear ramp costs 2.45 against 0.15
    bits/sample.
  - zstd and deflate collapse runs of zeros but do worse on noise.
  - FLAC's fitted predictors win on oscillation: 2.5–13× lower error on the sines and the chirp in
    the matrix.
- **Picking the best coder per block** more than halves the error again (0.019% → 0.0078%), but
  means shipping three codecs. Rejected to keep the design simple.
- **Single coder chosen: zstd.** It's simple, vectorizable, and native to Arrow/Parquet, and it has
  no 1-bit-per-sample minimum. Its known weakness is strong oscillation.

---

## 7. The one-shot encoder

The search in §6 bisects Δ, re-encoding the whole unit at each step. The one-shot encoder
(`delta0123-zstd-B`, `tslab/entropy/delta_zstd.py`) fixes all parameters from data that's already
available, so it compresses each unit once:

- **Step** = (max − min) / (2^B − 1), using the block's own range, stored in the unit. Codes fit in B
  bits, and min and max decode exactly.
- **Predictor order** (the fixed polynomial predictors of [Shorten](https://en.wikipedia.org/wiki/Shorten_(file_format)) and FLAC) = argmin over k of
  the variance of `diff(q, k)`, a cheap statistic with no compression.
- **Cap:** if a block's size, estimated from its residual variance, would exceed 8 bits/sample, B
  drops. Inside a shared stream a block can't be re-encoded until it fits, because its compressed
  size isn't known on its own; the original per-block encoder did that, and its noisy blocks at
  B = 10 dropped further (noisy-sine: 0.056% median RMSE per block, 0.028% now).

**The step decision.** Error is then constant relative to each block's own range, rather than in
absolute units as with a shared Δ. That's more robust to occasional very noisy blocks (they don't
change other blocks), at the cost of spending bits on noise in quiet blocks (§11).

**It comes close to the search.** On the regime minutes, delta0123-zstd-10 gives 3.84
bits/sample at 0.028% median RMSE, against 4.00 bits/sample and 0.019% for the searched zstd
pipeline: 4% smaller at 1.5× the error, about half a bit of precision short, with no search.

| B | bits/sample (regime minutes) | median RMSE | max error (guaranteed where B isn't capped) |
|---|---|---|---|
| 8 | 2.83 | 0.113% | 0.196% |
| 10 | 3.84 | 0.028% | 0.049% |
| 12 | 4.41 | 0.0071% | 0.012% |

In the matrix, the cap lowers B on 1% of blocks at B = 8, 12% at B = 10 and 41% at B = 12; there,
the worst max error is 0.39% (a block capped to B = 7).

**Order selection, step by step** (per-block (original evaluation); the comparison script is no
longer in the repository):

| statistic | worst / mean extra bits/sample vs the best order (B = 10) |
|---|---|
| mean \|r\| (first version) | 0.83 / 0.10 |
| **variance of r (final)** | **0.14 / 0.02** |
| histogram entropy | 0.31 / 0.05 |

Mean |r| is fooled by a constant offset, which zstd doesn't care about but order 0's residuals (the
raw codes) carry. Variance ignores it and picks order 0 correctly for impulses and steps. The typical
picks: order 3 for fast oscillation, 2 for slow smooth curves, 1 for noise and random walks, and 0
for impulses. A wrong order can cost 1–5 bits/sample, so choosing it matters; when variance misses,
the runner-up is within 0.14 bits/sample.

**Restricting to orders 1–2** (`delta12-zstd-8`, a performance option) costs 3% in size overall:
0.4 bits/sample on the chirp and on impulses, and almost nothing on the other kinds. Once the order
picker was native, it saved almost no time (§8), so it's not worth it.

**Versus the alternatives.** In bit gain (§5), delta0123-zstd-10 is the best of the classic set
(7.9). At B = 8 it's not: per minute, quant8-delta1-deflate is smaller in the median (1.48 against
1.86 bits/sample, the same samples' error), because raw deflate at level 9 over a whole unit finds
long matches in the clean periodic and smooth kinds (sin-4.12hz: 0.48 against 1.22). The one-shot
encoder is 7–12% smaller on the random walk, impulses and chirp. Per block (original evaluation)
it was smaller on 6 of the 12 kinds, the busy ones (chirp: 660 against 928 bytes per block). 6-bit
DPCM beats it on the 0.1-rounded sensor signal (2.17 bits/sample at 0.021% against 2.51 at 0.11%),
probably because the data sits on a 0.1 grid that DPCM's steps happen to match: a hint for the
resolution floor in §11.

---

## 8. Performance: native code and SIMD

Measured on one core of an Apple M3, single thread, µs per 1000-sample block, per-block (original
evaluation; not re-measured). `bench/native_speed.py` now times the per-minute codecs:

| codec | Python encode / decode | native encode / decode |
|---|---|---|
| quant8 | 388 / 343 | 1.4 / 1.7 |
| quant8-delta1-deflate (level 9) | 47 / 17 | 28 / 5.9 |
| dpcm6-linear-deflate | 1956 / 240 | 51 / 6.8 |
| delta0123-zstd-8 (per block) | 43 / 29 | 15 / 7.9 |
| delta12-zstd-8 | 36 / 28 | 15 / 7.7 |

**What runs natively.**
- **Python reference versions:** numpy calls (native, with SIMD inside) glued by Python. At 1000
  samples, the ~1–5 µs overhead of each call dominates. DPCM, Rice and the quant8 bit writer loop per
  sample in Python.
- **numba versions:** compiled. The NEON ([SIMD](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data)) instructions actually generated are:
  - **Vectorized:** quantizing, and residual + zigzag + max.
  - **Scalar:** the fused order picker (64-bit integer multiplies), byte-plane writes, prefix sums,
    and DPCM. DPCM is inherently sequential within a block.
- **zstd / zlib:** C.

**What made it faster:**
- **Compile the per-sample work.** 3–40× faster than the Python reference.
- **Compute variance from integer sums** instead of `np.std`: 3.7× faster order picking with
  identical picks. Fusing the four orders into one pass saved another 27%.
- **Avoid deflate level 9** (13 µs per block by itself).
- **Don't use threads within a signal.** The front end is memory- and interpreter-lock-bound;
  parallelism belongs across signals, as the orchestrator already does.

---

## 9. One-minute chunks

**Why the minute.** This section is the evidence for the report's evaluation unit (§1). A store
keeps one compressed unit per channel-minute: 60 block headers (each block's order, B, min and
max) followed by 60 blocks of residual planes, compressed together. The per-block min / max / mean
can also sit in index columns beside it, but the decoder doesn't need them.

**Effect of chunking** (delta0123-zstd-8 residual payload without the block headers, `bench/chunking.py`;
fastest of 5 runs):

| compressor | per-block bits/sample | 1-min chunk bits/sample | encode µs/block | decode µs/block |
|---|---|---|---|---|
| zstd-1 | 3.15 | 2.44 (−23%) | 3.6 → 1.2 | 2.0 → 0.7 |
| zstd-3 | 3.13 | 2.37 (−24%) | 4.1 → 1.5 | 2.0 → 0.6 |
| deflate-1 | 3.06 | 2.78 (−9%) | 11 → 5.9 | 2.7 → 1.1 |
| deflate-6 | 2.92 | 2.43 (−17%) | 20 → 28 | 2.6 → 1.0 |
| deflate-9 | 2.91 | 2.34 (−20%) | 24 → 108 | 2.6 → 0.9 |

- **zstd gets 2–3× faster** because its fixed per-call overhead is spread over 60 blocks.
- **Deflate at levels 6–9 gets slower** on 60 KB inputs.
- **Reading one block** now means decompressing its minute: about 40 µs.
- **Most of the −24% comes from unrealistically clean signals**, whose cycles repeat across the
  minute (sines 2–8× smaller, ramps and squares 6×). On noisy types the gain is modest:

  | type | per block → chunk (bits/sample) |
  |---|---|
  | random walk | 5.0 → 4.7 (−6%) |
  | noisy sine | 5.8 → 5.4 (−7%) |
  | gauss-spikes | 4.5 → 4.3 (−5%) |
  | sensor | 2.9 → 2.5 (−15%) |
  | impulses | 4.51 → 4.50 (0%) |

**delta0123-zstd per minute** (`tslab/entropy/delta_zstd.py`, `bench/delta_zstd.py`): the whole
per-block front end and header packing are compiled, and there's one compression call per unit.
On 120 one-minute signals (7,200 blocks); sizes are whole units, each block's min and max
included:

| codec | bits/sample | median RMSE | encode µs/block | decode µs/block | 200 signals × 1 day, core-s |
|---|---|---|---|---|---|
| delta0123-zstd-8-L1 (zstd level 1) | 2.59 | 0.113% | 5.2 | 2.0 | 91 |
| delta0123-zstd-8 (level 3) | 2.49 | 0.113% | 5.6 | 2.0 | 97 |
| delta0123-zstd-10-L1 | 3.46 | 0.028% | 5.4 | 2.1 | 94 |
| delta0123-zstd-10 | 3.37 | 0.028% | 5.6 | 2.0 | 97 |
| delta0123-deflate-8-L3 / L4 / L6 | 2.71 / 2.69 / 2.53 | 0.113% | 14 / 15 / 31 | ~2.3 | 250–537 |
| quant8-delta1-zstd (no order pick) | 2.44 | 0.113% | 2.1 | 1.5 | 36 |

**Encode time breakdown for delta0123-zstd-8, zstd-1** (per-block (original evaluation), 7.1
µs/block total; no longer measured):

| stage | µs/block |
|---|---|
| quantize | 0.7 |
| choose order | 2.7 |
| residual + zigzag | 0.6 |
| byte-plane write | 0.8 |
| headers + copy | 0.3 |
| zstd | 2.0–2.5 |

The index statistics (min / max / mean with numpy) take another 0.4 µs/block for every codec.

- **The targets are met easily.** 1:100 real time needs about 0.11 cores for 200 signals; 1:1000
  needs about 1.1.
- **Plain quant8 deltas + zstd is competitive at 8 bits:** about 2.7× faster and 2% smaller on
  this data. zstd across 60 blocks recovers much of what choosing the order buys, and order-1
  deltas of clean sines repeat almost exactly. delta0123-zstd still wins on the chirp and impulses,
  by 12% and 4%, and it's the only one of the two that supports B > 8.

---

## 10. Constant-width ~8-bit formats

**Question.** Can we say "64 → 8 bits per sample with no useful loss", in a format where every
sample costs the same, and ideally do better than plain 8-bit quantization? Code:
`tslab/constwidth/` (`delta_linear.py`, `delta_sqrt.py`, `delta_bfp.py`, `delta_tree.py`); head-to-head:
`bench/constwidth_8bit.py`.

All of these quantize to a 32-bit integer grid over the block's [min, max], and all are **closed
loop**: each residual is taken against the decoder's own running reconstruction, in integer
arithmetic, so error never accumulates.

### The formats

| format | bits/sample | how a sample is coded |
|---|---|---|
| **quant8** | 8.13 | 8-bit quantization of the value; baseline |
| **cw8-delta1-linear** | 8.18 | int8 code k, delta = round(D·k/127), with D = the block's largest step |
| **cw8-delta1-sqrt** ("semi-quadratic") | 8.15 | 1 bit odd_shift + 7-bit signed value; delta decoded with [CLZ](https://en.wikipedia.org/wiki/Find_first_set), so resolution scales like √\|delta\| (7 bits of range for the largest jumps, ~13 for small ones); wraps mod 2³² |
| **cw8-delta1-bfp-e2** | 8.28 | int8 code k (all 256 values), delta = k ≪ (24 − e); a 2-bit exponent e per 16-sample frame |
| **cw8-delta1-bfp-e4** | 8.41 | as cw8-delta1-bfp-e2 with a 4-bit exponent (e = 0..15) |
| cw6-delta1-bfp-e4 / cw4-delta1-bfp-e4 | 6.42 / 4.42 | as cw8-delta1-bfp-e4 with 6-bit / 4-bit codes (for comparison below 8 bits) |
| **cw-delta1-tree-7** | 8.23 | binary tree over the deltas, 1 bit per node (halve the subtree's scale), 7-bit sign-relative leaves |

(Every block's size is fixed; the bits include the block's float64 min and max, 0.128 bits/sample.)

### Findings per format

**cw8-delta1-sqrt.**
- **The encoder must be closed loop.** Encoding deltas of the original signal (open loop) gave a
  median error of 45% of range: rounding errors accumulate, and wraparound turns them into
  full-range jumps.
- **The shift must round down.** Shift left by ⌊lz/2⌋ when encoding, and right by CLZ − odd_shift
  when decoding. Rounding the half up pushes the largest deltas into the sign bit.
- **The decoder is equivalent to a fixed 256-entry table**, but the arithmetic form is better for
  SIMD: NEON has a vector CLZ (`clz.4s`) and no 32-bit gather. The compiled decoder does vectorize.
- **Encoder:** try the literal encoding of the residual plus the residual raised by ¼, ½ and 1 step
  at that magnitude. That matched a brute-force search over all 256 codes on every test block, at
  about 1/10 the cost. Fewer candidates missed codes near step-size boundaries: the literal
  encoding truncates away from zero for negatives, and the nearest level can lie in a finer class.
- **Correction:** its worst max error is 0.775% of range, not the 0.39% stated during the
  investigation. Near the ends of the range, wraparound makes the nearest code unusable, and the
  best one left can be a full step away.

**cw8-delta1-linear** (now in integer arithmetic).
- A rounding bias on negative residuals broke its error bound at first; after the fix it holds
  max error ≤ D/254 ≤ 0.394% of range.
- It's excellent on smooth data, but one large jump coarsens the whole block (impulses: 0.22% RMSE).

**cw8-delta1-bfp** is [block floating point](https://en.wikipedia.org/wiki/Block_floating_point): each 16-sample frame shares one step size.
- **The grid** maps [min, max] to [2²³, 2²³ + 255·2²⁴] on a uint32. At e = 0 the step is 2²⁴ and
  wraparound reaches any value, so it's exactly the quant8 grid: max error ≤ range/510 = 0.196%,
  guaranteed. The half-step offset keeps reconstructions within half a step of min/max from
  wrapping.
- **Rollover happens only at e = 0,** across the block's full quantization range; codes at e > 0
  are clamped, and the encoder asserts they never wrap.
- **Choosing e:** the finest step whose ±127.5 steps cover the frame's largest delta plus the error
  carried into the frame. Decode is shifts plus one running add.
- **Exponent width and frame size** (per-block (original evaluation), 12 blocks, min/max
  excluded; the sweep is no longer run):

  | exponent bits | frame | bits/sample | median RMSE | worst RMSE |
  |---|---|---|---|---|
  | 2 | 16 | 8.28 | 0.0142% | 0.066% |
  | 2 | 64–128 | 8.17–8.18 | 0.0142% | 0.066–0.072% |
  | 4 | 8 | 8.66 | 0.0081% | 0.062% |
  | **4** | **16** | **8.41** | **0.0090%** | 0.064% |
  | 4 | 32 | 8.28 | 0.0103% | 0.068% |

  - **With 2 bits, 76% of frames sat at the maximum e = 3.** The range was saturating, so frame
    size didn't matter.
  - **With 4 bits, only 4–8% reach e = 15.** Smooth signals improve up to 30×; busy ones (the
    ~50 Hz sine, chirp, noisy sine) don't change, since their limit is the 8-bit delta itself.
    Mean exponents of 1–9 suggest a 3-bit exponent would capture most of the gain.
  - **Every setting kept max error ≤ quant8's bound** on 3,000 randomized blocks (original
    evaluation; the report's bound check now asserts it on every block).

**Tree delta** is hierarchical block floating point: the original tree codec, applied to deltas.
- **The lopsided sign-relative alphabet** (2-bit: {−1, 0, 1, 2}; 4-bit: −8..7) means a leaf's
  range depends on whether it continues or reverses direction. A halving test that assumes one
  direction clips the other: reversal sizing ruins the 4-bit ramp (7.2%), continuation sizing
  ruins the 2-bit square wave (3.2%) (original evaluation, 12 blocks). Sizing each leaf for its
  actual direction ("aware", the `cw-delta1-tree-N` codecs) is best at every width.
- **Adding a half-step margin** cut clipping (13% → 6% of leaves, original evaluation) but made
  scales coarser, and was worse overall, so it was dropped.
- **Results** (the report's matrix, 3,600 blocks; median over the kinds, worst over every block):

  | leaf bits | bits/sample | median RMSE | worst RMSE | worst max err |
  |---|---|---|---|---|
  | 2 | 3.23 | 0.44% | 22% | 75% |
  | 4 | 5.23 | 0.105% | 2.9% | 12.1% |
  | 7 | 8.23 | 0.0119% | 0.35% | 1.55% |

  - **2-bit leaves beat the original tree codec** (bintree-center: 1.08% at 3.14 bits/sample), but
    delta0123-zstd-8 has 3.9× lower error (0.113%) at 1.86 bits/sample.
  - **The shared scale is the limit:** one scale per subtree must suit both its largest and
    smallest deltas, and 13–44% of leaves ended up at a code limit (original evaluation). The
    worst 2-bit blocks are unusable (75% of range).

### Head to head at ~8 bits/sample

7,200 blocks of continuous one-minute signals (`bench/constwidth_8bit.py`); sizes include each
block's min and max; single thread, µs per block:

| codec | bits/sample | median RMSE | 99th pct RMSE | worst RMSE | worst max err | encode | decode |
|---|---|---|---|---|---|---|---|
| quant8 | 8.13 | 0.113% | 0.117% | 0.144% | **0.196%** | 0.5 | 0.8 |
| cw8-delta1-linear | 8.18 | 0.0073% | 0.229% | 0.239% | 0.394% | 13.7 | 2.8 |
| cw8-delta1-sqrt | 8.15 | 0.037% | 0.108% | 0.184% | 0.775% | 16.1 | 2.7 |
| cw8-delta1-bfp-e2 | 8.28 | 0.0144% | 0.067% | 0.116% | **0.196%** | 11.0 | 2.7 |
| **cw8-delta1-bfp-e4** | 8.41 | **0.0071%** | **0.067%** | **0.116%** | **0.196%** | 10.1 | 2.7 |
| cw-delta1-tree-7 | 8.23 | 0.0121% | 0.095% | 0.351% | 1.554% | 77* | 18* |
| *compressed, for comparison:* | | | | | | | |
| cw8-delta1-bfp-e4 + zstd per unit | 4.36 | 0.0071% | 0.067% | 0.116% | 0.196% | 12.1 | 3.5 |
| delta0123-zstd-16 | 4.21 | 0.0037% | 0.057% | 0.234% | 0.394% | 7.4 | 2.4 |

\*The tree delta code isn't optimized: numpy per tree level, plus a numba leaf loop.

**Takeaways:**
- **cw8-delta1-bfp-e4 is the best constant-width format.** Its typical error is about 16× lower than quant8's,
  and it has the best worst-case RMSE of the group. Its max error is guaranteed never worse than
  quant8's, at 0.28 more bits/sample.
- **cw8-delta1-linear matches it typically**, but a single large jump coarsens the whole block (impulses).
- **cw-delta1-tree-7 and cw8-delta1-sqrt lose to cw8-delta1-bfp-e4 on every measure.** One scale per subtree, or a
  fixed √ law, can't adapt as locally as a scale per 16-sample frame.
- **Versus delta0123-zstd-16:**
  - **delta0123-zstd-16 stays near 8 bits/sample or less only because of its cap.** The cap
    estimates each block's size from its residual spread and lowers B where it would exceed 8.
    Here B was 16 for 33% of blocks, 9–13 for 66%, and 7 for 1%. Being an estimate, one block
    still reached 9.64 bits/sample when coded on its own.
  - **With entropy coding, delta0123-zstd-16 is half the size and 2× more precise typically.**
    It's much better on busy signals (the ~50 Hz sine: 0.007% vs 0.057%).
  - **Its capped B = 7 blocks double the worst-case max error** (0.394%).
- **cw8-delta1-bfp-e4's bytes also compress well:** zstd per unit takes it to 4.36 bits/sample.
  Once compressed, the two designs are about the same size.
- **Encode can't vectorize within a block** for any of the delta codecs, because the closed loop
  makes each sample depend on the previous reconstruction. Decode is shifts or CLZ plus one running
  sum.

### Narrower codes: cw6-delta1-bfp-e4 and cw4-delta1-bfp-e4

The same design with 6-bit or 4-bit codes and a 4-bit exponent per 16-sample frame. The base
(e = 0) grid has 64 or 16 levels, so max error is bounded by 6-bit or 4-bit quantization's (range/126 = 0.794%,
range/30 = 3.33%), and quiet frames get up to 15 extra bits. Codes are bit-packed; unpacking adds
about 1 µs to decode. Same 7,200 blocks:

| codec | bits/sample | median RMSE | 99th pct RMSE | worst RMSE | worst max err | encode | decode |
|---|---|---|---|---|---|---|---|
| cw8-delta1-bfp-e4 | 8.41 | 0.0071% | 0.067% | 0.116% | 0.196% | 10.1 | 2.7 |
| cw6-delta1-bfp-e4 | 6.42 | 0.043% | 0.277% | 0.477% | 0.794% | 11.4 | 4.0 |
| cw4-delta1-bfp-e4 | 4.42 | 0.198% | 1.27% | 1.99% | 3.33% | 11.0 | 3.8 |
| cw-delta1-tree-4 | 5.23 | 0.108% | 0.853% | 2.86% | 12.1% | 77* | 17* |
| *compressed, for comparison:* | | | | | | | |
| cw8-delta1-bfp-e4 + zstd per unit | 4.36 | 0.0071% | 0.067% | 0.116% | 0.196% | 12.1 | 3.5 |
| delta0123-zstd-8 | 2.49 | 0.113% | 0.123% | 0.234% | 0.394% | 5.6 | 1.9 |

- **Each 2 bits removed costs about 5–6× in typical error** (4.6× and 6×), a little worse than the 4× that
  6 dB/bit predicts: the busy signals that already saturate the exponent lose the full 2 bits,
  and fewer codes leave less room for the carried-in error.
- **cw6-delta1-bfp-e4 still beats quant8 typically** (0.043% vs 0.113% median RMSE at 6.42 vs 8.13 bits/sample),
  but its guarantee is 6-bit quantization's: worst max error 0.794%, 4× quant8's.
- **cw4-delta1-bfp-e4 vs cw-delta1-tree-4:** 0.8 bits/sample smaller, about 2× worse median RMSE (0.198% vs
  0.108%), but a better worst case (RMSE 1.99% vs 2.86%, max error 3.33% vs 12.1%) and 7× faster
  to encode.
- **Narrow constant-width codes lose badly to entropy coding at the same size.** cw8-delta1-bfp-e4 +
  zstd is 4.36 bits/sample, about the same as cw4-delta1-bfp-e4 (4.42), with 28× lower median error and
  a 17× lower worst max error. delta0123-zstd-8 is 56% of cw4-delta1-bfp-e4's size with lower error
  on every measure. So
  narrowing the code is the wrong way to save space: keep 8-bit codes and compress them.

**Smaller frames for cw6-delta1-bfp-e4** (original evaluation, same blocks, min/max excluded; the
sweep is no longer run; a smaller frame costs more exponent bits per sample):

| codec | frame | bits/sample | median RMSE | 99th pct RMSE | worst RMSE | worst max err |
|---|---|---|---|---|---|---|
| cw6-delta1-bfp-e4 | 16 | 6.29 | 0.043% | 0.277% | 0.477% | 0.794% |
| cw6-delta1-bfp-e4 | 8 | 6.54 | 0.037% | 0.275% | 0.477% | 0.794% |
| cw6-delta1-bfp-e4 | 4 | 7.03 | 0.033% | 0.258% | 0.469% | 0.794% |
| cw6-delta1-bfp-e2 | 4 | 6.54 | 0.059% | 0.258% | 0.469% | 0.794% |
| cw7-delta1-bfp-e2 | 16 | 7.16 | 0.029% | 0.135% | 0.234% | 0.394% |
| cw7-delta1-bfp-e4 | 16 | 7.29 | 0.020% | 0.135% | 0.234% | 0.394% |
| cw8-delta1-bfp-e4 | 4 | 9.02 | 0.0063% | 0.062% | 0.114% | 0.196% |

- **Frames of 4 samples help little:** 0.74 more bits/sample buys a 1.3× lower median RMSE.
  The worst case doesn't move, because it's set by frames that sit at e = 0 at any frame size.
- **The same bits are better spent on the code:** cw7-delta1-bfp-e4 (7.29 bits/sample) is 1.6×
  better typically than cw6-delta1-bfp-e4-f4 (7.03), and it halves the worst case. Frame size 16 stays the choice.

**Choose cw8-delta1-bfp-e4** when constant width matters (fixed offsets, size known in advance, a hard
"never worse than 8-bit" guarantee). **Choose entropy coding** (delta0123-zstd, or
fluxcode, §11) for the best storage and typical precision.

---

## 11. Power-of-two quantization (the fluxcode design)

**Proposal.** delta0123-zstd's pipeline, tightened for deployment. This section measures the
prototype, fluxproto (`tslab/flux/proto.py`; benchmark `bench/fluxproto_sweep.py`); the shipped
`fluxcode` package at the repository root implements the resulting design ([docs/SPEC.md](../docs/SPEC.md)).
- **Quantize on a power-of-two step:** e = k − B, where max − min ∈ [2ᵏ⁻¹, 2ᵏ), and
  q = round((x − min)·2⁻ᵉ). min is exact; max lands within half a step. B = 9..16 is fixed per
  run (default 16), with no cap.
- **Predict:** pick the order from 0–3 on the integers, as delta0123-zstd does.
- **Layout:** per block, the order (and mode), the exponent, then the low and high bytes of the
  residuals. A minute's blocks are interleaved by field (order/mode × N, exponent bytes × N,
  low × N, high × N), then zstd-3. The exponent field is now 8 bytes, one plane each, so it can hold
  the decimal parameter below; the unused planes compress to almost nothing.

**Implementation details** (my choices where the proposal left room):
- **Residuals are taken mod 2¹⁶ and read as int16,** then zigzagged. Order-k differences of 16-bit
  values need up to 16 + k + 1 bits, but the decoder integrates mod 2¹⁶, which is exact because
  every q is in [0, 2¹⁶). So two bytes per sample always suffice.
- **Zeros are prepended before differencing,** so the first residuals are the start values and
  there's no separate header.
- **If the max would round to 2ᴮ,** e goes up by 1.
- **B < 16 rounds directly to the coarser grid.** It's the same grid as a shift-right from 16
  bits, but without the half-step bias of truncation or double rounding.

### Continuous signals (7,200 blocks)

Byte planes, no decimal detection (the first version). Sizes are whole units: the prototype stores
each block's min in the unit, about 0.05 bits/sample.

| codec | bits/sample | median RMSE | worst RMSE | worst max err | encode µs | decode µs |
|---|---|---|---|---|---|---|
| fluxproto-16 | 6.93 | 0.00056% | 0.0009% | 0.0015% | 4.4 | 2.8 |
| fluxproto-14 | 5.75 | 0.0022% | 0.0037% | 0.0061% | 4.7 | 2.6 |
| fluxproto-12 | 4.29 | 0.0089% | 0.0147% | 0.0244% | 4.0 | 2.3 |
| fluxproto-10 | 3.16 | 0.036% | 0.058% | 0.098% | 3.6 | 2.1 |
| fluxproto-9 | 2.67 | 0.071% | 0.118% | 0.195% | 3.8 | 2.0 |
| fluxproto-16, per-block layout | 7.00 | same as fluxproto-16 | | | 5.1 | 2.6 |
| fluxproto-16, order fixed at 1 | 7.62 | same as fluxproto-16 | | | 3.0 | 2.3 |
| fluxproto-16, zstd-1 | 7.02 | same as fluxproto-16 | | | 4.0 | 2.9 |
| delta0123-zstd-16, no cap | 7.28 | 0.00044% | 0.0006% | 0.0008% | 7.6 | 3.0 |
| delta0123-zstd-12, no cap | 4.70 | 0.0070% | 0.0087% | 0.0122% | 6.7 | 2.5 |

(All B from 9 to 15 are in `bench/fluxproto_sweep.py`'s output; each bit removed saves about 0.5–0.7
bits/sample and doubles the error.)

- **A power-of-two step costs nothing in rate-distortion terms.** The range uses 2ᴮ⁻¹ to 2ᴮ levels
  instead of 2ᴮ − 1, so precision drops by ~0.35 bits on average. Size drops by about the same
  amount: fluxproto-12 is 0.41 bits/sample smaller than delta0123-zstd-12 without its cap, with
  1.27× the RMSE.
- **Max error is guaranteed ≤ 2⁻ᴮ of range** (half a step, with at least 2ᴮ⁻¹ steps in the range).
- **Interleaving by field helps, slightly:** 1.0% smaller than one block at a time.
- **Re-encoding decoded data gives identical bytes** at every B, on every minute.
- **Edits are where the design pays off.** I decoded each minute, raised one sample per block by
  1% of the range (a new max), re-encoded, and counted how many other samples changed (original
  evaluation; the edit experiment is no longer in the bench, and fluxcode's tests check the
  property):

  | codec | blocks where other samples changed | untouched samples that changed |
  |---|---|---|
  | fluxproto-16 | 6.2% | 3.1% |
  | delta0123-zstd-16 | 100% | 95.1% |
  | fluxproto-12 | 6.4% | 3.2% |
  | delta0123-zstd-12 | 100% | 94.9% |

  With a range-scaled step, any change to min or max moves the whole grid and re-rounds every
  sample. With a power-of-two step, the grid moves only when the range crosses a power of two
  (6% of blocks here).

**Profile of fluxproto-16's encode** (µs per block):

| total | quantize + order pick + residual + planes | with the order fixed at 1 | order pick alone | zstd-3 |
|---|---|---|---|---|
| 4.4 | 1.9 | 0.6 | 1.7 | 2.5 |

The order pick is about 40% of encode, not dominant. Fixing the order at 1 costs nothing on noise-like
signals and sensors: random walk, noisy sine, sensor-0.1 and all the discretized noisy signals are
unchanged. It costs +0.6 to +2.8 bits/sample on smooth signals (quadratic, sines, chirp; original
evaluation), so over this mix it's 10% bigger (7.62 vs 6.93). Keep the pick unless real data turns out to be
noise-dominated. If speed matters, picking on every 4th sample is a cheap middle ground (not tried).

### Discretized data

Real data is often already on a quantum (0.1 units, an ADC step), usually not a power of two.
Nine discretized signals, 600 blocks each (still byte planes, no decimal detection). Each cell is
bits/sample, with median RMSE in parentheses. "Ideal" is the same pipeline with step = the
quantum, i.e. lossless:

| signal | quantum | ideal (lossless) | fluxproto-16 | fluxproto-14 | fluxproto-12 | fluxproto-10 |
|---|---|---|---|---|---|---|
| random-walk | 0.001 | 12.64 | 13.77 (0.0006%) | 11.35 (0.0025%) | 8.67 (0.010%) | 6.33 (0.041%) |
| random-walk | 0.01 | 9.33 | 12.38 (0.0006%) | 10.97 (0.0025%) | 8.61 (0.010%) | 6.27 (0.040%) |
| random-walk | 0.1 | 5.82 | 9.97 (0.0006%) | 7.74 (0.0025%) | 7.06 (0.0099%) | 6.23 (0.039%) |
| random-walk | 2⁻⁴ | 6.48 | 7.71 (0) | 7.72 (0) | 7.08 (0) | 6.24 (0) |
| noisy-sine | 0.1 | 8.91 | 11.00 (0.0005%) | 11.14 (0.0020%) | 9.94 (0.0079%) | 7.25 (0.032%) |
| noisy-sine, 12-bit ADC over ±200 | 0.0977 | 8.93 | 14.52 (0) | 12.31 (0.0021%) | 9.99 (0.0081%) | 7.25 (0.032%) |
| sin-4.12hz | 0.01 | 2.63 | 3.64 (0.0006%) | 2.43 (0.0022%) | 1.70 (0.0090%) | 0.98 (0.036%) |
| sensor-0.1 | 0.1 | 1.79 | 4.18 (0.0006%) | 2.93 (0.0025%) | 2.43 (0.010%) | 2.43 (0.040%) |
| noisy-sine, 0.01 via float32 | 0.01 | 12.32 | 14.07 (0.0005%) | 12.86 (0.0020%) | 10.02 (0.0081%) | 7.26 (0.032%) |

- **fluxproto-16's error is far below every quantum** (median 0.0006% of the range), but it's
  bit-exact only where the quantum sits on its grid (the 2⁻⁴ walk and the ADC).
- **And it pays 1.0–5.6 bits/sample more than the lossless encoding at the quantum.** Where the
  grid is finer than the quantum, the quantized deltas are multiples of a non-power-of-two number
  (0.1 units = 102.4 steps, the ADC step = 25 × 2⁻⁸). Their low bits look random to zstd, which
  sees bytes, not multiples of 25.
- **Even a power-of-two quantum costs 1.2 bits** (random-walk 2⁻⁴): the q values have trailing
  zero bits that the byte planes carry anyway.
- **Coarser B isn't the fix.** fluxproto-12 and below cross the ideal size only by becoming lossy.

### Decimal step detection

**Design.** (Related: [ALP](https://doi.org/10.1145/3626717), which also detects decimals in floats.)
Real values usually sit on a decimal grid (0.1 units, 0.001, whole numbers).
So the detector only asks one question: is every value on a multiple of 10^p?
- **Tolerance:** tol = a quarter of the power-of-two step at the chosen B. A value within tol of
  the grid counts as on it, so detection never has a larger max error than the power-of-two grid
  at the same B.
- **Which p:** the coarsest p whose step 10^p is still coarser than the power-of-two step, such
  that every x·10^−p is within tol·10^−p of an integer. A failing p usually exits on its first
  sample.
- **Reconstruction:** x' = K / 10^−p with integer K. Float division is correctly rounded, so
  this is the same double as parsing the decimal text: bit-exact.
- **float32 sources (prototype only):** if every input is exactly a float32 (values that were
  stored as float32 upstream), a flag made the decoder round its output back to float32, so those
  also came back bit-exact. fluxcode dropped the flag: decimals stored as float32 are on the
  decimal grid to within tolerance, so they now decode as the decimal itself, and casting the
  output back to float32 recovers the input. The flag cost a head bit and a decoder pass for
  nothing a caller couldn't do ([docs/SPEC.md §6](../docs/SPEC.md#6-guarantees)). The float32 rows
  below are the prototype's.
- **Other steps** (ADC counts, 0.5, 2⁻⁴) fall back to the power-of-two grid.
- **Cost:** 0.5 µs per block on continuous data, 0.9 µs on discretized data (original run).

**A general step detector was tried first and dropped.** It took an approximate float GCD of the
deltas, but float noise compounds through [Euclid's](https://en.wikipedia.org/wiki/Euclidean_algorithm) quotients. Tracking the uncertainty fixed one
case (the float32 decimals) and broke another (0.001 steps, detected in 133 of 600 blocks
instead of 512; original evaluation). Detecting only powers of ten is simpler and handles every decimal case exactly.

**Four versions, B = 9..16** (plots and full table in `report/index.html`; the numbers here are
`bench/fluxproto_sweep.py`'s). Discretized data is now 9 signals, 5,400 blocks, including a 0.001
step and decimals stored as float32. These rows use bit-shuffled planes, the prototype's final
layout (next subsection); the decision was first made on byte planes, with the same outcome.

| version (B = 16) | continuous bits/sample | discretized bits/sample | bit-exact samples | encode µs (cont / disc) | decode µs |
|---|---|---|---|---|---|
| order 1, power-of-two only | 7.12 | 10.49 | 29.4% | 2.8 / 3.1 | 2.0 |
| order 1, decimal detect | 6.93 | 8.40 | 98.4% | 2.9 / 3.2 | 2.0 |
| order 0–3, power-of-two only | 6.31 | 10.30 | 29.4% | 4.0 / 4.3 | 2.4 |
| **order 0–3, decimal detect** | **6.12** | **8.17** | **98.4%** | 4.2 / 4.4 | 2.3 |

Discretized signals at B = 16, bits/sample:

| signal | lossless at the true quantum | order 1, power of two | order 1, decimal | order 0–3, power of two | order 0–3, decimal |
|---|---|---|---|---|---|
| random-walk, 0.001 | 12.64 | 12.63 | 12.51 | 12.63 | 12.51 |
| random-walk, 0.01 | 9.33 | 12.60 | 9.33 | 12.60 | 9.33 |
| random-walk, 0.1 | 5.82 | 11.53 | 5.82 | 11.53 | 5.82 |
| random-walk, 2⁻⁴ | 6.48 | 6.50 | 6.50 | 6.50 | 6.50 |
| noisy-sine, 0.1 | 8.91 | 13.75 | 8.91 | 13.75 | 8.91 |
| noisy-sine, 12-bit ADC | 8.93 | 13.68 | 13.68 | 13.68 | 13.68 |
| sin-4.12hz, 0.01 | 2.63 | 5.91 | 4.71 | 4.22 | 2.63 |
| sensor-0.1 | 1.79 | 4.01 | 1.79 | 4.01 | 1.79 |
| noisy-sine, 0.01 via float32 | 12.32 | 13.75 | 12.32 | 13.75 | 12.32 |

- **Decimal detection reaches the lossless-at-the-quantum size on every decimal signal,** and
  decodes it bit-exact. It saves up to 5.7 bits/sample (random-walk at 0.1).
- **It costs nothing elsewhere.** On continuous data it fires only on signals that really are
  decimal (the integer ramp, the ±100 square wave, sensor-0.1), which makes it 0.19 bits/sample
  smaller. Error bounds are unchanged at every B.
- **0.001 steps are only partly detected,** and correctly so: in 88 of 600 blocks (original
  evaluation) the range spans more than 2¹⁶ steps of 0.001, so they can't be exact in 16 bits and
  use the power-of-two grid.
- **As B drops,** more discretized blocks are too wide for their decimal step and fall back to
  power of two (bit-exact samples: 98% at B = 16, 56% at 13, 22% at 9), and the four versions
  converge.
- **The order pick matters on smooth data** (sin-4.12hz at 0.01: 4.71 → 2.63 bits/sample) and not
  on noisy data, as before. It adds about 1.3 µs per block to encode.
- **Still not covered:** non-decimal quanta (ADC counts in engineering units: 13.68 vs 8.93
  bits/sample, and power-of-two steps). If real data has many of those, extend the candidate
  list, e.g. {1, 2, 2.5, 5}·10^p, or configure a quantum per channel.
- **In the fluxcode package** (`bench/fluxcode_sweep.py`) the same sweep gives 6.11 / 8.16
  bits/sample at B = 16 with 87.7% of discretized samples bit-exact: without the float32 flag, the
  float32 signal decodes as its decimals rather than bit-exact float32 values.

### Byte planes vs bit-shuffle

`planes="bit"` in `tslab/flux/proto.py` replaces the low/high byte planes with 16 bit planes per block
([bit-shuffle](https://github.com/kiyo-masui/bitshuffle)):
an 8×8 bit transpose per 8 samples, compiled, and its own inverse. Orders 0–3, decimal
detection, B = 16, 12,600 blocks. Timings alternate between configurations, one thread
(fastest of 5 runs):

| layout | continuous bits/sample | discretized bits/sample | encode µs | decode µs |
|---|---|---|---|---|
| byte planes, zstd-3 | 6.76 | 8.70 | 4.71 | 2.69 |
| nibble planes, zstd-3 | 6.29 | 8.51 | 4.54 | 2.20 |
| **bit planes, zstd-3** | **6.12** | **8.17** | **4.31** | **2.29** |
| byte planes, zstd-1 | 6.84 | 8.74 | 4.34 | 2.79 |
| bit planes, zstd-1 | 6.16 | 8.18 | 4.18 | 2.32 |

- **Bit-shuffle is smaller and faster.** The shuffle costs 0.39 µs per block to encode and
  0.49 µs to decode. But zstd gets through the mostly-zero high bit planes faster than it gets
  through near-random bytes, which more than pays that back.
- **It isn't better for every input.** It's worse on smooth, periodic signals: slow sines +7% to
  +21%, the 0.1-step random walk +5%, the ramp and square wave +3% and +17% of almost nothing.
  zstd's repeated sequences in their residuals don't line up with 8-sample bit groups. The
  worst loss is 0.45 bits/sample, on signals that were cheap anyway. It's better on everything
  noisy or wide: −4% to −18%, and −70% on the quadratic, whose order-2 residuals are nearly
  constant.
- **zstd-1 vs zstd-3 barely matters with bit-shuffle:** 0.04 bits/sample for about 0.1 µs.
- **Across B (sweep in `report/index.html`),** bit-shuffle wins from B = 13 up (−0.64 bits/sample
  at B = 16 continuous). At B ≤ 12, byte planes are 0.06–0.25 bits/sample smaller on continuous
  data (and up to 0.2 on discretized data at B ≤ 11), since the residuals fit in about one byte and
  the byte split already separates random from mostly-zero bits.
- **Nibble planes** (four 4-bit planes, two samples per byte) land between the two on continuous
  data at every B from 13 up. At B = 16 they're 6.29 / 8.51 bits/sample (continuous /
  discretized), against bit-shuffle's 6.12 / 8.17 and byte planes' 6.76 / 8.70, with the fastest
  decode. At B ≤ 12 they're the worst of the three on discretized data.
  - **Per signal, nibbles win only on slow sines** (sin-4.12hz at 0.01: 2.00 vs 2.63
    bit-shuffle). Their 2-sample groups break fewer of zstd's repeats than 8-sample bit groups do.
  - **They lose badly on the 2⁻⁴ walk** (8.79 vs 6.50 bit-shuffle) and the 0.1 walk (6.21 vs 5.82).
  - **Not adopted:** they never beat bit-shuffle overall.
- **Plane order barely matters** (original evaluation). Storing the bit planes high bits first changes zstd-3's size by
  −0.8% on continuous data and +0.8% on discretized data. Most noisy signals are ~1% smaller;
  the 2⁻⁴ walk is 15% larger. Byte planes change by under 0.5%. Low bits first (as specified)
  is kept.
- **Near-random planes stored raw, outside zstd** (`bench/fluxproto_raw_planes.py`). A plane with
  |p − 0.5| < t is appended uncompressed, and the others (plus a 1-bit-per-plane mask) go through
  zstd-3. This bench and the next two leave out the stored block minima, so their baseline is
  6.08 / 8.14 bits/sample rather than 6.12 / 8.17.
  - **Noisy signals get 1–2% smaller** (random walk 12.60 → 12.28 bits/sample at t = 0.05, per
    block and plane). zstd spends slightly more than one bit per bit on random planes, and they
    dilute its model of the other planes.
  - **Structured signals get much bigger:** their near-random planes still repeat.
    - The 2⁻⁴ walk goes from 6.49 to 12.21. Its planes duplicate the sign plane.
    - The ~50 Hz sine goes from 6.64 to 8.37, the chirp from 7.80 to 8.57, and the quadratic
      from 0.21 to 1.33.
  - **Net loss:** 6.08 → 6.36 bits/sample continuous, 8.14 → 8.61 discretized.
  - **zstd time falls only ~10%** (1.14 → 1.01 µs/block at t = 0.05). zstd already handles
    incompressible data quickly.
  - **One mask per minute instead of per block** does no better: 6.14–6.24 / 8.04–8.66.
  - **Keeping a plane in zstd when it duplicates plane 0** fixed the 2⁻⁴ case (discretized
    8.08, −0.7%) but not the sines or the chirp, and the smaller of the two layouts per minute
    reached 5.99 / 7.98 (−1.5% / −2.0%) at twice the zstd cost (both original evaluation).
    **Not adopted.**

### Per-plane probability models instead of zstd

**Question.** Instead of zstd on the bit planes, store each plane's density of ones and
[arithmetic-code](https://en.wikipedia.org/wiki/Arithmetic_coding) the plane with it. On noisy data, do per-plane statistics beat zstd's local
modelling? Measured as ideal arithmetic-coding cost, i.e. entropy under each model, including
the stored parameters and block headers (`bench/fluxproto_plane_models.py`; no coder built). B = 16,
orders 0–3, decimal detection:

| model | continuous (all) | discretized (all) | noisy signals |
|---|---|---|---|
| bit-shuffle + zstd-3 (real) | **6.08** | **8.14** | baseline |
| static density per block and plane (8-bit p) | 6.98 | 8.55 | −2% to −5% |
| static density per minute and plane (12-bit p) | 7.30 | 8.45 | **−3% to −6%** |
| adaptive, [LZMA](https://en.wikipedia.org/wiki/Lempel%E2%80%93Ziv%E2%80%93Markov_chain_algorithm)-style (shift 4 / 5), across the minute | 6.95 / 6.91 | 8.74 / 8.64 | −1% to −3% |
| static per block and plane, 2 contexts (higher bit already set?) | 6.92 | 8.54 | −2% to −5% |

- **On noisy data, yes.** Per-plane statistics beat zstd by 0.3–0.5 bits/sample (random walks,
  noisy sines, spikes, the ADC and float32 cases). Global beats local: one density per plane for
  the whole minute is best, and adaptive (localized) models are worst of the three.
- **Overall, no: they lose badly on structured signals,** where zstd finds repeats.
  - sin-50.3hz: 11.0 vs 6.64 bits/sample; chirp: 12.2 vs 7.80.
  - The 2⁻⁴ walk: 12.2 vs 6.49. Its q values have trailing zero bits, but zigzag turns those
    into copies of the sign bit, so those planes look random to a density model while zstd sees
    repeated planes.
- **Speed is the other obstacle.** 13–15 planes per block aren't constant; about 9 of those sit
  near p = 0.5 and could be stored raw for free. That leaves ~5,000 binary decisions per block:
  roughly 10–25 µs with a binary arithmetic or [rANS](https://en.wikipedia.org/wiki/Asymmetric_numeral_systems) coder, against ~3.5 µs for zstd.
- **If pursued:** code only the skewed planes, with one static density per minute (rANS, raw
  planes copied), and fall back to zstd per minute when it's smaller. The gain is 3–6% on noisy
  data.

### Noise floor (`noise_f`)

Below the noise, the codec was spending bits on random numbers: the near-random bit planes.
The noise floor coarsens the step to at most f·σ on blocks that look like white measurement
noise (`bench/fluxproto_noise_floor.py`; details in
[docs/SPEC.md §3.1a](../docs/SPEC.md#31a-noise-floor-if-noise_floor_sigma--f-is-set-and-rng--0)). It's encoder-only, and the step stays a
power of two.

**The detector.**
- σ is a robust estimate from second differences: a clipped [mean absolute deviation](https://en.wikipedia.org/wiki/Average_absolute_deviation).
- The gate is the lag-1 [autocorrelation](https://en.wikipedia.org/wiki/Autocorrelation) of those differences: white noise gives −2/3, a random
  walk −1/2, and smooth or fast deterministic signals positive values. The threshold is −0.6.
- **Gated** (original evaluation): noisy-sine 100% (σ estimated 5.0, true 5.0), impulses 100% and gauss-spikes 90%
  (σ 6.0 against a true 5.8, their uniform base noise), sensor-0.1 20% (its 0.1 rounding noise;
  the decimal grid stays coarser, so it's still lossless).
- **Never gated:** random walks, sines, chirp, linear, square wave. The quadratic's gate fires
  on float rounding noise (σ ≈ 3.5e−11), far below the B-bit step, so it has no effect.

**Results** (B = 16, orders 0–3, decimal detection, bit-shuffle, block minima stored; errors in
units of the true noise σ):

| signal | f | bits/sample | RMS error vs input | RMS error vs clean signal |
|---|---|---|---|---|
| noisy-sine | off | 13.74 | 0.000 | 0.997 |
| noisy-sine | 0.25 | 5.48 | 0.058 | 0.999 |
| noisy-sine | 0.5 | 4.44 | 0.115 | 1.004 |
| noisy-sine | 1 | 3.42 | 0.231 | 1.025 |
| noisy-sine | 2 | 2.45 | 0.461 | 1.099 |
| impulses | off / 1 | 11.71 / 3.09 | 0.001 / 0.200 | 1.000 / 1.038 |
| gauss-spikes | off / 1 | 12.21 / 4.88 | 0.001 / 0.200 | 1.001 / 1.035 |

- **At f = 1, noisy signals shrink 3–4× and the added error is 0.2σ.** Against the clean signal
  the error grows only from 1.00σ to 1.03σ, since the noise was already there. The discretized
  noisy sines (0.1, ADC, float32) behave the same: 3.40–3.41 bits/sample at f = 1.
- **Clean signals are unchanged** at every f. Over all 12,600 blocks: 7.00 → 4.57 bits/sample
  (f = 1), 4.30 (f = 2). A fixed B = 10 reached a similar total (4.3, original evaluation) but
  coarsens every clean signal too: the random walk goes from 0.0006% to 0.04% RMSE. The noise
  floor spends its error only where noise already exceeds it.
- **It isn't denoising:** the output keeps the noise (error vs clean ≥ 1σ). It stops storing the
  noise precisely. Removing noise would need a filter ([Savitzky–Golay](https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter), wavelet shrinkage), which
  moves the output away from the input by about σ.
- **Max error as % of range is misleading on noise-only blocks:** 10% for spike signals at
  f = 1, from spike-free blocks whose whole range is the noise. It's 0.35σ.
- **Cost:** +0.8 µs/block encode (5.05 vs 4.26), with the estimator compiled with
  fastmath. Without fastmath it was +8 µs (original run).

**Truncation on the fine grid, and per-sample spike preservation** (`bench/fluxproto_truncation.py`).
Question: can the noise floor be applied per sample, zeroing low bits on the block's fine grid,
so that spikes keep full precision? And how much of the saving does zstd give back?

| signal, f = 1 | A: coarser block step | B: fine grid, low bits zeroed | B with sign-magnitude | C: per-sample, spikes ±2 kept | spike max error, A → C |
|---|---|---|---|---|---|
| noisy-sine | 3.37 | 3.36 | 3.63 | 3.38 | 0.40σ → 0.00σ |
| impulses | 3.04 | 3.08 | 3.13 | 3.69 | 0.35σ → 0.001σ |
| gauss-spikes | 4.82 | 4.83 | 4.99 | 4.83 | (smooth spikes aren't flagged) |

- **zstd gives back almost everything.** Zeroing low bits on the fine grid (B) is within 0–1.5% of
  a coarser step (A), at f = 0.5, 1 and 2. The decoder is unchanged.
- **Zigzag is fine.** Its sign-copied low planes compress well. Sign-magnitude, which keeps the
  zeroed bits zero, is 1–14% larger.
- **Keeping spikes exact is expensive:** impulses +0.6–0.7 bits/sample (+14–35%) to hold 3
  spikes per block (±2 samples) at full precision, about 200 bits per spike. Each exact sample
  carries ~13 random low bits into otherwise-zero planes, and with order-k prediction it spreads
  into its neighbours' residuals. Without it, a spike's error is at most half a step (0.35σ at
  f = 1), below the noise already on the spike.
- **Smooth spikes** (gauss-spikes, 10-sample width) aren't flagged by the second-difference test.
  They're treated as signal inside a noisy block and quantized at the block's step.

**Relative-sign zigzag** (`bench/fluxproto_relative_sign.py`): each residual is negated when the last nonzero
residual was negative, so the sign bit records "same/opposite sign as before". It's applied to
the codec's own residuals, and nothing else changes.
- **Overall it's a wash:** −0.1% continuous and −0.3% discretized without the noise floor; −0.4%
  / −0.4% with f = 1.
- **It helps slow sines,** whose order-3 residuals come in same-sign runs: −3.3% continuous,
  −8.5% at 0.01.
- **It hurts mostly-zero residuals:** sensor-0.1 +2%, the square wave +41% of almost nothing.
- **Noisy signals don't change.** The order picker already leaves little sign correlation. On
  noisy-sine at order 1, the sine's slope cancels the noise's −0.5 lag-1 correlation (measured
  −0.01, original evaluation). Impulses use order 0, whose values are never negative.
- **Not adopted.**

**Configuration at f = 0.5, whole block** (original evaluation, 12,600 blocks, block minima not
stored; timings alternate, best of 7):

| config | continuous bits/sample | discretized bits/sample | encode µs | decode µs |
|---|---|---|---|---|
| no noise floor | 6.08 | 8.14 | 6.64 | 3.90 |
| noise floor f = 0.5 | 4.13 | 5.73 | 8.03 | 4.06 |

- **Noisy signals shrink 2–3×:** noisy sines 13.7 → 4.4, impulses 11.7 → 4.1, gauss-spikes 12.2 → 5.7 (current run).
- **The added error is 0.10–0.12σ RMS.** Error against the clean signal goes from 1.000σ to 1.004–1.009σ.
- **Cost:** +1.4 µs encode, and decode is unchanged (the difference is timing noise).
- **Noisy blocks skip decimal detection:** on a noisy block, a decimal grid finer than the
  coarsened step isn't used. The noisy sine at 0.1 goes to 4.42 bits/sample, like its continuous
  version.

**In the per-minute matrix** (`report/index.html`, which runs the fluxcode package), sweeping f at
B = 16 moves the median over the 12 kinds from 5.23 bits/sample (f = 0, 0.01 and 0.03) to 4.86
(f = 0.1), 4.39 (f = 0.25), 3.92 (f = 0.5) and 3.32 (f = 1).
- **Only the noisy kinds move.** Clean kinds are identical at every f; noisy-sine goes 13.74 →
  10.71 → 8.64 → 7.12 → 5.48 → 4.44 → 3.42 bits/sample over f = 0, 0.01, 0.03, 0.1, 0.25, 0.5, 1:
  about 1.5–2 bits/sample for each ×3 of f.
- **fluxcode-16 at f = 0.5 (3.92) keeps full 16-bit precision on clean signals** (0.00056% RMSE
  on the sines, against delta0123-zstd-10's 0.028% at 2.07 bits/sample) and spends its error only
  on the noisy ones.
- Error against the clean signal stays within 0.3% of the noise-only floor up to f = 0.25, and
  rises 2.5–4% at f = 1 (`bench/fluxproto_noise_floor.py`).
- **Caveat:** against the noisy input, the noise-floor points sit on the one-shot curve
  (noisy-sine at f = 0.5: 4.44 bits/sample at 0.26% RMSE, the same bit gain as delta0123-zstd-8).
  That axis counts the noise as signal. Against the clean signal, the added error is 0.1σ.

**Recommendation for this codec:** orders 0–3 with decimal detection, B = 16, bit-shuffled
planes, zstd-3, and the noise floor for the whole block, turned off per channel where
high-frequency content matters. This investigation suggested f = 0.5; fluxcode ships with
f = 0.25, which against the clean signal is indistinguishable from storing the noise exactly
([docs/TUNING.md](../docs/TUNING.md)).

---

## 12. Recommendation

**Use fluxcode** (the package at the repository root; format in [docs/SPEC.md](../docs/SPEC.md)),
with its defaults: B = 16 (`max_quantize_bits`), a floor of 6 bits (`min_quantize_bits`), orders
0–3, decimal detection, bit-shuffled planes, zstd level 3, noise floor f = 0.25, one unit per
channel-minute (60 blocks of 1000 samples).

```
unit per minute:  16-byte header (version, block length, sample count)
                  || zstd( head[N] | param[N] | anchor[N] | bit planes 0..15 of zigzag(diff(q, order)) | nonfinite codes )
decode:           q = cumsum^order(residual) mod 2^16 ; x = lo + q·2^e  (or (K0 + q)·10^p on a decimal grid)
```

**Why this design:**
- **Error is bounded and predictable:** at most half the power-of-two step, 2⁻¹⁶ of the block's
  range on clean blocks (0.0015%), never worse than 1.6% of the range whatever the noise floor
  does, and min is exact.
- **Decimal data is lossless,** and costs what the lossless encoding at its quantum costs (§11).
- **Noisy channels don't pay for their noise:** at f = 0.25 the noise floor cuts the noisy kinds
  1.8–2.5× for 0.05–0.06σ of added RMS error, and leaves clean ones untouched.
- **Edits are stable,** and `update` replaces blocks without touching the others.
- **It's fast:** about 3.2 µs to encode and 1.9 µs to decode a block on one core
  ([bench/RESULTS.md](../bench/RESULTS.md)), one compressed call per unit.
- **It handles any float64:** NaN, ±inf, subnormal ranges and ±DBL_MAX spans.

**The step before it: delta0123-zstd** (§7, §9). A range-scaled step of (max − min)/(2ᴮ − 1),
the same order pick, byte planes and one zstd call per minute. At B = 10 it's 2.07 bits/sample in
the median at 0.028% RMSE. It's smaller than fluxcode in this matrix because the synthetic sines
repeat across a minute in byte planes (§3); on the noisy, spiky, random-walk and chirp kinds the two
are within a few percent at B = 10, or fluxcode is smaller. What fluxcode adds is what the
matrix doesn't score: exact decimals, a noise floor, stable edits, self-describing units and
non-finite values. If only 8-bit precision is needed and simplicity matters most, plain quant8 +
delta mod 256 + zstd per minute (quant8-delta1-zstd) is about 3× faster than delta0123-zstd-8 at
the same size (§9).

**If constant width is required,** use `cw8-delta1-bfp-e4` (§10): 8.41 bits/sample with the
block's min and max, median RMSE 0.007%, max error guaranteed ≤ 0.196% of range, about 10 µs
encode and 3 µs decode per block.

---

## 13. Open items

**Resolved in fluxcode** (the package at the repository root; [docs/SPEC.md](../docs/SPEC.md)):

- **Quiet periods and sensor resolution.** The proposed per-channel absolute floor became two
  per-block mechanisms that need no configuration: decimal detection makes channels already on a
  decimal grid (like `sensor-0.1`) lossless, and the noise floor stops blocks of white noise from
  spending bits on it (§11).
- **Deployment details:**
  - partial blocks: padded by repeating the last sample, trimmed on decode (SPEC §2)
  - NaN and ±inf: encoded exactly, at no cost to blocks without them (SPEC §2)
  - constant blocks: handled (range 0)
  - a format version: every unit starts with a 16-byte header carrying the version, block length
    and sample count, so units are self-describing (SPEC §5)
  - endianness: specified little-endian (SPEC §5)
- **Speed-ups.** With each unit encoded in one compiled call, the order picked on the first 250
  samples and bit-shuffled planes, encode takes about 3 µs per block on one thread ([docs/PERFORMANCE.md](../docs/PERFORMANCE.md)).
- **The size estimate for a cap.** fluxcode's optional `target_bits_per_sample` estimates each
  block from the class entropy of its residuals rather than log2(std) + 2.05, which fails on
  sparse residuals (SPEC §3.6).

**Still open:**

1. **Real data** (highest priority). Every result here is synthetic. Rerun the benchmarks (the
   report, and fluxcode's `bench/`) on a representative set of real channels: quiet, noisy,
   periodic/vibration, steppy/discrete. Check especially the order mix, how often the noise gate
   and decimal detection fire, and the size target's estimate against actual size
   (`bench/estimate.py` at the root).
2. **Non-decimal quanta** (ADC counts in engineering units, 0.5, 2⁻ᵏ steps) still use the
   power-of-two grid at 1.4–5 bits/sample above lossless (§11). A longer candidate list or a
   configured per-channel quantum would close that gap.
3. **Strong oscillation** (vibration, mains hum): fitted linear predictors (FLAC-style
   [LPC](https://en.wikipedia.org/wiki/Linear_predictive_coding)) feeding the same zstd back end
   would give 5–200× lower error on those signals (§6). Encoding stays vectorizable; decoding
   becomes a short recursive filter (fine in numba).
4. **Streaming rate control** (a shared Δ adjusted by feedback), only if absolute-error targets
   become more important than per-block relative error. fluxcode's target is a soft cap per unit,
   not a controller.
5. **Constant-width follow-ups** (§10), not pursued:
   - a 3-bit exponent for cw8-delta1-bfp (mean exponents are 1–9)
   - a floor on delta0123-zstd's cap (never below B = 8), so its worst case matches quant8 at the
     cost of noisy blocks exceeding 8 bits/sample
   - encoding several blocks side by side in SIMD lanes, the only way to vectorize closed-loop
     encoding

---

## 14. Files and how to run

Everything runs from `experimental/` (see [README.md](README.md) for the full module map):

| path | what it holds |
|---|---|
| `notes/` | the original request and the design prompts, in order |
| `tslab/common/` | `datasets.py` (every test signal: the 12 kinds as one-minute signals, the discretized sets, the report's units), `unit.py` (the unit codec harness), `bitio.py`, `intcode.py` (zigzag, varint, order pick) |
| `tslab/classic/` | quant8, quant*-delta1-deflate, companded DPCM, binary tree, piecewise linear / PCHIP, swinging door, Gorilla, DCT (§2, §4) |
| `tslab/entropy/` | rate control and its coders (`ratectl.py`, §6), the matrix wrappers (`ratectl_codecs.py`), delta0123-zstd (`delta_zstd.py`, §7, §9), numba ports (`native.py`, §8) |
| `tslab/constwidth/` | cw8-delta1-linear / -sqrt, cw*-delta1-bfp, cw-delta1-tree (§10) |
| `tslab/flux/` | fluxproto, the fluxcode prototype (`proto.py`, §11), and adapters for the fluxcode package (`adapters.py`) |
| `make_report.py` | the codec matrix → `report/index.html` (summary by size class, size-vs-error chart, fluxcode and noise-floor sweeps, DPCM sweep, per-kind tables and plots) |
| `make_rate_report.py` | the rate-control experiment (§6) → `report/rate.html` |
| `make_time_report.py` | fluxcode's time axis (timestamp shapes and their cost) → `report/time.html` |
| `bench/` | one script per experiment, below |

```sh
uv sync                                         # installs fluxcode from .. (editable)
uv run python make_report.py                    # report/index.html          (~3 min)
uv run python make_rate_report.py               # report/rate.html           (~3 min)
uv run python make_time_report.py               # report/time.html           (~5 s)
uv run python -m bench.native_speed             # §8: Python reference vs numba ports          (~45 s)
uv run python -m bench.chunking                 # §9: per-block vs one-minute chunks           (~11 s)
uv run python -m bench.delta_zstd               # §9: delta0123-zstd per minute, 200 signals/day (~7 s)
uv run python -m bench.constwidth_8bit          # §10: constant-width formats head to head     (~8 s)
uv run python -m bench.fluxproto_sweep          # §11: B sweep, discretized data, decimal detection, plane layouts (~22 s)
uv run python -m bench.fluxproto_noise_floor    # §11: noise floor                             (~2 s)
uv run python -m bench.fluxproto_truncation     # §11: block step vs low-bit truncation        (~1 s)
uv run python -m bench.fluxproto_plane_models   # §11: per-plane probability models vs zstd    (~2 s)
uv run python -m bench.fluxproto_raw_planes     # §11: near-random planes stored raw           (~1.5 s)
uv run python -m bench.fluxproto_relative_sign  # §11: relative-sign zigzag                    (~1 s)
uv run python -m bench.fluxcode_sweep           # §11: the same sweep on the fluxcode package  (~7 s)
```

`report/` is generated and not tracked. Versions used: Python 3.11, numpy 2.4, scipy 1.17,
matplotlib 3.11, zstandard 0.25, numba 0.67, pyflac 3.0. Timings: Apple M3, one thread,
fastest of 5 runs.

---

## References

Techniques and prior work referred to above.

- **Quantization and prediction:** [quantization](https://en.wikipedia.org/wiki/Quantization_(signal_processing)),
  [delta encoding](https://en.wikipedia.org/wiki/Delta_encoding), [DPCM](https://en.wikipedia.org/wiki/Differential_pulse-code_modulation),
  [companding](https://en.wikipedia.org/wiki/Companding) and [μ-law](https://en.wikipedia.org/wiki/%CE%9C-law_algorithm),
  [block floating point](https://en.wikipedia.org/wiki/Block_floating_point), fixed polynomial predictors as in
  [Shorten](https://en.wikipedia.org/wiki/Shorten_(file_format)) and
  [FLAC (RFC 9639 §9.2.5)](https://www.rfc-editor.org/rfc/rfc9639#name-fixed-predictor-subframe),
  [linear predictive coding](https://en.wikipedia.org/wiki/Linear_predictive_coding).
- **Curve and transform codecs:** [piecewise linear](https://en.wikipedia.org/wiki/Piecewise_linear_function),
  [PCHIP](https://en.wikipedia.org/wiki/Monotone_cubic_interpolation) (Fritsch–Carlson),
  [DCT](https://en.wikipedia.org/wiki/Discrete_cosine_transform), swinging door trending (E. H. Bristol;
  [US 4,669,097](https://patents.google.com/patent/US4669097A), expired).
- **Float codecs:** [Gorilla](https://www.vldb.org/pvldb/vol8/p1816-teller.pdf) (Pelkonen et al.,
  VLDB 2015), [Chimp](https://www.vldb.org/pvldb/vol15/p3058-liakos.pdf) (Liakos et al., VLDB 2022),
  [ALP](https://doi.org/10.1145/3626717) (Afroozeh et al., SIGMOD 2024;
  [code](https://github.com/cwida/ALP)).
- **Entropy coding:** [entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory)),
  [Golomb/Rice coding](https://en.wikipedia.org/wiki/Golomb_coding), [FLAC](https://en.wikipedia.org/wiki/FLAC),
  [deflate](https://en.wikipedia.org/wiki/Deflate), [Zstandard (RFC 8878)](https://www.rfc-editor.org/rfc/rfc8878),
  [arithmetic coding](https://en.wikipedia.org/wiki/Arithmetic_coding), [ANS / rANS](https://en.wikipedia.org/wiki/Asymmetric_numeral_systems),
  [LZMA](https://en.wikipedia.org/wiki/Lempel%E2%80%93Ziv%E2%80%93Markov_chain_algorithm).
- **Integer and bit layout:** [zigzag](https://protobuf.dev/programming-guides/encoding/#signed-ints),
  [LEB128](https://en.wikipedia.org/wiki/LEB128), [bit-shuffle](https://github.com/kiyo-masui/bitshuffle)
  (Masui et al., [arXiv:1503.00638](https://arxiv.org/abs/1503.00638)),
  [count leading zeros](https://en.wikipedia.org/wiki/Find_first_set), [SIMD](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data).
- **Statistics and filtering:** [rate-distortion theory](https://en.wikipedia.org/wiki/Rate%E2%80%93distortion_theory),
  [mean absolute deviation](https://en.wikipedia.org/wiki/Average_absolute_deviation),
  [autocorrelation](https://en.wikipedia.org/wiki/Autocorrelation), [Savitzky–Golay filter](https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter),
  [Euclidean algorithm](https://en.wikipedia.org/wiki/Euclidean_algorithm).
