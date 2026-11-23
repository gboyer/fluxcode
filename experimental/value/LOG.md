# value-exp: experiment log

Every hypothesis tried on branch `value-exp` (2026-11-21), how it was tested, what it measured and
what was decided. Written for a later review of production worthiness: what shipped, the evidence
behind it, the risks, and what was rejected (so it isn't retried).

The work was an open-ended brief from the project owner: make fluxcode better in accuracy,
performance and compression, value side first, format changes allowed (unreleased project),
accuracy assumptions open to question, nothing pushed to main.

## Contents

1. [Setup: corpus, tools, how to reproduce](#1-setup)
2. [Results against main](#2-results-against-main)
3. [Shipped: hypotheses that held](#3-shipped)
4. [Rejected: hypotheses that didn't](#4-rejected)
5. [Decisions taken with the owner](#5-decisions-taken-with-the-owner)
6. [Open decisions](#6-open-decisions)
7. [Production review checklist](#7-production-review-checklist)
8. [Review response (2026-11-22)](#8-review-response-2026-11-22)

## 1. Setup

**Corpus** (`corpus.py`): built because `tests/_signals.py` (the 1 GiB benchmark's 15 signals) is
mostly synthetic shapes. Three sets:

- **fast**: high-rate waveforms as a DAQ or condition-monitoring system records them, on their
  ADC's grid: an accelerometer with gear mesh and a resonance (and a sigma-delta variant behind a
  decimation filter), motor current and line voltage on 16-bit ADCs, pump pressure on a 12-bit
  4–20 mA input, a load cell, a servo axis on a 0.001 mm encoder, turbulent flow, a tachometer, a
  10 Hz thermocouple. 240,000 samples each, 1000-sample blocks.
- **slow**: the 19 slow 1 s tags of fluxprune's corpus (temperature, flow, tank level, valve,
  pH as float32, totalizer, batch, compressor, furnace, grid frequency...) in hourly blocks.
- **synth**: the benchmark's signals, kept on purpose including the pathological ones (chirp,
  exact sines): the owner asked that they not be compromised, but they need not compress well.

Plus: `archive_time.py` (simulated historian archives: swinging door at each tag's CompDev, a day
per group, hourly time blocks), `rates.py` (one process sampled from 1 kHz to 1/min),
`escapes_real.py` (glitch scenarios), `squares.py` (noisy square waves at off-grid levels).

**Tools**:

- `sizes.py out.json [sets]`: bits/sample, RMSE and worst error per signal; settings from env
  (`FX_F` noise floor factor, `FX_EFFORT`, `FX_BITS` max_quantize_bits, `FLUXCODE_X_*`
  experiment switches). `compare.py a.json b.json`: per-signal and per-set ratios.
- `harness.py`: encodes through the production encoder, swaps residual streams, measures through
  the real back end (planes + zstd). It sets a per-source `NUMBA_CACHE_DIR` and self-checks that
  the compiled noise kernel matches the settings (a stale numba cache silently mixes old callees;
  see `bench/_fresh_numba_cache.py`).
- `fluxcode.verify`: checks a call's result against the documented guarantees; used on every
  shipped change (e.g. tolerance: 0 violations in 309 group encodes of the corpus at 3 settings).
- Benchmarks (`bench/bench_gb.py`, 1 GiB) only on AC power on a quiet machine. The README table
  is the Python path (`FLUXCODE_RUST=0`), to compare with main's numbers, which were taken that way;
  with the Rust packer encode is about 0.3 µs/block faster (5.40 vs 5.71 µs).

**Reproduce**: from the repo root, `cd experimental/value && uv run python sizes.py /tmp/a.json`,
then the same on another checkout and `uv run python compare.py /tmp/main.json /tmp/a.json`.

## 2. Results against main

| measure | main | value-exp |
|---|---|---|
| 1 GiB benchmark, 15 signals | 4.76 bits/sample | 4.01 (−16%) |
| of which the 2 random walks (floored by the new definition, now lossy) | 12.62, 9.28 | 5.63, 5.63: 94% of the saving |
| the other 13 signals (same meaning as main) | 3.81 | 3.76 (−1.3%; gauss-spikes −8%, quadratic +0.07 b/s) |
| corpus fast (DAQ-like) | | −28% |
| corpus slow (1 s tags, hourly blocks) | | −13%; −28% more in 4-minute blocks |
| corpus synth | | −11% |
| sparse daily archives (tank level, valve) | 577 / 461 B/day | 473 / 355 (−18% / −23%) |
| glitched blocks (one −9999 or 1e6) | rest of block within 0.125 / 5 | within 0.000244 (as clean blocks) |
| encode, Python path | 5.12 µs/block | 5.71 (+12%) |
| decode (whole branch, 1 GiB run) | 1.79 µs/block | 1.75 (−2%; order-0 blocks alone about 6% faster, 3.3) |

Full benchmark tables: `bench/RESULTS.md`. Tests: 1871 pass, Rust extension included.

## 3. Shipped

Each entry: the hypothesis, the evidence, the decision, the commits, and what a reviewer should
look at. Docs: ENCODER.md and SPEC.md sections named; TUNING.md has the noise floor tables.

### 3.1 Noise floor: the top band's spectral level (12th differences)

- **Hypothesis.** main estimated noise from second differences and gated on their lag-1
  autocorrelation. Any signal that also shows in second differences (a slow process under the
  noise) fails that gate and keeps the full 16-bit step: the floor was missing most real noisy
  signals. Higher-order differences reject the signal and keep the top band, where sensor noise
  is flat.
- **Definition adopted** (the owner asked for one; see `noise-floor-purpose` in the agent's memory and
  ENCODER §5.2): the floor is the signal's own spectral density in the top of its band, where that
  band is flat; step ≤ f × that level. At 1 kHz it is sensor noise; at 1/min it is the process's
  move between samples. No sample-rate parameter.
- **Evidence.** `estimators.py`, `filters.py`, `gate.py`, `shortgate.py`, `bands.py`, `lags.py`:
  k = 2, 4, 6, 10, 12 compared; k = 6 left the accelerometer (a 2.8 kHz resonance at 10 kHz)
  unfloored, k = 10–12 floored it, past 12 nothing changed. Corpus: fast −26.1%, slow −10.3%,
  synth −11.8% (with the order pick, 3.2). Motor current −30%, turbulent flow −53%, thermocouple
  −55%, accelerometer −47%, pH float32 −52%; added RMS error 0.06σ, as on blocks floored before.
  `rates.py`: one process from 1 kHz to 1/min floored at every rate (13.3–14.5 → 5.2–6.4 b/s);
  main gated only at 10 Hz and above. Clean shapes byte-identical with the floor on and off.
- **Guards added along the way**, each from a failure it fixed:
  - lag-2 whiteness check: tones near half the sample rate read as white at lag 1;
  - first-difference cap (1.25×): dense steps (a square wave of period 20) put a burst into
    every 13-sample window; first differences see each step once (`squares.py`: errors within
    0.16σ of the noise at periods 6–1000);
  - lower bands (0.375, 0.25, 0.125 fs) when the top band is an anti-alias stopband, with a
    long-lag decay test (lags 20–23) against tones: sigma-delta accelerometer 13.7 → 6.5 b/s
    (`aafilter.py`);
  - non-finite blocks keep k = 12 while half their windows are finite; irregular times keep
    k = 2 chords on consecutive scans (cadence policy unchanged);
  - `keep_floor_grid`: decoded data re-encodes to itself (fixed point) when σ sits on a power of
    two (measured over 5 decode/encode rounds);
  - from 128 samples instead of 256 (12th differences reject random walks with fewer): 1 s tags
    in 4-minute blocks 5.0 → 3.6 b/s (`shortblocks.py`, `shortgate.py`);
  - early exit: a block whose first 128 differences read under 1/1024 of the smallest σ that
    could matter stops there (clean blocks 1.6 → 0.11 µs; same bytes).
- **Semantic consequence (accepted by the owner):** random walks are floored (σ_e/2 is their
  top-band level): random-walk 12.6 → 5.7 b/s, its 0.01-decimal variant lossy now (+0.31% of
  range max error).
- **Cost.** About 2 µs per noisy 1000-sample block (12th differences 0.76, robust spread 0.93,
  whiteness 0.36, cap 0.57 µs): most of the +12% encode against main.
- **Commits.** f03e3fb, cfd4dcc, db3915f, 100fa42, d1d0227, b287bf2, 661860e.
- **Review.** `fluxcode/_noise.py`. The gate thresholds were tuned on synthetic and simulated
  data: validate on real recordings (TUNING.md "Validating on your data"). Watch for false floors
  on deterministic signals (tones, PWM) and for lost detail in quiet stretches.

### 3.2 Order pick on 8 spread windows

- **Hypothesis.** main chose the predictor order on the first 256 samples; a block that changes
  part way (a servo move after a dwell, a batch ramp) was judged on its start.
- **Evidence.** `pickwin.py`, `pickwin2.py`: 8 windows of 32 spread over the block pick as well as
  the whole block, at the same cost: fast −1.4 to −1.7%, slow −5.4 to −6% (batch −13%,
  compressor −8%, a servo −13%). Whole-block variance no better; clipped variance or MAD under 0.1%.
- **Commit.** f340bed. ENCODER §5.6. Decoder unaffected (the order is stored).

### 3.3 Format: signed indices around a centered anchor

- **Hypothesis.** Order-0 residuals were indices from the block minimum, so each block's
  distribution sat at its own offset and zstd's per-group Huffman tables saw a mixture.
- **Evidence.** `center.py`: fast −1.5%, slow −2.9% (slow motor current −11%, a quiet tank level
  −18%, a load cell −10%, a servo's dwells −19%, the pyrometer −23%); decode about 6% faster
  (order-0 decode reads residuals directly). A trimmed mean, not a plain one: spikes pulled the
  plain mean off the bulk (about 1% on impulses and the pyrometer).
- **Commits.** 9319c2f, f0b2fab, 100fa42. SPEC §4, ENCODER §5.7.
- **Review.** A format change (int16 signed indices, anchor at the center). Exactness argument in
  ENCODER §5.7; blocks wider than 2^15 steps shift the center (quadratic 0.24 → 0.32 b/s).

### 3.4 Period predictor (effort ≥ 6)

- **Hypothesis.** Mains-frequency signals (motor current, line voltage) repeat every cycle; a
  prediction from one period earlier beats fixed differences.
- **Evidence.** `period.py` (an offline prototype: free-gain 2-tap least squares at the best of
  all lags, its own residual swap) first gave motor current −41%, line voltage −19%, fast −31%
  overall, on the corpus and encoder of the time; the shipped predictor (unit gain, fractional
  lag, crossing-based search, `ptime.py` for its speed: 8.5 → 3.5 µs/block) cost +30–90% encode
  on oscillating blocks, so it is gated to effort ≥ 6 and off with a target. The −41% is not a
  baseline for the shipped code: re-measured 2026-11-22 on the current branch, the shipped
  predictor gives motor current −15.2% at effort 4 (forced on) and −15.3% at effort 6, line
  voltage −18.7% / −17.7%, fast set −3.7% in total, nothing else; the prototype itself now gives
  motor current −2.7%, line voltage −12.3%, but −9.7% / −8.9% on the two accelerometers, where
  the shipped one finds nothing (a free-gain 2-tap predictor acting as short-term prediction: an
  open lead, §8). The faster search gave up nothing.
- **Size can grow.** An exactly repeating clean sine (zstd already matches its repeats) can come
  out larger at effort 6–9 than at effort 5: 4 of 45 sines tried (12 bits) grew, by up to +58%
  (908 → 1432 B). That breaks ENCODER §1's "no block group grows from effort 5 to 9"; the test
  of it uses signals the predictor doesn't touch (§8).
- **Commits.** 1d24bcf, 133d23f, 4af70cf. SPEC §4 (block flag bit 7), ENCODER §5.7.1. The
  parameters first lived in `grid_scale`'s high half (widened to 8 bytes for every block); since
  2026-11-22 they are their own 3-byte `period` column (§8), Rust interface version 6.
- **Review.** Decode: see §8 (period blocks now decode 8% faster than without the predictor).

### 3.5 Escaped samples (glitches)

- **Hypothesis.** One error code (−9999) or overrange value (1e6) stretches its block's range and
  coarsens every other sample's step: a −9999 on a 5 ± 0.3 bar signal left the rest within 0.125
  (42 noise σ), a 1e6 within 5 (`glitches.py`). Storing such samples apart restores the block.
- **Design** (discussed with the owner):
  - 2-bit codes: 00 on the grid, 01 NaN, 10 escaped (next float64 in the `escapes` field at the
    body's end), 11 repeat the previous escape (a dropout run stores its value once). ±inf
    became escapes, removing main's asymmetric +inf/−inf codes.
  - Payload: exact when float32 holds it (error codes), else rounded to the step the block would
    have had without escaping, so no sample decodes worse than before and the low mantissa bytes
    compress away (the owner's mantissa-limited suggestion, realised by rounding).
  - Detection by benefit, not by sigmas alone (the owner asked about IQR cost): skip unless the
    noise floor leaves room (free check); trigger on the mean within 5% of one end of the range
    or a range over 16 standard deviations; trimmed center; candidates beyond 16 robust
    deviations; take them only if at most n/16 and the bulk's step is at least 4× finer.
  - After escaping, the noise is re-estimated without the escaped samples (a dropout run hid it;
    glitches on both sides inflated it 4×).
- **Evidence.** `escapes.py` (prototype), `escapes_hard.py`, `escapes_real.py`: single,
  two-sided (−9999 and 1e6; ±1e6; ±1e6 with ±inf), dropout runs: the other samples within
  0.000244 (clean blocks' precision), codes exact; 0.01-decimal data with a −99999 code exact
  again; spikes in noise whose floor already sets the step aren't escaped. Realistic corpus
  unchanged in size. Encode +0.1–0.2 µs a block (first version cost 1.85 µs before reordering).
- **Commits.** 7392e6b, a0b2e23, ca330f5, b0172b2. SPEC §2, ENCODER §5.1.1.
- **Review.** Format change. Escaped groups pack through the Python path (the Rust compress path
  doesn't take escapes); `verify` loosens its step bound only in blocks with escapes. Fuzzing with
  glitches was added at the end (see 7).

### 3.6 `Params.tolerance` (opt-in absolute error permission)

- **Hypothesis.** Every precision setting was relative (bits of the block range, a share of the
  noise). Engineers often know an absolute accuracy (±0.1% of span, ±0.5 °C); storing finer is
  waste.
- **Design.** Each block's step may coarsen up to 2 × tolerance (never past min_quantize_bits);
  where tolerance sets the step and the block has no exact grid, a scaled step N·2^(e−4)
  (N 17–31) uses 94–100% of the allowance instead of the power of two's 50–100%. Decimal data
  stays exact while tolerance ≤ half its step.
- **Evidence.** `tolerance_span.py`, corpus mean bits/sample: 4.16 at 0; 3.86 at 0.01% of span
  (−7%); 2.92 at 0.1% (−30%); 2.05 at 1% (−51%). The scaled step is −3% of that. Worst error
  within tolerance on every signal; `verify`: 0 violations in 309 group encodes.
- **A bug caught on the way:** the first scaled-step version let the step pass min_quantize_bits'
  cap where tolerance's exponent equalled it (a random walk came out smaller at 0.1% than at 1%).
- **Naming** (owner's feedback): first `max_error`; renamed because it isn't a maximum (the noise
  floor may be coarser) and it needs long-horizon knowledge of a tag's range, so it is documented
  as a per-tag permission, not a set-and-forget default. fluxprune's `max_error` is unrelated.
- **Commits.** 49218b5, d76b360, 0d87987. ENCODER (Params), README.
- **Review.** API addition; default 0 changes nothing. A coarser step can break an exact grid (a
  ramp: +0.003 b/s).

### 3.7 Time quantum

- **Hypothesis.** Second-resolution data usually arrives in ns ticks (pandas' default): every
  block start increase and step is a multiple of 10^9, which isn't a power of two, so all eight
  byte planes of those columns look random. Sparse tags in fixed hourly buckets (about 2 points
  a block) pay most.
- **Design.** Header flag bit 4; the start increases (all but the first non-empty block's) and
  steps stored over their GCD q ≥ 2, q a uint64 at the body's end. Applied in `_compress.pack`
  and undone in `_group.decompress`, so decode and update splicing see plain ticks. The Rust
  packer copies trailing bytes, so it stays on.
- **Evidence.** Simulated first by dividing the columns in real day groups (−21/−23% sparse),
  then shipped: tank level 577 → 473 B/day (−18%), valve 461 → 355 (−23%), dense tags −0.1 to
  −0.5% (`archive_time.py`). An update off the seconds lowers q; restoring the block restores
  the group byte for byte (test).
- **Commit.** bfe5a7d. SPEC §3 and §6, ENCODER §4.

### 3.8 Benchmarks and measurement-only commits

cdced2e, 5655570 (1 GiB reruns on AC power), cfc4ef5, 44901e5 (studies), ef6e7f2 (block length),
065b844 (`FX_BITS`).

## 4. Rejected

| hypothesis | test | result | why rejected |
|---|---|---|---|
| Per-block integer LPC | `predictors.py`, `lpc2.py`, `lpc3.py` | side info eats the gain; on differences with ridge +3.6% (worse) on fast | no gain |
| Fixed smoothing predictors (LS polynomial over the last m) | `smoothing.py`, `family.py` | −0.6 to −2% after the windowed order pick | too small for a format change |
| A better entropy coder | `coder_gap.py`, `param_bound.py` | zstd on planes within 2–9% of the residuals' order-0 entropy | ceiling too low |
| Oracle noise estimate | `oracle_floor.py` | an upper bound: each fast signal quantized at its known white σ | a yardstick for the estimators, not a candidate |
| Better bit/byte plane rule | `layout_rule.py` | no rule beat the current one-pass heuristic meaningfully | no gain |
| Whole-block order pick | `pickwin.py` | no better than 8 spread windows | cost |
| Time residuals as byte planes | `time_planes.py`; implemented with a header bit | −7% of time bytes alone, −0.8% of archive groups | header bit, double compress, update choice |
| Noise on 4–6 spread segments of long blocks | quiet-stretch test | −20% row time on 3600-sample blocks, but misses quiet stretches of 500–800 samples (worst σ 1.0 vs 0.08) | loses shape |
| Exact floor step (scaled f·σ) | `FLUXCODE_X_FLOOR_EXACT` prototype | at equal bytes (f = 0.177) RMSE +1%, one signal 3.3× | same curve as changing f |
| Noise micro-optimisations | kernel timings | branch-free cap −0.1 µs, fused clamp+autocorrelation −0.03 µs; ~0 end to end | no gain |
| Per-block field coding | column marginal costs | per-block fields under 1% of compressed bytes (except already tiny signals) | no gain |
| Built-in zstd dictionary | trained on days 3–12, tested on 0–2 | −10% on sparse groups only | ties the spec to a trained blob |
| Magicless zstd frames | arithmetic | 4 B per group (under 1%) | too small |
| Anchors delta-coded across blocks | column simulation | sparse groups +8% (grids differ; float64 anchor bits) | worse |
| Closed-loop (SZ-style) prediction | analysis | identical residuals: integer predictions on the grid round to the same indices | no gain |
| Block length 2000 / 4000 | `blocklen.py` | −0.5% / −1.1%, but RMSE up to 2× / 4× on drifting signals (quadratic 4×, servo 2.6×) | accuracy; 1000 stays |
| Block length 500 | `blocklen.py` | +1.4% | worse |
| max_quantize_bits 14 / 12 | `sizes.py` with `FX_BITS` | −2.5% (err 0.006% of block range) / −7% (0.024%; quadratic doubles); clean signals only | the owner's call; left at 16, `tolerance` covers it per tag |

## 5. Decisions taken with the owner

- Random walks are floored (the spectral definition's consequence): accepted.
- Signed centered anchors (format change): accepted.
- Period predictor as an effort ≥ 6 option: accepted.
- Escaped samples: wanted; design discussed (detection cost, ±inf symmetry, payload size), and
  glitches on both sides confirmed handled.
- `max_error` → `tolerance`, documented as a per-tag permission.
- Block length stays 1000.
- The branch is pushed to `origin/value-exp` as a backup; never merged to main by the agent.

## 6. Open decisions

- **Default f** (noise floor factor): 0.25 kept. 0.5 is −5 to −9%, 1.0 −11 to −20% (TUNING.md:
  against the clean signal f ≤ 0.25 is indistinguishable from exact; 0.5 costs under 1%, 1.0
  2.5–4%). Decide on real data.
- **max_quantize_bits default:** 16 kept (see 4).
- **Encode speed:** if +12% matters, move `noise()` into the Rust extension; the Python/numba
  version has no cheap wins left.
- **Sparse tags:** remaining overhead is per-block fields in mostly empty hourly buckets; longer
  time blocks for slow tags would gain more than further format work.

## 7. Production review checklist

- **Real data.** Everything above was measured on synthetic and simulated data. Before
  production: run `sizes.py`-style measurements and `fluxcode.verify` on real recordings at
  several rates; inspect floored blocks for lost features (TUNING.md "Validating on your data").
- **Format changes to sign off** (no stored data exists yet, so no compatibility code was kept):
  signed centered indices (SPEC §4); 2-bit codes 10 escape / 11 repeat and the trailing
  `escapes` field (SPEC §2, §6); block flag bit 7 period and its 3-byte `period` column (SPEC §4,
  §6; `grid_scale` is back to 4 bytes);
  header flag bit 4 and the trailing `time_quantum` (SPEC §3, §6). Golden hashes were
  regenerated for these.
- **Rust.** Interface version bumped for the period predictor; the Rust compress path (rows to
  body) declines groups with a time axis or escapes, which are written in Python and then packed
  by the Rust packer (it copies the bytes after the residual planes: escapes, time quantum).
- **verify.** Step bounds account for escapes per sample (an escaped sample to the full range's
  bound, the rest of its block to the grid) and for tolerance; decimal exactness is checked only
  where tolerance permits it.
- **Fuzzing.** `tests/test_fuzz.py` covers random magnitudes, grids, non-finite values and
  random clocks (whose periods exercise the time quantum). At the end of the work it was extended
  with finite glitches (−9999, ±1e6, 32767: escapes) and a `tolerance` params entry, drawn after
  the existing draws so recorded regression cases keep their inputs. It found two hangs, an
  index wrap and fixed-point failures, all fixed and pinned (appendix below).
- **Performance.** Encode +12% (noise estimate), decode −2%; the period predictor adds encode
  time on oscillating blocks at effort ≥ 6 only.

## 8. Review response (2026-11-22)

An independent review of the branch at c8fbec7 was checked claim by claim against the code; every
factual finding held except the period predictor's numbers, which had a different cause (3.4).

**Fixed:**
- SPEC §4/§5 described period blocks wrong (said the head is first differences and the order is
  taken over all errors; the code writes the head as order-k residuals and the errors raw). SPEC
  now matches the decoder; `test_period_blocks_decode_as_the_spec_says` rebuilds period blocks
  from the rule alone.
- One function plans a block's step (`_encoder.plan_block_step`: max_bits, noise floor, tolerance,
  min_bits, keep_floor_grid); tolerance's scaled grid is its own helper (`tolerance_grid`), with
  why it only applies under `grids="rational"`; the noise gate test is one helper
  (`floor_exponent`), shared by escape detection. Output byte-identical (golden hashes).
- Experiment code out of production: the `FLUXCODE_X_NOISE_ORDER` switch and the order-2/6/generic
  branches of `noise()` (comparing orders now needs a checkout before 2026-11-22), the unused
  `_lag2_white`, `BLOCK_FLAG_RESERVED = 0` and its test, `_coded_diffs`' unused order 1.
- Groups with escapes (and updates of them) now go through the Rust packer too: it was byte-identical
  on 740 glitch groups at efforts 1, 4, 6, 9. `pack` no longer calls itself for the time quantum.
- `verify` checks escaped blocks per sample (was: the whole block loosened to the full range's
  bound, so the claim that the rest keeps clean precision was never checked); the identical
  ternary in the decimal check removed.
- Names left from `max_error`: `tolerance_step`, `TOLERANCE_STEP_BITS`, `tolerance_num/_fine_exp/_base`.
- Docstrings: `_noise` (module, `noise()`, second-difference wording, `MIN_ONE_SCAN_TRIPLETS`),
  `escape_outliers` (both triggers, robust standard deviations), `encode_block`/`encode_group`
  (new arguments, centered anchors, period bits in grid_scale), `try_period` and
  `PERIOD_SIDE_BITS` (a variance estimate, not class entropy), `GroupRows.grid_scales`, the
  decoder's reserved-bit comment, `Params.effort` (the period predictor from 6), `Params.tolerance`
  and ENCODER (the scaled step needs `grids="rational"`), TUNING.md (a tolerance section).
- This log: the headline split (random walks 94% of the 1 GiB saving), decode figures labelled,
  period numbers re-measured.

**Period predictor follow-up (2026-11-22, with the owner):**
- Kept at effort ≥ 6. The growth on exactly repeating groups is accepted and documented (ENCODER
  §1, §5.7.1; `test_period_predictor_growth_is_bounded`: under 0.1 bits/sample). A per-block repeat
  guard was rejected: one more tuned threshold, and it can't see repeats longer than a block, so
  the effort promise has to change anyway. "Try both and keep the smaller" was rejected by the
  owner (such comparisons multiply).
- Its parameters moved from `grid_scale`'s high half (4 extra bytes on every block) to their own
  3-byte `period` column after `value_anchor`: period length (uint16) and period fraction (int8).
  Decoded values identical on 67 cases (corpus at efforts 4 and 6, the test signals); sizes
  +0.01% in total, each within ±8 bytes per group. The high half had cost 0.2% on sparse daily
  groups and nothing measurable on dense ones. Docs and code name them period length and period
  fraction (P and w before).
- Decode: the loop now runs one period length of samples at a time, with no dependency between
  iterations: 36–43% faster than the single loop (`period_decode.py`, battery power, both in one
  process), and motor current's period blocks decode 8% faster than the same blocks without the
  predictor (they integrate only the head). It is not vectorized: the only form LLVM vectorized
  (predictions into a scratch array, then added: 4 × i32 lanes) was twice as slow on runs of a
  period's length (`period_simd.py`). The owner wanted SIMD if it paid; it doesn't here.

**Speed with the predictor on (2026-11-22, `period_speed.py`: effort 5 against effort 6, which
differ only by the predictor; interleaved in one process, battery power):** encode +14% over the
corpus: +35% on the 6 signals that use it (up to +65% on a clean 50.3 Hz sine), +10% on the 41 that
never take it (0 to +36%: oscillating blocks that pass the crossing check run the whole search and
then fail the gain test). Decode unchanged (−1% on the signals using it, 0 elsewhere). ENCODER's
"about +5% on a non-periodic block" understates it.

**Put on ice (2026-11-22, the owner):** the predictor lives on this branch, `period-predictor`
(from value-exp at ae1738d); value-exp itself no longer has it. To pick it up again: the gain is
−15% motor current, −18% line voltage, nothing elsewhere; the costs are above; open ideas are a
cheaper early reject for oscillating blocks it won't take, and the free-gain 2-tap lead on the
accelerometers (3.4).

**Script index** (timing microbenchmarks not cited above): `kernels.py`, `kernels2.py` (noise
kernel variants and the fused checks, the per-stage µs of 3.1), `time_noise.py` (noise estimate
cost per block), `ptime.py` (the period search, 3.4).

**Open, for the owner:**
- The period predictor at its re-measured value (motor current −15%, line voltage −18%, fast set
  −3.7%, nothing else) against its costs (the last block flag bit, grid_scale widened to 8 bytes,
  sequential decode of period blocks, a Rust interface bump), and, if kept, effort 5 → 9 growth
  on clean repeating signals (try both residual versions, or drop the guarantee).
- The two noise paths: a block with one dropout takes `noise_weighted`, which has the lag-2 check
  and the cap but not the long-lag test or the lower bands, so a dropout can change whether a
  block is floored (unify, or document).
- Held-out data: the estimator's guards were tuned on the corpus they are measured on.
- The prototype's free-gain 2-tap predictor on the accelerometers (−9%, 3.4).

**Sinusoidal baseline (2026-11-22, `sinusoid.py`, offline prototype).** Asked by the owner as an
alternative to the period predictor: per block, least squares fit of c + K harmonics of one
frequency (refined around the crossing estimate; optionally amplitudes linear in time) to the
block's indices q; residuals q − round(fit) at order 0, side info charged (32 bits for the
frequency, 24 per coefficient), taken per block on a Gaussian estimate. The grid is unchanged (the
baseline is subtracted in index space), so decoded values are the same. Against fixed orders,
effort 4 back end:

| signal | shipped period | K = 1 | K = 3 | K = 7 | K = 7, amplitudes linear in time |
|---|---|---|---|---|---|
| motor current | −15.2% | 0 | 0 | −14.0% | −13.3% |
| line voltage | −18.7% | 0 | 0 | −20.4% | −25.3% |
| sin-50.3hz (bench) | −40.1% | −32.6% | −30.9% | −27.9% | −82.8% |
| noisy-sine, accelerometer | 0 | 0 | 0 | 0 | 0 |

One or three harmonics never pay on the realistic signals (their 5th and 7th harmonics, load
changes within a block); seven do, about as well as the period predictor on motor current and
better on line voltage. Its decode has no dependency between samples (order 0, each fit value
from n alone), but needs a deterministic integer evaluation of up to 30 terms per sample and a
variable-size per-block parameter field. Dropped by the owner (2026-11-22): SIMD is the
parallelism wanted, which the period predictor allows (lag of a period), and its +20–30% decode
time on period blocks is acceptable.

## Appendix: final fuzz run

`tests/test_fuzz.py` extended (finite glitches −9999, ±1e6, 32767 in 15% of value cases; a
`tolerance=0.01` params entry in 10%; both drawn after the existing draws, so earlier regression
cases keep their inputs). First round, 20 seeds × 1500 cases per fuzzer (values, timed, decimal):

| finding | cases | cause | fix |
|---|---|---|---|
| **hang** (encode or re-encode never returns) | 9 of 20 value seeds | (1) a NaN after a glitch is held at the glitch's value (`fill_nonfinite`); escape detection skipped coded samples, so 1e6 stayed in a bulk at 1e−30. (2) The rational search's real Euclid (`_rational._approx_gcd`) assumes each remainder is at most half the divisor; past 2^53 multiples rounding breaks that and it cycled between 1e6 and 1e−35 forever | escaped blocks re-hold every coded sample from the bulk; the Euclid stops (no grid) when a remainder exceeds half the divisor |
| **wrapped indices on re-encode** (decoded data re-encoded off by 2^20) | 3 (binary grids, glitches not escaped) | `keep_floor_grid` (a value-exp addition for fixed points) went one level finer than the block's range fits in 16 bits when the noise floor sat at the range's own step; the indices wrapped. Reachable without glitches: any block whose floor equals its range's step (a large ramp with tiny noise) | never finer than `range_exponent(min, max, max_bits)` |
| **not a fixed point** (decoded data re-encodes to other bytes, same values) | 12 (tolerance, blocks of 9 to 2500) | tolerance's step N·2^(e−4) was found again on re-encode by the rational search, as an equal decimal-form step (125/64·10^−2 = 20·2^−10), and the grid was anchored at the minimum's fine index | a rational grid is taken over tolerance's only if strictly coarser; tolerance's grid is absolute (multiples of the step) |
| target moved | 10 | the soft per-group target (ENCODER §5.8), documented | none (counted, not a failure) |

Timed and decimal fuzzers: no failures. The failing cases are pinned in `REGRESSIONS`.

Second round on the fixes, same 90,000 cases: **0 failures, 0 hangs** (10 target moved). Tests:
1871 pass; golden hashes unchanged; the realistic corpus, the glitch scenarios and the tolerance
sweep are byte-for-byte or number-for-number unchanged.

**For the production review:** the hang and the wrap were real bugs in value-exp code paths
(escapes, keep_floor_grid) that the earlier fuzzer couldn't reach. A longer fuzz campaign (more
seeds, the timed fuzzer with glitches and tolerance too, updates on escaped groups) is
recommended before merging.
