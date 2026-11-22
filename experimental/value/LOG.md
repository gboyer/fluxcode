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
| 1 GiB benchmark, 15 signals | 4.76 bits/sample | 4.01 (−16%, mostly the random walks, now floored) |
| corpus fast (DAQ-like) | | −28% |
| corpus slow (1 s tags, hourly blocks) | | −13%; −28% more in 4-minute blocks |
| corpus synth | | −11% |
| sparse daily archives (tank level, valve) | 577 / 461 B/day | 473 / 355 (−18% / −23%) |
| glitched blocks (one −9999 or 1e6) | rest of block within 0.125 / 5 | within 0.000244 (as clean blocks) |
| encode, Python path | 5.12 µs/block | 5.71 (+12%) |
| decode | 1.79 µs/block | 1.75 (−2%) |

Full benchmark tables: `bench/RESULTS.md`. Tests: 1866 pass, Rust extension included.

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
- **Evidence.** `period.py`: at default effort it gave motor current −41%, line voltage −19%,
  fast −31% overall, but cost +30–90% encode on oscillating blocks; so it is gated to effort ≥ 6
  (effort never changes decoded values) and off with a target. Optimized search 8.5 → 3.5
  µs/block; at effort 6 motor current −15%, line voltage −18%, no other realistic signal changed.
  An exactly repeating synthetic sine can come out larger (zstd already matches its repeats).
- **Commits.** 1d24bcf, 133d23f, 4af70cf. SPEC §4 (block flag bit 7, grid_scale widened to
  int64), ENCODER §5.7.1, Rust interface version 5.
- **Review.** Decode is sequential for period blocks (+0.3–0.5 µs).

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
  `escapes` field (SPEC §2, §6); block flag bit 7 period and the int64 grid_scale (SPEC §4);
  header flag bit 4 and the trailing `time_quantum` (SPEC §3, §6). Golden hashes were
  regenerated for these.
- **Rust.** Interface version bumped for the period predictor; the Rust compress path declines
  groups with escapes (Python packs them); the Rust packer handles the time quantum.
- **verify.** Step bounds account for escapes and tolerance; decimal exactness is checked only
  where tolerance permits it.
- **Fuzzing.** `tests/test_fuzz.py` covers random magnitudes, grids, non-finite values and
  random clocks (whose periods exercise the time quantum). At the end of the work it was extended
  with finite glitches (−9999, ±1e6, 32767: escapes) and a `tolerance` params entry, drawn after
  the existing draws so recorded regression cases keep their inputs; its result is in the
  appendix below.
- **Performance.** Encode +12% (noise estimate), decode −2%; the period predictor adds encode
  time on oscillating blocks at effort ≥ 6 only.

## Appendix: final fuzz run

(Filled in when the run completes.)
