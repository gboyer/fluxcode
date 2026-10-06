# fluxcode — specification

<!-- FIXME: The specification needs to focus only on the file layout and best-case error guarantees. Create a separate doc that talks about the encoder. -->

<!-- FIXME: this long string of semicolon separated clauses is very hard to read. Use bullets, perhaps nested. Apply this stylistic change everywhere; bullets, not proose blocks. -->

<!-- LARGER CHANGE (after finishing the doc changes): Rename "unit" everywhere to "block group" (in prose) or "group" (when a shorter name is needed for code). In science and engineering, "unit" is overloaded as units of measurement and units of large industrial facilities. -->

Lossy (bounded-error) and, for decimal data, lossless compression of float64 time series,
designed around 1 kHz sensor data, with their timestamps stored exactly if given (§4). One
compressed unit holds blocks of 0 to 65,535 samples each; the unit records every block's size.
`encode` divides a series into units of 60 blocks of 1000 by default, one channel-minute at
1 kHz. Implementation: the `fluxcode` package (encoding: `encode_unit`, `encode_blocks`,
`encode_time_blocks`, bulk `encode`; decoding: `decode_unit`, bulk `decode`; updating: `update`,
`update_time_blocks`; and `Params`).

This document covers the format, the encoding algorithm and its guarantees. Speed and
implementation notes are in [PERFORMANCE.md](PERFORMANCE.md); measured noise-floor behaviour and
what to check on your data are in [TUNING.md](TUNING.md). The experiments behind the design
choices are in [REPORT.md §11](../experimental/REPORT.md#11-power-of-two-quantization-the-fluxcode-design) of the research in `experimental/`.

## 1. Parameters

All parameters are **encoder-only**. The unit is self-describing: its header records the block
count, the sample count, the time unit and the residual layout, and each block records its own
size, order, grid (power-of-two exponent or decimal step) and anchor, and its start time, time
step and reference, so a decoder needs only the unit. Timestamps are data, not parameters (§4),
and so is the division into blocks (§2).

| parameter | values | default | effect |
|---|---|---|---|
| `max_quantize_bits` | min–16 | 16 | hard: the finest step. The block's snapped power-of-two grid spans at most 2^max steps (65,535 at 16, §5.1; `B` in the formulas below) |
| `min_quantize_bits` | 1–max | 6 | hard: the coarsest step. Neither the noise floor nor the target coarsens a block past 2^min steps across its range |
| `diff_orders` | non-empty subset of {0, 1, 2, 3} | {0, 1, 2, 3} | predictor orders the encoder may choose from (§5.5) |
| `decimal_detection` | on / off | on | try a decimal grid (10^p) before the power-of-two grid |
| `noise_floor_sigma` | off, or f > 0; 0.1–0.5 recommended | 0.25 | noise floor: on blocks whose residual looks like white measurement noise, coarsen the step to at most f·σ for the whole block (§5.2; measured behaviour in TUNING.md). Turn off per tag where high-frequency content matters (vibration, harmonics) |
| `target_bits_per_sample` | off, or ≥ 6 | off | soft per-unit cap on the size (§5.7): a guard against unexpectedly high usage, not a way to squeeze signals whose shape you don't know |
| `effort` | 1–9 | 4 | encoder effort: how the body is compressed (below). It changes the size and the encode time, never the decoded values |

**Precedence.** e_fine is the §5.1 exponent at `max_quantize_bits`, e_coarse the one at
`min_quantize_bits`. A block starts at e_fine; the noise floor raises it on gated blocks; it is
clamped to e_coarse; decimal detection then looks for a decimal grid coarser than that step;
finally the target (if set and the unit is over budget) coarsens blocks further, still never past
e_coarse. With both the noise floor and the target on, each block takes the coarser step.

**Effort.** Each effort picks the residual layout (§7: 16 bit planes or 2 byte planes, the header
records which), whether the zstd frame ends a block after the per-block columns and after each
dense residual plane (more than 1/16 of its bytes non-zero; every zstd block has its own literal
Huffman table, and the frame is an ordinary one either way), and the zstd levels:

| effort | layout | blocks | zstd |
|---|---|---|---|
| 1 | heuristic | one run | 1 |
| 2 | heuristic | one run | 3 |
| 3–4 | best | one run | 3 |
| 5–8 | best | flushed | 3 |
| 9 | best | flushed | 3 and 9 |

<!--
  -- FIXME: this talks about "before", but this is an unpublished spec. Everything in
  -- this spec should focus on current-only.
  -->

*Heuristic*: byte planes if fewer than 1% of the unit's residuals (zigzagged) reach 128, else
bit planes, one compression. *Best*: both layouts, the smaller kept, ties to byte planes (they
decode faster). Efforts 3 and 4 are the encoder before efforts existed: their units are
byte-identical to it. Effort 9 compresses at both zstd levels and keeps the smaller (zstd 9
alone is larger on about a fifth of units), so no unit grows from effort 5 to 9; the other
steps are ordered only on average (a flushed frame is larger than one run on tiny units, and
effort 1 uses another zstd level). Efforts sharing settings leave room for later strategies;
their output may change when they get one. Measured size and speed per effort are in TUNING.md.

Fixed by this spec: the body stored field by field, each field holding every
block's bytes in block order (§7). Every block records its size, 0 to 65,535 samples (1000 in
most measurements), and a unit holds up to 65,535 blocks. Bit planes store each block in whole
bytes, so a block whose size isn't a multiple of 8 pads its last byte with zero bits.

## 2. Units and blocks

**A unit is one storage artifact** (e.g. a database row): blocks encoded and decoded together.
Units are independent. **Blocks have any size from 0 to 65,535 samples**, each recorded in the
unit, so a series can be divided in whatever way suits it:
- **Fixed size** (`encode_unit`, `encode`): dense, regularly sampled data in blocks of
  `block_len` samples, the last block holding the rest. The bulk `encode` splits a long series
  into units of `blocks_per_unit` blocks (60 of 1000 by default).
- **Explicit sizes** (`encode_blocks`): one flat array of samples and each block's size.
- **Fixed duration** (`encode_time_blocks`): block b holds the samples timed in
  `[start + b·duration, start + (b+1)·duration)`, for irregular data or data that arrives
  incomplete: a minute with no samples is an empty block, and `update_time_blocks` later
  replaces what the unit holds in given time ranges (§8). `start` and `duration` are the
  caller's (typically the start is part of the unit's storage key): the unit doesn't store them.

**Decoding needs only the unit.** The header records the block count and the sample count, the
body each block's size (§7), and each block its own anchor, so no index column or length has to
be kept beside it. Nothing is padded: a block holds exactly its samples.

**Empty and short blocks.** An empty block stores no samples: its flags, grid parameter and anchor
(and time columns) are 0, and its summary statistics NaN. A block of at most 8 samples isn't
analyzed: it takes the finest step (e_fine, §5.1), order 0 (the quantized values themselves), no
noise floor and no part in the target (§5.7). Decimal detection (§5.3) still runs, against the
finest step: it needs no analysis, so short blocks of decimal data stay bit-exact. Non-finite
values in them are still coded exactly.

Per block the encoder also returns `lo = min(x)`, `hi = max(x)` and the mean (over the finite
samples) as **summary statistics**. They come free with encoding (min and max size the
grid; the sum is the non-finite check), and the caller may store them as index columns, but
decoding never uses them.

`num_samples` is at most 2^26: not a format limit (the header field holds up to 2^32 − 1), but the
bound decoders use to reject implausible headers before decompressing.

**Any finite block encodes**, from ranges of a few subnormals (2^−1074) to ranges past 2^1023,
where `hi − lo` itself overflows (§5.1, §5.4, §6). Each extreme costs one decision per block;
ordinary blocks take the plain formulas.

## 3. Non-finite values

NaN, +inf and −inf are encoded exactly. A block holding any sets block flag bit 3 and stores a
2-bit code per sample in the `nonfinite_code_planes` field (§7): 00 finite, 01 NaN, 10 +inf,
11 −inf.
- **Held values.** Before the block goes through §5, each non-finite sample is replaced by the
  previous finite value (the first finite value, for a leading run). The analysis stages never see
  NaN or inf. A held run is a zero residual under order 1. Under orders 2 and 3 its start and end
  each leave a residual or two (the slope changes); the order pick isn't made aware of that.
- **Summary statistics** cover the finite samples. A block with none (all NaN, all +inf, all
  −inf, or a mix) gets NaN for min, max and mean, and encodes as a constant 0 (anchor 0.0) that
  the codes then overwrite. A −0.0 minimum, maximum or mean is reported as +0.0, matching what
  decodes.
- **Noise floor from runs of finite samples.** Held runs would bias the noise estimate, so a
  flagged block uses only the second differences whose three samples are all finite, and in ρ
  only pairs of consecutive such differences (each sum normalized by its own count). Joining the
  finite samples across gaps instead would make a regular gap pattern (every third sample
  missing, say) alternate the time step and look like white noise. The estimate needs at least
  n/2 such differences (500 at n = 1000): below that a random walk starts passing the gate, so
  the block keeps its full precision. One NaN per block on the noisy test signals costs nothing
  (noisy-sine 5.42 → 5.41 bits/sample); skipping the noise floor there would cost 2.5×.
- **Cost:** a block with no non-finite values stores no codes and encodes exactly as it would
  without them. A flagged block adds n/4 bytes of mostly-zero planes before compression.

## 4. Time axis

A unit may also store the samples' **timestamps, exactly**. They are optional: a unit without them
has time unit 0 in its header and no time fields in its body, and decodes with `times = None`.

**Ticks and units.** Timestamps are int64 ticks since 1970-01-01 in one of the Arrow and numpy
datetime64 units, recorded in the header (§7). The unit only fixes the range and meaning of a
tick; storage cost doesn't depend on it (the per-block GCD below absorbs a coarser grid):

| time unit | code | int64 range around 1970 |
|---|---|---|
| s | 1 | ± 2.9 · 10^11 years |
| ms | 2 | ± 2.9 · 10^8 years |
| µs | 3 | ± 292,000 years |
| ns | 4 | ± 292 years (1677–2262) |

Timestamps are **naive**: the format stores no time zone. Store UTC: local time can repeat or skip
an hour, which breaks the ordering rule. int64 minimum (datetime64's NaT) is never a timestamp.

**Ordering.** The encoder requires the whole series to be **non-decreasing** and rejects any
decrease. Equal consecutive timestamps are allowed. The format itself enforces the order within a
block (deltas are unsigned) and between block starts (start increases are unsigned), so any
decoded unit has non-decreasing times within each block and non-decreasing block starts.

**Per block** (`time_start`, `time_step`, `time_ref`, `time_residual_planes` in §7): every block
stores its start, a step and a reference. The step is the GCD of the block's deltas
`time[i] − time[i−1]`, and each sample i > 0 has the **quotient** `(time[i] − time[i−1]) / time_step`,
which is `time_ref` plus the sample's residual:
- **Regular block:** every delta is equal (possibly 0). Every quotient is `time_ref` = 1 (0 when
  all times are equal, with `time_step` = 0), so `time[i] = time_start + i · time_step`, and the block
  stores no per-sample data.
- **Irregular block:** any other block. It stores each sample's residual `quotient − time_ref`
  (mod 2^64, zigzagged; 0 for sample 0) in 32 bit planes, or 64 if any residual is 2^32 or more
  (a **long** block, block flag bit 5). The encoder picks `time_ref` per block:
  the rounded mean of the quotients when they spread around a center (clock jitter), and their
  minimum when they are skewed (gaps, events, deadband logging), where the mean would sit away
  from most of them. With σ² the quotients' variance, it takes the mean when
  3σ² < (mean − min)²: zigzagged residuals from the mean are about 2σ, while those from the minimum
  are non-negative, so zigzag only shifts them up a bit plane (plane 0 all zero) and they are about
  √(σ² + (mean − min)²). The decoder doesn't depend on the choice: any `time_ref` decodes.
  If a single delta exceeds int64 maximum (a block spanning more than half the int64 range), the
  GCD can't be stored and the block uses `time_step = 1` with the raw deltas as quotients.
  Long blocks are rare: in ns ticks with a GCD of 1, a gap of about 2 s or more; in µs ticks or on
  any coarser grid, a gap of over half an hour.
- **Tiny blocks:** a block of one sample has no deltas: `time_step` and `time_ref` are 0. A block
  of two samples is always regular; if its one delta exceeds int64 maximum, it is stored as
  `time_ref` over a `time_step` of 1.
- **Empty blocks** have 0 in all three columns and take no part in the chain of start
  increases: each non-empty block's start is stored relative to the previous non-empty block's.

A regular series costs only its per-block start, step and reference (about 65 bytes per 60-block
unit), and a gap costs only its own block. Measured costs per clock shape and the measurements
behind these choices (the GCD, plain deltas, the reference, 32 planes) are in
[TUNING.md](TUNING.md#time-axis).

## 5. Encoding one block

### 5.1 Power-of-two exponent

```
L  = 2^B if B < 16 else 65535            # q is stored mod 2^16: it must stay below 2^16
e  = the finest exponent in [−1074, 1023] with  rint(hi · 2^-e) − rint(lo · 2^-e) <= L
     (0 for a constant block, rng = 0)
ps = 2^e                                 # the power-of-two step
```

rint rounds half to even. The grid is absolute (§5.4): the points nearest lo and hi can sit up
to half a step outside [lo, hi], so the snapped grid can need one more level than the range alone,
and the rule counts the levels it actually uses. To compute it, start from the unsnapped rule

```
k  = frexp(rng).exponent                 # rng = hi − lo in [2^(k-1), 2^k)
e0 = k − B;  if floor(ldexp(rng, −e0) + 0.5) >= 2^B:  e0 += 1
```

and adjust by at most one level: e0 − 1 fits only when the snapped grid needs exactly 2^B levels
(so only for B < 16); if e0 itself doesn't fit (65,536 levels at B = 16), use e0 + 1. Coarser
exponents always fit.

- **Never coarser than the unsnapped rule, except at the storage edge.** When the unsnapped rule
  accepts e (round(rng/2^e) ≤ 2^B − 1), the snapped grid needs at most 2^B levels. So the rule
  differs only by one level finer for B < 16, which fixes the fencepost case (a range of exactly
  a power of two, like [1, 2], gets all 2^B steps instead of half), and one level coarser for
  about 1 block in 65,000 at B = 16, whose range is within half a step of 65,535 steps.
- **It depends on where the block sits on the grid,** not just its range: two blocks with the
  same range at different offsets can get different steps. It is still a function of lo and hi.

- **Wide blocks** (rng ≥ 2^1023, possibly inf): compute e0 from `rh = hi/2 − lo/2` (halving is
  exact) as e0(rh) + 1. That is the same e0 the exact range gives. The snapped count never
  overflows: lo · 2^−e and hi · 2^−e are each finite.
- **The clamps.** At e = −1074 every double is on the grid, so a block whose finest step would be
  finer encodes losslessly. At e = 1023 (only reachable with few bits on a range near 2^1024),
  q stays below 3.

The encoder computes e_fine with B = `max_quantize_bits` and e_coarse with B = `min_quantize_bits`.

### 5.2 Noise floor

If `noise_floor_sigma` = f is set, rng > 0 and the block has at least 256 samples:

```
d     = x[i+2] − 2·x[i+1] + x[i]  for i = 0..n−3,  minus its mean   # second differences
mad   = mean |d|; twice: mad = mean of |d| over |d| <= 4·sqrt(π/2)·mad  # clipped re-estimate
sd    = sqrt(π/2)·mad;   σ = sd / sqrt(6)                             # white-noise level
ρ     = lag-1 autocorrelation of d clipped to ±4·sd
if ρ < −0.6:   e = max(e, floor(log2(f·σ)))
e = min(e, e_coarse)                                                # min_quantize_bits is hard
```

The estimator runs on x scaled by the power of two that brings the range into [½, 1). That's exact,
so σ and ρ are unchanged, but the differences and their squares can't overflow near 1e308 or
underflow near 1e−300; without it, noise at 1e300 would never gate and a block at 1e−300 would
divide by zero. If every clipped square underflows anyway (differences below ~1e−154 of the
range), the block isn't gated. For ranges beyond 2^±1000, the scaling itself (whose factor
would overflow) is applied in two steps, the first outside fastmath so LLVM can't fold them.

White measurement noise gives ρ = −2/3 on second differences. A random walk (whose increments
are signal) gives −1/2, and smooth or fast deterministic signals give positive values, so they
keep the B-bit step. The step stays a power of two, so edit stability and the format are
unchanged. The decoder doesn't know or care. Added error is at most f·σ/√12 RMS (half that on
average, because of the power-of-two floor) on top of noise σ.

- **f is effectively rounded down to a power of two** (relative to σ): the step is
  2^floor(log2(f·σ)), so f values within a factor of 2 can give the same step (0.3 behaves like
  0.25).
- **On gated blocks max_quantize_bits stops mattering** whenever f·σ exceeds the finest step,
  which it does for f ≥ 0.01 on the measured noisy signals at B = 16. f alone sets their size, down to the
  `min_quantize_bits` clamp: a noisy block's noise-floor step spans range/(f·σ) steps (≈ 230, about
  7.8 bits, on the noisy sine at f = 0.25), so a minimum above that would override the noise floor.
  Six bits costs ≤ 1% on the noisy test signals.
- **Blocks of at least 256 samples.** The ρ test separates white noise (−2/3) from a random walk
  (−1/2) only on enough differences. On m differences, the estimate of ρ has a standard error of
  about √((1 − 3ρ² + 4ρ⁴)/m) (Bartlett's formula for a lag-1 autocorrelation): 0.044 for a random
  walk and 0.042 for white noise at m = 254. The threshold −0.6 is then 2.25 standard errors from
  −1/2 (random walks pass about 1% of the time) and 1.6 from −2/3 (white noise is missed about 6%
  of the time). At 64 samples random walks would pass about 13% of the time. Measured: pure
  random walks pass 21% of the time at 9 samples, 9.4% at 64, 4.3% at 128, 0.8% at 256 and never
  at 1000, while white noise passes 96% of the time at 256. So 256 is a chosen trade-off (about
  1% false positives, 5% missed white noise), not a hard limit. Smaller blocks keep the B-bit step.
- **Re-encoding decoded data** (a merging `update_time_blocks`) computes the noise floor again
  from the merged block: kept samples, now decoded, plus the new ones. Nothing is reused, as when
  encoding from scratch. Quantizing adds white noise of variance s²/12 with s ≤ f·σ = σ/4, so σ
  can rise by at most √(1 + 1/192) − 1 ≈ 0.26%; the estimate also moves by a little either way.
  The step 2^floor(log2(f·σ)) only changes when f·σ sits within that much of a power of two, and
  then by one level, once. Measured over 5 decode/re-encode rounds of 300 noisy 1000-sample
  blocks (all gated): σ of the decoded data averaged 1.0002× the original (at most 1.008×); 8
  blocks went one level finer on the first round and none changed after. Finer is exact on the
  snapped grid (§5.4), and coarser is one bounded rounding. A merged block can also cross 256
  samples, which switches the noise floor on or off between updates, with the same two cases.
- **The gate is per block and all-or-nothing.** Per-sample variants (full precision kept at
  spikes) cost 20–35% more for an error already below the noise on the spike ([REPORT.md §11](../experimental/REPORT.md#11-power-of-two-quantization-the-fluxcode-design)).

### 5.3 Decimal detection

If `decimal_detection` is on and rng > 0. Runs after the noise floor, with `ps` the (possibly
coarsened) step. A decimal grid finer than `ps` is never used: when exact decimals would need
more than B bits, or are finer than the noise, the block takes the power-of-two grid.

Find the coarsest decimal step 10^p that is coarser than `ps` and that every sample sits on:

```
tol = ps / 4
for p = min(floor(log10(rng)), 22) down to max(the smallest p with 10^p > ps, −22):
    s = 10^-p                                        # exact integer when p <= 0
    if all |x[i]·s − floor(x[i]·s + 0.5)| <= tol·s  (for every i)
       and max(|lo|, |hi|)·s < 2^52:
        use decimal step p; stop
    (a failing candidate stops at its first off-grid sample)
```

- **Absolute grid.** Values must sit on multiples of 10^p itself, not merely be spaced 10^p apart.
  That is what makes reconstruction bit-exact (§6).
- **Tolerance** is a quarter power-of-two step. The decimal grid is only chosen when every sample
  is within that of it, so decimal mode's max error is never larger than power-of-two mode's
  at the same step.
- **Exact powers of ten only:** −22 ≤ p ≤ 22, so 10^|p| is an exact double.
- **Worst-case cost.** A failing candidate normally stops at its first off-grid sample. A block that
  sits on a decimal grid except for one sample near its end defeats that: every candidate from the
  coarsest down scans almost the whole block: encode up to ~20% slower (PERFORMANCE.md).
  Real data rarely looks like this, so it's noted rather than guarded against. If it shows up,
  probing a few spread-out samples before each full scan would bound it.
- **Not detected:** other steps (ADC counts in engineering units, 0.5, 0.25, 2⁻ᵏ). They use the
  power-of-two grid.

### 5.4 Quantize

```
power of two:  K    = rint(lo · 2^-e)              anchor = K · 2^e
               q[i] = rint(x[i] · 2^-e) - K        (rint: round half to even)
decimal:       K0   = floor(lo · 10^-p + 0.5)
               q[i] = floor(x[i] · 10^-p + 0.5) - K0
```

**The power-of-two grid is absolute:** its points are the multiples of 2^e, and the anchor is the
one nearest the block's minimum. Every power-of-two grid is then a subset of the finer ones, so
re-encoding decoded values on the same or a finer step returns them bit for bit, wherever the
block sits and however its minimum moved. That bounds the error of repeated updates (§8).

- **One rounding for K and q.** The minimum maps to exactly K, and rounding is monotonic, so q = 0
  at the minimum and q ≥ 0 everywhere. With different roundings for the two, a minimum exactly
  halfway between grid points would get q = −1, which wraps to 65,535 and decodes as the maximum.
- **Ties to even.** Raw data almost never has exact ties, but decoded data re-encoded one level
  coarser does: half its values sit halfway between the coarser grid's points. Rounding them all
  up would shift the block up by a quarter of the finer step on average.
- **Exact.** x · 2^−e is a power-of-two scale: it can't overflow for a sample of a block whose
  step is 2^e, and where it underflows the index is 0 either way. For e < −1023, where 2^−e
  overflows, scale by 2^1023 and then by 2^(−e−1023), both exact. If |x · 2^−e| ≥ 2^52, x is
  already a multiple of 2^e and rint leaves it alone. The difference of two nearby integers held
  in floats is exact.
- **Exceptions: anchor = lo,** q[i] = floor((x[i] − lo) · 2^−e + 0.5):
  - a constant block (rng = 0): e = 0 is only a placeholder there, and snapping would round the
    value to a whole number;
  - a snapped anchor that isn't finite (K · 2^e beyond ±DBL_MAX). There e ≥ 1008, and x − lo may
    overflow, so it is evaluated at half scale, `floor((x/2 − lo/2) · 2^(1−e) + 0.5)` (halving is
    exact, so x/2 − lo/2 rounds exactly as (x − lo)/2).

Decimal detection is skipped for wide blocks: no 10^p grid with p ≤ 22 spans 2^1023. Decimal
grids already share one rounding between K0 and q, and can't have ties: detection only accepts
samples within a quarter step of the grid.

In both modes 0 ≤ q[i] ≤ L ≤ 65,535 (§5.1), and q = 0 at the minimum.
When decimal detection succeeds, its candidate pass has already computed these q; they are reused.

**The block's anchor** is what a decoder adds q to: a finite float64 on the power-of-two grid
(the reference encoder snaps it to the grid point nearest the minimum; decoders accept any finite
anchor), or the integer `K0` on a decimal grid (|K0| < 2^52, since decimal detection requires
max(|lo|, |hi|)·10^−p < 2^52). Both define the block's values exactly (anchor + q·2^e is a binary
fraction, (K0 + q)·10^p a decimal); float64 enters only at the final rounding.

**Snapping is encoder behaviour, not a format rule.** The decoder doesn't check that an anchor is
snapped, as it doesn't check the exponent or the difference order: an anchor carries meaning, and
any finite one decodes correctly. An unsnapped anchor from another writer costs at most half a
step on the first straddling update, and the bound of §8 holds from then on.

### 5.5 Order

For each k in `diff_orders`, compute the variance of `diff(q[0..M−1], k)` over the first
M = min(250, n) samples (M − k values, no padding):

```
v_k = (m·Σd² − (Σd)²) / m²,   d = diff(q[:M], k),   m = M − k
order = the k with the smallest v_k; ties go to the lower order
```

Sums are exact integers (int64). The order only has to be *a* valid choice: the decoder reads it
from the unit, so encoders may pick it differently. Picking on the first 250 samples agrees with
the full-block pick on 97% of blocks and costs 0.4% in size on continuous data (nothing on
discretized data).

### 5.6 Residuals

```
r = k-th difference of (0, …, 0, q[0], …, q[n-1])  with k = order zeros prepended; keep the last n
    # order 1: r[0] = q[0],  r[i] = q[i] − q[i−1]
    # order 2: r[0] = q[0],  r[1] = q[1] − 2q[0],  r[i] = q[i] − 2q[i−1] + q[i−2]
    # order 3: r[0] = q[0],  r[1] = q[1] − 3q[0],  r[2] = q[2] − 3q[1] + 3q[0],
    #          r[i] = q[i] − 3q[i−1] + 3q[i−2] − q[i−3]
v = ((r + 32768) mod 2^16) − 32768          # r mod 2^16, read as int16
u = ((v << 1) XOR (v >> 15)) & 0xFFFF       # zigzag -> uint16
```

Order-k differences of 16-bit values need up to 16 + k + 1 bits. But every q is in [0, 2^16),
so integrating mod 2^16 (§6) recovers q exactly. Two bytes per sample always suffice. The
prepended zeros make the first residuals the start values, so there's no separate header.

### 5.7 Per-unit target

If `target_bits_per_sample` = t is set. After every block of the unit has been through §5.1–5.6,
estimate each block's size from its residual, the **class entropy**: with L(u) the bit length of
the zigzagged residual (0 for 0),

```
h_b = −Σ_L p_L·log2 p_L  +  Σ_L p_L·max(L − 1, 0)      # entropy of L, plus the bits below the leading 1
```

The budget covers the analyzed blocks (more than 8 samples), each weighted by its size n_b, with
N their total samples. If Σ_b n_b·h_b ≤ t·N, nothing changes. Otherwise:

```
e_b     = the block's exponent (for a decimal block, floor(log2 10^p): its grid is already that coarse)
give_b  = min(round(max(h_b − 1, 0)), e_coarse_b − e_b)        # whole bits block b can give
k       = the smallest k ≥ 1 with Σ_b n_b·min(k, give_b) ≥ Σ_b n_b·h_b − t·N   (k ≤ 16)
k_b     = min(k, give_b)
blocks with k_b > 0: redo §5.3–5.6 at e_b + k_b; keep the result only if its h_b went down
```

- Coarsening a block by one bit saves about one bit per sample while its estimate is above ~1
  bit/sample, and little after that. So expensive blocks end up at equal precision, and cheap
  blocks (which would save almost nothing) aren't touched.
- The "keep only if h went down" check stops the cap from trading a decimal grid for a coarser
  power-of-two grid that costs more (a 0.01-grid sine: 2.64 → 3.01 bits/sample without it).
- zstd never runs in the loop. Blocks the cap coarsens are quantized a second time; without a
  target every block is quantized once.
- **It's a soft cap, and t ≥ 6.** The estimate is within ~0.5 bits/sample on noise-like data and
  high on signals zstd compresses by repeats (chirp 11.9 vs 7.8, quadratic 5.5 vs 0.2). At small
  targets it coarsens blocks that were already cheap, and can make them larger (the quadratic:
  0.21 → 0.54 bits/sample at t = 2). The target is meant to cap unexpectedly high usage, so t is
  required to be at least 6, where the quadratic's estimate is under budget. Supporting lower
  targets would need a check of the actual compressed size (a TODO in the encoder). Validate on
  real data (`bench/estimate.py`).
- In `update` and `update_time_blocks`, the cap applies to the re-encoded blocks only, with a
  budget of t × their samples.

## 6. Decoding one block

```
v = (u >> 1) XOR −(u & 1)                   # un-zigzag
repeat `order` times:  prefix-sum v, each partial sum mod 2^16     # q
order 0: q = v mod 2^16
power of two: y[i] = a + q[i] · 2^e, rounded once, clamped to the largest finite double   (a: the anchor)
decimal:      p < 0:  y[i] = float64(K0 + q[i]) / 10^-p   (K0: the anchor; integer divided by an exact power of ten)
              p >= 0: y[i] = float64(K0 + q[i]) · 10^p
```

Because IEEE division is correctly rounded, the decimal reconstruction is the same double that
parsing the decimal text (e.g. "12.345") produces.

**The clamp** only matters for e ≥ 971. There, a sample near the largest double can reconstruct
up to half a step above it, which would round to inf; clamping it moves it closer to the original.
Below 971, `a + q·2^e` can't overflow (the excess is under half an ulp of the largest double),
and decoders compute it directly. From 971 up, `q·2^e` itself can overflow while the sum doesn't,
so evaluate at half scale: `y = min(2·(a/2 + q·2^(e−1)), DBL_MAX)`, with `y = a` where q = 0
(a/2 rounds if a is subnormal). That is bit-identical to `a + q·2^e` wherever the latter is finite.

**Time axis** (header time unit ≠ 0): block starts are the running sum of the `time_start` field
(§7). Each sample i > 0 has `quotient[i] = time_ref + unzigzag(residual[i])` mod 2^64 (residuals
are 0 in a regular block, block flag bit 4 clear), and
`time[i] = start + time_step · (quotient[1] + … + quotient[i])`. A regular block is therefore
`time[i] = start + i · time_ref · time_step`. All arithmetic is exact in 64 bits; a decoder rejects
a unit whose times would exceed int64 maximum (§7).

**Non-finite codes** (block flag bit 3): after dequantizing, samples with code 01, 10 or 11 are
overwritten with NaN (the canonical quiet NaN), +inf or −inf. Groups of 8 samples whose two code
bytes are both 0 are skipped.

## 7. Unit format

A unit is an 8-byte header followed by one zstd frame holding the body (any zstd level and block
boundaries, content size recorded, no checksum). The header is uncompressed, so a decoder can validate it before
decompressing:

| offset | header field | type (little-endian) | contents |
|---|---|---|---|
| 0 | version | uint8 | 1 (a decoder rejects others) |
| 1 | flags | uint8 | bit 0: `residual_planes` are byte planes instead of bit planes; bits 1–3: `time_unit`, 0 for no time axis, 1 s, 2 ms, 3 µs, 4 ns (5–7 rejected); bits 4–7: 0 (rejected otherwise) |
| 2–3 | `num_blocks` | uint16 | block count, 0 to 65,535 |
| 4–7 | `num_samples` | uint32 | sample count, 0 to 2^26 (§2): the sum of the block sizes |

Each block b of `n_b` samples takes `g_b` = ceil(`n_b` / 8) bytes of every bit plane (none for an
empty block): when `n_b` isn't a multiple of 8, bits `n_b` mod 8 to 7 of its last byte are padding,
and byte planes likewise pad each block to 8 × `g_b` bytes. Writers zero the padding; decoders
ignore it. Within a plane the blocks' bytes follow each other in block order. Sums used below:
`G` = Σ `g_b` over all blocks, `G_nonfinite` over the blocks with block flag bit 3 set,
`G_irregular` over those with bit 4 and `G_long` over those with bit 5. The body is these fields,
in this order; the four time fields are present only when `time_unit` ≠ 0:

| field | size in bytes | contents |
|---|---|---|
| `block_flags` | `num_blocks` | bits 0–1: order; bit 2: decimal mode; bit 3: non-finite codes present (§3); bit 4: irregular times, with time residual planes (§4; rejected when `time_unit` = 0); bit 5: long time residuals, 64 planes instead of 32 (rejected without bit 4); bits 6–7: 0 (rejected otherwise). 0 for an empty block |
| `block_sizes` | 2 × `num_blocks` | each block's sample count as uint16, byte-planed: every low byte, then every high byte |
| `grid_params` | 2 × `num_blocks` | the block's grid parameter as int16: the power-of-two exponent (−1074 to 1023) or, in decimal mode, the decimal power (−22 to 22); 0 for an empty block. Byte-planed like `block_sizes` |
| `value_anchor` | 8 × `num_blocks` | the block's anchor (§5.4), byte-planed (byte 0 of every block, then byte 1, … byte 7): a finite float64 anchor (the reference encoder snaps the block minimum to the grid, §5.4; 0.0 for a block with no finite samples), or in decimal mode the int64 decimal grid index of the minimum (magnitude < 2^52); 0 for an empty block |
| `time_start` | 8 × `num_blocks` | the first non-empty block: its start time as int64; each later non-empty block: its start minus the previous non-empty block's start, as uint64; an empty block: 0. Byte-planed |
| `time_step` | 8 × `num_blocks` | the block's time step as int64, byte-planed: the GCD of its deltas (≥ 0; ≥ 1 in an irregular block; 0 in an empty block or one of a single sample) |
| `time_ref` | 8 × `num_blocks` | the block's reference quotient as uint64, byte-planed: 1 in a regular block (0 if its times are all equal or it has one sample or none; the delta itself in a 2-sample block whose delta exceeds int64 maximum, over a `time_step` of 1, §4); in an irregular block, the value the residuals are taken from (§4) |
| `residual_planes` | 16 × `G` | flags bit 0 clear: bit plane j = 0..15, then block, then byte i = 0..`g_b` − 1. Bit k of byte i (LSB = bit 0) is bit j of `u[8i + k]`. Flags bit 0 set: byte plane j = 0..1 (low, then high byte of `u`), then block, then sample 0..8 × `g_b` − 1 |
| `nonfinite_code_planes` | 2 × `G_nonfinite` | code plane j = 0..1, then the flagged blocks in block order, then byte i. Bit k of byte i is bit j of the code of sample 8i + k |
| `time_residual_planes` | 32 × (`G_irregular` + `G_long`) | per sample of an irregular block, the uint64 residual: `zigzag(quotient − time_ref)` mod 2^64, with the quotient `(time[i] − time[i−1]) / time_step`; 0 for sample 0. First bit plane j = 0..31, then the irregular blocks in block order, then byte i; then bit plane j = 32..63, then the long blocks in block order, then byte i. Bit k of byte i is bit j of the residual of sample 8i + k; a short block's residuals are below 2^32 |

```
body_size = 13 × num_blocks + 16 × G + 2 × G_nonfinite
          + (time_unit ≠ 0) × (24 × num_blocks + 32 × (G_irregular + G_long))
```

Repeated block sizes cost almost nothing: on 60 blocks of 1000, the `block_sizes` column adds
4–8 bytes after zstd.

A decoder checks the frame's recorded content size against the header before decompressing: it
must lie between the sizes with the fewest plane bytes (ceil(`num_samples` / 8) per plane, no
flags) and the most (up to 7 padding samples per block, every block flagged and long). Then it
checks that the block sizes add up to `num_samples`, the exact size once `block_flags` and
`block_sizes` give the sums, each block's flags, grid parameter and value anchor ranges (all 0 in
an empty block), and, for a time axis, rejects:
- a first non-empty start of int64 minimum, or a running sum of `time_start` above int64 maximum;
- an empty block with a nonzero `time_start`, `time_step` or `time_ref`, or a block of one sample
  with a nonzero `time_step` or `time_ref` or block flag bit 4;
- `time_step` < 0, or `time_step` = 0 on an irregular block;
- an irregular block whose first residual isn't 0, or a long block whose planes 32–63 are all zero
  (its residuals fit in a short block);
- any time above int64 maximum.

A unit with no flagged blocks has an empty `nonfinite_code_planes` field, and one with only
regular blocks an empty `time_residual_planes` field.

**Test vector for the bit order:** a block whose `u[3] = 0x0020` and `u[6] = 0x0400` (all others 0)
has byte 0 of plane 5 = `0b00001000` and byte 0 of plane 10 = `0b01000000`. All other plane bytes are 0.

**Worked example with a time axis** (`tests/test_time.py::test_worked_example_layout`):
20 samples in blocks of 8, so 3 blocks of 8, 8 and 4 samples; ns ticks (small, for readability);
`planes = "bit"` (with "best" the all-zero residuals tie and take byte planes, flags 0x09).

```
times   block 0: 1000 1010 1020 1030 1040 1050 1060 1070   regular
        block 1: 1080 1090 1100 1130 1140 1150 1160 1170   irregular: a gap after 1100
        block 2: 1180 1190 1200 1210                       regular
values  constant per block: 2.5, 2.25, 3.0   (power-of-two mode)
```

Header: `01 08 03 00 14 00 00 00` (flags 0x08: `time_unit` 4 in bits 1–3; 3 blocks; 20 samples).
Body: 191 bytes = 13 × 3 + 16 × 3 + 24 × 3 + 32 × 1.

```
offset   field                size   byte planes (3 bytes each: blocks 0, 1, 2)
0–2      block_flags          3      bit 4 set on block 1 only
3–8      block_sizes          6      low bytes 08 08 04, high bytes 00 00 00
9–14     grid_params          6      exponent 0 (constant blocks): all 00
15–38    value_anchor         24     raw bits 0x4004…, 0x4002…, 0x4008…: planes 0–5 = 00 00 00,
                                     plane 6 = 04 02 08, plane 7 = 40 40 40
39–62    time_start           24     stored 1000, 80, 100: plane 0 = E8 50 64, plane 1 = 03 00 00,
                                     planes 2–7 = 00 00 00
63–86    time_step            24     10, 10, 10: plane 0 = 0A 0A 0A, planes 1–7 = 00 00 00
87–110   time_ref             24     1, 1, 1: plane 0 = 01 01 01, planes 1–7 = 00 00 00
111–158  residual_planes      48     one byte per plane per block (block 2's last 4 bits padding)
159–190  time_residual_planes 32     block 1 (short: planes 0–31 only), quotients [1, 1, 3, 1, 1, 1, 1]
                                     from sample 1: minimum 1 (3σ² > (mean − min)²), residuals
                                     [0, 0, 0, 2, 0, 0, 0, 0], zigzagged [0, 0, 0, 4, 0, 0, 0, 0]:
                                     plane 2 = 0x08, all other planes 0x00
```

## 8. Guarantees

- **Max error ≤ half the step used = 2^(e−1)** (up to one rounding of the sample's magnitude).
  - **Always ≤ range / (2^min_quantize_bits − ½)**: range/63.5, 1.6% of the block's range, at the
    default of 6, whatever the noise floor or the target does. The ½ is from the exponent bump in §5.1.
  - Blocks at `max_quantize_bits` (neither noise floor nor target applied): ≤ range / (2^max − ½),
    about 2^−max of the block's range (0.0015% at 16).
  - Noise-floor blocks: ≤ f·σ/2, in the signal's units. This is not bounded relative to the
    range: a block that is all noise has a range of only a few σ.
  - Decimal mode is exact, or within `tol` = 2^e/4 if the samples were within tolerance of the
    grid.
- **One zero and one NaN.** −0.0 decodes as +0.0 (equal as floats, sign bit lost), and every NaN
  decodes as the canonical quiet NaN (payload and sign lost). These are the exceptions to
  bit-exactness for data on the grid. ±inf round-trip exactly.
- **min and max decode within half a step.** On the power-of-two grid the minimum decodes to the
  grid point nearest it (exactly when it is on the grid, as decoded data always is). In decimal
  mode, min decodes onto the decimal grid. That's exact for decimal data, but a min carrying float
  noise (e.g. 1000.0000000000291 on an integer grid) decodes to the grid value (1000.0): an error
  within `tol`, as for every sample. The block_min and block_max that encode returns are the
  samples' own; callers using them as hard bounds on decoded values (for pruning) should allow
  half a step either way.
- **Decimal data is bit-exact:** values that are decimals of p places (as parsed from text)
  decode to the identical float64. That holds when the block's range spans fewer than 2^B
  decimal steps; wider blocks use the power-of-two grid.
- **float32 rounding artifacts are intentionally lost.** A decimal stored as float32 upstream
  decodes as the decimal itself.
- **Decoded data is a fixed point:** decode(encode(y)) = y for any decoded y. The unit bytes are
  identical from the second encode on.
- **Timestamps are exact.** Given times decode to the identical int64 ticks, in the same unit.
  Decoded times are non-decreasing within every block, and block starts never decrease (§4).
- **Edits are stable.** Changing a sample leaves every other sample's q unchanged unless the step
  changes, which happens only when the range crosses a power of two: the grid is absolute, so a
  new minimum doesn't move it. (With the grid scaled to the range, a new max re-rounds every
  sample: in testing, 95% of untouched samples changed.)
- **`update` leaves other blocks untouched.** Replacing or appending whole blocks of a unit, of
  any size, carries the other blocks' flags, grid parameter, value anchor, residuals, codes and
  time rows over unchanged: they are neither dequantized nor re-encoded, and decode to identical
  values. Indices skipped past the unit's end are appended as empty blocks. With a time axis,
  `update` takes the new blocks' times (required exactly when the unit has one); the updated
  series must be non-decreasing throughout, which it checks where a new block meets its non-empty
  neighbours. Without a target, the updated unit is byte-identical to encoding the updated
  series from scratch (every block is encoded independently).
- **`update_time_blocks` is an upsert plus a range deletion, touching only the blocks it affects.**
  It first discards every sample timed in the optional delete ranges (`[start, end)` each), then
  adds the new samples, which may be timed anywhere (a new sample inside a range is kept). Every
  existing sample with the same timestamp as a new one is discarded too: the new sample replaces
  it. New samples sharing a timestamp are all kept, in their order. With no ranges it is an
  upsert; with no samples, a deletion. A block that no range meets and no new sample falls in is
  carried over as `update` carries it, without being decoded. A block wholly inside the ranges is
  encoded from the new samples alone. Any other affected block is decoded, keeps its surviving
  samples and is re-encoded with the new ones. The kept samples are already points of the
  absolute grid (§5.4), so on the same or a finer step they come back bit for bit, however the
  merged block's min and max moved. Only a
  coarser step rounds them again, once, without bias (ties to even); repeated coarsening adds a
  geometric series, so their error stays under one step of the coarsest grid the block has used.
  Decimal data on a decimal grid stays exact. New samples past the unit's end append blocks
  (empty ones to fill a gap); blocks are never removed, so a block emptied by an update stays,
  empty. The result is byte-identical to `encode_time_blocks` of the resulting series when no
  block is left empty at the end.

## 9. Conformance tests

1. **Round trip:** decode(encode(x)) has max error ≤ 2^(e−1) for every block (≤ f·σ/2 on
   noise-floor blocks) and ≤ range / (2^min_quantize_bits − ½) always; the decoded min is within
   half a step of the minimum.
2. **Decimal:** values generated as `K / 10^d` with a range under 2^B steps decode
   bit-identical.
3. **Fixed point:** y = decode(encode(x)) satisfies decode(encode(y)) = y,
   and encode(y) is byte-identical from the second encode on (§8).
4. **Bit order:** the test vector in §7.
5. **Order independence:** a unit written with any `diff_orders` setting decodes with the same decoder.
6. **Edge cases:** constant block (all q = 0, decodes to lo exactly, including non-integers and
   extreme magnitudes); range just below a power of two (exponent bump); a minimum exactly halfway
   between grid points (q = 0, not −1); [1, 2] at B = 12 (4,096 steps, grid-aligned data exact)
   and B = 16 (32,768 steps); the storage edge at B = 16 (a snap that would need q = 65,536 goes one
   level coarser); a block whose range exceeds 2^B decimal steps (falls back to the
   power-of-two grid); fewer than 60 blocks; a short last block; units of different block counts
   and block sizes; blocks of every size from 0 to 9 and over 1000 in one unit; units with no
   blocks or only empty ones; blocks of up to 8 samples at the finest step and order 0, whatever
   the parameters, on a decimal grid (bit-exact) when one fits and decimal detection is on.
   **Self-describing units:** decode_unit needs only the unit and returns exactly `num_samples`
   samples and each block's size; headers with another version, reserved flag bits, a time unit
   of 5–7, a sample count over 2^26 or than its blocks can hold, or a body size that doesn't fit
   them are rejected, as are block sizes that don't add up to the sample count, empty blocks with
   nonzero columns, non-finite float anchors and decimal anchors of 2^52 or more.
7. **Update:** untouched blocks decode identically and keep their rows; without a target,
   `update` (replacing blocks with ones of other sizes, emptying and appending past a gap) is
   byte-identical to encoding the updated blocks. `update_time_blocks` is byte-identical to
   `encode_time_blocks` of the expected series (old samples outside the ranges and not
   sharing a time with a new one, plus the new ones) for ranges filling empty blocks, covering
   whole blocks, straddling block edges, upserts with no ranges (including duplicate times in the
   new data and in the unit) and appending; it decodes only the blocks it merges and accepts ranges as one pair, a list, a
   `(k, 2)` array of datetime64 or ticks, and naive datetimes.
8. **Non-finite:** NaN / ±inf runs at a block's start, middle and end, scattered, alternating,
   and whole blocks, on every signal kind and with the target: exact NaN positions and infinities,
   finite samples within the bounds, summary statistics over the finite samples; blocks without
   non-finite values byte-identical to units without the field; `update` keeps existing codes.
9. **Limits:** min/max_quantize_bits are never violated, with or without the noise floor and target;
   the noise gate fires on white noise at 1e±300 and across ±DBL_MAX as at 1; ranges from a few
   subnormals to past 2^1023 round-trip within the bounds, with no inf, and exponents, anchors and
   q matching exact rational arithmetic.
10. **Noise gate:** it fires on white Gaussian noise (with or without a slow signal under it)
   and never on a random walk, a clean sine at any frequency, a chirp, a ramp or a square wave.
   It may fire on the float rounding noise of a smooth polynomial (the quadratic), with no
   effect, since that σ is far below the B-bit step. On all clean test signals the unit is
   byte-identical with the noise floor on (f = 0.01–1) and off.
11. **Time axis:** datetime64 in s, ms, µs and ns, and integer ticks with a time unit, round-trip
   exactly and in their unit; units without times decode `times = None`; regular series store no
   planes; a gap makes only its block irregular, with the GCD as its step; jitter takes the rounded
   mean as its reference and skewed deltas the minimum; any reference decodes; only a block whose
   residuals reach 2^32 is long; equal timestamps, all-equal blocks, short last blocks, blocks of
   0, 1 and 2 samples (a 2-sample block whose delta exceeds int64 maximum), leading empty blocks
   before negative ticks, ticks at both ends of int64 and a block spanning more than half of it
   round-trip; the worked example in §7 matches byte for byte. The encoder rejects decreasing
   times (within and across blocks), NaT, unsupported dtypes and units, and length mismatches;
   the decoder rejects each corrupt time field listed in §7, and the long flag without the
   irregular one. `update` with times is byte-identical to encoding the edited series, and
   rejects missing, unexpected, mis-shaped, wrong-unit or out-of-order times.
12. **Snapped grid** (`tests/test_grid.py`): the exponent rule picks at most one level finer than
   the unsnapped rule (only below 16 bits) and one coarser (only at 16), and its grid fits L;
   decoded values are fixed points block by block wherever the re-encode picks the same step,
   noisy, smooth, stepped, short and at extreme magnitudes; re-encoding decoded values one level
   coarser keeps their mean (ties to even); 200 straddling updates that move the block's min and
   max keep the kept samples' error under one step of the coarsest grid used, unchanged while the
   step is.

## References

Background for the techniques above (informative; the sections above are normative).

- Quantization: https://en.wikipedia.org/wiki/Quantization_(signal_processing)
- Fixed polynomial predictors (§5.5–5.6, §6): the fixed predictors of
  [Shorten](https://en.wikipedia.org/wiki/Shorten_(file_format)) and FLAC,
  [RFC 9639 §9.2.5](https://www.rfc-editor.org/rfc/rfc9639#name-fixed-predictor-subframe).
- Zigzag encoding of signed integers (§5.6):
  [Protocol Buffers encoding](https://protobuf.dev/programming-guides/encoding/#signed-ints).
- Bit-shuffle (§7): [bitshuffle](https://github.com/kiyo-masui/bitshuffle); K. Masui et al.,
  [arXiv:1503.00638](https://arxiv.org/abs/1503.00638).
- Zstandard (§7): [RFC 8878](https://www.rfc-editor.org/rfc/rfc8878).
- Noise estimate (§5.2): [mean absolute deviation](https://en.wikipedia.org/wiki/Average_absolute_deviation)
  and [autocorrelation](https://en.wikipedia.org/wiki/Autocorrelation).
- Decimal detection (§5.3) is related to ALP: A. Afroozeh, L. Kuffó, P. Boncz,
  [SIGMOD 2024](https://doi.org/10.1145/3626717).
- IEEE 754 binary64 and correctly rounded division (§6):
  [double-precision floating-point format](https://en.wikipedia.org/wiki/Double-precision_floating-point_format).
