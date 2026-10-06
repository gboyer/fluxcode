# fluxcode — reference encoder

How the `fluxcode` package encodes: its parameters, how it divides a series into blocks, the
algorithm for one block, and the guarantees that follow. The format it writes is specified in
[SPEC.md](SPEC.md); any writer of valid units is a conforming encoder, and this one's choices
(the step, the anchor, the order, the time reference) are not format rules. Implementation: the
`fluxcode` package (encoding: `encode_unit`, `encode_blocks`, `encode_time_blocks`, bulk `encode`;
decoding: `decode_unit`, bulk `decode`; updating: `update`, `update_time_blocks`; and `Params`).

Speed and implementation notes are in [PERFORMANCE.md](PERFORMANCE.md); measured noise-floor
behaviour and what to check on your data are in [TUNING.md](TUNING.md). The experiments behind the
design choices are in [REPORT.md §11](../experimental/REPORT.md#11-power-of-two-quantization-the-fluxcode-design) of the research in `experimental/`.

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
| `noise_floor_sigma` | default, off (0), or f > 0; 0.1–0.5 recommended | default: 0.25 for a unit without times, off for a unit with times | noise floor: on blocks whose residual looks like white measurement noise, coarsen the step to at most f·σ for the whole block (§5.2; measured behaviour in TUNING.md). Turn off per tag where high-frequency content matters (vibration, harmonics). Off by default with times: timed data is often irregular (a swinging-door archive, events), and its samples look like white noise without being noise |
| `target_bits_per_sample` | off, or ≥ 6 | off | soft per-unit cap on the size (§5.7): a guard against unexpectedly high usage, not a way to squeeze signals whose shape you don't know |
| `effort` | 1–9 | 4 | encoder effort: how the body is compressed (below). It changes the size and the encode time, never the decoded values |

**Precedence.** e_fine is the §5.1 exponent at `max_quantize_bits`, e_coarse the one at
`min_quantize_bits`. A block starts at e_fine; the noise floor raises it on gated blocks; it is
clamped to e_coarse; decimal detection then looks for a decimal grid coarser than that step;
finally the target (if set and the unit is over budget) coarsens blocks further, still never past
e_coarse. With both the noise floor and the target on, each block takes the coarser step.

**Effort.** Each effort picks the residual layout (SPEC.md §6: 16 bit planes or 2 byte planes, the header
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

## 2. Dividing a series into blocks

Every block records its size (SPEC.md §1), so a series can be divided in whatever way suits it:
- **Fixed size** (`encode_unit`, `encode`): dense, regularly sampled data in blocks of
  `block_len` samples, the last block holding the rest. The bulk `encode` splits a long series
  into units of `blocks_per_unit` blocks (60 of 1000 by default).
- **Explicit sizes** (`encode_blocks`): one flat array of samples and each block's size.
- **Fixed duration** (`encode_time_blocks`): block b holds the samples timed in
  `[start + b·duration, start + (b+1)·duration)`, for irregular data or data that arrives
  incomplete: a minute with no samples is an empty block, and `update_time_blocks` later
  replaces what the unit holds in given time ranges (§6). `start` and `duration` are the
  caller's (typically the start is part of the unit's storage key): the unit doesn't store them.

**Short blocks.** A block of at most 8 samples isn't
analyzed: it takes the finest step (e_fine, §5.1), order 0 (the quantized values themselves), no
noise floor and no part in the target (§5.7). Decimal detection (§5.3) still runs, against the
finest step: it needs no analysis, so short blocks of decimal data stay bit-exact. Non-finite
values in them are still coded exactly.

Per block the encoder also returns `lo = min(x)`, `hi = max(x)` and the mean (over the finite
samples) as **summary statistics**. They come free with encoding (min and max size the
grid; the sum is the non-finite check), and the caller may store them as index columns, but
decoding never uses them. An empty block's summary statistics are NaN.

**Any finite block encodes**, from ranges of a few subnormals (2^−1074) to ranges past 2^1023,
where `hi − lo` itself overflows (§5.1, §5.4, SPEC.md §5). Each extreme costs one decision per block;
ordinary blocks take the plain formulas.

## 3. Non-finite values

The format stores a 2-bit code per non-finite sample (SPEC.md §2). The encoder:
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
- **Cost:** a block with no non-finite values encodes exactly as it would without them.

## 4. Time axis

The encoder requires the whole series to be **non-decreasing** and rejects any decrease. Equal
consecutive timestamps are allowed.

**The time reference.** For an irregular block (SPEC.md §3) the encoder picks `time_ref`:
  the rounded mean of the quotients when they spread around a center (clock jitter), and their
  minimum when they are skewed (gaps, events, deadband logging), where the mean would sit away
  from most of them. With σ² the quotients' variance, it takes the mean when
  3σ² < (mean − min)²: zigzagged residuals from the mean are about 2σ, while those from the minimum
  are non-negative, so zigzag only shifts them up a bit plane (plane 0 all zero) and they are about
  √(σ² + (mean − min)²). The decoder doesn't depend on the choice: any `time_ref` decodes.

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

If the noise floor f is on (`noise_floor_sigma` > 0, or the default on a unit without times: f = 0.25),
rng > 0 and the block has at least 256 samples:

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
  That is what makes reconstruction bit-exact (SPEC.md §5).
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
block sits and however its minimum moved. That bounds the error of repeated updates (§6).

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

The residuals of the order picked in §5.5, mod 2^16 and zigzagged, as specified in SPEC.md §4.

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

## 6. Guarantees

- **Max error ≤ half the step used = 2^(e−1)** (up to one rounding of the sample's magnitude).
  - **Always ≤ range / (2^min_quantize_bits − ½)**: range/63.5, 1.6% of the block's range, at the
    default of 6, whatever the noise floor or the target does. The ½ is from the exponent bump in §5.1.
  - Blocks at `max_quantize_bits` (neither noise floor nor target applied): ≤ range / (2^max − ½),
    about 2^−max of the block's range (0.0015% at 16).
  - Noise-floor blocks: ≤ f·σ/2, in the signal's units. This is not bounded relative to the
    range: a block that is all noise has a range of only a few σ.
  - Decimal mode is exact, or within `tol` = 2^e/4 if the samples were within tolerance of the
    grid.
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

## 7. Conformance tests

1. **Round trip:** decode(encode(x)) has max error ≤ 2^(e−1) for every block (≤ f·σ/2 on
   noise-floor blocks) and ≤ range / (2^min_quantize_bits − ½) always; the decoded min is within
   half a step of the minimum.
2. **Decimal:** values generated as `K / 10^d` with a range under 2^B steps decode
   bit-identical.
3. **Fixed point:** y = decode(encode(x)) satisfies decode(encode(y)) = y,
   and encode(y) is byte-identical from the second encode on (§6).
4. **Edge cases:** constant block (all q = 0, decodes to lo exactly, including non-integers and
   extreme magnitudes); range just below a power of two (exponent bump); a minimum exactly halfway
   between grid points (q = 0, not −1); [1, 2] at B = 12 (4,096 steps, grid-aligned data exact)
   and B = 16 (32,768 steps); the storage edge at B = 16 (a snap that would need q = 65,536 goes one
   level coarser); a block whose range exceeds 2^B decimal steps (falls back to the
   power-of-two grid); fewer than 60 blocks; a short last block; units of different block counts
   and block sizes; blocks of every size from 0 to 9 and over 1000 in one unit; units with no
   blocks or only empty ones; blocks of up to 8 samples at the finest step and order 0, whatever
   the parameters, on a decimal grid (bit-exact) when one fits and decimal detection is on.
5. **Update:** untouched blocks decode identically and keep their rows; without a target,
   `update` (replacing blocks with ones of other sizes, emptying and appending past a gap) is
   byte-identical to encoding the updated blocks. `update_time_blocks` is byte-identical to
   `encode_time_blocks` of the expected series (old samples outside the ranges and not
   sharing a time with a new one, plus the new ones) for ranges filling empty blocks, covering
   whole blocks, straddling block edges, upserts with no ranges (including duplicate times in the
   new data and in the unit) and appending; it decodes only the blocks it merges and accepts ranges as one pair, a list, a
   `(k, 2)` array of datetime64 or ticks, and naive datetimes.
6. **Non-finite:** NaN / ±inf runs at a block's start, middle and end, scattered, alternating,
   and whole blocks, on every signal kind and with the target: exact NaN positions and infinities,
   finite samples within the bounds, summary statistics over the finite samples; blocks without
   non-finite values byte-identical to units without the field; `update` keeps existing codes.
7. **Limits:** min/max_quantize_bits are never violated, with or without the noise floor and target;
   the noise gate fires on white noise at 1e±300 and across ±DBL_MAX as at 1; ranges from a few
   subnormals to past 2^1023 round-trip within the bounds, with no inf, and exponents, anchors and
   q matching exact rational arithmetic.
8. **Noise gate:** it fires on white Gaussian noise (with or without a slow signal under it)
   and never on a random walk, a clean sine at any frequency, a chirp, a ramp or a square wave.
   It may fire on the float rounding noise of a smooth polynomial (the quadratic), with no
   effect, since that σ is far below the B-bit step. On all clean test signals the unit is
   byte-identical with the noise floor on (f = 0.01–1) and off.
9. **Time axis:** datetime64 in s, ms, µs and ns, and integer ticks with a time unit, round-trip
   exactly and in their unit; units without times decode `times = None`; regular series store no
   planes; a gap makes only its block irregular, with the GCD as its step; jitter takes the rounded
   mean as its reference and skewed deltas the minimum; only a block whose
   residuals reach 2^32 is long; equal timestamps, all-equal blocks, short last blocks, blocks of
   0, 1 and 2 samples (a 2-sample block whose delta exceeds int64 maximum), leading empty blocks
   before negative ticks, ticks at both ends of int64 and a block spanning more than half of it
   round-trip. The encoder rejects decreasing
   times (within and across blocks), NaT, unsupported dtypes and units, and length mismatches.
   `update` with times is byte-identical to encoding the edited series, and
   rejects missing, unexpected, mis-shaped, wrong-unit or out-of-order times.
10. **Snapped grid** (`tests/test_grid.py`): the exponent rule picks at most one level finer than
   the unsnapped rule (only below 16 bits) and one coarser (only at 16), and its grid fits L;
   decoded values are fixed points block by block wherever the re-encode picks the same step,
   noisy, smooth, stepped, short and at extreme magnitudes; re-encoding decoded values one level
   coarser keeps their mean (ties to even); 200 straddling updates that move the block's min and
   max keep the kept samples' error under one step of the coarsest grid used, unchanged while the
   step is.

## References

- Quantization: https://en.wikipedia.org/wiki/Quantization_(signal_processing)
- Noise estimate (§5.2): [mean absolute deviation](https://en.wikipedia.org/wiki/Average_absolute_deviation)
  and [autocorrelation](https://en.wikipedia.org/wiki/Autocorrelation).
- Decimal detection (§5.3) is related to ALP: A. Afroozeh, L. Kuffó, P. Boncz,
  [SIGMOD 2024](https://doi.org/10.1145/3626717).
