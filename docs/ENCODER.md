# fluxcode — reference encoder

How the `fluxcode` package encodes. The format it writes is specified in [SPEC.md](SPEC.md).

- **Scope:** its parameters, how it divides a series into blocks, the algorithm for one block, and
  the guarantees that follow.
- **Not format rules:** any writer of valid block groups conforms. This encoder's choices (the step,
  the anchor, the order, the time reference) are its own.
- **API:**
  - encoding: `encode_group`, `encode_blocks`, `encode_time_blocks`, bulk `encode`;
  - decoding: `decode_group`, bulk `decode`;
  - updating: `update`, `update_time_blocks`;
  - parameters: `Params`.
- **Related:**
  - speed and implementation notes: [PERFORMANCE.md](PERFORMANCE.md);
  - measured noise-floor behaviour, and what to check on your data: [TUNING.md](TUNING.md);
  - the experiments behind the design: [REPORT.md §11](../experimental/REPORT.md#11-power-of-two-quantization-the-fluxcode-design).

## 1. Parameters

- **All parameters are encoder-only.** The block group is self-describing (SPEC.md §1): a decoder
  needs only the block group.
- Timestamps are data, not parameters (§4), and so is the division into blocks (§2).

| parameter | values | default | effect |
|---|---|---|---|
| `max_quantize_bits` | min–16 | 16 | the finest step (hard), `B` in §5.1 |
| `min_quantize_bits` | 1–max | 6 | the coarsest step (hard) |
| `diff_orders` | non-empty subset of {0, 1, 2, 3} | {0, 1, 2, 3} | predictor orders to choose from (§5.5) |
| `decimal_detection` | on / off | on | try a decimal grid before the power-of-two grid (§5.3) |
| `noise_floor_sigma` | off (0), or f > 0 | 0.25 | the noise floor (§5.2) |
| `target_bits_per_sample` | off, or ≥ 6 | off | soft per-block-group size cap (§5.7) |
| `effort` | 1–9 | 4 | how the body is compressed (below) |
| `time_error` | 0–0.5 | 0 (exact) | how far a timestamp may move, per interval (§4) |

- **`max_quantize_bits`:** a block's snapped power-of-two grid spans at most 2^max steps
  (65,535 at 16, §5.1).
- **`min_quantize_bits`:** neither the noise floor nor the target coarsens a block past 2^min
  steps across its range.
- **`noise_floor_sigma`:** on blocks whose residual looks like white measurement noise, coarsen
  the step to at most f·σ for the whole block.
  - 0.1–0.5 recommended.
  - Turn it off per tag where high-frequency content matters (vibration, harmonics).
  - With irregular times it applies only to blocks with a cadence (§5.2): a jittered clock or
    one with gaps gets it, a sparse swinging-door archive or events get none.
- **`target_bits_per_sample`:** a guard against unexpectedly high usage, not a way to squeeze
  signals whose shape you don't know.
- **`effort`:** changes the size and the encode time, never the decoded values.
- **`time_error`:** 0 stores timestamps exactly. e > 0 lets each move by up to e times its block's
  interval, which puts a jittered clock back on its grid.
  - About 5× the clock's jitter (σ over the interval) makes it regular: 0.05 for 1% jitter.
  - The values see the rounded times: a block that becomes regular gets the noise floor of regular
    times (§5.2).

**Precedence.** With e_fine the §5.1 exponent at `max_quantize_bits` and e_coarse the one at
`min_quantize_bits`:
1. a block starts at e_fine;
2. the noise floor raises it on gated blocks;
3. it is clamped to e_coarse;
4. decimal detection looks for a decimal grid coarser than that step;
5. the target (if set and the block group is over budget) coarsens blocks further, never past
   e_coarse.

With both the noise floor and the target on, each block takes the coarser step.

**Effort.** Each effort picks:
- the residual layout: 16 bit planes or 2 byte planes (SPEC.md §6; the header records which);
- whether the zstd frame is **flushed**: it ends a zstd block after the per-block columns and
  after each dense residual plane (more than 1/16 of its bytes non-zero).
  - Every zstd block has its own literal Huffman table.
  - The frame is an ordinary one either way.
- the zstd levels.

| effort | layout | frame | zstd |
|---|---|---|---|
| 1 | heuristic | one run | 1 |
| 2 | heuristic | one run | 3 |
| 3–4 | best | one run | 3 |
| 5–8 | best | flushed | 3 |
| 9 | best | flushed | 3 and 9 |

- **Heuristic layout:** byte planes if fewer than 1% of the block group's zigzagged residuals reach
  128, else bit planes. One compression.
- **Best layout:** both layouts, the smaller kept. Ties go to byte planes (they decode faster).
- **Effort 9** compresses at both zstd levels and keeps the smaller (zstd 9 alone is larger on
  about a fifth of block groups).
- **Ordering of sizes:**
  - no block group grows from effort 5 to 9;
  - the other steps are ordered only on average: a flushed frame is larger than one run on tiny
    block groups, and effort 1 uses another zstd level.
- **Efforts sharing settings** leave room for later strategies; their output may change when
  they get one.
- Measured size and speed per effort are in [TUNING.md](TUNING.md#effort).

## 2. Dividing a series into blocks

Every block records its size (SPEC.md §1), so a series can be divided in whatever way suits it:
- **Fixed size** (`encode_group`, `encode`): dense, regularly sampled data.
  - Blocks of `block_len` samples; the last block holds the rest.
  - The bulk `encode` splits a long series into block groups of `blocks_per_group` blocks (60 of
    1000 by default: one channel-minute at 1 kHz).
- **Explicit sizes** (`encode_blocks`): one flat array of samples and each block's size.
- **Fixed duration** (`encode_time_blocks`): block b holds the samples timed in
  `[start + b·duration, start + (b+1)·duration)`.
  - For irregular data, or data that arrives incomplete: a minute with no samples is an empty block.
  - `update_time_blocks` later replaces what the block group holds in given time ranges (§6).
  - `start` and `duration` belong to the caller (typically the start is part of the block group's
    storage key). The block group doesn't store them.

**Short blocks** (at most 8 samples) aren't analyzed:
- they take the finest step (e_fine, §5.1) and order 0 (the quantized values themselves);
- no noise floor, and no part in the target (§5.7);
- decimal detection (§5.3) still runs, against the finest step: it needs no analysis, so short
  blocks of decimal data stay bit-exact;
- non-finite values in them are still coded exactly.

**Summary statistics.** Per block the encoder also returns `lo = min(x)`, `hi = max(x)` and the
mean, over the finite samples.
- They come free with encoding: min and max size the grid, and the sum is the non-finite check.
- The caller may store them as index columns. Decoding never uses them.
- An empty block's are NaN.

**Any finite block encodes**, from ranges of a few subnormals (2^−1074) to ranges past 2^1023,
where `hi − lo` itself overflows (§5.1, §5.4, SPEC.md §5).
- Each extreme costs one decision per block; ordinary blocks take the plain formulas.

## 3. Non-finite values

The format stores a 2-bit code per non-finite sample (SPEC.md §2). The encoder:
- **Holds values.** Before the block goes through §5, each non-finite sample is replaced by the
  previous finite value (the first finite value, for a leading run).
  - The analysis stages never see NaN or inf.
  - A held run is a zero residual under order 1. Under orders 2 and 3 its start and end each
    leave a residual or two (the slope changes); the order pick isn't made aware of that.
- **Summary statistics** cover the finite samples.
  - A block with none (all NaN, all +inf, all −inf, or a mix) gets NaN for min, max and mean. It
    encodes as a constant 0 (anchor 0.0) that the codes then overwrite.
  - A −0.0 minimum, maximum or mean is reported as +0.0, matching what decodes.
- **Estimates the noise floor from runs of finite samples.** Held runs would bias the estimate.
  - A flagged block uses only the second differences whose three samples are all finite, and in
    ρ only pairs of consecutive such differences (each sum normalized by its own count).
  - Why not join the finite samples across gaps: a regular gap pattern (every third sample
    missing, say) would alternate the time step and look like white noise.
  - The estimate needs at least n/2 such differences (500 at n = 1000). Below that a random walk
    starts passing the gate, so the block keeps its full precision.
  - One NaN per block on the noisy test signals costs nothing (noisy-sine 5.42 → 5.41
    bits/sample); skipping the noise floor there would cost 2.5×.
- **Cost:** a block with no non-finite values encodes exactly as it would without the feature.

## 4. Time axis

- **Ordering.** The encoder requires the whole series to be **non-decreasing** and rejects any
  decrease. Equal consecutive timestamps are allowed.
- **The time reference** of an irregular block (SPEC.md §3), with σ² the quotients' variance:
  - the **rounded mean** of the quotients when 3σ² < (mean − min)²: they spread around a center
    (clock jitter);
  - otherwise their **minimum**: they are skewed (gaps, events, deadband logging), and the mean
    would sit away from most of them.
  - Why: zigzagged residuals from the mean are about 2σ. Those from the minimum are
    non-negative, so zigzag only shifts them up a bit plane (plane 0 all zero), and they are about
    √(σ² + (mean − min)²).
- **Time error** (`time_error` = e > 0): each block's ticks are rounded before the analysis
  above. The format is unchanged.
  - **Interval** c: the median of 15 evenly spaced intervals, as for the noise floor's cadence
    (§5.2). If at least 90% of the intervals are one scan (strictly between c/2 and 3c/2), their
    mean instead: stable to about 0.2% on a 5%-jittered clock, where the median of 15 varies by
    about 2%.
  - **Quantum** q: the largest 1-2-5 × 10^k ticks at most 2·e·c·1.02.
    - The 2% slack: a nice e times a nice period is itself a 1-2-5 step (2 × 0.1 × 1 ms), so on a
      clock running slightly fast q would otherwise flip between 200 and 100 µs from block to
      block.
    - Any 1-2-5 step up to a fifth of a 1-2-5 period divides it: for e ≤ 0.1 a jittered clock
      rounds back onto its own grid.
    - A block isn't rounded if q would be 1, or it has fewer than 16 samples (the median of 15
      intervals needs 16: a smaller block's grid would be a guess) or a median interval of 0, or
      it is regular: a regular block stores as cheaply as it can already, so rounding could only
      move its times.
  - **Phase** φ: the block's grid is φ + k·q from the epoch, with φ one of ten multiples of q/10
    (q/10 is a 1-2-5 step too, so the times stay on a round grid, and series on the same clock
    pick the same one).
    - Each candidate is scored by the summed distance of 64 evenly spaced ticks to its grid. A
      clock on round times (as historians scan) scores best at 0; a free-running one at its own
      phase.
    - A block keeps the previous block's phase (that of its last rounded tick, if a candidate;
      the epoch's grid, 0, for the first) unless another scores better by more than half a
      step per tick. Two candidates either side of a clock's phase both round it cleanly, and
      noise would otherwise pick between them block by block, leaving the block starts uneven;
      on a noisy clock it would pick a phase a step off. A drifting clock moves on by a step
      once it has drifted about half of one.
    - A new block after a non-empty one (stored or earlier in the same encode) starts from that
      block's last tick, so encoding at once and appending block by block agree.
    - Quanta of 2^56 ticks or more (over two years in ns) keep the epoch's grid: the integer
      scores would overflow.
  - **Rounding:** to the nearest point of the block's grid (ties up). A tick moves by at most
    q/2 ≤ 1.02·e·c. The ticks' order is checked before rounding, which could hide a decrease.
  - **Following a regular clock:** a block of at least 16 samples after an exactly regular block
    of at least 16 (interval d) whose ticks are all within e/2·d of that block's lattice (its
    last tick + k·d, k ≥ 1) is rounded onto the lattice, so it is regular on the same clock
    (also when d has no 1-2-5 grid, like 1001 µs). Otherwise it gets its own grid as above.
  - **Order across blocks:** rounding with one quantum keeps ticks in order. Where neighbouring
    blocks' quanta differ and the earlier block's last ticks round past the later one's first,
    the block with the larger quantum gives way: the earlier block's trailing ticks are lowered
    to the later one's first, or the later block's leading ticks raised to the earlier one's last.
    Every tick stays within half its own block's quantum. That takes near-duplicate times across
    a block boundary where the rate changes.
  - **Time blocks** (`encode_time_blocks`): times are rounded with the quanta of the blocks they
    fall in, then assigned to blocks. A time that rounds across a boundary belongs to the block it
    lands in; one that would round below `start_time` is raised to it (between its time and its
    rounded one). Every block's times stay in its range.
  - **`update`** rounds the new blocks as encoding does. A stored block can't give way, so a new
    block that rounds past a stored neighbour (the stored times were rounded too) is clamped to
    it, raised to the previous block's last time or lowered to the next one's start, if its given
    time is within half the larger of the two blocks' quanta of it. A stored regular block's
    quantum comes from its columns (step × reference), an irregular one's from its expanded
    times. Re-sending a block's original times gives the original block group; larger overlaps
    are rejected as decreases.
  - **`update_time_blocks`** rounds each new sample with the quantum and phase of the block it
    falls in, from that block's stored and new samples, then assigns and merges: "the same
    timestamp" is the rounded one. Stored samples keep their times. A new sample that rounds into
    the next block joins it (decoded if stored).
    - Upsert and rounding are at odds: a new sample that rounds onto a stored one replaces it,
      whether it re-sends that sample or is a distinct event a fraction of a quantum away (Poisson
      events appended in small batches lose about 0.1% at e = 0.05). A caller using both should
      delete the range the new samples replace.
  - **Without a cadence** (events, sparse archives) c is the median of 15 intervals, a rough
    estimate: on Poisson events times moved by up to 1.45·e of the overall median interval.
  - Measured sizes and times are in [TUNING.md](TUNING.md#time-error).
- **Cost.**
  - A regular series costs only its per-block start, step and reference: about 65 bytes per
    block group of 60 blocks.
  - A gap costs only its own block.
  - Long blocks are rare: in ns ticks with a GCD of 1, a gap of about 2 s or more; in µs ticks
    or on any coarser grid, a gap of over half an hour.
  - Measured costs per clock shape, and the measurements behind the design (the GCD, plain
    deltas, the reference, 32 planes), are in [TUNING.md](TUNING.md#time-axis).

## 5. Encoding one block

### 5.1 Power-of-two exponent

```
L  = 2^B if B < 16 else 65535            # q is stored mod 2^16: it must stay below 2^16
e  = the finest exponent in [−1074, 1023] with  rint(hi · 2^-e) − rint(lo · 2^-e) <= L
     (0 for a constant block, rng = 0)
ps = 2^e                                 # the power-of-two step
```

- rint rounds half to even.
- **The rule counts the levels the snapped grid uses.** The grid is absolute (§5.4): the points
  nearest lo and hi can sit up to half a step outside [lo, hi], so the snapped grid can need one
  more level than the range alone.
- **To compute it,** start from the unsnapped rule and adjust by at most one level:

  ```
  k  = frexp(rng).exponent                 # rng = hi − lo in [2^(k-1), 2^k)
  e0 = k − B;  if floor(ldexp(rng, −e0) + 0.5) >= 2^B:  e0 += 1
  ```

  - e0 − 1 fits only when the snapped grid needs exactly 2^B levels (so only for B < 16);
  - if e0 itself doesn't fit (65,536 levels at B = 16), use e0 + 1;
  - coarser exponents always fit.
- **Never coarser than the unsnapped rule, except at the storage edge.** When the unsnapped rule
  accepts e (round(rng/2^e) ≤ 2^B − 1), the snapped grid needs at most 2^B levels. So the rule
  differs from it only by:
  - one level finer for B < 16, which fixes the fencepost case: a range of exactly a power of
    two, like [1, 2], gets all 2^B steps instead of half;
  - one level coarser for about 1 block in 65,000 at B = 16, whose range is within half a step of
    65,535 steps.
- **It depends on where the block sits on the grid,** not just its range: two blocks with the
  same range at different offsets can get different steps. It is still a function of lo and hi.
- **Wide blocks** (rng ≥ 2^1023, possibly inf): compute e0 from `rh = hi/2 − lo/2` (halving is
  exact) as e0(rh) + 1.
  - That is the same e0 the exact range gives.
  - The snapped count never overflows: lo · 2^−e and hi · 2^−e are each finite.
- **The clamps.**
  - At e = −1074 every double is on the grid, so a block whose finest step would be finer encodes
    losslessly.
  - At e = 1023 (only reachable with few bits on a range near 2^1024), q stays below 3.
- The encoder computes e_fine with B = `max_quantize_bits` and e_coarse with B = `min_quantize_bits`.

### 5.2 Noise floor

Runs when the noise floor f is on (`noise_floor_sigma` > 0), rng > 0 and the block has at least
256 samples:

```
d     = x[i+2] − 2·x[i+1] + x[i]  for i = 0..n−3,  minus its mean   # second differences
split d into W = max(1, n_d // 256) windows of n_d / W differences each
per window: mad = mean |d|; twice: mad = mean of |d| over |d| <= 4·sqrt(π/2)·mad  # clipped
            sd_w = sqrt(π/2)·mad;  clip the window's d to ±4·sd_w
σ     = min(mean_w sd_w, min_w sd_w / 0.75) / sqrt(6)
ρ     = lag-1 autocorrelation of the clipped d
if ρ < −0.6:   e = max(e, floor(log2(f·σ)))
e = min(e, e_coarse)                                                # min_quantize_bits is hard
```

On a block with irregular times (not all intervals equal), first:

```
Δ_i   = t[i+1] − t[i]
c     = the median of Δ_i at i = k·(n−2) // 14 for k = 0..14              # the scan interval
one_scan_i = c/2 < Δ_i < 3c/2                                             # in integers
if fewer than 90% of the Δ_i are one scan: no floor
d_i   = sqrt(6)·(b·x[i] + a·x[i+2] − (a + b)·x[i+1]) / sqrt((a + b)² + a² + b²)   # a = Δ_i, b = Δ_{i+1}
        only where one_scan_i and one_scan_{i+1}; at least 254 of them, else no floor
```

- **Times are given, or the samples are evenly spaced.** Without times the encoder assumes a
  regular grid. A block group of irregular samples (an archive exported without its time column)
  should be stored with its times, or with the noise floor off.
- **Irregular times: only blocks with a cadence, and consecutive scans only.** A historian's
  swinging-door (SDT) or deadband archive keeps only the points a straight line or the last value
  can't predict, so its points look like white noise without being noise: on the simulated
  archives of [sdt.html](https://gboyer.github.io/fluxcode/report/sdt.html) the plain estimate put
  σ at 4× the sensor's. One that kept nearly every scan is the sensor's raw samples. The rule is
  all or nothing: an unanticipated spread of intervals gets no floor rather than a wrong one.
  - The scan interval is the median of 15 evenly spaced intervals. Where 90% of the intervals are
    one scan, so is that median, wherever the gaps and two-scan intervals fall. A mean would sit
    between one and two scans: the mean of the two lowest occupied octaves passed exact 1×/2×
    mixes that kept 40–60% of scans. Where the samples miss the cadence (gaps aligned with their
    stride), the window misses too and the block gets no floor.
  - ±50% separates one scan from a skipped one (2×). Measured on blocks that kept every scan:
    jittered times up to ±20% uniform, sd 15% Gaussian or a 10% mean receive delay keep the floor
    on every block. At the edges (±30%, sd 20%, a 20% delay) a block whose 15-sample median lands
    off center loses it; beyond (±40%, sd 25%, a 30% delay) every block does.
  - Up to 10% of intervals off the cadence keep the floor: gaps (an outage is one interval),
    dropped scans, near-duplicate times. Anything sparser gets none: SDT or deadband archives
    that kept up to about 87% of scans, scans dropped at random or every 8th, events
    (exponential intervals) and mixes of one- and two-scan intervals.
  - It applies to an explicit f as to the default, so the noise floor is safe on data whose shape
    the caller doesn't know.
  - The difference is the middle sample's from the line through its neighbours, scaled to a
    regular second difference's variance on white noise (6σ² for any intervals), so a slope
    cancels across unequal intervals. On equal intervals it is the plain second difference.
  - Consecutive scans in a swinging-door archive are the ones that broke the line: σ measured on
    them reads about 1.3× the sensor's on the simulated archives, so the step there is up to
    f·1.3σ, on archives that kept at least 90% of scans.
  - A scan of 1–2 ticks is ambiguous (a 1.5 s scan in whole seconds gives intervals 1, 2, 1, 2,
    as a 1 s archive that dropped a third would) and gets no floor.
  - Regular blocks take the plain path: the same steps as without times.
- **Windows.** Noise that varies within a block (a quiet stretch, then a noisy one) gives one
  blended estimate over the whole block, which would coarsen the quiet part past f·σ of its own
  noise. Each window of 256 or more differences gets its own estimate (a block of under 512
  differences is one window). σ is their mean, unless the quietest is below 0.75 of it: then the
  quietest over 0.75, at most 4/3 of the quiet part's noise.
  - The plain minimum would be biased by the number of windows: on steady white noise it read 5%
    low over 3 windows (1,000 samples) and 18% low over 255 (65,536), a step one level finer on
    7% and 28% of blocks. Over up to 255 windows the quietest never measured below 0.77 of the
    mean, so steady noise takes the mean, unbiased.
- **What the gate separates,** by ρ on second differences:
  - white measurement noise: −2/3;
  - a random walk (whose increments are signal): −1/2;
  - smooth or fast deterministic signals: positive.
  - Only white noise passes; the rest keep the B-bit step.
- **Effect.**
  - The step stays a power of two, so edit stability and the format are unchanged. The decoder
    doesn't know or care.
  - Added error is at most f·σ/√12 RMS (half that on average, because of the power-of-two floor)
    on top of noise σ.
- **Scaling.** The estimator runs on x scaled by the power of two that brings the range into
  [½, 1).
  - That's exact, so σ and ρ are unchanged.
  - The differences and their squares then can't overflow near 1e308 or underflow near 1e−300.
    Without it, noise at 1e300 would never gate, and a block at 1e−300 would divide by zero.
  - If every clipped square underflows anyway (differences below ~1e−154 of the range), the block
    isn't gated.
  - For ranges beyond 2^±1000, whose scale factor would overflow, the scaling is applied in two
    steps, the first outside fastmath so LLVM can't fold them.
- **f is effectively rounded down to a power of two** relative to σ: the step is
  2^floor(log2(f·σ)), so f values within a factor of 2 can give the same step (0.3 behaves like
  0.25).
- **On gated blocks `max_quantize_bits` stops mattering** whenever f·σ exceeds the finest step:
  for f ≥ 0.01 on the measured noisy signals at B = 16.
  - f alone sets their size, down to the `min_quantize_bits` clamp.
  - A noisy block's noise-floor step spans range/(f·σ) steps: ≈ 230 (about 7.8 bits) on the
    noisy sine at f = 0.25. A minimum above that would override the noise floor.
  - The default minimum of 6 bits costs ≤ 1% on the noisy test signals.
- **Why 256 samples.** The ρ test separates white noise (−2/3) from a random walk (−1/2) only on
  enough differences.
  - On m differences, the estimate of ρ has a standard error of about √((1 − 3ρ² + 4ρ⁴)/m)
    (Bartlett's formula for a lag-1 autocorrelation): 0.044 for a random walk and 0.042 for white
    noise at m = 254.
  - The threshold −0.6 is then 2.25 standard errors from −1/2 (random walks pass about 1% of the
    time) and 1.6 from −2/3 (white noise is missed about 6% of the time). At 64 samples random
    walks would pass about 13% of the time.
  - Measured: pure random walks pass 21% of the time at 9 samples, 9.4% at 64, 4.3% at 128, 0.8%
    at 256 and never at 1000. White noise passes 96% of the time at 256.
  - So 256 is a chosen trade-off (about 1% false positives, 5% missed white noise), not a hard
    limit. Smaller blocks keep the B-bit step.
- **Re-encoding decoded data** (a merging `update_time_blocks`) computes the noise floor again
  from the merged block: kept samples, now decoded, plus the new ones. Nothing is reused, as when
  encoding from scratch.
  - Quantizing adds white noise of variance s²/12 with s ≤ f·σ = σ/4, so σ can rise by at most
    √(1 + 1/192) − 1 ≈ 0.26%. The estimate also moves by a little either way.
  - The step 2^floor(log2(f·σ)) only changes when f·σ sits within that much of a power of two,
    and then by one level, once.
  - Measured over 5 decode/re-encode rounds of 300 noisy 1000-sample blocks (all gated): σ of the
    decoded data averaged 1.0002× the original (at most 1.008×). 8 blocks went one level finer on
    the first round, and none changed after.
  - Finer is exact on the snapped grid (§5.4), and coarser is one bounded rounding.
  - A merged block can also cross 256 samples, which switches the noise floor on or off between
    updates, with the same two cases.
- **The gate is per block and all-or-nothing.** Per-sample variants (full precision kept at
  spikes) cost 20–35% more for an error already below the noise on the spike ([REPORT.md §11](../experimental/REPORT.md#11-power-of-two-quantization-the-fluxcode-design)).

### 5.3 Decimal detection

Runs when `decimal_detection` is on and rng > 0, after the noise floor, with `ps` the (possibly
coarsened) step. It finds the coarsest decimal step 10^p that is coarser than `ps` and that every
sample sits on:

```
tol = ps / 4
for p = min(floor(log10(rng)), 22) down to max(the smallest p with 10^p > ps, −22):
    s = 10^-p                                        # exact integer when p <= 0
    if all |x[i]·s − floor(x[i]·s + 0.5)| <= tol·s  (for every i)
       and max(|lo|, |hi|)·s < 2^52:
        use decimal step p; stop
    (a failing candidate stops at its first off-grid sample)
```

- **Never finer than `ps`.** When exact decimals would need more than B bits, or are finer than
  the noise, the block takes the power-of-two grid.
- **Absolute grid.** Values must sit on multiples of 10^p itself, not merely be spaced 10^p
  apart. That is what makes reconstruction bit-exact (SPEC.md §5).
- **Tolerance** is a quarter power-of-two step. The decimal grid is only chosen when every sample
  is within that of it, so its max error is never larger than the power-of-two grid's at the same
  step.
- **Exact powers of ten only:** −22 ≤ p ≤ 22, so 10^|p| is an exact double.
- **Skipped for wide blocks:** no 10^p grid with p ≤ 22 spans 2^1023.
- **Worst-case cost.** A block on a decimal grid except for one sample near its end defeats the
  early stop: every candidate from the coarsest down scans almost the whole block.
  - Encode is then up to ~20% slower ([PERFORMANCE.md](PERFORMANCE.md)).
  - Real data rarely looks like this, so it's noted rather than guarded against.
  - If it shows up, probing a few spread-out samples before each full scan would bound it.
- **Not detected:** other steps (ADC counts in engineering units, 0.5, 0.25, 2⁻ᵏ). They use the
  power-of-two grid.

### 5.4 Quantize

```
power of two:  K    = rint(lo · 2^-e)              anchor = K · 2^e
               q[i] = rint(x[i] · 2^-e) - K        (rint: round half to even)
decimal:       K0   = floor(lo · 10^-p + 0.5)       anchor = K0
               q[i] = floor(x[i] · 10^-p + 0.5) - K0
```

- **The power-of-two grid is absolute:** its points are the multiples of 2^e, and the anchor is
  the one nearest the block's minimum (it is **snapped**).
  - Every power-of-two grid is then a subset of the finer ones.
  - So re-encoding decoded values on the same or a finer step returns them bit for bit, wherever
    the block sits and however its minimum moved. That bounds the error of repeated updates (§6).
- **One rounding for K and q.** The minimum maps to exactly K, and rounding is monotonic, so
  q = 0 at the minimum and q ≥ 0 everywhere.
  - With different roundings for the two, a minimum exactly halfway between grid points would get
    q = −1, which wraps to 65,535 and decodes as the maximum.
  - Decimal grids share one rounding between K0 and q too.
- **Ties to even.** Raw data almost never has exact ties, but decoded data re-encoded one level
  coarser does: half its values sit halfway between the coarser grid's points.
  - Rounding them all up would shift the block up by a quarter of the finer step on average.
  - Decimal grids can't have ties: detection only accepts samples within a quarter step of the grid.
- **Exact arithmetic.**
  - x · 2^−e is a power-of-two scale: it can't overflow for a sample of a block whose step is 2^e,
    and where it underflows the index is 0 either way.
  - For e < −1023, where 2^−e overflows, scale by 2^1023 and then by 2^(−e−1023), both exact.
  - If |x · 2^−e| ≥ 2^52, x is already a multiple of 2^e and rint leaves it alone.
  - The difference of two nearby integers held in floats is exact.
- **Exceptions: anchor = lo,** q[i] = floor((x[i] − lo) · 2^−e + 0.5):
  - a constant block (rng = 0): e = 0 is only a placeholder there, and snapping would round the
    value to a whole number;
  - a snapped anchor that isn't finite (K · 2^e beyond ±DBL_MAX). There e ≥ 1008, and x − lo may
    overflow, so it is evaluated at half scale, `floor((x/2 − lo/2) · 2^(1−e) + 0.5)` (halving is
    exact, so x/2 − lo/2 rounds exactly as (x − lo)/2).
- **Range:** in both grids 0 ≤ q[i] ≤ L ≤ 65,535 (§5.1), and q = 0 at the minimum.
- When decimal detection succeeds, its candidate pass has already computed these q; they are reused.
- **An unsnapped anchor from another writer** decodes correctly (SPEC.md §4). Updating such a
  block costs at most half a step on its first straddling update; the bound of §6 holds from then
  on.

### 5.5 Order

For each k in `diff_orders`, compute the variance of `diff(q[0..M−1], k)` over the first
M = min(250, n) samples (M − k values, no padding):

```
v_k = (m·Σd² − (Σd)²) / m²,   d = diff(q[:M], k),   m = M − k
order = the k with the smallest v_k; ties go to the lower order
```

- Sums are exact integers (int64).
- The order only has to be *a* valid choice: the decoder reads it from the block group.
- Picking on the first 250 samples agrees with the full-block pick on 97% of blocks. It costs
  0.4% in size on continuous data, and nothing on discretized data.

### 5.6 Residuals

The residuals of the order picked in §5.5, mod 2^16 and zigzagged, as specified in SPEC.md §4.

### 5.7 Per-block-group target

Runs when `target_bits_per_sample` = t is set, after every block of the block group has been through
§5.1–5.6.

**Estimate.** Each block's size is estimated from its residual by the **class entropy**. With L(u)
the bit length of the zigzagged residual (0 for 0):

```
h_b = −Σ_L p_L·log2 p_L  +  Σ_L p_L·max(L − 1, 0)      # entropy of L, plus the bits below the leading 1
```

**Budget.** It covers the analyzed blocks (more than 8 samples), each weighted by its size n_b, with
N their total samples.
- If Σ_b n_b·h_b ≤ t·N, nothing changes.
- Otherwise:

  ```
  e_b     = the block's exponent (for a decimal block, floor(log2 10^p): its grid is already that coarse)
  give_b  = min(round(max(h_b − 1, 0)), e_coarse_b − e_b)        # whole bits block b can give
  k       = the smallest k ≥ 1 with Σ_b n_b·min(k, give_b) ≥ Σ_b n_b·h_b − t·N   (k ≤ 16)
  k_b     = min(k, give_b)
  blocks with k_b > 0: redo §5.3–5.6 at e_b + k_b; keep the result only if its h_b went down
  ```

- **Expensive blocks end up at equal precision.** Coarsening a block by one bit saves about one
  bit per sample while its estimate is above ~1 bit/sample, and little after that. Cheap blocks
  (which would save almost nothing) aren't touched.
- **"Keep only if h went down"** stops the cap from trading a decimal grid for a coarser
  power-of-two grid that costs more (a 0.01-grid sine: 2.64 → 3.01 bits/sample without it).
- **Cost.** zstd never runs in the loop. Blocks the cap coarsens are quantized a second time;
  without a target every block is quantized once.
- **It's a soft cap, and t ≥ 6.**
  - The estimate is within ~0.5 bits/sample on noise-like data, and high on signals zstd
    compresses by repeats (chirp 11.9 vs 7.8, quadratic 5.5 vs 0.2).
  - At small targets it coarsens blocks that were already cheap, and can make them larger (the
    quadratic: 0.21 → 0.54 bits/sample at t = 2).
  - The target is meant to cap unexpectedly high usage, so t must be at least 6, where the
    quadratic's estimate is under budget.
  - Supporting lower targets would need a check of the actual compressed size (a TODO in the
    encoder).
  - Validate on real data (`bench/estimate.py`).
- **In `update` and `update_time_blocks`** the cap applies to the re-encoded blocks only, with a
  budget of t × their samples.

## 6. Guarantees

- **Max error ≤ half the step used = 2^(e−1)**, up to one rounding of the sample's magnitude.
  - **Always ≤ range / (2^min_quantize_bits − ½)**, whatever the noise floor or the target does:
    range/63.5, 1.6% of the block's range, at the default of 6. The ½ is from the exponent bump
    in §5.1.
  - Blocks at `max_quantize_bits` (neither noise floor nor target applied):
    ≤ range / (2^max − ½), about 2^−max of the block's range (0.0015% at 16).
  - Noise-floor blocks: ≤ f·σ/2, in the signal's units. This is not bounded relative to the
    range: a block that is all noise has a range of only a few σ.
  - Decimal grid: exact, or within `tol` = 2^e/4 if the samples were within tolerance of the grid.
- **min and max decode within half a step.**
  - Power-of-two grid: the minimum decodes to the grid point nearest it (exactly, when it is on
    the grid, as decoded data always is).
  - Decimal grid: the minimum decodes onto the decimal grid. That's exact for decimal data, but a
    minimum carrying float noise (e.g. 1000.0000000000291 on an integer grid) decodes to the grid
    value (1000.0): an error within `tol`, as for every sample.
  - The `block_min` and `block_max` that encode returns are the samples' own. Callers using them
    as hard bounds on decoded values (for pruning) should allow half a step either way.
- **Decimal data is bit-exact:** values that are decimals of p places (as parsed from text)
  decode to the identical float64.
  - This holds when the block's range spans fewer than 2^B decimal steps; wider blocks use the
    power-of-two grid.
  - float32 rounding artifacts are intentionally lost: a decimal stored as float32 upstream
    decodes as the decimal itself.
- **Decoded data is a fixed point:** decode(encode(y)) = y for any decoded y. The block group bytes
  are identical from the second encode on.
- **Edits are stable.** Changing a sample leaves every other sample's q unchanged unless the step
  changes.
  - The step changes only when the range crosses a power of two: the grid is absolute, so a new
    minimum doesn't move it.
  - Why absolute: with a grid scaled to the range, a new maximum re-rounds every sample (95% of
    untouched samples changed in testing).
- **`update` leaves other blocks untouched.** It replaces or appends whole blocks of a block group,
  of any size.
  - The other blocks' flags, grid parameter, anchor, residuals, codes and time rows carry over
    unchanged: they are neither dequantized nor re-encoded, and decode to identical values.
  - Indices skipped past the block group's end are appended as empty blocks.
  - With a time axis, `update` takes the new blocks' times (required exactly when the block group
    has one). The updated series must be non-decreasing throughout, which it checks where a new
    block meets its non-empty neighbours.
  - Without a target, the updated block group is byte-identical to encoding the updated series from
    scratch (every block is encoded independently); with a time error, unless a new block was
    clamped to a stored neighbour (§4).
- **`update_time_blocks` is an upsert plus a range deletion, touching only the blocks it affects.**
  - **What it keeps:**
    - it first discards every sample timed in the optional delete ranges (`[start, end)` each);
    - it then adds the new samples, which may be timed anywhere (a new sample inside a range is
      kept);
    - every existing sample with the same timestamp as a new one is discarded too: the new sample
      replaces it;
    - new samples sharing a timestamp are all kept, in their order;
    - with no ranges it is an upsert; with no samples, a deletion.
  - **Which blocks it touches:**
    - a block that no range meets and no new sample falls in is carried over as `update` carries
      it, without being decoded;
    - a block wholly inside the ranges is encoded from the new samples alone;
    - any other affected block is decoded, keeps its surviving samples and is re-encoded with the
      new ones;
    - new samples past the block group's end append blocks (empty ones to fill a gap);
    - blocks are never removed: a block emptied by an update stays, empty.
  - **Error of kept samples:**
    - they are already points of the absolute grid (§5.4), so on the same or a finer step they
      come back bit for bit, however the merged block's min and max moved;
    - only a coarser step rounds them again, once, without bias (ties to even);
    - repeated coarsening adds a geometric series, so their error stays under one step of the
      coarsest grid the block has used;
    - decimal data on a decimal grid stays exact.
  - The result is byte-identical to `encode_time_blocks` of the resulting series when no block is
    left empty at the end, and no time error is set. With one, a block that has stored samples
    is rounded by its rule below, so a tick never moves further than e·c (+2%) in all, however
    often the block is updated: a tick moves twice at most, once onto a regular block's own
    lattice (by up to e/2·d), once onto a 1-2-5 grid, and a tick on a 1-2-5 grid stays.
    For each block with stored samples left and at least 16 samples after the update that is not
    exactly regular:
    1. A stored regular block (interval d): new ticks all within e/2·d of its lattice go onto it.
    2. Stored ticks on a 1-2-5 grid (90% of them, the coarsest step up to the block's
       interval): they stay; new ticks go to the coarsest 1-2-5 grid at most the merged block's
       quantum q* that nests with it (divides it or is a multiple), at its phase.
    3. Or the whole block, stored ticks included, is rounded as in an encode, to the coarsest
       quantum within the error budget: a tick may have moved already (half the stored grid's
       step, or e/2·d if the stored ticks lie on a lattice coarser than q* that rule 1 may have
       snapped to), and the quantum is at most 2·(1.02·e·d − that).
    Rule 3 is taken if its quantum is at least the common step rule 2 leaves the block on (so
    raw µs-precision ticks in ns, which lie on a 1 µs grid, are rounded whole), else rule 2 (a
    block already rounded to 200 µs has no budget left). A tick thus moves its prior movement plus
    half the quantum, ≤ e·d in all.
    Blocks without stored samples are rounded as in an encode, following the block before them.
- **Times** decode exactly at `time_error` 0. With e > 0 (§4):
  - each moves by at most half its block's quantum, ≤ 1.02·e times the block's interval (in an
    `update`, a block clamped to a stored neighbour with a larger quantum: half of that one);
  - the series stays non-decreasing, and every time block's times stay in its range;
  - `update_time_blocks` moves a stored time only when it rounds the whole block (rule 3).

## 7. Conformance tests

1. **Round trip:** decode(encode(x)) has
   - max error ≤ 2^(e−1) for every block (≤ f·σ/2 on noise-floor blocks);
   - max error ≤ range / (2^min_quantize_bits − ½) always;
   - the decoded minimum within half a step of the minimum.
2. **Decimal:** values generated as `K / 10^d` with a range under 2^B steps decode bit-identical.
3. **Fixed point:** y = decode(encode(x)) satisfies decode(encode(y)) = y, and encode(y) is
   byte-identical from the second encode on (§6).
4. **Edge cases:**
   - a constant block: all q = 0, decodes to lo exactly, including non-integers and extreme
     magnitudes;
   - a range just below a power of two (the exponent bump);
   - a minimum exactly halfway between grid points (q = 0, not −1);
   - [1, 2] at B = 12 (4,096 steps, grid-aligned data exact) and B = 16 (32,768 steps);
   - the storage edge at B = 16: a snap that would need q = 65,536 goes one level coarser;
   - a block whose range exceeds 2^B decimal steps falls back to the power-of-two grid;
   - fewer than 60 blocks; a short last block; block groups of different block counts and block
     sizes;
   - blocks of every size from 0 to 9 and over 1000 in one block group; block groups with no blocks
     or only empty ones;
   - blocks of up to 8 samples at the finest step and order 0, whatever the parameters, on a
     decimal grid (bit-exact) when one fits and decimal detection is on.
5. **Update:**
   - untouched blocks decode identically and keep their rows;
   - without a target, `update` (replacing blocks with ones of other sizes, emptying, and
     appending past a gap) is byte-identical to encoding the updated blocks;
   - `update_time_blocks` is byte-identical to `encode_time_blocks` of the expected series (old
     samples outside the ranges and not sharing a time with a new one, plus the new ones) for:
     ranges filling empty blocks, covering whole blocks and straddling block edges; upserts with
     no ranges (including duplicate times in the new data and in the block group); and appending;
   - it decodes only the blocks it merges;
   - it accepts ranges as one pair, a list, a `(k, 2)` array of datetime64 or ticks, and naive
     datetimes.
6. **Non-finite:** NaN / ±inf runs at a block's start, middle and end, scattered, alternating,
   and whole blocks, on every signal kind and with the target:
   - exact NaN positions and infinities;
   - finite samples within the bounds;
   - summary statistics over the finite samples;
   - blocks without non-finite values byte-identical to block groups without the field;
   - `update` keeps existing codes.
7. **Limits:**
   - min/max_quantize_bits are never violated, with or without the noise floor and target;
   - the noise gate fires on white noise at 1e±300 and across ±DBL_MAX as at 1;
   - ranges from a few subnormals to past 2^1023 round-trip within the bounds, with no inf, and
     exponents, anchors and q matching exact rational arithmetic.
8. **Noise gate:**
   - it fires on white Gaussian noise, with or without a slow signal under it;
   - it never fires on a random walk, a clean sine at any frequency, a chirp, a ramp or a square
     wave;
   - it may fire on the float rounding noise of a smooth polynomial (the quadratic), with no
     effect, since that σ is far below the B-bit step;
   - on all clean test signals the block group is byte-identical with the noise floor on
     (f = 0.01–1) and off.
9. **Time axis:**
   - datetime64 in s, ms, µs and ns, and integer ticks with a time unit, round-trip exactly and in
     their unit; block groups without times decode `times = None`;
   - regular series store no planes; a gap makes only its block irregular, with the GCD as its
     step;
   - jitter takes the rounded mean as its reference, and skewed deltas the minimum;
   - only a block whose residuals reach 2^32 is long;
   - these round-trip: equal timestamps, all-equal blocks, short last blocks, blocks of 0, 1 and
     2 samples (a 2-sample block whose delta exceeds int64 maximum), leading empty blocks before
     negative ticks, ticks at both ends of int64, and a block spanning more than half of it;
   - the encoder rejects decreasing times (within and across blocks), NaT, unsupported dtypes and
     units, and length mismatches;
   - `update` with times is byte-identical to encoding the edited series, and rejects missing,
     unexpected, mis-shaped, wrong-unit or out-of-order times.
10. **Snapped grid** (`tests/test_grid.py`):
    - the exponent rule picks at most one level finer than the unsnapped rule (only below 16
      bits) and one coarser (only at 16), and its grid fits L;
    - decoded values are fixed points block by block wherever the re-encode picks the same step:
      noisy, smooth, stepped, short and at extreme magnitudes;
    - re-encoding decoded values one level coarser keeps their mean (ties to even);
    - 200 straddling updates that move the block's min and max keep the kept samples' error under
      one step of the coarsest grid used, unchanged while the step is.
11. **Time error** (§4, `tests/test_time_error.py`):
    - every time within 1.02·e of its block's interval, at several jitters, time errors and
      periods (1 ms, 16.667 ms), and before 1970;
    - a clock with jitter of a fifth of e stores every block as regular; a regular grid's block
      group is byte-identical to the one at e = 0;
    - a decrease that rounding would hide, and NaT, are rejected, in one block group and between
      the block groups of `encode`;
    - at a rate change the block with the larger quantum gives way, in `encode_blocks` and across
      the block groups of `encode`;
    - re-sending either block of a rate change with `update` gives the same block group; an
      overlap beyond the clamp is rejected;
    - time blocks: times rounding across a boundary land in the next block, every block's times
      stay in its range, and a `start_time` off the grid still rounds;
    - `update_time_blocks` replaces a sample at its rounded time, moves a new sample onto a stored
      next block (keeping its other samples);
    - repeated `update_time_blocks` (jittered and 1001 µs clocks, Poisson events, re-sends, appends
      and off-lattice samples) never move a tick further than 1.02·e·c from the time it was given;
      a lattice snap within e instead of e/2 fails it; a block of fewer than 16 samples isn't
      rounded until it grows; a block continues the regular clock before it, and appending
      block by block gives the bytes of encoding at once;
    - free-running clocks (at a candidate phase, between two, and drifting) round onto their own
      phase, every block at one phase; a regular block off every grid keeps its times; `update`
      keeps a stored block's phase (re-sending gives the block group back) and
      `update_time_blocks` rounds new samples onto it.

## References

- Quantization: https://en.wikipedia.org/wiki/Quantization_(signal_processing)
- Noise estimate (§5.2): [mean absolute deviation](https://en.wikipedia.org/wiki/Average_absolute_deviation)
  and [autocorrelation](https://en.wikipedia.org/wiki/Autocorrelation).
- Decimal detection (§5.3) is related to ALP: A. Afroozeh, L. Kuffó, P. Boncz,
  [SIGMOD 2024](https://doi.org/10.1145/3626717).
