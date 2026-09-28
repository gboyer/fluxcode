# fluxcode — specification

Lossy (bounded-error) and, for decimal data, lossless compression of 1 kHz float64 time
series, with their timestamps stored exactly if given (§2a). One compressed unit holds up to
`blocks_per_unit` blocks of `block_len` samples (encoder parameters; the unit records `block_len` and
its sample count). The defaults, 60 blocks of 1000, make a unit one channel-minute. Implementation: the `fluxcode` package (`encode_unit` / `decode_unit` / `update`, bulk `encode` /
`decode`, `Params`).

This document covers the format, the encoding algorithm and its guarantees. Speed and
implementation notes are in [PERFORMANCE.md](PERFORMANCE.md); measured noise-floor behaviour and
what to check on your data are in [TUNING.md](TUNING.md). The experiments behind the design
choices are in [REPORT.md §11](../experimental/REPORT.md#11-power-of-two-quantization-the-fluxcode-design) of the research in `experimental/`.

## 1. Parameters

All parameters are **encoder-only**. The unit is self-describing: its header records the block
length, sample count and time unit, and each block records its own order, grid (power-of-two
exponent or decimal step) and anchor, and its start time and time step, so a decoder needs only
the unit. Timestamps are data, not parameters (§2a).

| parameter | values | default | effect |
|---|---|---|---|
| `max_quantize_bits` | min–16 | 16 | hard: the finest step. The power-of-two grid has at most 2^max steps across the block's range (`B` in the formulas below) |
| `min_quantize_bits` | 1–max | 6 | hard: the coarsest step. Neither the noise floor nor the target coarsens a block past 2^min steps across its range |
| `diff_orders` | non-empty subset of {0, 1, 2, 3} | {0, 1, 2, 3} | predictor orders the encoder may choose from (§3.4) |
| `decimal_detection` | on / off | on | try a decimal grid (10^p) before the power-of-two grid |
| `noise_floor_sigma` | off, or f > 0; 0.1–0.5 recommended | 0.25 | noise floor: on blocks whose residual looks like white measurement noise, coarsen the step to at most f·σ for the whole block (§3.1a; measured behaviour in TUNING.md). Turn off per tag where high-frequency content matters (vibration, harmonics) |
| `target_bits_per_sample` | off, or ≥ 6 | off | soft per-unit cap on the size (§3.6): a guard against unexpectedly high usage, not a way to squeeze signals whose shape you don't know |
| `try_byte_planes` | on / off | off | also compress the unit with byte planes instead of bit planes (§5) and keep the smaller; ties keep bit planes. Doubles the zstd work of encoding. Worth it on clean periodic signals whose cycles repeat across a unit (TUNING.md) |

**Precedence.** e_fine is the §3.1 exponent at `max_quantize_bits`, e_coarse the one at
`min_quantize_bits`. A block starts at e_fine; the noise floor raises it on gated blocks; it is
clamped to e_coarse; decimal detection then looks for a decimal grid coarser than that step;
finally the target (if set and the unit is over budget) coarsens blocks further, still never past
e_coarse. With both the noise floor and the target on, each block takes the coarser step.

Fixed by this spec: zstd level 3, blocks interleaved by field.
The block length `block_len` is fixed per deployment and must be a multiple of 8 (1000 in everything
measured); `num_blocks` ≤ `blocks_per_unit` (60) blocks per unit.

Residuals are bit-shuffled by default: smaller overall than byte planes, and the best bound on
the cost of noisy, wide signals. A unit may use byte planes instead (header flag, §5), which
`try_byte_planes` picks when they come out smaller (measurements in TUNING.md).

## 2. Units and summary statistics

**A unit is one storage artifact** (e.g. a database row): up to `blocks_per_unit` blocks of
`block_len` samples, encoded and decoded together. Units are independent. Each can hold a different
number of blocks (a stream's row may hold half an hour, then later be updated to the full hour), and
even a different `block_len`. The bulk `encode` just splits a long series into full units.

**Decoding needs only the unit.** The header records `block_len` and the sample count
`num_samples`, and each block stores its own anchor (§5), so no index column or length has to be
kept beside it. **A short last block is padded by repeating its last sample** (which leaves its min
and max unchanged), and decoding trims it back to `num_samples` samples.

Per block the encoder also returns `lo = min(x)`, `hi = max(x)` and the mean (over the real,
finite samples) as **summary statistics**. They come free with encoding (min and max size the
grid; the sum is the non-finite check), and the caller may store them as index columns, but
decoding never uses them.

`block_len` is at most 65,536 and `num_samples` at most 2^26: not format limits (the header field holds
up to 2^40 − 1 samples), but the bounds decoders use to reject
implausible headers before decompressing.
**Any finite block encodes**, from ranges of a few subnormals (2^−1074) to ranges past 2^1023,
where `hi − lo` itself overflows (§3.1, §3.3, §4). Each extreme costs one decision per block;
ordinary blocks take the plain formulas.

**Non-finite values (NaN, ±inf) are encoded exactly.** A block holding any gets head bit 3 and
two code planes in the unit's nonfinite field (§5): 00 finite, 01 NaN, 10 +inf, 11 −inf.
- **Held values.** Before the block goes through §3, each non-finite sample is replaced by the
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
- **Cost:** a block with no non-finite values encodes exactly as if the feature didn't exist. A
  flagged block adds n/4 bytes of mostly-zero planes before compression.

## 2a. Time axis

A unit may also store the samples' **timestamps, exactly**. They are optional: a unit without them
is byte-identical to one written before the time axis existed, and decodes with `times = None`.

**Ticks and units.** Timestamps are int64 ticks since 1970-01-01 in one of the Arrow and numpy
datetime64 units, recorded in the header (§5). The unit only fixes the range and meaning of a
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

**Per block** (`time_start`, `time_step`, `time_ref`, `time_residual_planes` in §5): every block
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
- **Padding** of a short last block extends the grid when its real samples are regular (so the
  block stays regular), and otherwise repeats the last timestamp. Decoding trims it.

A regular series costs only its per-block start, step and reference (about 45 bytes per 60-block unit). A
gap costs only its own block. Measured on one-minute units (`bench/time_axis.py`, Apple M3, AC
power, single thread); the first three rows are the common shapes, and
[experimental/report/time.html](../experimental/report/time.html) shows them in detail:

| timestamps | irregular blocks | bytes | bits/sample | encode time added | decode time added |
|---|---|---|---|---|---|
| perfect 1 kHz grid | 0/60 | 44 | 0.006 | +16 µs (+9%) | +12 µs (+11%) |
| grid with 2 gaps | 2/60 | 657 | 0.088 | +27 µs (+15%) | +21 µs (+19%) |
| noisy clock (σ = 20 µs, µs resolution) | 60/60 | 53,588 | 7.1 | +197 µs (+111%) | +128 µs (+114%) |
| 1 kHz, 20 gaps | 19/60 | 769 | 0.10 | +62 µs (+35%) | +49 µs (+44%) |
| 1 kHz, 1% dropped | 60/60 | 1,466 | 0.20 | +146 µs (+82%) | +105 µs (+94%) |
| drifting clock (0.99998 ms) | 60/60 | 277 | 0.037 | +124 µs (+70%) | +89 µs (+79%) |
| jitter σ = 10 µs, µs resolution | 60/60 | 46,077 | 6.1 | +199 µs (+112%) | +121 µs (+108%) |
| jitter σ = 10 µs, ns resolution | 60/60 | 120,960 | 16.1 | +258 µs (+146%) | +149 µs (+133%) |
| Poisson events (mean 1 ms), µs | 60/60 | 87,757 | 11.7 | +260 µs (+147%) | +170 µs (+152%) |
| deadband logging on a ms grid | 60/60 | 55,479 | 7.4 | +363 µs (+205%) | +220 µs (+196%) |

For scale, the same unit's values take 20,951 bytes, 177 µs to encode and 112 µs to decode; the
percentages are of those times.
Irregular timestamps can cost more than the values: their entropy is what it is. Even a perfect
grid adds 60,000 int64 ticks to read on encode and write on decode, as many bytes as the values:
on 4 threads, where memory bandwidth is shared, that costs 12–18% (docs/PERFORMANCE.md).

**Design notes** (measured while designing, on the timestamps above):
- The GCD matters: without it, µs or ms data stored in ns ticks costs 1.4–2.6× more (zstd alone
  doesn't find the grid). With it, the unit's resolution doesn't change the size.
- Plain deltas beat delta-of-deltas and residuals from the nominal grid overall: the grid
  residual saves about 1 bit/sample on jitter around a fixed grid but loses on gaps, drift and
  events. Delta-of-deltas measured worse again with the reference below (for example 2,682 bytes
  against 1,489 for 1% dropped samples).
- The per-block reference: bit planes of raw quotients around a center like 1000 waste about a
  bit per sample, because a spread of ±60 flips planes 3–10 together (and crossing 1024 flips
  more), and each plane is coded on its own. Residuals from the reference cut jitter by 4–16%,
  dropped samples by half and a drifting clock by 80%, and cost about 15 bytes per unit with gaps
  (the `time_ref` column). The mean alone lost 4–17% on skewed deltas, and the median matched
  the mean-or-minimum choice at the cost of a selection per block; the variance rule matched a
  per-block best of both on every shape measured.
- 32 planes, with 64 for long blocks only: zstd scans a run of zero bytes at about 9 GB/s, so
  64 planes per block spent 50–70 µs per irregular unit compressing the dead planes 32–63. Fewer
  planes lose size instead: this libzstd (1.5.7) compresses these bodies 1–3% better above 256 KB
  (splitting blocks where the statistics change), and a unit-wide plane count or per-block widths
  drop most bodies below it. At 32 planes a 60-block unit stays at about 360 KB and its size is
  unchanged; at 24 it drops to 300 KB and some shapes grew 1.6–1.8%.
- Bit planes and byte planes for the residuals are within a few percent either way; bit planes
  match the value residuals.
- Block starts are stored as unsigned increases (monotonic by requirement), which costs 45
  bytes per regular unit against 280 raw. The value anchor is not transformed: XOR with the
  previous anchor (+0.1% overall) and zigzagged integer deltas (−0.03%) were both noise.

## 3. Encoding one block

### 3.1 Power-of-two exponent

```
rng = hi - lo
if rng > 0:
    k  = frexp(rng).exponent            # rng in [2^(k-1), 2^k)
    e  = k - B
    if floor(ldexp(rng, -e) + 0.5) >= 2^B:  e += 1     # max would round up to 2^B
else:
    e  = 0                               # constant block
e  = min(max(e, −1074), 1023)            # 2^e is a double: the smallest subnormal .. 2^1023
ps = 2^e                                 # the power-of-two step
```

- **Wide blocks** (rng ≥ 2^1023, possibly inf): compute e from `rh = hi/2 − lo/2` (halving is
  exact) as e(rh) + 1. That is the same e the exact range gives.
- **The clamps.** At e = −1074 every double is on the grid, so a block whose finest step would be
  finer encodes losslessly. At e = 1023 (only reachable with few bits on a range near 2^1024),
  q stays below 3.

The encoder computes e_fine with B = `max_quantize_bits` and e_coarse with B = `min_quantize_bits`.

### 3.1a Noise floor (if `noise_floor_sigma` = f is set and rng > 0)

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
- **On gated blocks max_quantize_bits stops mattering** whenever f·σ exceeds the finest step, which it does for
  f ≥ 0.01 on the measured noisy signals at B = 16. f alone sets their size, down to the
  `min_quantize_bits` clamp: a noisy block's noise-floor step spans range/(f·σ) steps (≈ 230, about
  7.8 bits, on the noisy sine at f = 0.25), so a minimum above that would override the noise floor.
  Six bits costs ≤ 1% on the noisy test signals.
- **The gate is per block and all-or-nothing.** Per-sample variants (full precision kept at
  spikes) cost 20–35% more for an error already below the noise on the spike ([REPORT.md §11](../experimental/REPORT.md#11-power-of-two-quantization-the-fluxcode-design)).

### 3.2 Decimal detection (if `decimal_detection` is on and rng > 0)

Runs after the noise floor, with `ps` the (possibly coarsened) step. A decimal grid finer than
`ps` is never used: when exact decimals would need more than B bits, or are finer than the
noise, the block takes the power-of-two grid.

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
  That is what makes reconstruction bit-exact (§4).
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

### 3.3 Quantize

```
power of two:  q[i] = floor((x[i] - lo) · 2^-e + 0.5)
decimal:       K0   = floor(lo · 10^-p + 0.5)
               q[i] = floor(x[i] · 10^-p + 0.5) - K0
```

The power-of-two formula, evaluated so no intermediate overflows (one choice per block; each form
gives the same q wherever the plain one is finite):
- **−1023 ≤ e < 1008:** as written.
- **e < −1023** (2^−e overflows): multiply by 2^1023, then by 2^(−e−1023). Both are exact.
- **e ≥ 1008** (x − lo may overflow): `floor((x/2 − lo/2) · 2^(1−e) + 0.5)`. Halving is exact, so
  x/2 − lo/2 rounds exactly as (x − lo)/2.

Decimal detection is skipped for wide blocks: no 10^p grid with p ≤ 22 spans 2^1023.

In both modes 0 ≤ q[i] < 2^max_quantize_bits ≤ 2^16, and q = 0 at the minimum.
When decimal detection succeeds, its candidate pass has already computed these q; they are reused.

**The block's anchor** is what a decoder adds q to: the float64 `lo` on the power-of-two grid, or
the integer `K0` on a decimal grid (|K0| < 2^52, since decimal detection requires
max(|lo|, |hi|)·10^−p < 2^52). Both define the block's values exactly (lo + q·2^e is a binary
fraction, (K0 + q)·10^p a decimal); float64 enters only at the final rounding.

### 3.4 Order

For each k in `diff_orders`, compute the variance of `diff(q[0..M−1], k)` over the first M = min(250, n) samples (M − k values, no padding):

```
v_k = (m·Σd² − (Σd)²) / m²,   d = diff(q[:M], k),   m = M − k
order = the k with the smallest v_k; ties go to the lower order
```

Sums are exact integers (int64). The order only has to be *a* valid choice: the decoder reads it
from the unit, so encoders may pick it differently. Picking on the first 250 samples agrees with
the full-block pick on 97% of blocks and costs 0.4% in size on continuous data (nothing on
discretized data).

### 3.5 Residuals

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
so integrating mod 2^16 (§4) recovers q exactly. Two bytes per sample always suffice. The
prepended zeros make the first residuals the start values, so there's no separate header.

### 3.6 Per-unit target (if `target_bits_per_sample` = t is set)

After every block of the unit has been through §3.1–3.5, estimate each block's size from its
residual, the **class entropy**: with L(u) the bit length of the zigzagged residual (0 for 0),

```
h_b = −Σ_L p_L·log2 p_L  +  Σ_L p_L·max(L − 1, 0)      # entropy of L, plus the bits below the leading 1
```

If mean(h_b) ≤ t, nothing changes. Otherwise:

```
e_b     = the block's exponent (for a decimal block, floor(log2 10^p): its grid is already that coarse)
give_b  = min(round(max(h_b − 1, 0)), e_coarse_b − e_b)        # whole bits block b can give
k       = the smallest k ≥ 1 with Σ_b min(k, give_b) ≥ Σ_b h_b − t·N   (k ≤ 16)
k_b     = min(k, give_b)
blocks with k_b > 0: redo §3.2–3.5 at e_b + k_b; keep the result only if its h_b went down
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
- In `update`, the cap applies to the updated blocks only, with a budget of t × k.

## 4. Decoding one block

```
v = (u >> 1) XOR −(u & 1)                   # un-zigzag
repeat `order` times:  prefix-sum v, each partial sum mod 2^16     # q
order 0: q = v mod 2^16
power of two: y[i] = lo + q[i] · 2^e, rounded once, clamped to the largest finite double   (lo: the anchor)
decimal:      p < 0:  y[i] = float64(K0 + q[i]) / 10^-p   (K0: the anchor; integer divided by an exact power of ten)
              p >= 0: y[i] = float64(K0 + q[i]) · 10^p
```

Because IEEE division is correctly rounded, the decimal reconstruction is the same double that
parsing the decimal text (e.g. "12.345") produces.

**The clamp** only matters for e ≥ 971. There, a sample near the largest double can reconstruct
up to half a step above it, which would round to inf; clamping it moves it closer to the original.
Below 971, `lo + q·2^e` can't overflow (the excess is under half an ulp of the largest double),
and decoders compute it directly. From 971 up, `q·2^e` itself can overflow while the sum doesn't,
so evaluate at half scale: `y = min(2·(lo/2 + q·2^(e−1)), DBL_MAX)`, with `y = lo` where q = 0
(lo/2 rounds if lo is subnormal). That is bit-identical to `lo + q·2^e` wherever the latter is finite.

**Time axis** (header time unit ≠ 0): block starts are the running sum of the `time_start` field
(§5). Each sample i > 0 has `quotient[i] = time_ref + unzigzag(residual[i])` mod 2^64 (residuals
are 0 in a regular block, block flag bit 4 clear), and
`time[i] = start + time_step · (quotient[1] + … + quotient[i])`. A regular block is therefore
`time[i] = start + i · time_ref · time_step`. All arithmetic is exact in 64 bits; a decoder rejects
a unit whose times would exceed int64 maximum (§5).

**Non-finite codes** (head bit 3): after dequantizing, samples with code 01, 10 or 11 are
overwritten with NaN (the canonical quiet NaN), +inf or −inf. Groups of 8 samples whose two code
bytes are both 0 are skipped.

## 5. Unit format

A unit is a 16-byte header followed by one zstd frame holding the body (level 3, content size
recorded, no checksum). The header is uncompressed, so a decoder can validate it before
decompressing:

| offset | header field | type (little-endian) | contents |
|---|---|---|---|
| 0 | version | uint8 | 1 (a decoder rejects others) |
| 1 | flags | uint8 | bit 0: `residual_planes` are byte planes instead of bit planes; bits 1–3: `time_unit`, 0 for no time axis, 1 s, 2 ms, 3 µs, 4 ns (5–7 rejected); bits 4–7: 0 (rejected otherwise) |
| 2–3 | reserved | uint16 | 0 (a decoder rejects others) |
| 4–7 | `block_len` | uint32 | block length: a multiple of 8, 8 to 65,536 |
| 8–12 | `num_samples` | uint40 | sample count, 1 to 2^26 (§2). The unit holds `num_blocks` = ceil(`num_samples` / `block_len`) blocks |
| 13–15 | reserved | 3 bytes | 0 (a decoder rejects others): the top 3 bytes of a uint64 whose low 40 bits are `num_samples` |

Counts used below: `num_nonfinite_blocks` blocks have block flag bit 3 set,
`num_irregular_blocks` have bit 4 set and `num_long_blocks` have bit 5 set. The body is these fields, in this order; the four time
fields are present only when `time_unit` ≠ 0:

| field | size in bytes | contents |
|---|---|---|
| `block_flags` | `num_blocks` | bits 0–1: order; bit 2: decimal mode; bit 3: non-finite codes present (§2); bit 4: irregular times, with time residual planes (§2a; rejected when `time_unit` = 0); bit 5: long time residuals, 64 planes instead of 32 (rejected without bit 4); bits 6–7: 0 (rejected otherwise) |
| `grid_params` | 8 × `num_blocks` | the block's grid parameter as int64: the power-of-two exponent (−1074 to 1023) or, in decimal mode, the decimal power (−22 to 22). Byte-planed: byte 0 of every block, then byte 1 of every block, … byte 7. Eight bytes leave room for other grid parameters; the unused byte planes compress to almost nothing |
| `value_anchor` | 8 × `num_blocks` | the block's anchor (§3.3), byte-planed like `grid_params`: the bits of the float64 block minimum (finite; 0.0 for a block with no finite samples), or in decimal mode the int64 decimal grid index of the minimum (magnitude < 2^52) |
| `time_start` | 8 × `num_blocks` | block 0: its start time as int64; block b > 0: its start minus block b − 1's start, as uint64. Byte-planed |
| `time_step` | 8 × `num_blocks` | the block's time step as int64, byte-planed: the GCD of its deltas (≥ 0; ≥ 1 in an irregular block) |
| `time_ref` | 8 × `num_blocks` | the block's reference quotient as uint64, byte-planed: 1 in a regular block (0 if its times are all equal); in an irregular block, the value the residuals are taken from (§2a) |
| `residual_planes` | 2 × `num_blocks` × `block_len` | flags bit 0 clear: bit plane j = 0..15, then block, then byte i = 0..`block_len`/8 − 1. Bit k of byte i (LSB = bit 0) is bit j of `u[8i + k]`. Flags bit 0 set: byte plane j = 0..1 (low, then high byte of `u`), then block, then sample |
| `nonfinite_code_planes` | 2 × `num_nonfinite_blocks` × `block_len` / 8 | code plane j = 0..1, then the flagged blocks in block order, then byte i. Bit k of byte i is bit j of the code of sample 8i + k |
| `time_residual_planes` | 32 × (`num_irregular_blocks` + `num_long_blocks`) × `block_len` / 8 | per sample of an irregular block, the uint64 residual: `zigzag(quotient − time_ref)` mod 2^64, with the quotient `(time[i] − time[i−1]) / time_step`; 0 for sample 0. First bit plane j = 0..31, then the irregular blocks in block order, then byte i; then bit plane j = 32..63, then the long blocks in block order, then byte i. Bit k of byte i is bit j of the residual of sample 8i + k; a short block's residuals are below 2^32 |

```
body_size = 17 × num_blocks + 2 × num_blocks × block_len + num_nonfinite_blocks × block_len / 4
          + (time_unit ≠ 0) × (24 × num_blocks + 4 × (num_irregular_blocks + num_long_blocks) × block_len)
```

A decoder checks the frame's recorded content size against the header before decompressing (it
must lie between the sizes with no flagged and all blocks flagged), then the exact size once
`block_flags` give the counts. It then checks each block's flags, grid parameter and value anchor
ranges, and, for a time axis, rejects:
- a first start of int64 minimum, or a running sum of `time_start` above int64 maximum;
- `time_step` < 0, or `time_step` = 0 on an irregular block;
- an irregular block whose first residual isn't 0, or a long block whose planes 32–63 are all zero
  (its residuals fit in a short block);
- any time above int64 maximum.

A unit with no flagged blocks has an empty `nonfinite_code_planes` field, and one with only
regular blocks an empty `time_residual_planes` field.

**Test vector for the bit order:** a block whose `u[3] = 0x0020` and `u[6] = 0x0400` (all others 0)
has byte 0 of plane 5 = `0b00001000` and byte 0 of plane 10 = `0b01000000`. All other plane bytes are 0.

**Worked example with a time axis** (`tests/test_time.py::test_worked_example_layout`):
`block_len` = 8, `num_samples` = 20, so `num_blocks` = 3; ns ticks (small, for readability).

```
times   block 0: 1000 1010 1020 1030 1040 1050 1060 1070   regular
        block 1: 1080 1090 1100 1130 1140 1150 1160 1170   irregular: a gap after 1100
        block 2: 1180 1190 1200 1210 + 4 padding samples   regular: padding extends the grid, 1220 … 1250
values  constant per block: 2.5, 2.25, 3.0   (power-of-two mode)
```

Header: `01 08 00 00 08 00 00 00 14 00 00 00 00 00 00 00` (flags 0x08: `time_unit` 4 in bits 1–3).
Body: 203 bytes = 17 × 3 + 2 × 3 × 8 + 24 × 3 + 4 × 1 × 8.

```
offset   field                size   byte planes (3 bytes each: blocks 0, 1, 2)
0–2      block_flags          3      bit 4 set on block 1 only
3–26     grid_params          24
27–50    value_anchor         24     raw bits 0x4004…, 0x4002…, 0x4008…: planes 0–5 = 00 00 00,
                                     plane 6 = 04 02 08, plane 7 = 40 40 40
51–74    time_start           24     stored 1000, 80, 100: plane 0 = E8 50 64, plane 1 = 03 00 00,
                                     planes 2–7 = 00 00 00
75–98    time_step            24     10, 10, 10: plane 0 = 0A 0A 0A, planes 1–7 = 00 00 00
99–122   time_ref             24     1, 1, 1: plane 0 = 01 01 01, planes 1–7 = 00 00 00
123–170  residual_planes      48
171–202  time_residual_planes 32     block 1 (short: planes 0–31 only), quotients [1, 1, 3, 1, 1, 1, 1]
                                     from sample 1: minimum 1 (3σ² > (mean − min)²), residuals
                                     [0, 0, 0, 2, 0, 0, 0, 0], zigzagged [0, 0, 0, 4, 0, 0, 0, 0]:
                                     plane 2 = 0x08, all other planes 0x00
```

## 6. Guarantees

- **Max error ≤ half the step used = 2^(e−1)** (up to one rounding of the sample's magnitude).
  - **Always ≤ range / (2^min_quantize_bits − ½)**: range/63.5, 1.6% of the block's range, at the
    default of 6, whatever the noise floor or the target does. The ½ is from the exponent bump in §3.1.
  - Blocks at `max_quantize_bits` (neither noise floor nor target applied): ≤ range / (2^max − ½),
    about 2^−max of the block's range (0.0015% at 16).
  - Noise-floor blocks: ≤ f·σ/2, in the signal's units. This is not bounded relative to the
    range: a block that is all noise has a range of only a few σ.
  - Decimal mode is exact, or within `tol` = 2^e/4 if the samples were within tolerance of the
    grid.
- **One zero and one NaN.** −0.0 decodes as +0.0 (equal as floats, sign bit lost), and every NaN
  decodes as the canonical quiet NaN (payload and sign lost). These are the exceptions to
  bit-exactness for data on the grid. ±inf round-trip exactly.
- **min is exact in power-of-two mode.** In decimal mode, min decodes onto the decimal grid. That's
  exact for decimal data, but a min carrying float noise (e.g. 1000.0000000000291 on an
  integer grid) decodes to the grid value (1000.0): an error within `tol`, as for every
  sample. max decodes within half a step.
- **Decimal data is bit-exact:** values that are decimals of p places (as parsed from text)
  decode to the identical float64. That holds when the block's range spans fewer than 2^B
  decimal steps; wider blocks use the power-of-two grid.
- **float32 rounding artifacts are intentionally lost.** A decimal stored as float32 upstream
  decodes as the decimal itself.
- **Decoded data is a fixed point:** decode(encode(y)) = y for any decoded y. The unit bytes are
  identical from the second encode on.
- **Timestamps are exact.** Given times decode to the identical int64 ticks, in the same unit.
  Decoded times are non-decreasing within every block, and block starts never decrease (§2a).
- **Edits are stable.** Changing a sample moves the grid only when the range crosses a power of
  two. (With the grid scaled to the range, a new max re-rounds every sample: in testing, 95% of
  untouched samples changed, against 3% here.)
- **`update` leaves other blocks untouched.** Replacing or appending whole blocks of a unit
  carries the other blocks' flags, grid parameter, value anchor and residuals over unchanged, so
  they decode to identical values. With a time axis, `update` takes the new blocks' times (required
  exactly when the unit has one), decodes the other blocks' times exactly and re-derives every
  block's time rows, which reproduces the untouched blocks' rows byte for byte; the updated series
  must be non-decreasing throughout. Without a target, the updated unit is byte-identical to encoding the updated
  series from scratch (every block is encoded independently). The sample count follows the blocks:
  a partial last block stays partial until it is replaced by a full one, and blocks can be
  appended only after a full last block.

## 7. Conformance tests

1. **Round trip:** decode(encode(x)) has max error ≤ 2^(e−1) for every block (≤ f·σ/2 on
   noise-floor blocks) and ≤ range / (2^min_quantize_bits − ½) always; min is exact in power-of-two mode.
2. **Decimal:** values generated as `K / 10^d` with a range under 2^B steps decode
   bit-identical.
3. **Fixed point:** y = decode(encode(x)) satisfies decode(encode(y)) = y,
   and encode(y) is byte-identical from the second encode on (§6).
4. **Bit order:** the test vector in §5.
5. **Order independence:** a unit written with any `diff_orders` setting decodes with the same decoder.
6. **Edge cases:** constant block (all q = 0, decodes to lo exactly); range just below a power
   of two (exponent bump); a block whose range exceeds 2^B decimal steps (falls back to the
   power-of-two grid); fewer than 60 blocks; a short last block; units of different block counts
   and block lengths.
   **Self-describing units:** decode_unit needs only the unit and returns exactly `num_samples`
   samples; headers with another version, reserved flag bits, a time unit of 5–7, nonzero reserved
   bytes (including the top 3 bytes of the sample count), an invalid `block_len` or `num_samples`, or a body
   size that doesn't fit them are rejected, as are non-finite float anchors and decimal anchors
   of 2^52 or more.
7. **Update:** untouched blocks decode identically; without a target, `update` is byte-identical
   to encoding the updated series.
8. **Non-finite:** NaN / ±inf runs at a block's start, middle and end, scattered, alternating,
   and whole blocks, on every signal kind and with the target: exact NaN positions and infinities,
   finite samples within the bounds, summary statistics over the finite samples; blocks without
   non-finite values byte-identical to units without the field; `update` keeps existing codes.
9. **Limits:** min/max_quantize_bits are never violated, with or without the noise floor and target;
   the noise gate fires on white noise at 1e±300 and across ±DBL_MAX as at 1; ranges from a few
   subnormals to past 2^1023 round-trip within the bounds, with no inf, min exact, and exponents
   matching exact rational arithmetic.
10. **Noise gate:** it fires on white Gaussian noise (with or without a slow signal under it)
   and never on a random walk, a clean sine at any frequency, a chirp, a ramp or a square wave.
   It may fire on the float rounding noise of a smooth polynomial (the quadratic), with no
   effect, since that σ is far below the B-bit step. On all clean test signals the unit is
   byte-identical with the noise floor on (f = 0.01–1) and off.
11. **Time axis:** datetime64 in s, ms, µs and ns, and integer ticks with a time unit, round-trip
   exactly and in their unit; units without times decode `times = None`; regular series store no
   planes; a gap makes only its block irregular, with the GCD as its step; jitter takes the rounded
   mean as its reference and skewed deltas the minimum; any reference decodes; only a block whose
   residuals reach 2^32 is long; equal timestamps,
   all-equal blocks, short last blocks, ticks at both ends of int64 and a block spanning more than
   half of it round-trip; the worked example in §5 matches byte for byte. The encoder rejects
   decreasing times (within and across blocks), NaT, unsupported dtypes and units, and length
   mismatches; the decoder rejects each corrupt time field listed in §5, and the long flag without the irregular one. `update` with times is
   byte-identical to encoding the edited series, and rejects missing, unexpected, mis-shaped,
   wrong-unit or out-of-order times.

## References

Background for the techniques above (informative; the sections above are normative).

- Quantization: https://en.wikipedia.org/wiki/Quantization_(signal_processing)
- Fixed polynomial predictors (§3.4–3.5, §4): the fixed predictors of
  [Shorten](https://en.wikipedia.org/wiki/Shorten_(file_format)) and FLAC,
  [RFC 9639 §9.2.5](https://www.rfc-editor.org/rfc/rfc9639#name-fixed-predictor-subframe).
- Zigzag encoding of signed integers (§3.5):
  [Protocol Buffers encoding](https://protobuf.dev/programming-guides/encoding/#signed-ints).
- Bit-shuffle (§5): [bitshuffle](https://github.com/kiyo-masui/bitshuffle); K. Masui et al.,
  [arXiv:1503.00638](https://arxiv.org/abs/1503.00638).
- Zstandard (§5): [RFC 8878](https://www.rfc-editor.org/rfc/rfc8878).
- Noise estimate (§3.1a): [mean absolute deviation](https://en.wikipedia.org/wiki/Average_absolute_deviation)
  and [autocorrelation](https://en.wikipedia.org/wiki/Autocorrelation).
- Decimal detection (§3.2) is related to ALP: A. Afroozeh, L. Kuffó, P. Boncz,
  [SIGMOD 2024](https://doi.org/10.1145/3626717).
- IEEE 754 binary64 and correctly rounded division (§4):
  [double-precision floating-point format](https://en.wikipedia.org/wiki/Double-precision_floating-point_format).
