# fluxcode — Format Specification

The **fluxcode** block group format stores `float64` numeric series in self-contained blocks, with optional exact `int64` timestamps.

- **Section 0** is an informative overview of how fluxcode works and why each design choice was made.
- **Sections 1–8** are the normative specification of the binary layout, decoding rules, validation requirements, and conformance criteria.
- How the reference encoder chooses grids, anchors, predictor orders, and error bounds is documented in [ENCODER.md](ENCODER.md).

---

## 0. How fluxcode works (Overview & Design Rationale)

*This section is informative. Sections 1–8 are normative.*

A **block group** is a single self-describing binary blob (for example, one database row or object-store chunk) that holds up to 65,535 variable-sized **blocks** of `float64` samples, optionally paired with exact `int64` timestamps. Each block adapts independently to its local signal—choosing its own quantization grid, polynomial predictor order, exception mask, and time cadence—while the group packs all blocks column-by-column into one Zstandard frame so shared structure compresses across blocks.

### Pipeline overview

```text
ENCODE (per block b)                                   BODY LAYOUT (columnar, across all blocks)
────────────────────                                   ─────────────────────────────────────────
float64 samples y[0..n_b-1]                            Per-block metadata (byte-planed):
  │                                                      block_flags, block_sizes,
  ├─► Separate NaN / ±inf / outliers                     grid_exponents, grid_scales,
  │     └─► 2-bit sample_code_planes (00/01/10/11)       block_anchors,
  │         + escaped_values (float64 trailer)           [time_starts, time_steps, time_ref_quotients]
  │                                                    Per-sample planes (8-sample groups):
  ├─► Quantize on-grid samples to signed 16-bit          residual_planes         (16 bit or 2 byte planes)
  │   grid index q ∈ [-32768, 32767]                     sample_code_planes      (2 bit planes, flagged blocks)
  │   (binary 2^e, decimal 10^p, or scaled B^p/D)        time_residual_planes    (32 or 64 bit planes, irregular)
  │                                                    Trailers:
  ├─► Order-k polynomial differencing (k ∈ 0..3)         escaped_values          (float64, byte-planed)
  │   wrapping mod 2^16 → int16 residual v               [group_time_step]       (uint64, if header bit 4 set)
  │                                                                │
  ├─► Zigzag encode v → uint16 u                                   ▼
  │                                                      ┌───────────────────┐
  └─► Transpose u into 16 bit planes or 2 byte planes    │  Zstandard frame  │
      in 8-sample groups (ceil(n_b / 8) bytes/plane)     └─────────┬─────────┘
                                                                   ▼
                                                       [8-byte header] + [zstd frame]

DECODE: Validate header → check content-size bounds → zstd decompress → validate metadata & sizes
        → un-transpose planes → un-zigzag → k-fold prefix sum mod 2^16 → int16 q → dequantize
        → overlay sample_code_planes / escaped_values → reconstruct exact int64 timestamps.
```

### Design rationale

1. **Independent block groups of variable-sized blocks.**
   Block groups share no state with one another, enabling random access and partial updates at the group level without decoding untouched data. Within a group, blocks can vary freely in length (0 to 65,535 samples) so an encoder can align blocks to natural boundaries (e.g., hourly segments, regime changes, or gaps) and tailor each block's grid and predictor to local behavior. Because the header and per-block metadata are self-contained, a decoder never needs an external index or length sidecar.

2. **Fixed 16-bit grid indices with modular `2^16` polynomial differencing.**
   A `k`-th order finite difference of 16-bit signed integers normally requires up to `16 + k + 1` bits (up to 20 bits for order 3). Fluxcode avoids variable-width residuals by constraining every block's quantized grid index `q` to `[-32768, 32767]` (`int16`). When `q` is known to fit in 16 bits, wrapping each difference modulo `2^16` and applying `k` prefix sums modulo `2^16` on decode recovers `q` **losslessly**. Every predictor order (0–3) therefore produces uniform 16-bit residuals, and prepending `k` implicit zeros folds the startup values into the same residual stream without a separate predictor header.

3. **Three quantization grid families: binary, exact decimal, and scaled rational.**
   Numeric series arise from distinct physical and software sources, and a single power-of-two grid cannot represent all of them cleanly:
   - **Power-of-two (`a + q·2^e`)** handles native binary floats, arbitrary constant blocks, and lossy quantization to a binary step.
   - **Exact decimal (`(K0 + q)·10^p`)** handles decimal telemetry (such as `"12.345"`), whose `float64` values are not exact multiples of any power of two. Because IEEE-754 division is correctly rounded, dividing an exact integer by an exact power of 10 reproduces the *exact* `float64` bit pattern of the parsed decimal text.
   - **Scaled / rational (`(K0 + N·q)·B^p / D`)** handles ADC counts (e.g., `10 V / 4095`), affine unit conversions (e.g., `°C → °F`, `32 + 0.18·k = (1600 + 9k)·0.1 / 5`), and non-power-of-ten steps like `0.05` (`N = 5, D = 1, p = -2`) with a single IEEE-754 rounding.

4. **Bit planes vs. byte planes, and byte-planed metadata columns.**
   Zigzagged residuals cluster near zero, so their upper bits are overwhelmingly zero. Transposing residuals into 16 bit planes (8 samples per byte) groups those zero bits into contiguous all-zero planes that Zstandard compresses to a few bytes. When residuals are wider and bit-level transposition gives little benefit over byte-level grouping, header flag bit 0 switches `residual_planes` to 2 byte planes (low byte, then high byte), reducing transpose work while still grouping zero high bytes. The same principle applies to every multi-byte per-block metadata column (`block_sizes`, `grid_exponents`, `grid_scales`, `block_anchors`, and the time columns): storing byte 0 of all blocks, then byte 1 of all blocks, and so on lets Zstandard collapse shared high bytes (such as `0x40` float exponents or all-zero `grid_scales` when no block is scaled) while keeping every column's byte offset a simple function of `num_blocks`.

5. **2-bit exception codes, run-length repeat, and `escaped_values`.**
   A single `NaN`, `±inf`, sentinel error code (such as `-9999`), or rare outlier would otherwise force a block to widen its quantization step by orders of magnitude. Fluxcode pulls exceptional samples off the grid using 2-bit per-sample codes (`00` on-grid, `01` NaN, `10` escaped `float64`, `11` repeat previous escaped value). Runs of an identical sentinel store the `float64` only once, and blocks without exceptions pay zero bytes for exception codes.

6. **Exact timestamps with zero per-sample overhead on regular clocks.**
   Real-world time series are usually sampled on a regular cadence, occasionally with jitter or gaps. Each block factors out the GCD of its inter-sample deltas (`time_steps`) and subtracts a per-block reference quotient (`time_ref_quotients`). Regular blocks have all-zero residuals and store **0 bytes per sample**. Irregular blocks store zigzagged 64-bit residuals in 32 bit planes—or 64 bit planes only on wide-time blocks that actually need them. Finally, when an entire group shares a coarse tick multiple (for example, whole-second timestamps stored in nanosecond ticks, as in pandas), an optional 8-byte `group_time_step` trailer divides that common factor out of `time_starts` deltas and `time_steps`, zeroing their high byte planes.

---

## 1. Block groups and blocks

- **A block group is one storage artifact** (e.g., a database row): blocks encoded and decoded together.
  - Block groups are independent of each other.
  - A block group holds `0` to `65,535` blocks.
- **A block holds `0` to `65,535` samples.** Each block's sample count is recorded in `block_sizes` (§6.3).
- **Decoding needs only the block group.** No external index column or length sidecar is required:
  - the 8-byte header records `num_blocks` and `num_samples`;
  - the body records each block's size, flags, grid parameters, and anchor.
- **Nothing is padded at the sample level:** each block decodes to exactly its `n_b` samples.
  - Bit planes store each block in whole bytes (8-sample groups): a block whose sample count is not a multiple of 8 pads the trailing bits of its last byte with zeros (§6.2).
- **The body is stored column-by-column (field by field)** (§6.3). Each field holds every participating block's bytes in block order.
- **An empty block** (`n_b = 0`) stores no samples. Its `block_flags`, `grid_exponents`, `grid_scales`, `block_anchors`, and (if present) `time_starts`, `time_steps`, and `time_ref_quotients` entries are all `0`.
- **`num_samples` is at most `2^26` (67,108,864).**
  - This is a normative safety bound, not a wire-field limit (`num_samples` is a `uint32` holding up to `2^32 - 1`).
  - Decoders use it to reject implausible headers before allocating memory or decompressing.

---

## 2. Sample exception codes and escaped values

A block may code individual samples off its quantization grid using a 2-bit code per sample:

| Code (bit 1, bit 0) | Meaning |
|---|---|
| `00` | On the grid (use dequantized value) |
| `01` | `NaN` (canonical quiet NaN) |
| `10` | Escaped: consume the block's next `float64` from `escaped_values` |
| `11` | Repeat the block's previous escaped value |

- **Escaped values** are raw IEEE-754 `float64` values: `±inf`, or finite outliers that an encoder chooses to store apart (e.g., an error code or overrange reading) so the rest of the block keeps a fine grid ([ENCODER.md §5.1.1](ENCODER.md)). A run of identical escaped values stores the `float64` once (code `10`) followed by repeats (code `11`).
- **`escaped_values`** (§6.3) holds every block's escaped values in block order, then sample order. Its element count `E` is the total number of `10` codes across all blocks' actual samples (excluding padding bits).
- **Per-block opt-in:** only blocks containing at least one non-`00` sample code set `block_flags` bit 3 and contribute to `sample_code_planes` (§6.3), taking `2 × ceil(n_b / 8)` bytes (`≈ n_b / 4` bytes, mostly zero before compression).
- **Placeholders:** a block's grid residuals at non-`00` sample positions are placeholders that are overwritten when codes are applied during decode (§5).

---

## 3. Time axis

- **Timestamps are optional.**
  - A block group without timestamps has `time_unit = 0` in its header flags and omits all time fields from its body.
  - It decodes with `times = None`.
- **Ticks and units.** Timestamps are signed `int64` ticks since `1970-01-01T00:00:00`, in one of the Arrow / NumPy `datetime64` units recorded in header flags bits 1–3 (§6.1):

  | Time unit | Code | `int64` range around 1970 |
  |---|---|---|
  | `s` | `1` | `± 2.9 × 10^11` years |
  | `ms` | `2` | `± 2.9 × 10^8` years |
  | `µs` | `3` | `± 292,000` years |
  | `ns` | `4` | `± 292` years (1677–2262) |

  - The time unit fixes only the range and semantic meaning of a tick. Storage cost does not depend on it: per-block `time_steps` (and `group_time_step`) absorb coarser grids.
  - `INT64_MIN` (`-2^63`, `datetime64`'s `NaT`) is never a valid timestamp.
- **Timestamps are naive:** the format stores no time zone. Store UTC—local time can repeat or skip an hour across DST transitions, which breaks the monotonic ordering rule.
- **Monotonic ordering is structural.** Because within-block deltas and between-block start increases are unsigned, every valid decoded block group has:
  - non-decreasing timestamps within each block;
  - non-decreasing block start times across non-empty blocks.
- **Per-block representation.** Every block has entries in `time_starts`, `time_steps`, and `time_ref_quotients`, and irregular blocks additionally contribute to `time_residual_planes` (§6.3):
  - `time_step` is the GCD of the block's inter-sample deltas `time[i] - time[i - 1]` (for `i = 1..n_b - 1`).
  - Each sample `i > 0` has an unsigned **quotient** `quotient[i] = (time[i] - time[i - 1]) / time_step`.
  - Each quotient is represented as `time_ref + residual` (mod `2^64`).
- **Regular block** (`block_flags` bit 4 clear): every delta in the block is equal (possibly `0`).
  - Every quotient equals `time_ref = 1`, so `time[i] = start + i · time_step`.
  - When all timestamps in the block are equal, `time_step = 0` and `time_ref = 0`.
  - A regular block stores **no** per-sample time data in `time_residual_planes`.
- **Irregular block** (`block_flags` bit 4 set): any block with non-uniform deltas.
  - Stores per-sample residuals `zigzag((quotient[i] - time_ref) mod 2^64)` in `time_residual_planes`, with sample `0`'s residual fixed to `0`.
  - Uses 32 bit planes when all zigzagged residuals are `< 2^32`, or 64 bit planes when any zigzagged residual is `>= 2^32`—a **wide-time block** (`block_flags` bit 5 set).
  - Any `uint64` value of `time_ref` decodes identically; how the reference encoder chooses it is in [ENCODER.md §4](ENCODER.md#4-time-axis).
  - If any delta exceeds `INT64_MAX` (a block spanning more than half the `int64` range), that delta cannot be represented as a signed `int64` GCD; such a block uses `time_step = 1`, making the raw deltas the quotients.
- **Tiny blocks:**
  - **1-sample block:** has no deltas; `time_step = 0`, `time_ref = 0`, and `block_flags` bit 4 is clear.
  - **2-sample block:** always regular (bit 4 clear). If its single delta exceeds `INT64_MAX`, the delta is stored in `time_ref_quotients` with `time_step = 1`.
- **Empty blocks** have `0` in `time_starts`, `time_steps`, and `time_ref_quotients`, and are skipped in the running-sum chain of `time_starts`: each non-empty block after the first stores its start minus the previous *non-empty* block's start.
- **Group time step (`group_time_step`, header flags bit 4):**
  - When header flags bit 4 is set, the body ends with an 8-byte `uint64` trailer `q >= 2` (after `escaped_values`).
  - Every stored start increase (every non-empty block's `time_starts` entry after the first non-empty block) and every block's `time_steps` entry is stored divided by `q`; the decoder multiplies them back by `q`.
  - This prevents coarse timestamps stored in fine ticks (such as whole seconds in pandas' default `ns` ticks) from filling all 8 byte planes of `time_starts` and `time_steps`—for example, reducing a sparse tag's daily group of hourly 2-point blocks from 577 to 473 bytes.

---

## 4. Anchors, grid indices, and modular residuals

### Grid indices and anchors

Each on-grid sample in a block is represented by a signed 16-bit **grid index** `q` with `-32,768 <= q <= 32,767` (`-2^15 <= q < 2^15`), relative to the block's **anchor** (`block_anchors`, §6.3). A block therefore spans at most `2^16 - 1 = 65,535` grid steps. (For order 0, the reference encoder centers the anchor near the block's mean index; for higher orders, it places the anchor at the first sample's index so the first residual is 0, as far as 16 bits allow—see [ENCODER.md §5.7](ENCODER.md#57-residuals).)

`block_flags` bits 2 and 6 select one of four grid modes:

| `bit 6` (scaled) | `bit 2` (decimal) | Grid mode | Anchor (`block_anchors`) | Exact sample value |
|---|---|---|---|---|
| `0` | `0` | **Power-of-two** | Finite `float64` `a` | `a + q · 2^e` |
| `0` | `1` | **Decimal** | `int64` `K0` (`\|K0\| < 2^52`) | `(K0 + q) · 10^p` |
| `1` | `0` | **Scaled power-of-two** | `int64` `K0` (`\|K0\| < 2^52`) | `(K0 + N · q) · 2^e / D` |
| `1` | `1` | **Scaled decimal** | `int64` `K0` (`\|K0\| < 2^52`) | `(K0 + N · q) · 10^p / D` |

- Here `e` or `p` comes from `grid_exponents`, and `N`, `D` come from `grid_scales` (§6.3).
- On a **scaled grid**, `K0` is an integer index on the **fine grid** `B^p / D` (`B = 10` if bit 2 is set, else `B = 2`), and the block's grid step is `N` fine-grid points. `K0` need not be a multiple of `N`: for example, `32 + 0.18·k` (°F converted from `0.1 °C`) has `K0 = 1600 + 9c` and `N = 9` on the fine grid `0.1 / 5` (`p = -1, D = 5`). With `D = 1`, a scaled grid is `N` times a decimal or binary grid (e.g., a `0.05` grid is `N = 5, D = 1, p = -2`).
- Each mode defines the block's values as exact mathematical numbers (dyadic rationals, decimals, or general rationals); `float64` rounding occurs only at the final reconstruction step (§5).
- **Writer freedom:** the format does not constrain how an encoder chooses the grid mode, exponent, scale, predictor order, or anchor. A decoder does not verify that an anchor sits on the grid; any valid anchor decodes deterministically.

### Modular `2^16` residuals

With `k ∈ {0, 1, 2, 3}` the block's predictor order (`block_flags` bits 0–1), `residual_planes` stores `u`, the zigzagged modular `k`-th differences of `q`:

```text
# Prepend k zeros to q[0..n-1], take the k-th finite difference, and keep the last n values:
r = k-th difference of (0, ..., 0, q[0], ..., q[n-1])
    # order 0: r[i] = q[i]
    # order 1: r[0] = q[0],  r[i] = q[i] - q[i-1]
    # order 2: r[0] = q[0],  r[1] = q[1] - 2*q[0],
    #          r[i] = q[i] - 2*q[i-1] + q[i-2]
    # order 3: r[0] = q[0],  r[1] = q[1] - 3*q[0],  r[2] = q[2] - 3*q[1] + 3*q[0],
    #          r[i] = q[i] - 3*q[i-1] + 3*q[i-2] - q[i-3]

v = ((r + 32768) mod 2^16) - 32768          # wrap r mod 2^16 into signed int16
u = ((v << 1) XOR (v >> 15)) & 0xFFFF       # zigzag encode int16 -> uint16
```

*(Above, `v >> 15` is an arithmetic right shift of signed 16-bit `v`.)*

Because every `q[i]` lies in `[-2^15, 2^15)`, integrating `v` modulo `2^16` `k` times and reinterpreting the result as signed `int16` (§5) recovers `q[i]` exactly—even though an unwrapped `k`-th difference could need up to `16 + k + 1` bits. Prepending `k` zeros encodes the startup values directly in `r[0..k-1]`, eliminating any separate predictor-state header.

---

## 5. Decoding one block

### 1. Reconstruct grid indices `q`

From the block's `uint16` residual values `u[0..n_b-1]` unpacked from `residual_planes` (§6.3):

```text
v[i] = (u[i] >> 1) XOR -(u[i] & 1)                    # un-zigzag uint16 -> int16 (mod 2^16)
repeat `order` times:
    v = prefix_sum(v) mod 2^16                         # running sum, each partial sum mod 2^16
q[i] = ((v[i] + 32768) mod 2^16) - 32768               # reinterpret as signed int16: -32768 <= q[i] < 32768
```

### 2. Dequantize `q` to `float64`

Depending on the block's grid mode (`block_flags` bits 2 and 6):

```text
power-of-two:         y[i] = a + q[i] · 2^e, rounded once, clamped to [-DBL_MAX, +DBL_MAX]
decimal (p < 0):      y[i] = float64(K0 + q[i]) / 10^(-p)
decimal (p >= 0):     y[i] = float64(K0 + q[i]) · 10^p
scaled decimal:       y[i] = float64(K0 + N · q[i]) / (D · 10^(-p))        (p <= 0)
scaled power-of-two:  y[i] = (float64(K0 + N · q[i]) · 2^e) / D            (-1022 <= e <= 970)
```

- **Decimal single rounding:** `K0 + q[i]` is an exact integer with `|K0 + q[i]| < 2^53`, and `10^|p|` is an exact `float64` for `|p| <= 22`. Because IEEE-754 multiplication and division are correctly rounded, `y[i]` is the exact `float64` nearest `(K0 + q[i]) · 10^p`—bit-identical to parsing the decimal string (e.g., `"12.345"`).
- **Scaled single rounding:** `|K0| < 2^52` and `|N · q[i]| <= 65535 × 32768 < 2^31`, so `K0 + N · q[i]` is exact in `float64` (`< 2^53`).
  - On a **scaled decimal** grid, validation guarantees `D · 5^(-p) < 2^53` (§6.4), so `D · 10^(-p)` is an exact `float64` and the division rounds once.
  - On a **scaled power-of-two** grid, multiplying by `2^e` with `-1022 <= e <= 970` is exact (normal, non-overflowing since `2^53 · 2^970 < 2^1024`), so the division by `D` rounds once.
- **Overflow-safe clamping for power-of-two grids (`e >= 971`):**
  - The clamp to `±DBL_MAX` only matters when `e >= 971`. Near `±DBL_MAX`, a quantized sample can reconstruct up to half a step past `±DBL_MAX` and would otherwise round to `±inf`; clamping moves it closer to the original finite value.
  - For `e < 971`, `a + q · 2^e` cannot overflow (any excess is less than half an ulp of `DBL_MAX`), so decoders evaluate `a + q · 2^e` directly.
  - For `e >= 971`, `q · 2^e` itself can overflow even when `a + q · 2^e` is finite. Evaluate at half scale:
    ```text
    if q[i] == 0:
        y[i] = a                                       # avoid rounding a/2 when a is subnormal
    else:
        y[i] = clamp(2.0 · (a / 2.0 + q[i] · 2^(e - 1)), -DBL_MAX, +DBL_MAX)
    ```
    This is bit-identical to `a + q[i] · 2^e` wherever the latter is finite.

### 3. Overlay sample exception codes

If `block_flags` bit 3 is set, unpack the block's 2-bit codes from `sample_code_planes` (§6.3) and scan `i = 0..n_b - 1`:

- `00`: keep `y[i]`.
- `01`: set `y[i]` to the canonical quiet `NaN`.
- `10`: set `y[i]` to the next `float64` from `escaped_values`, and remember it as `last_escaped`.
- `11`: set `y[i]` to `last_escaped`.

Each block's escaped values immediately follow those of all preceding blocks.

### 4. Reconstruct timestamps

If `time_unit != 0` in the header:

- If header flags bit 4 is set, multiply every stored `time_starts` increase (every non-empty block after the first non-empty block) and every `time_steps` entry by `group_time_step` (checking for 64-bit overflow, §6.4).
- Compute each non-empty block's `start` as the running sum of `time_starts` across non-empty blocks (first non-empty block as signed `int64`; each subsequent non-empty block adds an unsigned `uint64` delta).
- For sample `0`: `time[0] = start`.
- For `i = 1..n_b - 1`:
  - `quotient[i] = (time_ref + unzigzag64(residual[i])) mod 2^64`, where `residual[i] = 0` on a regular block (`block_flags` bit 4 clear), and `unzigzag64(r) = (r >> 1) XOR -(r & 1)` in 64-bit unsigned/two's-complement arithmetic.
  - `time[i] = start + time_step · (quotient[1] + ... + quotient[i])`.
  - On a regular block this simplifies to `time[i] = start + i · time_ref · time_step`.
- All arithmetic is exact in 64 bits. A decoder MUST reject the block group if any intermediate or final timestamp exceeds `INT64_MAX` (§6.4).

---

## 6. Binary block group layout

A block group consists of an **8-byte uncompressed header** followed by **one Zstandard frame** containing the body:

- The header is uncompressed so a decoder can validate bounds before decompressing.
- The Zstandard frame may use any compression level and block boundaries. It MUST record its uncompressed content size in the frame header and MUST NOT include a content checksum.

### 6.1 Header

All multi-byte integers in the header and body are **little-endian**.

| Byte offset | Field | Type | Contents |
|---|---|---|---|
| `0` | `version` | `uint8` | Format version: `1` |
| `1` | `flags` | `uint8` | Bit flags (see below) |
| `2–3` | `num_blocks` | `uint16` | Number of blocks (`0` to `65,535`) |
| `4–7` | `num_samples` | `uint32` | Total sample count (`0` to `2^26`, §1): sum of `block_sizes` |

**Header `flags` byte:**

- **bit 0:** `0` = `residual_planes` uses 16 bit planes; `1` = `residual_planes` uses 2 byte planes.
- **bits 1–3:** `time_unit` (§3): `0` = no time axis, `1` = `s`, `2` = `ms`, `3` = `µs`, `4` = `ns`.
- **bit 4:** `1` = `group_time_step` 8-byte trailer is present (§3; valid only when `time_unit != 0`).
- **bits 5–7:** reserved, MUST be `0`.

### 6.2 8-sample groups, planes, and padding

- **8-sample groups.** Within every per-sample plane, each block's samples are grouped into **8-sample groups**: 8 consecutive samples pack into 1 byte of every bit plane (an 8×8 bit transpose per byte of residuals).
  - Block `b` with `n_b` samples occupies `p_b = plane_bytes(n_b) = ceil(n_b / 8)` bytes in every bit plane, and `8 × p_b` bytes in every byte plane.
  - An empty block (`n_b = 0`) has `p_b = 0`.
- **Padding.** When `n_b` is not a multiple of 8, bits `(n_b mod 8)` through `7` of the block's last byte in each bit plane are padding. Byte planes likewise pad each block to `8 × p_b` samples.
  - Writers MUST zero all padding bits and padding bytes; decoders MUST ignore them.
- **Block order within planes.** Inside any plane, participating blocks appear sequentially in ascending block index order (`0 .. num_blocks - 1`).
- **Byte-planed metadata columns.** A per-block column of `w`-byte little-endian values stores byte `0` (LSB) of every block in block order, followed by byte `1` of every block, and so on through byte `w - 1` (MSB).
- **Plane-byte sums** used in §6.3:
  - `P` = `Σ p_b` over all blocks;
  - `P_codes` = `Σ p_b` over blocks with `block_flags` bit 3 set;
  - `P_irregular` = `Σ p_b` over blocks with `block_flags` bit 4 set;
  - `P_wide` = `Σ p_b` over blocks with `block_flags` bit 5 set.

### 6.3 Body

The decompressed Zstandard frame payload contains the following fields in exact order. The four time fields (`time_starts`, `time_steps`, `time_ref_quotients`, `time_residual_planes`) are present **only** when `time_unit != 0`.

| Field | Size in bytes | Contents |
|---|---|---|
| `block_flags` | `num_blocks` | Per-block flags byte (see below) |
| `block_sizes` | `2 × num_blocks` | Sample count `n_b`, `uint16`, byte-planed |
| `grid_exponents` | `2 × num_blocks` | Grid exponent `e` or decimal power `p`, `int16`, byte-planed |
| `grid_scales` | `4 × num_blocks` | `N \| (D << 16)`, `uint32`, byte-planed (`0` on unscaled blocks) |
| `block_anchors` | `8 × num_blocks` | Anchor (§4): `float64` bits or `int64` `K0`, byte-planed |
| `time_starts` *(if `time_unit ≠ 0`)* | `8 × num_blocks` | Start time (`int64` first non-empty, then `uint64` deltas), byte-planed |
| `time_steps` *(if `time_unit ≠ 0`)* | `8 × num_blocks` | Per-block GCD step, `int64`, byte-planed |
| `time_ref_quotients` *(if `time_unit ≠ 0`)* | `8 × num_blocks` | Reference quotient, `uint64`, byte-planed |
| `residual_planes` | `16 × P` | Zigzagged residuals `u` (§4): 16 bit planes or 2 byte planes |
| `sample_code_planes` | `2 × P_codes` | 2-bit sample exception codes (§2) for blocks with bit 3 set |
| `time_residual_planes` *(if `time_unit ≠ 0`)* | `32 × (P_irregular + P_wide)` | Zigzagged time residuals (§3): planes 0–31 (irregular), then 32–63 (wide) |
| `escaped_values` | `8 × E` | Escaped `float64` values (§2), byte-planed (`E` = count of `10` codes) |
| `group_time_step` *(if header bit 4)* | `8` | Group-wide GCD time step `q >= 2` (§3), `uint64` (little-endian) |

Exact uncompressed body size:

```text
body_size = 17 × num_blocks + 16 × P + 2 × P_codes
          + (time_unit ≠ 0) × (24 × num_blocks + 32 × (P_irregular + P_wide))
          + 8 × E
          + 8 × (header flags bit 4)
```

Every per-block metadata column entry is `0` for an empty block (`n_b = 0`). Per-field specifications:

- **`block_flags`** (1 byte per block):
  - **bits 0–1:** predictor order `k ∈ {0, 1, 2, 3}` (§4);
  - **bit 2:** `1` = decimal grid, `0` = power-of-two grid;
  - **bit 3:** `1` = sample codes present in `sample_code_planes` (§2);
  - **bit 4:** `1` = irregular times, with time residuals in `time_residual_planes` (§3);
  - **bit 5:** `1` = wide time residuals (64 bit planes instead of 32; requires bit 4);
  - **bit 6:** `1` = scaled grid (combined with bit 2: scaled decimal grid);
  - **bit 7:** reserved, MUST be `0`.
- **`grid_exponents`** (`int16`, byte-planed):
  - Power-of-two grid: exponent `e`, `-1074 <= e <= 1023`.
  - Decimal grid: decimal power `p`, `-22 <= p <= 22`.
  - Scaled power-of-two grid: exponent `e`, `-1022 <= e <= 970`.
  - Scaled decimal grid: decimal power `p`, `-22 <= p <= 0`, with `D · 5^(-p) < 2^53` (guaranteeing `D · 10^(-p)` is an exact `float64`).
- **`grid_scales`** (`uint32`, byte-planed):
  - On a scaled block (`block_flags` bit 6 set): `N` in bits 0–15 (`1 <= N <= 65,535`) and `D` in bits 16–31 (`1 <= D <= 65,535`). `N` and `D` need not be coprime; any valid pair decodes.
  - On an unscaled block: `0`.
  - Stored as a fixed-width column for all blocks rather than only scaled blocks: Zstandard compresses all-zero columns to a few bytes, and every metadata column's offset remains a direct function of `num_blocks`.
- **`block_anchors`** (8 bytes per block, byte-planed):
  - Power-of-two grid: IEEE-754 `float64` bit pattern of a finite value `a`.
  - Decimal or scaled grid: signed `int64` fine-grid index `K0` with `|K0| < 2^52`.
- **`time_starts`** (8 bytes per block, byte-planed):
  - First non-empty block: its start timestamp as signed `int64`.
  - Each subsequent non-empty block: its start timestamp minus the previous non-empty block's start timestamp, as unsigned `uint64` (divided by `group_time_step` if header bit 4 is set).
- **`time_steps`** (`int64`, byte-planed):
  - GCD of the block's inter-sample deltas (divided by `group_time_step` if header bit 4 is set).
  - `>= 0` in all blocks, and `>= 1` in irregular blocks; `0` in 1-sample blocks.
- **`time_ref_quotients`** (`uint64`, byte-planed):
  - Regular block: `1`; or `0` if all timestamps in the block are equal or `n_b == 1`; or, in a 2-sample block whose single delta exceeds `INT64_MAX`, the delta itself (with `time_step = 1`).
  - Irregular block: the reference quotient from which residuals are measured (§3).
- **`residual_planes`** (`16 × P` bytes):
  - **Bit planes** (header flags bit 0 clear): for bit plane `j = 0..15` (LSB to MSB), for each block `b` in block order, for each 8-sample group byte `i = 0..p_b - 1`: bit `k` (LSB = bit `0`) of byte `i` is bit `j` of `u[8i + k]`.
  - **Byte planes** (header flags bit 0 set): for byte plane `j = 0..1` (low byte, then high byte of `u`), for each block `b` in block order, for sample `s = 0..8 × p_b - 1`: byte `s` is byte `j` of `u[s]` (`0` for padding samples `s >= n_b`).
- **`sample_code_planes`** (`2 × P_codes` bytes):
  - For code bit plane `j = 0..1` (bit 0, then bit 1 of the 2-bit code), for each block with `block_flags` bit 3 set in block order, for byte `i = 0..p_b - 1`: bit `k` of byte `i` is bit `j` of the code for sample `8i + k`.
  - Empty (0 bytes) when no block has bit 3 set.
- **`time_residual_planes`** (`32 × (P_irregular + P_wide)` bytes):
  - Per sample of an irregular block, stores the `uint64` residual `zigzag((quotient - time_ref) mod 2^64)`, with `0` for sample `0`.
  - First, bit planes `j = 0..31` across all irregular blocks (bit 4 set) in block order, `p_b` bytes per block per plane.
  - Then, bit planes `j = 32..63` across all wide-time blocks (bit 5 set) in block order, `p_b` bytes per block per plane.
  - Bit `k` of byte `i` is bit `j` of the residual for sample `8i + k`.
  - Non-wide irregular blocks have all residuals `< 2^32`.
  - Empty (0 bytes) when all blocks are regular.

### 6.4 Validation

A conforming decoder MUST reject a block group whenever any of the following checks fail:

- **Header:**
  - `version != 1`;
  - header `flags` bits 5–7 are non-zero, `time_unit > 4`, or bit 4 is set when `time_unit == 0`;
  - `num_samples > 2^26`, or `num_samples > 65,535 × num_blocks` (more samples than `num_blocks` blocks can hold).
- **Pre-decompression frame size:**
  - The Zstandard frame's recorded content size does not lie within the plausible `[min_body_size, max_body_size]` bounds derived from the header:
    - minimum plane bytes: `ceil(num_samples / 8)` per plane, no block flags set (and `E = 0`);
    - maximum plane bytes: up to 7 padding samples per block, every block flagged with sample codes and wide time residuals (plus up to `8 × num_samples` bytes of `escaped_values`).
- **Body sizes:**
  - The sum of `block_sizes` does not equal `num_samples`;
  - The decompressed content size does not equal the exact `body_size` computed once `block_flags`, `block_sizes`, and `E` are known.
- **Sample codes and escapes:**
  - Any block contains code `11` before its first code `10`;
  - `E` (computed over actual block samples only, ignoring padding bits) does not match the `escaped_values` byte count.
- **Per-block metadata:**
  - `block_flags` bit 7 is non-zero; bit 4 is set when `time_unit == 0`; bit 5 is set without bit 4;
  - `grid_exponents` is outside the valid range for the block's grid mode (§6.3);
  - `grid_scales` is non-zero on an unscaled block; or on a scaled block, `N == 0`, `D == 0`, or (for scaled decimal) `D · 5^(-p) >= 2^53`;
  - `block_anchors` is non-finite (`NaN` or `±inf`) on a power-of-two grid, or `|K0| >= 2^52` on a decimal or scaled grid;
  - Any metadata column is non-zero in an empty block (`n_b == 0`).
- **Time axis** (when `time_unit != 0`):
  - If header flags bit 4 is set: `group_time_step < 2`, or multiplying any stored start increase or `time_step` by `group_time_step` overflows 64 bits;
  - The first non-empty block's `time_starts` value is `INT64_MIN`;
  - The running sum of `time_starts` exceeds `INT64_MAX`;
  - In a 1-sample block: `time_step != 0`, `time_ref != 0`, or `block_flags` bit 4 is set;
  - `time_step < 0`, or `time_step < 1` in an irregular block;
  - An irregular block's sample-0 time residual is non-zero;
  - A wide-time block (`block_flags` bit 5 set) has all-zero bit planes 32–63 (it should have been encoded as a 32-bit irregular block);
  - Any decoded timestamp exceeds `INT64_MAX`.

### 6.5 Examples

#### Test vector for bit order

A block with `u[3] = 0x0020` (bit 5 set) and `u[6] = 0x0400` (bit 10 set), and all other `u[i] = 0`, encodes in bit-plane mode as:

- byte 0 of plane 5 = `0b00001000` (`0x08`, bit 3 set);
- byte 0 of plane 10 = `0b01000000` (`0x40`, bit 6 set);
- all other plane bytes = `0x00`.

#### Worked example with a time axis

From `tests/test_time.py::test_worked_example_layout`:

- 20 samples in 3 blocks of sizes `8`, `8`, and `4` (`p_0 = 1`, `p_1 = 1`, `p_2 = 1`; `P = 3`, `P_codes = 0`, `P_irregular = 1`, `P_wide = 0`, `E = 0`).
- `ns` ticks (`time_unit = 4`, small values for readability).
- Bit planes forced (header flags bit 0 clear). *(Note: the reference encoder's own layout heuristic selects byte planes here, header flags `0x09`, because all-zero residuals tie.)*

```text
times   block 0: 1000 1010 1020 1030 1040 1050 1060 1070   regular
        block 1: 1080 1090 1100 1130 1140 1150 1160 1170   irregular: a gap after 1100
        block 2: 1180 1190 1200 1210                       regular
values  constant per block: 2.5, 2.25, 3.0   (power-of-two grid)
```

- **Header (8 bytes):** `01 08 03 00 14 00 00 00`
  - `version = 0x01`, `flags = 0x08` (`time_unit = 4` in bits 1–3), `num_blocks = 3`, `num_samples = 20` (`0x14`).
- **Body (203 bytes):** `17 × 3 + 16 × 3 + 24 × 3 + 32 × 1 = 203`.

```text
offset   field                 size  byte planes (3 bytes per plane: blocks 0, 1, 2)
0–2      block_flags           3     00 10 00 (bit 4 set on block 1 only)
3–8      block_sizes           6     plane 0 = 08 08 04, plane 1 = 00 00 00
9–14     grid_exponents        6     exponent 0 (constant blocks): all 00
15–26    grid_scales           12    0 (unscaled): all 00
27–50    block_anchors         24    raw float64 bits 0x4004…, 0x4002…, 0x4008…:
                                     planes 0–5 = 00 00 00, plane 6 = 04 02 08, plane 7 = 40 40 40
51–74    time_starts           24    stored 1000, 80, 100:
                                     plane 0 = E8 50 64, plane 1 = 03 00 00, planes 2–7 = 00 00 00
75–98    time_steps            24    10, 10, 10: plane 0 = 0A 0A 0A, planes 1–7 = 00 00 00
99–122   time_ref_quotients    24    1, 1, 1: plane 0 = 01 01 01, planes 1–7 = 00 00 00
123–170  residual_planes       48    16 bit planes × 3 bytes (all 00; block 2's last 4 bits padding)
171–202  time_residual_planes  32    block 1 (32-bit irregular: planes 0–31 only):
                                     quotients [1, 1, 3, 1, 1, 1, 1] from sample 1, time_ref = 1
                                     → residuals [0, 0, 0, 2, 0, 0, 0, 0]
                                     → zigzagged [0, 0, 0, 4, 0, 0, 0, 0]
                                     → plane 2 = 0x08, all other planes = 0x00
```

---

## 7. Exactly representable values

This section states what the wire format can represent bit-for-bit. Whether a particular encoder discovers that representation is up to the encoder (see [ENCODER.md §6](ENCODER.md#6-guarantees) for the reference encoder's guarantees).

- **Power-of-two grids.** Any block whose finite values all take the form `a + q · 2^e` with:
  - `a` a finite `float64`,
  - `e` an integer in `[-1074, 1023]`,
  - `q` an integer in `[-32,768, 32,767]`.
  This includes any constant block of a finite `float64` (`q = 0`, `a` = the value) and any block on a binary grid `2^e` spanning at most `65,535` steps (e.g., integers or multiples of `2^-4`).
- **Decimal grids.** Any block whose finite values all take the form `(K0 + q) · 10^p` with:
  - `p` an integer in `[-22, 22]`,
  - `K0` an integer with `|K0| < 2^52`,
  - `q` an integer in `[-32,768, 32,767]`.
  Each sample decodes to the `float64` nearest the decimal—the same bit pattern produced by parsing its decimal text.
- **Scaled grids.** Any block whose finite values all take the form `(K0 + N · q) · B^p / D` with:
  - `B = 2` and `e ∈ [-1022, 970]`, or `B = 10` and `p ∈ [-22, 0]` with `D · 5^(-p) < 2^53`,
  - `N, D ∈ [1, 65,535]`,
  - `K0` an integer with `|K0| < 2^52`,
  - `q` an integer in `[-32,768, 32,767]`.
  Each sample decodes to the `float64` nearest the rational (e.g., thirds, `10 V / 4095` ADC counts, or `32 + 0.18·k`), bit-for-bit when the input was that `float64`.
- **Non-finite values.** `+inf` and `-inf` decode bit-for-bit (via `escaped_values`). `NaN` decodes as the canonical quiet `NaN`; its sign and payload bits are not preserved.
- **Negative zero.** `-0.0` on a grid decodes as `+0.0` (numerically equal; the sign bit is not preserved).
- **Timestamps.** Any non-decreasing sequence of `int64` ticks other than `INT64_MIN`, in `s`, `ms`, `µs`, or `ns`, decodes to the identical ticks in the same unit.

---

## 8. Conformance tests

A decoder conforms to this specification when:

1. **Bit order:** it decodes the bit-order test vector of §6.5.
2. **Writer-choice independence:** it decodes any valid block group identically regardless of the predictor order, grid mode, unsnapped finite anchor, or `time_ref_quotients` value chosen by the writer.
3. **Self-describing block groups:** it requires only the block group bytes and returns exactly `num_samples` samples and each block's size.
4. **Validation:** it rejects every failure condition listed in §6.4.
5. **Time axis:** it decodes the worked example of §6.5 byte-for-byte.

The reference encoder's conformance tests are in [ENCODER.md §7](ENCODER.md#7-conformance-tests).

---

## References

Background for the techniques used in this format (informative):

- Fixed polynomial predictors (§4, §5): the fixed predictors of [Shorten](https://en.wikipedia.org/wiki/Shorten_(file_format)) and FLAC, [RFC 9639 §9.2.5](https://www.rfc-editor.org/rfc/rfc9639#name-fixed-predictor-subframe).
- Zigzag encoding of signed integers (§4): [Protocol Buffers encoding](https://protobuf.dev/programming-guides/encoding/#signed-ints).
- Bit-shuffle (§6): [bitshuffle](https://github.com/kiyo-masui/bitshuffle); K. Masui et al., [arXiv:1503.00638](https://arxiv.org/abs/1503.00638).
- Zstandard (§6): [RFC 8878](https://www.rfc-editor.org/rfc/rfc8878).
- IEEE 754 binary64 and correctly rounded division (§5): [double-precision floating-point format](https://en.wikipedia.org/wiki/Double-precision_floating-point_format).
