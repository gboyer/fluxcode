# fluxcode — format specification

<!-- LARGER CHANGE (after finishing the doc changes): Rename "unit" everywhere to "block group" (in prose) or "group" (when a shorter name is needed for code). In science and engineering, "unit" is overloaded as units of measurement and units of large industrial facilities. -->

The fluxcode unit format: float64 samples in blocks, with optional exact timestamps.

- This document specifies the layout of a unit and how to decode it.
- How the reference encoder (the `fluxcode` package) chooses what to write, its parameters and the
  error bounds it guarantees are in [ENCODER.md](ENCODER.md).

## 1. Units and blocks

- **A unit is one storage artifact** (e.g. a database row): blocks encoded and decoded together.
  - Units are independent of each other.
  - A unit holds 0 to 65,535 blocks.
- **A block holds 0 to 65,535 samples.** Each block's size is recorded in the unit.
- **Decoding needs only the unit.** No index column or length has to be kept beside it:
  - the header records the block count and the sample count;
  - the body records each block's size (§6) and its own anchor.
- **Nothing is padded at the sample level:** a block decodes to exactly its samples.
  - Bit planes store each block in whole bytes: a block whose size isn't a multiple of 8 pads
    its last byte with zero bits (§6).
- **The body is stored field by field** (§6). Each field holds every block's bytes, in block order.
- **An empty block** stores no samples. Its flags, grid parameter, anchor and time columns are 0.
- **`num_samples` is at most 2^26.**
  - This is not a field limit (the header field holds up to 2^32 − 1).
  - It is the bound decoders use to reject implausible headers before decompressing.

## 2. Non-finite values

- NaN, +inf and −inf are stored exactly, as a 2-bit code per sample:

  | code | sample |
  |---|---|
  | 00 | finite |
  | 01 | NaN |
  | 10 | +inf |
  | 11 | −inf |

- Only a block holding a non-finite value stores codes.
  - It sets block flag bit 3.
  - Its codes go in the `nonfinite_code_planes` field (§6): n/4 bytes, mostly zero, before
    compression.
- The block's values at those samples are placeholders, overwritten by the codes on decode (§5).

## 3. Time axis

- **Timestamps are optional.**
  - A unit without them has time unit 0 in its header and no time fields in its body.
  - It decodes with `times = None`.
- **Ticks.** Timestamps are int64 ticks since 1970-01-01, in one of the Arrow and numpy datetime64
  units, recorded in the header (§6):

  | time unit | code | int64 range around 1970 |
  |---|---|---|
  | s | 1 | ± 2.9 · 10^11 years |
  | ms | 2 | ± 2.9 · 10^8 years |
  | µs | 3 | ± 292,000 years |
  | ns | 4 | ± 292 years (1677–2262) |

  - The time unit fixes only the range and meaning of a tick.
  - Storage cost doesn't depend on it: the per-block step (below) absorbs a coarser grid.
  - int64 minimum (datetime64's NaT) is never a timestamp.
- **Timestamps are naive:** the format stores no time zone.
  - Store UTC: local time can repeat or skip an hour, which breaks the ordering rule.
- **Ordering is built into the format.** Deltas within a block and increases between block starts
  are unsigned, so any decoded unit has:
  - non-decreasing times within each block;
  - non-decreasing block starts.
- **Per block** every block stores a start, a step and a reference (`time_start`, `time_step`,
  `time_ref`, and `time_residual_planes` in §6).
  - The **step** is the GCD of the block's deltas `time[i] − time[i−1]`.
  - Each sample i > 0 has the **quotient** `(time[i] − time[i−1]) / time_step`.
  - The quotient is `time_ref` plus the sample's **residual**.
- **Regular block:** every delta is equal (possibly 0).
  - Every quotient is `time_ref` = 1, so `time[i] = time_start + i · time_step`.
  - When all times are equal, `time_ref` and `time_step` are 0.
  - It stores no per-sample data.
- **Irregular block:** any other block (block flag bit 4).
  - It stores each sample's residual `quotient − time_ref`, mod 2^64 and zigzagged; 0 for sample 0.
  - The residuals take 32 bit planes, or 64 if any residual is 2^32 or more: a **long** block
    (block flag bit 5).
  - Any `time_ref` decodes. How the reference encoder picks it is in [ENCODER.md §4](ENCODER.md#4-time-axis).
  - A delta above int64 maximum (a block spanning more than half the int64 range) can't be a
    stored GCD. Such a block has `time_step` = 1, and its raw deltas are the quotients.
- **Tiny blocks.**
  - A block of one sample has no deltas: `time_step` and `time_ref` are 0.
  - A block of two samples is always regular. If its one delta exceeds int64 maximum, the delta
    is stored as `time_ref`, over a `time_step` of 1.
- **Empty blocks** have 0 in all three columns and are skipped in the chain of starts: each
  non-empty block's start is stored relative to the previous non-empty block's.

## 4. Anchors and residuals

- **Grid indices.** Each finite sample of a block is a grid index q, with 0 ≤ q < 2^16.
- **The anchor** is what a decoder adds q to. It is one of:
  - on a power-of-two grid: a finite float64, giving the value anchor + q·2^e;
  - on a decimal grid (block flag bit 2): an integer K0 with |K0| < 2^52, giving (K0 + q)·10^p.
  - Both define the block's values exactly (a binary fraction, or a decimal). float64 enters
    only at the final rounding (§5).
- **The format doesn't constrain how a writer picks the grid or the anchor.**
  - A decoder doesn't check that the anchor sits on the grid, nor the choice of exponent or order.
  - Any finite anchor decodes correctly.
- **Residuals.** With k the block's order (0–3), the body stores u, the zigzagged residuals of q:

  ```
  r = k-th difference of (0, …, 0, q[0], …, q[n-1])  with k zeros prepended; keep the last n
      # order 1: r[0] = q[0],  r[i] = q[i] − q[i−1]
      # order 2: r[0] = q[0],  r[1] = q[1] − 2q[0],  r[i] = q[i] − 2q[i−1] + q[i−2]
      # order 3: r[0] = q[0],  r[1] = q[1] − 3q[0],  r[2] = q[2] − 3q[1] + 3q[0],
      #          r[i] = q[i] − 3q[i−1] + 3q[i−2] − q[i−3]
  v = ((r + 32768) mod 2^16) − 32768          # r mod 2^16, read as int16
  u = ((v << 1) XOR (v >> 15)) & 0xFFFF       # zigzag -> uint16
  ```

  - Order-k differences of 16-bit values need up to 16 + k + 1 bits. But every q is in
    [0, 2^16), so integrating mod 2^16 (§5) recovers q exactly: two bytes per sample suffice.
  - The prepended zeros make the first residuals the start values, so there's no separate header.

## 5. Decoding one block

```
v = (u >> 1) XOR −(u & 1)                   # un-zigzag
repeat `order` times:  prefix-sum v, each partial sum mod 2^16     # q
order 0: q = v mod 2^16
power of two: y[i] = a + q[i] · 2^e, rounded once, clamped to the largest finite double   (a: the anchor)
decimal:      p < 0:  y[i] = float64(K0 + q[i]) / 10^-p   (K0: the anchor; integer divided by an exact power of ten)
              p >= 0: y[i] = float64(K0 + q[i]) · 10^p
```

- **Decimal reconstruction** is the same double that parsing the decimal text (e.g. "12.345")
  produces, because IEEE division is correctly rounded.
- **The clamp** only matters for e ≥ 971.
  - There, a sample near the largest double can reconstruct up to half a step above it, which
    would round to inf. Clamping moves it closer to the original.
  - Below 971, `a + q·2^e` can't overflow (the excess is under half an ulp of the largest
    double): decoders compute it directly.
  - From 971 up, `q·2^e` itself can overflow while the sum doesn't. Evaluate at half scale:
    `y = min(2·(a/2 + q·2^(e−1)), DBL_MAX)`, with `y = a` where q = 0 (a/2 rounds if a is
    subnormal). That is bit-identical to `a + q·2^e` wherever the latter is finite.
- **Non-finite codes** (block flag bit 3): after dequantizing, overwrite samples with code
  01, 10 or 11 with NaN (the canonical quiet NaN), +inf or −inf.
  - Groups of 8 samples whose two code bytes are both 0 can be skipped.
- **Time axis** (header time unit ≠ 0):
  - Block starts are the running sum of the `time_start` field (§6).
  - Each sample i > 0 has `quotient[i] = time_ref + unzigzag(residual[i])` mod 2^64. Residuals
    are 0 in a regular block (block flag bit 4 clear).
  - `time[i] = start + time_step · (quotient[1] + … + quotient[i])`. A regular block is therefore
    `time[i] = start + i · time_ref · time_step`.
  - All arithmetic is exact in 64 bits. A decoder rejects a unit whose times would exceed int64
    maximum (§6).

## 6. Unit format

A unit is an 8-byte header followed by one zstd frame holding the body.
- The header is uncompressed, so a decoder can validate it before decompressing.
- The frame may use any zstd level and block boundaries. It records its content size and has no
  checksum.

### 6.1 Header

All fields are little-endian.

| offset | field | type | contents |
|---|---|---|---|
| 0 | version | uint8 | 1 |
| 1 | flags | uint8 | see below |
| 2–3 | `num_blocks` | uint16 | block count, 0 to 65,535 |
| 4–7 | `num_samples` | uint32 | sample count, 0 to 2^26 (§1): the sum of the block sizes |

Header flags:
- bit 0: `residual_planes` are byte planes instead of bit planes;
- bits 1–3: `time_unit` (§3): 0 for no time axis, 1 s, 2 ms, 3 µs, 4 ns;
- bits 4–7: 0.

### 6.2 Planes and padding

- **Bytes per block.** Each block b of `n_b` samples takes `g_b` = ceil(`n_b` / 8) bytes of every
  bit plane. An empty block takes none.
- **Padding.** When `n_b` isn't a multiple of 8, bits `n_b` mod 8 to 7 of the block's last byte
  are padding. Byte planes likewise pad each block to 8 × `g_b` bytes.
  - Writers zero the padding; decoders ignore it.
- **Order.** Within a plane, the blocks' bytes follow each other in block order.
- **Byte-planed columns.** A per-block column of w-byte integers stores byte 0 of every block,
  then byte 1 of every block, and so on to byte w − 1.
- **Sums** used below:
  - `G` = Σ `g_b` over all blocks;
  - `G_nonfinite` over the blocks with block flag bit 3 set;
  - `G_irregular` over those with bit 4;
  - `G_long` over those with bit 5.

### 6.3 Body

The body is these fields, in this order. The four time fields are present only when
`time_unit` ≠ 0.

| field | size in bytes | contents |
|---|---|---|
| `block_flags` | `num_blocks` | per block: order, grid kind and what it stores (below) |
| `block_sizes` | 2 × `num_blocks` | sample count, uint16, byte-planed |
| `grid_params` | 2 × `num_blocks` | grid parameter, int16, byte-planed |
| `value_anchor` | 8 × `num_blocks` | anchor (§4), byte-planed |
| `time_start` | 8 × `num_blocks` | start time, byte-planed |
| `time_step` | 8 × `num_blocks` | time step, int64, byte-planed |
| `time_ref` | 8 × `num_blocks` | reference quotient, uint64, byte-planed |
| `residual_planes` | 16 × `G` | the residuals u (§4) |
| `nonfinite_code_planes` | 2 × `G_nonfinite` | the non-finite codes (§2) |
| `time_residual_planes` | 32 × (`G_irregular` + `G_long`) | the time residuals (§3) |

```
body_size = 13 × num_blocks + 16 × G + 2 × G_nonfinite
          + (time_unit ≠ 0) × (24 × num_blocks + 32 × (G_irregular + G_long))
```

Every column is 0 for an empty block. Per field:

- **`block_flags`**:
  - bits 0–1: the order (§4);
  - bit 2: decimal grid;
  - bit 3: non-finite codes present (§2);
  - bit 4: irregular times, with time residual planes (§3);
  - bit 5: long time residuals, 64 planes instead of 32;
  - bits 6–7: 0.
- **`grid_params`**:
  - power-of-two grid: the exponent e, −1074 to 1023;
  - decimal grid: the decimal power p, −22 to 22.
- **`value_anchor`**:
  - power-of-two grid: a finite float64;
  - decimal grid: the int64 K0, with |K0| < 2^52.
- **`time_start`**:
  - the first non-empty block: its start time, as int64;
  - each later non-empty block: its start minus the previous non-empty block's start, as uint64.
- **`time_step`**: the GCD of the block's deltas (§3).
  - ≥ 0, and ≥ 1 in an irregular block;
  - 0 in a block of one sample.
- **`time_ref`**:
  - regular block: 1; or 0 if its times are all equal or it has one sample; or, in a 2-sample
    block whose delta exceeds int64 maximum, the delta itself (over a `time_step` of 1);
  - irregular block: the value the residuals are taken from (§3).
- **`residual_planes`**:
  - bit planes (header flags bit 0 clear): bit plane j = 0..15, then block, then byte
    i = 0..`g_b` − 1. Bit k of byte i (LSB = bit 0) is bit j of `u[8i + k]`.
  - byte planes (header flags bit 0 set): byte plane j = 0..1 (low, then high byte of `u`), then
    block, then sample 0..8 × `g_b` − 1.
- **`nonfinite_code_planes`**: code plane j = 0..1, then the flagged blocks in block order, then
  byte i. Bit k of byte i is bit j of the code of sample 8i + k.
- **`time_residual_planes`**: per sample of an irregular block, the uint64 residual
  `zigzag(quotient − time_ref)` mod 2^64; 0 for sample 0.
  - First bit plane j = 0..31, then the irregular blocks in block order, then byte i.
  - Then bit plane j = 32..63, then the long blocks in block order, then byte i.
  - Bit k of byte i is bit j of the residual of sample 8i + k.
  - A short block's residuals are below 2^32.
- A unit with no flagged blocks has an empty `nonfinite_code_planes` field, and one with only
  regular blocks an empty `time_residual_planes` field.

### 6.4 Validation

A decoder rejects a unit when any of these fail.

- **Header:**
  - version 1;
  - header flags bits 4–7 zero, `time_unit` 0 to 4;
  - `num_samples` ≤ 2^26 and no more than `num_blocks` blocks can hold.
- **Before decompressing:** the frame's recorded content size lies between the body sizes with
  - the fewest plane bytes: ceil(`num_samples` / 8) per plane, no flags;
  - the most: up to 7 padding samples per block, every block flagged and long.
- **Sizes:** the block sizes add up to `num_samples`, and the content size is exactly
  `body_size` once `block_flags` and `block_sizes` give the sums.
- **Per block:**
  - `block_flags` bits 6–7 zero; bit 4 only with a time axis; bit 5 only with bit 4;
  - the grid parameter in range for its grid;
  - a finite float anchor, or a decimal anchor with |K0| < 2^52;
  - all columns 0 in an empty block.
- **Time axis:**
  - a first non-empty start other than int64 minimum;
  - a running sum of `time_start` at most int64 maximum;
  - in a block of one sample: `time_step` and `time_ref` 0, block flag bit 4 clear;
  - `time_step` ≥ 0, and ≥ 1 in an irregular block;
  - an irregular block's first residual 0;
  - a long block with a nonzero bit in planes 32–63 (otherwise it fits a short block);
  - every decoded time at most int64 maximum.

### 6.5 Examples

**Test vector for the bit order.** A block whose `u[3] = 0x0020` and `u[6] = 0x0400` (all others 0)
has:
- byte 0 of plane 5 = `0b00001000`;
- byte 0 of plane 10 = `0b01000000`;
- all other plane bytes 0.

**Worked example with a time axis** (`tests/test_time.py::test_worked_example_layout`):
- 20 samples in blocks of 8, 8 and 4;
- ns ticks (small, for readability);
- bit planes (header flags bit 0 clear).

```
times   block 0: 1000 1010 1020 1030 1040 1050 1060 1070   regular
        block 1: 1080 1090 1100 1130 1140 1150 1160 1170   irregular: a gap after 1100
        block 2: 1180 1190 1200 1210                       regular
values  constant per block: 2.5, 2.25, 3.0   (power-of-two grid)
```

- Header: `01 08 03 00 14 00 00 00` (flags 0x08: `time_unit` 4 in bits 1–3; 3 blocks; 20 samples).
- Body: 191 bytes = 13 × 3 + 16 × 3 + 24 × 3 + 32 × 1.

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
                                     from sample 1, time_ref 1: residuals [0, 0, 0, 2, 0, 0, 0, 0],
                                     zigzagged [0, 0, 0, 4, 0, 0, 0, 0]: plane 2 = 0x08,
                                     all other planes 0x00
```

## 7. Exactly representable values

What a unit can hold bit for bit. Whether a given encoder finds that representation is up to the
encoder ([ENCODER.md §6](ENCODER.md#6-guarantees) for the reference one).

- **Power-of-two grids.** A block whose finite values are all a + q·2^e, with
  - a a finite float64,
  - e an integer from −1074 to 1023,
  - q an integer from 0 to 65,535.
  - This covers a constant block of any finite value (q = 0, a the value), and any block of
    values on a grid of 2^e spanning at most 65,535 steps, e.g. integers, or multiples of 2^−4.
- **Decimal grids.** A block whose finite values are all decimals (K0 + q)·10^p, with
  - p an integer from −22 to 22,
  - K0 an integer with |K0| < 2^52,
  - q an integer from 0 to 65,535.
  - Each decodes to the double nearest the decimal: the same one parsing its text gives.
- **Non-finite values.** +inf and −inf decode exactly. NaN decodes as the canonical quiet NaN:
  its payload and sign are lost.
- **Negative zero** decodes as +0.0 (equal as floats; the sign bit is lost).
- **Timestamps.** Any non-decreasing int64 ticks other than int64 minimum, in s, ms, µs or ns,
  decode to the identical ticks in the same unit.

## 8. Conformance tests

A decoder conforms when:

1. **Bit order:** it decodes the test vector of §6.5.
2. **Any writer choice decodes:** a unit decodes the same whatever order, grid and anchor its
   writer picked, including any finite unsnapped anchor and any `time_ref`.
3. **Self-describing units:** it needs only the unit, and returns exactly `num_samples` samples
   and each block's size.
4. **Validation:** it rejects each failure listed in §6.4.
5. **Time axis:** it decodes the worked example of §6.5 byte for byte.

The reference encoder's own conformance tests are in [ENCODER.md §7](ENCODER.md#7-conformance-tests).

## References

Background for the techniques above (informative; the sections above are normative).

- Fixed polynomial predictors (§4, §5): the fixed predictors of
  [Shorten](https://en.wikipedia.org/wiki/Shorten_(file_format)) and FLAC,
  [RFC 9639 §9.2.5](https://www.rfc-editor.org/rfc/rfc9639#name-fixed-predictor-subframe).
- Zigzag encoding of signed integers (§4):
  [Protocol Buffers encoding](https://protobuf.dev/programming-guides/encoding/#signed-ints).
- Bit-shuffle (§6): [bitshuffle](https://github.com/kiyo-masui/bitshuffle); K. Masui et al.,
  [arXiv:1503.00638](https://arxiv.org/abs/1503.00638).
- Zstandard (§6): [RFC 8878](https://www.rfc-editor.org/rfc/rfc8878).
- IEEE 754 binary64 and correctly rounded division (§5):
  [double-precision floating-point format](https://en.wikipedia.org/wiki/Double-precision_floating-point_format).
