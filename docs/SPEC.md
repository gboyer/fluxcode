# fluxcode — format specification

<!-- FIXME: this long string of semicolon separated clauses is very hard to read. Use bullets, perhaps nested. Apply this stylistic change everywhere; bullets, not proose blocks. -->

<!-- LARGER CHANGE (after finishing the doc changes): Rename "unit" everywhere to "block group" (in prose) or "group" (when a shorter name is needed for code). In science and engineering, "unit" is overloaded as units of measurement and units of large industrial facilities. -->

The fluxcode unit format: a compressed series of float64 samples in blocks, with optional exact
timestamps. This document specifies the layout of a unit and how to decode it. How the reference
encoder (the `fluxcode` package) chooses what to write is in [ENCODER.md](ENCODER.md), along with
its parameters and the error bounds it guarantees.

## 1. Units and blocks

**A unit is one storage artifact** (e.g. a database row): blocks encoded and decoded together.
Units are independent. **Blocks have any size from 0 to 65,535 samples**, each recorded in the
unit, and a unit holds up to 65,535 blocks.

**Decoding needs only the unit.** The header records the block count and the sample count, the
body each block's size (§6), and each block its own anchor, so no index column or length has to
be kept beside it. Nothing is padded: a block holds exactly its samples.

An empty block stores no samples: its flags, grid parameter and anchor (and time columns) are 0.

`num_samples` is at most 2^26: not a format limit (the header field holds up to 2^32 − 1), but the
bound decoders use to reject implausible headers before decompressing.

Fixed by this spec: the body stored field by field, each field holding every block's bytes in
block order (§6). Bit planes store each block in whole bytes, so a block whose size isn't a
multiple of 8 pads its last byte with zero bits.

## 2. Non-finite values

NaN, +inf and −inf are encoded exactly. A block holding any sets block flag bit 3 and stores a
2-bit code per sample in the `nonfinite_code_planes` field (§6): 00 finite, 01 NaN, 10 +inf,
11 −inf.

A block with no non-finite values stores no codes. A flagged block adds n/4 bytes of mostly-zero
planes before compression.

## 3. Time axis

A unit may also store the samples' **timestamps, exactly**. They are optional: a unit without them
has time unit 0 in its header and no time fields in its body, and decodes with `times = None`.

**Ticks and units.** Timestamps are int64 ticks since 1970-01-01 in one of the Arrow and numpy
datetime64 units, recorded in the header (§6). The unit only fixes the range and meaning of a
tick; storage cost doesn't depend on it (the per-block GCD below absorbs a coarser grid):

| time unit | code | int64 range around 1970 |
|---|---|---|
| s | 1 | ± 2.9 · 10^11 years |
| ms | 2 | ± 2.9 · 10^8 years |
| µs | 3 | ± 292,000 years |
| ns | 4 | ± 292 years (1677–2262) |

Timestamps are **naive**: the format stores no time zone. Store UTC: local time can repeat or skip
an hour, which breaks the ordering rule. int64 minimum (datetime64's NaT) is never a timestamp.


**Ordering.** The format enforces non-decreasing times within a block (deltas are unsigned) and
between block starts (start increases are unsigned), so any decoded unit has non-decreasing times
within each block and non-decreasing block starts.

**Per block** (`time_start`, `time_step`, `time_ref`, `time_residual_planes` in §6): every block
stores its start, a step and a reference. The step is the GCD of the block's deltas
`time[i] − time[i−1]`, and each sample i > 0 has the **quotient** `(time[i] − time[i−1]) / time_step`,
which is `time_ref` plus the sample's residual:
- **Regular block:** every delta is equal (possibly 0). Every quotient is `time_ref` = 1 (0 when
  all times are equal, with `time_step` = 0), so `time[i] = time_start + i · time_step`, and the block
  stores no per-sample data.
- **Irregular block:** any other block. It stores each sample's residual `quotient − time_ref`
  (mod 2^64, zigzagged; 0 for sample 0) in 32 bit planes, or 64 if any residual is 2^32 or more
  (a **long** block, block flag bit 5). Any `time_ref` decodes; how the reference encoder picks
  it is in [ENCODER.md](ENCODER.md).
  If a single delta exceeds int64 maximum (a block spanning more than half the int64 range), the
  GCD can't be stored and the block uses `time_step = 1` with the raw deltas as quotients.
  Long blocks are rare: in ns ticks with a GCD of 1, a gap of about 2 s or more; in µs ticks or on
  any coarser grid, a gap of over half an hour.
- **Tiny blocks:** a block of one sample has no deltas: `time_step` and `time_ref` are 0. A block
  of two samples is always regular; if its one delta exceeds int64 maximum, it is stored as
  `time_ref` over a `time_step` of 1.
- **Empty blocks** have 0 in all three columns and take no part in the chain of start
  increases: each non-empty block's start is stored relative to the previous non-empty block's.

## 4. Anchors and residuals

**The block's anchor** is what a decoder adds q to: a finite float64 on the power-of-two grid
(the reference encoder snaps it to the grid point nearest the minimum; decoders accept any finite
anchor), or the integer `K0` on a decimal grid (|K0| < 2^52, since decimal detection requires
max(|lo|, |hi|)·10^−p < 2^52). Both define the block's values exactly (anchor + q·2^e is a binary
fraction, (K0 + q)·10^p a decimal); float64 enters only at the final rounding.

**Snapping is encoder behaviour, not a format rule.** The decoder doesn't check that an anchor is
snapped, as it doesn't check the exponent or the difference order: an anchor carries meaning, and
any finite one decodes correctly. An unsnapped anchor from another writer costs at most half a
step on the first straddling update of a reference encoder (ENCODER.md).

**Residuals.** With q[i] the block's grid indices (0 ≤ q[i] < 2^16) and k its order, the body
stores u, the zigzagged residuals:

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
so integrating mod 2^16 (§5) recovers q exactly. Two bytes per sample always suffice. The
prepended zeros make the first residuals the start values, so there's no separate header.

## 5. Decoding one block

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
(§6). Each sample i > 0 has `quotient[i] = time_ref + unzigzag(residual[i])` mod 2^64 (residuals
are 0 in a regular block, block flag bit 4 clear), and
`time[i] = start + time_step · (quotient[1] + … + quotient[i])`. A regular block is therefore
`time[i] = start + i · time_ref · time_step`. All arithmetic is exact in 64 bits; a decoder rejects
a unit whose times would exceed int64 maximum (§6).

**Non-finite codes** (block flag bit 3): after dequantizing, samples with code 01, 10 or 11 are
overwritten with NaN (the canonical quiet NaN), +inf or −inf. Groups of 8 samples whose two code
bytes are both 0 are skipped.

## 6. Unit format

A unit is an 8-byte header followed by one zstd frame holding the body (any zstd level and block
boundaries, content size recorded, no checksum). The header is uncompressed, so a decoder can validate it before
decompressing:

| offset | header field | type (little-endian) | contents |
|---|---|---|---|
| 0 | version | uint8 | 1 (a decoder rejects others) |
| 1 | flags | uint8 | bit 0: `residual_planes` are byte planes instead of bit planes; bits 1–3: `time_unit`, 0 for no time axis, 1 s, 2 ms, 3 µs, 4 ns (5–7 rejected); bits 4–7: 0 (rejected otherwise) |
| 2–3 | `num_blocks` | uint16 | block count, 0 to 65,535 |
| 4–7 | `num_samples` | uint32 | sample count, 0 to 2^26 (§1): the sum of the block sizes |

Each block b of `n_b` samples takes `g_b` = ceil(`n_b` / 8) bytes of every bit plane (none for an
empty block): when `n_b` isn't a multiple of 8, bits `n_b` mod 8 to 7 of its last byte are padding,
and byte planes likewise pad each block to 8 × `g_b` bytes. Writers zero the padding; decoders
ignore it. Within a plane the blocks' bytes follow each other in block order. Sums used below:
`G` = Σ `g_b` over all blocks, `G_nonfinite` over the blocks with block flag bit 3 set,
`G_irregular` over those with bit 4 and `G_long` over those with bit 5. The body is these fields,
in this order; the four time fields are present only when `time_unit` ≠ 0:

| field | size in bytes | contents |
|---|---|---|
| `block_flags` | `num_blocks` | bits 0–1: order; bit 2: decimal mode; bit 3: non-finite codes present (§2); bit 4: irregular times, with time residual planes (§3; rejected when `time_unit` = 0); bit 5: long time residuals, 64 planes instead of 32 (rejected without bit 4); bits 6–7: 0 (rejected otherwise). 0 for an empty block |
| `block_sizes` | 2 × `num_blocks` | each block's sample count as uint16, byte-planed: every low byte, then every high byte |
| `grid_params` | 2 × `num_blocks` | the block's grid parameter as int16: the power-of-two exponent (−1074 to 1023) or, in decimal mode, the decimal power (−22 to 22); 0 for an empty block. Byte-planed like `block_sizes` |
| `value_anchor` | 8 × `num_blocks` | the block's anchor (§4), byte-planed (byte 0 of every block, then byte 1, … byte 7): a finite float64 anchor (the reference encoder snaps the block minimum to the grid, ENCODER.md §5.4; 0.0 for a block with no finite samples), or in decimal mode the int64 decimal grid index of the minimum (magnitude < 2^52); 0 for an empty block |
| `time_start` | 8 × `num_blocks` | the first non-empty block: its start time as int64; each later non-empty block: its start minus the previous non-empty block's start, as uint64; an empty block: 0. Byte-planed |
| `time_step` | 8 × `num_blocks` | the block's time step as int64, byte-planed: the GCD of its deltas (≥ 0; ≥ 1 in an irregular block; 0 in an empty block or one of a single sample) |
| `time_ref` | 8 × `num_blocks` | the block's reference quotient as uint64, byte-planed: 1 in a regular block (0 if its times are all equal or it has one sample or none; the delta itself in a 2-sample block whose delta exceeds int64 maximum, over a `time_step` of 1, §3); in an irregular block, the value the residuals are taken from (§4) |
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

## 7. Exact values

The format represents these exactly:
- **Power-of-two grids.** A block on the grid anchor + q·2^e (q < 2^16) decodes bit for bit.
- **Decimals.** In decimal mode, (K0 + q)·10^p with |K0 + q| < 2^52 and −22 ≤ p ≤ 22 decodes to
  the same double as parsing the decimal text.
- **One zero and one NaN.** −0.0 decodes as +0.0 (equal as floats, sign bit lost), and every NaN
  decodes as the canonical quiet NaN (payload and sign lost). These are the exceptions to
  bit-exactness for data on the grid. ±inf round-trip exactly.
- **Timestamps are exact.** Given times decode to the identical int64 ticks, in the same unit.
  Decoded times are non-decreasing within every block, and block starts never decrease (§3).

## 8. Conformance tests

1. **Bit order:** the test vector in §6.
2. **Order independence:** a unit written with any `diff_orders` setting decodes with the same decoder.
3. **Self-describing units:** decode_unit needs only the unit and returns exactly `num_samples`
   samples and each block's size; headers with another version, reserved flag bits, a time unit
   of 5–7, a sample count over 2^26 or than its blocks can hold, or a body size that doesn't fit
   them are rejected, as are block sizes that don't add up to the sample count, empty blocks with
   nonzero columns, non-finite float anchors and decimal anchors of 2^52 or more.
4. **Time axis:** any reference decodes; the worked example in §6 matches byte for byte; the
   decoder rejects each corrupt time field listed in §6, and the long flag without the irregular
   one.

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
