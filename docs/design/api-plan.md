# fluxcode — API plan (v2)

> **History.** This is the design plan fluxcode was implemented from, kept for its reasoning. It
> is superseded by [SPEC.md](../SPEC.md) (format), [ENCODER.md](../ENCODER.md) (algorithm,
> guarantees) and by the docstrings
> of the `fluxcode` package (API). Where they differ, they win; the main differences are listed
> below.

**What changed during implementation:**

- **Names.** The per-block "head byte" below is the `block_flags` field, and the SPEC sections
  it cites have been renumbered (the format is now SPEC §6, the algorithm ENCODER.md).
- **Self-describing block groups.** A block group starts with a 16-byte header (version, block length, sample
  count), and each block stores its own anchor (SPEC §5). `decode(groups)` takes only the block groups and
  returns one array per block group; `decode_group(group)` returns the block group's samples as a 1-D array. No
  `block_min`, `n_samples` or `block_len` is passed to `decode` or `update`, and min/max/mean are
  summary statistics only.
- **`encode_group(x, params)`** encodes one block group. `encode` returns `(groups, block_mins,
  block_maxs, block_means)` with one array per group, and `update(group, indices, blocks, params)`
  returns min/max/mean for the updated blocks only, in `indices` order.
- **Non-finite values are encoded exactly** (head bit 3 with two code planes, SPEC §2) instead of
  raising (§6 below).
- **No float32 flag.** Decimals stored as float32 upstream decode as the decimal itself.
- **Validation:** `noise_floor_sigma` is None or a finite number ≥ 0; `target_bits_per_sample` is
  None or ≥ 6; `block_len` is a multiple of 8 up to 65,536; `blocks_per_group × block_len` is at
  most 2^26; `diff_orders` may be any set or sequence of orders.
- **More modules:** `_noise`, `_nonfinite` and `_extreme_magnitudes` joined the layout of §3. The
  conformance tests are SPEC §7.
- The prototype the implementation was checked against is now fluxproto,
  [`experimental/tslab/flux/proto.py`](../../experimental/tslab/flux/proto.py). The body keeps the
  old names: `pow2codec.py` is that prototype, and `bench_pow2.py` / `bench_noise.py` are now
  `experimental/bench/fluxproto_sweep.py` / `fluxproto_noise_floor.py`.

`fluxcode` packages the codec specified in [`docs/SPEC.md`](../SPEC.md): power-of-two / decimal
quantization → per-block delta order → bit-shuffled zstd, with a noise floor. The public
surface is small. The stages stay separate inside the implementation, as numba functions
organized into an encoder module and a decoder module, but they aren't a public API.
Performance and code clarity come first.

---

## 1. Public API

```python
import fluxcode

groups, block_min, block_max, block_mean = fluxcode.encode(x, params)     # x: float64[n]
x2 = fluxcode.decode(groups, block_min, n_samples=len(x))                 # float64[n]
group2, new_min, new_max, new_mean = fluxcode.update(groups[3], block_min[180:240],
                                                     indices, new_blocks, params)
```

### 1.1 `encode`

```python
def encode(x: np.ndarray, params: Params = Params()) -> tuple[list[bytes], np.ndarray, np.ndarray, np.ndarray]
```

- **Input:** `x` is float64, 1-D and non-empty. Anything `np.asarray(x, dtype=np.float64)`
  converts is accepted. numpy float64 C-contiguous arrays are the zero-copy fast path;
  pandas and pyarrow inputs go through `to_numpy()` on the caller's side or in that conversion.
- **Blocks and block groups:** `x` is split into blocks of `params.block_len` samples, and the blocks
  into block groups of up to `params.blocks_per_group` blocks (one zstd frame each). A short last block
  is padded by repeating its last value, which leaves min/max unchanged and costs almost
  nothing. The caller keeps `n_samples`.
- **Output:**
  - `groups`: one `bytes` per block group;
  - `block_min`, `block_max`, `block_mean`: float64 per block, for the index columns. Only
    `block_min` is needed to decode.
- **Non-finite values raise** `ValueError`, naming the first bad index (§6).

### 1.2 `decode`

```python
def decode(groups: Sequence[bytes], block_min: np.ndarray, n_samples: int | None = None,
           *, block_len: int = 1000) -> np.ndarray
```

- Returns float64[n_samples] (or every block's samples if `n_samples` is None).
- Block groups are self-describing apart from `block_len` and the minima. The block count comes from
  the block group's decompressed size.
- There's also `decode_group(group, block_min) -> float64[N, block_len]` for one block group.

### 1.3 `update`

```python
def update(group: bytes, block_min: np.ndarray, indices: np.ndarray, blocks: np.ndarray,
           params: Params = Params()) -> tuple[bytes, np.ndarray, np.ndarray, np.ndarray]
```

- **Arguments:**
  - `group`, `block_min`: one existing block group and the minima of its N blocks;
  - `indices`: int[k], the block positions to replace (0..N−1), or N, N+1, … to append;
  - `blocks`: float64[k, block_len], the new samples.
- **Returns:** the new block group, plus `block_min`, `block_max` and `block_mean` for the new block group's
  blocks (old values unchanged where not updated).
- **Untouched blocks decode to identical values.** Their quantized residuals are carried over,
  not re-derived from floats.
- **The per-block-group target** (§2.3), if set, is optimized over the updated blocks only, with a
  budget of `target × k`.

Implementation (clarity first):
1. Decompress the block group, and parse it with the decoder's reader into head bytes, parameters and
   int16 residuals. That's the unshuffle, which is fast.
2. Encode the k new blocks with the encoder's kernel into the same three arrays.
3. Replace or append those rows.
4. Write the block group with the encoder's writer (shuffle and serialize), then compress.

There's no byte-level splicing, even though each block touches 25 places in the block group (its head
byte, 8 byte-planed parameter bytes and 16 plane slices). Parse and write are the same functions
`encode` and `decode` use.
- **Cost for a block group of 60 blocks:** decompress ~120 µs, unshuffle and reshuffle ~90 µs, compress
  ~210 µs, plus a full encode of the k new blocks.
- **Hard limit:** `blocks_per_group` (an append past it raises; start a new block group).

### 1.4 `Params`

```python
@dataclass(frozen=True)
class Params:
    min_quantize_bits: int = 6                  # hard: steps are never coarser than this (1..16)
    max_quantize_bits: int = 16                 # hard: steps are never finer than this (min..16)
    diff_orders: frozenset[int] = frozenset({0, 1, 2, 3})   # one element: no order selection
    noise_floor_sigma: float | None = None      # f: step ≤ f·σ on blocks that look like white noise; None: 0.25 without times, off with them
    target_bits_per_sample: float | None = None # soft per-block-group cap
    decimal_detection: bool = True
    block_len: int = 1000                       # multiple of 8
    blocks_per_group: int = 60
```

Validated in `__post_init__`, raising `ValueError`:
- `1 ≤ min ≤ max ≤ 16` (16 is a format limit: residuals are mod 2^16);
- `diff_orders` is a non-empty subset of {0, 1, 2, 3};
- `noise_floor_sigma` is None or ≥ 0 (0 means off);
- `target_bits_per_sample` is None or > 0;
- `block_len` is a positive multiple of 8, and `blocks_per_group` ≥ 1.

---

## 2. How each block's step is chosen

Exponents: the step is 2^e (or 10^p in decimal mode), and a larger e is a coarser step, i.e.
fewer effective bits.

### 2.1 Precedence

```
e_fine   = exponent(range, max_quantize_bits)          # finest allowed step
e_coarse = exponent(range, min_quantize_bits)          # coarsest allowed step
e        = e_fine
noise floor on, block gated (ρ < −0.6):  e = max(e, floor(log2(f·σ)))
e        = min(e, e_coarse)                             # hard limit
decimal detection at step 2^e                            # a coarser 10^p grid, if the data sits on one
target set, block group over budget (§2.3):  e += k_b, clamped at e_coarse; decimal re-check; requantize
```

- **min/max bits are never violated.**
- **The noise floor** coarsens noisy blocks, for uniform data quality.
- **The target** coarsens expensive blocks whether they're noisy or not: clean high-frequency
  signals, which the noise floor rightly leaves alone. It's the knob for callers who care about
  rate.
- **With both set,** each block takes the coarser step.

### 2.2 Rounds

- **Without a target, every block is quantized exactly once.** The noise floor uses σ from the
  floats, before quantizing.
- **With a target,** blocks the cap coarsens are quantized a second time, once. zstd never
  runs in the loop.

### 2.3 Per-block-group target

A block's estimate h_b is the bit-length-class entropy of its provisional residuals (§9.2),
computed after the provisional quantize, order pick and residual. When the block group's mean estimate
exceeds the target, the cap spreads the block group's bit budget where it costs least:
- Coarsening a block by one bit saves about one bit/sample while its estimate is above ~1
  bit/sample, and little after that.
- Find the smallest k such that Σ_b (h_b − min(k, max(h_b − 1, 0))) ≤ target × N.
- Coarsen each block by k_b = min(k, max(⌈h_b − 1⌉, 0)), clamped at `min_quantize_bits`, then
  requantize those blocks.
- Expensive blocks end up at equal precision; cheap blocks, which would save almost nothing,
  aren't touched.
- It's a soft cap: the estimate is within ~0.5 bits/sample on noise-like data and high on
  signals zstd compresses by repeats (§9.2). To validate on real data.

---

## 3. Package layout

```
(repo root)                      uv; deps: numpy, numba, zstandard
  fluxcode/
    __init__.py                  encode, decode, decode_group, update, Params
    _api.py                      validation, splitting into blocks/block groups, zstd calls, update orchestration
    _encoder.py                  numba: the encoder stages + encode_group
    _decoder.py                  numba: the decoder stages + decode_group
    _format.py                   block group layout: head bits, parameter byte-planes, plane offsets; read_group / write_group
  tests/
    test_conformance.py          docs/SPEC.md §9
    test_stages.py               each numba stage against a small numpy reference
    test_update.py               untouched blocks bit-identical; update == re-encode of the changed blocks
    test_params.py               validation and precedence
  bench/                         ported from bench_pow2.py / bench_noise.py
  docs/SPEC.md
```

### 3.1 `_encoder.py`: one `@njit` function per stage, composed in `encode_group`

| stage | passes over | output |
|---|---|---|
| `block_stats` | floats | min, max, mean, finiteness |
| `noise` (fastmath) | floats | σ, ρ; only if the noise floor is on |
| `plan_step` | — | e from the precedence rules |
| `detect_decimal` | floats, early exit | p, float32 flag; **its q are kept** when it succeeds |
| `quantize` | floats → uint16 | q (skipped when decimal detection produced q) |
| `pick_order` | ints (first 250 samples) | order |
| `residual` | ints | int16 residual mod 2^16, one branch-free loop per order |
| `estimate_bits` | ints (only with a target) | class entropy via `ctlz` + a 17-bin histogram |

`encode_group(X, params...) -> (head, param, residual)` runs the table for all blocks of a block group
in one compiled call. `_format.write_group` does zigzag, bit-shuffle and serialization; zstd
follows in `_api`.

### 3.2 `_decoder.py`

- `read_group` (in `_format`): unshuffle and un-zigzag into (head, param, residual).
- `integrate`: mod 2^16 prefix sums per order.
- `dequantize`: power of two, or decimal, then the float32 rounding.
- `decode_group` fuses these per block.

Each stage is callable from tests. None is public.

---

## 4. Block group format

Unchanged from `docs/SPEC.md` §5: head bytes (order, decimal flag, float32 flag), byte-planed
int64 parameters, 16 bit planes, and one zstd-3 frame.
- **Head bit 3 is reserved for non-finite values** (§6). It's 0 in v1, and a v1 decoder rejects
  block groups that set it.
- **No container in v1.** The caller stores block groups and index columns (Arrow). A self-contained
  container can be added later without touching block groups.

---

## 5. Performance targets and rules

One thread, per 1000-sample block, zstd included, on this machine:
- **encode ≤ 6 µs** with the noise floor on. It's 8.0 µs in `pow2codec.py` today; the spec §7
  items (250-sample order pick, single-pass residual) save ~3 µs.
- **decode ≤ 4 µs.**

Rules:
- One compiled call per block group. There's no Python per block, and buffers are reused across block groups.
- **fastmath only for encoder-only estimates** (noise, entropy). Anything that defines the
  bytes (quantize, residual, dequantize) is compiled without it.
- `nogil=True` on the kernels, so callers can use threads. The library itself is single-threaded.
- `update` is fine at the cost of a full block group parse and write. It's not a hot path.

---

## 6. Non-finite values

**v1: raise.** The check is folded into `block_stats`, so the fast path doesn't pay.

The design is kept ready without touching the fast path:
- **Format:** head bit 3 = 1 marks a block carrying two extra bit planes. Codes: 00 valid,
  01 NaN, 10 +inf, 11 −inf. Finite blocks' bytes are unchanged.
- **Encoder:** replace each non-finite sample with the previous finite value (the next one at
  the block start) before `noise`/`quantize`. That's a zero residual under any order, so
  there's no order-0 forcing and analyze/quantize/delta don't change. min/max/mean are over the
  finite samples.
- **Decoder:** writes NaN/±inf back from the two planes after dequantizing.
- **Cost:** only blocks that contain non-finite values pay.

---

## 7. Changes to `docs/SPEC.md` that come with this

1. Replace `B` with `min_quantize_bits` / `max_quantize_bits`, with the precedence of §2.1.
2. Add `target_bits_per_sample` (per block group, §2.3) and the class-entropy estimate.
3. Noise floor default 0.25; recommended range 0.1–0.5 unchanged.
4. **Guarantee:** max error ≤ half the step used, at most range/(2^min_quantize_bits − ½).
   That's range/63.5 (1.6% of range) at the default of 6, whatever the noise floor or target
   does. Blocks at `max_quantize_bits` keep the ~2^−max bound.
5. Head bit 3 reserved (non-finite planes); the padding of the last block; and `update`
   semantics (untouched blocks decode identically).

---

## 8. Implementation order

1. `git init` the `fluxcode/` repo, then add `Params`, `_format` (read/write block group), `_decoder`
   and `_encoder` without the target. Check byte-identity against `pow2codec.py`, with the same
   parameters and order pick (then switch to the 250-sample pick).
2. `encode` / `decode` / `decode_group`, the conformance tests, and a port of the benchmarks.
3. `update`, with its tests.
4. `target_bits_per_sample`: `estimate_bits`, the per-block-group allocation, and validating the
   estimate on the synthetic sets.
5. The spec updates of §7.

---

## 9. Evidence behind the choices

### 9.1 `min_quantize_bits = 6`

Noise floor at f = 0.25, noisy signals, bits/sample:

| min_quantize_bits | noisy-sine | impulses | gauss-spikes |
|---|---|---|---|
| none | 5.42 | 5.01 | 6.60 |
| 9 | 6.46 | 5.23 | 6.74 |
| 8 | 5.42 | 5.18 | 6.71 |
| 6 | 5.42 | 5.06 | 6.63 |

A noisy block's noise-floor step spans range/(f·σ) steps (≈ 230, about 7.8 bits, on the noisy
sine), so a minimum above that overrides the noise floor. Six bits costs ≤ 1% and bounds the
max error at 1.6% of range.

### 9.2 The size estimate for the target

Mean over blocks at B = 16, compared with actual bit-shuffled zstd-3 size:

| signal | Gaussian (log2 std + 2.05) | class entropy | actual |
|---|---|---|---|
| square wave | 13.79 | 0.06 | 0.05 |
| sensor-0.1 | 10.24 | 1.76 | 1.76 |
| random-walk 0.1 | 12.01 | 5.46 | 5.78 |
| noisy-sine | 13.10 | 13.19 | 13.69 |
| sin-4.12hz | 2.46 | 2.66 | 2.74 |
| chirp | 12.75 | 11.90 | 7.80 |
| sin-50.3hz | 11.19 | 11.02 | 6.64 |

The Gaussian estimate (the old one-shot cap) fails on sparse residuals. The class entropy (the
entropy of each residual's bit length, plus the mantissa bits below its leading 1) tracks
everything not driven by repeats. It overestimates where zstd finds repeats, which is the safe
direction for a rate cap.

### 9.3 Order selection

The stored order is picked on the final integers (exact). A float-domain pick isn't needed
anywhere; the target estimate uses the provisional integer residual. Picking on the first 250
samples agrees with the full pick on 97% of blocks and costs 0.4% in size on continuous data
(nothing on discretized data).
