# Faster coders than zstd for the residual bit planes?

**Question.** zstd (level 3) is 28% of `encode()` with `planes="bit"`, 48% with the default
`planes="best"` (which compresses both layouts), and 23–28% of `decode()`. Would a simpler coder
for the 16 residual bit planes be meaningfully faster?

**Answer.** Not enough to justify leaving zstd. The best of three attempts, a raw/zero/0xFF
run-length coder, is 2× faster to encode and 4–5× faster to decode than zstd on the planes alone,
which speeds the whole call up by roughly 1.2–1.3×, and it makes the planes 17% (0.84 bits/sample)
larger. A custom format also costs complexity that a well-known codec doesn't. Not for the spec.

- Data: the report's synthetic signals (`tests/_signals.py`), 2 minutes of each of 14 kinds, real
  fluxcode block groups with `planes="bit"`. Only the residual planes field is coded (120,000 raw bytes
  per 60,000-sample block group, 16 bits/sample); the rest of the body and the header aren't.
- Timings: best of N, one thread, numba coders against python-zstandard, which has a little call
  overhead on ~100 µs jobs. Round trips are checked in every script.
- Related, earlier: `bench/fluxproto_plane_models.py` costs per-plane probability models (ideal
  arithmetic coding) against zstd for size only.

## Results

All relative to zstd level 3 on the same plane bytes; totals over the 14 signals.

| coder | size | encode time | decode time |
|---|---|---|---|
| density byte per plane + 4-way rANS (`arith_planes.py`) | 1.21 | 2.9× | 15–19× |
| alternating raw/zero runs (`zero_runs.py`) | 1.22 | 0.37 | 0.19 |
| raw / zero / 0xFF runs (`rle_variants.py`) | 1.17 | 0.49–0.53 | 0.21 |
| PackBits-style, any repeated byte (`rle_variants.py`) | 1.19 | 0.58–0.64 | 0.26–0.28 |

### Size in bits per sample (planes only, 60,000 samples per block group, raw = 16)

| signal | zstd 3 | alt-zero | zero/FF | any-byte | zero/FF − zstd |
|---|---|---|---|---|---|
| linear | 0.005 | 1.001 | 0.033 | 0.041 | +0.028 |
| quadratic | 0.214 | 4.229 | 2.015 | 1.991 | +1.801 |
| sin-4.12hz | 2.719 | 3.216 | 3.295 | 3.409 | +0.576 |
| sin-9.87hz | 3.690 | 4.296 | 4.167 | 4.481 | +0.477 |
| sin-50.3hz | 6.760 | 11.103 | 11.126 | 11.158 | +4.366 |
| gauss-spikes | 6.376 | 6.401 | 6.421 | 6.491 | +0.045 |
| impulses | 5.034 | 5.251 | 5.284 | 5.329 | +0.249 |
| square-2.24hz | 0.044 | 0.173 | 0.226 | 0.230 | +0.181 |
| random-walk | 12.631 | 12.687 | 12.695 | 12.811 | +0.064 |
| chirp | 7.830 | 10.489 | 10.611 | 10.696 | +2.781 |
| noisy-sine | 5.421 | 5.765 | 5.770 | 5.885 | +0.349 |
| sensor-0.1 | 1.764 | 2.188 | 2.223 | 2.323 | +0.460 |
| random-walk q0.01 | 9.259 | 9.276 | 9.288 | 9.363 | +0.029 |
| noisy-sine q0.1 | 5.423 | 5.765 | 5.771 | 5.884 | +0.348 |
| **mean** | 4.798 | 5.846 | 5.637 | 5.721 | +0.839 |

Noisy and random signals are within 0–6% of zstd; sines and sensor data lose 13–26%; sin-50.3hz,
chirp and quadratic lose the most (periodic structure that zstd matches and no run coder sees).

### How much of the call is zstd (`zstd_share.py`)

| Params | zstd share of encode | zstd share of decode |
|---|---|---|
| `planes="bit"` | 28% | 23% |
| default (`"best"`) | 48% | 27–28% |

Per signal it ranges from 6–11% (linear, square wave: planes are almost all runs) to ~60%
(sin-9.87hz, chirp). The rest of encode (quantize, delta, bit transposition) is untouched by the
choice of coder.

## What each coder is, and why it lost or won

- **Density + rANS.** One byte per plane with the density of 1 bits (0 = skip the plane), then the
  plane bytes coded as i.i.d. Bernoulli bytes by a 4-way interleaved 32-bit rANS. A byte's
  probability depends only on its popcount, so the model is nine frequencies per density, and no
  table is indexed by symbol. It is serial per symbol (a few ns/byte plus a division) while zstd
  runs near memory speed on mostly-zero planes and stores incompressible planes raw, so it loses on
  time; and a density can't see repeats, so it loses on size (quadratic ~25× larger).
- **Alternating raw/zero runs.** Varint lengths alternate raw and zero runs, the first raw. Faster
  than zstd on every signal, but all-ones planes (linear: 192× zstd's size) and periodic planes
  defeat it.
- **Raw / zero / 0xFF runs.** A token is a varint of `len << 2 | kind`; runs of 3+ zero or 0xFF
  bytes are split out. Fixes the all-ones planes (linear 192× → 6×, 78 vs ~490 bytes per block group).
- **PackBits-style.** `len << 1 | kind`, with a fill byte after a repeat token, runs of 4+ of any
  byte. Catches nothing more (the periodic planes have periods beyond one byte), spends a byte per
  run, and is slower.

## Lessons: numba pitfalls that made the first runs wrong

The first zero-run numbers (1.13× encode, 1.14× decode of zstd's time) were an artifact:

- Slice assignment `dst[a:b] = src[c:d]` is ~45× slower than memcpy (111 µs vs 2.5 µs for 120 KB).
- A loop `dst[d + k] = src[p + k]` isn't vectorized either (84 µs): the signed index needs a
  wraparound check. A loop over views (`a = dst[d:d+n]; b = src[p:p+n]; a[k] = b[k]`) is (2.6 µs).
- Scanning runs 8 bytes at a time (`uint64` views) took encode from slower to faster than zstd.

The rANS coder's serial dependency makes it less exposed, but its 2-D indexing wasn't audited.

## Reproduce

```sh
cd experimental
uv run python plane_coders/arith_planes.py    # density + rANS, ~20 s
uv run python plane_coders/zero_runs.py       # alternating raw/zero runs
uv run python plane_coders/rle_variants.py    # all three run coders side by side (size/enc/dec per signal)
uv run python plane_coders/zstd_share.py      # zstd's share of encode() and decode()
```
