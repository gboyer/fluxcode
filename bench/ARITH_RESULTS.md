# Arithmetic-coded bit planes vs zstd (experiment, branch `arith-planes`)

`bench/arith_planes.py` codes a unit's 16 residual bit planes (real units, `planes="bit"`) with one
density byte per plane (0 = all-zero plane, skipped) and a static 4-way interleaved rANS over the
plane bytes (Bernoulli(d/256) per bit; the byte's probability depends only on its popcount, so no
big tables), and times it against zstd level 3 on the same plane bytes. Round trip is checked.

2 minutes of each of 14 signals, best of 10, single thread, planes field only (120 KB per unit):

| | size | encode | decode |
|---|---|---|---|
| zstd 3 | 1.00 | 1.00 | 1.00 |
| density + rANS | 1.20 | 2.9x slower | 15-19x slower |

**Verdict: not worth considering.** Slower on both sides and larger. zstd runs near memory speed on
these planes (mostly-zero high planes become long matches or runs; ~0.1 ms per unit), while any
per-symbol arithmetic coder pays a few ns per byte plus a division, whatever the bit-level model.
A plane-density model also can't see the structure zstd uses (periodic signals: quadratic 25x,
chirp 1.6x larger); it only wins ~5% on noisy data. A branchless class search made decode slower.

# Zero-run coding vs zstd (`bench/zero_runs.py`)

Alternating raw/zero runs (first run raw, possibly empty), each length a protobuf varint; a zero run
is only split out of a raw run at 3+ bytes. The encoder skips 8 bytes per step through non-zero
and zero stretches. Same planes and method as above; round trip checked.

| | size | encode | decode |
|---|---|---|---|
| zstd 3 | 1.00 | 1.00 | 1.00 |
| zero runs | 1.22x | 0.37x (2.7x faster) | 0.19x (5x faster) |

Faster than zstd on every signal, both ways. Size is 1.0-1.2x on noisy signals but far worse where
zstd finds more than zeros: all-ones planes (linear: 192x, they're 0xFF), repeats (quadratic 20x),
square wave 3.9x.

**Correction.** A first run of this showed zero runs about as slow as zstd (1.13x enc, 1.14x dec).
That was a numba pitfall, not the format: the raw-run copy was slow. `dst[d:d+n] = src[p:p+n]`
is ~45x slower than memcpy, and a loop `dst[d + k] = src[p + k]` isn't vectorized (84 us vs 2.6 us
for 120 KB; the signed index needs a wraparound check), while a loop over views is. Zero-run
skipping by 8-byte words was the second fix. The rANS numbers above are less exposed (a serial
dependency chain per symbol), but its 2-D `planes[j, i]` indexing wasn't audited.

**Verdict: faster than zstd, but larger.** Worth considering only as a fast mode where the size
cost (about 20% here, unbounded on all-ones or repetitive planes) is acceptable; it would need a
fix for all-ones planes (e.g. XOR with the previous plane or a 0xFF run type) to be safe.

# How much of encode/decode is zstd (`bench/zstd_share.py`)

2 minutes of each of 14 signals, single thread, zstd level 3 on the unit bodies (for `planes="best"`
the encode charge includes the second layout's compress):

| Params | zstd share of encode | zstd share of decode |
|---|---|---|
| `planes="bit"` | 28% | 23% |
| default (`"best"`) | 48% | 28% |

Per signal it ranges from 6-11% (linear, square wave: planes are almost entirely runs) to 60%
(sin-9.87hz, chirp). Replacing zstd by zero runs (0.37x encode, 0.19x decode of zstd's time) would
cut total encode by about 18% (bit) / 30% (default) and decode by about 19% / 22%: at most a
1.2-1.4x speedup of the whole call, for ~20% larger units.
