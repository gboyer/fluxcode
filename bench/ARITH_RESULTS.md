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
is split out of a raw run only at 3+ bytes. The encoder skips 8 non-zero bytes per step. Same
planes, same method as above; round trip checked.

| | size | encode | decode |
|---|---|---|---|
| zstd 3 | 1.00 | 1.00 | 1.00 |
| zero runs | 1.22x | 1.13x slower | 1.14x slower |

Per signal, encode ranges from 0.55x (chirp) to 4.4x (linear), decode from 0.53x (noisy sine) to
5.9x (random walk, where zstd stores the plane raw and just memcpys). Size is about equal (1.0 to
1.2x) on noisy signals but far worse where zstd finds more than zeros: all-ones planes (linear:
192x, because those planes are 0xFF, not 0x00), repeats (quadratic 20x), square wave 3.9x.

**Verdict: not faster than zstd overall**, and larger. It wins decode on some mid-entropy signals
(up to ~2x), loses it on incompressible ones, so there's no case for it as a replacement.
