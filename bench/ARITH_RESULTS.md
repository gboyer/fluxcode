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
