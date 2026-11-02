# fluxcode update benchmark (bench/update.py, 2026-11-01, AC power, single thread)

With the scaled grids, default params (effort 4: both layouts compressed, one block run). Each update against encoding
the same series from scratch. The append row's update builds a block group of 61 blocks and is compared with encoding
the 60-block one. The main table is the Python path (FLUXCODE_RUST=0); a second table is the same run with the
optional Rust extension, which also packs the bodies of time-axis block groups and of updates (`pack_group`), so
at efforts 5 and up their block flushes don't hold the GIL (not measured here).

Against `main` at b48c7a3 in the same session (Python path): the updates of 60-block groups are within ±6 µs (±2%),
and an update re-encodes each block as a fresh encode would (no cross-block hints). The block group of 3,600 short
blocks encodes from scratch +275 µs (+20%) slower for the random walk and noisy sine and +99 µs (+10%) for
sensor-0.1, the per-block cost of the grid search, and so does the upsert into each of its blocks (+299 µs, +15%, for
the random walk). An update of 3 of its blocks is −211 µs (−21%), −245 µs (−24%) and +27 µs (+5%) against b48c7a3.

arm64 Darwin 25.6.0, Python 3.11.13, numpy 2.4.6; median of 300 calls (100 for the block group of 3,600 blocks)

| signal | operation | encode from scratch | update | vs encode |
|---|---|---|---|---|
| random-walk | `update`, 1 of 60 blocks | 294 µs | 270 µs | -24 µs (-8%) |
| random-walk | `update` with times, 1 of 60 | 309 µs | 288 µs | -21 µs (-7%) |
| random-walk | `update` with times, append 1 | 309 µs | 313 µs | +4 µs (+1%) |
| random-walk | `update_time_blocks`, 2 s straddling 3 of 60 blocks | 530 µs | 386 µs | -144 µs (-27%) |
| random-walk | `update_time_blocks`, 2 s straddling 3 of 3,600 blocks | 1683 µs | 816 µs | -867 µs (-52%) |
| random-walk | `update_time_blocks` upsert, 1 sample in 1 of 3,600 blocks | 1683 µs | 766 µs | -917 µs (-55%) |
| random-walk | `update_time_blocks` upsert, 1 sample in each of 3,600 blocks | 1683 µs | 2266 µs | +583 µs (+35%) |
| noisy-sine | `update`, 1 of 60 blocks | 244 µs | 193 µs | -51 µs (-21%) |
| noisy-sine | `update` with times, 1 of 60 | 265 µs | 213 µs | -52 µs (-20%) |
| noisy-sine | `update` with times, append 1 | 265 µs | 241 µs | -24 µs (-9%) |
| noisy-sine | `update_time_blocks`, 2 s straddling 3 of 60 blocks | 493 µs | 311 µs | -183 µs (-37%) |
| noisy-sine | `update_time_blocks`, 2 s straddling 3 of 3,600 blocks | 1668 µs | 772 µs | -896 µs (-54%) |
| noisy-sine | `update_time_blocks` upsert, 1 sample in 1 of 3,600 blocks | 1668 µs | 748 µs | -921 µs (-55%) |
| noisy-sine | `update_time_blocks` upsert, 1 sample in each of 3,600 blocks | 1668 µs | 2235 µs | +567 µs (+34%) |
| sensor-0.1 | `update`, 1 of 60 blocks | 344 µs | 293 µs | -51 µs (-15%) |
| sensor-0.1 | `update` with times, 1 of 60 | 362 µs | 315 µs | -47 µs (-13%) |
| sensor-0.1 | `update` with times, append 1 | 362 µs | 338 µs | -25 µs (-7%) |
| sensor-0.1 | `update_time_blocks`, 2 s straddling 3 of 60 blocks | 578 µs | 409 µs | -169 µs (-29%) |
| sensor-0.1 | `update_time_blocks`, 2 s straddling 3 of 3,600 blocks | 1087 µs | 569 µs | -518 µs (-48%) |
| sensor-0.1 | `update_time_blocks` upsert, 1 sample in 1 of 3,600 blocks | 1087 µs | 546 µs | -541 µs (-50%) |
| sensor-0.1 | `update_time_blocks` upsert, 1 sample in each of 3,600 blocks | 1087 µs | 1817 µs | +730 µs (+67%) |

## With the Rust extension

arm64 Darwin 25.6.0, Python 3.11.13, numpy 2.4.6; median of 300 calls (100 for the block group of 3,600 blocks)

| signal | operation | encode from scratch | update | vs encode |
|---|---|---|---|---|
| random-walk | `update`, 1 of 60 blocks | 276 µs | 232 µs | -44 µs (-16%) |
| random-walk | `update` with times, 1 of 60 | 293 µs | 250 µs | -43 µs (-15%) |
| random-walk | `update` with times, append 1 | 293 µs | 273 µs | -20 µs (-7%) |
| random-walk | `update_time_blocks`, 2 s straddling 3 of 60 blocks | 511 µs | 339 µs | -172 µs (-34%) |
| random-walk | `update_time_blocks`, 2 s straddling 3 of 3,600 blocks | 1550 µs | 710 µs | -839 µs (-54%) |
| random-walk | `update_time_blocks` upsert, 1 sample in 1 of 3,600 blocks | 1550 µs | 687 µs | -862 µs (-56%) |
| random-walk | `update_time_blocks` upsert, 1 sample in each of 3,600 blocks | 1550 µs | 2090 µs | +540 µs (+35%) |
| noisy-sine | `update`, 1 of 60 blocks | 221 µs | 163 µs | -58 µs (-26%) |
| noisy-sine | `update` with times, 1 of 60 | 239 µs | 180 µs | -59 µs (-25%) |
| noisy-sine | `update` with times, append 1 | 239 µs | 206 µs | -33 µs (-14%) |
| noisy-sine | `update_time_blocks`, 2 s straddling 3 of 60 blocks | 450 µs | 270 µs | -179 µs (-40%) |
| noisy-sine | `update_time_blocks`, 2 s straddling 3 of 3,600 blocks | 1544 µs | 709 µs | -835 µs (-54%) |
| noisy-sine | `update_time_blocks` upsert, 1 sample in 1 of 3,600 blocks | 1544 µs | 688 µs | -856 µs (-55%) |
| noisy-sine | `update_time_blocks` upsert, 1 sample in each of 3,600 blocks | 1544 µs | 2106 µs | +562 µs (+36%) |
| sensor-0.1 | `update`, 1 of 60 blocks | 322 µs | 260 µs | -62 µs (-19%) |
| sensor-0.1 | `update` with times, 1 of 60 | 338 µs | 277 µs | -61 µs (-18%) |
| sensor-0.1 | `update` with times, append 1 | 338 µs | 301 µs | -37 µs (-11%) |
| sensor-0.1 | `update_time_blocks`, 2 s straddling 3 of 60 blocks | 534 µs | 364 µs | -170 µs (-32%) |
| sensor-0.1 | `update_time_blocks`, 2 s straddling 3 of 3,600 blocks | 1004 µs | 509 µs | -495 µs (-49%) |
| sensor-0.1 | `update_time_blocks` upsert, 1 sample in 1 of 3,600 blocks | 1004 µs | 486 µs | -518 µs (-52%) |
| sensor-0.1 | `update_time_blocks` upsert, 1 sample in each of 3,600 blocks | 1004 µs | 1710 µs | +707 µs (+70%) |
