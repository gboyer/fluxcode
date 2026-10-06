# fluxcode update benchmark (bench/update.py, 2026-10-02, AC power, single thread)

At d039350, default params (effort 4: both layouts compressed, one block run). Each update against encoding
the same series from scratch. The append row's update builds a 61-block group and is compared with encoding
the 60-block one. The main table is the Python path (FLUXCODE_RUST=0); a second table is the same run with the
optional Rust extension, which now also packs the bodies of time-axis block groups and of updates (`pack_group`), so
at efforts 5 and up their block flushes no longer hold the GIL (not measured here).

Against the run at 5ee16fa (same session A/B of the code before this work, random-walk): the encode of
the 3,600-block group is 1617 → 1402 µs (-13%), because the second layout's body is derived from the first by
one transpose of the residual region instead of written from the rows (3,600 blocks of 10: `write_group` in bit planes 195 µs; in byte planes 52 µs plus the 24 µs transpose);
the 60-block straddle update 612 → 385 µs (+7% → -31% against its encode); the 3,600-block upsert of one
sample in each block 4532 → 1939 µs (+180% → +38% against its encode; this row had been added to the script
after the last run, so it was never recorded), by merging the decoded and new samples in one numba pass,
computing the covered blocks with array operations and checking block order with one batched call.
What is left of that row is the work itself: decoding, re-encoding and compressing every block.

The 3,600-block groups' bit-plane body is 248,400 bytes, below 256 KB, where libzstd picks its level-3
parameters for smaller inputs: compressing that body takes 348 µs instead of 148. Forcing the large-input
parameters for every block group was measured and doesn't pay: standard 60-block groups (about 120 KB)
come out 0.03% smaller with 4.6% more zstd time.

Block flushes (effort 5 and up) don't avoid that switch: zstd sizes its parameters from the input size
the frame declares, and a flushed frame declares the same size (measured 2026-10-02, 10-sample blocks,
random-walk bit planes: body 243 KB compresses in 335 µs in one run and 415 µs flushed; 256 KB in 130 µs and
244 µs). Byte planes don't show the jump.

arm64 Darwin 25.6.0, Python 3.11.13, numpy 2.4.6; median of 300 calls (100 for the 3,600-block group)

| signal | operation | encode from scratch | update | vs encode |
|---|---|---|---|---|
| random-walk | `update`, 1 of 60 blocks | 303 µs | 265 µs | -38 µs (-13%) |
| random-walk | `update` with times, 1 of 60 | 320 µs | 287 µs | -33 µs (-10%) |
| random-walk | `update` with times, append 1 | 320 µs | 313 µs | -7 µs (-2%) |
| random-walk | `update_time_blocks`, 2 s straddling 3 of 60 blocks | 555 µs | 385 µs | -170 µs (-31%) |
| random-walk | `update_time_blocks`, 2 s straddling 3 of 3,600 blocks | 1402 µs | 1012 µs | -390 µs (-28%) |
| random-walk | `update_time_blocks` upsert, 1 sample in 1 of 3,600 blocks | 1402 µs | 986 µs | -417 µs (-30%) |
| random-walk | `update_time_blocks` upsert, 1 sample in each of 3,600 blocks | 1402 µs | 1939 µs | +536 µs (+38%) |
| noisy-sine | `update`, 1 of 60 blocks | 234 µs | 191 µs | -43 µs (-18%) |
| noisy-sine | `update` with times, 1 of 60 | 255 µs | 212 µs | -43 µs (-17%) |
| noisy-sine | `update` with times, append 1 | 255 µs | 240 µs | -15 µs (-6%) |
| noisy-sine | `update_time_blocks`, 2 s straddling 3 of 60 blocks | 491 µs | 309 µs | -182 µs (-37%) |
| noisy-sine | `update_time_blocks`, 2 s straddling 3 of 3,600 blocks | 1394 µs | 1003 µs | -391 µs (-28%) |
| noisy-sine | `update_time_blocks` upsert, 1 sample in 1 of 3,600 blocks | 1394 µs | 979 µs | -415 µs (-30%) |
| noisy-sine | `update_time_blocks` upsert, 1 sample in each of 3,600 blocks | 1394 µs | 1956 µs | +562 µs (+40%) |
| sensor-0.1 | `update`, 1 of 60 blocks | 330 µs | 290 µs | -40 µs (-12%) |
| sensor-0.1 | `update` with times, 1 of 60 | 349 µs | 314 µs | -35 µs (-10%) |
| sensor-0.1 | `update` with times, append 1 | 349 µs | 334 µs | -16 µs (-4%) |
| sensor-0.1 | `update_time_blocks`, 2 s straddling 3 of 60 blocks | 580 µs | 406 µs | -174 µs (-30%) |
| sensor-0.1 | `update_time_blocks`, 2 s straddling 3 of 3,600 blocks | 873 µs | 541 µs | -333 µs (-38%) |
| sensor-0.1 | `update_time_blocks` upsert, 1 sample in 1 of 3,600 blocks | 873 µs | 517 µs | -356 µs (-41%) |
| sensor-0.1 | `update_time_blocks` upsert, 1 sample in each of 3,600 blocks | 873 µs | 1571 µs | +697 µs (+80%) |

## With the Rust extension

arm64 Darwin 25.6.0, Python 3.11.13, numpy 2.4.6; median of 300 calls (100 for the 3,600-block group)

| signal | operation | encode from scratch | update | vs encode |
|---|---|---|---|---|
| random-walk | `update`, 1 of 60 blocks | 282 µs | 248 µs | -35 µs (-12%) |
| random-walk | `update` with times, 1 of 60 | 302 µs | 268 µs | -34 µs (-11%) |
| random-walk | `update` with times, append 1 | 302 µs | 292 µs | -10 µs (-3%) |
| random-walk | `update_time_blocks`, 2 s straddling 3 of 60 blocks | 536 µs | 363 µs | -173 µs (-32%) |
| random-walk | `update_time_blocks`, 2 s straddling 3 of 3,600 blocks | 1383 µs | 985 µs | -398 µs (-29%) |
| random-walk | `update_time_blocks` upsert, 1 sample in 1 of 3,600 blocks | 1383 µs | 960 µs | -422 µs (-31%) |
| random-walk | `update_time_blocks` upsert, 1 sample in each of 3,600 blocks | 1383 µs | 1920 µs | +538 µs (+39%) |
| noisy-sine | `update`, 1 of 60 blocks | 213 µs | 172 µs | -40 µs (-19%) |
| noisy-sine | `update` with times, 1 of 60 | 237 µs | 192 µs | -45 µs (-19%) |
| noisy-sine | `update` with times, append 1 | 237 µs | 219 µs | -18 µs (-7%) |
| noisy-sine | `update_time_blocks`, 2 s straddling 3 of 60 blocks | 473 µs | 288 µs | -185 µs (-39%) |
| noisy-sine | `update_time_blocks`, 2 s straddling 3 of 3,600 blocks | 1398 µs | 993 µs | -404 µs (-29%) |
| noisy-sine | `update_time_blocks` upsert, 1 sample in 1 of 3,600 blocks | 1398 µs | 970 µs | -428 µs (-31%) |
| noisy-sine | `update_time_blocks` upsert, 1 sample in each of 3,600 blocks | 1398 µs | 1953 µs | +555 µs (+40%) |
| sensor-0.1 | `update`, 1 of 60 blocks | 318 µs | 281 µs | -37 µs (-12%) |
| sensor-0.1 | `update` with times, 1 of 60 | 340 µs | 305 µs | -35 µs (-10%) |
| sensor-0.1 | `update` with times, append 1 | 340 µs | 325 µs | -15 µs (-4%) |
| sensor-0.1 | `update_time_blocks`, 2 s straddling 3 of 60 blocks | 572 µs | 396 µs | -176 µs (-31%) |
| sensor-0.1 | `update_time_blocks`, 2 s straddling 3 of 3,600 blocks | 863 µs | 526 µs | -337 µs (-39%) |
| sensor-0.1 | `update_time_blocks` upsert, 1 sample in 1 of 3,600 blocks | 863 µs | 501 µs | -362 µs (-42%) |
| sensor-0.1 | `update_time_blocks` upsert, 1 sample in each of 3,600 blocks | 863 µs | 1565 µs | +702 µs (+81%) |
