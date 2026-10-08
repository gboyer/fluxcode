# fluxcode update benchmark (bench/update.py, 2026-10-07, AC power, single thread)

At 4e89772, default params (effort 4: both layouts compressed, one block run). Each update against encoding
the same series from scratch. The append row's update builds a block group of 61 blocks and is compared with encoding
the 60-block one. The main table is the Python path (FLUXCODE_RUST=0); a second table is the same run with the
optional Rust extension, which also packs the bodies of time-axis block groups and of updates (`pack_group`), so
at efforts 5 and up their block flushes don't hold the GIL (not measured here).

Against 0581c29 (the noise floor's cadence and window rule changed; these have regular 1 kHz times, so the same
steps as without times), everything is within run-to-run noise: encoding the 60-block random walk with times takes
333 → 325 µs, and `update` with times 288 → 291 µs (the load average was 2.1–2.4 from other apps). Updates stay
2–43% cheaper than encoding, and the upsert of one sample in each of 3,600 blocks 37–84% dearer.

Notes from the run at d039350 (2026-10-02), still accurate:

Against the run at 5ee16fa (same session A/B of the code before this work, random-walk): the encode of
the block group of 3,600 blocks is 1617 → 1402 µs (-13%), because the second layout's body is derived from the first by
one transpose of the residual region instead of written from the rows (3,600 blocks of 10: `write_group` in bit planes 195 µs; in byte planes 52 µs plus the 24 µs transpose);
the 60-block straddle update 612 → 385 µs (+7% → -31% against its encode); the 3,600-block upsert of one
sample in each block 4532 → 1939 µs (+180% → +38% against its encode; this row had been added to the script
after the last run, so it was never recorded), by merging the decoded and new samples in one numba pass,
computing the covered blocks with array operations and checking block order with one batched call.
What is left of that row is the work itself: decoding, re-encoding and compressing every block.

The bit-plane body of a block group of 3,600 blocks is 248,400 bytes, below 256 KB, where libzstd picks its level-3
parameters for smaller inputs: compressing that body takes 348 µs instead of 148. Forcing the large-input
parameters for every block group was measured and doesn't pay: standard block groups of 60 blocks (about 120 KB)
come out 0.03% smaller with 4.6% more zstd time.

Block flushes (effort 5 and up) don't avoid that switch: zstd sizes its parameters from the input size
the frame declares, and a flushed frame declares the same size (measured 2026-10-02, 10-sample blocks,
random-walk bit planes: body 243 KB compresses in 335 µs in one run and 415 µs flushed; 256 KB in 130 µs and
244 µs). Byte planes don't show the jump.

arm64 Darwin 25.6.0, Python 3.11.13, numpy 2.4.6; median of 300 calls (100 for the block group of 3,600 blocks)

| signal | operation | encode from scratch | update | vs encode |
|---|---|---|---|---|
| random-walk | `update`, 1 of 60 blocks | 309 µs | 269 µs | -40 µs (-13%) |
| random-walk | `update` with times, 1 of 60 | 325 µs | 291 µs | -33 µs (-10%) |
| random-walk | `update` with times, append 1 | 325 µs | 318 µs | -7 µs (-2%) |
| random-walk | `update_time_blocks`, 2 s straddling 3 of 60 blocks | 565 µs | 389 µs | -176 µs (-31%) |
| random-walk | `update_time_blocks`, 2 s straddling 3 of 3,600 blocks | 1424 µs | 1013 µs | -411 µs (-29%) |
| random-walk | `update_time_blocks` upsert, 1 sample in 1 of 3,600 blocks | 1424 µs | 988 µs | -436 µs (-31%) |
| random-walk | `update_time_blocks` upsert, 1 sample in each of 3,600 blocks | 1424 µs | 1956 µs | +532 µs (+37%) |
| noisy-sine | `update`, 1 of 60 blocks | 240 µs | 192 µs | -48 µs (-20%) |
| noisy-sine | `update` with times, 1 of 60 | 261 µs | 212 µs | -49 µs (-19%) |
| noisy-sine | `update` with times, append 1 | 261 µs | 240 µs | -21 µs (-8%) |
| noisy-sine | `update_time_blocks`, 2 s straddling 3 of 60 blocks | 497 µs | 309 µs | -189 µs (-38%) |
| noisy-sine | `update_time_blocks`, 2 s straddling 3 of 3,600 blocks | 1401 µs | 1001 µs | -400 µs (-29%) |
| noisy-sine | `update_time_blocks` upsert, 1 sample in 1 of 3,600 blocks | 1401 µs | 974 µs | -427 µs (-30%) |
| noisy-sine | `update_time_blocks` upsert, 1 sample in each of 3,600 blocks | 1401 µs | 1969 µs | +568 µs (+41%) |
| sensor-0.1 | `update`, 1 of 60 blocks | 337 µs | 299 µs | -38 µs (-11%) |
| sensor-0.1 | `update` with times, 1 of 60 | 356 µs | 317 µs | -39 µs (-11%) |
| sensor-0.1 | `update` with times, append 1 | 356 µs | 342 µs | -14 µs (-4%) |
| sensor-0.1 | `update_time_blocks`, 2 s straddling 3 of 60 blocks | 593 µs | 413 µs | -180 µs (-30%) |
| sensor-0.1 | `update_time_blocks`, 2 s straddling 3 of 3,600 blocks | 884 µs | 547 µs | -337 µs (-38%) |
| sensor-0.1 | `update_time_blocks` upsert, 1 sample in 1 of 3,600 blocks | 884 µs | 521 µs | -363 µs (-41%) |
| sensor-0.1 | `update_time_blocks` upsert, 1 sample in each of 3,600 blocks | 884 µs | 1600 µs | +715 µs (+81%) |

## With the Rust extension

arm64 Darwin 25.6.0, Python 3.11.13, numpy 2.4.6; median of 300 calls (100 for the block group of 3,600 blocks)

| signal | operation | encode from scratch | update | vs encode |
|---|---|---|---|---|
| random-walk | `update`, 1 of 60 blocks | 288 µs | 249 µs | -39 µs (-14%) |
| random-walk | `update` with times, 1 of 60 | 307 µs | 268 µs | -39 µs (-13%) |
| random-walk | `update` with times, append 1 | 307 µs | 296 µs | -11 µs (-4%) |
| random-walk | `update_time_blocks`, 2 s straddling 3 of 60 blocks | 533 µs | 366 µs | -166 µs (-31%) |
| random-walk | `update_time_blocks`, 2 s straddling 3 of 3,600 blocks | 1396 µs | 1009 µs | -386 µs (-28%) |
| random-walk | `update_time_blocks` upsert, 1 sample in 1 of 3,600 blocks | 1396 µs | 985 µs | -410 µs (-29%) |
| random-walk | `update_time_blocks` upsert, 1 sample in each of 3,600 blocks | 1396 µs | 1956 µs | +560 µs (+40%) |
| noisy-sine | `update`, 1 of 60 blocks | 220 µs | 176 µs | -45 µs (-20%) |
| noisy-sine | `update` with times, 1 of 60 | 244 µs | 194 µs | -50 µs (-20%) |
| noisy-sine | `update` with times, append 1 | 244 µs | 222 µs | -22 µs (-9%) |
| noisy-sine | `update_time_blocks`, 2 s straddling 3 of 60 blocks | 480 µs | 292 µs | -189 µs (-39%) |
| noisy-sine | `update_time_blocks`, 2 s straddling 3 of 3,600 blocks | 1406 µs | 998 µs | -408 µs (-29%) |
| noisy-sine | `update_time_blocks` upsert, 1 sample in 1 of 3,600 blocks | 1406 µs | 974 µs | -432 µs (-31%) |
| noisy-sine | `update_time_blocks` upsert, 1 sample in each of 3,600 blocks | 1406 µs | 1969 µs | +563 µs (+40%) |
| sensor-0.1 | `update`, 1 of 60 blocks | 324 µs | 283 µs | -41 µs (-13%) |
| sensor-0.1 | `update` with times, 1 of 60 | 341 µs | 300 µs | -41 µs (-12%) |
| sensor-0.1 | `update` with times, append 1 | 341 µs | 323 µs | -18 µs (-5%) |
| sensor-0.1 | `update_time_blocks`, 2 s straddling 3 of 60 blocks | 577 µs | 395 µs | -182 µs (-32%) |
| sensor-0.1 | `update_time_blocks`, 2 s straddling 3 of 3,600 blocks | 866 µs | 527 µs | -339 µs (-39%) |
| sensor-0.1 | `update_time_blocks` upsert, 1 sample in 1 of 3,600 blocks | 866 µs | 503 µs | -363 µs (-42%) |
| sensor-0.1 | `update_time_blocks` upsert, 1 sample in each of 3,600 blocks | 866 µs | 1594 µs | +728 µs (+84%) |
