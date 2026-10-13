# fluxcode update benchmark (bench/update.py, 2026-10-13, AC power, single thread)

At 7fa0991, default params (effort 4: both layouts compressed, one block run). Each update against encoding
the same series from scratch. The append row's update builds a block group of 61 blocks and is compared with encoding
the 60-block one. The main table is the Python path (FLUXCODE_RUST=0); a second table is the same run with the
optional Rust extension, which also packs the bodies of time-axis block groups and of updates (`pack_group`), so
at efforts 5 and up their block flushes don't hold the GIL (not measured here).

Against 4e89772 (since then timestamp rounding, which these exact updates don't use, and clean-ups), everything is
within run-to-run noise: encoding the 60-block random walk with times takes 325 → 321 µs, and `update` with times
291 → 287 µs (the load average was 1.5–2.6). Updates stay 3–43% cheaper than encoding, and the upsert of one sample
in each of 3,600 blocks 37–81% dearer. Lossy updates (`replace_time_blocks`) re-encode whole blocks and aren't timed
here.

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
| random-walk | `update`, 1 of 60 blocks | 291 µs | 266 µs | -24 µs (-8%) |
| random-walk | `update` with times, 1 of 60 | 321 µs | 287 µs | -34 µs (-11%) |
| random-walk | `update` with times, append 1 | 321 µs | 312 µs | -9 µs (-3%) |
| random-walk | `update_time_blocks`, 2 s straddling 3 of 60 blocks | 526 µs | 387 µs | -139 µs (-26%) |
| random-walk | `update_time_blocks`, 2 s straddling 3 of 3,600 blocks | 1411 µs | 1031 µs | -380 µs (-27%) |
| random-walk | `update_time_blocks` upsert, 1 sample in 1 of 3,600 blocks | 1411 µs | 1002 µs | -409 µs (-29%) |
| random-walk | `update_time_blocks` upsert, 1 sample in each of 3,600 blocks | 1411 µs | 1968 µs | +557 µs (+39%) |
| noisy-sine | `update`, 1 of 60 blocks | 241 µs | 194 µs | -47 µs (-20%) |
| noisy-sine | `update` with times, 1 of 60 | 263 µs | 214 µs | -48 µs (-18%) |
| noisy-sine | `update` with times, append 1 | 263 µs | 242 µs | -21 µs (-8%) |
| noisy-sine | `update_time_blocks`, 2 s straddling 3 of 60 blocks | 498 µs | 312 µs | -186 µs (-37%) |
| noisy-sine | `update_time_blocks`, 2 s straddling 3 of 3,600 blocks | 1399 µs | 1016 µs | -384 µs (-27%) |
| noisy-sine | `update_time_blocks` upsert, 1 sample in 1 of 3,600 blocks | 1399 µs | 990 µs | -409 µs (-29%) |
| noisy-sine | `update_time_blocks` upsert, 1 sample in each of 3,600 blocks | 1399 µs | 1963 µs | +564 µs (+40%) |
| sensor-0.1 | `update`, 1 of 60 blocks | 335 µs | 294 µs | -41 µs (-12%) |
| sensor-0.1 | `update` with times, 1 of 60 | 349 µs | 312 µs | -37 µs (-11%) |
| sensor-0.1 | `update` with times, append 1 | 349 µs | 335 µs | -14 µs (-4%) |
| sensor-0.1 | `update_time_blocks`, 2 s straddling 3 of 60 blocks | 585 µs | 405 µs | -180 µs (-31%) |
| sensor-0.1 | `update_time_blocks`, 2 s straddling 3 of 3,600 blocks | 863 µs | 538 µs | -325 µs (-38%) |
| sensor-0.1 | `update_time_blocks` upsert, 1 sample in 1 of 3,600 blocks | 863 µs | 512 µs | -352 µs (-41%) |
| sensor-0.1 | `update_time_blocks` upsert, 1 sample in each of 3,600 blocks | 863 µs | 1560 µs | +697 µs (+81%) |

## With the Rust extension

arm64 Darwin 25.6.0, Python 3.11.13, numpy 2.4.6; median of 300 calls (100 for the block group of 3,600 blocks)

| signal | operation | encode from scratch | update | vs encode |
|---|---|---|---|---|
| random-walk | `update`, 1 of 60 blocks | 269 µs | 232 µs | -37 µs (-14%) |
| random-walk | `update` with times, 1 of 60 | 286 µs | 250 µs | -36 µs (-12%) |
| random-walk | `update` with times, append 1 | 286 µs | 274 µs | -12 µs (-4%) |
| random-walk | `update_time_blocks`, 2 s straddling 3 of 60 blocks | 505 µs | 340 µs | -165 µs (-33%) |
| random-walk | `update_time_blocks`, 2 s straddling 3 of 3,600 blocks | 1367 µs | 978 µs | -389 µs (-28%) |
| random-walk | `update_time_blocks` upsert, 1 sample in 1 of 3,600 blocks | 1367 µs | 958 µs | -409 µs (-30%) |
| random-walk | `update_time_blocks` upsert, 1 sample in each of 3,600 blocks | 1367 µs | 1879 µs | +512 µs (+37%) |
| noisy-sine | `update`, 1 of 60 blocks | 219 µs | 173 µs | -46 µs (-21%) |
| noisy-sine | `update` with times, 1 of 60 | 242 µs | 191 µs | -51 µs (-21%) |
| noisy-sine | `update` with times, append 1 | 242 µs | 217 µs | -25 µs (-10%) |
| noisy-sine | `update_time_blocks`, 2 s straddling 3 of 60 blocks | 445 µs | 284 µs | -161 µs (-36%) |
| noisy-sine | `update_time_blocks`, 2 s straddling 3 of 3,600 blocks | 1368 µs | 974 µs | -394 µs (-29%) |
| noisy-sine | `update_time_blocks` upsert, 1 sample in 1 of 3,600 blocks | 1368 µs | 951 µs | -417 µs (-30%) |
| noisy-sine | `update_time_blocks` upsert, 1 sample in each of 3,600 blocks | 1368 µs | 1913 µs | +546 µs (+40%) |
| sensor-0.1 | `update`, 1 of 60 blocks | 319 µs | 274 µs | -45 µs (-14%) |
| sensor-0.1 | `update` with times, 1 of 60 | 336 µs | 294 µs | -42 µs (-12%) |
| sensor-0.1 | `update` with times, append 1 | 336 µs | 314 µs | -22 µs (-6%) |
| sensor-0.1 | `update_time_blocks`, 2 s straddling 3 of 60 blocks | 535 µs | 385 µs | -150 µs (-28%) |
| sensor-0.1 | `update_time_blocks`, 2 s straddling 3 of 3,600 blocks | 850 µs | 506 µs | -344 µs (-40%) |
| sensor-0.1 | `update_time_blocks` upsert, 1 sample in 1 of 3,600 blocks | 850 µs | 484 µs | -365 µs (-43%) |
| sensor-0.1 | `update_time_blocks` upsert, 1 sample in each of 3,600 blocks | 850 µs | 1539 µs | +689 µs (+81%) |
