# fluxcode update benchmark (bench/update.py, 2026-10-07, AC power, single thread)

At 0581c29, default params (effort 4: both layouts compressed, one block run). Each update against encoding
the same series from scratch. The append row's update builds a block group of 61 blocks and is compared with encoding
the 60-block one. The main table is the Python path (FLUXCODE_RUST=0); a second table is the same run with the
optional Rust extension, which also packs the bodies of time-axis block groups and of updates (`pack_group`), so
at efforts 5 and up their block flushes don't hold the GIL (not measured here).

Against d039350, the noise floor now runs on block groups with times (these have regular 1 kHz times, so the same
steps as without times): encoding the 60-block random walk with times takes 320 → 333 µs (+4%), and
`update` with times 287 → 288 µs. Everything else is within run-to-run noise (the load average was 2.3 from other
apps). Updates stay 6–43% cheaper than encoding, and the upsert of one sample in each of 3,600 blocks 39–81% dearer.

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
| random-walk | `update`, 1 of 60 blocks | 309 µs | 266 µs | -43 µs (-14%) |
| random-walk | `update` with times, 1 of 60 | 333 µs | 288 µs | -45 µs (-14%) |
| random-walk | `update` with times, append 1 | 333 µs | 315 µs | -19 µs (-6%) |
| random-walk | `update_time_blocks`, 2 s straddling 3 of 60 blocks | 569 µs | 386 µs | -182 µs (-32%) |
| random-walk | `update_time_blocks`, 2 s straddling 3 of 3,600 blocks | 1404 µs | 1011 µs | -392 µs (-28%) |
| random-walk | `update_time_blocks` upsert, 1 sample in 1 of 3,600 blocks | 1404 µs | 985 µs | -418 µs (-30%) |
| random-walk | `update_time_blocks` upsert, 1 sample in each of 3,600 blocks | 1404 µs | 1945 µs | +541 µs (+39%) |
| noisy-sine | `update`, 1 of 60 blocks | 240 µs | 191 µs | -49 µs (-20%) |
| noisy-sine | `update` with times, 1 of 60 | 269 µs | 213 µs | -56 µs (-21%) |
| noisy-sine | `update` with times, append 1 | 269 µs | 242 µs | -27 µs (-10%) |
| noisy-sine | `update_time_blocks`, 2 s straddling 3 of 60 blocks | 505 µs | 311 µs | -194 µs (-38%) |
| noisy-sine | `update_time_blocks`, 2 s straddling 3 of 3,600 blocks | 1392 µs | 1002 µs | -390 µs (-28%) |
| noisy-sine | `update_time_blocks` upsert, 1 sample in 1 of 3,600 blocks | 1392 µs | 976 µs | -416 µs (-30%) |
| noisy-sine | `update_time_blocks` upsert, 1 sample in each of 3,600 blocks | 1392 µs | 1955 µs | +562 µs (+40%) |
| sensor-0.1 | `update`, 1 of 60 blocks | 336 µs | 291 µs | -45 µs (-13%) |
| sensor-0.1 | `update` with times, 1 of 60 | 357 µs | 311 µs | -47 µs (-13%) |
| sensor-0.1 | `update` with times, append 1 | 357 µs | 335 µs | -22 µs (-6%) |
| sensor-0.1 | `update_time_blocks`, 2 s straddling 3 of 60 blocks | 595 µs | 407 µs | -188 µs (-32%) |
| sensor-0.1 | `update_time_blocks`, 2 s straddling 3 of 3,600 blocks | 873 µs | 539 µs | -334 µs (-38%) |
| sensor-0.1 | `update_time_blocks` upsert, 1 sample in 1 of 3,600 blocks | 873 µs | 513 µs | -360 µs (-41%) |
| sensor-0.1 | `update_time_blocks` upsert, 1 sample in each of 3,600 blocks | 873 µs | 1579 µs | +705 µs (+81%) |

## With the Rust extension

arm64 Darwin 25.6.0, Python 3.11.13, numpy 2.4.6; median of 300 calls (100 for the block group of 3,600 blocks)

| signal | operation | encode from scratch | update | vs encode |
|---|---|---|---|---|
| random-walk | `update`, 1 of 60 blocks | 288 µs | 248 µs | -41 µs (-14%) |
| random-walk | `update` with times, 1 of 60 | 315 µs | 267 µs | -47 µs (-15%) |
| random-walk | `update` with times, append 1 | 315 µs | 292 µs | -22 µs (-7%) |
| random-walk | `update_time_blocks`, 2 s straddling 3 of 60 blocks | 551 µs | 363 µs | -188 µs (-34%) |
| random-walk | `update_time_blocks`, 2 s straddling 3 of 3,600 blocks | 1393 µs | 986 µs | -407 µs (-29%) |
| random-walk | `update_time_blocks` upsert, 1 sample in 1 of 3,600 blocks | 1393 µs | 962 µs | -431 µs (-31%) |
| random-walk | `update_time_blocks` upsert, 1 sample in each of 3,600 blocks | 1393 µs | 1924 µs | +531 µs (+38%) |
| noisy-sine | `update`, 1 of 60 blocks | 221 µs | 173 µs | -48 µs (-22%) |
| noisy-sine | `update` with times, 1 of 60 | 252 µs | 192 µs | -60 µs (-24%) |
| noisy-sine | `update` with times, append 1 | 252 µs | 219 µs | -33 µs (-13%) |
| noisy-sine | `update_time_blocks`, 2 s straddling 3 of 60 blocks | 488 µs | 289 µs | -199 µs (-41%) |
| noisy-sine | `update_time_blocks`, 2 s straddling 3 of 3,600 blocks | 1380 µs | 976 µs | -405 µs (-29%) |
| noisy-sine | `update_time_blocks` upsert, 1 sample in 1 of 3,600 blocks | 1380 µs | 951 µs | -430 µs (-31%) |
| noisy-sine | `update_time_blocks` upsert, 1 sample in each of 3,600 blocks | 1380 µs | 1945 µs | +565 µs (+41%) |
| sensor-0.1 | `update`, 1 of 60 blocks | 319 µs | 274 µs | -45 µs (-14%) |
| sensor-0.1 | `update` with times, 1 of 60 | 348 µs | 296 µs | -52 µs (-15%) |
| sensor-0.1 | `update` with times, append 1 | 348 µs | 317 µs | -31 µs (-9%) |
| sensor-0.1 | `update_time_blocks`, 2 s straddling 3 of 60 blocks | 581 µs | 388 µs | -193 µs (-33%) |
| sensor-0.1 | `update_time_blocks`, 2 s straddling 3 of 3,600 blocks | 863 µs | 515 µs | -348 µs (-40%) |
| sensor-0.1 | `update_time_blocks` upsert, 1 sample in 1 of 3,600 blocks | 863 µs | 491 µs | -373 µs (-43%) |
| sensor-0.1 | `update_time_blocks` upsert, 1 sample in each of 3,600 blocks | 863 µs | 1559 µs | +696 µs (+81%) |
