# fluxcode update benchmark (bench/update.py, 2026-10-02, AC power, single thread)

At 5ee16fa, default params (effort 4: both layouts compressed, one block run: the encoder as it was at
8bfc6c6, so every row is within about 2% of the 2026-10-01 run's). Each update against encoding the same
series from scratch. The append row's update builds a 61-block unit and is compared with encoding the
60-block one. `update` and `update_time_blocks` use the python-zstandard path, including with the optional
Rust extension (it covers `encode` of units without a time axis only), so at efforts 5 and up their block
flushes hold the GIL; not measured here.

Against the previous run (8b7bb76, before the int16 `grid_params` column) every row is within
noise except the 3,600-block units of random-walk and noisy-sine: encode 1419 → 1612 µs and
update 1011 → 1198 µs (A/B against main in one session, +14% and +18%). Their bit-plane body
shrank from 270,000 to 248,400 bytes, below 256 KB, where libzstd picks its level-3 parameters
for smaller inputs: compressing that body takes 348 µs instead of 148. Forcing the large-input
parameters for every unit was measured and doesn't pay: standard 60-block units (about 120 KB)
come out 0.03% smaller with 4.6% more zstd time.

Block flushes (effort 5 and up) don't avoid that switch: zstd sizes its parameters from the input size
the frame declares, and a flushed frame declares the same size (measured 2026-10-02, 10-sample blocks,
random-walk bit planes: body 243 KB compresses in 335 µs in one run and 415 µs flushed; 256 KB in 130 µs and
244 µs). Byte planes don't show the jump.

arm64 Darwin 25.6.0, Python 3.11.13, numpy 2.4.6; median of 300 calls (100 for the 3,600-block unit)

| signal | operation | encode from scratch | update | vs encode |
|---|---|---|---|---|
| random-walk | `update`, 1 of 60 blocks | 314 µs | 265 µs | -50 µs (-16%) |
| random-walk | `update` with times, 1 of 60 | 332 µs | 286 µs | -47 µs (-14%) |
| random-walk | `update` with times, append 1 | 332 µs | 312 µs | -20 µs (-6%) |
| random-walk | `update_time_blocks`, 2 s straddling 3 of 60 blocks | 566 µs | 423 µs | -143 µs (-25%) |
| random-walk | `update_time_blocks`, 2 s straddling 3 of 3,600 blocks | 1599 µs | 1201 µs | -398 µs (-25%) |
| noisy-sine | `update`, 1 of 60 blocks | 243 µs | 209 µs | -33 µs (-14%) |
| noisy-sine | `update` with times, 1 of 60 | 266 µs | 232 µs | -33 µs (-13%) |
| noisy-sine | `update` with times, append 1 | 266 µs | 263 µs | -3 µs (-1%) |
| noisy-sine | `update_time_blocks`, 2 s straddling 3 of 60 blocks | 504 µs | 373 µs | -131 µs (-26%) |
| noisy-sine | `update_time_blocks`, 2 s straddling 3 of 3,600 blocks | 1607 µs | 1211 µs | -396 µs (-25%) |
| sensor-0.1 | `update`, 1 of 60 blocks | 345 µs | 294 µs | -51 µs (-15%) |
| sensor-0.1 | `update` with times, 1 of 60 | 361 µs | 315 µs | -46 µs (-13%) |
| sensor-0.1 | `update` with times, append 1 | 361 µs | 340 µs | -20 µs (-6%) |
| sensor-0.1 | `update_time_blocks`, 2 s straddling 3 of 60 blocks | 598 µs | 452 µs | -146 µs (-24%) |
| sensor-0.1 | `update_time_blocks`, 2 s straddling 3 of 3,600 blocks | 1085 µs | 715 µs | -371 µs (-34%) |
