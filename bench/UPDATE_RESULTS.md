# fluxcode update benchmark (bench/update.py, 2026-10-01, AC power, single thread)

At 8bfc6c6, default params (`planes="best"`: encode and update both compress twice). Each
update against encoding the same series from scratch. The append row's update builds a
61-block unit and is compared with encoding the 60-block one.

Against the previous run (8b7bb76, before the int16 `grid_params` column) every row is within
noise except the 3,600-block units of random-walk and noisy-sine: encode 1419 → 1612 µs and
update 1011 → 1198 µs (A/B against main in one session, +14% and +18%). Their bit-plane body
shrank from 270,000 to 248,400 bytes, below 256 KB, where libzstd picks its level-3 parameters
for smaller inputs: compressing that body takes 348 µs instead of 148. Forcing the large-input
parameters for every unit was measured and doesn't pay: standard 60-block units (about 120 KB)
come out 0.03% smaller with 4.6% more zstd time.

arm64 Darwin 25.6.0, Python 3.11.13, numpy 2.4.6; median of 300 calls (100 for the 3,600-block unit)

| signal | operation | encode from scratch | update | vs encode |
|---|---|---|---|---|
| random-walk | `update`, 1 of 60 blocks | 313 µs | 263 µs | -49 µs (-16%) |
| random-walk | `update` with times, 1 of 60 | 331 µs | 285 µs | -45 µs (-14%) |
| random-walk | `update` with times, append 1 | 331 µs | 314 µs | -16 µs (-5%) |
| random-walk | `update_time_blocks`, 2 s straddling 3 of 60 blocks | 570 µs | 427 µs | -143 µs (-25%) |
| random-walk | `update_time_blocks`, 2 s straddling 3 of 3,600 blocks | 1621 µs | 1223 µs | -399 µs (-25%) |
| noisy-sine | `update`, 1 of 60 blocks | 244 µs | 211 µs | -33 µs (-13%) |
| noisy-sine | `update` with times, 1 of 60 | 266 µs | 231 µs | -35 µs (-13%) |
| noisy-sine | `update` with times, append 1 | 266 µs | 260 µs | -6 µs (-2%) |
| noisy-sine | `update_time_blocks`, 2 s straddling 3 of 60 blocks | 503 µs | 372 µs | -131 µs (-26%) |
| noisy-sine | `update_time_blocks`, 2 s straddling 3 of 3,600 blocks | 1612 µs | 1212 µs | -400 µs (-25%) |
| sensor-0.1 | `update`, 1 of 60 blocks | 345 µs | 293 µs | -52 µs (-15%) |
| sensor-0.1 | `update` with times, 1 of 60 | 361 µs | 313 µs | -47 µs (-13%) |
| sensor-0.1 | `update` with times, append 1 | 361 µs | 338 µs | -23 µs (-6%) |
| sensor-0.1 | `update_time_blocks`, 2 s straddling 3 of 60 blocks | 598 µs | 450 µs | -148 µs (-25%) |
| sensor-0.1 | `update_time_blocks`, 2 s straddling 3 of 3,600 blocks | 1082 µs | 711 µs | -370 µs (-34%) |
