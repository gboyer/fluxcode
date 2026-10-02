# fluxcode update benchmark (bench/update.py, 2026-10-01, AC power, single thread)

At 8b7bb76, default params (`planes="best"`: encode and update both compress twice). Each
update against encoding the same series from scratch. The append row's update builds a
61-block unit and is compared with encoding the 60-block one.

arm64 Darwin 25.6.0, Python 3.11.13, numpy 2.4.6; median of 300 calls (100 for the 3,600-block unit)

| signal | operation | encode from scratch | update | vs encode |
|---|---|---|---|---|
| random-walk | `update`, 1 of 60 blocks | 313 µs | 263 µs | -51 µs (-16%) |
| random-walk | `update` with times, 1 of 60 | 331 µs | 287 µs | -45 µs (-14%) |
| random-walk | `update` with times, append 1 | 331 µs | 313 µs | -18 µs (-5%) |
| random-walk | `update_time_blocks`, 2 s straddling 3 of 60 blocks | 570 µs | 425 µs | -145 µs (-25%) |
| random-walk | `update_time_blocks`, 2 s straddling 3 of 3,600 blocks | 1424 µs | 1017 µs | -407 µs (-29%) |
| noisy-sine | `update`, 1 of 60 blocks | 244 µs | 211 µs | -33 µs (-14%) |
| noisy-sine | `update` with times, 1 of 60 | 267 µs | 231 µs | -36 µs (-13%) |
| noisy-sine | `update` with times, append 1 | 267 µs | 260 µs | -6 µs (-2%) |
| noisy-sine | `update_time_blocks`, 2 s straddling 3 of 60 blocks | 503 µs | 370 µs | -133 µs (-26%) |
| noisy-sine | `update_time_blocks`, 2 s straddling 3 of 3,600 blocks | 1401 µs | 995 µs | -407 µs (-29%) |
| sensor-0.1 | `update`, 1 of 60 blocks | 344 µs | 290 µs | -55 µs (-16%) |
| sensor-0.1 | `update` with times, 1 of 60 | 366 µs | 313 µs | -52 µs (-14%) |
| sensor-0.1 | `update` with times, append 1 | 366 µs | 335 µs | -30 µs (-8%) |
| sensor-0.1 | `update_time_blocks`, 2 s straddling 3 of 60 blocks | 596 µs | 449 µs | -147 µs (-25%) |
| sensor-0.1 | `update_time_blocks`, 2 s straddling 3 of 3,600 blocks | 1110 µs | 717 µs | -393 µs (-35%) |
