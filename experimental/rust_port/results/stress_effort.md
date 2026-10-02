# Day-scale stress test at efforts 4 and 5, python-zstandard and the Rust extension

`bench/stress.py --scale 0.25 --effort N` (250 tags x a day, 4 threads), AC power, one run each; the two paths back to back:

| effort | path | encode wall | float64 GB/s | µs/block/worker | compressed |
|---|---|---:|---:|---:|---:|
| 4 | python | 25.7 s | 6.71 GB/s | 4.77 | 9.91 GB |
| 4 | rust | 24.3 s | 7.10 GB/s | 4.51 | 9.91 GB |
| 5 | python | 47.8 s | 3.62 GB/s | 8.85 | 9.68 GB |
| 5 | rust | 28.9 s | 5.98 GB/s | 5.35 | 9.68 GB |
