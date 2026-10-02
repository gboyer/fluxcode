Now drawing from 'AC Power'

## sensor mix (100 units, weighted as bench/stress.py)

| settings | size | 1 thr python / rust | 4 thr python / rust | 8 thr python / rust |
|---|---:|---:|---:|---:|
| heuristic, one run (effort 2) | +1.44% | 2390 / 2389 MiB/s | 6865 / 7662 MiB/s | 6034 / 7681 MiB/s |
| best, one run (effort 4, default) | +0.00% | 1888 / 1875 MiB/s | 5917 / 6514 MiB/s | 5506 / 6725 MiB/s |
| heuristic, flushed | -0.44% | 2022 / 2046 MiB/s | 3809 / 6922 MiB/s | 2704 / 7261 MiB/s |
| best, flushed (effort 5) | -2.28% | 1536 / 1589 MiB/s | 2939 / 5497 MiB/s | 2003 / 6265 MiB/s |
| best, flushed, zstd 3 + 9 (effort 9) | -2.74% | 455 / 475 MiB/s | 1404 / 1726 MiB/s | 1265 / 2454 MiB/s |

## report signals (15 kinds x 3)

| settings | size | 1 thr python / rust | 4 thr python / rust | 8 thr python / rust |
|---|---:|---:|---:|---:|
| heuristic, one run (effort 2) | +0.00% | 2010 / 1977 MiB/s | 6414 / 6690 MiB/s | 5853 / 6818 MiB/s |
| best, one run (effort 4, default) | +0.00% | 1457 / 1465 MiB/s | 4842 / 5041 MiB/s | 4707 / 5918 MiB/s |
| heuristic, flushed | -2.22% | 1716 / 1756 MiB/s | 4061 / 6106 MiB/s | 3123 / 6348 MiB/s |
| best, flushed (effort 5) | -2.23% | 1200 / 1235 MiB/s | 2821 / 4373 MiB/s | 2095 / 5248 MiB/s |
| best, flushed, zstd 3 + 9 (effort 9) | -3.43% | 316 / 328 MiB/s | 1077 / 1222 MiB/s | 1076 / 1683 MiB/s |

power at the end: Now drawing from 'AC Power'
