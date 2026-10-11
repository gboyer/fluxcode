# fluxprune

A prepass for storage and processing: keep the points of a time series that carry its shape, read
as straight lines between them. Unlike fluxcode, which keeps every sample within a bound, fluxprune
returns a subset of the points, far fewer and much lossier, chosen for the time-weighted RMSE of
the linear interpolation.

- The first and last points (to stitch storage blocks back together), the minimum and the maximum
  are always kept, as are non-finite values with their neighbours.
- One O(n log n) pass (`rank`) orders every point by significance: bottom-up, it removes the point
  whose removal adds the least error, with each segment's error integrated exactly, and records the
  error of every count. Two O(n) refinement passes then move each kept point to where it fits best
  between its neighbours (about 6% above the optimal choice, against 25-35% without them).
- How many points to keep is read off that error curve: a count, a tolerance (RMSE relative to the
  block's range), their geometric mean, with or without a floor at the block's own noise level, or
  `error_per_point`, which trades error against points at a fixed rate.

```python
import fluxprune

idx = fluxprune.prune(t, x, error_per_point=0.0015)            # adaptive, one knob
idx = fluxprune.prune(t, x, points=200)                        # a fixed count
idx = fluxprune.prune(t, x, rmse=0.01)                         # within 1% of the range
idx = fluxprune.prune(t, x, points=200, rmse=0.01, noise_floor=True)  # a balance, capped at sqrt(200 n)

r = fluxprune.rank(t, x)                                       # or step by step
r.rmse()        # the error of every count
r.noise()       # the white-noise level, from the curve
keep = r.refine(r.keep(r.count(points=200)))
```

The experiments behind the choices are in
[experimental/make_prune_report.py](../experimental/make_prune_report.py) (`report/prune.html`):
ranking quality against the optimum, swinging door, LTTB and others; eight stopping rules on 12 slow
signals in 1 h blocks; sampling-rate and block-length stability; swinging-door archives as input; speed.
