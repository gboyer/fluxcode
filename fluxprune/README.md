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
- Errors are relative to each block's range (max - min) unless `scale` gives a span in the values'
  units. The range needs no knowledge of the series and gives small variations on a large offset
  (a furnace at 2,115 °F, grid frequency) the precision they need; a known span (a historian's
  per-tag span) makes errors absolute, so quiet blocks, glitches and the block length matter less.
  The largest |value| is not offered: it depends on where zero is (°C or K).

```python
import fluxprune

idx = fluxprune.prune(t, x, error_per_point=0.0015)            # adaptive, one knob (1 h blocks)
idx = fluxprune.prune(t, x, error_per_point=0.0015 / 12)       # the same density for 12 h blocks
idx = fluxprune.prune(t, x, error_per_point=0.0015, scale=30)  # errors relative to a known 30-unit span
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
signals in 1 h blocks; sampling-rate and block-length stability; what errors should be relative to, on
signals with small variations on a large offset; swinging-door archives as input; speed (0.6 ms for an
hour of 1 s scans on an M3).
