# fluxprune

A prepass for storage and processing: keep the points of a time series that carry its shape, read
as straight lines between them. Unlike fluxcode, which keeps every sample within a bound, fluxprune
returns a subset of the points, far fewer and much lossier.

```python
import fluxprune

p = fluxprune.prune(t, x)                                   # automatic
p.index, p.max_error, p.rmse                                # kept indices and the errors they leave
p = fluxprune.prune(t, x, max_points=360, max_error=0.5)    # with guards; p.capped if the cap won
p = fluxprune.prune(t, x, error_per_point=0.0015 / 12)      # the same density for 12 h blocks

pk = fluxprune.peaks(t, x, period=np.timedelta64(60, "s"), prominence=50.0)
p = fluxprune.prune(t, x, keep=pk)                          # every peak and trough, the rest pruned
```

- **Automatic** (`error_per_point`, default 0.0015): one O(n log n) pass (`rank`) orders every point
  by significance, bottom-up: it removes the point whose removal adds the least time-weighted
  error, integrated exactly per segment, and records the error of every count. The count is the
  first where the RMSE left, relative to the block's range, falls below `error_per_point` times the
  points added: error and points traded at a fixed rate, more points for complex blocks, fewer for
  simple ones, sublinearly, and nearly independent of the sampling rate. Two O(n) refinement passes
  then move each point to where it fits best between its neighbours (about 6% above the optimal
  choice). The first and last points (to stitch blocks back together), the minimum and the maximum
  are always kept, as are non-finite values with their neighbours. For blocks H times as long,
  `error_per_point / H` keeps the same density.
- **`max_error`**: no input point strays further than this, in value units, from the kept lines.
  It only adds points: each segment of the automatic choice that strays further is filled with
  swinging door's chords. At the same guarantee it keeps fewer points than swinging door on most
  signals, and it also covers what a range-relative error can't (a quiet block's own dither, a
  glitch that sets a block's range, the block length).
- **`max_points`**: a hard cap. Over it, the automatic points give way first; if `max_error` alone
  needs more, the cap is spent where the error is largest, and `capped` says so.
- **`keep`**: indices kept whatever else is, such as `peaks()`.
- **`peaks(t, x, period, prominence, max_points)`**: for periodic signals, where each peak and trough
  matters as a sample of its own. A peak is a local maximum at least `prominence` above the higher
  of the lowest points separating it from a higher one (set it to about 6σ of the noise or more),
  and of two peaks less than half a `period` apart only the higher is kept; troughs mirrored; a
  trough between every two peaks and a peak between every two troughs. O(n log n).

`rank(t, x)` gives the ranking step by step: `r.rmse()` (the error of every count), `r.keep(k)`
(nested), `r.refine(...)`, `r.prune(...)`.

The experiments behind the choices are in
[experimental/make_prune_report.py](../experimental/make_prune_report.py) (`report/prune.html`):
ranking quality against the optimum, swinging door, LTTB and others; nine stopping rules on 12
slow signals in 1 h blocks, with their stability under sampling rate and block length; what the
error should be relative to (six more signals); the guards against swinging door; peaks on two
periodic signals;
swinging-door archives as input; speed. [bench/prune.py](../bench/prune.py) times every call
([bench/PRUNE_RESULTS.md](../bench/PRUNE_RESULTS.md)): about 0.7 ms for an hour of 1 s scans and
25 ms for a day, on an M3.
