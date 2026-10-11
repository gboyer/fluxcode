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
  It only adds points (see below). At the same guarantee it keeps far fewer points than swinging
  door, and it also covers what a range-relative error can't (a quiet block's own dither, a glitch
  that sets a block's range, the block length).
- **`max_points`**: a hard cap. `max_error` takes precedence over the automatic points: over the
  cap, they give way (least significant first) while `max_error` still holds; if `max_error` can't
  be met within the cap even without them, it is relaxed to the smallest bound that fits, and
  `capped` says so (`max_error` of the result is the bound reached).
- **`keep`**: indices kept whatever else is, such as `peaks()`.
- **`peaks(t, x, period, prominence, max_points)`**: for periodic signals, where each peak and trough
  matters as a sample of its own. A peak is a local maximum at least `prominence` above the higher
  of the lowest points separating it from a higher one (set it to about 6σ of the noise or more),
  and of two peaks less than half a `period` apart only the higher is kept; troughs mirrored; a
  trough between every two peaks and a peak between every two troughs. O(n log n).

## How the guards work

`prune` with `max_error` and `max_points` spends its budget in a fixed order:

1. **The required points**: first and last, minimum and maximum, `keep`, non-finite values with
   their neighbours. If they alone exceed `max_points`, they are kept in that order up to the cap.
2. **`max_error`, if it can be met within the cap.** The fill below, run from the required points
   alone, says whether it can.
3. **The automatic points take what is left.** The k most significant points of the ranking
   (`keep(k)`, k from `error_per_point`), refined, then filled at `max_error`. If that exceeds the
   cap, k is lowered by bisection to the largest that fits; since the ranking is nested, the least
   significant automatic points give way first, and the fill is recomputed each time.
4. **If `max_error` can't be met within the cap** even without automatic points, it is relaxed:
   a geometric bisection between `max_error` (doesn't fit) and the error of the required points
   alone (fits without adding any) finds about the smallest bound whose fill fits, to 0.1%. The
   result's `max_error` is the bound reached and `capped` is set.

The fill is always the last step: adding a point can make the error worse (a vertex on a noise
spike tilts both lines next to it), so the guarantee must be established on the final set.

**The fill (farthest reach).** Each kept segment that strays more than `max_error` from the input
is filled from its left end: from the anchor, scan on while some line from it could still pass
within `max_error` of every point scanned so far (each point bounds the slope from below and
above; the bounds must overlap), note the last point whose own chord satisfies them, and when the
bounds cross, make that point the next anchor. Swinging door uses the same slope bounds but ends
the line at the first point the chord can't reach; with noise, a later point often can be reached
again. On the report's 18 slow signals (1 h blocks) this keeps 1-3x fewer points than the
swinging-door rule (vibration 66 against 170 per hour, batch 23 against 72, temperature 510
against 766) at the same cost, and 4% more than the fewest points that meet `max_error` (up to
20% more on the noisiest signals; exact by dynamic programming over all feasible lines, about 35x
slower).

**Cost.** Each step is linear: `keep(k)` is a mask over the removal order (one pass); a refine pass
scans each point between its kept neighbours twice; the fill scans each segment once for its worst
point and once more where it strays (points between a jump target and where the bounds crossed are
scanned again, so a pathological signal can cost more). Without a binding cap, `max_error` adds
about 2 passes; giving way adds a bisection of about log2(k) attempts of 6-7 passes; relaxing adds
about 12-15 fills of 2 passes each. `bench/PRUNE_RESULTS.md` has the measured cost: +5-35% over `rank`.

Measured and not adopted (report/prune.html): filling first and ranking again with the fill fixed
(same RMSE at the same count, 3-5x slower); a secant search for the relaxed bound (fewer fills on
average, twice as many at worst); trying every k instead of bisecting (0.4% lower RMSE on average);
splitting the worst segment at its worst point when the cap binds (1.3-1.7x larger errors).

## Step by step

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
