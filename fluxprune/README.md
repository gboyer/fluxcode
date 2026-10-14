# fluxprune

A prepass for storage and processing: keep the points of a time series that carry its shape, read
as straight lines between them. Unlike fluxcode, which keeps every sample within a bound, fluxprune
keeps a subset of the points, far fewer and much lossier.

```python
import fluxprune

p = fluxprune.prune(t, x)                                   # the rmse pass alone
p.times, p.values                                           # the kept points
p.rmse, p.max_error                                         # the errors they leave, in value units
p.noise                                                     # the white-noise σ the rmse pass reads above
p = fluxprune.prune(t, x, max_points=360, max_error=0.5)    # with guards
p.rmse_met, p.max_error_met, p.max_points_met               # which asks were met
p = fluxprune.prune(t, x, error_per_point=0.0015 / 12)      # 12 h blocks (fewer per hour)
p = fluxprune.prune(t, x, error_per_point=0, max_points=36) # the best 36 points
p = fluxprune.prune(t, x, error_per_point=np.inf, max_error=0.5)  # the max_error pass alone
```

Two passes choose the points:

- **The rmse pass** (`error_per_point`, default 0.0015): one O(n log n) pass orders every point by
  significance, bottom-up: it removes the point whose removal adds the least time-weighted error,
  integrated exactly per segment, and records the error of every count. The count is the first where
  the RMSE left, relative to the block's range, falls below `error_per_point` times the points
  added: error and points traded at a fixed rate, more points for complex blocks, fewer for simple
  ones, sublinearly, and nearly independent of the sampling rate. Two O(n) refinement passes then
  move each point to where it fits best between its neighbours (4-6% above the optimal choice on
  1 h blocks). The rate is per block: for blocks H times as long, start from `error_per_point / H`.
  That keeps fewer points per hour, not the same density: about a third as many in 12 h blocks as
  in 1 h blocks on the report's corpus (geometric mean over 18 signals), since a longer block has a
  wider range for the error to be relative to, and needs one set of required points instead of H.
  `error_per_point=0` makes every point worth keeping (with `max_points`: the best `max_points`);
  `np.inf` keeps only the required points (with `max_error`: the max_error pass alone).
  By default the RMSE is read above the input's own white noise (`noise_floor`, below), so the pass
  doesn't buy points that only trace the noise.
- **The max_error pass** (`max_error`): no input point strays further than this, in value units,
  from the kept lines. It only adds points (see below). At the same guarantee it keeps far fewer
  points than swinging door, and it also covers what a range-relative error can't (a quiet block's
  own dither, a glitch that sets a block's range, the block length). A `max_error` below the noise
  keeps most of the points, as swinging door does (at σ about half, at 0.6σ two thirds); `prune`
  warns then, with σ the noise floor's estimate. A `max_error` of a few σ is also what keeps narrow
  events on the noise (spikes of a few seconds in an hour), which a time-weighted RMSE counts as
  cheap.

And a cap:

- **`max_points`**: a hard cap, at least 4. `max_error` takes precedence over the rmse pass: over
  the cap, the rmse pass's points give way (least significant first) while `max_error` still holds
  (`rmse_met` is then false); if `max_error` can't be met within the cap even without them, it is
  relaxed to the smallest bound that fits (`max_error_met` false, and `max_error` of the result is
  the bound reached).

**Required points.** The first and last points (to stitch blocks back together), the minimum and
the maximum are always kept. A run of non-finite values (a gap) keeps its first point and the
finite points just before and after it; a kept non-finite value holds until the next kept point,
so the inside of a gap needs none. These gap points are kept even past `max_points`: then
`max_points_met` is false and `prune` warns.

**The noise floor** (`noise_floor=True`, the default). Lines through noisy samples leave about the
noise's σ against them however many points are kept, until nearly all are, so the rmse pass on a
quiet, noisy block keeps buying points that trace the noise: the range is a few σ, and the noise
looks like signal relative to it. With the floor it stops on √(e² - σ²) instead of e, an estimate of
the error against the signal under the noise. σ comes from one O(n) pass: each interior point's
distance from the chord of its two neighbours (normal with σ·√(1 + w² + (1 - w)²) for white noise, w
its position between them in time), median, times 1.4826 (`fluxprune/_noise.py`; `p.noise` reports
it). Irregular times are fine and the median ignores spikes and steps; quantized signals read 0
(more than half the chords are exact) and don't change. Nothing is smoothed: the kept points are
still samples, and the ranking is the same; only where it stops moves.

On the report's corpus (1 h blocks, 1 s scans), the noise-dominated signals keep a fraction of the
points (white noise 95 → 11 per hour, motor current 67 → 12, the quiet tank level 77 → 19, vibration
47 → 18) and smooth or quantized ones the same (temperature, tank level, valve, totalizer, sine), for
0-10% more RMSE against the scans. On pure noise (120 hours), the kept lines are closer to the signal
under it on average (0.26σ against 0.33σ RMSE; the worst hour 0.48σ against 0.38σ, where the lines
from the required minimum and maximum lean on the few points left). Calibrated to the same total
(36 per hour), it reaches the least mean RMSE possible for it, with half the median error of the rule
without the floor. Its cost is the sampling rate: correlated noise (a time constant of a few scans)
reads partly as signal at 1 s and as noise at 10 s, so such a signal keeps fewer points per hour at
10 s (flow: 0.14x; without the floor 0.82x). With the floor, a lower `error_per_point` no longer buys noise: noise-dominated
signals level off at what their content needs (white noise 17 per hour even at 1e-8), and only
signals with structure at finer scales keep more. Narrow spikes on the noise (a pyrometer's hot
items, about 10 s wide) are cheap in a time-weighted RMSE with or without the floor: a lower
`error_per_point` (1.5e-5: every peak within 7 °C, quiet hours still about 20 points) or a
`max_error` of a few σ keeps them. Scoring against a smoothed copy instead (Savitzky-Golay, a running
median) was measured and not adopted: it keeps as many points on noise (a line through noisy samples
is no closer to a smooth curve), needs a window, and shaves narrow peaks.

**`refit=True`** (experimental): the same times, with the values whose lines fit the input best in
time-weighted L2 instead of the samples (a tridiagonal solve, O(n)); the required points keep their
samples. On the report's corpus it lowers the RMSE by about a tenth, almost all of it on smooth or
step-like signals (0.5-0.7x), 1-4% on noisy ones. The values are no longer readings, and
`max_error` isn't checked again: `max_error_met` says whether it still holds.

## How the guards work

`prune` with `max_error` and `max_points` spends its budget in a fixed order:

1. **The required points.** They alone exceed `max_points` only through gaps (above).
2. **`max_error`, if it can be met within the cap.** The max_error pass, run from the required
   points alone, says whether it can.
3. **The rmse pass takes what is left.** The k most significant points of the ranking (k from
   `error_per_point`), refined, then the max_error pass. If that exceeds the cap, k is lowered by
   bisection to the largest that fits; since the ranking is nested, the least significant points
   give way first, and the max_error pass is rerun each time.
4. **If `max_error` can't be met within the cap** even without the rmse pass, it is relaxed: a
   geometric bisection between `max_error` (doesn't fit) and the error of the required points alone
   (fits without adding any) finds about the smallest bound whose max_error pass fits, to 0.1%.

The max_error pass is always the last step: adding a point can make the error worse (a vertex on a
noise spike tilts both lines next to it), so the guarantee must be established on the final set.

**The max_error pass (farthest reach).** Each line between kept points that strays more than
`max_error` from the input is split from its left end: from the anchor, scan on while some line
from it could still pass within `max_error` of every point scanned so far (each point bounds the
slope from below and above; the bounds must overlap), note the last point whose own chord satisfies
them, and when the bounds cross, make that point the next anchor. Swinging door uses the same slope
bounds but ends the line at the first point the chord can't reach; with noise, a later point often
can be reached again. On the report's 18 slow signals (1 h blocks) this keeps 1-5x fewer points
than the swinging-door rule (vibration 36 against 176 per hour, batch 23 against 73, temperature
509 against 766) at the same cost, and 4% more than the fewest points that meet `max_error` (up to
20% more on the noisiest signals; exact by dynamic programming over all feasible lines, about 35x
slower).

**Cost.** Each step is linear: picking the k most significant points is a mask over the removal
order (one pass); a refine pass scans each point between its kept neighbours twice; the max_error
pass scans each line once for its worst point and once more where it strays (points between a jump
target and where the bounds crossed are scanned again, so a pathological signal can cost more).
Without a binding cap, `max_error` adds about 2 passes; giving way adds a bisection of about
log2(k) attempts of 6-7 passes; relaxing adds about 12-15 max_error passes of 2 passes each.
`bench/PRUNE_RESULTS.md` has the measured cost: +13-53% over the ranking (+16-32% for the rmse pass alone at 3,600
points, +7-21% at 86,400 and more).

Measured and not adopted (report/prune.html): the max_error pass first and ranking again with its
points fixed (same RMSE at the same count, 3-5x slower); a secant search for the relaxed bound
(fewer max_error passes on average, twice as many at worst); trying every k instead of bisecting
(0.4% lower RMSE on average); splitting the worst segment at its worst point when the cap binds
(1.3-1.7x larger errors).

## Experiments

`fluxprune._api.rank(t, x)` gives the steps (not part of the API): `r.relative_rmse()` (the error of
every count, relative to the block's range, before the noise floor), `r.noise`, `r.top(k)` (nested),
`r.refine(...)`, `r.select(...)`, `r.refit(...)`, `r.errors(...)`.

The experiments behind the choices are in
[experimental/make_prune_report.py](../experimental/make_prune_report.py) (`report/prune.html`):
ranking quality against the optimum, top-down splitting, Visvalingam-Whyatt, swinging door, LTTB
and others; nine stopping rules on 12 slow signals in 1 h blocks, with their stability under
sampling rate and block length; what the error should be relative to (six more signals); the noise
floor (with a pyrometer: narrow spikes and quiet hours); the guards
against swinging door and the max_error pass against its alternatives; refit; swinging-door archives
as input; speed. [bench/prune.py](../bench/prune.py) times every call
([bench/PRUNE_RESULTS.md](../bench/PRUNE_RESULTS.md)): about 0.45 ms for an hour of 1 s scans and
17 ms for a day, on an M3 (one core).
