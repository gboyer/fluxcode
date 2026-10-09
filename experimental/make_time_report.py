# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Timestamp compression in fluxcode's time axis (docs/SPEC.md §3) -> report/time.html.

    uv run python make_time_report.py [--reps 30]

Only the time axis: what exact timestamps add to one-minute block groups (60 blocks of 1000 samples,
nominally 1 kHz), for the common clock shapes (a perfect grid, a grid with a few gaps, a noisy
clock) and a few harder ones. The values are fixed (a 4.12 Hz sine); every number is the block group
with times minus the same block group without. Timings are single thread, best of --reps; run on AC
power with nothing else busy. The last section rounds the timestamps first (Params.time_error).
"""

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "tests"), str(REPO / "bench")]

import _fresh_numba_cache  # noqa: E402, F401  (before fluxcode: a cache keyed on its sources)
import fluxcode  # noqa: E402
from _signals import CLOCKS, MINUTE, clock_minute, minute  # noqa: E402
from fluxcode import _format, _group  # noqa: E402
from time_axis import patterns  # noqa: E402

from tslab.report.page import clean_outputs, save_fig, shell  # noqa: E402

OUT = Path(__file__).parent / "report"
BLOCK, BLOCKS = 1000, 60
SERIES = "#2a78d6"  # one series per panel: the reference palette's first slot
IRREGULAR = "#f3c9b5"  # tint behind irregular blocks
PALETTE = ["#2a78d6", "#eb6834", "#1a9e77", "#8f5bd6", "#c43c7a"]  # one per series on the sweep
SEEDS = 64  # block groups per clock shape for the distributions
MS = 1_000_000
TIME_ERRORS = (0.0, 0.02, 0.05, 0.1)  # the table's columns
SWEEP_ERRORS = tuple(np.round(np.arange(41) * 0.0025, 4))  # the chart's points: every 0.0025 from 0 to 0.1
SWEPT = ("Noisy clock", "Jitter σ = 10 µs (1%), ns, on round times", "Drift and jitter σ = 10 µs, 37 µs off",
         "Poisson events, mean 1 ms, ns", "Deadband logging, ms grid")  # the chart's lines
SHAPE_LABELS = {
    "grid": "Perfect grid",
    "grid+gaps": "Grid with a few gaps",
    "noisy": "Noisy clock",
}
SHAPE_NOTES = {
    "grid": "Every interval exactly 1 ms: hardware-clocked acquisition, resampled or synthesized series.",
    "grid+gaps": "A perfect grid that sometimes skips: dropped packets, reconnects, a paused logger "
                 "(Poisson, mean 2 gaps per minute, each 5 ms to 3 s).",
    "noisy": "Host timestamps on arrival: 1 ms nominal, σ = 20 µs jitter, µs resolution.",
}


def best_us(func, reps):
    fastest = np.inf
    for _ in range(reps):
        started = time.perf_counter()
        func()
        fastest = min(fastest, time.perf_counter() - started)
    return 1e6 * fastest


def measure(values, ticks, plain_group, reps, plain_times):
    """Bytes, irregular blocks and µs the timestamps add to the values' block group."""
    times = ticks.view("datetime64[ns]")
    group = fluxcode.encode_group(values, times=times).group
    decoded = fluxcode.decode_group(group)
    assert decoded.times is not None and np.array_equal(decoded.times, times)
    irregular = (_group.decompress(group).block_flags & _format.BLOCK_FLAG_IRREGULAR_TIME) != 0
    row = {"bytes": len(group) - len(plain_group), "irregular": irregular}
    if reps:
        row["encode_us"] = best_us(lambda: fluxcode.encode_group(values, times=times), reps) - plain_times[0]
        row["decode_us"] = best_us(lambda: fluxcode.decode_group(group), reps) - plain_times[1]
    return row


def added(added_us, base_us):
    """An added time with its share of the values-only time, e.g. "+30 µs (+17%)"."""
    return f"+{added_us:.0f} µs (+{100 * added_us / base_us:.0f}%)"


def plot_shapes(examples):
    """Small multiples: the sample interval over one minute per shape, irregular blocks shaded."""
    fig, axes = plt.subplots(len(examples), 1, figsize=(11, 2.3 * len(examples)), layout="constrained")
    for ax, (shape, ticks, row) in zip(axes, examples):
        interval_ms = np.diff(ticks) / 1e6
        for block_idx in np.flatnonzero(row["irregular"]):
            ax.axvspan(block_idx, block_idx + 1, color=IRREGULAR, lw=0, zorder=0)
        ax.plot(np.arange(1, MINUTE) / BLOCK, interval_ms, color=SERIES, lw=1, zorder=2)
        ax.set_xlim(0, BLOCKS)
        if interval_ms.max() / max(interval_ms.min(), 1e-9) > 20:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda value, _: f"{value:g}"))
        else:
            pad = max(0.02, 0.1 * (interval_ms.max() - interval_ms.min()))
            ax.set_ylim(interval_ms.min() - pad, interval_ms.max() + pad)
        num_irregular = int(row["irregular"].sum())
        ax.set_title(f"{SHAPE_LABELS[shape]}: {row['bytes']:,} bytes for 60,000 timestamps, "
                     f"{num_irregular}/60 irregular blocks", loc="left", fontsize=10)
        ax.set_ylabel("interval (ms)")
        ax.grid(axis="y", color="0.9", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[-1].set_xlabel("block (1 s each); shaded: irregular blocks, which store per-sample deltas")
    return save_fig(fig, OUT, "time-shapes", "Sample interval over one minute for each clock shape")


def gaps_sweep(values, plain_group):
    """Bytes per block group as the number of gaps on a perfect grid grows."""
    rng = np.random.default_rng(11)
    rows = []
    for num_gaps in (0, 1, 2, 5, 10, 20, 60):
        sizes, irregular = [], []
        for _ in range(16):
            steps = np.ones(MINUTE, np.int64)
            steps[rng.choice(np.arange(1, MINUTE), num_gaps, replace=False)] += rng.integers(5, 3000, num_gaps)
            steps[0] = 0
            ticks = 1_790_000_000_000_000_000 + np.cumsum(steps) * 1_000_000
            row = measure(values, ticks, plain_group, 0, None)
            sizes.append(row["bytes"])
            irregular.append(int(row["irregular"].sum()))
        rows.append((num_gaps, float(np.median(irregular)), float(np.median(sizes))))
    return rows


def rounding_cases(examples):
    """(label, ticks) for the rounding section: the common shapes' examples, jittered clocks on round
    times and free-running, drift, and two shapes without a grid (as in bench/time_axis.py)."""
    harder = dict(patterns(np.random.default_rng(0)))
    index = np.arange(MINUTE)
    jitter = np.round(np.random.default_rng(1).normal(0, 10_000, MINUTE)).astype(np.int64)
    start = 1_790_000_000_000_000_000
    return [(SHAPE_LABELS[shape], ticks) for shape, ticks, _ in examples] + [
        ("Jitter σ = 10 µs (1%), ns, on round times", harder["jitter σ=10 µs, ns resolution"]),
        ("The same, free-running: 50 µs off the epoch's grid",
         np.maximum.accumulate(start + 50_000 + index * MS + jitter)),
        ("Drifting clock (0.99998 ms)", harder["drifting clock (0.99998 ms)"]),
        ("Drift and jitter σ = 10 µs, 37 µs off",
         np.maximum.accumulate(start + 37_000 + np.round(index * 0.99998731 * MS).astype(np.int64) + jitter)),
        ("Poisson events, mean 1 ms, ns", harder["Poisson events, mean 1 ms, ns"]),
        ("Deadband logging, ms grid", harder["deadband logging, ms grid"]),
    ]


def measure_rounded(values, ticks, plain_group, time_error, reps, plain_times):
    """Bytes, irregular blocks, the largest move (in median intervals) and µs the timestamps add when
    rounded to the time error."""
    times = ticks.view("datetime64[ns]")
    params = fluxcode.Params(time_error=time_error)
    group = fluxcode.encode_group(values, params, times=times).group
    decoded = fluxcode.decode_group(group)
    rounded = decoded.times.view(np.int64)
    assert np.all(np.diff(rounded) >= 0), "rounded times out of order"
    irregular = (_group.decompress(group).block_flags & _format.BLOCK_FLAG_IRREGULAR_TIME) != 0
    row = {"bytes": len(group) - len(plain_group), "irregular": irregular,
           "moved": float(np.abs(rounded - ticks).max() / np.median(np.diff(ticks)))}
    if reps:
        row["encode_us"] = best_us(lambda: fluxcode.encode_group(values, params, times=times), reps) - plain_times[0]
        row["decode_us"] = best_us(lambda: fluxcode.decode_group(group), reps) - plain_times[1]
    return row


def plot_rounding(sweep):
    """Bytes the timestamps add against the time error, one line per shape (log scale)."""
    fig, ax = plt.subplots(figsize=(11, 4.2), layout="constrained")
    for color, (label, sizes) in zip(PALETTE, sweep):
        ax.step(SWEEP_ERRORS, sizes, where="post", color=color, lw=1.5, label=label)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax.set_xlabel("time error e (Params.time_error): the largest move, as a share of the block's interval")
    ax.set_ylabel("bytes the timestamps add")
    ax.grid(color="0.9", lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    return save_fig(fig, OUT, "time-rounding", "Bytes the timestamps add per block group against the time error")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=30)
    args = parser.parse_args()
    values = minute("sin-4.12hz", 1)
    plain_group = fluxcode.encode_group(values).group
    plain_times = (best_us(lambda: fluxcode.encode_group(values), args.reps),
                   best_us(lambda: fluxcode.decode_group(plain_group), args.reps))

    examples = []
    for shape in CLOCKS:
        # The example: the first seed whose block group is typical of the shape (median size)
        rows = [(seed, measure(values, clock_minute(shape, seed), plain_group, 0, None)) for seed in range(SEEDS)]
        sizes = np.array([row["bytes"] for _, row in rows])
        seed = rows[int(np.argsort(sizes)[len(sizes) // 2])][0]
        ticks = clock_minute(shape, seed)
        example = measure(values, ticks, plain_group, args.reps, plain_times)
        example["sizes"] = sizes
        examples.append((shape, ticks, example))
        print(f"{shape}: example seed {seed}, {example['bytes']} bytes", flush=True)

    others = []
    for name, ticks in patterns(np.random.default_rng(0))[3:]:
        others.append((name, measure(values, ticks, plain_group, args.reps, plain_times)))
        print(f"{name}: {others[-1][1]['bytes']} bytes", flush=True)
    sweep = gaps_sweep(values, plain_group)

    cases = rounding_cases(examples)
    rounded = []
    for label, ticks in cases:
        rounded.append((label, [measure_rounded(values, ticks, plain_group, e, args.reps, plain_times)
                                for e in TIME_ERRORS]))
        print(f"{label}: " + ", ".join(f"{row['bytes']} B" for row in rounded[-1][1]), flush=True)
    rounding_sweep = [(label, [measure_rounded(values, ticks, plain_group, e, 0, None)["bytes"] for e in SWEEP_ERRORS])
                      for label, ticks in cases if label in SWEPT]
    write_report(examples, others, sweep, rounded, rounding_sweep, len(plain_group), plain_times, args.reps)


def write_report(examples, others, sweep, rounded, rounding_sweep, plain_bytes, plain_times, reps):
    clean_outputs(OUT, ("time.html", "time-*.svg"))
    chart = plot_shapes(examples)
    rounding_chart = plot_rounding(rounding_sweep)
    noisy_rounded = next(rows for label, rows in rounded if label == SHAPE_LABELS["noisy"])
    grid_bytes = examples[0][2]["bytes"]
    h = [f"""<p><a href="index.html">← codec comparison (index.html)</a> ·
<a href="https://github.com/gboyer/fluxcode/blob/main/docs/SPEC.md#3-time-axis">SPEC §3: time axis</a></p>
<p>fluxcode can store each sample's timestamp <b>exactly</b> next to its values (by default; the
<a href="#rounding">last section</a> lets them move by a share of an interval). This page covers only that time
axis: what the timestamps add to a one-minute block group (60 blocks of 1,000 samples, nominally 1 kHz, int64 ns ticks).
Every number here is the block group with times minus the same block group without them. For scale, that block group's values alone take
{plain_bytes:,} bytes (a 4.12 Hz sine at the default settings), {plain_times[0]:.0f} µs to encode and
{plain_times[1]:.0f} µs to decode.
<b>Regenerate:</b> <code>cd experimental &amp;&amp; uv run python make_time_report.py</code>.</p>
<h2 id="how">How timestamps are stored</h2>
<ul>
<li><b>Per block: a start, a step and a reference.</b> The step is the GCD of the block's intervals, and each
interval divided by it is a quotient: the reference plus a residual. A block whose 999 intervals are all equal is
<b>regular</b>: every quotient equals the reference, so it stores nothing else (24 bytes before compression, a few
after). Any other block is <b>irregular</b> and stores the residuals, zigzagged, as 32 bit planes (64 in the rare
<i>long</i> block, with a residual of 2<sup>32</sup> or more: in ns ticks, a gap of seconds). Most of those planes are
zero, and zstd removes them.</li>
<li><b>The reference is the rounded mean or the minimum,</b> chosen per block. The mean suits jitter around a
center; the minimum suits skewed intervals (a gap, events, deadband logging), whose mean sits away from most of
them. Without a reference, intervals near 1000 flip several bit planes together as they wobble, which costs about
a bit per sample.</li>
<li><b>A gap costs only its own block.</b> The blocks before and after it stay regular.</li>
<li><b>The resolution is free.</b> The GCD takes out the grid, so a millisecond grid costs the same in ns, µs or ms
ticks.</li>
<li>Block starts are stored as increases over the previous start, so a regular block group's 60 starts compress to a few
bytes.</li>
</ul>
<h2 id="shapes">The common shapes</h2>
<p>A perfect grid, and a perfect grid with a few gaps, are by far the most common timestamps in practice. A noisy
clock (host timestamps on arrival) is the third common shape, and the only one of the three where the timestamps
cost more than the values (here about two and a half times as much).</p>
{chart}
<table><tr><th class='l'>shape</th><th class='l'>what it is</th><th>bytes (example)</th><th>bits/sample</th>
<th>irregular blocks</th><th>vs values</th><th>bytes over {SEEDS} groups: median</th><th>max</th>
<th>encode time added</th><th>decode time added</th></tr>"""]
    for shape, _, row in examples:
        sizes = row["sizes"]
        h.append(f"<tr><td>{SHAPE_LABELS[shape]}</td><td class='l'>{SHAPE_NOTES[shape]}</td>"
                 f"<td>{row['bytes']:,}</td><td>{8 * row['bytes'] / MINUTE:.3f}</td>"
                 f"<td>{int(row['irregular'].sum())}/60</td><td>{100 * row['bytes'] / plain_bytes:.1f}%</td>"
                 f"<td>{int(np.median(sizes)):,}</td><td>{int(sizes.max()):,}</td>"
                 f"<td>{added(row['encode_us'], plain_times[0])}</td><td>{added(row['decode_us'], plain_times[1])}</td></tr>")
    h.append(f"""</table>
<p class='muted'>The example is the median-size block group of {SEEDS} seeds per shape. Times are single thread, best of
{reps}, Apple M3 on AC power; each added time's percentage is of the same block group's values-only encode
({plain_times[0]:.0f} µs) or decode ({plain_times[1]:.0f} µs). <i>vs values</i> is the timestamps' bytes as a share
of the values' {plain_bytes:,}.</p>
<ul>
<li><b>Perfect grid:</b> {grid_bytes} bytes for 60,000 timestamps, whatever the start, step or tick unit. Decoding
writes the 60,000 int64 ticks, and that is most of its cost.</li>
<li><b>A few gaps:</b> each gap makes one block irregular. Its intervals are all 1 except one, so the block's
planes are nearly all zero and it costs about 20 to 60 bytes (next section).</li>
<li><b>Noisy clock:</b> the jitter is real entropy, and no lossless coder can remove it: σ = 20 µs at µs
resolution carries about 6.4 bits per sample, and the block group spends 7.1. The rest comes from storing intervals
(each the difference of two jitters) with each bit plane coded on its own. If the jitter is measurement noise
rather than information, let fluxcode round it away: with <code>Params(time_error=0.1)</code> the same clock
costs {noisy_rounded[3]['bytes']:,} bytes (<a href="#rounding">last section</a>).</li>
</ul>
<h2 id="gaps">Cost per gap</h2>
<p>A grid with <i>k</i> gaps per minute, each 5 ms to 3 s; median of 16 block groups per row.</p>
<table><tr><th>gaps per minute</th><th>irregular blocks</th><th>bytes</th><th>bits/sample</th><th>bytes per gap</th></tr>""")
    base = sweep[0][2]
    for num_gaps, irregular, size in sweep:
        per_gap = f"{(size - base) / num_gaps:.0f}" if num_gaps else "—"
        h.append(f"<tr><td>{num_gaps}</td><td>{irregular:.0f}/60</td><td>{size:,.0f}</td>"
                 f"<td>{8 * size / MINUTE:.3f}</td><td>{per_gap}</td></tr>")
    h.append("""</table>
<h2 id="harder">Harder shapes</h2>
<p>For comparison, shapes with many more irregular blocks (from <code>bench/time_axis.py</code>). Only the
cases with random intervals cost more than the values. Percentages are of the values-only encode and decode, as
above.</p>
<table><tr><th class='l'>timestamps</th><th>bytes</th><th>bits/sample</th><th>irregular blocks</th>
<th>encode time added</th><th>decode time added</th></tr>""")
    for name, row in others:
        h.append(f"<tr><td>{name}</td><td>{row['bytes']:,}</td><td>{8 * row['bytes'] / MINUTE:.3f}</td>"
                 f"<td>{int(row['irregular'].sum())}/60</td><td>{added(row['encode_us'], plain_times[0])}</td>"
                 f"<td>{added(row['decode_us'], plain_times[1])}</td></tr>")
    h.append("""</table>
<p class='muted'>1% dropped and a drifting clock stay cheap: their intervals take two or three values. Jitter,
Poisson events and deadband logging have random intervals, and ns resolution costs more than µs when the jitter
really is at ns resolution.</p>""")
    h.append(f"""<h2 id="rounding">Rounding timestamps: Params.time_error</h2>
<p>When the jitter is measurement noise, <code>Params(time_error=e)</code> lets each timestamp move by up to
<i>e</i> times its block's interval (0 to 0.5; the default 0 keeps them exact). The values are kept as measured,
and the format doesn't change: the decoder sees ordinary times.</p>
<ul>
<li><b>Per block: a step and a phase.</b> The interval is the mean of the one-scan intervals where the block
has a cadence (90% of its intervals one scan), else the median of 15. The step is the largest 1-2-5 tick
count (1, 2, 5, 10, 20, …) at most 2<i>e</i> intervals (2% slack), and the grid sits at the clock's own phase
(a multiple of a tenth of the step, kept from the previous block unless another fits clearly better), so a
free-running or drifting clock rounds onto its own grid, not the epoch's. Each time rounds to its nearest grid
point.</li>
<li><b>A jittered clock becomes regular</b> once <i>e</i> is about 5 times its jitter (σ over the interval):
every time stays on its own grid point, and the block stores no residuals. Below that the size falls in steps,
one at each 1-2-5 step (for a 1 ms clock at e = …, 0.005, 0.01, 0.025, 0.05, 0.1), and each step pays off on its own:
there is no threshold below which rounding stops helping.</li>
<li><b>Regular blocks are left alone,</b> and times already on a grid don't change.</li>
<li><b>Order is kept</b> across blocks and block groups; <code>update</code> and <code>update_time_blocks</code>
round new samples onto the stored blocks' grids and never move stored times.</li>
</ul>
{rounding_chart}
<table><tr><th class='l'>timestamps</th>""" + "".join(
        f"<th>{'exact' if e == 0 else f'e = {e:g}'}</th>" for e in TIME_ERRORS) + "</tr>")
    for label, rows in rounded:
        cells = []
        for e, row in zip(TIME_ERRORS, rows):
            moved = "" if e == 0 else f" · max {row['moved']:.3f}"
            cells.append(f"<td>{row['bytes']:,} B · {int(row['irregular'].sum())}/60<br><span class='muted'>"
                         f"{added(row['encode_us'], plain_times[0])}{moved}</span></td>")
        h.append(f"<tr><td>{label}</td>{''.join(cells)}</tr>")
    h.append(f"""</table>
<p class='muted'>Each cell: bytes the timestamps add · irregular blocks of 60, then the encode time they add
(percentage of the values-only encode, {plain_times[0]:.0f} µs) and the largest move in median intervals.
Single thread, best of {reps}, Apple M3 on AC power. The chart samples e every 0.0025.</p>
<ul>
<li><b>Jittered clocks</b> drop from about 120 KB to 63–66 bytes once e reaches about 5σ (0.05 here), on
round times or free-running. With drift as well, the phase follows the clock block by block: 258 bytes at 0.05
and 108 at 0.1. Once regular they also encode faster: a regular block skips the residuals.</li>
<li><b>Times already on a grid</b> don't change, but checking them costs time: about +25 µs per block group on a
perfect grid (finding each block's step) and +60 µs with a few gaps (rounding the irregular blocks).</li>
<li><b>The noisy clock</b> (σ = 20 µs, 2%) needs e = 0.1 to land on its grid; at 0.05 about 1% of its
times fall to a neighbouring grid point.</li>
<li><b>Events and deadband logging</b> have no grid to land on. Rounding still takes out the low bits
(events get 3–4 times smaller), but a coarse grid (deadband logging's 1 ms) gains only once the step
passes it. Without a cadence the interval is a rough estimate, so a time can move by more than <i>e</i> of
the overall median (within about <i>e</i> of its block's own estimate).</li>
</ul>
<p class='muted'>Measurements behind the choices (1-2-5 steps over powers of two, no interpolation of values, no
resampling, per-block phases): <a href="https://github.com/gboyer/fluxcode/blob/main/docs/TUNING.md#time-error">docs/TUNING.md,
Time error</a>.</p>""")
    toc = [("how", "How timestamps are stored", []), ("shapes", "The common shapes", []),
           ("gaps", "Cost per gap", []), ("harder", "Harder shapes", []),
           ("rounding", "Rounding timestamps: Params.time_error", [])]
    (OUT / "time.html").write_text(shell("fluxcode timestamps: the time axis", "", toc, "\n".join(h)))
    print(f"wrote {OUT / 'time.html'}")


if __name__ == "__main__":
    main()
