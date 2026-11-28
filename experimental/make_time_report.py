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

from tslab.report.page import clean_outputs, collapsible_table, save_fig, shell  # noqa: E402

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
    group = fluxcode.encode_group(values, block_len=BLOCK, times=times).payload
    decoded = fluxcode.decode_group(group)
    assert decoded.times is not None and np.array_equal(decoded.times, times)
    irregular = (_group.decompress(group).block_flags & _format.BLOCK_FLAG_IRREGULAR_TIME) != 0
    row = {"bytes": len(group) - len(plain_group), "irregular": irregular}
    if reps:
        row["encode_us"] = best_us(lambda: fluxcode.encode_group(values, block_len=BLOCK, times=times), reps) - plain_times[0]
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
    lattice_jitter = np.random.default_rng(2).normal(0, 0.005, MINUTE)
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
    ] + [(f"{hz} Hz clock, jitter 0.5% (no 1-2-5 step divides the period)",
          np.maximum.accumulate(start + np.round(index * 1e9 / hz + lattice_jitter * 1e9 / hz).astype(np.int64)))
         for hz in (1024, 60)]


def measure_rounded(values, ticks, plain_group, time_error, reps, plain_times):
    """Bytes, irregular blocks, the largest move (in mean intervals) and µs the timestamps add when
    rounded to the time error."""
    times = ticks.view("datetime64[ns]")
    params = fluxcode.Params(time_error=time_error)
    group = fluxcode.encode_group(values, params, block_len=BLOCK, times=times).payload
    decoded = fluxcode.decode_group(group)
    rounded = decoded.times.view(np.int64)
    assert np.all(np.diff(rounded) >= 0), "rounded times out of order"
    irregular = (_group.decompress(group).block_flags & _format.BLOCK_FLAG_IRREGULAR_TIME) != 0
    row = {"bytes": len(group) - len(plain_group), "irregular": irregular,
           "moved": float(np.abs(rounded - ticks).max() / np.mean(np.diff(ticks)))}
    if reps:
        row["encode_us"] = best_us(lambda: fluxcode.encode_group(values, params, block_len=BLOCK, times=times), reps) - plain_times[0]
        row["decode_us"] = best_us(lambda: fluxcode.decode_group(group), reps) - plain_times[1]
    return row


def plot_rounding(sweep):
    """Bytes the timestamps add against the time error, one line per shape (log scale)."""
    fig, ax = plt.subplots(figsize=(11, 4.2), layout="constrained")
    for color, (label, sizes) in zip(PALETTE, sweep):
        ax.step(SWEEP_ERRORS, sizes, where="post", color=color, lw=1.5, label=label)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax.set_xlabel("time error e (Params.time_error)")
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
    plain_group = fluxcode.encode_group(values, block_len=BLOCK).payload
    plain_times = (best_us(lambda: fluxcode.encode_group(values, block_len=BLOCK), args.reps),
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
    intro = f"""<p>By default, fluxcode stores each sample's timestamp <b>exactly</b> alongside its values (the
<a href="#rounding">last section</a> covers optional rounding by a fraction of an interval). This report isolates the time
axis (<a href="spec.html#3-time-axis">SPEC §3</a>) and measures what timestamps add to a one-minute block group
(60 blocks of 1,000 samples at a nominal 1 kHz, stored as int64 nanosecond ticks). Every figure below is the difference
between encoding the block group with timestamps and encoding it without them. For reference, the values alone (a 4.12 Hz
sine wave at default settings) take {plain_bytes:,} bytes, {plain_times[0]:.0f} µs to encode, and
{plain_times[1]:.0f} µs to decode.</p>
<p class='muted'><b>Regenerate:</b> <code>cd experimental &amp;&amp; uv run python make_time_report.py</code>.</p>"""
    shapes_tbl = [f"<table><tr><th class='l'>shape</th><th class='l'>what it is</th><th>bytes (example)</th><th>bits/sample</th>"
                  f"<th>irregular blocks</th><th>vs values</th><th>bytes over {SEEDS} groups: median</th><th>max</th>"
                  "<th>encode time added</th><th>decode time added</th></tr>"]
    for shape, _, row in examples:
        sizes = row["sizes"]
        shapes_tbl.append(f"<tr><td>{SHAPE_LABELS[shape]}</td><td class='l'>{SHAPE_NOTES[shape]}</td>"
                          f"<td>{row['bytes']:,}</td><td>{8 * row['bytes'] / MINUTE:.3f}</td>"
                          f"<td>{int(row['irregular'].sum())}/60</td><td>{100 * row['bytes'] / plain_bytes:.1f}%</td>"
                          f"<td>{int(np.median(sizes)):,}</td><td>{int(sizes.max()):,}</td>"
                          f"<td>{added(row['encode_us'], plain_times[0])}</td><td>{added(row['decode_us'], plain_times[1])}</td></tr>")
    shapes_tbl.append(f"</table><p class='muted'>Each example is the median-size block group across {SEEDS} seeds for that shape. "
                      f"Timings are single-threaded, best of {reps} runs on an Apple M3 on AC power; added-time percentages "
                      f"are relative to the values-only encode ({plain_times[0]:.0f} µs) and decode ({plain_times[1]:.0f} µs). "
                      f"<i>vs values</i> shows timestamp bytes as a percentage of the {plain_bytes:,} value bytes.</p>")

    h = [f"""<h2 id="how">How timestamps are stored</h2>
<ul>
<li>Each block records a start timestamp, a step, and a reference. The step is the GCD of the block's intervals, and
dividing each interval by that step produces a quotient written as the reference plus a residual. When all 999 intervals
in a block are equal, every quotient matches the reference and the block is <b>regular</b>, needing no residual data
(24 bytes before compression, and only a few bytes after). Any other block is <b>irregular</b> and stores its zigzagged
residuals across 32 bit planes—or 64 in the rare <i>long</i> block where a residual reaches 2<sup>32</sup> or more (a
multi-second gap in nanosecond ticks). Most planes are zero, and zstd strips them away.</li>
<li>The reference is chosen per block as either the rounded mean or the minimum. The mean works best for symmetric
jitter around a center, while the minimum suits skewed distributions (such as gaps, events, or deadband logging) where
the mean sits away from most intervals. Without a reference, intervals wobbling near 1,000 flip several bit planes at
once, costing roughly a bit per sample.</li>
<li>A gap only affects its own block; the blocks before and after it remain regular.</li>
<li>Tick resolution is effectively free because the GCD factors out the grid, so a 1 ms grid costs the same whether
stored in nanoseconds, microseconds, or milliseconds.</li>
<li>Block starts are delta-encoded from the previous block's start, so the 60 starts of a regular block group compress
to just a few bytes.</li>
</ul>
<h2 id="shapes">The common shapes</h2>
<p>In practice, most time series fall on either a perfect grid or a grid with occasional gaps. A noisy clock—such as
host arrival timestamps—is the third common case, and the only one where timestamps take more space than the values
themselves (about two and a half times as much here).</p>
{chart}
{collapsible_table(f"Show common shapes table ({len(examples)} shapes)", "".join(shapes_tbl))}
<ul>
<li><b>Perfect grid</b> takes {grid_bytes} bytes for 60,000 timestamps regardless of the start time, step size, or tick
unit. Most of the decode time is simply writing out the 60,000 int64 ticks.</li>
<li><b>A few gaps</b> turn only the affected blocks irregular. Because all but one interval in each affected block equals
1, its bit planes are almost entirely zero, adding roughly 20 to 60 bytes per gap (see the next section).</li>
<li><b>Noisy clock</b> jitter is genuine entropy that no lossless coder can eliminate. At µs resolution, σ = 20 µs
carries about 6.4 bits of entropy per sample, and the block group uses 7.1 bits; the extra overhead comes from encoding
intervals (each the difference of two jittered times) with independent bit planes. When that jitter is measurement noise
rather than signal, <code>Params(time_error=0.1)</code> lets fluxcode round it away, bringing the same clock down to
{noisy_rounded[3]['bytes']:,} bytes (<a href="#rounding">last section</a>).</li>
</ul>"""]

    gaps_tbl = ["<table><tr><th>gaps per minute</th><th>irregular blocks</th><th>bytes</th><th>bits/sample</th><th>bytes per gap</th></tr>"]
    base = sweep[0][2]
    for num_gaps, irregular, size in sweep:
        per_gap = f"{(size - base) / num_gaps:.0f}" if num_gaps else "—"
        gaps_tbl.append(f"<tr><td>{num_gaps}</td><td>{irregular:.0f}/60</td><td>{size:,.0f}</td>"
                        f"<td>{8 * size / MINUTE:.3f}</td><td>{per_gap}</td></tr>")
    gaps_tbl.append("</table>")
    h.append(f"<h2 id='gaps'>Cost per gap</h2><p>A 1 ms grid with <i>k</i> gaps per minute (each lasting 5 ms to 3 s), showing the median across 16 block groups per row.</p>"
             + collapsible_table(f"Show gap sweep table ({len(sweep)} rows)", "".join(gaps_tbl)))

    harder_tbl = ["<table><tr><th class='l'>timestamps</th><th>bytes</th><th>bits/sample</th><th>irregular blocks</th>"
                  "<th>encode time added</th><th>decode time added</th></tr>"]
    for name, row in others:
        harder_tbl.append(f"<tr><td>{name}</td><td>{row['bytes']:,}</td><td>{8 * row['bytes'] / MINUTE:.3f}</td>"
                          f"<td>{int(row['irregular'].sum())}/60</td><td>{added(row['encode_us'], plain_times[0])}</td>"
                          f"<td>{added(row['decode_us'], plain_times[1])}</td></tr>")
    harder_tbl.append("</table>")
    h.append("""<h2 id="harder">Harder shapes</h2>
<p>For comparison, here are patterns from <code>bench/time_axis.py</code> that produce many more irregular blocks. Only
patterns with random intervals end up larger than the values. As above, percentages are relative to the values-only
encode and decode times.</p>"""
             + collapsible_table(f"Show harder shapes table ({len(others)} patterns)", "".join(harder_tbl))
             + """<p class='muted'>A 1% drop rate and a drifting clock stay compact because their intervals only take
two or three distinct values. Jitter, Poisson events, and deadband logging have random intervals, and nanosecond
resolution costs more than microsecond resolution whenever the jitter actually extends down to nanoseconds.</p>""")

    round_tbl = ["<table><tr><th class='l'>timestamps</th>" + "".join(
        f"<th>{'exact' if e == 0 else f'e = {e:g}'}</th>" for e in TIME_ERRORS) + "</tr>"]
    for label, rows in rounded:
        cells = []
        for e, row in zip(TIME_ERRORS, rows):
            moved = "" if e == 0 else f" · max {row['moved']:.3f}"
            cells.append(f"<td>{row['bytes']:,} B · {int(row['irregular'].sum())}/60<br><span class='muted'>"
                         f"{added(row['encode_us'], plain_times[0])}{moved}</span></td>")
        round_tbl.append(f"<tr><td>{label}</td>{''.join(cells)}</tr>")
    round_tbl.append(f"</table><p class='muted'>Each cell shows the bytes added by timestamps and the number of irregular "
                     f"blocks out of 60, followed by the added encode time (as a percentage of the {plain_times[0]:.0f} µs "
                     f"values-only encode) and the largest timestamp shift in mean intervals. Timings are single-threaded, "
                     f"best of {reps} runs on an Apple M3 on AC power. The chart samples <i>e</i> every 0.0025.</p>")
    by_label = dict(rounded)
    col = {e: idx for idx, e in enumerate(TIME_ERRORS)}

    def cell(label, e, key="bytes"):
        return by_label[label][col[e]][key]

    def check_cost(label):  # what rounding adds over exact times on a block group already on a grid
        return added(cell(label, 0.05, "encode_us") - cell(label, 0.0, "encode_us"), plain_times[0])
    jitter_bytes = [cell(label, 0.05) for label in ("Jitter σ = 10 µs (1%), ns, on round times",
                                                   "The same, free-running: 50 µs off the epoch's grid")]
    drift = "Drift and jitter σ = 10 µs, 37 µs off"
    events = "Poisson events, mean 1 ms, ns"
    events_ratio = cell(events, 0.0) / cell(events, 0.05)
    lattice_labels = [label for label in by_label if "no 1-2-5 step" in label]
    lattice_text = "; ".join(
        f"{label.split(',')[0]}: {cell(label, 0.0):,} → {cell(label, 0.05):,} bytes at 0.05 and {cell(label, 0.1):,} at 0.1"
        for label in lattice_labels)
    h.append(f"""<h2 id="rounding">Rounding timestamps: Params.time_error</h2>
<p>When timestamp jitter is just measurement noise, <code>Params(time_error=e)</code> allows fluxcode to round
timestamps (<i>e</i> ranges from 0 to 0.5, with 0 keeping them exact). Each block gets a coarse, consistent step where
at least 95% of timestamps stay within <i>e</i> of their own adjacent interval and no timestamp moves more than <i>e</i>
of the block's mean interval. Sample values are left untouched and the wire format stays identical, so the decoder simply
reads ordinary timestamps.</p>
<ul>
<li><b>Validation rule.</b> No timestamp may move by more than <i>e</i> mean intervals (the block span divided by its
interval count, with 2% slack), and at most 1 in 20 may move by more than <i>e</i> times its own local interval (the
shorter of the two intervals on either side). Only unusually close timestamps can exceed <i>e</i> relative to their local
interval, and even then they remain bounded by the block mean.</li>
<li><b>Candidate search.</b> The encoder tries candidates in order and takes the first one that passes. First, it checks
the lattice continuing the previous block if that block was regular on the same clock. Next, it tries a 1-2-5 grid
(1, 2, 5, 10, 20, … ticks), starting at the coarsest step up to 2<i>e</i> mean intervals (which clocks usually pass) or
stepping down from a short sampled interval (the 3rd smallest of 15) until it passes. Each grid uses the clock's own
phase—a multiple of one-tenth of the step, carried over from the previous block unless another fits clearly better—so
free-running and drifting clocks snap to their own grids rather than the epoch's. Finally, for clocks whose period no
1-2-5 step divides (such as 1024 Hz or 60 Hz, where rounded intervals would alternate), it tries a custom lattice using
the clock's typical interval as the period.</li>
<li><b>Smooth degradation.</b> Once <i>e</i> reaches roughly 5× the clock's relative jitter (σ divided by the interval),
a jittered clock snaps onto a regular grid and stores no residuals. Below that point, size drops in discrete steps as
each finer 1-2-5 grid becomes eligible, so even a small tolerance still helps.</li>
<li><b>No-op on clean grids.</b> Regular blocks are skipped entirely, and timestamps already on a grid are never
altered.</li>
<li><b>Ordering and mutations.</b> Timestamp order is preserved across blocks and block groups, and within time blocks
every sample stays in its original block (a timestamp that would round past the end is clamped to the end). Rounding
happens once during encoding: <code>update_time_blocks</code> stores newly added timestamps exactly, whereas
<code>replace_time_blocks</code> re-encodes whole blocks from the supplied samples.</li>
</ul>
{rounding_chart}
{collapsible_table(f"Show time_error table ({len(rounded)} patterns × {len(TIME_ERRORS)} values)", "".join(round_tbl))}
<ul>
<li><b>Jittered clocks</b> drop from roughly 120 KB to {min(jitter_bytes)}–{max(jitter_bytes)} bytes once <i>e</i>
reaches about 5σ (0.05 here), whether aligned to round times or free-running. When drift is also present, the phase
tracks the clock block by block, reaching {cell(drift, 0.05):,} bytes at 0.05 and {cell(drift, 0.1):,} bytes at 0.1.
Becoming regular speeds up encoding as well because regular blocks skip residual coding.</li>
<li><b>Non-1-2-5 clocks</b> snap to their own custom lattices once <i>e</i> covers the jitter (around 5σ):
{lattice_text}.</li>
<li><b>Already-regular grids</b> do not change, though verifying them has a small CPU cost: {check_cost("Perfect grid")}
per block group on a perfect grid (checking order and regularity) and {check_cost("Grid with a few gaps")} when there
are a few gaps (attempting to round the irregular blocks).</li>
<li><b>The noisy clock</b> (σ = 20 µs, or 2%) needs <i>e</i> = 0.1 to land cleanly on its grid; at 0.05, about 1% of its
timestamps still fall onto a neighboring grid point.</li>
<li><b>Events and deadband logging</b> have no underlying clock grid, though rounding still trims low-order bits
(Poisson events shrink by {events_ratio:.1f}× at 0.05). Because Poisson events contain many tight pairs and 95% of
timestamps must stay within <i>e</i> of their local interval, they get a finer step than a periodic clock at the same
<i>e</i>. Deadband timestamps already sit on a 1 ms grid mostly a few milliseconds apart, so any coarser step violates
the local-interval bound for too many points; below <i>e</i> = 0.5 it saves nothing and costs
{added(cell("Deadband logging, ms grid", 0.05, "encode_us") - cell("Deadband logging, ms grid", 0.0, "encode_us"), plain_times[0])}
per block group at 0.05 to check.</li>
</ul>
<p class='muted'>For the measurements behind these design choices (1-2-5 steps over powers of two, no value
interpolation or resampling, and per-block phases), see
<a href="https://github.com/gboyer/fluxcode/blob/main/docs/TUNING.md#time-error">docs/TUNING.md (Time error)</a>.</p>""")
    toc = [("how", "How timestamps are stored", []), ("shapes", "The common shapes", []),
           ("gaps", "Cost per gap", []), ("harder", "Harder shapes", []),
           ("rounding", "Rounding timestamps: Params.time_error", [])]
    (OUT / "time.html").write_text(shell("fluxcode timestamps: the time axis", intro, toc, "\n".join(h), active="time"))
    print(f"wrote {OUT / 'time.html'}")


if __name__ == "__main__":
    main()
