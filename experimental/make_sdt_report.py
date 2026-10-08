# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""fluxcode on what a swinging-door historian archives -> report/sdt.html.

    uv run python make_sdt_report.py

A historian that compresses with swinging-door trending (SDT) keeps only some of each tag's
scans, at irregular times (tslab.common.historian). This page encodes those archives, one
tag-day per block group with exact timestamps, and measures bytes per archived point (values and
times), fluxcode's error against the archived points and against the scans the historian saw,
the block layout, a B sweep (size vs accuracy) and a CompDev sweep against encoding every scan
with fluxcode instead of SDT. Sizes and errors only, no timings.
"""

import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
import _fresh_numba_cache  # noqa: E402, F401  (before fluxcode: a cache keyed on its sources)
import html
from pathlib import Path

import fluxcode
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import zstandard
from fluxcode import Params, _noise
from fluxcode._encoder import NOISE_MIN_LEN

from tslab.classic.gorilla import Gorilla
from tslab.common.historian import (
    DAY_S,
    DAY_START_NS,
    DAYS,
    SECOND_NS,
    TAG_INFO,
    TAGS,
    hold,
    interp,
    tag_day,
    times_ns,
)
from tslab.report.page import clean_outputs, save_fig, shell

OUT = Path(__file__).parent / "report"
SERIES = "#2a78d6"  # the reference palette's first two slots
SECOND = "#eb6834"
SCANS = "0.72"
RAW_BYTES = 16  # an int64 timestamp and a float64 value per point

DEFAULT = Params()  # noise floor 0.25, as much of it as the archive's share of consecutive scans allows
OFF = Params(noise_floor_sigma=0)
DAY_START = np.datetime64(DAY_START_NS, "ns")


def time_blocks(minutes):
    def encode(values, ticks, params):
        return fluxcode.encode_time_blocks(values, ticks.view("datetime64[ns]"), params, start_time=DAY_START,
                                           block_duration=np.timedelta64(minutes, "m")).group
    return encode


def point_blocks(size):
    def encode(values, ticks, params):
        return fluxcode.encode_group(values, params, block_len=size, times=ticks.view("datetime64[ns]")).group
    return encode


LAYOUTS = [("10 min blocks", time_blocks(10)), ("1 h blocks", time_blocks(60)),
           ("1000-point blocks", point_blocks(1000)), ("one block per day", point_blocks(65_535))]
LAYOUT = "1 h blocks"  # the layout of every other table: fixed time buckets, as a store would use
ENCODE = dict(LAYOUTS)[LAYOUT]
SWEEP_BITS = (4, 6, 8, 10, 12, 14, 16)
DEV_SCALES = (0.25, 0.5, 1, 2, 4)  # CompDev as a multiple of the tag's own
EXACT_FLOOR = 1e-5  # where an exact decode sits on the log error axes


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------

def gorilla_bytes(values, ticks, tick_ns):
    """Gorilla (VLDB 2015) with its own timestamps: a 64-bit first time, a 32-bit first delta,
    then each delta of deltas (in ticks of tick_ns) in its bucket: 0 -> 1 bit, |d| < 64 -> 9,
    < 256 -> 12, < 2048 -> 16, else 36 bits."""
    dod = np.abs(np.diff(np.diff(ticks // tick_ns)))
    bits = 96 + np.select([dod == 0, dod < 64, dod < 256, dod < 2048], [1, 9, 12, 16], 36).sum()
    return int(np.ceil(bits / 8)) + len(Gorilla().encode(values)[0])


def zstd_bytes(values, ticks):
    """Lossless baseline: zstd-3 of the ns time deltas (int64) followed by the values (float64)."""
    deltas = np.diff(ticks, prepend=0)
    return len(zstandard.ZstdCompressor(level=3).compress(deltas.tobytes() + values.tobytes()))


def flux_measure(scans, idx, ticks, params, encode=ENCODE):
    """Bytes (all, and values only), errors and floored blocks of one archive. The times' bytes are
    those of the same blocks with the noise floor off, minus the same without times (they don't
    depend on the floor); floored blocks are those that decode differently with it off."""
    values = scans[idx]
    group = encode(values, ticks, params)
    decoded = fluxcode.decode_group(group)
    assert np.array_equal(decoded.times.view(np.int64), ticks)
    off = dataclasses.replace(params, noise_floor_sigma=0)
    off_group = encode(values, ticks, off)
    time_bytes = len(off_group) - len(fluxcode.encode_blocks(values, decoded.block_sizes, off).group)
    ends = np.cumsum(decoded.block_sizes)
    differs = np.add.reduceat(decoded.values != fluxcode.decode_group(off_group).values, ends - decoded.block_sizes)
    trend = interp(idx, decoded.values, DAY_S)
    return {"bytes": len(group), "value_bytes": len(group) - time_bytes, "point_err": np.abs(decoded.values - values).max(),
            "exact": np.count_nonzero(decoded.values == values), "trend_err": np.abs(trend - scans).max(),
            "trend_rms": np.sqrt(np.mean((trend - scans) ** 2)),
            "floored": int(np.count_nonzero(differs[decoded.block_sizes > 0])),
            "blocks": int(np.count_nonzero(decoded.block_sizes >= NOISE_MIN_LEN))}


def one_scan_shares(ticks):
    """The share of one-scan intervals (as the noise floor sees them) of each 1 h block of at
    least NOISE_MIN_LEN points."""
    hours = (ticks - DAY_START_NS) // (3600 * SECOND_NS)
    shares = []
    for hour in np.unique(hours):
        block = ticks[hours == hour]
        if block.size >= NOISE_MIN_LEN:
            shares.append(1.0 if _noise.regular_times(block) else _noise.cadence(block, np.empty(block.size - 1), np.empty(_noise.CADENCE_COUNTS, np.int32))[0])
    return shares


def every_scan(scans, budgets):
    """fluxcode on every scan (a 1 s grid, times included, 1000-scan blocks), no SDT: (bytes, max
    error) at the default parameters, and per budget the smallest (bytes, max error, B) whose max
    error is within it."""
    ticks = (DAY_START_NS + np.arange(DAY_S, dtype=np.int64) * SECOND_NS).view("datetime64[ns]")

    def run(params):
        group = fluxcode.encode_group(scans, params, times=ticks).group
        return len(group), np.abs(fluxcode.decode_group(group).values - scans).max()

    sweep = [(*run(Params(min_quantize_bits=1, max_quantize_bits=bits, noise_floor_sigma=0)), bits)
             for bits in range(1, 17)]
    return run(DEFAULT), [min(fit for fit in sweep if fit[1] <= budget) for budget in budgets]


def resampled(read_back, idx, values):
    """fluxcode (defaults) on the archive read back at every scan: bytes (1 s times included, 1000-scan blocks) and the
    max error against that read-back."""
    series = read_back(idx, values, DAY_S)
    ticks = (DAY_START_NS + np.arange(DAY_S, dtype=np.int64) * SECOND_NS).view("datetime64[ns]")
    group = fluxcode.encode_group(series, DEFAULT, times=ticks).group
    return len(group), np.abs(fluxcode.decode_group(group).values - series).max()


READ_BACKS = (("interpolated", interp), ("held", hold))


def measure_tag(tag):
    _, _, dev, _ = TAG_INFO[tag]
    r = {"points": [], "sdt_trend_err": [], "sdt_trend_rms": [], "gorilla": 0, "zstd": 0, "gorilla_src": 0,
         "flux": {}, "flux_src": [], "layouts": {name: 0 for name, _ in LAYOUTS}, "sweep": {b: [] for b in SWEEP_BITS},
         "scans_default": [], "scans_budget": [], "scans_matched": [], "dev_sweep": {s: [] for s in DEV_SCALES},
         "shares": {s: [] for s in DEV_SCALES},
         "resampled": {name: [] for name, _ in READ_BACKS}}
    for day in range(DAYS):
        scans, idx = tag_day(tag, day)
        values, ticks, src = scans[idx], times_ns(idx, tag, day), times_ns(idx, tag, day, source_timestamps=True)
        trend = interp(idx, values, DAY_S)
        r["points"].append(len(idx))
        r["sdt_trend_err"].append(np.abs(trend - scans).max())
        r["sdt_trend_rms"].append(np.sqrt(np.mean((trend - scans) ** 2)))
        r["gorilla"] += gorilla_bytes(values, ticks, SECOND_NS)
        r["gorilla_src"] += gorilla_bytes(values, src, 1_000_000)
        r["zstd"] += zstd_bytes(values, ticks)
        for name, params in (("defaults", DEFAULT), ("noise floor off", OFF)):
            r["flux"].setdefault(name, []).append(flux_measure(scans, idx, ticks, params))
        r["flux_src"].append(flux_measure(scans, idx, src, DEFAULT))
        for name, read_back in READ_BACKS:
            r["resampled"][name].append(resampled(read_back, idx, values))
        for name, encode in LAYOUTS:
            r["layouts"][name] += flux_measure(scans, idx, ticks, DEFAULT, encode)["bytes"]
        for bits in SWEEP_BITS:
            r["sweep"][bits].append(flux_measure(scans, idx, ticks, Params(min_quantize_bits=1, max_quantize_bits=bits, noise_floor_sigma=0)))
        default, (budget, matched) = every_scan(scans, (dev, r["sdt_trend_err"][-1]))
        r["scans_default"].append(default)
        r["scans_budget"].append(budget)
        r["scans_matched"].append(matched)
        for scale in DEV_SCALES:
            s_scans, s_idx = tag_day(tag, day, comp_dev=scale * dev)
            s_ticks = times_ns(s_idx, tag, day)
            sdt = flux_measure(s_scans, s_idx, s_ticks, DEFAULT)
            sdt["off_bytes"] = len(ENCODE(s_scans[s_idx], s_ticks, OFF))
            sdt["n"] = len(s_idx)
            sdt["every_scan"] = every_scan(s_scans, (sdt["trend_err"],))[1][0][0]
            r["dev_sweep"][scale].append(sdt)
            r["shares"][scale] += one_scan_shares(s_ticks)
    r["n"] = sum(r["points"])
    print(f"{tag}: {r['n'] / DAYS:.0f} points/day", flush=True)
    return r


def total(rows, key):
    return sum(row[key] for row in rows)


def worst(rows, key):
    return max(row[key] for row in rows)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def tidy(ax):
    ax.grid(axis="y", color="0.9", lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def clock(hours, _=None):
    minutes = round(60 * hours)
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def plot_tags():
    """A window of each tag's first day (the whole day for the sparse tags, minutes for the noisy
    ones): the scans, the archived points and the trend between them."""
    fig, axes = plt.subplots(len(TAGS), 1, figsize=(11, 1.9 * len(TAGS)), layout="constrained")
    for ax, tag in zip(axes, TAGS):
        scans, idx = tag_day(tag, 0)
        start, hours = TAG_INFO[tag][3]
        s, e = round(3600 * start), round(3600 * (start + hours))
        sel = idx[(idx >= s) & (idx < e)]
        ax.plot(np.arange(s, e) / 3600, scans[s:e], color=SCANS, lw=0.7, label="scans (1 s)")
        ax.plot(np.arange(s, e) / 3600, interp(idx, scans[idx], DAY_S)[s:e], color=SERIES, lw=1.2,
                label="trend: straight lines between archived points")
        ax.plot(sel / 3600, scans[sel], "o", ms=3.5, color=SERIES, label="archived points")
        ax.set_xlim(start, start + hours)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(clock))
        length = f"{hours:g} h" if hours >= 1 else f"{round(60 * hours)} min"
        ax.set_title(f"{tag}: {len(sel):,} archived points in {length}, CompDev {TAG_INFO[tag][2]:g}",
                     loc="left", fontsize=10)
        tidy(ax)
    axes[-1].set_xlabel("time of day")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=9, frameon=False)
    return save_fig(fig, OUT, "historian-tags", "One hour of each tag: scans, archived points and the trend")


def panels():
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), layout="constrained")
    return fig, axes.flat


def plot_sweep(results):
    """Per tag: bytes per archived point vs fluxcode's max error (in CompDevs) for B = 4..16."""
    fig, axes = panels()
    for ax, tag in zip(axes, TAGS):
        r, dev = results[tag], TAG_INFO[tag][2]
        pts = [(total(r["sweep"][b], "bytes") / r["n"], max(worst(r["sweep"][b], "point_err") / dev, EXACT_FLOOR))
               for b in SWEEP_BITS]
        ax.plot(*zip(*pts), "-o", color=SERIES, lw=1.5, ms=5, label="B = 4..16, noise floor off")
        for b, (x, y) in zip(SWEEP_BITS, pts):
            if b in (SWEEP_BITS[0], SWEEP_BITS[-1]):
                ax.annotate(f"B={b}", (x, y), textcoords="offset points", xytext=(6, 3), fontsize=7.5, color="0.3")
        d = r["flux"]["defaults"]
        ax.plot(total(d, "bytes") / r["n"], max(worst(d, "point_err") / dev, EXACT_FLOOR), "D", color=SECOND, ms=8,
                mec="k", mew=0.6, label="defaults (B = 16, the noise floor as the archive allows)")
        ax.axhline(1, color="0.45", ls=":", lw=1)
        ax.axhline(EXACT_FLOOR, color="0.8", lw=0.8)
        ax.set_yscale("log")
        ax.set_ylim(EXACT_FLOOR / 3, 30)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: "exact" if v == EXACT_FLOOR else f"{v:g}"))
        ax.set_title(tag, fontsize=10)
        ax.set_xlabel("bytes per archived point (values + times)", fontsize=8)
        ax.set_ylabel("max error / CompDev", fontsize=8)
        ax.tick_params(labelsize=7)
        tidy(ax)
    handles, labels = fig.axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color="0.45", ls=":", lw=1))
    labels.append("1 CompDev, the historian's own tolerance; bottom line: exact")
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=9, frameon=False)
    return save_fig(fig, OUT, "historian-bsweep", "Bytes per archived point vs max error per tag, B sweep")


def plot_dev_sweep(results):
    """Per tag: kB per day vs CompDev, for SDT + fluxcode and for fluxcode on every scan."""
    fig, axes = panels()
    for ax, tag in zip(axes, TAGS):
        r, dev = results[tag], TAG_INFO[tag][2]
        devs = [s * dev for s in DEV_SCALES]
        sdt = [total(r["dev_sweep"][s], "bytes") / DAYS / 1000 for s in DEV_SCALES]
        scans = [total(r["dev_sweep"][s], "every_scan") / DAYS / 1000 for s in DEV_SCALES]
        ax.plot(devs, sdt, "-o", color=SERIES, lw=1.5, ms=5, label="SDT archive + fluxcode (defaults)")
        ax.plot(devs, scans, "-s", color=SECOND, lw=1.5, ms=5,
                label="every scan with fluxcode, no SDT: coarsest B whose max error is within SDT's")
        ax.axvline(dev, color="0.45", ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(devs, [f"{d:g}" for d in devs])
        ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1, 2, 5)))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.minorticks_off()
        ax.set_title(tag, fontsize=10)
        ax.set_xlabel("CompDev (engineering units); dotted: the tag's own", fontsize=8)
        ax.set_ylabel("kB per day", fontsize=8)
        ax.tick_params(labelsize=7)
        tidy(ax)
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=9, frameon=False)
    return save_fig(fig, OUT, "historian-compdev", "kB per day vs CompDev per tag, SDT + fluxcode vs every scan")


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------

def per_point(nbytes, n):
    return f"{nbytes / n:.2f}"


def rel(err, dev):
    return "exact" if err == 0 else f"{err / dev:.2g}"


def write_report(results):
    clean_outputs(OUT, ("sdt.html", "historian-*.svg"))
    days = f"{DAYS} days"
    h = [f"""<p><a href="index.html">← codec comparison (index.html)</a> · <a href="time.html">the time axis (time.html)</a></p>
<p>Process historians that compress with <b>swinging-door trending</b> (SDT; PI's compression, and many others) scan each
tag at a fixed rate, drop scans that are within an exception deadband (ExcDev) of the last reported one, and archive only
the points that swinging door keeps: actual scans, at irregular times, such that straight lines between them stay
close to every snapshot (nominally within CompDev; <a href="#tags">in practice up to twice that</a>). That is what an export of the archive gives you, and what this page encodes: one block group per tag
and day, values and exact timestamps. The data are six simulated tags ({days} each, 1 s scans) chosen to cover what SDT
archives look like: from 1 point in 2,000 scans (a valve) to nearly every scan (a noisy tag with a tight CompDev).</p>
<p>Store the archived points themselves. A historian can also export a regular series (interpolated, or held as a
forward fill gives), but that throws the points away and costs more: <a href="#resampled">resampled exports</a>.</p>
<p><b>Regenerate:</b> <code>cd experimental &amp;&amp; uv run python make_sdt_report.py</code> (a few minutes; sizes and
errors only, no timings). The simulator is <a href="../tslab/common/historian.py"><code>tslab/common/historian.py</code></a>.</p>
{findings(results)}
<h2 id="tags">The tags</h2>
<p>Each tag is scanned every second for a day, then filtered by exception (ExcDev = CompDev / 2, the usual rule; the scan
before each exception is reported too) and by swinging door (CompMax 8 h). Times are whole seconds, as a scan class gives.
Errors are in CompDevs, CompDev being the historian's nominal tolerance. The historian's trend (straight lines between
archived points) strays from the scans by up to about 2.5 CompDevs: swinging door archives the real scan where the doors
close, and the line to it is only guaranteed within CompDev at that scan, so earlier snapshots can be up to 2 CompDevs off
it (about 1.9 here); the exception filter adds the rest.</p>
<table><tr><th class='l'>tag</th><th class='l'>what it is</th><th>CompDev</th><th>points per day</th>
<th>scans per point</th><th>trend vs scans: max</th><th>RMS</th></tr>"""]
    for tag in TAGS:
        r, (desc, _, dev, _) = results[tag], TAG_INFO[tag]
        h.append(f"<tr><td>{tag}</td><td class='l'>{html.escape(desc)}</td><td>{dev:g}</td>"
                 f"<td>{r['n'] / DAYS:,.0f}</td><td>{DAY_S * DAYS / r['n']:,.1f}</td>"
                 f"<td>{max(r['sdt_trend_err']) / dev:.2f}</td><td>{np.mean(r['sdt_trend_rms']) / dev:.2f}</td></tr>")
    h.append("</table><p class='muted'>Points per day: mean over the days. Trend vs scans: the worst over the days, "
             "and the mean RMS, in CompDevs, before any fluxcode.</p>")
    h.append(plot_tags())

    h.append(f"""<h2 id="size">Size</h2>
<p>Bytes per archived point over all {days}, everything a decoder needs included. Raw is an int64 time and a float64 value
(16 bytes). zstd and Gorilla are lossless: zstd-3 of the time deltas and the values as two int64/float64 columns; Gorilla
with its own delta-of-delta timestamps (in seconds) and XOR values. fluxcode is one block group per tag-day in {LAYOUT}
(<a href="#layout">layout</a>), with exact times; its <i>values</i> and <i>times</i> columns split its bytes, the
times' being those of the same blocks with the noise floor off minus the same blocks without times.</p>
<table><tr><th class='l'>tag</th><th>points per day</th><th>raw</th><th>zstd</th><th>Gorilla</th>
<th>fluxcode defaults</th><th>values</th><th>times</th><th>max error</th><th>exact</th><th>kB per day</th>
<th>noise floor off</th><th>max error</th></tr>""")
    for tag in TAGS:
        r, dev = results[tag], TAG_INFO[tag][2]
        n, o, d = r["n"], r["flux"]["noise floor off"], r["flux"]["defaults"]
        h.append(f"<tr><td>{tag}</td><td>{n / DAYS:,.0f}</td><td>{RAW_BYTES}</td><td>{per_point(r['zstd'], n)}</td>"
                 f"<td>{per_point(r['gorilla'], n)}</td><td><b>{per_point(total(d, 'bytes'), n)}</b></td>"
                 f"<td>{per_point(total(d, 'value_bytes'), n)}</td>"
                 f"<td>{per_point(total(d, 'bytes') - total(d, 'value_bytes'), n)}</td>"
                 f"<td>{rel(worst(d, 'point_err'), dev)}</td><td>{100 * total(d, 'exact') / n:.0f}%</td>"
                 f"<td>{total(d, 'bytes') / DAYS / 1000:.2f}</td><td>{per_point(total(o, 'bytes'), n)}</td>"
                 f"<td>{rel(worst(o, 'point_err'), dev)}</td></tr>")
    h.append("""</table><p class='muted'>Max error: the worst |decoded − archived| over the days, in CompDevs. Exact:
the share of points that decode bit-exact. Values and times: the times' bytes are those of the same blocks with the noise
floor off minus the same without times. The <a href="#noise-floor">noise floor</a> section explains where the defaults
and the noise floor off differ.</p>""")

    h.append(f"""<h2 id="accuracy">Accuracy against the scans</h2>
<p>What matters downstream is the trend: straight lines between the decoded points, compared with the scans the historian
saw. SDT alone is the historian's own error; fluxcode's is added on top. In CompDevs, worst over the {days}.</p>
<table><tr><th class='l'>tag</th><th>SDT alone: max</th><th>RMS</th><th>+ fluxcode defaults: max</th><th>RMS</th>
<th>+ fluxcode, noise floor off: max</th><th>RMS</th></tr>""")
    for tag in TAGS:
        r, dev = results[tag], TAG_INFO[tag][2]
        cells = [f"{max(r['sdt_trend_err']) / dev:.3f}", f"{np.mean(r['sdt_trend_rms']) / dev:.3f}"]
        for name in ("defaults", "noise floor off"):
            rows = r["flux"][name]
            cells += [f"{worst(rows, 'trend_err') / dev:.3f}", f"{np.mean([x['trend_rms'] for x in rows]) / dev:.3f}"]
        h.append(f"<tr><td>{tag}</td>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    h.append("</table>")

    h.append(noise_floor_section(results))

    h.append(f"""<h2 id="sweep">Size vs accuracy</h2>
<p>fluxcode's B (<code>max_quantize_bits</code>) from 4 to 16 with the noise floor off, and the defaults (diamond: B = 16
and the noise floor).
The error is the worst |decoded − archived| over the {days}, in CompDevs; exact decodes sit on the bottom line. B sets the
step from each block's range, so the same B gives a different error on each tag: there is no B that means "a tenth of
CompDev".</p>""")
    h.append(plot_sweep(results))
    h.append("<table><tr><th class='l'>tag</th>" + "".join(f"<th>B = {b}</th>" for b in SWEEP_BITS) + "</tr>")
    for tag in TAGS:
        r, dev = results[tag], TAG_INFO[tag][2]
        h.append(f"<tr><td>{tag}</td>" + "".join(
            f"<td>{per_point(total(r['sweep'][b], 'bytes'), r['n'])} B<br>"
            f"<span class='muted'>{rel(worst(r['sweep'][b], 'point_err'), dev)}</span></td>" for b in SWEEP_BITS) + "</tr>")
    h.append("</table><p class='muted'>Each cell: bytes per archived point (values + times), and below it the max "
             "error in CompDevs.</p>")

    h.append(f"""<h2 id="layout">Block layout</h2>
<p>An archive has few points per hour, so how a day is cut into blocks matters more than for 1 kHz data: every block
stores its own anchors, sizes and time reference. Bytes per point at the defaults, all {days}.</p>
<table><tr><th class='l'>tag</th><th>points per day</th>""" + "".join(f"<th>{name}</th>" for name, _ in LAYOUTS) + "</tr>")
    for tag in TAGS:
        r = results[tag]
        sizes = [r["layouts"][name] for name, _ in LAYOUTS]
        best = min(sizes)
        h.append(f"<tr><td>{tag}</td><td>{r['n'] / DAYS:,.0f}</td>" + "".join(
            f"<td{' class=best' if s == best else ''}>{per_point(s, r['n'])}</td>" for s in sizes) + "</tr>")
    h.append("""</table><p class='muted'>Time blocks come from <code>encode_time_blocks</code> (empty hours cost almost
nothing); point blocks from <code>encode_group(block_len=…)</code>. One block per day is 65,535 points at most, so the
densest tag gets two.</p>""")

    h.append("""<h2 id="timestamps">Timestamps</h2>
<p>Scan-aligned times are whole seconds; with <i>source timestamps</i> each point keeps the device's time, a scan delayed
by 0–200 ms at ms resolution (as OPC timestamps often are). The values are the same. Bytes per point for the times only
(fluxcode defaults, the block group minus the same blocks without times), and Gorilla's whole size for comparison.</p>
<table><tr><th class='l'>tag</th><th>fluxcode times: scan-aligned</th><th>source timestamps</th>
<th>Gorilla (all): scan-aligned</th><th>source timestamps</th></tr>""")
    for tag in TAGS:
        r = results[tag]
        o, s = r["flux"]["defaults"], r["flux_src"]
        h.append(f"<tr><td>{tag}</td><td>{per_point(total(o, 'bytes') - total(o, 'value_bytes'), r['n'])}</td>"
                 f"<td>{per_point(total(s, 'bytes') - total(s, 'value_bytes'), r['n'])}</td>"
                 f"<td>{per_point(r['gorilla'], r['n'])}</td><td>{per_point(r['gorilla_src'], r['n'])}</td></tr>")
    h.append("</table>")

    h.append(f"""<h2 id="resampled">Resampled exports</h2>
<p>A historian can also be read at a fixed rate: interpolated (straight lines between the archived points) or held
(each scan takes the last archived value, as a forward fill gives). If that is all you have, fluxcode stores the regular
series (defaults, 1000-scan blocks, its 1 s times included), but the archived points are cheaper and keep the
information: an interpolated series is off the decimal grid, and its rounded ramps leave residuals of a step or so. kB per
day, mean over the {days}; errors are max |decoded − read-back| in CompDevs.</p>
<table><tr><th class='l'>tag</th><th>archived points + times: kB/day</th>""" + "".join(
        f"<th>{name} at 1 s: kB/day</th><th>max error</th>" for name, _ in READ_BACKS) + "</tr>")
    for tag in TAGS:
        r, dev = results[tag], TAG_INFO[tag][2]
        cells = [f"{total(r['flux']['defaults'], 'bytes') / DAYS / 1000:.2f}"]
        for name, _ in READ_BACKS:
            rows = r["resampled"][name]
            cells += [f"{sum(size for size, _ in rows) / DAYS / 1000:.2f}", rel(max(err for _, err in rows), dev)]
        h.append(f"<tr><td>{tag}</td>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    h.append("</table>")

    h.append(f"""<h2 id="no-sdt">SDT or every scan?</h2>
<p>If the scans are still available (at the source, or before the historian compresses), fluxcode can store every one
of them instead. Here every scan of the day is one block group (1000-scan blocks, the 1 s times included): at the defaults, and
at the coarsest B (noise floor off) whose max error is within a budget: one CompDev (SDT's nominal tolerance), or the
error SDT actually made that day (the same max error, a fair comparison). The SDT route is the archive with fluxcode
(defaults), its error that of the trend. kB per day, mean over the {days}; errors are the worst max |decoded −
scan| in CompDevs.</p>
<table><tr><th class='l'>tag</th><th>SDT + fluxcode: kB/day</th><th>max error</th>
<th>every scan, defaults: kB/day</th><th>max error</th><th>within 1 CompDev: kB/day</th><th>B</th><th>max error</th>
<th>within SDT's error: kB/day</th><th>B</th><th>max error</th></tr>""")
    for tag in TAGS:
        r, dev = results[tag], TAG_INFO[tag][2]
        o, dflt = r["flux"]["defaults"], r["scans_default"]
        cells = [f"{total(o, 'bytes') / DAYS / 1000:.2f}", f"{worst(o, 'trend_err') / dev:.2f}",
                 f"{sum(size for size, _ in dflt) / DAYS / 1000:.2f}", rel(max(err for _, err in dflt), dev)]
        for fits in (r["scans_budget"], r["scans_matched"]):
            bits = sorted({b for _, _, b in fits})
            cells += [f"{sum(size for size, _, _ in fits) / DAYS / 1000:.2f}",
                      f"{bits[0]}–{bits[-1]}" if len(bits) > 1 else f"{bits[0]}", rel(max(err for _, err, _ in fits), dev)]
        h.append(f"<tr><td>{tag}</td>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    h.append("</table><p>The same comparison as CompDev changes (ExcDev stays half of it), mean kB per day:</p>")
    h.append(plot_dev_sweep(results))

    toc = [("findings", "Findings", []), ("tags", "The tags", []), ("size", "Size", []), ("accuracy", "Accuracy against the scans", []),
           ("noise-floor", "Noise floor", []), ("sweep", "Size vs accuracy", []), ("layout", "Block layout", []), ("timestamps", "Timestamps", []), ("resampled", "Resampled exports", []),
           ("no-sdt", "SDT or every scan?", [])]
    (OUT / "sdt.html").write_text(shell("fluxcode on swinging-door historian data", "", toc, "\n".join(h)))
    print(f"wrote {OUT / 'sdt.html'}")


def noise_floor_section(results):
    """The noise floor on each tag as CompDev changes: one-scan share, floored blocks, size and error."""
    h = [f"""<h2 id="noise-floor">Noise floor</h2>
<p>fluxcode's noise floor (default 0.25σ) coarsens the step of blocks whose samples look like white measurement noise.
An SDT archive's sparse points look like that without being noise: swinging door keeps exactly the points a straight
line can't predict. So on a block with irregular times, fluxcode measures the noise on consecutive scans only (two
intervals of one scan, the scan being the block's 1st-percentile interval), and scales the floor by the share of
one-scan intervals: none at 50% or less, all of it from 90%. A sparse archive gets no floor and keeps its values; one
that kept most scans is raw sensor data, and gets it. Below, each tag's archive at five CompDevs (ExcDev half of it),
{LAYOUT}, all {DAYS} days, against the same with the noise floor off.</p>
<table><tr><th class='l'>tag</th><th>CompDev</th><th>points per day</th><th>one-scan share</th><th>blocks floored</th>
<th>kB per day</th><th>noise floor off</th><th>max error</th><th>exact</th></tr>"""]
    for tag in TAGS:
        r, dev = results[tag], TAG_INFO[tag][2]
        for scale in DEV_SCALES:
            rows, shares = r["dev_sweep"][scale], r["shares"][scale]
            n = total(rows, "n")
            blocks = total(rows, "blocks")
            floored = total(rows, "floored")
            saving = total(rows, "bytes") / total(rows, "off_bytes") - 1
            share = span(shares, "{:.0%}") if shares else "–"
            h.append(f"<tr><td>{tag if scale == DEV_SCALES[0] else ''}</td><td>{scale * dev:g}</td><td>{n / DAYS:,.0f}</td>"
                     f"<td>{share}</td><td>{f'{floored} of {blocks}' if blocks else '–'}</td>"
                     f"<td>{total(rows, 'bytes') / DAYS / 1000:.2f}</td><td>{total(rows, 'off_bytes') / DAYS / 1000:.2f}"
                     f"{f' ({saving:+.0%})' if floored else ''}</td><td>{rel(worst(rows, 'point_err'), dev)}</td>"
                     f"<td>{100 * total(rows, 'exact') / n:.0f}%</td></tr>")
    h.append("""</table><p class='muted'>One-scan share: the range over the 1 h blocks of at least 256 points (smaller
blocks get no noise floor). Blocks floored: those that decode differently with the noise floor off, of the blocks of at
least 256 points. Max error: the worst |decoded − archived| at the defaults, in CompDevs of the tag's own (the first table),
so that the rows of a tag compare. Exact: the share of points that decode bit-exact at the defaults.</p>""")
    return "\n".join(h)


def span(values, fmt="{:.2f}"):
    """"a–b" for the smallest and largest of values, or one number if they print the same."""
    lo, hi = fmt.format(min(values)), fmt.format(max(values))
    return lo if lo == hi else f"{lo}–{hi}"


def findings(results):
    """The summary at the top of the page, from the numbers."""
    dense = [t for t in TAGS if results[t]["n"] / DAYS >= 1000]
    sparse = [t for t in TAGS if t not in dense]

    def per_point_of(tag, key="bytes", name="defaults"):
        return total(results[tag]["flux"][name], key) / results[tag]["n"]

    own = {t: results[t]["dev_sweep"][1] for t in TAGS}  # each tag's own CompDev
    floored_tags = [t for t in TAGS if total(own[t], "floored")]
    sparse_archives = [t for t in TAGS if t not in floored_tags and results[t]["shares"][1]]
    max_sparse_share = max((max(results[t]["shares"][1]) for t in sparse_archives), default=0)
    floor_share = {t: span(results[t]["shares"][1], "{:.0%}") for t in floored_tags}
    floor_saving = {t: total(own[t], "bytes") / total(own[t], "off_bytes") - 1 for t in floored_tags}
    floor_err = {t: worst(own[t], "point_err") / TAG_INFO[t][2] for t in floored_tags}
    times = [per_point_of(t) - per_point_of(t, "value_bytes") for t in dense]
    source = [(total(results[t]["flux_src"], "bytes") - per_point_of(t) * results[t]["n"]) / results[t]["n"]
              for t in dense]
    zstd = [results[t]["zstd"] / results[t]["n"] / per_point_of(t) for t in dense]
    gorilla = [results[t]["gorilla"] / results[t]["n"] / per_point_of(t) for t in dense]
    every_scan_wins = [t for t in TAGS if sum(size for size, _, _ in results[t]["scans_matched"])
                       < total(results[t]["flux"]["defaults"], "bytes")]
    f32 = results["ph-float32"]["flux"]["defaults"]
    sparse_inflation = [results[t]["layouts"]["1 h blocks"] / results[t]["layouts"]["one block per day"] for t in sparse]
    resample = {name: [sum(size for size, _ in results[t]["resampled"][name]) / total(results[t]["flux"]["defaults"], "bytes")
                       for t in TAGS] for name, _ in READ_BACKS}
    return f"""<h2 id="findings">Findings</h2>
<ul>
<li><b>The noise floor adapts to the archive.</b> On the sparse archives of {", ".join(sparse_archives) or "no tag"}
(at most {max_sparse_share:.0%} of intervals one scan) it is off, and the decimal tags decode exactly as they do with it
off; swinging door's points look like white noise without being noise. Where the historian kept most scans
({", ".join(f"{t}: {floor_share[t]}" for t in floored_tags) or "no tag"} of intervals one scan) they are raw sensor data,
and the floor applies: {", ".join(f"{t} {floor_saving[t]:+.0%}" for t in floored_tags)} in size, at a max error of
{", ".join(f"{floor_err[t]:.2g}" for t in floored_tags)} CompDevs: CompDev is set far below the noise there
(<a href="#noise-floor">noise floor</a>). At the defaults the float32 tag is within
{worst(f32, "point_err") / TAG_INFO["ph-float32"][2]:.1g} CompDevs at {per_point_of("ph-float32"):.2f} bytes per point.</li>
<li><b>Store the archived points, not a resampled export.</b> Read back at the 1 s scan rate, the same archive takes
{span(resample["interpolated"], "{:.1f}")}× the bytes interpolated and {span(resample["held"], "{:.1f}")}× held
(<a href="#resampled">resampled exports</a>), and fluxcode can't recover the points from either.</li>
<li><b>Against lossless coders,</b> on the dense tags fluxcode is {span(zstd, "{:.1f}")}× smaller than zstd of the
columns and {span(gorilla, "{:.1f}")}× smaller than Gorilla. On the sparse tags ({", ".join(sparse)}: about 50 points a
day) a day costs a few hundred bytes whatever the coder, and block overhead decides: hourly blocks cost
{span(sparse_inflation, "{:.1f}")}× what one block per day does (<a href="#layout">layout</a>). Fixed time buckets keep a
store simple, and the inflation only hits data that is tiny anyway.</li>
<li><b>Timestamps are cheap when scan-aligned:</b> {span(times)} bytes per point on the dense tags, since every interval
is a whole number of scans. Device timestamps at ms resolution add {span(source)} bytes per point.</li>
<li><b>fluxcode adds almost nothing to SDT's error:</b> at the defaults, the trend through the decoded points
has the historian's own max error (<a href="#accuracy">accuracy</a>), which is about 2 CompDevs from swinging door
itself.</li>
<li><b>SDT doesn't always pay.</b> At the same max error, fluxcode on every scan is smaller than SDT + fluxcode on
{", ".join(every_scan_wins) or "none of the tags"} (<a href="#no-sdt">SDT or every scan</a>): on noisy tags SDT keeps
most scans anyway, and a regular grid's times cost almost nothing. On smooth or flat tags SDT wins by a wide margin.</li>
</ul>"""


def main():
    results = {tag: measure_tag(tag) for tag in TAGS}
    write_report(results)


if __name__ == "__main__":
    main()
