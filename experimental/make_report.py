# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Run every codec on one-minute block groups of every signal kind; write report/index.html with charts.

    uv run python make_report.py

Every codec encodes whole block groups (60 blocks of 1000 samples) and is charged every byte its decoder
reads (tslab.common.group). Errors are per block, relative to the block's range, summarized as the
median and worst over the kind's blocks. Plots show one second (KIND_INFO) of the first minute.

Ordering rules (shared with rate.html, see tslab/report/__init__.py):
  * codecs: one canonical order (tslab.report.order.ordered): family (classic, constwidth, entropy,
    flux), then nominal bits per sample, then the package's own order;
  * datasets: KINDS order;
  * colors: one hue per family, shades by position within the family; one marker per family.
Every table, legend and chart uses these. Every chart is checked for data at build time.
"""

import html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
import _fresh_numba_cache  # noqa: E402, F401  (before fluxcode: a cache keyed on its sources)
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from fluxcode import Params
from matplotlib.lines import Line2D

from tslab.classic import (
    CLASSIC_CODECS,
    BinaryTree,
    CompandedDpcm,
    Gorilla,
    Quant8,
    QuantDeltaDeflate,
    cubic_expander,
    mulaw_expander,
)
from tslab.common.datasets import BLOCK, KIND_INFO, KINDS, REPORT_MINUTES, noisy_sets, peaks, groups
from tslab.common.group import SharedBackend
from tslab.constwidth import CONSTWIDTH_CODECS, DELTA_TREE_CODECS, Delta1Bfp, Delta1Sqrt8, Delta1Tree
from tslab.constwidth.delta_sqrt import SQ_TABLE
from tslab.entropy.delta_zstd import DELTA_ZSTD_CODECS, DeltaZstd
from tslab.entropy.ratectl_codecs import RATECTL_CODECS, RateCtlCodec
from tslab.flux import adapters
from tslab.flux.adapters import FLUX_CODECS, FluxCodec, noise_score
from tslab.report.order import (
    CONST_WIDTH,
    FAMILIES,
    FAMILY_MARKERS,
    FAMILY_NAMES,
    family,
    lossless_kind,
    module_link,
    ordered,
    palette,
)
from tslab.report.page import clean_outputs, save_fig, shell

CODECS = ordered(CLASSIC_CODECS + RATECTL_CODECS + DELTA_ZSTD_CODECS + CONSTWIDTH_CODECS + DELTA_TREE_CODECS
                 + FLUX_CODECS)
COLORS = palette(CODECS)
NOISE_COLORS = plt.get_cmap("tab10").colors

OUT = Path(__file__).parent / "report"
N = BLOCK
RAW_BYTES = 8 * N  # one block of float64
GROUPS = groups()
METRICS = ("rmse", "mean", "max", "min", "maxabs")


def block_metrics(x, y):
    fs = (x.max() - x.min()) or 1.0
    return {"rmse": np.sqrt(np.mean((y - x) ** 2)) / fs, "mean": (y.mean() - x.mean()) / fs,
            "max": (y.max() - x.max()) / fs, "min": (y.min() - x.min()) / fs, "maxabs": np.abs(y - x).max() / fs}


def inner(codec):
    """The block codec behind a block group adapter (Concat / SharedBackend), else the codec itself."""
    return getattr(codec, "codec", codec)


def check_bounds(codec, x, y, info):
    """The per-block guarantees: exact for gorilla, <= step/2, and exact min (and max) where the
    codec promises it (fluxcode's power-of-two grid is absolute, so its min is within the bound)."""
    c = inner(codec)
    fs, err, tol = x.max() - x.min(), np.abs(y - x).max(), 1 + 1e-9
    if isinstance(c, Gorilla):
        assert np.array_equal(y, x), "gorilla-xor must be lossless"
    elif isinstance(c, (Quant8, QuantDeltaDeflate)):
        bits = 8 if isinstance(c, Quant8) else c.bits
        assert err <= fs / ((1 << bits) - 1) / 2 * tol and y.min() == x.min(), c.name
    elif isinstance(c, DeltaZstd):
        assert err <= fs / ((1 << info["bits"]) - 1) / 2 * tol, c.name
        assert y.min() == x.min() and y.max() == x.max(), c.name
    elif isinstance(c, RateCtlCodec):
        assert err <= info["step"] * fs / 2 * tol + 1e-12 * fs, c.name
    elif isinstance(c, FluxCodec):
        if info["decimal"]:  # values within a quarter power-of-two step (< 10^p / 4) of the grid
            assert err <= 10.0 ** info["param"] / 4 * tol, c.name
        else:
            # The grid is absolute (snapped): the min decodes to the grid point nearest it, within half a step
            assert err <= 2.0 ** info["param"] / 2 * tol, c.name


def summarize(r):
    """Median and worst over blocks; worst of a signed error is the one farthest from zero."""
    r["bps"] = 8 * r["bytes"] / (r["blocks"] * N)
    r["bytes_block"] = r["bytes"] / r["blocks"]
    for m in METRICS:
        v = np.array(r.pop(m))
        r[m] = float(np.median(v))
        r[m + "_worst"] = float(v.max() if m in ("rmse", "maxabs") else v[np.argmax(np.abs(v))])


def run_codec(codec, kind_groups, plot_seed=None, plot_block=None):
    """Encode and decode every block group; returns the per-block summary (plus the plotted block if asked)."""
    r = {"bytes": 0, "blocks": 0, "infos": [], **{m: [] for m in METRICS}}
    for seed, X in kind_groups:
        data, infos = codec.encode_group(X)
        Y = codec.decode_group(data, *X.shape)  # only the bytes and the shape
        assert Y.shape == X.shape
        r["bytes"] += len(data)
        r["blocks"] += len(X)
        r["infos"].extend(infos)
        for x, y, info in zip(X, Y, infos):
            check_bounds(codec, x, y, info)
            for m, v in block_metrics(x, y).items():
                r[m].append(v)
        if seed == plot_seed:
            x, y = X[plot_block], Y[plot_block]
            r["y"], r["info"], r["plot_rmse"] = y, infos[plot_block], block_metrics(x, y)["rmse"]
    summarize(r)
    return r


def by_kind():
    """kind -> [(seed, X)] for the report's block groups."""
    out = {k: [] for k in KINDS}
    for kind, seed, X in GROUPS:
        out[kind].append((seed, X))
    return out


def plotted(kind):
    """(seed, block index, zoom window, x, true peak offsets in the block or None) for a kind."""
    seed, X = by_kind()[kind][0]
    _, b, window = KIND_INFO[kind]
    pk = peaks(kind, seed)
    if pk is not None:
        pk = [int(p) - b * N for p in pk if b * N <= p < (b + 1) * N]
    return seed, b, window, X[b], pk


def run():
    results = {}
    for kind, kind_groups in by_kind().items():
        seed, b, _, x, pk = plotted(kind)
        results[kind] = {}
        for codec in CODECS:
            r = run_codec(codec, kind_groups, seed, b)
            if pk:
                # Location of the reconstructed local max within ±20 samples of each true peak.
                r["peak_shift"] = [int(np.argmax(r["y"][max(0, p - 20):p + 21]) - np.argmax(x[max(0, p - 20):p + 21]))
                                   for p in pk]
            results[kind][codec.name] = r
        print(kind, flush=True)
    return results



# Codecs overlaid in the per-dataset plots: a few per family, the rest are in the tables.
PLOT_CODECS = ("quant8", "dpcm4-linear-deflate", "bintree-center", "swingdoor-2%", "dct-top64", "ratectl-flac-4b",
               "delta0123-zstd-8", "cw4-delta1-bfp-e4", "fluxcode-16-f0.25")


def plot_kind(kind, x, res, window, b, seed):
    """Original second, then (zoomed to the window if there is one) the decoded traces and their errors."""
    s, e = window or (0, N)
    t = np.arange(s, e)
    fs = (x.max() - x.min()) or 1.0
    rows = 3 if window else 2
    fig, axes = plt.subplots(rows, 1, figsize=(11, 2.7 * rows + 0.8), layout="constrained")
    ax = list(axes)
    if window:
        top = ax.pop(0)
        top.plot(np.arange(N), x, color="0.35", lw=1.0)
        top.axvspan(s, e, color="tab:red", alpha=0.15, lw=0)
        top.set(title=f"Original, second {b} of seed {seed} (shaded: zoom window)", ylabel="value")
    sig, err = ax
    sig.plot(t, x[s:e], color="0.7", lw=3.0, label="original", zorder=1)
    for name in PLOT_CODECS:
        y = res[name]["y"]
        sig.plot(t, y[s:e], color=COLORS[name], lw=1.0, label=name, zorder=2)
        err.plot(t, 100 * (y[s:e] - x[s:e]) / fs, color=COLORS[name], lw=0.9, label=name)
    sig.set(title="Decoded (representative codecs)" + (f", samples {s}–{e}" if window else ""), ylabel="value")
    err.set(title="Error, decoded − original", ylabel="% of block range", xlabel="sample within the second (1 ms each)")
    for a in axes:
        a.grid(alpha=0.3, lw=0.4)
    handles, labels = sig.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, fontsize=8, frameon=False)
    fig.suptitle(f"{kind}: {KIND_INFO[kind][0]}", fontsize=11)
    return save_fig(fig, OUT, kind, f"{kind}: original, decoded traces and errors")


def group_stats(codec, kind_groups):
    """(bits/sample over whole block groups, median block RMSE as a fraction of range, worst block max |err|)."""
    r = run_codec(codec, kind_groups)
    return r["bps"], r["rmse"], r["maxabs_worst"]


def notes(c, r, base):
    """The notes column: a summary of the per-block choices over the kind's blocks."""
    infos, cb = r["infos"], inner(c)
    if isinstance(cb, BinaryTree) and cb.leaf == "center":
        return "codes 00/01/10/11 per block: " + "/".join(f"{np.mean([i['codes'][k] for i in infos]):.0f}" for k in (0, 1, 2, 3))
    if "knots" in infos[0]:
        return f"{np.mean([len(i['knots']) for i in infos]):.0f} knots per block"
    if isinstance(cb, DeltaZstd):
        orders = Counter(i["order"] for i in infos)
        capped = sum(i["bits"] < cb.bits for i in infos)
        return ("orders " + ", ".join(f"{k}: {orders[k]}" for k in sorted(orders))
                + (f"; B lowered by the cap on {capped} blocks (to {min(i['bits'] for i in infos)})" if capped else ""))
    if isinstance(cb, Delta1Sqrt8):
        fine = np.mean([np.mean(np.abs(SQ_TABLE[i["codes"]]) < (1 << 25)) for i in infos])
        return f"{100 * fine:.0f}% of samples use a finer-than-7-bit delta"
    if isinstance(cb, Delta1Bfp):
        return f"frame exponents: mean {np.mean([i['exps'].mean() for i in infos]):.1f}, max {max(i['exps'].max() for i in infos)}"
    if isinstance(cb, FluxCodec):
        orders = Counter(i["order"] for i in infos)
        dec = [i["param"] for i in infos if i["decimal"]]
        pw = [i["param"] for i in infos if not i["decimal"]]
        note = "orders " + ", ".join(f"{k}: {orders[k]}" for k in sorted(orders)) + "; "
        note += ", ".join(([f"decimal grid 10^{min(dec)}" + (f"..10^{max(dec)}" if max(dec) > min(dec) else "")] if dec else [])
                          + ([f"step 2^{min(pw)}" + (f"..2^{max(pw)}" if max(pw) > min(pw) else "")] if pw else []))
        if base is not None:
            changed = sum(i["param"] != j["param"] for i, j in zip(infos, base["infos"]))
            note += f" (noise floor applied on {changed} blocks)" if changed else ""
        return note
    if isinstance(cb, Delta1Tree):
        return (f"halving bits set {100 * np.mean([i['halved'] for i in infos]):.0f}%, leaves at code limit "
                f"{100 * np.mean([i['clipped'] for i in infos]):.0f}%")
    if isinstance(cb, Gorilla):
        return "per block: " + ", ".join(f"{k.replace('_', ' ')} {np.mean([i[k] for i in infos]):.0f}"
                                         for k in ("same", "reuse_window", "new_window"))
    return ""


def med_worst(r, m, fmt):
    return f"{fmt(r[m])} / {fmt(r[m + '_worst'])}"


def pct(v):
    return f"{100 * v:+.3g}%"



def rmse_class(rmse):
    """Cell shading of the summary matrix by median RMSE (% of range)."""
    p = 100 * rmse
    return "" if p < 0.2 and p >= 0.03 else " r1" if p < 0.03 else " r2" if p < 1 else " r3" if p < 5 else " r4"


def fam_style(c):
    r, g, b, _ = COLORS[c.name]
    return f" class='fam' style='--c:rgb({int(255 * r)},{int(255 * g)},{int(255 * b)})'"


def legend_table():
    h = ["<table><tr><th class='l'>codec</th><th class='l'>family</th><th class='l'>lossless?</th>"
         "<th class='l'>constant width?</th><th class='l'>description</th><th class='l'>module</th></tr>"]
    for c in CODECS:
        fam = family(c)
        href, mod = module_link(c)
        cw = "yes" if fam == "constwidth" or c.name in CONST_WIDTH else "no"
        h.append(f"<tr{fam_style(c)}><td><code>{c.name}</code></td><td class='l'>{FAMILY_NAMES[fam]}</td>"
                 f"<td class='l'>{lossless_kind(c)}</td><td class='l'>{cw}</td><td class='l'>{html.escape(c.label)}</td>"
                 f"<td class='l'><a href='{href}'><code>{mod}</code></a></td></tr>")
    h.append("</table>")
    return "".join(h)


def summary_matrix(results):
    h = ["<table><tr><th>codec</th>" + "".join(f"<th><a href='#{k}'>{k}</a></th>" for k in KINDS) + "</tr>"]
    for c in CODECS:
        cells = []
        for k in KINDS:
            r = results[k][c.name]
            cells.append(f"<td class='{rmse_class(r['rmse']).strip()}'>{r['bps']:.2f} b/s<br>"
                         f"<span class='muted'>{100 * r['rmse']:.3g}%</span></td>")
        h.append(f"<tr{fam_style(c)}><td><code>{c.name}</code></td>{''.join(cells)}</tr>")
    h.append("</table>")
    return "".join(h)


def family_handles():
    return [Line2D([], [], marker=FAMILY_MARKERS[f], ls="", color=COLORS[[c for c in CODECS if family(c) == f][len(
        [c for c in CODECS if family(c) == f]) // 2].name], markeredgecolor="k", markeredgewidth=0.4, ms=7,
        label=FAMILY_NAMES[f]) for f in FAMILIES]


EXACT_PCT = 1e-7  # RMSE at or below this (% of range) counts as lossless: float round-off, no place on a log axis
Y_TOP = 15  # RMSE axis top (% of range) on the charts that share the fixed axes: the 10% tick plus a margin
X_MAX = 16  # bits/sample axis limit; codecs beyond it are marked at the right edge
NOT_PLOTTED = {"gorilla-xor"}  # lossless at ~65 bits/sample: in the tables, but too far out for the charts
ZERO_PCT = EXACT_PCT  # where a lossless RMSE (0) is drawn on the log axes, labelled 0*: the same value as the threshold
ZERO_NOTE = "0*: lossless (RMSE ≤ $10^{-7}$%)"


def shown(rmse_pct):
    """RMSE (% of range) as drawn on a log axis: lossless values at ZERO_PCT."""
    return ZERO_PCT if rmse_pct <= EXACT_PCT else rmse_pct


def pct_label(v):
    """Tick label for a log axis in % units: 0* for the lossless row, 1%, 10%, then 10^-d %."""
    if v == ZERO_PCT:
        return "0*"
    d = round(np.log10(v))
    return f"{10 ** d}%" if d >= 0 else f"$10^{{{d}}}$%"


def rmse_axis(ax, fontsize, zero=False, top=None):
    """Log RMSE axis in % of range: decade ticks labelled with %, and with zero, a 0* row at ZERO_PCT in a shaded
    strip below the continuous scale (decades from one above EXACT_PCT up), so the row reads as separate from it."""
    bottom, top = ax.get_ylim()[0], top or ax.get_ylim()[1]
    if zero:
        bottom = ZERO_PCT / 3
        ax.set_ylim(bottom, top)
        ax.axhspan(bottom, 1.5 * ZERO_PCT, color="0.5", alpha=0.12, lw=0, zorder=0)
    first = round(np.log10(EXACT_PCT)) + 1 if zero else int(np.ceil(np.log10(bottom)))
    decades = [10.0 ** d for d in range(first, int(np.floor(np.log10(top))) + 1)]
    major = ([ZERO_PCT] if zero else []) + decades
    ax.set_yticks(major, [pct_label(v) for v in major], fontsize=fontsize)
    minor = [m * v for v in [decades[0] / 10] + decades for m in range(2, 10) if max(bottom, EXACT_PCT) < m * v < top]
    ax.set_yticks(minor, [""] * len(minor), minor=True)


# The fluxcode curves, in the summary scatter and the per-dataset panels: the only lines there, so fluxcode
# stands out from the markers. One per effort (fastest, default, smallest), sweeping B = 4..16 with the noise floor and
# the bits target off (both only steer the quantization the sweep sets directly) and the rest at the defaults. Colors run light to dark with effort, the default drawn heavier.
DEFAULT_EFFORT = Params().effort
CURVE_EFFORTS = (DEFAULT_EFFORT, 1, 9)  # legend order: the default first
_PURPLES = matplotlib.colormaps["Purples"]
CURVE_STYLES = {e: dict(color=_PURPLES(0.5 + 0.5 * (e - 1) / 8), lw=2.4 if e == DEFAULT_EFFORT else 1.3,
                        label=f"effort {e}" + {1: " (fastest)", 9: " (smallest)", DEFAULT_EFFORT: " (default)"}[e])
                for e in CURVE_EFFORTS}
FLUX_CURVES = {e: [FluxCodec(params=Params(max_quantize_bits=b, min_quantize_bits=min(b, 6), noise_floor_sigma=0,
                                        target_bits_per_sample=None, effort=e))
                   for b in range(4, 17)] for e in CURVE_EFFORTS}
DEFAULT_CODEC = "fluxcode-16-f0.25"  # Params() unchanged: the diamond
DEFAULT_LABEL = f"all defaults (B = 16, noise floor 0.25, effort {DEFAULT_EFFORT})"


def curve_stats():
    """Effort -> kind -> [(bits/sample, median RMSE in % of range)] for B = 4..16."""
    kind_rows = by_kind()
    return {e: {kind: [(bps, 100 * rmse) for bps, rmse, _ in (group_stats(c, kind_rows[kind]) for c in curve)]
                for kind in KINDS}
            for e, curve in FLUX_CURVES.items()}


def curve_handles():
    d = Line2D([], [], marker="D", ls="", color=CURVE_STYLES[DEFAULT_EFFORT]["color"], markeredgecolor="k",
               markeredgewidth=0.8, ms=8, label=DEFAULT_LABEL)
    return [d] + [Line2D([], [], marker=".", **st) for st in CURVE_STYLES.values()]


def plot_curve(ax, points, **kw):
    """One effort's curve: points = [(bits/sample, RMSE %)] for B = 4..16, in that effort's style."""
    e = kw.pop("effort")
    ax.plot(*zip(*[(b, shown(r)) for b, r in points]), color=CURVE_STYLES[e]["color"], lw=CURVE_STYLES[e]["lw"],
            marker=".", ms=4, zorder=2 + (e == DEFAULT_EFFORT), **kw)


# Fluxcode variants other than the defaults (other B, noise floors): the lines already sweep B, and the other noise floors
# only move sideways on the median summary, so they are left to the tables.
SCATTER_HIDDEN = {c.name for c in FLUX_CODECS if c.name != DEFAULT_CODEC}


def plot_scatter(results, curves):
    """Every codec: median bits/sample vs median RMSE over the 12 kinds; lossless ones on the 0* row. The
    fluxcode curves are built the same way: per B, the medians over the kinds."""
    x = {c.name: float(np.median([results[k][c.name]["bps"] for k in KINDS])) for c in CODECS}
    y = {c.name: 100 * float(np.median([results[k][c.name]["rmse"] for k in KINDS])) for c in CODECS}
    fig, ax = plt.subplots(figsize=(11, 6.6), layout="constrained")
    ax.set_yscale("log")
    ax.set_xlim(-0.2, X_MAX + 0.2)
    ax.set_xticks(range(0, X_MAX + 1, 2))
    med_rmse = []
    for e, per_kind in curves.items():
        by_b = list(zip(*per_kind.values()))  # per B: one (bps, rmse) per kind
        med = [(float(np.median([p[0] for p in ps])), float(np.median([p[1] for p in ps]))) for ps in by_b]
        med_rmse += [r for _, r in med]
        plot_curve(ax, med, effort=e)
    pts = []
    for c in (c for c in CODECS if c.name not in SCATTER_HIDDEN | NOT_PLOTTED):
        px, py = min(x[c.name], X_MAX - 0.2), shown(y[c.name])
        if c.name == DEFAULT_CODEC:  # the diamond: labelled in the legend, not on the chart
            ax.scatter(px, py, marker="D", s=70, color=CURVE_STYLES[DEFAULT_EFFORT]["color"], edgecolor="k",
                       linewidth=0.9, zorder=4)
            continue
        ax.scatter(px, py, marker=FAMILY_MARKERS[family(c)], s=48, color=COLORS[c.name], edgecolor="k", linewidth=0.4,
                   zorder=3)
        pts.append((c.name + (f" ({x[c.name]:.0f} b/s, off scale)" if x[c.name] > X_MAX else ""), px, py))
    zero = (any(y[c.name] <= EXACT_PCT for c in CODECS if c.name not in NOT_PLOTTED)
            or any(r <= EXACT_PCT for r in med_rmse))
    rmse_axis(ax, 9, True, Y_TOP)  # fixed axes, the 0* row always there
    fig.canvas.draw()  # label a point only where it does not overlap an earlier label
    taken = []
    for name, px, py in pts:
        u, v = ax.transData.transform((px, py))
        box = (u + 5, v + 2, u + 5 + 4.2 * len(name), v + 12)
        if u + box[2] - box[0] > ax.bbox.x1 + 10:  # near the right edge: to the left
            box = (u - 5 - 4.2 * len(name), v + 2, u - 5, v + 12)
        if any(box[0] < t[2] and t[0] < box[2] and box[1] < t[3] and t[1] < box[3] for t in taken):
            continue
        taken.append(box)
        ax.annotate(name, (px, py), textcoords="offset points", xytext=(-5, 2) if box[2] < u else (5, 2),
                    ha="right" if box[2] < u else "left", fontsize=6.5)
    if True:
        ax.text(0.01, 0.01, ZERO_NOTE, transform=ax.transAxes, ha="left", va="bottom", fontsize=8, color="0.4")
    ax.set(xlabel="bits / sample (median over the 12 kinds)", ylabel="median RMSE (% of block range, log)",
           title="Size vs error, all codecs (unlabeled points: see the summary matrix)")
    ax.grid(alpha=0.3, lw=0.4, which="both")
    ax.legend(handles=[h for h in family_handles() if h.get_label() != "fluxcode"] + curve_handles(),
              loc="upper right", fontsize=9)
    return save_fig(fig, OUT, "scatter", "Bits per sample vs RMSE, all codecs")


def plot_rd(results, curves):
    """Per kind: every codec by family, the fluxcode curves; lossless points on a 0* row where there are any."""
    cols = 3
    rows = -(-len(KINDS) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3.6 * rows), layout="constrained")
    for ax, kind in zip(axes.flat, KINDS):
        ax.set_yscale("log")
        lossless = False
        for e, per_kind in curves.items():
            plot_curve(ax, per_kind[kind], effort=e)
            lossless |= any(r <= EXACT_PCT for _, r in per_kind[kind])
        for c in (c for c in CODECS if c.name not in SCATTER_HIDDEN | NOT_PLOTTED):
            r = results[kind][c.name]
            lossless |= 100 * r["rmse"] <= EXACT_PCT
            is_default = c.name == DEFAULT_CODEC
            ax.scatter(r["bps"], shown(100 * r["rmse"]), marker="D" if is_default else FAMILY_MARKERS[family(c)],
                       s=60 if is_default else 26, color=CURVE_STYLES[DEFAULT_EFFORT]["color"] if is_default else COLORS[c.name],
                       edgecolor="k", linewidth=0.8 if is_default else 0.3, zorder=4 if is_default else 3)
        ax.axvline(4, color="0.6", ls=":", lw=1, gid="ref")
        ax.set_xlim(-0.2, X_MAX + 0.2)
        ax.set_xticks(range(0, X_MAX + 1, 2))
        rmse_axis(ax, 7, True, Y_TOP)
        ax.set_title(kind, fontsize=10)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3, which="both", lw=0.4)
    for ax in axes[:, 0]:
        ax.set_ylabel("median RMSE (% FS, log)", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("bits / sample", fontsize=8)
    handles = [h for h in family_handles() if h.get_label() != "fluxcode"] + curve_handles() + [Line2D([], [], ls="", label=ZERO_NOTE)]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=9, frameon=False)
    return save_fig(fig, OUT, "rd", "Bits per sample vs RMSE per signal kind")


def write_report(results):
    clean_outputs(OUT, ("*.png", "*.svg", "index.html"), keep_prefix=("rate", "time", "historian-"))  # the other scripts' pages and charts
    body = []
    body.append("<h2 id='legend'>Codec legend</h2><p>One row per codec, in the order used by every table and chart: family "
                "(classic, constant width, entropy, fluxcode), then nominal bits per sample. The colored bar is the family's "
                "chart color. “Lossless?”: bit-exact on any input; fluxcode is bit-exact when the data sits on a decimal grid. "
                "“Constant width”: the same number of bytes for every block.</p>" + legend_table())

    body.append("<h2 id='matrix'>Summary matrix</h2><p>Codecs × datasets: bits per sample over whole one-minute block groups "
                "(everything the decoder reads), and the median block RMSE (% of block range). Shading is by RMSE: green "
                "&lt; 0.03%, none 0.03–0.2%, yellow 0.2–1%, orange 1–5%, red ≥ 5%. Sizes differ, so read both numbers.</p>"
                + summary_matrix(results))

    curves = curve_stats()
    default, fast, small = (curves[e] for e in CURVE_EFFORTS)
    total = lambda curve: sum(curve[kind][-1][0] for kind in KINDS)  # bits/sample at B = 16, summed over the datasets
    effort_gap = (f"At B = 16, summed over the datasets, effort 1 is {100 * (total(fast) / total(default) - 1):+.1f}% and "
                  f"effort 9 {100 * (total(small) / total(default) - 1):+.1f}% against the default effort; the encode "
                  "times behind each effort are in docs/TUNING.md (Effort). ")
    scatter = plot_scatter(results, curves)
    rd = plot_rd(results, curves)
    body.append("<h2 id='scatter'>Size vs error</h2><p>Every codec except the fluxcode variants, one marker per family "
                "(shade by position in the family). Bits/sample and RMSE are medians over the 12 datasets, so this is a "
                "summary; the per-dataset panels below show the spread. gorilla-xor (lossless, about 65 bits/sample) is left "
                "out of the charts; it's in the tables. fluxcode is the purple lines: B = 4..16 with the noise floor and the bits target off, since both only steer the quantization that B sets "
                f"directly; one line per <code>effort</code>, darker for more effort, the default effort {DEFAULT_EFFORT} heavy. "
                f"The diamond is the all-defaults codec (<code>{DEFAULT_CODEC}</code>: B = 16, noise floor 0.25): the noise detector "
                "coarsens the step on white-noise blocks, so it sits left of the B = 16 end of the line. "
                "The three lines "
                "nearly coincide: effort trades encode time for a few percent of size. " + effort_gap +
                "The other B and noise-floor variants of fluxcode are in the matrix and the per-dataset tables.</p>" + scatter
                + f"<h3 id='scatter-kinds'>Per dataset</h3><p>Each panel is one dataset: bits/sample over its {REPORT_MINUTES} "
                "one-minute block groups, RMSE the median over their blocks. The lines are fluxcode at B = 4..16, noise floor and bits target off, "
                f"at efforts {DEFAULT_EFFORT} (heavy, the default), 1 and 9; the diamond is all defaults. A marker below "
                "the heavy line beats fluxcode at equal size. "
                "Lossless points (RMSE ≤ 10<sup>−7</sup>%) sit on the 0* row. The dotted line is 4 bits/sample. Errors are against the "
                "input, which for noisy-sine counts the noise as signal.</p>" + rd)

    body.append("<h2 id='datasets'>Per-dataset results</h2>")
    for kind in KINDS:
        res = results[kind]
        seed, b, window, x, pk = plotted(kind)
        body.append(f"<h3 id='{kind}'>{kind}</h3><p>{html.escape(KIND_INFO[kind][0])}. "
                    f"{REPORT_MINUTES} one-minute signals; the plots show second {b} of the first (seed {seed}), "
                    f"full scale {x.max() - x.min():.4g}. Rows are in legend order.</p>")
        body.append("<table><tr><th>codec</th><th>bytes/block</th><th>ratio</th><th>RMSE<br>median / worst</th>"
                    "<th>mean err<br>median / worst</th><th>max err<br>median / worst</th><th>min err<br>median / worst</th>"
                    "<th>max |err|<br>median / worst</th>"
                    + ("<th>peak shift (samples, plotted second)</th>" if pk else "") + "<th class='l'>notes</th></tr>")
        for c in CODECS:
            r = res[c.name]
            base = res.get(c.name.rsplit("-f", 1)[0]) if isinstance(c, FluxCodec) and "-f" in c.name else None
            body.append(f"<tr{fam_style(c)}><td><code>{c.name}</code></td><td>{r['bytes_block']:.0f}</td><td>{RAW_BYTES / r['bytes_block']:.1f}×</td>"
                        f"<td>{med_worst(r, 'rmse', lambda v: f'{100 * v:.3g}%')}</td><td>{med_worst(r, 'mean', pct)}</td>"
                        f"<td>{med_worst(r, 'max', pct)}</td><td>{med_worst(r, 'min', pct)}</td>"
                        f"<td>{med_worst(r, 'maxabs', lambda v: f'{100 * v:.3g}%')}</td>"
                        + (f"<td>{', '.join(f'{s:+d}' for s in r['peak_shift'])}</td>" if pk else "")
                        + f"<td class='l muted'>{notes(c, r, base)}</td></tr>")
        body.append("</table>")
        body.append(plot_kind(kind, x, res, window, b, seed))

    body.append("<h2 id='sweeps'>Parameter sweeps</h2>")
    body.append(fluxcode_section())
    body.append(noise_section())
    body.append(dpcm_sweep())
    body.append(DECISIONS)

    intro = f"""<p><b>See also:</b> <a href="rate.html">rate-controlled quantize → predict → entropy code experiment</a>; <a href="time.html">fluxcode timestamps: the time axis</a>; <a href="sdt.html">fluxcode on swinging-door historian data</a>.</p>
<p>A comparison of {len(CODECS)} time-series codecs on synthetic 1 kHz signals. <b>Data:</b> {REPORT_MINUTES} one-minute
signals of each of the {len(KINDS)} kinds ({REPORT_MINUTES * 60 * len(KINDS):,} blocks of 1000 samples, fixed seeds). Every codec
encodes whole one-minute block groups (60 blocks) and is charged every byte needed to decode the block group from those bytes alone;
anything that depends on a range (quantizer grids, % of range thresholds) works per 1-second block. Raw is 1000 × float64 =
8000 bytes per block.</p>
<p><b>Metrics:</b> all errors are signed, per block, and a percentage of that block's range (max − min); tables give the
median and the worst over the dataset's blocks. “Max err” is decoded max − true max (negative = peak clipped). Bits/sample
counts whole block groups.</p>
<p><b>Regenerate:</b> <code>cd experimental &amp;&amp; uv run python make_report.py</code> (writes this page and its SVGs
into <code>report/</code>; <code>make_rate_report.py</code> writes rate.html, <code>make_time_report.py</code> time.html, <code>make_sdt_report.py</code> sdt.html).</p>"""
    toc = [("legend", "Codec legend", []), ("matrix", "Summary matrix", []),
           ("scatter", "Size vs error", [("scatter-kinds", "per dataset")]),
           ("datasets", "Per-dataset results", [(k, k) for k in KINDS]),
           ("sweeps", "Parameter sweeps", [("fluxcode-sweep", "fluxcode B sweep"), ("noise-floor", "Noise floor sweep"),
                                            ("dpcm-sweep", "DPCM sweep")]),
           ("notes", "Design notes", [])]
    (OUT / "index.html").write_text(shell("Time series codecs", intro, toc, "\n".join(body)))


F_SWEEP_RD = (0.0, 0.01, 0.03, 0.1, 0.25, 0.5, 1.0)

FLUX_COLORS = {"order 1, power-of-two only": ("#ff7f0e", "--", "o"), "order 1, decimal detect": ("#d62728", "-", "o"),
               "order 0-3, power-of-two only": ("#1f77b4", "--", "s"), "order 0-3, decimal detect": ("#2ca02c", "-", "s")}


def fluxcode_section():
    """Four versions of fluxcode, B = 9..16, continuous and discretized data."""
    res, ideal = adapters.sweep()
    labels = [l for l, _ in adapters.VERSIONS]
    bits = list(adapters.SWEEP_BITS)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for l in labels:
        color, ls, mk = FLUX_COLORS[l]
        rc = [res[l, B][0] for B in bits]
        rd = [res[l, B][1] for B in bits]
        axes[0].plot([r["bps"] for r in rc], [r["rmse_med"] for r in rc], ls, color=color, marker=mk, ms=4, label=l)
        axes[1].plot(bits, [r["bps"] for r in rd], ls, color=color, marker=mk, ms=4, label=l)
        axes[2].plot(bits, [100 * r["exact"] for r in rd], ls, color=color, marker=mk, ms=4, label=l)
    axes[1].axhline(np.mean(list(ideal.values())), color="0.5", ls=":", lw=1, label="lossless at the true quantum (mean)")
    axes[0].set(xlabel="bits / sample", ylabel="median RMSE (% of range, log)", yscale="log",
                title="Continuous signals (7,200 blocks): B = 9..16")
    axes[1].set(xlabel="B", ylabel="bits / sample", title="Discretized signals (5,400 blocks): size")
    axes[2].set(xlabel="B", ylabel="% of samples decoded bit-exact", title="Discretized signals: exactness", ylim=(0, 100))
    rmse_axis(axes[0], 10)
    for ax in axes:
        ax.grid(alpha=0.3, lw=0.4, which="both")
        ax.legend(fontsize=8)
    fig.tight_layout()
    img = save_fig(fig, OUT, 'fluxcode', 'fluxcode B sweep')

    h = ["<h3 id='fluxcode-sweep'>fluxcode B sweep: power-of-two only vs decimal detection, order 1 vs orders 0–3, B = 9..16</h3>"
         "<p>The <code>fluxcode</code> package (at the repo root, format in <code>docs/SPEC.md</code>): "
         "quantize on a power-of-two step (min exact, step = 2<sup>e</sup> with the range in "
         "[2<sup>B−1</sup>, 2<sup>B</sup>) steps), residuals mod 2<sup>16</sup> as int16 → zigzag → <b>16 bit planes</b> "
         "XX B is <code>max_quantize_bits</code>; the noise floor is off "
         "here and has its own sweep below. "
         "<b>Decimal detect</b>: if every value is within a quarter power-of-two step of a multiple of 10<sup>p</sup> "
         "(coarsest p whose step is still coarser than the power-of-two step), quantize on that grid instead and "
         "reconstruct K / 10<sup>−p</sup>, bit-exact for decimal data; values that were float32 are rounded back to float32. "
         "Otherwise (ADC counts, other steps) it falls back to the power-of-two grid. <b>Order 1</b> fixes the predictor; "
         "<b>orders 0–3</b> picks the lowest-variance one per block, from its first 250 samples. Continuous data: the 7,200 "
         "blocks of 12 signal types used by the benchmarks. Discretized: 9 signals rounded to a quantum (0.001, 0.01, "
         "0.1, 2<sup>−4</sup>, a 12-bit ADC step, decimals via float32), 600 blocks each. Sizes are whole block groups (header, "
         "per-block anchors, zstd frame). Timings: one thread, µs per 1000-sample block, including zstd and the Python API.</p>",
         img]
    h.append("<table><tr><th>version</th><th>B</th><th>continuous bits/sample</th><th>continuous median RMSE</th>"
             "<th>continuous worst max err</th><th>discretized bits/sample</th><th>discretized median RMSE</th>"
             "<th>discretized worst max err</th><th>bit-exact samples</th><th>encode µs (cont / disc)</th><th>decode µs</th></tr>")
    for l in labels:
        for B in bits:
            rc, rd = res[l, B]
            h.append(f"<tr><td>{l}</td><td>{B}</td><td>{rc['bps']:.2f}</td><td>{rc['rmse_med']:.3g}%</td>"
                     f"<td>{rc['maxerr_max']:.3g}%</td><td>{rd['bps']:.2f}</td><td>{rd['rmse_med']:.3g}%</td>"
                     f"<td>{rd['maxerr_max']:.3g}%</td><td>{100 * rd['exact']:.1f}%</td>"
                     f"<td>{rc['enc']:.1f} / {rd['enc']:.1f}</td><td>{rc['dec']:.1f}</td></tr>")
    h.append("</table>")
    h.append("<h4>Discretized signals at B = 16 (bits/sample)</h4><table><tr><th>signal</th>"
             "<th>lossless at the true quantum</th>" + "".join(f"<th>{l}</th>" for l in labels) + "</tr>")
    for name, _, _ in adapters.DISCRETE:
        cells = [res[l, 16][1]["per_signal"][name] for l in labels]
        best = min(cells)
        h.append(f"<tr><td>{name}</td><td>{ideal[name]:.2f}</td>"
                 + "".join(f"<td{' class=best' if c == best else ''}>{c:.2f}</td>" for c in cells) + "</tr>")
    h.append("</table>")
    h.append("<p class='muted'>“Lossless at the true quantum” is the same pipeline told the quantum in advance. "
             "Decimal detection reaches it on every decimal signal; the ADC and 2<sup>−4</sup> steps aren't decimal "
             "and fall back to the power-of-two grid. random-walk q0.001 only partly: blocks whose range exceeds "
             "65,535 × 0.001 can't be exact in 16 bits. Byte and nibble plane layouts, which fluxcode dropped for "
             "bit planes, are compared in REPORT.md §11.</p>")
    return "".join(h)


NOISE_F = (0.0, 0.01, 0.03, 0.1, 0.25, 0.5, 1.0)


def noise_section():
    """fluxcode-16 noise-floor sweep on the noisy signals, scored against the input and the clean signal."""
    sets = noisy_sets()
    res = {name: [noise_score(FluxCodec(params=Params(noise_floor_sigma=f)), mins)
                  for f in NOISE_F] for name, mins in sets.items()}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for (name, rows), color in zip(res.items(), NOISE_COLORS):
        lossy = [r for r in rows if r["in"] > 0]  # lossless points (decimal grid) have no place on the log axis
        axes[0].plot([r["bps"] for r in lossy], [r["in"] for r in lossy], "-o", color=color, ms=4, label=name)
        axes[1].plot([r["bps"] for r in rows], [r["clean"] for r in rows], "-o", color=color, ms=4, label=name)
    for bx, ey, f in zip([r["bps"] for r in res["noisy-sine"]], [r["clean"] for r in res["noisy-sine"]], NOISE_F):
        axes[1].annotate(f"f={f:g}", (bx, ey), textcoords="offset points", xytext=(3, 4), fontsize=7)
    axes[0].set(xlabel="bits / sample", ylabel="RMS error vs input (σ of the noise, log)", yscale="log",
                title="Added error (what the noise floor changes)")
    axes[1].set(xlabel="bits / sample", ylabel="RMS error vs clean signal (σ)",
                title="Error against the noise-free signal (≥ 1: the noise stays)")
    for ax in axes:
        ax.grid(alpha=0.3, lw=0.4, which="both")
        ax.legend(fontsize=7)
    fig.tight_layout()
    img = save_fig(fig, OUT, 'noise', 'noise floor sweep')
    h = ["<h3 id='noise-floor'>Noise floor sweep: fluxcode-16, f = 0 … 1</h3>"
         "<p>On blocks whose second differences look like white noise (lag-1 autocorrelation &lt; −0.6), the step "
         "becomes at most f·σ, with σ a robust estimate from the same differences (docs/ENCODER.md §5.2). B stays 16; "
         "on these signals the noise-floor step is coarser than the 16-bit step for every f ≥ 0.01, so f alone sets "
         "the size. fluxcode's default is f = 0.25. Noisy signals only (600 blocks each, minute block groups); clean signals are unaffected at every f. Errors "
         "are median RMS over blocks in units of the true noise σ, against the input and against the same signal "
         "generated without noise.</p>",
         img,
         "<table><tr><th>signal</th>" + "".join(f"<th>f = {f:g}</th>" for f in NOISE_F) + "</tr>"]
    for name, rows in res.items():
        h.append(f"<tr><td>{name}</td>" + "".join(
            f"<td>{r['bps']:.2f} b/s<br><span class='muted'>+{r['in']:.3f}σ / {r['clean']:.3f}σ</span></td>"
            for r in rows) + "</tr>")
    h.append("</table><p class='muted'>Each cell: bits/sample, then RMS error vs input / vs clean signal (σ). "
             "Lossless points (noisy-sine q0.1 at f ≤ 0.01, still on its decimal grid) are left off the log axis. "
             "The step is 2^floor(log2(f·σ)), so f values between powers of two can give the same step.</p>")
    return "".join(h)


def dpcm_sweep():
    """Compare expander curves and step scale D for companded DPCM across all datasets."""
    cands = [(6, "byte", 1.0, f"cubic c={c:g}", cubic_expander(c)) for c in (0.0, 0.5, 0.8, 1.0)]
    cands += [(6, "byte", 1.0, f"μ-law μ={mu}", mulaw_expander(mu)) for mu in (7, 31, 255)]
    cands += [(6, "byte", 0.5, "cubic c=0", cubic_expander(0.0))]
    cands += [(4, "byte", 1.0, f"cubic c={c:g}", cubic_expander(c)) for c in (0.0, 0.5, 0.8, 1.0)]
    cands += [(4, "byte", 1.0, f"μ-law μ={mu}", mulaw_expander(mu)) for mu in (3, 7, 31)]
    cands += [(4, "nibble", 1.0, "cubic c=0", cubic_expander(0.0)),
              (4, "nibble", 1.0, "cubic c=0.8", cubic_expander(0.8))]
    kind_rows = by_kind()
    pts = []
    h = ["<h3 id='dpcm-sweep'>DPCM sweep: code bits, packing, curve, step scale (all signal kinds, + deflate)</h3>"
         "<p>D = step scale × max |x[i] − x[i−1]| per block. “nibble” packs two 4-bit codes per byte before deflate. "
         "One deflate call per one-minute block group. Per-kind median RMSE in % FS.</p><table><tr><th>bits</th><th class='l'>packing</th>"
         "<th class='l'>curve</th><th>D scale</th><th>bits/sample</th><th>median RMSE</th><th>worst RMSE</th><th>worst max |err|</th>"
         + "".join(f"<th>{k}</th>" for k in KINDS) + "</tr>"]
    for bits, pack, ds, label, g in cands:
        codec = SharedBackend(CompandedDpcm(label, label, g, ds, bits, pack))
        rs = [run_codec(codec, kind_rows[k]) for k in KINDS]
        bps = np.mean([r["bps"] for r in rs])
        pts.append((bps, 100 * np.median([r["rmse"] for r in rs]), bits, pack, label))
        h.append(f"<tr><td>{bits}</td><td class='l'>{pack}</td><td class='l'>{label}</td><td>{ds:g}</td><td>{bps:.2f}</td>"
                 f"<td>{100 * np.median([r['rmse'] for r in rs]):.3g}%</td><td>{100 * max(r['rmse_worst'] for r in rs):.3g}%</td>"
                 f"<td>{100 * max(r['maxabs_worst'] for r in rs):.3g}%</td>" + "".join(f"<td>{100 * r['rmse']:.2g}</td>" for r in rs) + "</tr>")
    h.append("</table><p class='muted'>D below the largest step causes slope overload on smooth signals "
             "(a line rising 1/sample can't be tracked with a max step of 0.5), regardless of curve. "
             "Fewer levels means less resolved noise, hence less entropy for deflate: "
             "4-bit codes land near quant6-delta1-deflate's size with much lower error on smooth signals. Nibble packing "
             "doesn't help because deflate matches on byte boundaries.</p>")
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for bits, mk in ((6, "o"), (4, "s")):
        sel = [p for p in pts if p[2] == bits]
        ax.scatter([p[0] for p in sel], [p[1] for p in sel], marker=mk, s=36, color=COLORS["dpcm6-linear-deflate"], edgecolor="k", linewidth=0.4, label=f"{bits}-bit codes")
        for p in sel:
            ax.annotate(f"{p[4]}{' (nibble)' if p[3] == 'nibble' else ''}", p[:2], textcoords="offset points", xytext=(4, 3), fontsize=6.5)
    ax.set(xlabel="bits / sample (mean over the kinds)", ylabel="median RMSE (% of range, log)", yscale="log",
           title="DPCM candidates: size vs error")
    rmse_axis(ax, 10)
    ax.grid(alpha=0.3, lw=0.4, which="both")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    h.append(save_fig(fig, OUT, "dpcm-sweep", "DPCM candidates, size vs error"))
    return "".join(h)


DECISIONS = """<h2 id='notes'>Design notes: decisions made where the spec was open</h2><ul>
<li><b>Framing:</b> each codec encodes a one-minute block group (60 blocks). Codecs with a general-purpose back end
(deflate, zstd, FLAC) make one stream per block group: per-block headers first, interleaved by field, then the per-block
bodies, with one compressor call (FLAC: one stream, one frame per block). Codecs without one store the per-block
encodings back to back, with a LEB128 length per block only where the size depends on the data (the knot codecs and
Gorilla). fluxcode is one real block group per minute. Per-block headers such as min and max are inside the block group.</li>
<li><b>Binary tree:</b> tree built bottom-up by pairing (odd node out gets a unary parent) →
internal levels 1+2+4+8+16+32+63+125+250+500 = 1001 nodes, 1000 leaves, 3002 bits + 16-byte header = 392 bytes.
Every internal node (including the root, which is always 00) gets a 2-bit code applied to the range it inherits.
Encoder picks the narrowing that still <i>contains</i> all samples under the node (ties → the one centred on the data), else 00.
Leaf bit selects lower/upper half of its final range and decodes to that half's <i>centre</i> (¼ or ¾) in <code>bintree-center</code>, or to the literal bottom/top of the range in <code>bintree-edge</code> (same bytes; better peaks, worse RMSE).</li>
<li><b>Min/max piecewise:</b> two knots per 16-sample group (one if argmin = argmax), plus sample 0 and 999 as anchors.
Knot-to-knot gaps can reach ~30 samples, which a 4-bit offset (max 16) can't express, so filler knots (actual sample value)
are inserted every 16 samples when needed. Knot values are 12-bit relative to the global min/max; each knot costs 2 bytes.
The stream ends when a knot lands on sample 999, so no count is stored.</li>
<li><b>Consolidation:</b> drop the first point of block i+1 when block i's last point is its final sample, block i+1's first
point is its first sample, they are opposite extremes, and the series keeps going the same direction across the boundary
(so real reversals are never dropped).</li>
<li><b>PCHIP:</b> same knots and bytes as <code>pwlinear-minmax16</code>; only the decoder's interpolation differs.</li>
<li><b>Industrial-historian bounded-error piecewise linear (swinging door):</b> implemented as swinging-door trending, the standard historian algorithm. Deviation is
a fraction of full scale (0.5% and 2% shown). Stored in the same knot format, so a knot is forced at least every 16 samples.
Archived knots are real samples, so a single-sample impulse lands on the right index (see <code>impulses</code> peak shift).
Error can slightly exceed the deviation: classic SDT anchors to the last in-band sample, and knots are then 12-bit quantized.</li>
<li><b>Gorilla:</b> the value half of Facebook’s Gorilla (XOR with previous, reuse leading/trailing-zero window). Lossless; verified bit-exact.
Timestamps are implicit here; Gorilla’s delta-of-delta timestamps would add ≈1 bit/sample (≈125 B) for a regular 1 ms clock.</li>
<li><b>Extras:</b> <code>quant8-delta1-deflate</code> (same samples as quant8, delta + raw deflate — lossless on top of quant8, shows how much
entropy coding buys), <code>dct-top64</code> (64 largest DCT coefficients, 10-bit index + 16-bit value; frequency-domain contrast).</li>
<li><b>quant6-delta1-deflate:</b> same as quant8-delta1-deflate with 64 levels; still one byte per sample before deflate.</li>
<li><b>Companded DPCM (dpcm6-*-deflate):</b> the S-curve is applied to the <i>residual</i> against the decoder's own running
reconstruction (closed loop), so a coarsely-quantized big jump is corrected over the next samples instead of accumulating.
Code k ∈ −31..31 (6 bits, signed byte so “no change” = 0x00) decodes to D·g(k/31), with g an odd expander that is flat near 0.
Your cubic c·u³+(1−c)·u is used as that expander (code → delta); a true sigmoid is the compressor side, and its inverse
blows up at ±1, so μ-law is used as the standard signed S-curve. Header: x[0] and D as float64.
<code>dpcm4-*-deflate</code> use 4-bit codes (k ∈ −7..7), still one signed byte per code before deflate.</li>
<li><b>ratectl-delta0123-zstd-4b / ratectl-flac-4b:</b> the rate-controlled pipelines from the <a href="rate.html">rate-control experiment</a>
as block group codecs: rescale each block to [0,1], quantize with one step Δ for the block group, per-block predictor, then zstd (byte-split
residuals, one call per block group) or libFLAC (one stream per block group). Δ is bisected so each one-minute block group lands at ≤ 4 bits/sample
including its header (Δ, and min and max of every block). Max error is Δ/2 of the block's range, guaranteed.</li>
<li><b>delta0123-zstd-8 / -10 / -12:</b> no search. Step = (max − min)/(2<sup>B</sup> − 1) on the block's own range
(min and max decode exactly), predictor order = argmin var(diff(q, k)) for k = 0..3 (counts in the notes column), one zstd (level 3) call
per block group. If a block's estimated size (from the residual variance) would exceed 8 bits/sample, B drops by the excess and the
block is requantized. <code>delta12-zstd-8</code> is the performance option: B = 8, only orders 1 and 2 considered.
Each block's min and max are in the block group.</li>
<li><b>cw8-delta1-linear / cw8-delta1-sqrt (constant width):</b> exactly one byte per sample, no entropy coding.
<code>cw8-delta1-linear</code> is closed-loop DPCM with 255 uniform levels D·k/127 (D = largest step in the block).
<code>cw8-delta1-sqrt</code> is the "semi-quadratic" delta: a 32-bit grid over [min, max], one byte = [odd_shift | 7-bit signed
dampened value], decoded with CLZ so delta resolution scales like √|delta| (7 bits for the largest jumps, ~13 for small ones),
wrapping mod 2<sup>32</sup>. Encoder is closed loop and picks the nearest decodable delta.
<code>cw8-delta1-bfp-e2</code> is 8-bit deltas (all 256 codes) with a 2-bit step exponent per 16-sample frame: step = 2<sup>−e</sup>
of the 8-bit step, decoded with a shift. e = 0 is exactly the quant8 grid with rollover across the block's full range, so max
error never exceeds quant8's (range/510); quiet frames get up to 3 extra bits. <code>cw8-delta1-bfp-e4</code> uses 4-bit exponents
(e = 0..15, up to 15 extra bits; 8.41 bits/sample). <code>cw4-delta1-bfp-e4</code> / <code>cw6-delta1-bfp-e4</code> are the same
design with 4-bit / 6-bit codes (base grid 16 / 64 levels, max error bounded by q4's / q6's: range/30, range/126) and 4-bit
exponents: 4.42 / 6.42 bits/sample including the 20-byte header. See <code>tslab/constwidth/</code>.</li>
<li><b>cw-delta1-tree-2 / -4 / -7:</b> hierarchical block floating point over the deltas. The original tree codec's
binary tree over the 999 deltas, 1 bit per internal node (1 = halve the scale for that subtree), and sign-relative leaf
codes: decoded delta = c · s · direction of the last nonzero delta, a negative code flips direction. 2-bit leaves use
{−1, 0, 1, 2}; wider leaves use two's-complement ranges (−8..7, −64..63). A node halves if every leaf below still fits,
sized for the direction each leaf will actually have. Leaves are closed loop with integer residuals. 7-bit leaves
amortize to ~8 bits/sample. See <code>tslab/constwidth/delta_tree.py</code>.</li>
<li>Periodic signals use irrational frequencies (√17, π², 16π Hz, …) so cycles never land on whole samples; integer Hz at 1 kHz repeats exactly and made deflate look far better than it would on real data.</li>
<li>Uniform noise used for “±10” noise. Random seeds are fixed so runs are reproducible.</li>
</ul>"""


if __name__ == "__main__":
    results = run()
    write_report(results)
    print(f"wrote {OUT / 'index.html'}")
