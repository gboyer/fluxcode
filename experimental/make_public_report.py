# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Render the public real-world datasets HTML report (report/public.html + public-*.svg).

Reads precomputed benchmark results from public_datasets/results.json and dataset
provenance/licenses from public_datasets/datasets.json (no re-benchmarking needed).

    cd experimental && uv run python make_public_report.py
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from tslab.report.page import clean_outputs, collapsible_table, save_fig, shell

HERE = Path(__file__).resolve().parent
PUB_DIR = HERE / "public_datasets"
RESULTS_PATH = PUB_DIR / "results.json"
CODEC_RESULTS_PATH = PUB_DIR / "codec_results.json"
DATASETS_PATH = PUB_DIR / "datasets.json"
OUT = HERE / "report"

EXACT_PCT = 1e-7
ZERO_PCT = EXACT_PCT
Y_TOP = 15.0
X_MAX = 20.0
ZERO_NOTE = "0*: lossless (RMSE ≤ $10^{-7}$%)"

FAM_STYLE = {
    "lossless": ("#525252", "P", "Lossless zstd baselines"),
    "classic": ("#3182bd", "o", "Classic (Gorilla, quant, DPCM, SDT, DCT)"),
    "constwidth": ("#e6550d", "^", "Constant width (BFP)"),
    "entropy": ("#31a354", "s", "Entropy (delta + zstd)"),
    "flux": ("#6a51a3", "D", "fluxcode"),
}

# House palette (aligned with make_time_report.py and make_report.py)
C_RAW = "#8e99a4"
C_BSHUF = "#d95f02"
C_BIN = "#9e9ac8"
C_NF0 = "#6a51a3"
C_DEF = "#3f007d"
C_ENC = "#2a78d6"
C_DEC = "#1a9e77"
C_T0 = "#2a78d6"
C_T05 = "#eb6834"
C_T15 = "#1a9e77"


def slug(name: str) -> str:
    return "grp-" + re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _shown_rmse(v: float) -> float:
    return ZERO_PCT if v <= EXACT_PCT else v


def _pct_label(v: float) -> str:
    if v == ZERO_PCT:
        return "0*"
    d = round(np.log10(v))
    return f"{10 ** d}%" if d >= 0 else f"$10^{{{d}}}$%"


def _rmse_axis(ax: Any, fontsize: int) -> None:
    bottom = ZERO_PCT / 3
    ax.set_ylim(bottom, Y_TOP)
    ax.axhspan(bottom, 1.5 * ZERO_PCT, color="0.5", alpha=0.12, lw=0, zorder=0)
    first = round(np.log10(EXACT_PCT)) + 1
    decades = [10.0 ** d for d in range(first, int(np.floor(np.log10(Y_TOP))) + 1)]
    major = [ZERO_PCT] + decades
    ax.set_yticks(major, [_pct_label(v) for v in major], fontsize=fontsize)
    minor = [m * v for v in [decades[0] / 10] + decades for m in range(2, 10) if max(bottom, EXACT_PCT) < m * v < Y_TOP]
    ax.set_yticks(minor, [""] * len(minor), minor=True)


def plot_real_scatter(codec_data: dict[str, Any]) -> str:
    """Real-world size vs error scatterplot: median bits/sample vs median RMSE across the 9 curated public series."""
    series = codec_data["series"]
    codec_names = list(series[0]["codecs"].keys())
    fig, ax = plt.subplots(figsize=(11, 6.2), layout="constrained")
    ax.set_yscale("log")
    ax.set_xlim(-0.3, X_MAX + 0.5)
    ax.set_xticks(range(0, int(X_MAX) + 1, 2))

    # fluxcode B=6..16 curve (median across the 9 series)
    n_b = len(codec_data["curve_bits"])
    curve_pts = [
        (
            float(np.median([s["fluxcode_curve"][i]["bps"] for s in series])),
            _shown_rmse(float(np.median([s["fluxcode_curve"][i]["rmse_med_pct"] for s in series]))),
        )
        for i in range(n_b)
    ]
    ax.plot(*zip(*curve_pts), color=C_DEF, lw=2.4, marker=".", ms=6, zorder=3, label="fluxcode (B = 6..16, nf=0)")

    pts = []
    for cname in codec_names:
        if cname == "gorilla-xor":
            continue
        fam = series[0]["codecs"][cname]["family"]
        color, mk, _ = FAM_STYLE[fam]
        bps = float(np.median([s["codecs"][cname]["bps"] for s in series]))
        rmse = float(np.median([s["codecs"][cname]["rmse_med_pct"] for s in series]))
        px, py = min(bps, X_MAX - 0.2), _shown_rmse(rmse)
        if cname == "fluxcode-default":
            ax.scatter(px, py, marker="D", s=78, color=C_DEF, edgecolor="k", linewidth=0.9, zorder=5)
            pts.append(("fluxcode (default)", px, py))
            continue
        if cname == "fluxcode-bin-nf0":
            ax.scatter(px, py, marker="D", s=54, color=C_BIN, edgecolor="k", linewidth=0.7, zorder=4)
            pts.append(("fluxcode (binary grid)", px, py))
            continue
        ax.scatter(px, py, marker=mk, s=52, color=color, edgecolor="k", linewidth=0.45, zorder=4)
        lbl = cname + (f" ({bps:.0f} b/s)" if bps > X_MAX else "")
        pts.append((lbl, px, py))

    _rmse_axis(ax, 9)
    fig.canvas.draw()
    taken: list[tuple[float, float, float, float]] = []
    for name, px, py in pts:
        u, v = ax.transData.transform((px, py))
        w = 4.5 * len(name)
        box = (u + 6, v + 2, u + 6 + w, v + 13)
        if box[2] > ax.bbox.x1 + 6:
            box = (u - 6 - w, v + 2, u - 6, v + 13)
        if any(box[0] < t[2] and t[0] < box[2] and box[1] < t[3] and t[1] < box[3] for t in taken):
            # Try below the point
            box2 = (u + 6, v - 13, u + 6 + w, v - 2)
            if any(box2[0] < t[2] and t[0] < box2[2] and box2[1] < t[3] and t[1] < box2[3] for t in taken):
                continue
            taken.append(box2)
            ax.annotate(name, (px, py), textcoords="offset points", xytext=(6, -9), ha="left", fontsize=7.5)
            continue
        taken.append(box)
        ax.annotate(
            name,
            (px, py),
            textcoords="offset points",
            xytext=(-6, 3) if box[2] < u else (6, 3),
            ha="right" if box[2] < u else "left",
            fontsize=7.5,
        )

    ax.text(0.01, 0.01, ZERO_NOTE, transform=ax.transAxes, ha="left", va="bottom", fontsize=8, color="0.4")
    ax.set(
        xlabel="bits / sample (median across 9 real-world datasets)",
        ylabel="median block RMSE (% of range, log)",
        title="Real-world size vs. error across 9 public datasets (lower-left is better)",
    )
    ax.grid(alpha=0.3, lw=0.4, which="both")
    handles = [
        Line2D([], [], color=C_DEF, lw=2.4, marker=".", ms=6, label="fluxcode (B = 6..16, nf=0)"),
        Line2D([], [], marker="D", ls="", color=C_DEF, markeredgecolor="k", ms=7, label="fluxcode (default)"),
        Line2D([], [], marker="D", ls="", color=C_BIN, markeredgecolor="k", ms=6, label="fluxcode (binary grid, nf=0)"),
    ] + [
        Line2D([], [], marker=mk, ls="", color=col, markeredgecolor="k", ms=6.5, label=lbl)
        for fam, (col, mk, lbl) in FAM_STYLE.items()
        if fam != "flux"
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8.5, frameon=True)
    return save_fig(fig, OUT, "public-scatter", "Real-world bits per sample vs RMSE across codecs")


def plot_real_rd(codec_data: dict[str, Any]) -> str:
    """3x3 per-dataset real-world size vs error panels."""
    series = codec_data["series"]
    cols = 3
    rows = -(-len(series) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(14.5, 3.6 * rows), layout="constrained")
    for ax, s in zip(axes.flat, series):
        ax.set_yscale("log")
        pts = [(p["bps"], _shown_rmse(p["rmse_med_pct"])) for p in s["fluxcode_curve"]]
        ax.plot(*zip(*pts), color=C_DEF, lw=2.2, marker=".", ms=5, zorder=3)
        for cname, st in s["codecs"].items():
            if cname == "gorilla-xor":
                continue
            fam = st["family"]
            col, mk, _ = FAM_STYLE[fam]
            px, py = min(st["bps"], X_MAX - 0.2), _shown_rmse(st["rmse_med_pct"])
            if cname == "fluxcode-default":
                ax.scatter(px, py, marker="D", s=62, color=C_DEF, edgecolor="k", linewidth=0.8, zorder=5)
            elif cname == "fluxcode-bin-nf0":
                ax.scatter(px, py, marker="D", s=44, color=C_BIN, edgecolor="k", linewidth=0.6, zorder=4)
            else:
                ax.scatter(px, py, marker=mk, s=34, color=col, edgecolor="k", linewidth=0.35, zorder=4)
        ax.set_xlim(-0.3, X_MAX + 0.5)
        ax.set_xticks(range(0, int(X_MAX) + 1, 4))
        _rmse_axis(ax, 7)
        ax.set_title(s["label"], fontsize=10)
        ax.tick_params(labelsize=7.5)
        ax.grid(alpha=0.3, which="both", lw=0.4)
    for ax in axes[:, 0]:
        ax.set_ylabel("median RMSE (% FS, log)", fontsize=8.5)
    for ax in axes[-1]:
        ax.set_xlabel("bits / sample", fontsize=8.5)
    handles = [
        Line2D([], [], color=C_DEF, lw=2.2, marker=".", ms=5, label="fluxcode (B = 6..16, nf=0)"),
        Line2D([], [], marker="D", ls="", color=C_DEF, markeredgecolor="k", ms=7, label="fluxcode (default)"),
        Line2D([], [], marker="D", ls="", color=C_BIN, markeredgecolor="k", ms=6, label="fluxcode (binary grid)"),
    ] + [
        Line2D([], [], marker=mk, ls="", color=col, markeredgecolor="k", ms=6, label=lbl)
        for fam, (col, mk, lbl) in FAM_STYLE.items()
        if fam != "flux"
    ] + [Line2D([], [], ls="", label=ZERO_NOTE)]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=8.5, frameon=False)
    return save_fig(fig, OUT, "public-rd", "Per-dataset real-world bits per sample vs RMSE")


def render_codec_matrix(codec_data: dict[str, Any]) -> str:
    series = codec_data["series"]
    codec_names = list(series[0]["codecs"].keys())
    h = ["<table><tr><th class='l'>codec</th>" + "".join(f"<th>{html.escape(s['label'])}</th>" for s in series) + "</tr>"]
    for cname in codec_names:
        cells = []
        for s in series:
            st = s["codecs"][cname]
            err = "EXACT" if st["rmse_med_pct"] <= EXACT_PCT else f"{st['rmse_med_pct']:.3g}%"
            cells.append(f"<td>{st['bps']:.2f} b/s<br><span class='muted'>{err}</span></td>")
        h.append(f"<tr><td><code>{html.escape(cname)}</code></td>{''.join(cells)}</tr>")
    h.append("</table>")
    return collapsible_table(f"Show multi-codec comparison table ({len(codec_names)} codecs × {len(series)} datasets)", "".join(h))


def load_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {RESULTS_PATH}. Run 'make public-bench' first to generate results.json."
        )
    if not DATASETS_PATH.exists():
        raise FileNotFoundError(f"Missing {DATASETS_PATH}.")
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(DATASETS_PATH.read_text(encoding="utf-8"))
    codec_data = json.loads(CODEC_RESULTS_PATH.read_text(encoding="utf-8")) if CODEC_RESULTS_PATH.exists() else None
    return results, manifest, codec_data


def plot_bps(groups: dict[str, dict[str, float]]) -> str:
    """Horizontal grouped bar chart comparing baselines and fluxcode configs across all 14 families."""
    names = list(groups.keys())
    y = np.arange(len(names))
    bar_h = 0.155
    series_spec = [
        ("bshuf", "bshuf zstd", C_BSHUF, -2),
        ("raw", "raw zstd", C_RAW, -1),
        ("bin_nf0", "fluxcode bin nf=0", C_BIN, 0),
        ("nf0", "fluxcode nf=0", C_NF0, 1),
        ("default", "fluxcode default", C_DEF, 2),
    ]
    fig, ax = plt.subplots(figsize=(11.5, 7.6), layout="constrained")
    for key, label, color, offset in series_spec:
        vals = [groups[g][key] for g in names]
        ax.barh(y + offset * bar_h, vals, height=bar_h, color=color, label=label, edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    tot_samples_m = sum(int(groups[g]["n"]) for g in names) / 1e6
    ax.set_xlabel("sample-weighted bits / sample (lower is smaller; float64 = 64)")
    ax.set_title(
        f"Compression size across {len(names)} public dataset families ({tot_samples_m:.2f}M float64 samples)",
        loc="left",
        fontsize=11,
    )
    ax.grid(axis="x", color="0.9", lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, ncol=5, loc="upper right", fontsize=8.5)
    return save_fig(fig, OUT, "public-bps", "Bits per sample across 14 public dataset families")


def plot_grid_gain(series_rows: list[dict[str, Any]]) -> str:
    """Horizontal bar chart of rational/decimal grid savings (% smaller vs binary grid) on key series."""
    picks = [
        ("UCI Appliances", "Press_mm_hg", "UCI Appliances / Press_mm_hg (1/3 & 1/4)"),
        ("UCI Appliances", "T_out", "UCI Appliances / T_out (1/3 & 1/4)"),
        ("NOAA Buoy", "PRES_raw", "NOAA Buoy / PRES_raw (0.1 hPa)"),
        ("Elf Corpus", "Wind-Speed", "Elf / Wind-Speed (decimal)"),
        ("ERA5 Weather", "pressure_f32", "ERA5 / pressure_f32 (float32 recovery)"),
        ("MIT-BIH ECG", "rec101_MLII", "MIT-BIH ECG / rec101_MLII (0.005 mV)"),
        ("Elf Corpus", "IR-bio-temp", "Elf / IR-bio-temp (decimal)"),
        ("UCI Hydraulic", "VS1_1Hz", "UCI Hydraulic / VS1_1Hz (PLC)"),
        ("NOAA Buoy", "WTMP_nan", "NOAA Buoy / WTMP_nan (NaN gaps)"),
        ("ERA5 Weather", "temp_2m_f32", "ERA5 / temp_2m_f32 (float32 recovery)"),
        ("UCI Power", "Global_reactive_power", "UCI Power / Global_reactive_power"),
        ("UCI Hydraulic", "PS1_100Hz", "UCI Hydraulic / PS1_100Hz (100 Hz)"),
        ("Binance Trades", "BTCUSDT_price", "Binance / BTCUSDT_price ($0.01)"),
        ("CWRU Vibration", "normal_48k_DE", "CWRU / normal_48k_DE (~2.086e-4 ADC)"),
        ("CWRU Vibration", "fault_B_12k_FE", "CWRU / fault_B_12k_FE (~2.086e-4 ADC)"),
    ]
    by_key = {(r["group"], r["series"]): r for r in series_rows}
    labels: list[str] = []
    gains: list[float] = []
    colors: list[str] = []
    for g, s, label in picks:
        r = by_key.get((g, s))
        if r is None:
            continue
        nf0 = float(r["results"]["nf0"]["bps"])
        bin_bps = float(r["results"]["bin_nf0"]["bps"])
        gain = 100.0 * (bin_bps - nf0) / bin_bps if bin_bps > 0 else 0.0
        labels.append(label)
        gains.append(gain)
        colors.append("#1a9e77" if "f32" in s else ("#eb6834" if g == "CWRU Vibration" else C_NF0))

    order = np.argsort(gains)[::-1]
    labels = [labels[i] for i in order]
    gains = [gains[i] for i in order]
    colors = [colors[i] for i in order]

    fig, ax = plt.subplots(figsize=(11, 5.2), layout="constrained")
    y = np.arange(len(labels))
    bars = ax.barh(y, gains, color=colors, height=0.65)
    for bar, val in zip(bars, gains):
        ax.text(val + 0.8, bar.get_y() + bar.get_height() / 2, f"+{val:.1f}%", va="center", ha="left", fontsize=8.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.8)
    ax.invert_yaxis()
    ax.set_xlim(0, max(gains) + 10)
    ax.set_xlabel("% smaller at noise_floor_sigma=0 vs binary-only grids (grids='binary')")
    ax.set_title("Rational, decimal, ADC, and float32-recovered grid savings in the wild", loc="left", fontsize=11)
    ax.grid(axis="x", color="0.9", lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return save_fig(fig, OUT, "public-grid-gain", "Rational and decimal grid savings vs binary grids")


def plot_throughput(groups: dict[str, dict[str, float]], series_rows: list[dict[str, Any]]) -> str:
    """Side-by-side encode and decode MB/s across all 14 dataset families (sample-weighted)."""
    names = list(groups.keys())
    enc_vals: list[float] = []
    dec_vals: list[float] = []
    for g in names:
        rows = [r for r in series_rows if r["group"] == g]
        tot_n = sum(int(r["n"]) for r in rows)
        # Harmonic mean by bytes (total MB / total seconds) is exact sample-weighted throughput
        tot_mb = 8.0 * tot_n / 1e6
        tot_enc_s = sum((8.0 * int(r["n"]) / 1e6) / float(r["results"]["default"]["enc_mbs"]) for r in rows)
        tot_dec_s = sum((8.0 * int(r["n"]) / 1e6) / float(r["results"]["default"]["dec_mbs"]) for r in rows)
        enc_vals.append(tot_mb / tot_enc_s if tot_enc_s > 0 else 0.0)
        dec_vals.append(tot_mb / tot_dec_s if tot_dec_s > 0 else 0.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5.4), sharey=True, layout="constrained")
    y = np.arange(len(names))
    b1 = ax1.barh(y, enc_vals, color=C_ENC, height=0.65)
    b2 = ax2.barh(y, dec_vals, color=C_DEC, height=0.65)
    for bar, val in zip(b1, enc_vals):
        ax1.text(val + 15, bar.get_y() + bar.get_height() / 2, f"{val:.0f}", va="center", ha="left", fontsize=8)
    for bar, val in zip(b2, dec_vals):
        ax2.text(val + 70, bar.get_y() + bar.get_height() / 2, f"{val:.0f}", va="center", ha="left", fontsize=8)
    ax1.set_yticks(y)
    ax1.set_yticklabels(names, fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlim(0, max(enc_vals) * 1.18)
    ax2.set_xlim(0, max(dec_vals) * 1.18)
    ax1.set_xlabel("encode throughput (MB/s of float64)")
    ax2.set_xlabel("decode throughput (MB/s of float64)")
    ax1.set_title("Encode MB/s (default Params(), 1 thread)", loc="left", fontsize=10)
    ax2.set_title("Decode MB/s (default Params(), 1 thread)", loc="left", fontsize=10)
    for ax in (ax1, ax2):
        ax.grid(axis="x", color="0.9", lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    return save_fig(fig, OUT, "public-throughput", "Encode and decode throughput across 14 dataset families")


def plot_timestamps(ts_rows: list[dict[str, Any]]) -> str:
    """Grouped horizontal bar chart of timestamp cost (bits/sample added) across real clocks."""
    labels = [f"{r['group']} / {r['series']} ({r['clock_type']})" for r in ts_rows]
    exact = [float(r["exact_times"]) - float(r["values_only"]) for r in ts_rows]
    t05 = [float(r["time_error_005"]) - float(r["values_only"]) for r in ts_rows]
    t15 = [float(r["time_error_015"]) - float(r["values_only"]) for r in ts_rows]

    y = np.arange(len(labels))
    bar_h = 0.25
    fig, ax = plt.subplots(figsize=(11, 4.6), layout="constrained")
    ax.barh(y - bar_h, exact, height=bar_h, color=C_T0, label="exact timestamps")
    ax.barh(y, t05, height=bar_h, color=C_T05, label="time_error=0.05")
    ax.barh(y + bar_h, t15, height=bar_h, color=C_T15, label="time_error=0.15")

    for yi, ex, r15 in zip(y, exact, t15):
        if ex < 0.1:
            ax.text(ex + 0.12, yi, f"+{ex:.3f} bps", va="center", ha="left", fontsize=8, color="0.35")
        elif abs(ex - r15) > 0.1:
            ax.text(ex + 0.12, yi - bar_h, f"+{ex:.2f}", va="center", ha="left", fontsize=8)
            ax.text(r15 + 0.12, yi + bar_h, f"+{r15:.2f} (−{ex - r15:.2f})", va="center", ha="left", fontsize=8, color=C_T15)
        else:
            ax.text(ex + 0.12, yi, f"+{ex:.2f} bps", va="center", ha="left", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.8)
    ax.invert_yaxis()
    ax.set_xlim(0, max(exact) + 2.2)
    ax.set_xlabel("timestamp cost added over values-only (bits / sample)")
    ax.set_title("Real-world clock cost: regular, gapped, jittered IMU, and irregular exchange ticks", loc="left", fontsize=11)
    ax.grid(axis="x", color="0.9", lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    return save_fig(fig, OUT, "public-timestamps", "Timestamp bits/sample added across real-world clocks")


def render_group_summary_table(groups: dict[str, dict[str, float]]) -> str:
    h = [
        "<table><tr><th class='l'>dataset family</th><th>samples</th><th>raw zstd</th><th>bshuf zstd</th>"
        "<th>fc default</th><th>fc eff=1</th><th>fc eff=5</th><th>fc nf=0</th><th>fc bin nf=0</th>"
        "<th>grid gain</th><th>ratio vs f64 (def)</th></tr>"
    ]
    tot_n = 0
    sums = {k: 0.0 for k in ("raw", "bshuf", "default", "eff1", "eff5", "nf0", "bin_nf0")}
    for gname, d in groups.items():
        n = int(d["n"])
        tot_n += n
        for k in sums:
            sums[k] += float(d[k]) * n
        gain = 100.0 * (d["bin_nf0"] - d["nf0"]) / d["bin_nf0"] if d["bin_nf0"] > 0 else 0.0
        h.append(
            f"<tr><td><a href='#{slug(gname)}'>{html.escape(gname)}</a></td><td>{n:,}</td>"
            f"<td>{d['raw']:.2f}</td><td>{d['bshuf']:.2f}</td><td class='best'>{d['default']:.2f}</td>"
            f"<td>{d['eff1']:.2f}</td><td>{d['eff5']:.2f}</td><td>{d['nf0']:.2f}</td><td>{d['bin_nf0']:.2f}</td>"
            f"<td>{gain:+.1f}%</td><td>{64.0 / d['default']:.1f}×</td></tr>"
        )
    tot_avg = {k: v / tot_n for k, v in sums.items()}
    tot_gain = 100.0 * (tot_avg["bin_nf0"] - tot_avg["nf0"]) / tot_avg["bin_nf0"]
    h.append(
        f"<tr><td><b>overall sample-weighted</b></td><td><b>{tot_n:,}</b></td>"
        f"<td><b>{tot_avg['raw']:.2f}</b></td><td><b>{tot_avg['bshuf']:.2f}</b></td>"
        f"<td class='best'><b>{tot_avg['default']:.2f}</b></td><td><b>{tot_avg['eff1']:.2f}</b></td>"
        f"<td><b>{tot_avg['eff5']:.2f}</b></td><td><b>{tot_avg['nf0']:.2f}</b></td>"
        f"<td><b>{tot_avg['bin_nf0']:.2f}</b></td><td><b>{tot_gain:+.1f}%</b></td>"
        f"<td><b>{64.0 / tot_avg['default']:.1f}×</b></td></tr>"
    )
    h.append("</table>")
    return "".join(h)


def render_timestamp_table(ts_rows: list[dict[str, Any]]) -> str:
    h = [
        "<table><tr><th class='l'>dataset (channel)</th><th>N</th><th class='l'>clock type</th>"
        "<th>values only</th><th>+ exact times</th><th>exact time cost</th>"
        "<th>+ time_error=0.05</th><th>+ time_error=0.15</th></tr>"
    ]
    for r in ts_rows:
        vo = float(r["values_only"])
        ex = float(r["exact_times"])
        t05 = float(r["time_error_005"])
        t15 = float(r["time_error_015"])
        cls = " class='best'" if ex - t15 > 0.1 else ""
        h.append(
            f"<tr><td>{html.escape(r['group'])} / <code>{html.escape(r['series'])}</code></td>"
            f"<td>{int(r['n']):,}</td><td class='l'>{html.escape(r['clock_type'])}</td>"
            f"<td>{vo:.2f}</td><td>{ex:.2f}</td><td>+{ex - vo:.3f}</td>"
            f"<td>{t05:.2f}</td><td{cls}>{t15:.2f}</td></tr>"
        )
    h.append("</table>")
    return "".join(h)


def render_series_tables(groups: dict[str, dict[str, float]], series_rows: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for gname, gavg in groups.items():
        rows = [r for r in series_rows if r["group"] == gname]
        tbl: list[str] = [
            "<table><tr><th class='l'>series</th><th>N</th><th>NaN%</th><th>raw zstd</th><th>bshuf zstd</th>"
            "<th>fc default</th><th>err% (def)</th><th>fc nf=0</th><th>exact? (nf=0)</th>"
            "<th>fc bin nf=0</th><th>grid gain</th><th>enc/dec MB/s (def)</th></tr>"
        ]
        for r in rows:
            res = r["results"]
            d_def = res["default"]
            d_nf0 = res["nf0"]
            d_bin = res["bin_nf0"]
            nf0_bps = float(d_nf0["bps"])
            bin_bps = float(d_bin["bps"])
            gain = 100.0 * (bin_bps - nf0_bps) / bin_bps if bin_bps > 0 else 0.0
            def_err = "EXACT" if d_def["exact"] else f"{float(d_def['worst_pct']):.3f}%"
            nf0_err = "EXACT" if d_nf0["exact"] else f"{float(d_nf0['worst_pct']):.4f}%"
            spd = f"{float(d_def['enc_mbs']):.0f} / {float(d_def['dec_mbs']):.0f}"
            tbl.append(
                f"<tr><td><code>{html.escape(r['series'])}</code></td><td>{int(r['n']):,}</td>"
                f"<td>{float(r['nan_pct']):.1f}%</td><td>{float(r['raw_zstd']):.2f}</td>"
                f"<td>{float(r['bshuf_zstd']):.2f}</td><td class='best'>{float(d_def['bps']):.2f}</td>"
                f"<td>{def_err}</td><td>{nf0_bps:.2f}</td><td>{nf0_err}</td>"
                f"<td>{bin_bps:.2f}</td><td>{gain:+.1f}%</td><td>{spd}</td></tr>"
            )
        g_gain = 100.0 * (gavg["bin_nf0"] - gavg["nf0"]) / gavg["bin_nf0"] if gavg["bin_nf0"] > 0 else 0.0
        tbl.append(
            f"<tr><td><b>group avg ({int(gavg['n']):,})</b></td><td></td><td></td>"
            f"<td><b>{gavg['raw']:.2f}</b></td><td><b>{gavg['bshuf']:.2f}</b></td>"
            f"<td class='best'><b>{gavg['default']:.2f}</b></td><td></td>"
            f"<td><b>{gavg['nf0']:.2f}</b></td><td></td><td><b>{gavg['bin_nf0']:.2f}</b></td>"
            f"<td><b>{g_gain:+.1f}%</b></td><td></td></tr></table>"
        )
        parts.append(f"<h3 id='{slug(gname)}'>{html.escape(gname)}</h3>" + "".join(tbl))
    return "".join(parts)


def render_provenance_table(groups: dict[str, dict[str, float]], manifest: dict[str, Any]) -> str:
    group_meta: dict[str, dict[str, str]] = manifest.get("groups", {})
    files: list[dict[str, Any]] = manifest.get("files", [])
    by_group: dict[str, list[dict[str, Any]]] = {g: [] for g in groups}
    for f in files:
        by_group.setdefault(f["group"], []).append(f)

    h = [
        "<table><tr><th class='l'>dataset family</th><th>files</th><th>download MB</th>"
        "<th class='l'>license</th><th class='l'>source &amp; citation</th></tr>"
    ]
    tot_files = 0
    tot_bytes = 0
    for gname in groups:
        meta = group_meta.get(gname, {})
        gfiles = by_group.get(gname, [])
        n_files = len(gfiles)
        sz = sum(int(f.get("size_bytes", 0)) for f in gfiles)
        tot_files += n_files
        tot_bytes += sz
        url = meta.get("source_url", "")
        src = meta.get("source", gname)
        lic = meta.get("license", "")
        cite = meta.get("citation", "")
        src_html = f"<a href='{html.escape(url)}'>{html.escape(src)}</a>" if url else html.escape(src)
        h.append(
            f"<tr><td><a href='#{slug(gname)}'>{html.escape(gname)}</a></td>"
            f"<td>{n_files}</td><td>{sz / 1e6:.2f}</td>"
            f"<td class='l'>{html.escape(lic)}</td>"
            f"<td class='l' style='white-space:normal;min-width:340px'>{src_html}<br>"
            f"<span class='muted'>{html.escape(cite)}</span></td></tr>"
        )
    h.append(
        f"<tr><td><b>total</b></td><td><b>{tot_files}</b></td><td><b>{tot_bytes / 1e6:.1f}</b></td>"
        f"<td class='l'></td><td class='l muted'>All {tot_files} files verified by SHA-256 in "
        f"<code>experimental/public_datasets/datasets.json</code>.</td></tr></table>"
    )
    return collapsible_table(f"Show dataset provenance & licenses ({len(groups)} families, {tot_files} files)", "".join(h))


def main() -> None:
    results, manifest, codec_data = load_inputs()
    groups: dict[str, dict[str, float]] = results["groups"]
    series_rows: list[dict[str, Any]] = results["series"]
    ts_rows: list[dict[str, Any]] = results["timestamps"]

    clean_outputs(OUT, ("public.html", "public-*.svg"))

    fig_scatter = plot_real_scatter(codec_data) if codec_data else ""
    fig_rd = plot_real_rd(codec_data) if codec_data else ""
    fig_bps = plot_bps(groups)
    fig_grid = plot_grid_gain(series_rows)
    fig_tp = plot_throughput(groups, series_rows)
    fig_ts = plot_timestamps(ts_rows)

    tot_series = len(series_rows)
    tot_samples = sum(int(r["n"]) for r in series_rows)
    tot_mb = 8.0 * tot_samples / 1e6

    intro = f"""<p>Real-world value compression benchmarks across <b>{len(groups)} public dataset families</b>
(<b>{tot_series} series</b>, <b>{tot_samples:,} float64 samples</b>, {tot_mb:.1f} MB uncompressed), covering
high-frequency bearing vibration, clinical ECG/PPG waveforms, 24-bit broadband seismology, multi-rate industrial PLC
telemetry, residential smart-meter power, oceanographic buoys, float32-originated ERA5 weather, irregular millisecond
exchange trades, jittered smartphone IMU clocks, and the VLDB/SIGMOD floating-point benchmark (Elf / Chimp / ALP).
Every block group passed <code>fluxcode.verify.encode_group</code> with <b>0 errors</b>.</p>
<p class='muted'>For idealized and extreme synthetic stress tests (pure sines, square waves, chirps, impulses), see
<a href='codecs.html'>Synthetic &amp; edge-case codec comparison</a> under <a href='experiments.html'>Experiments</a>.
Regenerate with <code>make datasets &amp;&amp; make public-bench &amp;&amp; make public-report</code>.</p>"""

    body: list[str] = []

    if codec_data:
        body.append(f"""<h2 id='scatter'>Size vs. error on real-world datasets</h2>
<p>Comparing <code>fluxcode</code> against classic, constant-width, entropy, and lossless codecs on 9 representative
real-world series (up to 60 blocks of 1,000 samples per series). The purple line sweeps <code>fluxcode</code>'s bit
budget from B = 6 to 16 with the noise floor off; the dark diamond is default <code>Params()</code>, and the light
diamond forces a binary-only grid. Markers below or to the left of the purple curve beat <code>fluxcode</code> at the
same error or size. <code>gorilla-xor</code> is omitted from the plot because it sits around 40–60 bits/sample, but it
is included in the table below.</p>
{fig_scatter}
<h3 id='scatter-per-series'>Per dataset</h3>
<p>Each panel shows one real-world series on the same axes. Lossless points (RMSE ≤ 10<sup>−7</sup>%) sit on the
<code>0*</code> band at the bottom.</p>
{fig_rd}
{render_codec_matrix(codec_data)}""")

    # 1. Summary & bshuf vs raw zstd insight
    body.append(f"""<h2 id='summary'>Group summary &amp; compression ratios</h2>
<p>All sizes are sample-weighted <b>bits per sample</b> over whole block groups (60 blocks of 1,000 samples,
where raw <code>float64</code> is 64 bits/sample). The baselines are <code>raw zstd</code> (zstd level 3 on
raw <code>float64</code> bytes) and <code>bshuf zstd</code> (byte-shuffled <code>float64</code> followed by
zstd level 3).</p>
{fig_bps}
{render_group_summary_table(groups)}
<h3>Why byte-shuffled zstd loses to raw zstd on calibrated-ADC &amp; decimal float64s</h3>
<p>Across several datasets, <code>bshuf zstd</code> is <b>2–3× larger</b> than <code>raw zstd</code>—for example,
52.48 vs 18.33 bps on CWRU Vibration, 29.63 vs 9.48 bps on MIT-BIH ECG, and 16.63 vs 6.93 bps on UCI Power.
Calibrated-ADC and decimal-rounded series tend to have relatively few distinct values (CWRU
<code>normal_48k_DE</code> has only 2,254 unique values across 485,643 samples), so <code>raw zstd</code>'s
LZ77 matcher can copy entire repeated 8-byte floats in a single match. Byte-shuffling splits those 8-byte values
across separate streams, and for non-power-of-two steps like <code>0.005</code> or <code>2.086e-4</code>, the
six low mantissa bytes look nearly random on their own and no longer match together. <code>fluxcode</code>
improves on both baselines by <b>2–7×</b> because it recovers the underlying integer grid first, then applies
per-block delta coding (orders 0–3) and bit-plane zstd.</p>""")

    # 2. Hidden rational/ADC grids & float32 recovery
    body.append(f"""<h2 id='grids'>Hidden rational/ADC grids &amp; float32 recovery</h2>
<p>Comparing <code>fluxcode nf=0</code> (<code>Params(noise_floor_sigma=0)</code>) against
<code>fluxcode bin nf=0</code> (<code>grids='binary'</code>) isolates the savings from decimal, rational, and
scaled-ADC grid detection on real data.</p>
{fig_grid}
<ul>
<li><b>CWRU Vibration (ADC calibration step):</b> Raw accelerometer waveforms sit on a ~<code>2.086e-4</code>
calibration step. Scaled-grid detection saves <b>+24.5%</b> across the group (up to <b>+32.1%</b> on
<code>normal_48k_DE</code>) at ~1 ULP (<code>2.8e-17</code>) error when <code>noise_floor_sigma=0</code>.</li>
<li><b>MIT-BIH ECG (<code>0.005</code> mV ADC step):</b> 11-bit WFDB-212 counts scaled by 200 counts/mV land
on an exact scaled decimal grid, compressing to <b>4.14 bps bit-exact</b> across all 2.6M samples, which is
<b>+56.7%</b> smaller than binary-only grids (9.57 bps).</li>
<li><b>UCI Appliances (averaged <code>1/3</code> &amp; <code>1/4</code> rational grids):</b> 10-minute averaged
weather and indoor sensors (<code>T_out</code>, <code>Press_mm_hg</code>, <code>Windspeed</code>) sit on exact
rational multiples of 1/3 and 1/4, compressing to <b>0.92–1.31 bps</b> at 0.0000% error (<b>+54.6% to +66.7%</b>
smaller than binary grids).</li>
<li><b>UCI Hydraulic (multi-rate industrial PLC):</b> All 5 channels (100 Hz pressure/motor power, 10 Hz flow,
1 Hz temperature/vibration; 3.99M samples) decode <b>bit-exact</b> at <code>nf=0</code> with a
<b>+33.4%</b> group-wide grid gain (up to +54.7% on <code>VS1_1Hz</code>).</li>
<li><b>ERA5 Weather (float32-to-decimal recovery):</b> <code>temp_2m_f32</code> and <code>pressure_f32</code>
are cast through <code>float32</code> and promoted back to <code>float64</code>, introducing ~10<sup>−7</sup>
binary rounding noise. <code>fluxcode</code> recovers the exact decimal grid at the <b>same</b> bit rates as
the clean float64 originals (<b>4.95 bps</b> and <b>4.10 bps</b>, <b>+52.4%</b> and <b>+57.9%</b> smaller than
binary grids).</li>
</ul>""")

    # 3. Hardware anti-alias, effort tradeoffs, sentinels vs NaN
    body.append("""<h2 id='insights'>Real-world signal findings</h2>
<ul>
<li><b>Band-limited sigma-delta ADCs (IRIS 24-bit seismic):</b> The 40 Hz broadband seismometer
(<code>IU.ANMO.00</code> during the 2023-02-06 M7.8 Kahramanmaras earthquake) uses a 24-bit sigma-delta ADC
with a steep hardware anti-alias filter. Because the signal is smooth and band-limited, higher-order delta
predictors work well: raw counts compress to <b>4.16–4.37 bps</b> (0.002% max error), and calibrated ground
velocity (<code>ANMO_BHZ_mps</code>, ~10<sup>−4</sup> m/s scale) compresses to <b>4.96 bps</b> (0.023% max
error), which is <b>7× smaller</b> than <code>raw zstd</code> (34.46 bps) and <b>9× smaller</b> than
<code>bshuf zstd</code> (44.95 bps).</li>
<li><b>Effort tradeoffs (BIDMC clinical):</b> On most datasets, <code>effort=1</code> (fastest) lands within
1–2% of the default <code>effort=4</code>. On 125 Hz clinical waveforms (BIDMC PPG, respiration, and ECG),
<code>effort=4</code> reaches <b>4.84 bps</b> versus <b>5.57 bps</b> at <code>effort=1</code> (a <b>13%</b>
reduction from better predictor and codec selection), while <code>effort=5</code> improves that by only
0.8% (4.80 bps).</li>
<li><b>Decimal sentinels vs <code>NaN</code> (NOAA Buoy):</b> Legacy buoy feeds encode missing values as
in-band sentinels (<code>99.0</code>, <code>999.0</code>, <code>9999.0</code>), which inflate the block range
and create large delta spikes. Replacing those sentinels with IEEE <code>NaN</code> moves missingness into
<code>fluxcode</code>'s non-finite mask: <code>WTMP</code> drops from <b>1.92 bps</b> (<code>WTMP_raw</code>)
to <b>1.28 bps</b> (<code>WTMP_nan</code>, −33%), and <code>WVHT</code> (66.7% missing) drops from
<b>4.08 bps</b> to <b>2.79 bps</b> (−32%), while remaining <b>bit-exact</b> on every finite sample.</li>
</ul>""")

    # 4. Throughput
    body.append(f"""<h2 id='throughput'>Encode &amp; decode throughput</h2>
<p>Single-threaded throughput (<code>NUMBA_NUM_THREADS=1</code>, default <code>Params()</code>) is measured in
MB/s of uncompressed <code>float64</code> input (8 bytes/sample). On full-length series, decode runs at
<b>2.5–5.0 GB/s</b> and encode at <b>400–1,000 MB/s</b>. Smartphone IMU is slower (140 MB/s encode, 480 MB/s
decode) because each trial contains only 761–1,008 samples, so per-call Python and zstd setup overhead dominates.</p>
{fig_tp}""")

    # 5. Timestamp axis on real clocks
    body.append(f"""<h2 id='timestamps'>Timestamp axis on real clocks</h2>
<p>This comparison measures the extra bits per sample needed to store timestamps alongside values, comparing
exact timestamps against tolerance-rounded timestamps (<code>time_error=0.05</code> and <code>0.15</code>)
across regular, gapped, jittered, and event-driven clocks.</p>
{fig_ts}
{render_timestamp_table(ts_rows)}
<ul>
<li><b>Regular and gapped clocks add minimal overhead:</b> UCI Power (2.08M rows, 1-min), UCI AirQuality (1-h),
UCI Appliances (10-min), ERA5 (1-h), and NOAA Buoy (10-min with gaps) add only <b>+0.006 to +0.028 bps</b>.
SKAB's 1 Hz industrial PLC clock has occasional drops and drift, adding <b>+0.863 bps</b>.</li>
<li><b>Jittered smartphone IMU clocks benefit from <code>time_error</code>:</b> Real ~50 Hz phone clocks
(<code>dt</code> 16–177 ms) cost <b>+7.92 to +8.98 bps</b> when stored exactly. Setting
<code>Params(time_error=0.15)</code> smooths out the hardware jitter and saves <b>~1.65–1.68 bps</b> while
preserving sample order and keeping every timestamp within tolerance.</li>
<li><b>Irregular event clocks are left exact (Binance trades):</b> Without timestamps, Binance trades average
<b>4.14 bps</b> across price and quantity, with <code>BTCUSDT_price</code> at <b>3.03 bps bit-exact</b> on
the $0.01 grid. Adding 1.11M exact irregular millisecond arrival timestamps adds <b>+5.973 bps</b>
(<b>9.00 bps</b> total for exact cent prices and millisecond timestamps). Because trade arrivals have no
periodic cadence, <code>time_error=0.05</code> and <code>0.15</code> leave them unmodified instead of
distorting event timing.</li>
</ul>""")

    # 6. Per-series tables
    body.append(f"""<h2 id='series'>Per-series results (all {tot_series} series)</h2>
<p class='muted'><code>fc default</code> = <code>Params()</code>; <code>fc nf=0</code> =
<code>Params(noise_floor_sigma=0)</code>; <code>fc bin nf=0</code> =
<code>Params(noise_floor_sigma=0, grids='binary')</code>. Error columns show worst block max absolute error
as % of block range (or <code>EXACT</code> when bit-exact across all samples).</p>
{render_series_tables(groups, series_rows)}""")

    # 7. Dataset provenance / licenses / citations
    body.append(f"""<h2 id='provenance'>Dataset provenance, licenses &amp; citations</h2>
<p>Datasets are not committed to git; <code>experimental/public_datasets/datasets.json</code> pins all 38
source URLs, extraction modes, and SHA-256 checksums. Run <code>make datasets</code> (or
<code>uv run python experimental/public_datasets/download.py --verify</code>) to fetch or verify them locally.</p>
{render_provenance_table(groups, manifest)}""")

    toc = (
        [("scatter", "Size vs. error on real-world datasets", [("scatter-per-series", "Per dataset")])]
        if codec_data
        else []
    ) + [
        ("summary", "Group summary & compression ratios", []),
        ("grids", "Hidden rational/ADC grids & float32 recovery", []),
        ("insights", "Real-world signal findings", []),
        ("throughput", "Encode & decode throughput", []),
        ("timestamps", "Timestamp axis on real clocks", []),
        ("series", f"Per-series results ({tot_series} series)", [(slug(g), g) for g in groups]),
        ("provenance", "Dataset provenance, licenses & citations", []),
    ]

    out_html = OUT / "public.html"
    out_html.write_text(
        shell("Benchmarks: real-world datasets", intro, toc, "\n".join(body), active="benchmarks"),
        encoding="utf-8",
    )
    print(f"Wrote {out_html} (+ {6 if codec_data else 4} SVG charts)")


if __name__ == "__main__":
    main()
