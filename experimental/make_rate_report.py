# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Rate-control experiment: every coder and planner on one-minute block groups -> report/rate.html.

    uv run python make_rate_report.py

Each block group is a "regime" minute: 5 consecutive seconds of each of the 12 signal kinds, every block
rescaled to [0, 1], standing in for one sensor passing through different regimes. One block group per report
seed (REPORT_MINUTES). The bit budget applies per block group; errors are per block.

Ordering (same rules as make_report.py): coders in the order of tslab.entropy.ratectl.CODERS, each
with its rate controls window then fixed, then the one-shot codecs in their package order; datasets
in KINDS order. Every chart is checked for data at build time.
"""

import time
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tslab.common.datasets import BLOCK, KINDS, PER_CHUNK, REPORT_MINUTES, minute
from tslab.entropy.delta_zstd import DELTA_ZSTD_CODECS
from tslab.entropy.ratectl import (
    CAP_BPS,
    CODERS,
    TARGET_BPS,
    EntropyBound,
    baseline_quant8_deflate,
    evaluate,
    evaluate_delta_zstd,
    fixed_plan,
    window_plan,
)
from tslab.report.page import clean_outputs, save_fig, shell

OUT = Path(__file__).parent / "report"
PER_KIND = PER_CHUNK // len(KINDS)  # seconds of each kind in a regime block group


def regime_groups():
    """REPORT_MINUTES block groups of [60, 1000]: PER_KIND seconds of every kind (report seeds), each block in [0, 1]."""
    out = []
    for j in range(REPORT_MINUTES):
        X = np.concatenate([minute(k, 1000 * i + j).reshape(PER_CHUNK, BLOCK)[:PER_KIND] for i, k in enumerate(KINDS)])
        lo, hi = X.min(1, keepdims=True), X.max(1, keepdims=True)
        out.append((X - lo) / (hi - lo))
    return out


def merge(runs):
    """Several block groups' results as one: bits over all block groups, rows concatenated."""
    return {"delta": [r["delta"] for r in runs], "bps": float(np.mean([r["bps"] for r in runs])),
            "rows": [row for r in runs for row in r["rows"]],
            "us_per_block": float(np.mean([r["us_per_block"] for r in runs]))}


def main():
    groups = regime_groups()
    results = {}
    for coder in CODERS:
        for label, planner in (("window", window_plan), ("fixed", fixed_plan)):
            t0 = time.perf_counter()
            results[(coder.name, label)] = r = merge([evaluate(coder, U, planner) for U in groups])
            rows = r["rows"]
            print(f"{coder.name:20s} {label:6s} {r['bps']:.3f} b/s  median RMSE {100 * np.median([x['rmse'] for x in rows]):.4f}%  "
                  f"worst {100 * max(x['rmse'] for x in rows):.3f}%  "
                  f"capped {sum(1 for x in rows if x['e']) if label == 'window' else '-'}  ({time.perf_counter() - t0:.0f} s)",
                  flush=True)
    for codec in DELTA_ZSTD_CODECS:
        results[(codec.name, "one-shot")] = r = merge([evaluate_delta_zstd(codec, U) for U in groups])
        print(f"{codec.name:20s} one-shot {r['bps']:.3f} b/s  median RMSE {100 * np.median([x['rmse'] for x in r['rows']]):.4f}%  "
              f"worst {100 * max(x['rmse'] for x in r['rows']):.3f}%  {r['us_per_block']:.0f} µs/block")
    base = [baseline_quant8_deflate(U) for U in groups]
    base = {"bps": float(np.mean([b["bps"] for b in base])), "rows": [row for b in base for row in b["rows"]]}
    write_report(results, base)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

BEST = "best: rice|zstd|flac"
PER_KIND_TABLES = (("delta0123-zstd-10", "one-shot"), ("delta0123-zstd", "window"), (BEST, "window"), (BEST, "fixed"))


def sect_id(key):
    return "kind-" + "".join(ch if ch.isalnum() else "-" for ch in f"{key[0]}-{key[1]}").strip("-")


def kind_rows(rows):
    """Rows grouped by signal kind (groups are PER_KIND seconds of each kind, in KINDS order)."""
    out = {k: [] for k in KINDS}
    for i, row in enumerate(rows):
        out[KINDS[(i % PER_CHUNK) // PER_KIND]].append(row)
    return out


RMSE_FLOOR = 1e-9  # % of range; zero error has no place on a log axis, so it is drawn at the floor and labeled


def plot_blocks(results):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, layout="constrained")
    xs = np.arange(len(KINDS))
    greens = matplotlib.colormaps["Greens"]  # the entropy family's hue, shades by series
    series = [(BEST, "window", greens(0.95)), (BEST, "fixed", greens(0.75)),
              ("delta0123-zstd", "window", greens(0.55)), ("delta0123-zstd-10", "one-shot", greens(0.35))]
    w = 0.2
    for i, (c, l, col) in enumerate(series):
        by = kind_rows(results[(c, l)]["rows"])
        off = (i - (len(series) - 1) / 2) * w
        ax1.bar(xs + off, [np.median([r["bps"] for r in by[k]]) for k in KINDS], w, color=col, edgecolor="k", linewidth=0.3,
                label=f"{c} / {l}")
        rm = [100 * np.median([r["rmse"] for r in by[k]]) for k in KINDS]
        ax2.bar(xs + off, [max(v, RMSE_FLOOR) for v in rm], w, color=col, edgecolor="k", linewidth=0.3)
        for x0, v in zip(xs + off, rm):
            if v <= RMSE_FLOOR:
                ax2.text(x0, RMSE_FLOOR * 1.5, "0", ha="center", fontsize=6)
    ax1.axhline(TARGET_BPS, color="0.4", ls="--", lw=1, gid="ref")
    ax1.axhline(CAP_BPS, color="0.4", ls=":", lw=1, gid="ref")
    ax1.set(ylabel="bits / sample (block alone, median)", title="Bits per sample (dashed: 4 bits/sample budget, dotted: 8 bit cap)")
    ax2.set_yscale("log")
    ax2.set(ylabel="median RMSE (% of block range, log)", title="Error (0 = lossless, drawn at the floor)")
    ax2.set_xticks(xs, KINDS, rotation=30, ha="right", fontsize=8)
    fig.legend(*ax1.get_legend_handles_labels(), loc="outside lower center", ncol=4, fontsize=8, frameon=False)
    return save_fig(fig, OUT, "rate-blocks", "Per-kind bits and RMSE")


def write_report(results, base):
    clean_outputs(OUT, ("rate.html", "rate-blocks.*"))
    chart = plot_blocks(results)
    h = [f"""<p><a href="index.html">← codec comparison (index.html)</a></p>
<p>Window = one block group: a minute of 60 blocks, 5 consecutive seconds of each of the 12 signal kinds, every block
rescaled to [0, 1], standing in for one sensor passing through different regimes. {REPORT_MINUTES} groups
(one per report seed); bits/sample are whole block groups, every byte counted. Errors are absolute, which equals % of each
block's full scale; they are per block, summarized over all blocks (or a kind's blocks).
<b>Regenerate:</b> <code>cd experimental &amp;&amp; uv run python make_rate_report.py</code>.</p>
<h2 id="method">Method</h2>
<ul>
<li><b>Pipeline per block:</b> q = round(x / Δ<sub>b</sub>) (the only lossy step, so |err| ≤ Δ<sub>b</sub>/2 is guaranteed),
residual = diff(q, order) with order ∈ {0,1,2,3} chosen per block, then an entropy coder. Everything but the
Rice/FLAC bit-packing is vectorized numpy.</li>
<li><b>window:</b> one Δ for the whole block group, bisected so the block group averages 4 bits/sample including all headers.
A block that would exceed 8 bits/sample on its own gets its own coarser step Δ·2<sup>e/8</sup> (1-byte e per block).
With uniform quantization, one shared step is the MSE-optimal way to spend a total budget.</li>
<li><b>fixed:</b> every block individually driven to 4 bits/sample coded on its own (same format), for comparison;
the block group then costs what it costs.</li>
<li><b>Framing:</b> the rice coder stores its block payloads back to back with a length each; zstd and deflate make one
compressor call per block group (flags, then start values, then byte planes); flac is one stream per block group, one frame per block;
<code>best: …</code> stores a coder id per block, then each coder's block group of the blocks it won.</li>
<li><b>Coders:</b> <code>delta0123-rice</code> — FLAC-style partitioned Rice (2<sup>p</sup> partitions, own k each; order/p/k chosen by exact cost);
<code>delta0123-zstd</code> (level 9) / <code>delta0123-deflate</code> — zigzag residuals, narrowest byte width, byte-stream-split, then compress;
<code>flac</code> — real libFLAC level 8 (LPC up to order 12) on the same integers, frame bytes only;
<code>best: …</code> — per block, keep the coder whose standalone encoding is smallest (+1 byte coder id);
<code>0-order entropy</code> — zero-order entropy of the best residual + headers. A reference, not a codec and not a
true floor: partitioned Rice, LPC and LZ matching adapt within a block and can beat it.</li>
<li><b>delta0123-zstd-B:</b> no search — step = block range/(2<sup>B</sup>−1), order = argmin var(diff(q, k)),
one zstd (level 3) call per block group; lowers B if a block's estimated size would exceed 8 bits/sample.
<code>delta12-zstd-8</code> only considers orders 1 and 2 (performance option).</li>
<li>All real coders are round-trip verified per block group.</li>
</ul>"""]
    h.append("<h2 id='summary'>Summary</h2><table><tr><th class='l'>coder</th><th class='l'>rate control</th><th>avg bits/sample</th>"
             "<th>overall RMSE</th><th>median block RMSE</th><th>worst block RMSE</th><th>worst max |err|</th>"
             "<th>capped blocks</th><th>encode µs/block</th></tr>")
    def overall(rows):
        return np.sqrt(np.mean([x["rmse"] ** 2 for x in rows]))

    real = [k for k in results if k[0] != EntropyBound.name]
    best = min(overall(results[k]["rows"]) for k in real)
    for (cname, label), r in results.items():
        rows = r["rows"]
        cls = ' class="best"' if overall(rows) == best else ""
        capped = sum(1 for x in rows if x["e"]) if label in ("window", "one-shot") else "—"
        h.append(f"<tr><td><code>{cname}</code></td><td class='l'>{label}</td><td>{r['bps']:.3f}</td><td{cls}>{100 * overall(rows):.4f}%</td>"
                 f"<td>{100 * np.median([x['rmse'] for x in rows]):.4f}%</td>"
                 f"<td>{100 * max(x['rmse'] for x in rows):.3f}%</td>"
                 f"<td>{100 * max(x['maxabs'] for x in rows):.3f}%</td><td>{capped}</td>"
                 f"<td>{r['us_per_block']:.0f}</td></tr>")
    br = base["rows"]
    h.append(f"<tr><td class='muted'><code>quant8-delta1-deflate</code> (earlier codec)</td><td class='l muted'>none</td><td>{base['bps']:.3f}</td>"
             f"<td>{100 * overall(br):.4f}%</td>"
             f"<td>{100 * np.median([b['rmse'] for b in br]):.4f}%</td>"
             f"<td>{100 * max(b['rmse'] for b in br):.3f}%</td>"
             f"<td>{100 * max(b['maxabs'] for b in br):.3f}%</td><td>—</td><td>—</td></tr></table>"
             "<p class='muted'>One-shot rows include each block's min and max, stored in the block group. "
             "Encode time is the full rate-control search for window/fixed rows (Python, single thread), per block; "
             "one-shot is a single pass. Every block spans [0, 1], so one-shot's per-block step "
             "and the shared window step mean the same thing.</p>")
    h.append(f"<h2 id='per-kind-chart'>Bits and error per signal kind</h2>{chart}")

    for key in PER_KIND_TABLES:
        r = results[key]
        h.append(f"<h2 id='{sect_id(key)}'>Per signal kind: {key[0]} / {key[1]}</h2>"
                 + (f"<p>Block group Δ = {', '.join(f'{d:.3g}' for d in r['delta'])}</p>" if r["delta"][0] else "")
                 + f"<p class='muted'>Median over the kind's {PER_KIND * REPORT_MINUTES} blocks; bits/sample is the block "
                 "coded on its own (inside the block group it only has a share of the total).</p>"
                 "<table><tr><th>signal</th><th class='l'>order (most blocks)</th><th>step</th><th>bits/sample</th><th>RMSE</th>"
                 "<th>mean err</th><th>max err</th><th>min err</th><th>max |err|</th></tr>")
        for kind, rows in kind_rows(r["rows"]).items():
            def med(k, rows=rows):
                return np.median([x[k] for x in rows])
            common = Counter(x["order"] for x in rows).most_common(1)[0][0]
            h.append(f"<tr><td>{kind}</td><td class='l'>{common}</td><td>{med('step'):.3g}</td><td>{med('bps'):.2f}</td>"
                     f"<td>{100 * med('rmse'):.4f}%</td><td>{100 * med('mean'):+.4f}%</td><td>{100 * med('max'):+.4f}%</td>"
                     f"<td>{100 * med('min'):+.4f}%</td><td>{100 * med('maxabs'):.4f}%</td></tr>")
        h.append("</table>")
    toc = [("method", "Method", []), ("summary", "Summary", []), ("per-kind-chart", "Bits and error per signal kind", [])] + [
        (sect_id(k), f"Per signal kind: {k[0]} / {k[1]}", []) for k in PER_KIND_TABLES]
    (OUT / "rate.html").write_text(shell("Rate control: quantize → predict → entropy code", "", toc, "\n".join(h)))
    print(f"wrote {OUT / 'rate.html'}")


if __name__ == "__main__":
    main()
