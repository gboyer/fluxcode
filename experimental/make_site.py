# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Generate the overview homepage (report/index.html) and rendered format spec (report/spec.html).

    cd experimental && uv run python make_site.py
"""

from __future__ import annotations

import html
import re
from pathlib import Path

from tslab.report.page import shell

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT = HERE / "report"
SPEC_MD = REPO / "docs" / "SPEC.md"
GITHUB_DOCS = "https://github.com/gboyer/fluxcode/blob/main/docs/"


def slugify(text: str) -> str:
    plain = re.sub(r"`([^`]*)`", r"\1", text)
    plain = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", plain)
    plain = re.sub(r"[^a-zA-Z0-9\s-]", "", plain).strip().lower()
    return re.sub(r"[\s-]+", "-", plain)


def render_inline(text: str) -> str:
    """Render inline Markdown (code, links, bold, italic) to HTML."""
    tokens: list[str] = []

    def stash(html_frag: str) -> str:
        idx = len(tokens)
        tokens.append(html_frag)
        return f"\x00TOK{idx}\x00"

    # Inline code first so nothing inside backticks is parsed further
    def repl_code(m: re.Match[str]) -> str:
        return stash(f"<code>{html.escape(m.group(1))}</code>")

    text = re.sub(r"`([^`]+)`", repl_code, text)

    # Links [label](url)
    def repl_link(m: re.Match[str]) -> str:
        label = render_inline(m.group(1))
        url = m.group(2).strip()
        if not url.startswith(("http://", "https://", "#", "/")):
            url = GITHUB_DOCS + url
        return stash(f"<a href='{html.escape(url, quote=True)}'>{label}</a>")

    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", repl_link, text)

    escaped = html.escape(text)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", escaped)

    for idx, frag in enumerate(tokens):
        escaped = escaped.replace(f"\x00TOK{idx}\x00", frag)
    return escaped


def markdown_to_html(md: str) -> tuple[str, list[tuple[str, str, list[tuple[str, str]]]]]:
    """Convert GitHub-flavored Markdown (SPEC.md) to HTML and extract a 2-level TOC."""
    lines = md.splitlines()
    out: list[str] = []
    toc: list[tuple[str, str, list[tuple[str, str]]]] = []
    used_ids: dict[str, int] = {}

    def unique_id(title: str) -> str:
        base = slugify(title) or "section"
        count = used_ids.get(base, 0)
        used_ids[base] = count + 1
        return base if count == 0 else f"{base}-{count}"

    i = 0
    n = len(lines)
    # Skip leading H1 (rendered by shell)
    if i < n and lines[i].startswith("# "):
        i += 1

    while i < n:
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        # Fenced code block
        if stripped.startswith("```"):
            i += 1
            code_lines: list[str] = []
            while i < n and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            if i < n:
                i += 1
            out.append(f"<pre><code>{html.escape(chr(10).join(code_lines))}</code></pre>")
            continue

        # Horizontal rule
        if re.fullmatch(r"[-*_]{3,}", stripped):
            out.append("<hr>")
            i += 1
            continue

        # Headings
        hm = re.match(r"^(#{2,6})\s+(.*)$", line)
        if hm:
            level = len(hm.group(1))
            raw_title = hm.group(2).strip()
            anchor = unique_id(raw_title)
            if level == 2:
                toc.append((anchor, re.sub(r"`([^`]*)`", r"\1", raw_title), []))
            elif level == 3 and toc:
                toc[-1][2].append((anchor, re.sub(r"`([^`]*)`", r"\1", raw_title)))
            out.append(f"<h{level} id='{anchor}'>{render_inline(raw_title)}</h{level}>")
            i += 1
            continue

        # Pipe table
        if "|" in line and i + 1 < n and re.match(r"^\s*\|?[\s:|-]+\|[\s:|-]*$", lines[i + 1]):
            def split_row(row: str) -> list[str]:
                r = row.strip()
                if r.startswith("|"):
                    r = r[1:]
                if r.endswith("|"):
                    r = r[:-1]
                return [c.strip() for c in r.split("|")]

            headers = split_row(line)
            aligns_raw = split_row(lines[i + 1])
            aligns = []
            for a in aligns_raw:
                if a.startswith(":") and a.endswith(":"):
                    aligns.append(" style='text-align:center'")
                elif a.endswith(":"):
                    aligns.append(" style='text-align:right'")
                else:
                    aligns.append(" class='l'")
            i += 2
            rows: list[list[str]] = []
            while i < n and "|" in lines[i] and lines[i].strip():
                rows.append(split_row(lines[i]))
                i += 1
            th = "".join(
                f"<th{aligns[c] if c < len(aligns) else ' class=l'}>{render_inline(cell)}</th>"
                for c, cell in enumerate(headers)
            )
            trs = []
            for row in rows:
                tds = "".join(
                    f"<td{aligns[c] if c < len(aligns) else ' class=l'}>{render_inline(cell)}</td>"
                    for c, cell in enumerate(row)
                )
                trs.append(f"<tr>{tds}</tr>")
            out.append(f"<table><tr>{th}</tr>{''.join(trs)}</table>")
            continue

        # Blockquote
        if stripped.startswith(">"):
            bq_lines: list[str] = []
            while i < n and lines[i].strip().startswith(">"):
                bq_lines.append(re.sub(r"^>\s?", "", lines[i].strip()))
                i += 1
            out.append(f"<blockquote><p>{render_inline(' '.join(bq_lines))}</p></blockquote>")
            continue

        # Lists (unordered or ordered, with simple 1-level nesting)
        ul_m = re.match(r"^(\s*)[-*]\s+(.*)$", line)
        ol_m = re.match(r"^(\s*)\d+\.\s+(.*)$", line)
        if ul_m or ol_m:
            tag = "ul" if ul_m else "ol"
            base_indent = len((ul_m or ol_m).group(1))
            items: list[str] = []
            while i < n:
                cur = lines[i]
                if not cur.strip():
                    # Peek ahead: if next non-empty line continues the same list, keep going
                    j = i + 1
                    while j < n and not lines[j].strip():
                        j += 1
                    if j < n and (
                        (tag == "ul" and re.match(r"^\s*[-*]\s+", lines[j]))
                        or (tag == "ol" and re.match(r"^\s*\d+\.\s+", lines[j]))
                    ):
                        i = j
                        continue
                    break
                cur_ul = re.match(r"^(\s*)[-*]\s+(.*)$", cur)
                cur_ol = re.match(r"^(\s*)\d+\.\s+(.*)$", cur)
                if cur_ul and len(cur_ul.group(1)) > base_indent:
                    # Nested unordered sub-list
                    sub_items: list[str] = []
                    sub_indent = len(cur_ul.group(1))
                    while i < n:
                        sm = re.match(r"^(\s*)[-*]\s+(.*)$", lines[i])
                        if sm and len(sm.group(1)) >= sub_indent:
                            sub_items.append(f"<li>{render_inline(sm.group(2))}</li>")
                            i += 1
                        else:
                            break
                    if items:
                        items[-1] = items[-1][:-5] + f"<ul>{''.join(sub_items)}</ul></li>"
                    continue
                if tag == "ul" and cur_ul and len(cur_ul.group(1)) == base_indent:
                    item_lines = [cur_ul.group(2)]
                    i += 1
                    while i < n and lines[i].strip() and not re.match(r"^\s*([-*]|\d+\.)\s+", lines[i]) and not lines[i].startswith("#"):
                        item_lines.append(lines[i].strip())
                        i += 1
                    items.append(f"<li>{render_inline(' '.join(item_lines))}</li>")
                    continue
                if tag == "ol" and cur_ol and len(cur_ol.group(1)) == base_indent:
                    item_lines = [cur_ol.group(2)]
                    i += 1
                    while i < n and lines[i].strip() and not re.match(r"^\s*([-*]|\d+\.)\s+", lines[i]) and not lines[i].startswith("#"):
                        item_lines.append(lines[i].strip())
                        i += 1
                    items.append(f"<li>{render_inline(' '.join(item_lines))}</li>")
                    continue
                break
            out.append(f"<{tag}>{''.join(items)}</{tag}>")
            continue

        # Paragraph
        para_lines: list[str] = [stripped]
        i += 1
        while i < n:
            nxt = lines[i]
            ns = nxt.strip()
            if (
                not ns
                or ns.startswith(("```", "#", ">"))
                or re.fullmatch(r"[-*_]{3,}", ns)
                or re.match(r"^\s*([-*]|\d+\.)\s+", nxt)
                or ("|" in nxt and i + 1 < n and re.match(r"^\s*\|?[\s:|-]+\|[\s:|-]*$", lines[i + 1]))
            ):
                break
            para_lines.append(ns)
            i += 1
        out.append(f"<p>{render_inline(' '.join(para_lines))}</p>")

    return "\n".join(out), toc


def write_spec() -> None:
    md = SPEC_MD.read_text(encoding="utf-8")
    body, toc = markdown_to_html(md)
    intro = (
        "<p class='muted'>Rendered from "
        "<a href='https://github.com/gboyer/fluxcode/blob/main/docs/SPEC.md'><code>docs/SPEC.md</code></a>. "
        "See also <a href='https://github.com/gboyer/fluxcode/blob/main/docs/ENCODER.md'><code>ENCODER.md</code></a> "
        "and <a href='https://github.com/gboyer/fluxcode/blob/main/docs/TUNING.md'><code>TUNING.md</code></a>.</p>"
    )
    (OUT / "spec.html").write_text(
        shell("fluxcode Format Specification", intro, toc, body, active="spec"),
        encoding="utf-8",
    )
    print(f"Wrote {OUT / 'spec.html'}")


def write_experiments() -> None:
    intro = """<p>Side experiments, synthetic stress tests, and applied case studies that informed the design of
<code>fluxcode</code> and <code>fluxprune</code>. These sit outside the main value, time, and cardinality benchmarks
so you can dig into extreme behaviors and design history when you want to.</p>"""

    cards = """<div class='card-grid'>
<a class='card' href='codecs.html'>
  <span class='tag'>Synthetic stress tests</span>
  <h3>Synthetic &amp; Edge-Case Codec Comparison</h3>
  <p>All 34 codecs across 12 synthetic 1 kHz signals—including idealized waves (pure sines, square waves, chirps, linear
  and quadratic ramps) alongside spikes, impulses, and random walks to expose extreme behaviors like slope overload and outlier clipping.</p>
</a>
<a class='card' href='sdt.html'>
  <span class='tag'>Applied case study</span>
  <h3>Swinging-Door Historian (SDT)</h3>
  <p>Encoding swinging-door historian archives with <code>fluxcode</code>, compared against zstd, Gorilla, resampled
  exports, and storing every scan directly.</p>
</a>
<a class='card' href='rate.html'>
  <span class='tag'>Design experiment</span>
  <h3>Rate Control</h3>
  <p>Windowed vs. fixed-rate quantize-predict-entropy experiments on mixed-regime minutes—the study behind
  <code>fluxcode</code>'s one-shot quantization design.</p>
</a>
</div>"""

    (OUT / "experiments.html").write_text(
        shell("Experiments & Case Studies", intro, [], cards, active="experiments"),
        encoding="utf-8",
    )
    print(f"Wrote {OUT / 'experiments.html'}")


def write_overview() -> None:
    intro = """<p>This repository contains two complementary tools for scientific and engineering time series:
<b><code>fluxcode</code></b> for storage compression and <b><code>fluxprune</code></b> for volume reduction. You can use
either one on its own, or chain them together—pruning a high-volume series first and then packing the remaining
<code>(time, value)</code> points with <code>fluxcode</code>.</p>
<div class='two-col'>
  <div class='pillar'>
    <h3><code>fluxcode</code> — reduce storage with minimal loss</h3>
    <p>Packs <code>float64</code> values and optional exact <code>int64</code> timestamps into self-describing block
    groups. Every sample is preserved within a strict error bound, exact binary, decimal, and rational/ADC grids decode
    bit-for-bit, and decoding runs at multi-GB/s on a single core.</p>
  </div>
  <div class='pillar'>
    <h3><code>fluxprune</code> — reduce data volume itself</h3>
    <p>A pre-pass downsampler that drops redundant points and keeps the subset that carries a series' shape for linear
    interpolation. Fewer stored points means faster scans, plots, and downstream analytics.</p>
  </div>
</div>"""

    cards = """<h2 id='explore'>Three dimensions, plus spec &amp; experiments</h2>
<div class='card-grid'>
<a class='card' href='public.html'>
  <span class='tag'>Value compression</span>
  <h3>Benchmarks</h3>
  <p>Real-world size-vs-error scatterplots comparing <code>fluxcode</code> against competing codecs on public datasets,
  plus hidden ADC/rational grid recovery and throughput across 14 dataset families (~29M samples).</p>
</a>
<a class='card' href='time.html'>
  <span class='tag'>Time compression</span>
  <h3>Timestamps</h3>
  <p>Exact and tolerance-rounded (<code>time_error</code>) timestamp storage across regular grids, gapped clocks,
  jittered sensors, and irregular event streams.</p>
</a>
<a class='card' href='prune.html'>
  <span class='tag'>Cardinality reduction</span>
  <h3>Fluxprune</h3>
  <p>Point-selection quality, stopping rules, noise floor, and speed compared against optimal DP, swinging-door
  trending, Visvalingam-Whyatt, and LTTB.</p>
</a>
<a class='card' href='spec.html'>
  <span class='tag'>Reference</span>
  <h3>Format Specification</h3>
  <p>The normative binary layout and design rationale for <code>fluxcode</code> block groups
  (<code>docs/SPEC.md</code>).</p>
</a>
<a class='card' href='experiments.html'>
  <span class='tag'>Deep dives</span>
  <h3>Experiments &amp; Case Studies</h3>
  <p>Synthetic and edge-case codec stress tests, a swinging-door historian (SDT) case study showing how pruning and
  <code>fluxcode</code> interact, and rate-control experiments.</p>
</a>
</div>"""

    highlights = """<h2 id='highlights'>Key results at a glance</h2>
<figure class='figure-card'>
  <a href='public.html#scatter'><img src='public-scatter.svg' alt='Real-world bits per sample vs RMSE across codecs'></a>
  <figcaption><b>Value compression (real-world size vs. error):</b> median bits per sample vs. median RMSE across 9
  diverse public datasets (vibration, ECG, PPG, seismic, hydraulic PLC, smart meters, weather, and trades). Purple lines
  sweep <code>fluxcode</code> from B = 6 to 16; the diamond marks default settings.
  <a href='public.html#scatter'>View full benchmarks &rarr;</a></figcaption>
</figure>
<figure class='figure-card'>
  <a href='public.html#summary'><img src='public-bps.svg' alt='Bits per sample across 14 public dataset families'></a>
  <figcaption><b>Real-world compression ratios:</b> sample-weighted bits per sample across all 14 public dataset families
  compared against raw zstd and byte-shuffled zstd.
  <a href='public.html#summary'>View all 14 families &rarr;</a></figcaption>
</figure>
<figure class='figure-card'>
  <a href='prune.html#curves'><img src='prune-curves.svg' alt='Fluxprune error vs points kept'></a>
  <figcaption><b>Cardinality reduction (<code>fluxprune</code>):</b> reconstruction error vs. points kept per hour
  compared against optimal DP, swinging-door trending, Visvalingam-Whyatt, and LTTB.
  <a href='prune.html#curves'>View fluxprune results &rarr;</a></figcaption>
</figure>
<h2 id='docs'>Source &amp; documentation</h2>
<ul>
  <li><a href='https://github.com/gboyer/fluxcode'>GitHub repository</a> — Python packages (<code>fluxcode</code>, <code>fluxprune</code>), optional Rust extension, and test suite.</li>
  <li><a href='spec.html'>Format specification</a> (<code>docs/SPEC.md</code>) — normative binary layout and design rationale.</li>
  <li><a href='https://github.com/gboyer/fluxcode/blob/main/docs/ENCODER.md'>Encoder design</a> (<code>docs/ENCODER.md</code>) — grid detection, noise floor, and predictor selection.</li>
  <li><a href='https://github.com/gboyer/fluxcode/blob/main/docs/TUNING.md'>Tuning guide</a> (<code>docs/TUNING.md</code>) — choosing <code>effort</code>, <code>noise_floor_sigma</code>, <code>time_error</code>, and grid modes.</li>
  <li><a href='https://github.com/gboyer/fluxcode/blob/main/docs/PERFORMANCE.md'>Performance notes</a> (<code>docs/PERFORMANCE.md</code>) — throughput measurements and architecture notes.</li>
</ul>"""

    (OUT / "index.html").write_text(
        shell("fluxcode & fluxprune", intro, [], cards + highlights, active="overview"),
        encoding="utf-8",
    )
    print(f"Wrote {OUT / 'index.html'}")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    write_spec()
    write_experiments()
    write_overview()


if __name__ == "__main__":
    main()
