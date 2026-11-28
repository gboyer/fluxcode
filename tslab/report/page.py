# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""HTML shell (inline CSS, light and dark, top nav, collapsible tables) and figure saving for the report pages."""

import html

import matplotlib.pyplot as plt

from tslab.report.order import assert_data

plt.rcParams["svg.fonttype"] = "none"  # text stays text: smaller and crisper than PNG

SITE_NAV = [
    ("overview", "Overview", "index.html"),
    ("codecs", "Codecs", "codecs.html"),
    ("public", "Public Data", "public.html"),
    ("time", "Timestamps", "time.html"),
    ("sdt", "Historian", "sdt.html"),
    ("prune", "Fluxprune", "prune.html"),
    ("rate", "Rate Control", "rate.html"),
    ("spec", "Spec", "spec.html"),
]

CSS = """
:root{--bg:#fff;--fg:#1a1a1a;--muted:#666;--line:#ddd;--surface:#f7f7f8;--surface-hover:#eeeef0;--accent:#2a5bd7;--accent-bg:#eaf0ff;--hl:#dcefdc;--l1:#e8f3e8;--l2:#fff3cc;--l3:#ffe0cc;--l4:#ffcccc}
@media (prefers-color-scheme:dark){:root{--bg:#161616;--fg:#e8e8e8;--muted:#999;--line:#333;--surface:#202022;--surface-hover:#2a2a2e;--accent:#7aa2ff;--accent-bg:#1c2740;--hl:#254025;--l1:#1f2e1f;--l2:#3a3418;--l3:#40291a;--l4:#452020}}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,-apple-system,sans-serif;max-width:1200px;margin:0 auto;padding:0 20px 48px}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
header.site-header{display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:12px;padding:14px 0;margin-bottom:20px;border-bottom:1px solid var(--line)}
header.site-header .brand{font-weight:700;font-size:16px;color:var(--fg);letter-spacing:-0.01em}
header.site-header .brand:hover{text-decoration:none}
nav.site-nav{display:flex;flex-wrap:wrap;gap:4px}
nav.site-nav a{padding:4px 10px;border-radius:6px;font-size:13.5px;color:var(--muted);font-weight:500}
nav.site-nav a:hover{background:var(--surface-hover);color:var(--fg);text-decoration:none}
nav.site-nav a.active{background:var(--accent-bg);color:var(--accent);font-weight:600}
h1{font-size:28px;line-height:1.2;margin:8px 0 12px;letter-spacing:-0.02em}
h2{margin-top:36px;border-top:1px solid var(--line);padding-top:16px;font-size:20px}
h3{margin-top:24px;font-size:16.5px}
table{border-collapse:collapse;margin:8px 0 12px;font-size:13.5px;font-variant-numeric:tabular-nums;display:block;overflow-x:auto}
th,td{border:1px solid var(--line);padding:4px 8px;text-align:right;white-space:nowrap}
th{background:var(--surface)}
th:first-child,td:first-child,td.l,th.l{text-align:left}
td.best{background:var(--hl);font-weight:600}
td.r1{background:var(--l1)}td.r2{background:var(--l2)}td.r3{background:var(--l3)}td.r4{background:var(--l4)}
tr.fam td:first-child{border-left:4px solid var(--c,var(--line))}
img{max-width:100%;height:auto;background:#fff;border-radius:4px}
nav.toc{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:12px 16px;margin:18px 0 24px}
nav.toc h2{margin:0 0 8px;padding:0;border:0;font-size:14px;text-transform:uppercase;letter-spacing:0.04em;color:var(--muted)}
nav.toc ul{columns:2;list-style:none;padding-left:0;margin:0;font-size:14px}
nav.toc li{margin:3px 0;break-inside:avoid}
nav.toc ul ul{columns:1;padding-left:16px;font-size:13px}
@media (max-width:640px){nav.toc ul{columns:1}}
details.data-table{margin:12px 0 18px;border:1px solid var(--line);border-radius:6px;background:var(--surface)}
details.data-table>summary{cursor:pointer;padding:8px 12px;font-size:13.5px;font-weight:500;color:var(--muted);user-select:none}
details.data-table>summary:hover{color:var(--fg)}
details.data-table[open]>summary{border-bottom:1px solid var(--line);color:var(--fg)}
details.data-table>.table-wrap{padding:8px 12px;background:var(--bg);border-radius:0 0 6px 6px}
.card-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(290px,1fr));gap:14px;margin:20px 0 28px}
.card{display:block;padding:16px;border:1px solid var(--line);border-radius:8px;background:var(--surface);color:var(--fg);transition:border-color 0.15s}
.card:hover{border-color:var(--accent);text-decoration:none}
.card h3{margin:0 0 6px;font-size:16px;color:var(--accent)}
.card p{margin:0;font-size:13.5px;color:var(--muted);line-height:1.45}
.figure-card{margin:18px 0 24px;padding:14px;border:1px solid var(--line);border-radius:8px;background:var(--surface)}
.figure-card figcaption{margin-top:8px;font-size:13.5px;color:var(--muted)}
pre{background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:12px 14px;overflow-x:auto;font-size:13px;line-height:1.45}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:0.9em}
pre code{font-size:13px}
blockquote{margin:12px 0;padding:8px 14px;border-left:3px solid var(--line);color:var(--muted)}
hr{border:0;border-top:1px solid var(--line);margin:24px 0}
.muted{color:var(--muted)}
"""


def nav_bar(active=None):
    """Top navigation bar shared across every report page."""
    links = []
    for key, label, href in SITE_NAV:
        cls = " class='active'" if key == active else ""
        links.append(f"<a href='{href}'{cls}>{html.escape(label)}</a>")
    return (
        "<header class='site-header'>"
        "<a class='brand' href='index.html'>fluxcode</a>"
        f"<nav class='site-nav' aria-label='Site navigation'>{''.join(links)}</nav>"
        "</header>"
    )


def collapsible_table(summary, table_html, open_by_default=False):
    """Wrap a data table (and optional notes) in a closed-by-default <details> block."""
    attr = " open" if open_by_default else ""
    return (
        f"<details class='data-table'{attr}>"
        f"<summary>{html.escape(summary)}</summary>"
        f"<div class='table-wrap'>{table_html}</div>"
        "</details>"
    )


def shell(title, intro, toc, body, active=None):
    """A complete page. toc: [(anchor, label, [(anchor, label), ...])]."""
    toc_html = ""
    if toc:
        items = []
        for a, lab, subs in toc:
            sub = (
                "<ul>"
                + "".join(f"<li><a href='#{sa}'>{html.escape(sl)}</a></li>" for sa, sl in subs)
                + "</ul>"
                if subs
                else ""
            )
            items.append(f"<li><a href='#{a}'>{html.escape(lab)}</a>{sub}</li>")
        toc_html = f"<nav class='toc'><h2 id='contents'>Contents</h2><ul>{''.join(items)}</ul></nav>\n"
    return (
        f"<!doctype html>\n<html lang='en'><head><meta charset='utf-8'>"
        f"<meta name='viewport' content='width=device-width,initial-scale=1'><title>{html.escape(title)}</title>"
        f"<style>{CSS}</style></head><body>\n"
        f"{nav_bar(active)}\n"
        f"<h1>{html.escape(title)}</h1>\n{intro}\n"
        f"{toc_html}{body}\n</body></html>\n"
    )


_SAVED = set()


def save_fig(fig, out_dir, name, alt):
    """Check that no chart is empty, write <name>.svg (once per run), and return the <img> tag."""
    assert (out_dir, name) not in _SAVED, f"two charts share the file {name}.svg"
    _SAVED.add((out_dir, name))
    assert_data(fig, name)
    fig.savefig(out_dir / f"{name}.svg", format="svg")
    plt.close(fig)
    return f"<img src='{name}.svg' alt='{html.escape(alt)}'>"


def clean_outputs(out_dir, patterns, keep_prefix=None):
    """Delete this script's earlier outputs (files in out_dir matching patterns) so stale charts don't pile up.
    Files starting with keep_prefix (a string or a tuple of them) belong to other scripts and are left alone.
    Only touches out_dir."""
    out_dir.mkdir(exist_ok=True)
    for pat in patterns:
        for f in out_dir.glob(pat):
            if f.is_file() and not (keep_prefix and f.name.startswith(keep_prefix)):
                f.unlink()
