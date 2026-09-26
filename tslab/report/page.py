# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""HTML shell (inline CSS, light and dark, table of contents) and figure saving for the report pages."""

import html

import matplotlib.pyplot as plt

from tslab.report.order import assert_data

plt.rcParams["svg.fonttype"] = "none"  # text stays text: smaller and crisper than PNG

CSS = """
:root{--bg:#fff;--fg:#1a1a1a;--muted:#666;--line:#ddd;--hl:#dcefdc;--l1:#e8f3e8;--l2:#fff3cc;--l3:#ffe0cc;--l4:#ffcccc}
@media (prefers-color-scheme:dark){:root{--bg:#161616;--fg:#e8e8e8;--muted:#999;--line:#333;--hl:#254025;--l1:#1f2e1f;--l2:#3a3418;--l3:#40291a;--l4:#452020}}
body{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;max-width:1200px;margin:0 auto;padding:16px}
table{border-collapse:collapse;margin:8px 0 16px;font-variant-numeric:tabular-nums;display:block;overflow-x:auto}
th,td{border:1px solid var(--line);padding:3px 7px;text-align:right;white-space:nowrap}
th:first-child,td:first-child,td.l,th.l{text-align:left}
td.best{background:var(--hl);font-weight:600}
td.r1{background:var(--l1)}td.r2{background:var(--l2)}td.r3{background:var(--l3)}td.r4{background:var(--l4)}
tr.fam td:first-child{border-left:4px solid var(--c,var(--line))}
img{max-width:100%;background:#fff}
nav ul{columns:2;list-style:none;padding-left:0}nav li{margin:2px 0;break-inside:avoid}nav ul ul{columns:1;padding-left:16px}
.muted{color:var(--muted)} code{font-size:13px} h2{margin-top:32px;border-top:1px solid var(--line);padding-top:12px}
"""


def shell(title, intro, toc, body):
    """A complete page. toc: [(anchor, label, [(anchor, label), ...])]."""
    items = []
    for a, lab, subs in toc:
        sub = "<ul>" + "".join(f"<li><a href='#{sa}'>{html.escape(sl)}</a></li>" for sa, sl in subs) + "</ul>" if subs else ""
        items.append(f"<li><a href='#{a}'>{html.escape(lab)}</a>{sub}</li>")
    return (f"<!doctype html>\n<html lang='en'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'><title>{html.escape(title)}</title>"
            f"<style>{CSS}</style></head><body>\n<h1>{html.escape(title)}</h1>\n{intro}\n"
            f"<nav><h2 id='contents'>Contents</h2><ul>{''.join(items)}</ul></nav>\n{body}\n</body></html>\n")


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
    Files starting with keep_prefix belong to the other script and are left alone. Only touches out_dir."""
    out_dir.mkdir(exist_ok=True)
    for pat in patterns:
        for f in out_dir.glob(pat):
            if f.is_file() and not (keep_prefix and f.name.startswith(keep_prefix)):
                f.unlink()
