# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""The layout heuristic and block flushing together, from the per-unit tables of table.py.

Columns of `sizes`: bit planes under [one block run, flush every plane, every 4, every 8], then the
same for byte planes (byte planes have only two plane cuts, so the three flush columns are equal).
Unit sizes include the 8-byte header. "passes" = zstd compress calls per unit.

    uv run python plane_layout/combine.py
"""

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NAMES = ["fit (2000)", "held out (1000)", "report (160)"]


def load():
    out = []
    for n in ("table_fit.npz", "table_test.npz", "table_std.npz"):
        d = np.load(HERE / n)
        out.append(dict(d))
    return out


def pick(s, byte, framing):
    """Size of the chosen layout (bool array byte) under framing 0..3."""
    return np.where(byte, s[:, 4 + framing], s[:, framing])


def table(sets, rows):
    print(f"{'policy':58s} {'passes':>6s} " + " ".join(f"{n:>16s}" for n in NAMES))
    base = [np.minimum(s["sizes"][:, 0], s["sizes"][:, 4]).sum() for s in sets]
    for name, passes, f in rows:
        cells = []
        for s, b in zip(sets, base):
            tot = f(s).sum()
            cells.append(f"{tot / b - 1:+15.2%} ")
        print(f"{name:58s} {passes:>6s} " + " ".join(cells))


if __name__ == "__main__":
    sets = load()
    hi = lambda t: (lambda s: s["hi_nz"] < t)
    rows = [("today: best of bit/byte, one block run", "2", lambda s: np.minimum(s["sizes"][:, 0], s["sizes"][:, 4]))]
    rows += [(f"heuristic hi_nz < {t:.0%} -> byte, one block run", "1", lambda s, t=t: pick(s["sizes"], s["hi_nz"] < t, 0)) for t in (0.02, 0.05, 0.1)]
    print("--- layout choice x framing (framing: flush every plane / 4 planes / 8 planes) ---")
    for fr, label in ((1, "every plane"), (2, "every 4"), (3, "every 8")):
        rows.append((f"best of bit/byte + flush {label}", "2", lambda s, fr=fr: np.minimum(s["sizes"][:, fr], s["sizes"][:, 4 + fr])))
    for fr, label in ((1, "every plane"), (2, "every 4"), (3, "every 8")):
        for t in (0.05, 0.1):
            rows.append((f"heuristic hi_nz < {t:.0%} + flush {label}", "1", lambda s, fr=fr, t=t: pick(s["sizes"], s["hi_nz"] < t, fr)))
    rows.append(("oracle over all 8 (layout x framing)", "8", lambda s: s["sizes"].min(axis=1)))
    table(sets, rows)

    print("\n--- hi_nz threshold, with flush every plane (fit set; held out and report for the picks) ---")
    for t in (0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.25, 1.01):
        cells = []
        for s in sets:
            b = np.minimum(s["sizes"][:, 0], s["sizes"][:, 4]).sum()
            cells.append(f"{pick(s['sizes'], s['hi_nz'] < t, 1).sum() / b - 1:+8.2%}")
        print(f"  hi_nz < {t:5.3f}: " + " ".join(cells))

    print("\n--- flush only large units: flush every plane when the plane bytes hold more than T non-zero bytes ---")
    print("(heuristic hi_nz < 5%; otherwise one block run)")
    for T in (0, 500, 1000, 2000, 4000, 8000, 16000, 32000):
        cells = []
        for s in sets:
            byte = s["hi_nz"] < 0.05
            fl = s["nonzero_bytes"] > T
            size = np.where(fl, pick(s["sizes"], byte, 1), pick(s["sizes"], byte, 0))
            b = np.minimum(s["sizes"][:, 0], s["sizes"][:, 4]).sum()
            cells.append(f"{size.sum() / b - 1:+8.2%} ({fl.mean():4.0%} flushed)")
        print(f"  T = {T:6d}: " + "  ".join(cells))
    print("\n--- by unit size: effect of flushing every plane (best layout chosen by the heuristic), fit set ---")
    s = sets[0]
    byte = s["hi_nz"] < 0.05
    pooled, fl = pick(s["sizes"], byte, 0), pick(s["sizes"], byte, 1)
    edges = [0, 1000, 3000, 10000, 30000, 60000, 1e9]
    for lo, hi_ in zip(edges[:-1], edges[1:]):
        m = (pooled >= lo) & (pooled < hi_)
        if m.any():
            print(f"  unit {lo:7.0f}-{hi_:9.0f} B: {m.sum():4d} units, flush saves {1 - fl[m].sum() / pooled[m].sum():+6.2%} in total; "
                  f"better on {np.mean(fl[m] < pooled[m]):4.0%} of them, median change {np.median(fl[m] - pooled[m]):+.0f} B")
