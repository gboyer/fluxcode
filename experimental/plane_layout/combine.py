# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""The layout heuristic and block flushing together, from the per-group tables of table.py.

Columns of `sizes`: bit planes under 5 framings [one block run, flush after every plane, every 4,
every 8, dense planes (the encoder's rule)], then the same 5 for byte planes. Block group sizes include
the 8-byte header. "retry" = the encoder's guard: a flushed block group whose frame is under 16 KB is also
compressed in one block run and the smaller kept.

    uv run python plane_layout/combine.py
"""

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NAMES = ["fit (2000)", "held out (1000)", "report (160)"]
POOLED, EVERY1, EVERY4, EVERY8, DENSE = range(5)
W = 5
RETRY = 16_384 + 8


def load():
    return [dict(np.load(HERE / n)) for n in ("table_fit.npz", "table_test.npz", "table_std.npz")]


def pick(s, byte, framing, retry=False):
    """Size of the chosen layout (bool array byte) under a framing, optionally with the retry guard."""
    size = np.where(byte, s["sizes"][:, W + framing], s["sizes"][:, framing])
    if retry:
        pooled = np.where(byte, s["sizes"][:, W + POOLED], s["sizes"][:, POOLED])
        size = np.where(size < RETRY, np.minimum(size, pooled), size)
    return size


def best_of_two(s, framing, retry=False):
    return np.minimum(pick(s, np.zeros(len(s["hi_nz"]), bool), framing, retry), pick(s, np.ones(len(s["hi_nz"]), bool), framing, retry))


def table(sets, rows):
    print(f"{'policy':58s} {'passes':>6s} " + " ".join(f"{n:>16s}" for n in NAMES))
    base = [best_of_two(s, POOLED).sum() for s in sets]
    for name, passes, f in rows:
        print(f"{name:58s} {passes:>6s} " + " ".join(f"{f(s).sum() / b - 1:+15.2%} " for s, b in zip(sets, base)))


if __name__ == "__main__":
    sets = load()
    heur = lambda s, t=0.05: s["hi_nz"] < t
    FR = {"one block run": POOLED, "flush every plane": EVERY1, "flush every 4 planes": EVERY4, "flush every 8 planes": EVERY8,
          "flush dense planes (encoder)": DENSE}
    rows = []
    for label, fr in FR.items():
        rows.append((f"best of bit/byte, {label}", "2", lambda s, fr=fr: best_of_two(s, fr)))
    rows.append(("best of bit/byte, flush dense planes + retry (encoder)", "2-3", lambda s: best_of_two(s, DENSE, True)))
    for label, fr in FR.items():
        rows.append((f"heuristic (hi_nz < 5%), {label}", "1", lambda s, fr=fr: pick(s, heur(s), fr)))
    rows.append(("heuristic (hi_nz < 5%), flush dense planes + retry (encoder)", "1-2", lambda s: pick(s, heur(s), DENSE, True)))
    rows.append(("always bit planes, flush dense planes + retry", "1-2", lambda s: pick(s, np.zeros(len(s["hi_nz"]), bool), DENSE, True)))
    rows.append(("oracle over all 10 (layout x framing)", "10", lambda s: s["sizes"].min(axis=1)))
    table(sets, rows)

    print("\n--- heuristic threshold (byte iff hi_nz < t) with the encoder's framing (dense planes + retry) ---")
    for t in (0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.25):
        cells = [f"{pick(s, heur(s, t), DENSE, True).sum() / best_of_two(s, POOLED).sum() - 1:+8.2%}" for s in sets]
        print(f"  hi_nz < {t:5.3f}: " + " ".join(cells))

    print("\n--- effect of flushing by block group size (heuristic layout), fit set: total bytes saved vs one block run ---")
    s = sets[0]
    byte = heur(s)
    pooled = pick(s, byte, POOLED)
    cols = {"every plane": pick(s, byte, EVERY1), "dense planes": pick(s, byte, DENSE), "dense + retry": pick(s, byte, DENSE, True)}
    edges = [0, 1000, 3000, 10000, 30000, 60000, 1e9]
    print(f"{'one-run group size':>20s} {'groups':>6s} " + " ".join(f"{k:>26s}" for k in cols))
    for lo, hi_ in zip(edges[:-1], edges[1:]):
        m = (pooled >= lo) & (pooled < hi_)
        if m.any():
            print(f"{lo:9.0f}-{hi_:9.0f} B {m.sum():6d} " + " ".join(
                f"{1 - v[m].sum() / pooled[m].sum():+8.2%} saved, {np.mean(v[m] > pooled[m]):4.0%} worse" for v in cols.values()))
    print(f"{'worst group vs one run':>27s} " + " ".join(f"{(v / pooled).max() - 1:+26.1%}" for v in cols.values()))
    print(f"{'mean per-group change':>27s} " + " ".join(f"{np.mean(v / pooled) - 1:+26.2%}" for v in cols.values()))
