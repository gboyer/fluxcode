# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Best-of-2 and best-of-3 over bit, byte and nibble planes, each framed three ways, zstd 3:

  pooled   all the layout's streams in one frame and one block run (today)
  frames   each group's stream compressed on its own, as separate zstd frames
  blocks   one frame, with a block boundary forced between groups (the streaming API's
           FLUSH_BLOCK): each group gets its own literal Huffman table, the window is shared, and
           a one-shot decoder reads it like any other frame (no section lengths needed)

Sizes are the plane bytes only. "vs today" is against the best of pooled bit and pooled byte.

    uv run python plane_layout/combos.py
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import zstandard

sys.path.insert(0, str(Path(__file__).resolve().parent))
from layouts import group_streams

HERE = Path(__file__).resolve().parent
LAYOUTS = {"bit": [1] * 16, "byte": [8, 8], "nibble": [4] * 4}
ZC = zstandard.ZstdCompressor(level=3, write_checksum=False, write_content_size=True)


def pooled(streams):
    return len(ZC.compress(b"".join(s.tobytes() for s in streams)))


def frames(streams):
    return sum(len(ZC.compress(s.tobytes())) for s in streams)


def blocks(streams):
    total = sum(len(s) for s in streams)
    c = zstandard.ZstdCompressor(level=3, write_checksum=False, write_content_size=True).compressobj(size=total)
    out = 0
    for s in streams:
        out += len(c.compress(s.tobytes()))
        out += len(c.flush(zstandard.COMPRESSOBJ_FLUSH_BLOCK))
    out += len(c.flush())
    return out


MODES = {"pooled": pooled, "frames": frames, "blocks": blocks}


def sizes(name):
    u = np.load(HERE / name)["u"]
    out = {(l, m): np.zeros(len(u)) for l in LAYOUTS for m in MODES}
    for i, x in enumerate(u):
        for l, w in LAYOUTS.items():
            st = group_streams(x, w)
            for m, f in MODES.items():
                out[(l, m)][i] = f(st)
    return out


def report(label, s):
    n = len(next(iter(s.values())))
    today = np.minimum(s[("bit", "pooled")], s[("byte", "pooled")])
    bps = lambda a: a.sum() * 8 / (60_000 * n)
    print(f"\n{label} ({n} units): bits/sample and vs today (best of pooled bit/byte = {bps(today):.3f})")
    print(f"{'':30s} " + " ".join(f"{m:>16s}" for m in MODES))
    for l in LAYOUTS:
        print(f"{l:30s} " + " ".join(f"{bps(s[(l, m)]):7.3f} {s[(l, m)].sum() / today.sum() - 1:+7.1%}" for m in MODES))
    for k in (2, 3):
        for combo in itertools.combinations(LAYOUTS, k):
            row = []
            for m in MODES:
                o = np.minimum.reduce([s[(l, m)] for l in combo])
                row.append(f"{bps(o):7.3f} {o.sum() / today.sum() - 1:+7.1%}")
            print(f"{'best of ' + '/'.join(combo):30s} " + " ".join(row))
    allm = np.minimum.reduce([v for k, v in s.items() if k[1] != "frames"])
    print(f"{'best of all 3 layouts x (pooled, blocks)':44s} {bps(allm):7.3f} {allm.sum() / today.sum() - 1:+7.1%}")
    allm = np.minimum.reduce(list(s.values()))
    print(f"{'best of all 9 combinations':44s} {bps(allm):7.3f} {allm.sum() / today.sum() - 1:+7.1%}")
    wins = np.argmin(np.stack(list(s.values())), axis=0)
    keys = list(s)
    print("which wins (of 9): " + ", ".join(f"{keys[k][0]}/{keys[k][1]} {np.mean(wins == k):.0%}" for k in range(9) if np.mean(wins == k) > 0.02))


if __name__ == "__main__":
    for label, name in (("fit", "corpus.npz"), ("held out", "test.npz"), ("report signals", "standard.npz")):
        report(label, sizes(name))
