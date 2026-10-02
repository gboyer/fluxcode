# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""How framing the residual planes separately (layouts.py) interacts with the rest of the body:
the per-block columns, the non-finite code planes and the time residual planes.

A unit body is: columns (flags, sizes, grid params, anchors, and with a time axis the time
columns), the 16 residual planes, the non-finite code planes (flagged blocks only), the time
residual planes (irregular blocks only). Today everything is one zstd frame. Here the same bytes
are framed differently and compressed with zstd 3:

  today     the whole body in one frame, bit or byte planes, whichever is smaller
  A         residuals as 4 nibble frames; columns + code planes + time residuals in one more frame
  B         residuals as 4 nibble frames; columns, code planes and time residuals each in a frame
  C         residuals as 4 nibble frames; everything else pooled into the first nibble frame

Sizes are compressed bytes per unit (no 8-byte header), averaged over 18 units per scenario.

    uv run python plane_layout/sections.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
import numpy as np
import zstandard

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fluxcode
from _signals import CLOCKS, MINUTE, clock_minute, minute
from fluxcode import Params, _format, _unit
from layouts import group_streams

ZC = zstandard.ZstdCompressor(level=3, write_checksum=False, write_content_size=True)
KINDS = ["sin-9.87hz", "random-walk", "noisy-sine", "sensor-0.1", "chirp", "gauss-spikes"]
Z = lambda *parts: len(ZC.compress(b"".join(bytes(p) for p in parts)))


def body(x, mode, times=None):
    units, *_ = fluxcode.encode(x, Params(planes=mode), times=times, time_unit=None if times is None else "ns")
    (unit,) = units
    return _unit.decompress(unit)


def sections(parsed):
    raw = parsed.raw_body
    nb, has_time = parsed.header.num_blocks, parsed.has_time
    start = _format.residual_start(nb, has_time)
    n = int(parsed.layout.sample_offsets[-1])
    g = int(parsed.layout.group_offsets[-1])
    code_len = int(parsed.layout.code_offsets[-1]) * _format.NONFINITE_BITS_PER_SAMPLE
    res_end = start + 16 * g
    return dict(columns=raw[:start], residual=raw[start:res_end], codes=raw[res_end:res_end + code_len],
                timeres=raw[res_end + code_len:], n=n)


def scenario(name):
    rng = np.random.default_rng(3)
    rows = []
    for k, kind in enumerate(KINDS):
        for seed in range(3):
            x = minute(kind, 900 + 10 * k + seed)
            times = None
            if name in CLOCKS or name.endswith("+noisy"):
                times = clock_minute(name if name in CLOCKS else "noisy", 40 + seed)
            if name.startswith("nan"):
                x = x.copy()
                for blk in rng.choice(60, 30, replace=False):
                    idx = blk * 1000 + rng.choice(1000, 5, replace=False)
                    x[idx] = np.nan
                x[rng.integers(0, MINUTE, 20)] = np.inf
            rows.append((x, times))
    return rows


def measure(rows):
    out = {k: 0 for k in ("columns", "residual (best of bit/byte)", "codes", "timeres", "today", "A", "B", "C")}
    for x, times in rows:
        pb, py = body(x, "bit", times), body(x, "byte", times)
        today = min(Z(pb.raw_body), Z(py.raw_body))
        sb, sy = sections(pb), sections(py)
        u = (np.frombuffer(sy["residual"][:sy["n"]], np.uint8).astype(np.uint16)
             | (np.frombuffer(sy["residual"][sy["n"]:], np.uint8).astype(np.uint16) << 8))
        nib = [s.tobytes() for s in group_streams(u, [4, 4, 4, 4])]
        rest = (sy["columns"], sy["codes"], sy["timeres"])
        out["columns"] += Z(sy["columns"])
        out["residual (best of bit/byte)"] += min(Z(sb["residual"]), Z(sy["residual"]))
        out["codes"] += Z(sy["codes"]) if len(sy["codes"]) else 0
        out["timeres"] += Z(sy["timeres"]) if len(sy["timeres"]) else 0
        out["today"] += today
        out["A"] += sum(Z(s) for s in nib) + Z(*rest)
        out["B"] += sum(Z(s) for s in nib) + sum(Z(p) for p in rest if len(p))
        out["C"] += Z(nib[0], *rest) + sum(Z(s) for s in nib[1:])
    return {k: v / len(rows) for k, v in out.items()}


if __name__ == "__main__":
    print(f"{'scenario':16s} | {'columns':>7s} {'resid.':>7s} {'codes':>6s} {'timeres':>7s} (each compressed alone) | "
          f"{'today':>7s} | {'A':>7s} {'B':>7s} {'C':>7s}   bytes/unit; A/B/C vs today")
    for name in ("plain", "grid", "grid+gaps", "noisy", "nan", "nan+noisy"):
        m = measure(scenario(name))
        print(f"{name:16s} | {m['columns']:7.0f} {m['residual (best of bit/byte)']:7.0f} {m['codes']:6.0f} {m['timeres']:7.0f} {'':26s}| "
              f"{m['today']:7.0f} | " + " ".join(f"{m[k]:7.0f}" for k in "ABC") + "   " + " ".join(f"{m[k] / m['today'] - 1:+.1%}" for k in "ABC"))
