# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Block boundaries inside the one zstd frame: no format change, an encoder-only trick.

zstd codes a frame as blocks (at most 128 KiB each), and every block carries its own literal
Huffman table. Ending a block at a field or plane boundary (the streaming API's FLUSH_BLOCK) gives
each plane its own table, and the result is still one ordinary frame that today's decoder reads
unchanged. This checks that on real unit bodies: the compressed unit is rebuilt with a flush after
the column fields and after each residual plane (bit planes) or byte plane (byte planes), the
real decoder (`fluxcode.decode`) must return the same values, and the unit sizes are compared.

    uv run python plane_layout/flush_blocks.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
import numpy as np
import zstandard

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
import fluxcode
from _signals import KINDS, minute
from fluxcode import Params, _format, _unit

ZD = zstandard.ZstdDecompressor()


POLICIES = {"every plane": 1, "every 4 planes": 4, "every 8 planes (halves)": 8}


def cuts(parsed, every):
    """Offsets in the body where a block should end: after the columns, after every `every` bit
    planes (byte planes: after each of the 2 byte planes), after the residuals."""
    raw_len = len(parsed.raw_body)
    nb, g = parsed.header.num_blocks, int(parsed.layout.group_offsets[-1])
    start = _format.residual_start(nb, parsed.has_time)
    if parsed.header.byte_planes:
        pts = [start + k * 8 * g for k in range(3)]
    else:
        pts = [start + k * g for k in range(0, 17, every)]
        pts.append(start + 16 * g)
    pts.append(pts[-1] + int(parsed.layout.code_offsets[-1]) * _format.NONFINITE_BITS_PER_SAMPLE)
    pts = [0] + pts + [raw_len]
    return sorted(set(p for p in pts if 0 <= p <= raw_len))


def reflush(unit, every=1):
    parsed = _unit.decompress(unit)
    body = bytes(parsed.raw_body)
    c = zstandard.ZstdCompressor(level=3, write_checksum=False, write_content_size=True).compressobj(size=len(body))
    out = []
    pts = cuts(parsed, every)
    for a, b in zip(pts[:-1], pts[1:]):
        out.append(c.compress(body[a:b]))
        out.append(c.flush(zstandard.COMPRESSOBJ_FLUSH_BLOCK))
    out.append(c.flush())
    return unit[:_format.HEADER_BYTES] + b"".join(out)


if __name__ == "__main__":
    from _signals import DISCRETE, discrete_minute
    names = KINDS + [n for n, _, _ in DISCRETE]
    print("unit bytes summed over each signal's 3 minutes (header included); today = best of bit/byte planes, one block run")
    cols = list(POLICIES)
    print(f"{'signal':26s} {'today':>8s} " + " ".join(f"{c:>24s}" for c in cols))
    tot = np.zeros(1 + len(cols))
    for k, name in enumerate(names):
        row = np.zeros(1 + len(cols))
        for seed in range(3):
            x = minute(name, 700 + 10 * k + seed) if name in KINDS else discrete_minute(name, 700 + 10 * k + seed)
            units = {mode: fluxcode.encode(x, Params(planes=mode))[0][0] for mode in ("bit", "byte")}
            ref = {m: fluxcode.decode([u])[0].values for m, u in units.items()}
            row[0] += min(len(u) for u in units.values())
            for j, every in enumerate(POLICIES.values()):
                new = {m: reflush(u, every) for m, u in units.items()}
                for m in new:
                    assert zstandard.frame_content_size(new[m][_format.HEADER_BYTES:]) == len(_unit.decompress(units[m]).raw_body)
                    assert np.array_equal(ref[m], fluxcode.decode([new[m]])[0].values, equal_nan=True), (name, m, every)
                row[1 + j] += min(len(u) for u in new.values())
        tot += row
        print(f"{name:26s} {row[0]:8.0f} " + " ".join(f"{row[1 + j]:12.0f} {row[1 + j] / row[0] - 1:+10.1%}" for j in range(len(cols))))
    print(f"{'TOTAL (size-weighted)':26s} {tot[0]:8.0f} " + " ".join(f"{tot[1 + j]:12.0f} {tot[1 + j] / tot[0] - 1:+10.1%}" for j in range(len(cols))))
    print("every re-framed unit decodes to the same values through fluxcode.decode")
