# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Beyond bit and byte planes: store the 16 bits of each residual as fields of w neighbouring
bits (w = 1: bit planes, 2: crumbs, 4: nibbles, 8: bytes), mixed per layout, and compress each
layout with zstd 3. Also the same bit/byte layouts through LZMA and bzip2 (bit-wise literal
coding, a BWT), to see whether a different compressor changes which layout wins.

A layout is a list of field widths from the least significant bits up, e.g. [8, 1, 1, 1, 1, 1, 1,
1, 1] = the low byte as a byte plane, the high byte as 8 bit planes. A field of width w becomes
one stream of w-bit symbols, packed 8 // w per byte (w = 8: one byte per sample). "pooled" puts
all fields in one zstd frame (one Huffman table, as `planes=` does today); "split" compresses
each field's stream in its own frame (frame header and table cost, no sharing).

Sizes are the plane bytes only (the rest of a block group is the same under every layout).

    uv run python plane_layout/layouts.py [--codecs]
"""

import argparse
import bz2
import lzma
import sys
import time
from pathlib import Path

import numpy as np
import zstandard

sys.path.insert(0, str(Path(__file__).resolve().parent))

HERE = Path(__file__).resolve().parent
ZC = zstandard.ZstdCompressor(level=3, write_checksum=False, write_content_size=True)

LAYOUTS = {
    "bit": [1] * 16,
    "byte": [8, 8],
    "nibble": [4] * 4,
    "crumb": [2] * 8,
    "byte-low + bit-high": [8] + [1] * 8,
    "bit-low + byte-high": [1] * 8 + [8],
    "nibble-low + bit-high": [4, 4] + [1] * 8,
    "bit-low + nibble-high": [1] * 8 + [4, 4],
    "byte-low + nibble-high": [8, 4, 4],
    "nibble-low + byte-high": [4, 4, 8],
    "byte-low + crumb-high": [8] + [2] * 4,
}


def field_streams(u, widths):
    """The packed stream of each field of a layout."""
    out, shift = [], 0
    for w in widths:
        sym = ((u >> shift) & ((1 << w) - 1)).astype(np.uint8)
        shift += w
        if w == 8:
            out.append(sym)
        elif w == 1:
            out.append(np.packbits(sym, bitorder="little"))
        else:
            per = 8 // w
            out.append(np.bitwise_or.reduce(sym.reshape(-1, per) << (w * np.arange(per, dtype=np.uint8)), axis=1).astype(np.uint8))
    return out


def pooled(u, widths):
    return len(ZC.compress(np.concatenate(field_streams(u, widths)).tobytes()))


def split(u, widths):
    """Each field's stream in its own zstd frame (its own Huffman table, no sharing)."""
    return sum(len(ZC.compress(s.tobytes())) for s in field_streams(u, widths))


def halves(u, widths):
    """One frame for the fields of the low byte of u, one for the fields of the high byte (fields
    never straddle bit 8 in the layouts used here)."""
    streams, bits = field_streams(u, widths), np.cumsum(widths)
    low = [s for s, b in zip(streams, bits) if b <= 8]
    high = [s for s, b in zip(streams, bits) if b > 8]
    return sum(len(ZC.compress(np.concatenate(part).tobytes())) for part in (low, high) if part)


MODES = {"pooled": pooled, "split": split, "halves": halves}


def evaluate(name):
    d = np.load(HERE / name)
    U = d["u"]
    keys = [(k, m) for k in LAYOUTS for m in MODES if m == "pooled" or (m == "split" and len(LAYOUTS[k]) > 1)
            or (m == "halves" and len(LAYOUTS[k]) > 2)]
    sizes = {key: np.array([MODES[key[1]](u, LAYOUTS[key[0]]) for u in U], float) for key in keys}
    return d, sizes


def report(label, d, sizes):
    n = len(d["u"])
    pair = np.minimum(sizes[("bit", "pooled")], sizes[("byte", "pooled")])
    allbest = np.minimum.reduce(list(sizes.values()))
    bps = lambda a: a.sum() * 8 / (60_000 * n)
    print(f"\n{label}: {n} block groups, plane bytes compressed with zstd 3; bits/sample, and size vs the best-of-bit-and-byte oracle")
    print(f"{'layout':44s} {'bits/sample':>11s} {'vs oracle(bit,byte)':>20s} {'wins':>6s}")
    for (k, mode), s in sorted(sizes.items(), key=lambda kv: kv[1].sum())[:12]:
        wins = np.mean(s <= allbest + 1e-9)
        print(f"{k + ' (' + mode + ')':44s} {bps(s):11.3f} {s.sum() / pair.sum() - 1:+19.2%} {wins:6.0%}")
    print(f"{'oracle(bit, byte), pooled (today)':44s} {bps(pair):11.3f} {0:+19.2%}")
    for m in ("split", "halves"):
        o = np.minimum(sizes[("bit", m)], sizes[("byte", "split")])  # a two-field layout has one frame per field
        print(f"{'oracle(bit, byte), ' + m:44s} {bps(o):11.3f} {o.sum() / pair.sum() - 1:+19.2%}")
    print(f"{'oracle(all of the above)':44s} {bps(allbest):11.3f} {allbest.sum() / pair.sum() - 1:+19.2%}")


def codecs():
    d = np.load(HERE / "corpus.npz")
    rng = np.random.default_rng(0)
    pick = rng.choice(len(d["u"]), 200, replace=False)
    import model
    ls = [model.streams(d["u"][i]) for i in pick]
    cs = {"zstd 3": lambda b: ZC.compress(b), "zstd 19": lambda b: zstandard.ZstdCompressor(level=19).compress(b),
          "bzip2 9": lambda b: bz2.compress(b, 9), "xz 6": lambda b: lzma.compress(b, preset=6),
          "xz 6, lc=0 lp=0 pb=0": lambda b: lzma.compress(b, format=lzma.FORMAT_RAW, filters=[{"id": lzma.FILTER_LZMA2, "preset": 6, "lc": 0, "lp": 0, "pb": 0}]),
          "xz 6, lc=4 lp=0 pb=0": lambda b: lzma.compress(b, format=lzma.FORMAT_RAW, filters=[{"id": lzma.FILTER_LZMA2, "preset": 6, "lc": 4, "lp": 0, "pb": 0}]),
          "xz 6, lc=0 lp=3 pb=3": lambda b: lzma.compress(b, format=lzma.FORMAT_RAW, filters=[{"id": lzma.FILTER_LZMA2, "preset": 6, "lc": 0, "lp": 3, "pb": 3}])}
    print("\nother compressors on the bit and byte layouts (200 fit block groups; bits/sample; ms per block group for both layouts)")
    print(f"{'codec':26s} {'bit':>7s} {'byte':>7s} {'byte/bit':>9s} {'oracle':>7s} {'ms':>7s}")
    for name, f in cs.items():
        t0 = time.perf_counter()
        sb = np.array([len(f(a.tobytes())) for a, _ in ls], float)
        t = time.perf_counter() - t0
        sy = np.array([len(f(b.tobytes())) for _, b in ls], float)
        print(f"{name:26s} {sb.sum() * 8 / (60000 * 200):7.3f} {sy.sum() * 8 / (60000 * 200):7.3f} {sy.sum() / sb.sum():9.3f} "
              f"{np.minimum(sb, sy).sum() * 8 / (60000 * 200):7.3f} {t / 200 * 1e3:7.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--codecs", action="store_true")
    a = ap.parse_args()
    if a.codecs:
        codecs()
    else:
        for label, name in (("fit", "corpus.npz"), ("held out", "test.npz"), ("report signals", "standard.npz")):
            report(label, *evaluate(name))
