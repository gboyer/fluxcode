# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Canonical codec order, families, colors and the legend metadata."""

import matplotlib
import numpy as np

FAMILIES = ("classic", "constwidth", "entropy", "flux")
FAMILY_NAMES = {"classic": "classic", "constwidth": "constant width", "entropy": "entropy",
                "flux": "fluxcode"}
FAMILY_MARKERS = {"classic": "o", "constwidth": "s", "entropy": "^", "flux": "D"}
_CMAPS = {"classic": "Blues", "constwidth": "Oranges", "entropy": "Greens", "flux": "Purples"}

# Nominal bits per sample of each codec (its name or design point), the sort key within a family.
NOMINAL_BITS = {
    "quant8": 8, "quant8-delta1-deflate": 8, "quant6-delta1-deflate": 6, "dpcm6-linear-deflate": 6,
    "dpcm6-cubic-deflate": 6, "dpcm6-mulaw-deflate": 6, "dpcm4-linear-deflate": 4, "dpcm4-cubic-deflate": 4,
    "bintree-center": 3.1, "bintree-edge": 3.1, "pwlinear-minmax16": 2.2, "pwlinear-minmax16-merged": 1.2,
    "pchip-minmax16": 2.2, "swingdoor-0.5%": 1.2, "swingdoor-2%": 1.2, "gorilla-xor": 64, "dct-top64": 1.7,
}
# Codecs that decode bit-exactly on any input / that spend the same bytes on every block.
LOSSLESS = {"gorilla-xor"}
CONST_WIDTH = {"quant8", "bintree-center", "bintree-edge", "dct-top64"}


def family(codec):
    """classic / constwidth / entropy / flux, from the module that defines the block codec."""
    c = getattr(codec, "codec", codec)
    mod = type(c).__module__.split(".")
    return mod[1] if mod[0] == "tslab" and mod[1] in FAMILIES else "classic"


def nominal_bits(codec):
    n = codec.name
    if n in NOMINAL_BITS:
        return NOMINAL_BITS[n]
    if n.startswith("ratectl-"):
        return 4
    last = n.split("-")  # cw4-delta1-..., delta0123-zstd-10, fluxcode-16-f0.25, cw-delta1-tree-7
    if n.startswith("cw") and last[0] != "cw":
        return int(last[0][2:])
    if n.startswith("cw-delta1-tree-"):
        return int(last[-1]) + 1
    if n.startswith("fluxcode-"):
        return int(last[1]) + 0.01 * float(last[2][1:]) if len(last) > 2 else int(last[1])
    return int(last[-1])  # delta0123-zstd-8, delta12-zstd-8, ...


def ordered(codecs):
    """The canonical order: family, then nominal bits, then the package's own order."""
    idx = {c.name: i for i, c in enumerate(codecs)}
    return sorted(codecs, key=lambda c: (FAMILIES.index(family(c)), nominal_bits(c), idx[c.name]))


def palette(codecs):
    """codec name -> RGBA: the family's hue, shaded by position within the family (never runs out)."""
    out = {}
    for fam in FAMILIES:
        members = [c for c in codecs if family(c) == fam]
        cmap = matplotlib.colormaps[_CMAPS[fam]]
        for i, c in enumerate(members):
            out[c.name] = cmap(0.45 + 0.5 * (i / max(1, len(members) - 1)))
    return out


def lossless_kind(codec):
    """The legend's lossless? cell."""
    if codec.name in LOSSLESS:
        return "yes"
    if family(codec) == "flux":
        return "decimal data"
    return "no"


def module_link(codec):
    """Path of the codec's module relative to report/ (../tslab/<family>/<module>.py)."""
    c = getattr(codec, "codec", codec)
    mod = type(c).__module__
    return "../" + mod.replace(".", "/") + ".py", mod.split(".", 1)[1]


def finite_points(ax):
    """Number of finite plotted data points on an axis (positive ones if an axis is logarithmic)."""
    n = 0

    def ok(v, log):
        v = np.asarray(v, dtype=float)
        return int(np.sum(np.isfinite(v) & (v > 0 if log else True)))

    for ln in ax.get_lines():
        if ln.get_gid() == "ref":
            continue
        xs, ys = ln.get_xdata(), ln.get_ydata()
        good = np.isfinite(np.asarray(xs, float)) & np.isfinite(np.asarray(ys, float))
        if ax.get_yscale() == "log":
            good &= np.asarray(ys, float) > 0
        n += int(good.sum())
    for coll in ax.collections:
        off = np.asarray(coll.get_offsets(), dtype=float)
        if off.size:
            good = np.isfinite(off).all(axis=1)
            if ax.get_yscale() == "log" and coll.get_transform() == ax.transData:
                good &= off[:, 1] > 0
            n += int(good.sum())
    for p in ax.patches:
        h = p.get_height() if hasattr(p, "get_height") else 0
        n += int(np.isfinite(h) and (h > 0 or ax.get_yscale() != "log"))
    return n


def assert_data(fig, name):
    """Fail the build if any visible axis of the figure has no finite data point (an empty chart)."""
    for i, ax in enumerate(fig.axes):
        if ax.axison:
            assert finite_points(ax) > 0, f"empty chart: {name}, axes #{i} ({ax.get_title()!r})"
