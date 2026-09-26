# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Shared pieces of the generated report pages (index.html, rate.html).

Ordering rules (every table, legend, chart and TOC follows them):

1. Codecs: one canonical order, `order.ordered(codecs)`. Family order is classic, constwidth,
   entropy, flux; within a family by nominal bits per sample (`order.NOMINAL_BITS`), ties in the
   order the package lists them.
2. Datasets: `tslab.common.datasets.KINDS` order, everywhere.
3. Colors: one hue per family (blue, orange, green, purple), shades within the family by position
   in the canonical order, so a codec has the same color in every chart. Markers: one per family.
"""
