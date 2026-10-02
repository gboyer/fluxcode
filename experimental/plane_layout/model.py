# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Cheap estimates of how well zstd will compress a residual layout, without running zstd.

streams(u) builds the bytes zstd sees for each layout. lz_estimate(stream) runs a greedy
hash-chain-free LZ77 pass (4-byte hash, one candidate, no lazy matching) and costs it like zstd
would roughly: Huffman-coded literals plus a fixed number of bits per match.
"""

import numpy as np
from numba import njit

MIN_MATCH = 4
HASH_BITS = 15


def streams(u):
    """(bit-plane bytes, byte-plane bytes) of a flat uint16 residual array (length a multiple of 8)."""
    bit = np.concatenate([np.packbits(((u >> j) & 1).astype(np.uint8), bitorder="little") for j in range(16)])
    byte = np.concatenate([(u & 0xFF).astype(np.uint8), (u >> 8).astype(np.uint8)])
    return bit, byte


@njit(cache=True)
def hash4(s, i):
    w = np.uint64(s[i]) | (np.uint64(s[i + 1]) << np.uint64(8)) | (np.uint64(s[i + 2]) << np.uint64(16)) | (np.uint64(s[i + 3]) << np.uint64(24))
    return np.int64(((w * np.uint64(2654435761)) & np.uint64(0xFFFFFFFF)) >> np.uint64(32 - HASH_BITS))


WAYS = 4


@njit(cache=True)
def lz_scan(s, ways=WAYS):
    """Greedy LZ77 with `ways` candidates per hash bucket (the longest wins) and zstd-style repeat
    offsets (the last offset is cheap to reuse). Returns (literal histogram, number of matches,
    number of repeat-offset matches, bytes covered by matches, sum of log2(offset) over the
    non-repeat matches, sum of log2(length - 3))."""
    n = s.shape[0]
    table = np.full((1 << HASH_BITS, WAYS), -1, np.int64)
    fill = np.zeros(1 << HASH_BITS, np.int64)
    hist = np.zeros(256, np.int64)
    matches = 0
    repeats = 0
    covered = 0
    log_off = 0.0
    log_len = 0.0
    last_off = 0
    i = 0
    while i + MIN_MATCH <= n:
        h = hash4(s, i)
        best_len = 0
        best_off = 0
        for w in range(ways):
            cand = table[h, w]
            if cand < 0:
                continue
            ln = 0
            while i + ln < n and s[cand + ln] == s[i + ln]:
                ln += 1
            if ln >= MIN_MATCH and (ln > best_len or (ln == best_len and i - cand == last_off)):
                best_len = ln
                best_off = i - cand
        table[h, fill[h] % ways] = i
        fill[h] += 1
        if best_len >= MIN_MATCH:
            matches += 1
            covered += best_len
            if best_off == last_off:
                repeats += 1
            else:
                log_off += np.log2(best_off + 1.0)
            last_off = best_off
            log_len += np.log2(best_len - 3.0)
            for k in range(1, min(best_len, 16)):
                if i + k + 4 <= n:
                    h2 = hash4(s, i + k)
                    table[h2, fill[h2] % ways] = i + k
                    fill[h2] += 1
            i += best_len
        else:
            hist[s[i]] += 1
            i += 1
    while i < n:
        hist[s[i]] += 1
        i += 1
    return hist, matches, repeats, covered, log_off, log_len


def entropy_bits(hist):
    p = hist[hist > 0] / hist.sum()
    return float(-(hist[hist > 0] * np.log2(p)).sum())


def lz_estimate(s, ways=WAYS, match_bits=8.0, repeat_bits=3.0):
    """Estimated compressed bytes: order-0 entropy of the literals plus, per match, the offset
    (log2, or a few bits for a repeat offset), the length and a fixed code cost."""
    hist, matches, repeats, covered, log_off, log_len = lz_scan(s, ways)
    lit_bits = entropy_bits(hist) if hist.sum() else 0.0
    bits = lit_bits + log_off + log_len + match_bits * (matches - repeats) + repeat_bits * repeats
    return bits / 8, int(hist.sum()), matches, covered
