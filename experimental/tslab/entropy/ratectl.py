# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Rate-controlled quantize -> predict -> entropy-code pipelines on one-minute windows.

A window is one block group of 60 blocks of 1000 samples, each block rescaled to [0, 1] (make_rate_report.py
builds them from seconds of every signal kind, standing in for one sensor passing through different
regimes). Because each block has block group range, absolute error = error as a fraction of FS.

Block group format (every byte counted):
  delta f64 | e u8 per block | coder block group (coder.encode_group)
  Block step is delta_b = delta * 2**(e/8).
Pipeline per block: q = round(x / delta_b) -> residual = diff(q, order) -> coder.
The only lossy step is the rounding, so |err| <= delta_b / 2 always.

Each coder has a block form, encode(q) -> payload / decode(payload, n) -> q, used to choose
per-block options and by the per-block cap, and a block group form, encode_group(Q) -> bytes /
decode_group(data, nb, n) -> Q, which is what gets stored and counted:
  delta0123-rice      block payloads back to back, each preceded by its LEB128 length
  delta0123-zstd/deflate  one compressor call per block group: per-block flags, then init varints,
                      then residual byte planes (order and width chosen per block by the
                      smallest standalone block)
  flac                per-block base varints, then one FLAC stream with one frame per block
  best: ...           a coder id byte per block, then each coder's block group of its blocks, length first
"""

import time
import zlib

import numpy as np
import pyflac
import zstandard

from tslab.classic.quant import QuantDeltaDeflate
from tslab.common.bitio import BitReader, BitWriter
from tslab.common.intcode import ORDERS, from_residual, pack_init, read_varint, to_residual, unpack_init, unzigzag, varint, zigzag
from tslab.common.group import SharedBackend

BLOCK = 1000
TARGET_BPS = 4.0
CAP_BPS = 8.0
E_STEPS = 8  # per-block step multiplier resolution: 2**(1/8)


# ---------------------------------------------------------------------------
# Block coders: encode(q) -> payload bytes (flags byte included), decode(payload, n) -> q
# ---------------------------------------------------------------------------

class RiceCoder:
    """FLAC-style: per block choose predictor order and partition order p
    (2**p partitions, each with its own Rice parameter k). Choice is by exact
    bit cost, which is computed vectorized."""

    name = "delta0123-rice"
    KS = np.arange(48)[:, None]

    def _best_k(self, u):
        if len(u) == 0:
            return 0, 0
        cost = (u[None, :] >> self.KS).sum(axis=1) + len(u) * (self.KS[:, 0] + 1)
        k = int(np.argmin(cost))
        return k, int(cost[k])

    def _plan(self, q):
        best = None
        for order in ORDERS:
            init, r = to_residual(q, order)
            u = zigzag(r)
            for p in range(4):
                parts = np.array_split(u, 1 << p)
                ks, bits = zip(*(self._best_k(part) for part in parts))
                total = 8 * len(pack_init(init)) + 6 * len(parts) + sum(bits)
                if best is None or total < best[0]:
                    best = (total, order, p, ks, init, parts)
        return best

    def encode(self, q):
        _, order, p, ks, init, parts = self._plan(q)
        w = BitWriter()
        for k, part in zip(ks, parts):
            w.write(k, 6)
            mask = (1 << k) - 1
            for v in part.tolist():
                w.write(1, (v >> k) + 1)  # unary quotient: zeros then a 1
                w.write(v & mask, k)
        return bytes([order << 2 | p]) + pack_init(init) + w.getvalue()

    def decode(self, payload, n):
        order, p = payload[0] >> 2, payload[0] & 3
        init, pos = unpack_init(payload, 1, order)
        r = BitReader(payload[pos:])
        sizes = [len(a) for a in np.array_split(np.empty(n - order), 1 << p)]
        u = []
        for size in sizes:
            k = r.read(6)
            for _ in range(size):
                qv = 0
                while not r.read(1):
                    qv += 1
                u.append((qv << k) | r.read(k))
        return from_residual(init, unzigzag(np.array(u, dtype=np.int64)))

    def encode_group(self, Q):
        parts = [self.encode(q) for q in Q]
        return b"".join(varint(len(p)) + p for p in parts)

    def decode_group(self, data, nb, n):
        out, pos = [], 0
        for _ in range(nb):
            size, pos = read_varint(data, pos)
            out.append(self.decode(data[pos:pos + size], n))
            pos += size
        return out


class ByteCoder:
    """Zigzag residuals -> narrowest of 1/2/4/8 bytes -> byte-stream-split
    (all low bytes, then all next bytes, ...) -> general-purpose compressor.
    Tries every predictor order and keeps the smallest."""

    WIDTHS = (1, 2, 4, 8)

    def __init__(self, name, compress, decompress):
        self.name, self._c, self._d = name, compress, decompress

    def _parts(self, q, order):
        """(flags + init varints, residual byte planes) for one order."""
        init, r = to_residual(q, order)
        u = zigzag(r)
        wc = next(i for i, w in enumerate(self.WIDTHS) if u.max(initial=0) < 1 << (8 * w))
        w = self.WIDTHS[wc]
        return bytes([order << 2 | wc]) + pack_init(init), u.astype(f"<u{w}").view(np.uint8).reshape(-1, w).T.tobytes()

    def _best(self, q):
        """The order whose block compresses smallest on its own."""
        return min((self._parts(q, o) for o in ORDERS), key=lambda hp: len(self._c(hp[1])))

    def encode(self, q):
        head, planes = self._best(q)
        body = self._c(planes)
        return head + varint(len(body)) + body

    def decode(self, payload, n):
        order, w = payload[0] >> 2, self.WIDTHS[payload[0] & 3]
        init, pos = unpack_init(payload, 1, order)
        size, pos = read_varint(payload, pos)
        planes = np.frombuffer(self._d(payload[pos:pos + size], (n - order) * w), dtype=np.uint8)
        u = planes.reshape(w, -1).T.copy().view(f"<u{w}").ravel().astype(np.int64)
        return from_residual(init, unzigzag(u))

    def encode_group(self, Q):
        parts = [self._best(q) for q in Q]
        flags = bytes(h[0] for h, _ in parts)
        raw = flags + b"".join(h[1:] for h, _ in parts) + b"".join(p for _, p in parts)
        return self._c(raw)

    def decode_group(self, data, nb, n):
        raw = self._d(data, None)
        pos, inits = nb, []
        for b in range(nb):
            init, pos = unpack_init(raw, pos, raw[b] >> 2)
            inits.append(init)
        out = []
        for b in range(nb):
            order, w = raw[b] >> 2, self.WIDTHS[raw[b] & 3]
            m = (n - order) * w
            planes = np.frombuffer(raw[pos:pos + m], dtype=np.uint8)
            pos += m
            u = planes.reshape(w, -1).T.copy().view(f"<u{w}").ravel().astype(np.int64)
            out.append(from_residual(inits[b], unzigzag(u)))
        return out


def _deflate(b):
    c = zlib.compressobj(9, zlib.DEFLATED, -15)
    return c.compress(b) + c.flush()


_ZC = zstandard.ZstdCompressor(level=9, write_content_size=False, write_checksum=False)
_ZD = zstandard.ZstdDecompressor()


class FlacCoder:
    """libFLAC at level 8 (LPC up to order 12, partitioned Rice) on the quantized
    integers, one frame per block. Only frame bytes are counted: the stream header
    (STREAMINFO etc.) depends only on the block count and length, so the decoder
    rebuilds it."""

    name = "flac"

    def _stream(self, q, n):
        bufs = []
        enc = pyflac.StreamEncoder(1000, lambda buf, nb, ns, fr: bufs.append((bytes(buf), ns)),
                                   compression_level=8, blocksize=n, streamable_subset=False)
        enc.process(np.asarray(q).astype(np.int32))
        enc.finish()
        return bufs

    def _decode_stream(self, frames, nb, n):
        header = b"".join(b for b, ns in self._stream(np.zeros(nb * n, dtype=np.int64), n) if not ns)
        out = []
        dec = pyflac.StreamDecoder(lambda audio, sr, ch, ns: out.append(audio.copy()))
        dec.process(header + frames)
        dec.finish()
        return np.concatenate(out).ravel().astype(np.int64)

    def encode(self, q):
        return self.encode_group([q])

    def decode(self, payload, n):
        return self.decode_group(payload, 1, n)[0]

    def encode_group(self, Q):
        bases = [int(q.min()) for q in Q]
        frames = b"".join(b for b, ns in self._stream(np.concatenate([q - b for q, b in zip(Q, bases)]), len(Q[0])) if ns)
        return b"".join(varint(b) for b in bases) + frames

    def decode_group(self, data, nb, n):
        bases, pos = [], 0
        for _ in range(nb):
            b, pos = read_varint(data, pos)
            bases.append(b)
        q = self._decode_stream(data[pos:], nb, n).reshape(nb, n)
        return [q[b] + bases[b] for b in range(nb)]


class BestOf:
    """Encode each block with every candidate and keep the smallest; one extra
    byte records which coder was used."""

    def __init__(self, name, coders):
        self.name, self.coders = name, coders

    def encode(self, q):
        return min((bytes([i]) + c.encode(q) for i, c in enumerate(self.coders)), key=len)

    def decode(self, payload, n):
        return self.coders[payload[0]].decode(payload[1:], n)

    def encode_group(self, Q):
        ids = [min(range(len(self.coders)), key=lambda i: len(self.coders[i].encode(q))) for q in Q]
        out = bytes(ids)
        for i, c in enumerate(self.coders):
            sub = [q for q, j in zip(Q, ids) if j == i]
            if sub:
                part = c.encode_group(sub)
                out += varint(len(part)) + part
        return out

    def decode_group(self, data, nb, n):
        ids, pos = list(data[:nb]), nb
        out = [None] * nb
        for i, c in enumerate(self.coders):
            idx = [b for b in range(nb) if ids[b] == i]
            if idx:
                size, pos = read_varint(data, pos)
                for b, q in zip(idx, c.decode_group(data[pos:pos + size], len(idx), n)):
                    out[b] = q
                pos += size
        return out


class EntropyBound:
    """Not a codec: zero-order empirical entropy of the best-order residual, plus
    the same header bytes a real block would carry. It is a reference, not a
    true floor: adaptive coders (partitioned Rice, LPC, LZ matches) can beat it."""

    name = "0-order entropy"

    def bits(self, q):
        best = np.inf
        for order in ORDERS:
            init, r = to_residual(q, order)
            _, counts = np.unique(r, return_counts=True)
            prob = counts / len(r)
            best = min(best, 8 * (1 + len(pack_init(init))) - (counts * np.log2(prob)).sum())
        return best


RICE = RiceCoder()
ZSTD = ByteCoder("delta0123-zstd", _ZC.compress,
                 lambda b, size: _ZD.decompress(b, max_output_size=size) if size else _ZD.decompressobj().decompress(b))
DEFLATE = ByteCoder("delta0123-deflate", _deflate, lambda b, size: zlib.decompress(b, -15))
FLAC = FlacCoder()
CODERS = [
    RICE,
    ZSTD,
    DEFLATE,
    FLAC,
    BestOf("best: rice|zstd", [RICE, ZSTD]),
    BestOf("best: rice|zstd|flac", [RICE, ZSTD, FLAC]),
    EntropyBound(),
]


# ---------------------------------------------------------------------------
# Rate control
# ---------------------------------------------------------------------------

def quantize(xb, step):
    return np.rint(xb / step).astype(np.int64)


def block_bits(coder, xb, step):
    """Bits for one block coded on its own: e byte + coder payload. Drives the per-block cap."""
    q = quantize(xb, step)
    if isinstance(coder, EntropyBound):
        return 8 + coder.bits(q)
    return 8 * (1 + len(coder.encode(q)))


def group_bits(coder, blocks, steps):
    """Bits for the whole block group: delta f64 + one e byte per block + the coder's block group."""
    Q = [quantize(xb, st) for xb, st in zip(blocks, steps)]
    if isinstance(coder, EntropyBound):
        return 64 + sum(8 + coder.bits(q) for q in Q)
    return 64 + 8 * len(blocks) + 8 * len(coder.encode_group(Q))


def cap_exponent(coder, xb, delta, cap_bits):
    """Smallest e in 0..255 with block bits <= cap (binary search; bits fall as e grows)."""
    if block_bits(coder, xb, delta) <= cap_bits:
        return 0
    lo, hi = 0, 255
    while lo < hi:
        mid = (lo + hi) // 2
        if block_bits(coder, xb, delta * 2 ** (mid / E_STEPS)) <= cap_bits:
            hi = mid
        else:
            lo = mid + 1
    return lo


def _steps(delta, es):
    return [delta * 2 ** (e / E_STEPS) for e in es]


def window_plan(coder, blocks):
    """One delta for the whole block group; blocks above the cap (coded on their own) get their own coarser
    step. Bisect log2(delta) so the block group averages TARGET_BPS, every byte included."""
    budget = TARGET_BPS * blocks.size
    cap = CAP_BPS * blocks.shape[1]
    es_at = {}

    def exps(log_delta):  # the cap exponents only depend on delta; cache them across the two passes
        if log_delta not in es_at:
            es_at[log_delta] = [cap_exponent(coder, xb, 2.0 ** log_delta, cap) for xb in blocks]
        return es_at[log_delta]

    lo, hi = -30.0, 0.0  # log2(delta); data are in [0, 1]
    for _ in range(30):
        mid = (lo + hi) / 2
        if group_bits(coder, blocks, _steps(2.0 ** mid, exps(mid))) <= budget:
            hi = mid
        else:
            lo = mid
    return 2.0 ** hi, exps(hi)


def fixed_plan(coder, blocks):
    """Every block individually (coded on its own) at TARGET_BPS; same format (delta = finest block
    step). Blocks that can't use their share leave it unspent, and the shared block group compresses better
    than the blocks alone, so the block group lands under the budget."""
    per_block = TARGET_BPS * blocks.shape[1] - 64 / len(blocks)  # share the block group header
    finest = []
    for xb in blocks:
        lo, hi = -30.0, 0.0
        for _ in range(30):
            mid = (lo + hi) / 2
            if block_bits(coder, xb, 2.0 ** mid) <= per_block:
                hi = mid
            else:
                lo = mid
        finest.append(hi)
    delta = 2.0 ** min(finest)
    return delta, [int(np.ceil(E_STEPS * (f - min(finest)))) for f in finest]


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

def describe(coder, payload):
    """Which coder / predictor a block ended up using."""
    if isinstance(coder, BestOf):
        sub = coder.coders[payload[0]]
        return sub.name + " " + describe(sub, payload[1:])
    return "LPC" if isinstance(coder, FlacCoder) else f"order {payload[0] >> 2}"


def error_row(xb, y, step):
    err = y - xb
    return {"step": step, "rmse": np.sqrt(np.mean(err ** 2)), "maxabs": np.abs(err).max(),
            "mean": err.mean(), "max": y.max() - xb.max(), "min": y.min() - xb.min()}


def evaluate(coder, blocks, planner):
    """Plan, encode and decode one block group (blocks: [nb, n] in [0, 1]). Per-block rows carry the error
    and the bits the block would take on its own (the block group's bits are only known as a whole)."""
    t0 = time.perf_counter()
    delta, es = planner(coder, blocks)
    steps = _steps(delta, es)
    Q = [quantize(xb, st) for xb, st in zip(blocks, steps)]
    if isinstance(coder, EntropyBound):
        total_bits = group_bits(coder, blocks, steps)
    else:
        data = coder.encode_group(Q)
        total_bits = 64 + 8 * len(blocks) + 8 * len(data)
    enc_s = time.perf_counter() - t0
    if not isinstance(coder, EntropyBound):
        Q2 = coder.decode_group(data, len(blocks), blocks.shape[1])
        assert all(np.array_equal(a, b) for a, b in zip(Q, Q2)), f"{coder.name}: lossless stage failed"
    rows = []
    for xb, e, st, q in zip(blocks, es, steps, Q):
        y = q * st
        assert np.abs(y - xb).max() <= st / 2 + 1e-12
        if isinstance(coder, EntropyBound):
            bits, order = 8 + coder.bits(q), ""
        else:
            payload = coder.encode(q)
            bits, order = 8 * (1 + len(payload)), describe(coder, payload)
        rows.append({"e": e, "bps": bits / len(xb), "order": order, **error_row(xb, y, st)})
    return {"delta": delta, "bps": total_bits / blocks.size, "rows": rows,
            "us_per_block": 1e6 * enc_s / len(blocks)}


def evaluate_delta_zstd(codec, blocks):
    """delta0123-zstd (one-shot encoder, no rate search) on the same block group: its block group includes each
    block's min and max."""
    codec.encode_group(blocks[:1])  # compile outside the timing
    t0 = time.perf_counter()
    data, infos = codec.encode_group(blocks)
    enc_s = time.perf_counter() - t0
    Y = codec.decode_group(data, *blocks.shape)
    rows = []
    for xb, y, info in zip(blocks, Y, infos):
        step = (xb.max() - xb.min()) / ((1 << info["bits"]) - 1)
        assert np.abs(y - xb).max() <= step / 2 * (1 + 1e-9)
        alone = 8 * len(codec.encode_group(xb[None])[0]) / len(xb)  # the block as a block group of its own
        rows.append({"e": int(info["bits"] < codec.bits), "bps": alone,
                     "order": f"B={info['bits']} order {info['order']}", **error_row(xb, y, step)})
    return {"delta": None, "bps": 8 * len(data) / blocks.size, "rows": rows, "us_per_block": 1e6 * enc_s / len(blocks)}


def baseline_quant8_deflate(blocks):
    c = SharedBackend(QuantDeltaDeflate(8))
    data, _ = c.encode_group(blocks)
    Y = c.decode_group(data, *blocks.shape)
    return {"bps": 8 * len(data) / blocks.size, "rows": [error_row(xb, y, None) for xb, y in zip(blocks, Y)]}
