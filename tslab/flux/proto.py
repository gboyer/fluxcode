# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""fluxproto, the fluxcode prototype: a one-minute block group codec, optionally with step (quantum) detection:
delta0123-zstd's quantize -> predict -> zstd with a power-of-two step, fixed B, and
blocks interleaved by field.

Per block (min and max - min of the block's values):
    k      : max - min in [2^(k-1), 2^k)            (frexp)
    e      = k - B; +1 if max - min would round to 2^B
    q      = round((x - min) * 2^-e)                  0 <= q < 2^B, min exact
    x'     = min + q * 2^e                            (power-of-two step: re-encoding x' gives q back)

Decimal step detection (step_detect=True): real values are often on a decimal
grid (0.1 units, 0.001, integers), which a power-of-two grid can't express compactly.
    tol    = 2^e / 4       a value within tol of the grid counts as on it, so detection
                           never gives a larger max error than the power-of-two grid at the same B
    p      : coarsest p with 10^p > 2^e (and 10^p <= max - min) such that every x is within
             tol of a multiple of 10^p (absolute grid); a failing p usually exits on its
             first sample, so continuous data costs almost nothing
    q      = round(x / 10^p) - K0,  K0 = round(min / 10^p)
    x'     = (K0 + q) / 10^-p  (p < 0: integer, then one correctly rounded division, the same
             double as parsing the decimal text)  or  (K0 + q) * 10^p  (p >= 0)
    float32: if every input is exactly a float32 (stored as float32 upstream), a flag makes
             the decoder round its output to float32, so those values come back bit-exact
    Other steps (ADC counts, 0.5, 0.25) fall back to the power-of-two grid.

Noise floor (noise_f = f, encoder-only; the format is unchanged):
    d      = second differences of x (a smooth signal mostly cancels)
    sigma  = robust std(d) / sqrt(6), std from the mean absolute deviation of d (minus its mean),
             re-estimated twice with values beyond 4 robust stds dropped (linear passes, no sort)
    rho    = lag-1 autocorrelation of d, clipped at 4 robust sigmas (so spikes don't dominate)
             white noise: -2/3; random walk (increments are signal): -1/2; smooth or fast
             deterministic signal: positive
    if rho < noise_rho (default -0.6): e = max(e, floor(log2(f * sigma)))
    i.e. the step is at most f * sigma and still a power of two; quantization error adds
    at most f * sigma / sqrt(12) (RMS) to noise already sigma.
    mode 0 = power of two (param = e), 1 = decimal (param = p)
    (A general step detector, approximate float GCD of the deltas, was tried first. Float
     noise compounds through Euclid, and tracking it made the detector fragile; dropped.)

    order  = argmin over k in 0..3 of var(diff(q, k))   (exact integer sums), or fixed
    r      = diff^order of (0,..,0, q), mod 2^16, read as int16, zigzag -> uint16
             (two bytes per sample for any order and any B <= 16; the leading zeros
              make the first residuals the start values, so no separate header)

Block group per minute (N blocks), then zstd-3, interleaved by field, block minima last:
    order | mode << 2 | float32 << 4 (x N) | param byte 0 (x N) .. param byte 7 (x N) | low bytes (x N) | high bytes (x N)
    | min byte 0 (x N) .. min byte 7 (x N)
raw() is the block group without the minima (what the plane benches analyze); encode_group / decode_group
are the whole block group (tslab.common.group's interface), decodable from its bytes alone.
("block" layout, for comparison: the same fields one block at a time.)
planes="bit": the low and high byte planes are bit-shuffled instead: 16 bit planes x N blocks
(an 8x8 bit transpose per 8 samples, its own inverse), so small residuals leave whole planes of zeros.
planes="nibble": 4 planes of 4 bits (bits 0-3, 4-7, 8-11, 12-15) x N blocks, two samples per byte
(sample 2i in the low nibble, 2i + 1 in the high nibble).
"""

import math

import numpy as np
import zstandard
from numba import njit

from tslab.common.intcode import pick_order

POWER_OF_TWO, DECIMAL = 0, 1


@njit(cache=True)
def _exponent(rng, bits):
    if not rng > 0:
        return 0  # constant block: q = 0, reconstructs min exactly
    _, k = math.frexp(rng)  # rng in [2^(k-1), 2^k)
    e = k - bits
    if math.floor(math.ldexp(rng, -e) + 0.5) >= (1 << bits):
        e += 1
    return e


@njit(cache=True, fastmath=True)  # encoder-only estimate: reassociated sums vectorize (~4x)
def _noise(x):
    """(sigma, rho): robust white-noise level from second differences, and their lag-1
    autocorrelation after clipping at 4 robust sigmas. Linear passes only (no sort):
    mean absolute deviation, re-estimated twice with values beyond 4 robust sigmas dropped."""
    n = x.shape[0]
    m = n - 2
    d = np.empty(m)
    mean = 0.0
    for i in range(m):
        d[i] = x[i + 2] - 2.0 * x[i + 1] + x[i]
        mean += d[i]
    mean /= m  # removes a constant second difference (quadratics)
    mad = 0.0
    for i in range(m):
        d[i] -= mean
        mad += abs(d[i])
    mad /= m
    k = math.sqrt(math.pi / 2)  # std / mean absolute deviation, Gaussian
    for _ in range(2):
        c = 4.0 * k * mad
        s = 0.0
        cnt = 0
        for i in range(m):
            v = abs(d[i])
            if v <= c:
                s += v
                cnt += 1
        mad = s / cnt if cnt > 0 else 0.0
    sd = k * mad
    if not sd > 0:
        return 0.0, 0.0
    c = 4.0 * sd
    s0 = s1 = 0.0
    prev = 0.0
    for i in range(m):
        v = min(max(d[i], -c), c)
        s0 += v * v
        if i > 0:
            s1 += v * prev
        prev = v
    return sd / math.sqrt(6.0), s1 / s0


@njit(cache=True)
def _detect_decimal(x, lo, hi, ps):
    """Coarsest p with 10^p > ps such that every x is within ps/4 of a multiple of 10^p; else 99."""
    tol = 0.25 * ps
    p = int(math.floor(math.log10(hi - lo)))
    while 10.0 ** p > ps:
        s = 10.0 ** -p  # exact for -22 <= p <= 0
        ts = tol * s
        ok = True
        for i in range(x.shape[0]):
            v = x[i] * s
            if abs(v - math.floor(v + 0.5)) > ts:
                ok = False
                break
        if ok and max(abs(lo), abs(hi)) * s < 2.0 ** 52:
            return p
        p -= 1
    return 99


@njit(cache=True)
def _quantize(x, lo, e, q):
    s = math.ldexp(1.0, -e)
    for i in range(x.shape[0]):
        q[i] = np.int32(math.floor((x[i] - lo) * s + 0.5))


@njit(cache=True)
def _quantize_decimal(x, p, K0, q):
    s = 10.0 ** -p
    for i in range(x.shape[0]):
        q[i] = np.int32(np.int64(math.floor(x[i] * s + 0.5)) - K0)


@njit(cache=True)
def _residual_u16(q, order, r, u):
    n = q.shape[0]
    for i in range(n):
        r[i] = q[i]
    for _ in range(order):  # diff with a zero prepended, in place from the end
        for i in range(n - 1, 0, -1):
            r[i] = r[i] - r[i - 1]
    for i in range(n):
        v = ((r[i] + 32768) & 0xFFFF) - 32768  # mod 2^16 as int16
        u[i] = np.uint16(((v << 1) ^ (v >> 15)) & 0xFFFF)


@njit(inline="always")
def _transpose8(x):
    """8x8 bit-matrix transpose of a uint64 (byte k, bit j) -> (byte j, bit k); its own inverse."""
    t = (x ^ (x >> np.uint64(7))) & np.uint64(0x00AA00AA00AA00AA)
    x = x ^ t ^ (t << np.uint64(7))
    t = (x ^ (x >> np.uint64(14))) & np.uint64(0x0000CCCC0000CCCC)
    x = x ^ t ^ (t << np.uint64(14))
    t = (x ^ (x >> np.uint64(28))) & np.uint64(0x00000000F0F0F0F0)
    return x ^ t ^ (t << np.uint64(28))


@njit(cache=True)
def _bitshuffle(low64, high64, out):
    """low64/high64: (nb, n/8) uint64 views of the byte planes -> out (16, nb, n/8): bit plane j of every block."""
    nb, g = low64.shape
    for b in range(nb):
        for i in range(g):
            x = _transpose8(low64[b, i])
            y = _transpose8(high64[b, i])
            for j in range(8):
                out[j, b, i] = np.uint8((x >> np.uint64(8 * j)) & np.uint64(255))
                out[8 + j, b, i] = np.uint8((y >> np.uint64(8 * j)) & np.uint64(255))


@njit(cache=True)
def _bitunshuffle(planes, low64, high64):
    nb, g = low64.shape
    for b in range(nb):
        for i in range(g):
            x = np.uint64(0)
            y = np.uint64(0)
            for j in range(8):
                x |= np.uint64(planes[j, b, i]) << np.uint64(8 * j)
                y |= np.uint64(planes[8 + j, b, i]) << np.uint64(8 * j)
            low64[b, i] = _transpose8(x)
            high64[b, i] = _transpose8(y)


@njit(cache=True)
def _nibbleshuffle(low, high, out):
    """low/high: (nb, n) byte planes -> out (4, nb, n/2): nibble plane j of every block."""
    nb, n = low.shape
    for b in range(nb):
        for i in range(n // 2):
            a0, a1 = low[b, 2 * i], low[b, 2 * i + 1]
            c0, c1 = high[b, 2 * i], high[b, 2 * i + 1]
            out[0, b, i] = (a0 & 15) | ((a1 & 15) << 4)
            out[1, b, i] = (a0 >> 4) | (a1 & 0xF0)
            out[2, b, i] = (c0 & 15) | ((c1 & 15) << 4)
            out[3, b, i] = (c0 >> 4) | (c1 & 0xF0)


@njit(cache=True)
def _nibbleunshuffle(planes, low, high):
    nb, n = low.shape
    for b in range(nb):
        for i in range(n // 2):
            p0, p1, p2_, p3 = planes[0, b, i], planes[1, b, i], planes[2, b, i], planes[3, b, i]
            low[b, 2 * i] = (p0 & 15) | ((p1 & 15) << 4)
            low[b, 2 * i + 1] = (p0 >> 4) | (p1 & 0xF0)
            high[b, 2 * i] = (p2_ & 15) | ((p3 & 15) << 4)
            high[b, 2 * i + 1] = (p2_ >> 4) | (p3 & 0xF0)


@njit(cache=True)
def _encode_group(X, lo, hi, bits, orders, detect, head, param_i, low, high, noise_f=0.0, noise_rho=-0.6):
    """head[b] = order | mode << 2 | float32 << 4; param[b] = e or p; byte planes."""
    nb, n = X.shape
    q = np.empty(n, np.int32)
    r = np.empty(n, np.int32)
    u = np.empty(n, np.uint16)
    for b in range(nb):
        e = _exponent(hi[b] - lo[b], bits)
        if noise_f > 0 and hi[b] > lo[b]:
            sigma, rho = _noise(X[b])
            if rho < noise_rho and sigma > 0:
                e = max(e, int(math.floor(math.log2(noise_f * sigma))))
        mode = POWER_OF_TWO
        if detect and hi[b] > lo[b]:
            p = _detect_decimal(X[b], lo[b], hi[b], math.ldexp(1.0, e))
            if p != 99:
                mode = DECIMAL
                param_i[b] = p
                _quantize_decimal(X[b], p, np.int64(math.floor(lo[b] * 10.0 ** -p + 0.5)), q)
                f32 = True  # every input is a float32: the decoder rounds back to float32
                for i in range(n):
                    if np.float64(np.float32(X[b, i])) != X[b, i]:
                        f32 = False
                        break
                if f32:
                    mode |= 4
        if mode == POWER_OF_TWO:
            param_i[b] = e
            _quantize(X[b], lo[b], e, q)
        mode_bits = mode
        mode &= 3
        if orders.shape[0] == 1:
            order = orders[0]
        else:
            order, _ = pick_order(q)  # orders 0..3
        head[b] = order | mode_bits << 2
        _residual_u16(q, order, r, u)
        for i in range(n):
            low[b, i] = np.uint8(u[i] & 255)
            high[b, i] = np.uint8(u[i] >> 8)


@njit(cache=True)
def _decode_group(head, param_i, low, high, lo, X):
    nb, n = X.shape
    v = np.empty(n, np.int64)
    for b in range(nb):
        order, mode, f32 = head[b] & 3, (head[b] >> 2) & 3, (head[b] >> 4) & 1
        for i in range(n):
            z = np.int64(low[b, i]) | (np.int64(high[b, i]) << 8)
            v[i] = (z >> 1) ^ -(z & 1)
        for _ in range(order):
            acc = np.int64(0)
            for i in range(n):
                acc = (acc + v[i]) & 0xFFFF
                v[i] = acc
        if order == 0:
            for i in range(n):
                v[i] &= 0xFFFF
        if mode == POWER_OF_TWO:
            step = math.ldexp(1.0, param_i[b])
            for i in range(n):
                X[b, i] = lo[b] + v[i] * step
        else:
            p = param_i[b]
            K0 = np.int64(math.floor(lo[b] * 10.0 ** -p + 0.5))
            if p < 0:
                den = 10.0 ** -p  # exact for p >= -22
                for i in range(n):
                    X[b, i] = float(K0 + v[i]) / den
            else:
                mul = 10.0 ** p
                for i in range(n):
                    X[b, i] = float(K0 + v[i]) * mul
        if f32:
            for i in range(n):
                X[b, i] = np.float64(np.float32(X[b, i]))


class FluxProto:
    """fluxproto[-decimal]-<bits>[...]: the prototype on one minute (blocks X[nb, n]); the name's
    suffixes list the non-default options (see the module docstring for the format)."""

    def __init__(self, bits=16, orders=(0, 1, 2, 3), step_detect=False, layout="field", level=3, n=1000, planes="byte",
                 noise_f=0.0, noise_rho=-0.6):
        assert 9 <= bits <= 16 and layout in ("field", "block") and tuple(orders) in ((0, 1, 2, 3), (0,), (1,), (2,), (3,))
        assert planes in ("byte", "nibble", "bit") and (planes == "byte" or (layout == "field" and n % 8 == 0))
        self.bits, self.n, self.detect, self.field, self.planes = bits, n, step_detect, layout == "field", planes
        self.bitplanes = planes == "bit"
        self.noise_f, self.noise_rho = float(noise_f), float(noise_rho)
        self.orders = np.array(orders, np.int64)
        self.name = (f"fluxproto{'-decimal' if step_detect else ''}-{bits}" + ("" if len(orders) == 4 else f"-o{orders[0]}")
                     + ("" if layout == "field" else "-blocklayout") + ("" if level == 3 else f"-zstd{level}") + {"byte": "", "bit": "-bitshuffle", "nibble": "-nibble"}[planes] + (f"-nf{noise_f:g}" if noise_f else ""))
        self._c = zstandard.ZstdCompressor(level=level, write_checksum=False).compress
        self._d = zstandard.ZstdDecompressor().decompress

    def raw(self, X, lo, hi):
        nb, n = X.shape
        head = np.empty(nb, np.uint8)
        param = np.zeros(nb, np.int64)
        low, high = np.empty((nb, n), np.uint8), np.empty((nb, n), np.uint8)
        _encode_group(X, lo, hi, self.bits, self.orders, self.detect, head, param, low, high, self.noise_f, self.noise_rho)
        pbytes = param.view(np.uint8).reshape(nb, 8)
        if self.bitplanes:
            planes = np.empty((16, nb, n // 8), np.uint8)
            _bitshuffle(low.view(np.uint64), high.view(np.uint64), planes)
            return np.concatenate([head, pbytes.T.ravel(), planes.ravel()])
        if self.planes == "nibble":
            planes = np.empty((4, nb, n // 2), np.uint8)
            _nibbleshuffle(low, high, planes)
            return np.concatenate([head, pbytes.T.ravel(), planes.ravel()])
        if self.field:
            return np.concatenate([head, pbytes.T.ravel(), low.ravel(), high.ravel()])
        return np.concatenate([head[:, None], pbytes, low, high], axis=1).ravel()

    def encode_group(self, X):
        """One minute X[nb, n] -> (group, infos): zstd(raw || block minima as byte-planed float64)."""
        lo, hi = X.min(1), X.max(1)
        raw = self.raw(X, lo, hi)
        anchor = np.ascontiguousarray(lo, "<f8").view(np.uint8).reshape(-1, 8).T.ravel()
        nb = len(X)
        infos = [{"order": int(h & 3), "decimal": bool(h >> 2 & 1)} for h in raw[:nb]] if self.field else None
        return self._c(np.concatenate([raw, anchor]).tobytes()), infos

    def decode_group(self, data, nb, n):
        assert n == self.n
        buf = np.frombuffer(self._d(data), np.uint8)
        raw = buf[:-8 * nb]
        lo = np.ascontiguousarray(buf[-8 * nb:].reshape(8, nb).T).view("<f8").ravel()
        if self.field:
            head = raw[:nb]
            param = np.ascontiguousarray(raw[nb:9 * nb].reshape(8, nb).T).view(np.int64).ravel()
            if self.bitplanes:
                low, high = np.empty((nb, n), np.uint8), np.empty((nb, n), np.uint8)
                _bitunshuffle(raw[9 * nb:].reshape(16, nb, n // 8), low.view(np.uint64), high.view(np.uint64))
            elif self.planes == "nibble":
                low, high = np.empty((nb, n), np.uint8), np.empty((nb, n), np.uint8)
                _nibbleunshuffle(raw[9 * nb:].reshape(4, nb, n // 2), low, high)
            else:
                low = raw[9 * nb:9 * nb + nb * n].reshape(nb, n)
                high = raw[9 * nb + nb * n:].reshape(nb, n)
        else:
            rows = raw.reshape(nb, 9 + 2 * n)
            head = np.ascontiguousarray(rows[:, 0])
            param = np.ascontiguousarray(rows[:, 1:9]).view(np.int64).ravel()
            low, high = np.ascontiguousarray(rows[:, 9:9 + n]), np.ascontiguousarray(rows[:, 9 + n:])
        X = np.empty((nb, n))
        _decode_group(head, param, low, high, lo, X)
        return X
