# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""A varied corpus of one-minute units with the true outcome of the bit-vs-byte choice, and the
unit's residuals, for plane_layout/analyze.py.

Signals are drawn from families with random parameters (frequencies, noise levels, quanta,
precision limits), so a predictor is tested on shapes it wasn't tuned on. Each unit is encoded
with planes="bit" and planes="byte"; the real unit sizes are the ground truth. The residuals
(uint16, as stored) come from the byte-plane body.

    uv run python plane_layout/corpus.py [--n 2000] [--seed 1] [--out corpus.npz]
    uv run python plane_layout/corpus.py --standard --out standard.npz   # the report's signal kinds, default params
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
import fluxcode
from fluxcode import Params, _format, _unit

N = 60_000
FS = 1000.0


def lowpass(rng, phi):
    """AR(1) noise with unit variance."""
    e = rng.normal(0, np.sqrt(1 - phi ** 2), N)
    out = np.empty(N)
    acc = rng.normal()
    for i in range(N):
        acc = phi * acc + e[i]
        out[i] = acc
    return out


def draw_signal(rng):
    """(family, description, one minute of signal)."""
    t = np.arange(N) / FS + rng.uniform(0, 1000)
    amp = 10 ** rng.uniform(0, 3)
    off = rng.uniform(-1, 1) * amp * rng.choice([0, 0, 5])
    fam = rng.choice(["sine", "int-period sine", "noisy sine", "multi sine", "random walk", "ar1", "white",
                      "square", "sawtooth", "triangle", "steps", "spikes", "ramp", "chirp"])
    if fam == "sine":
        f = 10 ** rng.uniform(-0.3, 1.8)
        x = amp * np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28))
    elif fam == "int-period sine":
        p = int(rng.choice([8, 16, 20, 25, 40, 50, 100, 125, 200, 250, 500, 7, 13, 99, 333]))
        x = amp * np.sin(2 * np.pi * np.arange(N) / p + rng.uniform(0, 6.28))
        fam += f" P={p}"
    elif fam == "noisy sine":
        f = 10 ** rng.uniform(-0.3, 1.8)
        snr = 10 ** rng.uniform(0.5, 3.5)
        x = amp * np.sin(2 * np.pi * f * t) + rng.normal(0, amp / snr, N)
    elif fam == "multi sine":
        x = sum(amp / (k + 1) * np.sin(2 * np.pi * 10 ** rng.uniform(-0.3, 1.8) * t + rng.uniform(0, 6.28)) for k in range(3))
    elif fam == "random walk":
        x = np.cumsum(rng.normal(0, amp / 100 * 10 ** rng.uniform(-1.5, 0.5), N))
    elif fam == "ar1":
        x = amp * lowpass(rng, float(rng.choice([0.0, 0.5, 0.9, 0.99, 0.999])))
    elif fam == "white":
        x = amp * (rng.normal(0, 1, N) if rng.random() < 0.5 else rng.uniform(-1, 1, N))
    elif fam in ("square", "sawtooth", "triangle"):
        p = float(rng.choice([8, 50, 125, 200, 500, 1000 / 2.24, 1000 / 7.3]))
        ph = (np.arange(N) / p) % 1
        x = amp * {"square": np.where(ph < 0.5, 1.0, -1.0), "sawtooth": 2 * ph - 1,
                   "triangle": 1 - 4 * np.abs(ph - 0.5)}[fam]
        x = x + (rng.normal(0, amp * 10 ** rng.uniform(-4, -1), N) if rng.random() < 0.5 else 0)
        fam += f" P={p:.4g}"
    elif fam == "steps":
        k = int(rng.integers(5, 200))
        x = np.repeat(rng.normal(0, amp, k), N // k + 1)[:N] + rng.normal(0, amp * 10 ** rng.uniform(-4, -2), N)
    elif fam == "spikes":
        x = rng.normal(0, amp * 0.05, N)
        x[rng.integers(0, N, int(rng.integers(20, 600)))] += amp * rng.choice([5, 20])
    elif fam == "ramp":
        x = amp * np.linspace(0, 1, N) ** rng.choice([1, 2, 3]) + rng.normal(0, amp * 10 ** rng.uniform(-5, -2), N)
    else:  # repeating chirp
        frac = (np.arange(N) % 1000) / 1000
        f = rng.uniform(1, 5) + rng.uniform(5, 50) * frac
        x = amp * np.sin(np.cumsum(2 * np.pi * f / FS))
        x = x + (rng.normal(0, amp * 10 ** rng.uniform(-4, -1.5), N) if rng.random() < 0.5 else 0)
    x = x + off
    q = None
    if rng.random() < 0.4:  # a sensor's resolution: decimal quantum or a power-of-two LSB
        q = float(rng.choice([0.1, 0.01, 0.001, 2.0 ** -4, 2.0 ** -7, 0.5, 1.0]))
        den = round(1 / q)
        x = np.round(x * den) / den if abs(den * q - 1) < 1e-12 and den % 10 == 0 else np.round(x / q) * q
    return fam, q, x


def configs(rng):
    mx = int(rng.choice([8, 10, 12, 14, 16]))
    nf = None if rng.random() < 0.3 else 0.25
    return Params(max_quantize_bits=mx, noise_floor_sigma=nf), mx, nf


def residuals(unit):
    parsed = _unit.decompress(unit)
    assert parsed.header.num_blocks * 1000 == parsed.header.num_samples == N and parsed.header.byte_planes
    start = _format.residual_start(parsed.header.num_blocks, parsed.has_time)
    body = parsed.raw_body
    return (body[start:start + N].astype(np.uint16) | (body[start + N:start + 2 * N].astype(np.uint16) << 8))


def record(x, mx=16, nf=0.25):
    """(real bit-layout unit size, real byte-layout unit size, residuals) of one minute."""
    (u_bit,), *_ = fluxcode.encode(x, Params(max_quantize_bits=mx, noise_floor_sigma=nf, planes="bit"))
    (u_byte,), *_ = fluxcode.encode(x, Params(max_quantize_bits=mx, noise_floor_sigma=nf, planes="byte"))
    return len(u_bit), len(u_byte), residuals(u_byte)


def standard():
    """The report's signal kinds and discretized variants (tests/_signals.py), 8 minutes each, default Params."""
    from _signals import DISCRETE, KINDS, discrete_minute, minute
    names = KINDS + [n for n, _, _ in DISCRETE]
    rows = [(name, *record(minute(name, 500 + j) if name in KINDS else discrete_minute(name, 500 + j))) for name in names for j in range(8)]
    return {"fam": np.array([r[0] for r in rows]), "q": np.full(len(rows), -1.0), "max_bits": np.full(len(rows), 16),
            "noise_floor": np.full(len(rows), 0.25), "bit": [r[1] for r in rows], "byte": [r[2] for r in rows],
            "u": np.stack([r[3] for r in rows])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="corpus.npz")
    ap.add_argument("--standard", action="store_true")
    a = ap.parse_args()
    out = Path(__file__).with_name(a.out)
    if a.standard:
        data = standard()
    else:
        rng = np.random.default_rng(a.seed)
        cols = {k: [] for k in ("fam", "q", "max_bits", "noise_floor", "bit", "byte", "u")}
        while len(cols["u"]) < a.n:
            fam, q, x = draw_signal(rng)
            _, mx, nf = configs(rng)
            bit, byte, u = record(x, mx, nf)
            for k, v in zip(cols, (fam, -1.0 if q is None else q, mx, -1.0 if nf is None else nf, bit, byte, u)):
                cols[k].append(v)
            if len(cols["u"]) % 500 == 0:
                print(len(cols["u"]), flush=True)
        data = {k: np.array(v) if k != "u" else np.stack(v) for k, v in cols.items()}
    np.savez_compressed(out, **data)
    bit, byte = np.array(data["bit"]), np.array(data["byte"])
    print(f"{len(bit)} units; byte smaller on {(byte < bit).mean():.0%}; always-bit {bit.sum()}, always-byte {byte.sum()}, "
          f"oracle {np.minimum(bit, byte).sum()} bytes -> {out}")


if __name__ == "__main__":
    main()
