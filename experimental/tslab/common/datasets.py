# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Every test signal used by the experiments: synthetic 1 kHz signals, one continuous minute
(60 blocks of 1000 samples, one unit) at a time.

- minute(kind, seed) / KINDS: one continuous minute of each signal type. Periodic signals use
  irrational frequencies so no cycle lands on a whole number of samples; integer Hz at 1 kHz
  repeats exactly and flatters dictionary coders like deflate.
- units(): the report's set, REPORT_MINUTES minutes of each kind (3,600 blocks).
- KIND_INFO: per kind, a description and the second (block) the report plots, with its zoom window.
- peaks(kind, seed): sample indices of the spikes / impulses, for the peak-shift column.
- load(): the 7,200-block set (10 minutes of each of the 12 kinds) used by the benchmarks.
- load_discrete() / DISCRETE: 5,400 blocks of signals rounded to a fixed quantum.
- noisy_sets(): the noisy signals, each with its noise-free part, for the noise-floor sweeps.

All seeds are fixed, so every run sees the same data.
"""

import numpy as np

# ---------------------------------------------------------------------------
# One-minute signals: 60 consecutive 1000-sample blocks of one channel
# ---------------------------------------------------------------------------

BLOCK, PER_CHUNK = 1000, 60
MINUTE = BLOCK * PER_CHUNK


def minute(kind, seed):
    """One continuous minute (60,000 samples at 1 kHz) of a dataset type."""
    rng = np.random.default_rng(seed)
    t = np.arange(MINUTE, dtype=np.float64) + rng.uniform(0, 1e6)  # random start time
    ph = rng.uniform(0, 2 * np.pi)
    if kind == "linear":
        return t - t[0]
    if kind == "quadratic":
        return (t - t[0]) ** 2 / 1000
    if kind.startswith("sin-"):
        f = {"sin-4.12hz": np.sqrt(17), "sin-9.87hz": np.pi ** 2, "sin-50.3hz": 16 * np.pi}[kind]
        return 100 * np.sin(2 * np.pi * f * t / 1000 + ph)
    if kind == "gauss-spikes":
        x = rng.uniform(-10, 10, MINUTE)
        for c in rng.uniform(0, MINUTE, 180):  # ~3 per second
            x += 1000 * np.exp(-((np.arange(MINUTE) - c) ** 2) / 200)
        return x
    if kind == "impulses":
        x = rng.uniform(-10, 10, MINUTE)
        x[rng.integers(0, MINUTE, 180)] += 500
        return x
    if kind == "square-2.24hz":
        return np.where(np.sin(2 * np.pi * np.sqrt(5) * t / 1000 + ph) >= 0, 100.0, -100.0)
    if kind == "random-walk":
        return np.cumsum(rng.normal(0, 1, MINUTE))
    if kind == "chirp":  # repeating 1-second sweep √2 -> 32π Hz, phase continuous
        frac = (np.arange(MINUTE) % 1000) / 1000
        f = np.sqrt(2) + (32 * np.pi - np.sqrt(2)) * frac
        return 100 * np.sin(np.cumsum(2 * np.pi * f / 1000) + ph)
    if kind == "noisy-sine":
        return 100 * np.sin(2 * np.pi * np.pi ** 2 * t / 1000 + ph) + rng.normal(0, 5, MINUTE)
    if kind == "sensor-0.1":
        return np.round(50 + 5 * np.sin(2 * np.pi * np.sqrt(2) * t / 1000 + ph)
                        + np.cumsum(rng.normal(0, 0.05, MINUTE)), 1)
    raise ValueError(kind)


KINDS = ["linear", "quadratic", "sin-4.12hz", "sin-9.87hz", "sin-50.3hz", "gauss-spikes", "impulses",
         "square-2.24hz", "random-walk", "chirp", "noisy-sine", "sensor-0.1"]

# kind -> (description, plotted block, zoom window within it or None). The report plots one second of
# the first report minute (seed 1000 * kind index), chosen to show the kind's feature: the second
# with the largest spike for gauss-spikes (two overlapping spikes) and impulses, a second with an
# edge in the zoom for square, the high-frequency end of the sweep for chirp; any second (the first)
# for the smooth signals. Zoom windows are sample offsets within that second.
KIND_INFO = {
    "linear": ("Straight line, slope 1 per sample", 0, None),
    "quadratic": ("t²/1000 from the start of the minute", 0, None),
    "sin-4.12hz": ("100·sin, √17 ≈ 4.123 Hz, random phase", 0, None),
    "sin-9.87hz": ("100·sin, π² ≈ 9.870 Hz, random phase", 0, None),
    "sin-50.3hz": ("100·sin, 16π ≈ 50.27 Hz (~19.9 samples/cycle), random phase", 0, (0, 100)),
    "gauss-spikes": ("Uniform noise ±10 + Gaussian spikes (σ = 10 samples, peak 1000), 180 per minute at random times",
                     32, (690, 770)),
    "impulses": ("Uniform noise ±10 + single-sample impulses of +500, 180 per minute at random samples", 56, (576, 611)),
    "square-2.24hz": ("±100 square wave, √5 ≈ 2.236 Hz (hard edges)", 0, (136, 236)),
    "random-walk": ("Gaussian random walk, σ = 1 per step", 0, None),
    "chirp": ("Linear chirp √2 → 32π Hz (≈1.41 → 100.5) repeating every second, amplitude 100", 0, (850, 1000)),
    "noisy-sine": ("π² ≈ 9.87 Hz sine, amplitude 100, + Gaussian noise σ = 5", 0, None),
    "sensor-0.1": ("Slow drift + √2 Hz wobble, rounded to 0.1 (decimal floats)", 0, None),
}

REPORT_MINUTES = 5  # minutes per kind in the report tables: the first 5 of load()'s seeds


def units():
    """The report's units: (kind, seed, X[60, 1000]) for REPORT_MINUTES minutes of each kind."""
    return [(k, 1000 * i + j, minute(k, 1000 * i + j).reshape(PER_CHUNK, BLOCK))
            for i, k in enumerate(KINDS) for j in range(REPORT_MINUTES)]


def peaks(kind, seed):
    """Sample indices (within the minute) of the spike centers / impulses, or None."""
    rng = np.random.default_rng(seed)
    rng.uniform(0, 1e6)  # start time
    rng.uniform(0, 2 * np.pi)  # phase
    if kind == "gauss-spikes":
        rng.uniform(-10, 10, MINUTE)
        return np.sort(np.rint(rng.uniform(0, MINUTE, 180)).astype(int))
    if kind == "impulses":
        rng.uniform(-10, 10, MINUTE)
        return np.unique(rng.integers(0, MINUTE, 180))
    return None


MINUTES_PER_KIND = 10


def load(minutes_per_kind=MINUTES_PER_KIND):
    """The 7,200-block continuous set: (kind, X[60, 1000], block minima, block maxima) per minute."""
    mins = []
    for i, k in enumerate(KINDS):
        for j in range(minutes_per_kind):
            X = minute(k, 1000 * i + j).reshape(60, 1000)
            mins.append((k, X, X.min(1), X.max(1)))
    return mins


# ---------------------------------------------------------------------------
# Discretized signals: values on a fixed quantum
# ---------------------------------------------------------------------------

# (name, source kind, quantum): signal rounded to a multiple of the quantum
DISCRETE = [
    ("random-walk q0.001", "random-walk", 0.001),
    ("random-walk q0.01", "random-walk", 0.01),
    ("random-walk q0.1", "random-walk", 0.1),
    ("random-walk q2^-4", "random-walk", 2.0 ** -4),
    ("noisy-sine q0.1", "noisy-sine", 0.1),
    ("noisy-sine adc12", "noisy-sine", 400 / 4096),  # 12-bit ADC over ±200 engineering units
    ("sin-4.12hz q0.01", "sin-4.12hz", 0.01),
    ("sensor-0.1", "sensor-0.1", 0.1),
    ("noisy-sine q0.01 via float32", "noisy-sine", 0.01),  # decimal values stored as float32 upstream
]
DISCRETE_MINUTES = 10


def load_discrete():
    out = []
    for i, (name, kind, qt) in enumerate(DISCRETE):
        for j in range(DISCRETE_MINUTES):
            x = minute(kind, 5000 + 100 * i + j)
            den = round(1 / qt)
            if abs(den * qt - 1) < 1e-12 and den % 10 == 0:  # decimal quantum: K / 10^p, as parsing "12.345" gives
                X = (np.round(x * den) / den).reshape(60, 1000)
            else:  # binary / ADC quantum: exact multiples
                X = (np.round(x / qt) * qt).reshape(60, 1000)
            if "float32" in name:
                X = X.astype(np.float32).astype(np.float64)
            out.append((name, X, X.min(1), X.max(1), qt))
    return out


# ---------------------------------------------------------------------------
# Noisy signals with a known clean part
# ---------------------------------------------------------------------------

UNIFORM_SIGMA = 20 / np.sqrt(12)  # uniform(-10, 10) base noise


def clean_minute(kind, seed):
    """The noise-free part of minute(kind, seed), same random draws. Returns (clean, sigma)."""
    rng = np.random.default_rng(seed)
    t = np.arange(MINUTE, dtype=np.float64) + rng.uniform(0, 1e6)
    ph = rng.uniform(0, 2 * np.pi)
    if kind == "noisy-sine":
        return 100 * np.sin(2 * np.pi * np.pi ** 2 * t / 1000 + ph), 5.0
    if kind == "gauss-spikes":
        rng.uniform(-10, 10, MINUTE)
        x = np.zeros(MINUTE)
        for c in rng.uniform(0, MINUTE, 180):
            x += 1000 * np.exp(-((np.arange(MINUTE) - c) ** 2) / 200)
        return x, UNIFORM_SIGMA
    if kind == "impulses":
        rng.uniform(-10, 10, MINUTE)
        x = np.zeros(MINUTE)
        x[rng.integers(0, MINUTE, 180)] += 500
        return x, UNIFORM_SIGMA
    raise ValueError(kind)


def noisy_sets():
    """(name, minutes as (X, lo, hi, clean, sigma)) for every signal with known clean part."""
    out = {}
    for i, kind in enumerate(["gauss-spikes", "impulses", "noisy-sine"]):
        k = KINDS.index(kind)
        for j in range(10):  # load() seeds: 1000 * kind index + j
            X = minute(kind, 1000 * k + j).reshape(60, 1000)
            c, s = clean_minute(kind, 1000 * k + j)
            out.setdefault(kind, []).append((X, X.min(1), X.max(1), c.reshape(60, 1000), s))
    for i, (name, kind, qt) in enumerate(DISCRETE):
        if kind != "noisy-sine":
            continue
        for j in range(DISCRETE_MINUTES):
            seed = 5000 + 100 * i + j
            x = minute(kind, seed)
            den = round(1 / qt)
            if abs(den * qt - 1) < 1e-12 and den % 10 == 0:
                X = np.round(x * den) / den
            else:
                X = np.round(x / qt) * qt
            if "float32" in name:
                X = X.astype(np.float32).astype(np.float64)
            X = X.reshape(60, 1000)
            c, s = clean_minute(kind, seed)
            out.setdefault(name, []).append((X, X.min(1), X.max(1), c.reshape(60, 1000), s))
    return out
