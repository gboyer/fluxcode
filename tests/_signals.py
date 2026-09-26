# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Test and benchmark signals: one continuous minute (60 blocks of 1000 samples) per kind,
the generators of the experiments' bench_chunks.minute, plus discretized variants."""

import numpy as np

MINUTE = 60_000

KINDS = ["linear", "quadratic", "sin-4.12hz", "sin-9.87hz", "sin-50.3hz", "gauss-spikes", "impulses",
         "square-2.24hz", "random-walk", "chirp", "noisy-sine"]
NOISY = ["gauss-spikes", "impulses", "noisy-sine"]
CLEAN = [k for k in KINDS if k not in NOISY]


def minute(kind, seed):
    """One continuous minute (60,000 samples at 1 kHz) of a signal kind."""
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
    if kind == "chirp":  # repeating 1-second sweep sqrt(2) -> 32 pi Hz, phase continuous
        frac = (np.arange(MINUTE) % 1000) / 1000
        f = np.sqrt(2) + (32 * np.pi - np.sqrt(2)) * frac
        return 100 * np.sin(np.cumsum(2 * np.pi * f / 1000) + ph)
    if kind == "noisy-sine":
        return 100 * np.sin(2 * np.pi * np.pi ** 2 * t / 1000 + ph) + rng.normal(0, 5, MINUTE)
    if kind == "sensor-0.1":  # slow drift rounded to 0.1
        return np.round(50 + 5 * np.sin(2 * np.pi * np.sqrt(2) * t / 1000 + ph)
                        + np.cumsum(rng.normal(0, 0.05, MINUTE)), 1)
    raise ValueError(kind)


# (name, source kind, quantum): the signal rounded to a multiple of the quantum
DISCRETE = [
    ("random-walk q0.001", "random-walk", 0.001),
    ("random-walk q0.01", "random-walk", 0.01),
    ("random-walk q0.1", "random-walk", 0.1),
    ("random-walk q2^-4", "random-walk", 2.0 ** -4),
    ("noisy-sine q0.1", "noisy-sine", 0.1),
    ("noisy-sine adc12", "noisy-sine", 400 / 4096),  # 12-bit ADC over +-200 engineering units
    ("sin-4.12hz q0.01", "sin-4.12hz", 0.01),
    ("sensor-0.1", "sensor-0.1", 0.1),
    ("noisy-sine q0.01 via float32", "noisy-sine", 0.01),  # decimal values stored as float32 upstream
]


def discrete_minute(name, seed):
    kind, qt = next((k, q) for n, k, q in DISCRETE if n == name)
    x = minute(kind, seed)
    den = round(1 / qt)
    if abs(den * qt - 1) < 1e-12 and den % 10 == 0:  # decimal quantum: K / 10^p, as parsing "12.345" gives
        x = np.round(x * den) / den
    else:  # binary / ADC quantum: exact multiples
        x = np.round(x / qt) * qt
    if "float32" in name:
        x = x.astype(np.float32).astype(np.float64)
    return x
