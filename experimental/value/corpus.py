# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""A corpus for value-compression experiments, closer to industrial data than tests/_signals.py.

Three sets, each a list of (name, samples, block_len):

- FAST: high-rate waveforms as a DAQ or condition-monitoring system records them, with the ADC's
  grid (counts times a gain, sometimes stored as float32), harmonics, drift and noise at realistic
  levels. 240,000 samples each (4 block groups of 60 blocks of 1000).
- SLOW: the fluxprune corpus (tslab.prune.signals): a day of 1 s scans per tag, mostly decimal
  resolutions, in hourly blocks (24 blocks of 3600 per block group).
- SYNTH: the benchmark's 15 signals (tests/_signals.py), including the pathological ones (chirp,
  pure sines): kept so nothing regresses badly there, but they needn't compress as well.

All seeds are fixed.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tests"), str(ROOT / "experimental")]

N_FAST = 240_000


def _f32(x):
    return x.astype(np.float32).astype(np.float64)


def _adc(x, full_scale, bits):
    """Counts of a bipolar ADC of `bits` over ±full_scale, times its gain (the doubles a driver returns)."""
    gain = full_scale / 2 ** (bits - 1)
    return np.clip(np.round(x / gain), -2 ** (bits - 1), 2 ** (bits - 1) - 1) * gain


def _ar1(rng, n, rho, sigma):
    from scipy.signal import lfilter

    return lfilter([1.0], [1.0, -rho], rng.normal(0, sigma * np.sqrt(1 - rho * rho), n))


def fast(name, seed=0):
    rng = np.random.default_rng(1000 + seed + 17 * FAST_NAMES.index(name))
    n = N_FAST
    if name == "accel-gearbox":  # 10 kHz accelerometer, g, IEPE + 24-bit ADC, stored as float32
        fs = 10_000.0
        t = np.arange(n) / fs
        shaft = 24.7 * (1 + 0.002 * np.sin(2 * np.pi * t / 7))  # slow speed wander
        ph = 2 * np.pi * np.cumsum(shaft) / fs
        x = 0.5 * np.sin(ph) + 0.2 * np.sin(2 * ph + 1) + 0.08 * np.sin(3 * ph + 2)
        mesh = 0.3 * (1 + 0.3 * np.sin(ph)) * np.sin(37 * ph)  # gear mesh with shaft sidebands
        x += mesh
        # bearing outer-race impacts at 3.57 x shaft ringing a 2.8 kHz resonance
        from scipy.signal import lfilter

        imp = np.zeros(n)
        k = 0.0
        while k < n:
            imp[int(k)] = rng.uniform(0.5, 1.5)
            k += fs / (3.57 * 24.7)
        r, w = np.exp(-2 * np.pi * 150 / fs), 2 * np.pi * 2800 / fs
        x += 0.4 * lfilter([1.0], [1, -2 * r * np.cos(w), r * r], imp)
        x += rng.normal(0, 0.03, n)
        return _f32(x)
    if name == "motor-current":  # 5 kHz phase current, A, 16-bit ADC over ±50 A
        fs = 5000.0
        t = np.arange(n) / fs
        f = 60 + 0.01 * np.cumsum(rng.normal(0, 0.002, n))
        ph = 2 * np.pi * np.cumsum(f) / fs
        load = 14 + 4 * np.repeat(rng.uniform(-1, 1, n // 20000 + 1), 20000)[:n]
        from scipy.signal import lfilter

        load = lfilter([0.0005], [1, -0.9995], load, zi=[load[0] * 0.9995])[0]
        x = load * (np.sin(ph) + 0.03 * np.sin(5 * ph + 0.3) + 0.02 * np.sin(7 * ph + 1.1))
        x += rng.normal(0, 0.05, n)
        return _adc(x, 50.0, 16)
    if name == "line-voltage":  # 7680 Hz (128/cycle), V, 16-bit ADC over ±500 V
        fs = 7680.0
        t = np.arange(n) / fs
        ph = 2 * np.pi * 60.003 * t
        x = 169.7 * (1 + 0.003 * np.sin(2 * np.pi * t / 3)) * (np.sin(ph) + 0.015 * np.sin(3 * ph) + 0.01 * np.sin(5 * ph))
        x += rng.normal(0, 0.03, n)
        return _adc(x, 500.0, 16)
    if name == "pressure-pump":  # 1 kHz, bar, 4-20 mA on a 12-bit ADC over 0-10 bar
        t = np.arange(n) / 1000.0
        x = 5 + _ar1(rng, n, 1 - 1 / 10_000, 0.15)
        x += 0.04 * np.sin(2 * np.pi * 18.3 * t) + 0.015 * np.sin(2 * np.pi * 36.6 * t + 0.5)
        x += rng.normal(0, 0.004, n)
        return np.round(x / (10 / 4095)) * (10 / 4095)
    if name == "loadcell":  # 1 kHz, kg, 24-bit bridge ADC, float32; loads every 5-20 s, ringing at 12 Hz
        from scipy.signal import lfilter

        target = np.zeros(n)
        k = 0
        level = 0.0
        while k < n:
            target[k:] = level
            k += int(rng.uniform(5000, 20000))
            level = 0.0 if level else rng.uniform(50, 400)
        fs, f0, zeta = 1000.0, 12.0, 0.08
        w0 = 2 * np.pi * f0 / fs
        r = np.exp(-zeta * w0)
        a1, a2 = -2 * r * np.cos(w0 * np.sqrt(1 - zeta ** 2)), r * r
        x = lfilter([1 + a1 + a2], [1, a1, a2], target)
        x += rng.normal(0, 0.02, n)
        return _f32(x)
    if name == "servo-position":  # 1 kHz, mm, encoder 0.001 mm: trapezoidal moves and dwells
        x = np.empty(n)
        k, pos = 0, 0.0
        while k < n:
            dwell = int(rng.uniform(200, 3000))
            x[k:k + dwell] = pos
            k += dwell
            dist = rng.uniform(-200, 200)
            vmax, acc = rng.uniform(100, 500) / 1000, 2.0 / 1000  # mm/ms, mm/ms^2
            ta = vmax / acc
            T = abs(dist) / vmax + ta
            tt = np.arange(int(T) + 1, dtype=float)
            s = np.where(tt < ta, 0.5 * acc * tt ** 2,
                         np.where(tt < T - ta, 0.5 * acc * ta ** 2 + vmax * (tt - ta),
                                  abs(dist) - 0.5 * acc * np.maximum(T - tt, 0) ** 2))
            s = np.minimum(s, abs(dist))
            seg = pos + np.sign(dist) * s
            m = min(len(seg), n - k)
            x[k:k + m] = seg[:m]
            k += m
            pos += dist
        x += rng.normal(0, 0.002, n)  # servo hunting
        return np.round(x, 3)
    if name == "flow-turbulent":  # 1 kHz, m3/h: mean flow with turbulent colored noise, float32
        x = 120 + _ar1(rng, n, 1 - 1 / 30_000, 3) + _ar1(rng, n, 0.95, 1.5) + rng.normal(0, 0.3, n)
        return _f32(x)
    if name == "tach-rpm":  # 1 kHz speed from an encoder count, rpm, 0.25 rpm resolution
        x = 1500 + _ar1(rng, n, 1 - 1 / 20_000, 20) + 3 * np.sin(2 * np.pi * 25 * np.arange(n) / 1000)
        return np.round((x + rng.normal(0, 0.4, n)) * 4) / 4
    if name == "thermocouple-10hz":  # 10 Hz, °C, float32 from a 24-bit converter, 50 Hz pickup aliased
        t = np.arange(n) / 10.0
        x = 350 + 20 * np.sin(2 * np.pi * t / 7200) + _ar1(rng, n, 1 - 1 / 600, 1.5)
        x += 0.05 * np.sin(2 * np.pi * 0.37 * t) + rng.normal(0, 0.015, n)
        return _f32(x)
    raise ValueError(name)


FAST_NAMES = ["accel-gearbox", "motor-current", "line-voltage", "pressure-pump", "loadcell", "servo-position",
              "flow-turbulent", "tach-rpm", "thermocouple-10hz"]


def slow_names():
    from tslab.prune import signals

    return list(signals.SIGNALS) + list(signals.OFFSET) + list(signals.NOISY)


def slow(name, day=0):
    from tslab.prune import signals

    return np.asarray(signals.scans(name, day), dtype=np.float64)


def synth_names():
    import _signals

    return list(_signals.KINDS) + [d[0] for d in _signals.DISCRETE if d[0] in (
        "sensor-0.1", "random-walk q0.01", "noisy-sine q0.1", "noisy-sine q0.01 via float32",
        "random-walk q0.05", "random-walk q1/3", "random-walk adc 10/4095")]


def synth(name, seed=0, minutes=4):
    import _signals

    parts = []
    for m in range(minutes):
        if name in _signals.KINDS:
            parts.append(_signals.minute(name, seed + m))
        else:
            parts.append(_signals.discrete_minute(name, seed + m))
    return np.concatenate(parts)


def corpus(sets=("fast", "slow", "synth")):
    """[(set, name, samples, block_len, blocks_per_group)]."""
    out = []
    if "fast" in sets:
        out += [("fast", n, fast(n), 1000, 60) for n in FAST_NAMES]
    if "slow" in sets:
        out += [("slow", n, slow(n), 3600, 24) for n in slow_names()]
    if "synth" in sets:
        out += [("synth", n, synth(n), 1000, 60) for n in synth_names()]
    return out
