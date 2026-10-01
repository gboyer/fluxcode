# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Golden bytes: SHA-256 of the unit and its index columns for fixed (signal, Params) cases. The
other tests check self-consistency (bounds, fixed point, update); this one catches any change to
what the encoder chooses (noise gate, order pick, decimal grid, target) or to the layout.

When the encoder or format is meant to change, regenerate and commit the new hashes:

    uv run python tests/test_golden.py --update
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from _signals import discrete_minute, minute

import fluxcode
from fluxcode import Params

GOLDEN = Path(__file__).with_name("golden.json")


def _gaps(x):
    y = x.copy()
    y[np.random.default_rng(0).integers(0, len(y), 300)] = np.nan  # scattered, most blocks flagged
    y[5_000:7_500] = np.inf
    y[20_000:21_000] = -np.inf  # a whole block
    return y


def _decimal_walk_f32():
    """A random walk on a 0.01 grid, stored as float32 upstream: decodes as the decimals."""
    k = np.cumsum(np.random.default_rng(2).integers(-300, 301, 60_000)) + 12_345
    return (k / 100).astype(np.float32).astype(np.float64)


def _scaled(x, s):
    return x / np.abs(x).max() * s


VARIABLE_SIZES = [1000, 0, 3, 8, 9, 500, 1001, 0, 0, 2048, 1, 2, 777]


def _encode_fixed(block_len=1000):
    """encode_unit in blocks of block_len."""
    return lambda x, params: fluxcode.encode_unit(x, params, block_len=block_len)


def _encode_variable(x, params):
    """Blocks of every kind of size: empty, short (no analysis), odd, and over 1000."""
    return fluxcode.encode_blocks(x[:sum(VARIABLE_SIZES)], VARIABLE_SIZES, params)


def _encode_time_blocks(x, params):
    """One-second blocks of jittered 1 kHz times (µs) with a missing stretch and a gap mid-block."""
    rng = np.random.default_rng(7)
    ticks = 1_790_000_000_000_000 + np.arange(x.size) * 1000 + rng.integers(-20, 21, x.size)
    keep = (np.arange(x.size) < 12_000) | (np.arange(x.size) >= 19_500)
    keep &= (np.arange(x.size) < 40_200) | (np.arange(x.size) >= 40_350)
    return fluxcode.encode_time_blocks(x[keep], ticks[keep].view("datetime64[us]"), params,
                                       start_time=np.datetime64(1_789_999_999_999_000, "us"),
                                       block_duration=np.timedelta64(1, "s"))


CASES = {
    "linear default": (lambda: minute("linear", 1), Params(), _encode_fixed()),
    "quadratic default": (lambda: minute("quadratic", 1), Params(), _encode_fixed()),
    "sin-50.3hz default": (lambda: minute("sin-50.3hz", 1), Params(), _encode_fixed()),
    "chirp default": (lambda: minute("chirp", 1), Params(), _encode_fixed()),
    "noisy-sine default": (lambda: minute("noisy-sine", 1), Params(), _encode_fixed()),
    "gauss-spikes default": (lambda: minute("gauss-spikes", 1), Params(), _encode_fixed()),
    "random-walk default": (lambda: minute("random-walk", 1), Params(), _encode_fixed()),
    "noisy-sine noise off": (lambda: minute("noisy-sine", 2), Params(noise_floor_sigma=None), _encode_fixed()),
    "chirp target 6": (lambda: minute("chirp", 2), Params(target_bits_per_sample=6.0), _encode_fixed()),
    "random-walk target 8": (lambda: minute("random-walk", 2), Params(target_bits_per_sample=8.0), _encode_fixed()),
    "square orders {2}": (lambda: minute("square-2.24hz", 1), Params(diff_orders={2}), _encode_fixed()),
    "sin-9.87hz 9 bits": (lambda: minute("sin-9.87hz", 1), Params(min_quantize_bits=4, max_quantize_bits=9), _encode_fixed()),
    "random-walk q0.01": (lambda: discrete_minute("random-walk q0.01", 1), Params(), _encode_fixed()),
    "sensor-0.1": (lambda: discrete_minute("sensor-0.1", 1), Params(), _encode_fixed()),
    "0.01-grid walk via float32": (lambda: _decimal_walk_f32(), Params(), _encode_fixed()),
    "noisy-sine adc12 decimal off": (lambda: discrete_minute("noisy-sine adc12", 1), Params(decimal_detection=False), _encode_fixed()),
    "noisy-sine with gaps": (lambda: _gaps(minute("noisy-sine", 3)), Params(), _encode_fixed()),
    "random-walk with gaps target 6": (lambda: _gaps(minute("random-walk", 3)), Params(target_bits_per_sample=6.0), _encode_fixed()),
    "white noise at 1e-310": (lambda: _scaled(np.random.default_rng(1).normal(size=60_000), 1e-310), Params(), _encode_fixed()),
    "white noise across +-DBL_MAX": (lambda: _scaled(np.random.default_rng(1).normal(size=60_000), 1.7e308),
                                     Params(), _encode_fixed()),
    "short last block": (lambda: minute("sin-4.12hz", 4)[:59_321], Params(), _encode_fixed()),
    "block_len 64, 7 blocks": (lambda: minute("random-walk", 5)[:7 * 64], Params(), _encode_fixed(64)),
    "block_len 1001": (lambda: minute("noisy-sine", 5), Params(), _encode_fixed(1001)),
    "variable blocks": (lambda: minute("random-walk", 6), Params(), _encode_variable),
    "variable blocks q0.01": (lambda: discrete_minute("random-walk q0.01", 6), Params(), _encode_variable),
    "variable blocks target 6": (lambda: minute("chirp", 6), Params(target_bits_per_sample=6.0), _encode_variable),
    "time blocks": (lambda: discrete_minute("sensor-0.1", 7), Params(),
                    _encode_time_blocks),
}


def digest(name):
    make, params, encoder = CASES[name]
    unit, lo, hi, mean = encoder(make(), params)
    h = hashlib.sha256(unit)
    for col in (lo, hi, mean):
        h.update(np.ascontiguousarray(col, np.float64).tobytes())
    return h.hexdigest()


@pytest.mark.parametrize("name", CASES)
def test_golden(name):
    assert digest(name) == json.loads(GOLDEN.read_text())[name], \
        "encoder output changed; if intended, run: uv run python tests/test_golden.py --update"


if __name__ == "__main__" and sys.argv[1:] == ["--update"]:
    GOLDEN.write_text(json.dumps({name: digest(name) for name in CASES}, indent=1) + "\n")
    print(f"wrote {len(CASES)} hashes to {GOLDEN}")
