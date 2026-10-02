# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Smoke test for the research code that reaches into fluxcode's private modules (_format, _unit,
_encoder, _bitpacking): run each such path on a tiny input, so a rename or layout change in the
package fails CI here instead of when the report is next regenerated.

    uv run --project experimental python experimental/smoke.py
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "tests"), str(HERE.parent / "bench")]

import fluxcode  # noqa: E402
import make_time_report  # noqa: E402
from tslab.flux.adapters import FluxCodec, ideal_quantum  # noqa: E402


def main():
    rng = np.random.default_rng(0)
    X = np.cumsum(rng.normal(size=(4, 1000)), axis=1)
    X[1] = np.round(X[1], 2)  # one decimal block
    unit, infos = FluxCodec(16).encode_unit(X)
    assert len(infos) == 4 and all(0 <= i["order"] <= 3 for i in infos), infos
    assert infos[1]["decimal"] and infos[1]["param"] == -2, infos[1]
    assert all(-1074 <= i["param"] <= 1023 for i in infos), infos
    assert np.abs(fluxcode.decode_unit(unit).values.reshape(4, 1000) - X).max() < 1e-3

    qt = 0.01
    Xq = np.round(X / qt) * qt
    bits = ideal_quantum([(None, Xq, Xq.min(axis=1), Xq.max(axis=1), qt)])
    assert 0 < bits < 16, bits

    values = X.ravel()
    ticks = np.arange(values.size, dtype=np.int64) * 1_000_000
    ticks[2500:] += 7_000_000  # one gap: one irregular block
    plain = fluxcode.encode_unit(values).unit
    row = make_time_report.measure(values, ticks, plain, 0, (0.0, 0.0))
    assert row["irregular"].tolist() == [False, False, True, False], row
    print("experimental smoke test passed")


if __name__ == "__main__":
    main()
