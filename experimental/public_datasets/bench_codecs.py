# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Benchmark fluxcode and competing codecs on a curated real-world public dataset suite.

Writes experimental/public_datasets/codec_results.json for make_public_report.py to render
the real-world size-vs-error scatterplots (public-scatter.svg and public-rd.svg).

    FLUXCODE_DATA_DIR=... cd experimental && uv run python public_datasets/bench_codecs.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = EXP.parent
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(REPO / "bench"))

import _fresh_numba_cache  # noqa: E402, F401
import numpy as np  # noqa: E402
import zstandard  # noqa: E402
from fluxcode import Params  # noqa: E402
from public_datasets import analyze  # noqa: E402

from tslab.classic import (  # noqa: E402
    CompandedDpcm,
    DctTopK,
    Gorilla,
    Quant8,
    QuantDeltaDeflate,
    SwingingDoor,
    cubic_expander,
)
from tslab.common.group import Concat, SharedBackend  # noqa: E402
from tslab.constwidth import Delta1Bfp  # noqa: E402
from tslab.entropy.delta_zstd import DeltaZstd  # noqa: E402
from tslab.flux.adapters import FluxCodec  # noqa: E402
from tslab.report.order import family  # noqa: E402

BLOCK = 1000
MAX_BLOCKS = 60  # up to 60,000 samples (one standard block group) per series
OUT_JSON = HERE / "codec_results.json"

# 9 diverse real-world series spanning vibration, ECG, clinical PPG, 24-bit seismic,
# industrial PLC, smart-meter power, rational-grid weather, float32 weather, and tick trades.
CURATED_SERIES: list[tuple[str, str, str]] = [
    ("CWRU Vibration", "normal_48k_DE", "CWRU Vibration (48 kHz)"),
    ("MIT-BIH ECG", "rec100_MLII", "MIT-BIH ECG (360 Hz)"),
    ("BIDMC Clinical", "PLETH", "BIDMC PPG (125 Hz)"),
    ("IRIS Seismic", "ANMO_BHZ_mps", "IRIS Seismic (40 Hz)"),
    ("UCI Hydraulic", "PS1_100Hz", "UCI Hydraulic (100 Hz)"),
    ("UCI Power", "Global_active_power", "UCI Power (1 min)"),
    ("UCI Appliances", "Press_mm_hg", "UCI Appliances (1/3 grid)"),
    ("ERA5 Weather", "temp_2m_f32", "ERA5 Weather (f32)"),
    ("Binance Trades", "BTCUSDT_price", "Binance BTC ($0.01)"),
]


class _RawZstd:
    name = "raw-zstd"
    label = "raw float64 bytes + zstd level 3"

    def encode_group(self, X: np.ndarray) -> tuple[bytes, list[dict[str, Any]]]:
        data = zstandard.ZstdCompressor(level=3).compress(np.ascontiguousarray(X, np.float64).tobytes())
        return data, [{} for _ in range(len(X))]

    def decode_group(self, data: bytes, nb: int, n: int) -> np.ndarray:
        raw = zstandard.ZstdDecompressor().decompress(data)
        return np.frombuffer(raw, np.float64).reshape(nb, n)


class _BshufZstd:
    name = "bshuf-zstd"
    label = "byte-shuffled float64 + zstd level 3"

    def encode_group(self, X: np.ndarray) -> tuple[bytes, list[dict[str, Any]]]:
        arr = np.ascontiguousarray(X, np.float64).view(np.uint8).reshape(-1, 8).T
        data = zstandard.ZstdCompressor(level=3).compress(np.ascontiguousarray(arr).tobytes())
        return data, [{} for _ in range(len(X))]

    def decode_group(self, data: bytes, nb: int, n: int) -> np.ndarray:
        raw = zstandard.ZstdDecompressor().decompress(data)
        arr = np.frombuffer(raw, np.uint8).reshape(8, nb * n).T
        return np.ascontiguousarray(arr).view(np.float64).reshape(nb, n)


def build_codecs() -> list[tuple[str, Any]]:
    """Representative competing codecs across classic, constant-width, entropy, lossless, and fluxcode."""
    return [
        ("lossless", _RawZstd()),
        ("lossless", _BshufZstd()),
        ("classic", Concat(Gorilla())),
        ("classic", Concat(Quant8())),
        ("classic", SharedBackend(QuantDeltaDeflate(8))),
        ("classic", SharedBackend(CompandedDpcm("dpcm4-linear-deflate", "4-bit linear DPCM + deflate", cubic_expander(0.0), 1.0, 4, "byte"))),
        ("classic", Concat(SwingingDoor(0.005))),
        ("classic", Concat(SwingingDoor(0.02))),
        ("classic", Concat(DctTopK(64))),
        ("constwidth", Concat(Delta1Bfp(4, 4))),
        ("constwidth", Concat(Delta1Bfp(8, 4))),
        ("entropy", DeltaZstd(8)),
        ("entropy", DeltaZstd(12)),
        ("flux", FluxCodec(16, 0.25, params=Params())),
        ("flux", FluxCodec(16, None, params=Params(noise_floor_sigma=0, grids="binary"))),
    ]


CURVE_BITS = (6, 8, 10, 12, 14, 16)


def eval_codec(codec: Any, X: np.ndarray) -> dict[str, float]:
    data, _ = codec.encode_group(X)
    Y = codec.decode_group(data, *X.shape)
    assert Y.shape == X.shape
    rmses = []
    maxerrs = []
    for x, y in zip(X, Y):
        fs = float(x.max() - x.min()) or 1.0
        rmses.append(float(np.sqrt(np.mean((y - x) ** 2)) / fs))
        maxerrs.append(float(np.abs(y - x).max() / fs))
    return {
        "bps": 8.0 * len(data) / X.size,
        "rmse_med_pct": 100.0 * float(np.median(rmses)),
        "rmse_worst_pct": 100.0 * float(np.max(rmses)),
        "maxerr_worst_pct": 100.0 * float(np.max(maxerrs)),
    }


def load_curated() -> list[tuple[str, str, str, np.ndarray]]:
    # Reuse analyze.py's loaders
    loaders = [
        analyze.load_cwru,
        analyze.load_mitdb,
        analyze.load_bidmc,
        analyze.load_iris_seismic,
        analyze.load_uci_hydraulic,
        analyze.load_uci_household_power,
        analyze.load_uci_appliances,
        analyze.load_era5,
        analyze.load_binance_trades,
    ]
    by_key: dict[tuple[str, str], np.ndarray] = {}
    for fn in loaders:
        for s in fn():
            by_key[(s.group, s.name)] = s.values

    out = []
    for group, name, label in CURATED_SERIES:
        vals = by_key.get((group, name))
        if vals is None:
            raise FileNotFoundError(f"Missing curated series {group}/{name} in {analyze.DATA_DIR}")
        finite = vals[np.isfinite(vals)]
        nb = min(MAX_BLOCKS, len(finite) // BLOCK)
        assert nb >= 8, f"Too few finite blocks ({nb}) for {group}/{name}"
        X = np.ascontiguousarray(finite[: nb * BLOCK].reshape(nb, BLOCK), dtype=np.float64)
        out.append((group, name, label, X))
    return out


def main() -> None:
    curated = load_curated()
    codecs = build_codecs()
    curve_codecs = [
        (b, FluxCodec(b, params=Params(max_quantize_bits=b, min_quantize_bits=min(b, 6), noise_floor_sigma=0, target_bits_per_sample=None)))
        for b in CURVE_BITS
    ]

    series_out: list[dict[str, Any]] = []
    for group, name, label, X in curated:
        print(f"Benchmarking {label} ({X.shape[0]} blocks x {X.shape[1]})...", flush=True)
        codec_stats: dict[str, dict[str, float]] = {}
        for fam, c in codecs:
            r = eval_codec(c, X)
            r["family"] = fam if c.name in ("raw-zstd", "bshuf-zstd") else family(c)
            cname = "fluxcode-bin-nf0" if c.name == "fluxcode-16" else ("fluxcode-default" if c.name == "fluxcode-16-f0.25" else c.name)
            codec_stats[cname] = r
        curve_stats = []
        for b, c in curve_codecs:
            r = eval_codec(c, X)
            r["B"] = b
            curve_stats.append(r)
        series_out.append({
            "group": group,
            "series": name,
            "label": label,
            "blocks": int(X.shape[0]),
            "samples": int(X.size),
            "codecs": codec_stats,
            "fluxcode_curve": curve_stats,
        })

    payload = {
        "block_len": BLOCK,
        "curve_bits": list(CURVE_BITS),
        "series": series_out,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
