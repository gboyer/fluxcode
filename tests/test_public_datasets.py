# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Opt-in verification tests against downloaded real-world public datasets.

Skips cleanly when the dataset directory (FLUXCODE_DATA_DIR or
experimental/public_datasets/data/) is absent or empty, so default offline
``uv run pytest`` never fails or touches the network.

Download datasets once with ``make datasets``, then run ``make test-public``.
"""

from __future__ import annotations

import csv
import os
import struct
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

import fluxcode
from fluxcode import Params, verify

pytestmark = pytest.mark.public_data

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "experimental" / "public_datasets" / "data"
MAX_SAMPLES = 60_000  # 1 full block group (60 x 1000) keeps runtime brisk


def _data_dir() -> Path:
    env = os.environ.get("FLUXCODE_DATA_DIR")
    return Path(env) if env else DEFAULT_DATA_DIR


DATA_DIR = _data_dir()
if not DATA_DIR.exists() or not any(DATA_DIR.iterdir()):
    pytest.skip(
        f"public datasets not present in {DATA_DIR}; run 'make datasets' to enable",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Minimal, self-contained loaders for representative series (all 14 families)
# ---------------------------------------------------------------------------


def _sext(val: int, bits: int) -> int:
    if val & (1 << (bits - 1)):
        val -= 1 << bits
    return val


def _decode_mitdb_212(dat_path: Path, max_samples: int = MAX_SAMPLES) -> np.ndarray:
    n_bytes = min(dat_path.stat().st_size, max_samples * 3)
    raw = np.fromfile(dat_path, dtype=np.uint8, count=n_bytes)
    triplets = raw.reshape(-1, 3).astype(np.int32)
    b0, b1 = triplets[:, 0], triplets[:, 1]
    s0 = b0 | ((b1 & 0x0F) << 8)
    s0 = np.where(s0 >= 2048, s0 - 4096, s0)
    return np.ascontiguousarray((s0 - 1024).astype(np.float64) / 200.0)


def _decode_mseed_steim2(path: Path, max_samples: int = MAX_SAMPLES) -> np.ndarray:
    buf = path.read_bytes()
    out: list[int] = []
    pos = 0
    while pos + 48 <= len(buf) and len(out) < max_samples:
        nsamp = struct.unpack(">H", buf[pos + 30 : pos + 32])[0]
        data_off, blk_off = struct.unpack(">HH", buf[pos + 44 : pos + 48])
        btype, _bnext, enc, _bo, reclen_exp = struct.unpack(
            ">HHBBB", buf[pos + blk_off : pos + blk_off + 7]
        )
        reclen = 1 << reclen_exp
        if btype != 1000 or enc != 11:
            raise ValueError(f"Unsupported MiniSEED blockette={btype} encoding={enc}")
        rec = buf[pos : pos + reclen]
        pos += reclen
        x0, xn = struct.unpack(">ii", rec[data_off + 4 : data_off + 12])
        cur = x0
        rec_vals: list[int] = []
        nframes = (reclen - data_off) // 64
        for fidx in range(nframes):
            f = rec[data_off + fidx * 64 : data_off + (fidx + 1) * 64]
            ctrl = struct.unpack(">I", f[0:4])[0]
            start_w = 3 if fidx == 0 else 1
            for w in range(start_w, 16):
                if len(rec_vals) >= nsamp:
                    break
                ck = (ctrl >> (30 - 2 * w)) & 0x3
                word = struct.unpack(">I", f[w * 4 : (w + 1) * 4])[0]
                if ck == 0:
                    continue
                if ck == 1:
                    diffs = [_sext((word >> (24 - 8 * k)) & 0xFF, 8) for k in range(4)]
                elif ck == 2:
                    dnib = (word >> 30) & 0x3
                    if dnib == 1:
                        diffs = [_sext(word & 0x3FFFFFFF, 30)]
                    elif dnib == 2:
                        diffs = [
                            _sext((word >> (15 - 15 * k)) & 0x7FFF, 15)
                            for k in range(2)
                        ]
                    elif dnib == 3:
                        diffs = [
                            _sext((word >> (20 - 10 * k)) & 0x3FF, 10) for k in range(3)
                        ]
                    else:
                        continue
                else:
                    dnib = (word >> 30) & 0x3
                    if dnib == 0:
                        diffs = [
                            _sext((word >> (24 - 6 * k)) & 0x3F, 6) for k in range(5)
                        ]
                    elif dnib == 1:
                        diffs = [
                            _sext((word >> (25 - 5 * k)) & 0x1F, 5) for k in range(6)
                        ]
                    elif dnib == 2:
                        diffs = [
                            _sext((word >> (24 - 4 * k)) & 0xF, 4) for k in range(7)
                        ]
                    else:
                        continue
                for d in diffs:
                    if not rec_vals:
                        rec_vals.append(x0)
                    else:
                        cur += d
                        rec_vals.append(cur)
                    if len(rec_vals) >= nsamp:
                        break
        assert len(rec_vals) == nsamp and rec_vals[-1] == xn
        out.extend(rec_vals)
    return np.ascontiguousarray(out[:max_samples], dtype=np.float64)


def _load_cwru() -> tuple[np.ndarray, np.ndarray | None]:
    p = DATA_DIR / "cwru" / "1730_Normal.npz"
    if not p.exists():
        pytest.skip(f"missing {p}")
    z = np.load(p)
    return np.ascontiguousarray(z["DE"].ravel()[:MAX_SAMPLES], dtype=np.float64), None


def _load_mitdb() -> tuple[np.ndarray, np.ndarray | None]:
    p = DATA_DIR / "mitdb" / "100.dat"
    if not p.exists():
        pytest.skip(f"missing {p}")
    return _decode_mitdb_212(p, MAX_SAMPLES), None


def _load_iris() -> tuple[np.ndarray, np.ndarray | None]:
    p = DATA_DIR / "iris" / "IU_ANMO_00_BHZ.mseed"
    if not p.exists():
        pytest.skip(f"missing {p}")
    return _decode_mseed_steim2(p, MAX_SAMPLES), None


def _load_bidmc() -> tuple[np.ndarray, np.ndarray | None]:
    p = DATA_DIR / "bidmc" / "bidmc_01_Signals.csv"
    if not p.exists():
        pytest.skip(f"missing {p}")
    vals: list[float] = []
    with open(p, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [k.strip() for k in (reader.fieldnames or [])]
        for row in reader:
            vals.append(float(row["PLETH"].strip()))
            if len(vals) >= MAX_SAMPLES:
                break
    return np.ascontiguousarray(vals, dtype=np.float64), None


def _load_uci_power() -> tuple[np.ndarray, np.ndarray | None]:
    p = DATA_DIR / "uci" / "household_power" / "household_power_consumption.txt"
    if not p.exists():
        pytest.skip(f"missing {p}")
    vals: list[float] = []
    ts_int: list[int] = []
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    with open(p, encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split(";")
            if len(parts) < 9:
                continue
            dd, mm, yy = parts[0].split("/")
            hh, mi, ss = parts[1].split(":")
            dt = datetime(
                int(yy),
                int(mm),
                int(dd),
                int(hh),
                int(mi),
                int(ss),
                tzinfo=timezone.utc,
            )
            ts_int.append(int((dt - epoch).total_seconds()))
            v = parts[2]
            vals.append(np.nan if (v == "?" or v == "") else float(v))
            if len(vals) >= MAX_SAMPLES:
                break
    ts = np.array(ts_int, dtype=np.int64).astype("datetime64[s]")
    return np.ascontiguousarray(vals, dtype=np.float64), ts


def _load_uci_hydraulic() -> tuple[np.ndarray, np.ndarray | None]:
    p = DATA_DIR / "uci" / "hydraulic_systems" / "PS1.txt"
    if not p.exists():
        pytest.skip(f"missing {p}")
    chunks: list[np.ndarray] = []
    total = 0
    with open(p, encoding="utf-8") as f:
        for line in f:
            arr = np.fromstring(line, sep="\t", dtype=np.float64)
            chunks.append(arr)
            total += len(arr)
            if total >= MAX_SAMPLES:
                break
    return np.ascontiguousarray(np.concatenate(chunks)[:MAX_SAMPLES]), None


def _load_skab() -> tuple[np.ndarray, np.ndarray | None]:
    p = DATA_DIR / "skab" / "anomaly-free.csv"
    if not p.exists():
        pytest.skip(f"missing {p}")
    vals: list[float] = []
    ts_list: list[np.datetime64] = []
    with open(p, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter=";"):
            ts_list.append(np.datetime64(row["datetime"].strip(), "s"))
            vals.append(float(row["Pressure"].strip()))
    return np.ascontiguousarray(vals, dtype=np.float64), np.array(
        ts_list, dtype="datetime64[s]"
    )


def _load_uci_air_quality() -> tuple[np.ndarray, np.ndarray | None]:
    p = DATA_DIR / "uci" / "air_quality" / "AirQualityUCI.csv"
    if not p.exists():
        pytest.skip(f"missing {p}")
    vals: list[float] = []
    ts_int: list[int] = []
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    with open(p, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter=";"):
            d_str = (row.get("Date") or "").strip()
            t_str = (row.get("Time") or "").strip()
            if not d_str or not t_str:
                continue
            dd, mm, yy = d_str.split("/")
            hh, mi, ss = t_str.split(".")
            dt = datetime(
                int(yy),
                int(mm),
                int(dd),
                int(hh),
                int(mi),
                int(ss),
                tzinfo=timezone.utc,
            )
            ts_int.append(int((dt - epoch).total_seconds()))
            s = (row.get("T") or "").strip().replace(",", ".")
            vals.append(float(s) if s else np.nan)
    ts = np.array(ts_int, dtype=np.int64).astype("datetime64[s]")
    return np.ascontiguousarray(vals, dtype=np.float64), ts


def _load_uci_appliances() -> tuple[np.ndarray, np.ndarray | None]:
    p = DATA_DIR / "uci" / "appliances_energy" / "energydata_complete.csv"
    if not p.exists():
        pytest.skip(f"missing {p}")
    vals: list[float] = []
    ts_list: list[np.datetime64] = []
    with open(p, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ts_list.append(np.datetime64(row["date"].strip(), "s"))
            vals.append(float(row["T_out"].strip()))
    return np.ascontiguousarray(vals, dtype=np.float64), np.array(
        ts_list, dtype="datetime64[s]"
    )


def _load_noaa_buoy() -> tuple[np.ndarray, np.ndarray | None]:
    p = DATA_DIR / "noaa" / "46026h2023.txt"
    if not p.exists():
        pytest.skip(f"missing {p}")
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    vals: list[float] = []
    ts_int: list[int] = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 15:
                continue
            dt = datetime(
                int(parts[0]),
                int(parts[1]),
                int(parts[2]),
                int(parts[3]),
                int(parts[4]),
                tzinfo=timezone.utc,
            )
            ts_int.append(int((dt - epoch).total_seconds()))
            v = float(parts[14])
            vals.append(np.nan if v == 999.0 else v)
    ts = np.array(ts_int, dtype=np.int64).astype("datetime64[s]")
    return np.ascontiguousarray(vals, dtype=np.float64), ts


def _load_era5_f32() -> tuple[np.ndarray, np.ndarray | None]:
    p = DATA_DIR / "era5" / "sf_2023_hourly.csv"
    if not p.exists():
        pytest.skip(f"missing {p}")
    lines = p.read_text(encoding="utf-8").splitlines()
    hdr_idx = next(i for i, line in enumerate(lines) if line.startswith("time,"))
    vals: list[float] = []
    ts_list: list[np.datetime64] = []
    for row in csv.DictReader(lines[hdr_idx:]):
        ts_list.append(np.datetime64(row["time"].strip(), "s"))
        vals.append(float(row["temperature_2m (°C)"]))
    arr = np.ascontiguousarray(np.array(vals, dtype=np.float32).astype(np.float64))
    return arr, np.array(ts_list, dtype="datetime64[s]")


def _load_binance() -> tuple[np.ndarray, np.ndarray | None]:
    p = (
        DATA_DIR
        / "binance"
        / "BTCUSDT-trades-2024-01-01"
        / "BTCUSDT-trades-2024-01-01.csv"
    )
    if not p.exists():
        pytest.skip(f"missing {p}")
    prices: list[float] = []
    ts_ms: list[int] = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            parts = line.split(",")
            if len(parts) < 5:
                continue
            prices.append(float(parts[1]))
            ts_ms.append(int(parts[4]))
            if len(prices) >= MAX_SAMPLES:
                break
    ts = np.array(ts_ms, dtype=np.int64).astype("datetime64[ms]")
    return np.ascontiguousarray(prices, dtype=np.float64), ts


def _load_smartphone_imu() -> tuple[np.ndarray, np.ndarray | None]:
    p = DATA_DIR / "imu" / "p001_hand_walking.csv"
    if not p.exists():
        pytest.skip(f"missing {p}")
    vals: list[float] = []
    ts_ms: list[int] = []
    with open(p, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ts_ms.append(int(row["elapsedMillis"]))
            vals.append(float(row["accelerometerAccelerationX"]))
    ts = np.array(ts_ms, dtype=np.int64).astype("datetime64[ms]")
    return np.ascontiguousarray(vals, dtype=np.float64), ts


def _load_elf() -> tuple[np.ndarray, np.ndarray | None]:
    p = DATA_DIR / "elf" / "City-temp.csv"
    if not p.exists():
        pytest.skip(f"missing {p}")
    vals: list[float] = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                vals.append(float(s))
            except ValueError:
                continue
            if len(vals) >= MAX_SAMPLES:
                break
    return np.ascontiguousarray(vals, dtype=np.float64), None


Loader = Callable[[], tuple[np.ndarray, np.ndarray | None]]

FAMILY_LOADERS: list[tuple[str, Loader]] = [
    ("cwru_vibration", _load_cwru),
    ("mit_bih_ecg", _load_mitdb),
    ("iris_seismic", _load_iris),
    ("bidmc_clinical", _load_bidmc),
    ("uci_power", _load_uci_power),
    ("uci_hydraulic", _load_uci_hydraulic),
    ("skab_industrial", _load_skab),
    ("uci_air_quality", _load_uci_air_quality),
    ("uci_appliances", _load_uci_appliances),
    ("noaa_buoy", _load_noaa_buoy),
    ("era5_weather_f32", _load_era5_f32),
    ("binance_trades", _load_binance),
    ("smartphone_imu", _load_smartphone_imu),
    ("elf_corpus", _load_elf),
]

TIMED_LOADERS: list[tuple[str, Loader]] = [
    ("uci_power", _load_uci_power),
    ("smartphone_imu", _load_smartphone_imu),
    ("binance_trades", _load_binance),
]


@pytest.mark.parametrize(
    ("name", "loader"), FAMILY_LOADERS, ids=[k for k, _ in FAMILY_LOADERS]
)
@pytest.mark.parametrize(
    "params",
    [Params(), Params(noise_floor_sigma=0)],
    ids=["default", "nf0"],
)
def test_public_family_verify_and_fixed_point(
    name: str, loader: Loader, params: Params
) -> None:
    x, _ = loader()
    assert len(x) > 0, f"{name} loaded empty"
    res = fluxcode.encode_group(x, params, block_len=1000)
    assert verify.encode_group(res, x, params, block_len=1000) == []
    assert verify.fixed_point(res.payload, params) == []


@pytest.mark.parametrize(
    ("name", "loader"), TIMED_LOADERS, ids=[k for k, _ in TIMED_LOADERS]
)
@pytest.mark.parametrize(
    "params",
    [Params(), Params(time_error=0.15)],
    ids=["exact_times", "time_error_015"],
)
def test_public_timestamps_verify_and_fixed_point(
    name: str, loader: Loader, params: Params
) -> None:
    x, times = loader()
    assert times is not None and len(times) == len(x), f"{name} missing timestamps"
    res = fluxcode.encode_group(x, params, times=times, block_len=1000)
    assert verify.encode_group(res, x, params, times=times, block_len=1000) == []
    assert verify.fixed_point(res.payload, params) == []
