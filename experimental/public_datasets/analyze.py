# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Analyze fluxcode performance on real-world public time-series datasets.

First download the datasets once into the gitignored data/ folder:
    uv run python experimental/public_datasets/download.py

Then run the analysis:
    uv run python experimental/public_datasets/analyze.py --verify --out experimental/public_datasets/RESULTS.md
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import struct
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
import _fresh_numba_cache  # noqa: E402, F401
import fluxcode  # noqa: E402
import numpy as np  # noqa: E402
import zstandard as zstd  # noqa: E402
from fluxcode import Params, verify  # noqa: E402

DATA_DIR = Path(os.environ.get("FLUXCODE_DATA_DIR") or (Path(__file__).resolve().parent / "data"))


@dataclass
class Series:
    group: str
    name: str
    values: np.ndarray  # float64, 1D
    times: np.ndarray | None = None  # datetime64[s|ms], optional
    note: str = ""


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------


def _decode_mitdb_212(dat_path: Path, gain: float = 200.0, baseline: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Decode PhysioNet WFDB format 212 (two 12-bit signed samples packed per 3 bytes)."""
    raw = np.fromfile(dat_path, dtype=np.uint8)
    triplets = raw.reshape(-1, 3).astype(np.int32)
    b0, b1, b2 = triplets[:, 0], triplets[:, 1], triplets[:, 2]
    s0 = b0 | ((b1 & 0x0F) << 8)
    s1 = b2 | ((b1 & 0xF0) << 4)
    s0 = np.where(s0 >= 2048, s0 - 4096, s0)
    s1 = np.where(s1 >= 2048, s1 - 4096, s1)
    ch0 = (s0 - baseline).astype(np.float64) / gain
    ch1 = (s1 - baseline).astype(np.float64) / gain
    return np.ascontiguousarray(ch0), np.ascontiguousarray(ch1)


def _sext(val: int, bits: int) -> int:
    if val & (1 << (bits - 1)):
        val -= 1 << bits
    return val


def _decode_mseed_steim2(path: Path) -> np.ndarray:
    """Decode big-endian MiniSEED Steim-2 records, verifying every record's xn checksum."""
    buf = path.read_bytes()
    out: list[int] = []
    pos = 0
    while pos + 48 <= len(buf):
        nsamp = struct.unpack(">H", buf[pos + 30 : pos + 32])[0]
        data_off, blk_off = struct.unpack(">HH", buf[pos + 44 : pos + 48])
        btype, _bnext, enc, _bo, reclen_exp = struct.unpack(">HHBBB", buf[pos + blk_off : pos + blk_off + 7])
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
                        diffs = [_sext((word >> (15 - 15 * k)) & 0x7FFF, 15) for k in range(2)]
                    elif dnib == 3:
                        diffs = [_sext((word >> (20 - 10 * k)) & 0x3FF, 10) for k in range(3)]
                    else:
                        continue
                else:
                    dnib = (word >> 30) & 0x3
                    if dnib == 0:
                        diffs = [_sext((word >> (24 - 6 * k)) & 0x3F, 6) for k in range(5)]
                    elif dnib == 1:
                        diffs = [_sext((word >> (25 - 5 * k)) & 0x1F, 5) for k in range(6)]
                    elif dnib == 2:
                        diffs = [_sext((word >> (24 - 4 * k)) & 0xF, 4) for k in range(7)]
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
    return np.ascontiguousarray(out, dtype=np.float64)


def load_cwru() -> list[Series]:
    out: list[Series] = []
    specs = [
        ("1730_Normal.npz", "normal_48k", ["DE", "FE"], "48 kHz normal baseline"),
        ("1730_IR_7_DE48.npz", "fault_IR_48k", ["DE", "FE"], "48 kHz inner-race fault"),
        ("1730_B_7_DE12.npz", "fault_B_12k", ["DE", "FE", "BA"], "12 kHz ball fault"),
    ]
    for fname, tag, cols, note in specs:
        p = DATA_DIR / "cwru" / fname
        if not p.exists():
            continue
        z = np.load(p)
        for col in cols:
            if col in z.files:
                x = np.ascontiguousarray(z[col].ravel(), dtype=np.float64)
                out.append(Series("CWRU Vibration", f"{tag}_{col}", x, note=note))
    return out


def load_mitdb() -> list[Series]:
    out: list[Series] = []
    for rec, ch_names in [("100", ("MLII", "V5")), ("101", ("MLII", "V1"))]:
        p = DATA_DIR / "mitdb" / f"{rec}.dat"
        if not p.exists():
            continue
        ch0, ch1 = _decode_mitdb_212(p)
        out.append(Series("MIT-BIH ECG", f"rec{rec}_{ch_names[0]}", ch0, note="360 Hz ECG, 0.005 mV ADC grid"))
        out.append(Series("MIT-BIH ECG", f"rec{rec}_{ch_names[1]}", ch1, note="360 Hz ECG, 0.005 mV ADC grid"))
    return out


def load_iris_seismic() -> list[Series]:
    out: list[Series] = []
    for cha in ("BHZ", "BH1"):
        p = DATA_DIR / "iris" / f"IU_ANMO_00_{cha}.mseed"
        if not p.exists():
            continue
        counts = _decode_mseed_steim2(p)
        out.append(Series("IRIS Seismic", f"ANMO_{cha}_counts", counts, note="40 Hz 24-bit sigma-delta ADC counts"))
        if cha == "BHZ":
            # Calibrated ground velocity (m/s) using IU.ANMO nominal stage-0 sensitivity
            vel = np.ascontiguousarray(counts / 3.35544e9, dtype=np.float64)
            out.append(Series("IRIS Seismic", "ANMO_BHZ_mps", vel, note="40 Hz calibrated velocity (m/s, ~1e-4 scale)"))
    return out


def load_uci_household_power() -> list[Series]:
    p = DATA_DIR / "uci" / "household_power" / "household_power_consumption.txt"
    if not p.exists():
        return []
    cols = [
        "Global_active_power",
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3",
    ]
    n_rows = 2_075_259
    arrs = [np.empty(n_rows, dtype=np.float64) for _ in cols]
    ts_int = np.empty(n_rows, dtype=np.int64)
    epoch = datetime(1970, 1, 1)
    idx = 0
    with open(p, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split(";")
            if len(parts) < 9:
                continue
            dd, mm, yy = parts[0].split("/")
            hh, mi, ss = parts[1].split(":")
            dt = datetime(int(yy), int(mm), int(dd), int(hh), int(mi), int(ss))
            ts_int[idx] = int((dt - epoch).total_seconds())
            for j in range(7):
                v = parts[2 + j]
                arrs[j][idx] = np.nan if (v == "?" or v == "") else float(v)
            idx += 1
    ts = ts_int[:idx].astype("datetime64[s]")
    return [
        Series("UCI Power", col, np.ascontiguousarray(arrs[j][:idx]), times=ts, note="1-min smart meter w/ NaN gaps")
        for j, col in enumerate(cols)
    ]


def load_uci_air_quality() -> list[Series]:
    p = DATA_DIR / "uci" / "air_quality" / "AirQualityUCI.csv"
    if not p.exists():
        return []
    keep = ["CO(GT)", "PT08.S1(CO)", "C6H6(GT)", "NOx(GT)", "T", "RH", "AH"]
    data: dict[str, list[float]] = {k: [] for k in keep}
    ts_list: list[int] = []
    epoch = datetime(1970, 1, 1)
    with open(p, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            d_str = (row.get("Date") or "").strip()
            t_str = (row.get("Time") or "").strip()
            if not d_str or not t_str:
                continue
            dd, mm, yy = d_str.split("/")
            hh, mi, ss = t_str.split(".")
            dt = datetime(int(yy), int(mm), int(dd), int(hh), int(mi), int(ss))
            ts_list.append(int((dt - epoch).total_seconds()))
            for k in keep:
                s = (row.get(k) or "").strip().replace(",", ".")
                data[k].append(float(s) if s else np.nan)
    ts = np.array(ts_list, dtype=np.int64).astype("datetime64[s]")
    return [
        Series("UCI AirQuality", k, np.ascontiguousarray(data[k], dtype=np.float64), times=ts, note="hourly sensor w/ -200 glitches")
        for k in keep
    ]


def load_uci_appliances() -> list[Series]:
    p = DATA_DIR / "uci" / "appliances_energy" / "energydata_complete.csv"
    if not p.exists():
        return []
    keep = ["Appliances", "T1", "RH_1", "T6", "T_out", "Press_mm_hg", "Windspeed", "rv1"]
    data: dict[str, list[float]] = {k: [] for k in keep}
    ts_list: list[np.datetime64] = []
    with open(p, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_list.append(np.datetime64(row["date"].strip(), "s"))
            for k in keep:
                data[k].append(float(row[k].strip()))
    ts = np.array(ts_list, dtype="datetime64[s]")
    return [
        Series(
            "UCI Appliances",
            k,
            np.ascontiguousarray(data[k], dtype=np.float64),
            times=ts,
            note="10-min sensors (1/3, 1/4 grids)" if k != "rv1" else "uniform random noise column",
        )
        for k in keep
    ]


def load_uci_hydraulic() -> list[Series]:
    hdir = DATA_DIR / "uci" / "hydraulic_systems"
    if not hdir.exists():
        return []
    # Cap 100 Hz sensors to first 200 cycles (1.2M samples each) to keep runtime brisk
    specs = [
        ("PS1.txt", "PS1_100Hz", 200, "100 Hz hydraulic pressure (200 cycles)"),
        ("EPS1.txt", "EPS1_100Hz", 200, "100 Hz motor power (200 cycles)"),
        ("FS1.txt", "FS1_10Hz", None, "10 Hz volume flow (all 2205 cycles)"),
        ("TS1.txt", "TS1_1Hz", None, "1 Hz temperature (all 2205 cycles)"),
        ("VS1.txt", "VS1_1Hz", None, "1 Hz vibration (all 2205 cycles)"),
    ]
    out: list[Series] = []
    for fname, label, max_cycles, note in specs:
        p = hdir / fname
        if not p.exists():
            continue
        chunks: list[np.ndarray] = []
        with open(p, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_cycles is not None and i >= max_cycles:
                    break
                chunks.append(np.fromstring(line, sep="\t", dtype=np.float64))
        if chunks:
            out.append(Series("UCI Hydraulic", label, np.ascontiguousarray(np.concatenate(chunks)), note=note))
    return out


def load_skab() -> list[Series]:
    p = DATA_DIR / "skab" / "anomaly-free.csv"
    if not p.exists():
        return []
    cols = [
        "Accelerometer1RMS",
        "Current",
        "Pressure",
        "Temperature",
        "Thermocouple",
        "Voltage",
        "Volume Flow RateRMS",
    ]
    short = {
        "Accelerometer1RMS": "Accel1RMS",
        "Volume Flow RateRMS": "FlowRateRMS",
    }
    data: dict[str, list[float]] = {c: [] for c in cols}
    ts_list: list[np.datetime64] = []
    with open(p, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            ts_list.append(np.datetime64(row["datetime"].strip(), "s"))
            for c in cols:
                data[c].append(float(row[c].strip()))
    ts = np.array(ts_list, dtype="datetime64[s]")
    return [
        Series("SKAB Industrial", short.get(c, c), np.ascontiguousarray(data[c], dtype=np.float64), times=ts, note="1 Hz PLC water testbed")
        for c in cols
    ]


def load_binance_trades() -> list[Series]:
    p = DATA_DIR / "binance" / "BTCUSDT-trades-2024-01-01" / "BTCUSDT-trades-2024-01-01.csv"
    if not p.exists():
        return []
    prices: list[float] = []
    qtys: list[float] = []
    ts_ms: list[int] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split(",")
            if len(parts) < 5:
                continue
            prices.append(float(parts[1]))
            qtys.append(float(parts[2]))
            ts_ms.append(int(parts[4]))
    ts = np.array(ts_ms, dtype=np.int64).astype("datetime64[ms]")
    return [
        Series("Binance Trades", "BTCUSDT_price", np.ascontiguousarray(prices, dtype=np.float64), times=ts, note="irregular ms trade ticks (cents)"),
        Series("Binance Trades", "BTCUSDT_qty", np.ascontiguousarray(qtys, dtype=np.float64), times=ts, note="irregular ms trade quantities"),
    ]


def load_smartphone_imu() -> list[Series]:
    out: list[Series] = []
    specs = [
        ("p001_hand_walking.csv", "walk"),
        ("p001_hand_running.csv", "run"),
    ]
    cols = [
        ("accelerometerAccelerationX", "ax"),
        ("accelerometerAccelerationZ", "az"),
        ("gyroRotationX", "gx"),
    ]
    for fname, prefix in specs:
        p = DATA_DIR / "imu" / fname
        if not p.exists():
            continue
        ts_ms: list[int] = []
        vals: dict[str, list[float]] = {short: [] for _, short in cols}
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts_ms.append(int(row["elapsedMillis"]))
                for col, short in cols:
                    vals[short].append(float(row[col]))
        ts = np.array(ts_ms, dtype=np.int64).astype("datetime64[ms]")
        for _, short in cols:
            out.append(
                Series("Smartphone IMU", f"{prefix}_{short}", np.ascontiguousarray(vals[short], dtype=np.float64), times=ts, note="jittered ~50 Hz phone clock")
            )
    return out


def load_era5() -> list[Series]:
    p = DATA_DIR / "era5" / "sf_2023_hourly.csv"
    if not p.exists():
        return []
    lines = p.read_text(encoding="utf-8").splitlines()
    # Header is after the metadata blank line (line 3)
    hdr_idx = next(i for i, l in enumerate(lines) if l.startswith("time,"))
    reader = csv.DictReader(lines[hdr_idx:])
    col_map = {
        "temperature_2m (°C)": "temp_2m",
        "surface_pressure (hPa)": "pressure",
        "wind_speed_10m (km/h)": "wind_10m",
        "shortwave_radiation (W/m²)": "shortwave_rad",
    }
    data: dict[str, list[float]] = {v: [] for v in col_map.values()}
    ts_list: list[np.datetime64] = []
    for row in reader:
        ts_list.append(np.datetime64(row["time"].strip(), "s"))
        for src, dst in col_map.items():
            data[dst].append(float(row[src]))
    ts = np.array(ts_list, dtype="datetime64[s]")
    out: list[Series] = []
    for name, vals in data.items():
        arr = np.ascontiguousarray(vals, dtype=np.float64)
        out.append(Series("ERA5 Weather", name, arr, times=ts, note="hourly ERA5 decimal"))
        # Float32-originated copy promoted to float64 (tests float32-to-decimal recovery)
        if name in ("temp_2m", "pressure"):
            arr_f32 = np.ascontiguousarray(arr.astype(np.float32).astype(np.float64))
            out.append(Series("ERA5 Weather", f"{name}_f32", arr_f32, times=ts, note="float32-originated ERA5 promoted to f64"))
    return out


def load_bidmc() -> list[Series]:
    p = DATA_DIR / "bidmc" / "bidmc_01_Signals.csv"
    if not p.exists():
        return []
    cols = [("RESP", "RESP"), ("PLETH", "PLETH"), ("II", "ECG_II"), ("V", "ECG_V")]
    data: dict[str, list[float]] = {dst: [] for _, dst in cols}
    with open(p, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Strip whitespace from header keys
        reader.fieldnames = [k.strip() for k in (reader.fieldnames or [])]
        for row in reader:
            for src, dst in cols:
                data[dst].append(float(row[src].strip()))
    return [
        Series("BIDMC Clinical", dst, np.ascontiguousarray(data[dst], dtype=np.float64), note="125 Hz PPG/Resp/ECG")
        for _, dst in cols
    ]


def load_noaa_buoy() -> list[Series]:
    p = DATA_DIR / "noaa" / "46026h2023.txt"
    if not p.exists():
        return []
    epoch = datetime(1970, 1, 1)
    col_idx = {"WSPD": 6, "GST": 7, "WVHT": 8, "PRES": 12, "WTMP": 14}
    sentinels = {"WSPD": 99.0, "GST": 99.0, "WVHT": 99.0, "PRES": 9999.0, "WTMP": 999.0}
    raw_cols: dict[str, list[float]] = {k: [] for k in col_idx}
    ts_list: list[int] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 15:
                continue
            dt = datetime(int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]))
            ts_list.append(int((dt - epoch).total_seconds()))
            for k, i in col_idx.items():
                raw_cols[k].append(float(parts[i]))
    ts = np.array(ts_list, dtype=np.int64).astype("datetime64[s]")
    out: list[Series] = []
    for k, vals in raw_cols.items():
        arr = np.ascontiguousarray(vals, dtype=np.float64)
        out.append(Series("NOAA Buoy", f"{k}_raw", arr, times=ts, note="10-min buoy w/ 99/999 sentinels"))
        clean = np.where(arr == sentinels[k], np.nan, arr)
        if np.isnan(clean).any():
            out.append(Series("NOAA Buoy", f"{k}_nan", clean, times=ts, note="10-min buoy w/ sentinels->NaN"))
    return out


def load_elf() -> list[Series]:
    elf_dir = DATA_DIR / "elf"
    if not elf_dir.exists():
        return []
    out: list[Series] = []
    for p in sorted(elf_dir.glob("*.csv")):
        vals: list[float] = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    vals.append(float(s))
                except ValueError:
                    continue
        if vals:
            out.append(Series("Elf Corpus", p.stem, np.ascontiguousarray(vals, dtype=np.float64), note="VLDB/SIGMOD float benchmark"))
    return out


# ---------------------------------------------------------------------------
# Baselines & evaluation helpers
# ---------------------------------------------------------------------------

_ZSTD_C = zstd.ZstdCompressor(level=3)


def raw_zstd_bps(x: np.ndarray) -> float:
    return 8.0 * len(_ZSTD_C.compress(x.tobytes())) / len(x)


def bshuf_zstd_bps(x: np.ndarray) -> float:
    b = x.view(np.uint8).reshape(-1, 8).T.copy()
    return 8.0 * len(_ZSTD_C.compress(b.tobytes())) / len(x)


def eval_fluxcode(
    x: np.ndarray,
    params: Params,
    *,
    times: np.ndarray | None = None,
    block_len: int = 1000,
    blocks_per_group: int = 60,
    run_verify: bool = False,
) -> dict[str, float | bool | int]:
    t0 = time.perf_counter()
    groups, mins, maxs, _means = fluxcode.encode(
        x, params, block_len=block_len, blocks_per_group=blocks_per_group, times=times
    )
    te = time.perf_counter() - t0

    t0 = time.perf_counter()
    decoded = fluxcode.decode(groups)
    td = time.perf_counter() - t0

    y = np.concatenate([d.values for d in decoded])
    total_bytes = sum(len(g) for g in groups)
    bps = 8.0 * total_bytes / len(x)

    exact = bool(np.array_equal(x, y, equal_nan=True))

    finite = np.isfinite(x) & np.isfinite(y)
    if finite.any():
        abs_err = np.abs(y[finite] - x[finite])
        rmse = float(np.sqrt(np.mean(abs_err ** 2)))
        span = float(np.max(x[finite]) - np.min(x[finite]))
        rmse_pct = (100.0 * rmse / span) if span > 0 else 0.0
    else:
        rmse_pct = 0.0

    mins_flat = np.concatenate(mins)
    maxs_flat = np.concatenate(maxs)
    worst_pct = 0.0
    pos = 0
    b_idx = 0
    for d in decoded:
        for sz in d.block_sizes:
            sz_i = int(sz)
            xb = x[pos : pos + sz_i]
            yb = y[pos : pos + sz_i]
            fb = np.isfinite(xb) & np.isfinite(yb)
            if fb.any():
                lo, hi = float(mins_flat[b_idx]), float(maxs_flat[b_idx])
                rng = hi - lo if (np.isfinite(hi) and np.isfinite(lo) and hi > lo) else 1.0
                err = float(np.max(np.abs(yb[fb] - xb[fb])))
                worst_pct = max(worst_pct, 100.0 * err / rng)
            pos += sz_i
            b_idx += 1

    verify_errs = 0
    if run_verify:
        pos = 0
        for g_idx, g in enumerate(groups):
            n_g = int(np.sum(decoded[g_idx].block_sizes))
            if g_idx in (0, len(groups) - 1):
                xg = x[pos : pos + n_g]
                tg = times[pos : pos + n_g] if times is not None else None
                res = fluxcode.encode_group(xg, params, block_len=block_len, times=tg)
                errs = verify.encode_group(res, xg, params, block_len=block_len, times=tg)
                verify_errs += len(errs)
            pos += n_g

    mb = 8.0 * len(x) / 1e6
    return {
        "bps": bps,
        "exact": exact,
        "worst_pct": worst_pct,
        "rmse_pct": rmse_pct,
        "enc_mbs": mb / te if te > 0 else 0.0,
        "dec_mbs": mb / td if td > 0 else 0.0,
        "verify_errs": verify_errs,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", action="store_true", help="Run fluxcode.verify on first & last block group of each series.")
    ap.add_argument("--out", type=Path, default=None, help="Optional path to write Markdown report.")
    ap.add_argument("--json", type=Path, default=None, help="Optional path to write structured JSON results.")
    args = ap.parse_args()

    t_load = time.perf_counter()
    all_series: list[Series] = []
    for loader in (
        load_cwru,
        load_mitdb,
        load_iris_seismic,
        load_bidmc,
        load_uci_household_power,
        load_uci_hydraulic,
        load_skab,
        load_uci_air_quality,
        load_uci_appliances,
        load_noaa_buoy,
        load_era5,
        load_binance_trades,
        load_smartphone_imu,
        load_elf,
    ):
        all_series.extend(loader())
    print(f"Loaded {len(all_series)} series ({sum(len(s.values) for s in all_series):,} total samples) in {time.perf_counter() - t_load:.1f}s")

    # Warm up numba kernels once
    warm = np.linspace(0.0, 10.0, 2000, dtype=np.float64)
    warm_t = np.datetime64("2026-01-01T00:00", "ms") + np.arange(2000) * np.timedelta64(10, "ms")
    for p in (Params(), Params(noise_floor_sigma=0), Params(noise_floor_sigma=0, grids="binary"), Params(effort=1), Params(effort=5)):
        fluxcode.decode(fluxcode.encode(warm, p)[0])
    fluxcode.decode(fluxcode.encode(warm, Params(time_error=0.05), times=warm_t)[0])

    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    emit("# fluxcode on Public Real-World Datasets\n")
    emit(f"Total: **{len(all_series)} series**, **{sum(len(s.values) for s in all_series):,} float64 samples** "
         f"({8 * sum(len(s.values) for s in all_series) / 1e6:.1f} MB uncompressed) across 14 dataset families.\n")
    emit("Columns: `raw zstd` (zstd L3 on float64 bytes), `bshuf zstd` (byte-shuffled float64 + zstd L3; note that on "
         "calibrated-ADC/decimal series with few unique float64 values, raw zstd's LZ77 matches whole 8-byte repeats while "
         "byte-shuffling separates low mantissa bytes), `fc default` (`Params()`), `fc nf=0` (`noise_floor_sigma=0`), "
         "`fc bin nf=0` (`grids='binary', noise_floor_sigma=0`), plus exactness and worst block max error (% of block range).\n")

    configs = {
        "default": Params(),
        "nf0": Params(noise_floor_sigma=0),
        "bin_nf0": Params(noise_floor_sigma=0, grids="binary"),
        "eff1": Params(effort=1),
        "eff5": Params(effort=5),
    }

    groups_order: list[str] = []
    for s in all_series:
        if s.group not in groups_order:
            groups_order.append(s.group)

    group_totals: dict[str, dict[str, float]] = {}
    json_rows: list[dict[str, object]] = []
    verify_failures: list[str] = []

    for gname in groups_order:
        gseries = [s for s in all_series if s.group == gname]
        emit(f"## {gname}\n")
        emit("| series | N | NaN% | raw zstd | bshuf zstd | fc default | err% (def) | fc nf=0 | exact? (nf=0) | fc bin nf=0 | grid gain | enc/dec MB/s (def) |")
        emit("|---|---:|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|")

        tot_n = 0
        tot_bits = {k: 0.0 for k in ("raw", "bshuf", "default", "nf0", "bin_nf0", "eff1", "eff5")}

        for s in gseries:
            x = s.values
            n = len(x)
            nan_pct = 100.0 * float(np.isnan(x).sum()) / n
            rz = raw_zstd_bps(x)
            bz = bshuf_zstd_bps(x)
            res = {k: eval_fluxcode(x, p, run_verify=(args.verify and k in ("default", "nf0"))) for k, p in configs.items()}

            for k in ("default", "nf0"):
                if res[k]["verify_errs"]:
                    verify_failures.append(f"{gname}/{s.name} ({k}): {res[k]['verify_errs']} errors")

            tot_n += n
            tot_bits["raw"] += rz * n
            tot_bits["bshuf"] += bz * n
            for k in configs:
                tot_bits[k] += float(res[k]["bps"]) * n

            nf0_bps = float(res["nf0"]["bps"])
            bin_bps = float(res["bin_nf0"]["bps"])
            grid_gain = 100.0 * (bin_bps - nf0_bps) / bin_bps if bin_bps > 0 else 0.0
            exact_str = "EXACT" if res["nf0"]["exact"] else f"{res['nf0']['worst_pct']:.4f}%"
            def_err = "EXACT" if res["default"]["exact"] else f"{res['default']['worst_pct']:.3f}%"
            spd = f"{res['default']['enc_mbs']:.0f} / {res['default']['dec_mbs']:.0f}"

            emit(
                f"| {s.name} | {n:,} | {nan_pct:.1f}% | {rz:.2f} | {bz:.2f} | "
                f"**{res['default']['bps']:.2f}** | {def_err} | {nf0_bps:.2f} | {exact_str} | "
                f"{bin_bps:.2f} | {grid_gain:+.1f}% | {spd} |"
            )
            json_rows.append(
                {
                    "group": gname,
                    "series": s.name,
                    "n": n,
                    "nan_pct": nan_pct,
                    "raw_zstd": rz,
                    "bshuf_zstd": bz,
                    "results": res,
                }
            )

        g_raw = tot_bits["raw"] / tot_n
        g_bsh = tot_bits["bshuf"] / tot_n
        g_def = tot_bits["default"] / tot_n
        g_nf0 = tot_bits["nf0"] / tot_n
        g_bin = tot_bits["bin_nf0"] / tot_n
        g_gain = 100.0 * (g_bin - g_nf0) / g_bin if g_bin > 0 else 0.0
        emit(
            f"| **group avg ({tot_n:,})** | | | **{g_raw:.2f}** | **{g_bsh:.2f}** | "
            f"**{g_def:.2f}** | | **{g_nf0:.2f}** | | **{g_bin:.2f}** | **{g_gain:+.1f}%** | |"
        )
        emit("")
        group_totals[gname] = {
            "n": tot_n,
            "raw": g_raw,
            "bshuf": g_bsh,
            "default": g_def,
            "nf0": g_nf0,
            "bin_nf0": g_bin,
            "eff1": tot_bits["eff1"] / tot_n,
            "eff5": tot_bits["eff5"] / tot_n,
        }

    emit("## Group Summary (Sample-Weighted bits/sample)\n")
    emit("| dataset group | samples | raw zstd | bshuf zstd | fc default | fc eff=1 | fc eff=5 | fc nf=0 | fc bin nf=0 | ratio vs f64 (def) |")
    emit("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for gname, d in group_totals.items():
        emit(
            f"| {gname} | {int(d['n']):,} | {d['raw']:.2f} | {d['bshuf']:.2f} | "
            f"**{d['default']:.2f}** | {d['eff1']:.2f} | {d['eff5']:.2f} | "
            f"{d['nf0']:.2f} | {d['bin_nf0']:.2f} | {64.0 / d['default']:.1f}x |"
        )
    emit("")

    # Timestamp Axis Analysis (regular + jittered + irregular real clocks)
    emit("## Timestamp Axis Analysis\n")
    emit("Comparing values-only vs values + exact timestamps vs `time_error=0.05` and `time_error=0.15` across regular, "
         "gapped, jittered (smartphone IMU), and irregular (exchange trade ticks) real-world clocks.\n")
    emit("| dataset (channel) | N | clock type | values only | + exact times | exact time cost | + `time_error=0.05` | + `time_error=0.15` |")
    emit("|---|---:|---|---:|---:|---:|---:|---:|")
    ts_reps = [
        ("UCI Power", "Global_active_power", "1-min regular"),
        ("UCI AirQuality", "T", "1-h regular"),
        ("UCI Appliances", "T1", "10-min regular"),
        ("SKAB Industrial", "Pressure", "1-s PLC regular"),
        ("ERA5 Weather", "temp_2m", "1-h regular"),
        ("NOAA Buoy", "WSPD_raw", "10-min w/ gaps"),
        ("Smartphone IMU", "walk_ax", "~50 Hz jittered ms"),
        ("Smartphone IMU", "run_ax", "~50 Hz jittered ms"),
        ("Binance Trades", "BTCUSDT_price", "irregular ms ticks"),
    ]
    ts_json: list[dict[str, object]] = []
    for gname, sname, ctype in ts_reps:
        matches = [s for s in all_series if s.group == gname and s.name == sname and s.times is not None]
        if not matches:
            continue
        s = matches[0]
        v_only = eval_fluxcode(s.values, Params())["bps"]
        v_exact_r = eval_fluxcode(s.values, Params(), times=s.times, run_verify=args.verify)
        v_t05_r = eval_fluxcode(s.values, Params(time_error=0.05), times=s.times, run_verify=args.verify)
        v_t15_r = eval_fluxcode(s.values, Params(time_error=0.15), times=s.times, run_verify=args.verify)
        for label, r in (("exact_t", v_exact_r), ("terr_0.05", v_t05_r), ("terr_0.15", v_t15_r)):
            if r["verify_errs"]:
                verify_failures.append(f"{gname}/{sname} ({label}): {r['verify_errs']} errors")
        v_exact = float(v_exact_r["bps"])
        v_t05 = float(v_t05_r["bps"])
        v_t15 = float(v_t15_r["bps"])
        emit(
            f"| {gname} / {sname} | {len(s.values):,} | {ctype} | {v_only:.2f} | {v_exact:.2f} | "
            f"+{v_exact - v_only:.3f} | {v_t05:.2f} | {v_t15:.2f} |"
        )
        ts_json.append(
            {
                "group": gname,
                "series": sname,
                "clock_type": ctype,
                "n": len(s.values),
                "values_only": v_only,
                "exact_times": v_exact,
                "time_error_005": v_t05,
                "time_error_015": v_t15,
            }
        )
    emit("")

    if args.verify:
        if verify_failures:
            emit(f"## Verification: FAILED ({len(verify_failures)} failures)\n")
            for f in verify_failures:
                emit(f"- {f}")
        else:
            emit("## Verification: ALL PASSED (`fluxcode.verify.encode_group` returned 0 errors on all tested groups, including timestamped & `time_error` runs)\n")

    if args.out is not None:
        args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Wrote report to {args.out}")

    if args.json is not None:
        args.json.write_text(
            json.dumps({"groups": group_totals, "series": json_rows, "timestamps": ts_json}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote JSON to {args.json}")


if __name__ == "__main__":
    main()
