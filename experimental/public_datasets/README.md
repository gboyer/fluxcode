# Public Dataset Experiments for fluxcode

This directory contains scripts to download and benchmark `fluxcode` against **real-world public float64 time-series datasets** spanning **14 dataset families** (94 series, 29.22M float64 samples, 233.8 MB uncompressed): high-frequency vibration, biomedical ECG/PPG waveforms, 24-bit sigma-delta broadband seismology, industrial hydraulic & water-plant PLC telemetry, residential smart-meter power, oceanographic buoys, float32-originated ERA5 reanalysis weather, high-frequency irregular exchange trade ticks, jittered smartphone IMU clocks, and the standard VLDB/SIGMOD floating-point compression corpus (Elf / Chimp / ALP).

Datasets are **not** committed to the repository. Instead, `experimental/public_datasets/data/` is gitignored (overrideable via `FLUXCODE_DATA_DIR`) and populated once locally from the SHA-256-verified manifest [datasets.json](datasets.json).

## Quickstart

From the repository root using the `Makefile`:

```sh
make init           # install locked root + experimental environments
make datasets       # download & SHA-256-verify all 38 files (~171 MB) into data/
make test-public    # run opt-in verify.encode_group + fixed_point tests across all 14 families
make public-bench   # regenerate RESULTS.md and results.json
make public-report  # render ../report/public.html and public-*.svg from results.json
```

Or run the scripts directly (e.g. `uv run python experimental/public_datasets/download.py --verify`).

Rendered HTML report: [../report/public.html](../report/public.html) (linked from [../report/index.html](../report/index.html)).

## Datasets (14 Families)

| Group | Source | Series | Sampling / Characteristics |
|---|---|---:|---|
| **CWRU Vibration** | [Case Western Reserve University Bearing Data](https://engineering.case.edu/bearingdatacenter) via [srigas/CWRU_Bearing_NumPy](https://github.com/srigas/CWRU_Bearing_NumPy) | 7 | 12 kHz & 48 kHz mechanical accelerometer waveforms (normal, inner-race fault, ball fault; hidden ADC calibration step). |
| **MIT-BIH ECG** | [PhysioNet MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/) (records 100, 101) | 4 | 360 Hz 2-channel ECG (650k samples/ch), 11-bit ADC scaled to mV (`0.005` mV scaled decimal grid). |
| **IRIS Seismic** | [IRIS/EarthScope FDSN](https://service.iris.edu/fdsnws/dataselect/1/) (`IU.ANMO.00.BHZ/BH1`, 2023-02-06 M7.8 earthquake) | 3 | 40 Hz 24-bit sigma-delta broadband seismometer (288k samples/ch, steep anti-alias filter; raw counts & calibrated `m/s` velocity). |
| **BIDMC Clinical** | [PhysioNet BIDMC PPG & Respiration](https://physionet.org/content/bidmc/1.0.0/) (record 01) | 4 | 125 Hz clinical monitor (60k samples/ch: RESP, PLETH, ECG II & V). |
| **UCI Power** | [UCI Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235) | 7 | 1-min smart-meter telemetry (~2.08M rows × 7 ch), exact timestamps + ~1.3% NaN gaps. |
| **UCI Hydraulic** | [UCI Condition Monitoring of Hydraulic Systems](https://archive.ics.uci.edu/dataset/447) | 5 | Multi-rate industrial hydraulic PLC/DAQ (100 Hz pressure/motor power, 10 Hz flow, 1 Hz temp/vibration; 3.99M samples). |
| **SKAB Industrial** | [Skoltech Anomaly Benchmark (SKAB)](https://github.com/waico/SKAB) (`anomaly-free.csv`) | 7 | 1 Hz industrial water-circulation plant sensors (accelerometers, current, pressure, temp, thermocouple, voltage, flow) w/ timestamps. |
| **UCI AirQuality** | [UCI Air Quality](https://archive.ics.uci.edu/dataset/360) | 7 | Hourly chemical sensor array (9,357 rows), exact timestamps + `-200` error-code glitches. |
| **UCI Appliances** | [UCI Appliances Energy Prediction](https://archive.ics.uci.edu/dataset/374) | 8 | 10-min indoor/outdoor temp & humidity sensors (averaged 1/3, 1/4 rational grids) + `rv1` uniform random noise. |
| **NOAA Buoy** | [NOAA NDBC Station 46026 (2023)](https://www.ndbc.noaa.gov/station_page.php?station=46026) | 10 | 10-min ocean buoy telemetry w/ gaps & `99`/`999` sentinels (tested both raw and sentinels → `NaN`). |
| **ERA5 Weather** | [Open-Meteo ERA5 Reanalysis](https://open-meteo.com/) (San Francisco 2023, hourly) | 6 | Hourly weather fields (8,760 rows), tested both as-parsed decimal and float32-promoted-to-float64 (`*_f32`). |
| **Binance Trades** | [Binance Public Data Vision](https://data.binance.vision/) (`BTCUSDT-trades-2024-01-01`) | 2 | 1.11M high-frequency exchange trades with irregular millisecond arrival timestamps, decimal prices, and quantities. |
| **Smartphone IMU** | [AZI350 Smartphone IMU Dataset](https://github.com/AZI350/Smartphone-IMU-Three-Positions-Dataset) (walking & running) | 6 | 3-axis accelerometer & gyroscope on a real jittered ~50 Hz millisecond smartphone clock (`dt` 16–177 ms). |
| **Elf Corpus** | [Spatio-Temporal-Lab/Elf](https://github.com/Spatio-Temporal-Lab/Elf) `ElfTestData` | 18 | Standard VLDB/SIGMOD float-compression benchmark columns (weather, air sensors, GPS/POI, stocks, crypto, dust). |

## Results

See [RESULTS.md](RESULTS.md) for the full per-series tables and timestamp-axis measurements.

### Highlights (94 series, 29.22M float64 samples, 233.8 MB uncompressed)

- **All guarantee checks passed:** `fluxcode.verify.encode_group` reported **0 errors** across all 94 real-world series, including exact timestamps and `time_error` runs.
- **Compression ratios vs float64 (default `Params()`):** **6.8×** on CWRU vibration (9.41 bps), **7.5×** on Smartphone IMU (8.50 bps), **12.1×** on Elf corpus (5.29 bps), **13.2×** on BIDMC clinical (4.84 bps), **14.2×** on IRIS 24-bit seismic (4.50 bps), **14.7×** on UCI Hydraulic PLC (4.35 bps), **15.4×** on MIT-BIH ECG (4.14 bps, exact) and Binance trades (4.14 bps), **19.0×** on NOAA buoy (3.37 bps), and **21.2×** on UCI Power (3.03 bps).
- **Hidden rational/scaled grids & float32 recovery in the wild:**
  - **CWRU Vibration** (`~2.086e-4` ADC step): +24.5% over binary grids, ~1 ULP (`2.8e-17`) at `nf=0`.
  - **MIT-BIH ECG** (`0.005` mV ADC step): **4.14 bps exact**, +56.7% over binary grids.
  - **UCI Hydraulic** (100 Hz–1 Hz PLC): +33.4% over binary grids (all 5 channels **exact** at `nf=0`).
  - **UCI Appliances** (averaged 1/3, 1/4 grids): +55–67% over binary grids (`0.92–1.31` bps at 0.0000% error).
  - **ERA5 Weather (`temp_2m_f32`, `pressure_f32`):** Float32-originated values promoted to float64 recover the exact decimal grid at **identical** bps (**4.95** and **4.10** bps, +52–58% over binary).
- **Sigma-delta anti-alias hardware (IRIS Seismic):** 40 Hz 24-bit broadband seismometer data compresses to **4.16–4.37 bps** (counts, 0.002% max error) and **4.96 bps** (`m/s` velocity, 0.023% max error — 7× smaller than raw zstd's 34.46 bps).
- **Timestamps (regular vs jittered vs irregular):**
  - Regular / gapped clocks (UCI Power, AirQuality, Appliances, ERA5, NOAA Buoy): **+0.006 to +0.028 bps**.
  - SKAB 1 Hz PLC (minor drops/drift): **+0.863 bps**.
  - Jittered ~50 Hz smartphone IMU (`walk_ax`/`run_ax`): exact times cost **+7.9 to +9.0 bps**; `time_error=0.15` cuts **~1.7 bps**.
  - Irregular exchange trade ticks (Binance, 1.11M trades): exact ms times cost **+5.97 bps** (9.00 bps total with exact cents prices), and `time_error` rightly leaves truly irregular arrivals untouched.
- **Why `bshuf zstd` is larger than `raw zstd` on calibrated-ADC/decimal series:** On series with relatively few unique quantized float64 values (e.g., CWRU `normal_48k_DE` has 2,254 unique values across 485k samples), `raw zstd`'s LZ77 matches entire repeated 8-byte float64 values at once (~16.8 bps), whereas byte-shuffling separates the 6 low mantissa bytes into independent near-random streams (~49.8 bps). `fluxcode` beats both by recovering the integer grid before delta + bit-plane coding.
