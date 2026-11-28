# Fluxcode Agent Guidelines

## Project Layout & Boundaries
- **`fluxcode/` and `fluxprune/`**: Core Python packages shipped in wheels/sdists. Keep dependencies minimal (`numpy`, `numba`, `zstandard`) and maintain strict typing (`mypy --strict`, `ty`).
- **`rust/`**: Optional `fluxcode-rs` PyO3 accelerator (`--extra rust`).
- **`bench/`**: Benchmark scripts and recorded `*_RESULTS.md` snapshots.
- **`experimental/`**: Separate `uv` project for exploration, public-dataset evaluation, and HTML/SVG report generation. Not packaged in releases.

## Python Environment
- Always use `uv` (`uv run`, `uv sync --locked`); do not invoke bare `python` or `pip`.
- For `experimental/`, run from `experimental/` or use `uv run --locked --project experimental ...`.

## Verification Checklist
Run before merging any change:
- **Standard CI checks**: `make check` (`ruff check`, `ruff format --check`, `experimental/` syntax/undefined-name lint + `experimental/smoke.py`, `mypy --strict fluxcode fluxprune rust/fluxcode_rs.pyi`, `ty check`, and offline `pytest`).
- **Rust dual-path checks** (required when touching `fluxcode/`, `rust/`, or encoding/decoding):
  - `make check-rust` (`cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, and `rust/src/planes.rs` SIMD tests)
  - `uv run --locked --extra rust pytest`
  - `FLUXCODE_RUST=0 uv run --locked --extra rust pytest`
- **Public-dataset tests** (when touching the wire format, quantization, or public-dataset pipeline): `make test-public`.

## Docs, Benchmarks & Reports
- Keep documentation aligned with code changes:
  - Wire format -> `docs/SPEC.md`
  - Encoder architecture & heuristics -> `docs/ENCODER.md`, `docs/TUNING.md`
  - Benchmarks & throughput -> `docs/PERFORMANCE.md`, `bench/*_RESULTS.md`
- When recording timing/throughput benchmarks, note whether the Rust accelerator was enabled or disabled (`FLUXCODE_RUST=0/1`).
- Regenerate affected reports in `experimental/report/` via:
  - `make public-report` (`experimental/make_public_report.py`)
  - `cd experimental && uv run python make_site.py` (overview `index.html` and rendered `spec.html`)
  - Topic-specific report generators as needed: `make_report.py`, `make_prune_report.py`, `make_rate_report.py`, `make_sdt_report.py`, `make_time_report.py`
