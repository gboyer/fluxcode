.PHONY: help init datasets public-bench public-report test test-public check

help: ## Show available targets
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z0-9_-]+:.*## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

init: ## Install locked root and experimental Python environments
	uv sync --locked
	cd experimental && uv sync --locked

datasets: ## Download and verify public real-world datasets (honors FLUXCODE_DATA_DIR)
	uv run python experimental/public_datasets/download.py

public-bench: ## Benchmark fluxcode on public datasets and write RESULTS.md + results.json
	uv run python experimental/public_datasets/analyze.py --verify \
		--out experimental/public_datasets/RESULTS.md \
		--json experimental/public_datasets/results.json

public-report: ## Render experimental/report/public.html and public-*.svg from results.json
	cd experimental && uv run python make_public_report.py

test: ## Run offline unit test suite (excludes public_data)
	uv run pytest

test-public: ## Run opt-in public-dataset verification tests
	uv run pytest -o addopts="" -m public_data

check: ## Run CI-equivalent checks (ruff, experimental smoke, mypy, ty, offline pytest)
	uv run ruff check
	uv run ruff check --isolated --select E9,F63,F7,F82 experimental
	uv run --locked --project experimental python experimental/smoke.py
	uv run mypy fluxcode fluxprune
	uv run ty check
	uv run pytest
