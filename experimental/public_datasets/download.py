# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Garry Boyer
"""Download and verify public real-world time-series datasets from datasets.json.

Run once:
    uv run python experimental/public_datasets/download.py

Verify already-downloaded files against SHA-256 checksums without downloading:
    uv run python experimental/public_datasets/download.py --verify

Set FLUXCODE_DATA_DIR to store datasets outside experimental/public_datasets/data/.
Already-downloaded files are skipped unless --force is passed.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

MANIFEST_PATH = Path(__file__).resolve().parent / "datasets.json"
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
USER_AGENT = "fluxcode-public-datasets/0.3"


def resolve_data_dir(override: Path | None = None) -> Path:
    if override is not None:
        return override
    env = os.environ.get("FLUXCODE_DATA_DIR")
    return Path(env) if env else DEFAULT_DATA_DIR


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path, expected_sha256: str | None = None, force: bool = False) -> bool:
    """Download url to dest and verify SHA-256. Returns True if downloaded, False if skipped."""
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as out:
            shutil.copyfileobj(resp, out)
        if expected_sha256:
            got = sha256_file(tmp)
            if got != expected_sha256:
                raise ValueError(f"SHA-256 mismatch: expected {expected_sha256[:12]}..., got {got[:12]}...")
        tmp.replace(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    return True


def extract_zip(zip_path: Path, force: bool = False) -> None:
    target_dir = zip_path.with_suffix("")
    marker = target_dir / ".extracted"
    if marker.exists() and not force:
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target_dir)
    marker.write_text("ok\n", encoding="utf-8")


def extract_gz(gz_path: Path, force: bool = False) -> None:
    out_path = gz_path.with_suffix("")
    if out_path.exists() and not force:
        return
    with gzip.open(gz_path, "rb") as fin, open(out_path, "wb") as fout:
        shutil.copyfileobj(fin, fout)


def verify_all(files: list[dict[str, Any]], data_dir: Path) -> int:
    """Verify every file in the manifest against its SHA-256. Returns failure count."""
    ok = missing = mismatched = 0
    total_bytes = 0
    for entry in files:
        rel = entry["path"]
        expected = entry.get("sha256")
        dest = data_dir / rel
        if not dest.exists():
            missing += 1
            print(f"[MISS] {rel:42s} not found")
            continue
        size = dest.stat().st_size
        total_bytes += size
        got = sha256_file(dest)
        if expected and got != expected:
            mismatched += 1
            print(f"[FAIL] {rel:42s} expected {expected[:12]}..., got {got[:12]}...")
        else:
            ok += 1
            print(f"[OK  ] {rel:42s} {size / 1e6:7.2f} MB  {got[:12]}")
    failures = missing + mismatched
    print(
        f"\nVerify: {ok}/{len(files)} OK, {missing} missing, {mismatched} mismatched "
        f"({total_bytes / 1e6:.1f} MB in {data_dir})"
    )
    return failures


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="Re-download and re-extract existing files.")
    ap.add_argument("--verify", action="store_true", help="Verify SHA-256 checksums of downloaded files without downloading.")
    ap.add_argument("--data-dir", type=Path, default=None, help="Override data directory (default: FLUXCODE_DATA_DIR or data/).")
    args = ap.parse_args()

    manifest = load_manifest()
    files: list[dict[str, Any]] = manifest["files"]
    data_dir = resolve_data_dir(args.data_dir)

    if args.verify:
        failures = verify_all(files, data_dir)
        sys.exit(1 if failures else 0)

    data_dir.mkdir(parents=True, exist_ok=True)
    downloaded = skipped = failed = 0
    total_bytes = 0
    for entry in files:
        rel = entry["path"]
        url = entry["url"]
        mode = entry.get("extract")
        expected = entry.get("sha256")
        dest = data_dir / rel
        try:
            did = fetch(url, dest, expected_sha256=expected, force=args.force)
            size = dest.stat().st_size
            total_bytes += size
            status = "GET " if did else "SKIP"
            if did:
                downloaded += 1
            else:
                skipped += 1
            print(f"[{status}] {rel:42s} {size / 1e6:7.2f} MB")
            if mode == "zip":
                extract_zip(dest, force=args.force or did)
            elif mode == "gz":
                extract_gz(dest, force=args.force or did)
        except Exception as exc:
            failed += 1
            print(f"[FAIL] {rel:42s} {exc}")

    print(
        f"\nDone: {downloaded} downloaded, {skipped} skipped, {failed} failed, "
        f"{total_bytes / 1e6:.1f} MB total in {data_dir}"
    )
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
