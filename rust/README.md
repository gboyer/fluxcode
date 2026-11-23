# fluxcode-rs

Optional accelerator for `fluxcode`'s encoder (`uv sync --extra rust` from a checkout; `pip install fluxcode[rust]` once it is published; or build here with
`maturin develop --release`). It replaces `_compress.compress_columns` for block groups without a time axis: layout choice,
body packing, flush points and the zstd frame in one call (`compress_group`) that doesn't hold the GIL; and,
for block groups with a time axis and for `update`, `_compress.pack_body` hands it the byte-plane body that Python built
(`pack_group`: layout choice, flush points and framing). python-zstandard
holds the GIL inside `flush(FLUSH_BLOCK)`, so block-flushed frames (efforts 5-9) don't scale with threads there:
8 threads encode about 2.5x faster with this extension (experimental/rust_port/README.md). Built as an
abi3 wheel, so one wheel per platform serves Python 3.10 and later.

> **Versioning policy:** `fluxcode` and `fluxcode-rs` are always developed and deployed in lockstep;
> no cross-version ABI compatibility is intended or promised. `INTERFACE_VERSION` is simply a guard
> against stale local builds (checked against `RUST_INTERFACE_VERSION` in `fluxcode/_compress.py`)
> and is bumped freely whenever the internal Python↔Rust boundary changes.

fluxcode works without it, and gives the same block groups: it falls back to Python if the module isn't importable,
and `FLUXCODE_RUST=0` turns it off. The block group bytes match the Python path's while both link the same libzstd
(`fluxcode_rs.zstd_version()` against `zstandard.ZSTD_VERSION`); otherwise the block groups differ but both decode.
