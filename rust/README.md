# fluxcode-rs

Optional accelerator for `fluxcode`'s encoder (`uv sync --extra rust` from a checkout; `pip install fluxcode[rust]` once it is published; or build here with
`maturin develop --release`). It replaces `_unit.compress` for units without a time axis: layout choice,
body packing, flush points and the zstd frame in one call that doesn't hold the GIL. python-zstandard
holds it inside `flush(FLUSH_BLOCK)`, so block-flushed frames (efforts 5-9) don't scale with threads there:
8 threads encode about 2.5x faster with this extension (experimental/rust_port/README.md). Built as an
abi3 wheel, so one wheel per platform serves Python 3.10 and later. `splice`/`update` and units with a time
axis still use python-zstandard.

fluxcode works without it, and gives the same units: it falls back to Python if the module isn't importable,
and `FLUXCODE_RUST=0` turns it off. The unit bytes match the Python path's while both link the same libzstd
(`fluxcode_rs.zstd_version()` against `zstandard.ZSTD_VERSION`); otherwise the units differ but both decode.
