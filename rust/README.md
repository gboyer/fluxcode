# fluxcode-rs

Optional accelerator for `fluxcode`'s encoder (`uv sync --extra rust` from a checkout; `pip install fluxcode[rust]` once it is published; or build here with
`maturin develop --release`). It replaces `_compress.compress` for units without a time axis: layout choice,
body packing, flush points and the zstd frame in one call (`compress_unit`) that doesn't hold the GIL; and,
for units with a time axis and for `update`, `_compress.pack` hands it the byte-plane body that Python built
(`pack_unit`: layout choice, flush points and framing). python-zstandard
holds the GIL inside `flush(FLUSH_BLOCK)`, so block-flushed frames (efforts 5-9) don't scale with threads there:
8 threads encode about 2.5x faster with this extension (experimental/rust_port/README.md). Built as an
abi3 wheel, so one wheel per platform serves Python 3.10 and later.

fluxcode works without it, and gives the same units: it falls back to Python if the module isn't importable,
and `FLUXCODE_RUST=0` turns it off. The unit bytes match the Python path's while both link the same libzstd
(`fluxcode_rs.zstd_version()` against `zstandard.ZSTD_VERSION`); otherwise the units differ but both decode.
