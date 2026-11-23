// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Garry Boyer
//! PyO3 extension module accelerating fluxcode's block-group encoder without holding the GIL.
//!
//! `python-zstandard` holds the GIL during `flush(FLUSH_BLOCK)`, which prevents multi-threaded
//! scaling when zstd block flushing is enabled. This crate ports the hot packing and compression
//! pipeline to Rust and releases the GIL across all packing, transposition, and zstd calls:
//!
//! - `format`: wire-format constants, section offsets, and 8-byte header encoding;
//! - `bitpacking`: columnar body serialization and byte-plane/bit-plane conversion;
//! - `planes`: SIMD-accelerated (NEON / SSE2) and scalar bit-plane transposition;
//! - `compress`: layout selection policy, density-based zstd block flushing, and framing.
//!
//! Output bytes are bit-for-bit identical to the Python encoder when both link the same libzstd
//! version (see `zstd_version`).

use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

mod bitpacking;
mod compress;
mod format;
mod planes;

use compress::ResidualLayout;
use format::{BlockColumns, GroupShape};

/// ABI version checked by `fluxcode/_compress.py` (`RUST_INTERFACE_VERSION`) before enabling the
/// extension.
const INTERFACE_VERSION: u32 = 6;

/// Encodes and compresses a block group without a time axis directly from columnar NumPy arrays.
///
/// `layout` must be `"heuristic"` or `"best"`. Releases the GIL during body packing and zstd
/// compression. Raises `ValueError` on mismatched array lengths, invalid layout names, empty
/// `zstd_levels`, or libzstd errors.
#[pyfunction]
#[pyo3(signature = (
    block_flags, block_sizes, grid_params, grid_scales, value_anchors, residuals, codes, escapes,
    layout, flush, zstd_levels
))]
#[allow(clippy::too_many_arguments)]
fn compress_group<'py>(
    py: Python<'py>,
    block_flags: PyReadonlyArray1<u8>,
    block_sizes: PyReadonlyArray1<i64>,
    grid_params: PyReadonlyArray1<i64>,
    grid_scales: PyReadonlyArray1<i64>,
    value_anchors: PyReadonlyArray1<i64>,
    residuals: PyReadonlyArray1<i16>,
    codes: PyReadonlyArray1<u8>,
    escapes: PyReadonlyArray1<f64>,
    layout: &str,
    flush: bool,
    zstd_levels: Vec<i32>,
) -> PyResult<Bound<'py, PyBytes>> {
    // Validate NumPy slices and layout policy under the GIL, then release the GIL for packing
    // and zstd compression.
    let columns = BlockColumns::new(
        block_flags.as_slice()?,
        block_sizes.as_slice()?,
        grid_params.as_slice()?,
        grid_scales.as_slice()?,
        value_anchors.as_slice()?,
        residuals.as_slice()?,
        codes.as_slice()?,
        escapes.as_slice()?,
    )
    .map_err(PyValueError::new_err)?;
    let layout = ResidualLayout::parse(layout).map_err(PyValueError::new_err)?;
    let group = py
        .detach(|| compress::compress_group(&columns, layout, flush, &zstd_levels))
        .map_err(PyValueError::new_err)?;
    Ok(PyBytes::new(py, &group))
}

/// Packs and compresses a pre-built uncompressed byte-plane body (with or without a time axis)
/// into a block group (8-byte header + zstd frame).
///
/// `num_octets` is the total number of 8-sample groups (plane bytes per bit plane) across all
/// blocks; `time_unit` is `0` when no time axis is present. Releases the GIL during transposition
/// and compression. Raises `ValueError` if the shape overflows the header, `layout` is invalid,
/// `body` is too short, or libzstd fails.
#[pyfunction]
#[pyo3(signature = (
    body, num_blocks, num_samples, num_octets, time_unit, layout, flush, zstd_levels,
    group_time_step = false
))]
#[allow(clippy::too_many_arguments)]
fn pack_group<'py>(
    py: Python<'py>,
    body: PyReadonlyArray1<u8>,
    num_blocks: usize,
    num_samples: usize,
    num_octets: usize,
    time_unit: u8,
    layout: &str,
    flush: bool,
    zstd_levels: Vec<i32>,
    group_time_step: bool,
) -> PyResult<Bound<'py, PyBytes>> {
    // Validate header bounds and layout policy under the GIL, then detach to transpose/compress.
    let shape = GroupShape {
        num_blocks,
        num_samples,
        num_plane_bytes: num_octets,
        time_unit,
        group_time_step,
    };
    if !shape.fits_header() {
        return Err(PyValueError::new_err(
            "block group shape doesn't fit the header",
        ));
    }
    let layout = ResidualLayout::parse(layout).map_err(PyValueError::new_err)?;
    let body = body.as_slice()?;
    let group = py
        .detach(|| compress::pack(&shape, body, layout, flush, &zstd_levels))
        .map_err(PyValueError::new_err)?;
    Ok(PyBytes::new(py, &group))
}

/// Returns the active bit-plane transposition backend: `"neon"`, `"sse2"`, or `"scalar"`.
#[pyfunction]
fn simd_path() -> &'static str {
    planes::SIMD_BACKEND
}

/// Returns the linked libzstd version as `(major, minor, release)`.
///
/// Encoded block groups are bit-for-bit identical to `python-zstandard`'s output only when both
/// link the same libzstd version.
#[pyfunction]
fn zstd_version() -> (u32, u32, u32) {
    let version = zstd_safe::version_number();
    (version / 10000, version / 100 % 100, version % 100)
}

/// Registers the `fluxcode_rs` Python extension functions and policy constants.
#[pymodule]
fn fluxcode_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compress_group, m)?)?;
    m.add_function(wrap_pyfunction!(pack_group, m)?)?;
    m.add_function(wrap_pyfunction!(zstd_version, m)?)?;
    m.add_function(wrap_pyfunction!(simd_path, m)?)?;
    // Version gate checked by `fluxcode/_compress.py` before loading this extension.
    m.add("INTERFACE_VERSION", INTERFACE_VERSION)?;
    // Compression-policy constants mirrored in `fluxcode/_compress.py` and verified by tests.
    m.add("FLUSH_MIN_DENSITY", compress::FLUSH_MIN_DENSITY)?;
    m.add("BYTE_PLANES_BIT", compress::BYTE_PLANES_BIT)?;
    m.add("BYTE_PLANES_MAX_SHARE", compress::BYTE_PLANES_MAX_SHARE)?;
    Ok(())
}
