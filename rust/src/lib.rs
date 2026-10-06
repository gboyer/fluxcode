// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Garry Boyer
//! Optional accelerator for fluxcode's encoder, without the GIL. The modules are named after the
//! Python ones they port, and this file holds only the bindings:
//!
//! - `format` (`_format.py`): constants, offsets and the header;
//! - `bitpacking` (`_bitpacking.py`): packing rows into the body, and the byte- and bit-plane
//!   layouts, on top of `planes` (the SIMD transposition, standard library only);
//! - `compress` (`_compress.py`): the compression policy: layout choice, flush points, framing and
//!   candidate choice.
//!
//! python-zstandard holds the GIL inside `flush(FLUSH_BLOCK)`, which is where the compression of
//! the buffered data happens, so flushed frames don't scale with threads there. `compress_unit`
//! builds a unit without a time axis from its rows, and `pack_unit` the unit of a body that Python
//! built (a unit with a time axis, or a splice); the unit bytes are the same as the Python path's
//! while both link the same libzstd (see `zstd_version`).

use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

mod bitpacking;
mod compress;
mod format;
mod planes;

use compress::Layout;
use format::{Rows, Shape};

/// The interface fluxcode/_compress.py checks (`RUST_INTERFACE_VERSION`) before using the
/// extension.
const INTERFACE_VERSION: u32 = 1;

/// A unit without a time axis from its rows: header plus zstd frame, as `_compress.compress` builds
/// it. `layout` is "byte", "bit", "heuristic" or "best" (`Effort.layout`).
#[pyfunction]
#[pyo3(signature = (
    block_flags, block_sizes, grid_params, value_anchors, residuals, codes, layout, flush,
    zstd_levels
))]
#[allow(clippy::too_many_arguments)]
fn compress_unit<'py>(
    py: Python<'py>,
    block_flags: PyReadonlyArray1<u8>,
    block_sizes: PyReadonlyArray1<i64>,
    grid_params: PyReadonlyArray1<i64>,
    value_anchors: PyReadonlyArray1<i64>,
    residuals: PyReadonlyArray1<i16>,
    codes: PyReadonlyArray1<u8>,
    layout: &str,
    flush: bool,
    zstd_levels: Vec<i32>,
) -> PyResult<Bound<'py, PyBytes>> {
    let rows = Rows::new(
        block_flags.as_slice()?,
        block_sizes.as_slice()?,
        grid_params.as_slice()?,
        value_anchors.as_slice()?,
        residuals.as_slice()?,
        codes.as_slice()?,
    )
    .map_err(PyValueError::new_err)?;
    let layout = Layout::parse(layout).map_err(PyValueError::new_err)?;
    let unit = py
        .detach(|| compress::compress_unit(&rows, layout, flush, &zstd_levels))
        .map_err(PyValueError::new_err)?;
    Ok(PyBytes::new(py, &unit))
}

/// The unit of an uncompressed body whose residuals are byte planes, with or without a time axis
/// (time_unit 0 for none): header plus zstd frame, as `_compress.pack` builds it. `num_octets` is
/// the octets (8 residual bytes each) of all blocks.
#[pyfunction]
#[pyo3(signature = (
    body, num_blocks, num_samples, num_octets, time_unit, layout, flush, zstd_levels
))]
#[allow(clippy::too_many_arguments)]
fn pack_unit<'py>(
    py: Python<'py>,
    body: PyReadonlyArray1<u8>,
    num_blocks: usize,
    num_samples: usize,
    num_octets: usize,
    time_unit: u8,
    layout: &str,
    flush: bool,
    zstd_levels: Vec<i32>,
) -> PyResult<Bound<'py, PyBytes>> {
    let shape = Shape {
        num_blocks,
        num_samples,
        num_octets,
        time_unit,
    };
    if !shape.fits_header() {
        return Err(PyValueError::new_err("unit shape doesn't fit the header"));
    }
    let layout = Layout::parse(layout).map_err(PyValueError::new_err)?;
    let body = body.as_slice()?;
    let unit = py
        .detach(|| compress::pack(&shape, body, layout, flush, &zstd_levels))
        .map_err(PyValueError::new_err)?;
    Ok(PyBytes::new(py, &unit))
}

/// Which bit-plane transposition runs: "neon", "sse2" or "scalar".
#[pyfunction]
fn simd_path() -> &'static str {
    planes::PATH
}

/// The version of the libzstd this extension links, as (major, minor, release): the unit bytes
/// match python-zstandard's only if its `ZSTD_VERSION` is the same.
#[pyfunction]
fn zstd_version() -> (u32, u32, u32) {
    let version = zstd_safe::version_number();
    (version / 10000, version / 100 % 100, version % 100)
}

#[pymodule]
fn fluxcode_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compress_unit, m)?)?;
    m.add_function(wrap_pyfunction!(pack_unit, m)?)?;
    m.add_function(wrap_pyfunction!(zstd_version, m)?)?;
    m.add_function(wrap_pyfunction!(simd_path, m)?)?;
    // fluxcode/_compress.py loads the extension only if its RUST_INTERFACE_VERSION is this
    m.add("INTERFACE_VERSION", INTERFACE_VERSION)?;
    // the policy constants, which fluxcode/_compress.py has too: tests/test_rust.py compares them
    m.add("FLUSH_MIN_DENSITY", compress::FLUSH_MIN_DENSITY)?;
    m.add("BYTE_PLANES_BIT", compress::BYTE_PLANES_BIT)?;
    m.add("BYTE_PLANES_MAX_SHARE", compress::BYTE_PLANES_MAX_SHARE)?;
    Ok(())
}
