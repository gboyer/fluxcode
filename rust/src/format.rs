// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Garry Boyer
//! The unit format's constants, offsets and header, as in `fluxcode/_format.py`.

pub const BLOCK_FLAG_NONFINITE: u8 = 0x08;
// BYTES_PER_FLAGS + BYTES_PER_SIZE + BYTES_PER_PARAM + BYTES_PER_ANCHOR
pub const METADATA_BYTES_PER_BLOCK: usize = 1 + 2 + 2 + 8;
// the three 8-byte time columns of a unit with a time axis
pub const TIME_BYTES_PER_BLOCK: usize = 3 * 8;
pub const HEADER_BYTES: usize = 8;
const FORMAT_VERSION: u8 = 1;
const UNIT_FLAG_BYTE_PLANES: u8 = 1;
const UNIT_FLAG_TIME_UNIT_SHIFT: u8 = 1;

/// Offset of the residual planes: after the per-block columns (and the time columns).
pub fn residual_start(n: usize, has_time: bool) -> usize {
    (METADATA_BYTES_PER_BLOCK + if has_time { TIME_BYTES_PER_BLOCK } else { 0 }) * n
}

/// What the header records, and the groups of residual bytes (8 samples each) of all blocks.
pub struct Shape {
    pub n: usize,
    pub num_samples: usize,
    pub ng: usize,
    pub time_unit: u8,
}

impl Shape {
    pub fn has_time(&self) -> bool {
        self.time_unit != 0
    }

    pub fn residual_start(&self) -> usize {
        residual_start(self.n, self.has_time())
    }

    /// The 8 header bytes of a unit (`_format.pack_header`).
    pub fn header(&self, byte_planes: bool) -> [u8; HEADER_BYTES] {
        let mut header = [0u8; HEADER_BYTES];
        header[0] = FORMAT_VERSION;
        header[1] = if byte_planes { UNIT_FLAG_BYTE_PLANES } else { 0 } | (self.time_unit << UNIT_FLAG_TIME_UNIT_SHIFT);
        header[2..4].copy_from_slice(&(self.n as u16).to_le_bytes());
        header[4..8].copy_from_slice(&(self.num_samples as u32).to_le_bytes());
        header
    }
}

/// A unit's rows and the offsets of each block in the samples, residual groups and code groups.
pub struct Rows<'a> {
    pub shape: Shape,
    pub flags: &'a [u8],
    pub sizes: &'a [i64],
    pub params: &'a [i64],
    pub anchors: &'a [i64],
    pub residuals: &'a [i16],
    pub codes: &'a [u8],
    pub so: Vec<usize>,
    pub go: Vec<usize>,
    pub co: Vec<usize>,
}

impl<'a> Rows<'a> {
    /// The rows of a unit without a time axis with their offsets (`_format._fill_layout`), or an
    /// error if the fields disagree.
    pub fn new(
        flags: &'a [u8], sizes: &'a [i64], params: &'a [i64], anchors: &'a [i64], residuals: &'a [i16],
        codes: &'a [u8],
    ) -> Result<Self, String> {
        let n = flags.len();
        let mismatch = || "rows don't match the layout".to_string();
        if n > u16::MAX as usize || sizes.len() != n || params.len() != n || anchors.len() != n {
            return Err(mismatch());
        }
        let (mut so, mut go, mut co) = (vec![0usize; n + 1], vec![0usize; n + 1], vec![0usize; n + 1]);
        for b in 0..n {
            if !(0..=u16::MAX as i64).contains(&sizes[b]) {
                return Err(mismatch());
            }
            let size = sizes[b] as usize;
            let groups = size.div_ceil(8);
            so[b + 1] = so[b] + size;
            go[b + 1] = go[b] + groups;
            co[b + 1] = co[b] + if flags[b] & BLOCK_FLAG_NONFINITE != 0 { groups } else { 0 };
        }
        if residuals.len() != so[n] || codes.len() != so[n] {
            return Err(mismatch());
        }
        let shape = Shape { n, num_samples: so[n], ng: go[n], time_unit: 0 };
        Ok(Rows { shape, flags, sizes, params, anchors, residuals, codes, so, go, co })
    }
}
