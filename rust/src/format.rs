// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Garry Boyer
//! The unit format's constants, offsets and header, as in `fluxcode/_format.py`.

/// Block flags bit 3: the block has non-finite code planes.
pub const BLOCK_FLAG_NONFINITE: u8 = 0x08;

/// Unit header flags bit 0: the residual planes are byte planes.
const UNIT_FLAG_BYTE_PLANES: u8 = 1;
/// Position of the 3-bit time unit in the unit header flags (bits 1-3).
const UNIT_FLAG_TIME_UNIT_SHIFT: u8 = 1;
/// Largest time unit code the header's 3 bits hold.
const MAX_TIME_UNIT: u8 = 7;
const FORMAT_VERSION: u8 = 1;
pub const HEADER_BYTES: usize = 8;

pub const BYTES_PER_FLAGS: usize = 1;
pub const BYTES_PER_SIZE: usize = 2;
pub const BYTES_PER_PARAM: usize = 2;
pub const BYTES_PER_ANCHOR: usize = 8;
/// Bytes of the value columns (flags, size, grid parameter, anchor) per block.
pub const METADATA_BYTES_PER_BLOCK: usize =
    BYTES_PER_FLAGS + BYTES_PER_SIZE + BYTES_PER_PARAM + BYTES_PER_ANCHOR;
/// Bytes of the three 8-byte time columns of a unit with a time axis, per block.
pub const TIME_BYTES_PER_BLOCK: usize = 3 * 8;

/// Samples per group: each plane holds one byte per group.
pub const GROUP_SAMPLES: usize = 8;
/// Bit planes of the residuals (16 bits each), or byte planes (2 bytes each).
pub const BIT_PLANES: usize = 16;
/// Bytes of the residual planes per group.
pub const RESIDUAL_BYTES_PER_GROUP: usize = BIT_PLANES;
/// Bytes of the non-finite code planes (2 bits per sample) per group of a flagged block.
pub const CODE_BYTES_PER_GROUP: usize = 2;

/// Offset of the block_sizes field in the body.
pub fn block_sizes_start(num_blocks: usize) -> usize {
    BYTES_PER_FLAGS * num_blocks
}

/// Offset of the grid_params field in the body.
pub fn grid_params_start(num_blocks: usize) -> usize {
    (BYTES_PER_FLAGS + BYTES_PER_SIZE) * num_blocks
}

/// Offset of the value_anchor field in the body.
pub fn value_anchor_start(num_blocks: usize) -> usize {
    (BYTES_PER_FLAGS + BYTES_PER_SIZE + BYTES_PER_PARAM) * num_blocks
}

/// Offset of the residual planes: after the per-block columns (and the time columns).
pub fn residual_start(num_blocks: usize, has_time: bool) -> usize {
    (METADATA_BYTES_PER_BLOCK + if has_time { TIME_BYTES_PER_BLOCK } else { 0 }) * num_blocks
}

/// What the header records, and the groups of residual bytes (8 samples each) of all blocks.
pub struct Shape {
    pub num_blocks: usize,
    pub num_samples: usize,
    pub num_groups: usize,
    pub time_unit: u8,
}

impl Shape {
    pub fn has_time(&self) -> bool {
        self.time_unit != 0
    }

    /// Whether the counts fit the header's fields.
    pub fn fits_header(&self) -> bool {
        self.num_blocks <= u16::MAX as usize
            && self.num_samples <= u32::MAX as usize
            && self.time_unit <= MAX_TIME_UNIT
    }

    pub fn residual_start(&self) -> usize {
        residual_start(self.num_blocks, self.has_time())
    }

    /// The 8 header bytes of a unit (`_format.pack_header`).
    pub fn header(&self, byte_planes: bool) -> [u8; HEADER_BYTES] {
        let mut header = [0u8; HEADER_BYTES];
        header[0] = FORMAT_VERSION;
        header[1] = if byte_planes {
            UNIT_FLAG_BYTE_PLANES
        } else {
            0
        } | (self.time_unit << UNIT_FLAG_TIME_UNIT_SHIFT);
        header[2..4].copy_from_slice(&(self.num_blocks as u16).to_le_bytes());
        header[4..8].copy_from_slice(&(self.num_samples as u32).to_le_bytes());
        header
    }
}

/// A unit's rows and the offsets of each block in the samples, residual groups and code groups.
///
/// Each offsets vector has `num_blocks + 1` entries: entry `b` is block `b`'s offset, and the last
/// the total.
pub struct Rows<'a> {
    pub shape: Shape,
    pub block_flags: &'a [u8],
    pub block_sizes: &'a [i64],
    pub grid_params: &'a [i64],
    pub value_anchors: &'a [i64],
    pub residuals: &'a [i16],
    pub codes: &'a [u8],
    pub sample_offsets: Vec<usize>,
    pub group_offsets: Vec<usize>,
    pub code_offsets: Vec<usize>,
}

impl<'a> Rows<'a> {
    /// The rows of a unit without a time axis with their offsets (`_format._fill_layout`), or an
    /// error if the fields disagree.
    pub fn new(
        block_flags: &'a [u8],
        block_sizes: &'a [i64],
        grid_params: &'a [i64],
        value_anchors: &'a [i64],
        residuals: &'a [i16],
        codes: &'a [u8],
    ) -> Result<Self, String> {
        let num_blocks = block_flags.len();
        let mismatch = || "rows don't match the layout".to_string();
        if num_blocks > u16::MAX as usize
            || block_sizes.len() != num_blocks
            || grid_params.len() != num_blocks
            || value_anchors.len() != num_blocks
        {
            return Err(mismatch());
        }
        let mut sample_offsets = vec![0usize; num_blocks + 1];
        let mut group_offsets = vec![0usize; num_blocks + 1];
        let mut code_offsets = vec![0usize; num_blocks + 1];
        for block_idx in 0..num_blocks {
            if !(0..=u16::MAX as i64).contains(&block_sizes[block_idx]) {
                return Err(mismatch());
            }
            let size = block_sizes[block_idx] as usize;
            let groups = size.div_ceil(GROUP_SAMPLES);
            let has_codes = block_flags[block_idx] & BLOCK_FLAG_NONFINITE != 0;
            sample_offsets[block_idx + 1] = sample_offsets[block_idx] + size;
            group_offsets[block_idx + 1] = group_offsets[block_idx] + groups;
            code_offsets[block_idx + 1] =
                code_offsets[block_idx] + if has_codes { groups } else { 0 };
        }
        let num_samples = sample_offsets[num_blocks];
        if residuals.len() != num_samples || codes.len() != num_samples {
            return Err(mismatch());
        }
        let shape = Shape {
            num_blocks,
            num_samples,
            num_groups: group_offsets[num_blocks],
            time_unit: 0,
        };
        Ok(Rows {
            shape,
            block_flags,
            block_sizes,
            grid_params,
            value_anchors,
            residuals,
            codes,
            sample_offsets,
            group_offsets,
            code_offsets,
        })
    }
}
