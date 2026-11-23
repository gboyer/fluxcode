// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Garry Boyer
//! Uncompressed block-group wire format: constants, section offsets, and header encoding.
//!
//! A block group consists of an 8-byte little-endian header followed by a zstd-compressed body.
//! The uncompressed body is laid out in five contiguous sections:
//! 1. Per-block metadata columns (flags, block sizes, grid exponents, grid scales, block anchors),
//!    each multi-byte column stored byte-planed (all byte 0s, then all byte 1s, ...).
//! 2. Optional per-block time columns (3 × 8 bytes per block) when a time axis is present.
//! 3. Residual planes: zigzag-encoded 16-bit signed residuals, padded per block to whole 8-sample
//!    groups (`SAMPLES_PER_PLANE_BYTE`), stored either as 2 byte planes or 16 bit planes.
//! 4. 2-bit sample-code planes for blocks with `BLOCK_FLAG_SAMPLE_CODES` set (non-finite or
//!    escaped samples), stored as 2 bit planes of 1 byte per 8-sample group.
//! 5. Escaped IEEE 754 float64 values, stored byte-planed (all byte 0s, ..., all byte 7s).

/// Block flag bit 3: indicates the block has 2-bit sample-code planes (non-finite or escaped values).
pub const BLOCK_FLAG_SAMPLE_CODES: u8 = 0x08;

/// Group header flag bit 0: residual section uses 2 byte planes instead of 16 bit planes.
const GROUP_FLAG_BYTE_PLANES: u8 = 1;
/// Bit shift of the 3-bit time-unit field inside the group header flags byte (bits 1–3).
const GROUP_FLAG_TIME_UNIT_SHIFT: u8 = 1;
/// Group header flag bit 4: a uint64 `group_time_step` trailer is present (time axis only).
const GROUP_FLAG_GROUP_TIME_STEP: u8 = 0x10;
/// Maximum time-unit code representable in the header's 3-bit field.
const MAX_TIME_UNIT: u8 = 7;
/// Wire-format version written to byte 0 of every block-group header.
const FORMAT_VERSION: u8 = 1;
/// Byte length of the uncompressed block-group header prepended before the zstd frame.
pub const HEADER_BYTES: usize = 8;

/// Byte width of each block's flags entry.
pub const BYTES_PER_FLAGS: usize = 1;
/// Byte width of each block's sample-count entry (`u16`).
pub const BYTES_PER_SIZE: usize = 2;
/// Byte width of each block's grid-exponent entry (`i16`).
pub const BYTES_PER_EXPONENT: usize = 2;
/// Byte width of each block's grid-scale entry (`i32`).
pub const BYTES_PER_SCALE: usize = 4;
/// Byte width of each block's anchor value (`i64`).
pub const BYTES_PER_ANCHOR: usize = 8;
/// Total metadata bytes per block across flags, sizes, grid exponents, grid scales, and anchors.
pub const METADATA_BYTES_PER_BLOCK: usize =
    BYTES_PER_FLAGS + BYTES_PER_SIZE + BYTES_PER_EXPONENT + BYTES_PER_SCALE + BYTES_PER_ANCHOR;
/// Total time-column bytes per block (three 8-byte columns) when a time axis is present.
pub const TIME_BYTES_PER_BLOCK: usize = 3 * 8;

/// Number of samples grouped into a single byte of each bit plane (8 samples → 1 plane byte).
pub const SAMPLES_PER_PLANE_BYTE: usize = 8;
/// Number of bit planes representing 16-bit zigzag-encoded residuals.
pub const BIT_PLANES: usize = 16;
/// Residual bytes per 8-sample group (16 bit planes × 1 byte, or 2 byte planes × 8 bytes).
pub const RESIDUAL_BYTES_PER_PLANE_BYTE: usize = BIT_PLANES;
/// Sample-code bytes per 8-sample group for a flagged block (2 bit planes × 1 byte).
pub const CODE_BYTES_PER_PLANE_BYTE: usize = 2;
/// Byte width of each escaped IEEE 754 float64 value in the escaped-values trailer.
pub const BYTES_PER_ESCAPED_VALUE: usize = 8;

/// Returns the byte offset of the `block_sizes` column in an uncompressed body.
pub fn block_sizes_start(num_blocks: usize) -> usize {
    BYTES_PER_FLAGS * num_blocks
}

/// Returns the byte offset of the `grid_exponents` column in an uncompressed body.
pub fn grid_exponents_start(num_blocks: usize) -> usize {
    (BYTES_PER_FLAGS + BYTES_PER_SIZE) * num_blocks
}

/// Returns the byte offset of the `grid_scales` column in an uncompressed body.
pub fn grid_scales_start(num_blocks: usize) -> usize {
    (BYTES_PER_FLAGS + BYTES_PER_SIZE + BYTES_PER_EXPONENT) * num_blocks
}

/// Returns the byte offset of the `block_anchors` column in an uncompressed body.
pub fn block_anchors_start(num_blocks: usize) -> usize {
    (BYTES_PER_FLAGS + BYTES_PER_SIZE + BYTES_PER_EXPONENT + BYTES_PER_SCALE) * num_blocks
}

/// Returns the byte offset where the residual planes begin, immediately after per-block metadata
/// (and optional per-block time columns).
pub fn residual_start(num_blocks: usize, has_time: bool) -> usize {
    (METADATA_BYTES_PER_BLOCK + if has_time { TIME_BYTES_PER_BLOCK } else { 0 }) * num_blocks
}

/// Dimensions and header flags describing a block group.
pub struct GroupShape {
    /// Number of blocks in the group (must fit in `u16` for header encoding).
    pub num_blocks: usize,
    /// Total number of samples across all blocks (must fit in `u32` for header encoding).
    pub num_samples: usize,
    /// Total number of 8-sample groups across all blocks; each bit plane is `num_plane_bytes` long.
    pub num_plane_bytes: usize,
    /// Encoded time-unit identifier (`0` means no time axis; `1..=7` identify concrete units).
    pub time_unit: u8,
    /// Whether a uint64 `group_time_step` trailer is present (valid only when `time_unit != 0`).
    pub group_time_step: bool,
}

impl GroupShape {
    /// Returns `true` if this block group carries a time axis.
    pub fn has_time(&self) -> bool {
        self.time_unit != 0
    }

    /// Returns `true` if all counts and flags satisfy the 8-byte header's field-width limits.
    pub fn fits_header(&self) -> bool {
        self.num_blocks <= u16::MAX as usize
            && self.num_samples <= u32::MAX as usize
            && self.time_unit <= MAX_TIME_UNIT
            && (!self.group_time_step || self.has_time())
    }

    /// Returns the byte offset of the residual planes within the uncompressed body.
    pub fn residual_start(&self) -> usize {
        residual_start(self.num_blocks, self.has_time())
    }

    /// Encodes the 8-byte little-endian block-group header for the chosen residual layout.
    ///
    /// Precondition: `self.fits_header()` must be `true`.
    pub fn header(&self, byte_planes: bool) -> [u8; HEADER_BYTES] {
        // Pack version, layout/time flags, uint16 block count, and uint32 sample count into 8 bytes.
        let mut header = [0u8; HEADER_BYTES];
        header[0] = FORMAT_VERSION;
        // Combine bit 0 (byte-plane layout), bits 1-3 (time unit), and bit 4 (group time step).
        header[1] = if byte_planes {
            GROUP_FLAG_BYTE_PLANES
        } else {
            0
        } | (self.time_unit << GROUP_FLAG_TIME_UNIT_SHIFT)
            | if self.group_time_step {
                GROUP_FLAG_GROUP_TIME_STEP
            } else {
                0
            };
        header[2..4].copy_from_slice(&(self.num_blocks as u16).to_le_bytes());
        header[4..8].copy_from_slice(&(self.num_samples as u32).to_le_bytes());
        header
    }
}

/// Validated columnar inputs for a block group without a time axis, together with prefix-sum
/// offsets for each block.
///
/// Each offset vector has length `num_blocks + 1`: entry `b` is block `b`'s starting index, and
/// entry `num_blocks` is the grand total across all blocks.
pub struct BlockColumns<'a> {
    /// Group dimensions derived from the validated columns.
    pub shape: GroupShape,
    /// Per-block bitmask flags (`num_blocks` entries).
    pub block_flags: &'a [u8],
    /// Per-block sample counts (`num_blocks` entries, each in `0..=u16::MAX`).
    pub block_sizes: &'a [i64],
    /// Per-block base-10/base-2 grid exponents (`num_blocks` entries).
    pub grid_exponents: &'a [i64],
    /// Per-block quantization grid scales (`num_blocks` entries).
    pub grid_scales: &'a [i64],
    /// Per-block integer anchor values (`num_blocks` entries).
    pub block_anchors: &'a [i64],
    /// Concatenated signed 16-bit residuals across all blocks (`num_samples` entries).
    pub residuals: &'a [i16],
    /// Concatenated 2-bit sample codes across all blocks (`num_samples` entries).
    pub sample_codes: &'a [u8],
    /// Escaped float64 values in block order, appended to the body trailer.
    pub escaped_values: &'a [f64],
    /// Prefix sums of `block_sizes`, indexing into `residuals` and `sample_codes`.
    pub sample_offsets: Vec<usize>,
    /// Prefix sums of per-block 8-sample group counts, indexing into each residual plane.
    pub plane_byte_offsets: Vec<usize>,
    /// Prefix sums of 8-sample group counts for blocks with `BLOCK_FLAG_SAMPLE_CODES` set.
    pub code_offsets: Vec<usize>,
}

impl<'a> BlockColumns<'a> {
    /// Validates columnar block-group inputs (without a time axis) and computes per-block prefix
    /// offsets.
    ///
    /// Returns an `Err` if slice lengths disagree, `num_blocks` exceeds `u16::MAX`, or any block
    /// size is outside `0..=u16::MAX`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        block_flags: &'a [u8],
        block_sizes: &'a [i64],
        grid_exponents: &'a [i64],
        grid_scales: &'a [i64],
        block_anchors: &'a [i64],
        residuals: &'a [i16],
        sample_codes: &'a [u8],
        escaped_values: &'a [f64],
    ) -> Result<Self, String> {
        // Validate per-block column lengths, accumulate prefix offsets for samples, residual
        // 8-sample groups, and coded-block groups, then verify per-sample array lengths match.
        let num_blocks = block_flags.len();
        let mismatch = || "rows don't match the layout".to_string();

        // Reject block counts that overflow the header or mismatched per-block metadata columns.
        if num_blocks > u16::MAX as usize
            || block_sizes.len() != num_blocks
            || grid_exponents.len() != num_blocks
            || grid_scales.len() != num_blocks
            || block_anchors.len() != num_blocks
        {
            return Err(mismatch());
        }

        let mut sample_offsets = vec![0usize; num_blocks + 1];
        let mut plane_byte_offsets = vec![0usize; num_blocks + 1];
        let mut code_offsets = vec![0usize; num_blocks + 1];

        for block_idx in 0..num_blocks {
            if !(0..=u16::MAX as i64).contains(&block_sizes[block_idx]) {
                return Err(mismatch());
            }
            let size = block_sizes[block_idx] as usize;
            // Each block's residuals are padded to whole 8-sample groups (1 plane byte each).
            let plane_bytes = size.div_ceil(SAMPLES_PER_PLANE_BYTE);
            let has_codes = block_flags[block_idx] & BLOCK_FLAG_SAMPLE_CODES != 0;
            sample_offsets[block_idx + 1] = sample_offsets[block_idx] + size;
            plane_byte_offsets[block_idx + 1] = plane_byte_offsets[block_idx] + plane_bytes;
            code_offsets[block_idx + 1] =
                code_offsets[block_idx] + if has_codes { plane_bytes } else { 0 };
        }

        let num_samples = sample_offsets[num_blocks];
        if residuals.len() != num_samples || sample_codes.len() != num_samples {
            return Err(mismatch());
        }

        let shape = GroupShape {
            num_blocks,
            num_samples,
            num_plane_bytes: plane_byte_offsets[num_blocks],
            time_unit: 0,
            group_time_step: false,
        };
        Ok(BlockColumns {
            shape,
            block_flags,
            block_sizes,
            grid_exponents,
            grid_scales,
            block_anchors,
            residuals,
            sample_codes,
            escaped_values,
            sample_offsets,
            plane_byte_offsets,
            code_offsets,
        })
    }
}
