//! Numeric constants ported from `constants/voxel_constants.h`.

/// SDF quantization scale for 8-bit channels.
/// Source: constants/voxel_constants.h:34
pub const QUANTIZED_SDF_8_BITS_SCALE: f32 = 0.1;
/// Source: constants/voxel_constants.h:35
pub const QUANTIZED_SDF_8_BITS_SCALE_INV: f32 = 1.0 / 0.1;
/// Source: constants/voxel_constants.h:41
pub const QUANTIZED_SDF_16_BITS_SCALE: f32 = 0.002;
/// Source: constants/voxel_constants.h:42
pub const QUANTIZED_SDF_16_BITS_SCALE_INV: f32 = 1.0 / 0.002;

/// Value written by remove-mode point edits.
/// Source: constants/voxel_constants.h:51
pub const SDF_FAR_OUTSIDE: f32 = 100.0;
/// Source: constants/voxel_constants.h:52
pub const SDF_FAR_INSIDE: f32 = -100.0;
