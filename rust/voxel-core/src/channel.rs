//! Channel identifiers and bit depths ported from `storage/voxel_buffer.h:40-221`.

use crate::constants::{
    QUANTIZED_SDF_16_BITS_SCALE, QUANTIZED_SDF_16_BITS_SCALE_INV, QUANTIZED_SDF_8_BITS_SCALE,
    QUANTIZED_SDF_8_BITS_SCALE_INV,
};
use crate::funcs::{s16_to_snorm, s8_to_snorm, snorm_to_s16, snorm_to_s8};

/// Number of channels a buffer can hold.
/// Source: storage/voxel_buffer.h:49-50
pub const MAX_CHANNELS: usize = 8;

/// Mask selecting every channel.
/// Source: storage/voxel_buffer.h:55
pub const ALL_CHANNELS_MASK: u8 = 0xff;

/// Maximum edge length of a buffer.
/// Source: storage/voxel_buffer.h:170
pub const MAX_SIZE: u32 = 65535;

/// Channel ids, values match the C++ enum ordering.
/// Source: storage/voxel_buffer.h:40-51
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ChannelId {
    Type = 0,
    Sdf = 1,
    Color = 2,
    Indices = 3,
    Weights = 4,
    Data5 = 5,
    Data6 = 6,
    Data7 = 7,
}

impl ChannelId {
    /// All channels in enum order.
    pub const ALL: [ChannelId; MAX_CHANNELS] = [
        ChannelId::Type,
        ChannelId::Sdf,
        ChannelId::Color,
        ChannelId::Indices,
        ChannelId::Weights,
        ChannelId::Data5,
        ChannelId::Data6,
        ChannelId::Data7,
    ];

    pub fn index(self) -> usize {
        self as usize
    }

    pub fn from_index(index: usize) -> Option<ChannelId> {
        ChannelId::ALL.get(index).copied()
    }

    /// Source: storage/voxel_buffer.cpp, get_channel_name
    pub fn name(self) -> &'static str {
        match self {
            ChannelId::Type => "type",
            ChannelId::Sdf => "sdf",
            ChannelId::Color => "color",
            ChannelId::Indices => "indices",
            ChannelId::Weights => "weights",
            ChannelId::Data5 => "data5",
            ChannelId::Data6 => "data6",
            ChannelId::Data7 => "data7",
        }
    }

    /// Default depth per channel.
    /// Source: storage/voxel_buffer.h:161-167
    pub fn default_depth(self) -> Depth {
        match self {
            ChannelId::Type => Depth::D16,
            ChannelId::Sdf => Depth::D16,
            // 32-bit so the Mixel4 index word holds 4 slots x 8 bits = 256 materials
            ChannelId::Indices => Depth::D32,
            ChannelId::Weights => Depth::D16,
            _ => Depth::D8,
        }
    }
}

/// Bit depth of a channel. Byte count is `1 << discriminant`.
/// Source: storage/voxel_buffer.h:63-69
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum Depth {
    D8 = 0,
    D16 = 1,
    D32 = 2,
    D64 = 3,
}

impl Depth {
    pub const ALL: [Depth; 4] = [Depth::D8, Depth::D16, Depth::D32, Depth::D64];

    /// Source: storage/voxel_buffer.h:80-83
    pub fn byte_count(self) -> u32 {
        1 << (self as u32)
    }

    /// Source: storage/voxel_buffer.h:85-88
    pub fn bit_count(self) -> u32 {
        self.byte_count() << 3
    }

    /// Source: storage/voxel_buffer.h:90-104
    pub fn from_byte_count(size: usize) -> Option<Depth> {
        match size {
            1 => Some(Depth::D8),
            2 => Some(Depth::D16),
            4 => Some(Depth::D32),
            8 => Some(Depth::D64),
            _ => None,
        }
    }

    /// Largest raw value representable at this depth.
    pub fn max_raw_value(self) -> u64 {
        match self {
            Depth::D8 => u8::MAX as u64,
            Depth::D16 => u16::MAX as u64,
            Depth::D32 => u32::MAX as u64,
            Depth::D64 => u64::MAX,
        }
    }
}

/// Encodes a real value into the raw representation of a depth.
/// Depths below 32 bit are normalized and scaled, 32 and 64 bit are bit casts.
/// Source: storage/voxel_buffer.h:106-130
pub fn real_to_raw_voxel(value: f32, depth: Depth) -> u64 {
    match depth {
        Depth::D8 => snorm_to_s8(value * QUANTIZED_SDF_8_BITS_SCALE) as u8 as u64,
        Depth::D16 => snorm_to_s16(value * QUANTIZED_SDF_16_BITS_SCALE) as u16 as u64,
        Depth::D32 => value.to_bits() as u64,
        Depth::D64 => (value as f64).to_bits(),
    }
}

/// Decodes a raw value of a depth back into a real value.
/// Source: storage/voxel_buffer.h:132-159
pub fn raw_voxel_to_real(value: u64, depth: Depth) -> f32 {
    match depth {
        Depth::D8 => s8_to_snorm(value as u8 as i8) * QUANTIZED_SDF_8_BITS_SCALE_INV,
        Depth::D16 => s16_to_snorm(value as u16 as i16) * QUANTIZED_SDF_16_BITS_SCALE_INV,
        Depth::D32 => f32::from_bits(value as u32),
        Depth::D64 => f64::from_bits(value) as f32,
    }
}

/// Default SDF raw value, meaning "fully outside".
/// The 32-bit case deliberately returns `SDF_FAR_OUTSIDE` as an integer rather
/// than a float bit pattern, matching the C++ implementation.
/// 64-bit SDF is unsupported there and yields 0 here.
/// Source: storage/voxel_buffer.cpp:113-125
pub fn default_sdf_raw_value(depth: Depth) -> u64 {
    match depth {
        Depth::D8 => snorm_to_s8(1.0) as u8 as u64,
        Depth::D16 => snorm_to_s16(1.0) as u16 as u64,
        Depth::D32 => crate::constants::SDF_FAR_OUTSIDE as u64,
        Depth::D64 => 0,
    }
}

/// Source: storage/voxel_buffer.cpp, get_default_sdf_value
pub fn default_sdf_value(depth: Depth) -> f32 {
    raw_voxel_to_real(default_sdf_raw_value(depth), depth)
}

/// Default raw value of a channel when it was never written.
/// Source: storage/voxel_buffer.cpp, get_default_raw_value
pub fn default_raw_value(channel: ChannelId, depth: Depth) -> u64 {
    match channel {
        ChannelId::Sdf => default_sdf_raw_value(depth),
        ChannelId::Indices => crate::mixel4::default_indices_raw_value(depth),
        ChannelId::Weights => crate::mixel4::make_encoded_weights_for_single_texture() as u64,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn depth_byte_counts_match_cpp() {
        assert_eq!(Depth::D8.byte_count(), 1);
        assert_eq!(Depth::D16.byte_count(), 2);
        assert_eq!(Depth::D32.byte_count(), 4);
        assert_eq!(Depth::D64.byte_count(), 8);
        assert_eq!(Depth::D64.bit_count(), 64);
    }

    #[test]
    fn default_depths_match_cpp() {
        assert_eq!(ChannelId::Type.default_depth(), Depth::D16);
        assert_eq!(ChannelId::Sdf.default_depth(), Depth::D16);
        assert_eq!(ChannelId::Indices.default_depth(), Depth::D32);
        assert_eq!(ChannelId::Weights.default_depth(), Depth::D16);
        assert_eq!(ChannelId::Color.default_depth(), Depth::D8);
        assert_eq!(ChannelId::Data5.default_depth(), Depth::D8);
    }

    #[test]
    fn sdf_round_trip_within_quantization_error() {
        for depth in Depth::ALL {
            for value in [0.0f32, 0.5, -0.5, 1.0, -1.0] {
                let raw = real_to_raw_voxel(value, depth);
                let back = raw_voxel_to_real(raw, depth);
                // One quantization step is scale_inv / max_int, so 10/127 at
                // 8 bit and 500/32767 at 16 bit. 32 and 64 bit are bit casts.
                let tolerance = match depth {
                    Depth::D8 => QUANTIZED_SDF_8_BITS_SCALE_INV / 127.0,
                    Depth::D16 => QUANTIZED_SDF_16_BITS_SCALE_INV / 32767.0,
                    _ => 0.0,
                };
                assert!(
                    (back - value).abs() <= tolerance,
                    "depth {depth:?} value {value} became {back}"
                );
            }
        }
    }

    #[test]
    fn float_depths_are_bit_exact() {
        let raw = real_to_raw_voxel(123.456, Depth::D32);
        assert_eq!(raw_voxel_to_real(raw, Depth::D32), 123.456);
    }

    #[test]
    fn eight_bit_sdf_saturates_at_scale_inverse() {
        // 0.1 scale means the representable range is about [-10, 10]
        let raw = real_to_raw_voxel(50.0, Depth::D8);
        assert!((raw_voxel_to_real(raw, Depth::D8) - 10.0).abs() < 0.1);
    }
}
