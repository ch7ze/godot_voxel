//! Dense voxel storage ported from `storage/voxel_buffer.{h,cpp}`.
//!
//! Only the parts the hybrid mesher path needs are present. Serialization,
//! metadata, the memory pool and the Godot binding are deliberately absent.

use glam::IVec3;

use crate::channel::{
    default_raw_value, raw_voxel_to_real, real_to_raw_voxel, ChannelId, Depth, MAX_CHANNELS,
    MAX_SIZE,
};
use crate::math::Box3i;

/// Per-channel storage. A channel holds either a single default value or a
/// dense array, mirroring `COMPRESSION_UNIFORM` and `COMPRESSION_NONE`.
/// Source: storage/voxel_buffer.h:57-61, :172-191
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChannelData {
    /// No voxels allocated, every position reads back `value`.
    Uniform { value: u64 },
    /// Dense array in ZXY order, `depth.byte_count()` bytes per voxel.
    Dense { bytes: Vec<u8> },
}

/// One channel of a buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Channel {
    pub depth: Depth,
    pub data: ChannelData,
}

impl Channel {
    pub fn is_uniform(&self) -> bool {
        matches!(self.data, ChannelData::Uniform { .. })
    }
}

/// Per-channel depth configuration of a buffer.
/// Source: storage/voxel_format.h, reduced to the depth table
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VoxelFormat {
    pub depths: [Depth; MAX_CHANNELS],
    /// Channels treated as SDF beyond `ChannelId::Sdf`, as a bit mask.
    /// Source: storage/voxel_buffer.cpp:241-252
    pub channels_acting_as_sdf: u8,
}

impl Default for VoxelFormat {
    fn default() -> VoxelFormat {
        let mut depths = [Depth::D8; MAX_CHANNELS];
        for channel in ChannelId::ALL {
            depths[channel.index()] = channel.default_depth();
        }
        VoxelFormat {
            depths,
            channels_acting_as_sdf: 0,
        }
    }
}

impl VoxelFormat {
    pub fn depth(&self, channel: ChannelId) -> Depth {
        self.depths[channel.index()]
    }

    pub fn with_depth(mut self, channel: ChannelId, depth: Depth) -> VoxelFormat {
        self.depths[channel.index()] = depth;
        self
    }

    pub fn acts_as_sdf(&self, channel: ChannelId) -> bool {
        channel == ChannelId::Sdf || (self.channels_acting_as_sdf >> channel.index()) & 1 == 1
    }
}

/// Error cases of buffer construction and access.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferError {
    /// An edge length was zero or exceeded `MAX_SIZE`.
    InvalidSize(IVec3),
    /// A position was outside the buffer.
    PositionOutOfBounds(IVec3),
    /// A box was not fully inside the buffer.
    BoxOutOfBounds,
    /// The requested depth does not match the channel's depth.
    DepthMismatch { expected: Depth, got: Depth },
}

/// Dense voxel data, organized in channels of configurable bit depth.
/// Memory layout per channel is ZXY, which makes vertical access contiguous.
/// Source: storage/voxel_buffer.h:35-38, :362-364
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VoxelBuffer {
    size: IVec3,
    format: VoxelFormat,
    channels: [Channel; MAX_CHANNELS],
}

/// ZXY index of a position inside a volume of `size`.
/// Source: storage/voxel_buffer.h:362-364
#[inline]
pub fn zxy_index(x: u32, y: u32, z: u32, size: IVec3) -> usize {
    (y as usize) + (size.y as usize) * ((x as usize) + (size.x as usize) * (z as usize))
}

impl VoxelBuffer {
    /// Creates a buffer with the default format.
    pub fn new(size: IVec3) -> Result<VoxelBuffer, BufferError> {
        VoxelBuffer::with_format(size, VoxelFormat::default())
    }

    /// Creates a buffer with an explicit format. All channels start uniform,
    /// holding their default value.
    /// Source: storage/voxel_buffer.cpp:147-152, :228-258
    pub fn with_format(size: IVec3, format: VoxelFormat) -> Result<VoxelBuffer, BufferError> {
        let valid = size.x > 0
            && size.y > 0
            && size.z > 0
            && (size.x as u32) <= MAX_SIZE
            && (size.y as u32) <= MAX_SIZE
            && (size.z as u32) <= MAX_SIZE;
        if !valid {
            return Err(BufferError::InvalidSize(size));
        }

        let channels = core::array::from_fn(|index| {
            let channel = ChannelId::from_index(index).expect("index within MAX_CHANNELS");
            let depth = format.depths[index];
            // Channels opted into SDF get the air default instead of plain 0,
            // otherwise untouched voxels would sit exactly on the iso-surface.
            // Source: storage/voxel_buffer.cpp:241-252
            let value = if format.acts_as_sdf(channel) {
                crate::channel::default_sdf_raw_value(depth)
            } else {
                default_raw_value(channel, depth)
            };
            Channel {
                depth,
                data: ChannelData::Uniform { value },
            }
        });

        Ok(VoxelBuffer {
            size,
            format,
            channels,
        })
    }

    pub fn size(&self) -> IVec3 {
        self.size
    }

    pub fn format(&self) -> &VoxelFormat {
        &self.format
    }

    /// Source: storage/voxel_buffer.h:511-513
    pub fn volume(&self) -> usize {
        self.size.x as usize * self.size.y as usize * self.size.z as usize
    }

    pub fn box_of_whole_buffer(&self) -> Box3i {
        Box3i::from_size(self.size)
    }

    /// Source: storage/voxel_buffer.h:499-501
    pub fn is_position_valid(&self, pos: IVec3) -> bool {
        pos.x >= 0
            && pos.y >= 0
            && pos.z >= 0
            && pos.x < self.size.x
            && pos.y < self.size.y
            && pos.z < self.size.z
    }

    /// Source: storage/voxel_buffer.h:507-509
    pub fn is_box_valid(&self, b: Box3i) -> bool {
        self.box_of_whole_buffer().contains_box(b)
    }

    pub fn channel(&self, channel: ChannelId) -> &Channel {
        &self.channels[channel.index()]
    }

    pub fn depth(&self, channel: ChannelId) -> Depth {
        self.channels[channel.index()].depth
    }

    pub fn is_uniform(&self, channel: ChannelId) -> bool {
        self.channels[channel.index()].is_uniform()
    }

    /// Default value of a uniform channel, `None` when the channel is dense.
    pub fn uniform_value(&self, channel: ChannelId) -> Option<u64> {
        match self.channels[channel.index()].data {
            ChannelData::Uniform { value } => Some(value),
            ChannelData::Dense { .. } => None,
        }
    }

    /// ZXY index of a position in this buffer.
    #[inline]
    pub fn index(&self, pos: IVec3) -> usize {
        zxy_index(pos.x as u32, pos.y as u32, pos.z as u32, self.size)
    }

    /// Reads a raw voxel. Out-of-bounds positions yield 0, as in C++.
    /// Source: storage/voxel_buffer.cpp:295-328
    pub fn get_voxel(&self, pos: IVec3, channel: ChannelId) -> u64 {
        if !self.is_position_valid(pos) {
            return 0;
        }
        let ch = &self.channels[channel.index()];
        match &ch.data {
            ChannelData::Uniform { value } => *value,
            ChannelData::Dense { bytes } => read_raw(bytes, self.index(pos), ch.depth),
        }
    }

    /// Reads a voxel as a real value, decoding per the channel's depth.
    /// Source: storage/voxel_buffer.cpp, get_voxel_f
    pub fn get_voxel_f(&self, pos: IVec3, channel: ChannelId) -> f32 {
        raw_voxel_to_real(self.get_voxel(pos, channel), self.depth(channel))
    }

    /// Writes a raw voxel, allocating the channel on first differing write.
    /// Source: storage/voxel_buffer.cpp:330-380
    pub fn set_voxel(
        &mut self,
        value: u64,
        pos: IVec3,
        channel: ChannelId,
    ) -> Result<(), BufferError> {
        if !self.is_position_valid(pos) {
            return Err(BufferError::PositionOutOfBounds(pos));
        }
        let index = self.index(pos);
        let volume = self.volume();
        let ch = &mut self.channels[channel.index()];

        match &mut ch.data {
            ChannelData::Uniform { value: defval } => {
                if *defval != value {
                    let mut bytes = vec![0u8; volume * ch.depth.byte_count() as usize];
                    fill_raw(&mut bytes, *defval, ch.depth);
                    write_raw(&mut bytes, index, ch.depth, value);
                    ch.data = ChannelData::Dense { bytes };
                }
            }
            ChannelData::Dense { bytes } => write_raw(bytes, index, ch.depth, value),
        }
        Ok(())
    }

    /// Writes a voxel from a real value, encoding per the channel's depth.
    pub fn set_voxel_f(
        &mut self,
        value: f32,
        pos: IVec3,
        channel: ChannelId,
    ) -> Result<(), BufferError> {
        let raw = real_to_raw_voxel(value, self.depth(channel));
        self.set_voxel(raw, pos, channel)
    }

    /// Resets a channel to a uniform raw value, releasing its storage.
    /// Source: storage/voxel_buffer.cpp, clear_channel
    pub fn clear_channel(&mut self, channel: ChannelId, value: u64) {
        self.channels[channel.index()].data = ChannelData::Uniform { value };
    }

    /// Resets a channel to a uniform real value.
    pub fn clear_channel_f(&mut self, channel: ChannelId, value: f32) {
        let raw = real_to_raw_voxel(value, self.depth(channel));
        self.clear_channel(channel, raw);
    }

    /// Resets every channel to its default value.
    /// Source: storage/voxel_buffer.cpp:228-258
    pub fn clear(&mut self) {
        for channel in ChannelId::ALL {
            let depth = self.depth(channel);
            let value = if self.format.acts_as_sdf(channel) {
                crate::channel::default_sdf_raw_value(depth)
            } else {
                default_raw_value(channel, depth)
            };
            self.clear_channel(channel, value);
        }
    }

    /// Allocates a channel and fills it with `value`, making it dense.
    pub fn decompress_channel(&mut self, channel: ChannelId, value: u64) {
        let volume = self.volume();
        let ch = &mut self.channels[channel.index()];
        if ch.is_uniform() {
            let mut bytes = vec![0u8; volume * ch.depth.byte_count() as usize];
            fill_raw(&mut bytes, value, ch.depth);
            ch.data = ChannelData::Dense { bytes };
        }
    }

    /// Collapses a channel back to uniform when every voxel holds the same
    /// value. Returns true when the channel is uniform afterwards.
    /// Source: storage/voxel_buffer.cpp, compress_uniform_channels
    pub fn compress_channel_if_uniform(&mut self, channel: ChannelId) -> bool {
        let ch = &mut self.channels[channel.index()];
        let uniform_value = match &ch.data {
            ChannelData::Uniform { value } => Some(*value),
            ChannelData::Dense { bytes } => {
                let count = bytes.len() / ch.depth.byte_count() as usize;
                let first = read_raw(bytes, 0, ch.depth);
                let mut all_equal = true;
                for i in 1..count {
                    if read_raw(bytes, i, ch.depth) != first {
                        all_equal = false;
                        break;
                    }
                }
                if all_equal {
                    Some(first)
                } else {
                    None
                }
            }
        };
        match uniform_value {
            Some(value) => {
                ch.data = ChannelData::Uniform { value };
                true
            }
            None => false,
        }
    }

    /// Collapses every channel that holds a single value.
    pub fn compress_uniform_channels(&mut self) {
        for channel in ChannelId::ALL {
            self.compress_channel_if_uniform(channel);
        }
    }

    /// Read-only byte view of a dense channel, `None` while it is uniform.
    /// Source: storage/voxel_buffer.h:518-519
    pub fn channel_as_bytes(&self, channel: ChannelId) -> Option<&[u8]> {
        match &self.channels[channel.index()].data {
            ChannelData::Uniform { .. } => None,
            ChannelData::Dense { bytes } => Some(bytes),
        }
    }

    /// Typed read-only view of an 8-bit channel.
    pub fn channel_as_u8(&self, channel: ChannelId) -> Result<Option<&[u8]>, BufferError> {
        self.typed_view(channel, Depth::D8).map(|opt| opt)
    }

    fn typed_view(
        &self,
        channel: ChannelId,
        expected: Depth,
    ) -> Result<Option<&[u8]>, BufferError> {
        let depth = self.depth(channel);
        if depth != expected {
            return Err(BufferError::DepthMismatch {
                expected,
                got: depth,
            });
        }
        Ok(self.channel_as_bytes(channel))
    }

    /// Typed read-only view of a 16-bit channel.
    pub fn channel_as_u16(&self, channel: ChannelId) -> Result<Option<Vec<u16>>, BufferError> {
        let view = self.typed_view(channel, Depth::D16)?;
        Ok(view.map(|bytes| {
            bytes
                .chunks_exact(2)
                .map(|c| u16::from_ne_bytes([c[0], c[1]]))
                .collect()
        }))
    }

    /// Typed read-only view of a 32-bit channel.
    pub fn channel_as_u32(&self, channel: ChannelId) -> Result<Option<Vec<u32>>, BufferError> {
        let view = self.typed_view(channel, Depth::D32)?;
        Ok(view.map(|bytes| {
            bytes
                .chunks_exact(4)
                .map(|c| u32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        }))
    }

    /// Fills a box with a raw value.
    /// Source: storage/voxel_buffer.cpp:460-505
    pub fn fill_box(
        &mut self,
        b: Box3i,
        value: u64,
        channel: ChannelId,
    ) -> Result<(), BufferError> {
        let clipped = b.clipped(self.box_of_whole_buffer());
        if clipped.is_empty() {
            return Ok(());
        }
        if clipped == self.box_of_whole_buffer() {
            self.clear_channel(channel, value);
            return Ok(());
        }
        self.decompress_channel(channel, self.uniform_value(channel).unwrap_or(0));

        let depth = self.depth(channel);
        let size = self.size;
        let ch = &mut self.channels[channel.index()];
        let ChannelData::Dense { bytes } = &mut ch.data else {
            return Err(BufferError::BoxOutOfBounds);
        };

        let end = clipped.end();
        for z in clipped.position.z..end.z {
            for x in clipped.position.x..end.x {
                let base = zxy_index(x as u32, clipped.position.y as u32, z as u32, size);
                for i in 0..clipped.size.y as usize {
                    write_raw(bytes, base + i, depth, value);
                }
            }
        }
        Ok(())
    }

    /// Fills a box with a real value.
    pub fn fill_box_f(
        &mut self,
        b: Box3i,
        value: f32,
        channel: ChannelId,
    ) -> Result<(), BufferError> {
        let raw = real_to_raw_voxel(value, self.depth(channel));
        self.fill_box(b, raw, channel)
    }

    /// Copies a channel from `src` into this buffer at `dst_min`.
    /// Depths of the channel must match on both sides.
    pub fn copy_channel_from(
        &mut self,
        src: &VoxelBuffer,
        src_box: Box3i,
        dst_min: IVec3,
        channel: ChannelId,
    ) -> Result<(), BufferError> {
        let src_depth = src.depth(channel);
        let dst_depth = self.depth(channel);
        if src_depth != dst_depth {
            return Err(BufferError::DepthMismatch {
                expected: dst_depth,
                got: src_depth,
            });
        }
        let clipped = src_box.clipped(src.box_of_whole_buffer());
        if clipped.is_empty() {
            return Ok(());
        }
        let dst_box = Box3i::new(dst_min, clipped.size);
        if !self.is_box_valid(dst_box) {
            return Err(BufferError::BoxOutOfBounds);
        }

        // A uniform source over the whole destination collapses to a fill
        if let (Some(value), true) = (
            src.uniform_value(channel),
            dst_box == self.box_of_whole_buffer(),
        ) {
            self.clear_channel(channel, value);
            return Ok(());
        }

        let offset = dst_min - clipped.position;
        let mut pending: Vec<(IVec3, u64)> = Vec::with_capacity(clipped.volume() as usize);
        clipped.for_each_cell_zxy(|pos| {
            pending.push((pos + offset, src.get_voxel(pos, channel)));
        });
        for (pos, value) in pending {
            self.set_voxel(value, pos, channel)?;
        }
        Ok(())
    }
}

/// Reads one raw voxel out of a dense byte array.
#[inline]
fn read_raw(bytes: &[u8], index: usize, depth: Depth) -> u64 {
    match depth {
        Depth::D8 => bytes[index] as u64,
        Depth::D16 => {
            let o = index * 2;
            u16::from_ne_bytes([bytes[o], bytes[o + 1]]) as u64
        }
        Depth::D32 => {
            let o = index * 4;
            u32::from_ne_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]) as u64
        }
        Depth::D64 => {
            let o = index * 8;
            u64::from_ne_bytes([
                bytes[o],
                bytes[o + 1],
                bytes[o + 2],
                bytes[o + 3],
                bytes[o + 4],
                bytes[o + 5],
                bytes[o + 6],
                bytes[o + 7],
            ])
        }
    }
}

/// Writes one raw voxel into a dense byte array, truncating to the depth.
#[inline]
fn write_raw(bytes: &mut [u8], index: usize, depth: Depth, value: u64) {
    match depth {
        Depth::D8 => bytes[index] = value as u8,
        Depth::D16 => {
            let o = index * 2;
            bytes[o..o + 2].copy_from_slice(&(value as u16).to_ne_bytes());
        }
        Depth::D32 => {
            let o = index * 4;
            bytes[o..o + 4].copy_from_slice(&(value as u32).to_ne_bytes());
        }
        Depth::D64 => {
            let o = index * 8;
            bytes[o..o + 8].copy_from_slice(&value.to_ne_bytes());
        }
    }
}

/// Fills a dense byte array with a repeated raw value.
/// Source: storage/voxel_buffer.cpp:408-440
fn fill_raw(bytes: &mut [u8], value: u64, depth: Depth) {
    let count = bytes.len() / depth.byte_count() as usize;
    for i in 0..count {
        write_raw(bytes, i, depth, value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channel::default_sdf_value;

    fn buffer() -> VoxelBuffer {
        VoxelBuffer::new(IVec3::splat(8)).expect("valid size")
    }

    #[test]
    fn zero_and_oversized_dimensions_are_rejected() {
        assert!(VoxelBuffer::new(IVec3::new(0, 4, 4)).is_err());
        assert!(VoxelBuffer::new(IVec3::new(4, -1, 4)).is_err());
        assert!(VoxelBuffer::new(IVec3::new(MAX_SIZE as i32 + 1, 1, 1)).is_err());
    }

    #[test]
    fn zxy_index_increments_along_y() {
        let size = IVec3::new(5, 7, 3);
        assert_eq!(zxy_index(0, 0, 0, size), 0);
        assert_eq!(zxy_index(0, 1, 0, size), 1);
        assert_eq!(zxy_index(1, 0, 0, size), 7);
        assert_eq!(zxy_index(0, 0, 1, size), 35);
    }

    #[test]
    fn index_covers_the_volume_bijectively() {
        let b = VoxelBuffer::new(IVec3::new(3, 4, 5)).unwrap();
        let mut seen = vec![false; b.volume()];
        b.box_of_whole_buffer().for_each_cell_zxy(|pos| {
            let i = b.index(pos);
            assert!(!seen[i], "index {i} produced twice");
            seen[i] = true;
        });
        assert!(seen.into_iter().all(|v| v));
    }

    #[test]
    fn channels_start_uniform_with_cpp_defaults() {
        let b = buffer();
        assert!(b.is_uniform(ChannelId::Sdf));
        assert!(b.is_uniform(ChannelId::Type));
        assert_eq!(b.get_voxel(IVec3::ZERO, ChannelId::Type), 0);
        assert_eq!(
            b.get_voxel(IVec3::ZERO, ChannelId::Indices),
            crate::mixel4::DEFAULT_INDICES as u64
        );
        assert_eq!(
            b.get_voxel(IVec3::ZERO, ChannelId::Weights),
            crate::mixel4::DEFAULT_WEIGHTS as u64
        );
        // SDF defaults to "fully outside"
        assert!(b.get_voxel_f(IVec3::ZERO, ChannelId::Sdf) > 0.0);
        assert_eq!(
            b.get_voxel_f(IVec3::ZERO, ChannelId::Sdf),
            default_sdf_value(Depth::D16)
        );
    }

    #[test]
    fn writing_the_default_value_keeps_the_channel_uniform() {
        let mut b = buffer();
        let defval = b.uniform_value(ChannelId::Type).unwrap();
        b.set_voxel(defval, IVec3::splat(1), ChannelId::Type).unwrap();
        assert!(b.is_uniform(ChannelId::Type));
    }

    #[test]
    fn writing_a_new_value_allocates_and_preserves_neighbours() {
        let mut b = buffer();
        b.set_voxel(7, IVec3::new(2, 3, 4), ChannelId::Type).unwrap();
        assert!(!b.is_uniform(ChannelId::Type));
        assert_eq!(b.get_voxel(IVec3::new(2, 3, 4), ChannelId::Type), 7);
        assert_eq!(b.get_voxel(IVec3::new(2, 3, 5), ChannelId::Type), 0);
        assert_eq!(b.get_voxel(IVec3::ZERO, ChannelId::Type), 0);
    }

    #[test]
    fn out_of_bounds_access_is_reported_and_reads_zero() {
        let mut b = buffer();
        assert_eq!(b.get_voxel(IVec3::splat(99), ChannelId::Type), 0);
        assert_eq!(
            b.set_voxel(1, IVec3::splat(99), ChannelId::Type),
            Err(BufferError::PositionOutOfBounds(IVec3::splat(99)))
        );
    }

    #[test]
    fn sdf_write_round_trips_through_quantization() {
        let mut b = buffer();
        b.set_voxel_f(-0.25, IVec3::splat(2), ChannelId::Sdf).unwrap();
        let read = b.get_voxel_f(IVec3::splat(2), ChannelId::Sdf);
        // One 16-bit step is 500/32767, about 0.0153
        let step = crate::constants::QUANTIZED_SDF_16_BITS_SCALE_INV / 32767.0;
        assert!((read + 0.25).abs() <= step, "read {read}");
    }

    #[test]
    fn compression_collapses_a_uniformly_written_channel() {
        let mut b = buffer();
        b.set_voxel(5, IVec3::ZERO, ChannelId::Type).unwrap();
        assert!(!b.is_uniform(ChannelId::Type));
        b.fill_box(b.box_of_whole_buffer(), 5, ChannelId::Type).unwrap();
        assert!(b.compress_channel_if_uniform(ChannelId::Type));
        assert_eq!(b.uniform_value(ChannelId::Type), Some(5));
    }

    #[test]
    fn compression_keeps_a_mixed_channel_dense() {
        let mut b = buffer();
        b.set_voxel(5, IVec3::ZERO, ChannelId::Type).unwrap();
        assert!(!b.compress_channel_if_uniform(ChannelId::Type));
    }

    #[test]
    fn fill_box_writes_only_inside_the_box() {
        let mut b = buffer();
        let target = Box3i::new(IVec3::splat(2), IVec3::splat(3));
        b.fill_box(target, 9, ChannelId::Type).unwrap();
        target.for_each_cell_zxy(|pos| {
            assert_eq!(b.get_voxel(pos, ChannelId::Type), 9, "at {pos}");
        });
        assert_eq!(b.get_voxel(IVec3::splat(1), ChannelId::Type), 0);
        assert_eq!(b.get_voxel(IVec3::splat(5), ChannelId::Type), 0);
    }

    #[test]
    fn fill_box_clips_to_the_buffer() {
        let mut b = buffer();
        b.fill_box(Box3i::new(IVec3::splat(-4), IVec3::splat(4)), 3, ChannelId::Type)
            .unwrap();
        // The clipped region is empty, nothing was written
        assert_eq!(b.get_voxel(IVec3::ZERO, ChannelId::Type), 0);
        b.fill_box(Box3i::new(IVec3::splat(-2), IVec3::splat(4)), 3, ChannelId::Type)
            .unwrap();
        assert_eq!(b.get_voxel(IVec3::ZERO, ChannelId::Type), 3);
        assert_eq!(b.get_voxel(IVec3::splat(2), ChannelId::Type), 0);
    }

    #[test]
    fn typed_views_reject_a_wrong_depth() {
        let b = buffer();
        // TYPE defaults to 16 bit
        assert!(matches!(
            b.channel_as_u8(ChannelId::Type),
            Err(BufferError::DepthMismatch { .. })
        ));
        assert!(b.channel_as_u16(ChannelId::Type).is_ok());
    }

    #[test]
    fn typed_view_is_none_while_uniform_and_some_once_dense() {
        let mut b = buffer();
        assert_eq!(b.channel_as_u16(ChannelId::Type).unwrap(), None);
        b.set_voxel(1, IVec3::ZERO, ChannelId::Type).unwrap();
        let view = b.channel_as_u16(ChannelId::Type).unwrap().unwrap();
        assert_eq!(view.len(), b.volume());
        assert_eq!(view[b.index(IVec3::ZERO)], 1);
    }

    #[test]
    fn copy_channel_transfers_values_at_an_offset() {
        let mut src = VoxelBuffer::new(IVec3::splat(4)).unwrap();
        src.set_voxel(11, IVec3::ZERO, ChannelId::Type).unwrap();
        src.set_voxel(12, IVec3::new(1, 2, 3), ChannelId::Type).unwrap();

        let mut dst = buffer();
        dst.copy_channel_from(
            &src,
            src.box_of_whole_buffer(),
            IVec3::splat(2),
            ChannelId::Type,
        )
        .unwrap();

        assert_eq!(dst.get_voxel(IVec3::splat(2), ChannelId::Type), 11);
        assert_eq!(dst.get_voxel(IVec3::new(3, 4, 5), ChannelId::Type), 12);
        assert_eq!(dst.get_voxel(IVec3::ZERO, ChannelId::Type), 0);
    }

    #[test]
    fn copy_channel_rejects_a_destination_overflow() {
        let src = VoxelBuffer::new(IVec3::splat(4)).unwrap();
        let mut dst = buffer();
        assert_eq!(
            dst.copy_channel_from(
                &src,
                src.box_of_whole_buffer(),
                IVec3::splat(6),
                ChannelId::Type
            ),
            Err(BufferError::BoxOutOfBounds)
        );
    }

    #[test]
    fn copy_channel_rejects_mismatched_depths() {
        let src = VoxelBuffer::with_format(
            IVec3::splat(4),
            VoxelFormat::default().with_depth(ChannelId::Type, Depth::D8),
        )
        .unwrap();
        let mut dst = buffer();
        assert!(matches!(
            dst.copy_channel_from(&src, src.box_of_whole_buffer(), IVec3::ZERO, ChannelId::Type),
            Err(BufferError::DepthMismatch { .. })
        ));
    }

    #[test]
    fn sdf_opted_in_data_channel_defaults_to_air() {
        let format = VoxelFormat {
            channels_acting_as_sdf: 1 << ChannelId::Data5.index(),
            ..VoxelFormat::default()
        };
        let b = VoxelBuffer::with_format(IVec3::splat(4), format).unwrap();
        assert!(b.get_voxel_f(IVec3::ZERO, ChannelId::Data5) > 0.0);
    }

    #[test]
    fn clear_restores_the_defaults() {
        let mut b = buffer();
        b.set_voxel(3, IVec3::ZERO, ChannelId::Type).unwrap();
        b.clear();
        assert!(b.is_uniform(ChannelId::Type));
        assert_eq!(b.get_voxel(IVec3::ZERO, ChannelId::Type), 0);
    }
}
