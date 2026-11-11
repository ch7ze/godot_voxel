//! Voxel generator interface.
//!
//! Replaces the Godot resource hierarchy around `VoxelGenerator`
//! (`generators/voxel_generator.h:50-99`) with a plain trait. The biome, cave
//! and plot logic of the separate `rame` project is not part of this port, see
//! doc/source/bevy_port_plan.md section 4, P9.

use glam::IVec3;

use crate::buffer::VoxelBuffer;
use crate::channel::ChannelId;
use crate::math::Box3i;

/// Outcome of one block generation.
/// Source: generators/voxel_generator.h:55-61
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GenerateResult {
    /// When true, no lower LOD index yields more detail for this block, which
    /// lets the streaming layer load fewer blocks.
    pub max_lod_hint: bool,
}

/// Produces voxel data for a block. Implementations must be thread safe, the
/// mesher and the streaming layer call them from several threads.
/// Source: generators/voxel_generator.h:48-69
pub trait VoxelGenerator: Send + Sync {
    /// Fills `buffer` with the content of the block whose minimum corner lies
    /// at `origin_in_voxels`, at the given LOD index.
    fn generate_block(
        &self,
        buffer: &mut VoxelBuffer,
        origin_in_voxels: IVec3,
        lod: u32,
    ) -> GenerateResult;

    /// Channels the generator writes, as a bit mask over `ChannelId` indices.
    /// Source: generators/voxel_generator.h:26 of the heightmap subclass
    fn used_channels_mask(&self) -> u8;
}

/// Reference generator writing a height field into the SDF channel.
///
/// The SDF convention matches the C++ generators, `sdf = world_y - height`, so
/// negative means solid.
/// Source: meshers/hybrid/voxel_mesher_hybrid.cpp:458-459
#[derive(Debug, Clone, Copy)]
pub struct FlatGenerator {
    /// World height the surface sits at.
    pub height: f32,
    /// Scales the distance field, matching `iso_scale` of the C++ heightmap
    /// generator.
    /// Source: generators/simple/voxel_generator_heightmap.h:34-35
    pub iso_scale: f32,
}

impl Default for FlatGenerator {
    fn default() -> FlatGenerator {
        FlatGenerator {
            height: 0.0,
            iso_scale: 1.0,
        }
    }
}

impl VoxelGenerator for FlatGenerator {
    fn generate_block(
        &self,
        buffer: &mut VoxelBuffer,
        origin_in_voxels: IVec3,
        lod: u32,
    ) -> GenerateResult {
        let step = 1i32 << lod;
        let size = buffer.size();

        // Whether the surface crosses this block at all. The C++ heightmap
        // generator makes the same test against the height range's bounds.
        // Source: generators/simple/voxel_generator_heightmap.h:59-66
        let block_min_y = origin_in_voxels.y as f32;
        let block_max_y = (origin_in_voxels.y + size.y * step) as f32;
        let surface_outside_block = block_min_y > self.height || block_max_y < self.height;

        // Values are written per row even for a block the surface misses.
        // A uniform fill would make neighbouring blocks disagree on their
        // shared border row, which the mesher reads through the padding and
        // would turn into a seam.
        let mut writes = Vec::with_capacity(buffer.volume());
        Box3i::from_size(size).for_each_cell_zxy(|pos| {
            let world_y = (origin_in_voxels.y + pos.y * step) as f32;
            writes.push((pos, self.iso_scale * (world_y - self.height)));
        });
        for (pos, value) in writes {
            buffer.set_voxel_f(value, pos, ChannelId::Sdf).ok();
        }

        GenerateResult {
            // No finer LOD brings more detail where the surface is absent
            max_lod_hint: surface_outside_block,
        }
    }

    fn used_channels_mask(&self) -> u8 {
        1 << ChannelId::Sdf.index()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channel::Depth;
    use crate::buffer::VoxelFormat;

    fn buffer(size: i32) -> VoxelBuffer {
        // A 32-bit SDF channel keeps the test values exact
        let format = VoxelFormat::default().with_depth(ChannelId::Sdf, Depth::D32);
        VoxelBuffer::with_format(IVec3::splat(size), format).unwrap()
    }

    #[test]
    fn the_channel_mask_lists_only_the_sdf_channel() {
        let g = FlatGenerator::default();
        assert_eq!(g.used_channels_mask(), 1 << ChannelId::Sdf.index());
    }

    #[test]
    fn the_surface_sits_at_the_requested_height() {
        let g = FlatGenerator {
            height: 4.0,
            iso_scale: 1.0,
        };
        let mut b = buffer(8);
        let result = g.generate_block(&mut b, IVec3::ZERO, 0);
        assert!(!result.max_lod_hint);
        // Below the surface is solid
        assert!(b.get_voxel_f(IVec3::new(0, 0, 0), ChannelId::Sdf) < 0.0);
        // Exactly at the surface the field is zero
        assert_eq!(b.get_voxel_f(IVec3::new(0, 4, 0), ChannelId::Sdf), 0.0);
        // Above the surface is air
        assert!(b.get_voxel_f(IVec3::new(0, 7, 0), ChannelId::Sdf) > 0.0);
    }

    #[test]
    fn the_origin_offsets_the_surface() {
        let g = FlatGenerator {
            height: 20.0,
            iso_scale: 1.0,
        };
        let mut b = buffer(8);
        g.generate_block(&mut b, IVec3::new(0, 16, 0), 0);
        // World y 20 falls on local y 4
        assert_eq!(b.get_voxel_f(IVec3::new(0, 4, 0), ChannelId::Sdf), 0.0);
        assert!(b.get_voxel_f(IVec3::new(0, 0, 0), ChannelId::Sdf) < 0.0);
    }

    #[test]
    fn a_block_fully_above_the_surface_is_air_and_reports_the_lod_hint() {
        let g = FlatGenerator {
            height: 4.0,
            iso_scale: 1.0,
        };
        let mut b = buffer(8);
        let result = g.generate_block(&mut b, IVec3::new(0, 100, 0), 0);
        assert!(result.max_lod_hint);
        assert!(b.get_voxel_f(IVec3::ZERO, ChannelId::Sdf) > 0.0);
        assert!(b.get_voxel_f(IVec3::splat(7), ChannelId::Sdf) > 0.0);
    }

    #[test]
    fn a_block_fully_below_the_surface_is_solid_and_reports_the_lod_hint() {
        let g = FlatGenerator {
            height: 400.0,
            iso_scale: 1.0,
        };
        let mut b = buffer(8);
        let result = g.generate_block(&mut b, IVec3::new(0, -100, 0), 0);
        assert!(result.max_lod_hint);
        assert!(b.get_voxel_f(IVec3::ZERO, ChannelId::Sdf) < 0.0);
        assert!(b.get_voxel_f(IVec3::splat(7), ChannelId::Sdf) < 0.0);
    }

    #[test]
    fn a_block_crossing_the_surface_reports_no_lod_hint() {
        let g = FlatGenerator {
            height: 4.0,
            iso_scale: 1.0,
        };
        let mut b = buffer(8);
        let result = g.generate_block(&mut b, IVec3::ZERO, 0);
        assert!(!result.max_lod_hint);
    }

    #[test]
    fn adjacent_blocks_agree_on_their_shared_border_row() {
        // The mesher reads a neighbour's values through the padding, so the
        // same world row must carry the same value in both blocks
        let g = FlatGenerator {
            height: 4.0,
            iso_scale: 1.0,
        };
        let mut lower = buffer(8);
        let mut upper = buffer(8);
        g.generate_block(&mut lower, IVec3::ZERO, 0);
        g.generate_block(&mut upper, IVec3::new(0, 8, 0), 0);
        // World y 8 is local y 8 in the lower block, which lies outside it,
        // so the comparison uses world y 7 against local y -1 of the upper
        // block. Instead the blocks are made to overlap by one row.
        let mut overlapping = buffer(8);
        g.generate_block(&mut overlapping, IVec3::new(0, 7, 0), 0);
        assert_eq!(
            lower.get_voxel_f(IVec3::new(0, 7, 0), ChannelId::Sdf),
            overlapping.get_voxel_f(IVec3::new(0, 0, 0), ChannelId::Sdf)
        );
        // The far block stays consistent with the formula as well
        assert_eq!(
            upper.get_voxel_f(IVec3::new(0, 0, 0), ChannelId::Sdf),
            8.0 - 4.0
        );
    }

    #[test]
    fn the_lod_index_strides_the_sampling() {
        let g = FlatGenerator {
            height: 8.0,
            iso_scale: 1.0,
        };
        let mut b = buffer(8);
        g.generate_block(&mut b, IVec3::ZERO, 1);
        // At lod 1 each voxel spans two world units, so world y 8 falls on
        // local y 4
        assert_eq!(b.get_voxel_f(IVec3::new(0, 4, 0), ChannelId::Sdf), 0.0);
        assert_eq!(b.get_voxel_f(IVec3::new(0, 5, 0), ChannelId::Sdf), 2.0);
    }

    #[test]
    fn the_iso_scale_scales_the_field() {
        let g = FlatGenerator {
            height: 4.0,
            iso_scale: 0.5,
        };
        let mut b = buffer(8);
        g.generate_block(&mut b, IVec3::ZERO, 0);
        // Two units below the surface, scaled by 0.5
        assert_eq!(b.get_voxel_f(IVec3::new(0, 2, 0), ChannelId::Sdf), -1.0);
    }

    #[test]
    fn the_field_is_constant_across_x_and_z() {
        let g = FlatGenerator {
            height: 4.0,
            iso_scale: 1.0,
        };
        let mut b = buffer(8);
        g.generate_block(&mut b, IVec3::ZERO, 0);
        let reference = b.get_voxel_f(IVec3::new(0, 3, 0), ChannelId::Sdf);
        for z in 0..8 {
            for x in 0..8 {
                assert_eq!(
                    b.get_voxel_f(IVec3::new(x, 3, z), ChannelId::Sdf),
                    reference,
                    "at x {x} z {z}"
                );
            }
        }
    }

    #[test]
    fn the_trait_is_usable_through_a_reference() {
        // Confirms the trait object bounds hold, which the chunk layer needs
        let g: &dyn VoxelGenerator = &FlatGenerator::default();
        let mut b = buffer(4);
        g.generate_block(&mut b, IVec3::ZERO, 0);
        assert_ne!(g.used_channels_mask(), 0);
    }
}
