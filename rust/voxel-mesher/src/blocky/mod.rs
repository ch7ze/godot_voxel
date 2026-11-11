//! Blocky model meshing.
//!
//! Ported from the parts of `meshers/blocky/` the hybrid mesher consumes, see
//! doc/source/bevy_port_plan.md section 1.6.

pub mod baked_library;
pub mod fluids;
pub mod library_desc;

use glam::{IVec3, Vec2, Vec3};
use std::collections::BTreeMap;
use voxel_core::cube::{SIDE_COUNT, SIDE_NORMALS};
use voxel_core::{ChannelId, VoxelBuffer};

use baked_library::{is_face_visible, BakedLibrary, AIR_ID};
use fluids::{generate_fluid_model, BufferJumps, SliceTypeBuffer};

/// Padding the model path needs on every side, matching the blocky mesher.
/// Source: meshers/hybrid/voxel_mesher_hybrid.cpp:28
pub const MODEL_PADDING: i32 = 1;

/// Geometry accumulated for one material id.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SurfaceArrays {
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub uvs: Vec<Vec2>,
    pub indices: Vec<u32>,
}

impl SurfaceArrays {
    /// Appends geometry, offsetting positions and rebasing indices.
    ///
    /// Missing normals are filled with the zero vector, matching the C++ call
    /// that passes an empty normal array for side surfaces.
    /// Source: meshers/hybrid/voxel_mesher_hybrid.cpp:866-888
    pub fn append_geometry(
        &mut self,
        positions: &[Vec3],
        normals: &[Vec3],
        uvs: &[Vec2],
        indices: &[u32],
        offset: Vec3,
    ) {
        if positions.is_empty() {
            return;
        }
        let base = self.positions.len() as u32;

        self.positions
            .extend(positions.iter().map(|p| *p + offset));

        if normals.len() == positions.len() {
            self.normals.extend_from_slice(normals);
        } else {
            self.normals.resize(self.positions.len(), Vec3::ZERO);
        }

        if uvs.len() == positions.len() {
            self.uvs.extend_from_slice(uvs);
        } else {
            self.uvs.resize(self.positions.len(), Vec2::ZERO);
        }

        self.indices.extend(indices.iter().map(|i| i + base));
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

/// Model geometry grouped by material id, ordered so output is deterministic.
pub type ModelMeshAccumulator = BTreeMap<u32, SurfaceArrays>;

/// Decodes the TYPE channel of a block into linear model ids.
///
/// The hybrid mesher needs raw access because `generate_fluid_model` indexes
/// the buffer linearly for its nine-neighbour reads. A uniform channel yields
/// an empty vector, in which case fluids fall back to their static geometry,
/// matching the C++ fallback.
/// Source: meshers/hybrid/voxel_mesher_hybrid.cpp:771-783, :918-929
pub fn decode_type_channel(voxels: &VoxelBuffer) -> Vec<u64> {
    match voxels.uniform_value(ChannelId::Type) {
        Some(_) => Vec::new(),
        None => {
            let volume = voxels.volume();
            let mut out = Vec::with_capacity(volume);
            voxels
                .box_of_whole_buffer()
                .for_each_cell_zxy(|_| out.push(0));
            // Fill by index so the order matches the buffer layout exactly
            voxels.box_of_whole_buffer().for_each_cell_zxy(|pos| {
                out[voxels.index(pos)] = voxels.get_voxel(pos, ChannelId::Type);
            });
            out
        }
    }
}

/// Step sizes through a block's channel data per axis.
/// Source: meshers/hybrid/voxel_mesher_hybrid.cpp:777-779
pub fn buffer_jumps(block_size: IVec3) -> BufferJumps {
    BufferJumps {
        y: 1,
        x: block_size.y as isize,
        z: (block_size.y * block_size.x) as isize,
    }
}

/// Builds the blocky model geometry of one block, grouped by material id.
///
/// Positions are relative to the unpadded mesh origin. `lod_scale` is applied
/// to positions so a model fills a coarse-LOD voxel.
/// Source: meshers/hybrid/voxel_mesher_hybrid.cpp:745-955
pub fn build_model_surfaces(
    voxels: &VoxelBuffer,
    library: &BakedLibrary,
    lod_index: u32,
) -> ModelMeshAccumulator {
    let mut acc = ModelMeshAccumulator::new();

    // No models anywhere in this block
    if let Some(value) = voxels.uniform_value(ChannelId::Type) {
        if value == AIR_ID as u64 {
            return acc;
        }
    }

    let block_size = voxels.size();
    let min = IVec3::splat(MODEL_PADDING);
    let max = block_size - IVec3::splat(MODEL_PADDING);
    if min.x >= max.x || min.y >= max.y || min.z >= max.z {
        return acc;
    }

    let type_values = decode_type_channel(voxels);
    let type_buffer = SliceTypeBuffer {
        values: &type_values,
    };
    let jumps = buffer_jumps(block_size);

    for z in min.z..max.z {
        for x in min.x..max.x {
            for y in min.y..max.y {
                let pos = IVec3::new(x, y, z);
                let voxel_id = voxels.get_voxel(pos, ChannelId::Type);

                if voxel_id == AIR_ID as u64 {
                    continue;
                }
                let Some(baked_model) = library.model(voxel_id) else {
                    continue;
                };
                if baked_model.empty {
                    continue;
                }
                let model = &baked_model.model;

                // Position relative to the unpadded mesh origin
                let voxel_offset = (pos - min).as_vec3();

                // Neighbour side culling. The padding border carries the
                // neighbour chunk's values, so this stays correct across chunk
                // boundaries. `empty_sides_mask` marks sides the model never
                // renders, fluids mark +Y because their top is an inner
                // surface.
                // Source: meshers/hybrid/voxel_mesher_hybrid.cpp:812-835
                let mut visible_sides_mask = 0u8;
                for side in 0..SIDE_COUNT {
                    if model.empty_sides_mask & (1 << side) != 0 {
                        continue;
                    }
                    let npos = pos + SIDE_NORMALS[side];
                    let neighbor_id = voxels.get_voxel(npos, ChannelId::Type);
                    if is_face_visible(library, baked_model, neighbor_id, side) {
                        visible_sides_mask |= 1 << side;
                    }
                }

                // Fluid path, replacing the static geometry with interpolated
                // top and side surfaces. Needs raw type-buffer access.
                // Source: meshers/hybrid/voxel_mesher_hybrid.cpp:837-864
                if baked_model.is_fluid() && type_buffer.values.len() > 0 {
                    let voxel_index = voxels.index(pos) as isize;
                    let Some(fluid_out) = generate_fluid_model(
                        baked_model,
                        &type_buffer,
                        voxel_index,
                        jumps,
                        visible_sides_mask,
                        library,
                    ) else {
                        // Fully covered by same-fluid neighbours
                        continue;
                    };

                    let material_id = if fluid_out.top_covered {
                        // The model's own surface supplies the material id
                        model.surfaces[0].material_id
                    } else {
                        fluid_out.top_surface.material_id
                    };

                    let entry = acc.entry(material_id).or_default();
                    entry.append_geometry(
                        &fluid_out.top_surface.positions,
                        &fluid_out.top_surface.normals,
                        &fluid_out.top_surface.uvs,
                        &fluid_out.top_surface.indices,
                        voxel_offset,
                    );
                    for side in 0..SIDE_COUNT {
                        if visible_sides_mask & (1 << side) == 0 {
                            continue;
                        }
                        let side_surface = &fluid_out.sides[side][0];
                        entry.append_geometry(
                            &side_surface.positions,
                            &[],
                            &side_surface.uvs,
                            &side_surface.indices,
                            voxel_offset,
                        );
                    }
                    continue;
                }

                for s in 0..model.surface_count {
                    let surface = &model.surfaces[s];
                    let entry = acc.entry(surface.material_id).or_default();
                    entry.append_geometry(
                        &surface.positions,
                        &surface.normals,
                        &surface.uvs,
                        &surface.indices,
                        voxel_offset,
                    );
                    // Side surfaces share their parent surface's material id
                    for side in 0..SIDE_COUNT {
                        if visible_sides_mask & (1 << side) == 0 {
                            continue;
                        }
                        let side_surface = &model.sides_surfaces[side][s];
                        entry.append_geometry(
                            &side_surface.positions,
                            &[],
                            &side_surface.uvs,
                            &side_surface.indices,
                            voxel_offset,
                        );
                    }
                }
            }
        }
    }

    // LOD scaling, same approach as the blocky mesher
    // Source: meshers/hybrid/voxel_mesher_hybrid.cpp:932, :949-955
    let lod_scale = if lod_index > 0 {
        (1u32 << lod_index) as f32
    } else {
        1.0
    };
    if lod_scale != 1.0 {
        for sa in acc.values_mut() {
            for p in &mut sa.positions {
                *p *= lod_scale;
            }
        }
    }

    acc.retain(|_, sa| !sa.is_empty());
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use baked_library::{BakedModel, SideSurface, Surface};

    /// Unit cube side quad, four vertices and two triangles.
    fn side_quad(material_uv: f32) -> SideSurface {
        SideSurface {
            positions: vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
            uvs: vec![Vec2::splat(material_uv); 4],
            indices: vec![0, 2, 1, 0, 3, 2],
            tangents: vec![0.0; 16],
        }
    }

    /// Model whose six sides are quads, using one material.
    fn cube_model(material_id: u32) -> BakedModel {
        let mut m = BakedModel {
            empty: false,
            ..BakedModel::default()
        };
        m.model.surface_count = 1;
        m.model.surfaces[0] = Surface {
            material_id,
            ..Surface::default()
        };
        for side in 0..SIDE_COUNT {
            m.model.sides_surfaces[side][0] = side_quad(side as f32);
        }
        m
    }

    fn library_with(models: Vec<BakedModel>) -> BakedLibrary {
        BakedLibrary {
            side_pattern_count: 1,
            side_pattern_culling: vec![true],
            models,
            fluids: Vec::new(),
            indexed_materials_count: 1,
        }
    }

    /// Block of `size` voxels plus one padding voxel per side.
    fn buffer(size: i32) -> VoxelBuffer {
        VoxelBuffer::new(IVec3::splat(size + 2 * MODEL_PADDING)).unwrap()
    }

    #[test]
    fn an_air_only_block_produces_nothing() {
        let b = buffer(8);
        let lib = library_with(vec![BakedModel::default(), cube_model(0)]);
        let acc = build_model_surfaces(&b, &lib, 0);
        assert!(acc.is_empty());
    }

    #[test]
    fn a_single_voxel_emits_its_six_sides() {
        let mut b = buffer(4);
        let lib = library_with(vec![BakedModel::default(), cube_model(0)]);
        b.set_voxel(1, IVec3::splat(2), ChannelId::Type).unwrap();
        let acc = build_model_surfaces(&b, &lib, 0);
        assert_eq!(acc.len(), 1);
        let sa = &acc[&0];
        // Six quads, four vertices and six indices each
        assert_eq!(sa.positions.len(), 24);
        assert_eq!(sa.indices.len(), 36);
        assert_eq!(sa.normals.len(), sa.positions.len());
        assert_eq!(sa.uvs.len(), sa.positions.len());
    }

    #[test]
    fn indices_stay_within_the_vertex_range() {
        let mut b = buffer(4);
        let lib = library_with(vec![BakedModel::default(), cube_model(0)]);
        b.set_voxel(1, IVec3::splat(2), ChannelId::Type).unwrap();
        b.set_voxel(1, IVec3::new(2, 2, 3), ChannelId::Type).unwrap();
        let acc = build_model_surfaces(&b, &lib, 0);
        let sa = &acc[&0];
        let count = sa.positions.len() as u32;
        for i in &sa.indices {
            assert!(*i < count, "index {i} exceeds {count}");
        }
    }

    #[test]
    fn touching_voxels_cull_the_shared_sides() {
        let mut b = buffer(4);
        let lib = library_with(vec![BakedModel::default(), cube_model(0)]);
        b.set_voxel(1, IVec3::new(2, 2, 2), ChannelId::Type).unwrap();
        b.set_voxel(1, IVec3::new(3, 2, 2), ChannelId::Type).unwrap();
        let acc = build_model_surfaces(&b, &lib, 0);
        let sa = &acc[&0];
        // Two cubes would give 12 quads, the shared pair is culled
        assert_eq!(sa.positions.len(), 10 * 4);
    }

    #[test]
    fn positions_are_relative_to_the_unpadded_origin() {
        let mut b = buffer(4);
        let lib = library_with(vec![BakedModel::default(), cube_model(0)]);
        // Voxel at padded position 1 is the unpadded origin
        b.set_voxel(1, IVec3::splat(1), ChannelId::Type).unwrap();
        let acc = build_model_surfaces(&b, &lib, 0);
        let sa = &acc[&0];
        let min_x = sa.positions.iter().map(|p| p.x).fold(f32::MAX, f32::min);
        assert_eq!(min_x, 0.0);
    }

    #[test]
    fn material_ids_become_separate_groups() {
        let mut b = buffer(4);
        let lib = library_with(vec![
            BakedModel::default(),
            cube_model(0),
            cube_model(7),
        ]);
        b.set_voxel(1, IVec3::new(1, 1, 1), ChannelId::Type).unwrap();
        b.set_voxel(2, IVec3::new(3, 3, 3), ChannelId::Type).unwrap();
        let acc = build_model_surfaces(&b, &lib, 0);
        assert_eq!(acc.len(), 2);
        assert!(acc.contains_key(&0));
        assert!(acc.contains_key(&7));
    }

    #[test]
    fn lod_scaling_multiplies_positions() {
        let mut b = buffer(4);
        let lib = library_with(vec![BakedModel::default(), cube_model(0)]);
        b.set_voxel(1, IVec3::splat(3), ChannelId::Type).unwrap();
        let acc0 = build_model_surfaces(&b, &lib, 0);
        let acc2 = build_model_surfaces(&b, &lib, 2);
        let max0 = acc0[&0].positions.iter().map(|p| p.x).fold(0.0f32, f32::max);
        let max2 = acc2[&0].positions.iter().map(|p| p.x).fold(0.0f32, f32::max);
        assert_eq!(max2, max0 * 4.0);
    }

    #[test]
    fn empty_sides_mask_suppresses_those_sides() {
        let mut model = cube_model(0);
        // Mark the top side as never rendered
        model.model.empty_sides_mask = 1 << voxel_core::cube::side_axis::SIDE_POSITIVE_Y;
        let lib = library_with(vec![BakedModel::default(), model]);
        let mut b = buffer(4);
        b.set_voxel(1, IVec3::splat(2), ChannelId::Type).unwrap();
        let acc = build_model_surfaces(&b, &lib, 0);
        // Five quads instead of six
        assert_eq!(acc[&0].positions.len(), 5 * 4);
    }

    #[test]
    fn a_model_id_outside_the_library_is_skipped() {
        let mut b = buffer(4);
        let lib = library_with(vec![BakedModel::default(), cube_model(0)]);
        b.set_voxel(999, IVec3::splat(2), ChannelId::Type).unwrap();
        let acc = build_model_surfaces(&b, &lib, 0);
        assert!(acc.is_empty());
    }

    #[test]
    fn the_padding_border_is_not_meshed() {
        let mut b = buffer(4);
        let lib = library_with(vec![BakedModel::default(), cube_model(0)]);
        // Position 0 is inside the padding border
        b.set_voxel(1, IVec3::ZERO, ChannelId::Type).unwrap();
        let acc = build_model_surfaces(&b, &lib, 0);
        assert!(acc.is_empty());
    }

    #[test]
    fn a_block_smaller_than_its_padding_produces_nothing() {
        let b = VoxelBuffer::new(IVec3::splat(2)).unwrap();
        let lib = library_with(vec![BakedModel::default(), cube_model(0)]);
        let acc = build_model_surfaces(&b, &lib, 0);
        assert!(acc.is_empty());
    }

    #[test]
    fn decoding_a_uniform_type_channel_yields_no_raw_access() {
        let b = buffer(4);
        assert!(decode_type_channel(&b).is_empty());
    }

    #[test]
    fn decoding_a_dense_type_channel_follows_the_buffer_index() {
        let mut b = buffer(4);
        b.set_voxel(5, IVec3::new(1, 2, 3), ChannelId::Type).unwrap();
        let values = decode_type_channel(&b);
        assert_eq!(values.len(), b.volume());
        assert_eq!(values[b.index(IVec3::new(1, 2, 3))], 5);
    }

    #[test]
    fn buffer_jumps_match_the_zxy_layout() {
        let j = buffer_jumps(IVec3::new(5, 7, 3));
        assert_eq!(j.y, 1);
        assert_eq!(j.x, 7);
        assert_eq!(j.z, 35);
    }

    #[test]
    fn append_geometry_rebases_indices_per_call() {
        let mut sa = SurfaceArrays::default();
        let positions = vec![Vec3::ZERO, Vec3::X, Vec3::Y];
        sa.append_geometry(&positions, &[], &[], &[0, 1, 2], Vec3::ZERO);
        sa.append_geometry(&positions, &[], &[], &[0, 1, 2], Vec3::splat(5.0));
        assert_eq!(sa.indices, vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(sa.positions[3], Vec3::splat(5.0));
    }

    #[test]
    fn append_geometry_pads_missing_normals_and_uvs() {
        let mut sa = SurfaceArrays::default();
        let positions = vec![Vec3::ZERO, Vec3::X];
        sa.append_geometry(&positions, &[], &[], &[0, 1], Vec3::ZERO);
        assert_eq!(sa.normals.len(), 2);
        assert_eq!(sa.uvs.len(), 2);
        assert_eq!(sa.normals[0], Vec3::ZERO);
    }
}
