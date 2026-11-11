//! Hybrid blocky and SDF mesher.
//!
//! Ported from `meshers/hybrid/voxel_mesher_hybrid.{h,cpp}`. The build order
//! and its stated reasons are preserved, see
//! doc/source/bevy_port_plan.md section 1.5.

pub mod output;

use glam::{IVec3, Vec2};
use voxel_core::{ChannelId, VoxelBuffer};

use crate::blocky::baked_library::{BakedLibrary, AIR_ID};
use crate::blocky::{build_model_surfaces, MODEL_PADDING};
use crate::transvoxel::materials::TexturingMode;
use crate::transvoxel::{self as tv, TransvoxelParams};

use output::{
    CollisionSurface, LodAttrib, MeshOutput, Surface, MODEL_MATERIAL_INDEX_BASE,
    TERRAIN_MATERIAL_INDEX, WATER_MATERIAL_INDEX,
};

/// Configuration of the hybrid mesher.
#[derive(Debug, Clone)]
pub struct HybridParams {
    /// Texturing mode forwarded to the terrain Transvoxel pass.
    /// Source: meshers/hybrid/voxel_mesher_hybrid.h:105-111
    pub texturing_mode: TexturingMode,
    /// Excludes air voxels from Mixel4 aggregation, which keeps the
    /// generator's default indices from bleeding into edits.
    /// Source: meshers/hybrid/voxel_mesher_hybrid.h:113-118
    pub textures_ignore_air_voxels: bool,
    /// Flattens the terrain to the bottom edge of voxels holding a model, so
    /// models sit flush on the ground.
    /// Source: meshers/hybrid/voxel_mesher_hybrid.h:100-103
    pub flatten_terrain_under_models: bool,
    /// Runs the fluid iso-surface pass on the DATA5 channel.
    /// Source: meshers/hybrid/voxel_mesher_hybrid.h:78-81
    pub build_fluid_surfaces: bool,
    pub lod_index: u32,
    /// Bit mask of sides that get a transition mesh.
    pub transition_mask: u8,
    pub edge_clamp_margin: f32,
    /// Requests collision geometry.
    pub collision_hint: bool,
}

impl Default for HybridParams {
    fn default() -> HybridParams {
        HybridParams {
            texturing_mode: TexturingMode::None,
            textures_ignore_air_voxels: false,
            // Source: meshers/hybrid/voxel_mesher_hybrid.h:174
            flatten_terrain_under_models: true,
            build_fluid_surfaces: true,
            lod_index: 0,
            transition_mask: 0,
            edge_clamp_margin: 0.0,
            collision_hint: true,
        }
    }
}

impl HybridParams {
    fn texturing_enabled(&self) -> bool {
        self.texturing_mode != TexturingMode::None
    }

    fn terrain_transvoxel_params(&self) -> TransvoxelParams {
        TransvoxelParams {
            sdf_channel: ChannelId::Sdf,
            lod_index: self.lod_index,
            texturing_mode: self.texturing_mode,
            edge_clamp_margin: self.edge_clamp_margin,
            textures_ignore_air_voxels: self.textures_ignore_air_voxels,
        }
    }

    /// The fluid pass always runs without texturing, fluid surfaces carry no
    /// Mixel4 material weights.
    /// Source: meshers/hybrid/voxel_mesher_hybrid.h:155-158
    fn fluid_transvoxel_params(&self) -> TransvoxelParams {
        TransvoxelParams {
            sdf_channel: ChannelId::Data5,
            lod_index: self.lod_index,
            texturing_mode: TexturingMode::None,
            edge_clamp_margin: self.edge_clamp_margin,
            textures_ignore_air_voxels: false,
        }
    }
}

/// Channels the mesher reads, as a bit mask over `ChannelId` indices.
/// Source: meshers/hybrid/voxel_mesher_hybrid.cpp, get_used_channels_mask
pub fn used_channels_mask(params: &HybridParams) -> u8 {
    let mut mask = (1 << ChannelId::Sdf.index()) | (1 << ChannelId::Type.index());
    if params.texturing_enabled() {
        mask |= (1 << ChannelId::Indices.index()) | (1 << ChannelId::Weights.index());
    }
    if params.build_fluid_surfaces {
        mask |= 1 << ChannelId::Data5.index();
    }
    mask as u8
}

/// Padding the mesher needs towards the negative axes.
/// Source: meshers/transvoxel/transvoxel.h:16, meshers/voxel_mesher.h:91-95
pub fn minimum_padding() -> i32 {
    tv::MIN_PADDING.max(MODEL_PADDING)
}

/// Padding the mesher needs towards the positive axes.
/// Source: meshers/transvoxel/transvoxel.h:18
pub fn maximum_padding() -> i32 {
    tv::MAX_PADDING.max(MODEL_PADDING)
}

/// Builds the SDF the smooth surface is meshed from, flattened under models.
///
/// Returns `None` when no flattening applies, in which case the original
/// buffer is used. Two passes run, the first collecting a per-column flatten
/// height from model stack bases, the second unioning the SDF with the support
/// plane of that height.
/// Source: meshers/hybrid/voxel_mesher_hybrid.cpp:340-540
pub fn build_flattened_sdf(src: &VoxelBuffer, library: &BakedLibrary) -> Option<VoxelBuffer> {
    // No models, the original SDF can be used directly
    if let Some(value) = src.uniform_value(ChannelId::Type) {
        if value == AIR_ID as u64 {
            return None;
        }
    }

    let block_size = src.size();
    let sx = block_size.x;
    let sz = block_size.z;

    // Pass 1, per-column flatten height.
    //
    // A model voxel at (mx, my, mz) occupies the world box
    // (mx..mx+1, my..my+1, mz..mz+1), its bottom face spanning the columns
    // mx..mx+1 by mz..mz+1. Transvoxel meshes the surface under that face from
    // the cell at (mx, my-1, mz), whose four upper corners are exactly those
    // four columns, so all four must be flattened. Flattening only the model's
    // own column tilts the surface, flattening the whole 3x3 neighbourhood
    // lifts a plateau.
    //
    // Only stack bases, model voxels with no model voxel directly below,
    // trigger flattening. Higher voxels of a stack rest on the lower ones.
    // The lowest base claiming a column wins, it is the one needing ground
    // contact.
    // Source: meshers/hybrid/voxel_mesher_hybrid.cpp:373-398
    let mut column_flatten_y = vec![-1i32; (sx * sz) as usize];
    let mut any_model = false;

    let is_rendered_model = |voxel_id: u64| -> bool {
        voxel_id != AIR_ID as u64
            && library
                .model(voxel_id)
                .map(|m| !m.empty)
                .unwrap_or(false)
    };

    for z in 0..sz {
        for x in 0..sx {
            // The y loop walks upwards, so the voxel below is the previous
            // iteration's value
            let mut below_is_model = false;
            for y in 0..block_size.y {
                let voxel_id = src.get_voxel(IVec3::new(x, y, z), ChannelId::Type);
                let is_model = is_rendered_model(voxel_id);
                // At y == 0 the bottom of the buffer including its padding is
                // reached, treated conservatively as a base. A stack
                // continuing into the chunk below is flattened by that chunk.
                let is_stack_base = is_model && !below_is_model;
                below_is_model = is_model;
                if !is_model {
                    continue;
                }
                any_model = true;
                if !is_stack_base || y == 0 {
                    continue;
                }

                // Ground proximity, read all four lower corners of the cell at
                // (x, y-1, z). One solid corner suffices, which matches the
                // claim below covering all four corner columns of that cell.
                // A single-sample test would reject cells the iso-surface
                // crosses diagonally, such as cliff edges.
                // Negative SDF means solid.
                // Source: meshers/hybrid/voxel_mesher_hybrid.cpp:436-468
                let mut ground_below = false;
                for gz in z..=(z + 1) {
                    for gx in x..=(x + 1) {
                        if gx >= sx || gz >= sz || ground_below {
                            continue;
                        }
                        ground_below =
                            src.get_voxel_f(IVec3::new(gx, y - 1, gz), ChannelId::Sdf) < 0.0;
                    }
                }
                if !ground_below {
                    continue;
                }

                for cz in z..=(z + 1) {
                    for cx in x..=(x + 1) {
                        if cx >= sx || cz >= sz {
                            continue;
                        }
                        let slot = &mut column_flatten_y[(cx + cz * sx) as usize];
                        if *slot < 0 || y < *slot {
                            *slot = y;
                        }
                    }
                }
            }
        }
    }

    if !any_model {
        return None;
    }

    // Work on an own copy so the source buffer stays untouched. Mixel4 data
    // must travel along, Transvoxel reads INDICES and WEIGHTS for per-vertex
    // materials and would otherwise fall back to the defaults.
    // Source: meshers/hybrid/voxel_mesher_hybrid.cpp:354-368
    let mut dst = src.clone();

    // Pass 2, union the SDF with the support plane,
    //   sdf(y) = min(original_sdf(y), y - flatten_y).
    // The union raises the ground only below the plane and keeps it where it
    // is already higher, so a model over a dip gets flush ground and a model
    // on a slope embeds into it. Untouched columns let Transvoxel interpolate
    // a smooth slope outwards.
    //
    // Only the two rows shaping the cell directly below the model are written.
    // Marching Cubes polygonizes a cell from its eight corner samples, so
    // deeper samples do not affect that surface. Writing the full column would
    // re-clamp cells below the model on every remesh, making an edit such as a
    // tunnel under the block invisible.
    // Source: meshers/hybrid/voxel_mesher_hybrid.cpp:493-512
    let mut any_flattened = false;
    for z in 0..sz {
        for x in 0..sx {
            let flatten_y = column_flatten_y[(x + z * sx) as usize];
            if flatten_y < 0 {
                continue;
            }
            for y in [flatten_y - 1, flatten_y] {
                if y < 0 || y >= block_size.y {
                    continue;
                }
                let pos = IVec3::new(x, y, z);
                let original = dst.get_voxel_f(pos, ChannelId::Sdf);
                let plane = (y - flatten_y) as f32;
                let unified = if original < plane { original } else { plane };
                if unified != original {
                    dst.set_voxel_f(unified, pos, ChannelId::Sdf).ok();
                    any_flattened = true;
                }
            }
        }
    }

    if any_flattened {
        Some(dst)
    } else {
        None
    }
}

/// Converts a Transvoxel pass into renderable surfaces of one material index.
fn transvoxel_to_surfaces(
    tv_out: tv::TransvoxelOutput,
    material_index: u16,
    texturing_enabled: bool,
) -> (Vec<Surface>, [Vec<Surface>; 6]) {
    let convert = |arrays: tv::MeshArrays| -> Option<Surface> {
        if arrays.indices.is_empty() {
            return None;
        }
        let vertex_count = arrays.vertices.len();
        let lod_data = Some(
            arrays
                .lod_data
                .iter()
                .map(|a| LodAttrib {
                    secondary_position: a.secondary_position,
                    cell_border_mask: a.cell_border_mask,
                    vertex_border_mask: a.vertex_border_mask,
                    transition: a.transition,
                })
                .collect(),
        );
        let texturing_data = if texturing_enabled && arrays.texturing_data_2f32.len() == vertex_count
        {
            Some(arrays.texturing_data_2f32)
        } else {
            None
        };
        let mut surface = Surface {
            positions: arrays.vertices,
            normals: arrays.normals,
            uvs: None,
            indices: arrays.indices,
            lod_data,
            texturing_data,
            material_id_stream: Vec::new(),
            material_index,
        };
        surface.set_material_id_stream(material_index as f32);
        Some(surface)
    };

    let mut surfaces = Vec::new();
    if let Some(s) = convert(tv_out.regular) {
        surfaces.push(s);
    }

    let mut transitions: [Vec<Surface>; 6] = Default::default();
    for (side, arrays) in tv_out.transitions.into_iter().enumerate() {
        if let Some(s) = convert(arrays) {
            transitions[side].push(s);
        }
    }

    (surfaces, transitions)
}

/// Builds the collision sub-range of the terrain surface.
///
/// Transvoxel reports its collider as the range `[0, end)` of render surface 0,
/// excluding the transition meshes.
/// Source: meshers/hybrid/voxel_mesher_hybrid.cpp:704-743
fn terrain_collision_submesh(surface: &Surface) -> CollisionSurface {
    CollisionSurface {
        positions: Vec::new(),
        indices: Vec::new(),
        submesh_vertex_end: Some(surface.positions.len() as u32),
        submesh_index_end: Some(surface.indices.len() as u32),
    }
}

/// Promotes the terrain collision sub-range into explicit arrays, so model
/// triangles can be appended to the same collider.
/// Source: meshers/hybrid/voxel_mesher_hybrid.cpp:704-743
fn promote_terrain_collision(out: &mut MeshOutput) {
    let Some(vertex_end) = out.collision.submesh_vertex_end else {
        // Nothing to promote, for example an empty terrain surface
        return;
    };
    let Some(index_end) = out.collision.submesh_index_end else {
        return;
    };
    let Some(surface) = out.surfaces.first() else {
        return;
    };

    let vertex_end = (vertex_end as usize).min(surface.positions.len());
    let index_end = (index_end as usize).min(surface.indices.len());

    let positions = surface.positions[..vertex_end].to_vec();
    let indices = surface.indices[..index_end].to_vec();

    out.collision.positions = positions;
    out.collision.indices = indices;
    // Switch to explicit arrays so appended model triangles are included
    out.collision.submesh_vertex_end = None;
    out.collision.submesh_index_end = None;
}

/// Builds one chunk mesh from a voxel block.
///
/// The order of the steps is fixed and its reasons are stated per step.
/// Source: meshers/hybrid/voxel_mesher_hybrid.cpp:542-599
pub fn build(
    voxels: &VoxelBuffer,
    library: Option<&BakedLibrary>,
    params: &HybridParams,
) -> MeshOutput {
    let mut out = MeshOutput::default();

    // Step 1, smooth SDF terrain. With models present the SDF is flattened
    // under them first, and the smooth surface is meshed from that copy.
    // Source: meshers/hybrid/voxel_mesher_hybrid.cpp:545-574
    let flattened = match (params.flatten_terrain_under_models, library) {
        (true, Some(lib)) => build_flattened_sdf(voxels, lib),
        _ => None,
    };
    let terrain_voxels = flattened.as_ref().unwrap_or(voxels);

    let terrain_out = tv::build_mesh(
        terrain_voxels,
        params.terrain_transvoxel_params(),
        params.transition_mask,
        false,
    );

    // Step 2, the material id stream of the terrain surfaces. Runs before the
    // fluid pass so the fluid surfaces are not tagged as terrain.
    // Source: meshers/hybrid/voxel_mesher_hybrid.cpp:576-581
    let (terrain_surfaces, terrain_transitions) = transvoxel_to_surfaces(
        terrain_out,
        TERRAIN_MATERIAL_INDEX,
        params.texturing_enabled(),
    );
    out.surfaces = terrain_surfaces;
    out.transition_surfaces = terrain_transitions;

    // Step 3, promote the terrain collision range to explicit arrays. Runs
    // before the fluid pass, which would otherwise overwrite the markers with
    // its own, and fluid carries no collision.
    // Source: meshers/hybrid/voxel_mesher_hybrid.cpp:583-587
    if params.collision_hint {
        if let Some(surface) = out.surfaces.first() {
            out.collision = terrain_collision_submesh(surface);
        }
        promote_terrain_collision(&mut out);
    }

    // Step 4, smooth fluid iso-surface from the DATA5 channel, tagged with the
    // water material index. It contributes no collision, so players can enter
    // the water.
    //
    // The C++ code saves and restores `submesh_*_end`, `mesh_flags` and the
    // surface count around the inner build because that build assigns
    // `mesh_flags` instead of or-ing it
    // (voxel_mesher_hybrid.cpp:599-610, :608-616). Writing the fluid pass into
    // its own output and merging afterwards makes that unnecessary. This
    // difference is deliberate, see doc/source/bevy_port_plan.md section 4, P5.
    // Source: meshers/hybrid/voxel_mesher_hybrid.cpp:589-593, :601-661
    if params.build_fluid_surfaces && !voxels.is_uniform_air_data5() {
        let fluid_out = tv::build_mesh(
            voxels,
            params.fluid_transvoxel_params(),
            params.transition_mask,
            false,
        );
        // The fluid pass runs without texturing, so it carries no texture data
        let (fluid_surfaces, fluid_transitions) =
            transvoxel_to_surfaces(fluid_out, WATER_MATERIAL_INDEX, false);
        out.surfaces.extend(fluid_surfaces);
        for (side, surfaces) in fluid_transitions.into_iter().enumerate() {
            out.transition_surfaces[side].extend(surfaces);
        }
    }

    // Step 5, sharp-edged models from the TYPE channel. Library material N
    // becomes surface material index 2 + N. Model triangles are appended to
    // the collider.
    // Source: meshers/hybrid/voxel_mesher_hybrid.cpp:595-598, :938-969
    if let Some(lib) = library {
        let model_surfaces = build_model_surfaces(voxels, lib, params.lod_index);
        for (material_id, sa) in model_surfaces {
            let material_index = MODEL_MATERIAL_INDEX_BASE + material_id as u16;

            if params.collision_hint {
                out.collision.append(&sa.positions, &sa.indices);
            }

            let vertex_count = sa.positions.len();
            // Models are not transition geometry and do not shift between LOD
            // levels, so their LOD attributes are inert, the secondary
            // position equals the primary one and every mask is zero.
            // Source: meshers/hybrid/voxel_mesher_hybrid.cpp:114-120
            let lod_data = sa
                .positions
                .iter()
                .map(|p| LodAttrib {
                    secondary_position: *p,
                    cell_border_mask: 0,
                    vertex_border_mask: 0,
                    transition: 0,
                })
                .collect();
            // With texturing enabled the models carry an inert texture stream
            // so every surface keeps one vertex format.
            // Source: meshers/hybrid/voxel_mesher_hybrid.cpp:971-987
            let texturing_data = if params.texturing_enabled() {
                Some(vec![Vec2::ZERO; vertex_count])
            } else {
                None
            };

            let mut surface = Surface {
                positions: sa.positions,
                normals: sa.normals,
                uvs: Some(sa.uvs),
                indices: sa.indices,
                lod_data: Some(lod_data),
                texturing_data,
                material_id_stream: Vec::new(),
                material_index,
            };
            surface.set_material_id_stream(material_index as f32);
            out.surfaces.push(surface);
        }
    }

    out
}

/// Helper on the buffer, true when the DATA5 channel holds no fluid at all.
trait Data5Check {
    fn is_uniform_air_data5(&self) -> bool;
}

impl Data5Check for VoxelBuffer {
    fn is_uniform_air_data5(&self) -> bool {
        // A uniform channel whose value is outside the surface produces no
        // geometry, so the pass can be skipped
        match self.uniform_value(ChannelId::Data5) {
            Some(_) => self.get_voxel_f(IVec3::ZERO, ChannelId::Data5) >= 0.0,
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blocky::library_desc::{bake, LibraryDesc, ModelDesc, ModelGeometry};
    use voxel_core::VoxelFormat;

    fn padded_size(size: i32) -> IVec3 {
        IVec3::splat(size + tv::MIN_PADDING + tv::MAX_PADDING)
    }

    /// Buffer whose SDF describes a plane at height `h`, negative below.
    fn plane_buffer(size: i32, h: f32) -> VoxelBuffer {
        let format = VoxelFormat {
            // DATA5 acts as SDF so the fluid pass starts from air
            channels_acting_as_sdf: 1 << ChannelId::Data5.index(),
            ..VoxelFormat::default()
        };
        let mut b = VoxelBuffer::with_format(padded_size(size), format).unwrap();
        b.box_of_whole_buffer().for_each_cell_zxy(|pos| {
            let y = (pos.y - tv::MIN_PADDING) as f32;
            b.set_voxel_f(y - h, pos, ChannelId::Sdf).unwrap();
        });
        b
    }

    /// Library with air at 0 and one cube model using material `material_id`.
    fn cube_library(material_id: u32) -> BakedLibrary {
        let desc = LibraryDesc {
            models: vec![
                ModelDesc {
                    name: "air".into(),
                    geometry: ModelGeometry::Empty,
                    ..ModelDesc::default()
                },
                ModelDesc {
                    name: "stone".into(),
                    geometry: ModelGeometry::Cube,
                    material_id,
                    ..ModelDesc::default()
                },
            ],
            fluids: Vec::new(),
        };
        bake(&desc).unwrap()
    }

    #[test]
    fn padding_covers_both_meshers() {
        assert_eq!(minimum_padding(), tv::MIN_PADDING.max(MODEL_PADDING));
        assert_eq!(maximum_padding(), tv::MAX_PADDING);
        assert!(minimum_padding() >= MODEL_PADDING);
    }

    #[test]
    fn channel_mask_lists_sdf_and_type_at_minimum() {
        let mask = used_channels_mask(&HybridParams {
            texturing_mode: TexturingMode::None,
            build_fluid_surfaces: false,
            ..HybridParams::default()
        });
        assert_ne!(mask & (1 << ChannelId::Sdf.index()), 0);
        assert_ne!(mask & (1 << ChannelId::Type.index()), 0);
        assert_eq!(mask & (1 << ChannelId::Indices.index()), 0);
        assert_eq!(mask & (1 << ChannelId::Data5.index()), 0);
    }

    #[test]
    fn channel_mask_adds_material_and_fluid_channels() {
        let mask = used_channels_mask(&HybridParams {
            texturing_mode: TexturingMode::Mixel4S4,
            build_fluid_surfaces: true,
            ..HybridParams::default()
        });
        assert_ne!(mask & (1 << ChannelId::Indices.index()), 0);
        assert_ne!(mask & (1 << ChannelId::Weights.index()), 0);
        assert_ne!(mask & (1 << ChannelId::Data5.index()), 0);
    }

    #[test]
    fn an_empty_block_produces_nothing() {
        let b = VoxelBuffer::new(padded_size(8)).unwrap();
        let out = build(&b, None, &HybridParams::default());
        assert!(out.is_empty());
        assert!(out.collision.is_empty());
    }

    #[test]
    fn a_plane_produces_a_terrain_surface() {
        let b = plane_buffer(8, 4.0);
        let out = build(&b, None, &HybridParams::default());
        assert_eq!(out.surfaces.len(), 1);
        let s = &out.surfaces[0];
        assert_eq!(s.material_index, TERRAIN_MATERIAL_INDEX);
        assert!(!s.is_empty());
        assert_eq!(s.material_id_stream.len(), s.positions.len());
        assert!(s.material_id_stream.iter().all(|v| *v == 0.0));
        assert!(s.lod_data.is_some());
    }

    #[test]
    fn terrain_collision_is_promoted_to_explicit_arrays() {
        let b = plane_buffer(8, 4.0);
        let out = build(&b, None, &HybridParams::default());
        assert!(out.collision.submesh_vertex_end.is_none());
        assert!(out.collision.submesh_index_end.is_none());
        assert!(!out.collision.positions.is_empty());
        assert_eq!(out.collision.indices.len() % 3, 0);
        // The collider matches the terrain surface's range
        assert_eq!(out.collision.positions.len(), out.surfaces[0].positions.len());
    }

    #[test]
    fn a_disabled_collision_hint_leaves_the_collider_empty() {
        let b = plane_buffer(8, 4.0);
        let out = build(
            &b,
            None,
            &HybridParams {
                collision_hint: false,
                ..HybridParams::default()
            },
        );
        assert!(out.collision.is_empty());
    }

    #[test]
    fn model_voxels_become_surfaces_above_the_model_base_index() {
        let mut b = plane_buffer(8, 4.0);
        let lib = cube_library(0);
        b.set_voxel(1, IVec3::new(4, 5, 4), ChannelId::Type).unwrap();
        let out = build(&b, Some(&lib), &HybridParams::default());
        let model_surfaces: Vec<&Surface> = out
            .surfaces
            .iter()
            .filter(|s| s.material_index >= MODEL_MATERIAL_INDEX_BASE)
            .collect();
        assert_eq!(model_surfaces.len(), 1);
        assert_eq!(model_surfaces[0].material_index, MODEL_MATERIAL_INDEX_BASE);
        assert!(!model_surfaces[0].is_empty());
    }

    #[test]
    fn library_material_ids_offset_by_two() {
        let mut b = plane_buffer(8, 4.0);
        let lib = cube_library(7);
        b.set_voxel(1, IVec3::new(4, 5, 4), ChannelId::Type).unwrap();
        let out = build(&b, Some(&lib), &HybridParams::default());
        assert!(out
            .surfaces
            .iter()
            .any(|s| s.material_index == MODEL_MATERIAL_INDEX_BASE + 7));
    }

    #[test]
    fn model_triangles_are_appended_to_the_collider() {
        let mut b = plane_buffer(8, 4.0);
        let lib = cube_library(0);
        let without = build(&b, Some(&lib), &HybridParams::default());
        b.set_voxel(1, IVec3::new(4, 5, 4), ChannelId::Type).unwrap();
        let with = build(&b, Some(&lib), &HybridParams::default());
        assert!(with.collision.indices.len() > without.collision.indices.len());
        // Appended indices stay within the collider's vertex range
        let count = with.collision.positions.len() as u32;
        for i in &with.collision.indices {
            assert!(*i < count, "collision index {i} exceeds {count}");
        }
    }

    #[test]
    fn model_surfaces_carry_inert_lod_attributes() {
        let mut b = plane_buffer(8, 4.0);
        let lib = cube_library(0);
        b.set_voxel(1, IVec3::new(4, 5, 4), ChannelId::Type).unwrap();
        let out = build(&b, Some(&lib), &HybridParams::default());
        let s = out
            .surfaces
            .iter()
            .find(|s| s.material_index >= MODEL_MATERIAL_INDEX_BASE)
            .unwrap();
        let lod = s.lod_data.as_ref().unwrap();
        assert_eq!(lod.len(), s.positions.len());
        for (attrib, position) in lod.iter().zip(&s.positions) {
            assert_eq!(attrib.secondary_position, *position);
            assert_eq!(attrib.cell_border_mask, 0);
            assert_eq!(attrib.vertex_border_mask, 0);
            assert_eq!(attrib.transition, 0);
        }
    }

    #[test]
    fn texturing_gives_every_surface_a_texture_stream() {
        let mut b = plane_buffer(8, 4.0);
        let lib = cube_library(0);
        b.set_voxel(1, IVec3::new(4, 5, 4), ChannelId::Type).unwrap();
        let out = build(
            &b,
            Some(&lib),
            &HybridParams {
                texturing_mode: TexturingMode::Mixel4S4,
                textures_ignore_air_voxels: true,
                ..HybridParams::default()
            },
        );
        for s in &out.surfaces {
            let data = s
                .texturing_data
                .as_ref()
                .unwrap_or_else(|| panic!("surface {} has no texture stream", s.material_index));
            assert_eq!(data.len(), s.positions.len());
        }
    }

    #[test]
    fn without_texturing_no_surface_carries_a_texture_stream() {
        let b = plane_buffer(8, 4.0);
        let out = build(&b, None, &HybridParams::default());
        for s in &out.surfaces {
            assert!(s.texturing_data.is_none());
        }
    }

    #[test]
    fn a_fluid_filled_data5_channel_produces_a_water_surface() {
        let mut b = plane_buffer(8, 4.0);
        // Fill DATA5 with a plane at height 2, so a fluid surface exists
        b.box_of_whole_buffer().for_each_cell_zxy(|pos| {
            let y = (pos.y - tv::MIN_PADDING) as f32;
            b.set_voxel_f(y - 2.0, pos, ChannelId::Data5).unwrap();
        });
        let out = build(&b, None, &HybridParams::default());
        let water: Vec<&Surface> = out
            .surfaces_with_material(WATER_MATERIAL_INDEX)
            .collect();
        assert_eq!(water.len(), 1);
        assert!(!water[0].is_empty());
        assert!(water[0].material_id_stream.iter().all(|v| *v == 1.0));
    }

    #[test]
    fn the_fluid_pass_does_not_contribute_collision() {
        let mut b = plane_buffer(8, 4.0);
        let without = build(&b, None, &HybridParams::default());
        b.box_of_whole_buffer().for_each_cell_zxy(|pos| {
            let y = (pos.y - tv::MIN_PADDING) as f32;
            b.set_voxel_f(y - 2.0, pos, ChannelId::Data5).unwrap();
        });
        let with = build(&b, None, &HybridParams::default());
        assert!(with
            .surfaces_with_material(WATER_MATERIAL_INDEX)
            .next()
            .is_some());
        assert_eq!(with.collision.indices.len(), without.collision.indices.len());
    }

    #[test]
    fn a_disabled_fluid_pass_produces_no_water_surface() {
        let mut b = plane_buffer(8, 4.0);
        b.box_of_whole_buffer().for_each_cell_zxy(|pos| {
            let y = (pos.y - tv::MIN_PADDING) as f32;
            b.set_voxel_f(y - 2.0, pos, ChannelId::Data5).unwrap();
        });
        let out = build(
            &b,
            None,
            &HybridParams {
                build_fluid_surfaces: false,
                ..HybridParams::default()
            },
        );
        assert!(out
            .surfaces_with_material(WATER_MATERIAL_INDEX)
            .next()
            .is_none());
    }

    #[test]
    fn surface_order_is_terrain_then_water_then_models() {
        let mut b = plane_buffer(8, 4.0);
        let lib = cube_library(0);
        b.box_of_whole_buffer().for_each_cell_zxy(|pos| {
            let y = (pos.y - tv::MIN_PADDING) as f32;
            b.set_voxel_f(y - 2.0, pos, ChannelId::Data5).unwrap();
        });
        b.set_voxel(1, IVec3::new(4, 5, 4), ChannelId::Type).unwrap();
        let out = build(&b, Some(&lib), &HybridParams::default());
        let indices: Vec<u16> = out.surfaces.iter().map(|s| s.material_index).collect();
        assert_eq!(
            indices,
            vec![
                TERRAIN_MATERIAL_INDEX,
                WATER_MATERIAL_INDEX,
                MODEL_MATERIAL_INDEX_BASE
            ]
        );
    }

    #[test]
    fn flattening_is_skipped_without_models() {
        let b = plane_buffer(8, 4.0);
        let lib = cube_library(0);
        assert!(build_flattened_sdf(&b, &lib).is_none());
    }

    #[test]
    fn flattening_raises_the_ground_under_a_model() {
        let mut b = plane_buffer(8, 4.0);
        let lib = cube_library(0);
        // The ground-proximity test requires a strictly negative SDF directly
        // below the model base (voxel_mesher_hybrid.cpp:466). With a plane at
        // height 4 the rows below y = 5 are solid, so a base at y = 7 floats
        // and gets no flattening.
        let model_pos = IVec3::new(4, 7, 4);
        b.set_voxel(1, model_pos, ChannelId::Type).unwrap();
        let flattened = build_flattened_sdf(&b, &lib);
        assert!(flattened.is_none());

        // A base resting on solid ground instead
        let mut b2 = plane_buffer(8, 4.0);
        let base = IVec3::new(4, 5, 4);
        b2.set_voxel(1, base, ChannelId::Type).unwrap();
        let flattened = build_flattened_sdf(&b2, &lib).expect("flattening expected");
        // The row at the model base is clamped to at most the plane value 0
        let at_base = flattened.get_voxel_f(base, ChannelId::Sdf);
        assert!(at_base <= 0.0, "sdf at base is {at_base}");
        // The row below is solid
        let below = flattened.get_voxel_f(base - IVec3::Y, ChannelId::Sdf);
        assert!(below < 0.0, "sdf below base is {below}");
    }

    #[test]
    fn flattening_claims_all_four_corner_columns() {
        let mut b = plane_buffer(8, 4.0);
        let lib = cube_library(0);
        let base = IVec3::new(4, 5, 4);
        b.set_voxel(1, base, ChannelId::Type).unwrap();
        let flattened = build_flattened_sdf(&b, &lib).unwrap();
        for dz in 0..=1 {
            for dx in 0..=1 {
                let pos = base + IVec3::new(dx, 0, dz);
                let value = flattened.get_voxel_f(pos, ChannelId::Sdf);
                assert!(value <= 0.0, "corner {pos} has sdf {value}");
            }
        }
        // A column outside the four corners keeps its original value
        let outside = base + IVec3::new(3, 0, 0);
        assert_eq!(
            flattened.get_voxel_f(outside, ChannelId::Sdf),
            b.get_voxel_f(outside, ChannelId::Sdf)
        );
    }

    #[test]
    fn flattening_only_touches_the_two_rows_of_the_cell_below() {
        let mut b = plane_buffer(8, 4.0);
        let lib = cube_library(0);
        let base = IVec3::new(4, 5, 4);
        b.set_voxel(1, base, ChannelId::Type).unwrap();
        let flattened = build_flattened_sdf(&b, &lib).unwrap();
        // Two rows below the cell stay untouched, so edits below a model stay
        // visible
        for dy in 2..5 {
            let pos = base - IVec3::new(0, dy, 0);
            assert_eq!(
                flattened.get_voxel_f(pos, ChannelId::Sdf),
                b.get_voxel_f(pos, ChannelId::Sdf),
                "row {pos} was modified"
            );
        }
    }

    #[test]
    fn only_stack_bases_trigger_flattening() {
        let mut b = plane_buffer(8, 4.0);
        let lib = cube_library(0);
        let base = IVec3::new(4, 5, 4);
        // A stack of three model voxels
        for dy in 0..3 {
            b.set_voxel(1, base + IVec3::new(0, dy, 0), ChannelId::Type)
                .unwrap();
        }
        let flattened = build_flattened_sdf(&b, &lib).unwrap();
        // The ground meets the lowest voxel, the higher ones leave no mark
        let at_base = flattened.get_voxel_f(base, ChannelId::Sdf);
        assert!(at_base <= 0.0);
        let above = base + IVec3::new(0, 2, 0);
        assert_eq!(
            flattened.get_voxel_f(above, ChannelId::Sdf),
            b.get_voxel_f(above, ChannelId::Sdf)
        );
    }

    #[test]
    fn a_model_id_without_geometry_does_not_flatten() {
        let mut b = plane_buffer(8, 4.0);
        let lib = cube_library(0);
        // Id 0 is air, id 2 does not exist in this library
        b.set_voxel(2, IVec3::new(4, 6, 4), ChannelId::Type).unwrap();
        assert!(build_flattened_sdf(&b, &lib).is_none());
    }

    #[test]
    fn disabled_flattening_leaves_the_sdf_alone() {
        // A model resting in a dip, so flattening has to lift the ground.
        // The plane is at height 4, the model base sits at padded y = 7 over a
        // column carved down to the plane's own level.
        let mut b = plane_buffer(8, 4.0);
        let lib = cube_library(0);
        let base = IVec3::new(4, 7, 4);
        // Make the four corner columns solid up to the model base, so the
        // ground-proximity test passes and the plane still lies below it
        for dz in 0..=1 {
            for dx in 0..=1 {
                let column = base + IVec3::new(dx, 0, dz);
                b.set_voxel_f(-0.5, column - IVec3::Y, ChannelId::Sdf).unwrap();
            }
        }
        b.set_voxel(1, base, ChannelId::Type).unwrap();

        let flattened = build_flattened_sdf(&b, &lib).expect("flattening expected");
        // The flattened copy differs from the source at the model base
        assert_ne!(
            flattened.get_voxel_f(base, ChannelId::Sdf),
            b.get_voxel_f(base, ChannelId::Sdf),
            "the sdf at the model base was not changed"
        );

        // Disabling the option leaves the terrain exactly as the plain
        // Transvoxel pass produces it
        let without = build(
            &b,
            Some(&lib),
            &HybridParams {
                flatten_terrain_under_models: false,
                ..HybridParams::default()
            },
        );
        let plain = tv::build_mesh(
            &b,
            HybridParams::default().terrain_transvoxel_params(),
            0,
            false,
        );
        assert_eq!(without.surfaces[0].positions, plain.regular.vertices);

        // Enabling it changes the terrain geometry
        let with = build(&b, Some(&lib), &HybridParams::default());
        assert_ne!(
            with.surfaces[0].positions, without.surfaces[0].positions,
            "flattening had no effect on the terrain surface"
        );
    }

    #[test]
    fn flattening_preserves_the_material_channels() {
        let mut b = plane_buffer(8, 4.0);
        let lib = cube_library(0);
        let base = IVec3::new(4, 5, 4);
        b.set_voxel(1, base, ChannelId::Type).unwrap();
        let marker = voxel_core::mixel4::make_encoded_indices_for_single_texture(42) as u64;
        b.set_voxel(marker, IVec3::splat(3), ChannelId::Indices).unwrap();
        let flattened = build_flattened_sdf(&b, &lib).unwrap();
        assert_eq!(
            flattened.get_voxel(IVec3::splat(3), ChannelId::Indices),
            marker
        );
    }

    #[test]
    fn transition_surfaces_follow_the_requested_mask() {
        let b = plane_buffer(8, 4.0);
        let out = build(
            &b,
            None,
            &HybridParams {
                transition_mask: 0b000001,
                ..HybridParams::default()
            },
        );
        assert!(!out.transition_surfaces[0].is_empty());
        for side in 1..6 {
            assert!(
                out.transition_surfaces[side].is_empty(),
                "side {side} not masked"
            );
        }
    }

    #[test]
    fn lod_index_scales_both_parts() {
        let mut b = plane_buffer(8, 4.0);
        let lib = cube_library(0);
        b.set_voxel(1, IVec3::new(4, 5, 4), ChannelId::Type).unwrap();
        let out0 = build(&b, Some(&lib), &HybridParams::default());
        let out1 = build(
            &b,
            Some(&lib),
            &HybridParams {
                lod_index: 1,
                ..HybridParams::default()
            },
        );
        let extent = |out: &MeshOutput| {
            out.surfaces
                .iter()
                .flat_map(|s| s.positions.iter())
                .map(|p| p.x)
                .fold(0.0f32, f32::max)
        };
        let e0 = extent(&out0);
        let e1 = extent(&out1);
        assert!(e1 > e0 * 1.5, "lod 0 extent {e0}, lod 1 extent {e1}");
    }

    #[test]
    fn every_surface_index_addresses_an_existing_vertex() {
        let mut b = plane_buffer(8, 4.0);
        let lib = cube_library(0);
        b.box_of_whole_buffer().for_each_cell_zxy(|pos| {
            let y = (pos.y - tv::MIN_PADDING) as f32;
            b.set_voxel_f(y - 2.0, pos, ChannelId::Data5).unwrap();
        });
        b.set_voxel(1, IVec3::new(4, 5, 4), ChannelId::Type).unwrap();
        let out = build(&b, Some(&lib), &HybridParams::default());
        for s in &out.surfaces {
            let count = s.positions.len() as u32;
            assert_eq!(s.normals.len(), s.positions.len());
            assert_eq!(s.material_id_stream.len(), s.positions.len());
            for i in &s.indices {
                assert!(*i < count, "index {i} exceeds {count}");
            }
        }
    }

    #[test]
    fn a_block_with_only_models_still_produces_a_mesh() {
        let mut b = VoxelBuffer::new(padded_size(8)).unwrap();
        let lib = cube_library(0);
        b.set_voxel(1, IVec3::new(4, 4, 4), ChannelId::Type).unwrap();
        let out = build(&b, Some(&lib), &HybridParams::default());
        // No smooth terrain, but a model surface
        assert_eq!(out.surfaces.len(), 1);
        assert_eq!(out.surfaces[0].material_index, MODEL_MATERIAL_INDEX_BASE);
        assert!(!out.collision.indices.is_empty());
    }
}
