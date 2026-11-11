//! Engine-neutral mesh output, replacing `VoxelMesher::Output`.
//!
//! The C++ structure stores Godot `Array` and `Mesh::PrimitiveType`
//! (`meshers/voxel_mesher.h:50-77`), which are replaced here. The per-vertex
//! material id is kept as its own field instead of being written into CUSTOM1
//! or CUSTOM2 depending on the texturing mode, which removes the mismatched
//! vertex format the C++ code documents repeatedly
//! (`meshers/hybrid/voxel_mesher_hybrid.cpp:592-598`, `:612-616`).
//! See doc/source/bevy_port_plan.md section 3.

use glam::{Vec2, Vec3};
use voxel_core::cube::SIDE_COUNT;

/// Material index of the smooth terrain surface.
/// Source: meshers/hybrid/voxel_mesher_hybrid.h:41-46
pub const TERRAIN_MATERIAL_INDEX: u16 = 0;
/// Material index of the smooth fluid iso-surface.
/// Source: meshers/hybrid/voxel_mesher_hybrid.h:53
pub const WATER_MATERIAL_INDEX: u16 = 1;
/// First material index the model library maps to. Library material N becomes
/// surface material index `MODEL_MATERIAL_INDEX_BASE + N`.
/// Source: meshers/hybrid/voxel_mesher_hybrid.cpp:966-969
pub const MODEL_MATERIAL_INDEX_BASE: u16 = 2;

/// LOD attributes a Transvoxel vertex carries, the C++ CUSTOM0 stream.
/// Source: meshers/transvoxel/transvoxel.h:37-54
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct LodAttrib {
    pub secondary_position: Vec3,
    pub cell_border_mask: u8,
    pub vertex_border_mask: u8,
    pub transition: u8,
}

/// One renderable surface.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Surface {
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub uvs: Option<Vec<Vec2>>,
    pub indices: Vec<u32>,
    /// Transvoxel LOD attributes, present on smooth surfaces only.
    pub lod_data: Option<Vec<LodAttrib>>,
    /// Packed Mixel4 indices and weights, present when texturing is enabled.
    pub texturing_data: Option<Vec<Vec2>>,
    /// Per-vertex material id, one value per position.
    pub material_id_stream: Vec<f32>,
    /// Which material this surface is rendered with.
    pub material_index: u16,
}

impl Surface {
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }

    /// Fills the material id stream with one value per vertex.
    /// Source: meshers/hybrid/voxel_mesher_hybrid.cpp, build_material_id_stream
    pub fn set_material_id_stream(&mut self, material_id: f32) {
        self.material_id_stream.clear();
        self.material_id_stream
            .resize(self.positions.len(), material_id);
    }
}

/// Collision geometry of a chunk.
/// Source: meshers/voxel_mesher.h:61-69
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CollisionSurface {
    pub positions: Vec<Vec3>,
    pub indices: Vec<u32>,
    /// When set, the collider is the sub-range `[0, end)` of render surface 0
    /// rather than the explicit arrays. `None` means the explicit arrays apply.
    pub submesh_vertex_end: Option<u32>,
    pub submesh_index_end: Option<u32>,
}

impl CollisionSurface {
    /// Appends a surface's triangles, rebasing indices.
    /// Source: meshers/hybrid/voxel_mesher_hybrid.cpp:73-81
    pub fn append(&mut self, positions: &[Vec3], indices: &[u32]) {
        let base = self.positions.len() as u32;
        self.positions.extend_from_slice(positions);
        self.indices.extend(indices.iter().map(|i| i + base));
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty() && self.submesh_index_end.is_none()
    }
}

/// Everything one chunk build produces.
/// Source: meshers/voxel_mesher.h:50-77
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MeshOutput {
    pub surfaces: Vec<Surface>,
    pub transition_surfaces: [Vec<Surface>; SIDE_COUNT],
    pub collision: CollisionSurface,
}

impl MeshOutput {
    pub fn is_empty(&self) -> bool {
        self.surfaces.iter().all(|s| s.is_empty())
    }

    /// Total triangle count across the render surfaces.
    pub fn triangle_count(&self) -> usize {
        self.surfaces.iter().map(|s| s.indices.len() / 3).sum()
    }

    /// Surfaces carrying a given material index.
    pub fn surfaces_with_material(&self, material_index: u16) -> impl Iterator<Item = &Surface> {
        self.surfaces
            .iter()
            .filter(move |s| s.material_index == material_index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn material_index_convention_matches_cpp() {
        assert_eq!(TERRAIN_MATERIAL_INDEX, 0);
        assert_eq!(WATER_MATERIAL_INDEX, 1);
        assert_eq!(MODEL_MATERIAL_INDEX_BASE, 2);
    }

    #[test]
    fn a_fresh_output_is_empty() {
        let out = MeshOutput::default();
        assert!(out.is_empty());
        assert_eq!(out.triangle_count(), 0);
        assert!(out.collision.is_empty());
    }

    #[test]
    fn material_id_stream_matches_the_vertex_count() {
        let mut s = Surface {
            positions: vec![Vec3::ZERO; 5],
            ..Surface::default()
        };
        s.set_material_id_stream(3.0);
        assert_eq!(s.material_id_stream.len(), 5);
        assert!(s.material_id_stream.iter().all(|v| *v == 3.0));
    }

    #[test]
    fn appending_to_collision_rebases_indices() {
        let mut col = CollisionSurface::default();
        col.append(&[Vec3::ZERO, Vec3::X, Vec3::Y], &[0, 1, 2]);
        col.append(&[Vec3::Z, Vec3::X, Vec3::Y], &[0, 1, 2]);
        assert_eq!(col.indices, vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(col.positions.len(), 6);
    }

    #[test]
    fn a_submesh_range_counts_as_non_empty_collision() {
        let col = CollisionSurface {
            submesh_vertex_end: Some(10),
            submesh_index_end: Some(30),
            ..CollisionSurface::default()
        };
        assert!(!col.is_empty());
    }

    #[test]
    fn surfaces_can_be_filtered_by_material() {
        let out = MeshOutput {
            surfaces: vec![
                Surface {
                    indices: vec![0, 1, 2],
                    material_index: TERRAIN_MATERIAL_INDEX,
                    ..Surface::default()
                },
                Surface {
                    indices: vec![0, 1, 2],
                    material_index: WATER_MATERIAL_INDEX,
                    ..Surface::default()
                },
                Surface {
                    indices: vec![0, 1, 2, 3, 4, 5],
                    material_index: WATER_MATERIAL_INDEX,
                    ..Surface::default()
                },
            ],
            ..MeshOutput::default()
        };
        assert_eq!(out.surfaces_with_material(WATER_MATERIAL_INDEX).count(), 2);
        assert_eq!(out.triangle_count(), 4);
        assert!(!out.is_empty());
    }
}
