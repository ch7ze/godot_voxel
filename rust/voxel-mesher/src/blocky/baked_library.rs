//! Baked model data consumed by the mesher.
//!
//! Ported from `meshers/blocky/blocky_baked_library.h`. Only the plain data
//! the mesher reads is present. The Godot resource classes that produce it
//! (`voxel_blocky_model*.cpp`, `voxel_blocky_library*.cpp`) are replaced by the
//! `library_desc` loader, see doc/source/bevy_port_plan.md section 4, P4.

use glam::{Vec2, Vec3};
use voxel_core::cube::SIDE_COUNT;

/// Maximum number of distinct models, bounded by the 16-bit TYPE channel.
/// Source: meshers/blocky/blocky_baked_library.h:22
pub const MAX_MODELS: usize = 65536;
/// Maximum number of fluids, bounded by how fluids are indexed.
/// Source: meshers/blocky/blocky_baked_library.h:24
pub const MAX_FLUIDS: usize = 256;
/// Maximum number of materials.
/// Source: meshers/blocky/blocky_baked_library.h:29
pub const MAX_MATERIALS: usize = 65536;
/// Model id meaning "nothing". No non-empty model may use it.
/// Source: meshers/blocky/blocky_baked_library.h:33
pub const AIR_ID: u16 = 0;
/// Fluid index meaning "not a fluid".
/// Source: meshers/blocky/blocky_baked_library.h:34
pub const NULL_FLUID_INDEX: u8 = 255;
/// Materials a single model may use.
/// Source: meshers/blocky/blocky_baked_library.h:35
pub const MAX_SURFACES: usize = 2;

/// Geometry of one cube side of a model. Normals are omitted, a side's normal
/// is constant.
/// Source: meshers/blocky/blocky_baked_library.h:42-55
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SideSurface {
    pub positions: Vec<Vec3>,
    pub uvs: Vec<Vec2>,
    pub indices: Vec<u32>,
    pub tangents: Vec<f32>,
}

impl SideSurface {
    pub fn clear(&mut self) {
        self.positions.clear();
        self.uvs.clear();
        self.indices.clear();
        self.tangents.clear();
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

/// Inner geometry of a model, carrying its own material.
/// Source: meshers/blocky/blocky_baked_library.h:57-75
#[derive(Debug, Clone, PartialEq)]
pub struct Surface {
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub uvs: Vec<Vec2>,
    pub indices: Vec<u32>,
    pub tangents: Vec<f32>,
    pub material_id: u32,
    pub collision_enabled: bool,
}

impl Default for Surface {
    fn default() -> Surface {
        Surface {
            positions: Vec::new(),
            normals: Vec::new(),
            uvs: Vec::new(),
            indices: Vec::new(),
            tangents: Vec::new(),
            material_id: 0,
            collision_enabled: true,
        }
    }
}

impl Surface {
    pub fn clear(&mut self) {
        self.positions.clear();
        self.normals.clear();
        self.uvs.clear();
        self.indices.clear();
        self.tangents.clear();
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

/// Geometry of a model, split into inner surfaces and per-side surfaces so
/// sides can be culled individually.
/// Source: meshers/blocky/blocky_baked_library.h:77-116
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Model {
    pub surfaces: [Surface; MAX_SURFACES],
    pub sides_surfaces: [[SideSurface; MAX_SURFACES]; SIDE_COUNT],
    pub surface_count: usize,
    /// Sides the model never renders, indexed by the side enum.
    pub empty_sides_mask: u8,
    pub full_sides_mask: u8,
    /// Shape id per side, used to cull sides against neighbours quickly.
    pub side_pattern_indices: [u32; SIDE_COUNT],
}

impl Model {
    pub fn clear(&mut self) {
        for surface in &mut self.surfaces {
            surface.clear();
        }
        for side in &mut self.sides_surfaces {
            for surface in side {
                surface.clear();
            }
        }
    }
}

/// One baked model of the library.
/// Source: meshers/blocky/blocky_baked_library.h:41-139
#[derive(Debug, Clone, PartialEq)]
pub struct BakedModel {
    pub model: Model,
    pub color: [f32; 4],
    pub transparency_index: u8,
    pub culls_neighbors: bool,
    pub contributes_to_ao: bool,
    pub empty: bool,
    pub is_random_tickable: bool,
    pub is_transparent: bool,
    pub fluid_index: u8,
    pub fluid_level: u8,
    pub lod_skirts: bool,
    pub box_collision_mask: u32,
    pub tags_mask: u32,
}

impl Default for BakedModel {
    fn default() -> BakedModel {
        BakedModel {
            model: Model::default(),
            color: [1.0, 1.0, 1.0, 1.0],
            transparency_index: 0,
            culls_neighbors: true,
            contributes_to_ao: true,
            empty: true,
            is_random_tickable: false,
            is_transparent: false,
            fluid_index: NULL_FLUID_INDEX,
            fluid_level: 0,
            lod_skirts: false,
            box_collision_mask: 0,
            tags_mask: 0,
        }
    }
}

impl BakedModel {
    pub fn is_fluid(&self) -> bool {
        self.fluid_index != NULL_FLUID_INDEX
    }
}

/// Side geometry of a fluid. UVs are generated for the flow animation, so none
/// are stored.
/// Source: meshers/blocky/blocky_baked_library.h:145-157
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FluidSurface {
    pub positions: Vec<Vec3>,
    pub indices: Vec<u32>,
    pub tangents: Vec<f32>,
}

impl FluidSurface {
    pub fn clear(&mut self) {
        self.positions.clear();
        self.indices.clear();
        self.tangents.clear();
    }
}

/// A baked fluid.
/// Source: meshers/blocky/blocky_baked_library.h:141-166
#[derive(Debug, Clone, PartialEq)]
pub struct BakedFluid {
    pub side_surfaces: [FluidSurface; SIDE_COUNT],
    pub material_id: u32,
    pub max_level: u8,
    pub dip_when_flowing_down: bool,
}

impl BakedFluid {
    /// Source: meshers/blocky/blocky_baked_library.h:142
    pub const TOP_HEIGHT: f32 = 0.9375;
    /// Source: meshers/blocky/blocky_baked_library.h:143
    pub const BOTTOM_HEIGHT: f32 = 0.0625;
}

impl Default for BakedFluid {
    fn default() -> BakedFluid {
        BakedFluid {
            side_surfaces: Default::default(),
            material_id: 0,
            max_level: 1,
            dip_when_flowing_down: false,
        }
    }
}

/// All baked data the mesher reads.
///
/// `side_pattern_culling` is a square bit matrix of `side_pattern_count` rows,
/// entry `(a, b)` telling whether pattern A occludes pattern B.
/// Source: meshers/blocky/blocky_baked_library.h:168-197
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BakedLibrary {
    pub side_pattern_culling: Vec<bool>,
    pub side_pattern_count: usize,
    pub models: Vec<BakedModel>,
    pub fluids: Vec<BakedFluid>,
    pub indexed_materials_count: usize,
}

impl BakedLibrary {
    /// Source: meshers/blocky/blocky_baked_library.h:186-188
    #[inline]
    pub fn has_model(&self, i: u64) -> bool {
        (i as usize) < self.models.len()
    }

    #[inline]
    pub fn model(&self, i: u64) -> Option<&BakedModel> {
        self.models.get(i as usize)
    }

    /// Source: meshers/blocky/blocky_baked_library.h:190-196
    #[inline]
    pub fn get_side_pattern_occlusion(&self, pattern_a: usize, pattern_b: usize) -> bool {
        let index = pattern_a + pattern_b * self.side_pattern_count;
        self.side_pattern_culling.get(index).copied().unwrap_or(false)
    }
}

/// True when a neighbour cannot occlude a side regardless of its shape.
/// Source: meshers/blocky/voxel_mesher_blocky.h:133-136
#[inline]
pub fn is_face_visible_regardless_of_shape(vt: &BakedModel, other_vt: &BakedModel) -> bool {
    other_vt.empty
        || other_vt.transparency_index > vt.transparency_index
        || !other_vt.culls_neighbors
}

/// True when the side patterns differ and B does not occlude A.
/// Source: meshers/blocky/voxel_mesher_blocky.h:139-149
#[inline]
pub fn is_face_visible_according_to_shape(
    lib: &BakedLibrary,
    vt: &BakedModel,
    other_vt: &BakedModel,
    side: usize,
) -> bool {
    let ai = vt.model.side_pattern_indices[side] as usize;
    let bi = other_vt.model.side_pattern_indices[voxel_core::cube::OPPOSITE_SIDE[side]] as usize;
    ai != bi && !lib.get_side_pattern_occlusion(bi, ai)
}

/// True when a model's side must be rendered given its neighbour.
/// An unknown neighbour id leaves the side visible.
/// Source: meshers/blocky/voxel_mesher_blocky.h:151-166
#[inline]
pub fn is_face_visible(
    lib: &BakedLibrary,
    vt: &BakedModel,
    other_voxel_id: u64,
    side: usize,
) -> bool {
    match lib.model(other_voxel_id) {
        Some(other_vt) => {
            is_face_visible_regardless_of_shape(vt, other_vt)
                || is_face_visible_according_to_shape(lib, vt, other_vt, side)
        }
        None => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_model(transparency_index: u8) -> BakedModel {
        BakedModel {
            empty: false,
            transparency_index,
            culls_neighbors: true,
            ..BakedModel::default()
        }
    }

    fn library(models: Vec<BakedModel>) -> BakedLibrary {
        BakedLibrary {
            side_pattern_count: 1,
            side_pattern_culling: vec![true],
            models,
            ..BakedLibrary::default()
        }
    }

    #[test]
    fn air_id_and_null_fluid_match_cpp() {
        assert_eq!(AIR_ID, 0);
        assert_eq!(NULL_FLUID_INDEX, 255);
        assert_eq!(MAX_SURFACES, 2);
        assert_eq!(MAX_FLUIDS, 256);
    }

    #[test]
    fn a_default_model_is_empty_and_not_a_fluid() {
        let m = BakedModel::default();
        assert!(m.empty);
        assert!(!m.is_fluid());
        assert_eq!(m.fluid_index, NULL_FLUID_INDEX);
    }

    #[test]
    fn an_empty_neighbour_leaves_the_side_visible() {
        let lib = library(vec![BakedModel::default(), solid_model(0)]);
        assert!(is_face_visible(&lib, &lib.models[1], 0, 0));
    }

    #[test]
    fn a_more_transparent_neighbour_leaves_the_side_visible() {
        let lib = library(vec![solid_model(0), solid_model(5)]);
        // Neighbour 1 is more transparent than model 0
        assert!(is_face_visible(&lib, &lib.models[0], 1, 0));
        // The reverse is occluded, same pattern and occlusion set
        assert!(!is_face_visible(&lib, &lib.models[1], 0, 0));
    }

    #[test]
    fn a_non_culling_neighbour_leaves_the_side_visible() {
        let mut neighbour = solid_model(0);
        neighbour.culls_neighbors = false;
        let lib = library(vec![solid_model(0), neighbour]);
        assert!(is_face_visible(&lib, &lib.models[0], 1, 0));
    }

    #[test]
    fn an_unknown_neighbour_id_leaves_the_side_visible() {
        let lib = library(vec![solid_model(0)]);
        assert!(is_face_visible(&lib, &lib.models[0], 999, 0));
    }

    #[test]
    fn differing_side_patterns_leave_the_side_visible() {
        let mut a = solid_model(0);
        let mut b = solid_model(0);
        a.model.side_pattern_indices = [0; SIDE_COUNT];
        b.model.side_pattern_indices = [1; SIDE_COUNT];
        let lib = BakedLibrary {
            side_pattern_count: 2,
            // Nothing occludes anything
            side_pattern_culling: vec![false; 4],
            models: vec![a, b],
            ..BakedLibrary::default()
        };
        assert!(is_face_visible(&lib, &lib.models[0], 1, 0));
    }

    #[test]
    fn an_identical_occluding_pattern_hides_the_side() {
        let lib = library(vec![solid_model(0), solid_model(0)]);
        // Same pattern index 0, so `ai != bi` already fails
        assert!(!is_face_visible(&lib, &lib.models[0], 1, 0));
    }

    #[test]
    fn side_pattern_occlusion_is_indexed_column_major() {
        let lib = BakedLibrary {
            side_pattern_count: 2,
            // (a=1, b=0) is set, meaning pattern 1 occludes pattern 0
            side_pattern_culling: vec![false, true, false, false],
            ..BakedLibrary::default()
        };
        assert!(lib.get_side_pattern_occlusion(1, 0));
        assert!(!lib.get_side_pattern_occlusion(0, 1));
        // Out-of-range reads report no occlusion
        assert!(!lib.get_side_pattern_occlusion(9, 9));
    }

    #[test]
    fn fluid_heights_match_cpp_constants() {
        assert_eq!(BakedFluid::TOP_HEIGHT, 0.9375);
        assert_eq!(BakedFluid::BOTTOM_HEIGHT, 0.0625);
        assert_eq!(BakedFluid::default().max_level, 1);
    }
}
