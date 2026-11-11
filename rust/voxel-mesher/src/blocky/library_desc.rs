//! Declarative model library, replacing the Godot resource path.
//!
//! The C++ classes `voxel_blocky_model*.cpp`, `voxel_blocky_library*.cpp` and
//! `voxel_blocky_fluid.cpp` build the baked data from editor resources. Those
//! are not ported, see doc/source/bevy_port_plan.md section 4, P4. This module
//! builds the same baked data from a plain description a caller can
//! deserialize with any format, so the Bevy side can load it as an asset.

use glam::{Vec2, Vec3};
use voxel_core::cube::{
    side_axis::SIDE_POSITIVE_Y, CORNER_POSITION, SIDE_CORNERS, SIDE_COUNT, SIDE_QUAD_TRIANGLES,
    SIDE_TANGENTS,
};

use super::baked_library::{
    BakedFluid, BakedLibrary, BakedModel, FluidSurface, Model, SideSurface, Surface, AIR_ID,
    MAX_FLUIDS, MAX_MODELS, MAX_SURFACES, NULL_FLUID_INDEX,
};

/// Geometry a model uses.
#[derive(Debug, Clone, PartialEq)]
pub enum ModelGeometry {
    /// Renders nothing, the air model.
    Empty,
    /// A full cube, one quad per side.
    Cube,
    /// Caller-supplied geometry.
    Custom {
        /// Inner geometry, rendered regardless of neighbours.
        surfaces: Vec<SurfaceDesc>,
        /// Per-side geometry, culled against neighbours.
        sides: Vec<SideDesc>,
    },
}

/// One inner surface of a custom model.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SurfaceDesc {
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub uvs: Vec<Vec2>,
    pub indices: Vec<u32>,
    pub material_id: u32,
    pub collision_enabled: bool,
}

/// One side surface of a custom model.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SideDesc {
    /// Side index, following `voxel_core::cube::side_axis`.
    pub side: usize,
    /// Which of the model's surfaces this side belongs to.
    pub surface_index: usize,
    pub positions: Vec<Vec3>,
    pub uvs: Vec<Vec2>,
    pub indices: Vec<u32>,
}

/// Description of one model.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelDesc {
    /// Name, for diagnostics only.
    pub name: String,
    pub geometry: ModelGeometry,
    /// Material of a `Cube` model. Custom models carry it per surface.
    pub material_id: u32,
    /// Higher values are more transparent and never occlude lower ones.
    /// Source: meshers/blocky/voxel_mesher_blocky.h:133-136
    pub transparency_index: u8,
    pub is_transparent: bool,
    /// When false, this model never hides a neighbour's side.
    pub culls_neighbors: bool,
    pub contributes_to_ao: bool,
    pub is_random_tickable: bool,
    pub lod_skirts: bool,
    pub box_collision_mask: u32,
    pub tags_mask: u32,
    /// Index into the library's fluid list, or `None` for a solid model.
    pub fluid: Option<FluidRef>,
    pub color: [f32; 4],
}

/// Which fluid a model belongs to and at what level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FluidRef {
    pub fluid_index: u8,
    pub level: u8,
}

impl Default for ModelDesc {
    fn default() -> ModelDesc {
        ModelDesc {
            name: String::new(),
            geometry: ModelGeometry::Empty,
            material_id: 0,
            transparency_index: 0,
            is_transparent: false,
            culls_neighbors: true,
            contributes_to_ao: true,
            is_random_tickable: false,
            lod_skirts: false,
            box_collision_mask: 0,
            tags_mask: 0,
            fluid: None,
            color: [1.0, 1.0, 1.0, 1.0],
        }
    }
}

/// Description of one fluid.
#[derive(Debug, Clone, PartialEq)]
pub struct FluidDesc {
    pub name: String,
    pub material_id: u32,
    /// Highest level a voxel of this fluid can carry.
    pub max_level: u8,
    /// Fakes level 0 on voxels that can flow downwards, steepening slopes.
    /// Source: meshers/blocky/blocky_fluids_meshing_impl.h:299-317
    pub dip_when_flowing_down: bool,
}

impl Default for FluidDesc {
    fn default() -> FluidDesc {
        FluidDesc {
            name: String::new(),
            material_id: 0,
            max_level: 1,
            dip_when_flowing_down: false,
        }
    }
}

/// A full library description.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LibraryDesc {
    /// Model at index 0 must be the air model.
    pub models: Vec<ModelDesc>,
    pub fluids: Vec<FluidDesc>,
}

/// Why a description could not be baked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LibraryError {
    /// The library holds no models at all.
    NoModels,
    /// Index 0 must render nothing, it is the air id.
    /// Source: meshers/blocky/blocky_baked_library.h:31-33
    AirSlotNotEmpty,
    /// More models than the TYPE channel can address.
    TooManyModels(usize),
    /// More fluids than the fluid index can address.
    TooManyFluids(usize),
    /// A model references a fluid that does not exist.
    UnknownFluid { model: usize, fluid_index: u8 },
    /// A fluid level exceeds its fluid's `max_level`.
    FluidLevelTooHigh { model: usize, level: u8, max: u8 },
    /// A model uses more surfaces than `MAX_SURFACES`.
    TooManySurfaces { model: usize, count: usize },
    /// A side description names a side or surface that does not exist.
    InvalidSideRef { model: usize, side: usize, surface: usize },
    /// A surface's vertex attribute arrays have differing lengths.
    AttributeLengthMismatch { model: usize, surface: usize },
    /// An index addresses a vertex the surface does not have.
    IndexOutOfRange { model: usize, surface: usize },
}

/// Builds the quad of one cube side from the shared corner tables.
/// Source: constants/cube_tables.cpp:35-105, winding per SIDE_QUAD_TRIANGLES
pub fn build_cube_side(side: usize) -> SideSurface {
    let corners = SIDE_CORNERS[side];
    let positions: Vec<Vec3> = corners
        .iter()
        .map(|c| {
            let p = CORNER_POSITION[*c];
            Vec3::new(p.x, p.y, p.z)
        })
        .collect();
    // Unit-square UVs in corner order
    let uvs = vec![
        Vec2::new(0.0, 0.0),
        Vec2::new(1.0, 0.0),
        Vec2::new(1.0, 1.0),
        Vec2::new(0.0, 1.0),
    ];
    let indices = SIDE_QUAD_TRIANGLES[side].to_vec();
    let tangent = SIDE_TANGENTS[side];
    let mut tangents = Vec::with_capacity(16);
    for _ in 0..4 {
        tangents.extend_from_slice(&tangent);
    }
    SideSurface {
        positions,
        uvs,
        indices,
        tangents,
    }
}

/// Builds the six side quads of a fluid, with the top and bottom at the
/// fluid's own heights so `generate_fluid_model` can raise the top corners.
/// Source: meshers/blocky/blocky_fluids_meshing_impl.h:363-391
fn build_fluid_side(side: usize) -> FluidSurface {
    let quad = build_cube_side(side);
    let positions = quad
        .positions
        .iter()
        .map(|p| {
            let y = if p.y > 0.5 {
                BakedFluid::TOP_HEIGHT
            } else {
                BakedFluid::BOTTOM_HEIGHT
            };
            Vec3::new(p.x, y, p.z)
        })
        .collect();
    FluidSurface {
        positions,
        indices: quad.indices,
        tangents: quad.tangents,
    }
}

/// Validates one surface description.
fn check_surface(
    model: usize,
    surface: usize,
    positions: &[Vec3],
    normals: &[Vec3],
    uvs: &[Vec2],
    indices: &[u32],
) -> Result<(), LibraryError> {
    if (!normals.is_empty() && normals.len() != positions.len())
        || (!uvs.is_empty() && uvs.len() != positions.len())
    {
        return Err(LibraryError::AttributeLengthMismatch { model, surface });
    }
    let count = positions.len() as u32;
    if indices.iter().any(|i| *i >= count) {
        return Err(LibraryError::IndexOutOfRange { model, surface });
    }
    Ok(())
}

/// Bakes a description into the data the mesher reads.
///
/// Side pattern culling is derived from the models themselves. Every model
/// that renders a full cube on a side gets pattern 1 there, anything else
/// pattern 0, and pattern 1 occludes pattern 1. The C++ baker derives richer
/// patterns from the actual geometry (`voxel_blocky_library_base.cpp:512-515`),
/// which matters only for partially covering models.
pub fn bake(desc: &LibraryDesc) -> Result<BakedLibrary, LibraryError> {
    if desc.models.is_empty() {
        return Err(LibraryError::NoModels);
    }
    if desc.models.len() > MAX_MODELS {
        return Err(LibraryError::TooManyModels(desc.models.len()));
    }
    if desc.fluids.len() > MAX_FLUIDS {
        return Err(LibraryError::TooManyFluids(desc.fluids.len()));
    }
    if desc.models[AIR_ID as usize].geometry != ModelGeometry::Empty {
        return Err(LibraryError::AirSlotNotEmpty);
    }

    let fluids: Vec<BakedFluid> = desc
        .fluids
        .iter()
        .map(|f| BakedFluid {
            side_surfaces: core::array::from_fn(build_fluid_side),
            material_id: f.material_id,
            max_level: f.max_level.max(1),
            dip_when_flowing_down: f.dip_when_flowing_down,
        })
        .collect();

    let mut models = Vec::with_capacity(desc.models.len());
    let mut max_material_id = 0u32;

    for (model_index, m) in desc.models.iter().enumerate() {
        let mut baked = BakedModel {
            color: m.color,
            transparency_index: m.transparency_index,
            culls_neighbors: m.culls_neighbors,
            contributes_to_ao: m.contributes_to_ao,
            empty: true,
            is_random_tickable: m.is_random_tickable,
            is_transparent: m.is_transparent,
            fluid_index: NULL_FLUID_INDEX,
            fluid_level: 0,
            lod_skirts: m.lod_skirts,
            box_collision_mask: m.box_collision_mask,
            tags_mask: m.tags_mask,
            model: Model::default(),
        };

        if let Some(fluid_ref) = m.fluid {
            let Some(fluid) = fluids.get(fluid_ref.fluid_index as usize) else {
                return Err(LibraryError::UnknownFluid {
                    model: model_index,
                    fluid_index: fluid_ref.fluid_index,
                });
            };
            if fluid_ref.level > fluid.max_level {
                return Err(LibraryError::FluidLevelTooHigh {
                    model: model_index,
                    level: fluid_ref.level,
                    max: fluid.max_level,
                });
            }
            baked.fluid_index = fluid_ref.fluid_index;
            baked.fluid_level = fluid_ref.level;
        }

        match &m.geometry {
            ModelGeometry::Empty => {
                // Nothing to render, and nothing occludes a neighbour
                baked.model.empty_sides_mask = (1 << SIDE_COUNT) - 1;
            }
            ModelGeometry::Cube => {
                baked.empty = false;
                baked.model.surface_count = 1;
                baked.model.surfaces[0] = Surface {
                    material_id: m.material_id,
                    collision_enabled: true,
                    ..Surface::default()
                };
                for side in 0..SIDE_COUNT {
                    baked.model.sides_surfaces[side][0] = build_cube_side(side);
                }
                baked.model.full_sides_mask = (1 << SIDE_COUNT) - 1;
                baked.model.side_pattern_indices = [1; SIDE_COUNT];
                max_material_id = max_material_id.max(m.material_id);
            }
            ModelGeometry::Custom { surfaces, sides } => {
                if surfaces.len() > MAX_SURFACES {
                    return Err(LibraryError::TooManySurfaces {
                        model: model_index,
                        count: surfaces.len(),
                    });
                }
                baked.model.surface_count = surfaces.len();
                for (s, sd) in surfaces.iter().enumerate() {
                    check_surface(
                        model_index,
                        s,
                        &sd.positions,
                        &sd.normals,
                        &sd.uvs,
                        &sd.indices,
                    )?;
                    baked.model.surfaces[s] = Surface {
                        positions: sd.positions.clone(),
                        normals: sd.normals.clone(),
                        uvs: sd.uvs.clone(),
                        indices: sd.indices.clone(),
                        tangents: Vec::new(),
                        material_id: sd.material_id,
                        collision_enabled: sd.collision_enabled,
                    };
                    max_material_id = max_material_id.max(sd.material_id);
                    if !sd.indices.is_empty() {
                        baked.empty = false;
                    }
                }

                for sd in sides {
                    if sd.side >= SIDE_COUNT || sd.surface_index >= surfaces.len().max(1) {
                        return Err(LibraryError::InvalidSideRef {
                            model: model_index,
                            side: sd.side,
                            surface: sd.surface_index,
                        });
                    }
                    check_surface(
                        model_index,
                        sd.surface_index,
                        &sd.positions,
                        &[],
                        &sd.uvs,
                        &sd.indices,
                    )?;
                    baked.model.sides_surfaces[sd.side][sd.surface_index] = SideSurface {
                        positions: sd.positions.clone(),
                        uvs: sd.uvs.clone(),
                        indices: sd.indices.clone(),
                        tangents: Vec::new(),
                    };
                    if !sd.indices.is_empty() {
                        baked.empty = false;
                    }
                }

                // A side with no geometry on any surface is marked empty so
                // the mesher skips it outright
                for side in 0..SIDE_COUNT {
                    let has_geometry = baked.model.sides_surfaces[side]
                        .iter()
                        .any(|s| !s.is_empty());
                    if !has_geometry {
                        baked.model.empty_sides_mask |= 1 << side;
                    }
                }
            }
        }

        // Fluids render their top as an inner surface, so the +Y side is empty
        // Source: meshers/hybrid/voxel_mesher_hybrid.cpp:820-824
        if baked.is_fluid() {
            baked.model.empty_sides_mask |= 1 << SIDE_POSITIVE_Y;
            baked.empty = false;
            if baked.model.surface_count == 0 {
                baked.model.surface_count = 1;
                let material_id = fluids[baked.fluid_index as usize].material_id;
                baked.model.surfaces[0] = Surface {
                    material_id,
                    collision_enabled: false,
                    ..Surface::default()
                };
                max_material_id = max_material_id.max(material_id);
            }
            for side in 0..SIDE_COUNT {
                if side != SIDE_POSITIVE_Y {
                    baked.model.sides_surfaces[side][0] = SideSurface {
                        positions: build_fluid_side(side).positions,
                        uvs: vec![Vec2::ZERO; 4],
                        indices: build_cube_side(side).indices,
                        tangents: Vec::new(),
                    };
                }
            }
            // Fluids do not occlude solid neighbours
            baked.model.side_pattern_indices = [0; SIDE_COUNT];
        }

        models.push(baked);
    }

    // Pattern 0 means "does not fully cover", pattern 1 means "full cube".
    // Only a full cube occludes a full cube.
    let side_pattern_count = 2;
    let mut side_pattern_culling = vec![false; side_pattern_count * side_pattern_count];
    side_pattern_culling[1 + 1 * side_pattern_count] = true;

    Ok(BakedLibrary {
        side_pattern_culling,
        side_pattern_count,
        models,
        fluids,
        indexed_materials_count: max_material_id as usize + 1,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn air() -> ModelDesc {
        ModelDesc {
            name: "air".into(),
            geometry: ModelGeometry::Empty,
            ..ModelDesc::default()
        }
    }

    fn cube(name: &str, material_id: u32) -> ModelDesc {
        ModelDesc {
            name: name.into(),
            geometry: ModelGeometry::Cube,
            material_id,
            ..ModelDesc::default()
        }
    }

    #[test]
    fn an_empty_description_is_rejected() {
        assert_eq!(bake(&LibraryDesc::default()), Err(LibraryError::NoModels));
    }

    #[test]
    fn a_non_empty_air_slot_is_rejected() {
        let desc = LibraryDesc {
            models: vec![cube("stone", 0)],
            fluids: Vec::new(),
        };
        assert_eq!(bake(&desc), Err(LibraryError::AirSlotNotEmpty));
    }

    #[test]
    fn a_cube_model_bakes_six_sides() {
        let desc = LibraryDesc {
            models: vec![air(), cube("stone", 2)],
            fluids: Vec::new(),
        };
        let lib = bake(&desc).unwrap();
        assert_eq!(lib.models.len(), 2);
        assert!(lib.models[0].empty);
        assert!(!lib.models[1].empty);
        assert_eq!(lib.models[1].model.surface_count, 1);
        assert_eq!(lib.models[1].model.surfaces[0].material_id, 2);
        for side in 0..SIDE_COUNT {
            let s = &lib.models[1].model.sides_surfaces[side][0];
            assert_eq!(s.positions.len(), 4);
            assert_eq!(s.indices.len(), 6);
            assert_eq!(s.uvs.len(), 4);
            assert_eq!(s.tangents.len(), 16);
        }
        assert_eq!(lib.indexed_materials_count, 3);
    }

    #[test]
    fn cube_sides_lie_on_their_own_plane() {
        for side in 0..SIDE_COUNT {
            let quad = build_cube_side(side);
            let normal = voxel_core::cube::SIDE_NORMALS[side];
            let axis = if normal.x != 0 {
                0
            } else if normal.y != 0 {
                1
            } else {
                2
            };
            let expected = quad.positions[0][axis];
            for p in &quad.positions {
                assert_eq!(p[axis], expected, "side {side}");
            }
        }
    }

    #[test]
    fn two_cubes_occlude_each_other() {
        let desc = LibraryDesc {
            models: vec![air(), cube("a", 0), cube("b", 1)],
            fluids: Vec::new(),
        };
        let lib = bake(&desc).unwrap();
        assert!(!super::super::baked_library::is_face_visible(
            &lib,
            &lib.models[1],
            2,
            0
        ));
    }

    #[test]
    fn air_never_occludes_a_cube() {
        let desc = LibraryDesc {
            models: vec![air(), cube("a", 0)],
            fluids: Vec::new(),
        };
        let lib = bake(&desc).unwrap();
        assert!(super::super::baked_library::is_face_visible(
            &lib,
            &lib.models[1],
            0,
            0
        ));
    }

    #[test]
    fn a_transparent_model_does_not_occlude_an_opaque_one() {
        let mut glass = cube("glass", 1);
        glass.transparency_index = 3;
        glass.is_transparent = true;
        let desc = LibraryDesc {
            models: vec![air(), cube("stone", 0), glass],
            fluids: Vec::new(),
        };
        let lib = bake(&desc).unwrap();
        assert!(super::super::baked_library::is_face_visible(
            &lib,
            &lib.models[1],
            2,
            0
        ));
    }

    #[test]
    fn a_fluid_model_marks_its_top_side_empty() {
        let desc = LibraryDesc {
            models: vec![
                air(),
                ModelDesc {
                    name: "water".into(),
                    geometry: ModelGeometry::Empty,
                    fluid: Some(FluidRef {
                        fluid_index: 0,
                        level: 4,
                    }),
                    ..ModelDesc::default()
                },
            ],
            fluids: vec![FluidDesc {
                name: "water".into(),
                material_id: 1,
                max_level: 4,
                dip_when_flowing_down: true,
            }],
        };
        let lib = bake(&desc).unwrap();
        let m = &lib.models[1];
        assert!(m.is_fluid());
        assert_eq!(m.fluid_level, 4);
        assert!(!m.empty);
        assert_ne!(m.model.empty_sides_mask & (1 << SIDE_POSITIVE_Y), 0);
        assert_eq!(m.model.surfaces[0].material_id, 1);
        assert!(!m.model.surfaces[0].collision_enabled);
        // Lateral sides carry geometry
        assert_eq!(
            lib.models[1].model.sides_surfaces[voxel_core::cube::side_axis::SIDE_NEGATIVE_X][0]
                .positions
                .len(),
            4
        );
    }

    #[test]
    fn fluid_side_quads_use_the_fluid_heights() {
        let surface = build_fluid_side(voxel_core::cube::side_axis::SIDE_NEGATIVE_X);
        for p in &surface.positions {
            assert!(
                p.y == BakedFluid::BOTTOM_HEIGHT || p.y == BakedFluid::TOP_HEIGHT,
                "height {}",
                p.y
            );
        }
    }

    #[test]
    fn an_unknown_fluid_reference_is_rejected() {
        let desc = LibraryDesc {
            models: vec![
                air(),
                ModelDesc {
                    fluid: Some(FluidRef {
                        fluid_index: 9,
                        level: 1,
                    }),
                    ..ModelDesc::default()
                },
            ],
            fluids: Vec::new(),
        };
        assert_eq!(
            bake(&desc),
            Err(LibraryError::UnknownFluid {
                model: 1,
                fluid_index: 9
            })
        );
    }

    #[test]
    fn a_fluid_level_above_the_maximum_is_rejected() {
        let desc = LibraryDesc {
            models: vec![
                air(),
                ModelDesc {
                    fluid: Some(FluidRef {
                        fluid_index: 0,
                        level: 9,
                    }),
                    ..ModelDesc::default()
                },
            ],
            fluids: vec![FluidDesc {
                max_level: 4,
                ..FluidDesc::default()
            }],
        };
        assert_eq!(
            bake(&desc),
            Err(LibraryError::FluidLevelTooHigh {
                model: 1,
                level: 9,
                max: 4
            })
        );
    }

    #[test]
    fn a_custom_model_keeps_its_geometry() {
        let desc = LibraryDesc {
            models: vec![
                air(),
                ModelDesc {
                    name: "slab".into(),
                    geometry: ModelGeometry::Custom {
                        surfaces: vec![SurfaceDesc {
                            positions: vec![Vec3::ZERO, Vec3::X, Vec3::Y],
                            normals: vec![Vec3::Y; 3],
                            uvs: vec![Vec2::ZERO; 3],
                            indices: vec![0, 1, 2],
                            material_id: 5,
                            collision_enabled: true,
                        }],
                        sides: vec![SideDesc {
                            side: 0,
                            surface_index: 0,
                            positions: vec![Vec3::ZERO, Vec3::X, Vec3::Y],
                            uvs: vec![Vec2::ZERO; 3],
                            indices: vec![0, 1, 2],
                        }],
                    },
                    ..ModelDesc::default()
                },
            ],
            fluids: Vec::new(),
        };
        let lib = bake(&desc).unwrap();
        let m = &lib.models[1];
        assert!(!m.empty);
        assert_eq!(m.model.surface_count, 1);
        assert_eq!(m.model.surfaces[0].material_id, 5);
        assert_eq!(m.model.sides_surfaces[0][0].indices, vec![0, 1, 2]);
        // Sides without geometry are flagged empty
        for side in 1..SIDE_COUNT {
            assert_ne!(m.model.empty_sides_mask & (1 << side), 0, "side {side}");
        }
        assert_eq!(m.model.empty_sides_mask & 1, 0);
    }

    #[test]
    fn too_many_surfaces_are_rejected() {
        let desc = LibraryDesc {
            models: vec![
                air(),
                ModelDesc {
                    geometry: ModelGeometry::Custom {
                        surfaces: vec![SurfaceDesc::default(); MAX_SURFACES + 1],
                        sides: Vec::new(),
                    },
                    ..ModelDesc::default()
                },
            ],
            fluids: Vec::new(),
        };
        assert_eq!(
            bake(&desc),
            Err(LibraryError::TooManySurfaces {
                model: 1,
                count: MAX_SURFACES + 1
            })
        );
    }

    #[test]
    fn mismatched_attribute_lengths_are_rejected() {
        let desc = LibraryDesc {
            models: vec![
                air(),
                ModelDesc {
                    geometry: ModelGeometry::Custom {
                        surfaces: vec![SurfaceDesc {
                            positions: vec![Vec3::ZERO; 3],
                            // One normal too few
                            normals: vec![Vec3::Y; 2],
                            indices: vec![0, 1, 2],
                            ..SurfaceDesc::default()
                        }],
                        sides: Vec::new(),
                    },
                    ..ModelDesc::default()
                },
            ],
            fluids: Vec::new(),
        };
        assert_eq!(
            bake(&desc),
            Err(LibraryError::AttributeLengthMismatch {
                model: 1,
                surface: 0
            })
        );
    }

    #[test]
    fn an_out_of_range_index_is_rejected() {
        let desc = LibraryDesc {
            models: vec![
                air(),
                ModelDesc {
                    geometry: ModelGeometry::Custom {
                        surfaces: vec![SurfaceDesc {
                            positions: vec![Vec3::ZERO; 3],
                            indices: vec![0, 1, 7],
                            ..SurfaceDesc::default()
                        }],
                        sides: Vec::new(),
                    },
                    ..ModelDesc::default()
                },
            ],
            fluids: Vec::new(),
        };
        assert_eq!(
            bake(&desc),
            Err(LibraryError::IndexOutOfRange {
                model: 1,
                surface: 0
            })
        );
    }

    #[test]
    fn an_invalid_side_reference_is_rejected() {
        let desc = LibraryDesc {
            models: vec![
                air(),
                ModelDesc {
                    geometry: ModelGeometry::Custom {
                        surfaces: vec![SurfaceDesc::default()],
                        sides: vec![SideDesc {
                            side: 99,
                            ..SideDesc::default()
                        }],
                    },
                    ..ModelDesc::default()
                },
            ],
            fluids: Vec::new(),
        };
        assert!(matches!(
            bake(&desc),
            Err(LibraryError::InvalidSideRef { model: 1, .. })
        ));
    }

    #[test]
    fn the_air_model_reports_every_side_empty() {
        let desc = LibraryDesc {
            models: vec![air()],
            fluids: Vec::new(),
        };
        let lib = bake(&desc).unwrap();
        assert_eq!(lib.models[0].model.empty_sides_mask, 0b111111);
    }
}
