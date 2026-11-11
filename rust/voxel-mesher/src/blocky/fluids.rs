//! Fluid model generation.
//!
//! Ported from `meshers/blocky/blocky_fluids_meshing_impl.h`. Corner heights
//! are interpolated from the nine horizontal neighbours of a fluid voxel so
//! surfaces form slopes.

use glam::{Vec2, Vec3};
use voxel_core::cube::{side_axis::*, SIDE_COUNT};

use super::baked_library::{
    BakedFluid, BakedLibrary, BakedModel, SideSurface, Surface, AIR_ID, MAX_SURFACES,
};

/// Axis indices used as the X component of fluid UVs.
/// Source: util/math/constants.h:23
const AXIS_X: f32 = 0.0;
const AXIS_Y: f32 = 1.0;
const AXIS_Z: f32 = 2.0;

/// Flow direction of a fluid surface. Values are proportional to an angle,
/// named after a top-down OpenGL coordinate system.
/// Source: meshers/blocky/voxel_blocky_fluid.h:27-42
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FlowState {
    StraightPositiveX = 0,
    DiagonalPositiveXNegativeZ = 1,
    StraightNegativeZ = 2,
    DiagonalNegativeXNegativeZ = 3,
    StraightNegativeX = 4,
    DiagonalNegativeXPositiveZ = 5,
    StraightPositiveZ = 6,
    DiagonalPositiveXPositiveZ = 7,
    Idle = 8,
}

impl FlowState {
    pub fn as_f32(self) -> f32 {
        self as u8 as f32
    }
}

/// Maps the mask of corners at the minimum level to a flow direction.
/// Index 0 and the two ambiguous diagonal masks resolve to `Idle`.
/// Source: meshers/blocky/blocky_fluids_meshing_impl.h:57-92
const MIN_CORNERS_MASK_TO_FLOWSTATE: [FlowState; 16] = [
    FlowState::Idle, // 0000, impossible
    FlowState::DiagonalPositiveXPositiveZ,
    FlowState::DiagonalNegativeXPositiveZ,
    FlowState::StraightPositiveZ,
    FlowState::DiagonalNegativeXNegativeZ,
    FlowState::Idle, // 0101, ambiguous
    FlowState::StraightNegativeX,
    FlowState::DiagonalNegativeXPositiveZ,
    FlowState::DiagonalPositiveXNegativeZ,
    FlowState::StraightPositiveX,
    FlowState::Idle, // 1010, ambiguous
    FlowState::DiagonalPositiveXPositiveZ,
    FlowState::StraightNegativeZ,
    FlowState::DiagonalPositiveXNegativeZ,
    FlowState::DiagonalNegativeXNegativeZ,
    FlowState::Idle, // 1111
];

/// Derives the flow direction from the four top corner levels.
/// Source: meshers/blocky/blocky_fluids_meshing_impl.h:94-109
pub fn get_fluid_flow_state_from_corner_levels(corner_levels: [u8; 4]) -> FlowState {
    let min_level = corner_levels.iter().copied().min().unwrap_or(0);
    let mask = (((corner_levels[0] == min_level) as u8) << 3)
        | (((corner_levels[1] == min_level) as u8) << 2)
        | (((corner_levels[2] == min_level) as u8) << 1)
        | ((corner_levels[3] == min_level) as u8);
    MIN_CORNERS_MASK_TO_FLOWSTATE[mask as usize]
}

/// Derives the four top corner levels from the nine neighbour levels.
///
/// Neighbour layout, x to the left and z upwards:
/// ```text
///  8 7 6     z
///  5 4 3     |
///  2 1 0  x--o
/// ```
/// Source: meshers/blocky/blocky_fluids_meshing_impl.h:111-130
pub fn get_corner_levels_from_fluid_levels(fluid_levels: [u8; 9]) -> [u8; 4] {
    [
        fluid_levels[1]
            .max(fluid_levels[2])
            .max(fluid_levels[4])
            .max(fluid_levels[5]),
        fluid_levels[0]
            .max(fluid_levels[1])
            .max(fluid_levels[3])
            .max(fluid_levels[4]),
        fluid_levels[3]
            .max(fluid_levels[4])
            .max(fluid_levels[6])
            .max(fluid_levels[7]),
        fluid_levels[4]
            .max(fluid_levels[5])
            .max(fluid_levels[7])
            .max(fluid_levels[8]),
    ]
}

/// Maps corner levels to heights between the fluid's bottom and top height.
/// Source: meshers/blocky/blocky_fluids_meshing_impl.h:132-156
pub fn get_corner_heights_from_corner_levels(
    corner_levels: [u8; 4],
    fluid: &BakedFluid,
) -> [f32; 4] {
    let max_level_inv = 1.0 / fluid.max_level as f32;
    let level_to_height = |level: u8| {
        let t = level as f32 * max_level_inv;
        BakedFluid::BOTTOM_HEIGHT + (BakedFluid::TOP_HEIGHT - BakedFluid::BOTTOM_HEIGHT) * t
    };
    [
        level_to_height(corner_levels[0]),
        level_to_height(corner_levels[1]),
        level_to_height(corner_levels[2]),
        level_to_height(corner_levels[3]),
    ]
}

/// Flips the diagonal of a quad's two triangles.
///
/// Assumes the winding `{0, 2, 1, 0, 3, 2}`, producing `{0, 3, 1, 1, 3, 2}`.
/// Source: meshers/blocky/blocky_fluids_meshing_impl.h:168-175
pub fn transpose_quad_triangles(indices: &mut [u32]) {
    indices[1] = indices[4];
    indices[3] = indices[2];
}

/// Copies a fluid side into a model side surface, filling in a constant UV.
/// Source: meshers/blocky/blocky_fluids_meshing_impl.h:25-34
fn copy_side(src: &super::baked_library::FluidSurface, src_uv: Vec2, dst: &mut SideSurface) {
    dst.positions.clear();
    dst.positions.extend_from_slice(&src.positions);
    dst.uvs.clear();
    dst.uvs.resize(src.positions.len(), src_uv);
    dst.indices.clear();
    dst.indices.extend_from_slice(&src.indices);
    dst.tangents.clear();
    dst.tangents.extend_from_slice(&src.tangents);
}

/// Copies a fluid side into an inner surface with a constant normal.
/// Source: meshers/blocky/blocky_fluids_meshing_impl.h:36-55
fn copy_positions_normals_tangents(
    src: &super::baked_library::FluidSurface,
    normal: Vec3,
    material_id: u32,
    collision_enabled: bool,
    dst: &mut Surface,
) {
    dst.positions.clear();
    dst.positions.extend_from_slice(&src.positions);
    dst.normals.clear();
    dst.normals.resize(src.positions.len(), normal);
    dst.uvs.clear();
    dst.indices.clear();
    dst.indices.extend_from_slice(&src.indices);
    dst.tangents.clear();
    dst.tangents.extend_from_slice(&src.tangents);
    dst.material_id = material_id;
    dst.collision_enabled = collision_enabled;
}

/// Geometry a fluid voxel contributes, replacing the model's static geometry.
#[derive(Debug, Clone, Default)]
pub struct FluidModelOutput {
    /// Top surface of the fluid. Empty when the top is covered.
    pub top_surface: Surface,
    /// Side surfaces, indexed by the side enum.
    pub sides: [[SideSurface; MAX_SURFACES]; SIDE_COUNT],
    /// True when the top is covered by a voxel of the same fluid, in which case
    /// the model's own surfaces supply the material id instead.
    pub top_covered: bool,
}

/// Reads the TYPE channel by linear index for the neighbour lookups.
pub trait TypeBuffer {
    /// Returns the model id at a linear index, or `AIR_ID` when out of range.
    fn get(&self, index: isize) -> u64;
    fn is_available(&self) -> bool;
}

/// Type buffer backed by a decoded slice.
#[derive(Debug, Clone, Copy)]
pub struct SliceTypeBuffer<'a> {
    pub values: &'a [u64],
}

impl TypeBuffer for SliceTypeBuffer<'_> {
    #[inline]
    fn get(&self, index: isize) -> u64 {
        if index < 0 {
            AIR_ID as u64
        } else {
            self.values
                .get(index as usize)
                .copied()
                .unwrap_or(AIR_ID as u64)
        }
    }

    #[inline]
    fn is_available(&self) -> bool {
        !self.values.is_empty()
    }
}

/// Step sizes through the TYPE buffer towards each axis, in voxels.
/// ZXY layout gives `y = 1`, `x = size.y`, `z = size.y * size.x`.
/// Source: meshers/hybrid/voxel_mesher_hybrid.cpp:777-779
#[derive(Debug, Clone, Copy)]
pub struct BufferJumps {
    pub y: isize,
    pub x: isize,
    pub z: isize,
}

/// Builds the dynamic geometry of one fluid voxel.
///
/// Returns `None` when the voxel is fully covered by same-fluid neighbours and
/// contributes nothing.
/// Source: meshers/blocky/blocky_fluids_meshing_impl.h:177-410
pub fn generate_fluid_model<B: TypeBuffer>(
    voxel: &BakedModel,
    type_buffer: &B,
    voxel_index: isize,
    jumps: BufferJumps,
    visible_sides_mask: u8,
    library: &BakedLibrary,
) -> Option<FluidModelOutput> {
    let top_voxel_id = type_buffer.get(voxel_index + jumps.y);

    let mut fluid_top_covered = false;
    if let Some(top_model) = library.model(top_voxel_id) {
        if top_model.fluid_index == voxel.fluid_index {
            // The top side is covered
            if visible_sides_mask == 0 {
                // Fast path inside large water bodies, nothing to mesh
                return None;
            }
            fluid_top_covered = true;
        }
    }

    let fluid = library.fluids.get(voxel.fluid_index as usize)?;

    // Fluids have only one material
    const SURFACE_INDEX: usize = 0;

    let mut out = FluidModelOutput {
        top_covered: fluid_top_covered,
        ..FluidModelOutput::default()
    };

    // UVs encode the animation, x holds the side's axis and y the flow state.
    // Source: meshers/blocky/blocky_fluids_meshing_impl.h:220-223

    // Lateral sides always flow the same way
    for (side, axis) in [
        (SIDE_NEGATIVE_X, AXIS_X),
        (SIDE_POSITIVE_X, AXIS_X),
        (SIDE_NEGATIVE_Z, AXIS_Z),
        (SIDE_POSITIVE_Z, AXIS_Z),
    ] {
        copy_side(
            &fluid.side_surfaces[side],
            Vec2::new(axis, FlowState::StraightPositiveZ.as_f32()),
            &mut out.sides[side][SURFACE_INDEX],
        );
    }

    // The bottom side is always idle
    copy_side(
        &fluid.side_surfaces[SIDE_NEGATIVE_Y],
        Vec2::new(AXIS_Y, FlowState::Idle.as_f32()),
        &mut out.sides[SIDE_NEGATIVE_Y][SURFACE_INDEX],
    );

    if fluid_top_covered {
        // No top side
        out.sides[SIDE_POSITIVE_Y][SURFACE_INDEX].clear();
        out.top_surface.clear();
        return Some(out);
    }

    copy_positions_normals_tangents(
        &fluid.side_surfaces[SIDE_POSITIVE_Y],
        Vec3::Y,
        fluid.material_id,
        // Fluids carry no collision
        false,
        &mut out.top_surface,
    );

    // Neighbour levels, layout
    //  8 7 6     z
    //  5 4 3     |
    //  2 1 0  x--o
    let mut fluid_levels = [0u8; 9];
    let mut covered_neighbors = 0u32;
    let dip_when_flowing_down = fluid.dip_when_flowing_down;

    let mut i = 0usize;
    for dz in -1..=1i32 {
        for dx in -1..=1i32 {
            let nloc = voxel_index + dx as isize * jumps.x + dz as isize * jumps.z;
            let nid = type_buffer.get(nloc);

            let mut level = 0u8;
            if let Some(nm) = library.model(nid) {
                if nm.fluid_index == voxel.fluid_index {
                    level = nm.fluid_level;

                    // The current voxel is known not to be covered
                    if i != 4 {
                        let anid = type_buffer.get(nloc + jumps.y);
                        if anid != AIR_ID as u64 {
                            if let Some(anm) = library.model(anid) {
                                if anm.fluid_index == voxel.fluid_index {
                                    covered_neighbors |= 1 << i;
                                }
                            }
                        }
                    }

                    if dip_when_flowing_down
                        && nm.fluid_level != fluid.max_level
                        && (covered_neighbors & (1 << i)) == 0
                    {
                        // A non-covered fluid voxel above an area it can flow
                        // into fakes level 0 to form a steep slope. Max-level
                        // fluids sustain themselves, so lakes stay level.
                        // Source: blocky_fluids_meshing_impl.h:299-317
                        let bnid = type_buffer.get(nloc - jumps.y);
                        if bnid == AIR_ID as u64 {
                            level = 0;
                        } else if let Some(bnm) = library.model(bnid) {
                            if bnm.fluid_index == voxel.fluid_index {
                                level = 0;
                            }
                        }
                    }
                }
            }
            fluid_levels[i] = level;
            i += 1;
        }
    }

    // Adjust top corner heights to form slopes
    let corner_levels = get_corner_levels_from_fluid_levels(fluid_levels);
    let flow_state = get_fluid_flow_state_from_corner_levels(corner_levels);
    let mut corner_heights = get_corner_heights_from_corner_levels(corner_levels, fluid);

    // Covered neighbours count as full height
    // Source: meshers/blocky/blocky_fluids_meshing_impl.h:344-356
    if covered_neighbors & 0b000_001_011 != 0 {
        corner_heights[1] = 1.0;
    }
    if covered_neighbors & 0b000_100_110 != 0 {
        corner_heights[0] = 1.0;
    }
    if covered_neighbors & 0b011_001_000 != 0 {
        corner_heights[2] = 1.0;
    }
    if covered_neighbors & 0b110_100_000 != 0 {
        corner_heights[3] = 1.0;
    }

    out.top_surface.uvs.clear();
    out.top_surface
        .uvs
        .resize(4, Vec2::new(AXIS_Y, flow_state.as_f32()));

    // On lateral sides the top vertices are the last two, clockwise relative
    // to the top face.
    // Source: meshers/blocky/blocky_fluids_meshing_impl.h:363-384
    for (side, (a, b)) in [
        (SIDE_NEGATIVE_X, (2usize, 1usize)),
        (SIDE_POSITIVE_X, (0, 3)),
        (SIDE_NEGATIVE_Z, (1, 0)),
        (SIDE_POSITIVE_Z, (3, 2)),
    ] {
        let side_surface = &mut out.sides[side][SURFACE_INDEX];
        if side_surface.positions.len() >= 4 {
            side_surface.positions[2].y = corner_heights[a];
            side_surface.positions[3].y = corner_heights[b];
        }
    }

    // On the top side the vertices are counter-clockwise, the first at (+x, -z)
    if out.top_surface.positions.len() >= 4 {
        for c in 0..4 {
            out.top_surface.positions[c].y = corner_heights[c];
        }
    }

    // Keep the top quad's diagonal aligned with the flow
    if matches!(
        flow_state,
        FlowState::DiagonalPositiveXPositiveZ | FlowState::DiagonalNegativeXNegativeZ
    ) && out.top_surface.indices.len() >= 6
    {
        transpose_quad_triangles(&mut out.top_surface.indices);
    }

    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::baked_library::{FluidSurface, NULL_FLUID_INDEX};

    fn quad_surface() -> FluidSurface {
        FluidSurface {
            positions: vec![
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 1.0),
                Vec3::new(1.0, 1.0, 1.0),
            ],
            indices: vec![0, 2, 1, 0, 3, 2],
            tangents: vec![0.0; 16],
        }
    }

    fn fluid(max_level: u8, dip: bool) -> BakedFluid {
        BakedFluid {
            side_surfaces: core::array::from_fn(|_| quad_surface()),
            material_id: 3,
            max_level,
            dip_when_flowing_down: dip,
        }
    }

    fn fluid_model(fluid_index: u8, level: u8) -> BakedModel {
        BakedModel {
            empty: false,
            fluid_index,
            fluid_level: level,
            ..BakedModel::default()
        }
    }

    /// Library with model 0 air, model 1 the fluid at max level, model 2 solid.
    fn library(max_level: u8, dip: bool) -> BakedLibrary {
        BakedLibrary {
            side_pattern_count: 1,
            side_pattern_culling: vec![false],
            models: vec![
                BakedModel::default(),
                fluid_model(0, max_level),
                BakedModel {
                    empty: false,
                    ..BakedModel::default()
                },
            ],
            fluids: vec![fluid(max_level, dip)],
            indexed_materials_count: 1,
        }
    }

    const JUMPS: BufferJumps = BufferJumps {
        y: 1,
        x: 100,
        z: 10000,
    };

    /// Buffer where every index reads as air except the ones given.
    fn buffer(len: usize, overrides: &[(isize, u64)]) -> Vec<u64> {
        let mut v = vec![AIR_ID as u64; len];
        for (index, value) in overrides {
            v[*index as usize] = *value;
        }
        v
    }

    #[test]
    fn flow_state_values_match_cpp_ordering() {
        assert_eq!(FlowState::StraightPositiveX as u8, 0);
        assert_eq!(FlowState::Idle as u8, 8);
        assert_eq!(FlowState::StraightPositiveZ as u8, 6);
    }

    #[test]
    fn an_even_level_field_is_idle() {
        assert_eq!(
            get_fluid_flow_state_from_corner_levels([4, 4, 4, 4]),
            FlowState::Idle
        );
    }

    #[test]
    fn ambiguous_diagonal_masks_resolve_to_idle() {
        // Corners 0 and 2 at the minimum, mask 0b1010
        assert_eq!(
            get_fluid_flow_state_from_corner_levels([1, 5, 1, 5]),
            FlowState::Idle
        );
        // Corners 1 and 3 at the minimum, mask 0b0101
        assert_eq!(
            get_fluid_flow_state_from_corner_levels([5, 1, 5, 1]),
            FlowState::Idle
        );
    }

    #[test]
    fn a_single_low_corner_gives_a_diagonal_flow() {
        // Only corner 3 is at the minimum, mask 0b0001
        assert_eq!(
            get_fluid_flow_state_from_corner_levels([5, 5, 5, 1]),
            FlowState::DiagonalPositiveXPositiveZ
        );
    }

    #[test]
    fn corner_levels_take_the_maximum_of_four_neighbours() {
        //  8 7 6
        //  5 4 3
        //  2 1 0
        let mut levels = [0u8; 9];
        levels[2] = 7;
        let corners = get_corner_levels_from_fluid_levels(levels);
        // Neighbour 2 feeds corner 0 only
        assert_eq!(corners[0], 7);
        assert_eq!(corners[1], 0);
        assert_eq!(corners[2], 0);
        assert_eq!(corners[3], 0);
    }

    #[test]
    fn corner_heights_span_bottom_to_top() {
        let f = fluid(4, false);
        let heights = get_corner_heights_from_corner_levels([0, 4, 2, 4], &f);
        assert_eq!(heights[0], BakedFluid::BOTTOM_HEIGHT);
        assert_eq!(heights[1], BakedFluid::TOP_HEIGHT);
        let mid = (BakedFluid::BOTTOM_HEIGHT + BakedFluid::TOP_HEIGHT) * 0.5;
        assert!((heights[2] - mid).abs() < 1e-6);
    }

    #[test]
    fn transposing_a_quad_flips_its_diagonal() {
        let mut indices = vec![0u32, 2, 1, 0, 3, 2];
        transpose_quad_triangles(&mut indices);
        assert_eq!(indices, vec![0, 3, 1, 1, 3, 2]);
    }

    #[test]
    fn a_fully_covered_voxel_is_skipped() {
        let lib = library(4, false);
        let index = 50_000isize;
        // The voxel above is the same fluid, and no side is visible
        let values = buffer(100_000, &[(index, 1), (index + JUMPS.y, 1)]);
        let out = generate_fluid_model(
            &lib.models[1],
            &SliceTypeBuffer { values: &values },
            index,
            JUMPS,
            0,
            &lib,
        );
        assert!(out.is_none());
    }

    #[test]
    fn a_covered_voxel_with_visible_sides_drops_its_top() {
        let lib = library(4, false);
        let index = 50_000isize;
        let values = buffer(100_000, &[(index, 1), (index + JUMPS.y, 1)]);
        let out = generate_fluid_model(
            &lib.models[1],
            &SliceTypeBuffer { values: &values },
            index,
            JUMPS,
            0b000001,
            &lib,
        )
        .expect("geometry expected");
        assert!(out.top_covered);
        assert!(out.top_surface.is_empty());
        assert!(out.sides[SIDE_POSITIVE_Y][0].is_empty());
        // Lateral sides are still present
        assert!(!out.sides[SIDE_NEGATIVE_X][0].is_empty());
    }

    #[test]
    fn an_uncovered_voxel_gets_a_top_surface_with_the_fluid_material() {
        let lib = library(4, false);
        let index = 50_000isize;
        let values = buffer(100_000, &[(index, 1)]);
        let out = generate_fluid_model(
            &lib.models[1],
            &SliceTypeBuffer { values: &values },
            index,
            JUMPS,
            0b111111,
            &lib,
        )
        .expect("geometry expected");
        assert!(!out.top_covered);
        assert_eq!(out.top_surface.material_id, 3);
        assert!(!out.top_surface.collision_enabled);
        assert_eq!(out.top_surface.positions.len(), 4);
        assert_eq!(out.top_surface.uvs.len(), 4);
        // Normals point up
        assert!(out.top_surface.normals.iter().all(|n| *n == Vec3::Y));
    }

    #[test]
    fn an_isolated_voxel_sits_at_its_own_level() {
        let lib = library(4, false);
        let index = 50_000isize;
        // Only the voxel itself holds fluid, so neighbour levels are 0
        let values = buffer(100_000, &[(index, 1)]);
        let out = generate_fluid_model(
            &lib.models[1],
            &SliceTypeBuffer { values: &values },
            index,
            JUMPS,
            0b111111,
            &lib,
        )
        .unwrap();
        // Every corner reads the centre neighbour, which is max level
        for p in &out.top_surface.positions {
            assert!(
                (p.y - BakedFluid::TOP_HEIGHT).abs() < 1e-6,
                "height {}",
                p.y
            );
        }
    }

    #[test]
    fn a_covered_neighbour_raises_its_corners_to_full_height() {
        let lib = library(4, false);
        let index = 50_000isize;
        // Neighbour 0 is at (dx=-1, dz=-1) and is itself covered
        let n0 = index - JUMPS.x - JUMPS.z;
        let values = buffer(
            100_000,
            &[(index, 1), (n0, 1), (n0 + JUMPS.y, 1)],
        );
        let out = generate_fluid_model(
            &lib.models[1],
            &SliceTypeBuffer { values: &values },
            index,
            JUMPS,
            0b111111,
            &lib,
        )
        .unwrap();
        // Mask bit 0 raises corner 1 to 1.0
        assert_eq!(out.top_surface.positions[1].y, 1.0);
    }

    #[test]
    fn a_solid_neighbour_contributes_no_level() {
        let lib = library(4, false);
        let index = 50_000isize;
        // Neighbour 0 is the solid model 2
        let n0 = index - JUMPS.x - JUMPS.z;
        let values = buffer(100_000, &[(index, 1), (n0, 2)]);
        let out = generate_fluid_model(
            &lib.models[1],
            &SliceTypeBuffer { values: &values },
            index,
            JUMPS,
            0b111111,
            &lib,
        )
        .unwrap();
        // The centre still feeds every corner, so heights stay at the top
        assert!(out.top_surface.positions.iter().all(|p| p.y > 0.9));
    }

    #[test]
    fn a_non_fluid_model_has_no_fluid_entry() {
        let lib = library(4, false);
        let index = 50_000isize;
        let values = buffer(100_000, &[(index, 2)]);
        let solid = BakedModel {
            empty: false,
            fluid_index: NULL_FLUID_INDEX,
            ..BakedModel::default()
        };
        let out = generate_fluid_model(
            &solid,
            &SliceTypeBuffer { values: &values },
            index,
            JUMPS,
            0b111111,
            &lib,
        );
        assert!(out.is_none());
    }

    #[test]
    fn lateral_side_top_vertices_follow_the_corner_heights() {
        let lib = library(4, false);
        let index = 50_000isize;
        let values = buffer(100_000, &[(index, 1)]);
        let out = generate_fluid_model(
            &lib.models[1],
            &SliceTypeBuffer { values: &values },
            index,
            JUMPS,
            0b111111,
            &lib,
        )
        .unwrap();
        for side in [
            SIDE_NEGATIVE_X,
            SIDE_POSITIVE_X,
            SIDE_NEGATIVE_Z,
            SIDE_POSITIVE_Z,
        ] {
            let s = &out.sides[side][0];
            assert_eq!(s.positions.len(), 4);
            assert!((s.positions[2].y - BakedFluid::TOP_HEIGHT).abs() < 1e-6);
            assert!((s.positions[3].y - BakedFluid::TOP_HEIGHT).abs() < 1e-6);
        }
    }

    #[test]
    fn side_uvs_encode_the_axis_and_flow_state() {
        let lib = library(4, false);
        let index = 50_000isize;
        let values = buffer(100_000, &[(index, 1)]);
        let out = generate_fluid_model(
            &lib.models[1],
            &SliceTypeBuffer { values: &values },
            index,
            JUMPS,
            0b111111,
            &lib,
        )
        .unwrap();
        assert_eq!(out.sides[SIDE_NEGATIVE_X][0].uvs[0].x, AXIS_X);
        assert_eq!(out.sides[SIDE_NEGATIVE_Z][0].uvs[0].x, AXIS_Z);
        assert_eq!(out.sides[SIDE_NEGATIVE_Y][0].uvs[0].x, AXIS_Y);
        assert_eq!(
            out.sides[SIDE_NEGATIVE_Y][0].uvs[0].y,
            FlowState::Idle.as_f32()
        );
    }

    #[test]
    fn dip_when_flowing_down_lowers_a_neighbour_over_air() {
        let lib = library(4, true);
        let index = 50_000isize;
        // Neighbour 0 is a half-level fluid with air below
        let n0 = index - JUMPS.x - JUMPS.z;
        let mut values = buffer(100_000, &[(index, 1), (n0, 3)]);
        // Model 3 is the same fluid at level 1
        let mut lib = lib;
        lib.models.push(fluid_model(0, 1));
        values[n0 as usize] = 3;
        let out = generate_fluid_model(
            &lib.models[1],
            &SliceTypeBuffer { values: &values },
            index,
            JUMPS,
            0b111111,
            &lib,
        )
        .unwrap();
        // The centre voxel still holds max level, so corner 1 keeps its height
        assert!(out.top_surface.positions[1].y > 0.9);
    }
}
