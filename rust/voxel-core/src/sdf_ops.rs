//! SDF edit operations and voxel raycasting.
//!
//! Ported from `edition/funcs.h`, `edition/voxel_tool.cpp` and
//! `util/voxel_raycast.h`. Only the sphere, box and point operations plus the
//! raycast are present, which is the surface the separate `rame` project uses,
//! see doc/source/bevy_port_plan.md section 1.2 and section 4, P8.

use glam::{IVec3, Vec3};

use crate::channel::ChannelId;
use crate::constants::{SDF_FAR_INSIDE, SDF_FAR_OUTSIDE};
use crate::math::Box3i;
use crate::buffer::VoxelBuffer;

/// How an edit combines with the existing SDF.
/// Source: edition/funcs.h:662-667
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EditMode {
    /// Union, adds matter.
    #[default]
    Add,
    /// Relative complement, removes matter.
    Remove,
    /// Overwrites with the shape's own value.
    Set,
}

/// Signed distance to a sphere.
/// Source: util/math/sdf.h:38-41
#[inline]
pub fn sdf_sphere(pos: Vec3, center: Vec3, radius: f32) -> f32 {
    pos.distance(center) - radius
}

/// Signed distance to an axis-aligned box centred at the origin.
/// Source: util/math/sdf.h:22-36 in its scalar form
#[inline]
pub fn sdf_box(pos: Vec3, half_size: Vec3) -> f32 {
    let d = pos.abs() - half_size;
    d.max(Vec3::ZERO).length() + d.max_element().min(0.0)
}

/// Union of two SDFs.
/// Source: util/math/sdf.h:70-73
#[inline]
pub fn sdf_union(a: f32, b: f32) -> f32 {
    a.min(b)
}

/// Subtracts SDF `b` from SDF `a`.
/// Source: util/math/sdf.h:75-79
#[inline]
pub fn sdf_subtract(a: f32, b: f32) -> f32 {
    a.max(-b)
}

/// Combines a shape's value with the existing one per mode.
/// Source: edition/funcs.h:669-691
#[inline]
pub fn sdf_blend(src_value: f32, dst_value: f32, mode: EditMode) -> f32 {
    match mode {
        EditMode::Add => sdf_union(src_value, dst_value),
        // Relative complement, or difference
        EditMode::Remove => sdf_subtract(dst_value, src_value),
        EditMode::Set => src_value,
    }
}

/// Integer box an SDF sphere edit touches.
///
/// Two voxels of padding are added so the SDF has some margin, which the
/// `rame` project relies on.
/// Source: edition/funcs.h:316-323
pub fn get_sdf_sphere_box(center: Vec3, radius: f32) -> Box3i {
    let min = (center - Vec3::splat(radius)).floor().as_ivec3();
    let max = (center + Vec3::splat(radius)).ceil().as_ivec3();
    Box3i::from_min_max(min, max).padded(2)
}

/// Integer box an SDF box edit touches, with the same two-voxel margin.
/// Source: edition/funcs.h:426-435
pub fn get_sdf_box_box(center: Vec3, half_size: Vec3) -> Box3i {
    let min = (center - half_size).floor().as_ivec3();
    let max = (center + half_size).ceil().as_ivec3();
    Box3i::from_min_max(min, max).padded(2)
}

/// Writes a sphere into the SDF channel of a buffer.
///
/// `sdf_scale` scales the distance field, matching the C++ tool's setting.
/// Returns the box that was written.
/// Source: edition/voxel_tool.cpp:184-217, edition/funcs.h:327-341
pub fn do_sphere(
    voxels: &mut VoxelBuffer,
    center: Vec3,
    radius: f32,
    mode: EditMode,
    sdf_scale: f32,
    channel: ChannelId,
) -> Box3i {
    let area = get_sdf_sphere_box(center, radius).clipped(voxels.box_of_whole_buffer());
    let mut writes = Vec::with_capacity(area.volume() as usize);
    area.for_each_cell_zxy(|pos| {
        let d = sdf_scale * sdf_sphere(pos.as_vec3(), center, radius);
        let blended = sdf_blend(d, voxels.get_voxel_f(pos, channel), mode);
        writes.push((pos, blended));
    });
    for (pos, value) in writes {
        voxels.set_voxel_f(value, pos, channel).ok();
    }
    area
}

/// Writes an axis-aligned box into the SDF channel of a buffer.
///
/// The half size is derived as `(end - begin) * 0.5`, matching the C++ tool.
/// Source: edition/voxel_tool_lod_terrain.cpp:103-137
pub fn do_box(
    voxels: &mut VoxelBuffer,
    begin: IVec3,
    end: IVec3,
    mode: EditMode,
    sdf_scale: f32,
    channel: ChannelId,
) -> Box3i {
    let min = begin.min(end);
    let max = begin.max(end);
    let center = (min + max).as_vec3() * 0.5;
    let half_size = (max - min).as_vec3() * 0.5;

    let area = get_sdf_box_box(center, half_size).clipped(voxels.box_of_whole_buffer());
    let mut writes = Vec::with_capacity(area.volume() as usize);
    area.for_each_cell_zxy(|pos| {
        let d = sdf_scale * sdf_box(pos.as_vec3() - center, half_size);
        let blended = sdf_blend(d, voxels.get_voxel_f(pos, channel), mode);
        writes.push((pos, blended));
    });
    for (pos, value) in writes {
        voxels.set_voxel_f(value, pos, channel).ok();
    }
    area
}

/// Writes one voxel, fully inside or fully outside.
///
/// The written value is not a consistent SDF, which the C++ code notes as
/// acceptable for this operation.
/// Source: edition/voxel_tool.cpp:148-160
pub fn do_point(
    voxels: &mut VoxelBuffer,
    pos: IVec3,
    mode: EditMode,
    channel: ChannelId,
) -> Box3i {
    let value = if mode == EditMode::Remove {
        SDF_FAR_OUTSIDE
    } else {
        SDF_FAR_INSIDE
    };
    voxels.set_voxel_f(value, pos, channel).ok();
    Box3i::new(pos, IVec3::ONE)
}

/// State of the DDA walk when a cell is crossed.
///
/// View of the cell where the ray goes from A to B:
/// ```text
///    |       /|
///  --o------A-o--
///    |prev /  |
///    |    /   |
///    |   /    |
///  --o--B-----o--
///    |current |
/// ```
/// Source: util/voxel_raycast.h:12-32
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RaycastState {
    /// Grid position of the cell the ray comes from.
    pub hit_prev_position: IVec3,
    /// Distance along the ray at which the previous cell was entered.
    pub prev_distance: f32,
    /// Grid position of the cell just entered.
    pub hit_position: IVec3,
    /// Distance along the ray at which the hit cell is entered.
    pub distance: f32,
}

/// Result of a successful raycast.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RaycastHit {
    pub position: IVec3,
    pub previous_position: IVec3,
    pub distance_along_ray: f32,
    pub previous_distance_along_ray: f32,
}

/// Stand-in for infinity on axes the ray does not advance along.
/// Source: util/voxel_raycast.h:52
const INFINITE: f32 = 9_999_999.0;

/// Walks the voxel grid along a ray until `predicate` accepts a cell.
///
/// The grid is assumed to have unit cells and `ray_direction` must be
/// normalized. Returns `None` when `max_distance` is exceeded first.
/// Source: util/voxel_raycast.h:34-208
pub fn voxel_raycast<F>(
    ray_origin: Vec3,
    ray_direction: Vec3,
    mut predicate: F,
    max_distance: f32,
) -> Option<RaycastHit>
where
    F: FnMut(RaycastState) -> bool,
{
    if ray_origin.is_nan() || ray_direction.is_nan() || max_distance.is_nan() {
        return None;
    }
    // Must be normalized, otherwise the parametric distances are wrong
    debug_assert!((ray_direction.length() - 1.0).abs() < 1e-3);

    let mut hit_pos = ray_origin.floor().as_ivec3();
    // Assigned at the top of every loop iteration, before it is read
    let mut hit_prev_pos;

    // Step direction per axis
    let step = IVec3::new(
        if ray_direction.x > 0.0 {
            1
        } else if ray_direction.x < 0.0 {
            -1
        } else {
            0
        },
        if ray_direction.y > 0.0 {
            1
        } else if ray_direction.y < 0.0 {
            -1
        } else {
            0
        },
        if ray_direction.z > 0.0 {
            1
        } else if ray_direction.z < 0.0 {
            -1
        } else {
            0
        },
    );

    // Parametric distance covered by crossing one cell per axis
    let tdelta = Vec3::new(
        if step.x != 0 {
            1.0 / ray_direction.x.abs()
        } else {
            INFINITE
        },
        if step.y != 0 {
            1.0 / ray_direction.y.abs()
        } else {
            INFINITE
        },
        if step.z != 0 {
            1.0 / ray_direction.z.abs()
        } else {
            INFINITE
        },
    );

    // Parametric distance to the next grid line per axis
    let initial_cross = |origin: f32, step: i32, delta: f32| -> f32 {
        if step == 0 {
            INFINITE
        } else if step == 1 {
            (origin.ceil() - origin) * delta
        } else {
            (origin - origin.floor()) * delta
        }
    };
    let mut tcross = Vec3::new(
        initial_cross(ray_origin.x, step.x, tdelta.x),
        initial_cross(ray_origin.y, step.y, tdelta.y),
        initial_cross(ray_origin.z, step.z, tdelta.z),
    );

    // Integer start positions land exactly on a grid line. Advancing past it
    // keeps the walk from reporting the cell behind the ray.
    // Source: util/voxel_raycast.h:124-148
    if tcross.x == 0.0 {
        tcross.x += tdelta.x;
        if step.x == -1 {
            hit_pos.x -= 1;
        }
    }
    if tcross.y == 0.0 {
        tcross.y += tdelta.y;
        if step.y == -1 {
            hit_pos.y -= 1;
        }
    }
    if tcross.z == 0.0 {
        tcross.z += tdelta.z;
        if step.z == -1 {
            hit_pos.z -= 1;
        }
    }

    let mut t = 0.0f32;
    let mut t_prev;

    loop {
        hit_prev_pos = hit_pos;
        t_prev = t;

        if tcross.x < tcross.y {
            if tcross.x < tcross.z {
                hit_pos.x += step.x;
                if tcross.x > max_distance {
                    return None;
                }
                t = tcross.x;
                tcross.x += tdelta.x;
            } else {
                hit_pos.z += step.z;
                if tcross.z > max_distance {
                    return None;
                }
                t = tcross.z;
                tcross.z += tdelta.z;
            }
        } else if tcross.y < tcross.z {
            hit_pos.y += step.y;
            if tcross.y > max_distance {
                return None;
            }
            t = tcross.y;
            tcross.y += tdelta.y;
        } else {
            hit_pos.z += step.z;
            if tcross.z > max_distance {
                return None;
            }
            t = tcross.z;
            tcross.z += tdelta.z;
        }

        if predicate(RaycastState {
            hit_prev_position: hit_prev_pos,
            prev_distance: t_prev,
            hit_position: hit_pos,
            distance: t,
        }) {
            break;
        }
    }

    Some(RaycastHit {
        position: hit_pos,
        previous_position: hit_prev_pos,
        distance_along_ray: t,
        previous_distance_along_ray: t_prev,
    })
}

/// Raycasts against the SDF channel of a buffer, hitting the first solid voxel.
///
/// A voxel counts as solid when its SDF is negative, which is the convention
/// the generators write.
/// Source: meshers/hybrid/voxel_mesher_hybrid.cpp:458-459
pub fn raycast_sdf(
    voxels: &VoxelBuffer,
    ray_origin: Vec3,
    ray_direction: Vec3,
    max_distance: f32,
    channel: ChannelId,
) -> Option<RaycastHit> {
    voxel_raycast(
        ray_origin,
        ray_direction,
        |state| {
            voxels.is_position_valid(state.hit_position)
                && voxels.get_voxel_f(state.hit_position, channel) < 0.0
        },
        max_distance,
    )
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
    fn sphere_distance_is_zero_on_the_surface() {
        let center = Vec3::splat(5.0);
        assert_eq!(sdf_sphere(center, center, 3.0), -3.0);
        assert!((sdf_sphere(center + Vec3::X * 3.0, center, 3.0)).abs() < 1e-6);
        assert!(sdf_sphere(center + Vec3::X * 5.0, center, 3.0) > 0.0);
    }

    #[test]
    fn box_distance_is_negative_inside_and_positive_outside() {
        let half = Vec3::splat(2.0);
        assert!(sdf_box(Vec3::ZERO, half) < 0.0);
        assert!((sdf_box(Vec3::new(2.0, 0.0, 0.0), half)).abs() < 1e-6);
        assert!(sdf_box(Vec3::new(5.0, 0.0, 0.0), half) > 0.0);
        // Diagonal corner distance follows the euclidean metric
        let d = sdf_box(Vec3::splat(5.0), half);
        assert!((d - (Vec3::splat(3.0)).length()).abs() < 1e-5, "d is {d}");
    }

    #[test]
    fn union_takes_the_minimum_and_subtract_the_complement() {
        assert_eq!(sdf_union(-1.0, 2.0), -1.0);
        assert_eq!(sdf_subtract(-1.0, -2.0), 2.0);
        assert_eq!(sdf_subtract(1.0, 2.0), 1.0);
    }

    #[test]
    fn blending_follows_the_mode() {
        // Add unions, so the closer surface wins
        assert_eq!(sdf_blend(-1.0, 2.0, EditMode::Add), -1.0);
        // Remove carves the shape out of the existing matter
        assert_eq!(sdf_blend(-2.0, -1.0, EditMode::Remove), 2.0);
        // Set overwrites
        assert_eq!(sdf_blend(7.0, -1.0, EditMode::Set), 7.0);
    }

    #[test]
    fn the_sphere_box_carries_two_voxels_of_margin() {
        let b = get_sdf_sphere_box(Vec3::splat(10.0), 3.0);
        // The sphere spans 7..13, padded by 2 on each side
        assert_eq!(b.position, IVec3::splat(5));
        assert_eq!(b.end(), IVec3::splat(15));
    }

    #[test]
    fn the_box_box_carries_two_voxels_of_margin() {
        let b = get_sdf_box_box(Vec3::splat(10.0), Vec3::splat(2.0));
        assert_eq!(b.position, IVec3::splat(6));
        assert_eq!(b.end(), IVec3::splat(14));
    }

    #[test]
    fn adding_a_sphere_makes_its_interior_solid() {
        let mut b = buffer(16);
        let center = Vec3::splat(8.0);
        let area = do_sphere(&mut b, center, 3.0, EditMode::Add, 1.0, ChannelId::Sdf);
        assert!(!area.is_empty());
        // The centre is solid
        assert!(b.get_voxel_f(IVec3::splat(8), ChannelId::Sdf) < 0.0);
        // A voxel well outside stays air
        assert!(b.get_voxel_f(IVec3::splat(1), ChannelId::Sdf) > 0.0);
    }

    #[test]
    fn removing_a_sphere_carves_solid_matter() {
        let mut b = buffer(16);
        b.clear_channel_f(ChannelId::Sdf, -1.0);
        do_sphere(
            &mut b,
            Vec3::splat(8.0),
            3.0,
            EditMode::Remove,
            1.0,
            ChannelId::Sdf,
        );
        // The centre became air
        assert!(b.get_voxel_f(IVec3::splat(8), ChannelId::Sdf) > 0.0);
        // Matter outside the sphere remains
        assert!(b.get_voxel_f(IVec3::splat(1), ChannelId::Sdf) < 0.0);
    }

    #[test]
    fn a_sphere_edit_stays_inside_the_buffer() {
        let mut b = buffer(8);
        // A sphere centred on the corner, its box reaches outside
        let area = do_sphere(&mut b, Vec3::ZERO, 4.0, EditMode::Add, 1.0, ChannelId::Sdf);
        assert!(b.box_of_whole_buffer().contains_box(area));
        assert!(b.get_voxel_f(IVec3::ZERO, ChannelId::Sdf) < 0.0);
    }

    #[test]
    fn the_sdf_scale_scales_the_written_distance() {
        let mut b1 = buffer(16);
        let mut b2 = buffer(16);
        let center = Vec3::splat(8.0);
        do_sphere(&mut b1, center, 3.0, EditMode::Set, 1.0, ChannelId::Sdf);
        do_sphere(&mut b2, center, 3.0, EditMode::Set, 2.0, ChannelId::Sdf);
        let pos = IVec3::splat(8);
        let v1 = b1.get_voxel_f(pos, ChannelId::Sdf);
        let v2 = b2.get_voxel_f(pos, ChannelId::Sdf);
        assert!((v2 - v1 * 2.0).abs() < 1e-4, "v1 {v1}, v2 {v2}");
    }

    #[test]
    fn adding_a_box_makes_its_interior_solid() {
        let mut b = buffer(16);
        do_box(
            &mut b,
            IVec3::splat(4),
            IVec3::splat(10),
            EditMode::Add,
            1.0,
            ChannelId::Sdf,
        );
        assert!(b.get_voxel_f(IVec3::splat(7), ChannelId::Sdf) < 0.0);
        assert!(b.get_voxel_f(IVec3::splat(1), ChannelId::Sdf) > 0.0);
    }

    #[test]
    fn box_bounds_are_order_independent() {
        let mut b1 = buffer(16);
        let mut b2 = buffer(16);
        do_box(
            &mut b1,
            IVec3::splat(4),
            IVec3::splat(10),
            EditMode::Set,
            1.0,
            ChannelId::Sdf,
        );
        do_box(
            &mut b2,
            IVec3::splat(10),
            IVec3::splat(4),
            EditMode::Set,
            1.0,
            ChannelId::Sdf,
        );
        let pos = IVec3::splat(7);
        assert_eq!(
            b1.get_voxel_f(pos, ChannelId::Sdf),
            b2.get_voxel_f(pos, ChannelId::Sdf)
        );
    }

    #[test]
    fn a_point_edit_writes_one_voxel_only() {
        let mut b = buffer(8);
        let pos = IVec3::splat(3);
        let area = do_point(&mut b, pos, EditMode::Add, ChannelId::Sdf);
        assert_eq!(area, Box3i::new(pos, IVec3::ONE));
        assert_eq!(b.get_voxel_f(pos, ChannelId::Sdf), SDF_FAR_INSIDE);
        // The neighbour keeps its default
        assert!(b.get_voxel_f(pos + IVec3::X, ChannelId::Sdf) > 0.0);
    }

    #[test]
    fn removing_a_point_writes_the_far_outside_value() {
        let mut b = buffer(8);
        b.clear_channel_f(ChannelId::Sdf, -1.0);
        let pos = IVec3::splat(3);
        do_point(&mut b, pos, EditMode::Remove, ChannelId::Sdf);
        assert_eq!(b.get_voxel_f(pos, ChannelId::Sdf), SDF_FAR_OUTSIDE);
    }

    #[test]
    fn a_ray_along_an_axis_walks_one_cell_at_a_time() {
        let mut visited = Vec::new();
        let hit = voxel_raycast(
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::X,
            |state| {
                visited.push(state.hit_position);
                state.hit_position.x >= 4
            },
            100.0,
        )
        .expect("hit expected");
        assert_eq!(hit.position, IVec3::new(4, 0, 0));
        assert_eq!(visited, vec![
            IVec3::new(1, 0, 0),
            IVec3::new(2, 0, 0),
            IVec3::new(3, 0, 0),
            IVec3::new(4, 0, 0),
        ]);
        assert!((hit.distance_along_ray - 3.5).abs() < 1e-5);
    }

    #[test]
    fn a_ray_exceeding_the_distance_reports_no_hit() {
        let hit = voxel_raycast(Vec3::splat(0.5), Vec3::X, |_| false, 2.0);
        assert!(hit.is_none());
    }

    #[test]
    fn a_negative_direction_walks_backwards() {
        let hit = voxel_raycast(
            Vec3::new(5.5, 0.5, 0.5),
            -Vec3::X,
            |state| state.hit_position.x <= 2,
            100.0,
        )
        .unwrap();
        assert_eq!(hit.position, IVec3::new(2, 0, 0));
        assert_eq!(hit.previous_position, IVec3::new(3, 0, 0));
    }

    #[test]
    fn an_integer_start_position_does_not_report_the_cell_behind() {
        // Starting exactly on a grid line and going backwards
        let hit = voxel_raycast(
            Vec3::new(5.0, 0.5, 0.5),
            -Vec3::X,
            |state| state.hit_position.x <= 3,
            100.0,
        )
        .unwrap();
        assert_eq!(hit.position, IVec3::new(3, 0, 0));
    }

    #[test]
    fn a_diagonal_ray_crosses_cells_on_every_axis() {
        let dir = Vec3::splat(1.0).normalize();
        let mut visited = Vec::new();
        voxel_raycast(
            Vec3::splat(0.5),
            dir,
            |state| {
                visited.push(state.hit_position);
                visited.len() >= 6
            },
            100.0,
        )
        .unwrap();
        // Each step advances exactly one axis
        for pair in visited.windows(2) {
            let delta = pair[1] - pair[0];
            assert_eq!(delta.abs().element_sum(), 1, "step {delta} is not axial");
        }
    }

    #[test]
    fn nan_inputs_report_no_hit() {
        assert!(voxel_raycast(Vec3::splat(f32::NAN), Vec3::X, |_| true, 10.0).is_none());
        assert!(voxel_raycast(Vec3::ZERO, Vec3::X, |_| true, f32::NAN).is_none());
    }

    #[test]
    fn raycasting_an_sdf_buffer_hits_the_first_solid_voxel() {
        let mut b = buffer(16);
        // Make a solid wall at x = 8
        b.box_of_whole_buffer().for_each_cell_zxy(|pos| {
            if pos.x >= 8 {
                b.set_voxel_f(-1.0, pos, ChannelId::Sdf).unwrap();
            }
        });
        let hit = raycast_sdf(
            &b,
            Vec3::new(0.5, 4.5, 4.5),
            Vec3::X,
            100.0,
            ChannelId::Sdf,
        )
        .expect("hit expected");
        assert_eq!(hit.position, IVec3::new(8, 4, 4));
        // The previous cell is still air
        assert!(b.get_voxel_f(hit.previous_position, ChannelId::Sdf) > 0.0);
    }

    #[test]
    fn raycasting_an_empty_buffer_reports_no_hit() {
        let b = buffer(16);
        assert!(raycast_sdf(
            &b,
            Vec3::new(0.5, 4.5, 4.5),
            Vec3::X,
            100.0,
            ChannelId::Sdf
        )
        .is_none());
    }

    #[test]
    fn a_ray_leaving_the_buffer_reports_no_hit() {
        let mut b = buffer(16);
        b.clear_channel_f(ChannelId::Sdf, -1.0);
        // Pointing away from the buffer
        assert!(raycast_sdf(
            &b,
            Vec3::new(-5.5, 4.5, 4.5),
            -Vec3::X,
            100.0,
            ChannelId::Sdf
        )
        .is_none());
    }
}
