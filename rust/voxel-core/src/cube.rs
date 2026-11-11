//! Cube lookup tables ported from `constants/cube_tables.{h,cpp}`.
//!
//! Corner, edge and side conventions are kept byte for byte, including the two
//! axis inversions marked as known issues in the C++ source
//! (constants/cube_tables.cpp:64, :71).

use glam::{IVec3, Vec3};

/// Source: constants/cube_tables.h:19
pub const SIDE_COUNT: usize = 6;
/// Source: constants/cube_tables.h:55
pub const EDGE_COUNT: usize = 12;
/// Source: constants/cube_tables.h:69
pub const CORNER_COUNT: usize = 8;
/// Source: constants/cube_tables.h:93
pub const MOORE_NEIGHBORING_3D_COUNT: usize = 26;
/// Source: constants/cube_tables.h:96
pub const MOORE_AREA_3D_COUNT: usize = 27;
/// Source: constants/cube_tables.h:97
pub const MOORE_AREA_3D_CENTRAL_INDEX: usize = 13;

/// Side ordering used by the lookup tables.
/// Source: constants/cube_tables.h:10-20
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Side {
    Left = 0,
    Right = 1,
    Bottom = 2,
    Top = 3,
    Back = 4,
    Front = 5,
}

impl Side {
    pub const ALL: [Side; SIDE_COUNT] = [
        Side::Left,
        Side::Right,
        Side::Bottom,
        Side::Top,
        Side::Back,
        Side::Front,
    ];

    pub fn index(self) -> usize {
        self as usize
    }

    pub fn from_index(index: usize) -> Option<Side> {
        Side::ALL.get(index).copied()
    }
}

/// Axis-named aliases of the same ordering, used by blocky side masks.
/// Source: constants/cube_tables.h:31-38
pub mod side_axis {
    pub const SIDE_POSITIVE_X: usize = 0;
    pub const SIDE_NEGATIVE_X: usize = 1;
    pub const SIDE_NEGATIVE_Y: usize = 2;
    pub const SIDE_POSITIVE_Y: usize = 3;
    pub const SIDE_NEGATIVE_Z: usize = 4;
    pub const SIDE_POSITIVE_Z: usize = 5;
}

/// Edge ordering used by the lookup tables.
/// Source: constants/cube_tables.h:41-56
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Edge {
    BottomBack = 0,
    BottomRight = 1,
    BottomFront = 2,
    BottomLeft = 3,
    BackLeft = 4,
    BackRight = 5,
    FrontRight = 6,
    FrontLeft = 7,
    TopBack = 8,
    TopRight = 9,
    TopFront = 10,
    TopLeft = 11,
}

/// Corner ordering used by the lookup tables.
/// Source: constants/cube_tables.h:59-70
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Corner {
    BottomBackLeft = 0,
    BottomBackRight = 1,
    BottomFrontRight = 2,
    BottomFrontLeft = 3,
    TopBackLeft = 4,
    TopBackRight = 5,
    TopFrontRight = 6,
    TopFrontLeft = 7,
}

/// Source: constants/cube_tables.cpp:35-44
pub const CORNER_POSITION: [Vec3; CORNER_COUNT] = [
    Vec3::new(1.0, 0.0, 0.0),
    Vec3::new(0.0, 0.0, 0.0),
    Vec3::new(0.0, 0.0, 1.0),
    Vec3::new(1.0, 0.0, 1.0),
    Vec3::new(1.0, 1.0, 0.0),
    Vec3::new(0.0, 1.0, 0.0),
    Vec3::new(0.0, 1.0, 1.0),
    Vec3::new(1.0, 1.0, 1.0),
];

/// Source: constants/cube_tables.cpp:49-56
pub const SIDE_QUAD_TRIANGLES: [[u32; 6]; SIDE_COUNT] = [
    [0, 2, 1, 0, 3, 2],
    [0, 2, 1, 0, 3, 2],
    [0, 2, 1, 0, 3, 2],
    [0, 2, 1, 0, 3, 2],
    [0, 2, 1, 0, 3, 2],
    [0, 2, 1, 0, 3, 2],
];

/// Side normals. X and Z entries are inverted relative to their names, which is
/// the documented behaviour of the C++ table and must not be corrected here.
/// Source: constants/cube_tables.cpp:63-74
pub const SIDE_NORMALS: [IVec3; SIDE_COUNT] = [
    IVec3::new(1, 0, 0),
    IVec3::new(-1, 0, 0),
    IVec3::new(0, -1, 0),
    IVec3::new(0, 1, 0),
    IVec3::new(0, 0, -1),
    IVec3::new(0, 0, 1),
];

/// Source: constants/cube_tables.cpp:76-85
pub const SIDE_TANGENTS: [[f32; 4]; SIDE_COUNT] = [
    [0.0, 0.0, -1.0, 1.0],
    [0.0, 0.0, 1.0, 1.0],
    [1.0, 0.0, 0.0, 1.0],
    [-1.0, 0.0, 0.0, 1.0],
    [-1.0, 0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0, 1.0],
];

/// Source: constants/cube_tables.cpp:89-96
pub const SIDE_CORNERS: [[usize; 4]; SIDE_COUNT] = [
    [3, 0, 4, 7],
    [1, 2, 6, 5],
    [1, 0, 3, 2],
    [4, 5, 6, 7],
    [0, 1, 5, 4],
    [2, 3, 7, 6],
];

/// Source: constants/cube_tables.cpp:98-105
pub const SIDE_EDGES: [[usize; 4]; SIDE_COUNT] = [
    [3, 7, 11, 4],
    [1, 6, 9, 5],
    [0, 1, 2, 3],
    [8, 9, 10, 11],
    [0, 5, 8, 4],
    [2, 6, 10, 7],
];

/// Source: constants/cube_tables.cpp:119-129
pub const CORNER_INORMALS: [IVec3; CORNER_COUNT] = [
    IVec3::new(1, -1, -1),
    IVec3::new(-1, -1, -1),
    IVec3::new(-1, -1, 1),
    IVec3::new(1, -1, 1),
    IVec3::new(1, 1, -1),
    IVec3::new(-1, 1, -1),
    IVec3::new(-1, 1, 1),
    IVec3::new(1, 1, 1),
];

/// Source: constants/cube_tables.cpp:131-146
pub const EDGE_INORMALS: [IVec3; EDGE_COUNT] = [
    IVec3::new(0, -1, -1),
    IVec3::new(-1, -1, 0),
    IVec3::new(0, -1, 1),
    IVec3::new(1, -1, 0),
    IVec3::new(1, 0, -1),
    IVec3::new(-1, 0, -1),
    IVec3::new(-1, 0, 1),
    IVec3::new(1, 0, 1),
    IVec3::new(0, 1, -1),
    IVec3::new(-1, 1, 0),
    IVec3::new(0, 1, 1),
    IVec3::new(1, 1, 0),
];

/// Source: constants/cube_tables.cpp:148-152
pub const EDGE_CORNERS: [[usize; 2]; EDGE_COUNT] = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 4],
];

/// Opposite side per side index, expressed with the axis aliases.
/// Source: constants/cube_tables.cpp:221-228
pub const OPPOSITE_SIDE: [usize; SIDE_COUNT] = [
    side_axis::SIDE_NEGATIVE_X,
    side_axis::SIDE_POSITIVE_X,
    side_axis::SIDE_POSITIVE_Y,
    side_axis::SIDE_NEGATIVE_Y,
    side_axis::SIDE_POSITIVE_Z,
    side_axis::SIDE_NEGATIVE_Z,
];

/// Source: constants/cube_tables.cpp:155-185
pub const MOORE_NEIGHBORING_3D: [IVec3; MOORE_NEIGHBORING_3D_COUNT] = [
    IVec3::new(-1, -1, -1),
    IVec3::new(0, -1, -1),
    IVec3::new(1, -1, -1),
    IVec3::new(-1, -1, 0),
    IVec3::new(0, -1, 0),
    IVec3::new(1, -1, 0),
    IVec3::new(-1, -1, 1),
    IVec3::new(0, -1, 1),
    IVec3::new(1, -1, 1),
    IVec3::new(-1, 0, -1),
    IVec3::new(0, 0, -1),
    IVec3::new(1, 0, -1),
    IVec3::new(-1, 0, 0),
    IVec3::new(1, 0, 0),
    IVec3::new(-1, 0, 1),
    IVec3::new(0, 0, 1),
    IVec3::new(1, 0, 1),
    IVec3::new(-1, 1, -1),
    IVec3::new(0, 1, -1),
    IVec3::new(1, 1, -1),
    IVec3::new(-1, 1, 0),
    IVec3::new(0, 1, 0),
    IVec3::new(1, 1, 0),
    IVec3::new(-1, 1, 1),
    IVec3::new(0, 1, 1),
    IVec3::new(1, 1, 1),
];

/// Ordering matters, it is relied upon to iterate blocks deadlock-free.
/// Source: constants/cube_tables.cpp:187-219
pub const ORDERED_MOORE_AREA_3D: [IVec3; MOORE_AREA_3D_COUNT] = [
    IVec3::new(-1, -1, -1),
    IVec3::new(0, -1, -1),
    IVec3::new(1, -1, -1),
    IVec3::new(-1, 0, -1),
    IVec3::new(0, 0, -1),
    IVec3::new(1, 0, -1),
    IVec3::new(-1, 1, -1),
    IVec3::new(0, 1, -1),
    IVec3::new(1, 1, -1),
    IVec3::new(-1, -1, 0),
    IVec3::new(0, -1, 0),
    IVec3::new(1, -1, 0),
    IVec3::new(-1, 0, 0),
    IVec3::new(0, 0, 0),
    IVec3::new(1, 0, 0),
    IVec3::new(-1, 1, 0),
    IVec3::new(0, 1, 0),
    IVec3::new(1, 1, 0),
    IVec3::new(-1, -1, 1),
    IVec3::new(0, -1, 1),
    IVec3::new(1, -1, 1),
    IVec3::new(-1, 0, 1),
    IVec3::new(0, 0, 1),
    IVec3::new(1, 0, 1),
    IVec3::new(-1, 1, 1),
    IVec3::new(0, 1, 1),
    IVec3::new(1, 1, 1),
];

/// Maps a unit direction to its side index. Returns `Side::Front` for unknown
/// directions, matching the C++ fallback.
/// Source: constants/cube_tables.cpp:232-240
pub fn dir_to_side(d: IVec3) -> Side {
    let mut result = Side::Front;
    for (index, normal) in SIDE_NORMALS.iter().enumerate() {
        if *normal == d {
            result = Side::from_index(index).unwrap_or(Side::Front);
            break;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn side_axis_aliases_match_side_enum_order() {
        assert_eq!(side_axis::SIDE_POSITIVE_X, Side::Left.index());
        assert_eq!(side_axis::SIDE_NEGATIVE_X, Side::Right.index());
        assert_eq!(side_axis::SIDE_NEGATIVE_Y, Side::Bottom.index());
        assert_eq!(side_axis::SIDE_POSITIVE_Y, Side::Top.index());
        assert_eq!(side_axis::SIDE_NEGATIVE_Z, Side::Back.index());
        assert_eq!(side_axis::SIDE_POSITIVE_Z, Side::Front.index());
    }

    #[test]
    fn opposite_side_is_an_involution() {
        for side in 0..SIDE_COUNT {
            assert_eq!(OPPOSITE_SIDE[OPPOSITE_SIDE[side]], side);
        }
    }

    #[test]
    fn opposite_side_normals_are_negated() {
        for side in 0..SIDE_COUNT {
            assert_eq!(SIDE_NORMALS[OPPOSITE_SIDE[side]], -SIDE_NORMALS[side]);
        }
    }

    #[test]
    fn dir_to_side_round_trips_every_normal() {
        for side in Side::ALL {
            assert_eq!(dir_to_side(SIDE_NORMALS[side.index()]), side);
        }
    }

    #[test]
    fn side_corners_lie_on_the_side_plane() {
        for side in 0..SIDE_COUNT {
            let normal = SIDE_NORMALS[side];
            let axis = if normal.x != 0 {
                0
            } else if normal.y != 0 {
                1
            } else {
                2
            };
            // On a side, the coordinate along its axis is constant across corners
            let expected = CORNER_POSITION[SIDE_CORNERS[side][0]][axis];
            for corner in SIDE_CORNERS[side] {
                assert_eq!(CORNER_POSITION[corner][axis], expected, "side {side}");
            }
        }
    }

    #[test]
    fn moore_area_center_index_is_the_origin() {
        assert_eq!(ORDERED_MOORE_AREA_3D[MOORE_AREA_3D_CENTRAL_INDEX], IVec3::ZERO);
    }

    #[test]
    fn moore_neighboring_excludes_the_origin() {
        assert!(!MOORE_NEIGHBORING_3D.contains(&IVec3::ZERO));
        assert_eq!(MOORE_NEIGHBORING_3D.len(), MOORE_AREA_3D_COUNT - 1);
    }
}
