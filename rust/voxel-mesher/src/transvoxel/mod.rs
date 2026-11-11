//! Transvoxel isosurface extraction.
//!
//! Ported from `meshers/transvoxel/transvoxel.{h,cpp}`. The lookup tables in
//! `tables` are machine-generated from the C++ source, see
//! `rust/tools/convert_transvoxel_tables.py`.

pub mod materials;
pub mod tables;

use glam::{IVec3, Vec2, Vec3};
use voxel_core::buffer::zxy_index;
use voxel_core::{ChannelId, Depth, VoxelBuffer};

use materials::{
    DefaultTextureIndicesData, MaterialProcessor, Mixel4Processor, NullProcessor, TextureIndicesData,
    TexturingMode, WeightSamplerPackedU16,
};

/// Extra voxels needed towards the negative axes.
/// Source: meshers/transvoxel/transvoxel.h:16
pub const MIN_PADDING: i32 = 1;
/// Extra voxels needed towards the positive axes.
/// Source: meshers/transvoxel/transvoxel.h:18
pub const MAX_PADDING: i32 = 2;
/// Upper bound of triangles per 2x2x2 cell.
/// Source: meshers/transvoxel/transvoxel.h:20
pub const MAX_TRIANGLES_PER_CELL: usize = 5;

/// Scale of a transition cell relative to a full cell.
/// Source: meshers/transvoxel/transvoxel.cpp:22
const TRANSITION_CELL_SCALE: f32 = 0.25;

/// Per-vertex data needed for LOD stitching.
/// Source: meshers/transvoxel/transvoxel.h:37-54
#[derive(Debug, Clone, Copy, Default)]
pub struct LodAttrib {
    pub secondary_position: Vec3,
    /// Set when the vertex's cell touches a block side. Bit 0 is -X, bit 1 +X,
    /// bit 2 -Y, bit 3 +Y, bit 4 -Z, bit 5 +Z.
    pub cell_border_mask: u8,
    /// Same convention, set when the vertex itself is on a block side.
    pub vertex_border_mask: u8,
    /// Non-zero when the vertex belongs to a transition mesh.
    pub transition: u8,
}

/// Geometry and attributes produced for one surface.
/// Source: meshers/transvoxel/transvoxel.h:67-103
#[derive(Debug, Clone, Default)]
pub struct MeshArrays {
    pub vertices: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub lod_data: Vec<LodAttrib>,
    /// Material attribute, layout depends on the texturing mode.
    pub texturing_data_2f32: Vec<Vec2>,
    pub indices: Vec<u32>,
}

impl MeshArrays {
    pub fn clear(&mut self) {
        self.vertices.clear();
        self.normals.clear();
        self.lod_data.clear();
        self.texturing_data_2f32.clear();
        self.indices.clear();
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// Source: meshers/transvoxel/transvoxel.h:89-102
    fn add_vertex(
        &mut self,
        primary: Vec3,
        normal: Vec3,
        cell_border_mask: u8,
        vertex_border_mask: u8,
        transition: u8,
        secondary: Vec3,
    ) -> i32 {
        let vi = self.vertices.len() as i32;
        self.vertices.push(primary);
        self.normals.push(normal);
        self.lod_data.push(LodAttrib {
            secondary_position: secondary,
            cell_border_mask,
            vertex_border_mask,
            transition,
        });
        vi
    }
}

/// Vertex indices a regular cell can hand to its successors.
/// Source: meshers/transvoxel/transvoxel.h:105-108
#[derive(Debug, Clone, Copy)]
struct ReuseCell {
    vertices: [i32; 4],
    packed_texture_indices: u32,
}

impl Default for ReuseCell {
    fn default() -> ReuseCell {
        ReuseCell {
            vertices: [-1; 4],
            packed_texture_indices: 0,
        }
    }
}

/// Vertex indices a transition cell can hand to its successors.
/// Source: meshers/transvoxel/transvoxel.h:110-113
#[derive(Debug, Clone, Copy)]
struct ReuseTransitionCell {
    vertices: [i32; 12],
    packed_texture_indices: u32,
}

impl Default for ReuseTransitionCell {
    fn default() -> ReuseTransitionCell {
        ReuseTransitionCell {
            vertices: [-1; 12],
            packed_texture_indices: 0,
        }
    }
}

/// Two-deck vertex reuse cache, reset per block.
/// Source: meshers/transvoxel/transvoxel.h:115-157
#[derive(Debug, Default)]
pub struct Cache {
    decks: [Vec<ReuseCell>; 2],
    rows_2d: [Vec<ReuseTransitionCell>; 2],
    block_size: IVec3,
}

impl Cache {
    /// Source: meshers/transvoxel/transvoxel.h:117-127
    fn reset_reuse_cells(&mut self, block_size: IVec3) {
        self.block_size = block_size;
        let deck_area = (block_size.x * block_size.y) as usize;
        for deck in &mut self.decks {
            deck.clear();
            deck.resize(deck_area, ReuseCell::default());
        }
    }

    /// Source: meshers/transvoxel/transvoxel.h:129-137
    fn reset_reuse_cells_2d(&mut self, block_size: IVec3) {
        for row in &mut self.rows_2d {
            row.clear();
            row.resize(block_size.x as usize, ReuseTransitionCell::default());
        }
    }

    /// Source: meshers/transvoxel/transvoxel.h:139-144
    #[inline]
    fn reuse_cell_index(&self, pos: IVec3) -> (usize, usize) {
        let j = (pos.z & 1) as usize;
        let i = (pos.y * self.block_size.x + pos.x) as usize;
        (j, i)
    }

    #[inline]
    fn get_reuse_cell(&self, pos: IVec3) -> ReuseCell {
        let (j, i) = self.reuse_cell_index(pos);
        self.decks[j][i]
    }

    #[inline]
    fn set_reuse_cell(&mut self, pos: IVec3, cell: ReuseCell) {
        let (j, i) = self.reuse_cell_index(pos);
        self.decks[j][i] = cell;
    }

    /// Source: meshers/transvoxel/transvoxel.h:146-151
    #[inline]
    fn get_reuse_cell_2d(&self, x: i32, y: i32) -> ReuseTransitionCell {
        self.rows_2d[(y & 1) as usize][x as usize]
    }

    #[inline]
    fn set_reuse_cell_2d(&mut self, x: i32, y: i32, cell: ReuseTransitionCell) {
        self.rows_2d[(y & 1) as usize][x as usize] = cell;
    }
}

/// Position and triangle count of one cell that produced geometry.
/// Source: meshers/transvoxel/transvoxel.h:167-170
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CellInfo {
    pub position: IVec3,
    pub triangle_count: u32,
}

/// SDF values considered negative have a sign bit of 1 in this algorithm.
/// Source: meshers/transvoxel/transvoxel.cpp:25-27
#[inline]
fn sign_f(v: f32) -> u8 {
    (v < 0.0) as u8
}

/// Offset making room for transition cells inside boundary cells.
/// Source: meshers/transvoxel/transvoxel.cpp:29-59
fn get_border_offset(pos_scaled: Vec3, lod_index: u32, block_size_non_scaled: IVec3) -> Vec3 {
    let mut delta = Vec3::ZERO;

    let p2k = (1 << lod_index) as f32; // 2 ^ lod
    let p2mk = 1.0 / p2k; // 2 ^ (-lod)
    let wk = TRANSITION_CELL_SCALE * p2k; // 2 ^ (lod - 2) at scale 0.25

    for i in 0..3 {
        let p = pos_scaled[i];
        let s = block_size_non_scaled[i] as f32;

        if p < p2k {
            // Inside the minimum cell
            delta[i] = (1.0 - p2mk * p) * wk;
        } else if p > (p2k * (s - 1.0)) {
            // Inside the maximum cell
            delta[i] = (s - 1.0 - p2mk * p) * wk;
        }
    }

    delta
}

/// Projects the border offset onto the plane of the normal.
/// Source: meshers/transvoxel/transvoxel.cpp:61-75
#[inline]
fn project_border_offset(delta: Vec3, normal: Vec3) -> Vec3 {
    Vec3::new(
        (1.0 - normal.x * normal.x) * delta.x
            - normal.y * normal.x * delta.y
            - normal.z * normal.x * delta.z,
        -normal.x * normal.y * delta.x + (1.0 - normal.y * normal.y) * delta.y
            - normal.z * normal.y * delta.z,
        -normal.x * normal.z * delta.x - normal.y * normal.z * delta.y
            + (1.0 - normal.z * normal.z) * delta.z,
    )
}

/// Secondary position a vertex moves to when a transition mesh is stitched in.
/// Source: meshers/transvoxel/transvoxel.cpp:77-92
fn get_secondary_position(
    primary: Vec3,
    normal: Vec3,
    lod_index: u32,
    block_size_non_scaled: IVec3,
) -> Vec3 {
    let mut delta = get_border_offset(primary, lod_index, block_size_non_scaled);
    delta = project_border_offset(delta, normal);

    // At very low LOD levels the error can send secondary positions far off
    let p2k = (1 << lod_index) as f32;
    delta = delta.clamp(Vec3::splat(-p2k), Vec3::splat(p2k));

    primary + delta
}

/// Bit mask of the block sides a position touches.
/// Source: meshers/transvoxel/transvoxel.cpp:94-116
#[inline]
fn get_border_mask(pos: IVec3, block_size: IVec3) -> u8 {
    let mut mask = 0u8;
    for i in 0..3 {
        if pos[i] == 0 {
            mask |= 1 << (i * 2);
        }
        if pos[i] == block_size[i] {
            mask |= 1 << (i * 2 + 1);
        }
    }
    mask
}

/// Normalizes a vector, falling back to +Y for the zero vector.
/// Source: meshers/transvoxel/transvoxel.cpp:118-126
#[inline]
fn normalized_not_null(n: Vec3) -> Vec3 {
    let length_sq = n.length_squared();
    if length_sq == 0.0 {
        Vec3::new(0.0, 1.0, 0.0)
    } else {
        n / length_sq.sqrt()
    }
}

/// Direction code to the offset of the preceding cell.
/// Source: meshers/transvoxel/transvoxel.cpp:128-131
#[inline]
fn dir_to_prev_vec(dir: u8) -> IVec3 {
    IVec3::new(
        -((dir & 1) as i32),
        -(((dir >> 1) & 1) as i32),
        -(((dir >> 2) & 1) as i32),
    )
}

/// SDF sample source of one block, decoded once up front.
///
/// Two forms are kept per sample, matching the C++ code which reads the raw
/// channel values for the early cell rejection (`transvoxel.cpp:242`) and the
/// negated values for case selection and interpolation (`:279`). Keeping only
/// the negated form would invert the rejection comparison, which the comment
/// at `transvoxel.cpp:235-241` states must stay `>` on the raw values so a
/// surface lining up with integer coordinates is still extracted.
///
/// The C++ version templates over the raw channel type to avoid per-sample
/// branches (`transvoxel.cpp:185`, `:1167-1230`). Decoding once up front keeps
/// the hot loops monomorphic here as well.
#[derive(Debug, Clone)]
pub struct SdfField {
    /// Negated real SDF values, in ZXY order. Negative means outside.
    /// Source: meshers/transvoxel/transvoxel.cpp:133-147, `sdf_as_float`
    values: Vec<f32>,
    /// Raw channel values as reals, in ZXY order, used for cell rejection.
    /// Source: meshers/transvoxel/transvoxel.cpp:242
    raw_values: Vec<f32>,
    size: IVec3,
}

impl SdfField {
    /// Decodes an SDF channel of a buffer, uniform or dense.
    /// Source: meshers/transvoxel/transvoxel.cpp:1092-1113
    pub fn from_buffer(voxels: &VoxelBuffer, channel: ChannelId) -> SdfField {
        let size = voxels.size();
        let volume = voxels.volume();
        let depth = voxels.depth(channel);

        let mut values = Vec::with_capacity(volume);
        let mut raw_values = Vec::with_capacity(volume);
        match voxels.uniform_value(channel) {
            Some(_) => {
                let raw = voxels.get_voxel_f(IVec3::ZERO, channel);
                // `sdf_as_float` negates, so the whole block holds one value
                values.resize(volume, -raw);
                raw_values.resize(volume, raw);
            }
            None => {
                let bytes = voxels
                    .channel_as_bytes(channel)
                    .expect("dense channel has bytes");
                let step = depth.byte_count() as usize;
                for i in 0..volume {
                    let raw_bits = read_raw_at(bytes, i, step, depth);
                    let raw = voxel_core::channel::raw_voxel_to_real(raw_bits, depth);
                    values.push(-raw);
                    raw_values.push(raw);
                }
            }
        }

        SdfField {
            values,
            raw_values,
            size,
        }
    }

    #[inline]
    pub fn size(&self) -> IVec3 {
        self.size
    }

    #[inline]
    pub fn get(&self, index: usize) -> f32 {
        self.values[index]
    }

    /// Raw channel value, used only for the early cell rejection comparison.
    /// Source: meshers/transvoxel/transvoxel.cpp:242
    #[inline]
    pub fn get_raw(&self, index: usize) -> f32 {
        self.raw_values[index]
    }

    /// Central-difference gradient at a sample, pointing out of the surface.
    /// Source: meshers/transvoxel/transvoxel.cpp:149-164
    #[inline]
    fn corner_gradient(&self, data_index: usize) -> Vec3 {
        let n010 = 1usize; // Y+1
        let n100 = self.size.y as usize; // X+1
        let n001 = (self.size.y * self.size.x) as usize; // Z+1

        let nx = self.values[data_index - n100];
        let px = self.values[data_index + n100];
        let ny = self.values[data_index - n010];
        let py = self.values[data_index + n010];
        let nz = self.values[data_index - n001];
        let pz = self.values[data_index + n001];

        Vec3::new(nx - px, ny - py, nz - pz)
    }
}

/// Reads one raw value out of a dense channel's bytes.
#[inline]
fn read_raw_at(bytes: &[u8], index: usize, step: usize, depth: Depth) -> u64 {
    let o = index * step;
    match depth {
        Depth::D8 => bytes[o] as u64,
        Depth::D16 => u16::from_ne_bytes([bytes[o], bytes[o + 1]]) as u64,
        Depth::D32 => {
            u32::from_ne_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]) as u64
        }
        Depth::D64 => u64::from_ne_bytes([
            bytes[o],
            bytes[o + 1],
            bytes[o + 2],
            bytes[o + 3],
            bytes[o + 4],
            bytes[o + 5],
            bytes[o + 6],
            bytes[o + 7],
        ]),
    }
}

/// Builds the regular mesh of one block.
///
/// `sdf` must cover the block plus `MIN_PADDING` on the negative and
/// `MAX_PADDING` on the positive axes.
/// Source: meshers/transvoxel/transvoxel.cpp:185-601
pub fn build_regular_mesh<P: MaterialProcessor>(
    sdf: &SdfField,
    material_processor: &mut P,
    lod_index: u32,
    cache: &mut Cache,
    output: &mut MeshArrays,
    mut cell_info: Option<&mut Vec<CellInfo>>,
    edge_clamp_margin: f32,
) {
    let edge_clamp_margin_max = 1.0 - edge_clamp_margin;

    let block_size_with_padding = sdf.size();
    let block_size = block_size_with_padding - IVec3::splat(MIN_PADDING + MAX_PADDING);
    let block_size_scaled = block_size << lod_index;

    cache.reset_reuse_cells(block_size_with_padding);

    // Cells reach one voxel further for normals, hence the padded bounds
    let min_pos = IVec3::splat(MIN_PADDING);
    let max_pos = block_size_with_padding - IVec3::splat(MAX_PADDING);

    // Steps in the data array towards neighbour voxels
    let n010 = 1usize;
    let n100 = block_size_with_padding.y as usize;
    let n001 = (block_size_with_padding.y * block_size_with_padding.x) as usize;
    let n110 = n010 + n100;
    let n101 = n100 + n001;
    let n011 = n010 + n001;
    let n111 = n100 + n010 + n001;

    // The isolevel of the negated field is zero
    let isolevel = 0.0f32;

    for z in min_pos.z..max_pos.z {
        for y in min_pos.y..max_pos.y {
            let mut data_index = zxy_index(
                min_pos.x as u32,
                y as u32,
                z as u32,
                block_size_with_padding,
            );

            for x in min_pos.x..max_pos.x {
                let pos = IVec3::new(x, y, z);

                // The comparison direction matters for cases where four
                // samples equal the isolevel and four do not. `>` must match
                // the comparison used for case selection.
                // Source: meshers/transvoxel/transvoxel.cpp:235-241
                let s = sdf.get_raw(data_index) > isolevel;
                if (sdf.get_raw(data_index + n010) > isolevel) == s
                    && (sdf.get_raw(data_index + n100) > isolevel) == s
                    && (sdf.get_raw(data_index + n110) > isolevel) == s
                    && (sdf.get_raw(data_index + n001) > isolevel) == s
                    && (sdf.get_raw(data_index + n011) > isolevel) == s
                    && (sdf.get_raw(data_index + n101) > isolevel) == s
                    && (sdf.get_raw(data_index + n111) > isolevel) == s
                {
                    // Not crossing the isolevel, no geometry here
                    data_index += n100;
                    continue;
                }

                //    6-------7
                //   /|      /|
                //  / |     / |  Corners
                // 4-------5  |
                // |  2----|--3
                // | /     | /   z y
                // |/      |/    |/
                // 0-------1     o--x
                let corner_data_indices: [usize; 8] = [
                    data_index,
                    data_index + n100,
                    data_index + n010,
                    data_index + n110,
                    data_index + n001,
                    data_index + n101,
                    data_index + n011,
                    data_index + n111,
                ];

                let mut cell_samples_sdf = [0.0f32; 8];
                for i in 0..8 {
                    cell_samples_sdf[i] = sdf.get(corner_data_indices[i]);
                }

                // Case code from the sign bits, corner 0 is the least
                // significant bit
                let mut case_code = sign_f(cell_samples_sdf[0]);
                for i in 1..8 {
                    case_code |= sign_f(cell_samples_sdf[i]) << i;
                }

                if case_code == 0 || case_code == 255 {
                    data_index += n100;
                    continue;
                }

                let mut current_reuse_cell = cache.get_reuse_cell(pos);

                let padded_corner_positions: [IVec3; 8] = [
                    IVec3::new(x, y, z),
                    IVec3::new(x + 1, y, z),
                    IVec3::new(x, y + 1, z),
                    IVec3::new(x + 1, y + 1, z),
                    IVec3::new(x, y, z + 1),
                    IVec3::new(x + 1, y, z + 1),
                    IVec3::new(x, y + 1, z + 1),
                    IVec3::new(x + 1, y + 1, z + 1),
                ];

                current_reuse_cell.packed_texture_indices =
                    material_processor.on_cell(&corner_data_indices, case_code);

                // Undo padding, from here positions are actual positions
                let mut corner_positions = [IVec3::ZERO; 8];
                for i in 0..8 {
                    corner_positions[i] = (padded_corner_positions[i] - min_pos) << lod_index;
                }

                // Cells on the minimal boundaries have no preceding cells, so
                // new vertex creation is allowed on additional edges there
                // Source: meshers/transvoxel/transvoxel.cpp:324-330
                let direction_validity_mask = (if x > min_pos.x { 1u8 } else { 0 })
                    | (if y > min_pos.y { 1u8 } else { 0 } << 1)
                    | (if z > min_pos.z { 1u8 } else { 0 } << 2);

                let regular_cell_class_index = tables::REGULAR_CELL_CLASS[case_code as usize];
                let regular_cell_data = &tables::REGULAR_CELL_DATA[regular_cell_class_index as usize];
                let triangle_count = regular_cell_data.triangle_count();
                let vertex_count = regular_cell_data.vertex_count();

                let mut cell_vertex_indices = [-1i32; 12];

                let cell_border_mask = get_border_mask(pos - min_pos, block_size - IVec3::ONE);

                for vertex_index in 0..vertex_count {
                    // Low byte holds the edge's corner indices, high byte the
                    // vertex reuse mapping
                    // Source: meshers/transvoxel/transvoxel.cpp:346-351
                    let rvd = tables::REGULAR_VERTEX_DATA[case_code as usize][vertex_index];
                    let edge_code_low = (rvd & 0xff) as u8;
                    let edge_code_high = ((rvd >> 8) & 0xff) as u8;

                    let v0 = ((edge_code_low >> 4) & 0xf) as usize;
                    let v1 = (edge_code_low & 0xf) as usize;
                    debug_assert!(v1 > v0);

                    let sample0 = cell_samples_sdf[v0]; // d0 in the paper
                    let sample1 = cell_samples_sdf[v1]; // d1 in the paper

                    let t = (sample1 / (sample1 - sample0))
                        .clamp(edge_clamp_margin, edge_clamp_margin_max);

                    let p0 = corner_positions[v0];
                    let p1 = corner_positions[v1];

                    if t > 0.0 && t < 1.0 {
                        // Vertex lies inside the edge
                        let reuse_dir = (edge_code_high >> 4) & 0xf;
                        let reuse_vertex_index = (edge_code_high & 0xf) as usize;

                        let present = (reuse_dir & direction_validity_mask) == reuse_dir;

                        if present {
                            let cache_pos = pos + dir_to_prev_vec(reuse_dir);
                            let prev_cell = cache.get_reuse_cell(cache_pos);
                            if prev_cell.packed_texture_indices
                                == current_reuse_cell.packed_texture_indices
                            {
                                cell_vertex_indices[vertex_index] =
                                    prev_cell.vertices[reuse_vertex_index];
                            }
                        }

                        if !present || cell_vertex_indices[vertex_index] == -1 {
                            let t0 = t;
                            let t1 = 1.0 - t;

                            let primaryf = p0.as_vec3() * t0 + p1.as_vec3() * t1;
                            let cg0 = sdf.corner_gradient(corner_data_indices[v0]);
                            let cg1 = sdf.corner_gradient(corner_data_indices[v1]);
                            let normal = normalized_not_null(cg0 * t0 + cg1 * t1);

                            let mut secondary = Vec3::ZERO;
                            let mut vertex_border_mask = 0u8;
                            if cell_border_mask > 0 {
                                secondary =
                                    get_secondary_position(primaryf, normal, lod_index, block_size);
                                vertex_border_mask = get_border_mask(p0, block_size_scaled)
                                    & get_border_mask(p1, block_size_scaled);
                            }

                            cell_vertex_indices[vertex_index] = output.add_vertex(
                                primaryf,
                                normal,
                                cell_border_mask,
                                vertex_border_mask,
                                0,
                                secondary,
                            );

                            material_processor.on_vertex(v0, v1, t1);

                            if reuse_dir & 8 != 0 {
                                current_reuse_cell.vertices[reuse_vertex_index] =
                                    cell_vertex_indices[vertex_index];
                            }
                        }
                    } else if t == 0.0 && v1 == 7 {
                        // The vertex sits on p1, the cell's maximum corner, so
                        // this cell owns it
                        let primaryf = p1.as_vec3();
                        let cg1 = sdf.corner_gradient(corner_data_indices[v1]);
                        let normal = normalized_not_null(cg1);

                        let mut secondary = Vec3::ZERO;
                        let mut vertex_border_mask = 0u8;
                        if cell_border_mask > 0 {
                            secondary =
                                get_secondary_position(primaryf, normal, lod_index, block_size);
                            vertex_border_mask = get_border_mask(p1, block_size_scaled);
                        }

                        cell_vertex_indices[vertex_index] = output.add_vertex(
                            primaryf,
                            normal,
                            cell_border_mask,
                            vertex_border_mask,
                            0,
                            secondary,
                        );

                        material_processor.on_vertex(v0, v1, 1.0);

                        current_reuse_cell.vertices[0] = cell_vertex_indices[vertex_index];
                    } else {
                        // The vertex is exactly on p0 or p1. Reuse is skipped
                        // here deliberately, ambiguous cases caused artifacts.
                        // Source: meshers/transvoxel/transvoxel.cpp:479-491
                        let vi = if t == 0.0 { v1 } else { v0 };
                        let primary = if t == 0.0 { p1 } else { p0 };
                        let primaryf = primary.as_vec3();
                        let cg = sdf.corner_gradient(corner_data_indices[vi]);
                        let normal = normalized_not_null(cg);

                        let mut secondary = Vec3::ZERO;
                        let mut vertex_border_mask = 0u8;
                        if cell_border_mask > 0 {
                            secondary =
                                get_secondary_position(primaryf, normal, lod_index, block_size);
                            vertex_border_mask = get_border_mask(primary, block_size_scaled);
                        }

                        cell_vertex_indices[vertex_index] = output.add_vertex(
                            primaryf,
                            normal,
                            cell_border_mask,
                            vertex_border_mask,
                            0,
                            secondary,
                        );

                        material_processor.on_vertex(v0, v1, 1.0 - t);
                    }
                }

                cache.set_reuse_cell(pos, current_reuse_cell);

                for t in 0..triangle_count {
                    let t0 = t * 3;
                    let i0 = cell_vertex_indices[regular_cell_data.get_vertex_index(t0) as usize];
                    let i1 =
                        cell_vertex_indices[regular_cell_data.get_vertex_index(t0 + 1) as usize];
                    let i2 =
                        cell_vertex_indices[regular_cell_data.get_vertex_index(t0 + 2) as usize];
                    output.indices.push(i0 as u32);
                    output.indices.push(i1 as u32);
                    output.indices.push(i2 as u32);
                }

                if let Some(infos) = cell_info.as_deref_mut() {
                    infos.push(CellInfo {
                        position: pos - min_pos,
                        triangle_count: triangle_count as u32,
                    });
                }

                data_index += n100;
            }
        }
    }
}

/// Maps face-space coordinates to block space for a given side.
/// Source: meshers/transvoxel/transvoxel.cpp:611-638
#[inline]
fn face_to_block(x: i32, y: i32, z: i32, dir: usize, bs: IVec3) -> IVec3 {
    use voxel_core::cube::side_axis::*;
    match dir {
        SIDE_NEGATIVE_X => IVec3::new(z, x, y),
        SIDE_POSITIVE_X => IVec3::new(bs.x - 1 - z, y, x),
        SIDE_NEGATIVE_Y => IVec3::new(y, z, x),
        SIDE_POSITIVE_Y => IVec3::new(x, bs.y - 1 - z, y),
        SIDE_NEGATIVE_Z => IVec3::new(x, y, z),
        _ => IVec3::new(y, x, bs.z - 1 - z),
    }
}

/// Which block axes the face-space X and Y axes map to.
/// Source: meshers/transvoxel/transvoxel.cpp:641-676
#[inline]
fn get_face_axes(dir: usize) -> (usize, usize) {
    use voxel_core::cube::side_axis::*;
    match dir {
        SIDE_NEGATIVE_X => (1, 2),
        SIDE_POSITIVE_X => (2, 1),
        SIDE_NEGATIVE_Y => (2, 0),
        SIDE_POSITIVE_Y => (0, 2),
        SIDE_NEGATIVE_Z => (0, 1),
        _ => (1, 0),
    }
}

/// Face index of a cube direction. Identity, kept for traceability against the
/// C++ switch.
/// Source: meshers/transvoxel/transvoxel.cpp:679-703
#[inline]
fn get_face_index(cube_dir: usize) -> u8 {
    cube_dir as u8
}

/// Builds the transition mesh of one block side.
///
/// Source: meshers/transvoxel/transvoxel.cpp:705-1090
pub fn build_transition_mesh<P: MaterialProcessor>(
    sdf: &SdfField,
    material_processor: &mut P,
    direction: usize,
    lod_index: u32,
    cache: &mut Cache,
    output: &mut MeshArrays,
    edge_clamp_margin: f32,
) {
    let edge_clamp_margin_max = 1.0 - edge_clamp_margin;

    let block_size_with_padding = sdf.size();
    let block_size_without_padding =
        block_size_with_padding - IVec3::splat(MIN_PADDING + MAX_PADDING);
    let block_size_scaled = block_size_without_padding << lod_index;

    if block_size_with_padding.x < 3
        || block_size_with_padding.y < 3
        || block_size_with_padding.z < 3
    {
        return;
    }

    cache.reset_reuse_cells_2d(block_size_with_padding);

    // Face space is 2D along local X and Y. -Z points at the half-resolution
    // cells, +Z at the full-resolution ones.
    let min_pos = IVec3::splat(MIN_PADDING);
    let max_pos = block_size_with_padding - IVec3::splat(MAX_PADDING);

    let (axis_x, axis_y) = get_face_axes(direction);
    let min_fpos_x = min_pos[axis_x];
    let min_fpos_y = min_pos[axis_y];
    // Another -1 because the 2D kernel is 3x3
    let max_fpos_x = max_pos[axis_x] - 1;
    let max_fpos_y = max_pos[axis_y] - 1;

    let n010 = 1isize;
    let n100 = block_size_with_padding.y as isize;
    let n001 = (block_size_with_padding.y * block_size_with_padding.x) as isize;

    let ftb_000 = face_to_block(0, 0, 0, direction, block_size_with_padding);
    let ftb_x00 = face_to_block(1, 0, 0, direction, block_size_with_padding);
    let ftb_0y0 = face_to_block(0, 1, 0, direction, block_size_with_padding);
    let fn00 = zxy_index(
        ftb_000.x as u32,
        ftb_000.y as u32,
        ftb_000.z as u32,
        block_size_with_padding,
    ) as isize;
    let fn10 = zxy_index(
        ftb_x00.x as u32,
        ftb_x00.y as u32,
        ftb_x00.z as u32,
        block_size_with_padding,
    ) as isize
        - fn00;
    let fn01 = zxy_index(
        ftb_0y0.x as u32,
        ftb_0y0.y as u32,
        ftb_0y0.z as u32,
        block_size_with_padding,
    ) as isize
        - fn00;
    let fn11 = fn10 + fn01;
    let fn21 = 2 * fn10 + fn01;
    let fn22 = 2 * fn10 + 2 * fn01;
    let fn12 = fn10 + 2 * fn01;
    let fn20 = 2 * fn10;
    let fn02 = 2 * fn01;

    let fz = MIN_PADDING;
    let isolevel = 0.0f32;
    let transition_hint_mask = 1u8 << get_face_index(direction);

    let mut fy = min_fpos_y;
    while fy < max_fpos_y {
        let mut fx = min_fpos_x;
        while fx < max_fpos_x {
            let mut cell_positions = [IVec3::ZERO; 13];
            // Temporarily includes padding, undone below
            cell_positions[0] = face_to_block(fx, fy, fz, direction, block_size_with_padding);

            let data_index = zxy_index(
                cell_positions[0].x as u32,
                cell_positions[0].y as u32,
                cell_positions[0].z as u32,
                block_size_with_padding,
            ) as isize;

            let at = |offset: isize| sdf.get_raw((data_index + offset) as usize);

            let s = at(0) > isolevel;
            if (at(fn10) > isolevel) == s
                && (at(fn20) > isolevel) == s
                && (at(fn01) > isolevel) == s
                && (at(fn11) > isolevel) == s
                && (at(fn21) > isolevel) == s
                && (at(fn02) > isolevel) == s
                && (at(fn12) > isolevel) == s
                && (at(fn22) > isolevel) == s
            {
                fx += 2;
                continue;
            }

            //  6---7---8
            //  |   |   |
            //  3---4---5
            //  |   |   |
            //  0---1---2
            let cell_data_indices: [usize; 9] = [
                data_index as usize,
                (data_index + fn10) as usize,
                (data_index + fn20) as usize,
                (data_index + fn01) as usize,
                (data_index + fn11) as usize,
                (data_index + fn21) as usize,
                (data_index + fn02) as usize,
                (data_index + fn12) as usize,
                (data_index + fn22) as usize,
            ];

            let mut cell_samples = [0.0f32; 13];
            for i in 0..9 {
                cell_samples[i] = sdf.get(cell_data_indices[i]);
            }
            // The half-resolution samples repeat the 3x3 side's corners
            cell_samples[0x9] = cell_samples[0];
            cell_samples[0xA] = cell_samples[2];
            cell_samples[0xB] = cell_samples[6];
            cell_samples[0xC] = cell_samples[8];

            // The case code order follows the lookup tables, which differs
            // from the sampling order
            // Source: meshers/transvoxel/transvoxel.cpp:848-858
            let mut case_code = sign_f(cell_samples[0]) as u16;
            case_code |= (sign_f(cell_samples[1]) as u16) << 1;
            case_code |= (sign_f(cell_samples[2]) as u16) << 2;
            case_code |= (sign_f(cell_samples[5]) as u16) << 3;
            case_code |= (sign_f(cell_samples[8]) as u16) << 4;
            case_code |= (sign_f(cell_samples[7]) as u16) << 5;
            case_code |= (sign_f(cell_samples[6]) as u16) << 6;
            case_code |= (sign_f(cell_samples[3]) as u16) << 7;
            case_code |= (sign_f(cell_samples[4]) as u16) << 8;

            if case_code == 0 || case_code == 511 {
                fx += 2;
                continue;
            }

            let mut current_reuse_cell = cache.get_reuse_cell_2d(fx, fy);
            current_reuse_cell.packed_texture_indices =
                material_processor.on_transition_cell(&cell_data_indices, case_code);

            let mut cell_gradients = [Vec3::ZERO; 13];
            for i in 0..9 {
                let di = cell_data_indices[i] as isize;
                let nx = sdf.get((di - n100) as usize);
                let ny = sdf.get((di - n010) as usize);
                let nz = sdf.get((di - n001) as usize);
                let px = sdf.get((di + n100) as usize);
                let py = sdf.get((di + n010) as usize);
                let pz = sdf.get((di + n001) as usize);
                cell_gradients[i] = Vec3::new(nx - px, ny - py, nz - pz);
            }
            cell_gradients[0x9] = cell_gradients[0];
            cell_gradients[0xA] = cell_gradients[2];
            cell_gradients[0xB] = cell_gradients[6];
            cell_gradients[0xC] = cell_gradients[8];

            cell_positions[1] = face_to_block(fx + 1, fy, fz, direction, block_size_with_padding);
            cell_positions[2] = face_to_block(fx + 2, fy, fz, direction, block_size_with_padding);
            cell_positions[3] =
                face_to_block(fx, fy + 1, fz, direction, block_size_with_padding);
            cell_positions[4] =
                face_to_block(fx + 1, fy + 1, fz, direction, block_size_with_padding);
            cell_positions[5] =
                face_to_block(fx + 2, fy + 1, fz, direction, block_size_with_padding);
            cell_positions[6] =
                face_to_block(fx, fy + 2, fz, direction, block_size_with_padding);
            cell_positions[7] =
                face_to_block(fx + 1, fy + 2, fz, direction, block_size_with_padding);
            cell_positions[8] =
                face_to_block(fx + 2, fy + 2, fz, direction, block_size_with_padding);
            for i in 0..9 {
                cell_positions[i] = (cell_positions[i] - min_pos) << lod_index;
            }
            cell_positions[0x9] = cell_positions[0];
            cell_positions[0xA] = cell_positions[2];
            cell_positions[0xB] = cell_positions[6];
            cell_positions[0xC] = cell_positions[8];

            let cell_class = tables::TRANSITION_CELL_CLASS[case_code as usize];
            let cell_data = &tables::TRANSITION_CELL_DATA[(cell_class & 0x7f) as usize];
            let flip_triangles = (cell_class & 128) != 0;

            let vertex_count = cell_data.vertex_count();
            let mut cell_vertex_indices = [-1i32; 12];

            let direction_validity_mask = (if fx > min_fpos_x { 1u8 } else { 0 })
                | (if fy > min_fpos_y { 1u8 } else { 0 } << 1);

            let cell_border_mask = get_border_mask(cell_positions[0], block_size_scaled);

            for vertex_index in 0..vertex_count {
                let edge_code = tables::TRANSITION_VERTEX_DATA[case_code as usize][vertex_index];
                let index_vertex_a = ((edge_code >> 4) & 0xf) as usize;
                let index_vertex_b = (edge_code & 0xf) as usize;

                let sample_a = cell_samples[index_vertex_a]; // d0 in the paper
                let sample_b = cell_samples[index_vertex_b]; // d1 in the paper

                let t = (sample_b / (sample_b - sample_a))
                    .clamp(edge_clamp_margin, edge_clamp_margin_max);
                let t0 = t;
                let t1 = 1.0 - t;

                if t > 0.0 && t < 1.0 {
                    // Vertex lies in the interior of the edge
                    let vertex_index_to_reuse_or_create = ((edge_code >> 8) & 0xf) as usize;

                    // Bit 0 subtracts one from X, bit 1 from Y, bit 2 marks an
                    // interior edge with no reuse, bit 3 a maximal edge that
                    // can be reused.
                    // Source: meshers/transvoxel/transvoxel.cpp:951-961
                    let reuse_direction = (edge_code >> 12) as u8;
                    let present = (reuse_direction & direction_validity_mask) == reuse_direction;

                    if present {
                        let prev = cache.get_reuse_cell_2d(
                            fx - (reuse_direction & 1) as i32,
                            fy - ((reuse_direction >> 1) & 1) as i32,
                        );
                        if prev.packed_texture_indices == current_reuse_cell.packed_texture_indices
                        {
                            cell_vertex_indices[vertex_index] =
                                prev.vertices[vertex_index_to_reuse_or_create];
                        }
                    }

                    if !present || cell_vertex_indices[vertex_index] == -1 {
                        let p0 = cell_positions[index_vertex_a];
                        let p1 = cell_positions[index_vertex_b];
                        let n0 = cell_gradients[index_vertex_a];
                        let n1 = cell_gradients[index_vertex_b];

                        let primaryf = p0.as_vec3() * t0 + p1.as_vec3() * t1;
                        let normal = normalized_not_null(n0 * t0 + n1 * t1);

                        let fullres_side = index_vertex_a < 9 || index_vertex_b < 9;

                        let mut secondary = Vec3::ZERO;
                        let mut cell_border_mask2 = cell_border_mask;
                        let mut vertex_border_mask = 0u8;
                        if fullres_side {
                            secondary = get_secondary_position(
                                primaryf,
                                normal,
                                lod_index,
                                block_size_without_padding,
                            );
                            vertex_border_mask = get_border_mask(p0, block_size_scaled)
                                & get_border_mask(p1, block_size_scaled);
                        } else {
                            // Half-res side vertices must never move, only the
                            // full-res side connects to the regular mesh
                            cell_border_mask2 = 0;
                        }

                        cell_vertex_indices[vertex_index] = output.add_vertex(
                            primaryf,
                            normal,
                            cell_border_mask2,
                            vertex_border_mask,
                            transition_hint_mask,
                            secondary,
                        );

                        material_processor.on_vertex(index_vertex_a, index_vertex_b, t1);

                        if reuse_direction & 0x8 != 0 {
                            current_reuse_cell.vertices[vertex_index_to_reuse_or_create] =
                                cell_vertex_indices[vertex_index];
                        }
                    }
                } else {
                    // The vertex is exactly on an endpoint, reuse follows
                    // `TRANSITION_CORNER_DATA`
                    let cell_index = if t == 0.0 {
                        index_vertex_b
                    } else {
                        index_vertex_a
                    };
                    let corner_data = tables::TRANSITION_CORNER_DATA[cell_index];
                    let vertex_index_to_reuse_or_create = (corner_data & 0xf) as usize;
                    let reuse_direction = (corner_data >> 4) & 0xf;

                    let present = (reuse_direction & direction_validity_mask) == reuse_direction;

                    if present {
                        let prev = cache.get_reuse_cell_2d(
                            fx - (reuse_direction & 1) as i32,
                            fy - ((reuse_direction >> 1) & 1) as i32,
                        );
                        cell_vertex_indices[vertex_index] =
                            prev.vertices[vertex_index_to_reuse_or_create];
                    }

                    if !present || cell_vertex_indices[vertex_index] == -1 {
                        let primary = cell_positions[cell_index];
                        let primaryf = primary.as_vec3();
                        let normal = normalized_not_null(cell_gradients[cell_index]);

                        let fullres_side = cell_index < 9;

                        let mut secondary = Vec3::ZERO;
                        let mut vertex_border_mask = 0u8;
                        let mut cell_border_mask2 = cell_border_mask;
                        if fullres_side {
                            secondary = get_secondary_position(
                                primaryf,
                                normal,
                                lod_index,
                                block_size_without_padding,
                            );
                            vertex_border_mask = get_border_mask(primary, block_size_scaled);
                        } else {
                            cell_border_mask2 = 0;
                        }

                        cell_vertex_indices[vertex_index] = output.add_vertex(
                            primaryf,
                            normal,
                            cell_border_mask2,
                            vertex_border_mask,
                            transition_hint_mask,
                            secondary,
                        );

                        material_processor.on_vertex(index_vertex_a, index_vertex_b, 1.0 - t);

                        // On a corner, so the vertex stays reusable
                        current_reuse_cell.vertices[vertex_index_to_reuse_or_create] =
                            cell_vertex_indices[vertex_index];
                    }
                }
            }

            cache.set_reuse_cell_2d(fx, fy, current_reuse_cell);

            let triangle_count = cell_data.triangle_count();
            for ti in 0..triangle_count {
                if flip_triangles {
                    output
                        .indices
                        .push(cell_vertex_indices[cell_data.get_vertex_index(ti * 3) as usize] as u32);
                    output.indices.push(
                        cell_vertex_indices[cell_data.get_vertex_index(ti * 3 + 1) as usize] as u32,
                    );
                    output.indices.push(
                        cell_vertex_indices[cell_data.get_vertex_index(ti * 3 + 2) as usize] as u32,
                    );
                } else {
                    output.indices.push(
                        cell_vertex_indices[cell_data.get_vertex_index(ti * 3 + 2) as usize] as u32,
                    );
                    output.indices.push(
                        cell_vertex_indices[cell_data.get_vertex_index(ti * 3 + 1) as usize] as u32,
                    );
                    output
                        .indices
                        .push(cell_vertex_indices[cell_data.get_vertex_index(ti * 3) as usize] as u32);
                }
            }

            fx += 2;
        }
        fy += 2;
    }
}

/// Reads the material index channel of a block, uniform or per voxel.
/// Source: meshers/transvoxel/transvoxel_materials_mixel4.h:295-322
pub fn get_texture_indices_data(
    voxels: &VoxelBuffer,
    indices_channel: ChannelId,
) -> Option<(TextureIndicesData, DefaultTextureIndicesData)> {
    if voxels.depth(indices_channel) != Depth::D32 {
        return None;
    }

    let mut data = TextureIndicesData::default();
    let mut default_data = DefaultTextureIndicesData::default();

    match voxels.uniform_value(indices_channel) {
        Some(encoded) => {
            let indices = voxel_core::mixel4::decode_indices_from_packed(encoded as u32);
            data.default_indices = indices;
            data.packed_default_indices = materials::pack_bytes(indices);
            default_data.indices = indices;
            default_data.packed_indices = data.packed_default_indices;
            default_data.use_it = true;
        }
        None => {
            data.buffer = voxels
                .channel_as_u32(indices_channel)
                .ok()
                .flatten()
                .unwrap_or_default();
            default_data.use_it = false;
        }
    }

    Some((data, default_data))
}

/// Reads the weight channel of a block into a sampler.
/// Source: meshers/transvoxel/transvoxel.cpp:1092-1113 combined with
/// `WeightSamplerPackedU16`
pub fn get_weights_sampler(voxels: &VoxelBuffer, weights_channel: ChannelId) -> WeightSamplerPackedU16 {
    let volume = voxels.volume();
    match voxels.channel_as_u16(weights_channel) {
        Ok(Some(data)) => WeightSamplerPackedU16 { data },
        _ => {
            let uniform = voxels.get_voxel(IVec3::ZERO, weights_channel) as u16;
            WeightSamplerPackedU16 {
                data: vec![uniform; volume],
            }
        }
    }
}

/// Everything one call needs beyond the block itself.
#[derive(Debug, Clone, Copy)]
pub struct TransvoxelParams {
    pub sdf_channel: ChannelId,
    pub lod_index: u32,
    pub texturing_mode: TexturingMode,
    /// Keeps interpolated vertices off exact cell corners, which avoids
    /// degenerate triangles.
    pub edge_clamp_margin: f32,
    /// Excludes air corners from material selection.
    pub textures_ignore_air_voxels: bool,
}

impl Default for TransvoxelParams {
    fn default() -> TransvoxelParams {
        TransvoxelParams {
            sdf_channel: ChannelId::Sdf,
            lod_index: 0,
            texturing_mode: TexturingMode::None,
            edge_clamp_margin: 0.0,
            textures_ignore_air_voxels: false,
        }
    }
}

/// Result of one full block build.
#[derive(Debug, Clone, Default)]
pub struct TransvoxelOutput {
    pub regular: MeshArrays,
    pub transitions: [MeshArrays; 6],
    pub cell_infos: Vec<CellInfo>,
    pub default_texture_indices: DefaultTextureIndicesData,
}

/// Builds the regular mesh plus the requested transition meshes of a block.
///
/// `transition_mask` selects sides, bit N corresponds to side N of
/// `voxel_core::cube::side_axis`.
pub fn build_mesh(
    voxels: &VoxelBuffer,
    params: TransvoxelParams,
    transition_mask: u8,
    want_cell_infos: bool,
) -> TransvoxelOutput {
    let sdf = SdfField::from_buffer(voxels, params.sdf_channel);
    let mut cache = Cache::default();
    let mut out = TransvoxelOutput::default();
    let mut cell_infos = Vec::new();

    match params.texturing_mode {
        TexturingMode::None => {
            let mut processor = NullProcessor::default();
            build_all(
                &sdf,
                &mut processor,
                params,
                transition_mask,
                &mut cache,
                &mut out,
                if want_cell_infos {
                    Some(&mut cell_infos)
                } else {
                    None
                },
            );
        }
        TexturingMode::Mixel4S4 => {
            let (indices_data, default_data) =
                get_texture_indices_data(voxels, ChannelId::Indices).unwrap_or_default();
            let weights = get_weights_sampler(voxels, ChannelId::Weights);
            out.default_texture_indices = default_data;
            let mut processor = Mixel4Processor::new(
                indices_data,
                weights,
                params.textures_ignore_air_voxels,
            );
            build_all(
                &sdf,
                &mut processor,
                params,
                transition_mask,
                &mut cache,
                &mut out,
                if want_cell_infos {
                    Some(&mut cell_infos)
                } else {
                    None
                },
            );
        }
    }

    out.cell_infos = cell_infos;
    out
}

/// Runs the regular pass and every selected transition pass.
fn build_all<P: MaterialProcessor>(
    sdf: &SdfField,
    processor: &mut P,
    params: TransvoxelParams,
    transition_mask: u8,
    cache: &mut Cache,
    out: &mut TransvoxelOutput,
    cell_infos: Option<&mut Vec<CellInfo>>,
) {
    build_regular_mesh(
        sdf,
        processor,
        params.lod_index,
        cache,
        &mut out.regular,
        cell_infos,
        params.edge_clamp_margin,
    );
    out.regular.texturing_data_2f32 = processor
        .take_material_data()
        .into_iter()
        .map(|v| Vec2::new(v[0], v[1]))
        .collect();

    for side in 0..6 {
        if transition_mask & (1 << side) == 0 {
            continue;
        }
        build_transition_mesh(
            sdf,
            processor,
            side,
            params.lod_index,
            cache,
            &mut out.transitions[side],
            params.edge_clamp_margin,
        );
        out.transitions[side].texturing_data_2f32 = processor
            .take_material_data()
            .into_iter()
            .map(|v| Vec2::new(v[0], v[1]))
            .collect();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use voxel_core::Box3i;

    /// Block of `size` voxels plus the padding Transvoxel requires.
    fn padded_size(size: i32) -> IVec3 {
        IVec3::splat(size + MIN_PADDING + MAX_PADDING)
    }

    /// Buffer whose SDF describes a plane at height `h`, negative below.
    fn plane_buffer(size: i32, h: f32) -> VoxelBuffer {
        let mut b = VoxelBuffer::new(padded_size(size)).unwrap();
        b.box_of_whole_buffer().for_each_cell_zxy(|pos| {
            let y = (pos.y - MIN_PADDING) as f32;
            b.set_voxel_f(y - h, pos, ChannelId::Sdf).unwrap();
        });
        b
    }

    /// Buffer whose SDF describes a sphere of radius `r` centred in the block.
    fn sphere_buffer(size: i32, r: f32) -> VoxelBuffer {
        let mut b = VoxelBuffer::new(padded_size(size)).unwrap();
        let center = Vec3::splat(size as f32 * 0.5);
        b.box_of_whole_buffer().for_each_cell_zxy(|pos| {
            let p = (pos - IVec3::splat(MIN_PADDING)).as_vec3();
            b.set_voxel_f((p - center).length() - r, pos, ChannelId::Sdf)
                .unwrap();
        });
        b
    }

    #[test]
    fn uniform_air_block_produces_no_geometry() {
        let b = VoxelBuffer::new(padded_size(8)).unwrap();
        let out = build_mesh(&b, TransvoxelParams::default(), 0, false);
        assert!(out.regular.is_empty());
        assert!(out.regular.vertices.is_empty());
    }

    #[test]
    fn uniform_solid_block_produces_no_geometry() {
        let mut b = VoxelBuffer::new(padded_size(8)).unwrap();
        b.clear_channel_f(ChannelId::Sdf, -1.0);
        let out = build_mesh(&b, TransvoxelParams::default(), 0, false);
        assert!(out.regular.is_empty());
    }

    #[test]
    fn a_plane_produces_a_horizontal_surface() {
        let b = plane_buffer(8, 4.0);
        let out = build_mesh(&b, TransvoxelParams::default(), 0, false);
        assert!(!out.regular.is_empty(), "no triangles generated");
        assert_eq!(out.regular.indices.len() % 3, 0);
        // Every vertex sits near the plane height
        for v in &out.regular.vertices {
            assert!((v.y - 4.0).abs() < 1.0, "vertex {v} off the plane");
        }
        // Normals point along +Y for a plane rising in Y
        for n in &out.regular.normals {
            assert!(n.y.abs() > 0.9, "normal {n} not axis aligned");
        }
    }

    #[test]
    fn every_index_addresses_an_existing_vertex() {
        let b = sphere_buffer(16, 5.0);
        let out = build_mesh(&b, TransvoxelParams::default(), 0, false);
        assert!(!out.regular.is_empty());
        let vertex_count = out.regular.vertices.len() as u32;
        for i in &out.regular.indices {
            assert!(*i < vertex_count, "index {i} exceeds {vertex_count}");
        }
        assert_eq!(out.regular.normals.len(), out.regular.vertices.len());
        assert_eq!(out.regular.lod_data.len(), out.regular.vertices.len());
    }

    #[test]
    fn normals_are_unit_length() {
        let b = sphere_buffer(16, 5.0);
        let out = build_mesh(&b, TransvoxelParams::default(), 0, false);
        for n in &out.regular.normals {
            assert!((n.length() - 1.0).abs() < 1e-4, "normal {n} not normalized");
        }
    }

    #[test]
    fn sphere_vertices_lie_on_the_sphere() {
        let size = 16;
        let r = 5.0f32;
        let b = sphere_buffer(size, r);
        let out = build_mesh(&b, TransvoxelParams::default(), 0, false);
        let center = Vec3::splat(size as f32 * 0.5);
        for v in &out.regular.vertices {
            let d = (*v - center).length();
            assert!((d - r).abs() < 1.0, "vertex {v} at distance {d}");
        }
    }

    #[test]
    fn sphere_normals_point_outwards() {
        let size = 16;
        let b = sphere_buffer(size, 5.0);
        let out = build_mesh(&b, TransvoxelParams::default(), 0, false);
        let center = Vec3::splat(size as f32 * 0.5);
        for (v, n) in out.regular.vertices.iter().zip(&out.regular.normals) {
            let outward = (*v - center).normalize_or_zero();
            assert!(outward.dot(*n) > 0.8, "normal {n} not outward at {v}");
        }
    }

    #[test]
    fn cell_infos_account_for_every_triangle() {
        let b = sphere_buffer(16, 5.0);
        let out = build_mesh(&b, TransvoxelParams::default(), 0, true);
        let total: u32 = out.cell_infos.iter().map(|c| c.triangle_count).sum();
        assert_eq!(total as usize * 3, out.regular.indices.len());
        // Cell positions are reported without padding
        let block = Box3i::from_size(IVec3::splat(16));
        for info in &out.cell_infos {
            assert!(block.contains_point(info.position), "cell {:?}", info.position);
        }
    }

    #[test]
    fn triangle_count_per_cell_respects_the_documented_maximum() {
        let b = sphere_buffer(16, 5.0);
        let out = build_mesh(&b, TransvoxelParams::default(), 0, true);
        for info in &out.cell_infos {
            assert!(
                info.triangle_count as usize <= MAX_TRIANGLES_PER_CELL,
                "cell {:?} produced {} triangles",
                info.position,
                info.triangle_count
            );
        }
    }

    #[test]
    fn transition_meshes_are_produced_per_requested_side() {
        // A plane crosses the four side faces of the block, so those produce
        // transition geometry. A shape not reaching a block side produces
        // none there, which is why a centred sphere is unsuitable here.
        let b = plane_buffer(8, 4.0);
        let out = build_mesh(&b, TransvoxelParams::default(), 0b111111, false);
        let produced = out.transitions.iter().filter(|m| !m.is_empty()).count();
        assert_eq!(produced, 4, "expected the four side faces to produce geometry");
        for mesh in &out.transitions {
            let vertex_count = mesh.vertices.len() as u32;
            for i in &mesh.indices {
                assert!(*i < vertex_count, "transition index {i} out of range");
            }
            // Transition vertices are flagged as such
            for attrib in &mesh.lod_data {
                assert_ne!(attrib.transition, 0);
            }
        }
    }

    #[test]
    fn a_masked_out_side_produces_nothing() {
        let b = plane_buffer(8, 4.0);
        let out = build_mesh(&b, TransvoxelParams::default(), 0b000001, false);
        assert!(!out.transitions[0].is_empty(), "the selected side is empty");
        for side in 1..6 {
            assert!(out.transitions[side].is_empty(), "side {side} not masked");
        }
    }

    #[test]
    fn geometry_away_from_the_block_sides_produces_no_transitions() {
        // A sphere well inside the block touches no side face
        let b = sphere_buffer(16, 5.0);
        let out = build_mesh(&b, TransvoxelParams::default(), 0b111111, false);
        assert!(!out.regular.is_empty(), "no regular geometry");
        for (side, mesh) in out.transitions.iter().enumerate() {
            assert!(mesh.is_empty(), "side {side} produced geometry");
        }
    }

    #[test]
    fn lod_index_scales_vertex_positions() {
        let b = plane_buffer(8, 4.0);
        let out0 = build_mesh(&b, TransvoxelParams::default(), 0, false);
        let out1 = build_mesh(
            &b,
            TransvoxelParams {
                lod_index: 1,
                ..TransvoxelParams::default()
            },
            0,
            false,
        );
        let max0 = out0
            .regular
            .vertices
            .iter()
            .map(|v| v.x)
            .fold(0.0f32, f32::max);
        let max1 = out1
            .regular
            .vertices
            .iter()
            .map(|v| v.x)
            .fold(0.0f32, f32::max);
        assert!((max1 - max0 * 2.0).abs() < 1.0, "max0 {max0} max1 {max1}");
    }

    #[test]
    fn border_masks_are_set_on_block_boundaries() {
        let b = plane_buffer(8, 4.0);
        let out = build_mesh(&b, TransvoxelParams::default(), 0, false);
        let with_border = out
            .regular
            .lod_data
            .iter()
            .filter(|a| a.cell_border_mask != 0)
            .count();
        assert!(with_border > 0, "no vertex reported as on a border");
    }

    #[test]
    fn mixel4_mode_emits_one_material_attribute_per_vertex() {
        let b = sphere_buffer(16, 5.0);
        let params = TransvoxelParams {
            texturing_mode: TexturingMode::Mixel4S4,
            textures_ignore_air_voxels: true,
            ..TransvoxelParams::default()
        };
        let out = build_mesh(&b, params, 0, false);
        assert!(!out.regular.is_empty());
        assert_eq!(
            out.regular.texturing_data_2f32.len(),
            out.regular.vertices.len()
        );
        // The indices channel is uniform here, so the default path is taken
        assert!(out.default_texture_indices.use_it);
    }

    #[test]
    fn mixel4_mode_reads_a_dense_indices_channel() {
        let mut b = sphere_buffer(16, 5.0);
        // Write a differing material somewhere to force a dense channel
        b.set_voxel(
            voxel_core::mixel4::make_encoded_indices_for_single_texture(42) as u64,
            IVec3::splat(4),
            ChannelId::Indices,
        )
        .unwrap();
        let params = TransvoxelParams {
            texturing_mode: TexturingMode::Mixel4S4,
            textures_ignore_air_voxels: true,
            ..TransvoxelParams::default()
        };
        let out = build_mesh(&b, params, 0, false);
        assert!(!out.default_texture_indices.use_it);
        assert_eq!(
            out.regular.texturing_data_2f32.len(),
            out.regular.vertices.len()
        );
    }

    #[test]
    fn edge_clamp_margin_keeps_vertices_off_the_corners() {
        // A plane exactly on integer coordinates puts t at 0 without clamping
        let b = plane_buffer(8, 4.0);
        let params = TransvoxelParams {
            edge_clamp_margin: 0.01,
            ..TransvoxelParams::default()
        };
        let out = build_mesh(&b, params, 0, false);
        assert!(!out.regular.is_empty());
        for v in &out.regular.vertices {
            let dy = (v.y - v.y.round()).abs();
            // Either clearly off a corner, or the clamp margin away from it
            assert!(dy > 0.001 || v.y.fract() == 0.0, "vertex {v}");
        }
    }

    #[test]
    fn sdf_field_decodes_a_uniform_channel() {
        let mut b = VoxelBuffer::new(padded_size(4)).unwrap();
        b.clear_channel_f(ChannelId::Sdf, 0.5);
        let field = SdfField::from_buffer(&b, ChannelId::Sdf);
        assert_eq!(field.size(), padded_size(4));
        // `sdf_as_float` negates, so a positive SDF reads back negative
        assert!(field.get(0) < 0.0);
    }

    #[test]
    fn sdf_field_decodes_a_dense_channel_in_zxy_order() {
        let mut b = VoxelBuffer::new(padded_size(4)).unwrap();
        b.set_voxel_f(-1.0, IVec3::new(2, 3, 4), ChannelId::Sdf).unwrap();
        let field = SdfField::from_buffer(&b, ChannelId::Sdf);
        let index = zxy_index(2, 3, 4, padded_size(4));
        assert!(field.get(index) > 0.0, "value {}", field.get(index));
    }

    #[test]
    fn eight_bit_sdf_depth_is_supported() {
        let format = voxel_core::VoxelFormat::default()
            .with_depth(ChannelId::Sdf, Depth::D8);
        let mut b = VoxelBuffer::with_format(padded_size(8), format).unwrap();
        b.box_of_whole_buffer().for_each_cell_zxy(|pos| {
            let y = (pos.y - MIN_PADDING) as f32;
            b.set_voxel_f(y - 4.0, pos, ChannelId::Sdf).unwrap();
        });
        let out = build_mesh(&b, TransvoxelParams::default(), 0, false);
        assert!(!out.regular.is_empty(), "no geometry at 8-bit depth");
    }

    #[test]
    fn thirty_two_bit_sdf_depth_is_supported() {
        let format = voxel_core::VoxelFormat::default()
            .with_depth(ChannelId::Sdf, Depth::D32);
        let mut b = VoxelBuffer::with_format(padded_size(8), format).unwrap();
        b.box_of_whole_buffer().for_each_cell_zxy(|pos| {
            let y = (pos.y - MIN_PADDING) as f32;
            b.set_voxel_f(y - 4.0, pos, ChannelId::Sdf).unwrap();
        });
        let out = build_mesh(&b, TransvoxelParams::default(), 0, false);
        assert!(!out.regular.is_empty(), "no geometry at 32-bit depth");
    }

    #[test]
    fn face_to_block_is_a_bijection_per_side() {
        let bs = IVec3::splat(6);
        for dir in 0..6 {
            let mut seen = std::collections::HashSet::new();
            for z in 0..3 {
                for y in 0..6 {
                    for x in 0..6 {
                        let p = face_to_block(x, y, z, dir, bs);
                        assert!(seen.insert(p), "side {dir} mapped twice to {p}");
                    }
                }
            }
        }
    }

    #[test]
    fn face_axes_are_distinct_per_side() {
        for dir in 0..6 {
            let (ax, ay) = get_face_axes(dir);
            assert_ne!(ax, ay, "side {dir}");
            assert!(ax < 3 && ay < 3);
        }
    }

    #[test]
    fn border_mask_flags_both_extremes() {
        let size = IVec3::splat(8);
        assert_eq!(get_border_mask(IVec3::ZERO, size), 0b010101);
        assert_eq!(get_border_mask(size, size), 0b101010);
        assert_eq!(get_border_mask(IVec3::splat(4), size), 0);
    }

    #[test]
    fn normalized_not_null_falls_back_to_up() {
        assert_eq!(normalized_not_null(Vec3::ZERO), Vec3::Y);
        assert!((normalized_not_null(Vec3::new(3.0, 0.0, 4.0)).length() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn dir_to_prev_vec_decodes_each_axis_bit() {
        assert_eq!(dir_to_prev_vec(0), IVec3::ZERO);
        assert_eq!(dir_to_prev_vec(1), IVec3::new(-1, 0, 0));
        assert_eq!(dir_to_prev_vec(2), IVec3::new(0, -1, 0));
        assert_eq!(dir_to_prev_vec(4), IVec3::new(0, 0, -1));
        assert_eq!(dir_to_prev_vec(7), IVec3::splat(-1));
    }
}
