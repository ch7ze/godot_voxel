//! Material processors for the Transvoxel mesher.
//!
//! Ported from `meshers/transvoxel/transvoxel_materials_null.h`,
//! `transvoxel_materials_common.h` and `transvoxel_materials_mixel4.h`.
//! The `single_s2` and `single_s4` variants are left out, the `rame` generator
//! writes Mixel4 materials (see doc/source/bevy_port_plan.md section 4, P3).

use voxel_core::mixel4::{decode_indices_from_packed, decode_weights_from_packed};

/// Blend slots per voxel.
/// Source: meshers/transvoxel/transvoxel_materials_mixel4.h:15
pub const MAX_TEXTURE_BLENDS: usize = 4;

/// Total addressable materials with 8-bit index slots.
/// Source: meshers/transvoxel/transvoxel_materials_mixel4.h:13
pub const MAX_TEXTURES: usize = 256;

/// Texturing mode of the mesher.
/// Source: meshers/transvoxel/transvoxel.h:22-35
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TexturingMode {
    #[default]
    None,
    /// 4 blended materials per vertex, indices and weights packed into two
    /// 32-bit words exposed as one 2-float attribute.
    Mixel4S4,
}

/// Packs four bytes into a 32-bit word, slot 0 in the low byte.
/// Source: meshers/transvoxel/transvoxel_materials_common.h:12-15
#[inline]
pub fn pack_bytes(a: [u8; 4]) -> u32 {
    (a[0] as u32) | ((a[1] as u32) << 8) | ((a[2] as u32) << 16) | ((a[3] as u32) << 24)
}

/// Copies the 9 full-resolution values of a transition cell and fills the four
/// redundant half-resolution slots 9, A, B, C.
/// Source: meshers/transvoxel/transvoxel_materials_common.h:37-46
pub fn fill_redundant_transition_cell_values<T: Copy + Default>(src: &[T; 9]) -> [T; 13] {
    let mut dst = [T::default(); 13];
    dst[..9].copy_from_slice(src);
    dst[0x9] = src[0];
    dst[0xA] = src[2];
    dst[0xB] = src[6];
    dst[0xC] = src[8];
    dst
}

/// Reorders a transition case code so its bits follow the voxel sampling order.
/// The lookup tables use a different corner order than the sampling loop.
/// Source: meshers/transvoxel/transvoxel_materials_mixel4.h:201-228
pub fn reorder_transition_case_code(case_code: u16) -> u16 {
    (case_code & 0b000000111)
        | ((case_code & 0b110000000) >> 4)
        | ((case_code & 0b000001000) << 2)
        | (case_code & 0b001000000)
        | ((case_code & 0b000100000) << 2)
        | ((case_code & 0b000010000) << 4)
}

/// What a mesher calls on every cell and vertex to accumulate material data.
/// Source: meshers/transvoxel/transvoxel_materials_null.h:10-25
pub trait MaterialProcessor {
    /// Called once per 2x2x2 cell that produces triangles. The return value
    /// gates vertex reuse, cells may only share vertices when it matches.
    fn on_cell(&mut self, corner_voxel_indices: &[usize; 8], case_code: u8) -> u32;

    /// Called once per 2x3x3 transition cell that produces triangles. Only the
    /// 9 distinct corners are passed, the cell is 2D data-wise.
    fn on_transition_cell(&mut self, corner_voxel_indices: &[usize; 9], case_code: u16) -> u32;

    /// Called per generated vertex, interpolating between corners `v0` and `v1`.
    fn on_vertex(&mut self, v0: usize, v1: usize, alpha: f32);

    /// Per-vertex material attribute produced so far, two 32-bit words each.
    fn material_data(&self) -> &[[f32; 2]];

    /// Takes the accumulated attribute out of the processor.
    fn take_material_data(&mut self) -> Vec<[f32; 2]>;
}

/// Processor producing no material data.
/// Source: meshers/transvoxel/transvoxel_materials_null.h:10-25
#[derive(Debug, Default)]
pub struct NullProcessor {
    empty: Vec<[f32; 2]>,
}

impl MaterialProcessor for NullProcessor {
    fn on_cell(&mut self, _corner_voxel_indices: &[usize; 8], _case_code: u8) -> u32 {
        0
    }

    fn on_transition_cell(&mut self, _corner_voxel_indices: &[usize; 9], _case_code: u16) -> u32 {
        0
    }

    fn on_vertex(&mut self, _v0: usize, _v1: usize, _alpha: f32) {}

    fn material_data(&self) -> &[[f32; 2]] {
        &self.empty
    }

    fn take_material_data(&mut self) -> Vec<[f32; 2]> {
        Vec::new()
    }
}

/// Material indices of a block, either per voxel or one set for the whole block.
/// Source: meshers/transvoxel/transvoxel_materials_mixel4.h:153-159
#[derive(Debug, Clone, Default)]
pub struct TextureIndicesData {
    /// Per-voxel packed index words. Empty when the channel is uniform.
    pub buffer: Vec<u32>,
    /// Used when `buffer` is empty.
    pub default_indices: [u8; 4],
    pub packed_default_indices: u32,
}

/// Reused across the regular and transition passes of one block.
/// Source: meshers/transvoxel/transvoxel.h:160-165
#[derive(Debug, Clone, Copy, Default)]
pub struct DefaultTextureIndicesData {
    pub indices: [u8; 4],
    pub packed_indices: u32,
    pub use_it: bool,
}

/// Reads per-voxel weights out of the packed 16-bit weight channel.
/// Source: meshers/transvoxel/transvoxel_materials_mixel4.h:194-199
#[derive(Debug, Clone, Default)]
pub struct WeightSamplerPackedU16 {
    pub data: Vec<u16>,
}

impl WeightSamplerPackedU16 {
    #[inline]
    pub fn get_weights(&self, i: usize) -> [u8; MAX_TEXTURE_BLENDS] {
        decode_weights_from_packed(self.data[i])
    }
}

/// Selected indices of a cell plus the per-corner weights remapped onto them.
/// Source: meshers/transvoxel/transvoxel_materials_mixel4.h:17-22
#[derive(Debug, Clone)]
pub struct CellTextureDatas<const N: usize> {
    pub packed_indices: u32,
    pub indices: [u8; MAX_TEXTURE_BLENDS],
    pub weights: [[u8; MAX_TEXTURE_BLENDS]; N],
}

impl<const N: usize> Default for CellTextureDatas<N> {
    fn default() -> CellTextureDatas<N> {
        CellTextureDatas {
            packed_indices: 0,
            indices: [0; MAX_TEXTURE_BLENDS],
            weights: [[0; MAX_TEXTURE_BLENDS]; N],
        }
    }
}

/// Picks the four highest-weight materials occurring in a cell.
///
/// Cost is bound by `N * 4` distinct indices per cell, independent of
/// `MAX_TEXTURES`. Air corners, marked in `case_code`, contribute nothing.
/// Source: meshers/transvoxel/transvoxel_materials_mixel4.h:24-151
pub fn select_textures_4_per_voxel<const N: usize>(
    voxel_indices: &[usize; N],
    indices_data: &[u32],
    weights_sampler: &WeightSamplerPackedU16,
    case_code: u32,
) -> CellTextureDatas<N> {
    let mut corner_indices = [[0u8; 4]; N];
    let mut corner_weights = [[0u8; 4]; N];

    // Accumulated weight per index actually occurring in this cell
    let mut occurring: Vec<(u32, u32)> = Vec::with_capacity(N * 4);

    for ci in 0..N {
        // Air voxels must not contribute
        if (case_code & (1 << ci)) != 0 {
            continue;
        }

        let data_index = voxel_indices[ci];
        corner_indices[ci] = decode_indices_from_packed(indices_data[data_index]);
        corner_weights[ci] = weights_sampler.get_weights(data_index);

        for j in 0..4 {
            let ti = corner_indices[ci][j] as u32;
            let w = corner_weights[ci][j] as u32;
            match occurring.iter_mut().find(|(index, _)| *index == ti) {
                Some(entry) => entry.1 += w,
                None => occurring.push((ti, w)),
            }
        }
    }

    let mut cell_textures = CellTextureDatas::<N>::default();

    // Partial selection sort over at most 4 passes
    let select_count = occurring.len().min(4);
    for i in 0..select_count {
        let mut best = i;
        for k in (i + 1)..occurring.len() {
            if occurring[k].1 > occurring[best].1 {
                best = k;
            }
        }
        occurring.swap(i, best);
        cell_textures.indices[i] = occurring[i].0 as u8;
    }

    // Fill unused slots with distinct indices so the four stay different,
    // which the blending requires. Their weight stays 0.
    if select_count < 4 {
        let mut next_fill = 0u8;
        for i in select_count..4 {
            while cell_textures.indices[..i].contains(&next_fill) {
                next_fill = next_fill.wrapping_add(1);
            }
            cell_textures.indices[i] = next_fill;
            next_fill = next_fill.wrapping_add(1);
        }
    }

    // Sorting removes blend ambiguity between permutations of the same set
    cell_textures.indices.sort_unstable();
    cell_textures.packed_indices = pack_bytes(cell_textures.indices);

    // Remap each corner's weights onto the selected index order
    for ci in 0..N {
        if (case_code & (1 << ci)) != 0 {
            cell_textures.weights[ci] = [0; MAX_TEXTURE_BLENDS];
            continue;
        }
        for i in 0..4 {
            let ti = cell_textures.indices[i];
            let mut w = 0u8;
            for j in 0..4 {
                if corner_indices[ci][j] == ti {
                    w = corner_weights[ci][j];
                    break;
                }
            }
            cell_textures.weights[ci][i] = w;
        }
    }

    cell_textures
}

/// Fills `cell_textures` either from the block-wide default indices or by
/// selecting the four dominant materials of the cell.
/// Source: meshers/transvoxel/transvoxel_materials_mixel4.h:161-192
pub fn get_cell_texture_data<const N: usize>(
    cell_textures: &mut CellTextureDatas<N>,
    texture_indices_data: &TextureIndicesData,
    voxel_indices: &[usize; N],
    weights_data: &WeightSamplerPackedU16,
    case_code: u32,
) {
    if texture_indices_data.buffer.is_empty() {
        // Indices are the same across the block, only weights are read
        cell_textures.indices = texture_indices_data.default_indices;
        cell_textures.packed_indices = texture_indices_data.packed_default_indices;
        for ci in 0..N {
            if (case_code & (1 << ci)) != 0 {
                cell_textures.weights[ci] = [0; MAX_TEXTURE_BLENDS];
            } else {
                cell_textures.weights[ci] = weights_data.get_weights(voxel_indices[ci]);
            }
        }
    } else {
        *cell_textures = select_textures_4_per_voxel(
            voxel_indices,
            &texture_indices_data.buffer,
            weights_data,
            case_code,
        );
    }
}

/// Mixel4 processor, writing one 2-float attribute per vertex holding the
/// packed indices in x and the packed interpolated weights in y.
/// Source: meshers/transvoxel/transvoxel_materials_mixel4.h:230-293
#[derive(Debug)]
pub struct Mixel4Processor {
    voxel_material_indices: TextureIndicesData,
    voxel_material_weights: WeightSamplerPackedU16,
    /// When set, air corners are excluded from material selection.
    /// Source: meshers/transvoxel/voxel_mesher_transvoxel, textures_ignore_air_voxels
    textures_skip_air_voxels: bool,
    cell_textures: CellTextureDatas<13>,
    output: Vec<[f32; 2]>,
}

impl Mixel4Processor {
    pub fn new(
        voxel_material_indices: TextureIndicesData,
        voxel_material_weights: WeightSamplerPackedU16,
        textures_skip_air_voxels: bool,
    ) -> Mixel4Processor {
        Mixel4Processor {
            voxel_material_indices,
            voxel_material_weights,
            textures_skip_air_voxels,
            cell_textures: CellTextureDatas::default(),
            output: Vec::new(),
        }
    }
}

impl MaterialProcessor for Mixel4Processor {
    fn on_cell(&mut self, corner_voxel_indices: &[usize; 8], case_code: u8) -> u32 {
        let mut partial = CellTextureDatas::<8>::default();
        get_cell_texture_data(
            &mut partial,
            &self.voxel_material_indices,
            corner_voxel_indices,
            &self.voxel_material_weights,
            if self.textures_skip_air_voxels {
                case_code as u32
            } else {
                0
            },
        );
        self.cell_textures.indices = partial.indices;
        self.cell_textures.packed_indices = partial.packed_indices;
        self.cell_textures.weights[..8].copy_from_slice(&partial.weights);
        self.cell_textures.packed_indices
    }

    fn on_transition_cell(&mut self, corner_voxel_indices: &[usize; 9], case_code: u16) -> u32 {
        let alt_case_code = if self.textures_skip_air_voxels {
            reorder_transition_case_code(case_code) as u32
        } else {
            0
        };

        let mut partial = CellTextureDatas::<9>::default();
        get_cell_texture_data(
            &mut partial,
            &self.voxel_material_indices,
            corner_voxel_indices,
            &self.voxel_material_weights,
            alt_case_code,
        );

        self.cell_textures.indices = partial.indices;
        self.cell_textures.packed_indices = partial.packed_indices;
        self.cell_textures.weights = fill_redundant_transition_cell_values(&partial.weights);
        self.cell_textures.packed_indices
    }

    fn on_vertex(&mut self, v0: usize, v1: usize, alpha: f32) {
        let weights0 = self.cell_textures.weights[v0];
        let weights1 = self.cell_textures.weights[v1];
        let mut weights = [0u8; MAX_TEXTURE_BLENDS];
        for i in 0..MAX_TEXTURE_BLENDS {
            let lerped = weights0[i] as f32 + (weights1[i] as f32 - weights0[i] as f32) * alpha;
            weights[i] = lerped.clamp(0.0, 255.0) as u8;
        }
        // Both words are bit-cast into floats, the shader reinterprets them
        self.output.push([
            f32::from_bits(self.cell_textures.packed_indices),
            f32::from_bits(pack_bytes(weights)),
        ]);
    }

    fn material_data(&self) -> &[[f32; 2]] {
        &self.output
    }

    fn take_material_data(&mut self) -> Vec<[f32; 2]> {
        core::mem::take(&mut self.output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use voxel_core::mixel4::encode_weights_to_packed_lossy;

    #[test]
    fn pack_bytes_keeps_slot_order_and_high_values() {
        assert_eq!(pack_bytes([1, 2, 3, 4]), 0x04030201);
        // Slot 3 above 127 must not overflow
        assert_eq!(pack_bytes([0, 0, 0, 255]), 0xff000000);
    }

    #[test]
    fn redundant_transition_slots_mirror_the_corners() {
        let src: [u8; 9] = [0, 1, 2, 3, 4, 5, 6, 7, 8];
        let dst = fill_redundant_transition_cell_values(&src);
        assert_eq!(&dst[..9], &src);
        assert_eq!(dst[0x9], src[0]);
        assert_eq!(dst[0xA], src[2]);
        assert_eq!(dst[0xB], src[6]);
        assert_eq!(dst[0xC], src[8]);
    }

    #[test]
    fn transition_case_reorder_maps_every_bit_bijectively() {
        let mut seen = std::collections::HashSet::new();
        for case in 0u16..512 {
            assert!(seen.insert(reorder_transition_case_code(case)), "case {case}");
        }
        assert_eq!(reorder_transition_case_code(0), 0);
        assert_eq!(reorder_transition_case_code(0b111111111), 0b111111111);
    }

    fn sampler(count: usize, packed: u16) -> WeightSamplerPackedU16 {
        WeightSamplerPackedU16 {
            data: vec![packed; count],
        }
    }

    #[test]
    fn uniform_indices_path_reads_weights_directly() {
        let data = TextureIndicesData {
            buffer: Vec::new(),
            default_indices: [3, 4, 5, 6],
            packed_default_indices: pack_bytes([3, 4, 5, 6]),
        };
        let weights = sampler(8, encode_weights_to_packed_lossy(255, 0, 0, 0));
        let mut cell = CellTextureDatas::<8>::default();
        get_cell_texture_data(&mut cell, &data, &[0, 1, 2, 3, 4, 5, 6, 7], &weights, 0);
        assert_eq!(cell.indices, [3, 4, 5, 6]);
        assert_eq!(cell.weights[0], [240, 0, 0, 0]);
    }

    #[test]
    fn air_corners_get_zero_weights() {
        let data = TextureIndicesData {
            buffer: Vec::new(),
            default_indices: [0, 1, 2, 3],
            packed_default_indices: pack_bytes([0, 1, 2, 3]),
        };
        let weights = sampler(8, encode_weights_to_packed_lossy(255, 0, 0, 0));
        let mut cell = CellTextureDatas::<8>::default();
        // Corner 0 marked as air
        get_cell_texture_data(&mut cell, &data, &[0, 1, 2, 3, 4, 5, 6, 7], &weights, 0b1);
        assert_eq!(cell.weights[0], [0, 0, 0, 0]);
        assert_eq!(cell.weights[1], [240, 0, 0, 0]);
    }

    #[test]
    fn selection_picks_the_dominant_materials_sorted() {
        // Corner 0 is material 200, corners 1..7 are material 7
        let mut indices = vec![voxel_core::mixel4::make_encoded_indices_for_single_texture(7); 8];
        indices[0] = voxel_core::mixel4::make_encoded_indices_for_single_texture(200);
        let weights = sampler(8, encode_weights_to_packed_lossy(255, 0, 0, 0));
        let cell =
            select_textures_4_per_voxel(&[0, 1, 2, 3, 4, 5, 6, 7], &indices, &weights, 0);
        assert!(cell.indices.contains(&7), "indices {:?}", cell.indices);
        assert!(cell.indices.contains(&200), "indices {:?}", cell.indices);
        // Sorted ascending
        let mut sorted = cell.indices;
        sorted.sort_unstable();
        assert_eq!(cell.indices, sorted);
    }

    #[test]
    fn selection_fills_unused_slots_with_distinct_indices() {
        // Every corner holds exactly material 10 with a weight, filler slots
        // of the single-texture encoding carry zero weight
        let indices = vec![voxel_core::mixel4::make_encoded_indices_for_single_texture(10); 8];
        let weights = sampler(8, encode_weights_to_packed_lossy(255, 0, 0, 0));
        let cell =
            select_textures_4_per_voxel(&[0, 1, 2, 3, 4, 5, 6, 7], &indices, &weights, 0);
        let mut unique = cell.indices.to_vec();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), 4, "indices {:?}", cell.indices);
    }

    #[test]
    fn null_processor_produces_nothing() {
        let mut p = NullProcessor::default();
        assert_eq!(p.on_cell(&[0; 8], 1), 0);
        assert_eq!(p.on_transition_cell(&[0; 9], 1), 0);
        p.on_vertex(0, 1, 0.5);
        assert!(p.material_data().is_empty());
    }

    #[test]
    fn mixel4_processor_emits_one_attribute_per_vertex() {
        let data = TextureIndicesData {
            buffer: Vec::new(),
            default_indices: [0, 1, 2, 3],
            packed_default_indices: pack_bytes([0, 1, 2, 3]),
        };
        let weights = sampler(8, encode_weights_to_packed_lossy(255, 0, 0, 0));
        let mut p = Mixel4Processor::new(data, weights, true);
        let packed = p.on_cell(&[0, 1, 2, 3, 4, 5, 6, 7], 0);
        assert_eq!(packed, pack_bytes([0, 1, 2, 3]));
        p.on_vertex(0, 1, 0.0);
        p.on_vertex(0, 1, 1.0);
        assert_eq!(p.material_data().len(), 2);
        // At alpha 0 the vertex takes corner 0's weights
        let first = p.material_data()[0];
        assert_eq!(first[0].to_bits(), pack_bytes([0, 1, 2, 3]));
        assert_eq!(first[1].to_bits(), pack_bytes([240, 0, 0, 0]));
    }

    #[test]
    fn mixel4_interpolates_weights_between_corners() {
        let data = TextureIndicesData {
            buffer: Vec::new(),
            default_indices: [0, 1, 2, 3],
            packed_default_indices: pack_bytes([0, 1, 2, 3]),
        };
        let mut sampler_data = vec![encode_weights_to_packed_lossy(255, 0, 0, 0); 8];
        sampler_data[1] = encode_weights_to_packed_lossy(0, 255, 0, 0);
        let weights = WeightSamplerPackedU16 { data: sampler_data };
        let mut p = Mixel4Processor::new(data, weights, false);
        p.on_cell(&[0, 1, 2, 3, 4, 5, 6, 7], 0);
        p.on_vertex(0, 1, 0.5);
        let w = p.material_data()[0][1].to_bits();
        // Halfway between [240,0,0,0] and [0,240,0,0]
        assert_eq!(w & 0xff, 120);
        assert_eq!((w >> 8) & 0xff, 120);
    }
}
