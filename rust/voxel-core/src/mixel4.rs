//! Material index and weight packing, ported from `storage/mixel4.h`.
//!
//! Indices occupy 8 bits per slot in a 32-bit word, allowing 256 distinct
//! materials (source: storage/mixel4.h:40-44). Weights keep 4 bits per slot in
//! a 16-bit word (source: storage/mixel4.h:16-38).

use crate::channel::Depth;

/// Number of blend slots per voxel.
pub const SLOT_COUNT: usize = 4;

/// Decodes the four 4-bit weights into 8-bit values.
/// Maximum decoded weight is 240, not 255, which preserves bijectivity.
/// Source: storage/mixel4.h:16-38
pub fn decode_weights_from_packed(packed_weights: u16) -> [u8; SLOT_COUNT] {
    [
        ((packed_weights & 0x0f) << 4) as u8,
        (packed_weights & 0xf0) as u8,
        ((packed_weights >> 4) & 0xf0) as u8,
        ((packed_weights >> 8) & 0xf0) as u8,
    ]
}

/// Decodes the four 8-bit material indices from a 32-bit word.
/// Source: storage/mixel4.h:45-52
pub fn decode_indices_from_packed(packed_indices: u32) -> [u8; SLOT_COUNT] {
    [
        (packed_indices & 0xff) as u8,
        ((packed_indices >> 8) & 0xff) as u8,
        ((packed_indices >> 16) & 0xff) as u8,
        ((packed_indices >> 24) & 0xff) as u8,
    ]
}

/// Source: storage/mixel4.h:54-56
pub const fn encode_indices_to_packed(a: u8, b: u8, c: u8, d: u8) -> u32 {
    (a as u32) | ((b as u32) << 8) | ((c as u32) << 16) | ((d as u32) << 24)
}

/// Encodes 0..255 weights into 4 bits each. Lower 4 bits of input are dropped.
/// Source: storage/mixel4.h:59-61
pub const fn encode_weights_to_packed_lossy(a: u8, b: u8, c: u8, d: u8) -> u16 {
    ((a >> 4) as u16)
        | (((b >> 4) as u16) << 4)
        | (((c >> 4) as u16) << 8)
        | (((d >> 4) as u16) << 12)
}

/// Returns true when no material index appears twice.
/// Source: storage/mixel4.h:66-74
pub fn indices_are_distinct(indices: [u8; SLOT_COUNT]) -> bool {
    let mut checked = [false; 256];
    let mut distinct = true;
    for index in indices {
        if checked[index as usize] {
            distinct = false;
        }
        checked[index as usize] = true;
    }
    distinct
}

/// Rescales the three non-preserved weights so the four sum to 1.
/// Source: storage/mixel4.h:76-98
fn normalize_weights_preserving_slots(
    weights: &mut [f32; SLOT_COUNT],
    preserved_index: usize,
    other0: usize,
    other1: usize,
    other2: usize,
) {
    let part_sum = weights[other0] + weights[other1] + weights[other2];
    // The preserved slot is assumed to be clamped to [0, 1] already
    let expected_part_sum = 1.0 - weights[preserved_index];

    if part_sum < 0.0001 {
        let third = expected_part_sum / 3.0;
        weights[other0] = third;
        weights[other1] = third;
        weights[other2] = third;
    } else {
        let scale = expected_part_sum / part_sum;
        weights[other0] *= scale;
        weights[other1] *= scale;
        weights[other2] *= scale;
    }
}

/// Slot ordering per preserved index matches the C++ switch exactly, including
/// the asymmetric orders for slots 2 and 3.
/// Source: storage/mixel4.h:100-115
pub fn normalize_weights_preserving(weights: &mut [f32; SLOT_COUNT], preserved_index: usize) {
    match preserved_index {
        0 => normalize_weights_preserving_slots(weights, 0, 1, 2, 3),
        1 => normalize_weights_preserving_slots(weights, 1, 0, 2, 3),
        2 => normalize_weights_preserving_slots(weights, 2, 1, 0, 3),
        _ => normalize_weights_preserving_slots(weights, 3, 1, 2, 0),
    }
}

/// Blends a material into the packed index and weight words in place.
/// Source: storage/mixel4.h:128-185
pub fn blend_texture_packed(
    texture_index: u8,
    target_weight: f32,
    encoded_indices: &mut u32,
    encoded_weights: &mut u16,
) {
    debug_assert!((0.0..=1.0).contains(&target_weight));

    let mut indices = decode_indices_from_packed(*encoded_indices);
    let mut weights = decode_weights_from_packed(*encoded_weights);

    // Search whether the material is already present
    let mut component_index = SLOT_COUNT;
    for (i, index) in indices.iter().enumerate() {
        if *index == texture_index {
            component_index = i;
            break;
        }
    }

    let mut index_was_changed = false;
    if component_index >= SLOT_COUNT {
        // Not present, replace the lowest weight. Default to slot 0 when all are maxed.
        let mut lowest_weight = u8::MAX;
        component_index = 0;
        for (i, weight) in weights.iter().enumerate() {
            if *weight < lowest_weight {
                lowest_weight = *weight;
                component_index = i;
            }
        }
        indices[component_index] = texture_index;
        index_was_changed = true;
    }

    let mut weights_f = [0.0f32; SLOT_COUNT];
    for i in 0..SLOT_COUNT {
        weights_f[i] = weights[i] as f32 / 255.0;
    }

    if weights_f[component_index] < target_weight || index_was_changed {
        weights_f[component_index] = target_weight;

        normalize_weights_preserving(&mut weights_f, component_index);

        for i in 0..SLOT_COUNT {
            weights[i] = (weights_f[i] * 255.0).clamp(0.0, 255.0) as u8;
        }

        *encoded_indices = encode_indices_to_packed(indices[0], indices[1], indices[2], indices[3]);
        *encoded_weights =
            encode_weights_to_packed_lossy(weights[0], weights[1], weights[2], weights[3]);
    }
}

/// Weight word putting all weight on slot 0.
/// Source: storage/mixel4.h:189-191
pub const fn make_encoded_weights_for_single_texture() -> u16 {
    encode_weights_to_packed_lossy(255, 0, 0, 0)
}

/// Index word for a single material. Filler slots are distinct so their weights
/// cannot override slot 0. Fillers wrap at 256 because slots hold 8 bits.
/// Source: storage/mixel4.h:193-206
pub const fn make_encoded_indices_for_single_texture(index: u8) -> u32 {
    let index1 = index.wrapping_add(1);
    let index2 = index.wrapping_add(2);
    let index3 = index.wrapping_add(3);
    encode_indices_to_packed(index, index1, index2, index3)
}

/// Default packed index word, slots 0, 1, 2, 3.
/// Source: storage/voxel_buffer.cpp:93
pub const DEFAULT_INDICES: u32 = encode_indices_to_packed(0, 1, 2, 3);

/// Default packed weight word, all weight on slot 0.
/// Source: storage/voxel_buffer.cpp:94
pub const DEFAULT_WEIGHTS: u16 = encode_weights_to_packed_lossy(255, 0, 0, 0);

/// Default raw value of the indices channel per depth.
/// The 16-bit case keeps the legacy 4-bit-per-slot layout.
/// Source: storage/voxel_buffer.cpp:127-141
pub fn default_indices_raw_value(depth: Depth) -> u64 {
    match depth {
        Depth::D8 => 0,
        Depth::D16 => (encode_indices_to_packed(0, 1, 2, 3) & 0xffff) as u64,
        Depth::D32 => DEFAULT_INDICES as u64,
        Depth::D64 => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_round_trip_covers_full_byte_range() {
        let packed = encode_indices_to_packed(0, 42, 200, 255);
        assert_eq!(decode_indices_from_packed(packed), [0, 42, 200, 255]);
    }

    #[test]
    fn weight_decode_caps_at_240() {
        let packed = encode_weights_to_packed_lossy(255, 255, 255, 255);
        assert_eq!(decode_weights_from_packed(packed), [240, 240, 240, 240]);
    }

    #[test]
    fn single_texture_weights_put_everything_on_slot_zero() {
        let weights = decode_weights_from_packed(make_encoded_weights_for_single_texture());
        assert_eq!(weights, [240, 0, 0, 0]);
    }

    #[test]
    fn single_texture_indices_are_distinct() {
        for index in [0u8, 1, 127, 253, 254, 255] {
            let packed = make_encoded_indices_for_single_texture(index);
            let indices = decode_indices_from_packed(packed);
            assert_eq!(indices[0], index);
            assert!(
                indices_are_distinct(indices),
                "index {index} produced duplicates {indices:?}"
            );
        }
    }

    #[test]
    fn default_constants_match_cpp() {
        assert_eq!(DEFAULT_INDICES, encode_indices_to_packed(0, 1, 2, 3));
        assert_eq!(default_indices_raw_value(Depth::D32), DEFAULT_INDICES as u64);
        assert_eq!(default_indices_raw_value(Depth::D16), 0x0100);
    }

    #[test]
    fn blend_inserts_new_material_into_lowest_weight_slot() {
        let mut indices = DEFAULT_INDICES;
        let mut weights = DEFAULT_WEIGHTS;
        blend_texture_packed(200, 1.0, &mut indices, &mut weights);
        let decoded = decode_indices_from_packed(indices);
        assert!(decoded.contains(&200), "indices were {decoded:?}");
    }

    #[test]
    fn normalize_preserves_requested_slot_and_sums_to_one() {
        let mut weights = [0.25f32, 0.25, 0.25, 0.25];
        weights[1] = 0.7;
        normalize_weights_preserving(&mut weights, 1);
        assert!((weights[1] - 0.7).abs() < 1e-6);
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum was {sum}");
    }

    #[test]
    fn normalize_distributes_evenly_when_others_are_zero() {
        let mut weights = [1.0f32, 0.0, 0.0, 0.0];
        weights[0] = 0.4;
        normalize_weights_preserving(&mut weights, 0);
        assert!((weights[1] - 0.2).abs() < 1e-6);
        assert!((weights[2] - 0.2).abs() < 1e-6);
        assert!((weights[3] - 0.2).abs() < 1e-6);
    }
}
