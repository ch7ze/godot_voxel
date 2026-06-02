#ifndef VOXEL_TRANSVOXEL_MATERIALS_MIXEL4_H
#define VOXEL_TRANSVOXEL_MATERIALS_MIXEL4_H

#include "../../storage/mixel4.h"
#include "../../util/containers/fixed_array.h"
#include "../../util/containers/std_vector.h"
#include "transvoxel.h"
#include "transvoxel_materials_common.h"

namespace zylann::voxel::transvoxel::materials::mixel4 {

// How many textures can be referred to in total (8-bit index slots -> 256).
static const unsigned int MAX_TEXTURES = 256;
// How many textures can blend at once (unchanged: 4 slots per voxel).
static const unsigned int MAX_TEXTURE_BLENDS = 4;

template <unsigned int NVoxels>
struct CellTextureDatas {
	uint32_t packed_indices = 0;
	FixedArray<uint8_t, MAX_TEXTURE_BLENDS> indices;
	FixedArray<FixedArray<uint8_t, MAX_TEXTURE_BLENDS>, NVoxels> weights;
};

template <unsigned int NVoxels, typename WeightSampler_T>
CellTextureDatas<NVoxels> select_textures_4_per_voxel(
		const FixedArray<unsigned int, NVoxels> &voxel_indices,
		const Span<const uint32_t> indices_data, // 8-bit-per-slot index words (256 materials)
		const WeightSampler_T &weights_sampler,
		const unsigned int case_code
) {
	// A cell has NVoxels corners, each referencing up to 4 material indices with weights. The total
	// number of DISTINCT indices in a cell is therefore at most NVoxels*4 (8*4 = 32), regardless of
	// how large the global material palette (MAX_TEXTURES) is. The previous implementation allocated
	// a MAX_TEXTURES-sized accumulator and sorted all of it per cell; with MAX_TEXTURES == 256 that
	// made this hot function (about half of non-empty-cell polygonization time) scale with the
	// palette size instead of the actual per-cell index count. This version accumulates weight sums
	// only over the indices that actually occur in the cell (a small compact list), so the cost is
	// bound by NVoxels*4, not by MAX_TEXTURES.
	constexpr unsigned int MAX_CELL_INDICES = NVoxels * 4;

	// Decoded per-corner indices/weights, reused below for the weight remap so they are decoded once.
	FixedArray<FixedArray<uint8_t, 4>, NVoxels> corner_indices;
	FixedArray<FixedArray<uint8_t, 4>, NVoxels> corner_weights;

	struct IndexAndWeight {
		unsigned int index;
		unsigned int weight;
	};
	FixedArray<IndexAndWeight, MAX_CELL_INDICES> occurring;
	unsigned int occurring_count = 0;

	for (unsigned int ci = 0; ci < voxel_indices.size(); ++ci) {
		// Air voxels should not contribute
		if ((case_code & (1 << ci)) != 0) {
			fill(corner_weights[ci], uint8_t(0));
			continue;
		}

		const unsigned int data_index = voxel_indices[ci];
		corner_indices[ci] = zylann::voxel::mixel4::decode_indices_from_packed_u16(indices_data[data_index]);
		corner_weights[ci] = weights_sampler.get_weights(data_index);

		for (unsigned int j = 0; j < 4; ++j) {
			const unsigned int ti = corner_indices[ci][j];
			const unsigned int w = corner_weights[ci][j];
			// Accumulate into the existing entry for this index, or append a new one. The list is at
			// most NVoxels*4 long, so this linear scan is small and bounded.
			unsigned int k = 0;
			for (; k < occurring_count; ++k) {
				if (occurring[k].index == ti) {
					occurring[k].weight += w;
					break;
				}
			}
			if (k == occurring_count) {
				occurring[occurring_count++] = IndexAndWeight{ ti, w };
			}
		}
	}

	CellTextureDatas<NVoxels> cell_textures;

	// Partial selection of the 4 highest-weight indices over the (<= NVoxels*4) occurring entries.
	// Selection sort limited to 4 passes -- cheaper than a full sort and independent of MAX_TEXTURES.
	// Missing slots (fewer than 4 distinct indices) are filled with distinct unused indices so the
	// 4 packed indices stay different (required by the blending), with zero weight.
	const unsigned int select_count = math::min(occurring_count, 4u);
	for (unsigned int i = 0; i < select_count; ++i) {
		unsigned int best = i;
		for (unsigned int k = i + 1; k < occurring_count; ++k) {
			if (occurring[k].weight > occurring[best].weight) {
				best = k;
			}
		}
		const IndexAndWeight tmp = occurring[i];
		occurring[i] = occurring[best];
		occurring[best] = tmp;
		cell_textures.indices[i] = static_cast<uint8_t>(occurring[i].index);
	}
	// Fill any unused slots with distinct indices not already selected (weight stays 0).
	if (select_count < 4) {
		uint8_t next_fill = 0;
		for (unsigned int i = select_count; i < 4; ++i) {
			// Advance next_fill until it is not one of the already-chosen indices.
			bool clash = true;
			while (clash) {
				clash = false;
				for (unsigned int s = 0; s < i; ++s) {
					if (cell_textures.indices[s] == next_fill) {
						clash = true;
						++next_fill;
						break;
					}
				}
			}
			cell_textures.indices[i] = next_fill;
			++next_fill;
		}
	}

	// Sort indices to avoid cases that are ambiguous for blending, like 1,2,3,4 and 2,1,3,4
	math::sort(cell_textures.indices[0], cell_textures.indices[1], cell_textures.indices[2], cell_textures.indices[3]);

	cell_textures.packed_indices = pack_bytes(cell_textures.indices);

	// Remap weights to follow the indices we selected, reading the decoded per-corner data directly
	// (no MAX_TEXTURES-sized scratch). For each chosen index, find its weight in the corner's 4 slots.
	for (unsigned int ci = 0; ci < NVoxels; ++ci) {
		FixedArray<uint8_t, 4> &dst_weights = cell_textures.weights[ci];

		// Skip air voxels
		if ((case_code & (1 << ci)) != 0) {
			fill(dst_weights, uint8_t(0));
			continue;
		}

		for (unsigned int i = 0; i < 4; ++i) {
			const uint8_t ti = cell_textures.indices[i];
			uint8_t w = 0;
			for (unsigned int j = 0; j < 4; ++j) {
				if (corner_indices[ci][j] == ti) {
					w = corner_weights[ci][j];
					break;
				}
			}
			dst_weights[i] = w;
		}
	}

	return cell_textures;
}

struct TextureIndicesData {
	// Texture indices for each voxel (8-bit-per-slot index words, 32-bit each)
	Span<const uint32_t> buffer;
	// Used if the buffer is empty
	FixedArray<uint8_t, 4> default_indices;
	uint32_t packed_default_indices;
};

template <unsigned int NVoxels, typename WeightSampler_T>
inline void get_cell_texture_data(
		CellTextureDatas<NVoxels> &cell_textures,
		const TextureIndicesData &texture_indices_data,
		const FixedArray<unsigned int, NVoxels> &voxel_indices,
		const WeightSampler_T &weights_data,
		// Used for rejecting air voxels. Can be set to 0 so all corners are always used.
		const unsigned int case_code
) {
	if (texture_indices_data.buffer.size() == 0) {
		// Indices are known for the whole block, just read weights directly
		cell_textures.indices = texture_indices_data.default_indices;
		cell_textures.packed_indices = texture_indices_data.packed_default_indices;
		for (unsigned int ci = 0; ci < voxel_indices.size(); ++ci) {
			if ((case_code & (1 << ci)) != 0) {
				// Force air voxels to not contribute
				// TODO This is not great, because every Transvoxel vertex interpolates between a matter and air corner.
				// This approach means we would always interpolate towards 0 as a result.
				// Maybe we'll have to use a different approach and remove this option in the future.
				fill(cell_textures.weights[ci], uint8_t(0));
			} else {
				const unsigned int wi = voxel_indices[ci];
				cell_textures.weights[ci] = weights_data.get_weights(wi);
			}
		}

	} else {
		// There can be more than 4 indices or they are not known, so we have to select them
		cell_textures =
				select_textures_4_per_voxel(voxel_indices, texture_indices_data.buffer, weights_data, case_code);
	}
}

struct WeightSamplerPackedU16 {
	Span<const uint16_t> u16_data;
	inline FixedArray<uint8_t, 4> get_weights(unsigned int i) const {
		return zylann::voxel::mixel4::decode_weights_from_packed_u16(u16_data[i]);
	}
};

inline uint16_t reorder_transition_case_code(const uint16_t case_code) {
	// Reorders the case code from a transition cell so bits corresponds to an XYZ iteration order through cell corners.
	//
	// The order of cell corners chosen in transition cells are dependent on how the Transvoxel tables are laid out (see
	// figures 4.16 and 4.17 of the paper), which unfortunately is different from the order in which we sample voxels.
	// That prevents from re-using it in texture selection. The reason for that choice seems to stem from convenience
	// when creating the lookup tables.
	//                       |436785210|
	const uint16_t alt_case_code = 0 //
			| ((case_code & 0b000000111)) // 210
			| ((case_code & 0b110000000) >> 4) // 43
			| ((case_code & 0b000001000) << 2) // 5
			| ((case_code & 0b001000000)) // 6
			| ((case_code & 0b000100000) << 2) // 7
			| ((case_code & 0b000010000) << 4) // 8
			;
	// uint16_t alt_case_code = sign_f(cell_samples[0]);
	// alt_case_code |= (sign_f(cell_samples[1]) << 1);
	// alt_case_code |= (sign_f(cell_samples[2]) << 2);
	// alt_case_code |= (sign_f(cell_samples[3]) << 3);
	// alt_case_code |= (sign_f(cell_samples[4]) << 4);
	// alt_case_code |= (sign_f(cell_samples[5]) << 5);
	// alt_case_code |= (sign_f(cell_samples[6]) << 6);
	// alt_case_code |= (sign_f(cell_samples[7]) << 7);
	// alt_case_code |= (sign_f(cell_samples[8]) << 8);

	return alt_case_code;
}

template <unsigned int NVoxels>
struct Processor {
	const TextureIndicesData voxel_material_indices;
	const WeightSamplerPackedU16 voxel_material_weights;
	const bool textures_skip_air_voxels;
	CellTextureDatas<NVoxels> cell_textures;
	StdVector<Vector2f> &output_mesh_material_data;

	Processor(
			const TextureIndicesData p_voxel_material_indices,
			const WeightSamplerPackedU16 p_voxel_material_weights,
			StdVector<Vector2f> &p_output_mesh_material_data,
			const bool p_textures_skip_air_voxels
	) :
			voxel_material_indices(p_voxel_material_indices),
			voxel_material_weights(p_voxel_material_weights),
			textures_skip_air_voxels(p_textures_skip_air_voxels),
			output_mesh_material_data(p_output_mesh_material_data) {}

	inline uint32_t on_cell(const FixedArray<uint32_t, NVoxels> &corner_voxel_indices, const uint8_t case_code) {
		get_cell_texture_data(
				cell_textures,
				voxel_material_indices,
				corner_voxel_indices,
				voxel_material_weights,
				textures_skip_air_voxels ? case_code : 0
		);

		return cell_textures.packed_indices;
	}

	inline uint32_t on_transition_cell(const FixedArray<uint32_t, 9> &corner_voxel_indices, const uint8_t case_code) {
		const uint16_t alt_case_code = textures_skip_air_voxels ? reorder_transition_case_code(case_code) : 0;

		// Get values from 9 significant corners
		CellTextureDatas<9> cell_textures_partial;
		get_cell_texture_data(
				cell_textures_partial,
				voxel_material_indices,
				corner_voxel_indices,
				voxel_material_weights,
				alt_case_code
		);

		// Fill in slots that are just repeating others

		cell_textures.indices = cell_textures_partial.indices;
		cell_textures.packed_indices = cell_textures_partial.packed_indices;

		fill_redundant_transition_cell_values(cell_textures_partial.weights, cell_textures.weights);

		return cell_textures.packed_indices;
	}

	inline void on_vertex(const unsigned int v0, const unsigned int v1, const float alpha) {
		const FixedArray<uint8_t, MAX_TEXTURE_BLENDS> weights0 = cell_textures.weights[v0];
		const FixedArray<uint8_t, MAX_TEXTURE_BLENDS> weights1 = cell_textures.weights[v1];
		FixedArray<uint8_t, MAX_TEXTURE_BLENDS> weights;
		for (unsigned int i = 0; i < MAX_TEXTURE_BLENDS; ++i) {
			weights[i] = static_cast<uint8_t>(math::clamp(Math::lerp(weights0[i], weights1[i], alpha), 0.f, 255.f));
		}
		add_4i8_4w8_texture_data(output_mesh_material_data, cell_textures.packed_indices, weights);
	}
};

TextureIndicesData get_texture_indices_data(
		const VoxelBuffer &voxels,
		const unsigned int indices_channel,
		DefaultTextureIndicesData &out_default_texture_indices_data
) {
	ZN_ASSERT_RETURN_V(voxels.get_channel_depth(indices_channel) == VoxelBuffer::DEPTH_32_BIT, TextureIndicesData());

	TextureIndicesData data;

	if (voxels.is_uniform(indices_channel)) {
		const uint32_t encoded_indices = voxels.get_voxel(Vector3i(), indices_channel);
		data.default_indices = zylann::voxel::mixel4::decode_indices_from_packed_u16(encoded_indices);
		data.packed_default_indices = pack_bytes(data.default_indices);

		out_default_texture_indices_data.indices = data.default_indices;
		out_default_texture_indices_data.packed_indices = data.packed_default_indices;
		out_default_texture_indices_data.use = true;

	} else {
		Span<const uint8_t> data_bytes;
		ZN_ASSERT(voxels.get_channel_as_bytes_read_only(indices_channel, data_bytes) == true);
		data.buffer = data_bytes.reinterpret_cast_to<const uint32_t>();

		out_default_texture_indices_data.use = false;
	}

	return data;
}

} // namespace zylann::voxel::transvoxel::materials::mixel4

#endif // VOXEL_TRANSVOXEL_MATERIALS_MIXEL4_H
