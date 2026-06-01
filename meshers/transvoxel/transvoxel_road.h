#ifndef VOXEL_TRANSVOXEL_ROAD_H
#define VOXEL_TRANSVOXEL_ROAD_H

#include "../../util/containers/fixed_array.h"
#include "../../util/containers/span.h"
#include "../../util/containers/std_vector.h"
#include "transvoxel.h"
#include "transvoxel_materials_common.h" // fill_redundant_transition_cell_values

namespace zylann::voxel::transvoxel::road {

// Optional per-vertex "road" scalar sampler, composed into the material processors.
//
// It reads one extra SDF channel (e.g. CHANNEL_DATA6 for land roads, CHANNEL_DATA7 later for
// water roads) and emits, for every generated vertex, the channel value interpolated along the
// same cell edge the position uses. The terrain/water shader reads it from ARRAY_CUSTOM3 and
// turns its sign into a sharp material edge (inside road < 0, outside >= 0).
//
// Channel-parametric: the channel is chosen by the caller (set_road_channel), so the exact same
// code serves land roads and, later, water roads -- only the channel and the consuming shader
// differ.
//
// Inactive by default: when the data span is empty (no road channel configured, or the channel is
// uniform far-outside), nothing is sampled and output.road_data stays empty, leaving non-road
// meshing byte-for-byte unchanged.
//
// Mirrors the mixel4 processor's cell/vertex split: on_cell caches the 8 corner road values of the
// current 2x2x2 cell (or 9 for a transition cell), on_vertex interpolates two of them with the
// edge alpha and appends one float to output.road_data. The vertex order in road_data therefore
// matches vertices/normals/lod_data exactly (one push per add_vertex).
template <unsigned int NVoxels>
struct RoadSampler {
	// Source channel values for the whole block, one float-equivalent per voxel. Empty = inactive.
	Span<const float> data;
	// Destination per-vertex stream (output.road_data).
	StdVector<float> &out;

	// Cached corner values for the current cell, filled by on_cell from the corner voxel indices.
	FixedArray<float, NVoxels> corner_values;
	bool active = false;

	RoadSampler(Span<const float> p_data, StdVector<float> &p_out) : data(p_data), out(p_out) {
		active = (data.size() > 0);
	}

	inline void on_cell(const FixedArray<uint32_t, NVoxels> &corner_voxel_indices) {
		if (!active) {
			return;
		}
		for (unsigned int i = 0; i < NVoxels; ++i) {
			corner_values[i] = data[corner_voxel_indices[i]];
		}
	}

	// Transition cells carry 9 significant corners; the remaining slots (up to NVoxels, normally
	// 13) just repeat four of them, mirroring fill_redundant_transition_cell_values used for the
	// material weights, so on_vertex's v0/v1 indices stay valid.
	inline void on_transition_cell(const FixedArray<uint32_t, 9> &corner_voxel_indices) {
		if (!active) {
			return;
		}
		FixedArray<float, 9> partial;
		for (unsigned int i = 0; i < 9; ++i) {
			partial[i] = data[corner_voxel_indices[i]];
		}
		materials::fill_redundant_transition_cell_values(partial, corner_values);
	}

	// alpha is the weight of corner v0 (same convention as the mixel4 processor's on_vertex).
	inline void on_vertex(const unsigned int v0, const unsigned int v1, const float alpha) {
		if (!active) {
			return;
		}
		const float value = corner_values[v0] * alpha + corner_values[v1] * (1.f - alpha);
		out.push_back(value);
	}
};

} // namespace zylann::voxel::transvoxel::road

#endif // VOXEL_TRANSVOXEL_ROAD_H
