#ifndef VOXEL_TRANSVOXEL_MATERIALS_NULL_H
#define VOXEL_TRANSVOXEL_MATERIALS_NULL_H

#include "transvoxel.h"
#include "transvoxel_road.h"

namespace zylann::voxel::transvoxel::materials {

// NVoxels is 8 for regular cells and 13 for transition cells, matching how the mixel4 processor is
// instantiated. The road sampler is templated on it so the same untextured pass (e.g. the water
// surface) can still carry a per-vertex road scalar -- inactive (no output) unless a road channel
// is configured, so the default water/None meshing is unchanged.
template <unsigned int NVoxels>
struct NullProcessorT {
	road::RoadSampler<NVoxels> road;

	explicit NullProcessorT(const road::RoadSampler<NVoxels> p_road) : road(p_road) {}

	// Called for every 2x2x2 cell containing triangles.
	// The returned value is used to determine if the next cell can re-use vertices from previous cells, when equal.
	inline uint32_t on_cell(const FixedArray<uint32_t, NVoxels> &corner_voxel_indices, const uint8_t case_code) {
		road.on_cell(corner_voxel_indices);
		return 0;
	}
	// Called for every 2x3x3 transition cell containing triangles.
	// Such cells are actually in 2D data-wise, so corners are the same value, so only 9 are passed in.
	// The returned value is used to determine if the next cell can re-use vertices from previous cells, when equal.
	inline uint32_t on_transition_cell(const FixedArray<uint32_t, 9> &corner_voxel_indices, const uint8_t case_code) {
		road.on_transition_cell(corner_voxel_indices);
		return 0;
	}
	// Called one or more times after each `on_cell` for every new vertex, to interpolate and add material data
	inline void on_vertex(const unsigned int v0, const unsigned int v1, const float alpha) {
		road.on_vertex(v0, v1, alpha);
	}
};

} // namespace zylann::voxel::transvoxel::materials

#endif // VOXEL_TRANSVOXEL_MATERIALS_NULL_H
