#ifndef VOXEL_BLOCKY_FLUIDS_H
#define VOXEL_BLOCKY_FLUIDS_H

#include "blocky_baked_library.h"

namespace zylann::voxel::blocky {

// Defined inline in blocky_fluids_meshing_impl.h.
// The matching `inline` here keeps the declaration consistent with the definition.
inline void generate_preview_fluid_model(
		const BakedModel &model,
		const uint16_t model_id,
		const BakedLibrary &library,
		Span<const BakedModel::Surface> &out_model_surfaces,
		const FixedArray<FixedArray<BakedModel::SideSurface, MAX_SURFACES>, Cube::SIDE_COUNT> *&out_model_sides_surfaces
);

} // namespace zylann::voxel::blocky

#endif // VOXEL_BLOCKY_FLUIDS_H
