#include "test_voxel_mesher_hybrid.h"
#include "../../meshers/blocky/voxel_blocky_library.h"
#include "../../meshers/blocky/voxel_blocky_model_cube.h"
#include "../../meshers/blocky/voxel_blocky_model_empty.h"
#include "../../meshers/hybrid/voxel_mesher_hybrid.h"
#include "../../storage/voxel_buffer.h"
#include "../../util/godot/classes/mesh.h"
#include "../../util/testing/test_macros.h"

namespace zylann::voxel::tests {

// Verifies that VoxelMesherHybrid produces both surfaces from a single buffer:
//   - Surface 0: smooth iso-surface from the SDF channel (Transvoxel),
//   - Surface 1: a blocky model baked from the TYPE channel.
void test_voxel_mesher_hybrid() {
	const int size = 16;

	VoxelBuffer vb(VoxelBuffer::ALLOCATOR_DEFAULT);
	vb.create(size, size, size);
	vb.set_channel_depth(VoxelBuffer::CHANNEL_SDF, VoxelBuffer::DEPTH_16_BIT);
	vb.set_channel_depth(VoxelBuffer::CHANNEL_TYPE, VoxelBuffer::DEPTH_16_BIT);

	// Fill the SDF channel so it crosses the isolevel: lower half solid (negative),
	// upper half air (positive). Without a sign change the Transvoxel mesher skips the block.
	for (int z = 0; z < size; ++z) {
		for (int x = 0; x < size; ++x) {
			for (int y = 0; y < size; ++y) {
				const float sdf = static_cast<float>(y) - static_cast<float>(size) * 0.5f;
				vb.set_voxel_f(sdf, Vector3i(x, y, z), VoxelBuffer::CHANNEL_SDF);
			}
		}
	}

	// Build a library: index 0 = air (empty), index 1 = a cube model.
	Ref<VoxelBlockyLibrary> library;
	library.instantiate();
	{
		Ref<VoxelBlockyModelEmpty> air;
		air.instantiate();
		library->add_model(air);

		Ref<VoxelBlockyModelCube> cube;
		cube.instantiate();
		library->add_model(cube);
	}
	library->bake();

	// Place a couple of model voxels (id 1) in the TYPE channel, away from the padding border.
	vb.set_voxel(1, Vector3i(6, 9, 6), VoxelBuffer::CHANNEL_TYPE);
	vb.set_voxel(1, Vector3i(7, 9, 6), VoxelBuffer::CHANNEL_TYPE);

	Ref<VoxelMesherHybrid> mesher;
	mesher.instantiate();
	mesher->set_library(library);

	VoxelMesher::Input input{ vb, nullptr, Vector3i(), 0, false };
	VoxelMesher::Output output;
	mesher->build(output, input);

	// Expect two surfaces: smooth terrain (0) and blocky models (1).
	ZN_TEST_ASSERT(output.surfaces.size() == 2);

	const VoxelMesher::Output::Surface &smooth_surface = output.surfaces[0];
	const VoxelMesher::Output::Surface &model_surface = output.surfaces[1];

	ZN_TEST_ASSERT(smooth_surface.material_index == 0);
	ZN_TEST_ASSERT(model_surface.material_index == 1);

	const PackedVector3Array smooth_vertices = smooth_surface.arrays[Mesh::ARRAY_VERTEX];
	const PackedVector3Array model_vertices = model_surface.arrays[Mesh::ARRAY_VERTEX];
	const PackedInt32Array model_indices = model_surface.arrays[Mesh::ARRAY_INDEX];

	// The smooth surface must contain the iso-surface geometry.
	ZN_TEST_ASSERT(smooth_vertices.size() > 0);

	// The model surface must contain the two baked cubes.
	ZN_TEST_ASSERT(model_vertices.size() > 0);
	ZN_TEST_ASSERT(model_indices.size() > 0);
	ZN_TEST_ASSERT((model_indices.size() % 3) == 0);

	// With no library set, the hybrid mesher must behave like plain Transvoxel (only surface 0).
	{
		Ref<VoxelMesherHybrid> mesher_no_lib;
		mesher_no_lib.instantiate();
		VoxelMesher::Output output_no_lib;
		mesher_no_lib->build(output_no_lib, input);
		ZN_TEST_ASSERT(output_no_lib.surfaces.size() == 1);
	}
}

} // namespace zylann::voxel::tests
