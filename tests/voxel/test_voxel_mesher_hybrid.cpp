#include "test_voxel_mesher_hybrid.h"
#include "../../meshers/blocky/voxel_blocky_library.h"
#include "../../meshers/blocky/voxel_blocky_model_cube.h"
#include "../../meshers/blocky/voxel_blocky_model_empty.h"
#include "../../meshers/hybrid/voxel_mesher_hybrid.h"
#include "../../storage/voxel_buffer.h"
#include "../../util/godot/classes/mesh.h"
#include "../../util/godot/classes/rendering_server.h"
#include "../../util/godot/classes/shader.h"
#include "../../util/godot/classes/shader_material.h"
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

	// The collision surface must use explicit arrays (submesh markers cleared) and contain
	// both the terrain geometry and the model triangles, so models are solid.
	const VoxelMesher::Output::CollisionSurface &col = output.collision_surface;
	ZN_TEST_ASSERT(col.submesh_vertex_end == -1);
	ZN_TEST_ASSERT(col.submesh_index_end == -1);
	ZN_TEST_ASSERT(col.positions.size() > 0);
	ZN_TEST_ASSERT(col.indices.size() > 0);
	ZN_TEST_ASSERT((col.indices.size() % 3) == 0);
	// It must hold more than the terrain alone: at least the model triangles were appended.
	ZN_TEST_ASSERT(col.positions.size() >= static_cast<size_t>(model_vertices.size()));
	// Every index must point at a valid vertex.
	for (const int idx : col.indices) {
		ZN_TEST_ASSERT(idx >= 0 && idx < static_cast<int>(col.positions.size()));
	}

	// With no library set, the hybrid mesher must behave like plain Transvoxel (only surface 0).
	{
		Ref<VoxelMesherHybrid> mesher_no_lib;
		mesher_no_lib.instantiate();
		VoxelMesher::Output output_no_lib;
		mesher_no_lib->build(output_no_lib, input);
		ZN_TEST_ASSERT(output_no_lib.surfaces.size() == 1);
	}

	// Neighbor side-culling: two touching cubes must produce fewer model vertices than two
	// isolated cubes, because their shared faces are culled.
	{
		// Buffer with a single isolated model cube.
		VoxelBuffer vb_one(VoxelBuffer::ALLOCATOR_DEFAULT);
		vb_one.create(size, size, size);
		vb_one.set_channel_depth(VoxelBuffer::CHANNEL_SDF, VoxelBuffer::DEPTH_16_BIT);
		vb_one.set_channel_depth(VoxelBuffer::CHANNEL_TYPE, VoxelBuffer::DEPTH_16_BIT);
		for (int z = 0; z < size; ++z) {
			for (int x = 0; x < size; ++x) {
				for (int y = 0; y < size; ++y) {
					const float sdf = static_cast<float>(y) - static_cast<float>(size) * 0.5f;
					vb_one.set_voxel_f(sdf, Vector3i(x, y, z), VoxelBuffer::CHANNEL_SDF);
				}
			}
		}
		vb_one.set_voxel(1, Vector3i(6, 9, 6), VoxelBuffer::CHANNEL_TYPE);

		VoxelMesher::Input input_one{ vb_one, nullptr, Vector3i(), 0, false };
		VoxelMesher::Output output_one;
		mesher->build(output_one, input_one);

		ZN_TEST_ASSERT(output_one.surfaces.size() == 2);
		const PackedVector3Array one_cube_vertices = output_one.surfaces[1].arrays[Mesh::ARRAY_VERTEX];

		// `output` above has two adjacent cubes. With culling, they share one face pair, so
		// their combined vertex count must be less than twice a single isolated cube.
		ZN_TEST_ASSERT(model_vertices.size() < 2 * one_cube_vertices.size());
	}

	// Shared vertex format: the terrain renders all surfaces with one material_override (the
	// default LOD shader), so every surface must carry CUSTOM0 (4 floats: Transvoxel LOD
	// attributes) and CUSTOM1 (1 float: material id), sized to the vertex count.
	{
		for (unsigned int s = 0; s < output.surfaces.size(); ++s) {
			const Array &arrays = output.surfaces[s].arrays;
			const PackedVector3Array verts = arrays[Mesh::ARRAY_VERTEX];
			const PackedFloat32Array custom0 = arrays[Mesh::ARRAY_CUSTOM0];
			const PackedFloat32Array custom1 = arrays[Mesh::ARRAY_CUSTOM1];
			ZN_TEST_ASSERT(custom0.size() == verts.size() * 4);
			ZN_TEST_ASSERT(custom1.size() == verts.size() * 1);
		}
		// mesh_flags must declare the format of BOTH custom streams. build_mesh() applies one
		// flags value to every surface; a missing CUSTOM0 field would make Godot read it as
		// RGBA8_UNORM and reject the PackedFloat32Array.
		const uint32_t custom0_format =
				(output.mesh_flags >> RenderingServer::ARRAY_FORMAT_CUSTOM0_SHIFT) & RenderingServer::ARRAY_FORMAT_CUSTOM_MASK;
		const uint32_t custom1_format =
				(output.mesh_flags >> RenderingServer::ARRAY_FORMAT_CUSTOM1_SHIFT) & RenderingServer::ARRAY_FORMAT_CUSTOM_MASK;
		ZN_TEST_ASSERT(custom0_format == RenderingServer::ARRAY_CUSTOM_RGBA_FLOAT);
		ZN_TEST_ASSERT(custom1_format == RenderingServer::ARRAY_CUSTOM_R_FLOAT);
	}

	// Regression: a chunk with models but no smooth terrain. The SDF is uniform, so the inner
	// Transvoxel mesher produces no surface and never sets mesh_flags. The hybrid mesher must
	// still declare the CUSTOM0 format itself -- otherwise the model surface's CUSTOM0 array
	// would be rejected by Godot ("Invalid array format for surface").
	//
	// Flattening is disabled here: it would union a model support plane into the uniform SDF,
	// introduce a sign change, and make Transvoxel produce a surface after all -- which would
	// defeat the "no smooth terrain" condition this case is meant to exercise.
	{
		VoxelBuffer vb_models_only(VoxelBuffer::ALLOCATOR_DEFAULT);
		vb_models_only.create(size, size, size);
		vb_models_only.set_channel_depth(VoxelBuffer::CHANNEL_SDF, VoxelBuffer::DEPTH_16_BIT);
		vb_models_only.set_channel_depth(VoxelBuffer::CHANNEL_TYPE, VoxelBuffer::DEPTH_16_BIT);
		// SDF left at its uniform default (no sign change -> Transvoxel produces nothing).
		vb_models_only.set_voxel(1, Vector3i(6, 8, 6), VoxelBuffer::CHANNEL_TYPE);

		Ref<VoxelMesherHybrid> mesher_models;
		mesher_models.instantiate();
		mesher_models->set_library(library);
		mesher_models->set_flatten_terrain_under_models(false);

		VoxelMesher::Input input_models{ vb_models_only, nullptr, Vector3i(), 0, false };
		VoxelMesher::Output output_models;
		mesher_models->build(output_models, input_models);

		// Only the model surface exists; no smooth terrain surface.
		ZN_TEST_ASSERT(output_models.surfaces.size() == 1);
		ZN_TEST_ASSERT(output_models.surfaces[0].material_index == 1);

		// The CUSTOM0 format field must be set even though Transvoxel produced no surface.
		const uint32_t custom0_format =
				(output_models.mesh_flags >> RenderingServer::ARRAY_FORMAT_CUSTOM0_SHIFT) &
				RenderingServer::ARRAY_FORMAT_CUSTOM_MASK;
		ZN_TEST_ASSERT(custom0_format == RenderingServer::ARRAY_CUSTOM_RGBA_FLOAT);

		// And the model surface must actually carry a matching CUSTOM0 array.
		const Array &model_arrays = output_models.surfaces[0].arrays;
		const PackedVector3Array mverts = model_arrays[Mesh::ARRAY_VERTEX];
		const PackedFloat32Array mcustom0 = model_arrays[Mesh::ARRAY_CUSTOM0];
		ZN_TEST_ASSERT(mverts.size() > 0);
		ZN_TEST_ASSERT(mcustom0.size() == mverts.size() * 4);
	}

	// The default LOD material must be a valid ShaderMaterial with a shader assigned: the
	// terrain uses it as the material_override that culls Transvoxel transition meshes.
	{
		Ref<VoxelMesherHybrid> mesher_mat;
		mesher_mat.instantiate();
		Ref<ShaderMaterial> lod_material = mesher_mat->get_default_lod_material();
		ZN_TEST_ASSERT(lod_material.is_valid());
		ZN_TEST_ASSERT(lod_material->get_shader().is_valid());
	}
}

} // namespace zylann::voxel::tests
