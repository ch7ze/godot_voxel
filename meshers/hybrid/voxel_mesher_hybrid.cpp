#include "voxel_mesher_hybrid.h"
#include "../../constants/cube_tables.h"
#include "../../storage/voxel_buffer.h"
#include "../../util/containers/std_vector.h"
#include "../../util/godot/core/packed_arrays.h"
#include "../../util/profiling.h"
#include "../blocky/blocky_baked_library.h"

using namespace zylann::godot;

namespace zylann::voxel {

namespace {

// Padding border to skip around the meshed block, matching VoxelMesherBlocky::PADDING (1).
// Kept as a local constant so this file does not need to pull in the whole blocky mesher header.
static constexpr int MODEL_PADDING = 1;

// Accumulates triangles of all baked models found in the TYPE channel into plain arrays.
// Each model's geometry is translated to its voxel position. No neighbor-culling is done:
// every side surface is baked, even if occluded. That keeps this step simple and correct;
// culling is an optimization that can be added later.
struct ModelMeshAccumulator {
	StdVector<Vector3f> positions;
	StdVector<Vector3f> normals;
	StdVector<Vector2f> uvs;
	StdVector<int32_t> indices;

	void append_geometry(
			const StdVector<Vector3f> &src_positions,
			const StdVector<Vector3f> &src_normals,
			const StdVector<Vector2f> &src_uvs,
			const StdVector<int> &src_indices,
			const Vector3f voxel_offset
	) {
		if (src_positions.size() == 0 || src_indices.size() == 0) {
			return;
		}
		const int base_index = static_cast<int>(positions.size());

		for (unsigned int i = 0; i < src_positions.size(); ++i) {
			positions.push_back(src_positions[i] + voxel_offset);
			// Side surfaces don't store normals; fall back to a zero normal in that case.
			normals.push_back(i < src_normals.size() ? src_normals[i] : Vector3f());
			uvs.push_back(i < src_uvs.size() ? src_uvs[i] : Vector2f());
		}
		for (const int idx : src_indices) {
			indices.push_back(base_index + idx);
		}
	}
};

} // namespace

VoxelMesherHybrid::VoxelMesherHybrid() {
	_transvoxel_mesher.instantiate();
	// Padding is dictated by the smooth part. The blocky model part needs no neighbor padding
	// (each model is self-contained within its voxel), so the Transvoxel requirements cover it.
	set_padding(_transvoxel_mesher->get_minimum_padding(), _transvoxel_mesher->get_maximum_padding());
}

VoxelMesherHybrid::~VoxelMesherHybrid() {}

int VoxelMesherHybrid::get_used_channels_mask() const {
	// SDF for the smooth surface, TYPE for the blocky model surface.
	int mask = _transvoxel_mesher->get_used_channels_mask();
	mask |= (1 << VoxelBuffer::CHANNEL_TYPE);
	return mask;
}

bool VoxelMesherHybrid::is_generating_collision_surface() const {
	return _transvoxel_mesher->is_generating_collision_surface();
}

void VoxelMesherHybrid::set_library(Ref<VoxelBlockyLibraryBase> library) {
	_library = library;
}

Ref<VoxelBlockyLibraryBase> VoxelMesherHybrid::get_library() const {
	return _library;
}

void VoxelMesherHybrid::build(VoxelMesher::Output &output, const VoxelMesher::Input &input) {
	ZN_PROFILE_SCOPE();

	// Surface 0: smooth SDF terrain, delegated to the reused Transvoxel mesher.
	// It appends its surface(s) and sets collision_surface / mesh_flags / primitive_type.
	_transvoxel_mesher->build(output, input);

	// Surface 1: sharp-edged 3D models baked from the TYPE channel.
	build_model_surface(output, input);
}

void VoxelMesherHybrid::build_model_surface(VoxelMesher::Output &output, const VoxelMesher::Input &input) {
	ZN_PROFILE_SCOPE();

	if (_library.is_null()) {
		// No models configured: hybrid mesher behaves like plain Transvoxel.
		return;
	}

	const VoxelBuffer &voxels = input.voxels;
	const VoxelBuffer::ChannelId type_channel = VoxelBuffer::CHANNEL_TYPE;

	if (voxels.is_uniform(type_channel) && voxels.get_voxel(Vector3i(), type_channel) == blocky::AIR_ID) {
		// No models anywhere in this block.
		return;
	}

	const Vector3i block_size = voxels.get_size();
	// Same padding convention as the blocky mesher: skip the padding border.
	const Vector3i min = Vector3iUtil::create(MODEL_PADDING);
	const Vector3i max = block_size - Vector3iUtil::create(MODEL_PADDING);
	if (min.x >= max.x || min.y >= max.y || min.z >= max.z) {
		return;
	}

	ModelMeshAccumulator acc;

	{
		// Baked data is the only library data safe for multithreaded access.
		RWLockRead lock(_library->get_baked_data_rw_lock());
		const blocky::BakedLibrary &baked = _library->get_baked_data();

		for (int z = min.z; z < max.z; ++z) {
			for (int x = min.x; x < max.x; ++x) {
				for (int y = min.y; y < max.y; ++y) {
					const uint64_t voxel_id = voxels.get_voxel(Vector3i(x, y, z), type_channel);

					if (voxel_id == blocky::AIR_ID || !baked.has_model(voxel_id)) {
						continue;
					}
					const blocky::BakedModel &baked_model = baked.models[voxel_id];
					if (baked_model.empty) {
						continue;
					}
					const blocky::BakedModel::Model &model = baked_model.model;

					// Position relative to the (unpadded) mesh origin.
					const Vector3f voxel_offset(x - min.x, y - min.y, z - min.z);

					for (unsigned int s = 0; s < model.surface_count; ++s) {
						// Inner geometry of the model.
						const blocky::BakedModel::Surface &surface = model.surfaces[s];
						acc.append_geometry(
								surface.positions, surface.normals, surface.uvs, surface.indices, voxel_offset
						);
						// All 6 cube sides (not culled against neighbors in this step).
						for (unsigned int side = 0; side < Cube::SIDE_COUNT; ++side) {
							const blocky::BakedModel::SideSurface &side_surface = model.sides_surfaces[side][s];
							acc.append_geometry(
									side_surface.positions,
									StdVector<Vector3f>(),
									side_surface.uvs,
									side_surface.indices,
									voxel_offset
							);
						}
					}
				}
			}
		}
	}

	if (acc.indices.size() == 0) {
		return;
	}

	// Apply LOD scaling, same approach as VoxelMesherBlocky: positions are scaled up so a
	// model occupies the larger world-space of a coarse-LOD voxel.
	if (input.lod_index > 0) {
		const float lod_scale = static_cast<float>(1 << input.lod_index);
		for (Vector3f &p : acc.positions) {
			p = p * lod_scale;
		}
	}

	PackedVector3Array gd_positions;
	PackedVector3Array gd_normals;
	PackedVector2Array gd_uvs;
	PackedInt32Array gd_indices;
	copy_to(gd_positions, to_span_const(acc.positions));
	copy_to(gd_normals, to_span_const(acc.normals));
	copy_to(gd_uvs, to_span_const(acc.uvs));
	copy_to(gd_indices, to_span_const(acc.indices));

	Array arrays;
	arrays.resize(Mesh::ARRAY_MAX);
	arrays[Mesh::ARRAY_VERTEX] = gd_positions;
	arrays[Mesh::ARRAY_NORMAL] = gd_normals;
	arrays[Mesh::ARRAY_TEX_UV] = gd_uvs;
	arrays[Mesh::ARRAY_INDEX] = gd_indices;

	// Material index 1: distinct from the smooth surface (index 0) so each gets its own material.
	output.surfaces.push_back({ arrays, 1 });
}

void VoxelMesherHybrid::_bind_methods() {
	using Self = VoxelMesherHybrid;

	ClassDB::bind_method(D_METHOD("set_library", "library"), &Self::set_library);
	ClassDB::bind_method(D_METHOD("get_library"), &Self::get_library);

	ADD_PROPERTY(
			PropertyInfo(
					Variant::OBJECT,
					"library",
					PROPERTY_HINT_RESOURCE_TYPE,
					VoxelBlockyLibraryBase::get_class_static()
			),
			"set_library",
			"get_library"
	);
}

} // namespace zylann::voxel
