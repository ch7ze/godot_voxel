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

Ref<Material> VoxelMesherHybrid::get_material_by_index(unsigned int index) const {
	// Surface 0 = smooth terrain, surface 1 = blocky models.
	if (index == 0) {
		return _terrain_material;
	}
	if (index == 1) {
		return _model_material;
	}
	return Ref<Material>();
}

unsigned int VoxelMesherHybrid::get_material_index_count() const {
	return 2;
}

void VoxelMesherHybrid::set_terrain_material(Ref<Material> material) {
	_terrain_material = material;
}

Ref<Material> VoxelMesherHybrid::get_terrain_material() const {
	return _terrain_material;
}

void VoxelMesherHybrid::set_model_material(Ref<Material> material) {
	_model_material = material;
}

Ref<Material> VoxelMesherHybrid::get_model_material() const {
	return _model_material;
}

void VoxelMesherHybrid::set_library(Ref<VoxelBlockyLibraryBase> library) {
	_library = library;
}

Ref<VoxelBlockyLibraryBase> VoxelMesherHybrid::get_library() const {
	return _library;
}

void VoxelMesherHybrid::set_flatten_terrain_under_models(bool enabled) {
	_flatten_terrain_under_models = enabled;
}

bool VoxelMesherHybrid::get_flatten_terrain_under_models() const {
	return _flatten_terrain_under_models;
}

bool VoxelMesherHybrid::build_flattened_sdf(VoxelBuffer &dst, const VoxelMesher::Input &input) const {
	ZN_PROFILE_SCOPE();

	const VoxelBuffer &src = input.voxels;
	const VoxelBuffer::ChannelId sdf_channel = VoxelBuffer::CHANNEL_SDF;
	const VoxelBuffer::ChannelId type_channel = VoxelBuffer::CHANNEL_TYPE;

	if (src.is_uniform(type_channel) && src.get_voxel(Vector3i(), type_channel) == blocky::AIR_ID) {
		// No models: no flattening needed, the original SDF can be used directly.
		return false;
	}

	const Vector3i block_size = src.get_size();

	// Work on an own copy of the SDF channel so the original (const) buffer is untouched.
	dst.create(block_size);
	dst.set_channel_depth(sdf_channel, src.get_channel_depth(sdf_channel));
	dst.copy_channel_from(src, sdf_channel);

	const int sx = block_size.x;
	const int sz = block_size.z;

	// Pass 1: for every (x,z) column, find the lowest model voxel (-1 = none).
	StdVector<int> column_model_y;
	column_model_y.resize(sx * sz, -1);
	bool any_model = false;
	for (int z = 0; z < sz; ++z) {
		for (int x = 0; x < sx; ++x) {
			for (int y = 0; y < block_size.y; ++y) {
				if (src.get_voxel(Vector3i(x, y, z), type_channel) != blocky::AIR_ID) {
					column_model_y[x + z * sx] = y;
					any_model = true;
					break;
				}
			}
		}
	}
	if (!any_model) {
		return false;
	}

	bool any_flattened = false;

	// Pass 2: adjust the SDF so each model sits flush, like a small step cut into the slope.
	//
	// A model column itself is hard-flattened to the model's bottom edge: the surface must be
	// exactly there, that is where the model rests.
	//
	// Neighbor columns are not replaced; instead the original SDF is combined with the model's
	// support plane via an SDF union (the minimum of the two signed distances). Union means the
	// ground is only raised where it lies below the plane, and stays original everywhere else.
	// This produces a smooth wedge from the flat model platform down to the surrounding terrain
	// instead of an abrupt hard-edged plateau, and several nearby models combine naturally.
	//
	// Transvoxel interpolates a cell from all 4 corner columns, so adjusting the immediate
	// neighbors is what makes the terrain meet the model cleanly.
	for (int z = 0; z < sz; ++z) {
		for (int x = 0; x < sx; ++x) {
			const int self_model_y = column_model_y[x + z * sx];

			if (self_model_y >= 0) {
				// Model column: hard flatten to the model's bottom edge.
				// Signed distance to that horizontal plane is (y - self_model_y).
				for (int y = 0; y < block_size.y; ++y) {
					dst.set_voxel_f(static_cast<float>(y - self_model_y), Vector3i(x, y, z), sdf_channel);
				}
				any_flattened = true;
				continue;
			}

			// Non-model column: find the nearest model voxel in the 3x3 neighborhood.
			// Picking the nearest (not the first found) keeps the support plane consistent
			// when several models of different heights are close together.
			int support_y = -1;
			int best_dist_sq = 0;
			for (int dz = -1; dz <= 1; ++dz) {
				for (int dx = -1; dx <= 1; ++dx) {
					const int nx = x + dx;
					const int nz = z + dz;
					if (nx < 0 || nx >= sx || nz < 0 || nz >= sz) {
						continue;
					}
					const int my = column_model_y[nx + nz * sx];
					if (my < 0) {
						continue;
					}
					const int dist_sq = dx * dx + dz * dz;
					if (support_y < 0 || dist_sq < best_dist_sq) {
						support_y = my;
						best_dist_sq = dist_sq;
					}
				}
			}
			if (support_y < 0) {
				continue;
			}

			// SDF union with the model's support plane: min(original, distance-to-plane).
			for (int y = 0; y < block_size.y; ++y) {
				const Vector3i pos(x, y, z);
				const float original = dst.get_voxel_f(pos, sdf_channel);
				const float plane = static_cast<float>(y - support_y);
				const float unified = original < plane ? original : plane;
				if (unified != original) {
					dst.set_voxel_f(unified, pos, sdf_channel);
					any_flattened = true;
				}
			}
		}
	}

	return any_flattened;
}

void VoxelMesherHybrid::build(VoxelMesher::Output &output, const VoxelMesher::Input &input) {
	ZN_PROFILE_SCOPE();

	// Surface 0: smooth SDF terrain, delegated to the reused Transvoxel mesher.
	// It appends its surface(s) and sets collision_surface / mesh_flags / primitive_type.
	if (_flatten_terrain_under_models && !_library.is_null()) {
		// Thread-local: build() may run on several threads at once.
		// ALLOCATOR_DEFAULT (not POOL): a thread_local buffer outlives single calls and would
		// otherwise hold VoxelMemoryPool blocks until thread exit, which can happen after the
		// pool singleton is torn down on module unregister (reported as a recycling leak).
		thread_local VoxelBuffer tls_flattened_sdf(VoxelBuffer::ALLOCATOR_DEFAULT);
		if (build_flattened_sdf(tls_flattened_sdf, input)) {
			// Mesh the smooth surface from the flattened SDF copy instead of the original.
			// Input.voxels is a reference, so a new Input must be constructed rather than
			// reassigned. Designated initializers are used so every field is named: if the
			// upstream Input struct gains a field, it keeps its default here instead of a
			// member being silently shifted to a wrong value.
			const VoxelMesher::Input flattened_input{
				.voxels = tls_flattened_sdf,
				.generator = input.generator,
				.origin_in_voxels = input.origin_in_voxels,
				.lod_index = input.lod_index,
				.collision_hint = input.collision_hint,
				.lod_hint = input.lod_hint,
				.detail_texture_hint = input.detail_texture_hint
			};
			_transvoxel_mesher->build(output, flattened_input);
		} else {
			_transvoxel_mesher->build(output, input);
		}
	} else {
		_transvoxel_mesher->build(output, input);
	}

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

	ClassDB::bind_method(D_METHOD("set_terrain_material", "material"), &Self::set_terrain_material);
	ClassDB::bind_method(D_METHOD("get_terrain_material"), &Self::get_terrain_material);

	ClassDB::bind_method(D_METHOD("set_model_material", "material"), &Self::set_model_material);
	ClassDB::bind_method(D_METHOD("get_model_material"), &Self::get_model_material);

	ClassDB::bind_method(
			D_METHOD("set_flatten_terrain_under_models", "enabled"), &Self::set_flatten_terrain_under_models
	);
	ClassDB::bind_method(
			D_METHOD("get_flatten_terrain_under_models"), &Self::get_flatten_terrain_under_models
	);

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
	ADD_PROPERTY(
			PropertyInfo(
					Variant::OBJECT, "terrain_material", PROPERTY_HINT_RESOURCE_TYPE, "Material"
			),
			"set_terrain_material",
			"get_terrain_material"
	);
	ADD_PROPERTY(
			PropertyInfo(
					Variant::OBJECT, "model_material", PROPERTY_HINT_RESOURCE_TYPE, "Material"
			),
			"set_model_material",
			"get_model_material"
	);
	ADD_PROPERTY(
			PropertyInfo(Variant::BOOL, "flatten_terrain_under_models"),
			"set_flatten_terrain_under_models",
			"get_flatten_terrain_under_models"
	);
}

} // namespace zylann::voxel
