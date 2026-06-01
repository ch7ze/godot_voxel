#include "voxel_tool_lod_terrain.h"
#include "../constants/voxel_string_names.h"
#include "../generators/graph/voxel_generator_graph.h"
#include "../meshers/blocky/voxel_mesher_blocky.h"
#include "../storage/mixel4.h"
#include "../storage/voxel_buffer_gd.h"
#include "../storage/voxel_data_grid.h"
#include "../terrain/variable_lod/voxel_lod_terrain.h"
#include "../util/containers/std_vector.h"
#include "../util/dstack.h"
#include "../util/island_finder.h"
#include "../util/math/conv.h"
#include "../util/string/format.h"
#include "../util/tasks/async_dependency_tracker.h"
#include "../util/voxel_raycast.h"
#include "floating_chunks.h"
#include "funcs.h"
#include "raycast.h"

#ifdef VOXEL_ENABLE_MESH_SDF
#include "voxel_mesh_sdf_gd.h"
#endif

namespace zylann::voxel {

namespace {

// Converts a world-space point + radius to terrain-local voxel space, given the terrain's
// global transform. Mirrors the convention already used by raycast_generic_world: the caller
// passes world coordinates, the operation works in voxel indices. Required for consistency
// with raycast; otherwise scaling the terrain node would silently break edits while raycasts
// keep working.
//
// For non-uniform scales the function uses the maximum-magnitude axis and warns once: edit
// operations conceptually need a uniform scale factor for radius, and non-uniform terrain
// transforms are an unsupported edge case in this fork.
inline void world_to_local_sphere_lod(const Transform3D &to_world, Vector3 &io_center, float &io_radius) {
	const Transform3D to_local = to_world.affine_inverse();
	io_center = to_local.xform(io_center);
	const Vector3 scale = to_world.basis.get_scale();
	const float sx = Math::abs(scale.x);
	const float sy = Math::abs(scale.y);
	const float sz = Math::abs(scale.z);
	if (!Math::is_equal_approx(sx, sy) || !Math::is_equal_approx(sy, sz)) {
		ZN_PRINT_WARNING_ONCE("Non-uniform terrain scale: edit radius will use max axis scale");
	}
	const float s = math::max(math::max(sx, sy), math::max(sz, 0.00001f));
	io_radius = io_radius / s;
}

} // namespace

VoxelToolLodTerrain::VoxelToolLodTerrain(VoxelLodTerrain *terrain) : _terrain(terrain) {
	ERR_FAIL_COND(terrain == nullptr);
	// At the moment, only LOD0 is supported.
	// Don't destroy the terrain while a voxel tool still references it
}

bool VoxelToolLodTerrain::is_area_editable(const Box3i &box) const {
	ERR_FAIL_COND_V(_terrain == nullptr, false);
	return _terrain->get_storage().is_area_loaded(box);
}

Ref<VoxelRaycastResult> VoxelToolLodTerrain::raycast(
		Vector3 pos,
		Vector3 dir,
		float max_distance,
		uint32_t collision_mask
) {
	return raycast_generic_world(
			_terrain->get_storage(),
			_terrain->get_mesher(),
			_terrain->get_global_transform(),
			pos,
			dir,
			max_distance,
			collision_mask,
			_raycast_binary_search_iterations,
			_raycast_normal_enabled
	);
}

void VoxelToolLodTerrain::do_box(Vector3i begin, Vector3i end) {
	ZN_PROFILE_SCOPE();
	ERR_FAIL_COND(_terrain == nullptr);

	ops::DoShapeChunked<ops::SdfAxisAlignedBox, ops::VoxelDataGridAccess> op;
	op.shape.center = to_vec3f(begin + end) * 0.5f;
	op.shape.half_size = to_vec3f(end - begin) * 0.5f;
	op.shape.sdf_scale = get_sdf_scale();
	op.box = op.shape.get_box().clipped(_terrain->get_voxel_bounds());
	op.mode = static_cast<ops::Mode>(get_mode());
	op.texture_params = _texture_params;
	op.blocky_value = _value;
	op.channel = get_channel();
	op.strength = get_sdf_strength();

	if (!is_area_editable(op.box)) {
		ZN_PRINT_WARNING("Area not editable");
		return;
	}

	VoxelData &data = _terrain->get_storage();

	data.pre_generate_box(op.box);

	VoxelDataGrid grid;
	data.get_blocks_grid(grid, op.box, 0);
	op.block_access.grid = &grid;

	{
		VoxelDataGrid::LockWrite wlock(grid);
		op();
	}

	_post_edit(op.box);
}

void VoxelToolLodTerrain::do_sphere(Vector3 center, float radius) {
	ZN_PROFILE_SCOPE();
	ERR_FAIL_COND(_terrain == nullptr);

	// Argument is in world space (consistent with raycast). Convert to terrain-local voxel
	// space so that scaling the terrain node (e.g. for finer voxel resolution) does not
	// silently break edits.
	world_to_local_sphere_lod(_terrain->get_global_transform(), center, radius);

	ops::DoSphere op;
	op.shape.center = to_vec3f(center);
	op.shape.radius = radius;
	op.shape.sdf_scale = get_sdf_scale();
	op.box = op.shape.get_box().clipped(_terrain->get_voxel_bounds());
	op.mode = static_cast<ops::Mode>(get_mode());
	op.texture_params = _texture_params;
	op.blocky_value = _value;
	op.channel = get_channel();
	op.strength = get_sdf_strength();

	const Box3i world_box = op.box;

	if (!is_area_editable(world_box)) {
		ZN_PRINT_WARNING("Area not editable");
		return;
	}

	VoxelData &data = _terrain->get_storage();

	data.pre_generate_box(world_box);
	data.get_blocks_grid(op.blocks, world_box, 0);

	// We can use floats by doing the operation in local space
	const Vector3i origin_in_voxels = op.blocks.get_origin_block_position_in_voxels();
	op.shape.center = to_vec3f(center - to_vec3(origin_in_voxels));
	op.box.position -= origin_in_voxels;
	op.blocks.use_relative_coordinates();

	op();

	_post_edit(world_box);
}

void VoxelToolLodTerrain::do_hemisphere(Vector3 center, float radius, Vector3 flat_direction, float smoothness) {
	ZN_PROFILE_SCOPE();
	ERR_FAIL_COND(_terrain == nullptr);

	// World -> local. flat_direction is rotated by the transform's basis (no translation
	// component). smoothness is a scalar in local space, but conceptually a soft-edge width;
	// scaling it with the radius keeps the look consistent.
	const Transform3D to_world = _terrain->get_global_transform();
	const Transform3D to_local = to_world.affine_inverse();
	const Vector3 center_local = to_local.xform(center);
	const Vector3 flat_dir_local = to_local.basis.xform(flat_direction).normalized();
	const Vector3 scale = to_world.basis.get_scale();
	const float scale_max = math::max(
			math::max(Math::abs(scale.x), Math::abs(scale.y)), math::max(Math::abs(scale.z), 0.00001f));
	const float radius_local = radius / scale_max;
	const float smoothness_local = smoothness / scale_max;

	ops::DoShapeChunked<ops::SdfHemisphere, ops::VoxelDataGridAccess> op;
	op.shape.center = to_vec3f(center_local);
	op.shape.radius = radius_local;
	op.shape.flat_direction = to_vec3f(flat_dir_local);
	op.shape.plane_d = flat_dir_local.dot(center_local);
	op.shape.smoothness = smoothness_local;
	op.shape.sdf_scale = get_sdf_scale();
	op.box = op.shape.get_box().clipped(_terrain->get_voxel_bounds());
	op.mode = static_cast<ops::Mode>(get_mode());
	op.texture_params = _texture_params;
	op.blocky_value = _value;
	op.channel = get_channel();
	op.strength = get_sdf_strength();

	if (!is_area_editable(op.box)) {
		ZN_PRINT_WARNING("Area not editable");
		return;
	}

	VoxelData &data = _terrain->get_storage();

	data.pre_generate_box(op.box);

	VoxelDataGrid grid;
	data.get_blocks_grid(grid, op.box, 0);
	op.block_access.grid = &grid;

	{
		VoxelDataGrid::LockWrite wlock(grid);
		op();
	}

	_post_edit(op.box);
}

void VoxelToolLodTerrain::do_flat_path(PackedVector3Array positions, float half_width, float thickness) {
	ZN_PROFILE_SCOPE();
	ERR_FAIL_COND(_terrain == nullptr);
	ERR_FAIL_COND(positions.size() < 2);

	// World -> local, same convention as do_path: positions and the radius-like extents are scaled
	// by the inverse terrain transform so node scaling (finer voxel resolution) does not break edits.
	const Transform3D to_world = _terrain->get_global_transform();
	const Transform3D to_local = to_world.affine_inverse();
	const Vector3 scale = to_world.basis.get_scale();
	const float scale_max = math::max(
			math::max(Math::abs(scale.x), Math::abs(scale.y)), math::max(Math::abs(scale.z), 0.00001f));
	const float half_width_local = half_width / scale_max;
	const float half_height_local = (thickness * 0.5f) / scale_max;

	const float sdf_scale = get_sdf_scale();
	const ops::Mode mode = static_cast<ops::Mode>(get_mode());

	// One chunked slab per segment. Each rasterizes independently (segments overlap at shared
	// points, which is fine for SdfUnion/Subtract). _post_edit is called once per segment over its
	// own box, matching how do_sphere/do_hemisphere report their edits.
	const int64_t point_count = positions.size();
	for (int64_t i = 1; i < point_count; ++i) {
		const Vector3 p0w = to_local.xform(positions[i - 1]);
		const Vector3 p1w = to_local.xform(positions[i]);

		ops::DoShapeChunked<ops::SdfFlatPathSegment, ops::VoxelDataGridAccess> op;
		op.shape.a = to_vec3f(p0w);
		op.shape.b = to_vec3f(p1w);
		op.shape.half_width = half_width_local;
		op.shape.half_height = half_height_local;
		// Slab vertical center: midpoint Y of the segment endpoints (the road follows the points'
		// height). A short segment is essentially flat in Y, so a single plane_y is enough.
		op.shape.plane_y = 0.5f * (static_cast<float>(p0w.y) + static_cast<float>(p1w.y));
		op.shape.sdf_scale = sdf_scale;
		op.box = op.shape.get_box().clipped(_terrain->get_voxel_bounds());
		op.mode = mode;
		op.texture_params = _texture_params;
		op.blocky_value = _value;
		op.channel = get_channel();
		op.strength = get_sdf_strength();

		if (!is_area_editable(op.box)) {
			ZN_PRINT_WARNING("Area not editable");
			continue;
		}

		VoxelData &data = _terrain->get_storage();
		data.pre_generate_box(op.box);

		VoxelDataGrid grid;
		data.get_blocks_grid(grid, op.box, 0);
		op.block_access.grid = &grid;

		{
			VoxelDataGrid::LockWrite wlock(grid);
			op();
		}

		_post_edit(op.box);
	}
}

void VoxelToolLodTerrain::do_path(Span<const Vector3> positions, Span<const float> radii) {
	ZN_ASSERT_RETURN(_terrain != nullptr);
	ZN_ASSERT_RETURN(positions.size() == radii.size());

	// World -> local. Copy into local vectors because the input Span is const. Each radius
	// is scaled by the max axis of the terrain transform basis (same as sphere ops).
	const Transform3D to_world = _terrain->get_global_transform();
	const Transform3D to_local = to_world.affine_inverse();
	const Vector3 scale = to_world.basis.get_scale();
	const float scale_max = math::max(
			math::max(Math::abs(scale.x), Math::abs(scale.y)), math::max(Math::abs(scale.z), 0.00001f));

	StdVector<Vector3> positions_local;
	StdVector<float> radii_local;
	positions_local.resize(positions.size());
	radii_local.resize(radii.size());
	for (size_t i = 0; i < positions.size(); ++i) {
		positions_local[i] = to_local.xform(positions[i]);
		radii_local[i] = radii[i] / scale_max;
	}

	do_path_chunked(
			_terrain->get_storage(),
			to_span(positions_local),
			to_span(radii_local),
			true);
}

PackedInt32Array VoxelToolLodTerrain::get_material_counts_in_sphere(Vector3 center, float radius) const {
	ZN_PROFILE_SCOPE();
	PackedInt32Array out;
	out.resize(16);
	int32_t *out_ptr = out.ptrw();
	for (int i = 0; i < 16; ++i) {
		out_ptr[i] = 0;
	}
	ERR_FAIL_COND_V(_terrain == nullptr, out);
	if (radius <= 0.0f) {
		return out;
	}

	// World-space sphere -> terrain-local voxel space, matching do_sphere(MODE_REMOVE).
	// The two functions must use the same transform, otherwise the histogram and the
	// actual carve disagree about which voxels are affected.
	world_to_local_sphere_lod(_terrain->get_global_transform(), center, radius);

	// Mirror what do_sphere(MODE_REMOVE) would do, voxel-by-voxel, and only count voxels that
	// actually change state from solid (old_sdf <= 0) to empty (new_sdf > 0). This matches the
	// visually disappearing volume rather than the sphere's geometric volume, which can be much
	// larger than the actually removed surface layer when the sphere sits inside the terrain.
	//
	// SdfOperation16bit samples the shape at integer voxel positions (to_vec3f(pos)), not at
	// voxel centers -- the same convention is used here for consistency with the edit.
	// SdfSubtract with strength=1 (Builder.cs default) is exactly max(old, -sphere_sdf).
	const float sdf_scale = get_sdf_scale();
	const Vector3i min_pos(
			Math::floor(center.x - radius),
			Math::floor(center.y - radius),
			Math::floor(center.z - radius)
	);
	const Vector3i max_pos(
			Math::ceil(center.x + radius),
			Math::ceil(center.y + radius),
			Math::ceil(center.z + radius)
	);

	const VoxelData &data = _terrain->get_storage();
	VoxelSingleValue sdf_default;
	sdf_default.f = constants::SDF_FAR_OUTSIDE;
	VoxelSingleValue idx_default;
	idx_default.i = 0;
	VoxelSingleValue weight_default;
	weight_default.i = 0;

	for (int z = min_pos.z; z <= max_pos.z; ++z) {
		for (int y = min_pos.y; y <= max_pos.y; ++y) {
			for (int x = min_pos.x; x <= max_pos.x; ++x) {
				const Vector3i pos(x, y, z);
				const float old_sdf = data.get_voxel(pos, VoxelBuffer::CHANNEL_SDF, sdf_default).f;
				if (old_sdf > 0.0f) {
					continue;
				}
				const Vector3 sample_pos(x, y, z);
				const float sphere_sdf = sdf_scale * ((sample_pos - center).length() - radius);
				const float new_sdf = math::max(old_sdf, -sphere_sdf);
				if (new_sdf <= 0.0f) {
					// Stays solid -- this voxel does not disappear from the mesh.
					continue;
				}
				const uint16_t packed_indices =
						static_cast<uint16_t>(data.get_voxel(pos, VoxelBuffer::CHANNEL_INDICES, idx_default).i);
				const uint16_t packed_weights =
						static_cast<uint16_t>(data.get_voxel(pos, VoxelBuffer::CHANNEL_WEIGHTS, weight_default).i);
				const FixedArray<uint8_t, 4> indices = mixel4::decode_indices_from_packed_u16(packed_indices);
				const FixedArray<uint8_t, 4> weights = mixel4::decode_weights_from_packed_u16(packed_weights);
				unsigned int best_slot = 0;
				uint8_t best_weight = weights[0];
				for (unsigned int i = 1; i < 4; ++i) {
					if (weights[i] > best_weight) {
						best_weight = weights[i];
						best_slot = i;
					}
				}
				const uint8_t dominant = indices[best_slot];
				if (dominant < 16) {
					out_ptr[dominant] += 1;
				}
			}
		}
	}
	return out;
}

int VoxelToolLodTerrain::do_sphere_add_with_material(Vector3 center, float radius, int material_index) {
	ZN_PROFILE_SCOPE();
	ERR_FAIL_COND_V(_terrain == nullptr, 0);
	ERR_FAIL_COND_V(material_index < 0 || material_index >= 16, 0);
	if (radius <= 0.0f) {
		return 0;
	}

	// World-space sphere -> terrain-local voxel space, matching do_sphere.
	world_to_local_sphere_lod(_terrain->get_global_transform(), center, radius);

	// Mirror what do_sphere(MODE_ADD) does, voxel-by-voxel, and only repaint Mixel4 material on
	// voxels that transition from empty (old_sdf > 0) to solid (new_sdf <= 0). Existing solid
	// voxels keep their original material -- this is the whole point of this method versus a
	// blind do_sphere + TexturePaint, which would repaint underlying terrain that the sphere
	// happens to overlap.
	//
	// MODE_ADD with SdfUnion at strength=1 is exactly min(old, sphere_sdf), see funcs.h.
	const float sdf_scale = get_sdf_scale();
	const Vector3i min_pos(
			Math::floor(center.x - radius),
			Math::floor(center.y - radius),
			Math::floor(center.z - radius)
	);
	const Vector3i max_pos(
			Math::ceil(center.x + radius),
			Math::ceil(center.y + radius),
			Math::ceil(center.z + radius)
	);

	const Box3i box = Box3i::from_min_max(min_pos, max_pos + Vector3i(1, 1, 1))
							  .clipped(_terrain->get_voxel_bounds());
	if (!is_area_editable(box)) {
		ZN_PRINT_WARNING("Area not editable");
		return 0;
	}

	VoxelData &data = _terrain->get_storage();
	data.pre_generate_box(box);

	VoxelSingleValue sdf_default;
	sdf_default.f = constants::SDF_FAR_OUTSIDE;

	// Encode the chosen material as Mixel4: dominant in slot 0 with full weight, the other 3 slots
	// get distinct filler indices with zero weight (Mixel4 requires the 4 indices to be different).
	uint8_t fill_slots[3];
	int next = 0;
	for (int candidate = 0; candidate < 16 && next < 3; ++candidate) {
		if (candidate == material_index) {
			continue;
		}
		fill_slots[next++] = static_cast<uint8_t>(candidate);
	}
	const uint16_t packed_indices = mixel4::encode_indices_to_packed_u16(
			static_cast<uint8_t>(material_index), fill_slots[0], fill_slots[1], fill_slots[2]
	);
	const uint16_t packed_weights = mixel4::encode_weights_to_packed_u16_lossy(255, 0, 0, 0);

	int filled_count = 0;

	for (int z = min_pos.z; z <= max_pos.z; ++z) {
		for (int y = min_pos.y; y <= max_pos.y; ++y) {
			for (int x = min_pos.x; x <= max_pos.x; ++x) {
				const Vector3i pos(x, y, z);
				const float old_sdf = data.get_voxel(pos, VoxelBuffer::CHANNEL_SDF, sdf_default).f;
				const Vector3 sample_pos(x, y, z);
				const float sphere_sdf = sdf_scale * ((sample_pos - center).length() - radius);
				const float new_sdf = math::min(old_sdf, sphere_sdf);
				if (new_sdf == old_sdf) {
					continue;
				}
				data.try_set_voxel_f(new_sdf, pos, VoxelBuffer::CHANNEL_SDF);
				const bool was_empty = old_sdf > 0.0f;
				const bool is_solid_now = new_sdf <= 0.0f;
				if (was_empty && is_solid_now) {
					data.try_set_voxel(packed_indices, pos, VoxelBuffer::CHANNEL_INDICES);
					data.try_set_voxel(packed_weights, pos, VoxelBuffer::CHANNEL_WEIGHTS);
					++filled_count;
				}
			}
		}
	}

	// Padding matches do_sphere (see get_sdf_sphere_box in funcs.h): _post_edit needs to also
	// cover neighbor blocks so their meshes regenerate at the seam. Without padding, seams
	// between freshly-filled voxels and untouched neighbor blocks may render with stale
	// indices/weights from the old mesh.
	_post_edit(box.padded(2));
	return filled_count;
}

void VoxelToolLodTerrain::paint_voxels_material(PackedVector3Array positions, int material_index) {
	ZN_PROFILE_SCOPE();
	ERR_FAIL_COND(_terrain == nullptr);
	ERR_FAIL_COND(material_index < 0 || material_index >= 16);
	if (positions.size() == 0) {
		return;
	}

	// Bounding box over all positions, so a single pre_generate / _post_edit covers them.
	Vector3i min_pos(
			Math::floor(positions[0].x), Math::floor(positions[0].y), Math::floor(positions[0].z)
	);
	Vector3i max_pos = min_pos;
	for (int i = 0; i < positions.size(); ++i) {
		const Vector3i p(Math::floor(positions[i].x), Math::floor(positions[i].y), Math::floor(positions[i].z));
		min_pos = min_pos.min(p);
		max_pos = max_pos.max(p);
	}

	const Box3i box = Box3i::from_min_max(min_pos, max_pos + Vector3i(1, 1, 1)).clipped(_terrain->get_voxel_bounds());
	if (!is_area_editable(box)) {
		ZN_PRINT_WARNING("Area not editable");
		return;
	}

	VoxelData &data = _terrain->get_storage();
	data.pre_generate_box(box);

	VoxelSingleValue indices_default;
	indices_default.i = 0;
	VoxelSingleValue weights_default;
	weights_default.i = 0;

	for (int i = 0; i < positions.size(); ++i) {
		const Vector3i pos(
				Math::floor(positions[i].x), Math::floor(positions[i].y), Math::floor(positions[i].z)
		);
		uint16_t encoded_indices =
				static_cast<uint16_t>(data.get_voxel(pos, VoxelBuffer::CHANNEL_INDICES, indices_default).i);
		uint16_t encoded_weights =
				static_cast<uint16_t>(data.get_voxel(pos, VoxelBuffer::CHANNEL_WEIGHTS, weights_default).i);

		// Reuse the engine's Mixel4 blending so the painted material drops into the lowest-weight
		// slot and weights stay normalized -- same logic the sphere/box texture-paint path uses.
		mixel4::blend_texture_packed_u16(material_index, 1.0f, encoded_indices, encoded_weights);

		data.try_set_voxel(encoded_indices, pos, VoxelBuffer::CHANNEL_INDICES);
		data.try_set_voxel(encoded_weights, pos, VoxelBuffer::CHANNEL_WEIGHTS);
	}

	_post_edit(box.padded(2));
}

// Shared pipeline for the fluid-channel sphere edits. Identical to do_sphere() but with the
// channel hardcoded to CHANNEL_DATA5 and the mode passed in, so the public methods do not
// depend on VoxelTool::set_channel / set_mode state and cannot accidentally write to the
// terrain SDF channel.
void VoxelToolLodTerrain::run_fluid_sphere_op(Vector3 center, float radius, ops::Mode mode) {
	ZN_PROFILE_SCOPE();
	ERR_FAIL_COND(_terrain == nullptr);

	world_to_local_sphere_lod(_terrain->get_global_transform(), center, radius);

	ops::DoSphere op;
	op.shape.center = to_vec3f(center);
	op.shape.radius = radius;
	op.shape.sdf_scale = get_sdf_scale();
	op.box = op.shape.get_box().clipped(_terrain->get_voxel_bounds());
	op.mode = mode;
	op.texture_params = _texture_params;
	op.blocky_value = 0;
	// Fluid channel: independent from the terrain SDF. The hybrid mesher reads it via a
	// second Transvoxel pass (see VoxelMesherHybrid::build_fluid_surfaces).
	op.channel = VoxelBuffer::CHANNEL_DATA5;
	op.strength = get_sdf_strength();

	const Box3i world_box = op.box;

	if (!is_area_editable(world_box)) {
		ZN_PRINT_WARNING("Area not editable");
		return;
	}

	VoxelData &data = _terrain->get_storage();
	data.pre_generate_box(world_box);
	data.get_blocks_grid(op.blocks, world_box, 0);

	// Operate in block-relative coordinates so the inner loop can stay in floats.
	const Vector3i origin_in_voxels = op.blocks.get_origin_block_position_in_voxels();
	op.shape.center = to_vec3f(center - to_vec3(origin_in_voxels));
	op.box.position -= origin_in_voxels;
	op.blocks.use_relative_coordinates();

	op();

	_post_edit(world_box);
}

void VoxelToolLodTerrain::do_sphere_fluid_add(Vector3 center, float radius) {
	run_fluid_sphere_op(center, radius, ops::MODE_ADD);
}

void VoxelToolLodTerrain::do_sphere_fluid_remove(Vector3 center, float radius) {
	run_fluid_sphere_op(center, radius, ops::MODE_REMOVE);
}

template <typename Op_T>
class VoxelToolAsyncEdit : public IThreadedTask {
public:
	VoxelToolAsyncEdit(Op_T op, std::shared_ptr<VoxelData> data) : _op(op), _data(data) {
		_tracker = make_shared_instance<AsyncDependencyTracker>(1);
	}

	const char *get_debug_name() const override {
		return "VoxelToolAsyncEdit";
	}

	void run(ThreadedTaskContext &ctx) override {
		ZN_PROFILE_SCOPE();
		ZN_ASSERT(_data != nullptr);
		// TODO May want to fail if not all blocks were found
		// TODO Need to apply modifiers
		_data->get_blocks_grid(_op.blocks, _op.box, 0);
		_op();
		_tracker->post_complete();
	}

	std::shared_ptr<AsyncDependencyTracker> get_tracker() {
		return _tracker;
	}

private:
	Op_T _op;
	// We reference this just to keep map pointers alive
	std::shared_ptr<VoxelData> _data;
	std::shared_ptr<AsyncDependencyTracker> _tracker;
};

void VoxelToolLodTerrain::do_sphere_async(Vector3 center, float radius) {
	ERR_FAIL_COND(_terrain == nullptr);

	// Same world->local convention as the synchronous do_sphere.
	world_to_local_sphere_lod(_terrain->get_global_transform(), center, radius);

	ops::DoSphere op;
	op.shape.center = to_vec3f(center);
	op.shape.radius = radius;
	op.shape.sdf_scale = get_sdf_scale();
	op.box = op.shape.get_box().clipped(_terrain->get_voxel_bounds());
	op.mode = ops::Mode(get_mode());
	op.texture_params = _texture_params;
	op.blocky_value = _value;
	op.channel = get_channel();
	op.strength = get_sdf_strength();

	if (!is_area_editable(op.box)) {
		ZN_PRINT_WARNING("Area not editable");
		return;
	}

	std::shared_ptr<VoxelData> data = _terrain->get_storage_shared();

	VoxelToolAsyncEdit<ops::DoSphere> *task = ZN_NEW(VoxelToolAsyncEdit<ops::DoSphere>(op, data));
	_terrain->push_async_edit(task, op.box, task->get_tracker());
}

void VoxelToolLodTerrain::copy(
		const Vector3i pos,
		VoxelBuffer &dst,
		const uint8_t p_channels_mask,
		const bool with_metadata
) const {
	ZN_PROFILE_SCOPE();
	ERR_FAIL_COND(_terrain == nullptr);
	const unsigned int channels_mask = (p_channels_mask == 0 ? (1 << _channel) : p_channels_mask);
	_terrain->get_storage().copy(pos, dst, channels_mask, with_metadata);
}

void VoxelToolLodTerrain::paste(Vector3i pos, const VoxelBuffer &src, uint8_t channels_mask) {
	ERR_FAIL_COND(_terrain == nullptr);
	if (channels_mask == 0) {
		channels_mask = (1 << _channel);
	}
	const Box3i box(pos, src.get_size());
	if (!is_area_editable(box)) {
		ZN_PRINT_WARNING("Area not editable");
		return;
	}

	VoxelData &data = _terrain->get_storage();

	data.pre_generate_box(box);
	data.paste(pos, src, channels_mask, false, true);

	_post_edit(box);
}

void VoxelToolLodTerrain::set_voxel_metadata(const Vector3i pos, const Variant &meta) {
	ZN_ASSERT_RETURN(_terrain != nullptr);
	VoxelData &data = _terrain->get_storage();
	data.set_voxel_metadata(pos, meta);
	_terrain->post_edit_area(Box3i(pos, Vector3i(1, 1, 1)), false);
}

Variant VoxelToolLodTerrain::get_voxel_metadata(const Vector3i pos) const {
	ZN_ASSERT_RETURN_V(_terrain != nullptr, Variant());
	VoxelData &data = _terrain->get_storage();
	return data.get_voxel_metadata(pos);
}

float VoxelToolLodTerrain::get_voxel_f_interpolated(Vector3 position) const {
	ZN_PROFILE_SCOPE();
	ERR_FAIL_COND_V(_terrain == nullptr, 0);
	const int channel = get_channel();
	VoxelData &data = _terrain->get_storage();
	// TODO Optimization: is it worth making a fast-path for this?
	return get_sdf_interpolated(
			[&data, channel](Vector3i ipos) {
				VoxelSingleValue defval;
				defval.f = constants::SDF_FAR_OUTSIDE;
				VoxelSingleValue value = data.get_voxel(ipos, channel, defval);
				return value.f;
			},
			position
	);
}

uint64_t VoxelToolLodTerrain::_get_voxel(Vector3i pos) const {
	ERR_FAIL_COND_V(_terrain == nullptr, 0);
	VoxelSingleValue defval;
	defval.i = 0;
	return _terrain->get_storage().get_voxel(pos, _channel, defval).i;
}

float VoxelToolLodTerrain::_get_voxel_f(Vector3i pos) const {
	ERR_FAIL_COND_V(_terrain == nullptr, 0);
	VoxelSingleValue defval;
	defval.f = constants::SDF_FAR_OUTSIDE;
	return _terrain->get_storage().get_voxel(pos, _channel, defval).f;
}

void VoxelToolLodTerrain::_set_voxel(Vector3i pos, uint64_t v) {
	ERR_FAIL_COND(_terrain == nullptr);
	_terrain->get_storage().try_set_voxel(v, pos, _channel);
	// No post_update, the parent class does it, it's a generic slow implementation.
}

void VoxelToolLodTerrain::_set_voxel_f(Vector3i pos, float v) {
	ERR_FAIL_COND(_terrain == nullptr);
	// TODO Format should be accessible from terrain
	_terrain->get_storage().try_set_voxel_f(v, pos, _channel);
	// No post_update, the parent class does it, it's a generic slow implementation.
}

void VoxelToolLodTerrain::_post_edit(const Box3i &box) {
	ERR_FAIL_COND(_terrain == nullptr);
	_terrain->post_edit_area(box, true);
}

int VoxelToolLodTerrain::get_raycast_binary_search_iterations() const {
	return _raycast_binary_search_iterations;
}

void VoxelToolLodTerrain::set_raycast_binary_search_iterations(int iterations) {
	_raycast_binary_search_iterations = math::clamp(iterations, 0, 16);
}

#if defined(ZN_GODOT)
Array VoxelToolLodTerrain::separate_floating_chunks(AABB world_box, Node *parent_node) {
#elif defined(ZN_GODOT_EXTENSION)
Array VoxelToolLodTerrain::separate_floating_chunks(AABB world_box, Object *parent_node_o) {
	Node *parent_node = Object::cast_to<Node>(parent_node_o);
#endif
	ERR_FAIL_COND_V(_terrain == nullptr, Array());
	ERR_FAIL_COND_V(!math::is_valid_size(world_box.size), Array());
	Ref<VoxelMesher> mesher = _terrain->get_mesher();
	Array materials;
	materials.append(_terrain->get_material());
	const Box3i int_world_box(math::floor_to_int(world_box.position), math::ceil_to_int(world_box.size));
	return zylann::voxel::separate_floating_chunks(
			*this, int_world_box, parent_node, _terrain->get_global_transform(), mesher, materials
	);
}

#ifdef VOXEL_ENABLE_MESH_SDF

// Combines a precalculated SDF with the terrain at a specific position, rotation and scale.
//
// `transform` is where the buffer should be applied on the terrain.
//
// `isolevel` alters the shape of the SDF: positive "puffs" it, negative "erodes" it. This is a applied after
// `sdf_scale`.
//
// `sdf_scale` scales SDF values (it doesn't make the shape bigger or smaller). Usually defaults to 1 but may be lower
// if artifacts show up due to scaling used in terrain SDF.
//
void VoxelToolLodTerrain::stamp_sdf(
		Ref<VoxelMeshSDF> mesh_sdf,
		Transform3D transform,
		float isolevel,
		float sdf_scale
) {
	ZN_PRINT_WARNING_ONCE("This method is deprecated. Use `do_mesh` instead.");
	ZN_PROFILE_SCOPE();

	ERR_FAIL_COND(_terrain == nullptr);
	ERR_FAIL_COND(mesh_sdf.is_null());
	ERR_FAIL_COND(!mesh_sdf->is_baked());
	Ref<godot::VoxelBuffer> buffer_ref = mesh_sdf->get_voxel_buffer();
	ERR_FAIL_COND(buffer_ref.is_null());
	const VoxelBuffer &buffer = buffer_ref->get_buffer();
	const VoxelBuffer::ChannelId channel = VoxelBuffer::CHANNEL_SDF;
	ERR_FAIL_COND(buffer.get_channel_compression(channel) == VoxelBuffer::COMPRESSION_UNIFORM);
	ERR_FAIL_COND(buffer.get_channel_depth(channel) != VoxelBuffer::DEPTH_32_BIT);

	const Transform3D &box_to_world = transform;
	const AABB local_aabb = mesh_sdf->get_aabb();

	// Note, transform is local to the terrain
	const AABB aabb = box_to_world.xform(local_aabb);
	const Box3i voxel_box = Box3i::from_min_max(aabb.position.floor(), (aabb.position + aabb.size).ceil());

	// TODO Sometimes it will fail near unloaded blocks, even though the transformed box does not intersect them.
	// This could be avoided with a box/transformed-box intersection algorithm. Might investigate if the use case
	// occurs. It won't happen with full load mode. This also affects other shapes.
	if (!is_area_editable(voxel_box)) {
		ZN_PRINT_WARNING("Area not editable");
		return;
	}

	VoxelData &data = _terrain->get_storage();

	data.pre_generate_box(voxel_box);

	// TODO Maybe more efficient to "rasterize" the box? We're going to iterate voxels the box doesn't intersect.
	// TODO Maybe we should scale SDF values based on the scale of the transform too

	const Transform3D buffer_to_box =
			Transform3D(Basis().scaled(Vector3(local_aabb.size / buffer.get_size())), local_aabb.position);
	const Transform3D buffer_to_world = box_to_world * buffer_to_box;

	// TODO Support other depths, format should be accessible from the volume
	ops::SdfOperation16bit<ops::SdfUnion, ops::SdfBufferShape> op;
	op.op.strength = get_sdf_strength();
	op.shape.world_to_buffer = buffer_to_world.affine_inverse();
	op.shape.buffer_size = buffer.get_size();
	op.shape.isolevel = isolevel;
	op.shape.sdf_scale = sdf_scale;
	// Note, the passed buffer must not be shared with another thread.
	// buffer.decompress_channel(channel);
	ZN_ASSERT_RETURN(buffer.get_channel_data_read_only(channel, op.shape.buffer));

	VoxelDataGrid grid;
	data.get_blocks_grid(grid, voxel_box, 0);
	grid.write_box(voxel_box, VoxelBuffer::CHANNEL_SDF, op);

	_post_edit(voxel_box);
}

void VoxelToolLodTerrain::do_mesh(const VoxelMeshSDF &mesh_sdf, const Transform3D &transform, const float isolevel) {
	ZN_ASSERT_RETURN(_terrain != nullptr);
	// transform is the brush pose in world space. Pre-multiply with terrain-to-local so the
	// operation runs in voxel coordinates, matching do_sphere and friends.
	const Transform3D to_local = _terrain->get_global_transform().affine_inverse();
	const Transform3D local_transform = to_local * transform;
	do_mesh_chunked(mesh_sdf, _terrain->get_storage(), local_transform, isolevel, true);
}

#endif

// Runs the given graph in a bounding box in the terrain.
// The graph must have an SDF output and can also have an SDF input to read source voxels.
// The transform contains the position of the edit, its orientation and scale.
// Graph base size is the original size of the brush, as designed in the graph. It will be scaled using the transform.
void VoxelToolLodTerrain::do_graph(Ref<VoxelGeneratorGraph> graph, Transform3D transform, Vector3 graph_base_size) {
	ZN_PROFILE_SCOPE();
	ZN_DSTACK();
	ERR_FAIL_COND(_terrain == nullptr);

	// transform is the brush pose in world space. Pre-multiply with terrain-to-local so the
	// operation runs in voxel coordinates, matching do_sphere and friends.
	const Transform3D to_local = _terrain->get_global_transform().affine_inverse();
	transform = to_local * transform;

	const Vector3 area_size = math::abs(transform.basis.xform(graph_base_size));

	const Box3i box = Box3i::from_min_max(
							  math::floor_to_int(transform.origin - 0.5 * area_size),
							  math::ceil_to_int(transform.origin + 0.5 * area_size)
	)
							  .padded(2)
							  .clipped(_terrain->get_voxel_bounds());

	if (!is_area_editable(box)) {
		ZN_PRINT_WARNING("Area not editable");
		return;
	}

	VoxelData &data = _terrain->get_storage();

	data.pre_generate_box(box);

	const unsigned int channel_index = VoxelBuffer::CHANNEL_SDF;

	VoxelBuffer buffer(VoxelBuffer::ALLOCATOR_POOL);
	buffer.create(box.size);
	data.copy(box.position, buffer, 1 << channel_index, false);

	buffer.decompress_channel(channel_index);

	// Convert input SDF
	static thread_local StdVector<float> tls_in_sdf_full;
	tls_in_sdf_full.resize(Vector3iUtil::get_volume_u64(buffer.get_size()));
	Span<float> in_sdf_full = to_span(tls_in_sdf_full);
	get_unscaled_sdf(buffer, in_sdf_full);

	static thread_local StdVector<float> tls_in_x;
	static thread_local StdVector<float> tls_in_y;
	static thread_local StdVector<float> tls_in_z;
	const unsigned int deck_area = box.size.x * box.size.y;
	tls_in_x.resize(deck_area);
	tls_in_y.resize(deck_area);
	tls_in_z.resize(deck_area);
	Span<float> in_x = to_span(tls_in_x);
	Span<float> in_y = to_span(tls_in_y);
	Span<float> in_z = to_span(tls_in_z);

	const Transform3D inv_transform = transform.affine_inverse();

	const int output_sdf_buffer_index = graph->get_sdf_output_port_address();
	ZN_ASSERT_RETURN_MSG(output_sdf_buffer_index != -1, "The graph has no SDF output, cannot use it as a brush");

	// The graph works at a fixed dimension, so if we scale the operation with the Transform3D then we have to also
	// scale the distance field the graph is working at
	const float graph_scale = transform.basis.get_scale().length();
	const float inv_graph_scale = 1.f / graph_scale;

	for (float &sd : in_sdf_full) {
		sd *= inv_graph_scale;
	}

	const float op_strength = get_sdf_strength();

	{
		ZN_PROFILE_SCOPE_NAMED("Slices");
		// For each deck of the box (doing this to reduce memory usage since the graph will allocate temporary buffers
		// for each operation, which can be a lot depending on the complexity of the graph)
		Vector3i pos;
		const Vector3i endpos = box.position + box.size;
		for (pos.z = box.position.z; pos.z < endpos.z; ++pos.z) {
			// Set positions
			for (unsigned int i = 0; i < deck_area; ++i) {
				in_z[i] = pos.z;
			}
			{
				unsigned int i = 0;
				for (pos.x = box.position.x; pos.x < endpos.x; ++pos.x) {
					for (pos.y = box.position.y; pos.y < endpos.y; ++pos.y) {
						in_x[i] = pos.x;
						in_y[i] = pos.y;
						++i;
					}
				}
			}

			// Transform positions to be local to the graph
			for (unsigned int i = 0; i < deck_area; ++i) {
				Vector3 graph_local_pos(in_x[i], in_y[i], in_z[i]);
				graph_local_pos = inv_transform.xform(graph_local_pos);
				in_x[i] = graph_local_pos.x;
				in_y[i] = graph_local_pos.y;
				in_z[i] = graph_local_pos.z;
			}

			// Get SDF input
			Span<float> in_sdf = in_sdf_full.sub(deck_area * (pos.z - box.position.z), deck_area);

			// Run graph
			graph->generate_series(in_x, in_y, in_z, in_sdf);

			// Read result
			const pg::Runtime::State &state = VoxelGeneratorGraph::get_last_state_from_current_thread();
			const pg::Runtime::Buffer &graph_buffer = state.get_buffer(output_sdf_buffer_index);

			// Apply strength and graph scale. Input serves as output too, shouldn't overlap
			for (unsigned int i = 0; i < in_sdf.size(); ++i) {
				in_sdf[i] = Math::lerp(in_sdf[i], graph_buffer.data[i], op_strength) * graph_scale;
			}
		}
	}

	scale_and_store_sdf(buffer, in_sdf_full);

	data.paste(box.position, buffer, 1 << channel_index, false, false);

	_post_edit(box);
}

void VoxelToolLodTerrain::run_blocky_random_tick(
		const AABB voxel_area,
		const int voxel_count,
		const Callable &callback,
		const int block_batch_count,
		const uint32_t tags_mask
) {
	ZN_PROFILE_SCOPE();

	ZN_ASSERT_RETURN(_terrain != nullptr);

	Ref<VoxelMesherBlocky> mesher = _terrain->get_mesher();
	ZN_ASSERT_RETURN_MSG(
			mesher.is_valid(),
			format("This function requires a volume using {} with a valid library", ZN_CLASS_NAME_C(VoxelMesherBlocky))
	);
	Ref<VoxelBlockyLibraryBase> library = mesher->get_library();
	ZN_ASSERT_RETURN_MSG(library.is_valid(), format("{} has no library assigned", ZN_CLASS_NAME_C(VoxelMesherBlocky)));

	ZN_ASSERT_RETURN(callback.is_valid());
	ZN_ASSERT_RETURN(block_batch_count > 0);
	ZN_ASSERT_RETURN(voxel_count >= 0);
	ZN_ASSERT_RETURN(math::is_valid_size(voxel_area.size));

	if (voxel_count == 0) {
		return;
	}

	VoxelData &data = _terrain->get_storage();

	zylann::voxel::run_blocky_random_tick(
			data, voxel_area, **library, _random, voxel_count, block_batch_count, tags_mask, callback
	);
}

VoxelFormat VoxelToolLodTerrain::get_format() const {
	ZN_ASSERT(_terrain != nullptr);
	return _terrain->get_storage().get_format();
}

void VoxelToolLodTerrain::_bind_methods() {
	using Self = VoxelToolLodTerrain;

	ClassDB::bind_method(
			D_METHOD("set_raycast_binary_search_iterations", "iterations"), &Self::set_raycast_binary_search_iterations
	);
	ClassDB::bind_method(D_METHOD("get_raycast_binary_search_iterations"), &Self::get_raycast_binary_search_iterations);
	ClassDB::bind_method(D_METHOD("get_voxel_f_interpolated", "position"), &Self::get_voxel_f_interpolated);
	ClassDB::bind_method(D_METHOD("separate_floating_chunks", "box", "parent_node"), &Self::separate_floating_chunks);
	ClassDB::bind_method(D_METHOD("do_sphere_async", "center", "radius"), &Self::do_sphere_async);
	ClassDB::bind_method(
			D_METHOD("get_material_counts_in_sphere", "center", "radius"), &Self::get_material_counts_in_sphere
	);
	ClassDB::bind_method(
			D_METHOD("do_sphere_add_with_material", "center", "radius", "material_index"),
			&Self::do_sphere_add_with_material
	);
	ClassDB::bind_method(
			D_METHOD("paint_voxels_material", "positions", "material_index"),
			&Self::paint_voxels_material
	);
	ClassDB::bind_method(
			D_METHOD("do_sphere_fluid_add", "center", "radius"), &Self::do_sphere_fluid_add
	);
	ClassDB::bind_method(
			D_METHOD("do_sphere_fluid_remove", "center", "radius"), &Self::do_sphere_fluid_remove
	);
#ifdef VOXEL_ENABLE_MESH_SDF
	ClassDB::bind_method(D_METHOD("stamp_sdf", "mesh_sdf", "transform", "isolevel", "sdf_scale"), &Self::stamp_sdf);
#endif
	ClassDB::bind_method(D_METHOD("do_graph", "graph", "transform", "area_size"), &Self::do_graph);
	ClassDB::bind_method(
			D_METHOD("do_hemisphere", "center", "radius", "flat_direction", "smoothness"),
			&Self::do_hemisphere,
			DEFVAL(0.0)
	);
	ClassDB::bind_method(
			D_METHOD("do_flat_path", "positions", "half_width", "thickness"), &Self::do_flat_path
	);
	ClassDB::bind_method(
			D_METHOD("run_blocky_random_tick", "area", "voxel_count", "callback", "batch_count", "tags_mask"),
			&Self::run_blocky_random_tick,
			DEFVAL(16),
			DEFVAL(0xffffffff)
	);
}

} // namespace zylann::voxel
