#include "voxel_mesher_hybrid.h"
#include "../../storage/voxel_buffer.h"
#include "../../util/profiling.h"

namespace zylann::voxel {

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
	// TODO Step 2: read CHANNEL_TYPE, fetch models from `_library`, bake their triangles into a
	// second surface. Left empty for now so the SDF path can be compiled and tested first.
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
