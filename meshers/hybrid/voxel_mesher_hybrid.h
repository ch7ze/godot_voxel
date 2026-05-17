#ifndef VOXEL_MESHER_HYBRID_H
#define VOXEL_MESHER_HYBRID_H

#include "../../util/godot/classes/mesh.h"
#include "../voxel_mesher.h"
#include "../blocky/voxel_blocky_library_base.h"
#include "../transvoxel/voxel_mesher_transvoxel.h"

namespace zylann::voxel {

// Hybrid mesher: produces a single chunk mesh containing both
//   - Surface 0: a smooth iso-surface built from the SDF channel (Transvoxel),
//   - Surface 1: sharp-edged 3D models baked from the TYPE channel (like VoxelMesherBlocky).
//
// The goal is to keep smooth terrain and placed blocky models in one terrain, one mesher.
// Models are NOT mixed into the SDF (which would round their edges): their original triangles
// are copied straight into the mesh output.
//
// The smooth part is delegated to an internally-held VoxelMesherTransvoxel instance, so the
// Transvoxel code is reused as-is rather than reimplemented.
class VoxelMesherHybrid : public VoxelMesher {
	GDCLASS(VoxelMesherHybrid, VoxelMesher)

public:
	VoxelMesherHybrid();
	~VoxelMesherHybrid();

	void build(VoxelMesher::Output &output, const VoxelMesher::Input &input) override;

	int get_used_channels_mask() const override;

	bool supports_lod() const override {
		return true;
	}

	bool is_generating_collision_surface() const override;

	// Library of models used for the blocky (TYPE channel) part of the mesh.
	void set_library(Ref<VoxelBlockyLibraryBase> library);
	Ref<VoxelBlockyLibraryBase> get_library() const;

private:
	// Builds the blocky model part into `output` (Surface 1). No-op if no library is set.
	void build_model_surface(VoxelMesher::Output &output, const VoxelMesher::Input &input);

	static void _bind_methods();

	// Reused as-is for the smooth SDF surface (Surface 0).
	Ref<VoxelMesherTransvoxel> _transvoxel_mesher;

	Ref<VoxelBlockyLibraryBase> _library;
};

} // namespace zylann::voxel

#endif // VOXEL_MESHER_HYBRID_H
