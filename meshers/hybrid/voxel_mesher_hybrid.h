#ifndef VOXEL_MESHER_HYBRID_H
#define VOXEL_MESHER_HYBRID_H

#include "../../util/godot/classes/material.h"
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

	// Provides the material for each surface: index 0 = smooth terrain, index 1 = models.
	// The terrain calls this per chunk surface to assign materials.
	Ref<Material> get_material_by_index(unsigned int index) const override;
	unsigned int get_material_index_count() const override;

	// Library of models used for the blocky (TYPE channel) part of the mesh.
	void set_library(Ref<VoxelBlockyLibraryBase> library);
	Ref<VoxelBlockyLibraryBase> get_library() const;

	// Material for the smooth terrain surface (surface 0).
	void set_terrain_material(Ref<Material> material);
	Ref<Material> get_terrain_material() const;

	// Material for the blocky model surface (surface 1).
	void set_model_material(Ref<Material> material);
	Ref<Material> get_model_material() const;

	// When enabled, the smooth terrain is flattened to the bottom edge of any voxel that
	// holds a model, so models sit flush on the ground instead of floating or sinking.
	void set_flatten_terrain_under_models(bool enabled);
	bool get_flatten_terrain_under_models() const;

private:
	// Builds the blocky model part into `output` (Surface 1). No-op if no library is set.
	void build_model_surface(VoxelMesher::Output &output, const VoxelMesher::Input &input);

	// Returns true if any model voxel exists, and writes a flattened SDF copy into `dst`.
	// `dst` is the SDF the smooth surface should be meshed from instead of the original.
	bool build_flattened_sdf(VoxelBuffer &dst, const VoxelMesher::Input &input) const;

	static void _bind_methods();

	// Reused as-is for the smooth SDF surface (Surface 0).
	Ref<VoxelMesherTransvoxel> _transvoxel_mesher;

	Ref<VoxelBlockyLibraryBase> _library;

	Ref<Material> _terrain_material;
	Ref<Material> _model_material;

	bool _flatten_terrain_under_models = true;
};

} // namespace zylann::voxel

#endif // VOXEL_MESHER_HYBRID_H
