#ifndef VOXEL_MESHER_HYBRID_H
#define VOXEL_MESHER_HYBRID_H

#include "../../util/godot/classes/material.h"
#include "../../util/godot/classes/mesh.h"
#include "../../util/godot/core/color.h"
#include "../../util/godot/core/packed_arrays.h"
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

	// Provides the material for each surface index. Index 0 is the smooth terrain;
	// indices >= 1 map to the model library's materials (index 1 -> library material 0, etc.).
	// The terrain calls this per chunk surface to assign materials.
	Ref<Material> get_material_by_index(unsigned int index) const override;
	unsigned int get_material_index_count() const override;

	// Material index that marks the smooth fluid iso-surface (CHANNEL_DATA5 pass). The
	// variable-LOD terrain reads this to split the chunk mesh into a terrain ArrayMesh and a
	// separate water ArrayMesh, each with its own ShaderMaterial. Keeping the value as a
	// virtual override avoids leaking the hybrid mesher's surface-id convention into
	// VoxelLodTerrain.
	static constexpr int WATER_MATERIAL_INDEX = 1;
	int get_water_material_index() const override {
		return WATER_MATERIAL_INDEX;
	}

	// Default material applied by the terrain as a single material_override over the whole
	// chunk mesh. It is a ShaderMaterial whose shader both handles Transvoxel LOD transitions
	// (so transition triangles are correctly culled instead of showing as artifacts) and picks
	// a per-vertex colour from the CUSTOM1 material id. Without this, the terrain would get no
	// LOD shader and the transition meshes would stay visible.
	//
	// The default shader is only meaningful when the inner Transvoxel mesher's texturing mode
	// is TEXTURES_NONE: it reads the material id from CUSTOM1. With texturing enabled the
	// material id lives in CUSTOM2 instead, so a project-specific user shader (assigned via
	// VoxelLodTerrain.material) is required to render the surfaces correctly.
	Ref<ShaderMaterial> get_default_lod_material() const override;

	// Library of models used for the blocky (TYPE channel) part of the mesh.
	void set_library(Ref<VoxelBlockyLibraryBase> library);
	Ref<VoxelBlockyLibraryBase> get_library() const;

	// Material for the smooth terrain surface (surface material id 0).
	void set_terrain_material(Ref<Material> material);
	Ref<Material> get_terrain_material() const;

	// Material for the smooth fluid iso-surface (surface material id 1). Meshed from
	// CHANNEL_DATA5 via a second Transvoxel pass inside build().
	void set_water_material(Ref<Material> material);
	Ref<Material> get_water_material() const;

	// Fallback material for model surfaces whose library material slot is null
	// (a library slot may be null to mean "the default material").
	void set_model_material(Ref<Material> material);
	Ref<Material> get_model_material() const;

	// Colours used by the default LOD shader (get_default_lod_material). The shader picks a
	// colour per vertex from the CUSTOM1 material id: id 0 -> terrain_color, id N>=1 ->
	// model_colors[N-1]. These are independent of terrain_material/model_material, which only
	// apply when the terrain is rendered without the default LOD shader.
	void set_terrain_color(Color color);
	Color get_terrain_color() const;

	void set_model_colors(PackedColorArray colors);
	PackedColorArray get_model_colors() const;

	// When enabled, the smooth terrain is flattened to the bottom edge of any voxel that
	// holds a model, so models sit flush on the ground instead of floating or sinking.
	void set_flatten_terrain_under_models(bool enabled);
	bool get_flatten_terrain_under_models() const;

	// Texturing mode is forwarded to the inner Transvoxel mesher. With TEXTURES_MIXEL4_S4 or
	// TEXTURES_SINGLE_S4, Transvoxel writes its 2-float texture data into CUSTOM1 of the
	// smooth surface, and the hybrid mesher relocates its per-vertex material id to CUSTOM2
	// so both data streams coexist. Models always store their material id in CUSTOM2 as well,
	// keeping every surface in the chunk on a uniform vertex format.
	void set_texturing_mode(VoxelMesherTransvoxel::TexturingMode mode);
	VoxelMesherTransvoxel::TexturingMode get_texturing_mode() const;

	// Forwarded to the inner Transvoxel mesher. When true, voxels with SDF > 0 (air) do not
	// contribute to the Mixel4 cell aggregation and per-vertex weight interpolation. Without
	// it, the generator's default indices on air voxels visually bleed into edits (e.g. an
	// earth sphere built above grass terrain renders with a strong grass tint).
	void set_textures_ignore_air_voxels(bool enabled);
	bool get_textures_ignore_air_voxels() const;

	// Forwarded to the inner terrain Transvoxel mesher: the SDF channel sampled per vertex into
	// CUSTOM3 as a "road" distance field (e.g. CHANNEL_DATA6), turned into a sharp material edge by
	// the terrain shader. -1 disables it. The fluid mesher keeps its own road channel (default -1)
	// for future water roads.
	void set_road_channel(int channel);
	int get_road_channel() const;

private:
	// True when the terrain road channel is configured (>= 0). Every surface in the chunk then has
	// to carry CUSTOM3 (real road data on road-crossing terrain, inert "outside" elsewhere).
	bool is_road_enabled() const;

	// Builds the blocky model part into `output` (one surface per material).
	// Also appends the model triangles to `output.collision_surface` so models are solid.
	// No-op if no library is set.
	void build_model_surface(VoxelMesher::Output &output, const VoxelMesher::Input &input);

	// Runs the fluid Transvoxel pass on the same buffer and tags its surfaces with material id 1
	// (water). The fluid pass does not contribute to collision -- collision_surface is preserved
	// across the call so players can swim through the fluid surface.
	void build_fluid_surfaces(VoxelMesher::Output &output, const VoxelMesher::Input &input);

	// Converts the Transvoxel terrain collision (provided as a submesh range of surface 0)
	// into explicit collision_surface arrays, so model triangles can be appended afterwards.
	void promote_terrain_collision(VoxelMesher::Output &output) const;

	// Adds the per-vertex material-id stream to the Transvoxel surface(s) and declares the
	// custom vertex stream formats in output.mesh_flags, so every surface -- terrain and
	// models -- ends up with the same vertex format. When the inner Transvoxel mesher's
	// texturing mode is TEXTURES_NONE the material id goes into CUSTOM1 (1 float); when
	// texturing is active CUSTOM1 is owned by Transvoxel's 2-float texture data and the
	// material id is written to CUSTOM2 instead.
	void finalize_terrain_surfaces(VoxelMesher::Output &output) const;

	// Returns true if any model voxel exists, and writes a flattened SDF copy into `dst`.
	// `dst` is the SDF the smooth surface should be meshed from instead of the original.
	bool build_flattened_sdf(VoxelBuffer &dst, const VoxelMesher::Input &input) const;

	// Pushes terrain_color / model_colors into the LOD shader material's u_material_colors.
	void update_lod_material_colors() const;

	static void _bind_methods();

	// Reused as-is for the smooth SDF surface (terrain, material id 0).
	Ref<VoxelMesherTransvoxel> _transvoxel_mesher;

	// Second Transvoxel instance configured to read CHANNEL_DATA5, producing the smooth fluid
	// iso-surface (water, material id 1) in the same build() call. Always TEXTURES_NONE: fluid
	// surfaces don't carry Mixel4 material weights.
	Ref<VoxelMesherTransvoxel> _fluid_transvoxel_mesher;

	Ref<VoxelBlockyLibraryBase> _library;

	Ref<Material> _terrain_material;
	Ref<Material> _water_material;
	Ref<Material> _model_material;

	// Colours fed into the default LOD shader (see set_terrain_color / set_model_colors).
	Color _terrain_color = Color(0.36f, 0.62f, 0.30f);
	PackedColorArray _model_colors;

	// The default LOD material, created once in the constructor (the terrain may request it
	// through a const reference). Colours are refreshed into it when they change.
	Ref<ShaderMaterial> _default_lod_material;

	bool _flatten_terrain_under_models = true;
};

} // namespace zylann::voxel

#endif // VOXEL_MESHER_HYBRID_H
