#ifndef VOXEL_MESHER_TRANSVOXEL_H
#define VOXEL_MESHER_TRANSVOXEL_H

#include "../../storage/voxel_buffer_gd.h"
#include "../../util/macros.h"
#include "../voxel_mesher.h"
#include "transvoxel.h"

ZN_GODOT_FORWARD_DECLARE(class ArrayMesh);
ZN_GODOT_FORWARD_DECLARE(class ShaderMaterial);

namespace zylann::voxel {

class VoxelMesherTransvoxel : public VoxelMesher {
	GDCLASS(VoxelMesherTransvoxel, VoxelMesher)

public:
	enum TexturingMode {
		TEXTURES_NONE = transvoxel::TEXTURES_NONE,
		TEXTURES_MIXEL4_S4 = transvoxel::TEXTURES_MIXEL4_S4,
		TEXTURES_SINGLE_S4 = transvoxel::TEXTURES_SINGLE_S4,
		TEXTURES_MODE_COUNT = transvoxel::TEXTURES_MODE_COUNT
	};

	static const int TEXTURES_BLEND_4_OVER_16 = 1;

	VoxelMesherTransvoxel();
	~VoxelMesherTransvoxel();

	void build(VoxelMesher::Output &output, const VoxelMesher::Input &input) override;
	Ref<ArrayMesh> build_transition_mesh(Ref<godot::VoxelBuffer> voxels, int direction);

	int get_used_channels_mask() const override;

	bool is_generating_collision_surface() const override;

	void set_texturing_mode(TexturingMode mode);
	TexturingMode get_texturing_mode() const;

	void set_textures_ignore_air_voxels(const bool enable);
	bool get_textures_ignore_air_voxels() const;

	void set_mesh_optimization_enabled(bool enabled);
	bool is_mesh_optimization_enabled() const;

	void set_mesh_optimization_error_threshold(float threshold);
	float get_mesh_optimization_error_threshold() const;

	void set_mesh_optimization_target_ratio(float ratio);
	float get_mesh_optimization_target_ratio() const;

	void set_transitions_enabled(bool enable);
	bool get_transitions_enabled() const;

	void set_edge_clamp_margin(float margin);
	float get_edge_clamp_margin() const;

	// Channel the SDF is read from. Default CHANNEL_SDF matches the historical behavior;
	// VoxelMesherHybrid uses this to mesh a second iso-surface from a different channel
	// (e.g. CHANNEL_DATA5 for fluid volumes) without duplicating the Transvoxel code path.
	void set_sdf_channel(VoxelBuffer::ChannelId channel);
	VoxelBuffer::ChannelId get_sdf_channel() const;

	// Optional extra SDF channel sampled per vertex into ARRAY_CUSTOM3 (a "road" distance field,
	// e.g. CHANNEL_DATA6 for land roads, later CHANNEL_DATA7 for water roads). -1 disables it: no
	// CUSTOM3 stream, meshing unchanged. The consuming shader turns its sign into a sharp material
	// edge. Channel-parametric so the same code serves different road kinds.
	void set_road_channel(int channel);
	int get_road_channel() const;

private:
	// Variant-bindable wrappers around set/get_sdf_channel: VARIANT_ENUM_CAST is registered
	// only for the godot::VoxelBuffer::ChannelId alias, so ClassDB::bind_method must accept
	// that namespaced type and forward to the internal one. Mirrors VoxelTool::_b_set_channel.
	void _b_set_sdf_channel(godot::VoxelBuffer::ChannelId p_channel);
	godot::VoxelBuffer::ChannelId _b_get_sdf_channel() const;
public:

	Ref<ShaderMaterial> get_default_lod_material() const override;

	// Internal

	static void load_static_resources();
	static void free_static_resources();

	// Exposed for a fast-path. Return values are only valid until the next invocation of build() in the calling thread.
	static const transvoxel::MeshArrays &get_mesh_cache_from_current_thread();
	// Exposed for a fast-path. Return values are only valid if `detail_texture_hint` is true in the input given to
	// `build`, and only remains valid until the next invocation of build() in the calling thread.
	static Span<const transvoxel::CellInfo> get_cell_info_from_current_thread();

	// Not sure if that's necessary, currently transitions are either combined or not generated
	// enum TransitionMode {
	// 	// No transition meshes will be generated
	// 	TRANSITION_NONE,
	// 	// Generates transition meshes as separate meshes
	// 	TRANSITION_SEPARATE,
	// 	// Transition meshes will be part of the main mesh
	// 	TRANSITION_COMBINED
	// };

protected:
	static void _bind_methods();

private:
	TexturingMode _texture_mode = TEXTURES_NONE;

	struct MeshOptimizationParams {
		bool enabled = false;
		float error_threshold = 0.005;
		float target_ratio = 0.0;
	};

	MeshOptimizationParams _mesh_optimization_params;

	// When a marching cube cell is computed, vertices may be placed anywhere on edges of the cell, including very close
	// to corners. This can lead to very thin or small triangles, which can be a problem notably for collision. this
	// margin is the minimum distance from corners, below which vertices will be clamped to it. Increasing this value
	// reduces quality of the mesh.
	float _edge_clamp_margin = 0.02f;

	bool _transitions_enabled = true;

	bool _textures_ignore_air_voxels = false;

	// SDF source channel. CHANNEL_SDF preserves the historical single-channel meshing path;
	// a second VoxelMesherTransvoxel instance configured with CHANNEL_DATA5 is used by
	// VoxelMesherHybrid to mesh a separate fluid iso-surface from the same buffer.
	VoxelBuffer::ChannelId _sdf_channel = VoxelBuffer::CHANNEL_SDF;

	// Optional road distance-field channel sampled into ARRAY_CUSTOM3. -1 = disabled (default), so
	// existing setups are unaffected until explicitly configured (e.g. CHANNEL_DATA6).
	int _road_channel = -1;
};

} // namespace zylann::voxel

VARIANT_ENUM_CAST(zylann::voxel::VoxelMesherTransvoxel::TexturingMode);

#endif // VOXEL_MESHER_TRANSVOXEL_H
