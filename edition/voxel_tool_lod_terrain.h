#ifndef VOXEL_TOOL_LOD_TERRAIN_H
#define VOXEL_TOOL_LOD_TERRAIN_H

#include "../util/godot/core/random_pcg.h"
#include "../util/macros.h"
#include "voxel_tool.h"

ZN_GODOT_FORWARD_DECLARE(class Node);

namespace zylann::voxel {

class VoxelLodTerrain;
class VoxelDataMap;
class VoxelGeneratorGraph;

class VoxelToolLodTerrain : public VoxelTool {
	GDCLASS(VoxelToolLodTerrain, VoxelTool)
public:
	VoxelToolLodTerrain() {}
	VoxelToolLodTerrain(VoxelLodTerrain *terrain);

	bool is_area_editable(const Box3i &box) const override;
	Ref<VoxelRaycastResult> raycast(Vector3 pos, Vector3 dir, float max_distance, uint32_t collision_mask) override;
	void do_box(Vector3i begin, Vector3i end) override;
	void do_sphere(Vector3 center, float radius) override;
	void do_path(Span<const Vector3> positions, Span<const float> radii) override;
#ifdef VOXEL_ENABLE_MESH_SDF
	void do_mesh(const VoxelMeshSDF &mesh_sdf, const Transform3D &transform, const float isolevel) override;
#endif
	void copy(
			const Vector3i pos,
			VoxelBuffer &dst,
			const uint8_t p_channels_mask,
			const bool with_metadata
	) const override;
	void paste(Vector3i pos, const VoxelBuffer &src, uint8_t channels_mask) override;

	void set_voxel_metadata(const Vector3i pos, const Variant &meta) override;
	Variant get_voxel_metadata(const Vector3i pos) const override;

	// Specialized API

	int get_raycast_binary_search_iterations() const;
	void set_raycast_binary_search_iterations(int iterations);
	void do_sphere_async(Vector3 center, float radius);
	void do_hemisphere(Vector3 center, float radius, Vector3 flat_direction, float smoothness);
	// Lays a FLAT slab strip along the polyline `positions` -- a road-like band of constant width
	// and thickness, rounded at the segment ends in the horizontal plane but flat on top and bottom
	// (unlike do_path, which makes a round tube). half_width is the horizontal half-extent, thickness
	// the full vertical extent; the slab center line slopes between each point's Y. `smoothness`
	// blends the slab into the existing terrain over a soft radius (world meters) so it merges with a
	// rounded fillet instead of a hard crease; 0 = plain union. The slab SDF raises CHANNEL_SDF along
	// the trace. Used for roads (see VoxelEdits.BuildRoadPath). PackedVector3Array (not Span) so it
	// binds directly to C#.
	// material_index (Mixel4 index 0..255, or -1 = leave material untouched): when >= 0, a per-column
	// weight is blended into CHANNEL_INDICES / CHANNEL_WEIGHTS so the painted material follows the
	// road. paint_smoothness is the soft-edge width of the MATERIAL weight (world meters), separate
	// from `smoothness` which feathers the height rise: a wider material edge spreads the asphalt
	// weight over several voxels/cells, giving the shader's threshold enough interpolation support to
	// render a straight (non-faceted) border, while the height feather can stay narrow. <= 0 falls
	// back to `smoothness` (material edge == height edge). Ignored when material_index < 0.
	void do_flat_path(
			PackedVector3Array positions, float half_width, float thickness, float smoothness,
			int material_index = -1, float paint_smoothness = -1.0f);
	float get_voxel_f_interpolated(Vector3 position) const override;

	// Counts solid voxels inside the SDF sphere and aggregates their dominant Mixel4 material.
	// Returns a PackedInt32Array of length 16, indexed by material id (0..15).
	// Each entry holds the number of solid voxels in which that material had the highest weight.
	// "Solid" means CHANNEL_SDF value <= 0. Useful to determine drops for a single do_sphere edit
	// in one C++ call instead of thousands of GDExtension round-trips.
	PackedInt32Array get_material_counts_in_sphere(Vector3 center, float radius) const;

	// Like do_sphere with MODE_ADD on CHANNEL_SDF, but additionally sets CHANNEL_INDICES /
	// CHANNEL_WEIGHTS to the given material on every voxel that transitions from empty
	// (old_sdf > 0) to solid (new_sdf <= 0). Voxels that were already solid keep their
	// existing Mixel4 material, so building a stone sphere into existing dirt does not
	// repaint the dirt.
	// Returns the number of newly filled voxels (useful for inventory accounting).
	int do_sphere_add_with_material(Vector3 center, float radius, int material_index);

	// Blends a Mixel4 material into the CHANNEL_INDICES/WEIGHTS of the given individual voxels at
	// full weight, WITHOUT touching the SDF (the shape is unchanged). Unlike do_box with
	// MODE_TEXTURE_PAINT -- whose blend weight follows the box-SDF and is 0 at a single cell --
	// this paints exactly the listed voxels, so a caller can mark just the border voxels of a
	// region (never a filled area). One _post_edit covers the bounding box of all positions.
	// positions are voxel indices. material_index is a Mixel4 index (0..15).
	void paint_voxels_material(PackedVector3Array positions, int material_index);

	// Fluid sphere edits on CHANNEL_DATA5 (the smooth water iso-surface read by
	// VoxelMesherHybrid's fluid Transvoxel pass). Independent from the terrain SDF -- the two
	// volumes coexist on the same voxel grid but never overwrite each other.
	//
	// add: SDF union with the sphere (sdf = min(old, sphere_sdf)) -> fills the sphere with fluid.
	// remove: SDF subtraction (sdf = max(old, -sphere_sdf))       -> carves fluid out of the sphere.
	//
	// Center/radius are in world space, consistent with do_sphere and raycast.
	void do_sphere_fluid_add(Vector3 center, float radius);
	void do_sphere_fluid_remove(Vector3 center, float radius);

	// Batch-writes fluid SDF values into CHANNEL_DATA5 for a list of individual voxels with a
	// single editability check, a single remesh, and WITHOUT emitting `area_edited`. Bulk path
	// for the fluid simulator's per-tick surface band. positions are voxel indices; sdf_values
	// must match positions in length. See the .cpp for the rationale on suppressing the signal.
	// Synchronously streams in / generates the LOD0 data blocks overlapping the given world-space
	// AABB so the area becomes editable, then returns whether it is now editable. Lets callers edit
	// areas the viewers have not streamed in yet (e.g. deep caves) without permanently raising the
	// LOD0 load radius. Mirrors the Aabb->Box3i convention of is_area_editable. Returns false if the
	// stream errored (the area then stays non-editable). See VoxelData::try_load_area for details.
	bool try_load_area_editable(AABB box);

	void set_voxels_fluid_sdf(PackedVector3Array positions, PackedFloat32Array sdf_values);

	// Batched terrain-surface query (read-side counterpart of set_voxels_fluid_sdf). For each
	// column Vector3(x_voxel, start_y_voxel, z_voxel) returns the sub-voxel terrain top in
	// voxel-Y (SDF zero-crossing), or NAN if no ground was found within search_depth voxel
	// steps. One pre_generate over the bounding box, then per-column SDF reads (pure read).
	PackedFloat32Array get_column_terrain_tops(PackedVector3Array columns, int search_depth);

	// TODO GDX: it seems binding a method taking a `Node*` fails to compile. It is supposed to be working.
#if defined(ZN_GODOT)
	Array separate_floating_chunks(AABB world_box, Node *parent_node);
#elif defined(ZN_GODOT_EXTENSION)
	Array separate_floating_chunks(AABB world_box, Object *parent_node_o);
#endif

#ifdef VOXEL_ENABLE_MESH_SDF
	void stamp_sdf(Ref<VoxelMeshSDF> mesh_sdf, Transform3D transform, float isolevel, float sdf_scale);
#endif

	void do_graph(Ref<VoxelGeneratorGraph> graph, Transform3D transform, Vector3 area_size);

	void run_blocky_random_tick(
			const AABB voxel_area,
			const int voxel_count,
			const Callable &callback,
			const int block_batch_count,
			const uint32_t tags_mask
	);

	VoxelFormat get_format() const override;

protected:
	uint64_t _get_voxel(Vector3i pos) const override;
	float _get_voxel_f(Vector3i pos) const override;
	void _set_voxel(Vector3i pos, uint64_t v) override;
	void _set_voxel_f(Vector3i pos, float v) override;
	void _post_edit(const Box3i &box) override;

private:
	static void _bind_methods();

	// Shared implementation behind do_sphere_fluid_add / do_sphere_fluid_remove.
	// Identical to do_sphere() but with channel hardcoded to CHANNEL_DATA5 and mode
	// passed in explicitly.
	void run_fluid_sphere_op(Vector3 center, float radius, ops::Mode mode);

	VoxelLodTerrain *_terrain = nullptr;
	int _raycast_binary_search_iterations = 0;
	RandomPCG _random;
};

} // namespace zylann::voxel

#endif // VOXEL_TOOL_LOD_TERRAIN_H
