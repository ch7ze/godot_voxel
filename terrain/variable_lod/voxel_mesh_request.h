#ifndef VOXEL_MESH_REQUEST_H
#define VOXEL_MESH_REQUEST_H

#include "../../engine/ids.h"
#include "../../util/containers/std_unordered_set.h"
#include "../../util/godot/classes/ref_counted.h"
#include "../../util/math/box3i.h"

namespace zylann::voxel {

// Handle returned by VoxelLodTerrain::request_area_meshing().
// Holds an engine viewer until release() is called and emits `completed` exactly once.
class VoxelMeshRequest : public RefCounted {
	GDCLASS(VoxelMeshRequest, RefCounted)
public:
	enum State : uint8_t { //
		STATE_PENDING = 0,
		STATE_COMPLETED,
		STATE_FAILED,
		STATE_RELEASED
	};

	~VoxelMeshRequest();

	// Removes the engine viewer. Before completion, this also cancels the request without emitting `completed`.
	void release();
	bool is_completed() const;

	// Internal, used by VoxelLodTerrain on the main thread

	void init(ViewerID viewer_id, Box3i block_box, bool require_visuals, bool require_collisions);
	// Emits `completed` immediately. Only valid once the caller had the chance to connect.
	void finish(bool success);
	// Emits `completed` on the next idle frame, for failures detected before the caller could connect.
	void finish_deferred(bool success);

	State get_state() const;
	const Box3i &get_block_box() const;
	bool requires_visuals() const;
	bool requires_collisions() const;

	// Set once the request was checked against all its blocks
	bool initial_check_done = false;
	// LOD0 mesh block positions that are not loaded and active yet
	StdUnorderedSet<Vector3i> pending_blocks;

private:
	void remove_viewer();
	void _b_emit_completed(bool success);

	static void _bind_methods();

	ViewerID _viewer_id;
	bool _has_viewer = false;
	Box3i _block_box;
	bool _require_visuals = false;
	bool _require_collisions = false;
	State _state = STATE_PENDING;
};

} // namespace zylann::voxel

#endif // VOXEL_MESH_REQUEST_H
