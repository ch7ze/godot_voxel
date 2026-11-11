#include "voxel_mesh_request.h"
#include "../../constants/voxel_string_names.h"
#include "../../engine/voxel_engine.h"

#ifdef ZN_GODOT
#include "../../util/godot/core/class_db.h"
#endif

namespace zylann::voxel {

VoxelMeshRequest::~VoxelMeshRequest() {
	remove_viewer();
}

void VoxelMeshRequest::init(ViewerID viewer_id, Box3i block_box, bool require_visuals, bool require_collisions) {
	_viewer_id = viewer_id;
	_has_viewer = true;
	_block_box = block_box;
	_require_visuals = require_visuals;
	_require_collisions = require_collisions;
}

void VoxelMeshRequest::release() {
	remove_viewer();
	if (_state == STATE_PENDING) {
		_state = STATE_RELEASED;
	}
}

bool VoxelMeshRequest::is_completed() const {
	return _state == STATE_COMPLETED;
}

void VoxelMeshRequest::finish(bool success) {
	if (!success) {
		remove_viewer();
	}
	_state = success ? STATE_COMPLETED : STATE_FAILED;
	emit_signal(VoxelStringNames::get_singleton().completed, success);
}

void VoxelMeshRequest::finish_deferred(bool success) {
	_state = success ? STATE_COMPLETED : STATE_FAILED;
	call_deferred(VoxelStringNames::get_singleton()._emit_completed, success);
}

VoxelMeshRequest::State VoxelMeshRequest::get_state() const {
	return _state;
}

const Box3i &VoxelMeshRequest::get_block_box() const {
	return _block_box;
}

bool VoxelMeshRequest::requires_visuals() const {
	return _require_visuals;
}

bool VoxelMeshRequest::requires_collisions() const {
	return _require_collisions;
}

// The engine can already be destroyed when a script releases its last reference during shutdown,
// in which case the viewer is gone with it.
void VoxelMeshRequest::remove_viewer() {
	if (_has_viewer && VoxelEngine::has_singleton()) {
		VoxelEngine::get_singleton().remove_viewer(_viewer_id);
	}
	_has_viewer = false;
}

// Intermediate method so deferred failures are emitted on the main thread after the caller connected
void VoxelMeshRequest::_b_emit_completed(bool success) {
	emit_signal(VoxelStringNames::get_singleton().completed, success);
}

void VoxelMeshRequest::_bind_methods() {
	ClassDB::bind_method(D_METHOD("release"), &VoxelMeshRequest::release);
	ClassDB::bind_method(D_METHOD("is_completed"), &VoxelMeshRequest::is_completed);

	// Internal
	ClassDB::bind_method(D_METHOD("_emit_completed", "success"), &VoxelMeshRequest::_b_emit_completed);

	ADD_SIGNAL(MethodInfo("completed", PropertyInfo(Variant::BOOL, "success")));
}

} // namespace zylann::voxel
