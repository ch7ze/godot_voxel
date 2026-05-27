#ifndef VOXEL_FORMAT_GD_H
#define VOXEL_FORMAT_GD_H

#include "../util/godot/classes/resource.h"
#include "voxel_buffer_gd.h"
#include "voxel_format.h"

namespace zylann::voxel::godot {

class VoxelFormat : public Resource {
	GDCLASS(VoxelFormat, Resource)
public:
	void set_channel_depth(const VoxelBuffer::ChannelId channel_index, const VoxelBuffer::Depth depth);
	VoxelBuffer::Depth get_channel_depth(const VoxelBuffer::ChannelId channel_index) const;

	// Channels that should be treated as SDF (Q16/Q8 sign-normalized encoding) in addition
	// to the implicit CHANNEL_SDF. Typical use: opt CHANNEL_DATA5..DATA7 into the SDF
	// pipeline so that VoxelData::get_voxel returns a properly decoded float instead of
	// the raw uint reinterpreted as float. See voxel_format.h::is_channel_acting_as_sdf.
	void set_channel_acting_as_sdf(const VoxelBuffer::ChannelId channel_index, const bool enable);
	bool is_channel_acting_as_sdf(const VoxelBuffer::ChannelId channel_index) const;

	void configure_buffer(Ref<VoxelBuffer> buffer) const;
	Ref<VoxelBuffer> create_buffer(const Vector3i size) const;

	zylann::voxel::VoxelFormat get_internal() const {
		return _internal;
	}

private:
	void _b_set_data(const Array &data);
	Array _b_get_data() const;

	static void _bind_methods();

	zylann::voxel::VoxelFormat _internal;
};

} // namespace zylann::voxel::godot

#endif // VOXEL_FORMAT_GD_H
