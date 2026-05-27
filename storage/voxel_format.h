#ifndef VOXEL_FORMAT_H
#define VOXEL_FORMAT_H

#include "voxel_buffer.h"
#include <array>

namespace zylann::voxel {

struct VoxelFormat {
	VoxelFormat();

	void configure_buffer(VoxelBuffer &vb) const;

	bool operator==(const VoxelFormat &other) const {
		return depths == other.depths && channels_acting_as_sdf == other.channels_acting_as_sdf;
	}

	// Per-channel flag (bitmask over ChannelId) marking generic data channels as SDF
	// (sign + Q16/Q8 encoding identical to CHANNEL_SDF). CHANNEL_SDF is always implicitly
	// treated as SDF -- this mask covers ADDITIONAL channels (DATA5..DATA7 typically).
	//
	// Consumed by get_voxel_sv() / set_voxel_f() / get_voxel_f() in voxel_data so the
	// generic float<->raw conversion path is taken on these channels too, instead of
	// returning the raw uint in VoxelSingleValue::i.
	//
	// Without this mask, a data channel storing SDF-encoded bytes would be writable via
	// set_voxel_f (which always Q-encodes) but unreadable via get_voxel_f (which only
	// dequantizes for CHANNEL_SDF -- everything else returns the raw bits reinterpreted
	// as float, producing sub-normal garbage).
	inline bool is_channel_acting_as_sdf(const VoxelBuffer::ChannelId channel_id) const {
		if (channel_id == VoxelBuffer::CHANNEL_SDF) {
			return true;
		}
		return (channels_acting_as_sdf & (uint8_t(1) << channel_id)) != 0;
	}

	struct DepthRange {
		uint32_t min;
		uint32_t max;

		inline bool contains(uint32_t i) const {
			return i >= min && i <= max;
		}
	};

	inline uint64_t get_default_raw_value(const VoxelBuffer::ChannelId channel_id) const {
		return VoxelBuffer::get_default_raw_value(channel_id, depths[channel_id]);
	}

	static DepthRange get_supported_depths(const VoxelBuffer::ChannelId channel_id);
	static uint64_t get_default_sdf_raw_value(const VoxelBuffer::Depth depth);

	std::array<VoxelBuffer::Depth, VoxelBuffer::MAX_CHANNELS> depths;

	// Bitmask: bit N set = channel N stores SDF-encoded values. CHANNEL_SDF is implicit
	// (always treated as SDF). Default 0 -- only generic data channels need this opt-in.
	uint8_t channels_acting_as_sdf = 0;
};

} // namespace zylann::voxel

#endif // VOXEL_FORMAT_H
