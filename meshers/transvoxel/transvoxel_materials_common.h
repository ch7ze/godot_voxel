#ifndef VOXEL_TRANSVOXEL_MATERIALS_COMMON_H
#define VOXEL_TRANSVOXEL_MATERIALS_COMMON_H

#include "transvoxel.h"

namespace zylann::voxel::transvoxel::materials {

// Each slot is cast to uint32_t before shifting. Without the cast, uint8_t is promoted to int, so a
// slot value of 128 or more shifted by 24 leaves the representable int range, which is undefined
// behavior. Material indices reach 255 since the Mixel4 index word moved to 8 bits per slot, and
// weights already span 0..255.
inline uint32_t pack_bytes(const FixedArray<uint8_t, 4> &a) {
	return (static_cast<uint32_t>(a[0]) << 0) | (static_cast<uint32_t>(a[1]) << 8) |
			(static_cast<uint32_t>(a[2]) << 16) | (static_cast<uint32_t>(a[3]) << 24);
}

inline void add_4i8_4w8_texture_data(
		StdVector<Vector2f> &uv,
		const uint32_t packed_indices,
		const FixedArray<uint8_t, 4> weights
) {
	struct IntUV {
		uint32_t x;
		uint32_t y;
	};
	static_assert(sizeof(IntUV) == sizeof(Vector2f), "Expected same binary size");
	uv.push_back(Vector2f());
	IntUV &iuv = *(reinterpret_cast<IntUV *>(&uv.back()));
	// print_line(String("{0}, {1}, {2}, {3}").format(varray(weights[0], weights[1], weights[2], weights[3])));
	iuv.x = packed_indices;
	iuv.y = pack_bytes(weights);
}

// Transition cells contain 2x2 values on one side, and 3x3 values on the other side.
// The side with 2x2 values only repeats the values at the corner of the 3x3 side.
// This function fills in an array of all values from the 3x3 "partial" side and fills in the redundant 2x2 ones.
template <typename TPartialArray9, typename TFullArray13>
void fill_redundant_transition_cell_values(const TPartialArray9 &src, TFullArray13 &dst) {
	for (unsigned int i = 0; i < src.size(); ++i) {
		dst[i] = src[i];
	}
	dst[0x9] = src[0];
	dst[0xA] = src[2];
	dst[0xB] = src[6];
	dst[0xC] = src[8];
}

} // namespace zylann::voxel::transvoxel::materials

#endif // VOXEL_TRANSVOXEL_MATERIALS_COMMON_H
