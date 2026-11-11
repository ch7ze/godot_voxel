#include "voxel_mesher_hybrid.h"
#include "../../constants/cube_tables.h"
#include "../../shaders/voxel_mesher_hybrid_shader.h"
#include "../../storage/voxel_buffer.h"
#include "../../util/containers/std_unordered_map.h"
#include "../../util/containers/std_vector.h"
#include "../../util/godot/classes/rendering_server.h"
#include "../../util/godot/classes/shader.h"
#include "../../util/godot/classes/shader_material.h"
#include "../../util/godot/core/packed_arrays.h"
#include "../../util/math/conv.h"
#include "../../util/math/funcs.h"
#include "../../util/profiling.h"
#include "../blocky/blocky_baked_library.h"
// generate_fluid_model template + side-culling helper come from the blocky mesher headers.
#include "../blocky/blocky_fluids_meshing_impl.h"
// For blocky::is_face_visible (neighbor side-culling), an exported inline helper.
#include "../blocky/voxel_mesher_blocky.h"

using namespace zylann::godot;

namespace zylann::voxel {

namespace {

// Padding border to skip around the meshed block, matching VoxelMesherBlocky::PADDING (1).
// Kept as a local constant so this file does not need to pull in the whole blocky mesher header.
static constexpr int MODEL_PADDING = 1;

// Plain geometry arrays for one mesh surface.
struct SurfaceArrays {
	StdVector<Vector3f> positions;
	StdVector<Vector3f> normals;
	StdVector<Vector2f> uvs;
	StdVector<int32_t> indices;

	void append_geometry(
			const StdVector<Vector3f> &src_positions,
			const StdVector<Vector3f> &src_normals,
			const StdVector<Vector2f> &src_uvs,
			const StdVector<int> &src_indices,
			const Vector3f voxel_offset
	) {
		if (src_positions.size() == 0 || src_indices.size() == 0) {
			return;
		}
		const int base_index = static_cast<int>(positions.size());

		for (unsigned int i = 0; i < src_positions.size(); ++i) {
			positions.push_back(src_positions[i] + voxel_offset);
			// Side surfaces don't store normals; fall back to a zero normal in that case.
			normals.push_back(i < src_normals.size() ? src_normals[i] : Vector3f());
			uvs.push_back(i < src_uvs.size() ? src_uvs[i] : Vector2f());
		}
		for (const int idx : src_indices) {
			indices.push_back(base_index + idx);
		}
	}
};

// Accumulates baked model geometry from the TYPE channel, grouped by material id so each
// material can become its own mesh surface. Cube sides occluded by a neighbor voxel are
// culled (see build_model_surface); inner model geometry is always kept.
//
// blocky::generate_mesh (VoxelMesherBlocky) does similar batching but is a file-local
// template, not exported in any header, and also does AO/tint this mesher does not need --
// so it cannot be reused as-is. The neighbor-culling check itself IS reused, via the
// exported blocky::is_face_visible helper.
using ModelMeshAccumulator = StdUnorderedMap<uint32_t, SurfaceArrays>;

// Appends a surface's triangles to the collision surface, offsetting indices by the
// triangles already present so the appended mesh stays consistent.
void append_to_collision_surface(VoxelMesher::Output::CollisionSurface &col, const SurfaceArrays &src) {
	const int base = static_cast<int>(col.positions.size());
	for (const Vector3f &p : src.positions) {
		col.positions.push_back(p);
	}
	for (const int32_t idx : src.indices) {
		col.indices.push_back(base + idx);
	}
}

// All surfaces of the hybrid mesh share one vertex format so the terrain can render them with a
// single material_override (the default LOD shader). Up to three custom vertex streams are used:
//   CUSTOM0 (4x float): Transvoxel LOD attributes -- secondary position + packed border masks.
//                       The shader needs it on surface 0 for the transition cull; model surfaces
//                       carry an inert value (secondary == primary, all masks 0, transition 0)
//                       so the same shader leaves them untouched.
//   CUSTOM1: either the per-vertex material id (1x float) when Transvoxel texturing is disabled,
//            or the packed Mixel4/Single material indices+weights produced by Transvoxel
//            (2x float) when texturing is enabled. Used by the shader for material lookup or
//            blend-texturing.
//   CUSTOM2 (1x float, only when texturing is enabled): per-vertex material id (0 = smooth
//            terrain surface, 1.. = model surfaces). This separate stream is needed because
//            CUSTOM1 is then owned by Transvoxel's texture data.
constexpr int CUSTOM0_FLOATS_PER_VERTEX = 4;
// CUSTOM1 (no-texturing mode) and CUSTOM2 (texturing mode) both store one float per vertex.
constexpr int MATERIAL_ID_FLOATS_PER_VERTEX = 1;

// Mesh_flags declaring the format of the custom streams when texturing is disabled. Both CUSTOM0
// and CUSTOM1 are always declared, so a chunk with models but no smooth terrain still gets a
// matching format on every surface ("Invalid array format" otherwise -- see build_model_surface).
constexpr int HYBRID_CUSTOM_MESH_FLAGS_NO_TEXTURING =
		(RenderingServerEnums::ARRAY_CUSTOM_RGBA_FLOAT << Mesh::ARRAY_FORMAT_CUSTOM0_SHIFT) |
		(RenderingServerEnums::ARRAY_CUSTOM_R_FLOAT << Mesh::ARRAY_FORMAT_CUSTOM1_SHIFT);

// Mesh_flags when Transvoxel texturing is enabled: CUSTOM1 carries Transvoxel's 2-float texture
// data; CUSTOM2 carries the per-vertex material id the hybrid mesher uses to pick a colour.
constexpr int HYBRID_CUSTOM_MESH_FLAGS_TEXTURED =
		(RenderingServerEnums::ARRAY_CUSTOM_RGBA_FLOAT << Mesh::ARRAY_FORMAT_CUSTOM0_SHIFT) |
		(RenderingServerEnums::ARRAY_CUSTOM_RG_FLOAT << Mesh::ARRAY_FORMAT_CUSTOM1_SHIFT) |
		(RenderingServerEnums::ARRAY_CUSTOM_R_FLOAT << Mesh::ARRAY_FORMAT_CUSTOM2_SHIFT);

// Builds the inert CUSTOM0 stream for a model surface: secondary position equals the primary
// position and every packed mask byte is 0, so the LOD shader neither moves nor culls these
// vertices (models are not transition geometry and do not shift between LOD levels).
PackedFloat32Array build_model_custom0(const StdVector<Vector3f> &positions) {
	PackedFloat32Array custom0;
	custom0.resize(static_cast<int64_t>(positions.size()) * CUSTOM0_FLOATS_PER_VERTEX);
	float *w = custom0.ptrw();
	for (const Vector3f &p : positions) {
		*w++ = p.x; // secondary_position.x
		*w++ = p.y; // secondary_position.y
		*w++ = p.z; // secondary_position.z
		*w++ = 0.0f; // packed masks (cell/vertex border, transition) all zero
	}
	return custom0;
}

// Builds a CUSTOM stream containing the same single float (a material id) for every vertex.
// Used for CUSTOM1 when texturing is disabled, and for CUSTOM2 when texturing is enabled.
PackedFloat32Array build_material_id_stream(int vertex_count, float material_id) {
	PackedFloat32Array stream;
	stream.resize(static_cast<int64_t>(vertex_count) * MATERIAL_ID_FLOATS_PER_VERTEX);
	float *w = stream.ptrw();
	for (int i = 0; i < vertex_count; ++i) {
		*w++ = material_id;
	}
	return stream;
}

// Builds an inert CUSTOM1 stream for model surfaces when texturing is enabled: Transvoxel's
// shader expects 2 floats per vertex (packed indices + weights), but model surfaces have no
// voxel-based material data. Zeros are written so the shader's texture sampler returns the
// first texture with no weight, which is harmless -- the shader's CUSTOM2 material id branches
// model surfaces away from texture sampling anyway.
PackedFloat32Array build_model_inert_custom1_texturing(int vertex_count) {
	PackedFloat32Array stream;
	stream.resize(static_cast<int64_t>(vertex_count) * 2);
	float *w = stream.ptrw();
	for (int i = 0; i < vertex_count * 2; ++i) {
		*w++ = 0.0f;
	}
	return stream;
}

} // namespace

VoxelMesherHybrid::VoxelMesherHybrid() {
	_transvoxel_mesher.instantiate();
	// Padding is dictated by the smooth part. The blocky model part needs no neighbor padding
	// (each model is self-contained within its voxel), so the Transvoxel requirements cover it.
	set_padding(_transvoxel_mesher->get_minimum_padding(), _transvoxel_mesher->get_maximum_padding());

	// Fluid pass: a second Transvoxel instance reading CHANNEL_DATA5. Texturing is left at
	// TEXTURES_NONE so the fluid surface stays a single, untextured surface (its shader will
	// drive the look from world-space coordinates instead of Mixel4 weights).
	_fluid_transvoxel_mesher.instantiate();
	_fluid_transvoxel_mesher->set_sdf_channel(VoxelBuffer::CHANNEL_DATA5);
	_fluid_transvoxel_mesher->set_texturing_mode(VoxelMesherTransvoxel::TEXTURES_NONE);

	// Create the default LOD material once. The terrain may request it through a const
	// reference (get_default_lod_material), so it cannot be lazily created there.
	{
		Ref<Shader> shader;
		shader.instantiate();
		shader->set_code(g_voxel_mesher_hybrid_shader);
		_default_lod_material.instantiate();
		_default_lod_material->set_shader(shader);
	}
	update_lod_material_colors();
}

VoxelMesherHybrid::~VoxelMesherHybrid() {}

int VoxelMesherHybrid::get_used_channels_mask() const {
	// SDF for the smooth terrain surface, fluid-SDF (CHANNEL_DATA5) for the smooth water
	// surface, TYPE for the blocky model surface.
	int mask = _transvoxel_mesher->get_used_channels_mask();
	if (_fluid_transvoxel_mesher.is_valid()) {
		mask |= _fluid_transvoxel_mesher->get_used_channels_mask();
	}
	mask |= (1 << VoxelBuffer::CHANNEL_TYPE);
	return mask;
}

bool VoxelMesherHybrid::is_generating_collision_surface() const {
	// Always true: build() produces an explicit collision_surface (terrain geometry promoted
	// from Transvoxel's submesh range, plus the model triangles). This does not depend on
	// what the inner Transvoxel mesher reports -- the hybrid mesher owns its collision output.
	return true;
}

Ref<Material> VoxelMesherHybrid::get_material_by_index(unsigned int index) const {
	// Index 0 is the smooth terrain surface, index 1 is the smooth water surface.
	if (index == 0) {
		return _terrain_material;
	}
	if (index == 1) {
		return _water_material;
	}
	// Indices >= 2 map to the model library's materials (index 2 -> library material 0).
	if (_library.is_valid()) {
		const Ref<Material> lib_material = _library->get_material_by_index(index - 2);
		// A library material slot may be null ("the default material"): fall back to
		// the configurable model material so such models still get something to render.
		if (lib_material.is_valid()) {
			return lib_material;
		}
	}
	return _model_material;
}

unsigned int VoxelMesherHybrid::get_material_index_count() const {
	// 1 for the terrain surface, 1 for the water surface, plus one slot per library material.
	unsigned int count = 2;
	if (_library.is_valid()) {
		count += _library->get_material_index_count();
	}
	return count;
}

void VoxelMesherHybrid::set_texturing_mode(VoxelMesherTransvoxel::TexturingMode mode) {
	ERR_FAIL_COND(_transvoxel_mesher.is_null());
	_transvoxel_mesher->set_texturing_mode(mode);
}

VoxelMesherTransvoxel::TexturingMode VoxelMesherHybrid::get_texturing_mode() const {
	if (_transvoxel_mesher.is_null()) {
		return VoxelMesherTransvoxel::TEXTURES_NONE;
	}
	return _transvoxel_mesher->get_texturing_mode();
}

void VoxelMesherHybrid::set_textures_ignore_air_voxels(bool enabled) {
	ERR_FAIL_COND(_transvoxel_mesher.is_null());
	_transvoxel_mesher->set_textures_ignore_air_voxels(enabled);
}

bool VoxelMesherHybrid::get_textures_ignore_air_voxels() const {
	if (_transvoxel_mesher.is_null()) {
		return false;
	}
	return _transvoxel_mesher->get_textures_ignore_air_voxels();
}

void VoxelMesherHybrid::set_terrain_material(Ref<Material> material) {
	_terrain_material = material;
}

Ref<Material> VoxelMesherHybrid::get_terrain_material() const {
	return _terrain_material;
}

void VoxelMesherHybrid::set_water_material(Ref<Material> material) {
	_water_material = material;
}

Ref<Material> VoxelMesherHybrid::get_water_material() const {
	return _water_material;
}

void VoxelMesherHybrid::set_model_material(Ref<Material> material) {
	_model_material = material;
}

Ref<Material> VoxelMesherHybrid::get_model_material() const {
	return _model_material;
}

void VoxelMesherHybrid::set_terrain_color(Color color) {
	_terrain_color = color;
	update_lod_material_colors();
}

Color VoxelMesherHybrid::get_terrain_color() const {
	return _terrain_color;
}

void VoxelMesherHybrid::set_model_colors(PackedColorArray colors) {
	_model_colors = colors;
	update_lod_material_colors();
}

PackedColorArray VoxelMesherHybrid::get_model_colors() const {
	return _model_colors;
}

Ref<ShaderMaterial> VoxelMesherHybrid::get_default_lod_material() const {
	return _default_lod_material;
}

void VoxelMesherHybrid::update_lod_material_colors() const {
	if (_default_lod_material.is_null()) {
		return;
	}
	// u_material_colors is a fixed-size shader array (MATERIAL_COLOR_COUNT == 8), indexed
	// directly by the per-vertex material id the surfaces emit: 0 = smooth terrain, 1 = smooth
	// water, 2.. = model surfaces (surface id = 2 + library material id, see
	// build_model_surface). _model_colors uses the SAME surface-id indexing -- slot 0 is unused
	// because the terrain colour wins there -- so no offset is applied between the array and the
	// vertex ids. The shader clamps the per-vertex id into range, so providing fewer colors than
	// slots is safe.
	constexpr int MATERIAL_COLOR_COUNT = 8;
	Array colors;
	colors.resize(MATERIAL_COLOR_COUNT);
	colors[0] = _terrain_color;
	for (int i = 1; i < MATERIAL_COLOR_COUNT; ++i) {
		colors[i] = i < _model_colors.size() ? _model_colors[i] : Color(1, 1, 1);
	}
	_default_lod_material->set_shader_parameter("u_material_colors", colors);
}

void VoxelMesherHybrid::set_library(Ref<VoxelBlockyLibraryBase> library) {
	_library = library;
}

Ref<VoxelBlockyLibraryBase> VoxelMesherHybrid::get_library() const {
	return _library;
}

void VoxelMesherHybrid::set_flatten_terrain_under_models(bool enabled) {
	_flatten_terrain_under_models = enabled;
}

bool VoxelMesherHybrid::get_flatten_terrain_under_models() const {
	return _flatten_terrain_under_models;
}

bool VoxelMesherHybrid::build_flattened_sdf(VoxelBuffer &dst, const VoxelMesher::Input &input) const {
	ZN_PROFILE_SCOPE();

	const VoxelBuffer &src = input.voxels;
	const VoxelBuffer::ChannelId sdf_channel = VoxelBuffer::CHANNEL_SDF;
	const VoxelBuffer::ChannelId type_channel = VoxelBuffer::CHANNEL_TYPE;

	if (src.is_uniform(type_channel) && src.get_voxel(Vector3i(), type_channel) == blocky::AIR_ID) {
		// No models: no flattening needed, the original SDF can be used directly.
		return false;
	}

	const Vector3i block_size = src.get_size();

	// Work on an own copy of the SDF channel so the original (const) buffer is untouched.
	dst.create(block_size);
	dst.set_channel_depth(sdf_channel, src.get_channel_depth(sdf_channel));
	dst.copy_channel_from(src, sdf_channel);
	// Mixel4 texture data must travel along: Transvoxel reads CHANNEL_INDICES/CHANNEL_WEIGHTS to
	// pick per-vertex material indices and blend weights. Without this copy, the flattened buffer
	// reverts to the default indices/weights (slot 0 = material id 0) and every chunk containing a
	// model voxel renders as a uniform fallback texture.
	if (_transvoxel_mesher.is_valid()
		&& _transvoxel_mesher->get_texturing_mode() != VoxelMesherTransvoxel::TEXTURES_NONE) {
		for (const VoxelBuffer::ChannelId ch : { VoxelBuffer::CHANNEL_INDICES, VoxelBuffer::CHANNEL_WEIGHTS }) {
			dst.set_channel_depth(ch, src.get_channel_depth(ch));
			dst.copy_channel_from(src, ch);
		}
	}

	const int sx = block_size.x;
	const int sz = block_size.z;

	// Pass 1: build the per-column flatten height.
	//
	// A model voxel at (mx, my, mz) occupies the world box (mx..mx+1, my..my+1, mz..mz+1):
	// its bottom face is the square at height `my` spanning columns mx..mx+1 x mz..mz+1.
	// Transvoxel meshes the surface under that face from the cell at (mx, my-1, mz), whose
	// four UPPER corners are the columns (mx,mz), (mx+1,mz), (mx,mz+1), (mx+1,mz+1).
	//
	// To make a model sit flush on all four of its bottom edges, all FOUR of those corner
	// columns must be flattened to `my` -- not just the model's own column (which would tilt
	// the surface toward one corner), and not the whole 3x3 neighborhood (which lifts a plateau
	// and leaves ridges).
	//
	// Only STACK BASES (model voxels with no model voxel directly below) trigger flattening:
	// the terrain only needs to meet the bottom of a stack; the higher voxels of the same stack
	// are supported by the lower ones, not by the ground. If every model voxel in a stack
	// claimed its own height, several blocks stacked over one cell would pull the ground up to
	// the topmost block and produce a visible cone of ground under the stack.
	//
	// `column_flatten_y` records, per column, the LOWEST stack-base height that claims it as a
	// corner. If two separate stacks (e.g. a floating block above a ground block) share a column,
	// the lower base wins -- it is the one that needs ground contact, the upper one floats.
	//
	// A model voxel only counts here if build_model_surface() would actually render it. The
	// same validity test is used (in library, not empty): a TYPE value that does not produce
	// geometry must not flatten the terrain either, otherwise the ground gets an axis-aligned
	// step with no model resting on it (a visible floating-triangle artifact).
	StdVector<int> column_flatten_y;
	column_flatten_y.resize(sx * sz, -1);
	bool any_model = false;
	{
		// Baked data is the only library data safe for multithreaded access.
		RWLockRead lock(_library->get_baked_data_rw_lock());
		const blocky::BakedLibrary &baked = _library->get_baked_data();

		auto is_rendered_model = [&baked](uint64_t voxel_id) -> bool {
			if (voxel_id == blocky::AIR_ID || !baked.has_model(voxel_id)) {
				return false;
			}
			return !baked.models[voxel_id].empty;
		};

		for (int z = 0; z < sz; ++z) {
			for (int x = 0; x < sx; ++x) {
				// The y loop walks a column upwards, so the voxel below is the one read in the
				// previous iteration. Carrying it over replaces the second TYPE read per voxel.
				bool below_is_model = false;
				for (int y = 0; y < block_size.y; ++y) {
					const uint64_t voxel_id = src.get_voxel(Vector3i(x, y, z), type_channel);
					const bool is_model = is_rendered_model(voxel_id);
					// Stack-base test: only the lowest voxel of a stacked column triggers
					// flattening. If `y == 0` we are at the bottom of the buffer (including its
					// padding border); we conservatively treat that as a base. If the stack
					// actually continues into the neighbour chunk below, that chunk owns the
					// real base position and will flatten there.
					const bool is_stack_base = is_model && !below_is_model;
					below_is_model = is_model;
					if (!is_model) {
						continue;
					}
					any_model = true;
					if (!is_stack_base) {
						continue;
					}
					// Ground-proximity test: the ground under the model voxel is the Transvoxel
					// cell at `(x, y - 1, z)`, whose four LOWER corners are the samples
					// (x..x+1, y-1, z..z+1). The test reads all four of them and passes as soon
					// as one is solid, which matches the claim below: the claim covers all four
					// corner columns of that same cell, so a one-sample test would reject cells
					// that the terrain iso-surface crosses diagonally (cliff edges, ridges) even
					// though the model does rest on ground there.
					//
					// Neighbour cells still do not count -- a block next to a cliff gets no
					// flattening from the cliff, only the cell it actually stands on is read.
					//
					// If the test is positive, ALL FOUR corner columns are claimed (the bottom
					// face is a quad and Transvoxel meshes it from 4 corner samples; claiming
					// fewer than 4 would tilt the cell and leave gaps to the block bottom).
					// If the test is negative, NO corner column is claimed -- the block floats.
					//
					// If `y == 0` we can't read `y - 1` and treat as "no ground". The neighbour
					// chunk below owns the real proximity decision if the base actually continues
					// into it.
					if (y == 0) {
						continue;
					}
					// SDF convention: negative = solid (matches Pass 2 below and the generators
					// that write sdf = world_y - height).
					bool ground_below = false;
					for (int gz = z; gz <= z + 1 && !ground_below; ++gz) {
						for (int gx = x; gx <= x + 1 && !ground_below; ++gx) {
							if (gx >= sx || gz >= sz) {
								continue;
							}
							ground_below = src.get_voxel_f(Vector3i(gx, y - 1, gz), sdf_channel) < 0.0f;
						}
					}
					if (!ground_below) {
						continue;
					}
					for (int cz = z; cz <= z + 1; ++cz) {
						for (int cx = x; cx <= x + 1; ++cx) {
							if (cx >= sx || cz >= sz) {
								continue;
							}
							int &slot = column_flatten_y[cx + cz * sx];
							if (slot < 0 || y < slot) {
								slot = y;
							}
						}
					}
				}
			}
		}
	}
	if (!any_model) {
		return false;
	}

	bool any_flattened = false;

	// Pass 2: in every claimed corner column, combine the SDF with the model support plane
	// via an SDF union:  sdf(y) = min( original_sdf(y), y - flatten_y ).
	// `y - flatten_y` is the signed distance to the horizontal plane at the model bottom.
	// The union (minimum) raises the ground only where it lies below that plane and keeps the
	// original ground where it is already higher, so:
	//   - model over a dip   -> ground is lifted up to the model bottom (flush on all 4 edges),
	//   - model on a slope   -> uphill ground stays higher and the model embeds into the slope.
	// Columns outside the claimed corners are left untouched, so Transvoxel interpolates a
	// smooth slope from the flattened square down to the surrounding terrain.
	//
	// Only the two y rows that actually shape the cell directly below the model are written:
	//   - y = flatten_y       (top samples of that cell, clamped to <= 0 -> surface ends here),
	//   - y = flatten_y - 1   (bottom samples of that cell, clamped to <= -1 -> guaranteed solid).
	// Marching Cubes / Transvoxel polygonize a cell from its 8 corner samples only, so samples
	// further down do not affect the surface at the model bottom. Writing the full column would
	// also overwrite samples deep below the block on every remesh -- making any sdf edit (e.g.
	// digging a tunnel under the block) invisible, because Pass 2 would re-clamp those cells to
	// `plane < 0` (solid) at every build. Restricting Pass 2 to these two rows leaves the rest
	// of the column on whatever the data stream has (generator or edits) and lets digging show
	// through under a placed block, the same way it shows through anywhere else.
	for (int z = 0; z < sz; ++z) {
		for (int x = 0; x < sx; ++x) {
			const int flatten_y = column_flatten_y[x + z * sx];
			if (flatten_y < 0) {
				continue;
			}
			// flatten_y - 1 can fall below 0 at chunk borders; skip those out-of-range writes.
			// The neighbour chunk below owns the actual bottom row in that case.
			const int y_lo = flatten_y - 1;
			const int y_hi = flatten_y;
			for (int y : { y_lo, y_hi }) {
				if (y < 0 || y >= block_size.y) {
					continue;
				}
				const Vector3i pos(x, y, z);
				const float original = dst.get_voxel_f(pos, sdf_channel);
				const float plane = static_cast<float>(y - flatten_y);
				const float unified = original < plane ? original : plane;
				if (unified != original) {
					dst.set_voxel_f(unified, pos, sdf_channel);
					any_flattened = true;
				}
			}
		}
	}

	return any_flattened;
}

void VoxelMesherHybrid::build(VoxelMesher::Output &output, const VoxelMesher::Input &input) {
	ZN_PROFILE_SCOPE();

	// Surface 0: smooth SDF terrain, delegated to the reused Transvoxel mesher.
	// It appends its surface(s) and sets collision_surface / mesh_flags / primitive_type.
	if (_flatten_terrain_under_models && !_library.is_null()) {
		// Thread-local: build() may run on several threads at once.
		// ALLOCATOR_DEFAULT (not POOL): a thread_local buffer outlives single calls and would
		// otherwise hold VoxelMemoryPool blocks until thread exit, which can happen after the
		// pool singleton is torn down on module unregister (reported as a recycling leak).
		thread_local VoxelBuffer tls_flattened_sdf(VoxelBuffer::ALLOCATOR_DEFAULT);
		if (build_flattened_sdf(tls_flattened_sdf, input)) {
			// Mesh the smooth surface from the flattened SDF copy instead of the original.
			// Input.voxels is a reference, so a new Input must be constructed rather than
			// reassigned. Designated initializers are used so every field is named: if the
			// upstream Input struct gains a field, it keeps its default here instead of a
			// member being silently shifted to a wrong value.
			const VoxelMesher::Input flattened_input{
				.voxels = tls_flattened_sdf,
				.generator = input.generator,
				.origin_in_voxels = input.origin_in_voxels,
				.lod_index = input.lod_index,
				.collision_hint = input.collision_hint,
				.lod_hint = input.lod_hint,
				.detail_texture_hint = input.detail_texture_hint
			};
			_transvoxel_mesher->build(output, flattened_input);
		} else {
			_transvoxel_mesher->build(output, input);
		}
	} else {
		_transvoxel_mesher->build(output, input);
	}

	// Give the Transvoxel terrain surface(s) the CUSTOM1 material-id stream so every surface of
	// the final mesh shares one vertex format (model surfaces get CUSTOM1 in build_model_surface).
	// This tags the surfaces produced so far (terrain only) with material id 0. Must run before
	// the fluid pass so that finalize_terrain_surfaces doesn't also tag the fluid surfaces as
	// terrain.
	finalize_terrain_surfaces(output);

	// Convert the Transvoxel terrain collision (a submesh range of surface 0) into explicit
	// collision_surface arrays, so the model triangles can be appended to the same collider.
	// Must run before the fluid pass, since the fluid pass would otherwise overwrite the
	// terrain's submesh_*_end markers with its own (fluid is not a collider).
	promote_terrain_collision(output);

	// Smooth fluid iso-surface: meshed from CHANNEL_DATA5 by a second Transvoxel instance.
	// Tagged with material id 1 (see get_material_by_index). The fluid surface is not part of
	// the collider -- players should be able to enter water -- so collision_surface stays as
	// promote_terrain_collision() left it.
	build_fluid_surfaces(output, input);

	// Sharp-edged 3D models baked from the TYPE channel. Each model material id N becomes
	// surface material index 2 + N (terrain=0, water=1, models from 2). Appends the model
	// triangles to output.collision_surface.
	build_model_surface(output, input);
}

void VoxelMesherHybrid::build_fluid_surfaces(VoxelMesher::Output &output, const VoxelMesher::Input &input) {
	ZN_PROFILE_SCOPE();

	if (_fluid_transvoxel_mesher.is_null()) {
		return;
	}

	// Snapshot state the inner Transvoxel build() may overwrite, so the terrain results are
	// preserved verbatim:
	//   - collision_surface submesh markers (the terrain promoted its collision already; fluid
	//     does not contribute to collision),
	//   - mesh_flags (the inner build() does `mesh_flags = ...` with `=`, not `|=`, so a second
	//     pass would otherwise blow away the CUSTOM1/CUSTOM2 declarations
	//     finalize_terrain_surfaces just made and the rendering server would reject the chunk
	//     with "Invalid array format for surface"),
	//   - the surface count (everything above this index is fluid output).
	const int64_t terrain_collision_submesh_vertex_end = output.collision_surface.submesh_vertex_end;
	const int64_t terrain_collision_submesh_index_end = output.collision_surface.submesh_index_end;
	const uint32_t mesh_flags_before = output.mesh_flags;
	const size_t surfaces_before = output.surfaces.size();

	_fluid_transvoxel_mesher->build(output, input);

	// Restore the snapshot. The fluid surfaces inherit the hybrid mesher's vertex format
	// convention -- see the per-surface CUSTOM stream writes below.
	output.collision_surface.submesh_vertex_end = terrain_collision_submesh_vertex_end;
	output.collision_surface.submesh_index_end = terrain_collision_submesh_index_end;
	output.mesh_flags = mesh_flags_before;

	// Tag every newly-added Transvoxel surface with material id 1 (water). The vertex format
	// must match what finalize_terrain_surfaces / build_model_surface produce, because
	// output.mesh_flags applies to every surface in the chunk: a mismatched CUSTOM slot
	// triggers "Invalid array format for surface" from the rendering server.
	//
	// The inner fluid mesher runs in TEXTURES_NONE mode (see ctor) and therefore did not
	// write CUSTOM1. The terrain mesher's mode decides where the material id has to live:
	//   - texturing off: CUSTOM1 carries the material id (1 float).
	//   - texturing on:  CUSTOM1 must be a 2-float texture-data stream (inert zeros), and
	//                    the material id moves to CUSTOM2 (1 float).
	const bool texturing_enabled = (_transvoxel_mesher.is_valid()
			&& _transvoxel_mesher->get_texturing_mode() != VoxelMesherTransvoxel::TEXTURES_NONE);

	for (size_t i = surfaces_before; i < output.surfaces.size(); ++i) {
		VoxelMesher::Output::Surface &surface = output.surfaces[i];
		Array &arrays = surface.arrays;
		if (arrays.size() < Mesh::ARRAY_MAX) {
			continue;
		}
		const PackedVector3Array vertices = arrays[Mesh::ARRAY_VERTEX];
		if (vertices.size() == 0) {
			continue;
		}
		if (texturing_enabled) {
			arrays[Mesh::ARRAY_CUSTOM1] = build_model_inert_custom1_texturing(vertices.size());
			arrays[Mesh::ARRAY_CUSTOM2] = build_material_id_stream(vertices.size(), 1.0f);
		} else {
			arrays[Mesh::ARRAY_CUSTOM1] = build_material_id_stream(vertices.size(), 1.0f);
		}
		surface.material_index = 1;
	}
}

void VoxelMesherHybrid::finalize_terrain_surfaces(VoxelMesher::Output &output) const {
	ZN_PROFILE_SCOPE();

	const bool texturing_enabled = (_transvoxel_mesher.is_valid()
			&& _transvoxel_mesher->get_texturing_mode() != VoxelMesherTransvoxel::TEXTURES_NONE);

	// All surfaces that exist at this point were produced by the inner Transvoxel mesher
	// (model surfaces are appended later). Transvoxel already provides CUSTOM0; the hybrid
	// mesher additionally needs a per-vertex material id on every surface, so the terrain
	// renders with the same vertex format -- and the same single material_override -- as the
	// models.
	//
	// Without Transvoxel texturing, the material id goes into CUSTOM1 (1 float). With Transvoxel
	// texturing (Mixel4 / Single), CUSTOM1 is already owned by Transvoxel's 2-float texture data
	// (indices+weights), so the material id is written to CUSTOM2 instead.
	for (VoxelMesher::Output::Surface &surface : output.surfaces) {
		Array &arrays = surface.arrays;
		if (arrays.size() < Mesh::ARRAY_MAX) {
			continue;
		}
		const PackedVector3Array vertices = arrays[Mesh::ARRAY_VERTEX];
		if (vertices.size() == 0) {
			continue;
		}
		// Material id 0 is the smooth terrain (see get_material_by_index / the LOD shader).
		if (texturing_enabled) {
			arrays[Mesh::ARRAY_CUSTOM2] = build_material_id_stream(vertices.size(), 0.0f);
		} else {
			arrays[Mesh::ARRAY_CUSTOM1] = build_material_id_stream(vertices.size(), 0.0f);
		}
	}

	// Declare the format of all custom streams. build_mesh() applies one mesh_flags value to
	// every surface, so every surface -- terrain and models -- must carry the same set of
	// custom streams. Streams are declared unconditionally even if a chunk has models but no
	// smooth terrain (Transvoxel's mesh_flags = 0 in that case), because an undeclared CUSTOM0
	// field would make Godot reject the model surface ("Invalid array format for surface").
	output.mesh_flags |= texturing_enabled ? HYBRID_CUSTOM_MESH_FLAGS_TEXTURED
										   : HYBRID_CUSTOM_MESH_FLAGS_NO_TEXTURING;
}

void VoxelMesherHybrid::promote_terrain_collision(VoxelMesher::Output &output) const {
	ZN_PROFILE_SCOPE();

	VoxelMesher::Output::CollisionSurface &col = output.collision_surface;

	// Transvoxel reports its collider as a sub-range [0 .. submesh_*) of render surface 0
	// (the range excludes the LOD transition meshes). To be able to append model triangles,
	// that range is copied into explicit positions/indices and the submesh markers cleared,
	// so make_collision_shape_from_mesher_output() uses the explicit arrays instead.
	if (col.submesh_vertex_end < 0 || output.surfaces.size() == 0) {
		// Nothing to promote (e.g. empty terrain surface): leave collision_surface as-is.
		return;
	}

	const Array &arrays = output.surfaces[0].arrays;
	if (arrays.size() < Mesh::ARRAY_MAX) {
		return;
	}
	const PackedVector3Array vertices = arrays[Mesh::ARRAY_VERTEX];
	const PackedInt32Array indices = arrays[Mesh::ARRAY_INDEX];

	const int vertex_end = math::min(col.submesh_vertex_end, static_cast<int32_t>(vertices.size()));
	const int index_end = math::min(col.submesh_index_end, static_cast<int32_t>(indices.size()));

	col.positions.clear();
	col.indices.clear();
	col.positions.reserve(vertex_end);
	col.indices.reserve(index_end);

	for (int i = 0; i < vertex_end; ++i) {
		col.positions.push_back(to_vec3f(vertices[i]));
	}
	for (int i = 0; i < index_end; ++i) {
		col.indices.push_back(indices[i]);
	}

	// Switch to explicit-arrays mode so appended model triangles are included.
	col.submesh_vertex_end = -1;
	col.submesh_index_end = -1;
}

void VoxelMesherHybrid::build_model_surface(VoxelMesher::Output &output, const VoxelMesher::Input &input) {
	ZN_PROFILE_SCOPE();

	if (_library.is_null()) {
		// No models configured: hybrid mesher behaves like plain Transvoxel.
		return;
	}

	const VoxelBuffer &voxels = input.voxels;
	const VoxelBuffer::ChannelId type_channel = VoxelBuffer::CHANNEL_TYPE;

	if (voxels.is_uniform(type_channel) && voxels.get_voxel(Vector3i(), type_channel) == blocky::AIR_ID) {
		// No models anywhere in this block.
		return;
	}

	const Vector3i block_size = voxels.get_size();
	// Same padding convention as the blocky mesher: skip the padding border.
	const Vector3i min = Vector3iUtil::create(MODEL_PADDING);
	const Vector3i max = block_size - Vector3iUtil::create(MODEL_PADDING);
	if (min.x >= max.x || min.y >= max.y || min.z >= max.z) {
		return;
	}

	ModelMeshAccumulator acc;

	// TYPE-channel raw access: needed by generate_fluid_model (which indexes the buffer
	// by linear index for its 9-neighbor reads). VoxelBuffer uses ZXY layout (see
	// voxel_buffer.h:361), so y_jump=1, x_jump=size.y, z_jump=size.y*size.x.
	// If the channel is uniform-compressed (no raw bytes available), we fall back to the
	// per-voxel get_voxel path -- fluids cannot occur in that case anyway because a
	// uniform block has the same id everywhere (no level variation).
	const int y_jump = 1;
	const int x_jump = block_size.y;
	const int z_jump = block_size.y * block_size.x;

	Span<const uint8_t> raw_channel;
	const bool has_raw_channel = voxels.get_channel_as_bytes_read_only(type_channel, raw_channel);
	const VoxelBuffer::Depth channel_depth = voxels.get_channel_depth(type_channel);

	{
		// Baked data is the only library data safe for multithreaded access.
		RWLockRead lock(_library->get_baked_data_rw_lock());
		const blocky::BakedLibrary &baked = _library->get_baked_data();

		// Per-voxel meshing body, templated on the TYPE-channel element type so the same code
		// works for both DEPTH_8_BIT and DEPTH_16_BIT. type_buffer may be empty when the
		// channel is uniform-compressed -- in that case the fluid branch is skipped (fluids
		// require neighbor reads through the buffer index).
		const auto mesh_voxel = [&](auto type_buffer_tag, int x, int y, int z) {
			using TModelID = typename decltype(type_buffer_tag)::type;
			const Span<const TModelID> type_buffer = type_buffer_tag.span;

			const uint64_t voxel_id = voxels.get_voxel(Vector3i(x, y, z), type_channel);

			if (voxel_id == blocky::AIR_ID || !baked.has_model(voxel_id)) {
				return;
			}
			const blocky::BakedModel &baked_model = baked.models[voxel_id];
			if (baked_model.empty) {
				return;
			}
			const blocky::BakedModel::Model &model = baked_model.model;

			// Position relative to the (unpadded) mesh origin.
			const Vector3f voxel_offset(x - min.x, y - min.y, z - min.z);

			// Neighbor side-culling: a cube side is only baked if its neighbor voxel
			// does not occlude it. blocky::is_face_visible is the exported helper that
			// VoxelMesherBlocky itself uses -- it handles air, transparency and the
			// side-pattern test for non-cube models. The neighbor is read from the
			// TYPE channel; the padding border carries the neighbor chunk's values,
			// so this stays correct across chunk boundaries. Computed once per voxel
			// since visibility depends on the neighbor, not on the surface/material.
			//
			// model.empty_sides_mask marks sides that the model itself never renders --
			// fluid models in particular mark SIDE_POSITIVE_Y as empty because their top
			// is rendered as an inner surface (fluid_top_surface) by generate_fluid_model,
			// not as a side surface. Without this skip, the side loop below would emit
			// stale TLS-cached top side geometry from a previous fluid voxel.
			uint8_t visible_sides_mask = 0;
			for (unsigned int side = 0; side < Cube::SIDE_COUNT; ++side) {
				if ((model.empty_sides_mask & (1 << side)) != 0) {
					continue;
				}
				const Vector3i npos = Vector3i(x, y, z) + Cube::g_side_normals[side];
				const uint64_t neighbor_id = voxels.get_voxel(npos, type_channel);
				if (blocky::is_face_visible(baked, baked_model, neighbor_id, side)) {
					visible_sides_mask |= (1 << side);
				}
			}

			// Fluid path: VoxelBlockyModelFluid baked models carry a fluid_index != NULL.
			// generate_fluid_model overrides the model's static geometry with dynamic top/side
			// surfaces whose corner heights are interpolated from the 9 horizontal neighbors,
			// matching VoxelMesherBlocky's behavior. Requires raw type-buffer access.
			Span<const blocky::BakedModel::Surface> model_surfaces = to_span(model.surfaces);
			const FixedArray<FixedArray<blocky::BakedModel::SideSurface, blocky::MAX_SURFACES>, Cube::SIDE_COUNT>
					*model_sides_surfaces = &model.sides_surfaces;
			unsigned int surface_count = model.surface_count;

			if (baked_model.fluid_index != blocky::NULL_FLUID_INDEX && type_buffer.size() > 0) {
				const int voxel_index = y * y_jump + x * x_jump + z * z_jump;
				if (!blocky::generate_fluid_model(
							baked_model,
							type_buffer,
							voxel_index,
							y_jump,
							x_jump,
							z_jump,
							visible_sides_mask,
							baked,
							model_surfaces,
							model_sides_surfaces
					)) {
					// Voxel completely covered by same-fluid neighbors: nothing to mesh.
					return;
				}
				surface_count = 1;
			}

			for (unsigned int s = 0; s < surface_count; ++s) {
				// Inner geometry of the model (or fluid top surface). Each surface has its own
				// material id; geometry is grouped by that id so it becomes its own mesh surface.
				const blocky::BakedModel::Surface &surface = model_surfaces[s];
				acc[surface.material_id].append_geometry(
						surface.positions, surface.normals, surface.uvs, surface.indices, voxel_offset
				);
				// Cube sides: only the ones not occluded by a neighbor. Side surfaces
				// share the material id of their parent model surface.
				for (unsigned int side = 0; side < Cube::SIDE_COUNT; ++side) {
					if ((visible_sides_mask & (1 << side)) == 0) {
						continue;
					}
					const blocky::BakedModel::SideSurface &side_surface = (*model_sides_surfaces)[side][s];
					acc[surface.material_id].append_geometry(
							side_surface.positions,
							StdVector<Vector3f>(),
							side_surface.uvs,
							side_surface.indices,
							voxel_offset
					);
				}
			}
		};

		// Tag struct to pass both the element type and the typed span to the generic lambda.
		// (Lambdas can't be explicit-template-parameter generic until C++20.)
		struct Tag8 { using type = uint8_t; Span<const uint8_t> span; };
		struct Tag16 { using type = uint16_t; Span<const uint16_t> span; };

		// Dispatch on channel depth once outside the loop so the inner loop stays tight.
		// When raw channel access is unavailable (uniform compression), we still iterate with
		// an empty span -- the fluid branch is skipped, the rest of the meshing works via
		// voxels.get_voxel() as before.
		if (has_raw_channel && channel_depth == VoxelBuffer::DEPTH_8_BIT) {
			const Tag8 tag{ raw_channel };
			for (int z = min.z; z < max.z; ++z) {
				for (int x = min.x; x < max.x; ++x) {
					for (int y = min.y; y < max.y; ++y) {
						mesh_voxel(tag, x, y, z);
					}
				}
			}
		} else if (has_raw_channel && channel_depth == VoxelBuffer::DEPTH_16_BIT) {
			const Tag16 tag{ raw_channel.reinterpret_cast_to<const uint16_t>() };
			for (int z = min.z; z < max.z; ++z) {
				for (int x = min.x; x < max.x; ++x) {
					for (int y = min.y; y < max.y; ++y) {
						mesh_voxel(tag, x, y, z);
					}
				}
			}
		} else {
			// Fallback: no raw access or unsupported depth -- meshing still works, but fluids
			// rendered as plain cubes (their static fallback). Empty span signals this.
			const Tag16 tag{ Span<const uint16_t>() };
			for (int z = min.z; z < max.z; ++z) {
				for (int x = min.x; x < max.x; ++x) {
					for (int y = min.y; y < max.y; ++y) {
						mesh_voxel(tag, x, y, z);
					}
				}
			}
		}
	}

	const float lod_scale = (input.lod_index > 0) ? static_cast<float>(1 << input.lod_index) : 1.0f;

	// Vertex-stream layout depends on Transvoxel's texturing mode (see finalize_terrain_surfaces).
	// Evaluated once because every surface in this loop uses the same layout.
	const bool texturing_enabled = (_transvoxel_mesher.is_valid()
			&& _transvoxel_mesher->get_texturing_mode() != VoxelMesherTransvoxel::TEXTURES_NONE);

	// Emit one mesh surface per material id. Surface material index is `2 + material_id`,
	// because index 0 is reserved for the smooth terrain surface and index 1 for the smooth
	// water surface (see get_material_by_index).
	for (auto &entry : acc) {
		const uint32_t material_id = entry.first;
		SurfaceArrays &sa = entry.second;
		if (sa.indices.size() == 0) {
			continue;
		}

		// Apply LOD scaling, same approach as VoxelMesherBlocky: positions are scaled up so a
		// model occupies the larger world-space of a coarse-LOD voxel.
		if (lod_scale != 1.0f) {
			for (Vector3f &p : sa.positions) {
				p = p * lod_scale;
			}
		}

		PackedVector3Array gd_positions;
		PackedVector3Array gd_normals;
		PackedVector2Array gd_uvs;
		PackedInt32Array gd_indices;
		copy_to(gd_positions, to_span_const(sa.positions));
		copy_to(gd_normals, to_span_const(sa.normals));
		copy_to(gd_uvs, to_span_const(sa.uvs));
		copy_to(gd_indices, to_span_const(sa.indices));

		// Surface material index: index 0 is reserved for the smooth terrain and index 1 for
		// the smooth water surface, so model material id N maps to surface material index
		// 2 + N (see get_material_by_index).
		const uint16_t surface_material_index = static_cast<uint16_t>(2 + material_id);

		// CUSTOM streams give the model surface the same vertex format as the smooth surface,
		// so one material_override (the default LOD shader) renders them all. Layout depends on
		// whether Transvoxel texturing is enabled (see finalize_terrain_surfaces for the matching
		// scheme on terrain surfaces):
		//   - texturing off: CUSTOM1 carries the material id directly.
		//   - texturing on:  CUSTOM1 carries an inert 2-float texture stream (zeros), and the
		//                    material id moves to CUSTOM2 so it stays separate from texture data.
		Array arrays;
		arrays.resize(Mesh::ARRAY_MAX);
		arrays[Mesh::ARRAY_VERTEX] = gd_positions;
		arrays[Mesh::ARRAY_NORMAL] = gd_normals;
		arrays[Mesh::ARRAY_TEX_UV] = gd_uvs;
		arrays[Mesh::ARRAY_INDEX] = gd_indices;
		arrays[Mesh::ARRAY_CUSTOM0] = build_model_custom0(sa.positions);
		if (texturing_enabled) {
			arrays[Mesh::ARRAY_CUSTOM1] = build_model_inert_custom1_texturing(gd_positions.size());
			arrays[Mesh::ARRAY_CUSTOM2] =
					build_material_id_stream(gd_positions.size(), static_cast<float>(surface_material_index));
		} else {
			arrays[Mesh::ARRAY_CUSTOM1] =
					build_material_id_stream(gd_positions.size(), static_cast<float>(surface_material_index));
		}

		output.surfaces.push_back({ arrays, surface_material_index });

		// Append the same model geometry to the collision surface so models are solid.
		// promote_terrain_collision() has already placed the terrain geometry there.
		append_to_collision_surface(output.collision_surface, sa);
	}
}

void VoxelMesherHybrid::_bind_methods() {
	using Self = VoxelMesherHybrid;

	ClassDB::bind_method(D_METHOD("set_library", "library"), &Self::set_library);
	ClassDB::bind_method(D_METHOD("get_library"), &Self::get_library);

	ClassDB::bind_method(D_METHOD("set_terrain_material", "material"), &Self::set_terrain_material);
	ClassDB::bind_method(D_METHOD("get_terrain_material"), &Self::get_terrain_material);

	ClassDB::bind_method(D_METHOD("set_water_material", "material"), &Self::set_water_material);
	ClassDB::bind_method(D_METHOD("get_water_material"), &Self::get_water_material);

	ClassDB::bind_method(D_METHOD("set_model_material", "material"), &Self::set_model_material);
	ClassDB::bind_method(D_METHOD("get_model_material"), &Self::get_model_material);

	ClassDB::bind_method(D_METHOD("set_terrain_color", "color"), &Self::set_terrain_color);
	ClassDB::bind_method(D_METHOD("get_terrain_color"), &Self::get_terrain_color);

	ClassDB::bind_method(D_METHOD("set_model_colors", "colors"), &Self::set_model_colors);
	ClassDB::bind_method(D_METHOD("get_model_colors"), &Self::get_model_colors);

	ClassDB::bind_method(
			D_METHOD("set_flatten_terrain_under_models", "enabled"), &Self::set_flatten_terrain_under_models
	);
	ClassDB::bind_method(
			D_METHOD("get_flatten_terrain_under_models"), &Self::get_flatten_terrain_under_models
	);

	ClassDB::bind_method(D_METHOD("set_texturing_mode", "mode"), &Self::set_texturing_mode);
	ClassDB::bind_method(D_METHOD("get_texturing_mode"), &Self::get_texturing_mode);

	ClassDB::bind_method(
			D_METHOD("set_textures_ignore_air_voxels", "enabled"), &Self::set_textures_ignore_air_voxels
	);
	ClassDB::bind_method(D_METHOD("get_textures_ignore_air_voxels"), &Self::get_textures_ignore_air_voxels);

	ADD_PROPERTY(
			PropertyInfo(
					Variant::OBJECT,
					"library",
					PROPERTY_HINT_RESOURCE_TYPE,
					VoxelBlockyLibraryBase::get_class_static()
			),
			"set_library",
			"get_library"
	);
	ADD_PROPERTY(
			PropertyInfo(
					Variant::OBJECT, "terrain_material", PROPERTY_HINT_RESOURCE_TYPE, "Material"
			),
			"set_terrain_material",
			"get_terrain_material"
	);
	ADD_PROPERTY(
			PropertyInfo(
					Variant::OBJECT, "water_material", PROPERTY_HINT_RESOURCE_TYPE, "Material"
			),
			"set_water_material",
			"get_water_material"
	);
	ADD_PROPERTY(
			PropertyInfo(
					Variant::OBJECT, "model_material", PROPERTY_HINT_RESOURCE_TYPE, "Material"
			),
			"set_model_material",
			"get_model_material"
	);
	ADD_PROPERTY(
			PropertyInfo(Variant::COLOR, "terrain_color"),
			"set_terrain_color",
			"get_terrain_color"
	);
	ADD_PROPERTY(
			PropertyInfo(Variant::PACKED_COLOR_ARRAY, "model_colors"),
			"set_model_colors",
			"get_model_colors"
	);
	ADD_PROPERTY(
			PropertyInfo(Variant::BOOL, "flatten_terrain_under_models"),
			"set_flatten_terrain_under_models",
			"get_flatten_terrain_under_models"
	);

	// Texturing mode is exposed as an enum hint with the same values as VoxelMesherTransvoxel
	// (which is where the work happens). The hybrid mesher itself only re-routes the per-vertex
	// material id from CUSTOM1 to CUSTOM2 when the mode is non-NONE, so Transvoxel's texture
	// data can live in CUSTOM1.
	ADD_PROPERTY(
			PropertyInfo(
					Variant::INT,
					"texturing_mode",
					PROPERTY_HINT_ENUM,
					"None,Mixel4 S4,Single S4"
			),
			"set_texturing_mode",
			"get_texturing_mode"
	);
}

} // namespace zylann::voxel
