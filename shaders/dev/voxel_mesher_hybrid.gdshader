shader_type spatial;

// Default LOD material for VoxelMesherHybrid.
//
// It does two jobs in one material, because the terrain applies a single
// material_override to the whole MeshInstance (all surfaces at once):
//
//  1. Transvoxel LOD transition handling (identical to transvoxel_minimal):
//     reads CUSTOM0 to move/cull the transition vertices of surface 0.
//  2. Per-voxel material colour: reads a material id from CUSTOM1.r and picks
//     a colour from u_material_colors. Id 0 is the smooth terrain, ids >= 1 are
//     the blocky model materials (id = 1 + library material id).
//
// Both the smooth surface and the model surfaces carry CUSTOM0 and CUSTOM1, so
// every surface has the same vertex format and this one shader renders them all.

// From Voxel Tools API
uniform int u_transition_mask;

// Colour table indexed by the per-vertex material id in CUSTOM1.r.
// Index 0 = smooth terrain, index 1.. = model materials.
const int MATERIAL_COLOR_COUNT = 8;
uniform vec4 u_material_colors[MATERIAL_COLOR_COUNT];
uniform float u_roughness = 1.0;

varying flat int v_material_id;

float get_transvoxel_secondary_factor(int idata) {
	int transition_mask = u_transition_mask & 0xff;

	int cell_border_mask = idata & 63; // Which sides the cell is touching
	int vertex_border_mask = (idata >> 8) & 63; // Which sides the vertex is touching
	// If the vertex is near a side where there is a low-resolution neighbor,
	// move it to secondary position
	int m = transition_mask & cell_border_mask;
	float t = float(m != 0);
	// If the vertex lies on one or more sides, and at least one side has no low-resolution neighbor,
	// don't move the vertex.
	t *= float((vertex_border_mask & ~transition_mask) == 0);

	return t;
}

vec3 get_transvoxel_position(vec3 vertex_pos, vec4 fdata) {
	int idata = floatBitsToInt(fdata.a);

	// Move vertices to smooth transitions
	float secondary_factor = get_transvoxel_secondary_factor(idata);
	vec3 secondary_position = fdata.xyz;
	vec3 pos = mix(vertex_pos, secondary_position, secondary_factor);

	// If the mesh combines transitions and the vertex belongs to a transition,
	// when that transition isn't active we change the position of the vertices so
	// all triangles will be degenerate and won't be visible.
	int itransition = (idata >> 16) & 0xff; // Is the vertex on a transition mesh?
	float transition_cull = float(itransition == 0 || (itransition & u_transition_mask) != 0);
	pos *= transition_cull;

	return pos;
}

void vertex() {
	VERTEX = get_transvoxel_position(VERTEX, CUSTOM0);
	// CUSTOM1.r holds the material id; clamp into the colour table range.
	v_material_id = clamp(int(CUSTOM1.r + 0.5), 0, MATERIAL_COLOR_COUNT - 1);
}

void fragment() {
	ALBEDO = u_material_colors[v_material_id].rgb;
	ROUGHNESS = u_roughness;
}
