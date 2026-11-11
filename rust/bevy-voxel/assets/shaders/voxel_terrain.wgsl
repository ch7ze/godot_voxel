// Voxel terrain material.
//
// Ported from `shaders/voxel_mesher_hybrid_shader.h` of the C++ module. Two
// jobs are performed, exactly as in the original.
//   1. Transvoxel LOD stitching, the vertex is moved to its secondary position
//      on sides facing a coarser chunk, and transition triangles of sides that
//      are not stitched are culled.
//   2. Per-vertex colour lookup from the material id.
// Source: shaders/voxel_mesher_hybrid_shader.h:16-50

#import bevy_pbr::{
    mesh_functions,
    view_transformations::position_world_to_clip,
    forward_io::VertexOutput,
}

const MATERIAL_COLOR_COUNT: u32 = 8u;

struct VoxelTerrainMaterial {
    // One colour per surface material index, id 0 is the smooth terrain,
    // id 1 the water, ids 2 and above the library materials.
    // Source: meshers/hybrid/voxel_mesher_hybrid.h:88-98
    material_colors: array<vec4<f32>, 8>,
    // Bit mask of the sides stitched to a coarser neighbour chunk.
    // Source: shaders/voxel_mesher_hybrid_shader.h:8
    transition_mask: u32,
    roughness: f32,
    _padding: vec2<f32>,
}

@group(2) @binding(0) var<uniform> material: VoxelTerrainMaterial;

struct Vertex {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    // Transvoxel LOD data, xyz the secondary position, w the packed masks
    @location(8) lod_data: vec4<f32>,
    // Per-vertex material id
    @location(10) material_id: f32,
}

struct VoxelVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) @interpolate(flat) material_id: u32,
}

// Blend factor between the primary and the secondary vertex position.
//
// The vertex moves only when its cell touches a stitched side and the vertex
// itself touches no side that is left unstitched.
// Source: shaders/voxel_mesher_hybrid_shader.h:16-26
fn get_transvoxel_secondary_factor(idata: u32) -> f32 {
    let transition_mask = material.transition_mask & 0xffu;

    let cell_border_mask = idata & 63u;
    let vertex_border_mask = (idata >> 8u) & 63u;
    let m = transition_mask & cell_border_mask;
    var t = f32(m != 0u);
    t *= f32((vertex_border_mask & ~transition_mask) == 0u);

    return t;
}

// Final vertex position, stitched and with unstitched transition triangles
// collapsed to the origin so they are not visible.
// Source: shaders/voxel_mesher_hybrid_shader.h:28-40
fn get_transvoxel_position(vertex_pos: vec3<f32>, fdata: vec4<f32>) -> vec3<f32> {
    let idata = bitcast<u32>(fdata.w);

    let secondary_factor = get_transvoxel_secondary_factor(idata);
    let secondary_position = fdata.xyz;
    var pos = mix(vertex_pos, secondary_position, secondary_factor);

    let itransition = (idata >> 16u) & 0xffu;
    let transition_cull = f32(itransition == 0u || (itransition & material.transition_mask) != 0u);
    pos *= transition_cull;

    return pos;
}

@vertex
fn vertex(vertex: Vertex) -> VoxelVertexOutput {
    var out: VoxelVertexOutput;

    let local_position = get_transvoxel_position(vertex.position, vertex.lod_data);

    let world_from_local = mesh_functions::get_world_from_local(vertex.instance_index);
    out.world_position = mesh_functions::mesh_position_local_to_world(
        world_from_local,
        vec4<f32>(local_position, 1.0),
    );
    out.position = position_world_to_clip(out.world_position.xyz);
    out.world_normal = mesh_functions::mesh_normal_local_to_world(
        vertex.normal,
        vertex.instance_index,
    );

    // Source: shaders/voxel_mesher_hybrid_shader.h:44
    out.material_id = clamp(u32(vertex.material_id + 0.5), 0u, MATERIAL_COLOR_COUNT - 1u);

    return out;
}

@fragment
fn fragment(in: VoxelVertexOutput) -> @location(0) vec4<f32> {
    // Source: shaders/voxel_mesher_hybrid_shader.h:47-50
    let albedo = material.material_colors[in.material_id].rgb;

    // Simple directional shading so the surface reads as three-dimensional
    // without pulling in the whole PBR pipeline
    let light_dir = normalize(vec3<f32>(0.4, 1.0, 0.3));
    let ndl = max(dot(normalize(in.world_normal), light_dir), 0.0);
    let ambient = 0.35;
    let lit = albedo * (ambient + (1.0 - ambient) * ndl);

    return vec4<f32>(lit, material.material_colors[in.material_id].a);
}
