//! Conversion of the engine-neutral mesh output into `bevy_mesh::Mesh`.
//!
//! The C++ mesher writes its extra vertex streams into Godot's fixed
//! ARRAY_CUSTOM0..2 slots, whose layout changes with the texturing mode
//! (`meshers/hybrid/voxel_mesher_hybrid.cpp:83-112`). Bevy takes freely chosen
//! attribute ids from `FIRST_AVAILABLE_CUSTOM_ATTRIBUTE` upwards, verified as 8
//! in `bevy_mesh-0.20.1/src/mesh.rs:407`, so every stream gets its own
//! attribute and the mode no longer moves data between slots.

use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, Mesh, MeshVertexAttribute, PrimitiveTopology, VertexFormat};
use voxel_mesher::hybrid::output::{MeshOutput, Surface};

/// Transvoxel LOD attributes, the C++ CUSTOM0 stream. XYZ hold the secondary
/// position, W the packed masks as a bit-cast integer.
/// Source: meshers/hybrid/voxel_mesher_hybrid.cpp:85-88,
/// shaders/voxel_mesher_hybrid_shader.h:28-40
pub const ATTRIBUTE_LOD_DATA: MeshVertexAttribute =
    MeshVertexAttribute::new("Vertex_VoxelLodData", 8, VertexFormat::Float32x4);

/// Packed Mixel4 material indices and weights, the C++ CUSTOM1 stream in
/// texturing mode.
/// Source: meshers/hybrid/voxel_mesher_hybrid.cpp:89-93
pub const ATTRIBUTE_TEXTURING_DATA: MeshVertexAttribute =
    MeshVertexAttribute::new("Vertex_VoxelTexturingData", 9, VertexFormat::Float32x2);

/// Per-vertex material id, the C++ CUSTOM1 or CUSTOM2 stream depending on the
/// texturing mode.
/// Source: meshers/hybrid/voxel_mesher_hybrid.cpp:94-98
pub const ATTRIBUTE_MATERIAL_ID: MeshVertexAttribute =
    MeshVertexAttribute::new("Vertex_VoxelMaterialId", 10, VertexFormat::Float32);

/// Packs the three mask bytes the shader reads out of the W component.
///
/// The layout matches the C++ shader, the cell border mask in bits 0..5, the
/// vertex border mask in bits 8..13 and the transition mask in bits 16..23.
/// Source: shaders/voxel_mesher_hybrid_shader.h:16-26, :35
#[inline]
pub fn pack_lod_masks(cell_border_mask: u8, vertex_border_mask: u8, transition: u8) -> f32 {
    let packed = (cell_border_mask as u32)
        | ((vertex_border_mask as u32) << 8)
        | ((transition as u32) << 16);
    f32::from_bits(packed)
}

/// Builds a Bevy mesh from one surface.
///
/// Returns `None` for a surface without triangles.
pub fn surface_to_mesh(surface: &Surface, asset_usage: RenderAssetUsages) -> Option<Mesh> {
    if surface.is_empty() {
        return None;
    }

    let vertex_count = surface.positions.len();
    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, asset_usage);

    let positions: Vec<[f32; 3]> = surface.positions.iter().map(|p| p.to_array()).collect();
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);

    if surface.normals.len() == vertex_count {
        let normals: Vec<[f32; 3]> = surface.normals.iter().map(|n| n.to_array()).collect();
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    }

    match &surface.uvs {
        Some(uvs) if uvs.len() == vertex_count => {
            let uvs: Vec<[f32; 2]> = uvs.iter().map(|uv| uv.to_array()).collect();
            mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
        }
        _ => {}
    }

    // The LOD attribute must exist on every surface of a chunk, the shader
    // reads it unconditionally. Model surfaces carry an inert value, which the
    // mesher already filled in.
    // Source: meshers/hybrid/voxel_mesher_hybrid.cpp:114-120, :695-701
    let lod_data: Vec<[f32; 4]> = match &surface.lod_data {
        Some(lod) if lod.len() == vertex_count => lod
            .iter()
            .map(|a| {
                let s = a.secondary_position;
                [
                    s.x,
                    s.y,
                    s.z,
                    pack_lod_masks(a.cell_border_mask, a.vertex_border_mask, a.transition),
                ]
            })
            .collect(),
        _ => surface
            .positions
            .iter()
            .map(|p| [p.x, p.y, p.z, pack_lod_masks(0, 0, 0)])
            .collect(),
    };
    mesh.insert_attribute(ATTRIBUTE_LOD_DATA, lod_data);

    let material_ids: Vec<f32> = if surface.material_id_stream.len() == vertex_count {
        surface.material_id_stream.clone()
    } else {
        vec![surface.material_index as f32; vertex_count]
    };
    mesh.insert_attribute(ATTRIBUTE_MATERIAL_ID, material_ids);

    match &surface.texturing_data {
        Some(data) if data.len() == vertex_count => {
            let values: Vec<[f32; 2]> = data.iter().map(|v| v.to_array()).collect();
            mesh.insert_attribute(ATTRIBUTE_TEXTURING_DATA, values);
        }
        _ => {}
    }

    mesh.insert_indices(Indices::U32(surface.indices.clone()));

    Some(mesh)
}

/// One mesh plus the material index it is rendered with.
#[derive(Debug)]
pub struct ChunkMesh {
    pub mesh: Mesh,
    pub material_index: u16,
}

/// Builds one Bevy mesh per non-empty surface of a chunk.
pub fn output_to_meshes(out: &MeshOutput, asset_usage: RenderAssetUsages) -> Vec<ChunkMesh> {
    let mut meshes = Vec::new();
    for surface in &out.surfaces {
        if let Some(mesh) = surface_to_mesh(surface, asset_usage) {
            meshes.push(ChunkMesh {
                mesh,
                material_index: surface.material_index,
            });
        }
    }
    meshes
}

/// Builds the transition meshes of a chunk, grouped per side.
pub fn output_to_transition_meshes(
    out: &MeshOutput,
    asset_usage: RenderAssetUsages,
) -> [Vec<ChunkMesh>; 6] {
    let mut per_side: [Vec<ChunkMesh>; 6] = Default::default();
    for (side, surfaces) in out.transition_surfaces.iter().enumerate() {
        for surface in surfaces {
            if let Some(mesh) = surface_to_mesh(surface, asset_usage) {
                per_side[side].push(ChunkMesh {
                    mesh,
                    material_index: surface.material_index,
                });
            }
        }
    }
    per_side
}

/// Splits a chunk's meshes into terrain and water, as `VoxelLodTerrain` does.
///
/// The terrain is rendered opaque, the water separately and transparent, so an
/// opaque material override does not cover the water surface.
/// Source: meshers/hybrid/voxel_mesher_hybrid.h:48-56, meshers/voxel_mesher.h:115-123
pub fn split_terrain_and_water(meshes: Vec<ChunkMesh>) -> (Vec<ChunkMesh>, Vec<ChunkMesh>) {
    let water_index = voxel_mesher::WATER_MATERIAL_INDEX;
    meshes
        .into_iter()
        .partition(|m| m.material_index != water_index)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::math::{Vec2, Vec3};
    use bevy::mesh::VertexAttributeValues;
    use voxel_mesher::hybrid::output::{LodAttrib, MODEL_MATERIAL_INDEX_BASE};
    use voxel_mesher::{TERRAIN_MATERIAL_INDEX, WATER_MATERIAL_INDEX};

    fn usage() -> RenderAssetUsages {
        RenderAssetUsages::RENDER_WORLD
    }

    /// Surface of one triangle.
    fn triangle_surface(material_index: u16) -> Surface {
        let positions = vec![Vec3::ZERO, Vec3::X, Vec3::Y];
        let mut surface = Surface {
            positions: positions.clone(),
            normals: vec![Vec3::Z; 3],
            uvs: None,
            indices: vec![0, 1, 2],
            lod_data: Some(vec![LodAttrib::default(); 3]),
            texturing_data: None,
            material_id_stream: Vec::new(),
            material_index,
        };
        surface.set_material_id_stream(material_index as f32);
        surface
    }

    #[test]
    fn custom_attribute_ids_start_at_the_first_available_slot() {
        // Verified in bevy_mesh-0.20.1/src/mesh.rs:407
        assert_eq!(Mesh::FIRST_AVAILABLE_CUSTOM_ATTRIBUTE, 8);
        // Every custom attribute must sit at or above that bound and be distinct
        let ids = [
            ATTRIBUTE_LOD_DATA.id,
            ATTRIBUTE_TEXTURING_DATA.id,
            ATTRIBUTE_MATERIAL_ID.id,
        ];
        let mut unique = ids.to_vec();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), 3, "attribute ids collide");
    }

    #[test]
    fn packing_masks_round_trips_through_the_bit_cast() {
        let packed = pack_lod_masks(0b101010, 0b010101, 0b00001000);
        let bits = packed.to_bits();
        assert_eq!(bits & 63, 0b101010);
        assert_eq!((bits >> 8) & 63, 0b010101);
        assert_eq!((bits >> 16) & 0xff, 0b00001000);
    }

    #[test]
    fn packing_zero_masks_yields_zero_bits() {
        assert_eq!(pack_lod_masks(0, 0, 0).to_bits(), 0);
    }

    #[test]
    fn an_empty_surface_produces_no_mesh() {
        let surface = Surface::default();
        assert!(surface_to_mesh(&surface, usage()).is_none());
    }

    #[test]
    fn a_triangle_surface_produces_a_triangle_list() {
        let surface = triangle_surface(TERRAIN_MATERIAL_INDEX);
        let mesh = surface_to_mesh(&surface, usage()).unwrap();
        assert_eq!(mesh.primitive_topology(), PrimitiveTopology::TriangleList);
        assert_eq!(mesh.count_vertices(), 3);
        match mesh.indices() {
            Some(Indices::U32(indices)) => assert_eq!(indices, &vec![0u32, 1, 2]),
            other => panic!("unexpected indices {other:?}"),
        }
    }

    #[test]
    fn positions_and_normals_are_transferred() {
        let surface = triangle_surface(TERRAIN_MATERIAL_INDEX);
        let mesh = surface_to_mesh(&surface, usage()).unwrap();
        match mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
            Some(VertexAttributeValues::Float32x3(values)) => {
                assert_eq!(values[1], [1.0, 0.0, 0.0]);
            }
            other => panic!("unexpected positions {other:?}"),
        }
        match mesh.attribute(Mesh::ATTRIBUTE_NORMAL) {
            Some(VertexAttributeValues::Float32x3(values)) => {
                assert_eq!(values.len(), 3);
                assert_eq!(values[0], [0.0, 0.0, 1.0]);
            }
            other => panic!("unexpected normals {other:?}"),
        }
    }

    #[test]
    fn uvs_are_transferred_when_present() {
        let mut surface = triangle_surface(MODEL_MATERIAL_INDEX_BASE);
        surface.uvs = Some(vec![Vec2::new(0.25, 0.75); 3]);
        let mesh = surface_to_mesh(&surface, usage()).unwrap();
        match mesh.attribute(Mesh::ATTRIBUTE_UV_0) {
            Some(VertexAttributeValues::Float32x2(values)) => {
                assert_eq!(values[0], [0.25, 0.75]);
            }
            other => panic!("unexpected uvs {other:?}"),
        }
    }

    #[test]
    fn a_surface_without_uvs_gets_no_uv_attribute() {
        let surface = triangle_surface(TERRAIN_MATERIAL_INDEX);
        let mesh = surface_to_mesh(&surface, usage()).unwrap();
        assert!(mesh.attribute(Mesh::ATTRIBUTE_UV_0).is_none());
    }

    #[test]
    fn the_lod_attribute_is_always_present() {
        // Even a surface carrying no LOD data gets an inert stream, the shader
        // reads the attribute on every surface
        let mut surface = triangle_surface(MODEL_MATERIAL_INDEX_BASE);
        surface.lod_data = None;
        let mesh = surface_to_mesh(&surface, usage()).unwrap();
        match mesh.attribute(ATTRIBUTE_LOD_DATA) {
            Some(VertexAttributeValues::Float32x4(values)) => {
                assert_eq!(values.len(), 3);
                // The secondary position equals the primary one
                assert_eq!(values[1], [1.0, 0.0, 0.0, 0.0]);
            }
            other => panic!("unexpected lod data {other:?}"),
        }
    }

    #[test]
    fn lod_masks_reach_the_attribute() {
        let mut surface = triangle_surface(TERRAIN_MATERIAL_INDEX);
        surface.lod_data = Some(vec![
            LodAttrib {
                secondary_position: Vec3::splat(2.0),
                cell_border_mask: 0b000011,
                vertex_border_mask: 0b000001,
                transition: 0b00000010,
            };
            3
        ]);
        let mesh = surface_to_mesh(&surface, usage()).unwrap();
        match mesh.attribute(ATTRIBUTE_LOD_DATA) {
            Some(VertexAttributeValues::Float32x4(values)) => {
                assert_eq!(&values[0][..3], &[2.0, 2.0, 2.0]);
                let bits = values[0][3].to_bits();
                assert_eq!(bits & 63, 0b000011);
                assert_eq!((bits >> 8) & 63, 0b000001);
                assert_eq!((bits >> 16) & 0xff, 0b00000010);
            }
            other => panic!("unexpected lod data {other:?}"),
        }
    }

    #[test]
    fn the_material_id_attribute_carries_the_surface_index() {
        let surface = triangle_surface(WATER_MATERIAL_INDEX);
        let mesh = surface_to_mesh(&surface, usage()).unwrap();
        match mesh.attribute(ATTRIBUTE_MATERIAL_ID) {
            Some(VertexAttributeValues::Float32(values)) => {
                assert_eq!(values, &vec![1.0f32; 3]);
            }
            other => panic!("unexpected material ids {other:?}"),
        }
    }

    #[test]
    fn a_missing_material_stream_falls_back_to_the_surface_index() {
        let mut surface = triangle_surface(MODEL_MATERIAL_INDEX_BASE);
        surface.material_id_stream.clear();
        let mesh = surface_to_mesh(&surface, usage()).unwrap();
        match mesh.attribute(ATTRIBUTE_MATERIAL_ID) {
            Some(VertexAttributeValues::Float32(values)) => {
                assert_eq!(values, &vec![2.0f32; 3]);
            }
            other => panic!("unexpected material ids {other:?}"),
        }
    }

    #[test]
    fn texturing_data_reaches_its_own_attribute() {
        let mut surface = triangle_surface(TERRAIN_MATERIAL_INDEX);
        surface.texturing_data = Some(vec![Vec2::new(1.5, 2.5); 3]);
        let mesh = surface_to_mesh(&surface, usage()).unwrap();
        match mesh.attribute(ATTRIBUTE_TEXTURING_DATA) {
            Some(VertexAttributeValues::Float32x2(values)) => {
                assert_eq!(values[0], [1.5, 2.5]);
            }
            other => panic!("unexpected texturing data {other:?}"),
        }
    }

    #[test]
    fn a_surface_without_texturing_gets_no_texturing_attribute() {
        let surface = triangle_surface(TERRAIN_MATERIAL_INDEX);
        let mesh = surface_to_mesh(&surface, usage()).unwrap();
        assert!(mesh.attribute(ATTRIBUTE_TEXTURING_DATA).is_none());
    }

    #[test]
    fn an_output_yields_one_mesh_per_non_empty_surface() {
        let out = MeshOutput {
            surfaces: vec![
                triangle_surface(TERRAIN_MATERIAL_INDEX),
                Surface::default(),
                triangle_surface(WATER_MATERIAL_INDEX),
            ],
            ..MeshOutput::default()
        };
        let meshes = output_to_meshes(&out, usage());
        assert_eq!(meshes.len(), 2);
        assert_eq!(meshes[0].material_index, TERRAIN_MATERIAL_INDEX);
        assert_eq!(meshes[1].material_index, WATER_MATERIAL_INDEX);
    }

    #[test]
    fn transition_meshes_stay_grouped_per_side() {
        let mut out = MeshOutput::default();
        out.transition_surfaces[0].push(triangle_surface(TERRAIN_MATERIAL_INDEX));
        out.transition_surfaces[3].push(triangle_surface(TERRAIN_MATERIAL_INDEX));
        let per_side = output_to_transition_meshes(&out, usage());
        assert_eq!(per_side[0].len(), 1);
        assert_eq!(per_side[3].len(), 1);
        for side in [1, 2, 4, 5] {
            assert!(per_side[side].is_empty(), "side {side} not empty");
        }
    }

    #[test]
    fn terrain_and_water_are_split_by_material_index() {
        let meshes = output_to_meshes(
            &MeshOutput {
                surfaces: vec![
                    triangle_surface(TERRAIN_MATERIAL_INDEX),
                    triangle_surface(WATER_MATERIAL_INDEX),
                    triangle_surface(MODEL_MATERIAL_INDEX_BASE),
                ],
                ..MeshOutput::default()
            },
            usage(),
        );
        let (terrain, water) = split_terrain_and_water(meshes);
        assert_eq!(terrain.len(), 2);
        assert_eq!(water.len(), 1);
        assert_eq!(water[0].material_index, WATER_MATERIAL_INDEX);
    }
}
