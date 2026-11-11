//! Minimal terrain viewer, wiring the chunk layer to the material.
//!
//! Run from the crate directory so the asset folder is found:
//! `cargo run -p bevy-voxel --example terrain`

use bevy::prelude::*;
use bevy_voxel::chunk::{ChunkState, ChunkSurface};
use bevy_voxel::{
    FlatGenerator, VoxelChunkPlugin, VoxelMaterialPlugin, VoxelTerrainMaterial,
    VoxelWorldSettings, VoxelWorldSources, VoxelWorldViewer,
};
use std::sync::Arc;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins((VoxelMaterialPlugin, VoxelChunkPlugin))
        .insert_resource(VoxelWorldSettings {
            chunk_size: 16,
            view_distance_in_chunks: 4,
            ..VoxelWorldSettings::default()
        })
        .insert_resource(VoxelWorldSources {
            generator: Arc::new(FlatGenerator {
                height: 8.0,
                iso_scale: 1.0,
            }),
            library: None,
        })
        .add_systems(Startup, setup)
        .add_systems(Update, (assign_materials, orbit_camera))
        .run();
}

#[derive(Resource)]
struct TerrainMaterials {
    terrain: Handle<VoxelTerrainMaterial>,
    water: Handle<VoxelTerrainMaterial>,
}

fn setup(
    mut commands: Commands,
    mut materials: ResMut<Assets<VoxelTerrainMaterial>>,
) {
    // The default shader reads the colour per vertex from the material id,
    // slot 0 is the terrain, slot 1 the water, slots 2 and above the models.
    // Source: meshers/hybrid/voxel_mesher_hybrid.h:88-98
    let terrain = materials.add(VoxelTerrainMaterial::terrain(
        LinearRgba::rgb(0.36, 0.62, 0.30),
        &[
            LinearRgba::rgb(0.1, 0.35, 0.7),
            LinearRgba::rgb(0.6, 0.6, 0.6),
        ],
    ));
    let water = materials.add(VoxelTerrainMaterial::water(LinearRgba::new(
        0.1, 0.35, 0.7, 0.6,
    )));
    commands.insert_resource(TerrainMaterials { terrain, water });

    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(60.0, 40.0, 60.0).looking_at(Vec3::new(0.0, 8.0, 0.0), Vec3::Y),
        VoxelWorldViewer,
    ));

    commands.spawn((
        DirectionalLight {
            // Field name per bevy_light-0.20.1/src/directional_light.rs:93
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.9, 0.6, 0.0)),
    ));
}

/// Attaches the matching material to every chunk surface the chunk layer added.
fn assign_materials(
    mut commands: Commands,
    materials: Res<TerrainMaterials>,
    surfaces: Query<(Entity, &ChunkSurface), Without<MeshMaterial3d<VoxelTerrainMaterial>>>,
) {
    for (entity, surface) in surfaces.iter() {
        let handle = if surface.material_index == voxel_mesher::WATER_MATERIAL_INDEX {
            materials.water.clone()
        } else {
            materials.terrain.clone()
        };
        commands
            .entity(entity)
            .insert(MeshMaterial3d(handle));
    }
}

/// Moves the camera so chunk loading and unloading can be observed.
fn orbit_camera(
    time: Res<Time>,
    mut cameras: Query<&mut Transform, With<VoxelWorldViewer>>,
    chunks: Query<&ChunkState>,
) {
    let ready = chunks.iter().filter(|s| **s == ChunkState::Ready).count();
    let total = chunks.iter().count();
    if total > 0 && ready == total {
        // Orbit only once every chunk in view is meshed, which keeps the
        // loading behaviour observable
        let angle = time.elapsed_secs() * 0.1;
        for mut transform in cameras.iter_mut() {
            let radius = 80.0;
            transform.translation = Vec3::new(
                angle.cos() * radius,
                40.0,
                angle.sin() * radius,
            );
            transform.look_at(Vec3::new(0.0, 8.0, 0.0), Vec3::Y);
        }
    }
}
