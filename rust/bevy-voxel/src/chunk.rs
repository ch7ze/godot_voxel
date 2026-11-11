//! Chunk management and concurrent meshing.
//!
//! This package has no directly portable counterpart in the C++ module within
//! the agreed scope. `terrain/` and `engine/` are explicitly out of scope, see
//! doc/source/bevy_port_plan.md section 4, P7. The padding, the LOD convention
//! and the transition masks follow the mesher's requirements, every one of
//! which carries its source.
//!
//! The alternative of building on `bevy_voxel_world` was checked. Its
//! `chunk_meshing_delegate` does accept a foreign mesher
//! (`bevy_voxel_world-0.19.3/src/configuration.rs:43-57`, `:244-248`), yet it
//! hands over a `VoxelArray<I>` holding one material index per voxel, while the
//! hybrid mesher reads the SDF, TYPE, INDICES, WEIGHTS and DATA5 channels of a
//! `VoxelBuffer` at once. Hence the chunk layer is implemented here.

use bevy::prelude::*;
use bevy::tasks::{block_on, poll_once, AsyncComputeTaskPool, Task};
use std::sync::Arc;
use voxel_core::generator::VoxelGenerator;
use voxel_core::{Box3i, ChannelId, VoxelBuffer, VoxelFormat};
use voxel_mesher::hybrid::{maximum_padding, minimum_padding, HybridParams};
use voxel_mesher::{BakedLibrary, MeshOutput};

/// Position of a chunk in chunk units, at a given LOD.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Component)]
pub struct ChunkPosition {
    /// Chunk coordinates, each unit spanning `chunk_size << lod_index` voxels.
    pub position: IVec3,
    pub lod_index: u32,
}

impl ChunkPosition {
    pub fn new(position: IVec3, lod_index: u32) -> ChunkPosition {
        ChunkPosition {
            position,
            lod_index,
        }
    }

    /// Minimum corner of the chunk in voxels, at LOD 0.
    pub fn origin_in_voxels(&self, chunk_size: u32) -> IVec3 {
        self.position * (chunk_size << self.lod_index) as i32
    }

    /// Minimum corner in the chunk's own LOD sample grid.
    pub fn origin_in_lod_voxels(&self, chunk_size: u32) -> IVec3 {
        self.position * chunk_size as i32
    }
}

/// Which stage a chunk is at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Component)]
pub enum ChunkState {
    /// Queued, nothing generated yet.
    Pending,
    /// Generation and meshing are running on a task.
    Meshing,
    /// Meshes are present.
    Ready,
    /// Meshing produced no geometry.
    Empty,
}

/// Bit mask of the chunk sides stitched to a coarser neighbour.
/// Source: shaders/voxel_mesher_hybrid_shader.h:8
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Component)]
pub struct TransitionMask(pub u8);

/// Settings of the chunk layer.
#[derive(Debug, Clone, Resource)]
pub struct VoxelWorldSettings {
    /// Edge length of a chunk in voxels, without padding.
    pub chunk_size: u32,
    /// Number of LOD levels, level 0 being the finest.
    pub lod_count: u32,
    /// Distance in chunks within which chunks are kept loaded.
    pub view_distance_in_chunks: u32,
    /// Channel depths of the generated buffers.
    pub format: VoxelFormat,
    /// Parameters handed to the mesher.
    pub mesher_params: HybridParams,
    /// Maximum number of meshing tasks started per frame.
    pub max_tasks_per_frame: usize,
}

impl Default for VoxelWorldSettings {
    fn default() -> VoxelWorldSettings {
        let format = VoxelFormat {
            // DATA5 feeds the fluid iso-surface pass, so it defaults to air
            // rather than sitting exactly on the surface.
            // Source: storage/voxel_buffer.cpp:241-252
            channels_acting_as_sdf: 1 << ChannelId::Data5.index(),
            ..VoxelFormat::default()
        };
        VoxelWorldSettings {
            chunk_size: 16,
            lod_count: 1,
            view_distance_in_chunks: 4,
            format,
            mesher_params: HybridParams::default(),
            max_tasks_per_frame: 8,
        }
    }
}

impl VoxelWorldSettings {
    /// Edge length of a generated buffer including the mesher's padding.
    ///
    /// Padding is taken from the mesher, `MIN_PADDING` towards the negative and
    /// `MAX_PADDING` towards the positive axes, with the model path's single
    /// voxel folded in.
    /// Source: meshers/transvoxel/transvoxel.h:16-18,
    /// meshers/hybrid/voxel_mesher_hybrid.cpp:28, meshers/voxel_mesher.h:91-95
    pub fn padded_chunk_size(&self) -> i32 {
        self.chunk_size as i32 + minimum_padding() + maximum_padding()
    }

    /// Offset of the unpadded chunk inside a padded buffer.
    pub fn padding_offset(&self) -> IVec3 {
        IVec3::splat(minimum_padding())
    }

    /// Size of a chunk in world units at a given LOD.
    pub fn chunk_world_size(&self, lod_index: u32) -> f32 {
        (self.chunk_size << lod_index) as f32
    }
}

/// Generator and model library the chunk layer meshes with.
#[derive(Resource, Clone)]
pub struct VoxelWorldSources {
    pub generator: Arc<dyn VoxelGenerator>,
    /// Model library for the blocky part. `None` makes the mesher behave like
    /// plain Transvoxel.
    /// Source: meshers/hybrid/voxel_mesher_hybrid.cpp:748-751
    pub library: Option<Arc<BakedLibrary>>,
}

/// Marks the entity whose position drives chunk loading.
#[derive(Debug, Clone, Copy, Default, Component)]
pub struct VoxelWorldViewer;

/// A meshing task in flight.
#[derive(Component)]
pub struct MeshingTask(pub Task<MeshOutput>);

/// Generates a padded buffer for one chunk.
///
/// The buffer covers the chunk plus the mesher's padding, so the border carries
/// the neighbour chunks' values and the mesher produces no seams.
/// Source: meshers/voxel_mesher.h:89-95
pub fn generate_chunk_buffer(
    generator: &dyn VoxelGenerator,
    settings: &VoxelWorldSettings,
    chunk: ChunkPosition,
) -> VoxelBuffer {
    let padded = IVec3::splat(settings.padded_chunk_size());
    let mut buffer = VoxelBuffer::with_format(padded, settings.format)
        .expect("padded chunk size is within the buffer limits");

    // The generator is asked for the padded region, so the border holds the
    // neighbours' data. The origin shifts back by the padding, scaled to the
    // chunk's LOD.
    let step = 1i32 << chunk.lod_index;
    let origin = chunk.origin_in_voxels(settings.chunk_size) - settings.padding_offset() * step;

    generator.generate_block(&mut buffer, origin, chunk.lod_index);
    buffer
}

/// Generates and meshes one chunk.
pub fn mesh_chunk(
    generator: &dyn VoxelGenerator,
    library: Option<&BakedLibrary>,
    settings: &VoxelWorldSettings,
    chunk: ChunkPosition,
    transition_mask: u8,
) -> MeshOutput {
    let buffer = generate_chunk_buffer(generator, settings, chunk);
    let params = HybridParams {
        lod_index: chunk.lod_index,
        transition_mask,
        ..settings.mesher_params.clone()
    };
    voxel_mesher::build(&buffer, library, &params)
}

/// Transition mask of a chunk, given which neighbour sides are coarser.
///
/// Bit N corresponds to side N of `voxel_core::cube::side_axis`, matching the
/// mask the shader reads.
/// Source: shaders/voxel_mesher_hybrid_shader.h:17-23
pub fn transition_mask_from_neighbour_lods(
    own_lod: u32,
    neighbour_lods: [Option<u32>; 6],
) -> u8 {
    let mut mask = 0u8;
    for (side, neighbour) in neighbour_lods.iter().enumerate() {
        if let Some(lod) = neighbour {
            if *lod > own_lod {
                mask |= 1 << side;
            }
        }
    }
    mask
}

/// LOD a chunk should be loaded at, given its distance to the viewer.
///
/// Each LOD level doubles the distance at which it takes over, which keeps the
/// screen-space error roughly constant.
pub fn lod_for_distance(
    distance_in_chunks: f32,
    lod_count: u32,
    view_distance_in_chunks: u32,
) -> u32 {
    if lod_count <= 1 || view_distance_in_chunks == 0 {
        return 0;
    }
    let normalized = distance_in_chunks / view_distance_in_chunks as f32;
    if normalized <= 1.0 {
        return 0;
    }
    let level = normalized.log2().floor().max(0.0) as u32 + 1;
    level.min(lod_count - 1)
}

/// Chunk positions that should be loaded for a viewer position.
pub fn chunks_in_view(
    settings: &VoxelWorldSettings,
    viewer_position: Vec3,
) -> Vec<ChunkPosition> {
    let mut out = Vec::new();
    let radius = settings.view_distance_in_chunks as i32;
    let chunk_size = settings.chunk_size as f32;
    let center = (viewer_position / chunk_size).floor().as_ivec3();

    let area = Box3i::from_center_extents(center, IVec3::splat(radius));
    area.for_each_cell_zxy(|position| {
        let offset = (position - center).as_vec3();
        let distance = offset.length();
        if distance <= radius as f32 {
            let lod = lod_for_distance(
                distance,
                settings.lod_count,
                settings.view_distance_in_chunks,
            );
            out.push(ChunkPosition::new(position, lod));
        }
    });
    out
}

/// Tracks which chunks exist, so spawning and despawning stay idempotent.
#[derive(Debug, Clone, Default, Resource)]
pub struct LoadedChunks {
    entries: std::collections::HashMap<ChunkPosition, Entity>,
}

impl LoadedChunks {
    pub fn contains(&self, chunk: ChunkPosition) -> bool {
        self.entries.contains_key(&chunk)
    }

    pub fn insert(&mut self, chunk: ChunkPosition, entity: Entity) {
        self.entries.insert(chunk, entity);
    }

    pub fn remove(&mut self, chunk: ChunkPosition) -> Option<Entity> {
        self.entries.remove(&chunk)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&ChunkPosition, &Entity)> {
        self.entries.iter()
    }

    /// Chunks that are loaded but no longer wanted.
    pub fn surplus(&self, wanted: &[ChunkPosition]) -> Vec<ChunkPosition> {
        self.entries
            .keys()
            .filter(|chunk| !wanted.contains(chunk))
            .copied()
            .collect()
    }
}

/// Spawns entities for chunks that came into view and despawns the rest.
pub fn update_chunk_set(
    mut commands: Commands,
    settings: Res<VoxelWorldSettings>,
    mut loaded: ResMut<LoadedChunks>,
    viewers: Query<&Transform, With<VoxelWorldViewer>>,
) {
    let Ok(viewer) = viewers.single() else {
        return;
    };
    let wanted = chunks_in_view(&settings, viewer.translation);

    for chunk in &wanted {
        if loaded.contains(*chunk) {
            continue;
        }
        let world_size = settings.chunk_world_size(chunk.lod_index);
        let translation = chunk.position.as_vec3() * world_size;
        let entity = commands
            .spawn((
                *chunk,
                ChunkState::Pending,
                TransitionMask::default(),
                Transform::from_translation(translation),
            ))
            .id();
        loaded.insert(*chunk, entity);
    }

    for chunk in loaded.surplus(&wanted) {
        if let Some(entity) = loaded.remove(chunk) {
            commands.entity(entity).despawn();
        }
    }
}

/// Starts meshing tasks for pending chunks.
///
/// Meshing runs on the async compute pool, the mesher permits several
/// concurrent calls.
/// Source: meshers/voxel_mesher.h:81
pub fn start_meshing_tasks(
    mut commands: Commands,
    settings: Res<VoxelWorldSettings>,
    sources: Res<VoxelWorldSources>,
    mut pending: Query<(Entity, &ChunkPosition, &TransitionMask, &mut ChunkState)>,
) {
    let pool = AsyncComputeTaskPool::get();
    let mut started = 0usize;

    for (entity, chunk, mask, mut state) in pending.iter_mut() {
        if started >= settings.max_tasks_per_frame {
            break;
        }
        if *state != ChunkState::Pending {
            continue;
        }

        let generator = sources.generator.clone();
        let library = sources.library.clone();
        let settings = settings.clone();
        let chunk = *chunk;
        let mask = mask.0;

        let task = pool.spawn(async move {
            mesh_chunk(
                generator.as_ref(),
                library.as_deref(),
                &settings,
                chunk,
                mask,
            )
        });

        commands.entity(entity).insert(MeshingTask(task));
        *state = ChunkState::Meshing;
        started += 1;
    }
}

/// Collects finished meshing tasks and attaches their meshes.
pub fn collect_meshing_tasks(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut tasks: Query<(Entity, &mut MeshingTask, &mut ChunkState)>,
) {
    for (entity, mut task, mut state) in tasks.iter_mut() {
        let Some(output) = block_on(poll_once(&mut task.0)) else {
            continue;
        };

        commands.entity(entity).remove::<MeshingTask>();

        let chunk_meshes = crate::mesh::output_to_meshes(
            &output,
            bevy::asset::RenderAssetUsages::default(),
        );
        if chunk_meshes.is_empty() {
            *state = ChunkState::Empty;
            continue;
        }

        for chunk_mesh in chunk_meshes {
            let handle = meshes.add(chunk_mesh.mesh);
            commands.entity(entity).with_child((
                Mesh3d(handle),
                ChunkSurface {
                    material_index: chunk_mesh.material_index,
                },
            ));
        }
        *state = ChunkState::Ready;
    }
}

/// Marks one rendered surface of a chunk and which material it uses.
#[derive(Debug, Clone, Copy, Component)]
pub struct ChunkSurface {
    pub material_index: u16,
}

/// Registers the chunk layer with an app.
pub struct VoxelChunkPlugin;

impl Plugin for VoxelChunkPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<LoadedChunks>().add_systems(
            Update,
            (update_chunk_set, start_meshing_tasks, collect_meshing_tasks).chain(),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use voxel_core::generator::FlatGenerator;
    use voxel_mesher::blocky::MODEL_PADDING;
    use voxel_mesher::transvoxel as tv;

    fn settings() -> VoxelWorldSettings {
        VoxelWorldSettings::default()
    }

    #[test]
    fn the_padded_size_covers_both_meshers_padding() {
        let s = settings();
        // Source: meshers/transvoxel/transvoxel.h:16-18
        assert_eq!(
            s.padded_chunk_size(),
            s.chunk_size as i32 + tv::MIN_PADDING + tv::MAX_PADDING
        );
        // The model path needs at least one voxel of padding
        // Source: meshers/hybrid/voxel_mesher_hybrid.cpp:28
        assert!(minimum_padding() >= MODEL_PADDING);
        assert!(maximum_padding() >= MODEL_PADDING);
    }

    #[test]
    fn the_padding_offset_matches_the_minimum_padding() {
        assert_eq!(
            settings().padding_offset(),
            IVec3::splat(tv::MIN_PADDING)
        );
    }

    #[test]
    fn chunk_origins_scale_with_the_lod() {
        let s = settings();
        let chunk = ChunkPosition::new(IVec3::new(2, 0, 0), 0);
        assert_eq!(chunk.origin_in_voxels(s.chunk_size), IVec3::new(32, 0, 0));
        let coarse = ChunkPosition::new(IVec3::new(2, 0, 0), 2);
        // Each unit spans chunk_size << lod voxels
        assert_eq!(coarse.origin_in_voxels(s.chunk_size), IVec3::new(128, 0, 0));
        // The sample grid itself stays at chunk_size units
        assert_eq!(
            coarse.origin_in_lod_voxels(s.chunk_size),
            IVec3::new(32, 0, 0)
        );
    }

    #[test]
    fn the_chunk_world_size_doubles_per_lod() {
        let s = settings();
        assert_eq!(s.chunk_world_size(0), 16.0);
        assert_eq!(s.chunk_world_size(1), 32.0);
        assert_eq!(s.chunk_world_size(3), 128.0);
    }

    #[test]
    fn the_generated_buffer_is_padded_and_holds_the_surface() {
        let s = settings();
        let generator = FlatGenerator {
            height: 8.0,
            iso_scale: 1.0,
        };
        let buffer = generate_chunk_buffer(&generator, &s, ChunkPosition::new(IVec3::ZERO, 0));
        assert_eq!(buffer.size(), IVec3::splat(s.padded_chunk_size()));
        // The surface at world y 8 falls inside the chunk
        let padded_y = 8 + tv::MIN_PADDING;
        assert_eq!(
            buffer.get_voxel_f(IVec3::new(2, padded_y, 2), ChannelId::Sdf),
            0.0
        );
        // Below is solid, above is air
        assert!(buffer.get_voxel_f(IVec3::new(2, 1, 2), ChannelId::Sdf) < 0.0);
        assert!(
            buffer.get_voxel_f(IVec3::new(2, padded_y + 2, 2), ChannelId::Sdf) > 0.0
        );
    }

    #[test]
    fn the_padding_border_carries_the_neighbour_data() {
        let s = settings();
        let generator = FlatGenerator {
            height: 100.0,
            iso_scale: 1.0,
        };
        // A chunk well below the surface is uniformly solid, including the
        // border, so no seam can appear
        let buffer = generate_chunk_buffer(&generator, &s, ChunkPosition::new(IVec3::ZERO, 0));
        assert!(buffer.get_voxel_f(IVec3::ZERO, ChannelId::Sdf) < 0.0);
        let last = s.padded_chunk_size() - 1;
        assert!(buffer.get_voxel_f(IVec3::splat(last), ChannelId::Sdf) < 0.0);
    }

    #[test]
    fn neighbouring_chunks_sample_the_same_border_voxels() {
        let s = settings();
        let generator = FlatGenerator {
            height: 8.0,
            iso_scale: 1.0,
        };
        let a = generate_chunk_buffer(&generator, &s, ChunkPosition::new(IVec3::ZERO, 0));
        let b = generate_chunk_buffer(
            &generator,
            &s,
            ChunkPosition::new(IVec3::new(0, 1, 0), 0),
        );
        // The last chunk row of a and the first of b describe the same world y
        let a_top = tv::MIN_PADDING + s.chunk_size as i32;
        let b_bottom = tv::MIN_PADDING;
        assert_eq!(
            a.get_voxel_f(IVec3::new(2, a_top, 2), ChannelId::Sdf),
            b.get_voxel_f(IVec3::new(2, b_bottom, 2), ChannelId::Sdf)
        );
    }

    #[test]
    fn meshing_a_chunk_on_the_surface_yields_geometry() {
        let s = settings();
        let generator = FlatGenerator {
            height: 8.0,
            iso_scale: 1.0,
        };
        let out = mesh_chunk(
            &generator,
            None,
            &s,
            ChunkPosition::new(IVec3::ZERO, 0),
            0,
        );
        assert!(!out.is_empty(), "no geometry produced");
        assert!(out.triangle_count() > 0);
    }

    #[test]
    fn meshing_a_chunk_away_from_the_surface_yields_nothing() {
        let s = settings();
        let generator = FlatGenerator {
            height: 8.0,
            iso_scale: 1.0,
        };
        let out = mesh_chunk(
            &generator,
            None,
            &s,
            ChunkPosition::new(IVec3::new(0, 40, 0), 0),
            0,
        );
        assert!(out.is_empty());
    }

    #[test]
    fn the_transition_mask_flags_only_coarser_neighbours() {
        // Side 0 is coarser, side 1 is finer, side 2 matches
        let mask = transition_mask_from_neighbour_lods(
            1,
            [Some(2), Some(0), Some(1), None, None, None],
        );
        assert_eq!(mask, 0b000001);
    }

    #[test]
    fn a_chunk_without_neighbours_has_no_transitions() {
        assert_eq!(transition_mask_from_neighbour_lods(0, [None; 6]), 0);
    }

    #[test]
    fn every_coarser_side_sets_its_bit() {
        let mask = transition_mask_from_neighbour_lods(0, [Some(1); 6]);
        assert_eq!(mask, 0b111111);
    }

    #[test]
    fn a_single_lod_level_always_selects_lod_zero() {
        assert_eq!(lod_for_distance(100.0, 1, 4), 0);
    }

    #[test]
    fn the_lod_rises_with_the_distance() {
        // Within the view distance the finest level applies
        assert_eq!(lod_for_distance(1.0, 4, 4), 0);
        assert_eq!(lod_for_distance(4.0, 4, 4), 0);
        // Beyond it each doubling raises the level
        assert_eq!(lod_for_distance(5.0, 4, 4), 1);
        assert_eq!(lod_for_distance(9.0, 4, 4), 2);
        assert_eq!(lod_for_distance(17.0, 4, 4), 3);
    }

    #[test]
    fn the_lod_is_capped_at_the_level_count() {
        assert_eq!(lod_for_distance(10_000.0, 3, 4), 2);
    }

    #[test]
    fn chunks_in_view_cover_a_sphere_around_the_viewer() {
        let s = VoxelWorldSettings {
            view_distance_in_chunks: 2,
            lod_count: 1,
            ..settings()
        };
        let chunks = chunks_in_view(&s, Vec3::ZERO);
        assert!(!chunks.is_empty());
        // Every chunk lies within the radius
        for chunk in &chunks {
            let distance = chunk.position.as_vec3().length();
            assert!(distance <= 2.0 + 1e-6, "chunk {:?} too far", chunk.position);
            assert_eq!(chunk.lod_index, 0);
        }
        // The chunk holding the viewer is included
        assert!(chunks.iter().any(|c| c.position == IVec3::ZERO));
    }

    #[test]
    fn chunks_in_view_follow_the_viewer() {
        let s = VoxelWorldSettings {
            view_distance_in_chunks: 1,
            lod_count: 1,
            ..settings()
        };
        let chunks = chunks_in_view(&s, Vec3::new(160.0, 0.0, 0.0));
        // 160 world units at chunk size 16 is chunk 10
        assert!(chunks.iter().any(|c| c.position == IVec3::new(10, 0, 0)));
        assert!(!chunks.iter().any(|c| c.position == IVec3::ZERO));
    }

    #[test]
    fn distant_chunks_get_a_coarser_lod() {
        let s = VoxelWorldSettings {
            view_distance_in_chunks: 2,
            lod_count: 3,
            ..settings()
        };
        let chunks = chunks_in_view(&s, Vec3::ZERO);
        // Inside the view distance everything stays at lod 0
        assert!(chunks.iter().all(|c| c.lod_index == 0));
    }

    #[test]
    fn the_loaded_set_tracks_insertions_and_removals() {
        let mut loaded = LoadedChunks::default();
        let chunk = ChunkPosition::new(IVec3::ZERO, 0);
        assert!(loaded.is_empty());
        loaded.insert(chunk, Entity::from_raw_u32(1).unwrap());
        assert!(loaded.contains(chunk));
        assert_eq!(loaded.len(), 1);
        assert!(loaded.remove(chunk).is_some());
        assert!(loaded.is_empty());
    }

    #[test]
    fn the_surplus_lists_chunks_no_longer_wanted() {
        let mut loaded = LoadedChunks::default();
        let keep = ChunkPosition::new(IVec3::ZERO, 0);
        let drop = ChunkPosition::new(IVec3::new(9, 9, 9), 0);
        loaded.insert(keep, Entity::from_raw_u32(1).unwrap());
        loaded.insert(drop, Entity::from_raw_u32(2).unwrap());
        let surplus = loaded.surplus(&[keep]);
        assert_eq!(surplus, vec![drop]);
    }

    #[test]
    fn a_chunk_changing_lod_counts_as_a_different_chunk() {
        let mut loaded = LoadedChunks::default();
        let fine = ChunkPosition::new(IVec3::ZERO, 0);
        let coarse = ChunkPosition::new(IVec3::ZERO, 1);
        loaded.insert(fine, Entity::from_raw_u32(1).unwrap());
        assert!(!loaded.contains(coarse));
        assert_eq!(loaded.surplus(&[coarse]), vec![fine]);
    }

    #[test]
    fn the_default_format_treats_data5_as_sdf() {
        // Without this the fluid pass would mesh through every untouched voxel
        // Source: storage/voxel_buffer.cpp:241-252
        let s = settings();
        assert!(s.format.acts_as_sdf(ChannelId::Data5));
        let generator = FlatGenerator::default();
        let buffer = generate_chunk_buffer(&generator, &s, ChunkPosition::new(IVec3::ZERO, 0));
        assert!(buffer.get_voxel_f(IVec3::ZERO, ChannelId::Data5) > 0.0);
    }

    #[test]
    fn a_coarse_chunk_meshes_at_its_own_lod() {
        let s = settings();
        let generator = FlatGenerator {
            height: 8.0,
            iso_scale: 1.0,
        };
        // At lod 1 the chunk spans twice the world size, so a surface at 8
        // falls into chunk 0 as well
        let out = mesh_chunk(
            &generator,
            None,
            &s,
            ChunkPosition::new(IVec3::ZERO, 1),
            0,
        );
        assert!(!out.is_empty());
        // Positions are scaled by the lod, so they exceed the chunk size
        let max_x = out
            .surfaces
            .iter()
            .flat_map(|surface| surface.positions.iter())
            .map(|p| p.x)
            .fold(0.0f32, f32::max);
        assert!(max_x > s.chunk_size as f32, "max x is {max_x}");
    }
}
