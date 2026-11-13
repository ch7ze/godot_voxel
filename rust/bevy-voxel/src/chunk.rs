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

/// LOD level the ring layout assigns to a world position.
///
/// The finest ring whose reach contains the position wins, which mirrors how
/// `chunks_in_view` drops chunks the finer ring already covers.
pub fn lod_at_world_position(
    settings: &VoxelWorldSettings,
    world_position: Vec3,
    viewer_position: Vec3,
) -> u32 {
    let lod_count = settings.lod_count.max(1);
    let mut level = lod_count - 1;
    // The finest level whose chunk at this position is actually loaded wins.
    // Testing the chunk the position falls into, rather than the position
    // itself, keeps this in step with `chunks_in_view`, which drops a chunk
    // only when the finer ring spans all of it.
    for lod_index in 0..lod_count {
        let world_size = settings.chunk_world_size(lod_index);
        let chunk_position = (world_position / world_size).floor().as_ivec3();
        let chunk = ChunkPosition::new(chunk_position, lod_index);
        if covered_by_finer_lod(settings, chunk, viewer_position) {
            continue;
        }
        if chunk_is_in_ring(settings, chunk, viewer_position) {
            level = lod_index;
            break;
        }
    }
    level
}

/// Whether a chunk overlaps its own LOD ring around the viewer.
///
/// The ring is the axis aligned cube of half extent
/// `lod_reach_in_world_units` centred on the viewer. A cube keeps the rings
/// nesting exactly, since every reach is a power of two multiple of the finer
/// levels' chunk sizes, which a sphere would break.
pub fn chunk_is_in_ring(
    settings: &VoxelWorldSettings,
    chunk: ChunkPosition,
    viewer_position: Vec3,
) -> bool {
    let reach = lod_reach_in_world_units(settings, chunk.lod_index);
    let (min_corner, max_corner) = chunk_box_around_viewer(settings, chunk, viewer_position);
    min_corner.max_element() < reach && max_corner.min_element() > -reach
}

/// Transition mask of one chunk under the ring layout.
///
/// Each of the six sides is probed at the centre of the neighbouring chunk.
/// A side whose neighbour sits at a coarser level gets its bit set, so the
/// mesher stitches that side.
/// Source: shaders/voxel_mesher_hybrid_shader.h:17-23
pub fn transition_mask_for_chunk(
    settings: &VoxelWorldSettings,
    chunk: ChunkPosition,
    viewer_position: Vec3,
) -> u8 {
    let world_size = settings.chunk_world_size(chunk.lod_index);
    let centre = (chunk.position.as_vec3() + Vec3::splat(0.5)) * world_size;

    let mut neighbour_lods = [None; 6];
    for (side, normal) in voxel_core::cube::SIDE_NORMALS.iter().enumerate() {
        let neighbour_centre = centre + normal.as_vec3() * world_size;
        neighbour_lods[side] =
            Some(lod_at_world_position(settings, neighbour_centre, viewer_position));
    }
    transition_mask_from_neighbour_lods(chunk.lod_index, neighbour_lods)
}

/// World space reach of one LOD ring, in world units.
///
/// Every level spans `view_distance_in_chunks` of its own chunks, which are
/// twice as large per level, so each ring reaches twice as far as the one
/// below it.
pub fn lod_reach_in_world_units(settings: &VoxelWorldSettings, lod_index: u32) -> f32 {
    settings.view_distance_in_chunks as f32 * settings.chunk_world_size(lod_index)
}

/// World space box a chunk occupies, relative to the viewer.
fn chunk_box_around_viewer(
    settings: &VoxelWorldSettings,
    chunk: ChunkPosition,
    viewer_position: Vec3,
) -> (Vec3, Vec3) {
    let world_size = settings.chunk_world_size(chunk.lod_index);
    let min_corner = chunk.position.as_vec3() * world_size - viewer_position;
    (min_corner, min_corner + Vec3::splat(world_size))
}

/// Whether the next finer LOD level already covers all of a chunk.
///
/// Each ring is an axis aligned cube in world space, and its reach is a power
/// of two multiple of every finer level's chunk size. The rings therefore
/// nest exactly, and a chunk fully inside the finer cube is covered without
/// any remainder.
pub fn covered_by_finer_lod(
    settings: &VoxelWorldSettings,
    chunk: ChunkPosition,
    viewer_position: Vec3,
) -> bool {
    if chunk.lod_index == 0 {
        return false;
    }
    let inner_reach = lod_reach_in_world_units(settings, chunk.lod_index - 1);
    let (min_corner, max_corner) = chunk_box_around_viewer(settings, chunk, viewer_position);
    min_corner.min_element() >= -inner_reach && max_corner.max_element() <= inner_reach
}

/// Chunk positions that should be loaded for a viewer position.
///
/// One ring is produced per LOD level. Level 0 sits closest to the viewer,
/// each coarser level forms a shell around the finer ones, and chunks the
/// finer ring already covers are left out.
pub fn chunks_in_view(
    settings: &VoxelWorldSettings,
    viewer_position: Vec3,
) -> Vec<ChunkPosition> {
    let mut out = Vec::new();
    // One chunk of slack, a chunk may overlap the ring cube without its own
    // origin lying inside the searched box
    let radius = settings.view_distance_in_chunks as i32 + 1;
    let lod_count = settings.lod_count.max(1);

    for lod_index in 0..lod_count {
        let world_size = settings.chunk_world_size(lod_index);
        let center = (viewer_position / world_size).floor().as_ivec3();
        let area = Box3i::from_center_extents(center, IVec3::splat(radius));

        area.for_each_cell_zxy(|position| {
            let chunk = ChunkPosition::new(position, lod_index);
            if !chunk_is_in_ring(settings, chunk, viewer_position) {
                return;
            }
            if covered_by_finer_lod(settings, chunk, viewer_position) {
                return;
            }
            out.push(chunk);
        });
    }
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
    ///
    /// The wanted set is a hash set so the scan stays linear in the number of
    /// loaded chunks, a slice would make it quadratic.
    pub fn surplus(
        &self,
        wanted: &std::collections::HashSet<ChunkPosition>,
    ) -> Vec<ChunkPosition> {
        self.entries
            .keys()
            .filter(|chunk| !wanted.contains(*chunk))
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
    let wanted_set: std::collections::HashSet<ChunkPosition> = wanted.iter().copied().collect();

    for chunk in &wanted {
        if loaded.contains(*chunk) {
            continue;
        }
        let world_size = settings.chunk_world_size(chunk.lod_index);
        let translation = chunk.position.as_vec3() * world_size;
        let mask = transition_mask_for_chunk(&settings, *chunk, viewer.translation);
        let entity = commands
            .spawn((
                *chunk,
                ChunkState::Pending,
                TransitionMask(mask),
                Transform::from_translation(translation),
                // The surfaces are spawned as children, so the chunk itself
                // carries the visibility its children inherit.
                // Source: bevy_camera-0.20.1/src/visibility/mod.rs:499,
                // bevy_transform-0.20.1/src/plugins.rs:24
                Visibility::default(),
            ))
            .id();
        loaded.insert(*chunk, entity);
    }

    for chunk in loaded.surplus(&wanted_set) {
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
    mut tasks: Query<(Entity, &mut MeshingTask, &TransitionMask, &mut ChunkState)>,
) {
    for (entity, mut task, mask, mut state) in tasks.iter_mut() {
        let Some(output) = block_on(poll_once(&mut task.0)) else {
            continue;
        };

        commands.entity(entity).remove::<MeshingTask>();

        let usage = bevy::asset::RenderAssetUsages::default();
        let mut chunk_meshes = crate::mesh::output_to_meshes(&output, usage);

        // The transition meshes of the stitched sides are rendered alongside
        // the regular ones. The shader culls the triangles of any side whose
        // bit is absent from the mask, so only the stitched sides show.
        // Source: shaders/voxel_mesher_hybrid_shader.h:28-40
        let per_side = crate::mesh::output_to_transition_meshes(&output, usage);
        for (side, side_meshes) in per_side.into_iter().enumerate() {
            if mask.0 & (1 << side) == 0 {
                continue;
            }
            chunk_meshes.extend(side_meshes);
        }

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
                    transition_mask: mask.0,
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
    /// Transition mask of the owning chunk, which the shader needs so it can
    /// stitch the stitched sides and cull the rest.
    /// Source: shaders/voxel_mesher_hybrid_shader.h:17-23
    pub transition_mask: u8,
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
    fn each_lod_ring_reaches_twice_as_far_as_the_one_below() {
        let s = VoxelWorldSettings {
            chunk_size: 16,
            view_distance_in_chunks: 4,
            lod_count: 3,
            ..settings()
        };
        assert_eq!(lod_reach_in_world_units(&s, 0), 64.0);
        assert_eq!(lod_reach_in_world_units(&s, 1), 128.0);
        assert_eq!(lod_reach_in_world_units(&s, 2), 256.0);
    }

    #[test]
    fn a_single_lod_level_always_selects_lod_zero() {
        let s = VoxelWorldSettings {
            lod_count: 1,
            view_distance_in_chunks: 4,
            ..settings()
        };
        assert_eq!(
            lod_at_world_position(&s, Vec3::new(10_000.0, 0.0, 0.0), Vec3::ZERO),
            0
        );
    }

    #[test]
    fn the_lod_rises_with_the_distance() {
        let s = VoxelWorldSettings {
            chunk_size: 16,
            view_distance_in_chunks: 4,
            lod_count: 3,
            ..settings()
        };
        // The rings are world space cubes, level 0 reaching 64 units
        assert_eq!(lod_at_world_position(&s, Vec3::new(10.0, 0.0, 0.0), Vec3::ZERO), 0);
        assert_eq!(lod_at_world_position(&s, Vec3::new(63.0, 0.0, 0.0), Vec3::ZERO), 0);
        // Level 1 takes over at the boundary and reaches 128
        assert_eq!(lod_at_world_position(&s, Vec3::new(64.0, 0.0, 0.0), Vec3::ZERO), 1);
        assert_eq!(lod_at_world_position(&s, Vec3::new(127.0, 0.0, 0.0), Vec3::ZERO), 1);
        // Level 2 from there on
        assert_eq!(lod_at_world_position(&s, Vec3::new(128.0, 0.0, 0.0), Vec3::ZERO), 2);
    }

    #[test]
    fn the_lod_is_capped_at_the_level_count() {
        let s = VoxelWorldSettings {
            view_distance_in_chunks: 4,
            lod_count: 3,
            ..settings()
        };
        assert_eq!(
            lod_at_world_position(&s, Vec3::new(10_000.0, 0.0, 0.0), Vec3::ZERO),
            2
        );
    }

    #[test]
    fn the_lod_follows_the_viewer() {
        let s = VoxelWorldSettings {
            chunk_size: 16,
            view_distance_in_chunks: 4,
            lod_count: 3,
            ..settings()
        };
        let viewer = Vec3::new(1000.0, 0.0, 0.0);
        // Right at the viewer the finest level applies
        assert_eq!(lod_at_world_position(&s, viewer, viewer), 0);
        // The same world position is coarse when the viewer is far away
        assert_eq!(lod_at_world_position(&s, viewer, Vec3::ZERO), 2);
    }

    #[test]
    fn chunks_in_view_cover_a_cube_around_the_viewer() {
        let s = VoxelWorldSettings {
            chunk_size: 16,
            view_distance_in_chunks: 2,
            lod_count: 1,
            ..settings()
        };
        let chunks = chunks_in_view(&s, Vec3::ZERO);
        assert!(!chunks.is_empty());

        // Level 0 reaches 32 world units, so chunks -2..1 are covered
        let reach = lod_reach_in_world_units(&s, 0);
        assert_eq!(reach, 32.0);
        for chunk in &chunks {
            assert_eq!(chunk.lod_index, 0);
            for axis in 0..3 {
                let min_corner = chunk.position[axis] as f32 * s.chunk_size as f32;
                let max_corner = min_corner + s.chunk_size as f32;
                assert!(
                    min_corner < reach && max_corner > -reach,
                    "chunk {:?} lies outside the ring",
                    chunk.position
                );
            }
        }
        // The chunk holding the viewer is included
        assert!(chunks.iter().any(|c| c.position == IVec3::ZERO));
        // The cube spans exactly four chunks per axis
        assert_eq!(chunks.len(), 4 * 4 * 4);
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
            chunk_size: 16,
            view_distance_in_chunks: 3,
            lod_count: 3,
            ..settings()
        };
        let chunks = chunks_in_view(&s, Vec3::ZERO);

        // Every level produces chunks
        for lod_index in 0..3 {
            assert!(
                chunks.iter().any(|c| c.lod_index == lod_index),
                "no chunk at lod {lod_index}"
            );
        }

        // The chunk holding the viewer is the finest one
        assert!(chunks
            .iter()
            .any(|c| c.position == IVec3::ZERO && c.lod_index == 0));

        // Coarse chunks sit outside the reach of the finer ring
        let inner_reach = lod_reach_in_world_units(&s, 0);
        for chunk in chunks.iter().filter(|c| c.lod_index == 1) {
            let world_size = s.chunk_world_size(1);
            let min_corner = chunk.position.as_vec3() * world_size;
            let max_corner = min_corner + Vec3::splat(world_size);
            assert!(
                min_corner.abs().max(max_corner.abs()).max_element() > inner_reach,
                "chunk {:?} at lod 1 lies inside the lod 0 ring",
                chunk.position
            );
        }
    }

    #[test]
    fn a_coarse_chunk_inside_the_finer_ring_is_dropped() {
        let s = VoxelWorldSettings {
            chunk_size: 16,
            view_distance_in_chunks: 4,
            lod_count: 2,
            ..settings()
        };
        // Lod 0 reaches 64 world units, a lod 1 chunk spans 32, so chunk
        // (0,0,0) at lod 1 covers 0..32 and lies fully inside
        assert!(covered_by_finer_lod(
            &s,
            ChunkPosition::new(IVec3::ZERO, 1),
            Vec3::ZERO
        ));
        // Chunk (2,0,0) at lod 1 covers 64..96, which reaches beyond
        assert!(!covered_by_finer_lod(
            &s,
            ChunkPosition::new(IVec3::new(2, 0, 0), 1),
            Vec3::ZERO
        ));
        // Level 0 is never covered
        assert!(!covered_by_finer_lod(
            &s,
            ChunkPosition::new(IVec3::ZERO, 0),
            Vec3::ZERO
        ));
    }

    #[test]
    fn no_two_chunks_in_view_cover_the_same_volume_twice() {
        let s = VoxelWorldSettings {
            chunk_size: 16,
            view_distance_in_chunks: 3,
            lod_count: 3,
            ..settings()
        };
        let chunks = chunks_in_view(&s, Vec3::ZERO);
        // Each entry is unique, so the loaded set stays idempotent
        let unique: std::collections::HashSet<ChunkPosition> =
            chunks.iter().copied().collect();
        assert_eq!(unique.len(), chunks.len());
    }

    #[test]
    fn a_chunk_bordering_a_coarser_ring_gets_a_transition_mask() {
        let s = VoxelWorldSettings {
            chunk_size: 16,
            view_distance_in_chunks: 4,
            lod_count: 2,
            ..settings()
        };
        // Lod 0 reaches 64 world units. Chunk (3,0,0) spans 48..64, its
        // positive x neighbour centre sits at 72, which falls into lod 1.
        let mask = transition_mask_for_chunk(
            &s,
            ChunkPosition::new(IVec3::new(3, 0, 0), 0),
            Vec3::ZERO,
        );
        // Side 0 is positive x
        // Source: voxel-core/src/cube.rs:57, :124-131
        assert_ne!(mask & (1 << 0), 0, "mask {mask:#08b}");
        // The negative x neighbour stays at lod 0
        assert_eq!(mask & (1 << 1), 0, "mask {mask:#08b}");
    }

    #[test]
    fn a_chunk_well_inside_the_finest_ring_has_no_transitions() {
        let s = VoxelWorldSettings {
            chunk_size: 16,
            view_distance_in_chunks: 4,
            lod_count: 2,
            ..settings()
        };
        let mask =
            transition_mask_for_chunk(&s, ChunkPosition::new(IVec3::ZERO, 0), Vec3::ZERO);
        assert_eq!(mask, 0);
    }

    #[test]
    fn no_two_loaded_chunks_overlap_in_world_space() {
        // Overlapping chunks of different levels would draw the same terrain
        // twice, which shows as z-fighting and doubled geometry
        let s = VoxelWorldSettings {
            chunk_size: 16,
            view_distance_in_chunks: 2,
            lod_count: 3,
            ..settings()
        };
        let chunks = chunks_in_view(&s, Vec3::ZERO);

        let world_box = |chunk: &ChunkPosition| {
            let size = s.chunk_world_size(chunk.lod_index);
            let min_corner = chunk.position.as_vec3() * size;
            (min_corner, min_corner + Vec3::splat(size))
        };

        for (index, a) in chunks.iter().enumerate() {
            let (a_min, a_max) = world_box(a);
            for b in &chunks[index + 1..] {
                if a.lod_index == b.lod_index {
                    continue;
                }
                let (b_min, b_max) = world_box(b);
                let overlaps = (0..3).all(|axis| {
                    a_min[axis] < b_max[axis] && b_min[axis] < a_max[axis]
                });
                assert!(
                    !overlaps,
                    "chunk {:?} lod {} overlaps {:?} lod {}",
                    a.position, a.lod_index, b.position, b.lod_index
                );
            }
        }
    }

    #[test]
    fn the_rings_leave_no_gap_between_levels() {
        // A gap would show as missing terrain at the ring boundary
        let s = VoxelWorldSettings {
            chunk_size: 16,
            view_distance_in_chunks: 2,
            lod_count: 3,
            ..settings()
        };
        let chunks = chunks_in_view(&s, Vec3::ZERO);
        let loaded: std::collections::HashSet<ChunkPosition> =
            chunks.iter().copied().collect();

        // Sample the whole covered volume, every point must belong to exactly
        // one loaded chunk
        let outer_reach = lod_reach_in_world_units(&s, s.lod_count - 1);
        let step = s.chunk_size as f32;
        let samples = (outer_reach / step) as i32;
        for iz in -samples..samples {
            for iy in -samples..samples {
                for ix in -samples..samples {
                    let point = Vec3::new(
                        ix as f32 * step + step * 0.5,
                        iy as f32 * step + step * 0.5,
                        iz as f32 * step + step * 0.5,
                    );
                    let count = (0..s.lod_count)
                        .filter(|lod_index| {
                            let size = s.chunk_world_size(*lod_index);
                            let position = (point / size).floor().as_ivec3();
                            loaded.contains(&ChunkPosition::new(position, *lod_index))
                        })
                        .count();
                    assert_eq!(count, 1, "point {point:?} covered {count} times");
                }
            }
        }
    }

    #[test]
    fn every_probed_neighbour_lod_matches_a_loaded_chunk() {
        // The transition mask is only correct when the level reported for a
        // neighbour position is the level that position is actually loaded at
        let s = VoxelWorldSettings {
            chunk_size: 16,
            view_distance_in_chunks: 4,
            lod_count: 3,
            ..settings()
        };
        let viewer = Vec3::ZERO;
        let loaded: std::collections::HashSet<ChunkPosition> =
            chunks_in_view(&s, viewer).into_iter().collect();

        for chunk in &loaded {
            let world_size = s.chunk_world_size(chunk.lod_index);
            let centre = (chunk.position.as_vec3() + Vec3::splat(0.5)) * world_size;

            for normal in voxel_core::cube::SIDE_NORMALS.iter() {
                let probe = centre + normal.as_vec3() * world_size;
                let predicted = lod_at_world_position(&s, probe, viewer);

                // The chunk that actually holds this position, if any
                let actual = (0..s.lod_count).find(|lod_index| {
                    let size = s.chunk_world_size(*lod_index);
                    let position = (probe / size).floor().as_ivec3();
                    loaded.contains(&ChunkPosition::new(position, *lod_index))
                });

                if let Some(actual) = actual {
                    assert_eq!(
                        predicted, actual,
                        "probe {probe:?} from chunk {:?} lod {}",
                        chunk.position, chunk.lod_index
                    );
                }
            }
        }
    }

    #[test]
    fn a_single_lod_level_produces_no_transitions() {
        let s = VoxelWorldSettings {
            view_distance_in_chunks: 4,
            lod_count: 1,
            ..settings()
        };
        for x in -4..=4 {
            let mask = transition_mask_for_chunk(
                &s,
                ChunkPosition::new(IVec3::new(x, 0, 0), 0),
                Vec3::ZERO,
            );
            assert_eq!(mask, 0, "at x {x}");
        }
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
        let wanted: std::collections::HashSet<ChunkPosition> = [keep].into_iter().collect();
        let surplus = loaded.surplus(&wanted);
        assert_eq!(surplus, vec![drop]);
    }

    #[test]
    fn a_chunk_changing_lod_counts_as_a_different_chunk() {
        let mut loaded = LoadedChunks::default();
        let fine = ChunkPosition::new(IVec3::ZERO, 0);
        let coarse = ChunkPosition::new(IVec3::ZERO, 1);
        loaded.insert(fine, Entity::from_raw_u32(1).unwrap());
        assert!(!loaded.contains(coarse));
        let wanted: std::collections::HashSet<ChunkPosition> = [coarse].into_iter().collect();
        assert_eq!(loaded.surplus(&wanted), vec![fine]);
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
