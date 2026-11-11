//! Bevy plugin exposing the hybrid voxel mesher.
//!
//! Ported from the C++ module `godot_voxel` (MIT, Copyright 2016-2024
//! Marc Gilleron). The meshing itself lives in `voxel-mesher`, this crate only
//! bridges it to Bevy.

pub mod chunk;
pub mod material;
pub mod mesh;

pub use chunk::{
    ChunkPosition, ChunkState, ChunkSurface, LoadedChunks, TransitionMask, VoxelChunkPlugin,
    VoxelWorldSettings, VoxelWorldSources, VoxelWorldViewer,
};
pub use material::{VoxelMaterialPlugin, VoxelTerrainMaterial, VoxelTerrainUniform};
pub use mesh::{
    output_to_meshes, output_to_transition_meshes, split_terrain_and_water, surface_to_mesh,
    ChunkMesh, ATTRIBUTE_LOD_DATA, ATTRIBUTE_MATERIAL_ID, ATTRIBUTE_TEXTURING_DATA,
};
pub use voxel_core::generator::{FlatGenerator, GenerateResult, VoxelGenerator};
pub use voxel_core::{Box3i, ChannelId, Depth, VoxelBuffer, VoxelFormat};
pub use voxel_mesher::{
    build, BakedLibrary, HybridParams, LibraryDesc, MeshOutput, Surface, TexturingMode,
};
