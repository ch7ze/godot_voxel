//! Engine-neutral hybrid blocky and SDF voxel mesher.
//!
//! Ported from the C++ module `godot_voxel` (MIT, Copyright 2016-2024
//! Marc Gilleron). The Transvoxel lookup tables additionally carry Eric
//! Lengyel's notice, see `transvoxel::tables`.
//!
//! This crate depends on neither Bevy nor Godot.

pub mod blocky;
pub mod hybrid;
pub mod transvoxel;

pub use blocky::baked_library::BakedLibrary;
pub use blocky::library_desc::{bake, LibraryDesc, LibraryError};
pub use hybrid::output::{
    CollisionSurface, LodAttrib, MeshOutput, Surface, MODEL_MATERIAL_INDEX_BASE,
    TERRAIN_MATERIAL_INDEX, WATER_MATERIAL_INDEX,
};
pub use hybrid::{build, HybridParams};
pub use transvoxel::materials::TexturingMode;
pub use transvoxel::{build_mesh, MeshArrays, TransvoxelOutput, TransvoxelParams};
