//! Engine-neutral voxel storage, material packing and SDF editing.
//!
//! Ported from the C++ module `godot_voxel` (MIT, Copyright 2016-2024
//! Marc Gilleron). Each item carries the file and line range it came from.
//!
//! This crate depends on neither Bevy nor Godot.

pub mod buffer;
pub mod channel;
pub mod constants;
pub mod cube;
pub mod funcs;
pub mod generator;
pub mod math;
pub mod mixel4;
pub mod sdf_ops;

pub use buffer::{BufferError, Channel, ChannelData, VoxelBuffer, VoxelFormat};
pub use channel::{ChannelId, Depth};
pub use generator::{FlatGenerator, GenerateResult, VoxelGenerator};
pub use math::Box3i;
pub use sdf_ops::{
    do_box, do_point, do_sphere, raycast_sdf, voxel_raycast, EditMode, RaycastHit, RaycastState,
};
