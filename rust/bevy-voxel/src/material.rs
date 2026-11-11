//! Voxel terrain material.
//!
//! Replaces `get_default_lod_material` of the C++ mesher
//! (`meshers/hybrid/voxel_mesher_hybrid.h:58-68`). The shader handles the
//! Transvoxel LOD transitions, so transition triangles are culled instead of
//! showing as artifacts, and picks a per-vertex colour from the material id.

use bevy::asset::{Asset, Handle};
use bevy::image::Image;
use bevy::pbr::{Material, MaterialPlugin};
use bevy::prelude::*;
use bevy::render::render_resource::{AsBindGroup, ShaderType};
use bevy::shader::ShaderRef;

use crate::mesh::{ATTRIBUTE_LOD_DATA, ATTRIBUTE_MATERIAL_ID};

/// Path of the shader inside the asset folder.
pub const SHADER_PATH: &str = "shaders/voxel_terrain.wgsl";

/// Number of colour slots the shader holds.
/// Source: shaders/voxel_mesher_hybrid_shader.h:10
pub const MATERIAL_COLOR_COUNT: usize = 8;

/// Default colour of the smooth terrain surface.
/// Source: meshers/hybrid/voxel_mesher_hybrid.h:167
pub const DEFAULT_TERRAIN_COLOR: LinearRgba = LinearRgba::rgb(0.36, 0.62, 0.30);

/// Uniform block the shader reads.
#[derive(Clone, Copy, Debug, ShaderType)]
pub struct VoxelTerrainUniform {
    /// One colour per surface material index.
    /// Source: meshers/hybrid/voxel_mesher_hybrid.h:88-98
    pub material_colors: [Vec4; MATERIAL_COLOR_COUNT],
    /// Bit mask of the sides stitched to a coarser neighbour chunk.
    /// Source: shaders/voxel_mesher_hybrid_shader.h:8
    pub transition_mask: u32,
    pub roughness: f32,
    pub _padding: Vec2,
}

impl Default for VoxelTerrainUniform {
    fn default() -> VoxelTerrainUniform {
        let mut material_colors = [Vec4::ONE; MATERIAL_COLOR_COUNT];
        material_colors[0] = DEFAULT_TERRAIN_COLOR.to_vec4();
        VoxelTerrainUniform {
            material_colors,
            transition_mask: 0,
            // Source: shaders/voxel_mesher_hybrid_shader.h:12
            roughness: 1.0,
            _padding: Vec2::ZERO,
        }
    }
}

/// Material rendering a voxel chunk.
#[derive(Asset, AsBindGroup, Clone, Debug, TypePath)]
pub struct VoxelTerrainMaterial {
    #[uniform(0)]
    pub uniform: VoxelTerrainUniform,
    /// Optional texture array for the Mixel4 material path. Unused by the
    /// default shader, kept so a project shader can bind it.
    #[texture(1, dimension = "2d_array")]
    #[sampler(2)]
    pub texture_array: Option<Handle<Image>>,
    /// Renders the water surface transparently.
    pub alpha_mode: AlphaMode,
}

impl Default for VoxelTerrainMaterial {
    fn default() -> VoxelTerrainMaterial {
        VoxelTerrainMaterial {
            uniform: VoxelTerrainUniform::default(),
            texture_array: None,
            alpha_mode: AlphaMode::Opaque,
        }
    }
}

impl VoxelTerrainMaterial {
    /// Material for the opaque terrain and the models.
    pub fn terrain(terrain_color: LinearRgba, model_colors: &[LinearRgba]) -> VoxelTerrainMaterial {
        let mut material = VoxelTerrainMaterial::default();
        material.set_colors(terrain_color, model_colors);
        material
    }

    /// Material for the transparent water surface.
    /// Source: meshers/voxel_mesher.h:115-123
    pub fn water(color: LinearRgba) -> VoxelTerrainMaterial {
        let mut material = VoxelTerrainMaterial {
            alpha_mode: AlphaMode::Blend,
            ..VoxelTerrainMaterial::default()
        };
        // The water surface carries material id 1
        material.uniform.material_colors[voxel_mesher::WATER_MATERIAL_INDEX as usize] =
            color.to_vec4();
        material
    }

    /// Fills the colour slots. Slot 0 is the terrain, slots 1 and above take
    /// `model_colors` in order, matching the C++ convention where slot 1 is
    /// water and slots 2 and above are the library materials.
    /// Source: meshers/hybrid/voxel_mesher_hybrid.h:88-98
    pub fn set_colors(&mut self, terrain_color: LinearRgba, model_colors: &[LinearRgba]) {
        self.uniform.material_colors[0] = terrain_color.to_vec4();
        for slot in 1..MATERIAL_COLOR_COUNT {
            let color = model_colors
                .get(slot - 1)
                .copied()
                .unwrap_or(LinearRgba::WHITE);
            self.uniform.material_colors[slot] = color.to_vec4();
        }
    }

    /// Sets which chunk sides are stitched to a coarser neighbour.
    pub fn set_transition_mask(&mut self, mask: u8) {
        self.uniform.transition_mask = mask as u32;
    }
}

impl Material for VoxelTerrainMaterial {
    fn vertex_shader() -> ShaderRef {
        SHADER_PATH.into()
    }

    fn fragment_shader() -> ShaderRef {
        SHADER_PATH.into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        self.alpha_mode
    }

    fn specialize(
        _pipeline: &bevy::pbr::MaterialPipeline,
        descriptor: &mut bevy::render::render_resource::RenderPipelineDescriptor,
        layout: &bevy::mesh::MeshVertexBufferLayoutRef,
        _key: bevy::pbr::MaterialPipelineKey<Self>,
    ) -> Result<(), bevy::render::render_resource::SpecializedMeshPipelineError> {
        // The shader reads position, normal, the LOD data and the material id,
        // at the locations the WGSL declares
        let vertex_layout = layout.0.get_layout(&[
            bevy::mesh::Mesh::ATTRIBUTE_POSITION.at_shader_location(0),
            bevy::mesh::Mesh::ATTRIBUTE_NORMAL.at_shader_location(1),
            ATTRIBUTE_LOD_DATA.at_shader_location(8),
            ATTRIBUTE_MATERIAL_ID.at_shader_location(10),
        ])?;
        descriptor.vertex.buffers = vec![vertex_layout];
        Ok(())
    }
}

/// Registers the material with an app.
pub struct VoxelMaterialPlugin;

impl Plugin for VoxelMaterialPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<VoxelTerrainMaterial>::default());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_colour_slot_count_matches_the_shader() {
        // Source: shaders/voxel_mesher_hybrid_shader.h:10
        assert_eq!(MATERIAL_COLOR_COUNT, 8);
    }

    #[test]
    fn the_default_terrain_colour_matches_cpp() {
        // Source: meshers/hybrid/voxel_mesher_hybrid.h:167
        assert_eq!(DEFAULT_TERRAIN_COLOR.red, 0.36);
        assert_eq!(DEFAULT_TERRAIN_COLOR.green, 0.62);
        assert_eq!(DEFAULT_TERRAIN_COLOR.blue, 0.30);
    }

    #[test]
    fn a_default_material_starts_opaque_with_the_terrain_colour() {
        let m = VoxelTerrainMaterial::default();
        assert_eq!(m.alpha_mode, AlphaMode::Opaque);
        assert_eq!(m.uniform.material_colors[0], DEFAULT_TERRAIN_COLOR.to_vec4());
        assert_eq!(m.uniform.transition_mask, 0);
        assert_eq!(m.uniform.roughness, 1.0);
    }

    #[test]
    fn colours_fill_slot_zero_then_the_model_colours() {
        let mut m = VoxelTerrainMaterial::default();
        let water = LinearRgba::rgb(0.0, 0.3, 0.8);
        let stone = LinearRgba::rgb(0.5, 0.5, 0.5);
        m.set_colors(LinearRgba::RED, &[water, stone]);
        assert_eq!(m.uniform.material_colors[0], LinearRgba::RED.to_vec4());
        // Slot 1 is water, slot 2 the first library material
        assert_eq!(m.uniform.material_colors[1], water.to_vec4());
        assert_eq!(m.uniform.material_colors[2], stone.to_vec4());
    }

    #[test]
    fn unspecified_colour_slots_fall_back_to_white() {
        let m = VoxelTerrainMaterial::terrain(LinearRgba::RED, &[]);
        for slot in 1..MATERIAL_COLOR_COUNT {
            assert_eq!(
                m.uniform.material_colors[slot],
                LinearRgba::WHITE.to_vec4(),
                "slot {slot}"
            );
        }
    }

    #[test]
    fn more_colours_than_slots_are_truncated() {
        let colors = vec![LinearRgba::BLUE; 20];
        let m = VoxelTerrainMaterial::terrain(LinearRgba::RED, &colors);
        assert_eq!(m.uniform.material_colors.len(), MATERIAL_COLOR_COUNT);
    }

    #[test]
    fn the_water_material_blends_and_uses_slot_one() {
        let color = LinearRgba::new(0.0, 0.3, 0.8, 0.5);
        let m = VoxelTerrainMaterial::water(color);
        assert_eq!(m.alpha_mode, AlphaMode::Blend);
        assert_eq!(
            m.uniform.material_colors[voxel_mesher::WATER_MATERIAL_INDEX as usize],
            color.to_vec4()
        );
    }

    #[test]
    fn the_transition_mask_reaches_the_uniform() {
        let mut m = VoxelTerrainMaterial::default();
        m.set_transition_mask(0b101010);
        assert_eq!(m.uniform.transition_mask, 0b101010);
    }

    #[test]
    fn the_shader_path_points_at_the_wgsl_file() {
        assert_eq!(SHADER_PATH, "shaders/voxel_terrain.wgsl");
    }
}
