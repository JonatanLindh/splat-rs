use std::{fs::File, io::BufReader, path::Path};

use glam::{Quat, Vec3};
use seq_macro::seq;
use serde::Deserialize;

use crate::shaders::splat_common::{GpuSplat, GpuSplatInit};

seq!(N in 0..=44 {
    /// One Gaussian splat as stored in a 3DGS `.ply` file.
    #[derive(Debug, Clone, Deserialize)]
    struct PlyGaussian {
        // Position
        pub x: f32,
        pub y: f32,
        pub z: f32,

        // DC spherical-harmonics coefficients (one per RGB channel)
        pub f_dc_0: f32,
        pub f_dc_1: f32,
        pub f_dc_2: f32,

        // Higher-order SH rest coefficients, degree 1–3 (15 per channel × 3 channels = 45).
        // All default to 0.0 so the struct accepts files with any SH degree.
        #(
            #[serde(default)]
            pub f_rest_~N : f32,
        )*

        // Pre-sigmoid opacity
        pub opacity: f32,

        // Log-space scale
        pub scale_0: f32,
        pub scale_1: f32,
        pub scale_2: f32,

        // Rotation quaternion stored as (w, x, y, z)
        pub rot_0: f32,
        pub rot_1: f32,
        pub rot_2: f32,
        pub rot_3: f32,
    }
});

impl PlyGaussian {
    pub fn position(&self) -> Vec3 {
        Vec3::new(self.x, self.y, self.z)
    }

    pub fn rotation(&self) -> Quat {
        Quat::from_xyzw(self.rot_1, self.rot_2, self.rot_3, self.rot_0)
    }

    pub fn log_scale(&self) -> Vec3 {
        Vec3::new(self.scale_0, self.scale_1, self.scale_2)
    }

    pub fn sh_dc(&self) -> Vec3 {
        Vec3::new(self.f_dc_0, self.f_dc_1, self.f_dc_2)
    }

    pub fn sh_rest(&self) -> [f32; 45] {
        // 0..=14 red, 15..=29 green, 30..=44 blue
        seq!(N in 0..=44 {
            [ #( self.f_rest_~N, )* ]
        })
    }
}

impl From<PlyGaussian> for GpuSplat {
    fn from(s: PlyGaussian) -> Self {
        let sign = Vec3::new(1., -1., -1.);
        let position = s.position() * sign;

        let rotation = {
            let q = s.rotation();
            Quat::from_xyzw(
                q.x * sign.x,
                q.y * sign.y,
                q.z * sign.z,
                q.w * sign.element_product(),
            )
            .normalize()
        };

        let flip_indices = [0, 1, 3, 6, 8, 10, 11, 13];
        let mut sh_rest = s.sh_rest();

        // Flip the coefficient for Red, Green, and Blue planes
        for &base_idx in &flip_indices {
            for color_offset in [0, 15, 30] {
                let idx = base_idx + color_offset;
                if idx < sh_rest.len() {
                    sh_rest[idx] = -sh_rest[idx];
                }
            }
        }

        GpuSplatInit {
            position,
            opacity: s.opacity,
            scale: s.log_scale(),
            rotation: rotation.into(),
            sh_dc: s.sh_dc(),
            sh_rest,
        }
        .build()
    }
}

/// Load all Gaussian splats from a binary PLY file.
pub fn load_splats(path: &Path) -> color_eyre::Result<Vec<GpuSplat>> {
    let file = BufReader::new(File::open(path)?);
    let mut reader = serde_ply::PlyReader::from_reader(file)?;
    let splats: Vec<PlyGaussian> = reader.next_element()?;

    Ok(splats.into_iter().map(GpuSplat::from).collect())
}
