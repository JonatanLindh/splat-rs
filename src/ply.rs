use std::{fs::File, io::BufReader, path::Path};

use glam::{Quat, Vec3};
use seq_macro::seq;
use serde::Deserialize;

seq!(N in 0..=44 {
    /// One Gaussian splat as stored in a 3DGS `.ply` file.
    #[derive(Debug, Clone, Deserialize)]
    pub struct PlyGaussian {
        // Position
        pub x: f32,
        pub y: f32,
        pub z: f32,

        #[serde(default)]
        pub nx: f32,
        #[serde(default)]
        pub ny: f32,
        #[serde(default)]
        pub nz: f32,

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

/// Load all Gaussian splats from a binary PLY file.
pub fn load_splats(path: &Path) -> color_eyre::Result<Vec<PlyGaussian>> {
    let file = BufReader::new(File::open(path)?);
    let mut reader = serde_ply::PlyReader::from_reader(file)?;
    let splats: Vec<PlyGaussian> = reader.next_element()?;
    Ok(splats)
}

/// Maximum splats sent to Rerun in one call
const MAX_RERUN_SPLATS: usize = 200_000;

/// Log splats to a Rerun recording stream as `Ellipsoids3D`.
pub fn log_splats_to_rerun(rec: &rerun::RecordingStream, splats: &[PlyGaussian]) {
    // SH DC → linear RGB: color = 0.5 + SH_C0 * f_dc,  SH_C0 = 1/sqrt(4π)
    #[allow(clippy::excessive_precision)]
    const SH_C0: f32 = 0.28209479177387814_f32;

    let step = (splats.len() / MAX_RERUN_SPLATS).max(1);
    let sampled: Vec<&PlyGaussian> = splats.iter().step_by(step).collect();

    if step > 1 {
        eprintln!(
            "Rerun: subsampling {total} splats to {shown} (every {step}th)",
            total = splats.len(),
            shown = sampled.len(),
        );
    }

    let centers: Vec<[f32; 3]> = sampled.iter().map(|s| [s.x, s.y, s.z]).collect();

    let half_sizes: Vec<[f32; 3]> = sampled
        .iter()
        .map(|s| [s.scale_0.exp(), s.scale_1.exp(), s.scale_2.exp()])
        .collect();

    // 3DGS stores quaternion as (w, x, y, z) = (rot_0, rot_1, rot_2, rot_3).
    // Rerun expects (x, y, z, w).
    let quaternions: Vec<[f32; 4]> = sampled
        .iter()
        .map(|s| [s.rot_1, s.rot_2, s.rot_3, s.rot_0])
        .collect();

    let colors: Vec<rerun::Color> = sampled
        .iter()
        .map(|s| {
            let r = ((0.5 + SH_C0 * s.f_dc_0).clamp(0.0, 1.0) * 255.0) as u8;
            let g = ((0.5 + SH_C0 * s.f_dc_1).clamp(0.0, 1.0) * 255.0) as u8;
            let b = ((0.5 + SH_C0 * s.f_dc_2).clamp(0.0, 1.0) * 255.0) as u8;
            rerun::Color::from_rgb(r, g, b)
        })
        .collect();

    rec.log(
        "splats",
        &rerun::Ellipsoids3D::from_centers_and_half_sizes(centers, half_sizes)
            .with_quaternions(quaternions)
            .with_colors(colors)
            .with_fill_mode(rerun::FillMode::Solid),
    )
    .ok();
}
