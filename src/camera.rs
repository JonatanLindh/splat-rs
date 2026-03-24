use glam::{Mat4, Vec2, Vec3};

use crate::shader_bindings::splat::{CameraUniform, CameraUniformInit};

/// Controls:
/// - Right-click drag  :  look (yaw / pitch)
/// - W / S             :  fly forward / backward
/// - A / D             :  strafe left / right
/// - E / Space         :  fly up
/// - Q / LShift        :  fly down
/// - Scroll            :  Zoom (ish, moves camera forward/backward)
pub struct Camera {
    /// Eye position in world space.
    pub position: Vec3,
    /// Horizontal look angle in radians.
    pub yaw: f32,
    /// Vertical look angle in radians, clamped away from the poles.
    pub pitch: f32,
    /// Vertical field-of-view in radians.
    pub fov_y: f32,
    pub near: f32,
    pub far: f32,
    /// Movement speed in world-units per second.
    pub move_speed: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            position: Vec3::new(0.0, 0.0, 5.0),
            yaw: std::f32::consts::PI, // look toward -Z (into the scene)
            pitch: 0.0,
            fov_y: 45_f32.to_radians(),
            near: 0.01,
            far: 1000.0,
            move_speed: 15.0,
        }
    }
}

impl Camera {
    /// Unit vector pointing in the direction the camera is looking.
    pub fn forward(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        Vec3::new(cp * sy, sp, -cp * cy)
    }

    /// Horizontal right vector
    pub fn right(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        Vec3::new(cy, 0.0, sy)
    }

    pub fn view_matrix(&self) -> Mat4 {
        let flip_y = Mat4::from_scale(Vec3::new(1.0, -1.0, 1.0));
        Mat4::look_at_rh(self.position, self.position + self.forward(), Vec3::Y) * flip_y
    }

    pub fn proj_matrix(&self, width: u32, height: u32) -> Mat4 {
        Mat4::perspective_rh(
            self.fov_y,
            width as f32 / height as f32,
            self.near,
            self.far,
        )
    }

    /// Rotate the look direction
    pub fn look(&mut self, dx: f32, dy: f32) {
        const SENSITIVITY: f32 = 0.003;
        self.yaw += dx * SENSITIVITY;
        self.pitch = (self.pitch - dy * SENSITIVITY).clamp(
            -std::f32::consts::FRAC_PI_2 + 0.01,
            std::f32::consts::FRAC_PI_2 - 0.01,
        );
    }
}

impl CameraUniform {
    pub fn from_camera(camera: &Camera, width: u32, height: u32) -> Self {
        CameraUniformInit {
            view: camera.view_matrix(),
            proj: camera.proj_matrix(width, height),
            viewport: Vec2::new(width as f32, height as f32),
            camera_pos: camera.position,
        }
        .build()
    }
}
