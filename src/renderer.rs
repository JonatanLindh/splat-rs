use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use wgpu::util::{BufferInitDescriptor, DeviceExt};

use crate::{
    camera::Camera,
    gpu::GpuContext,
    ply::PlyGaussian,
    radix_sort_cpu::{self, RadixSort},
    shader_bindings::{
        self,
        splat::{
            CameraUniform, GpuSplat, GpuSplatInit, WgpuBindGroup0, WgpuBindGroup0Entries,
            WgpuBindGroup0EntriesParams,
        },
    },
};

impl From<&PlyGaussian> for GpuSplat {
    fn from(s: &PlyGaussian) -> Self {
        GpuSplatInit {
            position: s.position(),
            opacity: s.opacity,
            scale: s.log_scale(),
            rotation: s.rotation().into(),
            sh_dc: s.sh_dc(),
            sh_rest: s.sh_rest(),
        }
        .build()
    }
}

pub struct SplatRenderer {
    pipeline: wgpu::RenderPipeline,
    bind_group: WgpuBindGroup0,
    camera_buf: wgpu::Buffer,
    splat_buf: wgpu::Buffer,
    splat_count: u32,

    /// unsorted GPU splats
    gpu_splats: Vec<GpuSplat>,
    /// scratch for sorted gpu splats
    sorted_splats: Vec<GpuSplat>,
    /// scratch for depth-sorting indices
    depth_indices: Vec<(u32, usize)>,
}

impl SplatRenderer {
    pub fn new(ctx: &GpuContext, splats: &[PlyGaussian]) -> Self {
        let gpu_splats: Vec<GpuSplat> = splats.iter().map(GpuSplat::from).collect();

        // Camera uniform buffer
        let camera_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Camera Uniform"),
            size: size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Splat storage buffer
        let splat_buf = ctx.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Splat Storage"),
            contents: bytemuck::cast_slice(&gpu_splats),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // Shader & pipeline
        let se = shader_bindings::ShaderEntry::Splat;
        let shader = se.create_shader_module_embed_source(&ctx.device);
        let pipeline_layout = se.create_pipeline_layout(&ctx.device);

        let pipeline = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Splat Pipeline"),
                layout: Some(&pipeline_layout),
                vertex: shader_bindings::splat::vertex_state(
                    &shader,
                    &shader_bindings::splat::vs_main_entry(),
                ),
                fragment: Some(shader_bindings::splat::fragment_state(
                    &shader,
                    &shader_bindings::splat::fs_main_entry([Some(wgpu::ColorTargetState {
                        format: ctx.surface_format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })]),
                )),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: None, // painter's algorithm — no depth test
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });

        // Bind group
        let bind_group = WgpuBindGroup0::from_bindings(
            &ctx.device,
            WgpuBindGroup0Entries::new(WgpuBindGroup0EntriesParams {
                camera: camera_buf.as_entire_buffer_binding(),
                splats: splat_buf.as_entire_buffer_binding(),
            }),
        );

        let splat_count = gpu_splats.len() as u32;
        let sorted_splats = Vec::with_capacity(gpu_splats.len());
        let depth_indices = Vec::with_capacity(gpu_splats.len());

        Self {
            pipeline,
            bind_group,
            camera_buf,
            splat_buf,
            splat_count,
            gpu_splats,
            sorted_splats,
            depth_indices,
        }
    }

    pub fn prepare(&mut self, ctx: &GpuContext, camera: &Camera, width: u32, height: u32) {
        let cam_uniform = CameraUniform::from_camera(camera, width, height);
        ctx.queue
            .write_buffer(&self.camera_buf, 0, bytemuck::cast_slice(&[cam_uniform]));

        let view = camera.view_matrix();
        let z_row = view.row(2).truncate();

        self.depth_indices.clear();
        self.gpu_splats
            .par_iter()
            .enumerate()
            .map(|(i, splat)| {
                let depth = z_row.dot(splat.position);
                (radix_sort_cpu::f32_sortable_bits(depth), i)
            })
            .collect_into_vec(&mut self.depth_indices);

        self.depth_indices.par_radix_sort_unstable();

        self.sorted_splats.clear();
        self.sorted_splats.extend(
            self.depth_indices
                .iter()
                .map(|&(_, original_idx)| self.gpu_splats[original_idx]),
        );

        ctx.queue.write_buffer(
            &self.splat_buf,
            0,
            bytemuck::cast_slice(&self.sorted_splats),
        );
    }

    /// Record the splat draw call into an active render pass.
    pub fn render<'p>(&'p self, rpass: &mut wgpu::RenderPass<'p>) {
        rpass.set_pipeline(&self.pipeline);
        self.bind_group.set(rpass);
        rpass.draw(0..self.splat_count * 6, 0..1);
    }
}
