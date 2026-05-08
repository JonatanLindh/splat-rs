use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use wgpu::util::{BufferInitDescriptor, DeviceExt};

use crate::{
    camera::Camera,
    gpu::GpuContext,
    radix_sort_cpu::{self, RadixSort},
    shaders::{
        self,
        splat_common::{
            CameraUniform, GpuSplat, WgpuBindGroup0, WgpuBindGroup0Entries,
            WgpuBindGroup0EntriesParams,
        },
    },
};

pub struct SplatRenderer {
    pipeline: wgpu::RenderPipeline,
    bind_group: WgpuBindGroup0,
    camera_buf: wgpu::Buffer,
    splat_buf: wgpu::Buffer,
    splat_count: u32,
    stochastic_transparency: bool,

    /// unsorted GPU splats
    splats: Vec<GpuSplat>,
    /// scratch for sorted gpu splats
    sorted_splats: Vec<GpuSplat>,
    /// scratch for depth-sorting indices
    depth_indices: Vec<(u32, usize)>,
}

impl SplatRenderer {
    pub fn new(
        ctx: &GpuContext,
        splats: Vec<GpuSplat>,
        stochastic_transparency: bool,
        msaa_samples: u32,
    ) -> Self {
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
            contents: bytemuck::cast_slice(&splats),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // Shader & pipeline
        let (se, shader) = if stochastic_transparency {
            let se = shaders::ShaderEntry::SplatStochastic;
            let shader = se.create_shader_module_embed_source(&ctx.device);
            (se, shader)
        } else {
            let se = shaders::ShaderEntry::Splat;
            let shader = se.create_shader_module_embed_source(&ctx.device);
            (se, shader)
        };

        let pipeline_layout = se.create_pipeline_layout(&ctx.device);

        let pipeline = if stochastic_transparency {
            ctx.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("Splat Pipeline (Stochastic)"),
                    layout: Some(&pipeline_layout),
                    vertex: shaders::splat_stochastic::vertex_state(
                        &shader,
                        &shaders::splat_stochastic::vs_main_entry(),
                    ),
                    fragment: Some(shaders::splat_stochastic::fragment_state(
                        &shader,
                        &shaders::splat_stochastic::fs_main_entry([Some(wgpu::ColorTargetState {
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
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: wgpu::TextureFormat::Depth32Float,
                        depth_write_enabled: Some(true),
                        depth_compare: Some(wgpu::CompareFunction::Less),
                        stencil: wgpu::StencilState::default(),
                        bias: wgpu::DepthBiasState::default(),
                    }),
                    multisample: wgpu::MultisampleState {
                        count: msaa_samples,
                        mask: !0,
                        alpha_to_coverage_enabled: true,
                    },
                    multiview_mask: None,
                    cache: None,
                })
        } else {
            ctx.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("Splat Pipeline (Standard)"),
                    layout: Some(&pipeline_layout),
                    vertex: shaders::splat::vertex_state(&shader, &shaders::splat::vs_main_entry()),
                    fragment: Some(shaders::splat::fragment_state(
                        &shader,
                        &shaders::splat::fs_main_entry([Some(wgpu::ColorTargetState {
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
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    multiview_mask: None,
                    cache: None,
                })
        };

        // Bind group
        let bind_group = WgpuBindGroup0::from_bindings(
            &ctx.device,
            WgpuBindGroup0Entries::new(WgpuBindGroup0EntriesParams {
                camera: camera_buf.as_entire_buffer_binding(),
                splats: splat_buf.as_entire_buffer_binding(),
            }),
        );

        let splat_count = splats.len() as u32;
        let sorted_splats = Vec::with_capacity(splats.len());
        let depth_indices = Vec::with_capacity(splats.len());

        Self {
            pipeline,
            bind_group,
            camera_buf,
            splat_buf,
            splat_count,
            stochastic_transparency,
            splats,
            sorted_splats,
            depth_indices,
        }
    }

    pub fn prepare(&mut self, ctx: &GpuContext, camera: &Camera, width: u32, height: u32) {
        let cam_uniform = CameraUniform::from_camera(camera, width, height);
        ctx.queue
            .write_buffer(&self.camera_buf, 0, bytemuck::cast_slice(&[cam_uniform]));

        if self.stochastic_transparency {
            return;
        }

        let view = camera.view_matrix();
        let z_row = view.row(2).truncate();

        self.depth_indices.clear();
        self.splats
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
                .map(|&(_, original_idx)| self.splats[original_idx]),
        );

        ctx.queue.write_buffer(
            &self.splat_buf,
            0,
            bytemuck::cast_slice(&self.sorted_splats),
        );
    }

    /// Rebuild the pipeline with a new shader module (for hot-reloading).
    pub fn rebuild_pipeline(&mut self, ctx: &GpuContext, shader: &wgpu::ShaderModule) {
        tracing::info!(
            "Rebuilding pipeline (mode: {})",
            if self.stochastic_transparency {
                "stochastic"
            } else {
                "standard"
            }
        );

        let se = if self.stochastic_transparency {
            shaders::ShaderEntry::SplatStochastic
        } else {
            shaders::ShaderEntry::Splat
        };

        let pipeline_layout = se.create_pipeline_layout(&ctx.device);

        self.pipeline = if self.stochastic_transparency {
            ctx.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("Splat Pipeline (Stochastic)"),
                    layout: Some(&pipeline_layout),
                    vertex: shaders::splat_stochastic::vertex_state(
                        shader,
                        &shaders::splat_stochastic::vs_main_entry(),
                    ),
                    fragment: Some(shaders::splat_stochastic::fragment_state(
                        shader,
                        &shaders::splat_stochastic::fs_main_entry([Some(wgpu::ColorTargetState {
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
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: wgpu::TextureFormat::Depth32Float,
                        depth_write_enabled: Some(true),
                        depth_compare: Some(wgpu::CompareFunction::Less),
                        stencil: wgpu::StencilState::default(),
                        bias: wgpu::DepthBiasState::default(),
                    }),
                    multisample: wgpu::MultisampleState {
                        count: 4,
                        mask: !0,
                        alpha_to_coverage_enabled: true,
                    },
                    multiview_mask: None,
                    cache: None,
                })
        } else {
            ctx.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("Splat Pipeline (Standard)"),
                    layout: Some(&pipeline_layout),
                    vertex: shaders::splat::vertex_state(shader, &shaders::splat::vs_main_entry()),
                    fragment: Some(shaders::splat::fragment_state(
                        shader,
                        &shaders::splat::fs_main_entry([Some(wgpu::ColorTargetState {
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
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    multiview_mask: None,
                    cache: None,
                })
        };
    }

    /// Record the splat draw call into an active render pass.
    pub fn render<'p>(&'p self, rpass: &mut wgpu::RenderPass<'p>) {
        rpass.set_pipeline(&self.pipeline);
        self.bind_group.set(rpass);
        rpass.draw(0..self.splat_count * 6, 0..1);
    }
}
