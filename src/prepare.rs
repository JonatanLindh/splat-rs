use crate::{
    gpu::GpuContext,
    shaders::{
        prepare::{
            PushConstants, WG_SIZE, WgpuBindGroup0, WgpuBindGroup0Entries,
            WgpuBindGroup0EntriesParams, compute::create_cs_prepare_pipeline_embed_source,
        },
        splat_common::CameraUniform,
    },
};

pub struct Preparer {
    // Pipelines
    pipeline: wgpu::ComputePipeline,
    bind_group: WgpuBindGroup0,
}

impl Preparer {
    pub fn new(
        GpuContext { device, .. }: &GpuContext,
        depth_buffer: wgpu::Buffer,
        index_buffer: wgpu::Buffer,
        splat_buffer: wgpu::Buffer,
    ) -> Self {
        let bind_group = WgpuBindGroup0::from_bindings(
            device,
            WgpuBindGroup0Entries::new(WgpuBindGroup0EntriesParams {
                u32_depths: depth_buffer.as_entire_buffer_binding(),
                u32_indices: index_buffer.as_entire_buffer_binding(),
                splats: splat_buffer.as_entire_buffer_binding(),
            }),
        );

        // Create the 3 pipelines mapped to your Slang entry points
        let pipeline = create_cs_prepare_pipeline_embed_source(device);

        Self {
            bind_group,
            pipeline,
        }
    }

    pub fn run(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        num_elements: u32,
        camera_uniform: &CameraUniform,
    ) {
        if num_elements == 0 {
            return;
        }

        let push_constants = PushConstants(camera_uniform.view.row(2), num_elements);

        let blocks = num_elements.div_ceil(WG_SIZE);

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Prepare depths pass"),
                timestamp_writes: None,
            });

            cpass.set_pipeline(&self.pipeline);
            cpass.set_immediates(0, bytemuck::cast_slice(&[push_constants]));
            self.bind_group.set(&mut cpass);

            cpass.dispatch_workgroups(blocks, 1, 1);
        }
    }
}
