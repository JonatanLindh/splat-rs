use std::mem;

use crate::{
    gpu::GpuContext,
    shaders::{
        prepare::{
            PushConstantsInit, WG_SIZE, WgpuBindGroup0, WgpuBindGroup0Entries,
            WgpuBindGroup0EntriesParams,
            compute::{
                create_cs_prepare_indirect_args_pipeline_embed_source,
                create_cs_prepare_pipeline_embed_source,
            },
        },
        splat_common::CameraUniform,
    },
};

pub struct Preparer {
    prep_pipeline: wgpu::ComputePipeline,
    indirect_args_pipeline: wgpu::ComputePipeline,
    bind_group: WgpuBindGroup0,

    cull_atomic_count_buffer: wgpu::Buffer,
}

impl Preparer {
    pub fn new(
        GpuContext { device, .. }: &GpuContext,
        depth_buffer: wgpu::Buffer,
        index_buffer: wgpu::Buffer,
        splat_buffer: wgpu::Buffer,
        cull_atomic_count_buffer: wgpu::Buffer,
        dispatch_indirect_args: wgpu::Buffer,
        draw_indexed_indirect_args: wgpu::Buffer,
    ) -> Self {
        let bind_group = WgpuBindGroup0::from_bindings(
            device,
            WgpuBindGroup0Entries::new(WgpuBindGroup0EntriesParams {
                u32_depths: depth_buffer.as_entire_buffer_binding(),
                u32_indices: index_buffer.as_entire_buffer_binding(),
                splats: splat_buffer.as_entire_buffer_binding(),
                atomic_count: cull_atomic_count_buffer.as_entire_buffer_binding(),
                dispatch_indirect_args: dispatch_indirect_args.as_entire_buffer_binding(),
                draw_indexed_indirect_args: draw_indexed_indirect_args.as_entire_buffer_binding(),
            }),
        );

        let prep_pipeline = create_cs_prepare_pipeline_embed_source(device);
        let indirect_args_pipeline = create_cs_prepare_indirect_args_pipeline_embed_source(device);

        Self {
            prep_pipeline,
            indirect_args_pipeline,
            bind_group,
            cull_atomic_count_buffer,
        }
    }

    pub fn run(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        num_elements: u32,
        camera_uniform: &CameraUniform,
        use_culling: bool,
        invert_culling: bool,
    ) {
        if num_elements == 0 {
            return;
        }

        let push_constants = PushConstantsInit {
            camera_view_row_z: camera_uniform.view.row(2),
            view_proj: camera_uniform.view_proj,
            count: num_elements,
            use_culling: use_culling as u32,
            invert_culling: invert_culling as u32,
        }
        .build();

        let blocks = num_elements.div_ceil(WG_SIZE);

        encoder.clear_buffer(
            &self.cull_atomic_count_buffer,
            0,
            Some(mem::size_of::<u32>() as u64),
        );

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Prepare depths pass"),
                timestamp_writes: None,
            });

            cpass.set_pipeline(&self.prep_pipeline);
            cpass.set_immediates(0, bytemuck::cast_slice(&[push_constants]));
            self.bind_group.set(&mut cpass);

            cpass.dispatch_workgroups(blocks, 1, 1);
        }
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Prepare indirect args pass"),
                timestamp_writes: None,
            });

            cpass.set_pipeline(&self.indirect_args_pipeline);
            self.bind_group.set(&mut cpass);

            cpass.dispatch_workgroups(1, 1, 1);
        }
    }
}
