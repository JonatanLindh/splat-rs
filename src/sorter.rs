use std::mem;

use crate::{
    gpu::GpuContext,
    shaders::sort::{
        BIN_PART_SIZE, PushConstants, RADIX, WgpuBindGroup0, WgpuBindGroup0Entries,
        WgpuBindGroup0EntriesParams,
        compute::{
            create_cs_count_pass_pipeline_embed_source, create_cs_scan_pass_pipeline_embed_source,
            create_cs_scatter_pass_pipeline_embed_source,
        },
    },
};

pub struct Sorter {
    // Pipelines
    count_pipeline: wgpu::ComputePipeline,
    scan_pipeline: wgpu::ComputePipeline,
    scatter_pipeline: wgpu::ComputePipeline,

    // Bind Groups
    bind_group_even: WgpuBindGroup0, // Reads sort_buffer, writes alt_buffer
    bind_group_odd: WgpuBindGroup0,  // Reads alt_buffer, writes sort_buffer

    // State Buffers (Need to be cleared before each full sort)
    pass_histograms: wgpu::Buffer,

    // Data Buffers
    pub in_keys: wgpu::Buffer,
    pub in_payload: wgpu::Buffer,
    pub out_keys: wgpu::Buffer,
    pub out_payload: wgpu::Buffer,

    max_elements: u32,
}

impl Sorter {
    pub fn new(GpuContext { device, .. }: &GpuContext, max_elements: u32) -> Self {
        let max_blocks_binning = max_elements.div_ceil(BIN_PART_SIZE);
        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;

        // --- Buffer Creation ---

        // Spine buffer for decoupled lookback
        let pass_histograms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pass Histograms (Spine)"),
            size: (4 * (max_blocks_binning + 1) * RADIX * mem::size_of::<u32>() as u32)
                as wgpu::BufferAddress,
            usage,
            mapped_at_creation: false,
        });

        // padded size
        let size = (max_elements.next_multiple_of(4) * mem::size_of::<u32>() as u32)
            as wgpu::BufferAddress;

        let in_keys = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("In Keys Buffer"),
            size,
            usage,
            mapped_at_creation: false,
        });

        let in_payload = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("In Payload Buffer"),
            size,
            usage,
            mapped_at_creation: false,
        });

        let out_keys = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Out Buffer"),
            size,
            usage,
            mapped_at_creation: false,
        });

        let out_payload = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Out Payload Buffer"),
            size,
            usage,
            mapped_at_creation: false,
        });

        // --- Load SPIR-V ---

        let bind_group_even = WgpuBindGroup0::from_bindings(
            device,
            WgpuBindGroup0Entries::new(WgpuBindGroup0EntriesParams {
                in_keys: in_keys.as_entire_buffer_binding(),
                in_payload: in_payload.as_entire_buffer_binding(),
                out_keys: out_keys.as_entire_buffer_binding(),
                out_payload: out_payload.as_entire_buffer_binding(),
                pass_histograms: pass_histograms.as_entire_buffer_binding(),
            }),
        );

        let bind_group_odd = WgpuBindGroup0::from_bindings(
            device,
            WgpuBindGroup0Entries::new(WgpuBindGroup0EntriesParams {
                //
                in_keys: out_keys.as_entire_buffer_binding(),
                in_payload: out_payload.as_entire_buffer_binding(),
                // SWAPPED
                out_keys: in_keys.as_entire_buffer_binding(),
                out_payload: in_payload.as_entire_buffer_binding(),
                //
                pass_histograms: pass_histograms.as_entire_buffer_binding(),
            }),
        );

        // Create the 3 pipelines mapped to your Slang entry points
        let count_pipeline = create_cs_count_pass_pipeline_embed_source(device);
        let scan_pipeline = create_cs_scan_pass_pipeline_embed_source(device);
        let scatter_pipeline = create_cs_scatter_pass_pipeline_embed_source(device);

        Self {
            count_pipeline,
            scan_pipeline,
            scatter_pipeline,
            bind_group_even,
            bind_group_odd,
            pass_histograms,
            in_keys,
            in_payload,
            out_keys,
            out_payload,
            max_elements,
        }
    }

    fn get_bind_group_for_pass(&self, pass: u32) -> &WgpuBindGroup0 {
        match pass.is_multiple_of(2) {
            true => &self.bind_group_even,
            false => &self.bind_group_odd,
        }
    }

    pub fn sort(&self, encoder: &mut wgpu::CommandEncoder, num_elements: u32) {
        if num_elements == 0 {
            return;
        }

        assert!(
            num_elements <= self.max_elements,
            "Exceeded pre-allocated sorting capacity"
        );

        // 1. Clear state buffers
        encoder.clear_buffer(&self.pass_histograms, 0, None);

        let mut push_constants = PushConstants {
            size: num_elements,
            shift: 0,
            pass_index: 0,
        };

        // 4. Digit Binning Passes (The OneSweep)
        let binning_blocks = num_elements.div_ceil(BIN_PART_SIZE);

        for pass in 0..4 {
            push_constants.pass_index = pass;
            push_constants.shift = pass * 8; // Radix Log = 8

            // Ping-pong the bind groups
            let bind_group = self.get_bind_group_for_pass(pass);

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("(Sort) Count Pass {pass}")),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.count_pipeline);
                cpass.set_immediates(0, bytemuck::cast_slice(&[push_constants]));
                bind_group.set(&mut cpass);

                cpass.dispatch_workgroups(binning_blocks, 1, 1);
            }

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("(Sort) Scan Pass {pass}")),
                    timestamp_writes: None,
                });

                cpass.set_pipeline(&self.scan_pipeline);
                cpass.set_immediates(0, bytemuck::cast_slice(&[push_constants]));
                bind_group.set(&mut cpass);

                cpass.dispatch_workgroups(1, 1, 1);
            }

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("(Sort) Scatter Pass {pass}")),
                    timestamp_writes: None,
                });

                cpass.set_pipeline(&self.scatter_pipeline);
                cpass.set_immediates(0, bytemuck::cast_slice(&[push_constants]));
                bind_group.set(&mut cpass);

                cpass.dispatch_workgroups(binning_blocks, 1, 1);
            }
        }
    }
}
