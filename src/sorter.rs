use std::mem;

use lampshade::KeyValueSoaSorter;

use crate::gpu::GpuContext;

pub struct Sorter {
    sorter: KeyValueSoaSorter,
    capacity: u32,
    pub in_keys: wgpu::Buffer,
    pub in_payload: wgpu::Buffer,
    pub count_buf: wgpu::Buffer,
}

impl Sorter {
    pub fn new(
        GpuContext { device, queue, .. }: &GpuContext,
        max_elements: u32,
        count_buf: wgpu::Buffer,
    ) -> Self {
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
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
        let mut sorter = KeyValueSoaSorter::new(device, queue);
        sorter
            .prepare_counted_from_word(&in_keys, &in_payload, &count_buf, 0, max_elements)
            .expect("splat sort buffers satisfy Lampshade's requirements");
        Self {
            sorter,
            capacity: max_elements,
            in_keys,
            in_payload,
            count_buf,
        }
    }

    pub fn sort(&self, encoder: &mut wgpu::CommandEncoder, _dispatch_indirect_args: &wgpu::Buffer) {
        self.sorter
            .record_reserved_sort_counted_from_word(
                encoder,
                &self.in_keys,
                &self.in_payload,
                &self.count_buf,
                0,
                self.capacity,
            )
            .expect("the splat sort plan remains valid");
    }
}

#[cfg(test)]
mod tests {
    use std::{error::Error, sync::mpsc};

    use wgpu::util::DeviceExt;

    use super::Sorter;
    use crate::gpu::GpuContext;

    const BIN_PART_SIZE: u32 = 2_048;

    #[test]
    fn sorts_gpu_counted_keys_and_values_stably() -> Result<(), Box<dyn Error>> {
        pollster::block_on(run_sort_regression())
    }

    async fn run_sort_regression() -> Result<(), Box<dyn Error>> {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter =
            match wgpu::util::initialize_adapter_from_env_or_default(&instance, None).await {
                Ok(adapter) => adapter,
                Err(error) => {
                    eprintln!("skipping GPU sort regression: {error}");
                    return Ok(());
                }
            };
        let required = wgpu::Features::IMMEDIATES | wgpu::Features::SUBGROUP;
        if !adapter.features().contains(required) {
            eprintln!("skipping GPU sort regression: required features are unavailable");
            return Ok(());
        }
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("splat-rs sort regression"),
                required_features: required,
                required_limits: adapter.limits(),
                memory_hints: wgpu::MemoryHints::Performance,
                ..Default::default()
            })
            .await?;
        let context = GpuContext {
            device,
            queue,
            surface_format: wgpu::TextureFormat::Rgba8Unorm,
        };

        let count = 1_100_003_u32;
        let count_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sort regression GPU count"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let count_source = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("sort regression count producer"),
                contents: bytemuck::bytes_of(&count),
                usage: wgpu::BufferUsages::COPY_SRC,
            });
        let dispatch = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("sort regression indirect dispatch"),
                contents: bytemuck::cast_slice(&[count.div_ceil(BIN_PART_SIZE), 1_u32, 1_u32]),
                usage: wgpu::BufferUsages::INDIRECT,
            });
        let sorter = Sorter::new(&context, count, count_buffer.clone());
        let keys: Vec<_> = (0..count).map(key_for_index).collect();
        let values: Vec<_> = (0..count).collect();
        context
            .queue
            .write_buffer(&sorter.in_keys, 0, bytemuck::cast_slice(&keys));
        context
            .queue
            .write_buffer(&sorter.in_payload, 0, bytemuck::cast_slice(&values));

        let mut encoder = context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(&count_source, 0, &sorter.count_buf, 0, 4);
        sorter.sort(&mut encoder, &dispatch);
        submit(&context, encoder)?;

        let actual_keys = read_u32(&context, &sorter.in_keys, count)?;
        let actual_values = read_u32(&context, &sorter.in_payload, count)?;
        for index in 0..actual_keys.len() {
            let value = actual_values[index];
            assert_eq!(
                actual_keys[index],
                key_for_index(value),
                "key/value association failed at {index}"
            );
            if index > 0 {
                let previous_key = actual_keys[index - 1];
                let previous_value = actual_values[index - 1];
                assert!(
                    actual_keys[index] > previous_key
                        || (actual_keys[index] == previous_key && value > previous_value),
                    "key order or duplicate stability failed at {index}"
                );
            }
        }
        Ok(())
    }

    fn key_for_index(index: u32) -> u32 {
        index.wrapping_mul(747_796_405).wrapping_add(2_891_336_453) & 0x000f_ffff
    }

    fn submit(context: &GpuContext, encoder: wgpu::CommandEncoder) -> Result<(), wgpu::PollError> {
        let submission = context.queue.submit([encoder.finish()]);
        context.device.poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        })?;
        Ok(())
    }

    fn read_u32(
        context: &GpuContext,
        source: &wgpu::Buffer,
        count: u32,
    ) -> Result<Vec<u32>, Box<dyn Error>> {
        let bytes = u64::from(count) * 4;
        let staging = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sort regression readback"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(source, 0, &staging, 0, bytes);
        submit(context, encoder)?;
        let slice = staging.slice(..);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        context.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })?;
        receiver.recv()??;
        let values = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
        staging.unmap();
        Ok(values)
    }
}
