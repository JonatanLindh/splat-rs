use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use wgpu::Features;
use winit::{dpi::PhysicalSize, window::Window};

#[derive(Clone, Debug)]
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface_format: wgpu::TextureFormat,
}

pub struct GpuState {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub ctx: GpuContext,
    pub surface: wgpu::Surface<'static>,
    pub surface_config: wgpu::SurfaceConfiguration,
    pub window: Arc<Window>,
    pub device_lost: Arc<AtomicBool>,
}

impl GpuState {
    pub async fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to find an appropriate adapter");

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Primary Device"),
                required_limits: adapter.limits(),
                required_features: Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
                ..Default::default()
            })
            .await
            .expect("Failed to create device");

        let device_lost = Arc::new(AtomicBool::new(false));
        device.set_device_lost_callback({
            let flag = Arc::clone(&device_lost);
            move |_reason, _msg| flag.store(true, Ordering::Relaxed)
        });

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .cloned()
            .unwrap_or(surface_caps.formats[0]);

        let alpha_mode = surface_caps
            .alpha_modes
            .iter()
            .find(|&&m| m == wgpu::CompositeAlphaMode::Opaque)
            .copied()
            .unwrap_or(surface_caps.alpha_modes[0]);

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        let mut state = Self {
            instance,
            adapter,
            ctx: GpuContext {
                device,
                queue,
                surface_format,
            },
            surface,
            surface_config,
            window,
            device_lost,
        };
        state.reconfigure_surface();
        state
    }

    pub fn resize(&mut self, PhysicalSize { width, height }: PhysicalSize<u32>) {
        if width > 0 && height > 0 {
            self.surface_config.width = width;
            self.surface_config.height = height;
            self.reconfigure_surface();
        }
    }

    pub fn reconfigure_surface(&mut self) {
        self.surface
            .configure(&self.ctx.device, &self.surface_config);
    }

    pub fn recreate_surface(&mut self) {
        self.surface = self
            .instance
            .create_surface(self.window.clone())
            .expect("Failed to recreate surface");
        self.reconfigure_surface();
    }
}
