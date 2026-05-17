use std::sync::{Arc, atomic::Ordering};

use tap::Pipe;
use winit::{
    application::ApplicationHandler, event::WindowEvent, event_loop::ActiveEventLoop,
    window::Window,
};

use glam::Vec3;

use crate::{
    camera::Camera,
    gpu::{GpuContext, GpuState},
    hot_reload::ShaderWatcher,
    renderer::SplatRenderer,
    shaders::splat_common::GpuSplat,
};

// ── UI output ─────────────────────────────────────────────────────────────────

/// Camera input collected from the egui viewport each frame.
#[derive(Default)]
struct UiOutput {
    load_file: bool,
    transparency_changed: bool,
    /// Right-drag delta in screen pixels.
    look_delta: egui::Vec2,
    /// Scroll wheel, positive = up (used to scale move speed).
    scroll: f32,
    /// Forward/back from W/S: +1 / -1.
    move_fwd: f32,
    /// Strafe from D/A: +1 / -1.
    move_right: f32,
    /// Vertical from E or Space / Q or Shift: +1 / -1.
    move_up: f32,
    /// Time since last frame from egui's stable_dt.
    dt: f32,
}

/// Shared app state that needs to be accessed from UI.
struct AppState<'a> {
    splat_count: usize,
    ply_path: &'a Option<std::path::PathBuf>,
    stochastic_transparency: &'a mut bool,
    msaa_samples: &'a mut u32,
    view_frustum_culling: &'a mut bool,
    invert_culling: &'a mut bool,
}

// ── Active GPU + UI state ─────────────────────────────────────────────────────

/// Transparency rendering mode with associated resources
#[allow(clippy::large_enum_variant)]
enum TransparencyMode {
    /// Standard alpha blending (no MSAA, no depth buffer)
    Normal,
    /// Stochastic transparency with MSAA + depth buffer
    Stochastic(MsaaDepth),
}

#[allow(unused)]
struct MsaaDepth {
    msaa_texture: wgpu::Texture,
    msaa_view: wgpu::TextureView,
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
}

impl MsaaDepth {
    fn new(gpu_ctx: &GpuContext, size: wgpu::Extent3d, msaa_samples: u32) -> MsaaDepth {
        let msaa_texture = gpu_ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("MSAA Texture"),
            size,
            mip_level_count: 1,
            sample_count: msaa_samples,
            dimension: wgpu::TextureDimension::D2,
            format: gpu_ctx.surface_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        let depth_texture = gpu_ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size,
            mip_level_count: 1,
            sample_count: msaa_samples,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        let msaa_view = msaa_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        MsaaDepth {
            msaa_texture,
            msaa_view,
            depth_texture,
            depth_view,
        }
    }
}

/// Exists only while the app is resumed.
struct ActiveState {
    gpu: GpuState,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
    splat_renderer: SplatRenderer,
    camera: Camera,

    watcher: ShaderWatcher,

    transparency_mode: TransparencyMode,
    msaa_samples: u32,
}

impl ActiveState {
    async fn new(
        window: Arc<Window>,
        egui_ctx: &egui::Context,
        splats: Vec<GpuSplat>,
        stochastic_transparency: bool,
        msaa_samples: u32,
    ) -> Self {
        let gpu = GpuState::new(window).await;

        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::viewport::ViewportId::ROOT,
            &gpu.window,
            Some(gpu.window.scale_factor() as f32),
            None,
            None,
        );

        let egui_renderer =
            egui_wgpu::Renderer::new(&gpu.ctx.device, gpu.ctx.surface_format, Default::default());

        let splat_renderer =
            SplatRenderer::new(&gpu.ctx, splats, stochastic_transparency, msaa_samples);
        let camera = Camera::default();

        let watcher = ShaderWatcher::new();

        // Create transparency mode with resources
        let transparency_mode = if stochastic_transparency {
            let size = wgpu::Extent3d {
                width: gpu.surface_config.width,
                height: gpu.surface_config.height,
                depth_or_array_layers: 1,
            };

            TransparencyMode::Stochastic(MsaaDepth::new(&gpu.ctx, size, msaa_samples))
        } else {
            TransparencyMode::Normal
        };

        Self {
            gpu,
            egui_state,
            egui_renderer,
            splat_renderer,
            camera,
            watcher,
            transparency_mode,
            msaa_samples,
        }
    }

    fn resize_render_targets(&mut self, width: u32, height: u32) {
        if let TransparencyMode::Stochastic { .. } = self.transparency_mode {
            let size = wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            };

            self.transparency_mode =
                TransparencyMode::Stochastic(MsaaDepth::new(&self.gpu.ctx, size, self.msaa_samples))
        }
    }

    fn try_reload_shaders(&mut self, stochastic_transparency: bool) {
        use crate::hot_reload::reload_shader_module;
        use crate::shaders::ShaderEntry;

        let entry = if stochastic_transparency {
            ShaderEntry::SplatStochastic
        } else {
            ShaderEntry::Splat
        };

        if let Some(module) = reload_shader_module(&self.gpu.ctx, entry) {
            self.splat_renderer.rebuild_pipeline(&self.gpu.ctx, &module);
        }
    }

    /// Run the egui frame, collect camera input, returns [`egui::FullOutput`].
    /// Does not record any GPU commands — call `encode_egui` after the scene pass.
    fn run_ui(
        &mut self,
        egui_ctx: &egui::Context,
        app_state: &mut AppState,
    ) -> (egui::FullOutput, UiOutput) {
        let raw_input = self.egui_state.take_egui_input(&self.gpu.window);

        let mut ui_out = UiOutput::default();

        let full_output = egui_ctx.run_ui(raw_input, |ui| {
            // ── Side panel ────────────────────────────────────────────────────
            egui::Panel::right("controls")
                .min_size(260.0)
                .max_size(380.0)
                .show_inside(ui, |ui| {
                    ui.add_space(6.0);

                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new("Gaussian Splatting")
                                .size(16.0)
                                .strong()
                                .color(egui::Color32::from_gray(220)),
                        );
                    });

                    ui.add_space(4.0);
                    let fps = ui.input(|i| 1.0 / i.stable_dt);
                    ui.label(format!("{} splats  |  {fps:.0} fps", app_state.splat_count));

                    ui.add_space(8.0);
                    ui.separator();
                    ui.add_space(4.0);

                    // File loading
                    ui.label(egui::RichText::new("File").strong());
                    if let Some(path) = &app_state.ply_path {
                        ui.label(format!(
                            "Loaded: {}",
                            path.file_name().unwrap_or_default().to_string_lossy()
                        ));
                    } else {
                        ui.label("No file loaded");
                    }

                    if ui.button("Load PLY file...").clicked() {
                        ui_out.load_file = true;
                    }

                    ui.add_space(8.0);
                    ui.separator();
                    ui.add_space(4.0);

                    // Rendering options
                    ui.label(egui::RichText::new("Rendering").strong());
                    if ui
                        .checkbox(app_state.stochastic_transparency, "Stochastic Transparency")
                        .changed()
                    {
                        ui_out.transparency_changed = true;
                    }

                    ui.checkbox(app_state.view_frustum_culling, "View Frustum Culling");
                    if *app_state.view_frustum_culling {
                        ui.checkbox(app_state.invert_culling, "Invert Culling (Debug)");
                    }

                    if *app_state.stochastic_transparency {
                        ui.label(egui::RichText::new("MSAA Samples").strong());
                        for sample_count in self
                            .gpu
                            .adapter
                            .get_texture_format_features(self.gpu.surface_config.format)
                            .flags
                            .supported_sample_counts()
                        {
                            if ui
                                .radio_value(
                                    app_state.msaa_samples,
                                    sample_count,
                                    sample_count.to_string(),
                                )
                                .clicked()
                            {
                                ui_out.transparency_changed = true;
                            }
                        }
                    }
                });

            // Viewport drag area
            let response = ui.allocate_response(ui.available_size(), egui::Sense::click_and_drag());

            // Right-drag
            if response.dragged_by(egui::PointerButton::Secondary) {
                ui_out.look_delta = response.drag_delta();
            }

            // Only collect scroll and keyboard input when the pointer is over the viewport.
            if response.hovered() || response.dragged() {
                ui.input(|i| {
                    ui_out.scroll = i.smooth_scroll_delta.y;
                    ui_out.dt = i.stable_dt;

                    if i.key_down(egui::Key::W) {
                        ui_out.move_fwd += 1.0;
                    }
                    if i.key_down(egui::Key::S) {
                        ui_out.move_fwd -= 1.0;
                    }
                    if i.key_down(egui::Key::D) {
                        ui_out.move_right += 1.0;
                    }
                    if i.key_down(egui::Key::A) {
                        ui_out.move_right -= 1.0;
                    }
                    if i.key_down(egui::Key::E) || i.key_down(egui::Key::Space) {
                        ui_out.move_up += 1.0;
                    }
                    if i.key_down(egui::Key::Q) || i.modifiers.shift {
                        ui_out.move_up -= 1.0;
                    }
                });
            }
        });

        (full_output, ui_out)
    }

    /// Tessellate the egui output and record the egui render pass into `encoder`.
    /// Must be called after the scene pass so the overlay composites on top.
    fn encode_egui(
        &mut self,
        egui_ctx: &egui::Context,
        full_output: egui::FullOutput,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
    ) {
        self.egui_state
            .handle_platform_output(&self.gpu.window, full_output.platform_output);

        let paint_jobs = egui_ctx.tessellate(full_output.shapes, full_output.pixels_per_point);
        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [
                self.gpu.surface_config.width,
                self.gpu.surface_config.height,
            ],
            pixels_per_point: self.gpu.window.scale_factor() as f32,
        };

        for (id, image_delta) in &full_output.textures_delta.set {
            self.egui_renderer.update_texture(
                &self.gpu.ctx.device,
                &self.gpu.ctx.queue,
                *id,
                image_delta,
            );
        }

        self.egui_renderer.update_buffers(
            &self.gpu.ctx.device,
            &self.gpu.ctx.queue,
            encoder,
            &paint_jobs,
            &screen_descriptor,
        );

        {
            let render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Egui Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load, // composite over the scene pass
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            self.egui_renderer.render(
                &mut render_pass.forget_lifetime(),
                &paint_jobs,
                &screen_descriptor,
            );
        }

        for id in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }
    }
}

// ── Application ───────────────────────────────────────────────────────────────

fn spawn_gpu_init(
    tx: flume::Sender<(ActiveState, egui::Context)>,
    window: Arc<Window>,
    splats: Vec<GpuSplat>,
    stochastic_transparency: bool,
    msaa_samples: u32,
) {
    let win = Arc::clone(&window);
    let future = async move {
        // Create fresh egui context to avoid timing issues during reinit
        let egui_ctx = egui::Context::default();
        egui_ctx.set_visuals(egui::Visuals::dark());

        let state = ActiveState::new(
            win,
            &egui_ctx,
            splats,
            stochastic_transparency,
            msaa_samples,
        )
        .await;
        let _ = tx.send_async((state, egui_ctx)).await;
        window.request_redraw();
    };

    std::thread::spawn(|| pollster::block_on(future));
}

pub struct SplatApp {
    egui_ctx: egui::Context,
    state: Option<ActiveState>,
    tx: flume::Sender<(ActiveState, egui::Context)>,
    rx: flume::Receiver<(ActiveState, egui::Context)>,
    splats: Vec<GpuSplat>,
    stochastic_transparency: bool,
    msaa_samples: u32,
    ply_path: Option<std::path::PathBuf>,
    needs_reinit: bool,
    view_frustum_culling: bool,
    invert_culling: bool,
}

impl Default for SplatApp {
    fn default() -> Self {
        Self::new()
    }
}

impl SplatApp {
    pub fn new() -> Self {
        let egui_ctx = egui::Context::default();
        egui_ctx.set_visuals(egui::Visuals::dark());
        let (tx, rx) = flume::bounded(1);

        // Try to load default PLY file if it exists
        let (splats, ply_path) = Self::try_load_default_ply();

        Self {
            egui_ctx,
            state: None,
            tx,
            rx,
            splats,
            stochastic_transparency: false,
            msaa_samples: 4,
            ply_path,
            needs_reinit: false,
            view_frustum_culling: true,
            invert_culling: false,
        }
    }

    fn try_load_default_ply() -> (Vec<GpuSplat>, Option<std::path::PathBuf>) {
        let default_files = ["point_cloud.ply", "train.ply"];

        for file in &default_files {
            let path = std::path::PathBuf::from(file);
            if path.exists() {
                match crate::ply::load_splats(&path) {
                    Ok(splats) => {
                        tracing::info!("Loaded {} splats from {}", splats.len(), path.display());
                        return (splats, Some(path));
                    }
                    Err(e) => {
                        tracing::warn!("Failed to load {}: {}", path.display(), e);
                    }
                }
            }
        }

        (Vec::new(), None)
    }

    fn load_ply_file(&mut self, path: std::path::PathBuf) {
        match crate::ply::load_splats(&path) {
            Ok(splats) => {
                tracing::info!("Loaded {} splats from {}", splats.len(), path.display());
                self.splats = splats;
                self.ply_path = Some(path);
                // Caller should set needs_reinit = true after calling this
            }
            Err(e) => {
                tracing::error!("Failed to load {}: {}", path.display(), e);
            }
        }
    }
}

impl ApplicationHandler for SplatApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_none() {
            let window = event_loop
                .create_window(Window::default_attributes().with_title("Splat"))
                .unwrap()
                .pipe(Arc::new);

            spawn_gpu_init(
                self.tx.clone(),
                window,
                self.splats.clone(),
                self.stochastic_transparency,
                self.msaa_samples,
            );
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        // Pick up newly initialized state from the async init channel
        if self.state.is_none()
            && let Ok((state, egui_ctx)) = self.rx.try_recv()
        {
            self.state = Some(state);
            self.egui_ctx = egui_ctx;
        }

        let Some(state) = &mut self.state else { return };

        if state.gpu.window.id() != window_id {
            return;
        }

        // Forward every event to egui
        let _ = state.egui_state.on_window_event(&state.gpu.window, &event);

        match event {
            WindowEvent::CloseRequested => {
                self.state = None;
                event_loop.exit();
            }

            WindowEvent::Resized(size) => {
                state.gpu.resize(size);
                state.resize_render_targets(size.width, size.height);
                state.gpu.window.request_redraw();
            }

            WindowEvent::RedrawRequested => {
                // Handle reinitialization request from previous frame
                if self.needs_reinit {
                    self.needs_reinit = false;
                    let window = Arc::clone(&state.gpu.window);
                    self.state = None;
                    spawn_gpu_init(
                        self.tx.clone(),
                        window,
                        self.splats.clone(),
                        self.stochastic_transparency,
                        self.msaa_samples,
                    );
                    return;
                }

                // Hot-reload shaders in debug builds
                if state.watcher.poll() {
                    state.try_reload_shaders(self.stochastic_transparency);
                }

                let output = match state.gpu.surface.get_current_texture() {
                    wgpu::CurrentSurfaceTexture::Success(t) => t,
                    wgpu::CurrentSurfaceTexture::Suboptimal(t) => {
                        state.gpu.reconfigure_surface();
                        t
                    }
                    wgpu::CurrentSurfaceTexture::Outdated => {
                        state.gpu.reconfigure_surface();
                        return;
                    }
                    wgpu::CurrentSurfaceTexture::Lost => {
                        if state.gpu.device_lost.load(Ordering::Relaxed) {
                            let window = Arc::clone(&state.gpu.window);
                            self.state = None;
                            spawn_gpu_init(
                                self.tx.clone(),
                                window,
                                self.splats.clone(),
                                self.stochastic_transparency,
                                self.msaa_samples,
                            );
                        } else {
                            state.gpu.recreate_surface();
                        }
                        return;
                    }
                    wgpu::CurrentSurfaceTexture::Timeout => {
                        tracing::warn!("Surface texture timeout");
                        return;
                    }
                    _ => {
                        tracing::warn!("Dropped frame");
                        return;
                    }
                };

                let view = output
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());

                let mut encoder =
                    state
                        .gpu
                        .ctx
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Frame Encoder"),
                        });

                let w = state.gpu.surface_config.width;
                let h = state.gpu.surface_config.height;

                // Run egui UI logic, collect camera input
                let mut app_state = AppState {
                    splat_count: self.splats.len(),
                    ply_path: &self.ply_path,
                    stochastic_transparency: &mut self.stochastic_transparency,
                    msaa_samples: &mut self.msaa_samples,
                    view_frustum_culling: &mut self.view_frustum_culling,
                    invert_culling: &mut self.invert_culling,
                };
                let (full_output, ui_out) = state.run_ui(&self.egui_ctx, &mut app_state);

                // Handle UI events that require reinit - defer until after this frame
                let mut load_file_path = None;
                if ui_out.load_file
                    && let Some(path) = rfd::FileDialog::new()
                        .add_filter("PLY files", &["ply"])
                        .pick_file()
                {
                    load_file_path = Some(path);
                    self.needs_reinit = true;
                }

                if ui_out.transparency_changed {
                    // Defer reinitialization until after this frame is presented
                    self.needs_reinit = true;
                }

                // Apply camera input
                state.camera.look(ui_out.look_delta.x, ui_out.look_delta.y);

                let speed = state.camera.move_speed * ui_out.dt.min(0.1);
                let fwd = state.camera.forward();
                let right = state.camera.right();

                state.camera.position += fwd * (ui_out.move_fwd + ui_out.scroll) * speed;
                state.camera.position += right * ui_out.move_right * speed;
                state.camera.position += Vec3::Y * ui_out.move_up * speed;

                // Scene pass (clears + draws splats)
                state.splat_renderer.prepare(
                    &state.gpu.ctx,
                    &state.camera,
                    w,
                    h,
                    self.view_frustum_culling,
                    self.invert_culling,
                );
                {
                    let (color_view, resolve_target, depth_stencil_attachment) =
                        match &state.transparency_mode {
                            TransparencyMode::Stochastic(MsaaDepth {
                                msaa_view,
                                depth_view,
                                ..
                            }) => {
                                // MSAA + depth for stochastic transparency
                                let color_view = msaa_view;
                                let resolve_target = Some(&view);
                                let depth_stencil_attachment =
                                    Some(wgpu::RenderPassDepthStencilAttachment {
                                        view: depth_view,
                                        depth_ops: Some(wgpu::Operations {
                                            load: wgpu::LoadOp::Clear(1.0),
                                            store: wgpu::StoreOp::Store,
                                        }),
                                        stencil_ops: None,
                                    });
                                (color_view, resolve_target, depth_stencil_attachment)
                            }
                            TransparencyMode::Normal => {
                                // Standard rendering (no MSAA, no depth)
                                (&view, None, None)
                            }
                        };

                    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Scene Pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: color_view,
                            resolve_target,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        })],
                        depth_stencil_attachment,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    state.splat_renderer.render(&mut rpass);
                }

                // Egui overlay
                state.encode_egui(&self.egui_ctx, full_output, &mut encoder, &view);

                state.gpu.ctx.queue.submit(Some(encoder.finish()));
                output.present();
                state.gpu.window.request_redraw();

                // Load file after presenting (to avoid holding state borrow)
                if let Some(path) = load_file_path {
                    self.load_ply_file(path);
                }
            }

            _ => {}
        }
    }
}
