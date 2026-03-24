use std::sync::{Arc, atomic::Ordering};

use pollster::FutureExt;
use tap::Pipe;
use winit::{
    application::ApplicationHandler, event::WindowEvent, event_loop::ActiveEventLoop,
    window::Window,
};

use glam::Vec3;

use crate::{camera::Camera, gpu::GpuState, ply::PlyGaussian, renderer::SplatRenderer};

// ── UI output ─────────────────────────────────────────────────────────────────

/// Camera input collected from the egui viewport each frame.
#[derive(Default)]
struct UiOutput {
    open_rerun: bool,
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

// ── Active GPU + UI state ─────────────────────────────────────────────────────

/// Exists only while the app is resumed.
struct ActiveState {
    gpu: GpuState,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
    splat_renderer: SplatRenderer,
    camera: Camera,
}

impl ActiveState {
    async fn new(window: Arc<Window>, egui_ctx: &egui::Context, splats: &[PlyGaussian]) -> Self {
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

        let splat_renderer = SplatRenderer::new(&gpu.ctx, splats);
        let camera = Camera::default();

        Self {
            gpu,
            egui_state,
            egui_renderer,
            splat_renderer,
            camera,
        }
    }

    /// Run the egui frame, collect camera input, returns [`egui::FullOutput`].
    /// Does not record any GPU commands — call `encode_egui` after the scene pass.
    fn run_ui(
        &mut self,
        egui_ctx: &egui::Context,
        splat_count: usize,
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
                    ui.label(format!("{splat_count} splats  |  {fps:.0} fps"));

                    ui.add_space(8.0);
                    if ui.button("Visualize in Rerun").clicked() {
                        ui_out.open_rerun = true;
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

pub struct SplatApp {
    egui_ctx: egui::Context,
    state: Option<ActiveState>,
    splats: Vec<PlyGaussian>,
    rerun_rec: Option<rerun::RecordingStream>,
}

impl SplatApp {
    pub fn new(splats: Vec<PlyGaussian>) -> Self {
        let egui_ctx = egui::Context::default();
        egui_ctx.set_visuals(egui::Visuals::dark());
        Self {
            egui_ctx,
            state: None,
            splats,
            rerun_rec: None,
        }
    }

    fn setup_pipeline(&mut self) {
        eprintln!("Setting up pipeline");
    }

    /// Takes fields as parameters so it can be called while `self.state` is mutably borrowed.
    fn open_in_rerun(rerun_rec: &mut Option<rerun::RecordingStream>, splats: &[PlyGaussian]) {
        if rerun_rec.is_none() {
            match rerun::RecordingStreamBuilder::new("splat-rs").spawn() {
                Ok(rec) => *rerun_rec = Some(rec),
                Err(e) => eprintln!("Failed to spawn Rerun viewer: {e}"),
            }
        }
        if let Some(rec) = rerun_rec.as_ref() {
            crate::ply::log_splats_to_rerun(rec, splats);
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

            let state = ActiveState::new(window, &self.egui_ctx, &self.splats).block_on();

            self.setup_pipeline();
            state.gpu.window.request_redraw();
            self.state = Some(state);
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = &mut self.state else { return };

        if state.gpu.window.id() != window_id {
            return;
        }

        // Forward every event to egui
        let _ = state.egui_state.on_window_event(&state.gpu.window, &event);

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }

            WindowEvent::Resized(size) => {
                state.gpu.resize(size);
                state.gpu.window.request_redraw();
            }

            WindowEvent::RedrawRequested => {
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
                            *state =
                                ActiveState::new(window, &self.egui_ctx, &self.splats).block_on();
                            self.setup_pipeline();
                        } else {
                            state.gpu.recreate_surface();
                        }
                        return;
                    }
                    wgpu::CurrentSurfaceTexture::Timeout => {
                        eprintln!("Surface texture timeout");
                        return;
                    }
                    _ => {
                        eprintln!("Dropped frame");
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
                let (full_output, ui_out) = state.run_ui(&self.egui_ctx, self.splats.len());

                // Apply camera input
                state.camera.look(ui_out.look_delta.x, ui_out.look_delta.y);

                let speed = state.camera.move_speed * ui_out.dt.min(0.1);
                let fwd = state.camera.forward();
                let right = state.camera.right();

                state.camera.position += fwd * (ui_out.move_fwd + ui_out.scroll) * speed;
                state.camera.position += right * ui_out.move_right * speed;
                state.camera.position += Vec3::Y * ui_out.move_up * speed;

                // Scene pass (clears + draws splats)
                state
                    .splat_renderer
                    .prepare(&state.gpu.ctx, &state.camera, w, h);
                {
                    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Scene Pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        })],
                        depth_stencil_attachment: None,
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

                if ui_out.open_rerun {
                    Self::open_in_rerun(&mut self.rerun_rec, &self.splats);
                }
            }

            _ => {}
        }
    }
}
