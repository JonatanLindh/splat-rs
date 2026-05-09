#[allow(
    unused,
    non_snake_case,
    non_camel_case_types,
    non_upper_case_globals,
    unused_doc_comments,
    clippy::all
)]
pub mod shaders {
    include!(concat!(env!("OUT_DIR"), "/shaders.rs"));
}

pub mod app;
pub mod camera;
pub mod gpu;
pub mod hot_reload;
pub mod ply;
pub mod radix_sort_cpu;
pub mod renderer;
pub mod sorter;

use winit::event_loop::{ControlFlow, EventLoop};

fn main() -> color_eyre::Result<()> {
    tracing_subscriber::fmt::init();
    color_eyre::install()?;

    let mut app = app::SplatApp::new();

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut app)?;

    Ok(())
}
