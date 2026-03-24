#[allow(
    unused,
    non_snake_case,
    non_camel_case_types,
    non_upper_case_globals,
    unused_doc_comments
)]
pub mod shader_bindings {
    include!(concat!(env!("OUT_DIR"), "/shader_bindings.rs"));
}

pub mod app;
pub mod camera;
pub mod gpu;
pub mod ply;
pub mod radix_sort_cpu;
pub mod renderer;

use std::path::PathBuf;

use app::SplatApp;
use winit::event_loop::{ControlFlow, EventLoop};

fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;

    let path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .ok_or_else(|| color_eyre::eyre::eyre!("Usage: splat-rs <path/to/splat.ply>"))?;

    let splats = ply::load_splats(&path)?;
    eprintln!("Loaded {} splats from {}", splats.len(), path.display());

    let mut app = SplatApp::new(splats);

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut app)?;

    Ok(())
}
