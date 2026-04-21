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
use clap::Parser;
use winit::event_loop::{ControlFlow, EventLoop};

#[derive(Parser)]
struct Args {
    ply: PathBuf,

    #[arg(short, long)]
    stochastic_transparency: bool,
}

fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;

    let args = Args::parse();

    let splats = ply::load_splats(&args.ply)?;
    eprintln!("Loaded {} splats from {}", splats.len(), args.ply.display());

    let mut app = SplatApp::new(splats, args.stochastic_transparency);

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut app)?;

    Ok(())
}
