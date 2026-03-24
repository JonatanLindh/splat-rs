use std::{env, path::PathBuf};

use wgsl_bindgen::{GlamWgslTypeMap, WgslBindgenOptionBuilder, WgslTypeSerializeStrategy};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    print!("cargo:rerun-if-changed=shaders/splat.wgsl");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let out = out_dir.join("shader_bindings.rs");

    let src = WgslBindgenOptionBuilder::default()
        .workspace_root("shaders")
        .add_entry_point("shaders/splat.wgsl")
        .serialization_strategy(WgslTypeSerializeStrategy::Bytemuck)
        .type_map(GlamWgslTypeMap)
        .skip_header_comments(true)
        .build()?
        .generate_string()?
        .lines() // Fix some wierd bug with inner attributes
        .filter(|line| !line.trim().starts_with("#![allow"))
        .collect::<Vec<_>>()
        .join("\n");

    std::fs::write(out, src)?;

    Ok(())
}
