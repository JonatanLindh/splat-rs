use std::{env, path::PathBuf};

use wgsl_bindgen::{
    GlamWgslTypeMap, WgslBindgenOptionBuilder, WgslShaderIrCapabilities, WgslShaderSourceType,
    WgslTypeSerializeStrategy,
};

const SHADERS: &[&str] = &[
    "splat.wgsl",
    "splat_stochastic.wgsl",
    "sort.wgsl",
    "prepare.wgsl",
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let out = out_dir.join("shaders.rs");

    let mut builder = WgslBindgenOptionBuilder::default();
    builder
        .workspace_root("shaders")
        .shader_source_type(
            WgslShaderSourceType::EmbedSource | WgslShaderSourceType::ComposerWithRelativePath,
        )
        .serialization_strategy(WgslTypeSerializeStrategy::Bytemuck)
        .type_map(GlamWgslTypeMap)
        .short_constructor(4)
        .skip_header_comments(true)
        .ir_capabilities(WgslShaderIrCapabilities::all());

    for shader in SHADERS {
        builder.add_entry_point(format!("shaders/{}", shader));
    }

    let src = builder
        .build()?
        .generate_string()?
        .lines()
        .filter(|line| !line.trim().starts_with("#![allow"))
        .collect::<Vec<_>>()
        .join("\n");

    std::fs::write(out, src)?;

    Ok(())
}
