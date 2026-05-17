#import splat_common

// ── Standard alpha-blended Gaussian splatting ─────────────────────────────────

@vertex
fn vs_main(
    @builtin(vertex_index) vi: u32,
    @builtin(instance_index) ii: u32
) -> splat_common::VertexOut {
    return splat_common::compute_vertex(vi, ii);
}

@fragment
fn fs_main(in: splat_common::VertexOut) -> @location(0) vec4<f32> {
    let gauss = splat_common::evaluate_gaussian(in);
    let alpha = in.color_opacity.a * gauss;

    if alpha < 1.0 / 255.0 { discard; }

    return vec4<f32>(in.color_opacity.rgb, alpha);
}
