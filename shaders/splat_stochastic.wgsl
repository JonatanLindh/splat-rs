#import splat_common

// ── Stochastic (ish) transparency Gaussian splatting ────────────────────────────────

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> splat_common::VertexOut {
    return splat_common::compute_vertex(vi);
}

@fragment
fn fs_main(in: splat_common::VertexOut) -> @location(0) vec4<f32> {
    let gauss = splat_common::evaluate_gaussian(in);
    let alpha = in.color_opacity.a * gauss;

    // Reject very transparent pixels early (stochastic transparency handles the rest)
    if alpha < 0.01 { discard; }

    return vec4<f32>(in.color_opacity.rgb, alpha);
}
