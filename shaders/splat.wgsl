struct CameraUniform {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    viewport: vec2<f32>,
    camera_pos: vec3<f32>,   // world-space eye position
}

@group(0) @binding(0) var<uniform> camera: CameraUniform;

struct GpuSplat {
    position: vec3<f32>,
    opacity: f32,
    scale: vec3<f32>,           // log-space
    rotation: vec4<f32>,           // x, y, z, w
    sh_dc: vec3<f32>,
    sh_rest: array<f32, 45>,      // higher-order SH coefficients
}

@group(0) @binding(1) var<storage, read> splats: array<GpuSplat>;

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) screen_offset: vec2<f32>, // pixels from splat centre
    @location(1) cov2d_inv: vec4<f32>,     // packed [a, b, b, d] of inv(Σ2D)
    @location(2) color_opacity: vec4<f32>, // rgb + sigmoid(opacity)
}

// Build a 3×3 rotation matrix from quaternion stored as (w, x, y, z)
fn quat_to_mat3(q: vec4<f32>) -> mat3x3<f32> {
    let x = q.x;  let y = q.y;  let z = q.z; let w = q.w;
    return mat3x3<f32>(
        vec3<f32>(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)),
        vec3<f32>(2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x)),
        vec3<f32>(2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)),
    );
}

fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

// Spherical Harmonics (SH) basis coefficients.
//
// Sources:
// 1. "3D Gaussian Splatting for Real-Time Radiance Field Rendering" (Kerbl et al., 2023)
//    - Matches the math used in the official `diff-gaussian-rasterization` CUDA implementation.
// 2. "Spherical Harmonic Lighting: The Gritty Details" (Robin Green, 2003)
//    - Standard reference for the derivation of these orthonormal basis constants.

const PI: f32 = 3.141592653589793;

// Degree 0
const SH_C0: f32 = 0.5 * sqrt(1.0 / PI);

// Degree 1
const SH_C1: f32 = 0.5 * sqrt(3.0 / PI);

// Degree 2
const SH_C2_0: f32 = 0.5 * sqrt(15.0 / PI);
const SH_C2_1: f32 = -0.5 * sqrt(15.0 / PI);
const SH_C2_2: f32 = 0.25 * sqrt(5.0 / PI);
const SH_C2_3: f32 = -0.5 * sqrt(15.0 / PI);
const SH_C2_4: f32 = 0.25 * sqrt(15.0 / PI);

// Degree 3
const SH_C3_0: f32 = -0.25 * sqrt(35.0 / (2.0 * PI));
const SH_C3_1: f32 = 0.5 * sqrt(105.0 / PI);
const SH_C3_2: f32 = -0.25 * sqrt(21.0 / (2.0 * PI));
const SH_C3_3: f32 = 0.25 * sqrt(7.0 / PI);
const SH_C3_4: f32 = -0.25 * sqrt(21.0 / (2.0 * PI));
const SH_C3_5: f32 = 0.25 * sqrt(105.0 / PI);
const SH_C3_6: f32 = -0.25 * sqrt(35.0 / (2.0 * PI));

// Two triangles = one quad; corners in local 2D space.
const QUAD: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
);

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOut {
    let splat_id = vi / 6u;
    let corner_id = vi % 6u;
    let splat = splats[splat_id];

    // We can fuse some math here to avoid a matrix multiplication per gaussian
    // S = diag(exp(scale))
    // Σ = R * S * S^T * R^T
    // WΣ = W * Σ * W^T = W * (R * S * S^T * R^T) * W^T = (W * R * S) * (W * R * S)^T
    let R = quat_to_mat3(splat.rotation);
    let s = exp(splat.scale);
    let RS = mat3x3<f32>(
        R[0] * s.x,
        R[1] * s.y,
        R[2] * s.z
    );

    // W = upper-left 3x3 of the view matrix (rotation + scale, no translation).
    let W = mat3x3<f32>(
        camera.view[0].xyz,
        camera.view[1].xyz,
        camera.view[2].xyz,
    );

    let WRS = W * RS;
    let Wcov = WRS * transpose(WRS);

    // View-space position of the splat centre.
    let view_pos = camera.view * vec4<f32>(splat.position, 1.0);
    let t = view_pos.xyz;
    let tz = -t.z;  // positive depth in right-handed view space

    // Guard: skip splats behind or on the camera plane.
    if tz < 0.001 {
        var out: VertexOut;
        out.clip_pos = vec4<f32>(0.0, 0.0, 2.0, 1.0);  // clip away
        out.screen_offset = vec2<f32>(0.0);
        out.cov2d_inv = vec4<f32>(0.0);
        out.color_opacity = vec4<f32>(0.0);
        return out;
    }

    // Extract focal lengths from the projection matrix.
    //   proj[0][0] = 2*fx / width   →  fx = proj[0][0] * width/2
    let fx = camera.proj[0][0] * camera.viewport.x * 0.5;
    let fy = camera.proj[1][1] * camera.viewport.y * 0.5;

    // clamp perspective projection in camera space to avoid Jacobian blowup at frustum edges
    let limx = 1.3 * camera.viewport.x * 0.5 / fx;
    let limy = 1.3 * camera.viewport.y * 0.5 / fy;
    let tx_c = clamp(t.x / tz, -limx, limx) * tz;
    let ty_c = clamp(t.y / tz, -limy, limy) * tz;

    // Jacobian rows: J = [[fx/tz, 0, -fx*tx/tz²], [0, fy/tz, -fy*ty/tz²]]
    let jx = vec3<f32>(fx / tz, 0.0, -fx * tx_c / (tz * tz));
    let jy = vec3<f32>(0.0, fy / tz, -fy * ty_c / (tz * tz));

    // Σ2D = J * WΣ * J^T  (compute component-wise to avoid WGSL mat confusion)
    var a = dot(jx, Wcov * jx);
    let b = dot(jx, Wcov * jy);
    var d = dot(jy, Wcov * jy);

    // low-pass anti-alias filter (from 3DGS paper's impl)
    a += 0.3;
    d += 0.3;

    // The eigenvalues are the solutions to λ² - tr(Σ2D)λ + det(Σ2D) = 0
    // We have a 2×2 symmetric matrix [a, b; b, d]:
    // tr = a + d
    // mid = tr/2
    // det = ad - b²
    // => λ = mid ± sqrt(mid² - det)
    let mid = 0.5 * (a + d);
    let det = a * d - b * b;
    let disc = sqrt(max(0.0, mid * mid - det));
    let lam1 = mid + disc;
    let lam2 = max(0.0, mid - disc);

    // variance = σ² = λ => σ = sqrt(λ)
    // Gaussian distr: 99.7% of energy within 3σ, don't bother with energy outside
    let extent1 = 3.0 * sqrt(lam1);
    let extent2 = 3.0 * sqrt(lam2);

    // fragment needs to eval exp(-0.5 * d^T * inv(Σ2D) * d), precompute inv(Σ2D) = [a, b; b, d]⁻¹ = 1/det * [d, -b; -b, a]
    let inv_det = 1.0 / max(det, 1e-6);
    let cov2d_inv = inv_det * vec4<f32>(d, -b, -b, a);

    // rotate quad to align with oval, need eigenvectors of Σ2D
    // [a, b; b, d] has an eigenvector (b, λ - a) for eigenvalue λ, unless b=0 or λ=a, then just use (1, 0)
    // axis 2 is perpendicular to axis 1, so (a - λ, b) or (0, 1)
    var axis1 = vec2<f32>(1.0, 0.0);
    let denom = lam1 - a;
    if abs(b) > 1e-5 || abs(denom) > 1e-5 {
        axis1 = normalize(vec2<f32>(b, denom));
    }
    let axis2 = vec2<f32>(-axis1.y, axis1.x);

    let corner = QUAD[corner_id];
    let screen_offset = corner.x * axis1 * extent1 + corner.y * axis2 * extent2;

    // project to clip space
    let clip_centre = camera.proj * view_pos;
    let ndc_centre = clip_centre.xy / clip_centre.w;

    // screen-pixel offset to ndc offset
    // we don't want the gpu to do perspective division, we do it ourselves and pass w=1
    let ndc_offset = screen_offset / (camera.viewport * 0.5);
    let clip_pos = vec4<f32>(ndc_centre + ndc_offset, clip_centre.z / clip_centre.w, 1.0);

    // color — evaluate SH up to degree 3 for view-dependent appearance
    let dir = normalize(splat.position - camera.camera_pos);
    let x = dir.x; let y = dir.y; let z = dir.z;

    var color = vec3<f32>(0.5) + SH_C0 * splat.sh_dc;

    // Degree 1
    color -= SH_C1 * y * vec3(splat.sh_rest[0], splat.sh_rest[15], splat.sh_rest[30]);
    color += SH_C1 * z * vec3(splat.sh_rest[1], splat.sh_rest[16], splat.sh_rest[31]);
    color -= SH_C1 * x * vec3(splat.sh_rest[2], splat.sh_rest[17], splat.sh_rest[32]);

    // Degree 2
    let xx = x * x; let yy = y * y; let zz = z * z;
    let xy = x * y; let yz = y * z; let xz = x * z;
    color += SH_C2_0 * xy * vec3(splat.sh_rest[3], splat.sh_rest[18], splat.sh_rest[33]);
    color += SH_C2_1 * yz * vec3(splat.sh_rest[4], splat.sh_rest[19], splat.sh_rest[34]);
    color += SH_C2_2 * (2. * zz - xx - yy) * vec3(splat.sh_rest[5], splat.sh_rest[20], splat.sh_rest[35]);
    color += SH_C2_3 * xz * vec3(splat.sh_rest[6], splat.sh_rest[21], splat.sh_rest[36]);
    color += SH_C2_4 * (xx - yy) * vec3(splat.sh_rest[7], splat.sh_rest[22], splat.sh_rest[37]);

    // Degree 3
    color += SH_C3_0 * y * (3. * xx - yy) * vec3(splat.sh_rest[8], splat.sh_rest[23], splat.sh_rest[38]);
    color += SH_C3_1 * xy * z * vec3(splat.sh_rest[9], splat.sh_rest[24], splat.sh_rest[39]);
    color += SH_C3_2 * y * (4. * zz - xx - yy) * vec3(splat.sh_rest[10], splat.sh_rest[25], splat.sh_rest[40]);
    color += SH_C3_3 * z * (2. * zz - 3. * xx - 3. * yy) * vec3(splat.sh_rest[11], splat.sh_rest[26], splat.sh_rest[41]);
    color += SH_C3_4 * x * (4. * zz - xx - yy) * vec3(splat.sh_rest[12], splat.sh_rest[27], splat.sh_rest[42]);
    color += SH_C3_5 * z * (xx - yy) * vec3(splat.sh_rest[13], splat.sh_rest[28], splat.sh_rest[43]);
    color += SH_C3_6 * x * (xx - 3. * yy) * vec3(splat.sh_rest[14], splat.sh_rest[29], splat.sh_rest[44]);

    let rgb = clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
    let alpha = sigmoid(splat.opacity);

    var out: VertexOut;
    out.clip_pos = clip_pos;
    out.screen_offset = screen_offset;
    out.cov2d_inv = cov2d_inv;
    out.color_opacity = vec4<f32>(rgb, alpha);
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let d = in.screen_offset;
    let ia = in.cov2d_inv.x;
    let ib = in.cov2d_inv.y;
    let id = in.cov2d_inv.w;

    // -0.5 * d^T * inv(Σ2D) * d
    let power = -0.5 * (d.x * d.x * ia
                      + 2.0 * d.x * d.y * ib
                      + d.y * d.y * id);

    // exp(-4.6) ≈ 0.01 — below this the contribution is invisible.
    if power < -4.6 { discard; }

    let gauss = exp(power);
    let alpha = in.color_opacity.a * gauss;

    if alpha < 1.0 / 255.0 { discard; }

    return vec4<f32>(in.color_opacity.rgb, alpha);
}

@fragment
fn fs_main_stochastic(in: VertexOut) -> @location(0) vec4<f32> {
    let d = in.screen_offset;
    let ia = in.cov2d_inv.x;
    let ib = in.cov2d_inv.y;
    let id = in.cov2d_inv.w;

    let power = -0.5 * (d.x * d.x * ia + 2.0 * d.x * d.y * ib + d.y * d.y * id);

    if power < -4.6 { discard; }

    let gauss = exp(power);
    let alpha = in.color_opacity.a * gauss;

    // Reject very transparent pixels early
    if alpha < 0.01 { discard; }

    return vec4<f32>(in.color_opacity.rgb, alpha);
}
