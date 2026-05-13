#import splat_common::GpuSplat

struct PushConstants {
    camera_view_row_z: vec4<f32>,
    count: u32,
}

var<immediate> pc: PushConstants;

@group(0) @binding(0) var<storage, read> splats: array<GpuSplat>;
@group(0) @binding(1) var<storage, read_write> u32_depths: array<u32>;
@group(0) @binding(2) var<storage, read_write> u32_indices: array<u32>;

fn f32_sortable_bits(f: f32) -> u32 {
    let bits = bitcast<u32>(f);
    let mask = bitcast<u32>(bitcast<i32>(bits) >> 31u) | 0x80000000u;
    return bits ^ mask;
}

const WG_SIZE: u32 = 256;

@compute @workgroup_size(WG_SIZE)
fn cs_prepare(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= pc.count { return; }

    let splat = splats[idx];
    // view_z = dot(view_row_2, [pos, 1.0])
    // More negative = further away. Sorting ascending gives Back-to-Front.
    let depth = dot(pc.camera_view_row_z, vec4<f32>(splat.position, 1.0));
    u32_depths[idx] = f32_sortable_bits(depth);
    u32_indices[idx] = idx;
}
