#import splat_common::{GpuSplat, div_ceil, BIN_PART_SIZE}

struct PushConstants {
    camera_view_row_z: vec4<f32>,
    view_proj: mat4x4<f32>,
    count: u32,
    use_culling: u32,
    invert_culling: u32,
}

struct DispatchIndirectArgs {
    x: u32,
    y: u32,
    z: u32,
}

struct DrawIndexedIndirectArgs {
    /// The number of indices to draw.
    index_count: u32,
    /// The number of instances to draw.
    instance_count: u32,
    /// The first index within the index buffer.
    first_index: u32,
    /// The value added to the vertex index before indexing into the vertex buffer.
    base_vertex: i32,
    /// The instance ID of the first instance to draw.
    ///
    /// Has to be 0, unless [`Features::INDIRECT_FIRST_INSTANCE`](crate::Features::INDIRECT_FIRST_INSTANCE) is enabled.
    first_instance: u32,
}

var<immediate> pc: PushConstants;

@group(0) @binding(0) var<storage, read> splats: array<GpuSplat>;
@group(0) @binding(1) var<storage, read_write> u32_depths: array<u32>;
@group(0) @binding(2) var<storage, read_write> u32_indices: array<u32>;

@group(0) @binding(3) var<storage, read_write> atomic_count: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> dispatch_indirect_args: DispatchIndirectArgs;
@group(0) @binding(5) var<storage, read_write> draw_indexed_indirect_args: DrawIndexedIndirectArgs;

fn f32_sortable_bits(f: f32) -> u32 {
    let bits = bitcast<u32>(f);
    let mask = bitcast<u32>(bitcast<i32>(bits) >> 31u) | 0x80000000u;
    return bits ^ mask;
}

fn in_frustum(pos: vec3<f32>) -> bool {
    let clip_pos = pc.view_proj * vec4<f32>(pos, 1.0);

    // reject points behind the near plane or camera plane
    if clip_pos.w <= 0.0 {
        return false;
    }

    let margin = 0.2 * clip_pos.w;
    let limit = clip_pos.w + margin;

    return clip_pos.x >= -limit && clip_pos.x <= limit &&
           clip_pos.y >= -limit && clip_pos.y <= limit &&
           clip_pos.z >= 0.0 && clip_pos.z <= limit;
}

const WG_SIZE: u32 = 256;
@compute @workgroup_size(WG_SIZE)
fn cs_prepare(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(subgroup_invocation_id) sid: u32
) {
    let idx = gid.x;

    var visible = false;
    var depth = 0.0;

    if idx < pc.count {
        let splat = splats[idx];

        if pc.use_culling == 1u {
            let inside = in_frustum(splat.position);
            if pc.invert_culling == 1u {
                visible = !inside;
            } else {
                visible = inside;
            }
        } else {
            visible = true;
        }

        if visible {
            depth = dot(pc.camera_view_row_z, vec4<f32>(splat.position, 1.0));
        }
    }

    let visible_u32 = u32(visible);
    let local_offset = subgroupExclusiveAdd(visible_u32);
    let subgroup_total = subgroupAdd(visible_u32);

    var subgroup_base_idx: u32 = 0;
    if sid == subgroupMin(sid) {
        if subgroup_total > 0 {
            subgroup_base_idx = atomicAdd(&atomic_count[0], subgroup_total);
        }
    }
    subgroup_base_idx = subgroupBroadcastFirst(subgroup_base_idx);

    if visible {
        let compacted_idx = subgroup_base_idx + local_offset;

        u32_depths[compacted_idx] = f32_sortable_bits(depth);
        u32_indices[compacted_idx] = idx;
    }
}

@compute @workgroup_size(1)
fn cs_prepare_indirect_args() {
    let total_count = atomicLoad(&atomic_count[0]);
    // The sorter workgroups process BIN_PART_SIZE (7680) elements each.
    dispatch_indirect_args.x = div_ceil(total_count, BIN_PART_SIZE);
    dispatch_indirect_args.y = 1;
    dispatch_indirect_args.z = 1;

    draw_indexed_indirect_args.index_count = 6;
    draw_indexed_indirect_args.instance_count = total_count;
    draw_indexed_indirect_args.first_index = 0;
    draw_indexed_indirect_args.base_vertex = 0;
    draw_indexed_indirect_args.first_instance = 0;
}
