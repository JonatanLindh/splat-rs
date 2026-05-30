#import splat_common::{div_ceil, BIN_PART_SIZE}

struct PushConstants {
    shift: u32,
    pass_index: u32,
}

var<immediate> pc: PushConstants;

const RADIX: u32 = 256u;
const RADIX_MASK: u32 = 255u;

@group(0) @binding(0) var<storage, read_write> in_keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> in_payload: array<u32>;

@group(0) @binding(2) var<storage, read_write> out_keys: array<u32>;
@group(0) @binding(3) var<storage, read_write> out_payload: array<u32>;

@group(0) @binding(4) var<storage, read_write> pass_histograms: array<atomic<u32>>;

@group(0) @binding(5) var<storage, read> count: u32;

var<workgroup> s_localHistogram: array<atomic<u32>, RADIX>;
var<workgroup> s_scan_temp: array<u32, RADIX>;

@compute @workgroup_size(RADIX)
fn cs_count_pass(
    @builtin(local_invocation_id) tid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let block_id = wid.x;
    let start_idx = block_id * BIN_PART_SIZE;

    // clear shared memory histogram
    atomicStore(&s_localHistogram[tid.x], 0u);
    workgroupBarrier();

    // count elements in this partition
    for (var i = tid.x; i < BIN_PART_SIZE; i += RADIX) {
        let global_idx = start_idx + i;
        if global_idx < count {
            let key = in_keys[global_idx];
            let digit = (key >> pc.shift) & RADIX_MASK;
            atomicAdd(&s_localHistogram[digit], 1u);
        }
    }
    workgroupBarrier();

    // write local counts to the global spine
    let num_blocks = div_ceil(count, BIN_PART_SIZE);
    let pass_offset = pc.pass_index * (num_blocks * RADIX);
    let local_count = atomicLoad(&s_localHistogram[tid.x]);
    atomicStore(&pass_histograms[pass_offset + (block_id * RADIX) + tid.x], local_count);
}

@compute @workgroup_size(RADIX)
fn cs_scan_pass(
    @builtin(local_invocation_id) tid: vec3<u32>,
    @builtin(subgroup_size) sg_size: u32,
    @builtin(subgroup_id) subgroup_id: u32,
    @builtin(subgroup_invocation_id) lane_in_sg: u32,
    @builtin(num_subgroups) num_subgroups: u32,
) {
    let digit = tid.x;
    let num_blocks = div_ceil(count, BIN_PART_SIZE);
    let pass_offset = pc.pass_index * (num_blocks * RADIX);

    // find the total count of this digit across all blocks
    var my_digit_total = 0u;
    for (var b = 0u; b < num_blocks; b++) {
        my_digit_total += atomicLoad(&pass_histograms[pass_offset + (b * RADIX) + digit]);
    }

    // Inclusive prefix sum over the 256 digit totals

    // subgroup scan
    let local_prefix = subgroupInclusiveAdd(my_digit_total);

    // last lane of each subgroup holds the whole subgroup's sum — collect them
    if lane_in_sg == sg_size - 1 {
        s_scan_temp[subgroup_id] = local_prefix;
    }
    workgroupBarrier();

    // convert the list of subgroup totals to an exclusive prefix sum
    if digit == 0u {
        var acc = 0u;
        for (var i = 0u; i < num_subgroups; i++) {
            let v = s_scan_temp[i];
            s_scan_temp[i] = acc;
            acc += v;
        }
    }
    workgroupBarrier();

    // combine
    let global_prefix = local_prefix + s_scan_temp[subgroup_id];
    workgroupBarrier();
    s_scan_temp[digit] = global_prefix;
    workgroupBarrier();

    var base_offset = 0u;
    if digit > 0u {
        base_offset = s_scan_temp[digit - 1u];
    }

    // distribute prefix sum back to the spine
    var running_offset = base_offset;
    for (var b = 0u; b < num_blocks; b++) {
        let idx = pass_offset + (b * RADIX) + digit;
        let block_count = atomicLoad(&pass_histograms[idx]);
        atomicStore(&pass_histograms[idx], running_offset);
        running_offset += block_count;
    }
}

@compute @workgroup_size(RADIX)
fn cs_scatter_pass(
    @builtin(local_invocation_id) tid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let block_id = wid.x;
    let num_blocks = div_ceil(count, BIN_PART_SIZE);
    let start_idx = block_id * BIN_PART_SIZE;
    let pass_offset = pc.pass_index * (num_blocks * RADIX);

    // fetch global base offsets for this block into shared memory
    if tid.x < RADIX {
        let global_base = atomicLoad(&pass_histograms[pass_offset + (block_id * RADIX) + tid.x]);
        atomicStore(&s_localHistogram[tid.x], global_base);
    }
    workgroupBarrier();

    for (var i = tid.x; i < BIN_PART_SIZE; i += RADIX) {
        let global_idx = start_idx + i;
        if global_idx >= count { continue; }

        let key = in_keys[global_idx];
        let payload = in_payload[global_idx];
        let digit = (key >> pc.shift) & RADIX_MASK;

        // AtomicAdd returns the value BEFORE the addition (the rank)
        let local_rank = atomicAdd(&s_localHistogram[digit], 1u);

        out_keys[local_rank] = key;
        out_payload[local_rank] = payload;
    }
}
