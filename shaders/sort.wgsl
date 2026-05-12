struct PushConstants {
    size: u32,
    shift: u32,
    pass_index: u32,
}

var<immediate> pc: PushConstants;

const RADIX: u32 = 256u;
const RADIX_MASK: u32 = 255u;

const SEC_RADIX_START: u32 = RADIX;
const THIRD_RADIX_START: u32 = RADIX * 2;
const FOURTH_RADIX_START: u32 = RADIX * 3;

const BIN_PART_SIZE: u32 = 7680u;
const BIN_HISTS_SIZE: u32 = 4096u;
const BIN_KEYS_PER_THREAD: u32 = 15u;

const MIN_SUBGROUP_SIZE = 4u;
const MAX_REDUCE_SIZE = RADIX / MIN_SUBGROUP_SIZE;

@group(0) @binding(0) var<storage, read_write> in_keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> in_payload: array<u32>;

@group(0) @binding(2) var<storage, read_write> out_keys: array<u32>;
@group(0) @binding(3) var<storage, read_write> out_payload: array<u32>;

@group(0) @binding(4) var<storage, read_write> pass_histograms: array<atomic<u32>>;

var<workgroup> s_wlms_hists: array<u32, BIN_HISTS_SIZE>;
var<workgroup> s_localHistogram: array<atomic<u32>, RADIX>;

// Helper for div_ceil
fn div_ceil(a: u32, b: u32) -> u32 {
    return (a + b - 1u) / b;
}

struct WLMSResult {
    bits: u32,
    total_matches: u32,
    leader_lane: u32,
}

fn get_wlms_offsets(digit: u32, sid: u32, subgroup_size: u32) -> WLMSResult {
    // find all lanes that have the exact same digit
    var m = vec4<u32>(0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu);
    for (var k = 0u; k < 8u; k++) {
        let bit_set = ((digit >> k) & 1u) != 0u;
        let b = subgroupBallot(bit_set);
        m &= select(~b, b, bit_set);
    }

    let full = subgroup_size >> 5u; // number of fully-used u32 words
    let rem = subgroup_size & 31u;
    for (var w = 0u; w < 4u; w++) {
        if w > full {
            m[w] = 0u;
        } else if w == full && rem != 0u {
            m[w] &= (1u << rem) - 1u;
        }
    }

    let word = sid >> 5u;
    let bit = sid & 31u;
    var bits = 0u;
    for (var w = 0u; w < 4u; w++) {
        if w < word {
            bits += countOneBits(m[w]);
        } else if w == word {
            bits += countOneBits(m[w] & ((1u << bit) - 1u));
        }
    }

    let total_matches = countOneBits(m.x) + countOneBits(m.y) + countOneBits(m.z) + countOneBits(m.w);

    var leader_lane = 0u;
    if m.x != 0u {
        leader_lane = countTrailingZeros(m.x);
    } else if m.y != 0u {
        leader_lane = 32u + countTrailingZeros(m.y);
    } else if m.z != 0u {
        leader_lane = 64u + countTrailingZeros(m.z);
    } else {
        leader_lane = 96u + countTrailingZeros(m.w);
    }

    return WLMSResult(bits, total_matches, leader_lane);
}

@compute @workgroup_size(RADIX)
fn cs_count_pass(
    @builtin(local_invocation_id) tid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let block_id = wid.x;
    let num_blocks = div_ceil(pc.size, BIN_PART_SIZE);
    let start_idx = block_id * BIN_PART_SIZE;

    // clear shared memory histogram
    atomicStore(&s_localHistogram[tid.x], 0u);
    workgroupBarrier();

    // count elements in this partition
    for (var i = tid.x; i < BIN_PART_SIZE; i += RADIX) {
        let global_idx = start_idx + i;
        if global_idx < pc.size {
            let key = in_keys[global_idx];
            let digit = (key >> pc.shift) & RADIX_MASK;
            atomicAdd(&s_localHistogram[digit], 1u);
        }
    }
    workgroupBarrier();

    // write local counts to the global spine
    let pass_offset = pc.pass_index * (num_blocks * RADIX);
    let count = atomicLoad(&s_localHistogram[tid.x]);
    atomicStore(&pass_histograms[pass_offset + (block_id * RADIX) + tid.x], count);
}

@compute @workgroup_size(RADIX)
fn cs_scan_pass(
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let digit = tid.x;
    let num_blocks = div_ceil(pc.size, BIN_PART_SIZE);
    let pass_offset = pc.pass_index * (num_blocks * RADIX);

    // find the total count of this digit across all blocks
    var my_digit_total = 0u;
    for (var b = 0u; b < num_blocks; b++) {
        my_digit_total += atomicLoad(&pass_histograms[pass_offset + (b * RADIX) + digit]);
    }

    // store totals to shared memory
    atomicStore(&s_localHistogram[digit], my_digit_total);
    workgroupBarrier();

    // global base offset for this digit
    var base_offset = 0u;
    for (var i = 0u; i < digit; i++) {
        base_offset += atomicLoad(&s_localHistogram[i]);
    }

    // distribute prefix sum back to the spine
    var running_offset = base_offset;
    for (var b = 0u; b < num_blocks; b++) {
        let idx = pass_offset + (b * RADIX) + digit;
        let count = atomicLoad(&pass_histograms[idx]);

        // overwrite the count with the absolute global starting offset
        atomicStore(&pass_histograms[idx], running_offset);
        running_offset += count;
    }
}

@compute @workgroup_size(RADIX * 2)
fn cs_scatter_pass(
    @builtin(local_invocation_id) tid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_invocation_id) sid: u32,
    @builtin(subgroup_size) subgroup_size: u32,
    @builtin(subgroup_id) subgroup_id: u32
) {
    let block_id = wid.x;
    let num_blocks = div_ceil(pc.size, BIN_PART_SIZE);
    let start_idx = block_id * BIN_PART_SIZE;
    let pass_offset = pc.pass_index * (num_blocks * RADIX);

    // fetch global base offsets from the spine
    if tid.x < RADIX {
        let global_base = atomicLoad(&pass_histograms[pass_offset + (block_id * RADIX) + tid.x]);
        atomicStore(&s_localHistogram[tid.x], global_base);
    }

    // clear WLMS shmem
    for (var i = tid.x; i < BIN_HISTS_SIZE; i += 512u) {
        s_wlms_hists[i] = 0;
    }
    workgroupBarrier();

    let sg_hist_offset = subgroup_id * RADIX;
    var keys: array<u32, BIN_KEYS_PER_THREAD>;
    var payloads: array<u32, BIN_KEYS_PER_THREAD>;
    var offsets: array<u32, BIN_KEYS_PER_THREAD>;

    let local_start = subgroup_id * subgroup_size * BIN_KEYS_PER_THREAD + sid;

    // load keys and payload
    for (var i = 0u; i < BIN_KEYS_PER_THREAD; i++) {
        let fetch_idx = start_idx + local_start + (i * subgroup_size);
        if fetch_idx < pc.size {
            keys[i] = in_keys[fetch_idx];
            payloads[i] = in_payload[fetch_idx];
        } else {
            keys[i] = 0xFFFFFFFFu; // dummy for out-of-bounds
        }
    }

    // local offsets
    for (var i = 0u; i < BIN_KEYS_PER_THREAD; i++) {
        let digit = (keys[i] >> pc.shift) & RADIX_MASK;
        let res = get_wlms_offsets(digit, sid, subgroup_size);

        var pre_increment_val: u32 = 0;
        if res.bits == 0u {
            pre_increment_val = s_wlms_hists[sg_hist_offset + digit];
            s_wlms_hists[sg_hist_offset + digit] += res.total_matches;
        }

        offsets[i] = subgroupShuffle(pre_increment_val, res.leader_lane) + res.bits;
    }
    workgroupBarrier();

    // prefix Sum wlms subgroup hists
    if tid.x < RADIX {
        var reduction = 0u;
        for (var i = tid.x; i < BIN_HISTS_SIZE; i += RADIX) {
            let sh_i = s_wlms_hists[i];
            s_wlms_hists[i] = reduction;
            reduction += sh_i;
        }
    }
    workgroupBarrier();

    // scatter to global memory
    for (var i = 0u; i < BIN_KEYS_PER_THREAD; i++) {
        if start_idx + local_start + (i * subgroup_size) >= pc.size { continue; }

        let digit = (keys[i] >> pc.shift) & RADIX_MASK;

        // rank of this key inside this block
        let local_wlms_offset = offsets[i] + s_wlms_hists[sg_hist_offset + digit];

        // global absolute base + local block rank
        let dest_idx = atomicLoad(&s_localHistogram[digit]) + local_wlms_offset;

        out_keys[dest_idx] = keys[i];
        out_payload[dest_idx] = payloads[i];
    }
}
