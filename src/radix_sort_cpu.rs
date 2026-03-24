use rayon::prelude::*;

pub fn f32_sortable_bits(f: f32) -> u32 {
    let bits = f.to_bits();
    let mask = ((bits as i32) >> 31) as u32 | (1 << 31);
    bits ^ mask
}

pub fn parallel_radix_sort(data: &mut [(u32, usize)]) {
    // 4 levels => 8 bits each
    const N_BUCKETS: usize = 1 << 8;

    let n = data.len();
    if n <= 1 {
        return;
    }

    let mut buf = vec![(0u32, 0usize); n];
    let mut src = &mut data[..];
    let mut dst = &mut buf[..];

    let num_threads = rayon::current_num_threads();
    let chunk_size = n.div_ceil(num_threads);

    // 4 levels, 8 bits in each
    for shift in [0, 8, 16, 24] {
        let thread_buckets_counts: Vec<[usize; _]> = src
            .par_chunks(chunk_size)
            .map(|thread_chunk| {
                let mut buckets = [0; N_BUCKETS];
                for &(val, _) in thread_chunk {
                    let bucket = ((val >> shift) & 0xFF) as usize;
                    buckets[bucket] += 1;
                }

                buckets
            })
            .collect();

        let num_chunks = thread_buckets_counts.len();
        let mut offsets = vec![[0; N_BUCKETS]; num_chunks];
        let mut total_bucket_counts = [0; N_BUCKETS];

        for bucket in 0..N_BUCKETS {
            for t in 0..num_chunks {
                offsets[t][bucket] = total_bucket_counts[bucket];
                total_bucket_counts[bucket] += thread_buckets_counts[t][bucket];
            }
        }

        let mut global_base = 0;
        for bucket in 0..N_BUCKETS {
            let current_base = global_base;
            global_base += total_bucket_counts[bucket];

            for thread_buckets in &mut offsets {
                thread_buckets[bucket] += current_base;
            }
        }

        // scatter
        let dst_ptr_addr = dst.as_mut_ptr() as usize;

        src.par_chunks(chunk_size)
            .zip(offsets.par_iter_mut())
            .for_each(|(chunk, thread_offsets)| {
                let dst_ptr = dst_ptr_addr as *mut (u32, usize);

                for &item in chunk {
                    let bucket = ((item.0 >> shift) & 0xFF) as usize;
                    let dst_index = thread_offsets[bucket];

                    // SAFETY: Every thread writes to disjoint indices
                    unsafe {
                        dst_ptr.add(dst_index).write(item);
                    }

                    thread_offsets[bucket] += 1;
                }
            });

        // Swap the buffers for the next pass
        std::mem::swap(&mut src, &mut dst);
    }
}

pub trait RadixSort {
    fn par_radix_sort_unstable(&mut self);
}

impl RadixSort for [(u32, usize)] {
    #[inline(always)]
    fn par_radix_sort_unstable(&mut self) {
        parallel_radix_sort(self);
    }
}

#[test]
fn radix_sort() {
    let original_floats = [3.20, -1.0, 2.71, 0.0, -3.20];

    let mut data: Vec<_> = original_floats
        .iter()
        .enumerate()
        .map(|(i, &f)| (f32_sortable_bits(f), i))
        .collect();

    parallel_radix_sort(&mut data);

    let sorted_floats: Vec<f32> = data.iter().map(|&(_, i)| original_floats[i]).collect();

    assert_eq!(sorted_floats, vec![-3.20, -1.0, 0.0, 2.71, 3.20]);
}
