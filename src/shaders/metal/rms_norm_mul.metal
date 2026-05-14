#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct Params {
    uint n;
    float eps;
};

// Multi-simdgroup RMS norm with threadgroup memory reduction.
// Uses up to 1024 threads (32 simdgroups) for better float32 precision
// when accumulating sum-of-squares over large vectors with wide value ranges.
kernel void main0(
    constant Params& p [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    device const float* weights [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    uint subgroup_size [[thread_execution_width]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    threadgroup float shmem[32]; // one slot per simdgroup

    const uint base = group_id * p.n;

    float sum_sq = 0.0f;
    for (uint i = tid; i < p.n; i += tg_size) {
        const float v = input[base + i];
        sum_sq += v * v;
    }

    // Reduce within simdgroup
    sum_sq = simd_sum(sum_sq);

    // Write per-simdgroup result to shared memory
    if (simd_lane == 0) {
        shmem[simd_group] = sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Every simdgroup does the final reduction + rms_inv independently.
    // Avoids the rms_inv broadcast barrier (used to be: thread 0 computes
    // rsqrt, writes to threadgroup mem, barrier, all 1024 threads read).
    // Work is duplicated 32× but Apple7 issues the same fast::rsqrt in
    // parallel across simdgroups; the saved barrier — ~144 dispatches/token
    // on Qwen3-8B — is net positive.
    const uint n_groups = (tg_size + subgroup_size - 1) / subgroup_size;
    float total = (simd_lane < n_groups) ? shmem[simd_lane] : 0.0f;
    total = simd_sum(total);
    const float rms_inv = fast::rsqrt(fast::divide(total, float(p.n)) + p.eps);

    for (uint i = tid; i < p.n; i += tg_size) {
        output[base + i] = weights[i] * (input[base + i] * rms_inv);
    }
}
