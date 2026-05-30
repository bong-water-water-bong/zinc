#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;

// Fused single-token Q/K-norm + Q/K-RoPE + Q8 KV-cache write for the kv_cache_q8
// decode path. Extends cycle-22's `rope_qk_norm_inplace.metal` by folding the
// follow-up `kv_cache_write_q8` dispatch (and its barrier) into this kernel:
//   - Q-heads (head < n_q_heads): Q-norm + Q-rope, in place in q_inout.
//   - K-heads (head < n_q_heads + n_kv_heads): K-norm + K-rope in place in
//     k_inout, then Q8-quantize this head's rotated K row directly into
//     kv_k_cache at `dst_offset_bytes` (current token slot).
//   - V-heads (head < n_q_heads + 2*n_kv_heads): Q8-quantize the corresponding
//     v_inout row directly into kv_v_cache at `dst_offset_bytes`.
//
// Adapts llama.cpp `ggml_metal_op_concurrency_check/reset` single-consumer
// fusion (ggml-metal-ops.cpp:159, 175) to the QK-norm → rope → kv-write chain
// that the Q8 KV decode path of Qwen3.6 walks every dense full-attn layer:
// since the only consumer of the rotated K (and the v_buf row) is the KV cache
// write that immediately follows, the rope→write barrier becomes redundant
// once both ops live in the same dispatch. Saves one dispatch + one barrier
// per dense full-attn layer (≈10/decode token on Qwen3.6-35B), extending the
// cycle-21/22 single-consumer fusion discipline to the KV-materialization edge.
//
// Requires head_dim % 64 == 0 so each K/V head splits evenly into 32-element
// Q8_0 blocks across both simdgroups of the 64-thread TG (blocks_per_head =
// head_dim / 32, blocks_per_simd = blocks_per_head / 2). Falls back to the
// cycle-22 unfused path otherwise. The pre-flash-attention barrier (which
// already includes kv_k_cache and kv_v_cache) makes the cache writes visible
// before flash_attn_q8 reads them.

struct RopeQkNormKvQ8Push {
    uint stride;            // head_dim
    uint rope_dim;
    uint n_q_heads;
    uint n_kv_heads;
    uint position;
    uint dst_offset_bytes;  // per-token byte offset within the layer's KV cache
    float eps;
};

kernel void main0(
    constant RopeQkNormKvQ8Push& p [[buffer(0)]],
    device float* q_inout       [[buffer(1)]],
    device float* k_inout       [[buffer(2)]],
    device const float* v_inout [[buffer(3)]],
    device const float* freqs   [[buffer(4)]],
    device const float* q_norm_w [[buffer(5)]],
    device const float* k_norm_w [[buffer(6)]],
    device uchar* kv_k_cache    [[buffer(7)]],
    device uchar* kv_v_cache    [[buffer(8)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    const uint stride = p.stride;
    const uint half_rot = p.rope_dim / 2;
    const uint blocks_per_head = stride / 32u;

    if (head < p.n_q_heads) {
        // Q branch — identical to rope_qk_norm_inplace.metal Q path.
        const uint base = head * stride;

        float sum_sq = 0.0f;
        for (uint i = simd_lane; i < stride; i += 32u) {
            const float v = q_inout[base + i];
            sum_sq += v * v;
        }
        sum_sq = simd_sum(sum_sq);
        const float q_rms_inv = fast::rsqrt(fast::divide(sum_sq, float(stride)) + p.eps);

        for (uint i = tid; i < half_rot; i += 64) {
            const float theta = float(p.position) * freqs[i];
            const float cos_t = fast::cos(theta);
            const float sin_t = fast::sin(theta);
            const float x0 = q_inout[base + i] * q_rms_inv * q_norm_w[i];
            const float x1 = q_inout[base + i + half_rot] * q_rms_inv * q_norm_w[i + half_rot];
            q_inout[base + i] = x0 * cos_t - x1 * sin_t;
            q_inout[base + i + half_rot] = x0 * sin_t + x1 * cos_t;
        }
        for (uint i = p.rope_dim + tid; i < stride; i += 64) {
            q_inout[base + i] = q_inout[base + i] * q_rms_inv * q_norm_w[i];
        }
        return;
    }

    if (head < p.n_q_heads + p.n_kv_heads) {
        // K branch — norm + rope (in place to k_inout) + Q8 quantize+write.
        const uint kv_head = head - p.n_q_heads;
        const uint base = kv_head * stride;

        float sum_sq = 0.0f;
        for (uint i = simd_lane; i < stride; i += 32u) {
            const float v = k_inout[base + i];
            sum_sq += v * v;
        }
        sum_sq = simd_sum(sum_sq);
        const float k_rms_inv = fast::rsqrt(fast::divide(sum_sq, float(stride)) + p.eps);

        for (uint i = tid; i < half_rot; i += 64) {
            const float theta = float(p.position) * freqs[i];
            const float cos_t = fast::cos(theta);
            const float sin_t = fast::sin(theta);
            const float x0 = k_inout[base + i] * k_rms_inv * k_norm_w[i];
            const float x1 = k_inout[base + i + half_rot] * k_rms_inv * k_norm_w[i + half_rot];
            k_inout[base + i] = x0 * cos_t - x1 * sin_t;
            k_inout[base + i + half_rot] = x0 * sin_t + x1 * cos_t;
        }
        for (uint i = p.rope_dim + tid; i < stride; i += 64) {
            k_inout[base + i] = k_inout[base + i] * k_rms_inv * k_norm_w[i];
        }

        // All K writes from both simdgroups in this TG must be visible before
        // re-reading for Q8 quantization (simdgroup 0 wrote indices 0..31 and
        // 64..95, simdgroup 1 wrote 32..63 and 96..127 plus the +half_rot
        // pairs — and the quantize block assignment crosses the simdgroup
        // boundary). The threadgroup_barrier is TG-local and far cheaper than
        // the encoder-scope barrier that the standalone kv_cache_write_q8
        // dispatch would otherwise need.
        threadgroup_barrier(mem_flags::mem_device);

        // Quantize this K head's rotated values into the layer's KV cache:
        // each simdgroup of the 64-thread TG owns half the head's blocks.
        const uint blocks_per_simd = blocks_per_head / 2u;
        const uint kv_block_base = kv_head * blocks_per_head;
        for (uint bi = 0; bi < blocks_per_simd; bi++) {
            const uint block_in_head = simd_id * blocks_per_simd + bi;
            const uint elem_offset = block_in_head * 32u + simd_lane;
            const float k_val = k_inout[base + elem_offset];
            const float k_abs_max = simd_max(fast::abs(k_val));
            const float k_scale = k_abs_max > 0.0f ? k_abs_max * (1.0f / 127.0f) : 0.0f;
            const float k_inv_scale = k_scale > 0.0f ? 1.0f / k_scale : 0.0f;

            device uchar* k_dst = kv_k_cache + p.dst_offset_bytes + (kv_block_base + block_in_head) * 34u;
            if (simd_lane == 0u) {
                *(device ushort*)(k_dst) = as_type<ushort>(half(k_scale));
            }
            const int q = clamp(int(rint(k_val * k_inv_scale)), -127, 127);
            k_dst[2u + simd_lane] = as_type<uchar>(char(q));
        }
        return;
    }

    // V branch — Q8 quantize+write only (V needs neither rope nor norm on the
    // Qwen3.6 Q8 KV path). Runs concurrently with the Q/K work above when the
    // hardware has enough TG slack.
    const uint v_head = head - p.n_q_heads - p.n_kv_heads;
    const uint base = v_head * stride;
    const uint blocks_per_simd = blocks_per_head / 2u;
    const uint kv_block_base = v_head * blocks_per_head;
    for (uint bi = 0; bi < blocks_per_simd; bi++) {
        const uint block_in_head = simd_id * blocks_per_simd + bi;
        const uint elem_offset = block_in_head * 32u + simd_lane;
        const float v_val = v_inout[base + elem_offset];
        const float v_abs_max = simd_max(fast::abs(v_val));
        const float v_scale = v_abs_max > 0.0f ? v_abs_max * (1.0f / 127.0f) : 0.0f;
        const float v_inv_scale = v_scale > 0.0f ? 1.0f / v_scale : 0.0f;

        device uchar* v_dst = kv_v_cache + p.dst_offset_bytes + (kv_block_base + block_in_head) * 34u;
        if (simd_lane == 0u) {
            *(device ushort*)(v_dst) = as_type<ushort>(half(v_scale));
        }
        const int q = clamp(int(rint(v_val * v_inv_scale)), -127, 127);
        v_dst[2u + simd_lane] = as_type<uchar>(char(q));
    }
}
