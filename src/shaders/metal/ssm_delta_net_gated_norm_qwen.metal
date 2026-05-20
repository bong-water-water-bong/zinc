#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct Params {
    uint d_inner;
    uint dt_rank;
    uint head_v_dim;
    uint d_state;
    uint n_group;
    uint ssm_a_is_f16;
    uint dt_bias_is_f16;
    uint has_dt_bias;
    uint has_ssm_a;
    uint alpha_offset;
    uint beta_offset;
    uint z_offset;
    uint output_offset;
    uint norm_per_head;
};

kernel void main0(
    constant Params& p [[buffer(0)]],
    device const float* conv_out [[buffer(1)]],
    device const float* alpha [[buffer(2)]],
    device const float* dt_bias [[buffer(3)]],
    device const float* ssm_a [[buffer(4)]],
    device const float* beta [[buffer(5)]],
    device float* state [[buffer(6)]],
    device const float* z_gate [[buffer(7)]],
    device const float* norm_weight [[buffer(8)]],
    device float* output [[buffer(9)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint simd_width [[thread_execution_width]],
    uint simdgroups_per_tg [[simdgroups_per_threadgroup]]
) {
    if (head >= 32u) {
        return;
    }

    constexpr uint head_v_dim = 128u;
    constexpr uint d_state = 128u;
    constexpr uint qk_dim = 2048u;
    constexpr uint v_base0 = 4096u;
    constexpr float inv_sqrt_d_state = 0.08838834764831845f;

    threadgroup float q[128];
    threadgroup float k[128];
    threadgroup float delta_out[128];
    threadgroup float partial_q[4];
    threadgroup float partial_k[4];
    threadgroup float partial_sq[4];
    threadgroup float head_scalars[4]; // q_scale, k_scale, decay, beta

    const uint tg_threads = simd_width * simdgroups_per_tg;
    const uint simd_lane = tid & 31u;
    const uint simd_idx = tid >> 5u;
    const uint group = head & 15u;
    const uint q_base = group * d_state;
    const uint k_base = qk_dim + group * d_state;
    const uint v_base = v_base0 + head * head_v_dim;
    const uint head_state_base = head * head_v_dim * head_v_dim;

    float q_ss = 0.0f;
    float k_ss = 0.0f;
    for (uint i = tid; i < head_v_dim; i += tg_threads) {
        const float qv = conv_out[q_base + i];
        const float kv = conv_out[k_base + i];
        q[i] = qv;
        k[i] = kv;
        q_ss = fma(qv, qv, q_ss);
        k_ss = fma(kv, kv, k_ss);
    }

    const float q_sum = simd_sum(q_ss);
    const float k_sum = simd_sum(k_ss);
    if (simd_lane == 0u) {
        partial_q[simd_idx] = q_sum;
        partial_k[simd_idx] = k_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float q_norm_sq = 0.0f;
        float k_norm_sq = 0.0f;
        for (uint i = 0u; i < simdgroups_per_tg; ++i) {
            q_norm_sq += partial_q[i];
            k_norm_sq += partial_k[i];
        }
        const float alpha_raw = alpha[p.alpha_offset + head] + dt_bias[head];
        const float softplus_alpha = log(1.0f + exp(alpha_raw));
        head_scalars[0] = rsqrt(fast::max(q_norm_sq, 1.0e-13f)) * inv_sqrt_d_state;
        head_scalars[1] = rsqrt(fast::max(k_norm_sq, 1.0e-13f));
        head_scalars[2] = exp(softplus_alpha * ssm_a[head]);
        head_scalars[3] = 1.0f / (1.0f + exp(-beta[p.beta_offset + head]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float q_scale = head_scalars[0];
    const float k_scale = head_scalars[1];
    const float decay = head_scalars[2];
    const float beta_val = head_scalars[3];

    float local_sq = 0.0f;
    for (uint row = tid; row < head_v_dim; row += tg_threads) {
        const uint row_base = head_state_base + row * head_v_dim;
        float sk_raw = 0.0f;
        for (uint col = 0u; col < head_v_dim; ++col) {
            const uint state_idx = row_base + col;
            const float decayed = state[state_idx] * decay;
            state[state_idx] = decayed;
            sk_raw = fma(decayed, k[col], sk_raw);
        }

        const float v = conv_out[v_base + row];
        const float delta = beta_val * (v - sk_raw * k_scale);
        const float scaled_delta = delta * k_scale;
        float out_v = 0.0f;
        for (uint col = 0u; col < head_v_dim; ++col) {
            const float updated = fma(k[col], scaled_delta, state[row_base + col]);
            state[row_base + col] = updated;
            out_v = fma(updated, q[col], out_v);
        }
        out_v *= q_scale;
        delta_out[row] = out_v;
        local_sq = fma(out_v, out_v, local_sq);
    }

    const float sq_sum = simd_sum(local_sq);
    if (simd_lane == 0u) {
        partial_sq[simd_idx] = sq_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float total_sq = 0.0f;
        for (uint i = 0u; i < simdgroups_per_tg; ++i) {
            total_sq += partial_sq[i];
        }
        partial_sq[0] = total_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float rms = rsqrt((partial_sq[0] / float(head_v_dim)) + 1.0e-6f);
    for (uint row = tid; row < head_v_dim; row += tg_threads) {
        const uint idx = head * head_v_dim + row;
        const uint weight_idx = (p.norm_per_head != 0u) ? idx : row;
        const float z = z_gate[p.z_offset + idx];
        const float silu_z = z / (1.0f + exp(-z));
        output[p.output_offset + idx] = delta_out[row] * rms * norm_weight[weight_idx] * silu_z;
    }
}
