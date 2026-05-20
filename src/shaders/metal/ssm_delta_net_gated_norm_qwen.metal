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
    uint tid [[thread_position_in_threadgroup]]
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

    const uint simd_lane = tid & 31u;
    const uint simd_idx = tid >> 5u;
    const uint group = head & 15u;
    const uint q_base = group * d_state;
    const uint k_base = qk_dim + group * d_state;
    const uint v_base = v_base0 + head * head_v_dim;
    const uint head_state_base = head * head_v_dim * head_v_dim;

    const float qv = conv_out[q_base + tid];
    const float kv = conv_out[k_base + tid];
    q[tid] = qv;
    k[tid] = kv;

    const float q_sum = simd_sum(qv * qv);
    const float k_sum = simd_sum(kv * kv);
    if (simd_lane == 0u) {
        partial_q[simd_idx] = q_sum;
        partial_k[simd_idx] = k_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        partial_q[0] = partial_q[0] + partial_q[1] + partial_q[2] + partial_q[3];
        partial_k[0] = partial_k[0] + partial_k[1] + partial_k[2] + partial_k[3];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float q_scale = rsqrt(fast::max(partial_q[0], 1.0e-13f)) * inv_sqrt_d_state;
    const float k_scale = rsqrt(fast::max(partial_k[0], 1.0e-13f));
    const float alpha_raw = alpha[p.alpha_offset + head] + dt_bias[head];
    const float softplus_alpha = log(1.0f + exp(alpha_raw));
    const float decay = exp(softplus_alpha * ssm_a[head]);
    const float beta_val = 1.0f / (1.0f + exp(-beta[p.beta_offset + head]));

    const uint row_base = head_state_base + tid * head_v_dim;
    float sk_raw = 0.0f;
    for (uint col = 0u; col < head_v_dim; ++col) {
        const uint state_idx = row_base + col;
        const float decayed = state[state_idx] * decay;
        state[state_idx] = decayed;
        sk_raw = fma(decayed, k[col], sk_raw);
    }

    const float v = conv_out[v_base + tid];
    const float delta = beta_val * (v - sk_raw * k_scale);
    const float scaled_delta = delta * k_scale;
    float out_v = 0.0f;
    for (uint col = 0u; col < head_v_dim; ++col) {
        const float updated = fma(k[col], scaled_delta, state[row_base + col]);
        state[row_base + col] = updated;
        out_v = fma(updated, q[col], out_v);
    }
    out_v *= q_scale;
    delta_out[tid] = out_v;

    const float sq_sum = simd_sum(out_v * out_v);
    if (simd_lane == 0u) {
        partial_sq[simd_idx] = sq_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        partial_sq[0] = partial_sq[0] + partial_sq[1] + partial_sq[2] + partial_sq[3];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float rms = rsqrt((partial_sq[0] / float(head_v_dim)) + 1.0e-6f);
    const uint idx = head * head_v_dim + tid;
    const uint weight_idx = (p.norm_per_head != 0u) ? idx : tid;
    const float z = z_gate[p.z_offset + idx];
    const float silu_z = z / (1.0f + exp(-z));
    output[p.output_offset + idx] = delta_out[tid] * rms * norm_weight[weight_idx] * silu_z;
}
