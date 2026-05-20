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

static inline float load_f32_or_f16(device const char* raw, uint idx, bool is_f16) {
    if (is_f16) {
        device const half* ptr = (device const half*)raw;
        return float(ptr[idx]);
    }
    device const float* ptr = (device const float*)raw;
    return ptr[idx];
}

kernel void main0(
    constant Params& p [[buffer(0)]],
    device const float* conv_out [[buffer(1)]],
    device const float* alpha [[buffer(2)]],
    device const char* dt_bias [[buffer(3)]],
    device const char* ssm_a [[buffer(4)]],
    device const float* beta [[buffer(5)]],
    device float* state [[buffer(6)]],
    device const float* z_gate [[buffer(7)]],
    device const float* norm_weight [[buffer(8)]],
    device float* output [[buffer(9)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint simd_width [[thread_execution_width]]
) {
    if (head >= p.dt_rank || p.head_v_dim > 128u || p.d_state > 128u) {
        return;
    }

    threadgroup float q[128];
    threadgroup float k[128];
    threadgroup float delta_out[128];
    threadgroup float partial[4];

    const uint qk_dim = p.d_state * p.n_group;
    const uint group = (p.n_group == p.dt_rank) ? head : (head % p.n_group);
    const uint k_len = min(p.d_state, p.head_v_dim);
    const uint head_state_base = head * p.head_v_dim * p.head_v_dim;
    const uint q_base = group * p.d_state;
    const uint k_base = qk_dim + group * p.d_state;
    const uint v_base = 2u * qk_dim + head * p.head_v_dim;

    float q_ss = 0.0f;
    float k_ss = 0.0f;
    for (uint i = tid; i < k_len; i += 64u) {
        const float qv = conv_out[q_base + i];
        const float kv = conv_out[k_base + i];
        q[i] = qv;
        k[i] = kv;
        q_ss = fma(qv, qv, q_ss);
        k_ss = fma(kv, kv, k_ss);
    }

    float q_sum = simd_sum(q_ss);
    if ((tid % simd_width) == 0u) {
        partial[tid / simd_width] = q_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float total = 0.0f;
        const uint n_sg = (64u + simd_width - 1u) / simd_width;
        for (uint i = 0u; i < n_sg; ++i) {
            total += partial[i];
        }
        partial[0] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float q_scale = rsqrt(fast::max(partial[0], 1.0e-13f)) / sqrt(float(p.d_state));
    for (uint i = tid; i < k_len; i += 64u) {
        q[i] *= q_scale;
    }

    float k_sum = simd_sum(k_ss);
    if ((tid % simd_width) == 0u) {
        partial[tid / simd_width] = k_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float total = 0.0f;
        const uint n_sg = (64u + simd_width - 1u) / simd_width;
        for (uint i = 0u; i < n_sg; ++i) {
            total += partial[i];
        }
        partial[0] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float k_scale = rsqrt(fast::max(partial[0], 1.0e-13f));
    for (uint i = tid; i < k_len; i += 64u) {
        k[i] *= k_scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float alpha_raw = alpha[p.alpha_offset + head] +
        ((p.has_dt_bias != 0u) ? load_f32_or_f16(dt_bias, head, p.dt_bias_is_f16 != 0u) : 0.0f);
    const float softplus_alpha = log(1.0f + exp(alpha_raw));
    const float decay_arg = (p.has_ssm_a != 0u) ?
        (softplus_alpha * load_f32_or_f16(ssm_a, head, p.ssm_a_is_f16 != 0u)) :
        (-softplus_alpha);
    const float decay = exp(decay_arg);
    const float beta_val = 1.0f / (1.0f + exp(-beta[p.beta_offset + head]));

    float local_sq = 0.0f;
    for (uint row = tid; row < p.head_v_dim; row += 64u) {
        const uint row_base = head_state_base + row * p.head_v_dim;
        for (uint col = 0u; col < p.head_v_dim; ++col) {
            state[row_base + col] *= decay;
        }

        float sk = 0.0f;
        for (uint col = 0u; col < k_len; ++col) {
            sk = fma(state[row_base + col], k[col], sk);
        }

        const float v = conv_out[v_base + row];
        const float delta = beta_val * (v - sk);
        for (uint col = 0u; col < k_len; ++col) {
            state[row_base + col] = fma(k[col], delta, state[row_base + col]);
        }

        float out_v = 0.0f;
        for (uint col = 0u; col < k_len; ++col) {
            out_v = fma(state[row_base + col], q[col], out_v);
        }
        delta_out[row] = out_v;
        local_sq = fma(out_v, out_v, local_sq);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sq_sum = simd_sum(local_sq);
    if ((tid % simd_width) == 0u) {
        partial[tid / simd_width] = sq_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float total = 0.0f;
        const uint n_sg = (64u + simd_width - 1u) / simd_width;
        for (uint i = 0u; i < n_sg; ++i) {
            total += partial[i];
        }
        partial[0] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float rms = rsqrt((partial[0] / float(p.head_v_dim)) + 1.0e-6f);

    for (uint row = tid; row < p.head_v_dim; row += 64u) {
        const uint head_base = head * p.head_v_dim;
        const uint idx = head_base + row;
        const uint weight_idx = (p.norm_per_head != 0u) ? idx : (row % p.d_state);
        const float z = z_gate[p.z_offset + idx];
        const float silu_z = z / (1.0f + exp(-z));
        output[p.output_offset + idx] = delta_out[row] * rms * norm_weight[weight_idx] * silu_z;
    }
}
