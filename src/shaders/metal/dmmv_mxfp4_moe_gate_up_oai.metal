#include <metal_stdlib>
using namespace metal;

struct MoeGateUpOaiPush {
    uint M;
    uint K;
    uint gate_offset;
    uint up_offset;
    uint gate_expert_stride;
    uint up_expert_stride;
    uint x_expert_stride;
    uint x_offset;
    uint y_offset;
    uint gate_bias_offset;
    uint up_bias_offset;
};

constant float kvalues_mxfp4[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

static inline float e8m0_to_fp32(uchar x) {
    uint bits;
    if (x == 0u) {
        bits = 0x00400000u;
    } else {
        bits = uint(x) << 23;
    }
    return as_type<float>(bits);
}

static inline float oai_swiglu(float gate, float up) {
    const float alpha = 1.702f;
    const float limit = 7.0f;
    const float x = min(gate, limit);
    const float y = clamp(up, -limit, limit);
    return (x / (1.0f + exp(alpha * (-x)))) * (y + 1.0f);
}

static inline float dot_mxfp4_row(
    device const uchar* row,
    threadgroup const float* x_cache,
    uint blocks_per_row
) {
    float sum = 0.0f;
    for (uint b = 0; b < blocks_per_row; b++) {
        device const uchar* block = row + ulong(b) * 17ul;
        const float d = e8m0_to_fp32(block[0]);
        device const uchar* qs = block + 1;
        const uint base = b * 32u;

        for (uint j = 0; j < 16u; j++) {
            const uchar q = qs[j];
            const float v_lo = d * kvalues_mxfp4[q & 0x0Fu];
            const float v_hi = d * kvalues_mxfp4[q >> 4];
            sum += v_lo * x_cache[base + j] + v_hi * x_cache[base + 16u + j];
        }
    }
    return sum;
}

kernel void main0(
    constant MoeGateUpOaiPush& p [[buffer(0)]],
    device const uchar* W_gate [[buffer(1)]],
    device const uchar* W_up [[buffer(2)]],
    device const float* X [[buffer(3)]],
    device float* Y [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]],
    device const float* gate_bias [[buffer(6)]],
    device const float* up_bias [[buffer(7)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint2 local_pos [[thread_position_in_threadgroup]]
) {
    threadgroup float x_cache[4096];

    const uint local_id = local_pos.x;
    const uint expert_slot = tg_pos.y;
    const uint expert_id = expert_ids[expert_slot];
    device const float* input = X + (p.x_offset / 4) + expert_slot * p.x_expert_stride;

    for (uint i = local_id; i < p.K; i += 64u) {
        x_cache[i] = input[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint row = tg_pos.x * 64u + local_id;
    if (row >= p.M) return;

    const uint blocks_per_row = p.K / 32u;
    const ulong row_bytes = ulong(blocks_per_row) * 17ul;
    const ulong gate_base = ulong(p.gate_offset) + ulong(expert_id) * ulong(p.gate_expert_stride);
    const ulong up_base = ulong(p.up_offset) + ulong(expert_id) * ulong(p.up_expert_stride);

    device const uchar* gate_row = W_gate + gate_base + ulong(row) * row_bytes;
    device const uchar* up_row = W_up + up_base + ulong(row) * row_bytes;

    const uint gate_bias_base = p.gate_bias_offset / 4u;
    const uint up_bias_base = p.up_bias_offset / 4u;
    const uint bias_idx = expert_id * p.M + row;

    const float gate = dot_mxfp4_row(gate_row, x_cache, blocks_per_row) + gate_bias[gate_bias_base + bias_idx];
    const float up = dot_mxfp4_row(up_row, x_cache, blocks_per_row) + up_bias[up_bias_base + bias_idx];

    Y[(p.y_offset / 4) + expert_slot * p.M + row] = oai_swiglu(gate, up);
}
