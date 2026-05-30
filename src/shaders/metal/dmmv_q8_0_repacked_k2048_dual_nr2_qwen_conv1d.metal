#include <metal_stdlib>
using namespace metal;

struct DualQ8DmmvConv1dPush {
    uint M0;
    uint M1;
    uint K;
    uint a0_offset;
    uint a1_offset;
    uint x_offset;
    uint y0_offset;
    uint y1_offset;
    uint conv_channels;   // for SSM qkv path: 8192 (== M0)
    uint d_conv;          // hard-gated to 4 by the dispatch site
};

// Sibling of dmmv_q8_0_repacked_k2048_dual_nr2_qwen.metal that fuses the
// per-channel SSM conv1d postlude into the qkv slice of the dual matvec.
// Used by the `prev_fused_attn_norm` SSM path (most layers after layer 0):
// the MoE finalizer of the previous layer already materialized norm_buf
// with this layer's attn_norm weights, so this kernel reads norm_buf
// directly and skips the inline RMSNorm — but the qkv → conv1d join
// pattern is identical to dmmv_q8_0_dual_fused_norm_conv1d.metal.
//
// Saves one standalone `ssm_conv1d_qwen_d4` dispatch + one `.qkv` barrier
// per SSM layer per decode token (≈29/decode token after layer 0 on
// Qwen3.6-35B). The conv1d postlude is run by the same lane (lane 0) that
// produces the conv1d input (sum0, sum1), and each simdgroup owns disjoint
// channel indices, so state[] writes do not race.
//
// Layout invariants (held by the dispatch site):
//   - M0 == conv_channels (the qkv slice goes through conv1d; the z slice
//     in M1 does not).
//   - M0 % 2 == 0 so the qkv→z boundary never splits a simdgroup (`first`
//     is uniform within the simdgroup; each simdgroup writes 2 contiguous
//     rows).
//   - d_conv == 4 (matches `ssm_conv1d_qwen_d4.metal`'s float4 layout).
//   - input_offset implicitly 0 (decode path; the prefill path uses a
//     different conv input buffer and is not eligible for this fusion).

kernel void main0(
    constant DualQ8DmmvConv1dPush& p [[buffer(0)]],
    device const uchar* W0 [[buffer(1)]],
    device const uchar* W1 [[buffer(2)]],
    device const float* X [[buffer(3)]],
    device float* Y0 [[buffer(4)]],
    device float* Y1 [[buffer(5)]],
    device const float* conv_kernel [[buffer(6)]],
    device float* conv_state [[buffer(7)]],
    device float* conv_out [[buffer(8)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroups_per_tg [[simdgroups_per_threadgroup]]
) {
    const uint base_pair = tg_id * simdgroups_per_tg + sg_idx;
    const uint base_row = base_pair * 2u;
    const uint total_rows = p.M0 + p.M1;
    if (base_row >= total_rows) {
        return;
    }

    device const float* input = X + (p.x_offset >> 2);

    constexpr ulong group_bytes = 1088ul;
    constexpr ulong row_bytes = 2176ul;

    const bool first = base_row < p.M0;
    const uint row = first ? base_row : (base_row - p.M0);
    device const uchar* weights = first ? W0 : W1;
    device float* output = first ? (Y0 + (p.y0_offset >> 2)) : (Y1 + (p.y1_offset >> 2));
    const uint a_offset = first ? p.a0_offset : p.a1_offset;

    device const uchar* row0 = weights + a_offset + ulong(row) * row_bytes;
    device const uchar* row1 = row0 + row_bytes;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    #pragma unroll
    for (uint gi = 0u; gi < 2u; ++gi) {
        device const uchar* g0 = row0 + ulong(gi) * group_bytes;
        device const uchar* g1 = row1 + ulong(gi) * group_bytes;

        const float s0 = float(as_type<half>(*(device const ushort*)(g0 + lane * 2u)));
        const float s1 = float(as_type<half>(*(device const ushort*)(g1 + lane * 2u)));
        const uint x_base = (gi * 32u + lane) << 5;

        #pragma unroll
        for (uint vi = 0u; vi < 8u; ++vi) {
            const uint qo = 64u + vi * 128u + lane * 4u;
            const char4 q0 = as_type<char4>(*(device const int*)(g0 + qo));
            const char4 q1 = as_type<char4>(*(device const int*)(g1 + qo));
            const float4 x = *(device const float4*)(input + x_base + (vi << 2));

            acc0 = fma(s0, dot(float4(q0), x), acc0);
            acc1 = fma(s1, dot(float4(q1), x), acc1);
        }
    }

    const float sum0 = simd_sum(acc0);
    const float sum1 = simd_sum(acc1);
    if (lane == 0u) {
        output[row] = sum0;
        output[row + 1u] = sum1;

        // Conv1d postlude: identical math to ssm_conv1d_qwen_d4.metal, run
        // by the same lane that just produced the conv1d input for this row.
        // Each conv channel is owned by exactly one simdgroup (row/row+1 are
        // unique across the dispatch), so state[] writes do not race.
        if (first) {
            const uint c = p.conv_channels;
            const uint r0 = row;
            const uint r1 = row + 1u;
            const float a0 = conv_state[r0];
            const float a1 = conv_state[c + r0];
            const float a2 = conv_state[2u * c + r0];
            const float4 w0 = *(device const float4*)(conv_kernel + r0 * 4u);
            const float ss0 = dot(w0, float4(a0, a1, a2, sum0));
            conv_out[r0] = ss0 * fast::divide(1.0f, 1.0f + fast::exp(-ss0));
            conv_state[r0] = a1;
            conv_state[c + r0] = a2;
            conv_state[2u * c + r0] = sum0;

            const float b0 = conv_state[r1];
            const float b1 = conv_state[c + r1];
            const float b2 = conv_state[2u * c + r1];
            const float4 w1 = *(device const float4*)(conv_kernel + r1 * 4u);
            const float ss1 = dot(w1, float4(b0, b1, b2, sum1));
            conv_out[r1] = ss1 * fast::divide(1.0f, 1.0f + fast::exp(-ss1));
            conv_state[r1] = b1;
            conv_state[c + r1] = b2;
            conv_state[2u * c + r1] = sum1;
        }
    }
}
