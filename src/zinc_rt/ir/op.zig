//! ZINC_RT IR opcode definitions.
//! This table mirrors Appendix B of the design and gives each opcode stable
//! metadata for verification, lowering, and T-CPU dispatch.
//! @section Decode Planning
const std = @import("std");

pub const Stage = enum {
    decode,
    prefill,
    both,
};

pub const Milestone = enum {
    m0,
    m2,
    m3,
    m4,
    m6,
};

pub const Opcode = enum(u8) {
    embed,
    rms_norm,
    rms_norm_fused_qkv,
    rms_norm_fused_mlp_gate_up,
    rope,
    rms_norm_fused_rope_kv_write,
    flash_attn,
    flash_attn_batched,
    moe_gate_topk,
    moe_gate_up,
    moe_swiglu,
    moe_down_acc,
    shared_expert,
    ssm_conv1d,
    ssm_delta_net,
    ssm_gated_norm,
    residual_rms_norm,
    lm_head,
    argmax,
    sample,
    kv_write_batched,
    matmul_wmma_f16,
    matmul_wmma_q4k,
    barrier,
    stream_out,
    load_request_state,
    store_request_state,
    verify_k,
};

pub const Info = struct {
    name: []const u8,
    stage: Stage,
    milestone: Milestone,
};

pub fn info(opcode: Opcode) Info {
    return switch (opcode) {
        .embed => .{ .name = "EMBED", .stage = .decode, .milestone = .m0 },
        .rms_norm => .{ .name = "RMS_NORM", .stage = .both, .milestone = .m0 },
        .rms_norm_fused_qkv => .{ .name = "RMS_NORM_FUSED_QKV", .stage = .decode, .milestone = .m2 },
        .rms_norm_fused_mlp_gate_up => .{ .name = "RMS_NORM_FUSED_MLP_GATE_UP", .stage = .decode, .milestone = .m2 },
        .rope => .{ .name = "ROPE", .stage = .both, .milestone = .m0 },
        .rms_norm_fused_rope_kv_write => .{ .name = "RMS_NORM_FUSED_ROPE_KV_WRITE", .stage = .decode, .milestone = .m2 },
        .flash_attn => .{ .name = "FLASH_ATTN", .stage = .decode, .milestone = .m0 },
        .flash_attn_batched => .{ .name = "FLASH_ATTN_BATCHED", .stage = .prefill, .milestone = .m2 },
        .moe_gate_topk => .{ .name = "MOE_GATE_TOPK", .stage = .decode, .milestone = .m0 },
        .moe_gate_up => .{ .name = "MOE_GATE_UP", .stage = .decode, .milestone = .m0 },
        .moe_swiglu => .{ .name = "MOE_SWIGLU", .stage = .decode, .milestone = .m0 },
        .moe_down_acc => .{ .name = "MOE_DOWN_ACC", .stage = .decode, .milestone = .m0 },
        .shared_expert => .{ .name = "SHARED_EXPERT", .stage = .decode, .milestone = .m0 },
        .ssm_conv1d => .{ .name = "SSM_CONV1D", .stage = .decode, .milestone = .m0 },
        .ssm_delta_net => .{ .name = "SSM_DELTA_NET", .stage = .decode, .milestone = .m0 },
        .ssm_gated_norm => .{ .name = "SSM_GATED_NORM", .stage = .decode, .milestone = .m0 },
        .residual_rms_norm => .{ .name = "RESIDUAL_RMS_NORM", .stage = .decode, .milestone = .m0 },
        .lm_head => .{ .name = "LM_HEAD", .stage = .decode, .milestone = .m0 },
        .argmax => .{ .name = "ARGMAX", .stage = .decode, .milestone = .m0 },
        .sample => .{ .name = "SAMPLE", .stage = .decode, .milestone = .m2 },
        .kv_write_batched => .{ .name = "KV_WRITE_BATCHED", .stage = .prefill, .milestone = .m2 },
        .matmul_wmma_f16 => .{ .name = "MATMUL_WMMA_F16", .stage = .prefill, .milestone = .m4 },
        .matmul_wmma_q4k => .{ .name = "MATMUL_WMMA_Q4K", .stage = .prefill, .milestone = .m4 },
        .barrier => .{ .name = "BARRIER", .stage = .both, .milestone = .m0 },
        .stream_out => .{ .name = "STREAM_OUT", .stage = .decode, .milestone = .m3 },
        .load_request_state => .{ .name = "LOAD_REQUEST_STATE", .stage = .decode, .milestone = .m3 },
        .store_request_state => .{ .name = "STORE_REQUEST_STATE", .stage = .decode, .milestone = .m3 },
        .verify_k => .{ .name = "VERIFY_K", .stage = .decode, .milestone = .m6 },
    };
}

pub fn name(opcode: Opcode) []const u8 {
    return info(opcode).name;
}

pub fn fromName(value: []const u8) ?Opcode {
    inline for (@typeInfo(Opcode).@"enum".fields) |field| {
        const opcode: Opcode = @enumFromInt(field.value);
        if (std.ascii.eqlIgnoreCase(value, field.name) or std.mem.eql(u8, value, name(opcode))) {
            return opcode;
        }
    }
    return null;
}

pub fn isM0(opcode: Opcode) bool {
    return info(opcode).milestone == .m0;
}

test "opcode table contains M0 argmax" {
    try std.testing.expectEqual(Opcode.argmax, fromName("ARGMAX").?);
    try std.testing.expect(isM0(.argmax));
}
