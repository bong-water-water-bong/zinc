//! T-CPU SwiGLU implementation.
//! This is the scalar reference activation used by MoE and dense MLP paths
//! before tier-specific kernels are trusted.
//! @section Inference Runtime
const std = @import("std");

pub const Params = struct {
    gate: []const f32,
    up: []const f32,
    output: []f32,
};

pub fn run(params: Params) !void {
    if (params.gate.len != params.up.len or params.output.len != params.gate.len) {
        return error.ShapeMismatch;
    }

    for (params.gate, params.up, params.output) |gate, up, *out| {
        const silu = gate / (1.0 + std.math.exp(-gate));
        out.* = silu * up;
    }
}

test "swiglu multiplies silu gate by up projection" {
    const gate = [_]f32{ 0.0, 2.0 };
    const up = [_]f32{ 3.0, 4.0 };
    var output = [_]f32{ 0.0, 0.0 };

    try run(.{ .gate = &gate, .up = &up, .output = &output });

    try std.testing.expectApproxEqAbs(@as(f32, 0.0), output[0], 0.00001);
    try std.testing.expectApproxEqAbs(@as(f32, 7.0463767), output[1], 0.00001);
}
