//! T-CPU element-wise vector addition implementation.
//! Computes: output[i] = a[i] + b[i]
//! @section Inference Runtime
const std = @import("std");

pub const Params = struct {
    a: []const f32,
    b: []const f32,
    output: []f32,
};

pub fn run(params: Params) !void {
    if (params.a.len == 0) return error.EmptyInput;
    if (params.output.len < params.a.len) return error.ShapeMismatch;
    if (params.b.len < params.a.len) return error.ShapeMismatch;
    for (0..params.a.len) |i| {
        params.output[i] = params.a[i] + params.b[i];
    }
}

test "vadd computes element-wise sum" {
    const a = [_]f32{ 1.0, -2.0, 3.0 };
    const b = [_]f32{ 4.0, 5.0, -1.0 };
    var output = [_]f32{ 0.0, 0.0, 0.0 };

    try run(.{
        .a = &a,
        .b = &b,
        .output = &output,
    });

    try std.testing.expectEqual(@as(f32, 5.0), output[0]);
    try std.testing.expectEqual(@as(f32, 3.0), output[1]);
    try std.testing.expectEqual(@as(f32, 2.0), output[2]);
}
