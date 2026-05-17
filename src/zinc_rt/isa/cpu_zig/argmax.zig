//! T-CPU ARGMAX implementation.
//! The CPU oracle selects token IDs from logits with deterministic tie
//! behavior that GPU tiers can match during validation.
//! @section Sampling
const std = @import("std");

pub const Params = struct {
    logits: []const f32,
    output: *u32,
};

pub fn run(params: Params) !void {
    if (params.logits.len == 0) return error.EmptyInput;

    var best_index: u32 = 0;
    var best_value = params.logits[0];
    for (params.logits[1..], 1..) |value, index| {
        if (value > best_value) {
            best_value = value;
            best_index = @intCast(index);
        }
    }
    params.output.* = best_index;
}

test "argmax returns index of largest logit" {
    const logits = [_]f32{ -1.0, 3.5, 2.0 };
    var output: u32 = 0;
    try run(.{ .logits = &logits, .output = &output });
    try std.testing.expectEqual(@as(u32, 1), output);
}
