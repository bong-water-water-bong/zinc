//! T-CPU EMBED implementation.
//! Reads one token row from a GGUF tensor into f32 hidden state.
//! @section Inference Runtime
const dequant = @import("dequant.zig");
const gguf = @import("gguf");

pub const Params = struct {
    raw_data: []const u8,
    tensor_type: gguf.GGMLType,
    token_id: u32,
    hidden_dim: u32,
    vocab_size: u32,
    output: []f32,
};

pub fn run(params: Params) !void {
    if (params.token_id >= params.vocab_size) return error.TokenOutOfRange;
    if (params.output.len != params.hidden_dim) return error.ShapeMismatch;
    try dequant.row(params.raw_data, params.token_id, params.hidden_dim, params.tensor_type, params.output);
}

test "embed reads f32 token row" {
    const raw = [_]f32{
        1.0, 2.0,
        3.0, 4.0,
    };
    var output = [_]f32{ 0.0, 0.0 };
    try run(.{
        .raw_data = @import("std").mem.sliceAsBytes(&raw),
        .tensor_type = .f32,
        .token_id = 1,
        .hidden_dim = 2,
        .vocab_size = 2,
        .output = &output,
    });
    try @import("std").testing.expectEqual(@as(f32, 3.0), output[0]);
    try @import("std").testing.expectEqual(@as(f32, 4.0), output[1]);
}
