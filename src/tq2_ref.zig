//! CPU reference for the Q2_0 (prism-ml Bonsai ternary) GEMV — spec and
//! correctness oracle for `shaders/dmmv_q2_0.comp` / `dmmv_q2_0_wave32.comp`.
//! Per 128-weight block: 2 bytes f16 scale `d` (first), then 32 bytes of
//! 2-bit codes (4/byte, LSB-first). Code map: value(code) = code - 1, i.e.
//! 00->-1, 01->0, 10->+1, 11->+2 -- verified bit-exact (cos=1.000000 vs F16)
//! against the real Ternary-Bonsai-1.7B-Q2_0.gguf via q2_0_decode.py; code 3
//! does not occur in real Bonsai-1.7B weight data but is not reserved, it's
//! a genuine 4th quantization level.
const std = @import("std");

pub const block_weights: u32 = 128;
pub const block_bytes: u32 = 34;

pub fn ternaryValue(code: u2) f32 {
    return @as(f32, @floatFromInt(code)) - 1.0;
}

/// Dot product of one packed Q2_0 row (`(k/128)*34` bytes) against a
/// `k`-length f32 activation vector, accumulated in f32.
pub fn rowDot(row: []const u8, act: []const f32, k: u32) f32 {
    std.debug.assert(k % block_weights == 0);
    std.debug.assert(row.len == @as(usize, k / block_weights) * block_bytes);
    std.debug.assert(act.len == k);

    var sum: f32 = 0.0;
    const num_blocks = k / block_weights;
    var block: u32 = 0;
    while (block < num_blocks) : (block += 1) {
        const blk = row[block * block_bytes ..][0..block_bytes];
        const d: f32 = @as(f16, @bitCast(std.mem.readInt(u16, blk[0..2], .little)));

        var partial: f32 = 0.0;
        var byte_idx: u32 = 0;
        while (byte_idx < 32) : (byte_idx += 1) {
            const qbyte = blk[2 + byte_idx];
            var j: u32 = 0;
            while (j < 4) : (j += 1) {
                const code: u2 = @intCast((qbyte >> @intCast(j * 2)) & 3);
                const w = block * block_weights + byte_idx * 4 + j;
                partial += ternaryValue(code) * act[w];
            }
        }
        sum += d * partial;
    }
    return sum;
}

/// Full GEMV: `weights` holds `m` packed rows of `(k/128)*34` bytes each.
pub fn gemv(weights: []const u8, act: []const f32, out: []f32, m: u32, k: u32) void {
    const row_bytes: usize = @as(usize, k / block_weights) * block_bytes;
    var row: u32 = 0;
    while (row < m) : (row += 1) {
        out[row] = rowDot(weights[row * row_bytes ..][0..row_bytes], act, k);
    }
}

fn packCode(dst: []u8, byte_idx: u32, code0: u2, code1: u2, code2: u2, code3: u2) void {
    dst[byte_idx] = @as(u8, code0) | (@as(u8, code1) << 2) | (@as(u8, code2) << 4) | (@as(u8, code3) << 6);
}

test "ternaryValue covers all four 2-bit codes (linear code-1 mapping)" {
    try std.testing.expectEqual(@as(f32, -1.0), ternaryValue(0));
    try std.testing.expectEqual(@as(f32, 0.0), ternaryValue(1));
    try std.testing.expectEqual(@as(f32, 1.0), ternaryValue(2));
    try std.testing.expectEqual(@as(f32, 2.0), ternaryValue(3));
}

test "rowDot: single block, all +1 codes, unit scale, unit activations" {
    var row: [block_bytes]u8 = undefined;
    const scale: f16 = 1.0;
    std.mem.writeInt(u16, row[0..2], @bitCast(scale), .little);
    var byte_idx: u32 = 0;
    while (byte_idx < 32) : (byte_idx += 1) packCode(row[2..], byte_idx, 2, 2, 2, 2);

    var act: [128]f32 = undefined;
    @memset(&act, 1.0);

    const result = rowDot(&row, &act, 128);
    try std.testing.expectApproxEqAbs(@as(f32, 128.0), result, 1e-6);
}

test "rowDot: code 3 contributes +2 (not reserved/zero)" {
    var row: [block_bytes]u8 = undefined;
    const scale: f16 = 1.0;
    std.mem.writeInt(u16, row[0..2], @bitCast(scale), .little);
    var byte_idx: u32 = 0;
    while (byte_idx < 32) : (byte_idx += 1) packCode(row[2..], byte_idx, 3, 3, 3, 3);

    var act: [128]f32 = undefined;
    @memset(&act, 5.0);

    const result = rowDot(&row, &act, 128);
    try std.testing.expectApproxEqAbs(@as(f32, 128.0 * 2.0 * 5.0), result, 1e-3);
}

test "rowDot: mixed codes and non-unit scale" {
    var row: [block_bytes]u8 = undefined;
    const scale: f16 = 0.5;
    std.mem.writeInt(u16, row[0..2], @bitCast(scale), .little);
    // First byte: codes -1, 0, +1, +2 -> weights 0..3.
    packCode(row[2..], 0, 0, 1, 2, 3);
    var byte_idx: u32 = 1;
    while (byte_idx < 32) : (byte_idx += 1) packCode(row[2..], byte_idx, 1, 1, 1, 1); // all zero-contribution

    var act: [128]f32 = undefined;
    @memset(&act, 2.0);

    // partial = (-1)*2 + 0*2 + 1*2 + 2*2 = 4; scaled by 0.5 -> 2.0.
    const result = rowDot(&row, &act, 128);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), result, 1e-6);
}

test "gemv matches rowDot per row" {
    const m: u32 = 4;
    const k: u32 = 128;
    const row_bytes = block_bytes;
    var weights: [4 * row_bytes]u8 = undefined;
    for (0..m) |r| {
        const scale: f16 = @floatFromInt(r + 1);
        const row = weights[r * row_bytes ..][0..row_bytes];
        std.mem.writeInt(u16, row[0..2], @bitCast(scale), .little);
        var byte_idx: u32 = 0;
        while (byte_idx < 32) : (byte_idx += 1) packCode(row[2..], byte_idx, 2, 0, 1, 3);
    }

    var act: [128]f32 = undefined;
    for (&act, 0..) |*v, i| v.* = @floatFromInt((i % 3) + 1);

    var out: [4]f32 = undefined;
    gemv(&weights, &act, &out, m, k);

    for (0..m) |r| {
        const expected = rowDot(weights[r * row_bytes ..][0..row_bytes], &act, k);
        try std.testing.expectEqual(expected, out[r]);
    }
}
