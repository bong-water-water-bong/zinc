//! Storage accounting for the opt-in ROCm Qwen KV cache.
//!
//! Computes checked allocation sizes for F32 rows and independently scaled
//! 32-value Q8 blocks. Cache storage is independent of GGUF weight quantization.
//! @section Inference Runtime
const std = @import("std");

pub const Format = enum {
    f32,
    q8,

    /// Parse an explicit cache format, defaulting to F32 when unset.
    pub fn parse(raw: ?[]const u8) !Format {
        const value = raw orelse return .f32;
        return std.meta.stringToEnum(Format, value) orelse error.InvalidKvCacheFormat;
    }

    /// Checked bytes per token's K or V row, including Q8 scales.
    pub fn rowBytes(self: Format, dimensions: u32) !usize {
        if (dimensions == 0) return error.InvalidKvDimensions;
        return switch (self) {
            .f32 => std.math.mul(usize, dimensions, @sizeOf(f32)),
            // One f32 scale and 32 signed bytes per independently scaled block.
            .q8 => if (dimensions % 32 == 0) std.math.mul(usize, dimensions / 32, 36) else error.InvalidKvDimensions,
        };
    }

    /// Checked size or row offset; zero tokens produces offset zero.
    pub fn bytes(self: Format, tokens: usize, dimensions: u32) !usize {
        return std.math.mul(usize, tokens, try self.rowBytes(dimensions));
    }
};

test "KV accounting includes scales at full native context" {
    try std.testing.expectEqual(@as(usize, 4 * 1024 * 1024 * 1024), 32 * try Format.f32.bytes(32768, 1024));
    try std.testing.expectEqual(@as(usize, 32 * 1024 * 1024 * 1024), 32 * try Format.f32.bytes(262144, 1024));
    try std.testing.expectEqual(@as(usize, 9 * 1024 * 1024 * 1024), 32 * try Format.q8.bytes(262144, 1024));
}

test "KV format rejects unsupported names, misalignment and overflow" {
    try std.testing.expectEqual(Format.f32, try Format.parse(null));
    try std.testing.expectEqual(Format.q8, try Format.parse("q8"));
    try std.testing.expectError(error.InvalidKvCacheFormat, Format.parse("q4"));
    try std.testing.expectError(error.InvalidKvDimensions, Format.q8.rowBytes(513));
    try std.testing.expectError(error.InvalidKvDimensions, Format.f32.rowBytes(0));
    try std.testing.expectError(error.Overflow, Format.q8.bytes(std.math.maxInt(usize), 1024));
}
