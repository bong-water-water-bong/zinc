//! Host-side launch sizes shared by accelerator paths.
//!
//! Computes bounded shared-memory byte counts before kernel dispatch. Explicit
//! integer widths keep range inference from narrowing intermediate arithmetic.
//! @section Inference Runtime
const std = @import("std");

/// Must match the bounded gate/beta tile in ssm_delta_net_warp_impl.
/// Widen @min's range-narrowed result before multiplying: its inferred integer
/// type cannot hold the byte count, even when the destination is a u32.
pub fn ssmWarpSharedBytes(tokens: u32) u32 {
    const tile_tokens: u32 = @min(tokens, 4096);
    return tile_tokens * (2 * @sizeOf(f32));
}

test "SSM shared memory remains valid across tile and hardware boundaries" {
    const cases = [_][2]u32{
        .{ 0, 0 },         .{ 1, 8 },                        .{ 127, 1016 },   .{ 4095, 32760 },
        .{ 4096, 32768 },  .{ 4097, 32768 },                 .{ 8192, 32768 }, .{ 8193, 32768 },
        .{ 32768, 32768 }, .{ std.math.maxInt(u32), 32768 },
    };
    for (cases) |case| try std.testing.expectEqual(case[1], ssmWarpSharedBytes(case[0]));
}
