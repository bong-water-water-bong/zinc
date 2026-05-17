//! ZINC_RT — the ZINC Runtime.
//! Owns tier selection and the top-level runtime handle used by future IR
//! emitters and ring backends.
//! @section Inference Runtime
// SPDX-FileCopyrightText: ZINC Authors
const std = @import("std");
const umq = @import("ring/umq.zig");
const kfd = @import("ring/kfd.zig");

pub const Tier = enum {
    t1_pm4,
    t2_umq,
    t_cpu,
    t_metal,
    t_intel,
    t_cuda,
};

pub const Options = struct {
    tier: Tier = .t_cpu,
};

pub const Engine = struct {
    allocator: std.mem.Allocator,
    tier: Tier,

    pub fn init(allocator: std.mem.Allocator, options: Options) !Engine {
        return .{
            .allocator = allocator,
            .tier = options.tier,
        };
    }

    pub fn deinit(self: *Engine) void {
        self.* = undefined;
    }
};

pub fn parseTier(value: []const u8) !Tier {
    if (std.mem.eql(u8, value, "auto")) return autoTier();
    if (std.mem.eql(u8, value, "t1") or std.mem.eql(u8, value, "t1_pm4")) return .t1_pm4;
    if (std.mem.eql(u8, value, "t2") or std.mem.eql(u8, value, "t2_umq")) return .t2_umq;
    if (std.mem.eql(u8, value, "t_cpu")) return .t_cpu;
    if (std.mem.eql(u8, value, "t_metal")) return .t_metal;
    if (std.mem.eql(u8, value, "t_intel")) return .t_intel;
    if (std.mem.eql(u8, value, "t_cuda")) return .t_cuda;
    return error.UnknownZincRtTier;
}

pub fn tierFromEnv() !Tier {
    const value = std.posix.getenv("ZINC_RT_TIER") orelse return autoTier();
    return parseTier(value);
}

pub fn autoTier() Tier {
    // T2 UMQ (DRM_IOCTL_AMDGPU_USERQ) is the blessed direct path, but the
    // amdgpu firmware on the bench node rejects compute user queues, so we
    // fall through to the T1 PM4 path on `/dev/kfd` (the ROCm/tinygrad ABI)
    // whenever it is reachable, and only then to the T-CPU reference.
    if (umq.admissionProbeDefault()) return .t2_umq;
    if (kfd.reachable()) return .t1_pm4;
    return .t_cpu;
}

test "parseTier maps explicit CPU tier" {
    try std.testing.expectEqual(Tier.t_cpu, try parseTier("t_cpu"));
}

test "auto tier remains CPU on non-Linux hosts" {
    if (@import("builtin").os.tag == .linux) return error.SkipZigTest;
    try std.testing.expectEqual(Tier.t_cpu, try parseTier("auto"));
}
