//! Dynamic packet list for building per-token decode sequences.
//! T-CPU forward passes use this to accumulate packets before submitting
//! them to the CPU ring for execution.
//! @section Inference Runtime
const std = @import("std");
const ring = @import("mod.zig");

pub const PacketList = struct {
    allocator: std.mem.Allocator,
    items: std.ArrayList(ring.Packet),

    pub fn init(allocator: std.mem.Allocator) PacketList {
        return .{
            .allocator = allocator,
            .items = std.ArrayList(ring.Packet).init(allocator),
        };
    }

    pub fn deinit(self: *PacketList) void {
        self.items.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn append(self: *PacketList, packet: ring.Packet) !void {
        try self.items.append(self.allocator, packet);
    }

    pub fn appendBarrier(self: *PacketList) !void {
        try self.items.append(self.allocator, .barrier);
    }

    pub fn slice(self: *const PacketList) []const ring.Packet {
        return self.items.items;
    }

    pub fn len(self: *const PacketList) usize {
        return self.items.items.len;
    }

    pub fn clear(self: *PacketList) void {
        self.items.clearRetainingCapacity();
    }
};
