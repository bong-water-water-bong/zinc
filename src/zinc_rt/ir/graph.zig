//! Shape-static ZINC_RT IR graph builder.
//! Graphs contain logical buffers and opcode nodes before any tier lowers them
//! into packets, shaders, or pure Zig calls.
//! @section Decode Planning
const std = @import("std");
const op = @import("op.zig");

pub const BufferId = u32;
pub const NodeId = u32;
pub const max_bindings = 8;

pub const BindingList = struct {
    items: [max_bindings]BufferId = undefined,
    len: u8 = 0,

    pub fn init(values: []const BufferId) !BindingList {
        if (values.len > max_bindings) return error.TooManyBindings;
        var result = BindingList{};
        result.len = @intCast(values.len);
        for (values, 0..) |value, index| {
            result.items[index] = value;
        }
        return result;
    }

    pub fn slice(self: *const BindingList) []const BufferId {
        return self.items[0..self.len];
    }
};

pub const Node = struct {
    opcode: op.Opcode,
    inputs: BindingList,
    outputs: BindingList,
};

pub const Graph = struct {
    allocator: std.mem.Allocator,
    buffers: u32 = 0,
    nodes: std.ArrayList(Node) = .{},

    pub fn init(allocator: std.mem.Allocator) Graph {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *Graph) void {
        self.nodes.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn addBuffer(self: *Graph) BufferId {
        const id = self.buffers;
        self.buffers += 1;
        return id;
    }

    pub fn addNode(
        self: *Graph,
        opcode: op.Opcode,
        inputs: []const BufferId,
        outputs: []const BufferId,
    ) !NodeId {
        const id: NodeId = @intCast(self.nodes.items.len);
        try self.nodes.append(self.allocator, .{
            .opcode = opcode,
            .inputs = try BindingList.init(inputs),
            .outputs = try BindingList.init(outputs),
        });
        return id;
    }

    pub fn verify(self: *const Graph) !void {
        if (self.nodes.items.len == 0) return error.EmptyGraph;

        for (self.nodes.items) |node| {
            for (node.inputs.slice()) |buffer| {
                if (buffer >= self.buffers) return error.UnknownInputBuffer;
            }
            for (node.outputs.slice()) |buffer| {
                if (buffer >= self.buffers) return error.UnknownOutputBuffer;
            }
            if (node.outputs.len == 0 and node.opcode != .barrier and node.opcode != .stream_out) {
                return error.NodeWithoutOutput;
            }
        }
    }
};

test "graph rejects unknown buffers" {
    var graph = Graph.init(std.testing.allocator);
    defer graph.deinit();

    const input = graph.addBuffer();
    const output = graph.addBuffer();
    _ = try graph.addNode(.rms_norm, &.{ input, 99 }, &.{output});
    try std.testing.expectError(error.UnknownInputBuffer, graph.verify());
}
