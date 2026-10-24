const std = @import("std");
const host_io = @import("host_io");

fn dumpStruct(comptime T: type, name: []const u8, writer: anytype) !void {
    const info = @typeInfo(T);
    if (info != .@"struct") return;

    try writer.print("### {s}\n\n", .{name});
    try writer.print("- **Total Size**: {d} bytes\n", .{@sizeOf(T)});
    try writer.print("- **Alignment**: {d} bytes\n\n", .{@alignOf(T)});

    try writer.print("| Offset | Size | Align | Field | Type |\n", .{});
    try writer.print("|--------|------|-------|-------|------|\n", .{});

    inline for (info.@"struct".field_names, info.@"struct".field_types, info.@"struct".field_attrs) |field_name, FieldType, attrs| {
        if (attrs.@"comptime") continue;
        try writer.print("| {d} | {d} | {d} | `{s}` | `{s}` |\n", .{
            @offsetOf(T, field_name),
            @sizeOf(FieldType),
            attrs.@"align" orelse @alignOf(FieldType),
            field_name,
            @typeName(FieldType),
        });
    }
    try writer.print("\n", .{});
}

const main_mod = @import("main.zig");
const modules = .{
    .{ "loader", @import("model/loader.zig") },
    .{ "forward", @import("compute/forward.zig") },
};

pub fn main(process_init: std.process.Init) !void {
    host_io.init(process_init);
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();

    var stdout_buffer: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(host_io.io, &stdout_buffer);

    try stdout.interface.print("# ZINC Struct Memory Layouts\n\n", .{});

    inline for (modules) |mod| {
        try stdout.interface.print("## Module: {s}\n\n", .{mod[0]});

        const mod_info = @typeInfo(mod[1]);
        if (mod_info == .@"struct") {
            inline for (mod_info.@"struct".decl_names) |decl_name| {
                const decl_val = @field(mod[1], decl_name);
                const T = @TypeOf(decl_val);
                if (T == type) {
                    const t_info = @typeInfo(decl_val);
                    if (t_info == .@"struct") {
                        try dumpStruct(decl_val, decl_name, &stdout.interface);
                    }
                }
            }
        }
    }

    try stdout.interface.flush();
}
