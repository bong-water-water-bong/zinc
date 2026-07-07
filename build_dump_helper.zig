const std = @import("std");

pub fn build(b: *std.Build) void {
    const exe = b.addExecutable(.{
        .name = "gguf_dump",
        .root_source_file = b.path("src/tools/gguf_dump.zig"),
        .target = b.standardTargetOptions(.{}),
        .optimize = .Debug,
    });
    b.installArtifact(exe);
}
