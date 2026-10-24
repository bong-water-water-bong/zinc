//! Runtime asset discovery for installed and source-tree ZINC layouts.
//! @section Shader Dispatch
//! Release archives place shader assets next to the executable under
//! `share/zinc/shaders`, while development builds often run from the repository
//! with assets under `zig-out` or `src/shaders`. This module centralizes that
//! lookup and keeps backend code from hardcoding source-checkout paths.
const std = @import("std");
const host_io = @import("host_io");

/// Which compiled shader family to locate.
pub const ShaderKind = enum {
    /// SPIR-V modules consumed by the Vulkan backend.
    spirv,
    /// Metallib/`.metal` sources consumed by the Apple Silicon backend.
    metal,
};

const spirv_cwd_candidates = [_][]const u8{
    "zig-out/share/zinc/shaders",
    "share/zinc/shaders",
};

const metal_cwd_candidates = [_][]const u8{
    "zig-out/share/zinc/shaders/metal",
    "share/zinc/shaders/metal",
    "src/shaders/metal",
};

/// Resolve the directory holding compiled shaders of `kind`, honoring `ZINC_SHADER_DIR`.
///
/// Checks the `ZINC_SHADER_DIR` environment override first, then falls back to the
/// executable-relative install layout and the in-tree development paths.
/// @param allocator Allocator for the returned path; the caller owns the result.
/// @param kind Shader family to locate (SPIR-V or Metal).
/// @returns Newly allocated path to the shader directory.
/// @note Returns `error.ShaderDirOverrideNotFound` when the override is set but absent,
///   or `error.ShaderDirNotFound` when no known layout exists.
pub fn resolveShaderDir(allocator: std.mem.Allocator, kind: ShaderKind) ![]u8 {
    if (host_io.getenv("ZINC_SHADER_DIR")) |override| {
        if (!dirExists(std.Io.Dir.cwd(), override)) return error.ShaderDirOverrideNotFound;
        return allocator.dupe(u8, override);
    }
    return resolveShaderDirFrom(allocator, std.Io.Dir.cwd(), null, kind);
}

/// Resolve the shader directory against explicit base/exe dirs — the testable core of `resolveShaderDir`.
///
/// Tries the executable-relative install layout first, then the working-directory
/// candidates for the requested shader family.
/// @param allocator Allocator for the returned path; the caller owns the result.
/// @param base_dir Directory the cwd-relative candidates are resolved against.
/// @param exe_dir_override Optional executable directory; when null the real exe path is queried.
/// @param kind Shader family to locate (SPIR-V or Metal).
/// @returns Newly allocated path to the shader directory.
/// @note Returns `error.ShaderDirNotFound` when no known layout exists.
pub fn resolveShaderDirFrom(
    allocator: std.mem.Allocator,
    base_dir: std.Io.Dir,
    exe_dir_override: ?[]const u8,
    kind: ShaderKind,
) ![]u8 {
    if (try exeRelativeShaderDir(allocator, kind, exe_dir_override)) |dir| {
        return dir;
    }

    const candidates = switch (kind) {
        .spirv => spirv_cwd_candidates[0..],
        .metal => metal_cwd_candidates[0..],
    };
    for (candidates) |candidate| {
        if (dirExists(base_dir, candidate)) {
            return allocator.dupe(u8, candidate);
        }
    }

    return error.ShaderDirNotFound;
}

fn exeRelativeShaderDir(
    allocator: std.mem.Allocator,
    kind: ShaderKind,
    exe_dir_override: ?[]const u8,
) !?[]u8 {
    if (exe_dir_override) |exe_dir| {
        return try exeRelativeShaderDirFromExeDir(allocator, kind, exe_dir);
    }

    const exe_path = try std.process.executablePathAlloc(host_io.io, allocator);
    defer allocator.free(exe_path);
    const exe_dir = std.fs.path.dirname(exe_path) orelse ".";
    return try exeRelativeShaderDirFromExeDir(allocator, kind, exe_dir);
}

fn exeRelativeShaderDirFromExeDir(
    allocator: std.mem.Allocator,
    kind: ShaderKind,
    exe_dir: []const u8,
) !?[]u8 {
    const dir = switch (kind) {
        .spirv => try std.fs.path.join(allocator, &.{ exe_dir, "..", "share", "zinc", "shaders" }),
        .metal => try std.fs.path.join(allocator, &.{ exe_dir, "..", "share", "zinc", "shaders", "metal" }),
    };
    errdefer allocator.free(dir);
    if (dirExists(std.Io.Dir.cwd(), dir)) {
        return dir;
    }
    allocator.free(dir);
    return null;
}

fn dirExists(base_dir: std.Io.Dir, path: []const u8) bool {
    if (std.fs.path.isAbsolute(path)) {
        var dir = std.Io.Dir.openDirAbsolute(host_io.io, path, .{}) catch return false;
        dir.close(host_io.io);
        return true;
    }

    var dir = base_dir.openDir(host_io.io, path, .{}) catch return false;
    dir.close(host_io.io);
    return true;
}

test "resolveShaderDirFrom finds SPIR-V cwd candidate" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.createDirPath(host_io.io, "zig-out/share/zinc/shaders");

    const result = try resolveShaderDirFrom(std.testing.allocator, tmp.dir, "/nonexistent", .spirv);
    defer std.testing.allocator.free(result);

    try std.testing.expectEqualStrings("zig-out/share/zinc/shaders", result);
}

test "resolveShaderDirFrom finds Metal cwd candidate" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.createDirPath(host_io.io, "share/zinc/shaders/metal");

    const result = try resolveShaderDirFrom(std.testing.allocator, tmp.dir, "/nonexistent", .metal);
    defer std.testing.allocator.free(result);

    try std.testing.expectEqualStrings("share/zinc/shaders/metal", result);
}

test "resolveShaderDirFrom falls back to SPIR-V exe-relative layout" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.createDirPath(host_io.io, "install/bin");
    try tmp.dir.createDirPath(host_io.io, "install/share/zinc/shaders");

    const bin_dir = try tmp.dir.realPathFileAlloc(host_io.io, "install/bin", std.testing.allocator);
    defer std.testing.allocator.free(bin_dir);

    var empty = std.testing.tmpDir(.{});
    defer empty.cleanup();

    const result = try resolveShaderDirFrom(std.testing.allocator, empty.dir, bin_dir, .spirv);
    defer std.testing.allocator.free(result);

    try std.testing.expect(std.mem.endsWith(u8, result, "share/zinc/shaders"));
}

test "resolveShaderDirFrom falls back to Metal exe-relative layout" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.createDirPath(host_io.io, "install/bin");
    try tmp.dir.createDirPath(host_io.io, "install/share/zinc/shaders/metal");

    const bin_dir = try tmp.dir.realPathFileAlloc(host_io.io, "install/bin", std.testing.allocator);
    defer std.testing.allocator.free(bin_dir);

    var empty = std.testing.tmpDir(.{});
    defer empty.cleanup();

    const result = try resolveShaderDirFrom(std.testing.allocator, empty.dir, bin_dir, .metal);
    defer std.testing.allocator.free(result);

    try std.testing.expect(std.mem.endsWith(u8, result, "share/zinc/shaders/metal"));
}

test "resolveShaderDirFrom reports missing shader directory" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try std.testing.expectError(
        error.ShaderDirNotFound,
        resolveShaderDirFrom(std.testing.allocator, tmp.dir, "/this/path/has/no/shaders", .spirv),
    );
}
