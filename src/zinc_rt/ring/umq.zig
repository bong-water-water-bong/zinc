//! AMDGPU user-mode queue (T2) availability and create/free smoke gate.
//!
//! M1 uses UMQ for direct submission on Linux kernels that expose the AMDGPU
//! user queue ABI. The preflight is intentionally cheap, while the smoke gate
//! exercises the actual GEM/VA/USERQ_CREATE/USERQ_FREE path required before
//! lowering decode packets onto T2.
//! @section Inference Runtime
const std = @import("std");
const builtin = @import("builtin");
const kmd = @import("../kmd.zig");

const linux = std.os.linux;

pub const min_linux_major: u32 = 6;
pub const min_linux_minor: u32 = 16;
pub const default_render_node = "/dev/dri/renderD128";
pub const user_queue_param_path = "/sys/module/amdgpu/parameters/user_queue";

pub const ProbeStatus = enum {
    preflight_ok,
    unsupported_os,
    kernel_too_old,
    render_node_missing,
    user_queue_param_missing,
    user_queue_disabled,
};

pub const KernelVersion = struct {
    major: u32,
    minor: u32,
    patch: u32 = 0,
};

pub const ProbeResult = struct {
    status: ProbeStatus,
    kernel: ?KernelVersion = null,
    render_node: []const u8 = default_render_node,
    user_queue_mode: ?i32 = null,

    pub fn preflightOk(self: ProbeResult) bool {
        return self.status == .preflight_ok;
    }
};

pub const SmokeStatus = enum {
    passed,
    unsupported_os,
    preflight_failed,
    compute_userq_unavailable,
    render_node_open_failed,
    gem_create_failed,
    gem_mmap_failed,
    gem_va_failed,
    userq_create_failed,
    userq_free_failed,
};

pub const SmokeResult = struct {
    status: SmokeStatus,
    queue_id: ?u32 = null,
    preflight_status: ?ProbeStatus = null,
    query_status: ?kmd.QueryStatus = null,
    errno: ?linux.E = null,
    userq_slots: u32 = 0,
    eop_size: u32 = 0,
    eop_alignment: u32 = 0,

    pub fn ok(self: SmokeResult) bool {
        return self.status == .passed;
    }
};

pub fn probeDefault() ProbeResult {
    return probePath(default_render_node);
}

pub fn admissionProbeDefault() bool {
    const preflight = probeDefault();
    if (!preflight.preflightOk()) return false;

    const query = kmd.queryComputeUserq(default_render_node);
    return query.status == .available;
}

pub fn createFreeSmokeDefault() SmokeResult {
    return createFreeSmokePath(default_render_node);
}

pub fn createFreeSmokePath(render_node: []const u8) SmokeResult {
    if (builtin.os.tag != .linux) {
        return .{ .status = .unsupported_os };
    }

    const preflight = probePath(render_node);
    if (!preflight.preflightOk()) {
        return .{
            .status = .preflight_failed,
            .preflight_status = preflight.status,
        };
    }

    const query = kmd.queryComputeUserq(render_node);
    if (query.status != .available) {
        return .{
            .status = .compute_userq_unavailable,
            .query_status = query.status,
            .errno = query.errno,
        };
    }

    return createFreeSmokeWithInfo(render_node, query.info.?);
}

pub fn probePath(render_node: []const u8) ProbeResult {
    if (builtin.os.tag != .linux) {
        return .{ .status = .unsupported_os, .render_node = render_node };
    }

    const kernel = currentKernelVersion() catch {
        return .{ .status = .kernel_too_old, .render_node = render_node };
    };
    if (!kernelSupportsUmq(kernel)) {
        return .{ .status = .kernel_too_old, .kernel = kernel, .render_node = render_node };
    }

    var file = std.fs.openFileAbsolute(render_node, .{}) catch {
        return .{ .status = .render_node_missing, .kernel = kernel, .render_node = render_node };
    };
    file.close();

    const mode = readUserQueueMode() catch {
        return .{ .status = .user_queue_param_missing, .kernel = kernel, .render_node = render_node };
    };
    if (!userQueueModeEnablesUmq(mode)) {
        return .{ .status = .user_queue_disabled, .kernel = kernel, .render_node = render_node, .user_queue_mode = mode };
    }

    return .{ .status = .preflight_ok, .kernel = kernel, .render_node = render_node, .user_queue_mode = mode };
}

fn createFreeSmokeWithInfo(render_node: []const u8, info: kmd.ComputeUserqInfo) SmokeResult {
    var file = std.fs.openFileAbsolute(render_node, .{ .mode = .read_write }) catch {
        return smokeFailure(.render_node_open_failed, info);
    };
    defer file.close();

    const queue_size: u64 = 0x10000;
    const page_size: u64 = 4096;
    const base_va: u64 = 0x1_0000_0000;
    const queue_va = base_va;
    const wptr_va = base_va + 0x2_0000;
    const rptr_va = base_va + 0x3_0000;
    const eop_va = base_va + 0x4_0000;
    const doorbell_offset: u32 = 4;
    const va_flags = kmd.AMDGPU_VM_PAGE_READABLE |
        kmd.AMDGPU_VM_PAGE_WRITEABLE |
        kmd.AMDGPU_VM_PAGE_EXECUTABLE |
        kmd.AMDGPU_VM_MTYPE_DEFAULT;

    const queue_bo = kmd.createGem(
        file,
        queue_size,
        256,
        kmd.AMDGPU_GEM_DOMAIN_GTT,
        kmd.AMDGPU_GEM_CREATE_CPU_GTT_USWC | kmd.AMDGPU_GEM_CREATE_VM_ALWAYS_VALID,
    ) catch return smokeIoctlFailure(.gem_create_failed, info);

    const wptr_bo = kmd.createGem(
        file,
        page_size,
        256,
        kmd.AMDGPU_GEM_DOMAIN_GTT,
        kmd.AMDGPU_GEM_CREATE_CPU_GTT_USWC | kmd.AMDGPU_GEM_CREATE_VM_ALWAYS_VALID,
    ) catch return smokeIoctlFailure(.gem_create_failed, info);

    const rptr_bo = kmd.createGem(
        file,
        page_size,
        256,
        kmd.AMDGPU_GEM_DOMAIN_VRAM,
        kmd.AMDGPU_GEM_CREATE_VRAM_CLEARED | kmd.AMDGPU_GEM_CREATE_VM_ALWAYS_VALID,
    ) catch return smokeIoctlFailure(.gem_create_failed, info);

    const eop_size = alignUp(@max(@as(u64, info.eop_size), page_size), @max(@as(u64, info.eop_alignment), page_size));
    const eop_bo = kmd.createGem(
        file,
        eop_size,
        @max(@as(u64, info.eop_alignment), 256),
        kmd.AMDGPU_GEM_DOMAIN_VRAM,
        kmd.AMDGPU_GEM_CREATE_VRAM_CLEARED | kmd.AMDGPU_GEM_CREATE_VM_ALWAYS_VALID,
    ) catch return smokeIoctlFailure(.gem_create_failed, info);

    const doorbell_bo = kmd.createGem(
        file,
        page_size,
        256,
        kmd.AMDGPU_GEM_DOMAIN_DOORBELL,
        kmd.AMDGPU_GEM_CREATE_VM_ALWAYS_VALID,
    ) catch return smokeIoctlFailure(.gem_create_failed, info);

    const queue_map = kmd.mmapGem(file, queue_bo, std.posix.PROT.READ | std.posix.PROT.WRITE) catch {
        return smokeIoctlFailure(.gem_mmap_failed, info);
    };
    defer std.posix.munmap(queue_map);
    @memset(queue_map, 0);

    const wptr_map = kmd.mmapGem(file, wptr_bo, std.posix.PROT.READ | std.posix.PROT.WRITE) catch {
        return smokeIoctlFailure(.gem_mmap_failed, info);
    };
    defer std.posix.munmap(wptr_map);
    @memset(wptr_map, 0);

    const doorbell_map = kmd.mmapGem(file, doorbell_bo, std.posix.PROT.WRITE) catch {
        return smokeIoctlFailure(.gem_mmap_failed, info);
    };
    defer std.posix.munmap(doorbell_map);

    kmd.mapGemVa(file, queue_bo, queue_va, va_flags) catch return smokeIoctlFailure(.gem_va_failed, info);
    kmd.mapGemVa(file, wptr_bo, wptr_va, va_flags) catch return smokeIoctlFailure(.gem_va_failed, info);
    kmd.mapGemVa(file, rptr_bo, rptr_va, va_flags) catch return smokeIoctlFailure(.gem_va_failed, info);
    kmd.mapGemVa(file, eop_bo, eop_va, va_flags) catch return smokeIoctlFailure(.gem_va_failed, info);

    const queue_id = kmd.createComputeUserq(
        file,
        doorbell_bo.handle,
        doorbell_offset,
        queue_va,
        queue_size,
        rptr_va,
        wptr_va,
        eop_va,
        0,
    ) catch return smokeIoctlFailure(.userq_create_failed, info);

    kmd.freeUserq(file, queue_id) catch {
        var result = smokeIoctlFailure(.userq_free_failed, info);
        result.queue_id = queue_id;
        return result;
    };

    return .{
        .status = .passed,
        .queue_id = queue_id,
        .userq_slots = info.userq_slots,
        .eop_size = info.eop_size,
        .eop_alignment = info.eop_alignment,
    };
}

fn smokeFailure(status: SmokeStatus, info: kmd.ComputeUserqInfo) SmokeResult {
    return .{
        .status = status,
        .userq_slots = info.userq_slots,
        .eop_size = info.eop_size,
        .eop_alignment = info.eop_alignment,
    };
}

fn smokeIoctlFailure(status: SmokeStatus, info: kmd.ComputeUserqInfo) SmokeResult {
    var result = smokeFailure(status, info);
    result.errno = kmd.lastErrno();
    return result;
}

fn alignUp(value: u64, alignment: u64) u64 {
    if (alignment == 0) return value;
    const rem = value % alignment;
    return if (rem == 0) value else value + (alignment - rem);
}

pub fn kernelSupportsUmq(version: KernelVersion) bool {
    return version.major > min_linux_major or
        (version.major == min_linux_major and version.minor >= min_linux_minor);
}

pub fn parseKernelRelease(release: []const u8) ?KernelVersion {
    var it = std.mem.splitScalar(u8, release, '.');
    const major_raw = it.next() orelse return null;
    const minor_raw = it.next() orelse return null;
    const patch_raw = it.next() orelse "0";

    return .{
        .major = parseLeadingU32(major_raw) orelse return null,
        .minor = parseLeadingU32(minor_raw) orelse return null,
        .patch = parseLeadingU32(patch_raw) orelse 0,
    };
}

fn currentKernelVersion() !KernelVersion {
    const uts = std.posix.uname();
    const release = std.mem.sliceTo(&uts.release, 0);
    return parseKernelRelease(release) orelse error.UnsupportedKernelRelease;
}

fn readUserQueueMode() !i32 {
    var file = try std.fs.openFileAbsolute(user_queue_param_path, .{});
    defer file.close();

    var buf: [32]u8 = undefined;
    const n = try file.readAll(&buf);
    const value = std.mem.trim(u8, buf[0..n], " \t\r\n");
    return parseLeadingI32(value) orelse error.InvalidUserQueueMode;
}

pub fn userQueueModeEnablesUmq(mode: i32) bool {
    return mode == -1 or mode == 1 or mode == 2;
}

fn parseLeadingU32(value: []const u8) ?u32 {
    var end: usize = 0;
    while (end < value.len and std.ascii.isDigit(value[end])) : (end += 1) {}
    if (end == 0) return null;
    return std.fmt.parseInt(u32, value[0..end], 10) catch null;
}

fn parseLeadingI32(value: []const u8) ?i32 {
    if (value.len == 0) return null;
    var end: usize = if (value[0] == '-' or value[0] == '+') 1 else 0;
    while (end < value.len and std.ascii.isDigit(value[end])) : (end += 1) {}
    if (end == 0 or (end == 1 and (value[0] == '-' or value[0] == '+'))) return null;
    return std.fmt.parseInt(i32, value[0..end], 10) catch null;
}

test "parseKernelRelease accepts distro suffixes and rc tags" {
    try std.testing.expectEqualDeep(KernelVersion{ .major = 6, .minor = 16, .patch = 0 }, parseKernelRelease("6.16.0-24-generic").?);
    try std.testing.expectEqualDeep(KernelVersion{ .major = 6, .minor = 16, .patch = 0 }, parseKernelRelease("6.16-rc4").?);
    try std.testing.expectEqualDeep(KernelVersion{ .major = 6, .minor = 15, .patch = 12 }, parseKernelRelease("6.15.12").?);
}

test "kernelSupportsUmq gates T2 at Linux 6.16" {
    try std.testing.expect(!kernelSupportsUmq(.{ .major = 6, .minor = 15 }));
    try std.testing.expect(kernelSupportsUmq(.{ .major = 6, .minor = 16 }));
    try std.testing.expect(kernelSupportsUmq(.{ .major = 7, .minor = 0 }));
}

test "userQueueModeEnablesUmq accepts kernel plus user queue modes" {
    try std.testing.expect(userQueueModeEnablesUmq(-1));
    try std.testing.expect(!userQueueModeEnablesUmq(0));
    try std.testing.expect(userQueueModeEnablesUmq(1));
    try std.testing.expect(userQueueModeEnablesUmq(2));
    try std.testing.expect(!userQueueModeEnablesUmq(3));
}

test "parseLeadingI32 accepts amdgpu auto module parameter" {
    try std.testing.expectEqual(@as(i32, -1), parseLeadingI32("-1\n").?);
    try std.testing.expectEqual(@as(i32, 2), parseLeadingI32("+2").?);
    try std.testing.expectEqual(@as(i32, 16), parseLeadingI32("16-garbage").?);
    try std.testing.expectEqual(@as(?i32, null), parseLeadingI32("-"));
}

test "alignUp handles exact and non-exact UMQ metadata sizes" {
    try std.testing.expectEqual(@as(u64, 4096), alignUp(4096, 4096));
    try std.testing.expectEqual(@as(u64, 8192), alignUp(4097, 4096));
}

test "probePath does not report UMQ available on non-Linux hosts" {
    if (builtin.os.tag == .linux) return error.SkipZigTest;
    const result = probePath(default_render_node);
    try std.testing.expectEqual(ProbeStatus.unsupported_os, result.status);
    try std.testing.expect(!result.preflightOk());
}

test "createFreeSmokePath does not run USERQ ioctls on non-Linux hosts" {
    if (builtin.os.tag == .linux) return error.SkipZigTest;
    const result = createFreeSmokePath(default_render_node);
    try std.testing.expectEqual(SmokeStatus.unsupported_os, result.status);
    try std.testing.expect(!result.ok());
}
