//! Thin AMDGPU kernel-driver queries used by direct ZINC_RT tiers.
//!
//! This file intentionally starts with capability discovery only. T2 UMQ queue
//! creation needs the same UAPI definitions, but selection must first prove the
//! kernel exposes compute user queues instead of relying on kernel version alone.
//! @section Inference Runtime
const std = @import("std");
const builtin = @import("builtin");

const linux = std.os.linux;

pub const QueryStatus = enum {
    available,
    unsupported_os,
    render_node_open_failed,
    hw_ip_info_failed,
    compute_userq_unsupported,
    compute_userq_slots_missing,
    fw_metadata_failed,
    invalid_fw_metadata,
};

pub const ComputeUserqInfo = struct {
    userq_slots: u32,
    eop_size: u32,
    eop_alignment: u32,
};

pub const QueryResult = struct {
    status: QueryStatus,
    info: ?ComputeUserqInfo = null,
    errno: ?linux.E = null,
};

const drm_ioctl_base: u8 = 'd';
const drm_command_base: u8 = 0x40;

pub const AMDGPU_HW_IP_COMPUTE: u32 = 1;
pub const AMDGPU_INFO_HW_IP_INFO: u32 = 0x02;
pub const AMDGPU_INFO_UQ_FW_AREAS: u32 = 0x24;
pub const AMDGPU_USERQ_OP_CREATE: u32 = 1;
pub const AMDGPU_USERQ_OP_FREE: u32 = 2;

pub const AMDGPU_GEM_DOMAIN_GTT: u64 = 0x2;
pub const AMDGPU_GEM_DOMAIN_VRAM: u64 = 0x4;
pub const AMDGPU_GEM_DOMAIN_DOORBELL: u64 = 0x40;

pub const AMDGPU_GEM_CREATE_CPU_GTT_USWC: u64 = 1 << 2;
pub const AMDGPU_GEM_CREATE_VRAM_CLEARED: u64 = 1 << 3;
pub const AMDGPU_GEM_CREATE_VM_ALWAYS_VALID: u64 = 1 << 6;

pub const AMDGPU_VA_OP_MAP: u32 = 1;
pub const AMDGPU_VM_PAGE_READABLE: u32 = 1 << 1;
pub const AMDGPU_VM_PAGE_WRITEABLE: u32 = 1 << 2;
pub const AMDGPU_VM_PAGE_EXECUTABLE: u32 = 1 << 3;
pub const AMDGPU_VM_MTYPE_DEFAULT: u32 = 0 << 5;

const drm_amdgpu_gem_create_nr: u8 = 0x00;
const drm_amdgpu_gem_mmap_nr: u8 = 0x01;
const drm_amdgpu_info_nr: u8 = 0x05;
const drm_amdgpu_gem_va_nr: u8 = 0x08;
const drm_amdgpu_userq_nr: u8 = 0x16;

pub const DrmAmdgpuGemCreateIn = extern struct {
    bo_size: u64,
    alignment: u64,
    domains: u64,
    domain_flags: u64,
};

pub const DrmAmdgpuGemCreateOut = extern struct {
    handle: u32,
    _pad: u32,
};

pub const DrmAmdgpuGemCreate = extern union {
    in: DrmAmdgpuGemCreateIn,
    out: DrmAmdgpuGemCreateOut,
};

pub const DrmAmdgpuGemMmapIn = extern struct {
    handle: u32,
    _pad: u32,
};

pub const DrmAmdgpuGemMmapOut = extern struct {
    addr_ptr: u64,
};

pub const DrmAmdgpuGemMmap = extern union {
    in: DrmAmdgpuGemMmapIn,
    out: DrmAmdgpuGemMmapOut,
};

pub const DrmAmdgpuGemVa = extern struct {
    handle: u32,
    _pad: u32,
    operation: u32,
    flags: u32,
    va_address: u64,
    offset_in_bo: u64,
    map_size: u64,
    vm_timeline_point: u64,
    vm_timeline_syncobj_out: u32,
    num_syncobj_handles: u32,
    input_fence_syncobj_handles: u64,
};

pub const DrmAmdgpuInfo = extern struct {
    return_pointer: u64,
    return_size: u32,
    query: u32,
    query_data: extern union {
        mode_crtc: extern struct {
            id: u32,
            _pad: u32,
        },
        query_hw_ip: extern struct {
            type: u32,
            ip_instance: u32,
        },
        read_mmr_reg: extern struct {
            dword_offset: u32,
            count: u32,
            instance: u32,
            flags: u32,
        },
        query_fw: extern struct {
            fw_type: u32,
            ip_instance: u32,
            index: u32,
            _pad: u32,
        },
        vbios_info: extern struct {
            type: u32,
            offset: u32,
        },
        sensor_info: extern struct {
            type: u32,
        },
        video_cap: extern struct {
            type: u32,
        },
    },
};

pub const DrmAmdgpuInfoHwIp = extern struct {
    hw_ip_version_major: u32,
    hw_ip_version_minor: u32,
    capabilities_flags: u64,
    ib_start_alignment: u32,
    ib_size_alignment: u32,
    available_rings: u32,
    ip_discovery_version: u32,
    userq_num_slots: u32,
};

pub const DrmAmdgpuInfoUqMetadata = extern union {
    gfx: extern struct {
        shadow_size: u32,
        shadow_alignment: u32,
        csa_size: u32,
        csa_alignment: u32,
    },
    compute: extern struct {
        eop_size: u32,
        eop_alignment: u32,
    },
    sdma: extern struct {
        csa_size: u32,
        csa_alignment: u32,
    },
};

pub const DrmAmdgpuUserqIn = extern struct {
    op: u32,
    queue_id: u32,
    ip_type: u32,
    doorbell_handle: u32,
    doorbell_offset: u32,
    flags: u32,
    queue_va: u64,
    queue_size: u64,
    rptr_va: u64,
    wptr_va: u64,
    mqd: u64,
    mqd_size: u64,
};

pub const DrmAmdgpuUserqOut = extern struct {
    queue_id: u32,
    _pad: u32,
};

pub const DrmAmdgpuUserq = extern union {
    in: DrmAmdgpuUserqIn,
    out: DrmAmdgpuUserqOut,
};

pub const DrmAmdgpuUserqMqdComputeGfx11 = extern struct {
    eop_va: u64,
};

pub const Bo = struct {
    handle: u32,
    size: u64,
};

const drm_ioctl_amdgpu_gem_create = linux.IOCTL.IOWR(
    drm_ioctl_base,
    drm_command_base + drm_amdgpu_gem_create_nr,
    DrmAmdgpuGemCreate,
);

const drm_ioctl_amdgpu_gem_mmap = linux.IOCTL.IOWR(
    drm_ioctl_base,
    drm_command_base + drm_amdgpu_gem_mmap_nr,
    DrmAmdgpuGemMmap,
);

const drm_ioctl_amdgpu_info = linux.IOCTL.IOW(
    drm_ioctl_base,
    drm_command_base + drm_amdgpu_info_nr,
    DrmAmdgpuInfo,
);

const drm_ioctl_amdgpu_gem_va = linux.IOCTL.IOW(
    drm_ioctl_base,
    drm_command_base + drm_amdgpu_gem_va_nr,
    DrmAmdgpuGemVa,
);

const drm_ioctl_amdgpu_userq = linux.IOCTL.IOWR(
    drm_ioctl_base,
    drm_command_base + drm_amdgpu_userq_nr,
    DrmAmdgpuUserq,
);

pub fn queryComputeUserq(render_node: []const u8) QueryResult {
    if (builtin.os.tag != .linux) {
        return .{ .status = .unsupported_os };
    }

    var file = std.fs.openFileAbsolute(render_node, .{ .mode = .read_write }) catch {
        return .{ .status = .render_node_open_failed };
    };
    defer file.close();

    const hw_ip = queryHwIp(file, AMDGPU_HW_IP_COMPUTE) catch |err| {
        return .{
            .status = .hw_ip_info_failed,
            .errno = errorToErrno(err),
        };
    };

    if (hw_ip.available_rings == 0) {
        return .{ .status = .compute_userq_unsupported };
    }
    if (hw_ip.userq_num_slots == 0) {
        return .{ .status = .compute_userq_slots_missing };
    }

    const metadata = queryUqMetadataCompute(file) catch |err| {
        return .{
            .status = .fw_metadata_failed,
            .errno = errorToErrno(err),
        };
    };

    if (metadata.compute.eop_size == 0 or metadata.compute.eop_alignment == 0) {
        return .{ .status = .invalid_fw_metadata };
    }

    return .{
        .status = .available,
        .info = .{
            .userq_slots = hw_ip.userq_num_slots,
            .eop_size = metadata.compute.eop_size,
            .eop_alignment = metadata.compute.eop_alignment,
        },
    };
}

pub fn queryHwIp(file: std.fs.File, ip_type: u32) !DrmAmdgpuInfoHwIp {
    var out: DrmAmdgpuInfoHwIp = std.mem.zeroes(DrmAmdgpuInfoHwIp);
    var query: DrmAmdgpuInfo = std.mem.zeroes(DrmAmdgpuInfo);
    query.return_pointer = @intFromPtr(&out);
    query.return_size = @sizeOf(DrmAmdgpuInfoHwIp);
    query.query = AMDGPU_INFO_HW_IP_INFO;
    query.query_data.query_hw_ip = .{
        .type = ip_type,
        .ip_instance = 0,
    };
    try ioctlRaw(file, drm_ioctl_amdgpu_info, @intFromPtr(&query));
    return out;
}

fn queryUqMetadataCompute(file: std.fs.File) !DrmAmdgpuInfoUqMetadata {
    var out: DrmAmdgpuInfoUqMetadata = std.mem.zeroes(DrmAmdgpuInfoUqMetadata);
    var query: DrmAmdgpuInfo = std.mem.zeroes(DrmAmdgpuInfo);
    query.return_pointer = @intFromPtr(&out);
    query.return_size = @sizeOf(DrmAmdgpuInfoUqMetadata);
    query.query = AMDGPU_INFO_UQ_FW_AREAS;
    query.query_data.query_hw_ip = .{
        .type = AMDGPU_HW_IP_COMPUTE,
        .ip_instance = 0,
    };
    try ioctlRaw(file, drm_ioctl_amdgpu_info, @intFromPtr(&query));
    return out;
}

pub fn createGem(file: std.fs.File, size: u64, alignment: u64, domains: u64, flags: u64) !Bo {
    var create: DrmAmdgpuGemCreate = std.mem.zeroes(DrmAmdgpuGemCreate);
    create.in = .{
        .bo_size = size,
        .alignment = alignment,
        .domains = domains,
        .domain_flags = flags,
    };
    try ioctlRaw(file, drm_ioctl_amdgpu_gem_create, @intFromPtr(&create));
    return .{
        .handle = create.out.handle,
        .size = size,
    };
}

pub fn mmapGem(file: std.fs.File, bo: Bo, prot: u32) ![]align(std.heap.page_size_min) u8 {
    var mmap_args: DrmAmdgpuGemMmap = std.mem.zeroes(DrmAmdgpuGemMmap);
    mmap_args.in = .{
        .handle = bo.handle,
        ._pad = 0,
    };
    try ioctlRaw(file, drm_ioctl_amdgpu_gem_mmap, @intFromPtr(&mmap_args));

    return std.posix.mmap(
        null,
        @intCast(bo.size),
        prot,
        .{ .TYPE = .SHARED },
        file.handle,
        mmap_args.out.addr_ptr,
    );
}

pub fn mapGemVa(file: std.fs.File, bo: Bo, va: u64, flags: u32) !void {
    var args: DrmAmdgpuGemVa = std.mem.zeroes(DrmAmdgpuGemVa);
    args.handle = bo.handle;
    args.operation = AMDGPU_VA_OP_MAP;
    args.flags = flags;
    args.va_address = va;
    args.offset_in_bo = 0;
    args.map_size = bo.size;
    try ioctlRaw(file, drm_ioctl_amdgpu_gem_va, @intFromPtr(&args));
}

pub fn createComputeUserq(
    file: std.fs.File,
    doorbell_handle: u32,
    doorbell_offset: u32,
    queue_va: u64,
    queue_size: u64,
    rptr_va: u64,
    wptr_va: u64,
    eop_va: u64,
    flags: u32,
) !u32 {
    var mqd: DrmAmdgpuUserqMqdComputeGfx11 = .{ .eop_va = eop_va };
    var args: DrmAmdgpuUserq = std.mem.zeroes(DrmAmdgpuUserq);
    args.in = .{
        .op = AMDGPU_USERQ_OP_CREATE,
        .queue_id = 0,
        .ip_type = AMDGPU_HW_IP_COMPUTE,
        .doorbell_handle = doorbell_handle,
        .doorbell_offset = doorbell_offset,
        .flags = flags,
        .queue_va = queue_va,
        .queue_size = queue_size,
        .rptr_va = rptr_va,
        .wptr_va = wptr_va,
        .mqd = @intFromPtr(&mqd),
        .mqd_size = @sizeOf(DrmAmdgpuUserqMqdComputeGfx11),
    };
    try ioctlRaw(file, drm_ioctl_amdgpu_userq, @intFromPtr(&args));
    return args.out.queue_id;
}

pub fn freeUserq(file: std.fs.File, queue_id: u32) !void {
    var args: DrmAmdgpuUserq = std.mem.zeroes(DrmAmdgpuUserq);
    args.in = .{
        .op = AMDGPU_USERQ_OP_FREE,
        .queue_id = queue_id,
        .ip_type = 0,
        .doorbell_handle = 0,
        .doorbell_offset = 0,
        .flags = 0,
        .queue_va = 0,
        .queue_size = 0,
        .rptr_va = 0,
        .wptr_va = 0,
        .mqd = 0,
        .mqd_size = 0,
    };
    try ioctlRaw(file, drm_ioctl_amdgpu_userq, @intFromPtr(&args));
}

const IoctlError = error{IoctlFailed};

var last_ioctl_errno: ?linux.E = null;

pub fn lastErrno() ?linux.E {
    return last_ioctl_errno;
}

fn ioctlRaw(file: std.fs.File, request: u32, arg: usize) IoctlError!void {
    last_ioctl_errno = null;
    const rc = linux.ioctl(file.handle, request, arg);
    const err = linux.E.init(rc);
    if (err != .SUCCESS) {
        last_ioctl_errno = err;
        return error.IoctlFailed;
    }
}

fn errorToErrno(err: anyerror) ?linux.E {
    return switch (err) {
        error.IoctlFailed => last_ioctl_errno,
        else => null,
    };
}

test "amdgpu info ioctl uapi layout is stable for userq discovery" {
    try std.testing.expectEqual(@as(usize, 32), @sizeOf(DrmAmdgpuGemCreate));
    try std.testing.expectEqual(@as(usize, 8), @sizeOf(DrmAmdgpuGemMmap));
    try std.testing.expectEqual(@as(usize, 64), @sizeOf(DrmAmdgpuGemVa));
    try std.testing.expectEqual(@as(usize, 32), @sizeOf(DrmAmdgpuInfo));
    try std.testing.expectEqual(@as(usize, 40), @sizeOf(DrmAmdgpuInfoHwIp));
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(DrmAmdgpuInfoUqMetadata));
    try std.testing.expectEqual(@as(usize, 72), @sizeOf(DrmAmdgpuUserqIn));
    try std.testing.expectEqual(@as(usize, 72), @sizeOf(DrmAmdgpuUserq));
    try std.testing.expectEqual(@as(usize, 8), @sizeOf(DrmAmdgpuUserqMqdComputeGfx11));
}

test "queryComputeUserq is disabled on non-Linux hosts" {
    if (builtin.os.tag == .linux) return error.SkipZigTest;
    const result = queryComputeUserq("/dev/dri/renderD128");
    try std.testing.expectEqual(QueryStatus.unsupported_os, result.status);
}
