//! CUDA buffer wrapper — device-local allocations with optional pinned staging.
//!
//! Unlike Metal (Apple unified memory), CUDA device memory is NOT CPU-visible;
//! host<->device transfers are explicit (`upload`/`download`), staged through
//! pinned host memory. Mirrors src/metal/buffer.zig.
//! @section CUDA Runtime
const std = @import("std");
const shim = @import("c.zig").shim;

/// CUDA device buffer handle plus optional pinned-host staging mirror.
pub const CudaBuffer = struct {
    handle: ?*shim.CudaBuf,
    size: usize,
    /// Pinned host staging pointer for staged buffers (null for plain device buffers).
    host_ptr: ?[*]u8 = null,
    /// False for lightweight aliases into a larger buffer owned elsewhere.
    owns_handle: bool = true,

    /// Raw device pointer (CUdeviceptr as u64) for kernel arg packing / aliasing.
    pub fn devicePtr(self: *const CudaBuffer) u64 {
        if (self.handle) |h| return shim.cuda_buffer_device_ptr(h);
        return 0;
    }

    /// Pinned host staging pointer, if this buffer was created staged.
    pub fn contents(self: *const CudaBuffer) ?[*]u8 {
        return self.host_ptr;
    }
};

/// Allocate a device-local buffer (the common case for weights/activations/state).
pub fn createBuffer(ctx: ?*shim.CudaCtx, size: usize) !CudaBuffer {
    const handle = shim.cuda_create_buffer(ctx, size);
    if (handle == null) return error.CudaBufferAllocFailed;
    return .{ .handle = handle, .size = size };
}

/// Allocate a device buffer paired with a pinned-host staging mirror for fast
/// `upload`/`download`. The host pointer is exposed via `contents()`.
pub fn createBufferStaged(ctx: ?*shim.CudaCtx, size: usize) !CudaBuffer {
    var cpu_ptr: ?*anyopaque = null;
    const handle = shim.cuda_create_buffer_staged(ctx, size, &cpu_ptr);
    if (handle == null) return error.CudaBufferAllocFailed;
    return .{ .handle = handle, .size = size, .host_ptr = @ptrCast(cpu_ptr) };
}

/// Register an existing host mapping (e.g. mmap'd weights) and copy to device —
/// the CUDA analogue of Metal's zero-copy wrapMmap.
pub fn uploadMmap(ctx: ?*shim.CudaCtx, host_ptr: *const anyopaque, size: usize) !CudaBuffer {
    const handle = shim.cuda_upload_mmap(ctx, host_ptr, size);
    if (handle == null) return error.CudaMmapUploadFailed;
    return .{ .handle = handle, .size = size };
}

/// Create a lightweight view into an existing buffer's device allocation. The
/// returned handle must not free the parent's device memory (the shim tracks
/// ownership on the C side).
pub fn aliasBuffer(base: *const CudaBuffer, offset: usize, size: usize) !CudaBuffer {
    const handle = shim.cuda_alias_buffer(base.handle, offset, size);
    if (handle == null) return error.CudaBufferAllocFailed;
    return .{ .handle = handle, .size = size, .owns_handle = false };
}

/// Free a buffer handle (the shim only releases device memory if this buffer
/// owns it — aliases just free the wrapper). Safe with a null handle.
pub fn freeBuffer(buf: *CudaBuffer) void {
    if (buf.handle) |h| {
        shim.cuda_free_buffer(h);
        buf.handle = null;
    }
}

/// Host->device copy (synchronous on the context stream).
pub fn upload(ctx: ?*shim.CudaCtx, buf: *const CudaBuffer, data: []const u8) void {
    shim.cuda_upload(ctx, buf.handle, @ptrCast(data.ptr), data.len);
}

/// Device->host copy (synchronous on the context stream).
pub fn download(ctx: ?*shim.CudaCtx, buf: *const CudaBuffer, dst: []u8) void {
    shim.cuda_download(ctx, buf.handle, @ptrCast(dst.ptr), dst.len);
}
