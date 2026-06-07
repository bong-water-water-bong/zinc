//! CUDA compute pipeline wrapper — NVRTC-compiled CUfunction (mirrors
//! src/metal/pipeline.zig). Compiles a `.cu` source string for the running
//! device's arch (sm_XY) or loads a precompiled cubin/PTX image.
//! @section CUDA Runtime
const std = @import("std");
const shim = @import("c.zig").shim;

/// A compiled CUDA kernel ready for dispatch.
pub const CudaPipeline = struct {
    /// Opaque handle to the C shim pipeline object (CUmodule + CUfunction).
    handle: ?*shim.CudaPipe,
    /// Maximum threads per block the kernel supports.
    max_threads: u32,
    /// Bytes of static shared memory the kernel declares.
    shared_mem: u32,
    /// Optional debug label (typically the kernel name).
    name: ?[]const u8 = null,
};

/// NVRTC-compile `cu_source` and resolve `fn_name` for dispatch.
pub fn createPipeline(ctx: ?*shim.CudaCtx, cu_source: [*:0]const u8, fn_name: [*:0]const u8) !CudaPipeline {
    const handle = shim.cuda_create_pipeline(ctx, cu_source, fn_name, null, 0);
    if (handle == null) return error.CudaPipelineCreateFailed;
    return .{
        .handle = handle,
        .max_threads = shim.cuda_pipeline_max_threads(handle),
        .shared_mem = shim.cuda_pipeline_shared_mem(handle),
    };
}

/// Load a kernel from a precompiled cubin/PTX image (offline nvcc path).
pub fn createPipelineFromImage(ctx: ?*shim.CudaCtx, image: [*]const u8, image_size: usize, fn_name: [*:0]const u8) !CudaPipeline {
    const handle = shim.cuda_create_pipeline_from_image(ctx, @ptrCast(image), image_size, fn_name);
    if (handle == null) return error.CudaPipelineCreateFailed;
    return .{
        .handle = handle,
        .max_threads = shim.cuda_pipeline_max_threads(handle),
        .shared_mem = shim.cuda_pipeline_shared_mem(handle),
    };
}

/// Opt this kernel into a larger dynamic shared-memory cap (Ada/Blackwell).
pub fn setMaxDynamicShared(pipe: *CudaPipeline, bytes: u32) void {
    if (pipe.handle) |h| shim.cuda_pipeline_set_max_dynamic_shared(h, bytes);
}

/// Release the pipeline handle. Safe to call with a null handle.
pub fn freePipeline(pipe: *CudaPipeline) void {
    if (pipe.handle) |h| {
        shim.cuda_free_pipeline(h);
        pipe.handle = null;
    }
}
