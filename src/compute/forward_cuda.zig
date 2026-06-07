//! CUDA forward pass for the dense `qwen35` hybrid (Qwen 3.5 9B).
//!
//! M2 bring-up — incremental. The CUDA backend modules (device/buffer/pipeline/
//! command) and the kernel library (`src/shaders/cuda/kernels.cu`, NVRTC-compiled)
//! are wired here into a real forward pass. See `docs/cuda-backend.md` and
//! Effort 20/21. Weights upload VERBATIM-quantized (Q4_K/Q5_K/Q6_K/Q8_0 blocks)
//! and are dequantized inside the DMMV/GEMM kernels.
//!
//! @section Inference Runtime
const std = @import("std");
const device = @import("../cuda/device.zig");
const buffer = @import("../cuda/buffer.zig");
const pipeline = @import("../cuda/pipeline.zig");
const command = @import("../cuda/command.zig");
const gguf = @import("../model/gguf.zig");

const log = std.log.scoped(.cuda_fwd);

/// The CUDA kernel library, bundled into the binary and NVRTC-compiled on load.
const KERNELS_CU = @embedFile("../shaders/cuda/kernels.cu");

/// Exact `qwen35-9b-q4k-m` config (from the GGUF; general.file_type=15 = Q4_K_M).
pub const Cfg = struct {
    pub const arch = "qwen35";
    pub const n_layers: u32 = 32;
    pub const n_embd: u32 = 4096;
    pub const n_ff: u32 = 12288;
    pub const n_head: u32 = 16;
    pub const n_head_kv: u32 = 4;
    pub const head_dim: u32 = 256;
    pub const rms_eps: f32 = 1e-6;
    pub const vocab: u32 = 248320;
    pub const full_attention_interval: u32 = 4; // (L+1)%4==0 → full attn (8); else SSM (24)
    pub const rope_dim: u32 = 64;
    pub const rope_freq_base: f32 = 1.0e7;
    pub const ssm_d_conv: u32 = 4;
    pub const ssm_d_inner: u32 = 4096;
    pub const ssm_d_state: u32 = 128;
    pub const ssm_dt_rank: u32 = 32;
    pub const ssm_group_count: u32 = 16;
};

// ---- kernel push-constant structs (must match kernels.cu byte layout) -------
const RmsPush = extern struct { N: u32, eps: f32 };
const DmmvPush = extern struct {
    M: u32, // output rows  = weight dims[1]
    K: u32, // input cols   = weight dims[0]
    a_offset: u32 = 0, // byte offsets within the bound buffers
    x_offset: u32 = 0,
    y_offset: u32 = 0,
    acc_mode: u32 = 0, // 0 = set, 1 = accumulate
};

/// Map a GGUF quant type to its DMMV kernel name.
/// @param t GGUF/GGML storage type of the weight tensor.
/// @returns The kernel name in `kernels.cu`, or null if no DMMV kernel handles it.
fn dmmvKernel(t: gguf.GGMLType) ?[:0]const u8 {
    return switch (t) {
        .q4_k => "dmmv_q4k",
        .q5_k => "dmmv_q5k",
        .q6_k => "dmmv_q6k",
        .q8_0 => "dmmv_q8_0",
        .f32 => "dmmv_f32",
        else => null,
    };
}

/// Upload one GGUF tensor's raw (quantized) bytes to a device buffer.
/// @param ctx CUDA context handle (`CudaDevice.ctx`).
/// @param gf Parsed GGUF file providing tensor offsets/sizes.
/// @param mmap The memory-mapped model file backing the tensor data.
/// @param name Tensor name to locate and upload.
/// @returns A device buffer holding the verbatim (quantized) tensor bytes.
fn uploadTensor(
    ctx: ?*anyopaque,
    gf: *const gguf.GGUFFile,
    mmap: []const u8,
    name: []const u8,
) !buffer.CudaBuffer {
    const info = gf.findTensor(name) orelse {
        log.err("tensor not found: {s}", .{name});
        return error.TensorNotFound;
    };
    const off: usize = @intCast(gf.tensor_data_offset + info.offset);
    const sz: usize = @intCast(info.sizeBytes());
    return buffer.uploadMmap(@ptrCast(ctx), &mmap[off], sz);
}

/// M2 increment 1 — validate the CUDA forward foundation on real layer-0 weights:
/// GGUF→GPU upload, NVRTC kernel compile, and `rms_norm` + `dmmv_q4k` on the GPU,
/// asserting the output is finite. Called from `main.zig` for `-Dbackend=cuda`.
/// @param allocator Backing allocator for host scratch and parsing.
/// @param config CLI config; `config.model_path` and `config.device_index` are read.
/// @returns Void on success; errors on missing model, tensor, or non-finite output.
pub fn run(allocator: std.mem.Allocator, config: anytype) !void {
    const model_path = config.model_path orelse {
        log.err("CUDA backend needs a model path (-m <gguf>)", .{});
        return error.NoModel;
    };

    var dev = try device.CudaDevice.init(allocator, config.device_index);
    defer dev.deinit();
    const ctx = dev.ctx;
    var nb: [128]u8 = undefined;
    log.info("CUDA forward: {s} (cc={d}, {d} MiB)", .{ dev.name(&nb), dev.computeCapability(), dev.totalMemory() / (1024 * 1024) });

    // mmap + parse the gguf
    const file = try std.fs.cwd().openFile(model_path, .{});
    defer file.close();
    const fsize = (try file.stat()).size;
    const mmap = try std.posix.mmap(null, fsize, std.posix.PROT.READ, .{ .TYPE = .PRIVATE }, file.handle, 0);
    defer std.posix.munmap(mmap);
    var gf = try gguf.parseWithOptions(mmap, allocator, .{ .log_summary = false });
    defer gf.deinit();
    log.info("gguf: {d} tensors, data_offset={d}", .{ gf.tensors.items.len, gf.tensor_data_offset });

    // compile kernels (NVRTC wants a null-terminated source)
    const src = try allocator.dupeZ(u8, KERNELS_CU);
    defer allocator.free(src);
    var pipe_rms = try pipeline.createPipeline(ctx, src.ptr, "rms_norm");
    defer pipeline.freePipeline(&pipe_rms);
    var pipe_dmmv_q4k = try pipeline.createPipeline(ctx, src.ptr, "dmmv_q4k");
    defer pipeline.freePipeline(&pipe_dmmv_q4k);
    log.info("nvrtc: compiled rms_norm + dmmv_q4k", .{});

    // ---- foundation slice: hidden -> rms_norm -> attn_q DMMV ----------------
    const N = Cfg.n_embd;
    const q_info = gf.findTensor("blk.3.attn_q.weight") orelse return error.TensorNotFound;
    const M: u32 = @intCast(q_info.dims[1]); // output rows
    const K: u32 = @intCast(q_info.dims[0]); // input cols (== n_embd)
    log.info("blk.3.attn_q.weight: type={s} dims=[{d},{d}] -> M={d} K={d}", .{ @tagName(q_info.type_), q_info.dims[0], q_info.dims[1], M, K });

    var w_norm = try uploadTensor(ctx, &gf, mmap, "blk.3.attn_norm.weight");
    defer buffer.freeBuffer(&w_norm);
    var w_q = try uploadTensor(ctx, &gf, mmap, "blk.3.attn_q.weight");
    defer buffer.freeBuffer(&w_q);

    var hidden = try buffer.createBufferStaged(ctx, N * @sizeOf(f32));
    defer buffer.freeBuffer(&hidden);
    var norm = try buffer.createBuffer(ctx, N * @sizeOf(f32));
    defer buffer.freeBuffer(&norm);
    var qout = try buffer.createBuffer(ctx, M * @sizeOf(f32));
    defer buffer.freeBuffer(&qout);

    // synthetic input (increment 1 validates plumbing, not numerics yet)
    {
        const h: [*]f32 = @ptrCast(@alignCast(hidden.host_ptr.?));
        var i: usize = 0;
        while (i < N) : (i += 1) h[i] = @as(f32, @floatFromInt(i % 17)) * 0.1 - 0.7;
        buffer.upload(ctx, &hidden, std.mem.sliceAsBytes(h[0..N]));
    }

    var cmd = try command.beginCommand(ctx);
    const rms_push = RmsPush{ .N = N, .eps = Cfg.rms_eps };
    cmd.dispatch(&pipe_rms, .{ 1, 1, 1 }, .{ 256, 1, 1 }, &.{ &hidden, &w_norm, &norm }, &rms_push, @sizeOf(RmsPush), 0);
    cmd.barrier();
    const dmmv_push = DmmvPush{ .M = M, .K = K };
    cmd.dispatch(&pipe_dmmv_q4k, .{ M, 1, 1 }, .{ 256, 1, 1 }, &.{ &w_q, &norm, &qout }, &dmmv_push, @sizeOf(DmmvPush), 0);
    cmd.commitAndWait();

    // verify finite + sane
    const out = try allocator.alloc(f32, M);
    defer allocator.free(out);
    buffer.download(ctx, &qout, std.mem.sliceAsBytes(out));
    var mn: f32 = std.math.inf(f32);
    var mx: f32 = -std.math.inf(f32);
    var bad: usize = 0;
    for (out) |v| {
        if (!std.math.isFinite(v)) bad += 1;
        mn = @min(mn, v);
        mx = @max(mx, v);
    }
    log.info("attn_q out[0..4] = {d:.4} {d:.4} {d:.4} {d:.4}", .{ out[0], out[1], out[2], out[3] });
    log.info("attn_q out: min={d:.4} max={d:.4} non-finite={d}/{d}", .{ mn, mx, bad, M });
    if (bad != 0) return error.NonFiniteOutput;
    _ = dmmvKernel; // used by later increments (full weight set)
    log.info("=== M2 increment 1 OK: GGUF->GPU upload + NVRTC + rms_norm + dmmv_q4k on real weights ===", .{});
}
