//! CUDA forward pass for the dense `qwen35` hybrid (Qwen 3.5 9B) — SCAFFOLD.
//!
//! Status: M1 bring-up scaffold. Establishes the structure + integration points
//! for ZINC's CUDA decode/prefill, but is **not yet wired** into the compute
//! dispatch (`gpu/interface.zig` + `main.zig` three-way) — that lands with the
//! M1 kernel set. See `docs/cuda-backend.md` §5 and Efforts 20 (prefill) / 21
//! (decode) under `loops/efforts/`.
//!
//! Mirrors `src/compute/forward_metal.zig` (raw-pointer binds; an async
//! stream/event command ring) using the CUDA backend modules surfaced by
//! `gpu/interface.zig` when `is_cuda`:
//!   - device   : `../cuda/device.zig`   — CudaDevice: caps (cc/SMs/vram), ctx
//!   - buffer   : `../cuda/buffer.zig`   — device buffers + pinned H2D/D2H staging
//!   - pipeline : `../cuda/pipeline.zig` — NVRTC compile `.cu` → CUfunction
//!   - command  : `../cuda/command.zig`  — CUstream dispatch; commitAsync/wait ring
//! Kernels live in `src/shaders/cuda/kernels.cu`, NVRTC-compiled on load for the
//! running arch (sm_89 on the 4090, sm_120 on the 5090).
//!
//! Target is the **dense** 9B — no MoE — so the expert path of the 35B plan
//! (`softmax_topk`, routed/shared experts) is intentionally absent.
//!
//! @section Inference Runtime
const std = @import("std");
const gpu = @import("../gpu/interface.zig");

/// CUDA forward pass for the dense qwen35 9B.
/// Milestones: M1 one correct token → M2 full prefill+decode → M3 fused/async perf.
pub const CudaForward = struct {
    allocator: std.mem.Allocator,
    // TODO M1 state (mirrors the Metal/Vulkan engines):
    //   device     : gpu.backend.CudaDevice
    //   pipelines  : compiled CUfunctions for the kernel set in the checklist below
    //   weights    : device buffers (mmap-staged H2D; quant-typed)
    //   kv_pool    : paged KV cache (attention layers)
    //   ssm_state  : conv ring `(d_conv-1)*inner` f32/layer + recurrent `dt_rank·…` f32/layer
    //   ring       : [N]command.CudaCommand pending ring over CUstream+CUevent (M3)

    pub fn init(allocator: std.mem.Allocator) !CudaForward {
        // TODO M1: select device (by UUID/cc), NVRTC-compile kernels.cu for the
        // running arch, allocate weight/scratch/KV/SSM buffers, stage weights H2D.
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *CudaForward) void {
        _ = self;
        // TODO: free buffers; destroy streams/events; unload modules; pop ctx.
    }

    /// M1 — one correct decode token, validated token-for-token vs the
    /// Metal/Vulkan reference. Forward order for the **dense** qwen35 9B
    /// (docs/cuda-backend.md §5 M1, minus the MoE path):
    ///   1. embedding gather (host) → hidden
    ///   per layer L:
    ///     2. rms_norm (input)                                    [done]
    ///     3. is_full_attn = ((L+1) % full_attention_interval == 0)
    ///          attention: DMMV Q/K/V → qk_norm + RoPE → kv_cache_write →
    ///                     softmax(QKᵀ)V (single query) → DMMV O
    ///          SSM      : DMMV in → ssm_conv1d → ssm_delta_net (recurrent) →
    ///                     ssm_gated_norm → DMMV out
    ///     4. scale_accumulate (residual)                         [done]
    ///     5. rms_norm (post-mixer)                               [done]
    ///     6. dense FFN: DMMV gate + DMMV up → swiglu → DMMV down [swiglu done]
    ///     7. scale_accumulate (residual)                         [done]
    ///   8. final rms_norm → DMMV lm_head (Q6_K) → argmax
    pub fn decodeStep(self: *CudaForward) !u32 {
        _ = self;
        return error.NotImplemented; // M1 TODO
    }

    /// M2 — batched prefill over the prompt (layer-major DMMV→GEMM; batched SSM
    /// selective scan). See Effort 20 for the prefill-specific kernel plan.
    pub fn prefill(self: *CudaForward, n_tokens: usize) !void {
        _ = self;
        _ = n_tokens;
        return error.NotImplemented; // M2 TODO
    }
};

// Reference the backend surface so the scaffold documents (and the compiler
// checks, once wired) the dependency on the CUDA modules.
comptime {
    if (gpu.is_cuda) {
        _ = gpu.backend; // ../cuda/device.zig
        _ = gpu.buffer_mod; // ../cuda/buffer.zig
        _ = gpu.pipeline_mod; // ../cuda/pipeline.zig
        _ = gpu.command_mod; // ../cuda/command.zig
    }
}

// Kernel checklist — `src/shaders/cuda/kernels.cu`, dense-9B M1 set:
//   done : rms_norm, dmmv_q4k, dmmv_f32, dmmv_q8_0, swiglu, scale_accumulate,
//          sigmoid_scale_acc          (all validated on 4090 sm_89 + 5090 sm_120)
//   todo : dmmv_q6k (LM head), qk_norm, RoPE (partial/IMRoPE), kv_cache_write,
//          attention (naive softmax(QKᵀ)V → flash), ssm_conv1d,
//          ssm_delta_net (recurrent step + batched prefill), ssm_gated_norm,
//          argmax
