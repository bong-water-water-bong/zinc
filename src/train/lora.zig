//! LoRA (Low-Rank Adaptation) adapter structures and forward pass integration.
//!
//! LoRA freezes pre-trained weights W and injects a learnable low-rank
//! decomposition into the forward pass:
//!     W'·x = W·x + (alpha / r) * B·A·x
//!
//! Only A and B receive gradients — typically ~0.1% of parameters trainable.
//! This keeps the memory footprint small enough for Strix Halo's 32 GB shared
//! memory.
//!
//! @section Training Integration
//! During forward: after each DMMV dispatch for a LoRA-targeted weight, the
//! trainer dispatches lora_fwd (computes B·A·x, scales, accumulates).
//!
//! During backward: the trainer dispatches lora_bwd (gradients for A and B).
//! Then adamw_update applies the parameter update.

const std = @import("std");

/// Maximum number of LoRA adapters supported in one training session.
pub const MAX_ADAPTERS: u32 = 128;

/// Configuration for one LoRA adapter attached to a specific weight tensor.
pub const LoraConfig = struct {
    /// Human-readable name matching the GGUF tensor (e.g. "blk.0.attn_q.weight").
    weight_name: []const u8,
    /// LoRA rank r (number of low-rank dimensions, typical 8-64).
    rank: u32,
    /// LoRA scaling alpha (typical: 16, 32, 64).
    alpha: f32,
    /// Layer index this adapter targets.
    layer_index: u32,
    /// Projection index within the layer:
    ///   0 = attn_q, 1 = attn_k, 2 = attn_v, 3 = attn_o,
    ///   4 = ffn_gate, 5 = ffn_up, 6 = ffn_down
    projection_index: u32,
};

/// Runtime GPU-side state for one LoRA adapter.
pub const LoraAdapter = struct {
    /// Human-readable name.
    name: []const u8,
    /// LoRA rank.
    rank: u32,
    /// Scaling factor: alpha / rank.
    scale: f32,
    /// Input dimension (hidden_dim, inter_dim, etc.).
    in_dim: u32,
    /// Output dimension.
    out_dim: u32,
    /// Layer index.
    layer_index: u32,
    /// Projection index.
    projection_index: u32,

    // ── GPU buffer indices (into ForwardState's buffer table) ────────────

    /// A matrix buffer index: [out_dim × r] F16 — task-specific, random init.
    a_buf: u32,
    /// B matrix buffer index: [r × in_dim] F16 — zero-initialized.
    b_buf: u32,

    /// M (Adam first moment, B params): [r × in_dim] F32
    m_b_buf: u32,
    /// V (Adam second moment, B params): [r × in_dim] F32
    v_b_buf: u32,
    /// M (Adam first moment, A params): [out_dim × r] F32
    m_a_buf: u32,
    /// V (Adam second moment, A params): [out_dim × r] F32
    v_a_buf: u32,

    /// Gradient buffer for A: [out_dim × r] F32
    grad_a_buf: u32,
    /// Gradient buffer for B: [r × in_dim] F32
    grad_b_buf: u32,

    /// Hidden state saved for this adapter's input (x).
    hidden_buf: u32,

    /// Number of F32 elements in the A matrix.
    a_params: u32,
    /// Number of F32 elements in the B matrix.
    b_params: u32,
    /// Total trainable parameters.
    total_params: u32,
};

/// Descriptor of where to inject a LoRA forward pass in the compute graph.
pub const LoraInjectionPoint = struct {
    /// Which layer in the model (0-based).
    layer_index: u32,
    /// Which projection in the layer.
    projection_index: u32,
    /// Index into the training session's adapter table.
    adapter_index: u32,
};
