//! Training loop orchestrator for LoRA fine-tuning on-device.
//!
//! @section Training Pipeline
//! The trainer owns:
//!   1. GPU state for all LoRA adapters (A, B, optimizer moments, gradients)
//!   2. Training data buffer (tokenized text)
//!   3. Cross-entropy loss computation shaders
//!   4. AdamW optimizer update shaders
//!   5. Backward pass shaders for LoRA adapters
//!
//! Each training step:
//!   a) Forward pass via existing ZINC compute graph (frozen weights)
//!      + injected lora_fwd dispatches after each targeted DMMV
//!   b) Cross-entropy loss shader (softmax + NLL)
//!   c) Backward pass: dlogits → gradient through LoRA adapters
//!   d) AdamW update on A and B
//!   e) Advance to next batch

const std = @import("std");
const vk = @import("../vulkan/vk.zig");
const Instance = @import("../vulkan/instance.zig").Instance;
const buffer_mod = @import("../vulkan/buffer.zig");
const Buffer = buffer_mod.Buffer;
const Pipeline = @import("../vulkan/pipeline.zig").Pipeline;
const pipeline_mod = @import("../vulkan/pipeline.zig");
const GpuConfig = @import("../vulkan/gpu_detect.zig").GpuConfig;
const CommandPool = @import("../vulkan/command.zig").CommandPool;
const CommandBuffer = @import("../vulkan/command.zig").CommandBuffer;
const InferenceEngine = @import("../compute/forward.zig").InferenceEngine;
const DecodeState = @import("../compute/forward.zig").DecodeState;
const lora_mod = @import("lora.zig");
const LoraConfig = lora_mod.LoraConfig;
const LoraAdapter = lora_mod.LoraAdapter;

const log = std.log.scoped(.trainer);

// ── Push constant structs (must match shader layouts exactly) ────────────

pub const LoraFwdPush = extern struct {
    M: u32,
    K: u32,
    R: u32,
    scale: f32,
    a_off: u32,
    b_off: u32,
    x_off: u32,
    y_off: u32,
};

pub const CrossEntropyPush = extern struct {
    N: u32,
    V: u32,
    logits_off: u32,
    targets_off: u32,
    probs_off: u32,
    loss_off: u32,
};

pub const LoraBwdPush = extern struct {
    M: u32,
    K: u32,
    R: u32,
    scale: f32,
    a_off: u32,
    b_off: u32,
    x_off: u32,
    dy_off: u32,
    grad_a_off: u32,
    grad_b_off: u32,
};

pub const AdamWUpdatePush = extern struct {
    N: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    step: u32,
    params_off: u32,
    grad_off: u32,
    m_off: u32,
    v_off: u32,
};

// ── Training configuration ───────────────────────────────────────────────

pub const TrainingConfig = struct {
    learning_rate: f32 = 2e-4,
    beta1: f32 = 0.9,
    beta2: f32 = 0.999,
    eps: f32 = 1e-8,
    weight_decay: f32 = 0.01,
    batch_size: u32 = 64,
    max_steps: u32 = 1000,
    log_interval: u32 = 10,
    save_interval: u32 = 0,
    data_path: ?[]const u8 = null,
};

// ── Training pipelines ───────────────────────────────────────────────────

pub const TrainingPipelines = struct {
    lora_fwd: Pipeline,
    cross_entropy: Pipeline,
    lora_bwd: Pipeline,
    adamw_update: Pipeline,

    pub fn load(instance: *const Instance, shader_dir: []const u8, allocator: std.mem.Allocator, options: pipeline_mod.PipelineOptions) !TrainingPipelines {
        var path_buf: [512]u8 = undefined;

        const lora_fwd = try pipeline_mod.createFromSpirvWithOptions(
            instance,
            std.fmt.bufPrint(&path_buf, "{s}/lora_fwd.spv", .{shader_dir}) catch unreachable,
            4, @sizeOf(LoraFwdPush), &.{}, options, allocator,
        );
        const cross_entropy = try pipeline_mod.createFromSpirvWithOptions(
            instance,
            std.fmt.bufPrint(&path_buf, "{s}/cross_entropy.spv", .{shader_dir}) catch unreachable,
            4, @sizeOf(CrossEntropyPush), &.{}, options, allocator,
        );
        const lora_bwd = try pipeline_mod.createFromSpirvWithOptions(
            instance,
            std.fmt.bufPrint(&path_buf, "{s}/lora_bwd.spv", .{shader_dir}) catch unreachable,
            6, @sizeOf(LoraBwdPush), &.{}, options, allocator,
        );
        const adamw_update = try pipeline_mod.createFromSpirvWithOptions(
            instance,
            std.fmt.bufPrint(&path_buf, "{s}/adamw_update.spv", .{shader_dir}) catch unreachable,
            4, @sizeOf(AdamWUpdatePush), &.{}, options, allocator,
        );

        return .{ .lora_fwd = lora_fwd, .cross_entropy = cross_entropy, .lora_bwd = lora_bwd, .adamw_update = adamw_update };
    }
};

// ── Buffer allocation helper ─────────────────────────────────────────────

fn addBuffer(bufs: *std.ArrayList(Buffer), alc: std.mem.Allocator, instance: *const Instance, size: u64, usage: u64, mem_props: u64) !u32 {
    var buf = try Buffer.init(instance, size, @as(u32, @intCast(usage)), @as(u32, @intCast(mem_props)));
    errdefer buf.deinit();
    const idx = @as(u32, @intCast(bufs.items.len));
    try bufs.append(alc, buf);
    return idx;
}

// ── Training session ─────────────────────────────────────────────────────

pub const Trainer = struct {
    allocator: std.mem.Allocator,
    instance: *const Instance,
    gpu_config: *const GpuConfig,
    pipelines: TrainingPipelines,
    cmd_pool: CommandPool,
    descriptor_pool: vk.c.VkDescriptorPool,

    adapters: []LoraAdapter,
    config: TrainingConfig,
    current_step: u32,
    moving_loss: f32,
    train_data: []const u32,

    // All GPU buffers are stored in this list. Individual fields below
    // store the index into this list.
    buffers: std.ArrayList(Buffer),

    targets_buf: u32,
    probs_buf: u32,
    loss_buf: u32,
    total_loss_buf: u32,

    data_len: u32,
    data_pos: u32,

    // Cached descriptor set layouts for quick reference
    layout_cross_entropy: vk.c.VkDescriptorSetLayout,
    layout_adamw_update: vk.c.VkDescriptorSetLayout,
    layout_lora_bwd: vk.c.VkDescriptorSetLayout,
    layout_lora_fwd: vk.c.VkDescriptorSetLayout,

    pub fn init(
        allocator: std.mem.Allocator,
        instance: *const Instance,
        gpu_config: *const GpuConfig,
        shader_dir: []const u8,
        adapter_configs: []const LoraConfig,
        model_hidden_dim: u32,
        model_vocab_size: u32,
        prompt_tokens: []const u32,
        config: TrainingConfig,
    ) !Trainer {
        const use_push = instance.push_descriptor_fn != null;
        const pip_options = pipeline_mod.PipelineOptions{ .push_descriptors = use_push };
        const pipelines = try TrainingPipelines.load(instance, shader_dir, allocator, pip_options);
        var cmd_pool = try CommandPool.init(instance);
        errdefer cmd_pool.deinit();

        // Create descriptor pool (only needed for non-push-descriptor path)
        var descriptor_pool: vk.c.VkDescriptorPool = null;
        if (!use_push) {
            const max_sets: u32 = @intCast(1024);
            const pool_size = vk.c.VkDescriptorPoolSize{
                .type = vk.c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = max_sets * 6,
            };
            const pool_info = vk.c.VkDescriptorPoolCreateInfo{
                .sType = vk.c.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                .pNext = null,
                .flags = vk.c.VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
                .maxSets = max_sets,
                .poolSizeCount = 1,
                .pPoolSizes = &pool_size,
            };
            const result = vk.c.vkCreateDescriptorPool(instance.device, &pool_info, null, &descriptor_pool);
            if (result != vk.c.VK_SUCCESS) return error.DescriptorPoolCreateFailed;
        }

        const n_adapter = adapter_configs.len;
        const adapters = try allocator.alloc(LoraAdapter, n_adapter);
        errdefer allocator.free(adapters);

        var buffers = std.ArrayList(Buffer){};
        errdefer { for (buffers.items) |*b| b.deinit(); buffers.deinit(allocator); }

        const su = vk.c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                    vk.c.VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                    vk.c.VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        const dm = vk.c.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        const hm = vk.c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                    vk.c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

        for (adapter_configs, 0..) |cfg, i| {
            const out_dim = model_hidden_dim;
            const in_dim = model_hidden_dim;
            const rank = cfg.rank;
            const scale = cfg.alpha / @as(f32, @floatFromInt(cfg.rank));

            const a_buf = try addBuffer(&buffers, allocator, instance, out_dim * rank * 2, su, dm);
            const b_buf = try addBuffer(&buffers, allocator, instance, rank * in_dim * 2, su, dm);
            const m_b_buf = try addBuffer(&buffers, allocator, instance, rank * in_dim * 4, su, dm);
            const v_b_buf = try addBuffer(&buffers, allocator, instance, rank * in_dim * 4, su, dm);
            const m_a_buf = try addBuffer(&buffers, allocator, instance, out_dim * rank * 4, su, dm);
            const v_a_buf = try addBuffer(&buffers, allocator, instance, out_dim * rank * 4, su, dm);
            const grad_a_buf = try addBuffer(&buffers, allocator, instance, out_dim * rank * 4, su, dm);
            const grad_b_buf = try addBuffer(&buffers, allocator, instance, rank * in_dim * 4, su, dm);
            const hidden_buf = try addBuffer(&buffers, allocator, instance, in_dim * 4, su, dm);

            adapters[i] = .{
                .name = cfg.weight_name,
                .rank = rank,
                .scale = scale,
                .in_dim = in_dim,
                .out_dim = out_dim,
                .layer_index = cfg.layer_index,
                .projection_index = cfg.projection_index,
                .a_buf = a_buf,
                .b_buf = b_buf,
                .m_b_buf = m_b_buf,
                .v_b_buf = v_b_buf,
                .m_a_buf = m_a_buf,
                .v_a_buf = v_a_buf,
                .grad_a_buf = grad_a_buf,
                .grad_b_buf = grad_b_buf,
                .hidden_buf = hidden_buf,
                .a_params = out_dim * rank,
                .b_params = rank * in_dim,
                .total_params = out_dim * rank + rank * in_dim,
            };
        }

        // Training data buffer + transfer-dst
        const data_buf = try addBuffer(&buffers, allocator, instance, prompt_tokens.len * 4,
            su | vk.c.VK_BUFFER_USAGE_TRANSFER_DST_BIT, dm);

        // Cross-entropy scratch buffers
        const batch_size = config.batch_size;
        const targets_buf = try addBuffer(&buffers, allocator, instance, batch_size * 4,
            su | vk.c.VK_BUFFER_USAGE_TRANSFER_DST_BIT, dm);
        const probs_buf = try addBuffer(&buffers, allocator, instance, batch_size * model_vocab_size * 4, su, dm);
        const loss_buf = try addBuffer(&buffers, allocator, instance, batch_size * 4, su, dm);
        const total_loss_buf = try addBuffer(&buffers, allocator, instance, 4, su | vk.c.VK_BUFFER_USAGE_TRANSFER_SRC_BIT, hm);

        // Upload training data via staging buffer
        {
            var staging = try Buffer.initStaging(instance, prompt_tokens.len * 4);
            defer staging.deinit();
            staging.upload(std.mem.sliceAsBytes(prompt_tokens));
            try buffer_mod.copyBuffer(instance, cmd_pool.handle, &staging, &buffers.items[data_buf], prompt_tokens.len * 4);
        }

        // Initialize A (random) and B (zero) on GPU
        if (adapters.len > 0) {
            var rng = std.Random.DefaultPrng.init(@intCast(std.time.milliTimestamp()));
            for (adapters) |*ad| {
                {
                    var staging = try Buffer.initStaging(instance, ad.a_params * 2);
                    defer staging.deinit();
                    const mapped = staging.mapped.?;
                    const ptr = @as([*]u16, @ptrCast(@alignCast(mapped)));
                    for (0..ad.a_params) |j| {
                        const rv: f32 = rng.random().float(f32) * 0.04 - 0.02;
                        ptr[j] = floatToF16(rv);
                    }
                    try buffer_mod.copyBuffer(instance, cmd_pool.handle, &staging, &buffers.items[ad.a_buf], ad.a_params * 2);
                }
                {
                    var staging = try Buffer.initStaging(instance, ad.b_params * 2);
                    defer staging.deinit();
                    @memset(staging.mapped.?[0..ad.b_params * 2], 0);
                    try buffer_mod.copyBuffer(instance, cmd_pool.handle, &staging, &buffers.items[ad.b_buf], ad.b_params * 2);
                }
                {
                    const zero_buf_indices = [_]u32{ ad.m_b_buf, ad.v_b_buf, ad.m_a_buf, ad.v_a_buf, ad.grad_a_buf, ad.grad_b_buf };
                    for (zero_buf_indices) |zbi| {
                        const sz: u64 = buffers.items[zbi].size;
                        var staging = try Buffer.initStaging(instance, sz);
                        defer staging.deinit();
                        @memset(staging.mapped.?[0..sz], 0);
                        try buffer_mod.copyBuffer(instance, cmd_pool.handle, &staging, &buffers.items[zbi], sz);
                    }
                }
            }
        }

        return .{
            .allocator = allocator,
            .instance = instance,
            .gpu_config = gpu_config,
            .pipelines = pipelines,
            .cmd_pool = cmd_pool,
            .descriptor_pool = descriptor_pool,
            .adapters = adapters,
            .config = config,
            .current_step = 0,
            .moving_loss = 0.0,
            .train_data = prompt_tokens,
            .buffers = buffers,
            .targets_buf = targets_buf,
            .probs_buf = probs_buf,
            .loss_buf = loss_buf,
            .total_loss_buf = total_loss_buf,
            .data_len = @intCast(prompt_tokens.len),
            .data_pos = 0,
            .layout_cross_entropy = pipelines.cross_entropy.descriptor_set_layout,
            .layout_adamw_update = pipelines.adamw_update.descriptor_set_layout,
            .layout_lora_bwd = pipelines.lora_bwd.descriptor_set_layout,
            .layout_lora_fwd = pipelines.lora_fwd.descriptor_set_layout,
        };
    }

    pub fn deinit(self: *Trainer) void {
        if (self.descriptor_pool != null) {
            vk.c.vkDestroyDescriptorPool(self.instance.device, self.descriptor_pool, null);
        }
        for (self.buffers.items) |*b| b.deinit();
        self.buffers.deinit(self.allocator);
        self.allocator.free(self.adapters);
        self.pipelines.lora_fwd.deinit();
        self.pipelines.cross_entropy.deinit();
        self.pipelines.lora_bwd.deinit();
        self.pipelines.adamw_update.deinit();
        self.cmd_pool.deinit();
    }

    pub fn step(self: *Trainer, instance: *const Instance, engine: *InferenceEngine) !f32 {
        const use_push = self.instance.push_descriptor_fn != null;

        // ── 1. Set up LoRA injection config on engine ─────────────────
        engine.lora_active = (self.adapters.len > 0 and engine.lora_fwd_pipeline != null);
        engine.lora_injection_count = 0;
        if (engine.lora_active) {
            for (self.adapters) |*ad| {
                if (engine.lora_injection_count >= engine.lora_injections.len) break;
                engine.lora_injections[engine.lora_injection_count] = .{
                    .layer = ad.layer_index,
                    .proj_idx = ad.projection_index,
                    .a_buf = self.buffers.items[ad.a_buf].handle,
                    .b_buf = self.buffers.items[ad.b_buf].handle,
                    .a_size = self.buffers.items[ad.a_buf].size,
                    .b_size = self.buffers.items[ad.b_buf].size,
                    .rank = ad.rank,
                    .scale = ad.scale,
                };
                engine.lora_injection_count += 1;
            }
            log.debug("LoRA active: {d} injection points", .{engine.lora_injection_count});
        }

        engine.capture_lora_attn_input = true;
        engine.capture_lora_ffn_input = true;

        const batch_size = self.config.batch_size;
        var batch_tokens: [256]u32 = undefined;
        const n_tokens = @min(batch_size, self.data_len - self.data_pos);
        for (0..n_tokens) |i| {
            batch_tokens[i] = self.train_data[self.data_pos + i];
        }

        // Forward-pass each token through the model (builds KV cache, produces logits)
        var state = DecodeState.init(self.allocator);
        defer state.deinit();
        for (0..n_tokens) |i| {
            try engine.decodeStep(&state, batch_tokens[i], i == n_tokens - 1);
        }

        // Check ZINC_CAPTURE_LORA_HIDDEN twice: at engine init time and at step.
        // If capture buffers were not allocated, log a warning on first step.
        if (engine.lora_capture_n_layers == 0 and self.current_step == 0) {
            log.warn("LoRA hidden capture not enabled. Set ZINC_CAPTURE_LORA_HIDDEN=1 and restart.", .{});
        }

        // ── 2. Cross-entropy loss dispatch on engine's logits_buf ──────
        {
            const pip = &self.pipelines.cross_entropy;
            const push_data = std.mem.asBytes(&CrossEntropyPush{
                .N = self.config.batch_size,
                .V = engine.model.config.vocab_size,
                .logits_off = 0,
                .targets_off = 0,
                .probs_off = 0,
                .loss_off = 0,
            });
            if (use_push) {
                var cmd_buf = try CommandBuffer.init(instance, &self.cmd_pool);
                defer cmd_buf.deinit(&self.cmd_pool);
                try cmd_buf.begin();
                const infos = [4]vk.c.VkDescriptorBufferInfo{
                    .{ .buffer = engine.logits_buf.handle, .offset = 0, .range = engine.logits_buf.size },
                    .{ .buffer = self.buffers.items[self.targets_buf].handle, .offset = 0, .range = self.buffers.items[self.targets_buf].size },
                    .{ .buffer = self.buffers.items[self.probs_buf].handle, .offset = 0, .range = self.buffers.items[self.probs_buf].size },
                    .{ .buffer = self.buffers.items[self.loss_buf].handle, .offset = 0, .range = self.buffers.items[self.loss_buf].size },
                };
                cmd_buf.pushDescAndDispatch(pip, self.instance.push_descriptor_fn, infos[0..], push_data, self.config.batch_size, 1, 1);
                try cmd_buf.end();
                try cmd_buf.submit(instance.compute_queue);
                try cmd_buf.waitForCompletion();
            } else {
                const ds = try self.allocDescSet(self.layout_cross_entropy);
                self.writeDescSet4(ds,
                    engine.logits_buf.handle, engine.logits_buf.size,
                    self.buffers.items[self.targets_buf].handle, self.buffers.items[self.targets_buf].size,
                    self.buffers.items[self.probs_buf].handle, self.buffers.items[self.probs_buf].size,
                    self.buffers.items[self.loss_buf].handle, self.buffers.items[self.loss_buf].size);
                var cmd_buf = try CommandBuffer.init(instance, &self.cmd_pool);
                defer cmd_buf.deinit(&self.cmd_pool);
                try cmd_buf.begin();
                cmd_buf.dispatchWithPush(pip, ds, push_data, self.config.batch_size, 1, 1);
                try cmd_buf.end();
                try cmd_buf.submit(instance.compute_queue);
                try cmd_buf.waitForCompletion();
            }
        }

        // ── 3. Copy captured hidden states into adapter hidden_bufs ────
        for (self.adapters) |*ad| {
            const src_buf = if (ad.projection_index <= 3)
                engine.lora_attn_input_capture_buf
            else
                engine.lora_ffn_input_capture_buf;
            if (src_buf.handle == null or engine.lora_capture_n_layers == 0) continue;
            const slot_off = @as(vk.c.VkDeviceSize, ad.layer_index) *
                @as(vk.c.VkDeviceSize, ad.in_dim) * @sizeOf(f32);
            const copy_size = @as(vk.c.VkDeviceSize, ad.in_dim) * @sizeOf(f32);
            if (slot_off + copy_size > src_buf.size) continue;
            const region = vk.c.VkBufferCopy{
                .srcOffset = slot_off,
                .dstOffset = 0,
                .size = copy_size,
            };
            var cmd_buf = try CommandBuffer.init(instance, &self.cmd_pool);
            defer cmd_buf.deinit(&self.cmd_pool);
            try cmd_buf.begin();
            vk.c.vkCmdCopyBuffer(
                cmd_buf.handle,
                src_buf.handle,
                self.buffers.items[ad.hidden_buf].handle,
                1,
                &region,
            );
            try cmd_buf.end();
            try cmd_buf.submit(instance.compute_queue);
            try cmd_buf.waitForCompletion();
        }

        // ── 4. Per-adapter backward + update ───────────────────────────
        for (self.adapters) |*ad| {
            const a_buf = self.buffers.items[ad.a_buf];
            const b_buf = self.buffers.items[ad.b_buf];
            const hidden_buf = self.buffers.items[ad.hidden_buf];
            const grad_a_buf = self.buffers.items[ad.grad_a_buf];
            const grad_b_buf = self.buffers.items[ad.grad_b_buf];
            const m_b_buf = self.buffers.items[ad.m_b_buf];
            const v_b_buf = self.buffers.items[ad.v_b_buf];
            const m_a_buf = self.buffers.items[ad.m_a_buf];
            const v_a_buf = self.buffers.items[ad.v_a_buf];

            // lora_bwd: 6 bindings — 0:A, 1:B, 2:x(captured), 3:dy, 4:grad_a, 5:grad_b
            {
                const pip = &self.pipelines.lora_bwd;
                const push_data = std.mem.asBytes(&LoraBwdPush{
                    .M = ad.out_dim,
                    .K = ad.in_dim,
                    .R = ad.rank,
                    .scale = ad.scale,
                    .a_off = 0,
                    .b_off = 0,
                    .x_off = 0,
                    .dy_off = 0,
                    .grad_a_off = 0,
                    .grad_b_off = 0,
                });
                if (use_push) {
                    var cmd_buf = try CommandBuffer.init(instance, &self.cmd_pool);
                    defer cmd_buf.deinit(&self.cmd_pool);
                    try cmd_buf.begin();
                    const infos = [6]vk.c.VkDescriptorBufferInfo{
                        .{ .buffer = a_buf.handle, .offset = 0, .range = a_buf.size },
                        .{ .buffer = b_buf.handle, .offset = 0, .range = b_buf.size },
                        .{ .buffer = hidden_buf.handle, .offset = 0, .range = hidden_buf.size },
                        .{ .buffer = hidden_buf.handle, .offset = 0, .range = hidden_buf.size },
                        .{ .buffer = grad_a_buf.handle, .offset = 0, .range = grad_a_buf.size },
                        .{ .buffer = grad_b_buf.handle, .offset = 0, .range = grad_b_buf.size },
                    };
                    cmd_buf.pushDescAndDispatch(pip, self.instance.push_descriptor_fn, infos[0..], push_data, ad.rank, 1, 1);
                    try cmd_buf.end();
                    try cmd_buf.submit(instance.compute_queue);
                    try cmd_buf.waitForCompletion();
                } else {
                    const ds = try self.allocDescSet(self.layout_lora_bwd);
                    self.writeDescSet6(ds,
                        a_buf.handle, a_buf.size,
                        b_buf.handle, b_buf.size,
                        hidden_buf.handle, hidden_buf.size,
                        hidden_buf.handle, hidden_buf.size,
                        grad_a_buf.handle, grad_a_buf.size,
                        grad_b_buf.handle, grad_b_buf.size);
                    var cmd_buf = try CommandBuffer.init(instance, &self.cmd_pool);
                    defer cmd_buf.deinit(&self.cmd_pool);
                    try cmd_buf.begin();
                    cmd_buf.dispatchWithPush(pip, ds, push_data, ad.rank, 1, 1);
                    try cmd_buf.end();
                    try cmd_buf.submit(instance.compute_queue);
                    try cmd_buf.waitForCompletion();
                }
            }

            // adamw_update on B params
            {
                const pip = &self.pipelines.adamw_update;
                const push_data = std.mem.asBytes(&AdamWUpdatePush{
                    .N = ad.b_params,
                    .lr = self.config.learning_rate,
                    .beta1 = self.config.beta1,
                    .beta2 = self.config.beta2,
                    .eps = self.config.eps,
                    .weight_decay = self.config.weight_decay,
                    .step = self.current_step + 1,
                    .params_off = 0,
                    .grad_off = 0,
                    .m_off = 0,
                    .v_off = 0,
                });
                if (use_push) {
                    var cmd_buf = try CommandBuffer.init(instance, &self.cmd_pool);
                    defer cmd_buf.deinit(&self.cmd_pool);
                    try cmd_buf.begin();
                    const infos = [4]vk.c.VkDescriptorBufferInfo{
                        .{ .buffer = b_buf.handle, .offset = 0, .range = b_buf.size },
                        .{ .buffer = grad_b_buf.handle, .offset = 0, .range = grad_b_buf.size },
                        .{ .buffer = m_b_buf.handle, .offset = 0, .range = m_b_buf.size },
                        .{ .buffer = v_b_buf.handle, .offset = 0, .range = v_b_buf.size },
                    };
                    cmd_buf.pushDescAndDispatch(pip, self.instance.push_descriptor_fn, infos[0..], push_data, (ad.b_params + 63) / 64, 1, 1);
                    try cmd_buf.end();
                    try cmd_buf.submit(instance.compute_queue);
                    try cmd_buf.waitForCompletion();
                } else {
                    const ds = try self.allocDescSet(self.layout_adamw_update);
                    self.writeDescSet4(ds,
                        b_buf.handle, b_buf.size,
                        grad_b_buf.handle, grad_b_buf.size,
                        m_b_buf.handle, m_b_buf.size,
                        v_b_buf.handle, v_b_buf.size);
                    var cmd_buf = try CommandBuffer.init(instance, &self.cmd_pool);
                    defer cmd_buf.deinit(&self.cmd_pool);
                    try cmd_buf.begin();
                    cmd_buf.dispatchWithPush(pip, ds, push_data, (ad.b_params + 63) / 64, 1, 1);
                    try cmd_buf.end();
                    try cmd_buf.submit(instance.compute_queue);
                    try cmd_buf.waitForCompletion();
                }
            }

            // adamw_update on A params
            {
                const pip = &self.pipelines.adamw_update;
                const push_data = std.mem.asBytes(&AdamWUpdatePush{
                    .N = ad.a_params,
                    .lr = self.config.learning_rate,
                    .beta1 = self.config.beta1,
                    .beta2 = self.config.beta2,
                    .eps = self.config.eps,
                    .weight_decay = self.config.weight_decay,
                    .step = self.current_step + 1,
                    .params_off = 0,
                    .grad_off = 0,
                    .m_off = 0,
                    .v_off = 0,
                });
                if (use_push) {
                    var cmd_buf = try CommandBuffer.init(instance, &self.cmd_pool);
                    defer cmd_buf.deinit(&self.cmd_pool);
                    try cmd_buf.begin();
                    const infos = [4]vk.c.VkDescriptorBufferInfo{
                        .{ .buffer = a_buf.handle, .offset = 0, .range = a_buf.size },
                        .{ .buffer = grad_a_buf.handle, .offset = 0, .range = grad_a_buf.size },
                        .{ .buffer = m_a_buf.handle, .offset = 0, .range = m_a_buf.size },
                        .{ .buffer = v_a_buf.handle, .offset = 0, .range = v_a_buf.size },
                    };
                    cmd_buf.pushDescAndDispatch(pip, self.instance.push_descriptor_fn, infos[0..], push_data, (ad.a_params + 63) / 64, 1, 1);
                    try cmd_buf.end();
                    try cmd_buf.submit(instance.compute_queue);
                    try cmd_buf.waitForCompletion();
                } else {
                    const ds = try self.allocDescSet(self.layout_adamw_update);
                    self.writeDescSet4(ds,
                        a_buf.handle, a_buf.size,
                        grad_a_buf.handle, grad_a_buf.size,
                        m_a_buf.handle, m_a_buf.size,
                        v_a_buf.handle, v_a_buf.size);
                    var cmd_buf = try CommandBuffer.init(instance, &self.cmd_pool);
                    defer cmd_buf.deinit(&self.cmd_pool);
                    try cmd_buf.begin();
                    cmd_buf.dispatchWithPush(pip, ds, push_data, (ad.a_params + 63) / 64, 1, 1);
                    try cmd_buf.end();
                    try cmd_buf.submit(instance.compute_queue);
                    try cmd_buf.waitForCompletion();
                }
            }
        }

        // ── 5. Read loss ──────────────────────────────────────────────
        var loss_val: f32 = 0.0;
        {
            const lb = &self.buffers.items[self.total_loss_buf];
            if (lb.mapped) |mapped| {
                loss_val = @as(*volatile f32, @ptrCast(@alignCast(mapped))).*;
            }
        }

        self.current_step += 1;
        self.moving_loss = if (self.current_step == 1) loss_val else 0.9 * self.moving_loss + 0.1 * loss_val;

        if (self.current_step % self.config.log_interval == 0) {
            log.info("step={d:>6}/{d}  loss={d:.6}  moving_loss={d:.6}", .{
                self.current_step, self.config.max_steps, loss_val, self.moving_loss,
            });
        }

        self.data_pos += batch_size;
        if (self.data_pos + batch_size > self.data_len) {
            self.data_pos = 0;
        }

        return loss_val;
    }
    pub fn saveCheckpoint(self: *const Trainer, path: []const u8) !void {
        // Download each adapter's A and B matrices from GPU to host and write as ZINC checkpoint binary.
        // Format: header (magic, n_adapters) + per-adapter { name_len, name, rank, in_dim, out_dim,
        // A_data_f16[out_dim*rank], B_data_f16[rank*in_dim] }
        const file = std.fs.cwd().createFile(path, .{}) catch |err| {
            log.err("Failed to create checkpoint '{s}': {s}", .{ path, @errorName(err) });
            return err;
        };
        defer file.close();

        const n_adapters: u32 = @intCast(self.adapters.len);
        const header = [2]u32{ 0x5A494E43, n_adapters }; // "ZINC" magic
        try file.writeAll(std.mem.sliceAsBytes(&header));

        for (self.adapters) |ad| {
            // Download A matrix (F16)
            const a_buf = &self.buffers.items[ad.a_buf];
            const a_bytes = ad.a_params * 2; // F16 = 2 bytes per element
            var staging_a = try buffer_mod.Buffer.initStaging(self.instance, a_bytes);
            defer staging_a.deinit();
            try buffer_mod.copyBuffer(self.instance, self.cmd_pool.handle, a_buf, &staging_a, a_bytes);
            // Wait for transfer
            var trans_cmd = try CommandBuffer.init(self.instance, &self.cmd_pool);
            defer trans_cmd.deinit(&self.cmd_pool);
            try trans_cmd.begin();
            try trans_cmd.end();
            try trans_cmd.submit(self.instance.compute_queue);
            try trans_cmd.waitForCompletion();

            // Download B matrix (F16)
            const b_buf = &self.buffers.items[ad.b_buf];
            const b_bytes = ad.b_params * 2;
            var staging_b = try buffer_mod.Buffer.initStaging(self.instance, b_bytes);
            defer staging_b.deinit();
            try buffer_mod.copyBuffer(self.instance, self.cmd_pool.handle, b_buf, &staging_b, b_bytes);
            var trans_cmd2 = try CommandBuffer.init(self.instance, &self.cmd_pool);
            defer trans_cmd2.deinit(&self.cmd_pool);
            try trans_cmd2.begin();
            try trans_cmd2.end();
            try trans_cmd2.submit(self.instance.compute_queue);
            try trans_cmd2.waitForCompletion();

            const name_bytes = ad.name;
            const name_len: u32 = @intCast(name_bytes.len);
            try file.writeAll(std.mem.asBytes(&name_len));
            try file.writeAll(name_bytes);
            try file.writeAll(std.mem.asBytes(&ad.rank));
            try file.writeAll(std.mem.asBytes(&ad.in_dim));
            try file.writeAll(std.mem.asBytes(&ad.out_dim));
            try file.writeAll(staging_a.mapped.?[0..a_bytes]);
            try file.writeAll(staging_b.mapped.?[0..b_bytes]);
        }

        log.info("Checkpoint saved to {s}: {d} adapters", .{ path, n_adapters });
    }

    fn allocDescSet(self: *Trainer, layout: vk.c.VkDescriptorSetLayout) !vk.c.VkDescriptorSet {
        const alloc_info = vk.c.VkDescriptorSetAllocateInfo{
            .sType = vk.c.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = null,
            .descriptorPool = self.descriptor_pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &layout,
        };
        var ds: vk.c.VkDescriptorSet = null;
        const result = vk.c.vkAllocateDescriptorSets(self.instance.device, &alloc_info, &ds);
        if (result != vk.c.VK_SUCCESS) return error.DescriptorSetAllocFailed;
        return ds;
    }

    fn writeDescSet4(self: *Trainer, ds: vk.c.VkDescriptorSet,
        buf0: vk.c.VkBuffer, size0: vk.c.VkDeviceSize,
        buf1: vk.c.VkBuffer, size1: vk.c.VkDeviceSize,
        buf2: vk.c.VkBuffer, size2: vk.c.VkDeviceSize,
        buf3: vk.c.VkBuffer, size3: vk.c.VkDeviceSize,
    ) void {
        var buffer_infos = [4]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = buf0, .offset = 0, .range = size0 },
            .{ .buffer = buf1, .offset = 0, .range = size1 },
            .{ .buffer = buf2, .offset = 0, .range = size2 },
            .{ .buffer = buf3, .offset = 0, .range = size3 },
        };
        var writes: [4]vk.c.VkWriteDescriptorSet = undefined;
        for (&writes, 0..) |*w, i| {
            w.* = .{
                .sType = vk.c.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = null,
                .dstSet = ds,
                .dstBinding = @intCast(i),
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk.c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pImageInfo = null,
                .pBufferInfo = &buffer_infos[i],
                .pTexelBufferView = null,
            };
        }
        vk.c.vkUpdateDescriptorSets(self.instance.device, 4, &writes, 0, null);
    }

    fn writeDescSet6(self: *Trainer, ds: vk.c.VkDescriptorSet,
        buf0: vk.c.VkBuffer, size0: vk.c.VkDeviceSize,
        buf1: vk.c.VkBuffer, size1: vk.c.VkDeviceSize,
        buf2: vk.c.VkBuffer, size2: vk.c.VkDeviceSize,
        buf3: vk.c.VkBuffer, size3: vk.c.VkDeviceSize,
        buf4: vk.c.VkBuffer, size4: vk.c.VkDeviceSize,
        buf5: vk.c.VkBuffer, size5: vk.c.VkDeviceSize,
    ) void {
        var buffer_infos = [6]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = buf0, .offset = 0, .range = size0 },
            .{ .buffer = buf1, .offset = 0, .range = size1 },
            .{ .buffer = buf2, .offset = 0, .range = size2 },
            .{ .buffer = buf3, .offset = 0, .range = size3 },
            .{ .buffer = buf4, .offset = 0, .range = size4 },
            .{ .buffer = buf5, .offset = 0, .range = size5 },
        };
        var writes: [6]vk.c.VkWriteDescriptorSet = undefined;
        for (&writes, 0..) |*w, i| {
            w.* = .{
                .sType = vk.c.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = null,
                .dstSet = ds,
                .dstBinding = @intCast(i),
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk.c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pImageInfo = null,
                .pBufferInfo = &buffer_infos[i],
                .pTexelBufferView = null,
            };
        }
        vk.c.vkUpdateDescriptorSets(self.instance.device, 6, &writes, 0, null);
    }
};

fn floatToF16(v: f32) u16 {
    const bits = @as(u32, @bitCast(v));
    const sign = @as(u16, @truncate((bits >> 16) & 0x8000));
    const exp = (bits >> 23) & 0xFF;
    const mant = bits & 0x7FFFFF;
    if (exp == 0xFF) return sign | 0x7C00 | @as(u16, @truncate(mant >> 13));
    if (exp == 0) return sign;
    const exp_f16 = @as(i32, @intCast(exp)) - 127 + 15;
    if (exp_f16 >= 0x1F) return sign | 0x7C00;
    if (exp_f16 <= 0) return sign;
    return sign | @as(u16, @intCast(exp_f16)) << 10 | @as(u16, @truncate(mant >> 13));
}
