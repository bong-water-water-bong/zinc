//! Correctness + throughput spike for the TQ2_0_g128 ternary Bonsai GEMV
//! Vulkan port (shaders/dmmv_tq2_bonsai.comp), the Vulkan side of the
//! ROCm-vs-Vulkan dual-backend prove-out. Validates GPU dispatch output
//! against the CPU reference in tq2_ref.zig at a small shape, then
//! benchmarks throughput at 6912x6912 -- the same shape as the ROCm
//! reference kernel (bonsai_tq2_gemv.hip, 48.5 GB/s, see
//! docs/gpu/HANDOFF-ROCM-GPU.md) -- so the two numbers are directly
//! comparable.
//! Run via `zig build bench-tq2-gemv -Doptimize=ReleaseFast`.
const std = @import("std");
const vk = @import("vulkan/vk.zig");
const instance_mod = @import("vulkan/instance.zig");
const Instance = instance_mod.Instance;
const CommandPool = @import("vulkan/command.zig").CommandPool;
const CommandBuffer = @import("vulkan/command.zig").CommandBuffer;
const Buffer = @import("vulkan/buffer.zig").Buffer;
const buffer_mod = @import("vulkan/buffer.zig");
const gpu_detect = @import("vulkan/gpu_detect.zig");
const Pipeline = @import("vulkan/pipeline.zig").Pipeline;
const dmmv_mod = @import("compute/dmmv.zig");
const DmmvDispatch = dmmv_mod.DmmvDispatch;
const tq2_ref = @import("tq2_ref.zig");

const log = std.log.scoped(.bench_tq2_gemv);

const BLOCK_WEIGHTS = tq2_ref.block_weights;
const BLOCK_BYTES = tq2_ref.block_bytes;
const ROWS_PER_WG: u32 = 8;

const TimestampTimer = struct {
    pool: vk.c.VkQueryPool,
    period_ns: f64,
    device: vk.c.VkDevice,

    fn init(instance: *const Instance) !TimestampTimer {
        const pool_info = vk.c.VkQueryPoolCreateInfo{
            .sType = vk.c.VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .queryType = vk.c.VK_QUERY_TYPE_TIMESTAMP,
            .queryCount = 2,
            .pipelineStatistics = 0,
        };
        var pool: vk.c.VkQueryPool = null;
        const result = vk.c.vkCreateQueryPool(instance.device, &pool_info, null, &pool);
        if (result != vk.c.VK_SUCCESS) return error.QueryPoolCreateFailed;
        return .{
            .pool = pool,
            .period_ns = @as(f64, instance.device_props.limits.timestampPeriod),
            .device = instance.device,
        };
    }

    fn deinit(self: *TimestampTimer) void {
        vk.c.vkDestroyQueryPool(self.device, self.pool, null);
        self.* = undefined;
    }

    fn writeStart(self: *const TimestampTimer, cmd: *const CommandBuffer) void {
        vk.c.vkCmdResetQueryPool(cmd.handle, self.pool, 0, 2);
        vk.c.vkCmdWriteTimestamp(cmd.handle, vk.c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, self.pool, 0);
    }

    fn writeEnd(self: *const TimestampTimer, cmd: *const CommandBuffer) void {
        vk.c.vkCmdWriteTimestamp(cmd.handle, vk.c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, self.pool, 1);
    }

    fn elapsedMs(self: *const TimestampTimer) !f64 {
        var timestamps: [2]u64 = undefined;
        const qr = vk.c.vkGetQueryPoolResults(
            self.device,
            self.pool,
            0,
            2,
            2 * @sizeOf(u64),
            &timestamps,
            @sizeOf(u64),
            vk.c.VK_QUERY_RESULT_64_BIT | vk.c.VK_QUERY_RESULT_WAIT_BIT,
        );
        if (qr != vk.c.VK_SUCCESS) return error.QueryReadFailed;
        const elapsed_ns = @as(f64, @floatFromInt(timestamps[1] -| timestamps[0])) * self.period_ns;
        return elapsed_ns / 1e6;
    }
};

fn resolveShaderDir(allocator: std.mem.Allocator) ![]u8 {
    const candidates = [_][]const u8{
        "zig-out/share/zinc/shaders",
        "share/zinc/shaders",
    };
    for (candidates) |candidate| {
        std.fs.cwd().access(candidate, .{}) catch continue;
        return allocator.dupe(u8, candidate);
    }
    const exe_path = try std.fs.selfExePathAlloc(allocator);
    defer allocator.free(exe_path);
    const exe_dir = std.fs.path.dirname(exe_path) orelse ".";
    const derived = try std.fs.path.join(allocator, &.{ exe_dir, "..", "share", "zinc", "shaders" });
    errdefer allocator.free(derived);
    std.fs.cwd().access(derived, .{}) catch return error.ShaderDirNotFound;
    return derived;
}

fn createStorageBuffer(instance: *const Instance, size: vk.c.VkDeviceSize) !Buffer {
    return Buffer.initDeviceLocal(instance, size, vk.c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
}

fn initDeviceLocalWithBytes(instance: *const Instance, cmd_pool: *const CommandPool, bytes: []const u8) !Buffer {
    var staging = try Buffer.initStaging(instance, bytes.len);
    defer staging.deinit();
    staging.upload(bytes);
    var device_buf = try createStorageBuffer(instance, bytes.len);
    errdefer device_buf.deinit();
    try buffer_mod.copyBuffer(instance, cmd_pool.handle, &staging, &device_buf, bytes.len);
    return device_buf;
}

fn allocDescSet(device: vk.c.VkDevice, pool: vk.c.VkDescriptorPool, layout: vk.c.VkDescriptorSetLayout) !vk.c.VkDescriptorSet {
    const alloc_info = vk.c.VkDescriptorSetAllocateInfo{
        .sType = vk.c.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .pNext = null,
        .descriptorPool = pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &layout,
    };
    var ds: vk.c.VkDescriptorSet = null;
    const result = vk.c.vkAllocateDescriptorSets(device, &alloc_info, &ds);
    if (result != vk.c.VK_SUCCESS) return error.DescriptorSetAllocFailed;
    return ds;
}

fn writeDescSet(comptime N: usize, device: vk.c.VkDevice, ds: vk.c.VkDescriptorSet, infos: *[N]vk.c.VkDescriptorBufferInfo) void {
    var writes: [N]vk.c.VkWriteDescriptorSet = undefined;
    for (0..N) |i| {
        writes[i] = .{
            .sType = vk.c.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = null,
            .dstSet = ds,
            .dstBinding = @intCast(i),
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk.c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pImageInfo = null,
            .pBufferInfo = &infos[i],
            .pTexelBufferView = null,
        };
    }
    vk.c.vkUpdateDescriptorSets(device, N, &writes, 0, null);
}

fn tq2RowBytes(k: u32) usize {
    return @as(usize, k / BLOCK_WEIGHTS) * BLOCK_BYTES;
}

fn tq2MatrixBytes(m: u32, k: u32) u64 {
    return @as(u64, m) * @as(u64, tq2RowBytes(k));
}

fn tq2BytesPerIter(m: u32, k: u32) u64 {
    return tq2MatrixBytes(m, k) + @as(u64, k) * @sizeOf(f16) + @as(u64, m) * @sizeOf(f32);
}

/// Deliberately exercises all 4 two-bit codes (including the reserved
/// 0b11 -> 0 case) across lanes/blocks/rows/salt, and varies the f16 scale
/// per (row, block), rather than filling with a degenerate constant.
fn fillTq2Weights(dst: []u8, m: u32, k: u32, salt: u32) void {
    const row_bytes = tq2RowBytes(k);
    const num_blocks = k / BLOCK_WEIGHTS;
    var row: u32 = 0;
    while (row < m) : (row += 1) {
        var block: u32 = 0;
        while (block < num_blocks) : (block += 1) {
            const off = row * row_bytes + block * BLOCK_BYTES;
            const scale: f16 = @floatCast(0.03125 * @as(f32, @floatFromInt(1 + (row + block) % 8)));
            std.mem.writeInt(u16, dst[off..][0..2], @bitCast(scale), .little);
            var byte_idx: u32 = 0;
            while (byte_idx < 32) : (byte_idx += 1) {
                var packed_byte: u8 = 0;
                var j: u32 = 0;
                while (j < 4) : (j += 1) {
                    const code: u8 = @intCast((byte_idx + j + block + row + salt) % 4);
                    packed_byte |= code << @intCast(j * 2);
                }
                dst[off + 2 + byte_idx] = packed_byte;
            }
        }
    }
}

fn fillActivations(dst: []f16, salt: u32) void {
    for (dst, 0..) |*v, i| {
        const lane: f32 = @floatFromInt(((i + salt) % 11) + 1);
        v.* = @floatCast(lane * 0.0625);
    }
}

const DmmvSlot = struct {
    weights: Buffer,
    x: Buffer,
    y: Buffer,
    descriptor_set: ?vk.c.VkDescriptorSet = null,

    fn deinit(self: *DmmvSlot) void {
        self.weights.deinit();
        self.x.deinit();
        self.y.deinit();
        self.* = undefined;
    }
};

fn recordRepeated(
    cmd: *CommandBuffer,
    timer: *const TimestampTimer,
    pipeline: *const Pipeline,
    push_desc_fn: ?instance_mod.PushDescriptorFn,
    slots: []const DmmvSlot,
    push: dmmv_mod.DmmvPushConstants,
    wg_x: u32,
    iterations: u32,
) void {
    timer.writeStart(cmd);
    var i: u32 = 0;
    while (i < iterations) : (i += 1) {
        const slot = slots[i % slots.len];
        if (pipeline.uses_push_descriptors) {
            const infos = [3]vk.c.VkDescriptorBufferInfo{
                .{ .buffer = slot.weights.handle, .offset = 0, .range = slot.weights.size },
                .{ .buffer = slot.x.handle, .offset = 0, .range = slot.x.size },
                .{ .buffer = slot.y.handle, .offset = 0, .range = slot.y.size },
            };
            cmd.pushDescAndDispatch(pipeline, push_desc_fn, infos[0..], std.mem.asBytes(&push), wg_x, 1, 1);
        } else {
            const ds = slot.descriptor_set orelse unreachable;
            cmd.dispatchWithPush(pipeline, ds, std.mem.asBytes(&push), wg_x, 1, 1);
        }
    }
    timer.writeEnd(cmd);
}

fn runCorrectness(
    allocator: std.mem.Allocator,
    instance: *const Instance,
    cmd_pool: *const CommandPool,
    cmd: *CommandBuffer,
    dispatch: *const DmmvDispatch,
    push_desc_fn: ?instance_mod.PushDescriptorFn,
) !void {
    const m: u32 = 16;
    const k: u32 = 256;
    const pipeline = dispatch.pipeline_tq2_bonsai orelse return error.ShaderNotLoaded;

    const weight_blob = try allocator.alloc(u8, @intCast(tq2MatrixBytes(m, k)));
    defer allocator.free(weight_blob);
    fillTq2Weights(weight_blob, m, k, 42);

    const act_host = try allocator.alloc(f16, k);
    defer allocator.free(act_host);
    fillActivations(act_host, 7);

    var weights_buf = try initDeviceLocalWithBytes(instance, cmd_pool, weight_blob);
    defer weights_buf.deinit();
    var x_buf = try initDeviceLocalWithBytes(instance, cmd_pool, std.mem.sliceAsBytes(act_host));
    defer x_buf.deinit();
    var y_buf = try Buffer.initHostVisibleStorage(instance, @as(vk.c.VkDeviceSize, m) * @sizeOf(f32));
    defer y_buf.deinit();

    var descriptor_set: ?vk.c.VkDescriptorSet = null;
    if (!pipeline.uses_push_descriptors) {
        const ds = try allocDescSet(instance.device, dispatch.descriptor_pool, pipeline.descriptor_set_layout);
        var infos = [3]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = weights_buf.handle, .offset = 0, .range = weights_buf.size },
            .{ .buffer = x_buf.handle, .offset = 0, .range = x_buf.size },
            .{ .buffer = y_buf.handle, .offset = 0, .range = y_buf.size },
        };
        writeDescSet(3, instance.device, ds, &infos);
        descriptor_set = ds;
    }

    const push = dmmv_mod.DmmvPushConstants{ .M = m, .K = k, .a_offset = 0, .x_offset = 0, .y_offset = 0, .acc_mode = 0 };
    const wg_x = (m + ROWS_PER_WG - 1) / ROWS_PER_WG;

    try cmd.reset();
    try cmd.beginOneTime();
    if (pipeline.uses_push_descriptors) {
        const infos = [3]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = weights_buf.handle, .offset = 0, .range = weights_buf.size },
            .{ .buffer = x_buf.handle, .offset = 0, .range = x_buf.size },
            .{ .buffer = y_buf.handle, .offset = 0, .range = y_buf.size },
        };
        cmd.pushDescAndDispatch(&pipeline, push_desc_fn, infos[0..], std.mem.asBytes(&push), wg_x, 1, 1);
    } else {
        cmd.dispatchWithPush(&pipeline, descriptor_set.?, std.mem.asBytes(&push), wg_x, 1, 1);
    }
    try cmd.end();
    try cmd.submitAndWait(instance.compute_queue);

    const y_gpu = @as([*]const f32, @alignCast(@ptrCast(y_buf.mapped.?)))[0..m];

    const y_ref = try allocator.alloc(f32, m);
    defer allocator.free(y_ref);
    tq2_ref.gemv(weight_blob, act_host, y_ref, m, k);

    var max_abs_err: f32 = 0.0;
    var fail_count: u32 = 0;
    for (0..m) |i| {
        const err = @abs(y_gpu[i] - y_ref[i]);
        const tol = @max(@as(f32, 1e-3), 1e-3 * @abs(y_ref[i]));
        if (err > tol) fail_count += 1;
        max_abs_err = @max(max_abs_err, err);
    }
    if (fail_count > 0) {
        log.err("TQ2 correctness FAILED: {d}/{d} rows out of tolerance, max_abs_err={d:.6}", .{ fail_count, m, max_abs_err });
        return error.CorrectnessFailed;
    }
    log.info("TQ2 correctness PASSED: M={d} K={d}, max_abs_err={d:.6}", .{ m, k, max_abs_err });
}

fn runThroughput(
    allocator: std.mem.Allocator,
    instance: *const Instance,
    cmd_pool: *const CommandPool,
    cmd: *CommandBuffer,
    timer: *const TimestampTimer,
    dispatch: *const DmmvDispatch,
    gpu_config: *const gpu_detect.GpuConfig,
    push_desc_fn: ?instance_mod.PushDescriptorFn,
) !void {
    const m: u32 = 6912;
    const k: u32 = 6912;
    const iterations: u32 = 200;
    const warmup: u32 = 25;
    const working_set: usize = 16;

    const pipeline = dispatch.pipeline_tq2_bonsai orelse return error.ShaderNotLoaded;

    const weight_blob = try allocator.alloc(u8, @intCast(tq2MatrixBytes(m, k)));
    defer allocator.free(weight_blob);
    const act_host = try allocator.alloc(f16, k);
    defer allocator.free(act_host);

    const slots = try allocator.alloc(DmmvSlot, working_set);
    defer allocator.free(slots);
    var init_count: usize = 0;
    errdefer for (slots[0..init_count]) |*slot| slot.deinit();

    while (init_count < working_set) : (init_count += 1) {
        fillTq2Weights(weight_blob, m, k, @intCast(init_count * 13));
        fillActivations(act_host, @intCast(init_count * 17));

        slots[init_count].weights = try initDeviceLocalWithBytes(instance, cmd_pool, weight_blob);
        slots[init_count].x = try initDeviceLocalWithBytes(instance, cmd_pool, std.mem.sliceAsBytes(act_host));
        slots[init_count].y = try createStorageBuffer(instance, @as(vk.c.VkDeviceSize, m) * @sizeOf(f32));
        if (!pipeline.uses_push_descriptors) {
            const ds = try allocDescSet(instance.device, dispatch.descriptor_pool, pipeline.descriptor_set_layout);
            slots[init_count].descriptor_set = ds;
            var infos = [3]vk.c.VkDescriptorBufferInfo{
                .{ .buffer = slots[init_count].weights.handle, .offset = 0, .range = slots[init_count].weights.size },
                .{ .buffer = slots[init_count].x.handle, .offset = 0, .range = slots[init_count].x.size },
                .{ .buffer = slots[init_count].y.handle, .offset = 0, .range = slots[init_count].y.size },
            };
            writeDescSet(3, instance.device, ds, &infos);
        }
    }
    defer for (slots[0..working_set]) |*slot| slot.deinit();

    const wg_x = (m + ROWS_PER_WG - 1) / ROWS_PER_WG;
    const push = dmmv_mod.DmmvPushConstants{ .M = m, .K = k, .a_offset = 0, .x_offset = 0, .y_offset = 0, .acc_mode = 0 };

    try cmd.reset();
    try cmd.beginOneTime();
    recordRepeated(cmd, timer, &pipeline, push_desc_fn, slots, push, wg_x, warmup);
    try cmd.end();
    try cmd.submitAndWait(instance.compute_queue);

    try cmd.reset();
    try cmd.beginOneTime();
    recordRepeated(cmd, timer, &pipeline, push_desc_fn, slots, push, wg_x, iterations);
    try cmd.end();
    const wall_start = std.time.nanoTimestamp();
    try cmd.submitAndWait(instance.compute_queue);
    const wall_end = std.time.nanoTimestamp();
    const gpu_ms = try timer.elapsedMs();
    const wall_ms = @as(f64, @floatFromInt(wall_end - wall_start)) / 1_000_000.0;

    const bytes_per_iter = tq2BytesPerIter(m, k);
    const eff_gbps = (@as(f64, @floatFromInt(bytes_per_iter)) * @as(f64, @floatFromInt(iterations))) / (gpu_ms / 1000.0) / 1_000_000_000.0;
    const utilization = if (gpu_config.bandwidth_gbps > 0) eff_gbps / @as(f64, @floatFromInt(gpu_config.bandwidth_gbps)) * 100.0 else 0.0;

    log.info("tq2_bonsai M={d} K={d}: gpu={d:.3} ms/iter wall={d:.3} ms/iter | {d:.1} GB/s | {d:.1}% of peak | {d} B/iter", .{
        m,
        k,
        gpu_ms / @as(f64, @floatFromInt(iterations)),
        wall_ms / @as(f64, @floatFromInt(iterations)),
        eff_gbps,
        utilization,
        bytes_per_iter,
    });
    log.info("ROCm reference (bonsai_tq2_gemv.hip, same shape family): 48.5 GB/s (docs/gpu/HANDOFF-ROCM-GPU.md)", .{});
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        const leaked = gpa.deinit();
        if (leaked == .leak) log.err("memory leak detected", .{});
    }
    const allocator = gpa.allocator();

    const shader_dir = try resolveShaderDir(allocator);
    defer allocator.free(shader_dir);

    var instance = try Instance.init(allocator, instance_mod.auto_select_device_index);
    defer instance.deinit();
    const gpu_cfg = gpu_detect.detect(&instance);

    var cmd_pool = try CommandPool.init(&instance);
    defer cmd_pool.deinit();
    var cmd = try CommandBuffer.init(&instance, &cmd_pool);
    defer cmd.deinit(&cmd_pool);
    var timer = try TimestampTimer.init(&instance);
    defer timer.deinit();

    var dmmv = try DmmvDispatch.init(&instance, &gpu_cfg, shader_dir, 6912, allocator);
    defer dmmv.deinit();

    log.info("GPU: {s} | BW {d} GB/s | shader_dir={s}", .{ gpu_cfg.nameSlice(), gpu_cfg.bandwidth_gbps, shader_dir });

    try runCorrectness(allocator, &instance, &cmd_pool, &cmd, &dmmv, instance.push_descriptor_fn);
    try runThroughput(allocator, &instance, &cmd_pool, &cmd, &timer, &dmmv, &gpu_cfg, instance.push_descriptor_fn);
}
