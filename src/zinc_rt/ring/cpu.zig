//! T-CPU ring backend.
//! Walks packet batches and executes pure Zig kernels as the validation oracle.
const ring = @import("mod.zig");
const kernels = @import("../isa/cpu_zig/mod.zig");

pub const CpuRing = struct {
    pub fn init() CpuRing {
        return .{};
    }

    pub fn deinit(_: *CpuRing) void {}

    pub fn submitAndWait(_: *CpuRing, batch: ring.PacketBatch) !void {
        for (batch.packets) |packet| {
            switch (packet) {
                .embed => |params| try kernels.embed.run(params),
                .rms_norm => |params| try kernels.rms_norm.run(params),
                .lm_head => |params| try kernels.lm_head.run(params),
                .swiglu => |params| try kernels.swiglu.run(params),
                .argmax => |params| try kernels.argmax.run(params),
                .barrier => {},
            }
        }
    }
};
