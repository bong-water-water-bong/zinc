//! ZINC_RT reference-runtime module.
//! Exposes the M0 engine, IR, CPU ring, and scalar CPU kernels through one
//! importable package so `forward_zinc_rt` can exercise the runtime without
//! pulling the same files into multiple Zig modules.
//! @section Inference Runtime
pub const engine = @import("engine.zig");
pub const ir_op = @import("ir/op.zig");
pub const ir_graph = @import("ir/graph.zig");
pub const kmd = @import("kmd.zig");
pub const ring = @import("ring/mod.zig");
pub const cpu_ring = @import("ring/cpu.zig");
pub const umq = @import("ring/umq.zig");
pub const kfd = @import("ring/kfd.zig");
pub const cs = @import("ring/cs.zig");
pub const pm4_packet = @import("ring/packet.zig");
pub const kernels = @import("isa/cpu_zig/mod.zig");
pub const fast_pool = @import("fast_pool.zig");

comptime {
    _ = @import("tests/ir_smoke.zig");
    _ = @import("fast_pool.zig");
}
