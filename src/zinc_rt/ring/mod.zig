//! Backend-neutral packet batch types for ZINC_RT rings.
//! These packet structs are the handoff point between lowered IR and concrete
//! ring implementations such as T-CPU, T2 UMQ, and future direct tiers.
//! @section Inference Runtime
const rms_norm = @import("../isa/cpu_zig/rms_norm.zig");
const swiglu = @import("../isa/cpu_zig/swiglu.zig");
const argmax = @import("../isa/cpu_zig/argmax.zig");
const embed = @import("../isa/cpu_zig/embed.zig");
const lm_head = @import("../isa/cpu_zig/lm_head.zig");

pub const Packet = union(enum) {
    embed: embed.Params,
    rms_norm: rms_norm.Params,
    lm_head: lm_head.Params,
    swiglu: swiglu.Params,
    argmax: argmax.Params,
    barrier: void,
};

pub const PacketBatch = struct {
    packets: []const Packet,
};
