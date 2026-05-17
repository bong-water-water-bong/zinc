//! Pure Zig T-CPU opcode implementations.
//! The modules exported here are clarity-first reference kernels used by the
//! CPU ring and by future cross-tier validation tests.
//! @section Inference Runtime
pub const rms_norm = @import("rms_norm.zig");
pub const swiglu = @import("swiglu.zig");
pub const argmax = @import("argmax.zig");
pub const dequant = @import("dequant.zig");
pub const embed = @import("embed.zig");
pub const lm_head = @import("lm_head.zig");
