//! ZINC_RT IR verifier entrypoints.
//! Verification stays separate from graph construction so future passes can
//! reject malformed shapes and bindings before any backend executes them.
//! @section Decode Planning
const graph_mod = @import("graph.zig");

pub fn graph(ir: *const graph_mod.Graph) !void {
    try ir.verify();
}
