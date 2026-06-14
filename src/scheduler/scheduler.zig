//! Continuous-batching scheduler groundwork for concurrent inference requests.
//! @section Scheduler
//! Today this module owns request slot accounting and state collection only.
//! The HTTP serving hot path still serializes generation behind
//! ServerState.generation_mutex; the batched prefill/decode dispatch loop is
//! not wired yet.
const std = @import("std");
const Request = @import("request.zig").Request;
const RequestState = @import("request.zig").RequestState;
const GenerationParams = @import("request.zig").GenerationParams;

const log = std.log.scoped(.scheduler);

/// Fixed-capacity pool of request slots used to track concurrent inference requests.
/// Each slot holds at most one active `Request`; slots are reused once released.
pub const Scheduler = struct {
    /// Active requests indexed by slot ID.
    slots: []?Request,
    /// Maximum number of concurrent requests.
    max_parallel: u32,
    /// Next request ID counter.
    next_id: u64,
    /// Allocator for owned resources.
    allocator: std.mem.Allocator,

    /// Initialize the scheduler with a fixed number of concurrent request slots.
    /// @param allocator Allocator for the slot array.
    /// @param max_parallel Maximum number of concurrent requests.
    /// @returns A Scheduler with all slots initially empty.
    pub fn init(allocator: std.mem.Allocator, max_parallel: u32) !Scheduler {
        const slots = try allocator.alloc(?Request, max_parallel);
        @memset(slots, null);
        log.info("Scheduler ready: {d} slots", .{max_parallel});
        return .{
            .slots = slots,
            .max_parallel = max_parallel,
            .next_id = 1,
            .allocator = allocator,
        };
    }

    /// Submit a new request and assign it to the first free slot.
    /// @param self Scheduler to submit to.
    /// @param prompt_tokens Tokenized prompt for the request.
    /// @param params Generation parameters (max_tokens, temperature, etc.).
    /// @returns The slot index that was assigned; pass this value to `release` when the request completes.
    /// @note Returns `error.AllSlotsBusy` if every slot is occupied.
    pub fn submit(self: *Scheduler, prompt_tokens: []const u32, params: GenerationParams) !u32 {
        // Find a free slot
        for (self.slots, 0..) |*slot, i| {
            if (slot.* == null) {
                const id = self.next_id;
                self.next_id += 1;
                var req = Request.init(self.allocator, id, prompt_tokens, params);
                req.slot_id = @intCast(i);
                slot.* = req;
                log.info("Request {d} assigned to slot {d} ({d} prompt tokens)", .{ id, i, prompt_tokens.len });
                return @intCast(i);
            }
        }
        return error.AllSlotsBusy;
    }

    /// Check if all slots are occupied.
    /// @param self Scheduler to query.
    /// @returns True if every slot holds an active request.
    pub fn isFull(self: *const Scheduler) bool {
        return self.activeCount() >= self.max_parallel;
    }

    /// Get the number of active (non-null) requests.
    /// @param self Scheduler to query.
    /// @returns Count of occupied slots.
    pub fn activeCount(self: *const Scheduler) u32 {
        var count: u32 = 0;
        for (self.slots) |slot| {
            if (slot != null) count += 1;
        }
        return count;
    }

    /// Transition a live slot through the request state machine.
    /// @param self Scheduler to query.
    /// @param slot_id Slot index to update.
    /// @param new_state Target request state.
    /// @returns error.InvalidSlot if the slot is out of range or empty.
    pub fn transition(self: *Scheduler, slot_id: u32, new_state: RequestState) !void {
        if (slot_id >= self.slots.len) return error.InvalidSlot;
        if (self.slots[slot_id]) |*req| {
            try req.transition(new_state);
            return;
        }
        return error.InvalidSlot;
    }

    /// Collect slot IDs whose request currently has `state`.
    /// @param self Scheduler to query.
    /// @param state Request state to match.
    /// @param out Caller-owned scratch buffer for slot IDs.
    /// @returns A slice of `out` containing the collected slot IDs.
    pub fn collectByState(self: *const Scheduler, state: RequestState, out: []u32) []u32 {
        var count: usize = 0;
        for (self.slots, 0..) |slot, i| {
            if (count == out.len) break;
            if (slot) |req| {
                if (req.state == state) {
                    out[count] = @intCast(i);
                    count += 1;
                }
            }
        }
        return out[0..count];
    }

    /// Return slot IDs of requests waiting for prefill admission.
    /// @param self Scheduler to query.
    /// @param out Caller-owned scratch buffer for slot IDs.
    /// @returns A slice of `out` containing pending prefill slot IDs.
    pub fn pendingPrefill(self: *const Scheduler, out: []u32) []u32 {
        return self.collectByState(.pending, out);
    }

    /// Return slot IDs of requests currently in decode.
    /// @param self Scheduler to query.
    /// @param out Caller-owned scratch buffer for slot IDs.
    /// @returns A slice of `out` containing decoding slot IDs.
    pub fn activeDecoding(self: *const Scheduler, out: []u32) []u32 {
        return self.collectByState(.decoding, out);
    }

    /// Release a completed or cancelled request's slot, freeing its resources.
    /// @param self Scheduler to release from.
    /// @param slot_id Slot index to free (the value returned by `submit`).
    /// @note Silently does nothing if `slot_id` is out of range or the slot is already empty.
    pub fn release(self: *Scheduler, slot_id: u32) void {
        if (slot_id < self.slots.len) {
            if (self.slots[slot_id]) |*req| {
                req.deinit();
                self.slots[slot_id] = null;
                log.info("Released slot {d}", .{slot_id});
            }
        }
    }

    /// Tear down all active requests and free the slot array.
    /// @param self Scheduler to destroy.
    pub fn deinit(self: *Scheduler) void {
        for (self.slots) |*slot| {
            if (slot.*) |*req| req.deinit();
        }
        self.allocator.free(self.slots);
    }
};

test "Scheduler submit and release" {
    const allocator = std.testing.allocator;
    var sched = try Scheduler.init(allocator, 4);
    defer sched.deinit();

    try std.testing.expectEqual(@as(u32, 0), sched.activeCount());

    const slot0 = try sched.submit(&.{ 1, 2, 3 }, .{});
    try std.testing.expectEqual(@as(u32, 0), slot0);
    try std.testing.expectEqual(@as(u32, 1), sched.activeCount());

    const slot1 = try sched.submit(&.{ 4, 5 }, .{});
    try std.testing.expectEqual(@as(u32, 1), slot1);
    try std.testing.expectEqual(@as(u32, 2), sched.activeCount());

    sched.release(0);
    try std.testing.expectEqual(@as(u32, 1), sched.activeCount());
}

test "Scheduler full" {
    const allocator = std.testing.allocator;
    var sched = try Scheduler.init(allocator, 2);
    defer sched.deinit();

    _ = try sched.submit(&.{1}, .{});
    _ = try sched.submit(&.{2}, .{});
    try std.testing.expectError(error.AllSlotsBusy, sched.submit(&.{3}, .{}));
}

test "Scheduler isFull" {
    const allocator = std.testing.allocator;
    var sched = try Scheduler.init(allocator, 2);
    defer sched.deinit();

    try std.testing.expect(!sched.isFull());
    _ = try sched.submit(&.{1}, .{});
    try std.testing.expect(!sched.isFull());
    _ = try sched.submit(&.{2}, .{});
    try std.testing.expect(sched.isFull());
    sched.release(0);
    try std.testing.expect(!sched.isFull());
}

test "Scheduler release and reuse slot" {
    const allocator = std.testing.allocator;
    var sched = try Scheduler.init(allocator, 1);
    defer sched.deinit();

    const s1 = try sched.submit(&.{10}, .{});
    try std.testing.expectEqual(@as(u32, 0), s1);
    sched.release(s1);

    // Same slot should be reusable
    const s2 = try sched.submit(&.{20}, .{});
    try std.testing.expectEqual(@as(u32, 0), s2);
    sched.release(s2);
}

test "Scheduler request IDs increment" {
    const allocator = std.testing.allocator;
    var sched = try Scheduler.init(allocator, 4);
    defer sched.deinit();

    _ = try sched.submit(&.{1}, .{});
    _ = try sched.submit(&.{2}, .{});

    // Request IDs should be 0 and 1 (or some incrementing sequence)
    // Check slots have different request objects
    try std.testing.expect(sched.slots[0] != null);
    try std.testing.expect(sched.slots[1] != null);
    try std.testing.expect(sched.slots[0].?.id != sched.slots[1].?.id);
}

test "Scheduler collects pending prefill and active decoding slots" {
    const allocator = std.testing.allocator;
    var sched = try Scheduler.init(allocator, 4);
    defer sched.deinit();

    const prefill_slot = try sched.submit(&.{1}, .{});
    const decode_slot = try sched.submit(&.{2}, .{});
    const other_slot = try sched.submit(&.{3}, .{});

    try sched.transition(decode_slot, .prefilling);
    try sched.transition(decode_slot, .decoding);
    try sched.transition(other_slot, .cancelled);

    var scratch: [4]u32 = undefined;
    const pending = sched.pendingPrefill(&scratch);
    try std.testing.expectEqual(@as(usize, 1), pending.len);
    try std.testing.expectEqual(prefill_slot, pending[0]);

    const decoding = sched.activeDecoding(&scratch);
    try std.testing.expectEqual(@as(usize, 1), decoding.len);
    try std.testing.expectEqual(decode_slot, decoding[0]);
}

test "Scheduler state collection respects scratch capacity" {
    const allocator = std.testing.allocator;
    var sched = try Scheduler.init(allocator, 3);
    defer sched.deinit();

    _ = try sched.submit(&.{1}, .{});
    _ = try sched.submit(&.{2}, .{});
    _ = try sched.submit(&.{3}, .{});

    var scratch: [2]u32 = undefined;
    const pending = sched.pendingPrefill(&scratch);
    try std.testing.expectEqual(@as(usize, 2), pending.len);
    try std.testing.expectEqual(@as(u32, 0), pending[0]);
    try std.testing.expectEqual(@as(u32, 1), pending[1]);
}
