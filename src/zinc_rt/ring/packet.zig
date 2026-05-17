//! PM4 packet builder shared by direct AMD ZINC_RT tiers.
//!
//! This is intentionally syntax-only: it does not know about model shapes or
//! IR op semantics. M1 lowering hands already-decided register writes and
//! dispatch dimensions to this builder, then T2/T1 copy the resulting dwords
//! into their user queue rings.
//! @section Inference Runtime
const std = @import("std");

pub const Error = error{OutOfSpace};

pub const Opcode = enum(u8) {
    nop = 0x10,
    dispatch_direct = 0x15,
    write_data = 0x37,
    wait_reg_mem = 0x3c,
    copy_data = 0x40,
    release_mem = 0x49,
    acquire_mem = 0x58,
    set_context_reg = 0x69,
    set_sh_reg = 0x76,
    set_uconfig_reg = 0x79,
};

// SH register offsets, expressed as `(byte_addr - 0xB000) >> 2`.
// The direct CS path programs these before a raw DISPATCH_DIRECT.
pub const sh_reg_num_thread_x: u32 = 0x207;
pub const sh_reg_pgm_lo: u32 = 0x20c;
pub const sh_reg_pgm_rsrc1: u32 = 0x212;
pub const sh_reg_resource_limits: u32 = 0x215;
pub const sh_reg_pgm_rsrc3: u32 = 0x228;
pub const compute_user_data_0: u32 = 0x240;
pub const dispatch_initiator_compute: u32 = 5;

pub const PacketBuilder = struct {
    words: []u32,
    len: usize = 0,

    pub fn init(words: []u32) PacketBuilder {
        return .{ .words = words };
    }

    pub fn reset(self: *PacketBuilder) void {
        self.len = 0;
    }

    pub fn written(self: *const PacketBuilder) []const u32 {
        return self.words[0..self.len];
    }

    pub fn writeNop(self: *PacketBuilder, payload_dwords: u32) Error!void {
        const body_dwords = @max(payload_dwords, 1);
        const start = try self.reservePacket(body_dwords);
        for (0..body_dwords) |i| self.words[start + 1 + i] = 0;
        self.publishPkt3Header(start, .nop, body_dwords);
    }

    pub fn setShReg(self: *PacketBuilder, reg_offset: u32, values: []const u32) Error!void {
        if (values.len == 0) return;
        const body_dwords: u32 = @intCast(values.len + 1);
        const start = try self.reservePacket(body_dwords);
        self.words[start + 1] = reg_offset;
        for (values, 0..) |value, i| self.words[start + 2 + i] = value;
        self.publishPkt3Header(start, .set_sh_reg, body_dwords);
    }

    pub fn setShRegOne(self: *PacketBuilder, reg_offset: u32, value: u32) Error!void {
        const values = [_]u32{value};
        try self.setShReg(reg_offset, &values);
    }

    pub fn setUserData64(self: *PacketBuilder, slot: u32, value: u64) Error!void {
        const values = [_]u32{ lo32(value), hi32(value) };
        try self.setShReg(compute_user_data_0 + slot, &values);
    }

    pub fn dispatchDirect(self: *PacketBuilder, dim_x: u32, dim_y: u32, dim_z: u32) Error!void {
        try self.dispatchDirectInitiator(dim_x, dim_y, dim_z, 0);
    }

    pub fn dispatchDirectInitiator(
        self: *PacketBuilder,
        dim_x: u32,
        dim_y: u32,
        dim_z: u32,
        dispatch_initiator: u32,
    ) Error!void {
        const start = try self.reservePacket(4);
        self.words[start + 1] = dim_x;
        self.words[start + 2] = dim_y;
        self.words[start + 3] = dim_z;
        self.words[start + 4] = dispatch_initiator;
        self.publishPkt3Header(start, .dispatch_direct, 4);
    }

    pub fn releaseMemSignal(self: *PacketBuilder, gpu_addr: u64, value: u64) Error!void {
        // Minimal fence signal packet shape used by the bring-up gate. The
        // event/data selectors are intentionally conservative placeholders;
        // kernel validation of executable streams happens in the UMQ smoke path.
        const start = try self.reservePacket(6);
        self.words[start + 1] = 0;
        self.words[start + 2] = 0;
        self.words[start + 3] = lo32(gpu_addr);
        self.words[start + 4] = hi32(gpu_addr);
        self.words[start + 5] = lo32(value);
        self.words[start + 6] = hi32(value);
        self.publishPkt3Header(start, .release_mem, 6);
    }

    pub fn writeData64(self: *PacketBuilder, gpu_addr: u64, value: u64) Error!void {
        // PKT3_WRITE_DATA, dst_sel=5 (memory async/direct), WR_CONFIRM=1,
        // engine_sel=0 (ME). This is the simplest in-band memory scribble for
        // validating CS-submitted fence/output-ring writes before real kernels.
        const dst_sel_memory_async: u32 = 5 << 8;
        const wr_confirm: u32 = 1 << 20;
        const engine_sel_me: u32 = 0 << 30;
        const start = try self.reservePacket(5);
        self.words[start + 1] = dst_sel_memory_async | wr_confirm | engine_sel_me;
        self.words[start + 2] = lo32(gpu_addr);
        self.words[start + 3] = hi32(gpu_addr);
        self.words[start + 4] = lo32(value);
        self.words[start + 5] = hi32(value);
        self.publishPkt3Header(start, .write_data, 5);
    }

    pub fn copyData32(self: *PacketBuilder, src_gpu_addr: u64, dst_gpu_addr: u64) Error!void {
        // PKT3_COPY_DATA, src_sel=1 (memory), dst_sel=5 (memory),
        // count_sel=0 (32 bits), WR_CONFIRM=1. This is the smallest
        // command-processor dataflow primitive available before shader
        // dispatch lowering is wired.
        const src_sel_memory: u32 = 1;
        const dst_sel_memory: u32 = 5 << 8;
        const count_sel_32: u32 = 0 << 16;
        const wr_confirm: u32 = 1 << 20;
        const start = try self.reservePacket(5);
        self.words[start + 1] = src_sel_memory | dst_sel_memory | count_sel_32 | wr_confirm;
        self.words[start + 2] = lo32(src_gpu_addr);
        self.words[start + 3] = hi32(src_gpu_addr);
        self.words[start + 4] = lo32(dst_gpu_addr);
        self.words[start + 5] = hi32(dst_gpu_addr);
        self.publishPkt3Header(start, .copy_data, 5);
    }

    pub fn padToAlignment(self: *PacketBuilder, dword_alignment: usize) Error!void {
        if (dword_alignment == 0) return;
        while (self.len % dword_alignment != 0) {
            var packet_dwords: usize = 2;
            while ((self.len + packet_dwords) % dword_alignment != 0) : (packet_dwords += 1) {}
            try self.writeNop(@intCast(packet_dwords - 1));
        }
    }

    fn reservePacket(self: *PacketBuilder, body_dwords: u32) Error!usize {
        std.debug.assert(body_dwords > 0);
        const total = @as(usize, body_dwords) + 1;
        if (self.len + total > self.words.len) return error.OutOfSpace;
        const start = self.len;
        self.words[start] = 0;
        self.len += total;
        return start;
    }

    fn publishPkt3Header(self: *PacketBuilder, start: usize, opcode: Opcode, body_dwords: u32) void {
        const count = body_dwords - 1;
        const header = (@as(u32, 3) << 30) | ((count & 0x3fff) << 16) | (@as(u32, @intFromEnum(opcode)) << 8);
        const header_ptr: *volatile u32 = @ptrCast(&self.words[start]);
        header_ptr.* = header;
    }
};

pub fn lo32(value: u64) u32 {
    return @truncate(value);
}

pub fn hi32(value: u64) u32 {
    return @truncate(value >> 32);
}

test "packet builder emits PM4 type-3 dispatch packet" {
    var words = [_]u32{0} ** 8;
    var builder = PacketBuilder.init(&words);
    try builder.dispatchDirect(7, 2, 1);

    const out = builder.written();
    try std.testing.expectEqual(@as(usize, 5), out.len);
    try std.testing.expectEqual((@as(u32, 3) << 30) | (@as(u32, 3) << 16) | (@as(u32, 0x15) << 8), out[0]);
    try std.testing.expectEqual(@as(u32, 7), out[1]);
    try std.testing.expectEqual(@as(u32, 2), out[2]);
    try std.testing.expectEqual(@as(u32, 1), out[3]);
    try std.testing.expectEqual(@as(u32, 0), out[4]);
}

test "packet builder emits contiguous user data register writes" {
    var words = [_]u32{0} ** 8;
    var builder = PacketBuilder.init(&words);
    try builder.setUserData64(4, 0x11223344_aabbccdd);

    const out = builder.written();
    try std.testing.expectEqual(@as(usize, 4), out.len);
    try std.testing.expectEqual((@as(u32, 3) << 30) | (@as(u32, 2) << 16) | (@as(u32, 0x76) << 8), out[0]);
    try std.testing.expectEqual(compute_user_data_0 + 4, out[1]);
    try std.testing.expectEqual(@as(u32, 0xaabbccdd), out[2]);
    try std.testing.expectEqual(@as(u32, 0x11223344), out[3]);
}

test "packet builder emits write-data memory signal" {
    var words = [_]u32{0} ** 8;
    var builder = PacketBuilder.init(&words);
    try builder.writeData64(0x11223344_aabbccdd, 0x01020304_05060708);

    const out = builder.written();
    try std.testing.expectEqual(@as(usize, 6), out.len);
    try std.testing.expectEqual((@as(u32, 3) << 30) | (@as(u32, 4) << 16) | (@as(u32, 0x37) << 8), out[0]);
    try std.testing.expectEqual(@as(u32, (5 << 8) | (1 << 20)), out[1]);
    try std.testing.expectEqual(@as(u32, 0xaabbccdd), out[2]);
    try std.testing.expectEqual(@as(u32, 0x11223344), out[3]);
    try std.testing.expectEqual(@as(u32, 0x05060708), out[4]);
    try std.testing.expectEqual(@as(u32, 0x01020304), out[5]);
}

test "packet builder emits copy-data memory to memory dword" {
    var words = [_]u32{0} ** 8;
    var builder = PacketBuilder.init(&words);
    try builder.copyData32(0x1000_0040, 0x1000_0080);

    const out = builder.written();
    try std.testing.expectEqual(@as(usize, 6), out.len);
    try std.testing.expectEqual((@as(u32, 3) << 30) | (@as(u32, 4) << 16) | (@as(u32, 0x40) << 8), out[0]);
    try std.testing.expectEqual(@as(u32, 1 | (5 << 8) | (1 << 20)), out[1]);
    try std.testing.expectEqual(@as(u32, 0x1000_0040), out[2]);
    try std.testing.expectEqual(@as(u32, 0), out[3]);
    try std.testing.expectEqual(@as(u32, 0x1000_0080), out[4]);
    try std.testing.expectEqual(@as(u32, 0), out[5]);
}

test "packet builder reports fixed buffer exhaustion" {
    var words = [_]u32{0} ** 2;
    var builder = PacketBuilder.init(&words);
    try std.testing.expectError(error.OutOfSpace, builder.dispatchDirect(1, 1, 1));
}

test "packet builder pads to dword alignment with valid NOP packets" {
    var words = [_]u32{0} ** 16;
    var builder = PacketBuilder.init(&words);
    try builder.dispatchDirect(1, 1, 1);
    try builder.padToAlignment(4);
    try std.testing.expectEqual(@as(usize, 8), builder.written().len);
}
