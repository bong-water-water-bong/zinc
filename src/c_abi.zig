// c_abi.zig — C ABI shim for embedding ZINC as a library backend inside
// 1bit-systems' unified_server (see 1bit-systems/src/backend_zinc.cpp).
//
// Thin wrapper only — no changes to ZINC's compute internals. Mirrors the
// exact single-prompt initialization sequence in src/main.zig's Vulkan path
// (Instance -> gpu_detect -> CommandPool -> loader.load -> InferenceEngine.init
// -> Tokenizer.initFromGGUF), packaged as a persistent opaque state with an
// init/generate/destroy lifecycle (same convention 1bit-systems' own
// zaya_engine.h extern "C" ABI already uses).
//
// Exposes a single fused generate call (prompt text in, response text out)
// rather than raw per-token stepping: ZINC's own compute.forward.generate()
// already does a real batched prefill + full decode loop in one call, which
// is both correct and efficient — no benefit to re-exposing it token-by-token
// across the FFI boundary, and it sidesteps needing the caller to juggle
// ZINC's native GGUF-derived vocabulary directly (encode/decode stays fully
// inside this file, using ZINC's own tokenizer.zig, so IDs never leak across
// the boundary in a vocabulary that could mismatch whatever tokenizer the
// C++ side happens to be using for its own token-ID space).

const std = @import("std");
const instance_mod = @import("vulkan/instance.zig");
const gpu_detect = @import("vulkan/gpu_detect.zig");
const CommandPool = @import("vulkan/command.zig").CommandPool;
const loader_mod = @import("model/loader.zig");
const forward_mod = @import("compute/forward.zig");
const tokenizer_mod = @import("model/tokenizer.zig");
const gguf_mod = @import("model/gguf.zig");

const ZincState = struct {
    allocator: std.mem.Allocator,
    vk_instance: instance_mod.Instance,
    cmd_pool: CommandPool,
    model: loader_mod.Model,
    engine: forward_mod.InferenceEngine,
    tokenizer: tokenizer_mod.Tokenizer,
};

fn resolveShaderDir(allocator: std.mem.Allocator) ![]u8 {
    // main.zig's resolveShaderDir is private and CLI-exe-relative, which
    // doesn't apply here (this .so gets loaded into unified_server, not the
    // zinc CLI binary). Configurable via env var to match this codebase's
    // existing convention (NPU_XCLBIN_DIR, NPU_ENGINE_BIN); falls back to
    // the standard zig-out install location under this checkout.
    if (std.process.getEnvVarOwned(allocator, "ZINC_SHADER_DIR")) |dir| {
        return dir;
    } else |_| {}
    return allocator.dupe(u8, "/opt/zinc/share/zinc/shaders");
}

export fn zinc_init(gguf_path: [*:0]const u8) ?*ZincState {
    const allocator = std.heap.c_allocator;
    const state = allocator.create(ZincState) catch return null;

    // Construct every field DIRECTLY on the heap-allocated `state`, not into
    // local stack variables copied in afterward. InferenceEngine.init (and
    // possibly Model/CommandPool internals) stores pointers to what's passed
    // in (&model, &vk_instance) rather than copying — main.zig's CLI path
    // gets away with local variables because those locals live in main()'s
    // own frame for the whole program lifetime. Here, zinc_init() returns,
    // so any locals would go out of scope and leave the engine holding
    // dangling pointers (this was a real bug: first-ever generate() call
    // crashed with a Vulkan "Invalid device" error from a stale &vk_instance).
    state.allocator = allocator;

    state.vk_instance = instance_mod.Instance.init(allocator, 0) catch {
        allocator.destroy(state);
        return null;
    };
    const gpu_config = gpu_detect.detect(&state.vk_instance);

    state.cmd_pool = CommandPool.init(&state.vk_instance) catch {
        state.vk_instance.deinit();
        allocator.destroy(state);
        return null;
    };

    const path_slice = std.mem.span(gguf_path);
    state.model = loader_mod.load(path_slice, &state.vk_instance, &state.cmd_pool, allocator) catch {
        state.cmd_pool.deinit();
        state.vk_instance.deinit();
        allocator.destroy(state);
        return null;
    };

    const shader_dir = resolveShaderDir(allocator) catch {
        state.model.deinit(&state.vk_instance);
        state.cmd_pool.deinit();
        state.vk_instance.deinit();
        allocator.destroy(state);
        return null;
    };
    defer allocator.free(shader_dir);

    state.engine = forward_mod.InferenceEngine.init(&state.model, &state.vk_instance, gpu_config, shader_dir, allocator) catch {
        state.model.deinit(&state.vk_instance);
        state.cmd_pool.deinit();
        state.vk_instance.deinit();
        allocator.destroy(state);
        return null;
    };

    state.tokenizer = tokenizer_mod.Tokenizer.initFromGGUF(&state.model.gguf_file, allocator) catch {
        state.engine.deinit();
        state.model.deinit(&state.vk_instance);
        state.cmd_pool.deinit();
        state.vk_instance.deinit();
        allocator.destroy(state);
        return null;
    };

    return state;
}

export fn zinc_destroy(state: ?*ZincState) void {
    const s = state orelse return;
    s.tokenizer.deinit();
    s.engine.deinit();
    s.model.deinit(&s.vk_instance);
    s.cmd_pool.deinit();
    s.vk_instance.deinit();
    s.allocator.destroy(s);
}

// Fused generate: prompt text in, response text out (caller-supplied buffer).
// Returns bytes written on success, -1 on error, -2 if out_buf was too small
// (nothing written in that case — caller should retry with a bigger buffer).
export fn zinc_generate_text(
    state: ?*ZincState,
    prompt_text: [*:0]const u8,
    max_tokens: u32,
    out_buf: [*]u8,
    out_cap: usize,
) i64 {
    const s = state orelse return -1;
    const prompt_slice = std.mem.span(prompt_text);

    const prompt_tokens = s.tokenizer.encodePrompt(prompt_slice, s.allocator) catch return -1;
    defer s.tokenizer.freeEncoded(prompt_tokens);

    const output_tokens = forward_mod.generate(
        &s.engine,
        prompt_tokens,
        max_tokens,
        s.tokenizer.eosId(),
        s.allocator,
    ) catch return -1;
    defer s.allocator.free(output_tokens);

    var text_buf: std.ArrayList(u8) = .{};
    defer text_buf.deinit(s.allocator);
    for (output_tokens) |tid| {
        var dec_buf: [256]u8 = undefined;
        const decoded = s.tokenizer.decodeToken(tid, &dec_buf);
        text_buf.appendSlice(s.allocator, decoded) catch return -1;
    }

    if (text_buf.items.len > out_cap) return -2;
    @memcpy(out_buf[0..text_buf.items.len], text_buf.items);
    return @intCast(text_buf.items.len);
}

// ── Tokenizer-only C ABI ──────────────────────────────────────────────────
// Lets the C++ side use ZINC's real GGUF-embedded BPE tokenizer for ANY
// GGUF model, without needing a matching Vulkan/GPU engine (or any external
// tokenizer.json — the vocab is self-contained in the GGUF file, same as
// zinc_generate_text's internal tokenizer, just exposed for callers that
// need to encode/decode independently of running generation through ZINC
// itself, e.g. so other backends can produce coherent text too). Metadata
// parse only (mmap + header scan) — doesn't touch tensor weight data, so
// this is cheap even for large models.

const TokenizerState = struct {
    allocator: std.mem.Allocator,
    mmap_data: []align(std.heap.page_size_min) const u8,
    gguf_file: gguf_mod.GGUFFile,
    tokenizer: tokenizer_mod.Tokenizer,
};

export fn zinc_tokenizer_init(gguf_path: [*:0]const u8) ?*TokenizerState {
    const allocator = std.heap.c_allocator;
    const state = allocator.create(TokenizerState) catch return null;
    state.allocator = allocator;

    const path_slice = std.mem.span(gguf_path);
    const file = std.fs.openFileAbsolute(path_slice, .{}) catch {
        allocator.destroy(state);
        return null;
    };
    defer file.close();
    const stat = file.stat() catch {
        allocator.destroy(state);
        return null;
    };
    const mmap_data = std.posix.mmap(
        null,
        stat.size,
        std.posix.PROT.READ,
        .{ .TYPE = .PRIVATE },
        file.handle,
        0,
    ) catch {
        allocator.destroy(state);
        return null;
    };
    state.mmap_data = mmap_data;

    state.gguf_file = gguf_mod.parseWithOptions(mmap_data, allocator, .{ .log_summary = false }) catch {
        std.posix.munmap(mmap_data);
        allocator.destroy(state);
        return null;
    };

    state.tokenizer = tokenizer_mod.Tokenizer.initFromGGUF(&state.gguf_file, allocator) catch {
        state.gguf_file.deinit();
        std.posix.munmap(mmap_data);
        allocator.destroy(state);
        return null;
    };

    return state;
}

export fn zinc_tokenizer_destroy(state: ?*TokenizerState) void {
    const s = state orelse return;
    s.tokenizer.deinit();
    s.gguf_file.deinit();
    std.posix.munmap(s.mmap_data);
    s.allocator.destroy(s);
}

export fn zinc_tokenizer_bos_id(state: ?*TokenizerState) u32 {
    const s = state orelse return 0;
    return s.tokenizer.bosId();
}

export fn zinc_tokenizer_eos_id(state: ?*TokenizerState) u32 {
    const s = state orelse return 0;
    return s.tokenizer.eosId();
}

// Returns token count on success, -1 on error, -2 if out_ids was too small
// (nothing written — caller should retry with a bigger buffer).
export fn zinc_tokenizer_encode(
    state: ?*TokenizerState,
    text: [*:0]const u8,
    out_ids: [*]u32,
    out_cap: usize,
) i64 {
    const s = state orelse return -1;
    const text_slice = std.mem.span(text);
    const tokens = s.tokenizer.encodePrompt(text_slice, s.allocator) catch return -1;
    defer s.tokenizer.freeEncoded(tokens);
    if (tokens.len > out_cap) return -2;
    @memcpy(out_ids[0..tokens.len], tokens);
    return @intCast(tokens.len);
}

// Returns byte count on success, -1 on error, -2 if out_buf was too small.
export fn zinc_tokenizer_decode(
    state: ?*TokenizerState,
    ids: [*]const u32,
    n_ids: usize,
    out_buf: [*]u8,
    out_cap: usize,
) i64 {
    const s = state orelse return -1;
    var text_buf: std.ArrayList(u8) = .{};
    defer text_buf.deinit(s.allocator);
    for (ids[0..n_ids]) |tid| {
        var dec_buf: [256]u8 = undefined;
        const decoded = s.tokenizer.decodeToken(tid, &dec_buf);
        text_buf.appendSlice(s.allocator, decoded) catch return -1;
    }
    if (text_buf.items.len > out_cap) return -2;
    @memcpy(out_buf[0..text_buf.items.len], text_buf.items);
    return @intCast(text_buf.items.len);
}
