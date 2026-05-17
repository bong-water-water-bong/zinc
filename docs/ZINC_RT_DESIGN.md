# ZINC_RT — The ZINC Runtime

**Status:** Design proposal, not yet implemented
**Audience:** AI coding agents executing on this design; senior contributors reviewing it
**Last updated:** 2026-05-10
**Owner:** ZINC core
**Primary target:** AMD RDNA4 (Radeon AI PRO R9700, RX 9070 family). Portable path to RDNA3 first, Intel Arc Xe2 and NVIDIA later. **Apple Silicon Metal is folded in as a tier.**

> Vulkan is a graphics API that grew a compute mode. It was designed to keep a game running at 144 Hz on a thermal budget while also juggling a swapchain, ray tracing, and a thousand draw calls. We are doing none of that. We run one shape of compute, on one queue, for one workload — LLM inference — and we pay the full graphics-era tax for the privilege.
>
> ZINC_RT is the layer that ends that tax. It is ZINC's own GPU runtime — the equivalent of CUDA Runtime or HSA Runtime, scoped to inference and owned end-to-end. Vulkan stays available alongside it as a fully supported alternative backend, but it is no longer the only path to the GPU.

---

## 0. Reader's Map

This document is long on purpose. It is meant to be executable by a coding agent that has read only the repo and this file. The sections are:

| § | Title | What it answers |
|---|---|---|
| 1 | Executive Summary | One page — what ZINC_RT is and why |
| 2 | The Problem | Why Vulkan is the wrong abstraction for ZINC's hot path |
| 3 | Performance Targets | Concrete, falsifiable numbers we commit to |
| 4 | Non-Goals | What ZINC_RT is **not** trying to be |
| 5 | The Name | Why "ZINC_RT" — and what kind of thing it is |
| 6 | Architecture Overview | The diagram and the call stack |
| 7 | Design Principles | The seven rules every decision is checked against |
| 8 | Tiered Backends | T1 PM4-direct, T2 UMQ, T-CPU, T-Metal, T-Intel, T-CUDA |
| 9 | The IR | The compute-graph IR ZINC's runtime emits |
| 10 | Memory Model | Heaps, BAR, paged KV, weight residency |
| 11 | Kernel ABI | What replaces SPIR-V descriptor sets |
| 12 | Submission Model | From 49 fence waits per token → 1 → 0 |
| 13 | PM4 Direct Path | The packet stream, doorbells, fences |
| 14 | UMQ Path | The portable AMD path on kernel 6.16+ |
| 15 | T-CPU Reference Backend | Bring-up, validation oracle, laptop development |
| 16 | T-Metal Tier | Folding the existing Metal work into ZINC_RT |
| 17 | WMMA on RDNA4 | Why we keep wave64 *and* add a wave32 WMMA path |
| 18 | Continuous Batching | The vLLM-class scheduler, built in |
| 19 | Paged KV v2 | What the new cache looks like under ZINC_RT |
| 20 | Speculative Decoding | Hooks (not a 1.0 feature, but designed in) |
| 21 | The Megakernel | The end-state where decode is one GPU launch |
| 22 | Portability Plan | Intel and NVIDIA |
| 23 | Validation | How we know it's right and how we know it's fast |
| 24 | Coexistence | How `forward.zig` (Vulkan) and `forward_zinc_rt.zig` ship side-by-side |
| 25 | Milestones | M0 → M8, exit criteria, predicted tok/s |
| 26 | Risks | What could kill this, and the mitigation |
| 27 | Open Questions | What we don't decide today |
| A | Appendix: PM4 packet quick-ref | |
| B | Appendix: IR opcode table | |
| C | Appendix: Source-tree diff plan | |

---

## 1. Executive Summary

ZINC today runs LLM inference through Mesa RADV → Vulkan 1.3 → SPIR-V compute shaders on Linux, and through Metal on macOS. On RDNA4 the engine reaches **117 tok/s decode on Qwen 3.6 35B-A3B** (1.12× llama.cpp, ~31% of peak DRAM bandwidth). The remaining gap to bandwidth-saturation has two sources:

1. **Submission overhead.** A single decode step still calls `submitAndWait` 1–4 times on the fast path, and ~25–50 times when MoE router/topk readback is uncovered. Each `vkQueueSubmit + vkWaitForFences` round-trip costs ~33 µs on RADV. Continuous batching on top of Vulkan multiplies this — every joining/leaving request forces a re-record of a static graph (~80 µs / 1500-node graph), every concurrent decode multiplies the fence traffic.
2. **The Vulkan abstraction itself.** Descriptor sets, pipeline layouts, command-pool resets, validation-driver layers, `vkUpdateDescriptorSets` / `vkResetDescriptorPool` on devices without `VK_KHR_push_descriptor`, the SPIR-V → ACO pipeline whose performance regresses by 5× on a newer glslc release — these are all costs we did not choose. They are the price of running on a graphics API.

**ZINC_RT** is ZINC's own GPU runtime. Same OS-level category as the CUDA Runtime, ROCm's HSA Runtime, or Vulkan itself — a *userspace* layer sitting on top of the kernel driver (`amdgpu` on Linux, the Apple Metal stack on macOS). Not a kernel driver. Not a graphics API. **No Vulkan anywhere in ZINC_RT.**

It is built on three observations:

* **Tinygrad already submits PM4 packets directly to the AMD command processor**, with no Mesa, no libdrm. The model works. It is missing everything around it: scheduling, memory planning, an IR, a kernel ABI portable to other vendors.
* **Linux 6.16 added experimental user-mode queues (UMQs) for GFX11 and GFX12**, exactly the architectures we run on. UMQ is the stable, supported way to bypass per-submit ioctls.
* **All current state-of-the-art megakernel inference engines are CUDA-only** (Mirage MPK, Hazy Research Llama-1B megakernel, Modular MAX). The AMD inference ecosystem has not crossed this threshold. The first project to do so on consumer RDNA4 owns that ground.

ZINC_RT collapses the submission tower from `forward.zig → vk_command_buffer → RADV → amdgpu_cs → kernel → CP` to:

```
forward.zig → zinc_rt.Engine → [IR → ZINC_RT codegen] → user-mapped ring → CP doorbell
```

The buildout is staged. Today, ZINC's decode step traces ~1500 shader dispatches and 49 `submitAndWait` waits per token. After **M1**, decode is one submit per token on ZINC_RT. After **M3**, prefill and decode share a continuous-batching scheduler that maintains a single resident kernel graph across thousands of requests. After **M5**, decode is *one* kernel launch — the megakernel. The Vulkan backend ships in parallel throughout — same product, two GPU paths, both first-class, both in CI.

The expected final state for Qwen 3.6 35B-A3B on R9700, decode steady state with 4 concurrent slots:

| State | Decode tok/s (per slot) | Aggregate tok/s | BW utilization |
|---|---:|---:|---:|
| Today (Vulkan) | 117 | ~432 | 31 % |
| ZINC_RT M1 (single-submit, T2 UMQ) | 145 | 580 | 39 % |
| ZINC_RT M3 (CB scheduler + paged KV v2) | 195 | 780 | 52 % |
| ZINC_RT M5 (full megakernel) | 240 | **960** | 65 % |
| Theoretical bandwidth ceiling | 365 | 1460 | 100 % |

ZINC_RT is RDNA-first, but its IR is hardware-vendor-neutral. M7 brings up Intel Arc as a direct-submission tier. M8 brings up NVIDIA via the CUDA Driver API + CUDA Graphs. The Vulkan backend remains the supported fallback for anything ZINC_RT doesn't yet have a direct tier for, and remains a first-class CI target indefinitely.

---

## 2. The Problem

### 2.1 Where the time actually goes on RDNA4

From `docs/RDNA4_PERFORMANCE_JOURNEY.md` and the in-tree GPU profiler, the per-token decode budget on Qwen 3.6 35B-A3B at the current 110-117 tok/s peak is approximately:

```
GPU compute ........................... 7.8 ms (87 %)
  matmul (DMMV + MoE + LM head) ....... 5.2 ms
  attention (flash + RoPE + KV write) . 1.3 ms
  SSM (conv1d + delta-net + gnorm) .... 0.8 ms
  elementwise + RMS norm + activations  0.5 ms
CPU embed + record .................... 0.6 ms ( 7 %)
Submit + fence wait ................... 0.5 ms ( 6 %)
Total ................................. 8.9 ms
```

At first glance, "submission overhead is 6 %" looks small enough to ignore. **It is not.** Two effects compound:

1. **Tail latency is dominated by submit-and-wait.** When the GPU finishes 12 µs before the CPU sets up the next descriptor write, the GPU sits idle for 12 µs. When this happens at MoE-router boundaries (where the CPU reads router logits before dispatching experts), the GPU goes idle for ~30 µs *per layer with router-readback*. On the *current* fast path this is gated away from the hot loop, but **any path that needs CPU intervention mid-decode reintroduces it**. Continuous batching on top of Vulkan is exactly such a path.

2. **The numbers above are for a single stream.** With four concurrent streams (vLLM-class continuous batching), the Vulkan path must either (a) record one shared command buffer that does all four streams' work, or (b) submit four separate buffers. Option (a) requires re-record on every set of joining/leaving requests — re-record is ~80 µs on a 1500-node graph. Option (b) multiplies fence-wait latency by 4. **ZINC_RT lets us do (a) without the re-record cost**, by making graph mutations cheap user-space ring edits instead of API calls.

### 2.2 The Vulkan API tax we don't choose

Vulkan was designed for a use case ZINC does not have:

| Vulkan feature | Why it exists | Cost to ZINC |
|---|---|---|
| Descriptor sets | Bind 100s of textures per draw call | We bind 3–7 SSBOs per dispatch — pure overhead |
| Pipeline layout | Reuse layouts across pipelines | We have 60 pipelines, mostly one-off layouts |
| Render passes / framebuffers | Tile-based renderers | Compute-only — we don't use these, but pay for the API width |
| VkSubmitInfo + semaphores | GPU-GPU sync across queues | We use one queue |
| VkInstance + layers | Validation, RenderDoc | Loaded on every run, 30 ms cold-start |
| `vkUpdateDescriptorSets` | Bind dynamic resources | Hot path on push-descriptor-less devices |
| `VkPipelineCache` blob format | Cross-vendor portable shaders | RADV cache is per-driver-version; rebuilds on every `mesa` upgrade |

Numerically, the cost is small *per call*. Architecturally, it is enormous: every one of these primitives forces an API edge between ZINC and the GPU, and every edge limits how much ZINC can know about its own workload. We can't fuse two dispatches across a pipeline boundary. We can't keep a kernel resident across user requests. We can't even know whether two consecutive dispatches' barriers can be merged without re-parsing our own command stream.

### 2.3 Why "just replace Vulkan with Vulkan again" doesn't help

Two adjacent options exist and are explicitly rejected:

* **ROCm + HIP.** ROCm 7.0 supports RDNA4 (gfx1201). Using HIP would replace SPIR-V with HSAIL and the Mesa stack with ROCr. *But:* (a) HIP brings the entire ROCm runtime (~600 MB) and a C++ dependency we don't want in a Zig project; (b) HIP Graphs functionally match CUDA Graphs but do not yet expose persistent-kernel control on RDNA4; (c) ROCm has a worse track record on consumer cards than RADV — Mesa is faster *and* more compatible. The README's "no ROCm, no MLX" promise is part of the project's identity.
* **A second SPIR-V compiler.** Swap glslc → DXC → glslang etc. We tried this. Newer toolchains regressed RADV performance by **5×**. The toolchain is fragile precisely *because* the API is a portability surface we don't need.

### 2.4 What we actually want

A userspace GPU runtime that:

1. **Knows it is running LLM inference.** Treats the decode forward pass as a first-class object, not a generic compute graph.
2. **Submits at most once per token in steady state, ideally zero times.**
3. **Lets the scheduler mutate the work in flight** without re-recording anything.
4. **Has a kernel ABI under our own control**, with no toolchain landmines.
5. **Is portable in *principle*** so we don't paint ourselves into an AMD-only corner.

ZINC_RT is exactly that.

---

## 3. Performance Targets

These are commitment-level targets. Each is falsifiable by a single command on the RDNA4 test node.

### 3.1 Single-stream decode (Qwen 3.6 35B-A3B Q4_K_XL, ctx 4096, greedy)

| Milestone | Target tok/s | vs ZINC today (117) | vs llama.cpp (104) |
|---|---:|---:|---:|
| M0 (T-CPU, behind feature flag) | n/a (CPU reference, not perf target) | n/a | n/a |
| M1 (T2 UMQ single-submit decode) | 140 | 1.20× | 1.35× |
| M2 (T1 PM4 + chunked prefill) | 165 | 1.41× | 1.59× |
| M3 (continuous batching + paged KV v2) | 195 | 1.67× | 1.88× |
| M5 (full megakernel) | 240 | 2.05× | 2.31× |

These numbers assume DRAM-bandwidth saturation curves:

```
tok/s_max = bw_gbps / weight_bytes_per_token
         = 576 / 1.575           # Qwen 3.6 35B-A3B = ~1.57 GiB read per token
         ≈ 365 tok/s             # 100% BW utilization, theoretical
```

M5's 240 tok/s = 66 % bandwidth utilization. This is below the 73 % H100 megakernel ceiling reported by Hazy Research, leaving headroom for the kinds of RDNA4-specific bottlenecks (scalar load contention, L2 stride aliasing) that always show up once the obvious wins are gone.

### 3.2 Continuous-batching aggregate throughput

| Concurrent slots | Today | M3 | M5 |
|---|---:|---:|---:|
| 1 | 117 | 195 | 240 |
| 4 | 432 (linear) | 760 | 960 |
| 16 | OOM | 2 100 | 2 800 |
| 64 | OOM | 5 800 | 7 800 |

### 3.3 Prefill

Closing the documented gap on hybrid MoE+SSM models (currently 88 tok/s vs llama.cpp's 182 tok/s on Qwen 3.6 35B-A3B) is **M2's primary correctness-and-perf gate**:

| Model | ZINC today | ZINC_RT M2 | llama.cpp |
|---|---:|---:|---:|
| Qwen 3 8B dense | 115 | 180 | 84 |
| Qwen 3.6 35B-A3B MoE+SSM | 88 | 280 | 182 |
| Gemma 4 31B dense | 44 | 145 | 139 |
| GPT-OSS 20B MoE | 82 | 320 | 271 |

### 3.4 Time-to-first-token (TTFT)

For a 128-token prompt, decoding 1 token, on Qwen 3.6 35B-A3B:

| State | TTFT (ms) |
|---|---:|
| Today | 1850 |
| M2 | 480 |
| M5 (with prefix cache, warm) | < 60 |
| M5 (cold) | 420 |

### 3.5 Power

R9700 nominal is 200 W; M5 should hold at 195 W ± 5 W under sustained load.

---

## 4. Non-Goals

ZINC_RT is intentionally narrow. Things it **will not** do:

* **General-purpose compute.** Not a CUDA replacement. Not a numpy. Not a Triton. The IR knows about transformers; it does not know about reductions on arbitrary tensors.
* **Graphics.** No rasterization, no images (except optionally for vision encoders, which would not use the hot-path API), no swapchains.
* **Cross-process GPU sharing.** ZINC owns the GPU when it is running. The existing `/tmp/zinc-gpu.lock` mechanism continues to apply.
* **Training.** No backward pass. No optimizer state. Inference only.
* **Tensor parallelism across multiple GPUs.** Single-GPU only in 1.0. Multi-GPU is a deliberate later target.
* **Quantization-aware training, or any kind of recompile of model weights.** GGUF in, tokens out. Quantization formats stay in shaders.
* **Be a Vulkan implementation.** ZINC_RT does not have a Vulkan tier. The Vulkan backend (`src/vulkan/`, `src/compute/forward.zig`, `src/shaders/*.comp`) lives alongside ZINC_RT as a parallel codebase, selected with `-Dbackend=vulkan`. It is permanently supported, CI-tested, and shipped — not a deprecated path.

What ZINC_RT *will* do that's adjacent but distinct:

* Expose a stable Zig API so other ZINC components (vision encoders, draft models for speculative decoding) can build on it.

---

## 5. The Name — ZINC_RT

**ZINC_RT — the ZINC Runtime.**

The naming reflects what this thing actually is: ZINC's own GPU runtime, in the same OS-level category as the CUDA Runtime API, ROCm's HSA Runtime, or Vulkan itself. The CUDA analogy is exact:

| CUDA layer | ZINC_RT analogue |
|---|---|
| `libcuda.so` (Driver API) | `zinc_rt` (this project) |
| `nvidia.ko` (kernel driver) | `amdgpu.ko` / `i915.ko` / Metal IOKit (unchanged) |
| `libcudart.so` (Runtime API) | folded into `zinc_rt` — we have one userspace layer, not two |
| `nvcc` (compiler) | the IR lowering pass + `gas` for ISA kernels |
| CUDA Graphs | the static IR; the megakernel at M5 |

In code:
* Zig package: `zinc_rt`
* Directory: `src/zinc_rt/`
* Public API: `zinc_rt.Engine`, `zinc_rt.Ring`, etc.
* Environment variables: `ZINC_RT_TIER`, `ZINC_RT_CHUNK_CAP`, etc.
* Binary tools: `zinc-rt-trace`, `zinc-rt-replay`, `zinc-rt-validate`, etc.
* Kernel binary extension: `.zrt`
* Build flag: `zig build -Dbackend=zinc_rt`

The existing `src/vulkan/`, `src/compute/forward.zig`, and `src/shaders/*.comp` are the **Vulkan backend**, selected via `-Dbackend=vulkan`. It is a peer of `-Dbackend=zinc_rt`, not a deprecation target. CI tests both. Users pick. See §24 for the coexistence policy.

What ZINC_RT is *not*:

* Not a kernel driver. `amdgpu` is the kernel driver, in the Linux kernel, unchanged.
* Not a graphics API. Compute-only, inference-specific.
* Not a general compute API. The IR's opcodes are transformer-shaped (RMS-norm-fused-projection, MoE gate-up, paged flash attention) — not arbitrary tensor algebra.
* Not a wrapper around something else. It owns submission, memory, the IR, the kernel ABI, the scheduler.

---

## 6. Architecture Overview

### 6.1 The call stack, now and after ZINC_RT

**Today (Vulkan backend, the only path):**

```
+----------------------------------+
| ZINC HTTP / CLI                  |
+----------------------------------+
| Scheduler stubs (request.zig)    |
+----------------------------------+
| forward.zig — decodeStep         |   12 676 lines, owns descriptor sets
+----------------------------------+
| compute/{dmmv,elementwise,attn}  |   per-op dispatchers
+----------------------------------+
| vulkan/{command,pipeline,buffer} |   Vulkan API wrapper
+----------------------------------+
| RADV (Mesa Vulkan compute)       |
+----------------------------------+
| amdgpu / libdrm_amdgpu           |
+----------------------------------+
| CP firmware on GPU               |
+----------------------------------+
```

49 `submitAndWait` per decode token. 87 `computeBarrier()`. ~1500 dispatches. **This Vulkan stack stays alive in `src/vulkan/` and `src/compute/forward.zig` permanently** — it is the parallel backend selected by `-Dbackend=vulkan`, not a deprecation target.

**After M3 (ZINC_RT default on RDNA Linux):**

```
+----------------------------------+
| ZINC HTTP / CLI                  |
+----------------------------------+
| zinc_rt.Scheduler (continuous batch)
+----------------------------------+
| forward_zinc_rt.zig — emits IR   |   ~2 000 lines, no descriptor sets
+----------------------------------+
| zinc_rt.Engine                   |   IR → ring packet sequence
+----------------------------------+
| zinc_rt.Ring (tier-specific)     |   one of:
|   • T1: PM4 → CP doorbell        |     direct (tinygrad style)
|   • T2: UMQ MQD → MES            |     kernel 6.16+, default
|   • T-CPU: pure Zig              |     validation oracle, no GPU
|   • T-Metal: MTLComputeEncoder   |     macOS
|   • T-Intel: i915/xe doorbell    |     Intel Arc Xe2+ (M7)
|   • T-CUDA: CUDA Driver API      |     NVIDIA (M8)
+----------------------------------+
| kernel driver (amdgpu/Metal/etc.)|   transient — touched only at queue create
+----------------------------------+
| CP firmware / Apple GPU / etc.   |
+----------------------------------+
```

1 submit per decode token (T2 path). 0 submits per token after the initial ring fill (T1 path). 1 PM4 dispatch packet per shader (still ~1500 packets/token at M3, but ring writes cost ~10 ns each — total ~15 µs vs. today's 33 µs *per* fence wait).

**After M5:**

```
... unchanged above the engine ...
| zinc_rt.Engine emits one packet  |
+----------------------------------+
| zinc_rt megakernel — one ISA blob|   loads all layers, all experts, attention,
|                                  |   LM head, argmax in one persistent dispatch
+----------------------------------+
```

The megakernel is one dispatch. The CP runs it. Until the megakernel signals "done", the engine does nothing — it just consumes streamed output via a host-mapped result ring.

### 6.2 Source-tree shape

```
src/
├── zinc_rt/                    # NEW
│   ├── engine.zig              # Engine: owns device, ring, weight residency
│   ├── ring/                   # Ring backends (one per tier)
│   │   ├── mod.zig             # Backend interface
│   │   ├── pm4.zig             # T1: direct PM4 over KFD
│   │   ├── umq.zig             # T2: amdgpu user-mode queues (6.16+)
│   │   ├── cpu.zig             # T-CPU: pure Zig reference impl
│   │   ├── metal.zig           # T-Metal: Apple Silicon (wraps existing metal/ work)
│   │   ├── i915.zig            # T-Intel (M7)
│   │   ├── cuda.zig            # T-CUDA (M8)
│   │   └── packet.zig          # PM4 packet builder (shared T1/T2)
│   ├── kmd.zig                 # Kernel-driver thin shim: ioctls, BO mgmt
│   ├── ir/                     # ZINC_RT Intermediate Representation
│   │   ├── op.zig              # Opcode enum + per-op shape rules
│   │   ├── graph.zig           # IR graph builder (replaces compute/graph.zig)
│   │   ├── lower.zig           # IR → ZINC_RT-ISA blob (codegen)
│   │   └── verify.zig          # Static shape/dtype/binding checker
│   ├── isa/                    # Kernel ABI binaries
│   │   ├── gfx1201/            # RDNA4 (Navi 48/44)
│   │   │   ├── *.s             # GAS asm source
│   │   │   └── *.bin           # Pre-assembled blobs (committed)
│   │   ├── gfx1151/            # RDNA4 APU (Strix Halo)
│   │   ├── gfx1100/            # RDNA3
│   │   ├── xe2hpg/             # Intel Xe2 (M7)
│   │   ├── sm_90/              # NVIDIA Hopper (M8)
│   │   ├── apple_msl/          # Metal Shading Language source (T-Metal)
│   │   └── cpu_zig/            # Pure Zig kernels for T-CPU
│   ├── mem/
│   │   ├── plan.zig            # Replaces gpu/memory_plan.zig
│   │   ├── pool.zig            # Suballocator over BOs
│   │   ├── pagetable.zig       # Paged KV / activations
│   │   └── bar.zig             # Host-visible mapped VRAM (large BAR)
│   ├── sched/                  # Continuous-batching scheduler
│   │   ├── batcher.zig         # Forms decode/prefill batches
│   │   ├── slot.zig            # Request slot lifecycle
│   │   └── policy.zig          # Admission, preemption, fairness
│   ├── spec/                   # Speculative decoding (hooks only in 1.0)
│   │   └── eagle_stub.zig
│   ├── trace.zig               # Tracing/profiling
│   └── tests/                  # Standalone ZINC_RT unit tests
├── compute/
│   ├── forward.zig             # Vulkan backend's decode loop — permanent
│   ├── forward_zinc_rt.zig     # NEW — emits IR
│   └── ...
├── vulkan/                     # Vulkan backend infrastructure — permanent
├── shaders/                    # Vulkan backend SPIR-V shaders — permanent
└── gpu/
    └── interface.zig           # NOW: dispatches to zinc_rt|vulkan|metal
```

**Two backends, side by side, both first-class.** The Vulkan stack (`src/vulkan/`, `src/compute/forward.zig`, `src/shaders/*.comp`) stays exactly where it is and continues to ship as `-Dbackend=vulkan`. ZINC_RT (`src/zinc_rt/`, `src/compute/forward_zinc_rt.zig`) is the new path, selected via `-Dbackend=zinc_rt`. Both are tested in CI, both are documented, both are supported for users. The Vulkan backend is the safety net any time ZINC_RT regresses or doesn't yet have a tier for a given device. The existing Metal stack (`src/metal/`, `src/compute/forward_metal.zig`, `src/shaders/metal/`) is *folded* into ZINC_RT as T-Metal at M2 — see §16.

---

## 7. Design Principles

Every ZINC_RT design decision is checked against these seven rules. When two conflict, the higher-numbered one loses.

1. **One workload, one path.** The fast path is decode. Every primitive is shaped around the decode loop first; prefill, eval, and admin paths reuse those primitives without distorting them.
2. **Submission is free in the limit.** The steady-state cost of advancing one token must be a constant number of ring-buffer writes — no ioctls, no driver calls, no kernel transitions.
3. **The IR knows about transformers.** IR opcodes describe transformer-shaped ops (attention, RMS-norm-fused-projection, MoE gate-up). They do not describe arbitrary tensor algebra.
4. **The kernel ABI is ours.** We do not depend on a SPIR-V toolchain whose semantics may shift between releases. The ABI is a binary blob format we define, plus a stable assembler.
5. **The scheduler is in ZINC_RT.** Continuous batching is a first-class concern, not a wrapper over an unaware engine.
6. **Portability is layered, not amortized.** RDNA gets the entire optimization budget. Intel, Apple, and NVIDIA get the same IR, the same ABI shape, but a separate kernel set. There is no "lowest-common-denominator" kernel.
7. **Every layer is replaceable.** No layer depends on the implementation of the layer below — only its contract. We can swap T1 for T2, swap PM4 for AQL, swap ZINC for another front-end, without rewriting ZINC_RT.

---

## 8. Tiered Backends

ZINC_RT exposes one front-end interface (`zinc_rt.Engine`) and chooses one of six back-ends at runtime. The choice is auto-detected; an environment variable forces it for testing.

```
                            ZINC_RT_TIER env var:  auto | t1 | t2 | t_cpu | t_metal | t_intel | t_cuda

           ┌──────────────────────────────────────────────────────────┐
           │  T1: PM4 Direct                                          │
           │   - opens /dev/kfd, creates HSA queue, writes PM4 packets│
           │   - lowest latency (~10 ns / packet ring write)          │
           │   - requires KFD enabled (consumer RDNA4: yes, since     │
           │     ROCm 6.4+; verified on 9070/R9700)                   │
           │   - KFD ABI is "evolving"; we pin to a verified shape    │
           └──────────────────────────────────────────────────────────┘
           ┌──────────────────────────────────────────────────────────┐
           │  T2: AMDGPU User-Mode Queues                             │
           │   - kernel 6.16+ feature, MES schedules user queues      │
           │   - same doorbell-on-ring model, but supported ABI       │
           │   - identical PM4 packet stream as T1                    │
           │   - preferred default on supported kernels               │
           └──────────────────────────────────────────────────────────┘
           ┌──────────────────────────────────────────────────────────┐
           │  T-CPU: Pure Zig Reference                               │
           │   - runs the IR on the CPU, bit-correct, very slow       │
           │   - validation oracle for every other tier               │
           │   - runs on a laptop with no GPU                         │
           │   - first tier delivered (M0); never deleted             │
           └──────────────────────────────────────────────────────────┘
           ┌──────────────────────────────────────────────────────────┐
           │  T-Metal: Apple Silicon                                  │
           │   - wraps MTLComputeCommandEncoder                       │
           │   - reuses existing src/metal/* infrastructure           │
           │   - reuses existing src/shaders/metal/*.metal kernels    │
           │   - inherits continuous batching from ZINC_RT scheduler  │
           └──────────────────────────────────────────────────────────┘
           ┌──────────────────────────────────────────────────────────┐
           │  T-Intel: Intel Arc Xe2+                                 │
           │   - i915/xe doorbell submission (M7)                     │
           │   - separate ISA kernel set; same IR                     │
           └──────────────────────────────────────────────────────────┘
           ┌──────────────────────────────────────────────────────────┐
           │  T-CUDA: NVIDIA                                          │
           │   - CUDA Driver API + CUDA Graphs (M8)                   │
           │   - separate ISA path; same IR                           │
           └──────────────────────────────────────────────────────────┘
```

**There is no Vulkan tier in ZINC_RT.** Vulkan ships as its own parallel backend (`-Dbackend=vulkan`), not as a tier of ZINC_RT. Both backends are supported permanently. A device that doesn't yet have a ZINC_RT tier of its own (NVIDIA before M8, Intel before M7) uses the Vulkan backend — same product, same CLI, same API, different GPU path.

### 8.1 Auto-detection logic

```zig
pub fn pick_tier(allocator, env) !Tier {
    if (env.get("ZINC_RT_TIER")) |forced| return parse(forced);

    // Platform-specific direct tiers
    if (builtin.os.tag == .macos) return .t_metal;
    if (is_amd_rdna_linux()) {
        if (umq.is_available()) return .t2;
        if (kfd.is_available()) return .t1;
    }
    if (is_intel_arc_xe2_linux()) return .t_intel; // M7+; before M7 we error out

    // No direct tier matches.
    // ZINC_RT has no Vulkan fallback inside itself. The caller should
    // detect this and switch the build to the Vulkan backend:
    //   zig build -Dbackend=vulkan
    return error.NoDirectTierForThisDevice;
}
```

A specific RDNA3 box on kernel 6.14 with KFD loaded lands on T1. A vanilla Ubuntu 26.04 box on kernel 6.18 with RDNA4 lands on T2. An Apple M3 Max lands on T-Metal. An Intel B770 (post-M7) lands on T-Intel. An NVIDIA card (post-M8) lands on T-CUDA. **Anything else gets a clear error and the build-flag suggestion**, not a silent fall-through to Vulkan.

### 8.2 Behavioral parity across tiers

All tiers MUST produce **bit-identical** logits for the same inputs to T-CPU on a fixed test set in CI. This is a hard test. Validation methodology in §23.

The interface every tier implements:

```zig
pub const RingBackend = struct {
    // Submit a sequence of dispatches to the GPU/CPU. Returns immediately.
    submit: *const fn (*Self, *const PacketBatch) anyerror!Fence,

    // Wait for a previously-returned fence to retire.
    wait: *const fn (*Self, Fence) anyerror!void,

    // Map device memory into the host address space (if supported).
    map: *const fn (*Self, *Buffer) anyerror![]u8,

    // Allocate a BO in a specific heap (DEVICE_LOCAL, HOST_VISIBLE, BAR).
    alloc_bo: *const fn (*Self, size: u64, heap: Heap) anyerror!Buffer,

    // Free a BO.
    free_bo: *const fn (*Self, *Buffer) void,

    // Load a kernel binary and return a callable handle.
    load_kernel: *const fn (*Self, isa_blob: []const u8) anyerror!Kernel,

    // Tier-specific tear-down.
    deinit: *const fn (*Self) void,
};
```

Note that `submit` takes a `PacketBatch` — *not* "a list of dispatch commands". The engine is in charge of building the entire packet stream (memory barriers, dispatches, fence signals) and handing it to the backend as one blob. On T1/T2, the backend literally `memcpy`s that blob into the ring buffer. On T-CPU it walks it and runs each dispatch's Zig implementation. On T-Metal it translates each dispatch into an MTLComputeCommandEncoder call. On T-Intel/T-CUDA similarly.

---

## 9. The IR

The IR is what `forward_zinc_rt.zig` emits. It is **not** a general-purpose tensor IR — it is shaped specifically around transformer decode and prefill. In the source it lives under `src/zinc_rt/ir/`.

### 9.1 Design constraints

* **Shape-static.** Every IR graph is static at construction. Shape parameters (sequence length, batch size, active expert IDs) are *parameters*, not graph topology. Graph rebuilds are reserved for model swap.
* **Stage-aware.** An IR node knows whether it runs in prefill or decode or both. The scheduler reuses one IR per (model, stage) tuple.
* **Stream-friendly.** An IR graph is serializable to a flat byte buffer. The engine `mmap`s prebuilt graphs from disk on cold start.
* **Verifiable.** `zinc_rt.ir.verify` rejects malformed IR (shape mismatch, dangling buffer, unsupported dtype combo) before any GPU code runs. This catches almost every shader bug as a host-side error.

### 9.2 Opcodes (selection — full table in Appendix B)

| Opcode | Meaning | Notes |
|---|---|---|
| `EMBED` | Read row from token embedding table | Currently CPU; moves to GPU at M2 |
| `RMS_NORM` | RMS normalization | Plain |
| `RMS_NORM_FUSED_QKV` | Fused RMS norm + Q,K,V projections | Replaces the existing 4-op chain |
| `RMS_NORM_FUSED_MLP_GATE_UP` | RMS norm + MLP gate + up | Fused 3-into-1 |
| `ROPE` | Rotary positional embedding | Variants: NeoX, IMRoPE, partial |
| `RMS_NORM_FUSED_ROPE_KV_WRITE` | Norm + RoPE + KV-cache write | Already exists in shaders/qk_norm_rope_kv_write.comp |
| `FLASH_ATTN` | Paged flash attention | Decode (one query) |
| `FLASH_ATTN_BATCHED` | Paged flash attention, multi-query | Prefill / batched decode |
| `MOE_GATE_TOPK` | Router + softmax + top-k | Fully on GPU; no readback |
| `MOE_GATE_UP` | Per-expert gate + up projections | Batched across experts |
| `MOE_SWIGLU` | Activation | |
| `MOE_DOWN_ACC` | Per-expert down projection + weighted accumulate | Fused |
| `SHARED_EXPERT` | Always-active expert | Fused gate+up+swiglu+down |
| `SSM_CONV1D` | 1-D causal conv | |
| `SSM_DELTA_NET` | Recurrent state update | The hot SSM op |
| `SSM_GATED_NORM` | Gated RMS norm | |
| `RESIDUAL_RMS_NORM` | Residual add + post-norm | Already shader-fused |
| `LM_HEAD` | Final projection to vocab logits | |
| `ARGMAX` / `SAMPLE` | Token selection | |
| `KV_WRITE_BATCHED` | Write K/V into paged cache | Prefill |
| `BARRIER` | Memory dependence | Explicit, but IR-lower may elide |
| `STREAM_OUT` | Emit a token to the host result ring | Used by megakernel |
| `LOAD_REQUEST_STATE` | Pull request-local state from page table | CB only |
| `STORE_REQUEST_STATE` | Push request-local state back | CB only |

### 9.3 Buffer-binding model

An IR node references buffers by **logical name**, not by descriptor binding. Names are model-aware:

* `model.attn.q_weight[layer]` — quantized Q-projection weight, layer N
* `model.embed_table` — token embedding table
* `state.kv_k[layer, page]` — page in the paged KV cache for layer N
* `state.hidden` — per-token activation scratch (reused across layers)
* `state.router_logits` — MoE router output scratch
* `request[slot].position` — sequence position counter (scalar, in scratchpad)
* `request[slot].expert_ids` — top-k expert IDs for this token's MoE
* `host_ring.tokens` — output ring for STREAM_OUT

The IR → ISA lowering layer resolves these names to actual GPU virtual addresses. At IR build time, names are validated against the model's tensor map. At lower time, they become byte offsets.

This is the single biggest structural simplification over Vulkan. We never compute descriptor sets. We never call `vkUpdateDescriptorSets`. We never reset a descriptor pool.

### 9.4 Buffer aliasing & in-place ops

The IR allows multiple writers to a buffer *within a single op chain* if and only if the writes are disjoint by construction (e.g. each workgroup writes one row, no row is written twice). The verifier checks this.

In-place ops (`RMS_NORM` that writes to its input buffer, `SCALE_ACC` that mutates the residual stream) declare it explicitly via an `inplace = true` flag. The verifier rejects any read-after-in-place-write that is not guarded by a barrier.

### 9.5 Concrete IR for one decode step (Qwen 3.6 35B-A3B)

Approximately (compressed):

```
graph "qwen36-35b-a3b-decode" {
  inputs:  request[slot].token_id, request[slot].position
  outputs: request[slot].next_token

  hidden = EMBED(model.embed_table, request.token_id)

  for layer in 0..40 {
    is_full_attn = (layer + 1) % 4 == 0

    norm = RMS_NORM_FUSED_QKV(hidden, model.attn.qkv_w[layer], model.attn.norm_w[layer])

    if is_full_attn {
      qkv = norm.qkv
      rope_kv = RMS_NORM_FUSED_ROPE_KV_WRITE(qkv, state.kv_k[layer], state.kv_v[layer],
                                             request.position)
      attn_out = FLASH_ATTN(qkv.q, state.kv_k[layer], state.kv_v[layer], state.page_table)
      hidden = RESIDUAL_RMS_NORM(hidden, attn_out * model.attn.o_w[layer], model.ffn.norm_w[layer])
    } else {
      qkv = norm.qkv
      ssm_conv = SSM_CONV1D(qkv, model.ssm.conv_w[layer], state.ssm_conv[layer])
      ssm_state = SSM_DELTA_NET(ssm_conv, model.ssm.{alpha,beta,A}[layer], state.ssm_state[layer])
      ssm_out = SSM_GATED_NORM(ssm_state, model.ssm.norm_w[layer]) * model.ssm.out_w[layer]
      hidden = RESIDUAL_RMS_NORM(hidden, ssm_out, model.ffn.norm_w[layer])
    }

    router = MOE_GATE_TOPK(hidden, model.moe.gate_w[layer], k=8)
    expert_outs = MOE_GATE_UP(hidden, model.moe.gate_up_w[layer], router.ids)
    expert_outs = MOE_SWIGLU(expert_outs)
    hidden = MOE_DOWN_ACC(expert_outs, model.moe.down_w[layer], router.ids, router.weights, hidden)
    hidden = SHARED_EXPERT(hidden, model.moe.shared_{gate_up,down}_w[layer])
  }

  out_norm = RMS_NORM(hidden, model.out_norm_w)
  logits = LM_HEAD(out_norm, model.lm_head_w)
  next = ARGMAX(logits)
  STREAM_OUT(host_ring.tokens, next)
}
```

At graph-construction time this is **a few hundred IR nodes**. After lowering, it becomes a packet stream of approximately the same length (one PM4 dispatch packet per IR node, plus barriers). The graph builder is ~2 000 lines of Zig — replacing the 12 676-line `forward.zig` decode loop.

---

## 10. Memory Model

### 10.1 Heaps

ZINC_RT exposes three GPU memory heaps:

| Heap | Backing | Purpose | Mapping |
|---|---|---|---|
| `DEVICE_LOCAL` | VRAM, not host-visible | Weights, KV cache, large scratch | Not mapped |
| `BAR` | VRAM via large-BAR PCIe BAR | Small CPU-readable: logits, profile timestamps, output ring | Mapped, coherent |
| `HOST_VISIBLE` | System RAM | Staging uploads, host-side ring buffers | Mapped, coherent |

### 10.2 BAR memory

RDNA4 supports resizable BAR. R9700 / RX 9070 expose **the entire VRAM** through the BAR on supported boards. ZINC_RT exploits this for:

* **Output ring.** Tokens flow from the megakernel into a small host-visible ring in VRAM. The CPU reads them with no `vkMapMemory`, no fence — just a memory load. Inside the kernel they're written with `global_store_dword` + `global_wb`.
* **Profile timestamps.** Same channel; megakernel writes `s_memrealtime` into a small VRAM region the CPU polls.
* **Logits readback** for the temperature/top-p sampling path (when needed).

The plain `HOST_VISIBLE` staging route stays for slow-path uploads; BAR is only for the small, hot CPU↔GPU exchange channels.

### 10.3 Weight residency

The current `loader.zig` mmaps the GGUF on disk and uploads each tensor to its own VRAM allocation. ZINC_RT preserves this on the input side but changes the allocation strategy:

* **One big VRAM allocation per layer-class.** All Q-projections across all 40 layers live in one contiguous BO; same for K-projections, V-projections, gate weights, etc. The IR lowering uses byte offsets into these BOs. This gets us better address-stride patterns and lets the L2 prefetcher work across layers.
* **Cold weights live in HOST_VISIBLE.** For models where MoE experts don't all fit in VRAM, the inactive experts live in system RAM and the GPU reads them over PCIe. The IR opcode `MOE_GATE_UP` carries an "expert is host-resident" flag.
* **mmap pages aren't faulted in until used.** Already works; preserved.

### 10.4 Paged KV cache

Today's KV cache is per-layer flat allocation. ZINC_RT's paged KV (§19) is layer-cross-cutting with GPU-resident metadata.

### 10.5 Memory plan output

`zinc_rt/mem/plan.zig` computes a fixed memory layout at engine-init time, given:

* Model config (n_layers, n_experts, hidden_dim, etc.)
* Concurrency target (max parallel requests)
* Max context per request
* Speculative-decoding lookahead window (0 if disabled)

It returns a static memory map. The engine then issues `alloc_bo` calls to the ring backend for each region. The map can be exported as JSON for `zinc-rt-trace` to display.

---

## 11. Kernel ABI

This is the single most important section of the design. It is what we own that we did not before.

### 11.1 The ABI in one paragraph

A ZINC_RT kernel is a chunk of GPU ISA (RDNA4 for gfx1201, RDNA3 for gfx110x, etc.) plus a small declarative header. The header says: "I take N buffer pointers, a push-constant block of M bytes, and a workgroup count of (X, Y, Z). I expect subgroup size W." That is the entire ABI. There are no descriptor sets, no descriptor set layouts, no push-constant ranges, no pipeline layouts.

### 11.2 Concrete header format

A kernel binary is a flat little-endian file (extension `.zrt`):

```
offset  size  field
0       4     magic 0x005A5254 = "ZRT\0"
4       2     abi_version (=1)
6       2     gpu_target  (1=gfx1201, 2=gfx1200, 3=gfx1151, 4=gfx1100,
                          0x10=xe2hpg, 0x20=sm_90, 0x30=apple_metal, 0xFF=cpu_zig)
8       4     code_size (bytes of ISA / SPIR-V / MSL / pure Zig blob)
12      4     entry_offset (within code_size; usually 0)
16      4     buffer_count (number of SSBO-like pointer args)
20      4     push_const_bytes
24      4     subgroup_size (32 or 64 or 0 for vendor-agnostic)
28      4     workgroup_size_x
32      4     workgroup_size_y
36      4     workgroup_size_z
40      4     lds_bytes (statically allocated LDS)
44      4     vgpr_count
48      4     sgpr_count
52      4     occupancy_hint
56      8     name_offset, name_size
64      ...   ISA / kernel blob
...     ...   debug info (DWARF if -Ddebug)
```

The information that matters at dispatch time (buffer pointers, push constants) is passed via PM4 `SET_SH_REG` packets on T1/T2, via Metal argument buffers on T-Metal, via CUDA kernel parameters on T-CUDA, and via plain function arguments on T-CPU.

### 11.3 Passing buffer pointers as 64-bit values (RDNA)

RDNA exposes scalar GPRs (s0–s127). The convention ZINC_RT uses:

* `s[0:1]` — pointer to buffer 0 (64-bit)
* `s[2:3]` — pointer to buffer 1
* ...up to `s[N*2:N*2+1]` — pointer to buffer N-1
* `s[N*2+2:N*2+2 + push_dwords]` — push constants
* `s[K:K+15]` — workgroup ID, dispatch dim, etc. (per existing RDNA convention)

This is exactly the convention the AMDGPU LLVM backend already uses when emitting compute kernels. We are not inventing — we are bypassing the wrapper that hides it.

At dispatch time, the engine emits PM4 packets:

```
PKT3(SET_SH_REG, 2)  SH_REG = COMPUTE_USER_DATA_0,  value = buffer0_ptr_lo
PKT3(SET_SH_REG, 2)  SH_REG = COMPUTE_USER_DATA_1,  value = buffer0_ptr_hi
... and so on for each buffer ...
PKT3(DISPATCH_DIRECT, 3) DIM_X, DIM_Y, DIM_Z
```

### 11.4 The kernel author's POV

A ZINC_RT kernel for the Q4_K DMMV could look like (excerpt, RDNA4 GAS asm):

```asm
.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
.text
.globl zinc_rt_dmmv_q4k
.type  zinc_rt_dmmv_q4k, @function

zinc_rt_dmmv_q4k:
    ;; s[0:1] = pointer to A (Q4_K weight)
    ;; s[2:3] = pointer to X (f32 activation)
    ;; s[4:5] = pointer to Y (f32 output)
    ;; s[6]   = M
    ;; s[7]   = K
    ;; s[8]   = a_offset, s[9] = x_offset, s[10] = y_offset, s[11] = acc_mode

    s_mov_b32 s_row, s_workgroup_id_x
    ;; ... K-parallel reduction over wave64, subgroupAdd via ds_swizzle/v_add ...
    ;; ... store result to Y[row] ...
    s_endpgm
```

Today the same kernel exists as GLSL in `src/shaders/dmmv_q4k_batch_kpar.comp` and is compiled by `glslc`. Bringing it up as ISA is a one-time port. The benefits:

* **No SPIR-V toolchain.** We have already been bitten by glslc-version regressions (5× perf cliff documented in `docs/RDNA4_TUNING.md`).
* **We can hand-tune** `s_waitcnt`, v_clause boundary, VGPR allocation.
* **We pick the v_clause shape.** RDNA4's clause optimizer benefits from explicit clause hints.

The downside: we own the kernels. M0–M2 keeps the GLSL/SPIR-V path (running on T-Metal and via the legacy backend for cross-validation); M4 ports the top-10-by-time kernels to native ISA on each direct tier.

### 11.5 Specialization constants

The IR's lowering pass bakes shape constants (hidden_dim, head_dim, max_experts, etc.) directly into the kernel binary by patching constant slots before the kernel is uploaded. The ISA blob has small `mov32` immediate fields the assembler emits with magic placeholder values (e.g. `0xCAFEBABE`); ZINC_RT's lowering finds them by symbol name and patches them.

This is faster than Vulkan specialization constants (which require re-creating a pipeline) and is purely a host-side memcpy.

### 11.6 Per-tier kernel sources

| Tier | Kernel source format | Notes |
|---|---|---|
| T1, T2 | RDNA GAS asm → assembled to ISA blob | M4+; M0–M3 use SPIR-V via libamdgpu-codegen |
| T-CPU | Pure Zig functions, one per opcode | hand-written, optimized for clarity not speed |
| T-Metal | MSL source from `src/shaders/metal/` | reused as-is at M2 |
| T-Intel | Intel ASM (xe2hpg) or SPIR-V on i915 | M7 |
| T-CUDA | PTX or CUDA C++ via NVRTC | M8 |

All tiers consume the same IR. The lowering pass is per-tier; the kernel binaries are per-tier.

### 11.7 What about cooperative matrix?

Cooperative matrix on RDNA4 is wave32 with 16x16x16 tiles. ZINC_RT exposes this via specific IR opcodes:

* `MATMUL_WMMA_F16` (and bf16/fp8/int8 variants)

These ops can only appear in an IR graph if the device claims WMMA support. The codegen for them uses the `v_wmma_*` instructions directly. (See §17.)

---

## 12. Submission Model

### 12.1 What submission looks like today

```
forward.zig:
  decode_cmd.reset()
  decode_cmd.begin()
  for layer in 0..40:
    dispatch RMS_NORM
    barrier
    dispatch DMMV_Q...
    ...
  decode_cmd.end()
  decode_cmd.submitAndWait(queue)        # <<< 33 µs ioctl + fence
```

Plus 25–48 mid-step `submitAndWait` calls under the various MoE/SSM paths.

### 12.2 What ZINC_RT submission looks like at M2

```
forward_zinc_rt.zig:
  let ir = build_decode_graph(token, state)         # ~5 µs CPU
  let packets = engine.lower(ir, slot=0)            # ~3 µs CPU
  let fence = ring.submit(packets)                  # writes to ring buffer
                                                    # rings doorbell
                                                    # ~200 ns total
  ring.wait(fence)                                  # one fence per decode token
```

One submit per token. The submit is a `memcpy` of ~30 KB into the user-mapped ring buffer, followed by a doorbell write (one MMIO store). The wait is a poll on a host-mapped fence dword that the GPU writes via `RELEASE_MEM`.

### 12.3 What submission looks like at M5

```
forward_zinc_rt.zig — once per session:
  let mega = engine.load_megakernel("qwen36_35b_a3b.zrt")
  let ring = engine.start(mega, host_input_ring, host_output_ring)
  # ring is running. Megakernel polls input_ring for new tokens.

forward_zinc_rt.zig — per token:
  host_input_ring.push(slot_id, token_id)            # one cacheline write
  while host_output_ring.empty(): __builtin_amd_uthreadfence_acquire()
  next_token = host_output_ring.pop(slot_id)         # one cacheline read
```

No submits. No fences. The GPU is *resident*. The CPU communicates by writing to a host-mapped, BAR-backed input ring and reading from a host-mapped output ring.

### 12.4 Cancellation / preemption

The megakernel checks a "cancel" flag in shared GPU memory at every layer boundary. Setting that flag from the CPU causes the kernel to drain its current step and return. Resuming = setting the flag back and resubmitting. Latency: ~one layer (~200 µs at M5 rates).

---

## 13. PM4 Direct Path (T1)

### 13.1 What PM4 is

PM4 is AMD's command-processor packet format. The CP firmware on the GPU reads PM4 packets from a ring buffer and executes them. Every Vulkan dispatch on AMD bottoms out as a PM4 `DISPATCH_DIRECT` or `DISPATCH_INDIRECT` packet. ZINC_RT T1 writes those packets directly, bypassing Mesa.

This is well-trodden ground for AMD-internal tools and for Tinygrad. The risk is not "is it possible" but "is the ABI stable across kernel releases." We mitigate this by:

1. **Pinning to one PM4 packet schema** that has been stable since GFX10 (RDNA1, 2019). Every packet we use exists on RDNA1–RDNA4.
2. **Probing at queue-create time** for new opcodes; falling back to T2 if anything we need is missing on this kernel.
3. **A `zinc-rt-pm4-test` tool** that runs every packet variant we use against the active kernel and refuses to start if any fail.

### 13.2 Queue setup (T1)

```
1. open /dev/kfd
2. ioctl KFD_IOC_ACQUIRE_VM    (one-time per process)
3. ioctl KFD_IOC_CREATE_QUEUE  (one HSA queue, KFD_IOC_QUEUE_TYPE_COMPUTE_AQL)
   -> kernel returns:
        - ring_base_address (we mmap it)
        - read_pointer_address (we mmap it)
        - write_pointer_address (we mmap it)
        - doorbell_offset (we mmap doorbell page)
        - eop_buffer_address
4. allocate VRAM via KFD_IOC_ALLOC_MEMORY_OF_GPU
   -> returns DMABUF fd, mmap to userspace if HOST_VISIBLE
5. allocate signal buffers (32B fence slots) via same path
```

This is exactly what tinygrad's `ops_amd.py` does. We just do it in Zig.

### 13.3 The packet builder

`zinc_rt/ring/packet.zig` is a thin wrapper that writes well-formed PM4 packets into a backing buffer. It does NOT understand the workload — it only knows packet syntax.

```zig
pub fn dispatch_direct(self: *PacketBuilder, dim_x: u32, dim_y: u32, dim_z: u32) void {
    self.write_pkt3(PKT3_DISPATCH_DIRECT, 3);
    self.write_u32(dim_x);
    self.write_u32(dim_y);
    self.write_u32(dim_z);
}

pub fn set_user_data(self: *PacketBuilder, slot: u32, value: u32) void {
    self.write_pkt3(PKT3_SET_SH_REG, 1);
    self.write_u32(R_COMPUTE_USER_DATA_0 + slot);
    self.write_u32(value);
}

pub fn release_mem_signal(self: *PacketBuilder, gpu_addr: u64, value: u64) void {
    self.write_pkt3(PKT3_RELEASE_MEM, 6);
    self.write_u32_lo(gpu_addr);
    self.write_u32_hi(gpu_addr);
    self.write_u64(value);
}
```

The set of packets used in the decode hot path is small (~12). All documented in the AMDGPU register reference for GFX10/11/12.

### 13.4 Doorbell ringing

After writing packets to the ring buffer up to `write_ptr_new`, we update the user-visible write pointer:

```zig
@atomicStore(u64, self.ring.write_ptr_mapped, write_ptr_new, .Release);
```

Then ring the doorbell — a single 64-bit MMIO store to the doorbell page:

```zig
@atomicStore(u64, self.doorbell_mapped, write_ptr_new, .SeqCst);
```

The CP wakes up, processes packets up to `write_ptr_new`, and stalls again. Total latency from CPU `submit` call to GPU "executing first packet": ~150–500 ns on RDNA4.

### 13.5 Fences

Two kinds:

* **In-band fences** — written by `RELEASE_MEM` packets at chosen points in the stream. The GPU writes a monotonic value to a host-visible memory address; the CPU polls it. Used for "submission N has retired".
* **Doorbell-only barrier** — no fence at all. Used when we don't care about completion (fire and forget — e.g. the megakernel input ring).

### 13.6 Risks specific to T1

* **KFD ABI drift.** The KFD ioctl numbers have been stable since 5.x kernels, but the *contents* of some ioctl structs changed. We pin to a version, probe at startup.
* **VRAM allocator quirks.** Some kernel versions allocate from a different heap depending on flags. We test all paths in CI.
* **Doorbell page mapping.** Must be `MAP_FIXED` for some kernels. Handled by `zinc_rt/kmd.zig`.
* **Packet-header-last gotcha.** PM4 packets must have their header written *last* so the CP doesn't try to execute a half-written packet. The packet builder enforces this with a 2-phase write and a compiler `volatile`.

The known-good reference is tinygrad's `ops_amd.py`.

---

## 14. AMDGPU UMQ Path (T2)

### 14.1 What's different from T1

UMQ is the "blessed" path: same doorbell-on-ring model as T1, but the kernel scheduler (MES) is in the loop. Pros:

* **No KFD-specific assumptions.** The amdgpu driver guarantees ABI stability across kernel releases for the UMQ path.
* **Better tool support over time.** Mesa is moving its compute path toward UMQ.
* **Same packet stream.** A T2 ring takes the exact same PM4 packets as T1.

Cons:

* **Kernel 6.16+ required.** On older kernels we fall back to T1.
* **Initial creation overhead is higher.** First `USERQ_CREATE` ioctl is ~10 ms.
* **Fewer existing references.** We are early; tinygrad's path is still KFD-based.

### 14.2 Queue setup (T2)

```
1. open /dev/dri/renderD128
2. amdgpu_device_initialize (libdrm_amdgpu — used minimally)
3. ioctl AMDGPU_USERQ_CREATE
   -> queue_id, mqd pointer, ring address, doorbell offset
4. allocate VRAM/GTT BOs via AMDGPU_GEM_CREATE (libdrm_amdgpu)
5. map the doorbell page
```

### 14.3 Submission flow

Identical to T1 from the packet stream's POV.

### 14.4 When to prefer T1 vs T2

* **T2 if kernel >= 6.16.** Future-safe, supported.
* **T1 if kernel < 6.16 but KFD works.** Saves the MES scheduler hop, lower steady-state latency.
* **Override via `ZINC_RT_TIER`** for testing.

---

## 15. T-CPU Reference Backend

T-CPU is the first tier delivered (M0) and the most important one nobody runs in production. It exists for three reasons:

1. **It's the bring-up path.** Before any of the GPU tiers work, T-CPU lets us run the entire ZINC_RT stack — IR builder, scheduler, memory plan — on a laptop with no GPU. M0 is end-to-end working software the day it ships.
2. **It's the validation oracle.** Every other tier's outputs are bit-checked against T-CPU on a fixed test set in CI. When T1 produces a different logit than T-CPU, T-CPU wins by definition. T-CPU's correctness is verified against the existing `forward.zig` CPU reference (`runDecodeStepReference`) at M0.
3. **It's the development environment.** Contributors can build, run, test, and debug ZINC_RT on macOS Intel, Linux x86 without a discrete GPU, or any other dev box. The IR can be iterated on without an RDNA card on the desk.

### 15.1 Implementation

`src/zinc_rt/ring/cpu.zig` walks the packet stream and executes each dispatch on the CPU. Each IR opcode has a hand-written Zig function in `src/zinc_rt/isa/cpu_zig/`:

```
src/zinc_rt/isa/cpu_zig/
├── rms_norm.zig
├── rms_norm_fused_qkv.zig
├── rope.zig
├── flash_attn.zig
├── moe_gate_topk.zig
├── moe_gate_up.zig
├── ssm_delta_net.zig
├── lm_head.zig
└── ...
```

These are written for clarity, not speed. They use plain f32 (no SIMD intrinsics, no threading). A 35B-A3B decode token on T-CPU takes ~30 seconds on a laptop. That's fine — T-CPU is not a performance target.

### 15.2 Why not just use the existing forward.zig CPU reference?

The existing `forward.zig` has a CPU reference path for *parts* of the decode (`runDecodeStepReference`) but it's coupled to the Vulkan structure. T-CPU is a clean Zig implementation of every IR opcode, decoupled from Vulkan. It is also where new IR ops are *first* implemented — having T-CPU light up means the IR contract is well-defined before any GPU code is written.

### 15.3 Speed targets

T-CPU should run Qwen 3 8B (the smallest catalog model) end-to-end in under one minute per decode token on a modern laptop. Anything faster is gravy; anything slower means we should look at the implementation. It is **not** "a CPU inference engine" — llama.cpp's CPU path exists, is fast, and is not what we're building.

### 15.4 Permanence

T-CPU is never deleted. It is the foundation. Every IR change starts with a T-CPU implementation; only after that lights up do we wire the GPU lowerings.

---

## 16. T-Metal Tier

Apple Silicon support is folded into ZINC_RT rather than left as a parallel stack. The current `src/metal/`, `src/compute/forward_metal.zig`, and `src/shaders/metal/` infrastructure becomes the implementation of T-Metal.

### 16.1 Why not bypass Metal?

Apple does not expose anything below Metal on macOS. No PM4 equivalent. No doorbell-on-ring user-mode submission. No documented IOKit path that would be sane to maintain across macOS versions. Metal *is* the lowest practical layer on Mac, in the same way the kernel driver is the lowest practical layer on Linux. ZINC_RT respects this.

### 16.2 What changes for Apple users

The existing Metal kernel collection (`src/shaders/metal/*.metal`, 57 shaders) is reused as-is. The existing Objective-C bridge (`src/metal/shim.{h,m}`) is reused as-is. What changes is the layer above: `forward_metal.zig`'s monolithic decode loop is replaced by `forward_zinc_rt.zig` emitting IR, with the T-Metal backend translating each IR opcode into a `MTLComputeCommandEncoder` dispatch.

Apple users gain — for free — everything ZINC_RT adds on top of bare submission:

* The continuous-batching scheduler.
* Chunked prefill (currently absent from Metal).
* Paged KV v2.
* Prefix caching.
* The Metal prefill performance roadmap collapses into "do the same things ZINC_RT does on RDNA but with MTL encoders".

### 16.3 What does NOT change

* Metal kernel sources (.metal files). They are the kernel ABI for T-Metal.
* The mmap-based zero-copy weight loading on Apple Silicon.
* Apple Silicon-specific tuning (simdgroup widths, unified memory assumptions).

### 16.4 Timeline

T-Metal lands at M2. Before M2, Apple users continue on the existing `forward_metal.zig` via `-Dbackend=metal`. At M2, `forward_zinc_rt.zig` + T-Metal becomes the default on Apple Silicon. The standalone Metal path (`-Dbackend=metal`) stays buildable and supported indefinitely as a peer of the Vulkan path on Linux — the same "two-backend, both first-class" policy applies.

---

## 17. WMMA on RDNA4

RDNA4 ships third-generation matrix cores: 16×16 wave32 with FP16/BF16/FP8/INT8 accumulators. Peak FP16 = 194.6 TFLOPS, peak FP8/INT8 = 389.3 TFLOPS (R9700). The current ZINC shader path is wave64 with subgroupAdd — **it does not use WMMA at all**. This is a significant unused resource for prefill and the LM head.

### 17.1 The plan

ZINC_RT adds WMMA on the prefill side only. Decode stays wave64 subgroupAdd because:

* Decode is bandwidth-bound, not compute-bound. WMMA doesn't help.
* Decode is M=1. WMMA wants M=16 minimum.

Prefill batches N tokens at once; M = N at the matmul level. For N ≥ 16 the tile is fully utilized.

### 17.2 The WMMA kernel set

`mul_mm_wmma_f16.zrt` and `mul_mm_wmma_q4k.zrt` (Q4_K-dequant-on-the-fly into WMMA tiles) targeting RDNA4 gfx1201. These are wave32; they coexist with the wave64 path.

The IR opcode `MATMUL_WMMA` is only selected when:
* Device claims WMMA support
* M ≥ 16
* dtype is supported (F16, BF16, F8, INT8 accumulators)

For Q4_K weights we dequantize into LDS at the inner loop start, then run WMMA on the dequantized fp16 tile.

### 17.3 Equivalents on other tiers

* T-Intel: DPAS (8×8×16 systolic) on Xe2. Same IR opcode (`MATMUL_WMMA`), different lowering.
* T-Metal: simdgroup_matrix on Apple Silicon. Same shape.
* T-CUDA: WMMA (the original); identical concept.
* T-CPU: a naive GEMM; no special handling.

---

## 18. Continuous Batching Architecture

This is what the user explicitly asked for. The design must beat vLLM on a single RDNA4 node.

### 18.1 Definitions

* **Slot** — one active request.
* **Batch** — a set of (slot, op) pairs the engine submits together.
* **Step** — one engine cycle.
* **Admission** — moving a request from queued → active.
* **Preemption** — moving an active slot back to queued.

### 18.2 The scheduling loop (megakernel-aware)

```
loop {
  pending = drain_request_queue()
  for r in pending:
    if can_admit(r):
      admit(r, alloc_kv_pages(r))
    else:
      queue_or_preempt(r)

  decoding = [s for s in slots if s.state == DECODE]
  prefilling = [s for s in slots if s.state == PREFILL]

  if decoding.is_empty() and prefilling.is_empty():
    park()
    continue

  prefill_chunk = min(remaining_prefill_tokens, CHUNK_CAP)

  step_input = {
    decode_slots: decoding,
    prefill_slots: prefilling,
    prefill_chunk_per_slot: ...,
  }

  // No submit — write to input ring (M5).
  // Or submit once (M3) if megakernel hasn't shipped.
  input_ring.push(step_input)

  for evt in output_ring.drain():
    slot = slots[evt.slot_id]
    slot.append_token(evt.token)
    if slot.should_stop():
      stream_close(slot)
      release(slot)
}
```

Differences from vLLM:

* **No re-record.** The IR graph is constructed once for (model, max_batch_size). Slots populate buffers; the graph runs unchanged.
* **No driver dispatch per slot.** vLLM submits one CUDA Graph replay per (num_tokens, num_reqs) bucket. ZINC_RT submits one packet stream (M3) or zero (M5).
* **Lock-free CPU↔GPU communication.** SPSC rings, BAR-backed.
* **No Python.** ZINC_RT's scheduler is Zig.
* **One process.** No IPC overhead.

### 18.3 The slot table

A flat structure in VRAM the megakernel reads at each step boundary:

```c
struct Slot {
  u32  state;
  u32  position;
  u32  page_count;
  u32  page_ids[MAX_PAGES_PER_SLOT];
  u32  prompt_len;
  u32  generated_len;
  u32  speculative_len;
  u32  active_expert_ids[K];
  u32  sampling_seed;
  f32  temperature, top_p, repetition_penalty;
  u64  client_handle;
}
```

Updated by the host via BAR.

### 18.4 Chunked prefill

Each step processes up to `ZINC_RT_CHUNK_CAP` input tokens (default 8192 on R9700, 4096 on 16-GB cards), distributed across decode and prefill slots. Prefill chunks are split across steps when prompts are larger than the cap.

### 18.5 Prefix caching (RadixAttention-lite)

The host maintains `hash(prompt_prefix_tokens) → list[page_id]`. New requests look up each 16-token prefix page; hits go to refcount instead of copy.

Hit rate on chat workloads is documented at 30–60% by SGLang.

### 18.6 Streaming

Output tokens land in the host-mapped output ring with `(slot_id, token, eot_flag, logprobs?)`. A drainer thread parks on a futex; one cache-line read per token. No syscall.

---

## 19. Paged KV Cache v2

Changes from today's KV cache:

* **Pages are layer-cross-cutting.** A "page" holds (K, V) for *all layers* of 16 tokens. The flash-attn shader indexes by `page_id × n_layers` plus `layer * stride`.
* **Page metadata is in GPU memory.** Currently host-side; v2 mirrors to a small VRAM table.
* **Page IDs are stable across a slot's lifetime.** Reallocation only at teardown.
* **Quantized KV (TurboQuant)** is a single shader specialization.

### 19.1 Sizing

For Qwen 3.6 35B-A3B, 16-token page, all layers (10 attention layers × 256-dim KV × 2 × 4 bytes):

```
page bytes = 10 * 256 * 2 * 4 * 16 = 320 KB per page (F32)
           = 80 KB per page (Q8_0 quantized KV)
```

R9700 has 32 GB. After model (21 GB) and scratch (~2 GB), ~9 GB free for KV. At 320 KB/page that's ~28 000 pages or ~450 000 tokens of context. At Q8_0 KV, ~1.8M tokens.

---

## 20. Speculative Decoding Hooks

Not a 1.0 feature, but designed in.

### 20.1 What it needs

* A second IR graph for the draft model.
* An IR opcode `VERIFY_K` that takes K draft tokens and runs the K-prefill against the target model.
* Page-table support for "speculative pages" committed only after acceptance.

All three are designable today, deferred to M6.

### 20.2 What it doesn't need

Architectural changes to the megakernel. Spec decoding adds a different *input shape* (K tokens per slot instead of 1), handled by the existing prefill path.

---

## 21. The Megakernel (M5)

End-state. After M5, ZINC_RT submits one PM4 stream once per session (or once per model change), and the GPU executes a persistent kernel that runs until told to stop.

### 21.1 Shape

One ISA blob. Internally:

* **Input-ring poller** — top-level loop polling host-mapped input ring.
* **Per-step driver** — given slot table, decides what to do this token.
* **Per-layer body** — transformer math.
* **Output emitter** — writes one entry to host output ring per generated token.

In RDNA4 wave64 wavefront terms: one dispatch with grid sized for the largest per-step kernel. Smaller kernels run on subsets.

### 21.2 What's hard

* **Scratch-space sizing.** All per-token activations must fit in static scratch budget (~256 KB per slot).
* **Long-running kernel timeouts.** UMQ supports long-running kernels with no host-side timeout; KFD also supports this with the right flags.
* **Cancellation.** Already covered (§12.4).
* **Resilience.** A bug hangs the GPU. M3 "one submit per token" mode stays available for debugging.

### 21.3 Why we don't start with this

* It's the hardest part. De-risk easier wins first.
* It depends on stable IR and ISA pipelines (M2 builds both).
* Most throughput win (M1→M3) doesn't need it.

M5 is the destination; M3 is the useful milestone.

---

## 22. Portability Plan

### 22.1 Intel Arc (Xe2 / Battlemage)

i915/xe drivers expose user-mode submission with similar primitives. T-Intel mirrors T2 in structure: ring buffer, doorbell, kernel binary format. DPAS for matmul. Wave size 32 in Vulkan terms (SIMD16 native).

Expected effort: 4–8 weeks. M7.

### 22.2 NVIDIA

CUDA Driver API + CUDA Graphs as T-CUDA. BIR lowers to a CUDA-Graph build. Performance match: should be close to vLLM out of the box; ZINC_RT scheduler gives the continuous-batching advantage.

Expected effort: 3–6 weeks once the IR is stable. M8.

### 22.3 Apple Silicon

T-Metal. Already covered in §16. Lands at M2.

### 22.4 What we do not do

* Build kernels that work on every vendor. Each tier has its own kernel set, hand-tuned.
* Write a portable PTX/SPIR-V/HSAIL emitter.
* Spend optimization budget on portability. The promise is "the same model runs"; not "at the same speed."

---

## 23. Validation Methodology

Correctness must hold across every backend tier and every milestone. **The oracle is T-CPU, not Vulkan.**

### 23.1 The validation pyramid

1. **Primary oracle: T-CPU.** Every T1/T2/T-Metal/T-Intel/T-CUDA output is bit-checked against T-CPU on a fixed prompt set in CI. T-CPU's correctness is itself verified against the existing `forward.zig` CPU reference (`runDecodeStepReference`) at M0.
2. **Cross-tier diff:** T1 ↔ T2 on RDNA4. T-Metal validates against T-CPU.
3. **End-to-end:** existing `tests/test_qwen_smoke.ts` runs against ZINC_RT — token-equality against golden traces.
4. **Cross-backend comparison (permanent):** a CI job runs the same prompts through the Vulkan build *and* the ZINC_RT build, diffs logits. This stays in CI forever — Vulkan is the human-readable reference because its kernel set is the older, more battle-tested one, and any time ZINC_RT diverges, that's a bug the contributor needs to look at.

### 23.2 The harness

```
zinc-rt-validate --model qwen3-8b-q4k-m --prompt "Hello" \
  --tiers t1,t2,t_cpu --layers all
```

emits a per-layer histogram of `hidden_buf` values; failure if any tier diverges from T-CPU.

### 23.3 ZINC end-to-end regression

* `zig build test`
* `tests/test_qwen_smoke.ts`
* `tests/chat_ui_markdown.test.ts`
* `tools/benchmark_api.mjs`

Plus new ZINC_RT-specific:
* `zinc-rt-pm4-test` — packet schema validity
* `zinc-rt-mem-test` — VRAM accounting under stress
* `zinc-rt-sched-test` — race tests for continuous batching

### 23.4 Performance gates

Each milestone has an exit criterion measured by `tools/performance_suite.mjs` on the RDNA4 test node. M1: ≥ 140 tok/s steady-state decode on Qwen 3.6 35B-A3B. M2: ≥ 165. Etc.

### 23.5 Trace + replay

`zinc-rt-trace` records every IR submission and every host-input-ring write to a binary log. `zinc-rt-replay` reconstructs an execution from the log and re-runs it on a fresh engine.

---

## 24. Coexistence Plan

### 24.1 Two backends, side by side, both permanent

ZINC ships two GPU backends. Both are first-class. Both are tested in CI. Both are supported for users. Neither is "legacy."

```
src/
├── zinc_rt/             ← new path (T1, T2, T-CPU, T-Metal, T-Intel, T-CUDA)
├── compute/
│   ├── forward.zig      ← Vulkan-backend decode loop — permanent
│   └── forward_zinc_rt.zig  ← ZINC_RT decode loop emitting IR
├── vulkan/              ← Vulkan backend infrastructure — permanent
├── shaders/             ← Vulkan backend's SPIR-V kernels — permanent
├── metal/               ← Apple's standalone path; folded into T-Metal at M2 (still buildable)
└── shaders/metal/       ← Metal kernels — used by T-Metal at M2; also by standalone metal backend
```

Selection at build time:

```
zig build -Dbackend=zinc_rt    # ZINC's own runtime (default after M2 where a tier exists)
zig build -Dbackend=vulkan     # Vulkan backend (permanently supported)
zig build -Dbackend=metal      # Standalone Metal backend (permanently supported)
```

### 24.2 Default backend per platform/milestone

The table below shows which backend the build picks by default, with no flag. Both Vulkan and ZINC_RT are always available on every Linux/AMD setup — the default just changes as ZINC_RT becomes the better choice.

| Phase | RDNA Linux | macOS | Intel | NVIDIA |
|---|---|---|---|---|
| Today | vulkan | metal | vulkan | vulkan |
| M0 | vulkan default; T-CPU via `ZINC_RT_TIER=t_cpu` | metal | vulkan | vulkan |
| M1 | zinc_rt (T2) | metal | vulkan | vulkan |
| M2 | zinc_rt | zinc_rt (T-Metal) | vulkan | vulkan |
| M3–M6 | zinc_rt | zinc_rt | vulkan | vulkan |
| M7 | zinc_rt | zinc_rt | zinc_rt (T-Intel) | vulkan |
| M8 | zinc_rt | zinc_rt | zinc_rt | zinc_rt (T-CUDA) |

At every row, `-Dbackend=vulkan` keeps working. If ZINC_RT regresses or fails on a specific user's hardware, they have a documented one-flag fallback.

### 24.3 The op-by-op port

For each IR opcode:

1. **Implement the IR node** in `zinc_rt/ir/op.zig`.
2. **Add an emitter** in `forward_zinc_rt.zig`.
3. **Wire T-CPU implementation** (in `zinc_rt/isa/cpu_zig/`).
4. **Wire one GPU tier** (T2 first, then T1, then T-Metal at M2).
5. **Validate** — cross-tier diff against T-CPU.
6. **Optimize** — port the kernel from SPIR-V/MSL to native ISA at M4+.

### 24.4 What stays put forever

* Tokenizer.
* GGUF parsing.
* HTTP server, OpenAI API, chat UI.
* `build.zig` (touched only to add the `zinc_rt` target).

---

## 25. Phased Milestones

Each milestone has: scope, exit criteria, expected calendar weeks for one focused engineer, and risk level.

| Milestone | Scope | Exit criterion (Qwen 3.6 35B-A3B, R9700) | Eng-weeks | Risk |
|---|---|---|---:|---|
| **M0** | `src/zinc_rt/` scaffolding; IR opcode table; verify pass; **T-CPU tier complete**; `ZINC_RT_TIER=t_cpu` runs end-to-end on a laptop | T-CPU output matches `forward.zig` token-for-token on test suite. Laptop dev workflow works. | 5 | Low |
| **M1** | T2 UMQ backend; PM4 packet builder; one-submit-per-decode-token; default-on for RDNA + kernel 6.16+ | ≥ 140 tok/s decode, output bit-identical to T-CPU. | 5 | Med |
| **M2** | T1 KFD-PM4 backend; **T-Metal tier** (folds in existing Metal work); IR-lowering of all decode ops; BAR-backed output ring; chunked prefill | ≥ 165 tok/s decode RDNA; ≥ 250 tok/s prefill (closes hybrid-MoE+SSM gap). Apple Silicon: metal parity. | 7 | Med |
| **M3** | Continuous-batching scheduler (16 concurrent slots); paged KV v2; prefix caching | ≥ 195 tok/s single-slot, ≥ 760 tok/s aggregate at 4 slots, ≥ 30% prefix hit rate on chat replays. | 5 | Med |
| **M4** | Hand-written ISA kernels for top-10 ops; WMMA on prefill path; cross-layer prefill fusion | Decode ≥ 220 tok/s; prefill ≥ 350 tok/s. WMMA path activates on RDNA4. | 8 | High |
| **M5** | Megakernel for decode; persistent input/output rings; zero-submit decode | Decode ≥ 240 tok/s; idle-to-streaming latency < 20 µs. Megakernel survives 100k tokens with no host intervention. | 10 | High |
| **M6** | Speculative decoding (EAGLE-3 draft); RadixAttention prefix tree; quantized KV via TurboQuant | 1.5× decode speedup on aligned prompts; KV memory −60% at iso-accuracy. | 8 | Med |
| **M7** | **T-Intel** backend (Xe2); IR vendor-neutral validation; Intel users get a direct-submission tier (Vulkan stays as their fallback) | Qwen 3 8B runs on Intel Arc B770 at ≥ 50 tok/s decode via T-Intel; same model via `-Dbackend=vulkan` keeps working with no regression. | 8 | High |
| **M8** | **T-CUDA** backend; NVIDIA users get a CUDA Driver API tier (Vulkan stays as their fallback) | NVIDIA path works on at least one consumer card via T-CUDA; `-Dbackend=vulkan` build still passes CI on NVIDIA. | 6 | Med |

Total: ~62 engineer-weeks to M8. M0–M3 is ~22 weeks and lands the user-facing win. **The Vulkan backend is never removed** — it remains in `src/vulkan/`, `src/compute/forward.zig`, and `src/shaders/*.comp` as a peer of ZINC_RT, kept in CI indefinitely.

### 25.1 The first commit

Before any code, the agent executing this design files a stub `zinc_rt/engine.zig`:

```zig
// SPDX-FileCopyrightText: ZINC Authors
//! ZINC_RT — the ZINC Runtime
//! Design: docs/ZINC_RT_DESIGN.md
const std = @import("std");

pub const Tier = enum {
    t1_pm4,
    t2_umq,
    t_cpu,
    t_metal,
    t_intel,
    t_cuda,
};

pub const Engine = struct {
    tier: Tier,

    pub fn init(allocator: std.mem.Allocator, opts: InitOptions) !Engine {
        // M0: only T-CPU is implemented. T1/T2/T-Metal land at M1/M2.
        return error.NotImplemented;
    }
};

pub const InitOptions = struct {
    forced_tier: ?Tier = null,
    max_concurrent_slots: u32 = 4,
    chunk_cap: u32 = 8192,
};
```

This is the first PR. It declares the namespace and locks the design.

### 25.2 What the agent ships per milestone

* Passing `zig build test`
* Passing performance gate
* Docs update — this file annotated with achieved numbers
* Blog post in `writing/` cadence

---

## 26. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| KFD ABI changes break T1 across kernel update | Med | High | T2 fallback; pinned kernel version in CI |
| UMQ has functional bugs on early 6.16 releases | Med | Med | T1 fallback; auto-detect at startup |
| WMMA path doesn't beat subgroupAdd on prefill | Low | Med | Keep subgroupAdd path; measure |
| Megakernel hits a watchdog timeout | Low | High | UMQ supports long-running kernels; configure timeout; alive-pings |
| Hand-written ISA kernels regress relative to glslc | Med | Low | A/B harness; revert per-kernel |
| Scheduler bugs cause head-of-line blocking | Med | High | Fuzz tester; lock-free invariants; CI stress test |
| Output ring overflows under bursty workload | Med | Med | Sized for 2× steady-state peak; backpressure via slot-table flag |
| `forward_zinc_rt.zig` too coupled to RDNA4 assumptions | Med | High | T-CPU mandatory for every op; cross-validation in CI |
| KFD requires CAP_SYS_NICE on some kernel configurations | Low | Low | Probe at startup; actionable error |
| Linux 6.16's UMQ ioctl numbers shift in 6.17 | Low | Med | Probe + ABI version check |
| Vulkan backend rots from neglect once ZINC_RT becomes the default | Med | Med | Vulkan stays in CI indefinitely (§23.1 #4 — cross-backend logit diff). Treat any Vulkan regression as a release blocker, same as ZINC_RT. |
| T-Metal regresses Apple performance during fold-in | Med | High | Standalone metal backend stays buildable as `-Dbackend=metal` indefinitely; revert default to it if T-Metal regressions exceed 5%. |
| Two-backend maintenance burden grows unbounded | Low | Med | Both backends share GGUF parser, tokenizer, server, scheduler — only the dispatch + kernels differ. ~15% extra maintenance cost is the price of always having a working fallback. |

---

## 27. Open Questions

1. **Should T1 or T2 be the steady-state default on M2 ship?** Decide at M2 close based on kernel-version adoption.
2. **Q4_K-on-the-fly dequant into WMMA tiles vs pre-dequant to a staging buffer?** Measure.
3. **Per-slot vs shared MoE expert dispatch in the megakernel?** Profile.
4. **Should the IR be exposed as a public C ABI?** Defer to M6.
5. **Default chunk-cap for prefill?** 8192 is a guess; runtime knob with per-(model, GPU) defaults.
6. **TurboQuant integration timing.** Wired into ZINC_RT's KV path at M6.
7. **Should the standalone Metal backend keep its build flag indefinitely, or fold into T-Metal completely at some point?** Default: keep `-Dbackend=metal` indefinitely as a peer of `-Dbackend=vulkan`. Revisit at M5 close — if T-Metal has zero regressions for two months, we *can* deprecate the standalone flag, but we don't have to.

---

## Appendix A — PM4 packet quick-reference

(All packets are little-endian. `PKT3(opcode, count)` = type-3 packet, body length = `(count+1) * 4` bytes after header.)

| Packet | Opcode | Body | Used for |
|---|---|---|---|
| `NOP` | 0x10 | 0+ dwords | padding |
| `SET_CONTEXT_REG` | 0x69 | reg-offset, value... | one-time setup |
| `SET_SH_REG` | 0x76 | reg-offset, value... | per-dispatch user data |
| `SET_UCONFIG_REG` | 0x79 | reg-offset, value... | one-time scratch / EOP buffer |
| `DISPATCH_DIRECT` | 0x15 | dim_x, dim_y, dim_z, dispatch_init | every kernel launch |
| `ACQUIRE_MEM` | 0x58 | coher_cntl, size_lo/hi, base_lo/hi, poll_interval | cache invalidate |
| `RELEASE_MEM` | 0x49 | event, data_sel, addr_lo/hi, value_lo/hi | fence signal |
| `WAIT_REG_MEM` | 0x3C | engine_sel, mem_space, function, addr_lo/hi, ref, mask, poll_interval | poll for fence |
| `WRITE_DATA` | 0x37 | engine_sel, dst_sel, addr_lo/hi, data... | scribble fence value |
| `EVENT_WRITE` | 0x46 | event_type, ... | barrier event |

The full PM4 packet definitions are in `<linux>/drivers/gpu/drm/amd/include/asic_reg/gfx10/pm4_pfp.h` and `pm4_compute.h`. Tinygrad's `runtime/autogen/pm4.py` is a tested user-space copy.

---

## Appendix B — Full IR opcode table

Status: M = milestone the opcode becomes mandatory.

| Opcode | Category | Inputs | Outputs | Notes | M |
|---|---|---|---|---|---|
| `EMBED` | input | token_id, table | hidden | CPU → GPU at M2 | M0 |
| `RMS_NORM` | norm | x, weight | y | eps push-const | M0 |
| `RMS_NORM_FUSED_QKV` | norm+proj | x, norm_w, qkv_w | q,k,v | replaces 4 ops | M2 |
| `RMS_NORM_FUSED_MLP_GATE_UP` | norm+proj | x, norm_w, gate_w, up_w | gate,up | dense FFN | M2 |
| `ROPE` | attn | q or k | rotated | partial-dim variants | M0 |
| `RMS_NORM_FUSED_ROPE_KV_WRITE` | attn | qkv, norm_w, freq, page_table | q (out) | already shader-fused | M2 |
| `FLASH_ATTN` | attn | q, kv_cache, page_table | attn_out | decode | M0 |
| `FLASH_ATTN_BATCHED` | attn | q (n queries), kv_cache, page_table | attn_out | prefill / multi-query | M2 |
| `MOE_GATE_TOPK` | moe | hidden, gate_w | logits, ids, weights | on-GPU softmax | M0 |
| `MOE_GATE_UP` | moe | hidden, gate_w, up_w, ids | gate_out, up_out | batched per expert | M0 |
| `MOE_SWIGLU` | act | gate, up | y | | M0 |
| `MOE_DOWN_ACC` | moe | x, down_w, ids, weights, hidden | hidden (in-place) | fused weighted-acc | M0 |
| `SHARED_EXPERT` | moe | hidden, sh_gate_up_w, sh_down_w | hidden (in-place) | always-active | M0 |
| `SSM_CONV1D` | ssm | qkv, conv_w, state | conv_out | causal | M0 |
| `SSM_DELTA_NET` | ssm | conv_out, abc_w, state | y, state' | recurrent | M0 |
| `SSM_GATED_NORM` | ssm | x, gate_w | y | | M0 |
| `RESIDUAL_RMS_NORM` | residual+norm | x, residual, norm_w | y | shader-fused | M0 |
| `LM_HEAD` | output | x, output_w | logits | | M0 |
| `ARGMAX` | sample | logits | token | | M0 |
| `SAMPLE` | sample | logits, seed, temp, top_p | token | | M2 |
| `KV_WRITE_BATCHED` | kv | k, v, page_table | (none) | prefill batch | M2 |
| `MATMUL_WMMA_F16` | matmul | a, b | c | RDNA4 + Xe2 + Apple + CUDA | M4 |
| `MATMUL_WMMA_Q4K` | matmul | a (q4k), b | c | dequant on the fly | M4 |
| `BARRIER` | sync | (none) | (none) | hint, may be elided | M0 |
| `STREAM_OUT` | output | token, slot_id | (none) | host ring write | M3 |
| `LOAD_REQUEST_STATE` | sched | slot_id, field | scalar | from slot table | M3 |
| `STORE_REQUEST_STATE` | sched | slot_id, field, value | (none) | into slot table | M3 |
| `VERIFY_K` | spec | k draft tokens, kv_cache | accept_len | spec decoding | M6 |

---

## Appendix C — Source-tree diff plan

**Commit 1 — M0 scaffolding:**
* +`src/zinc_rt/engine.zig` (stub)
* +`src/zinc_rt/ir/op.zig`
* +`src/zinc_rt/ir/graph.zig`
* +`src/zinc_rt/ring/mod.zig`
* +`src/zinc_rt/ring/cpu.zig` (T-CPU implementation)
* +`src/zinc_rt/isa/cpu_zig/*.zig` (T-CPU kernel implementations)
* +`src/zinc_rt/tests/ir_smoke.zig`
* `build.zig` updated to compile `src/zinc_rt/*`
* `src/gpu/interface.zig` updated for backend dispatch (`zinc_rt`/`vulkan`/`metal`)

**Commit 2 — M0 forward_zinc_rt:**
* +`src/compute/forward_zinc_rt.zig` — emits IR for the existing decode graph
* `gpu/interface.zig` routes `-Dbackend=zinc_rt` to `forward_zinc_rt`

**Commit 3 — M0 validation:**
* +`tests/zinc_rt_vs_vulkan.test.ts` — runs the same prompt on T-CPU and the Vulkan backend, asserts logit equality. This test stays in CI permanently as the cross-backend gate (§23.1 #4).
* CI updated to require both backends passing on every PR

**Commit 4–6 — M1 UMQ:**
* +`src/zinc_rt/kmd.zig` — amdgpu ioctl shim
* +`src/zinc_rt/ring/umq.zig`
* +`src/zinc_rt/ring/packet.zig`
* Default-on for kernel 6.16+
* `docs/ZINC_RT_DESIGN.md` annotated with M1 achieved numbers

**Commit 7–9 — M2 PM4 + T-Metal + BAR + chunked prefill:**
* +`src/zinc_rt/ring/pm4.zig` — KFD path
* +`src/zinc_rt/ring/metal.zig` — wraps existing src/metal/ work
* +`src/zinc_rt/isa/apple_msl/` — relink to existing src/shaders/metal/
* +`src/zinc_rt/mem/bar.zig`
* +`src/zinc_rt/sched/batcher.zig` initial chunked-prefill plumbing
* `docs/ZINC_RT_DESIGN.md` annotated with M2 achieved numbers

**Commit 10+ — M3 CB:**
* +`src/zinc_rt/sched/{slot,policy,batcher}.zig` full CB
* +`src/zinc_rt/mem/pagetable.zig` paged KV v2
* prefix cache hash table on the host

**Commit cluster — M4 ISA kernels:**
* +`src/zinc_rt/isa/gfx1201/dmmv_q4k.s`
* +`src/zinc_rt/isa/gfx1201/flash_attn.s`
* ... etc.
* +`tools/zinc-rt-asm.zig` — assembler/lower wrapper

**Commit cluster — M5 megakernel:**
* +`src/zinc_rt/isa/gfx1201/megakernel_qwen36.s`
* +`src/zinc_rt/ring/persistent.zig` (host-side input/output ring runtime)

**Commit cluster — M6 spec decoding:**
* +`src/zinc_rt/spec/eagle.zig`
* +`src/zinc_rt/ir/verify_k.zig`

**Commit cluster — M7 Intel:**
* +`src/zinc_rt/ring/i915.zig`
* +`src/zinc_rt/isa/xe2hpg/`

**Commit cluster — M8 NVIDIA tier:**
* +`src/zinc_rt/ring/cuda.zig`
* +`src/zinc_rt/isa/sm_90/`
* NVIDIA users get a direct ZINC_RT tier alongside the existing `-Dbackend=vulkan` option
* CI now runs both `vulkan` and `zinc_rt` builds on RDNA, Intel, and NVIDIA hardware
* **Nothing is deleted.** The Vulkan and standalone Metal backends remain in place permanently.

---

## Closing

ZINC_RT is the ZINC Runtime — ZINC's own userspace GPU layer, in the same OS-level category as the CUDA Runtime or HSA Runtime, scoped to inference and owned end-to-end. It exists because Vulkan is the wrong shape of API for ZINC's hot path. ZINC_RT is not a "thin Vulkan-alike"; it is a workload-specific submission and execution stack designed around LLM inference on RDNA-first GPUs, with first-class tiers for Apple Silicon, Intel Arc, and NVIDIA.

**The Vulkan backend is not going away.** It remains a peer of ZINC_RT, selected via `-Dbackend=vulkan`, tested in CI on every PR, and shipped to every user. ZINC ships *two* GPU paths long-term: ZINC_RT for the lowest-overhead route to peak performance, Vulkan as the broadly compatible, well-trodden fallback that every Linux GPU user has worked with for a decade. The two share GGUF parsing, tokenizer, server, scheduler, model catalog — only the GPU dispatch and kernels differ. The cost of keeping both is modest; the benefit (always a working fallback when ZINC_RT regresses or hits new hardware) is permanent.

The current ZINC + Vulkan stack peaks at 117 tok/s decode and 31 % bandwidth utilization on R9700. ZINC_RT, fully realized, targets 240 tok/s decode and 65 % bandwidth utilization, with continuous-batching aggregate throughput approaching 1 000 tok/s on 4 concurrent slots — outperforming vLLM on this hardware class while running in a single Zig binary with no Python and no ROCm.

This document is the contract. The agent should follow §25 milestone-by-milestone, annotating this file as each gate is cleared. Open questions in §27 are decisions the agent should escalate, not invent. **A milestone is not "complete" unless both `-Dbackend=zinc_rt` AND `-Dbackend=vulkan` build cleanly and pass CI.**
