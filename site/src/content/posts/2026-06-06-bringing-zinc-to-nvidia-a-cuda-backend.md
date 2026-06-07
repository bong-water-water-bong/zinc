---
title: "Bringing ZINC to NVIDIA: a CUDA backend, because WSL2 only speaks CUDA"
date: "2026-06-06"
tags:
  - zinc
  - nvidia
  - cuda
  - rtx-5090
  - blackwell
  - zig
  - llm-inference
  - gpu-kernels
  - qwen3-6
  - wsl2
keywords:
  - NVIDIA CUDA LLM inference
  - RTX 5090 local inference
  - Blackwell sm_120 inference
  - CUDA backend in Zig
  - WSL2 CUDA vs Vulkan
  - __dp4a int8 dot product
  - NVRTC runtime compilation
  - local LLM NVIDIA GPU
  - Vulkan to CUDA kernel port
  - ZINC CUDA backend
  - MoE CUDA kernels
  - delta-net SSM CUDA
excerpt: "ZINC is getting a fourth GPU backend: native CUDA for NVIDIA. The surprise that forced it — on Windows + WSL2, NVIDIA exposes only CUDA, not Vulkan, so the one Vulkan device ZINC can see is a CPU. The reassurance that makes it tractable — ZINC's matmuls are int8 dot products, not tensor-core GEMMs, so they map 1:1 onto __dp4a. This is the plan, what we've already validated on an RTX 5090, and the road to a first token."
seoTitle: "NVIDIA CUDA LLM Inference: ZINC's CUDA Backend Plan"
seoDescription: "Why ZINC needs a native CUDA backend for NVIDIA (WSL2 hides the Vulkan driver), what's already validated on an RTX 5090, and the milestone plan from __dp4a to a coherent Qwen3.6-35B token."
draft: false
---

To run local LLM inference on NVIDIA GPUs under Windows + WSL2, ZINC is getting a native CUDA backend rather than reusing its Vulkan path. The reason is not preference. It is that **WSL2 exposes only CUDA to the GPU, not Vulkan** — the sole Vulkan device the engine can enumerate is a software rasterizer running on the CPU. A native `src/cuda` backend is the only way ZINC's kernels ever touch a GeForce card on that box.

ZINC already runs on three backends: `vulkan` (AMD RDNA), `metal` (Apple Silicon), and `zinc_rt` (a from-scratch AMD direct-submission runtime). NVIDIA looked like the easy one. We have a working Vulkan backend; NVIDIA has excellent Vulkan compute drivers; surely you just point the existing backend at the green card and move on.

Then we tried, and the GPU was nowhere to be found.

This is the story of why the NVIDIA path is CUDA and not Vulkan, what we have already proven on an actual RTX 5090, and the milestone plan from "an `__dp4a` returns 70" to "Qwen3.6-35B emits a coherent token." It is a plan-in-progress post, not a ships-today post — the toolchain and the full primitive layer (C *and* Zig) are validated on the hardware; the build integration and the kernels come next.

<figure class="diagram-card diagram-wide">

| | AMD (Vulkan) | Apple Silicon (Metal) | NVIDIA (CUDA) |
|---|---|---|---|
| **API** | Vulkan 1.3 compute | Metal | CUDA Driver + Runtime |
| **Memory model** | Discrete VRAM + staging DMA | Unified, shared pages | Discrete VRAM + staging DMA |
| **Shader language** | GLSL 4.60 → SPIR-V (offline) | MSL (runtime compiled) | CUDA C → PTX via NVRTC (runtime) |
| **Int8 matmul primitive** | `dotPacked4x8AccSatEXT` | simdgroup int dot | `__dp4a` |
| **Subgroup width** | wave64 / wave32 | 32-lane simdgroup | 32-lane warp |
| **Async model** | queue submit + fences | `commitAsync`/`wait` | streams + events |
| **Reference to mirror** | — | **this one** | — |

  <figcaption>The CUDA backend mirrors Metal, not Vulkan. Metal already uses raw-pointer binds and an async commit/wait model that maps 1:1 onto CUDA streams and events. From Vulkan it borrows exactly one idea Metal never needed: explicit host-to-device staging, because CUDA has no unified memory.</figcaption>
</figure>

## The constraint: a Vulkan engine that can't use Vulkan

The deployment box is a Windows workstation running CUDA workloads through WSL2. NVIDIA's WSL passthrough is real GPU acceleration — but it is **CUDA-only**. The graphics/compute Vulkan stack does not come through.

We proved it empirically. Vulkan device enumeration on the box returns exactly one device:

```
vendor=0x10005  type=CPU  name=llvmpipe
```

`llvmpipe` is Mesa's software rasterizer. It runs Vulkan on the CPU. There is no NVIDIA Vulkan device to be found, because the driver files that would advertise one are not in the passthrough:

<figure class="diagram-card diagram-wide">

| In `/usr/lib/wsl/lib` (present) | Conspicuously absent |
|---|---|
| `libcuda.so` (CUDA driver) | `libGLX_nvidia.so` |
| `libnvidia-ml.so` (NVML) | `nvidia_icd.json` (the Vulkan loader manifest) |
| `libnvidia-gpucomp.so` | any NVIDIA Vulkan ICD |
| NVENC encode libs | a usable Vulkan compute queue |

  <figcaption>WSL2's GPU passthrough ships the CUDA stack and the NVML/encode libraries, but no NVIDIA Vulkan ICD. Without the ICD manifest, the Vulkan loader never even tries the GPU — it falls back to llvmpipe. Installing the ICD requires the operator/sudo, which is off the table on this box.</figcaption>
</figure>

The Vulkan loader discovers drivers through ICD manifest files in `/usr/share/vulkan/icd.d/`. On the box that directory is not empty — it ships `lvp`, `nouveau`, `radeon`, `intel`, `asahi`, `virtio`, and `gfxstream_vk` manifests — but there is **no proprietary `nvidia_icd.json`**, and the only manifest that can actually enumerate a device under WSL passthrough is `lvp` (lavapipe, the LLVM software rasterizer). `nouveau` is present but is not a working compute path for these cards under WSL; the proprietary NVIDIA ICD that would expose the real GPU simply isn't installed, and installing it requires the operator/sudo. Dozen (the Vulkan-on-D3D12 layer) is absent too, and would not run ZINC's compute shaders correctly even if it were present. ZINC is a Vulkan/Metal engine, and on this machine both doors are shut.

So either NVIDIA support means a CUDA backend, or it means nothing. We chose the backend.

## The reassuring result: ZINC's matmuls are `__dp4a`, not tensor cores

A new backend for a new vendor sounds like a rewrite. It is not, and the reason is a single architectural decision ZINC made years ago: **its quantized matmuls are integer dot products, not cooperative-matrix / tensor-core GEMMs.**

On Vulkan, every hot matmul-vector kernel is built on `GL_EXT_integer_dot_product` — specifically `dotPacked4x8AccSatEXT`, which multiplies two packed `int8×4` vectors and accumulates into `int32`. On AMD that lowers to `v_dot4_i32_i8`. On NVIDIA, the exact same operation is the CUDA intrinsic `__dp4a`. One instruction, same semantics, both vendors.

That means the matmul port is **mechanical, not creative**. We are not redesigning around `mma.sync` or `wmma`; we are re-expressing the same int8-dot kernels in CUDA C. The M0 smoke test on the box confirms the foundation:

```
device[0]: NVIDIA GeForce RTX 4090  cc=8.9  SMs=128  vram=25.8GB
vadd:  c[0]=3.0  c[N-1]=3.0  status=no error          # launch + H2D/D2H OK
dp4a:  1*5+2*6+3*7+4*8 = 70 (expect 70)                # __dp4a OK
libnvrtc.so.13  libcublas.so.13  libcudart.so.13       # runtime-compile + BLAS present
```

`70` is the whole thesis in one number. The instruction ZINC's GEMMs depend on works, runtime compilation is available, and the GGML quantization block layouts (Q4_K/Q5_K/Q6_K 256-element super-blocks; Q8_0/Q5_1 32-element; MXFP4) are **bit-for-bit identical** to what the Vulkan shaders already unpack. The dequant code ports verbatim.

## How ZINC picks a backend — and the one seam we change

Backend selection in ZINC is resolved at **compile time** (the inactive backend is never compiled into the binary), but today the discriminant is the target OS, not a flag:

```zig
// src/gpu/interface.zig — the real dispatcher, keyed on the OS
const is_metal  = (builtin.os.tag == .macos);
const is_vulkan = (builtin.os.tag == .linux);
pub const backend = if (is_metal) @import("../metal/device.zig")
                    else          @import("../vulkan/instance.zig");
```

This is the problem in one stanza: **Linux unconditionally means Vulkan.** CUDA also lives on Linux, so it cannot be selected by OS. The fix is to thread the existing `-Dbackend` build option (already an `enum { auto, vulkan, metal, zinc_rt }` in `build.zig`, but currently consumed only for `--version`) into `interface.zig` as an `is_cuda` discriminant, so `cuda` and `vulkan` coexist as two Linux backends chosen explicitly with `-Dbackend=cuda`.

Every site that today reads `if (gpu.is_vulkan) … else …` assumes exactly two backends and becomes three-way — a handful of spots in `main.zig` and `server/model_manager_runtime.zig`. Mechanical, but it has to be complete: miss one branch and you get a Vulkan loader on a CUDA build.

## The backend contract: mirror Metal, not Vulkan

ZINC has no backend vtable. The contract is a **duck-typed module surface** — the compute layer imports `device`, `buffer`, `pipeline`, and `command` modules and calls their methods directly. To add CUDA, you provide modules with the same method names and the compiler does the rest.

Metal is the reference to mirror, for two concrete reasons. First, Metal binds raw device pointers the way CUDA does (Vulkan hides everything behind descriptor sets). Second, Metal's async `commitAsync` / `wait` / `releaseCompleted` triad — the mechanism that overlaps GPU execution with CPU command-building — maps **1:1 onto CUDA streams + events**. From Vulkan we borrow exactly one thing Metal never needed: explicit H2D/D2H staging, because unlike Apple's unified memory, a discrete GeForce needs weights copied across PCIe.

<figure class="diagram-card diagram-wide">

| Metal (reference) | CUDA equivalent | Responsibility |
|---|---|---|
| `metal/shim.h` | `cuda/cuda_shim.h` | C ABI contract — the backend boundary |
| `metal/shim.m` (ObjC) | `cuda/cuda_shim.c` | Driver/Runtime impl: `cuCtx*`, `cudaMalloc`, `cuMemcpy*`, NVRTC, `cuLaunchKernel`, streams/events |
| `metal/c.zig` | `cuda/c.zig` | shared `@cImport` of the shim header |
| `metal/device.zig` | `cuda/device.zig` | context init + caps (SM count, compute capability, VRAM) |
| `metal/buffer.zig` | `cuda/buffer.zig` | `cudaMalloc` + pinned-host staging + `upload`/`download` |
| `metal/pipeline.zig` | `cuda/pipeline.zig` | NVRTC compile source → PTX → `CUfunction` |
| `metal/command.zig` | `cuda/command.zig` | a `CUstream`: `dispatch`=`cuLaunchKernel`; sync + async commit over `CUevent` |

  <figcaption>Seven files, each a direct analog of a Metal file that already works. The async command ring that overlaps exec with command-building — `[256]MetalCommand` pending in `forward_metal.zig` — gets mirrored in `forward_cuda.zig` over `CUstream` + `CUevent`.</figcaption>
</figure>

The dispatch ABI is deliberately uniform: every kernel is authored `__global__ void k(P0* b0, …, Push pc)` — buffers first, one trailing by-value push-constant struct. That is the same shape the Metal and Vulkan dispatch paths already use, so the host-side dispatch glue is shared in spirit across all three.

<figure class="diagram-card diagram-wide">

```
┌──────────────────────────────────────────────────────────────────┐
│                          ZINC Engine                             │
├────────────────────────┬─────────────────────────────────────────┤
│   Shared layers        │  Tokenizer, GGUF parser, HTTP API,     │
│                        │  model catalog, chat UI, sampling       │
├────────────────────────┼─────────────────────────────────────────┤
│   gpu/interface.zig    │  Compile-time backend switch            │
│                        │  (OS  +  -Dbackend)                     │
├───────────┬────────────┼───────────────┬─────────────────────────┤
│  Vulkan   │   Metal    │   zinc_rt     │   CUDA  (new)           │
│  (AMD)    │  (Apple)   │   (AMD)       │   (NVIDIA)              │
├───────────┼────────────┼───────────────┼─────────────────────────┤
│ instance  │ device     │ engine        │ device.zig    ✅        │
│ buffer    │ buffer     │ submit        │ buffer.zig    ✅        │
│ pipeline  │ pipeline   │ tiers         │ pipeline.zig  ✅        │
│ command   │ command    │               │ command.zig   ✅        │
│           │ shim.m     │               │ cuda_shim.c   ✅        │
├───────────┼────────────┼───────────────┼─────────────────────────┤
│ 110 GLSL  │ 175 MSL    │ —             │ ~14 → 110 .cu  ⏳       │
│ → SPIR-V  │ (runtime)  │               │ (NVRTC runtime)         │
├───────────┼────────────┼───────────────┼─────────────────────────┤
│ forward   │ forward_   │               │ forward_cuda.zig ⏳     │
│ .zig      │ metal.zig  │               │                         │
└───────────┴────────────┴───────────────┴─────────────────────────┘
```

  <figcaption>Four backends, one engine. Everything above the interface line is shared. The CUDA column is the new work: ✅ done / validated, ⏳ planned. The shim and all four Zig wrappers exist today and pass a smoke test on the 5090; what remains is build wiring, the kernels, and forward_cuda.zig.</figcaption>
</figure>

## Porting the kernels: 110 shaders, but ~14 for a first token

The Vulkan backend is **110 `.comp` shaders**, and they are the authoritative spec for the CUDA kernel set: GEMM/dequant (~58), MoE routing (10), SSM (9), norm/activation (13), attention (4), quantize (4), elementwise (6), RoPE (2), KV (2). You do not port all 110 to get a first token — you port roughly **14 unfused reference kernels** and validate correctness, then expand and fuse.

The port is variations on one cheat-sheet:

<figure class="diagram-card diagram-wide">

| Vulkan / GLSL | CUDA |
|---|---|
| `dotPacked4x8AccSatEXT` (int8 dot) | `__dp4a` |
| wave64 workgroup (`local_size_x=64`) | warp32 block — reuse the **existing wave32 fallback path** as the template |
| `subgroupAdd` / `subgroupClusteredAdd` | `__shfl_down_sync` reductions / `__reduce_add_sync` |
| `subgroupMax` / `subgroupShuffle` / `subgroupBroadcastFirst` | `__reduce_max_sync` / `__shfl_sync` |
| `subgroupBallot` / `subgroupElect` | `__ballot_sync` / `__activemask` |
| `vec4` / `uvec4` SSBO (128-bit loads) | `float4` / `int4` |
| push constants | kernel params or `__constant__` block |
| specialization constants (K=2048/4096/12288) | template params or runtime-tuned variants |
| `vkCmdDispatchIndirect` | host indirection (or device launch) |

  <figcaption>The single most useful fact: ZINC already has a wave32 fallback for AMD parts without wave64. That path — cross-subgroup merges through shared memory — is exactly the warp32 shape CUDA needs. The hard reduction-width problem was already solved once.</figcaption>
</figure>

We start with **NVRTC runtime compilation** — kernels live as `.cu` sources, compiled to PTX for whatever architecture is present at load (sm_120 on the 5090, sm_89 on the 4090) transparently. This mirrors Metal's runtime MSL compilation and keeps the bring-up loop tight: edit a kernel, rebuild the Zig binary, run. An offline `nvcc → cubin` step (parallel to the existing `glslc` shader build) is a later optimization, not a prerequisite.

The kernels we port last, and carefully: the tiled `mul_mm` / `*_full_dp4a` GEMMs (register blocking, shared-memory bank padding); `ssm_delta_net` (an autoregressive selective scan with register-resident per-row state and clustered subgroup reductions); `flash_attn` (paged KV, split-K, attention sinks); and the `softmax_topk` → route-pack → indirect-dispatch MoE chain. These are where wave64→warp32 numerics actually bite, so each is validated against the Vulkan/Metal reference output, not merely compiled.

## The target model: Qwen3.6-35B-A3B, a hybrid SSM + MoE

The bring-up target is `qwen36-35b-a3b-q4k-xl` — "Qwen3.6 35B-A3B," about **18 GiB**, which fits comfortably in the 5090's 32 GB. It is deliberately the hard case, not a toy: internally `.qwen2_moe` **with a state-space twist**.

- **Layer pattern.** `full_attention_interval = 4`, so **three of every four layers are delta-net SSM** and every fourth is attention. A CUDA backend that only did attention would stall on layer 1.
- **MoE everywhere.** Every layer has a Mixture-of-Experts FFN: top-k routed experts plus one sigmoid-gated shared expert.
- **Mixed quantization.** This is the part that bit the Metal backend hard, and will bite CUDA the same way:

<figure class="diagram-card diagram-wide">

| Tensor group | Quant |
|---|---|
| Experts — gate / up | **Q4_K** |
| Experts — down | **Q5_K** |
| Shared expert + SSM in/out | **Q8_0** |
| Norms, α/β, router | **F32** |
| conv1d, A_log | **F16** |
| LM head | **Q6_K** |

  <figcaption>The batched-MoE fast path is only legal if every quant format in every expert tensor is supported. Miss Q5_K and the whole layer drops to the slow per-expert path — which, as we learned on Metal, disables the fast path for the entire decode step. So Q4_K, Q5_K, Q6_K, and Q8_0 DMMV kernels are all M1 work, not nice-to-haves.</figcaption>
</figure>

## The roadmap: from `70` to a coherent token

<figure class="diagram-card diagram-wide">

| Milestone | Scope | Status |
|---|---|---|
| **M0 — toolchain** | `nvcc`, `__dp4a`, NVRTC/cuBLAS/cudart reachable on the box | ✅ **done** |
| **M0.5 — primitive layer** | shim + the four Zig wrappers (`device`/`buffer`/`pipeline`/`command`): device select, staged H2D/D2H, NVRTC compile for sm_120, buffers+push dispatch ABI, sync **and** async commit, `__dp4a` | ✅ **validated on 5090** — both the C smoke and a Zig smoke (through the wrappers, async path) PASS |
| **M1 — one correct token** | `forward_cuda.zig` minimal decode, ~14 unfused kernels, token-for-token parity vs Metal/Vulkan | ⏳ next |
| **M2 — full model** | paged KV pool, SSM conv/recurrent state rings, all quant formats, prefill, chat template, server path | ⏳ |
| **M3 — performance** | fused kernels, async stream/event ring, DP4a tiled GEMMs (optional `mma.sync` as a 5090-only win), Blackwell tuning; benchmark vs llama.cpp CUDA | ⏳ |

  <figcaption>The unusual shape here is M0.5: rather than build Zig wrappers and discover the C primitives don't work, we wrote the entire primitive layer in plain C first and proved it on the actual 5090. The risky, vendor-specific part is already retired.</figcaption>
</figure>

M1's forward order, concretely: embedding dequant (host gather) → `rms_norm_mul` → one templated **DMMV** covering Q4_K/Q5_K/Q6_K/Q8_0/F32 (it serves Q/K/V/O, SSM in/out, router, and the LM head) → attention (per-head qk-rmsnorm + partial/IM-RoPE + paged `kv_cache_write` + `flash_attn`, or naive `softmax(QKᵀ)V` for a single query) → SSM (`ssm_conv1d` + unfused `ssm_delta_net` + `ssm_gated_norm`) → MoE (`softmax_topk` + `dmmv_q4k_moe` gate/up + `swiglu` + `dmmv_q5k_moe` down + `moe_weighted_acc` + the Q8_0 shared expert with `sigmoid_scale_acc`) → residual `scale_accumulate` → `argmax`. Then we diff the chosen token against the Vulkan and Metal references on the same prompt. Same token, or it isn't done.

## Where we are right now

M0 and M0.5 are both validated on the hardware — and not just in C. `cuda_shim.c` implements the CUDA Driver API + NVRTC behind the same ABI as `metal/shim.h`, and a `smoke.c` proved it first. Then the four Zig wrappers — `device.zig`, `buffer.zig`, `pipeline.zig`, and `command.zig` — went on top, mirroring their Metal counterparts, and a `smoke.zig` exercises the whole stack *through the Zig abstraction*. On the 5090 (sm_120) it passes end-to-end:

```
device: NVIDIA GeForce RTX 5090  cc=120  SMs=170  vramGB=31
nvrtc: compiled vadd + dp4a_k for sm_120
vadd: c[1]=3 (expect 3)  c[100]=300 (expect 300)
dp4a (via abstraction, async path): 70 (expect 70)
RESULT: PASS
```

That is device selection (highest compute capability — 5090 over 4090), buffers staged across PCIe both directions, a kernel compiled at runtime for sm_120, the buffers+push dispatch ABI, the **async commit path** over `CUstream`/`CUevent`, and `__dp4a` — all green, all through Zig. The risky, vendor-specific part of the backend is retired.

What is *not* done yet is the integration: `build.zig` has no `cuda` module wiring and `gpu/interface.zig` has no `is_cuda` discriminant, so the wrappers currently live as a standalone smoke rather than a selectable backend. The immediate next steps are exactly those two seams — `configureCudaModule` + a `zig build cuda-smoke` target in `build.zig`, and the build-option selector in `interface.zig` — after which the kernel port (M1) begins. No `forward_cuda.zig` and no ported kernels exist yet; the first token is still ahead.

## The hard parts we already know about

No plan survives contact, but these are the named risks, not surprises:

- **wave64 → warp32 numerics.** Reductions change width when a 64-lane workgroup becomes a 32-lane warp. Float reduction order changes results. Every ported kernel is validated against the reference output, not just compiled clean. The existing AMD wave32 fallback path is the proven template for getting this right.
- **Host RAM, not VRAM, is the tight resource.** The box runs under a 24 GB cgroup cap on host memory. The 18 GiB model lives fine in the 5090's 32 GB VRAM, but host-side staging and mmap have to stay under that cap — weights stream to the device rather than fully materializing in host RAM.
- **Device order.** `CUDA_DEVICE_ORDER=PCI_BUS_ID` pins `cuda:0` to the 5090; non-login SSH doesn't inherit it, so the backend also selects by compute capability at runtime (which is why `initBest` probes for the highest-cc device).
- **Toolchain on the box.** Zig 0.15.2 isn't preinstalled — on-box builds just need the official tarball dropped into `$HOME` (no sudo anywhere in the path). The repo itself is public, so it clones over HTTPS with no auth.
- **Out of scope, on purpose.** `zinc_rt`'s `t_cuda` tier is a *different* idea — direct submission with no CUDA driver — and is not what this backend is. This plan rides the standard cudart/Driver API.

## The bigger picture

When ZINC started, the thesis was narrow: AMD consumer GPUs are ignored by the AI stack, and someone should fix that. Apple Silicon broadened it. NVIDIA — the vendor that is the opposite of ignored — broadens it again, and for an unexpected reason.

The interesting part of NVIDIA support is not "now it runs on the popular cards." It is what the WSL2 finding exposes: even on the most supported GPU in the world, the path you assume is open can be closed, and an engine that owns its kernels can route around it. A translation layer would have been stuck at `llvmpipe`. Because ZINC compiles its own kernels and chooses its own backend, the answer to "Vulkan is missing" is "then we speak CUDA" — and the matmuls come along for free, because they were int8 dot products all along.

Four backends. One engine. The next milestone is a single number: a Qwen3.6-35B token that matches the reference, emitted by an RTX 5090 that, as far as Vulkan is concerned, does not exist.

```bash
# the target, once M1 lands
zig build -Doptimize=ReleaseFast -Dbackend=cuda
./zig-out/bin/zinc chat
```
