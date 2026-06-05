---
title: "The first ZINC_RT model slices that actually count"
seoTitle: "ZINC_RT Direct GPU Model Slices on RDNA4"
date: "2026-06-04"
tags:
  - zinc
  - zinc-rt
  - rdna4
  - amd
  - amdgpu
  - pm4
  - local-llm
  - llm-inference
  - gpu-runtime
  - dmmv
  - q8-0
  - ssm
keywords:
  - ZINC_RT
  - ZINC_RT direct GPU runtime
  - AMDGPU CS local LLM inference
  - RDNA4 direct GPU runtime
  - PM4 packet LLM inference
  - Q8_0 DMMV kernel
  - SSM alpha beta row range
  - direct model slice
  - host assisted GPU decode
  - local LLM runtime
  - Radeon AI PRO R9700 inference
faqs:
  - question: "Is ZINC_RT faster than the Vulkan backend now?"
    answer: "No. This is not a speed claim. The current ZINC_RT path is still host-assisted and exists to prove direct AMDGPU command-stream execution can consume real model values safely before the runtime tries to replace larger parts of the Vulkan decode graph."
  - question: "What does a consumed direct model slice mean?"
    answer: "It means a direct AMDGPU CS kernel computes values from real model weights and activations, validates them against the CPU oracle when required, writes them into the live decode buffers, and the next stage of inference uses those GPU-produced values."
  - question: "Why use a one-wave Q8_0 DMMV kernel instead of one workgroup per row?"
    answer: "Earlier row-parallel attempts depended on unvalidated workgroup-id delivery through system SGPRs and left outputs at NaN sentinels. The one-wave path uses the lane id inside a single wave64, so each lane owns one row without depending on the broken TGID path."
excerpt: "ZINC_RT is still not the fast backend. The important change this week is smaller and more useful: direct AMDGPU CS kernels are starting to compute real model rows that the live decode path actually consumes. This post explains the difference between a diagnostic GPU probe and a consumed model slice, why the first row-parallel DMMV attempts failed, why the one-wave Q8_0 path works, and what this says about the road from host-assisted decode to a real direct runtime."
seoDescription: "How ZINC_RT began consuming real direct GPU model slices on RDNA4 with PM4, AMDGPU CS, and a one-wave Q8_0 DMMV row-range kernel."
---

Quick answer: today the interesting ZINC_RT story is not "we beat Vulkan." We did not. The interesting story is that ZINC_RT has started consuming direct GPU model slices that actually count.

That phrase needs unpacking, because it is the difference between a runtime demo and an inference engine.

A diagnostic GPU probe proves that a command stream can ring the doorbell, launch a tiny shader, write memory, and retire a fence. That is necessary bring-up work, but the model does not care. If the shader computes an isolated value and the runtime throws it away, the benchmark can still be almost entirely CPU-backed.

A consumed model slice is stricter. A direct AMDGPU command-stream kernel reads real model weights and the current activation, computes rows from the same projection the decoder needs, validates them when the path is not trusted yet, writes those values into the live decode buffers, and lets the next stage use them. If the GPU result is wrong, the text can change. That is why it counts.

That is the line ZINC_RT started crossing this week.

## The honest state

ZINC_RT is still an opt-in runtime. The production backend for published performance numbers is still Vulkan on AMD and Metal on Apple Silicon. The broader performance picture is in the [June 1 ZINC performance overview](/blog/2026-06-01-zinc-performance-where-it-is-fast-and-where-it-is-not).

The point of ZINC_RT is different. The [runtime-below-Vulkan post](/blog/2026-05-12-the-runtime-below-vulkan-that-local-llms-needed) explained the motivation: once the kernels get fast enough, the runtime layer starts showing through the profile. The longer [ROCm vs Vulkan vs ZINC_RT decision record](/blog/2026-05-18-inside-the-decision-to-write-our-own-gpu-runtime-for-local-llm-inference) explained why the direct path exists beside Vulkan instead of replacing it overnight.

This week is the first practical answer to the obvious question after those posts: what is the smallest unit of real model work ZINC_RT can own?

The answer is not "the whole token." Not yet. It is a DMMV row range.

## What changed

The relevant decode path is the Qwen 3.6 hybrid model path, where most layers are SSM layers and each decode token needs small source-format projections for state updates. The current ZINC_RT bridge is still host-assisted, but the tracked decode slice can now hand specific row ranges to direct AMDGPU CS kernels.

The important pieces are:

| Slice | Format | What changed |
| --- | --- | --- |
| LM-head prefix | Q4_0 | A 64-row direct prefix can produce scores that the selected token path can consume for low-id tokens. |
| SSM alpha/beta | Q8_0 | The tracked decode slice can consume paired alpha and beta row ranges from direct GPU DMMV. |
| SSM alpha/beta parallel64 | Q8_0 | A one-wave wave64 kernel now computes 64 rows with one lane per row for the 32-alpha plus 32-beta shape. |

The last row is the most interesting. It is the first row-parallel-ish source-format DMMV slice that avoids the broken multi-workgroup path and still feeds the live model path.

The shape is intentionally narrow: 64 rows, Q8_0 weights, `cols` divisible by 32, one wave, one lane owns one row. For the current SSM alpha/beta slice, that is exactly enough: 32 alpha rows plus 32 beta rows. The GPU output is not a side log. It is copied into the live alpha/beta buffers after validation, and the decode step continues with those values.

That does not make the whole runtime fast. It makes the runtime honest.

## Why the first row-parallel path failed

The obvious way to make a row-range DMMV less silly is one workgroup per row.

That attempt failed.

The earlier direct DMMV row-range probes computed compact Q4_0 and Q8_0 ranges by letting one GPU workitem serially loop over rows. That was useful for correctness, but not a real shape for performance. The next attempt added gfx1201 kernels that used `workgroup_id_x` as the row id and dispatched `rows` workgroups.

The symptom was blunt: outputs stayed at the NaN sentinel. Row 0 was not even finite. That means the failure happened before numeric validation could say anything interesting about quantized math.

The next diagnostic dumped candidate SGPRs for several workgroups. It proved two things at once:

1. The shader dispatched and wrote memory. The fixed marker and shader-written signal matched.
2. The expected system SGPRs did not contain workgroup ids `0, 1, 2, 3` under the packet shape we were using.

That matters because it rules out a lazy explanation. The problem was not simply "the command stream did not run" or "the CPU read before the GPU wrote." The command stream ran. Memory became visible. The row id just was not where the kernel assumed it was.

There was also a tempting packet-side fix: replace the existing async `WRITE_DATA` completion signal with a stronger memory-sync-shaped signal. That failed too. The signal itself stopped matching, and all shader-backed direct model-slice evidence disappeared from the run. The working path stayed on the known signal ABI.

Those failures are useful because they narrowed the problem. Multi-workgroup DMMV is blocked on the correct gfx1201 system-SGPR and packet setup for workgroup ids. It is not blocked on Q8_0 math.

## The one-wave workaround

The workaround is modest: do not ask for workgroup ids yet.

The new Q8_0 parallel64 kernel launches one wave64 and uses the lane id as the row id. Each lane owns one output row, then serially walks the K dimension for that row. In other words, the row id comes from `v0`, not from an unproven system SGPR.

That gives the runtime three practical benefits:

| Problem | One-wave answer |
| --- | --- |
| Workgroup id delivery is not validated | Avoid it entirely for this shape. |
| Serial row loop is too artificial | Compute 64 rows in parallel across lanes. |
| The model needs a small paired alpha/beta slice | Use exactly 32 alpha rows plus 32 beta rows. |

This is not the final DMMV kernel ZINC_RT needs. It is not tiled across K. It does not solve Q4_K. It does not replace the Vulkan DMMV family. But it turns a stalled bring-up problem into a consumed source-format slice, and that is the right kind of progress.

The kernel is deliberately simple enough to audit. Its ABI is:

```text
s[0:1] = input f32 vector pointer
s[2:3] = output f32 row-result pointer
s[4:5] = Q8_0 weight rows pointer
s6     = cols, multiple of 32
s7     = rows, currently 64
v0     = workitem_id_x, row id
```

The packet side switches to a 64-thread dispatch for the parallel path, uses the parallel shader blob, and signals completion with the release-memory signal path. The forward path only takes it when the combined alpha/beta row count is exactly 64. Otherwise it stays on the older serial row-range path.

That shape is conservative. It means the path can be enabled without pretending it generalizes to every tensor.

## Why this counts as model work

The runtime tracks these direct slices with annoying names on purpose:

```text
direct_compute_ops
direct_compute_kind
direct_decode_model_slices
real_model_slice
consumed_gpu_model_value
```

Those fields are there to stop us from lying to ourselves.

A GPU probe can retire and still leave `real_model_slice=0`. A debug kernel can compute a value and still leave `consumed_gpu_model_value=0`. A shortcut can make text coherent while the actual model math stays on the host. None of those are enough for M1.

The consumed row-range path has a higher bar. It uses the real source-format tensor bytes, checks shape and row bytes, computes the CPU oracle until trust is earned, validates finite output and delta thresholds, then writes the GPU values into the decode buffers. Only then does it increment the model-slice counters.

That is the discipline ZINC_RT needs because direct runtimes are easy to fake accidentally. A custom runtime has a lot of ways to look alive:

- ring buffers advance
- fences retire
- command packets parse
- tiny kernels write markers
- text remains coherent because the CPU path still does the hard part

The only metric that matters at this stage is whether the GPU produced a value the model actually used. Everything else is scaffolding.

## The deeper lesson

The direct-runtime story is moving from architecture to contracts.

The architecture was the easy part to write about: no Vulkan, PM4 packets, user-mapped rings, eventually user-mode queues, eventually persistent decode. Those ideas are in the [ZINC_RT design doc](/zinc/docs/zinc-rt-design), and they are still the reason the runtime exists.

The contracts are harder:

| Contract | What has to be true |
| --- | --- |
| ABI contract | The shader receives user SGPRs and lane ids exactly where the packet builder says it will. |
| Memory contract | GPU writes are visible before the CPU consumes output. |
| Math contract | Source-format Q8_0/Q4_0 rows match the CPU oracle within a fixed tolerance. |
| Model contract | The direct result feeds the live decode path, not a diagnostic side channel. |
| Reporting contract | Benchmarks say whether the path was host-assisted, shortcut-free, and model-slice-consuming. |

The failed TGID attempts were contract failures. The one-wave Q8_0 slice is a contract success in a narrow shape.

That is why this is worth a post. Not because 64 rows change the performance story by themselves, but because the runtime now has a way to promote direct GPU work one verified slice at a time.

## What comes next

The next work should stay boring and exact.

First, keep expanding the consumed SSM slices. The next edge is QKV row ranges: more rows, earlier in the SSM layer, still small enough to validate against the CPU oracle. That is the right target because it moves direct execution upstream without requiring the whole graph to move at once.

Second, fix multi-workgroup row ids properly. The one-wave path is a useful escape hatch, not a substitute for understanding gfx1201 system SGPR setup. A future DMMV kernel needs one workgroup per row or one workgroup per row block. That requires workgroup id delivery to be boring and tested.

Third, keep the reporting strict. If a run says `host_assisted_model_slice`, it should mean exactly that. If it says `real_model_slice=1`, the GPU value should have crossed into the model. If it says `cpu_fallback`, no amount of coherent text should make the run sound like a direct runtime win.

The best version of ZINC_RT is not built by jumping straight from a scalar host-assisted path to a megakernel. It is built by moving one dependency at a time from "the CPU computed this" to "the GPU computed this and the model used it."

The first slices are small. That is fine. Small is how a direct runtime becomes trustworthy.
