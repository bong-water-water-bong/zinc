---
title: "How ZINC's RDNA4 batched prefill went from 42 to 208 tok/s"
seoTitle: "ZINC RDNA4 Batched Prefill: 42 to 208 tok/s"
date: "2026-06-05"
tags:
  - zinc
  - rdna4
  - amd
  - vulkan
  - prefill
  - qwen3-8b
  - gemma
  - local-llm
  - llm-inference
  - gpu-kernels
  - dmmv
  - q4-k
  - q6-k
keywords:
  - ZINC RDNA4 batched prefill
  - Qwen3-8B prefill R9700
  - Vulkan batched prefill local LLM
  - dmmv_q4k_batch_kpar
  - dmmv_q6k_batch_kpar
  - ZINC_BATCHED_PREFILL
  - MAX_COLS 40 RDNA4
  - Radeon AI PRO R9700 prefill
  - Q4_K Q6_K batched DMMV
  - local LLM time to first token
  - Gemma 4 batched prefill
faqs:
  - question: "What changed in ZINC's RDNA4 batched prefill path?"
    answer: "The Vulkan path stopped bypassing prefillBatched, learned to dispatch Q6_K batched projections, fixed stale GPU-argmax sampling after prefill, and replaced serial-over-K batched DMMV shaders with K-parallel Q4_K and Q6_K variants."
  - question: "How fast did the shipped RDNA4 batched prefill path get?"
    answer: "On Qwen3-8B Q4_K_M on an AMD Radeon AI PRO R9700, the 658-token prompt moved from 42.9 tok/s on the per-token path to 207.9 tok/s on the shipped batched path, a 4.85x speedup."
  - question: "Why did the first correct batched path run slower than per-token prefill?"
    answer: "The original batched shader reused weights across columns but assigned one thread to a whole output row and walked K serially. The per-token shader was K-parallel across a wave64. The final batched path had to keep weight reuse while restoring K-parallel row work and subgroup reductions."
  - question: "Does this solve Qwen 35B MoE plus SSM prefill?"
    answer: "No. This post is the dense Qwen3-8B and dense Gemma batched-prefill story. Qwen3.5/3.6 35B-A3B still needs batched MoE routing, per-expert accumulation, and block-resident SSM state before the same gate can come down."
excerpt: "The old RDNA4 batched-prefill docs note was really a blog post hiding in the docs tree. This is the cleaned-up version: how ZINC discovered that ZINC_BATCHED_PREFILL was a no-op, used validate mode to prove the forward pass was correct, fixed a stale GPU argmax sampler bug, replaced serial-over-K DMMV with K-parallel Q4_K/Q6_K batched shaders, and moved Qwen3-8B prompt ingestion on the R9700 from 42.9 to 207.9 tok/s."
seoDescription: "How ZINC's Vulkan RDNA4 batched prefill path moved Qwen3-8B on the Radeon AI PRO R9700 from 42.9 to 207.9 tok/s with K-parallel Q4_K and Q6_K DMMV shaders."
---

The short version: ZINC's RDNA4 batched-prefill path was supposed to be the obvious fix for time-to-first-token. For a while it was not a fix at all. `ZINC_BATCHED_PREFILL` was effectively a no-op on Vulkan, the first real run produced a Wikipedia image URL instead of "Paris", and the first correct version was still slower than the per-token path.

Then the useful thing happened: validation split the problem into correctness and throughput, and the real bottleneck became visible. The final shipped path moved Qwen3-8B Q4_K_M on the Radeon AI PRO R9700 from **42.9 tok/s** on a 658-token per-token prefill to **207.9 tok/s** with batched prefill. That is a **4.85x** speedup on the prompt work users actually wait on.

This is the post version of what used to live as a raw docs note at `/zinc/docs/rdna4-batched-prefill-2x/`. The old note was too narrative and too measured to be API documentation. It belongs in the blog.

## The measured result

All numbers below are from the same AMD Radeon AI PRO R9700 box: RADV `gfx1201`, 32 GB VRAM, 576 GB/s memory bandwidth, `ReleaseFast`, no profiling overhead, Qwen3-8B Q4_K_M.

For a 105-token prompt:

| path | prefill median |
| --- | ---: |
| per-token `prefillBatch` | 72.3 tok/s |
| batched, serial-over-K original shader | 61.5 tok/s |
| batched, Q4_K kpar plus Q6_K serial, `MAX_COLS=32` | 143.1 tok/s |
| batched, Q4_K plus Q6_K kpar, `MAX_COLS=32` | 172.9 tok/s |
| **batched, Q4_K plus Q6_K kpar, `MAX_COLS=40` shipped** | **187.1 tok/s** |

For a longer 658-token prompt:

| path | prefill median |
| --- | ---: |
| per-token `prefillBatch` | 42.9 tok/s |
| **batched, Q4_K plus Q6_K kpar, `MAX_COLS=40` shipped** | **207.9 tok/s** |

The longer prompt is the more important number. The per-token path degrades with prompt length because it runs a decode-shaped step over and over, paying fixed state-management and dispatch costs for every token. The batched path amortizes weight reads and command-buffer work across a chunk of prompt tokens, so it crosses 200 tok/s around the 300-token mark and stays there through at least 658 tokens.

The output text in the validated rows stayed boring: "The capital of France is Paris." That matters more than the speedup. A fast prefill path that changes the first sampled token is not a prefill optimization. It is a different model.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/rdna4-prefill-three-regimes.svg" alt="Two stacked bar charts comparing per-token DMMV, 32-column batched DMMV, and tiled GEMM for a Qwen3-8B prefill on RDNA4. The chart shows the large drop in repeated weight traffic and dispatch count when moving from per-token prefill to column-batched DMMV." loading="lazy" />
  <figcaption>The April 22 design argument was right about the first jump. The measured work here is the story of making that jump real in the Vulkan path.</figcaption>
</figure>

## Why this mattered

The direct target of the session was not Qwen3-8B. It was Qwen3.6-35B-A3B. On the same R9700, ZINC decode was already near llama.cpp parity, but prefill was badly behind. A typical chat session feels that gap as time-to-first-token.

The annoying discovery was that Qwen3.6 could not use the batched path yet. It is a hybrid MoE plus SSM architecture, and `canUseBatchedPrefillRdna` rejected any model with experts or SSM state. That is the subject of the later [Qwen 35B prefill gate post](/blog/2026-04-26-the-gate-that-keeps-qwen-35b-prefill-at-half-of-llama-cpp-on-rdna4).

But testing the dense model was not wasted work. Qwen3-8B was supposed to be the easy case. It had no MoE router, no SSM recurrence, and a clean dense transformer schedule. If the dense batched path was not helping, there was no honest path to opening the hybrid gate.

The dense path exposed three bugs and one wrong shader shape.

## The two prefill paths

The Vulkan backend has two relevant entry points:

| Function | Shape |
| --- | --- |
| `prefillBatch(state, prompt_tokens)` | The proven per-token path. It runs the decode graph once per prompt token and appends one KV entry at a time. |
| `prefillBatched(state, prompt_tokens)` | The intended prompt path. It processes a chunk of tokens in one command-buffer flow with batched projection, RoPE, KV write, and flash-attention kernels. |

On paper, the batched path saves the repeated weight reads that dominate prompt ingestion. Qwen3-8B has seven quantized projections per layer across 36 layers. A 105-token prompt asks the per-token path to reread the same projection weights 105 times. Batched DMMV reads each row once per prompt chunk and applies it to many activation columns.

That was the theory. The actual Vulkan path had to earn it.

## Bug 1: batched prefill was dead code

The first surprise from calling `prefillBatched` in the CLI and server paths was a compile error:

```text
src/compute/forward.zig:7525:36: error: no field named 'position'
    in struct 'compute.forward.InferenceEngine'
```

The body had been ported from Metal and still referenced `self.position`. Metal's engine has that field. Vulkan's engine keeps request position in `state.position`. Because the CLI and server both called the old per-token path directly, Zig's lazy compilation never compiled the dead function body.

The fix was simple: remove the stale `self.position` check, make `state.position` authoritative, and route the live CLI/server prefill calls through `prefillBatched` when the gate allows it.

The lesson was less simple. A string-marker regression test can prove a function name exists in source. It cannot prove the function is compiled, called, or numerically correct.

## Bug 2: Q6_K had a gate but no dispatcher

The next failure was `UnsupportedQuantType`.

`canUseBatchedPrefillRdna` accepted Q4_K and Q6_K projections, which is necessary for Q4_K_M checkpoints. In those models, `ffn_down` and `attn_v` are commonly Q6_K. But the batched dispatcher only selected the Q4_K pipeline:

```zig
pub fn recordBatchDispatchPush(...) !void {
    const pip = switch (quant_type) {
        .q4_k => if (self.pipeline_q4k_batch) |*p| p
                 else return error.UnsupportedQuantType,
        else => return error.UnsupportedQuantType,
    };
    ...
}
```

The gate said yes. The dispatcher said no. No logits.

The safe sequence was to tighten the gate while Q6_K was missing, add `dmmv_q6k_batch.comp`, plumb it through `DmmvDispatch`, then reopen the gate. That made Qwen3-8B Q4_K_M capable of reaching the batched body at all.

## Bug 3: the garbage output was the sampler, not the forward pass

Once Qwen3-8B finally ran through `prefillBatched`, the output looked like this:

```text
Prompt:   The capital of France is Paris, and the capital of Italy is Rome, ...
Expected: The capital of France is Paris...
Got:      ![](https://upload.wikimedia.org/wikipedia/commons/...)
```

That looked like a math bug. Wrong RoPE position, wrong causal mask, bad KV cache indexing, Q6_K decode error: all plausible.

Instead of guessing, the fix was to add `ZINC_BATCHED_PREFILL=validate`. Validate mode runs the batched path, snapshots the last-token logits, resets state, runs the per-token reference path, and diffs the logits. The first useful run printed:

```text
warn(forward): prefillBatched validate[ok]: last-token logits
    max_abs_diff=0.000000 at idx=0 (ref=14.0986 batched=14.0986)
    tol=0.001000 n_tokens=17
```

Across a 151,936-wide vocabulary, the max absolute logit difference was `0.000000`. The forward pass was bit-identical. The bug had to be after forward.

That led to `sampleGreedy`. The fast path reads `argmax_result_staging`, a host-visible buffer populated by the GPU argmax shader at the end of normal decode steps. The batched prefill tail ran the LM head and copied logits, but never ran the argmax shader or copied its result. The next sample read stale bytes, often token zero, which explains the leading `!`.

The fix was about 20 lines: record GPU argmax and copy `argmax_result_buf` to `argmax_result_staging` at the end of `prefillBatched`, mirroring the decode-step tail.

After that, the model said Paris again.

## Correct but slower is still wrong

Correctness did not make the first batched implementation fast.

| prompt tokens | per-token | batched, correct but serial |
| ---: | ---: | ---: |
| 17 | 42 tok/s | 23 tok/s |
| 105 | 74 tok/s | 62 tok/s |

The reason was the shader shape. The original `dmmv_q4k_batch.comp` batched input columns but walked K serially:

```glsl
// 64 threads per workgroup; each thread owns one output row.
uint row = gl_WorkGroupID.x * 64u + gl_LocalInvocationID.x;
float sums[MAX_COLS];

for (uint blk = 0u; blk < blocks_per_row; blk++) {
    for (uint sp = 0u; sp < 4u; sp++) {
        for (uint e = 0u; e < 32u; e++) {
            float w = decode(blk, sp, e);
            for (uint c = 0u; c < num_cols; c++) {
                sums[c] += w * x_data[c * K + blk * 256u + ...];
            }
        }
    }
}
```

That saves weight rereads, but one thread still does the whole row reduction. The per-token kernels that were winning on RDNA4 did something different: one wave64 cooperated on a row, split K across lanes, then used `subgroupAdd` to reduce partial sums.

For `K=4096`, serial-over-K means thousands of element updates in one thread. K-parallel means each lane handles a stripe, and the wave reduction is cheap. The batched shader had optimized the reuse axis and thrown away the row-parallel axis.

## The fix: K-parallel batched DMMV

The winning shader shape combines both properties:

| Property | Why it matters |
| --- | --- |
| Batched columns | Read each quantized weight row once per prompt chunk, not once per token. |
| K-parallel row work | Use a wave64 to split the row reduction across lanes. |
| Per-column accumulators | Keep `sums[MAX_COLS]` in VGPRs and reduce once per output column. |
| Q4_K and Q6_K variants | Cover the actual Q4_K_M checkpoint layout instead of only the easy projection subset. |

The Q4_K kpar batched inner loop keeps the same dequant layout as the per-token kpar shader, then applies each decoded stripe to all active columns:

```glsl
for (uint c = 0u; c < num_cols; c++) {
    vec4 by0 = x_v4[col_base_v4 + b_idx];
    vec4 by1 = x_v4[col_base_v4 + b_idx + 8u];
    vec4 by2 = x_v4[col_base_v4 + b_idx2];
    vec4 by3 = x_v4[col_base_v4 + b_idx2 + 8u];

    float partial = dot(vec4(factor0) * q0_lo - vec4(bias0), by0)
                  + dot(vec4(factor1) * q0_hi - vec4(bias1), by1)
                  + dot(vec4(factor2) * q1_lo - vec4(bias2), by2)
                  + dot(vec4(factor3) * q1_hi - vec4(bias3), by3);
    sums[c] += partial;
}
```

At the end:

```glsl
for (uint c = 0u; c < num_cols; c++) {
    float reduced = subgroupAdd(sums[c]);
    if (tid == 0u)
        y_data[y_offset / 4u + c * M + row] = reduced;
}
```

Every weight nibble is still decoded once per output row per chunk. The difference is that 64 lanes now cooperate on that row, and the same decoded work feeds many prompt columns.

## Why `MAX_COLS=40` shipped

The first kpar result was already a win: Q4_K kpar plus Q6_K serial moved the 105-token prompt to 143.1 tok/s. Adding Q6_K kpar moved it to 172.9 tok/s at `MAX_COLS=32`.

The final sweep was the chunk size:

| `MAX_COLS` | chunks for 105 tokens | tok/s |
| ---: | --- | ---: |
| 16 | 7 chunks | 155 |
| 32 | 4 chunks | 173 |
| 35 | 3 chunks | 176 |
| 40 | 3 chunks | **187** |
| 44 | 3 chunks | 179 |
| 48 | 3 chunks | 165 |
| 64 | 2 chunks | 153 |

The curve is exactly what RDNA4 register pressure predicts. Larger chunks reduce dispatch count until the accumulator array starts hurting occupancy. `MAX_COLS=40` is the largest measured point that kept the VGPR footprint healthy on `gfx1201` wave64 while reducing a 105-token prompt to three chunks.

That is why the shipped default is 40, not the rounder-looking 32 or 64.

## End-to-end impact

For a 105-token prompt plus 8 generated tokens:

| path | total wall time |
| --- | ---: |
| per-token | 1.59 s |
| batched serial | 1.78 s |
| **batched plus kpar** | **0.88 s** |

This is why prefill work matters even when decode looks good. Users do not experience "prefill tok/s" as an abstract benchmark. They experience it as the pause before the first token streams.

## Dense Gemma came along for the ride

The same infrastructure later opened the dense Gemma 4 31B batched path, but not through a one-line gate relaxation. Gemma exposed two model-specific correctness bugs:

1. Gemma 4 applies a plain unit-weight RMS norm to V on every attention layer. Per-token and Metal did it; Vulkan batched prefill had skipped it.
2. On full-attention layers where Gemma omits `attn_v`, V must be derived from the raw K projection. The batched path was accidentally feeding post-norm, post-RoPE K as V.

The fix mirrored the per-token path: project Q and K, feed raw `scratch_k` through the V unit-norm dispatch into `scratch_v`, then let K norm and K RoPE mutate `scratch_k` afterward.

On the R9700, dense Gemma 4 31B moved like this:

| prompt | per-token baseline | batched | speedup |
| --- | ---: | ---: | ---: |
| 18 tokens | 4.96 tok/s | 31.23 tok/s | 6.3x |
| 43 tokens | 4.96 tok/s | 35.46 tok/s | 7.1x |
| 113 tokens | 4.96 tok/s | 42.98 tok/s | 8.7x |
| 313 tokens | 4.96 tok/s | 44.50 tok/s | 9.0x |
| **613 tokens** | **4.96 tok/s** | **57.58 tok/s** | **11.6x** |

Validate delta was `max_abs_diff=0.000064`, which is float noise. This is why the [Gemma honesty post](/blog/2026-06-02-gemma-is-the-model-family-that-keeps-zinc-honest) calls out the batched-prefill V-handling bug: it is a clean example of a model-family assumption that Qwen did not punish.

## What this did not solve

This work solved the dense batched path. It did not solve the flagship hybrid path.

Qwen3.5 and Qwen3.6 35B-A3B still need different structural work:

| Missing piece | Why dense batching is not enough |
| --- | --- |
| Batched MoE routing | Tokens must be grouped by selected expert, counted, dispatched, scattered, and weighted. |
| Batched per-expert matmul | Per-token expert DMMV keeps dispatch count high and cannot amortize expert weights across routed tokens. |
| Block-resident SSM state | Gated delta-net should walk prompt tokens inside one workgroup/block, not reload recurrent state once per token. |

That is the split between this post, the [Qwen 35B prefill gate post](/blog/2026-04-26-the-gate-that-keeps-qwen-35b-prefill-at-half-of-llama-cpp-on-rdna4), and the follow-up on [why Qwen 35B cannot use the 208 tok/s path yet](/blog/2026-06-06-why-qwen-35b-cannot-use-zincs-208-tok-s-batched-prefill-path-yet). Dense Qwen3-8B proved the Vulkan batched prefill machinery and shader shape. Hybrid Qwen needs MoE and SSM batching before it gets the same benefit.

## The lessons worth keeping

The first lesson is that validate modes are not optional for GPU inference work. The garbage output looked like a forward-pass math bug until logit validation proved the forward pass was bit-identical. That saved hours of shader archaeology and pointed directly at stale sampler state.

The second lesson is that batching one axis can regress another. The serial batched shader did reduce weight traffic, but it threw away K parallelism. On RDNA4, the correct shape was not "batch columns" or "parallelize K"; it was both.

The third lesson is that gates should be treated as product surface. `ZINC_BATCHED_PREFILL=1` sounding enabled while Vulkan still called `prefillBatch` was worse than a missing feature. It made the performance story untrustworthy until measurement found the dead path.

The fourth lesson is that dense models are the right staging ground for hybrid work. Qwen3-8B gave us a simpler correctness surface. Gemma dense then caught architecture details. Only after both pass does it make sense to reopen the MoE plus SSM gate.

For the design argument that predicted this direction, read [Why RDNA4 prefill wants a 32-column DMMV before it wants a GEMM](/blog/2026-04-22-why-rdna4-prefill-wants-a-32-column-dmmv-before-a-gemm). For the part that remains on the flagship model, read [The gate that keeps Qwen 35B prefill at half of llama.cpp on RDNA4](/blog/2026-04-26-the-gate-that-keeps-qwen-35b-prefill-at-half-of-llama-cpp-on-rdna4). This post sits between them: the measured dense-model proof that the batched path can be correct, fast, and worth turning on by default.
