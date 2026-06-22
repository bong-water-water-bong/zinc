---
title: "Gemma is the model family that keeps ZINC honest"
seoTitle: "Gemma 4 Local Inference Benchmarks"
date: "2026-06-02"
tags:
  - zinc
  - gemma
  - gemma4
  - local-llm
  - local-llm-inference
  - llm-inference
  - performance
  - benchmark
  - rdna4
  - metal
  - apple-silicon
  - prefill
  - decode
  - moe
  - gpu-kernels
keywords:
  - Gemma 4 local inference
  - Gemma 4 ZINC benchmark
  - Gemma 4 31B Q4_K_M
  - Gemma 4 26B A4B MoE
  - Gemma 4 RDNA4 benchmark
  - Gemma 4 Metal benchmark
  - Gemma 4 MoE inference
  - Gemma 4 dense inference
  - Gemma RDNA4 inference
  - Gemma Apple Silicon Metal
  - Gemma prefill decode benchmark
  - Gemma sliding window attention
  - Gemma asymmetric GQA
  - Gemma GEGLU MoE
  - ZINC vs llama.cpp Gemma
  - local LLM Gemma benchmark
faqs:
  - question: "Why is Gemma important for ZINC if Qwen has the headline numbers?"
    answer: "Gemma exercises different engine assumptions: sliding-window attention, asymmetric Q/KV dimensions, Gemma-specific norm placement, GEGLU FFNs, and a dense-vs-MoE split. It is the model family that proves ZINC is not only tuned for Qwen-shaped workloads."
  - question: "Is ZINC already faster than llama.cpp on Gemma?"
    answer: "Not generally. In the June 1 dashboard, ZINC is close on Gemma decode in several rows and ahead on Metal dense Gemma prefill, but it still trails badly on RDNA prefill, Metal Gemma MoE, and end-to-end latency."
  - question: "Which Gemma model is the better release target?"
    answer: "For v0.1, dense Gemma 4 31B is the better validation target because it exercises Gemma attention and norm rules without the extra MoE routing surface. Gemma 4 26B-A4B is still useful, but it should be treated as the harder follow-up."
excerpt: "Qwen is where ZINC has the cleanest headline wins. Gemma is where the engine gets audited. The current dashboard shows ZINC close to llama.cpp on Gemma decode and even ahead on Metal dense prefill, but prefill and MoE coverage still expose the weak spots. This post explains why the two Gemma rows matter, what they reveal about sliding-window attention, asymmetric GQA, GEGLU, and command scheduling, and what has to improve before Gemma becomes a release-strength target."
seoDescription: "Gemma 4 local inference benchmarks in ZINC: dense vs MoE, RDNA4 and Apple Metal results, prefill gaps, decode behavior, and ZINC vs llama.cpp context."
---

Every inference engine has a model family that flatters it and a model family that audits it. For ZINC right now, Qwen is the flattering one. Gemma is the audit.

Quick answer: Gemma is the model family that keeps ZINC honest. Qwen is where the current headline wins are cleanest. Gemma is where the engine has to prove those wins are not just Qwen-shaped special cases.

The current [ZINC benchmark dashboard](/zinc/benchmarks/) makes that distinction useful instead of philosophical. On Gemma 4, ZINC is not universally faster than llama.cpp. It is close on several decode rows, unexpectedly ahead on one Metal dense prefill row, and still far behind on the broad prompt and end-to-end story. That unevenness is exactly why Gemma is worth writing about.

If you only test Qwen, you can accidentally optimize for one architecture family. If you test Gemma too, the shortcuts show up: one shared head dimension, one FFN activation, one norm order, one attention window, one MoE routing shape. Gemma breaks enough of those assumptions that it becomes less like another benchmark row and more like a code review for the engine.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-06-02-gemma-dashboard-shapes.svg" alt="Grouped bar chart showing ZINC as percent of llama.cpp for Gemma 4 rows. RDNA Gemma MoE reaches 88 percent of llama.cpp decode but 18 percent of prefill. RDNA dense reaches 86 percent decode and 21 percent prefill. Metal Gemma MoE reaches 34 percent decode and 9 percent prefill. Metal dense reaches 94 percent decode and 157 percent prefill." loading="lazy" />
  <figcaption>The dashboard shape is the point. Gemma decode is already within reach in the dense rows, but prefill and MoE routing still decide whether the model feels good locally.</figcaption>
</figure>

## The two Gemmas are different tests

ZINC currently carries two managed Gemma 4 entries:

| Model | Catalog id | File size | Working-set note | What it tests |
| --- | --- | ---: | --- | --- |
| Gemma 4 31B dense Q4_K_M | `gemma4-31b-q4k-m` | 19.65 GB | 21 GB required VRAM | Dense Gemma attention, norm, tokenizer, and LM-head paths |
| Gemma 4 26B-A4B MoE Q4_K_M | `gemma4-26b-a4b-q4k-m` | 16.87 GB | 16 GB required VRAM, about 11 GB offloadable active footprint | Gemma-specific MoE routing, GEGLU experts, and sparse FFN scheduling |

Those are not two sizes of the same problem. Dense Gemma 4 31B is the cleaner architecture-port test. Every layer matters. Every dense FFN path matters. The big question is whether the engine understands Gemma's attention and normalization rules without falling back to a generic transformer path.

Gemma 4 26B-A4B is a different kind of stress test. The active parameter count is much smaller than the resident parameter count, so the decode story should be friendlier than a dense 26B would be. But the runtime now has to route experts, use GEGLU instead of SwiGLU, handle Gemma's post-FFN norm behavior, and keep the selected expert path from becoming a dispatch machine. A sparse model only helps if the runtime can keep sparsity cheap.

That distinction is why a single "Gemma support" label is not enough. Dense Gemma can look healthy while Gemma MoE is still weak. Gemma MoE can fit in memory while still losing badly on prompt processing. The model family has to be read as a matrix, not a checkbox.

## Gemma 4 local inference benchmark

The June 1 dashboard has four useful Gemma ZINC rows and two Intel baseline-only rows.

| Target | Model | ZINC decode | llama.cpp decode | ZINC prefill | llama.cpp prefill | Read |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| RDNA / R9700 | Gemma 4 26B-A4B MoE | `89.73` | `102.00` | `89.1` | `497.08` | Decode is plausible, prefill is not |
| RDNA / R9700 | Gemma 4 31B dense | `24.65` | `28.55` | `41.64` | `201.97` | Dense decode is close, prompt path trails |
| Metal / M4 Max | Gemma 4 26B-A4B MoE | `30.01` | `88.44` | `34.0` | `365.45` | MoE path is still immature |
| Metal / M4 Max | Gemma 4 31B dense | `21.86` | `23.30` | `132.6` | `84.47` | Dense prefill is the bright spot |

All numbers are median tokens per second from the dashboard data in `site/src/data/zinc-performance.json`. They are same-model comparisons against llama.cpp on the same target class.

The first thing to notice is that decode is not the scary column. RDNA Gemma MoE decode is `88.0%` of llama.cpp. RDNA dense Gemma decode is `86.3%`. Metal dense Gemma decode is `93.8%`. Those are not wins, but they are close enough to say the forward path is real.

The second thing to notice is that prefill is the problem almost everywhere. RDNA Gemma MoE prefill is only `17.9%` of llama.cpp. RDNA dense Gemma prefill is `20.6%`. Metal Gemma MoE prefill is `9.3%`. The exception is Metal dense Gemma 31B, where ZINC is `157.0%` of llama.cpp on prefill and still loses end-to-end because the full request path includes more than prompt ingestion.

The third thing is that Intel should not be folded into the release story yet. The dashboard has llama.cpp Gemma rows on Arc Pro B70, but no successful ZINC Gemma rows there. That is useful roadmap data. It is not a support promise.

## Why Gemma catches Qwen-shaped shortcuts

The Gemma rows are useful because they expose assumptions that Qwen does not always punish.

The first is attention shape. Gemma 4 uses sliding-window attention for most layers and full attention at a lower cadence. That changes long-context memory behavior and prefill scheduling. A kernel that assumes every layer has the same attention window can be correct on Qwen and wrong or wasteful on Gemma.

The second is grouped-query attention detail. The earlier [single push constant post](/blog/2026-04-24-the-single-push-constant-blocking-gemma-4-prefill-on-rdna4/) covered the hard version: full-attention Gemma layers can force the engine to distinguish Q head dimensions from KV head dimensions. A `head_dim` push constant that worked for LLaMA-shaped and Qwen-shaped paths was too vague for Gemma. The fix was not a better inner loop. It was a more honest kernel interface.

The third is V handling. Dense Gemma 4 has attention layers where V is derived from K and then receives a plain unit-weight RMS norm. The [RDNA4 batched prefill post](/blog/2026-06-05-how-zinc-rdna4-batched-prefill-went-from-42-to-208-tok-s/) calls this out explicitly: the dense Gemma gate came off after the batched path stopped feeding a post-norm, post-RoPE K buffer as V. That is the kind of bug that can pass simple shape checks and still corrupt model behavior.

The fourth is FFN structure. Qwen-style paths use SwiGLU. Gemma uses GEGLU. Gemma MoE also has different norm placement and can apply post-FFN normalization before the residual add. A generic "run the expert FFN" abstraction hides exactly the details that decide correctness.

The fifth is vocabulary and head cost. Gemma 4 carries a large vocabulary and a big LM-head path. The shader comment in `dmmv_q4k_wide.comp` exists because Gemma 4 31B's LM head was an explicit bottleneck. Even when decode is memory-bound, the final projection can still be large enough to deserve special treatment.

None of those make Gemma exotic. They make it representative of where open models are going: more architecture variation, more attention variants, more tokenizer behavior, more family-specific cleanup. Engines that hardcode one transformer shape will keep finding out late.

## Dense Gemma is the better v0.1 gate

For v0.1, dense Gemma 4 31B is the better Gemma gate than Gemma 4 26B-A4B.

That is not because the dense model is easier in absolute terms. It is larger at runtime and puts more pressure on memory bandwidth. It is better as a gate because the failure surface is narrower. If dense Gemma works, then the tokenizer, chat template, embedding scale, attention shape, RoPE, V handling, post-attention norm, dense FFN, LM head, and sampling boundary are all being exercised without the extra MoE routing layer.

Gemma 4 26B-A4B should stay in the matrix, but it should be treated as the harder follow-up. It tests a different part of the engine: selected expert dispatch, GEGLU fused kernels, expert down accumulation, router scaling, and architecture-specific cleanup around the sparse FFN. The current Metal MoE row says that path is not release-strength yet. The RDNA row says it is at least in the neighborhood on decode, but the prompt path still needs structural work.

The release discipline should be simple:

| Question | Better Gemma target |
| --- | --- |
| Does ZINC understand Gemma's architecture? | Gemma 4 31B dense |
| Does ZINC handle Gemma prompt ingestion well? | Gemma 4 31B dense and 26B-A4B |
| Does ZINC have production-grade Gemma MoE routing? | Gemma 4 26B-A4B |
| Should Gemma MoE be a v0.1 promise? | Not yet |

That keeps the message honest. "Gemma dense is supported and measured" is useful. "Every Gemma shape is solved" is not true.

## What needs to improve next

The next Gemma work is not mysterious.

On RDNA, prefill is the obvious target. The dense batched path is already coherent, and the old per-token numbers show why batching matters: Gemma 4 31B moved from roughly `4.96 tok/s` per-token prefill to `31.23` to `35.46 tok/s` in the dense batched path on short prompts. The public dashboard is now higher at `41.64 tok/s`, but still only one fifth of llama.cpp on the same model. That gap is large enough that small shader tuning will not close it alone. The prompt path needs fewer dispatch boundaries, better batching, and less overhead between attention and projection phases.

On Metal, dense Gemma is the encouraging row. A `132.6 tok/s` prefill median against llama.cpp at `84.47 tok/s` means the unified-memory path and the dense Gemma prefill kernel are doing real work. The problem is that this does not generalize to Gemma MoE yet. Metal Gemma 26B-A4B at `30.01 tok/s` decode against `88.44` for llama.cpp is the row that says the sparse expert path still needs the same treatment the dense path got.

On Intel, the right move is restraint. The Arc Pro B70 data is strategically interesting because the bandwidth and VRAM class match the 32 GB local-inference problem. But without successful ZINC Gemma rows, Intel belongs in roadmap language, not binary-release language. That matches the broader [performance overview](/blog/2026-06-01-zinc-performance-where-it-is-fast-and-where-it-is-not/).

Across all backends, Gemma needs to become a regression gate. If a change improves Qwen 3.6 35B-A3B decode but breaks Gemma dense prefill, the engine got narrower. If a new fused kernel assumes SwiGLU everywhere, Gemma should catch it. If a descriptor shortcut assumes one head dimension for Q and KV, Gemma should catch it. That is the practical value of this model family.

## Reference trail

This post is meant to be read with the measured dashboard open, not as a standalone claim. The live source of truth is the [ZINC benchmark dashboard](/zinc/benchmarks/), and the broader context is the June 1 [ZINC performance overview](/blog/2026-06-01-zinc-performance-where-it-is-fast-and-where-it-is-not/). The Gemma-specific kernel history starts with the [single push constant that blocked Gemma 4 prefill](/blog/2026-04-24-the-single-push-constant-blocking-gemma-4-prefill-on-rdna4/), then continues into the [one-submit-per-prompt prefill work](/blog/2026-04-25-why-one-vkqueuesubmit-per-prompt-is-the-next-quiet-rdna4-prefill-unlock/).

For readers who want the architecture mechanics, the most useful companion is [How MoE models work in ZINC](/blog/2026-04-04-how-moe-models-work-in-zinc/). For readers who want to reproduce local runs, start with [Getting Started with ZINC](/zinc/docs/getting-started/) and the RDNA4 [batched prefill post](/blog/2026-06-05-how-zinc-rdna4-batched-prefill-went-from-42-to-208-tok-s/). Those references are the reason the Gemma story can stay precise: the claims tie back to model ids, measured rows, and named kernel work.

## The broader lesson

Gemma is not important because it is another model people might want to run. It is important because it turns implicit engine assumptions into test failures.

Qwen tells us whether ZINC can win on the flagship local workload it has optimized hardest. Gemma tells us whether the engine is general enough to deserve the word "engine." Dense Gemma tests the non-Qwen architecture path. Gemma MoE tests whether sparse inference stays cheap outside the Qwen family. Metal Gemma tests whether unified-memory wins survive a different model shape. RDNA Gemma tests whether Vulkan batching is real and not just lucky.

That is the standard ZINC should use going into v0.1. The release can have a Qwen headline. It should have a Gemma audit.

The best version of this project is not "fast on the model we tuned yesterday." It is an inference engine that absorbs the next model family without discovering that half of its kernels were secretly named after the last one. Gemma is how we find that out now, while the fixes are still local and the release promise is still under our control.
