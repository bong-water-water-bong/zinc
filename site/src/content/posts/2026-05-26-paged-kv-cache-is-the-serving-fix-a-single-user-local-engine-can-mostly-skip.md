---
title: "Paged KV cache is the serving fix a single-user local engine can mostly skip"
date: "2026-05-26"
tags:
  - zinc
  - rdna4
  - amd
  - kv-cache
  - paged-attention
  - pagedattention
  - vllm
  - memory-management
  - prefix-cache
  - radixattention
  - vattention
  - llm-inference
keywords:
  - PagedAttention single user local inference
  - paged KV cache vs contiguous KV cache
  - vLLM KV cache fragmentation 20 to 38 percent
  - block table attention kernel overhead
  - vAttention contiguous virtual memory KV cache
  - RadixAttention prefix sharing radix tree
  - batch of one KV cache allocation RDNA4
  - Radeon AI PRO R9700 KV cache memory
  - amdgpu virtual memory KV cache paging
  - prefix reuse branching conversation block sharing
excerpt: "PagedAttention is one of the best systems ideas in LLM serving, and for a single-user local engine it is mostly the wrong default. Paging earns its keep by packing many concurrent, variable-length sequences into GPU memory with near-zero waste; a desktop assistant runs one conversation at a time and has nothing to pack, so it inherits the cost of a block-table lookup inside the attention kernel without the benefit. zinc's default reflects this: it borrows vLLM's memory-budget math and then allocates one contiguous arena, not a pool of blocks. The exception is the case that pays for paging anyway, and it is the same feature we called the cheapest speedup left on local chat: the moment conversations branch and share a prefix, block-granular allocation stops being overhead and becomes the only clean way to let siblings share a parent's KV without copying it."
---

The KV cache is the largest thing standing between a 30B-class model and a 32 GB card, and the most cited way to manage it was designed for a problem a desktop assistant almost never has. PagedAttention, the idea behind [vLLM](https://arxiv.org/abs/2309.06180), borrowed the operating system's trick of paging and applied it to the key-value cache. It landed at SOSP 2023 and became one of the most influential systems ideas in LLM serving. It is also, for a single-user local engine, mostly the wrong default.

That is a strong claim, so here is the shape of it before the details. Paging earns its keep by packing many concurrent, variable-length sequences into GPU memory without waste. An engine running one conversation at a time has no concurrency to pack. What it inherits from paging is the cost, a block-table lookup inside the attention kernel on every step, without the benefit that justified the cost in the first place. zinc's default path reflects exactly this. It borrows vLLM's memory-budget arithmetic and then allocates a single contiguous arena instead of a pool of blocks.

The part worth staying for is the exception. There is one place a single-user engine does inherit the fragmentation paging was built to fight, and it is the same feature we called [the cheapest five-x left on local chat](/blog/2026-05-06-why-prefix-kv-reuse-is-the-cheapest-five-x-left-on-local-qwen3-chat/): prefix reuse. Once conversations branch and start sharing a common prefix, block-granular allocation stops being overhead and becomes the only clean answer. This post is about where that line sits, and why it decides how a local engine should lay out its cache.

## Why this matters on a 32 GB card

Long context is the reason to run Qwen3 on a [Radeon AI PRO R9700](https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9700.html), and the KV cache is what turns that context into occupied VRAM. Every token you keep in the window costs a slice of memory in every layer, and past a certain length the cache, not the weights, is the thing filling the card. We have spent most of the month on the consequences of that, from [the 16k crossover where KV reads start to outweigh the active weights](/blog/2026-04-27-the-16k-crossover-where-kv-reads-outweigh-active-weights-on-rdna4-decode/) to [cutting the cache to fp8](/blog/2026-05-19-fp8-kv-cache-is-the-next-decode-bandwidth-cut-rdna4-already-has-the-wmma-for/).

If the cache is the budget, how you allocate it is not a detail. The question is whether you hand each sequence one contiguous block sized to its maximum possible length, or parcel memory out in small fixed pages as the sequence grows. That choice is the subject of the PagedAttention paper, and the answer that is right for a busy inference server is not automatically right for a machine serving one person.

## What paging actually solved

The vLLM paper starts from a measurement, and the measurement is the argument. In the serving systems that came before it, the KV cache for a request was a single contiguous reservation sized to the maximum sequence length the request might reach. Most requests never reach that length, so most of the reservation sits empty for the life of the request. Add many requests of different lengths arriving and finishing at different times, and the pool develops gaps too small to reuse. The paper measured the result directly: across existing systems, only 20.4 to 38.2 percent of the allocated KV memory held actual token state. The other 60 to 80 percent was lost to reservation and fragmentation.

PagedAttention fixes this the way an operating system fixes the same problem for process memory. It splits the cache into fixed-size blocks, hands them to a sequence one at a time as it generates, and keeps a per-sequence block table mapping logical token positions to physical blocks. A sequence only ever holds the blocks it has filled plus one partial block at the end, so waste drops to near zero, under 4 percent in the paper. Because blocks are uniform and freely placed, two sequences that share a prefix can point at the same physical blocks, which is the second win. The combined effect was a 2 to 4 times throughput improvement over FasterTransformer and Orca, larger on longer sequences, precisely because higher memory utilization let the server keep more requests in flight at once.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-05-26-paged-vs-contiguous-kv-cache-prefix-tree.svg" alt="A memory-layout schematic on a deep forest-green background. The top band, titled what paging was built to remove, KV waste under concurrent serving, shows two horizontal memory bars. The upper bar, reserve every request to its max length, is mostly a hatched copper region labeled reserved plus internal fragmentation with only a small gold segment at the left labeled actual tokens, annotated only 20.4 to 38.2 percent holds real tokens. The lower bar, PagedAttention fixed blocks committed on demand, is a dense row of small gold blocks with thin gaps and a tiny waste stub at the end, annotated near-zero waste under 4 percent and 2 to 4 times serving throughput, paid for by a block-table lookup in the attention kernel each step. The bottom band, titled the local case, batch of one and the prefix tree that brings the problem back, has two halves. The left half, one sequence and one contiguous arena, is a single gold bar filled about sixty percent with an arrow labeled grows as decode extends, annotated no inter-request fragmentation to fight, attention reads one contiguous KV buffer, zinc default vLLM block-budget math allocated as a contiguous arena at 0.85 device-memory utilization. The right half, a branching conversation with RadixAttention, is a small tree with a mint shared system prompt node and a mint shared chat prefix node that splits into three gold leaf branches, annotated siblings must share the parent KV, storing each branch contiguously would duplicate the shared prefix so block-granular sharing is back on the table. A footer credits the vLLM PagedAttention paper and vAttention and notes proportions are illustrative." loading="lazy" />
  <figcaption>Top: the waste PagedAttention removes. A reserve-to-max layout leaves most of the cache empty; fixed blocks fill it densely and let sequences share physical blocks, at the cost of a block-table lookup in the attention kernel. Bottom: the local case. One sequence needs only a contiguous arena, but a branching conversation that shares a prefix forces block-granular sharing back into the design. Waste and throughput figures are from the vLLM paper; the layout is schematic.</figcaption>
</figure>

The top band is the problem and the fix; the bottom band is why neither transfers cleanly to a local engine. Every benefit in the top band is a benefit of running many sequences at once, because the dense packing is what raises the batch size the throughput comes from. That is the hinge of the whole argument.

## The waste is a concurrency waste

Look again at what the 60 to 80 percent was actually lost to. Internal fragmentation comes from reserving each request's full maximum length when most requests are short. External fragmentation comes from variable-length holes opening up between requests that arrive and complete at different times. Over-reservation comes from holding space for tokens an in-flight request has not generated yet. Every one of those is a property of many concurrent sequences of different lengths competing for one pool of memory.

A single-user local engine has one sequence in flight. There is no second request to fragment the space against, so external fragmentation does not arise. There is no fleet of short requests each holding a long reservation, so the internal fragmentation across requests does not arise either. The engine has exactly one context to size for, which is the context the user is actually in, so reserving for it is the work rather than waste. The 20-to-38-percent utilization problem is a multi-tenant problem. At a batch of one it simply is not there to solve.

What is true at any batch size is that the cache grows during generation and you do not know the final length in advance. A single sequence handles that by growing one contiguous region, the way a stack grows, with none of the cross-request packing that motivates a block pool. Paging buys density across requests, and with one request there is nothing to densify.

## What zinc does by default

This is not a hypothetical preference. It is written into how zinc sizes its cache. The memory planner deliberately mirrors vLLM's startup arithmetic and then diverges on the layout. The comment in `memory_plan.zig` says it plainly: the routine is "analogous to vLLM's `determine_available_memory` then `get_num_blocks` flow, adapted for a contiguous (non-paged) KV cache." It reserves a fraction of the device budget, subtracts the weights and fixed runtime overhead, divides the remainder by the per-token KV cost, and clamps to the model's architectural ceiling. The one tuning difference is the utilization fraction: vLLM uses 0.9, zinc uses 0.85 to leave headroom for attention workspace and per-request scratch that the profiler does not account for.

The result is a single contiguous arena sized to a real context length, not a pool of blocks behind a translation table. The payoff is in the attention kernel. A contiguous cache means the flash-attention kernel reads keys and values as one linear span, with no per-step indirection through a block table to find where each token physically lives. That is the same kernel path our work on [filling RDNA4's idle compute units](/blog/2026-05-20-the-kv-split-that-fills-rdna4-idle-compute-units-on-long-context-decode/) tunes, and keeping its memory access contiguous is one less thing fighting the bandwidth ceiling that decode lives under.

zinc does carry a paged manager as well. `kv_cache.zig` is, in its own words, a "paged KV cache manager for concurrent request serving," a pool of fixed-size pages allocated per request from a free list and released on completion. That code is the right tool for the day zinc serves many requests at once. It is just not the right default for the batch-of-one case the engine spends most of its life in, which is why the default sizes a contiguous arena instead.

## The cost paging adds

It would be easy to treat the block table as free, since it is only an extra lookup. The people who build these kernels do not treat it as free. The whole premise of [vAttention](https://arxiv.org/abs/2405.04437), from a Microsoft Research group in 2024, is that PagedAttention's block layout forces the attention kernel to be rewritten to walk non-contiguous memory, which adds software complexity and leaves performance on the table. Their alternative keeps the cache virtually contiguous and lets unmodified attention kernels run over it, and they report up to a 1.23 times throughput gain over the PagedAttention versions of FlashAttention and FlashInfer. That gap is the standing cost of the block table, measured by people whose goal was to remove it.

| Strategy | Waste at batch of one | Waste under heavy concurrency | Attention kernel | Prefix sharing |
| --- | --- | --- | --- | --- |
| Contiguous arena (zinc default) | low | high: reservation and fragmentation | reads one contiguous span | hard, needs copies |
| Paged blocks (PagedAttention) | block-table overhead, little to gain | near-zero, under 4% | walks a block table each step | natural, siblings share blocks |
| Contiguous virtual, paged physical (vAttention) | low | near-zero | reads contiguous virtual span | via shared physical pages |

Read the table by column rather than by row. At a batch of one, the left column is what matters, and the contiguous arena wins because it has the least to lose and the cheapest kernel. Under heavy concurrency, the middle column flips the result, and paging's near-zero waste is worth its kernel tax. The bottom row is the one that refuses to choose, and it is where this is heading.

## Where the problem comes back: the prefix tree

Here is the exception that complicates the clean story. A single-user engine does not stay at a simple linear conversation if it is any good. The cheapest speedup available on local chat is reusing the KV of a shared prefix, the system prompt and the conversation so far, instead of recomputing it every turn. [SGLang's RadixAttention](https://arxiv.org/abs/2312.07104) is the canonical version: it keeps the KV cache of past requests in a radix tree, matches each new request against the longest cached prefix, and reuses it, with an LRU policy evicting the cold branches. We argued for the same move on local chat, and it changes the allocation problem.

A radix tree of conversations is a branching structure. Two replies to the same message, a few-shot prompt expanded into several completions, an edit that forks a chat at turn ten: all of these are sibling branches that share an interior prefix and then diverge. If each branch owns a contiguous arena, the shared prefix has to be duplicated into every branch, which throws away the saving that made prefix reuse worth doing. To let the siblings share the prefix's KV without copying it, the prefix has to live in units that more than one branch can point at. That is a block. The branching tree is exactly the variable-length, shared, dynamically allocated structure that paging was built for, and it reappears the moment prefix reuse goes from a straight line to a tree. The honest rule is not that local engines never need paging; it is that the linear case pays for it without benefit while the branching case genuinely needs block-granular sharing.

## The third option: contiguous virtual, paged physical

The bottom row of the table is the way out, and it is the reason vAttention is more than a critique. Physical memory should be paged, so the engine never reserves what a branch has not used and siblings can share a prefix's pages. Virtual memory should stay contiguous, so the attention kernel reads a flat span and needs no block table. You get both by decoupling the two, reserving a contiguous virtual address range per sequence and committing physical pages into it on demand, with shared prefixes mapped into more than one sequence's virtual range.

vAttention does this with CUDA's virtual memory APIs, which is an NVIDIA path. The same idea has an AMD path, and it is one zinc is unusually well placed to take because of the runtime work we described in [the decision to write our own GPU runtime](/blog/2026-05-18-inside-the-decision-to-write-our-own-gpu-runtime-for-local-llm-inference/). An engine that submits directly through the amdgpu kernel driver owns its GPU virtual address space, so it can map and unmap physical buffer objects into a reserved virtual range itself, which is the same decoupling vAttention gets from CUDA. On the Vulkan backend the analog is sparse residency, a large sparse buffer whose pages are bound on demand and can be backed by shared physical memory. Both let the prefix tree share pages at the driver level instead of through a software block table that every attention dispatch has to consult.

## What comes next

The takeaway is narrow and it is about defaults. PagedAttention solved a real and severe problem, but the problem is concurrency, and its waste numbers are the numbers of a multi-tenant server. A local engine at a batch of one does not have that waste to recover, so importing the block table imports the cost without the benefit, which is why zinc's default sizes a contiguous arena from the same budget math vLLM uses and keeps its paged manager for the day it serves a crowd.

The line to watch is prefix reuse. A linear chat wants a contiguous cache. A branching tree of shared prefixes wants block-granular sharing, and the cleanest way to give it that without taxing the attention kernel is virtual-memory paging, contiguous in the address space and paged in physical memory, which an engine that owns its amdgpu submission path can build directly. The right question is not "paged or contiguous." It is whether your single user's conversations are a line or a tree, and a local engine that means to be good at chat should plan for the tree while defaulting to the line.
