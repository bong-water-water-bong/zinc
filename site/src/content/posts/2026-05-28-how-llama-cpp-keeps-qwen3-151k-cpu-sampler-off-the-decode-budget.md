---
title: "How llama.cpp keeps Qwen3's 151k CPU sampler chain off the decode budget"
date: "2026-05-28"
tags:
  - zinc
  - rdna4
  - amd
  - sampling
  - sampler-chain
  - min-p
  - dry
  - top-k
  - bucket-sort
  - qwen3
  - flashsampling
  - llm-inference
keywords:
  - llama.cpp sampler chain Qwen3 151k vocab
  - bucket sort top-k sampler llama.cpp
  - CPU sampler cost per decode token
  - min-p DRY temperature stacked cost
  - FlashSampling fused GPU sampler
  - Qwen3 151936 vocab decode budget
  - sampler chain order penalties dry top_n_sigma
  - RDNA4 decode CPU sampler tax
  - Radeon AI PRO R9700 sampler chain
  - PR 5109 PR 15665 bucket sort reuse
excerpt: "Every decode step on Qwen3 hands the CPU a vector of 151,936 logits, and the default llama.cpp sampler chain walks that vector five times before it picks a token. At 100 tok/s on a Radeon AI PRO R9700, the GPU forward pass is around 10 ms and the naive sampler chain is no longer a rounding error inside it. The reason the chain still fits is a single optimization: ikawrakow's bucket sort from PR #5109, reused by PR #15665 across top-k, top-p, and min-p, so the chain pays one histogram and then walks only the buckets that contain the survivors. FlashSampling, the March 2026 paper that fuses sampling into the LM-head matmul epilogue, is the next move and the reason the CPU chain has a ceiling. On a single-user local engine the choice is not whether to sample on the GPU, it is which samplers can move there and which ones the CPU still has to carry."
---

A decode step on Qwen3 produces a logit per token in a vocabulary of 151,936, and the CPU has to do something with all of them before the next forward pass starts. The default llama.cpp sampler chain on that vector is `penalties;dry;top_n_sigma;top_k;typ_p;top_p;min_p;xtc;temperature`, which is nine stages, of which most walk the whole array. That is a measurable amount of work to do on the CPU between every token a fast GPU produces, and on a single-user local engine the GPU has finally gotten fast enough that the sampler can be heard from inside the decode budget.

It is also the reason a quiet sampler change last summer, [PR #15665 by ggerganov](https://github.com/ggml-org/llama.cpp/pull/15665), matters more than it looks. That PR did not introduce a new sampling algorithm. It made llama.cpp's existing top-k bucket sort, originally a one-shot trick added by ikawrakow in [PR #5109](https://github.com/ggerganov/llama.cpp/pull/5109), into a shared scaffold the rest of the chain runs on. The chain pays for one histogram per token and then never sees the bulk of the vocabulary again. Without that, a 151k vocab and a 100 tok/s decode would have made the CPU sampler a visible tax on local Qwen3 chat. With it, the tax is small enough to keep ignoring for one more model generation.

We have spent the month on the parts of the [Radeon AI PRO R9700](https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9700.html) that move tensors. This is a post about the part that moves tokens, which by now is the only piece of the decode loop that still lives on the CPU.

## Why the sampler stopped being free

For most of llama.cpp's history, the cost of sampling was a rounding error against the cost of the forward pass. A model that decoded at twenty tokens per second left fifty milliseconds between samples, and any reasonable CPU implementation finished its work in a few hundred microseconds of that. The forward pass was the budget and the sampler was the change.

Two things changed at once. The forward pass got faster. The wave32 attention fix we wrote about in the [wave32 commit that closed RDNA4's long-context flash attention gap](/blog/2026-05-11-the-wave32-commit-that-closes-rdna4-long-context-flash-attention-gap), the [decode bandwidth cut FP8 KV cache is teeing up](/blog/2026-05-19-fp8-kv-cache-is-the-next-decode-bandwidth-cut-rdna4-already-has-the-wmma-for), and the LMHead path that we sized in [what Qwen3's 151k LMHead costs on RDNA4 decode](/blog/2026-05-16-what-qwen3-151k-lmhead-costs-on-rdna4-decode) all push the per-token GPU budget down toward ten milliseconds on a 32B model. The other change is the vocabulary itself. Qwen3 ships with 151,936 tokens, confirmed in the [Qwen3 vocab size issue](https://github.com/QwenLM/Qwen3/issues/727), and that is the array the sampler runs over.

A naive sampler that touches the full logit vector once does about 608 kilobytes of work per pass, which is already at the edge of a typical CPU's per-core L2 cache. Five passes does it five times, and the chain becomes a stream of cold reads rather than a hot working set. None of the arithmetic is hard. The damage is in how often the cache gets refilled and how many independent kernels the CPU launches per token.

This is the same observation the [FlashSampling paper](https://arxiv.org/abs/2603.15854) makes about GPU sampling on a larger scale, and the line is worth reading directly: sampling already accounts for over ten percent of token generation time on a single GPU and twenty to thirty-eight percent in tensor-parallel settings, not because the math is heavy but because of the chain of separate kernels that materialize, normalize, and scan the logits tensor. CPU sampling on a 151k vocab is the same problem on smaller hardware.

## The default chain, in order

The chain llama.cpp installs by default for a server with no `--samplers` override is the one published in the [llama.cpp server flags reference](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md), and the order is not arbitrary. Penalties go first because they edit logits based on the recent history, before any truncation has discarded the candidates they want to penalize. DRY runs next for the same reason. The truncation samplers come after, with top-k usually narrowing the field first because it is the cheapest, then the probability-shape samplers, then temperature last so that scaling does not change the rank order the truncators used. This is the same reason [we argued temperature belongs at the tail of the chain](/blog/2026-05-04-why-min-p-is-the-right-default-sampler-for-local-qwen3-decode).

Each row in the chain, except top-k, used to be its own pass over the 151,936-element logit array, and three of them (top-p, min-p, top-n-sigma) also wanted the vector sorted. The naive cost of one decode step's sampler chain on a 151k vocab is roughly the sum of one O(V log V) sort and four to five O(V) passes, which on a single core at a few GFLOPS lands somewhere between half a millisecond and two milliseconds depending on cache and SIMD. Against a ten-millisecond GPU step that is a real slice of decode budget, and the slice grows with every kernel the GPU wins back.

## Bucket sort, reused

The piece that keeps this in check is small and not new. The [bucket-sort top-k routine](https://codepointer.substack.com/p/llamacpp-accelerate-top-k-sampling) that ikawrakow added in early 2024 replaces `std::partial_sort` with a two-pass histogram over a 128-bucket logit range, then sorts only the buckets above the cutoff. The substack writeup measures it at 2.9 times faster than `std::partial_sort` at k=8000 and breaks even at about k=128, which is the threshold llama.cpp uses to choose between the two paths today.

What [PR #15665](https://github.com/ggerganov/llama.cpp/pull/15665) added is the observation that the histogram and the bucketed scatter are reusable. Once the chain has classified all 151,936 tokens into 128 buckets and identified the cutoff bucket, top-p and min-p do not need to walk the whole array again. They walk the highest non-empty buckets, accumulate probability mass or compare against the top-token threshold, and stop. The same histogram that drives top-k becomes the index the rest of the truncation chain uses.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-05-28-cpu-sampler-chain-bucket-sort-reuse.svg" alt="A two-band schematic on a cool gunmetal-grey background with warm orange accents, titled five passes over 151,936 logits versus one histogram and four bucket walks. The top band, naive sampler chain, no bucket sharing, shows a horizontal logit strip labeled V equals 151,936 floats, about 608 kilobytes per pass, then five arrows arcing across the strip, each tagged with a sampler name in order: penalties, DRY, top-k partial sort O of V log k, top-p sort and cumulative sum, min-p compare against top token. Each arc is colored copper-orange and labeled one full V pass. A small inset at the right reads naive cost, roughly half a millisecond to two milliseconds per decode step on a 151k vocab CPU sampler. The bottom band, llama.cpp PR 15665, bucket-sort reuse, shows the same logit strip carved into 128 vertical bins by a single sweep labeled histogram, one pass over V, 128 buckets. The top eight buckets at the right are highlighted in gold and marked survivors above cutoff. Four short arrows now sit only above those buckets, labeled top-k sort, top-p mass scan, min-p threshold, top-n-sigma sweep. An annotation reads four samplers walk only the highest buckets, not the whole vocabulary, the rest of the array is touched once. A footer credits PR 5109 by ikawrakow and PR 15665 by ggerganov, notes the crossover at k equals 128, and marks the layout as schematic." loading="lazy" />
  <figcaption>Top: the naive chain pays five full V-sized passes over a 151,936-element logit array, which is the cost local engines were quietly absorbing before bucket sort was reused. Bottom: one histogram identifies the survivor buckets, and the rest of the chain walks only those, so the bulk of the vocabulary is touched once per decode step rather than five times. The 128-bucket count and the k=128 crossover are the values in llama.cpp's source; the proportions are schematic.</figcaption>
</figure>

The point of the diagram is the difference in how often the bulk of the vocabulary is read. The top band reads it five times. The bottom band reads it once, then the chain spends its time inside a few hundred surviving tokens. That is the entire reason a 151k vocab has not already pushed the CPU sampler to the top of the local decode flame graph.

## What it costs and what it does not

This is not free, even after the optimization. The order in the table below is the same order the chain runs, and the bucket reuse only helps the rows in the lower half.

| Sampler stage | Walks all 151,936 logits? | Cost shape | Notes |
| --- | :---: | --- | --- |
| Penalties (frequency, presence, repetition) | yes | O(V) scan with history lookup | runs before truncation; can mutate any logit |
| DRY | yes | O(V) with N-gram match against context | adds string-match cost outside the array |
| Top-k | no, after bucket sort | one histogram, sort top buckets | the path PR #5109 introduced |
| Top-n-sigma | no, reuses histogram | sweep top buckets to find the σ threshold | new sampler that benefits from PR #15665 |
| Top-p | no, reuses histogram | accumulate mass over highest buckets | naive version is O(V log V) |
| Min-p | no, reuses histogram | compare bucket maxima against top token | naive version is O(V) |
| Temperature | yes, but on survivors only | scale a few hundred values | runs after truncation, so V is small |

The first two rows are the bill the optimization does not touch. Penalties have to scan the whole array because any token in the vocabulary might be in the recent history and need its logit adjusted. DRY is the same shape plus an extra string-match step against the context, which is why it is the slowest sampler in practice even on a small vocab, and why we wrote about [DRY earning its slot ahead of min-p](/blog/2026-05-05-why-dry-earns-the-slot-before-min-p-on-qwen3-long-context-decode) before discussing whether it should run at all. Bucket sort is what keeps the rest of the chain from doubling that bill.

A reasonable rule of thumb for a single-user local engine on Qwen3 is that the CPU sampler chain costs somewhere on the order of a few hundred microseconds per decode token after bucket reuse. It is small. It is also no longer below the noise floor, and it grows linearly with the vocabulary the next model picks.

## Why this is a ceiling, not a fix

The pattern the bucket sort reveals is that most of the chain does not care about most of the vocabulary, which is exactly why a GPU-side sampler can win. FlashSampling, from a team at LMU Munich and Princeton in [the March 2026 paper](https://arxiv.org/abs/2603.15854), goes further: it never materializes the [1, 151936] logit tensor at all. The kernel computes logits tile by tile in shared memory, adds Gumbel noise, keeps one candidate per row per vocabulary tile, and reduces over tiles to a single sample. The reported end-to-end win in vLLM is up to a nineteen percent reduction in time per output token on the models tested, which is roughly the size of the sampler slice they cite for tensor-parallel inference.

A single-user local engine on RDNA4 has the simpler version of this problem. There is no logit gather across ranks. The LMHead matmul we sized at about half a millisecond at Q6_K is the largest non-MoE memory read in the decode loop, and its epilogue is the natural place to do argmax-style sampling without ever shipping the logits back to system memory. The Vulkan compute shader that produces the logits can also keep a Gumbel-noised running maximum per tile, and the CPU only sees the chosen token. That moves top-k, top-p, min-p, top-n-sigma, and temperature into the same kernel that produced the logits in the first place.

What it does not move is penalties and DRY. Both of those need the recent token history, both can mutate any logit in the vocabulary, and both fit naturally on the CPU because the history is small. The clean split is the one [vLLM's logits-processors design](https://docs.vllm.ai/en/latest/design/logits_processors/) gestures at: a small number of history-dependent processors stay on the host, and the dense truncation chain moves to the device. The CPU chain shrinks to a narrow path that runs only the samplers that genuinely need the vocabulary in CPU memory.

## What comes next on zinc

The piece zinc has to build is the LMHead-fused sampler kernel on Vulkan, with the bucket-reuse CPU chain as the fallback for penalties, DRY, and any sampler a future model wants that does not fit the Gumbel-max formulation. The fallback already exists in llama.cpp form and is the right thing to share; the fused kernel is the new code, and it is small because the math is small. The decision to write it is the same decision we walked through in [why we wrote our own runtime](/blog/2026-05-18-inside-the-decision-to-write-our-own-gpu-runtime-for-local-llm-inference): the seam between forward pass and sampler is the kind of place a local engine has to own end to end if it wants the decode budget to keep shrinking.

The thing to carry forward is the framing. A 151k vocab made CPU sampling visible. Bucket-sort reuse hid it again. FlashSampling shows that the same logic, applied one rung higher in the kernel hierarchy, takes sampling off the host entirely. The CPU chain is not going away, but it is becoming the narrow tail, and the rest of the chain belongs in the same kernel that produced the logits.
