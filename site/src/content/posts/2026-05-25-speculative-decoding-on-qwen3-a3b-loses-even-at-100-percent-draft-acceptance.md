---
title: "Speculative decoding on Qwen3-A3B loses even at 100% draft acceptance"
date: "2026-05-25"
tags:
  - zinc
  - rdna4
  - amd
  - speculative-decoding
  - mixture-of-experts
  - moe
  - qwen3
  - qwen3-a3b
  - expert-routing
  - decode
  - prompt-lookup-decoding
  - llm-inference
keywords:
  - speculative decoding MoE slowdown
  - Qwen3-A3B speculative decoding
  - expert saturation threshold MoE
  - 100 percent draft acceptance still slower
  - MoESD batch saturation threshold
  - Cascade utility-driven speculative decoding MoE
  - prompt lookup decoding n-gram draft
  - SuffixDecoding agentic speculation
  - Radeon AI PRO R9700 MoE decode
  - verify step union of experts Qwen3
excerpt: "The textbook says a draft that is free and always accepted should make decode several times faster. On Qwen3.6-35B-A3B it does the opposite. A public RTX 3090 benchmark ran n-gram and draft-model speculation at 100 percent acceptance on code and reasoning prompts and watched single-request decode fall from 135.7 tok/s to as low as 59. The reason is not the draft and not the acceptance rate. It is that a sparse mixture-of-experts verifies K drafted tokens by loading the union of the experts they route to, and below an expert-saturation threshold of about 94 tokens every extra drafted token wakes a fresh expert. That single fact decides whether speculation helps or hurts an A3B model, and it is why the right default on a single-user local engine is usually no speculation at all."
---

Speculative decoding has one promise that has held up for three years: if you can produce candidate tokens cheaply and the big model accepts most of them, decode gets faster. The cheapest possible draft is no model at all, just a search of the prompt for a matching n-gram. The best possible acceptance rate is 100 percent. Put those two together and the textbook says you should win by a wide margin.

On Qwen3.6-35B-A3B, you lose. A [public benchmark on a single RTX 3090](https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090) ran nineteen speculative configurations after llama.cpp merged its MoE draft path, and every one of them came in below the non-speculative baseline. The n-gram cache draft and the vocab-matched 0.8B draft model both hit 100 percent acceptance on the code and reasoning prompts, and on those exact prompts decode collapsed from 135.7 tok/s to as low as 59. A free draft, perfectly accepted, and the model got almost twice as slow.

We wrote about [why speculative decoding does not net out on this model in April](/blog/2026-04-28-why-speculative-decoding-does-not-net-out-on-qwen-35b-a3b), and framed it through the draft's cost ratio. That framing is right for a draft model, but it cannot explain the n-gram result, because an n-gram draft has a cost ratio of essentially zero. This post is about the term the dense-model math leaves out, the one that turns a free, fully accepted draft into a slowdown on a sparse mixture of experts.

## What the textbook says should happen

The standard speedup formula comes from [Leviathan, Kalman, and Matias](https://arxiv.org/abs/2211.17192). With per-token acceptance probability `α`, lookahead `γ`, and a cost ratio `c` between one draft step and one verifier step, the expected speedup over greedy decoding is `(1 - α^(γ+1)) / [(1 - α)(γc + 1)]`.

An n-gram draft, also called [prompt lookup decoding](https://github.com/apoorvumang/prompt-lookup-decoding), replaces the draft model with a string match. It takes the last few generated tokens, finds where they appeared earlier in the context, and returns the next several tokens from that earlier spot as the candidate. There is no second forward pass, so `c` is a rounding error. Set `c` to zero and the formula simplifies to `(1 - α^(γ+1)) / (1 - α)`, which is just `1 + α + α² + ... + α^γ`. At 100 percent acceptance with a lookahead of eight, that is nine. The verifier should be emitting as many as nine tokens per forward pass, and decode should run close to nine times faster.

That is the prediction the benchmark falsifies. The draft was free, the acceptance was perfect, and the measured result was a slowdown. When a clean formula and a clean measurement disagree this hard, the formula is missing a variable, and the missing variable is sitting inside the verifier.

## What the benchmark measured

The numbers are worth looking at directly, because the shape of the table is the whole argument. All runs are single-request greedy decode of `Qwen3.6-35B-A3B-UD-Q4_K_XL` through llama-server on one RTX 3090.

| Configuration | Mean tok/s | Worst prompt | Draft acceptance |
| --- | ---: | ---: | :---: |
| Baseline, no speculation | 135.7 | 135.3 | — |
| n-gram cache draft | 119.1 | 65.3 | 100% (96/96) |
| Vocab-matched 0.8B draft, K=8 | 121.1 | 59.2 | 100% (270/270) |
| Vocab-matched 0.8B draft, K=32 | 120.3 | 59.5 | 100% |

Every speculative row loses on the mean, and the worst prompts, the reasoning and code prompts where the draft actually found matches and fired, are the ones that fall to roughly half the baseline. Acceptance is not the problem. The repository reports 100 percent acceptance across hundreds of drafted tokens in those collapsing configurations. The classical intuition that high acceptance buys speed simply does not hold here, and that is the signal that something structural is going on rather than a tuning miss.

## The weight a sparse MoE reads to verify K tokens

Here is the part the dense-model formula quietly assumes away. On a dense model, verifying `K` drafted tokens in one forward pass reads the model's weights exactly once, the same weights it would read to decode a single token. That is why `c` and the verify step are treated as fixed costs in the textbook. The verifier processes one token or sixteen for the same weight traffic, so pulling more tokens per pass is close to free.

A mixture of experts breaks that assumption. Qwen3.6-35B-A3B keeps about 3.3 billion parameters active per token, but it picks them: a router selects 8 of 256 experts in each layer, and a different token routes to a different set. We covered this resident-versus-active split in [why an A3B fills the card like a 30B](/blog/2026-05-22-qwen3-30b-a3b-decodes-like-a-3b-and-fills-the-card-like-a-30b). The consequence for speculation is direct. When the verifier processes `K` drafted tokens at once, it does not read one token's experts. It reads the union of the experts that all `K` tokens route to, and that union grows with `K`.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-05-25-moe-expert-saturation-speculative-decode.svg" alt="A two-panel data visualization on a deep oxblood-burgundy background. The left panel, titled the verify step is not free on a sparse MoE, is a line chart. Its horizontal axis is draft width K, the number of tokens verified in one forward pass, from 1 to 128. Its vertical axis is the number of distinct experts the verify pass must load, out of 256. A rising gold curve starts at 8 experts when K equals 1, climbs steeply through about 57 experts at K equals 8 and 102 at K equals 16, and bends over to a plateau near 243 experts, about 95 percent of all of them, at K equals 94, where a vertical dashed cream line is labeled expert-saturation threshold, about 94 tokens. A flat dotted cream line sits along the bottom at the 8-expert level and is labeled the dense-model assumption, verify cost stays flat in K. A shaded copper band covers K equals 2 to 16 and is labeled real n-gram and draft widths live here, every extra token wakes a fresh expert. The right panel, titled what the RTX 3090 benchmark measured, is a horizontal bar chart of single-request decode in tokens per second on Qwen3.6-35B-A3B, greedy. A tall gold bar reads baseline 135.7. Below it three shorter bars read n-gram cache mean 119.1, 0.8B draft K=8 mean 121.1, and 0.8B draft K=32 mean 120.3, each with a copper whisker extending left to a worst-prompt marker near 59 to 65 tokens per second and each annotated 100 percent draft acceptance. A dashed vertical line marks the 135.7 baseline. A callout reads textbook at 100 percent acceptance and lookahead 8 predicts up to 9 times faster; measured is slower. A second smaller callout reads the same drafts on Qwen3.5-122B-A10B gain 15 to 45 percent, because a 10B active footprint clears the threshold. A footer notes the expert curve is the uniform-routing expectation 256 times one minus 0.96875 to the K, illustrative of the MoESD saturation mechanism, and the decode numbers are from the thc1006 RTX 3090 benchmark." loading="lazy" />
  <figcaption>Left: on a dense model the verify cost would be the flat dotted line, the same weight read at any draft width. On an A3B the verify pass loads the union of routed experts, the rising gold curve, which reaches about 95 percent of all 256 experts by a draft width near 94 tokens. Right: real drafts are only a handful of tokens wide, deep in the steep part of the curve, which is why every measured configuration loses despite 100 percent acceptance. The curve is the uniform-routing expectation and is illustrative; the decode numbers are measured.</figcaption>
</figure>

The left panel is the mechanism and the right panel is the receipt. A draft of eight tokens does not cost the verifier one expert read, it costs roughly fifty-seven experts' worth of weight traffic, because eight independently routed tokens scatter across the expert pool. On a bandwidth-bound card, where decode time is dominated by moving weights, that is the slowdown. The verifier is reading several times the data it would read for a single token, and the few accepted tokens it gets back do not pay for it.

## The threshold, and the model that clears it

There is a clean number for where this stops hurting. [MoESD](https://arxiv.org/abs/2505.19645) defines the batch size needed to saturate the expert set, the point past which adding tokens stops pulling in new experts because you have already touched almost all of them. For 8-of-256 routing the sparsity is about 0.031, and the threshold works out to roughly 94 tokens. Below it, every drafted token has a high chance of waking an expert no other token in the batch needed. Above it, the experts are already loaded and extra tokens really are close to free, exactly as the dense formula assumes.

Real speculation lives far below 94. An n-gram draft of three to sixteen tokens, or a draft model running at lookahead eight, sits in the steep early part of the curve where the union of experts is still climbing fast. That is the regime the benchmark measured, and it is why the answer is a slowdown.

The counter-example confirms the mechanism instead of contradicting it. The same n-gram machinery applied to Qwen3.5-122B-A10B, a mixture of experts with about 10 billion active parameters, gains 15 to 45 percent on the same hardware family. A larger active footprint means a lower saturation threshold, so a modest draft width gets closer to the point where the experts are already loaded. The [Cascade paper](https://arxiv.org/abs/2506.20675) puts the general result plainly: on MoE models, drafted tokens collectively activate more weights and increase verification time by two to three times, which turns into slowdowns of up to 1.5 times when the throughput gain cannot cover it. The same paper notes that the optimal draft width varies by task, by model, and even between requests, which is the next problem.

## Where the free draft still wins

None of this means prompt lookup is a bad idea. On a dense model it is one of the best latency tricks available. The original [prompt lookup decoding](https://github.com/apoorvumang/prompt-lookup-decoding) results show a consistent 2.4x speedup on summarization and context-grounded question answering with Mistral-7B, and 2x to 4x on input-grounded tasks generally, with no change to the output at all because the verifier still decides every token. Hugging Face's [assisted generation](https://huggingface.co/blog/assisted-generation) write-up reached the same conclusion from the dense side: the bottleneck is memory bandwidth, and a free draft that the model accepts is close to free latency. The method is in [vLLM and transformers](https://docs.vllm.ai/en/latest/features/speculative_decoding/) precisely because it works so well on dense models, where the verify step is flat in `K`.

The interesting case is the one that bridges the two worlds. [SuffixDecoding](https://arxiv.org/abs/2411.04975), a NeurIPS 2025 spotlight, builds a suffix tree over the prompt and previous outputs and speculates more tokens when acceptance looks likely and fewer when it does not. On agentic workloads with heavy repetition, the kind where a model quotes long spans of its own earlier output, it reaches up to 5.3x. The reason it can win where a fixed three-token n-gram cannot is that its adaptive long speculation occasionally drafts wide enough to approach the saturation threshold, so the experts it pays to load actually get used by many accepted tokens. Width is the lever a sparse MoE rewards, and only workloads with long, predictable continuations can supply it.

## What this means for zinc

The practical rule for a single-user local engine is the same per-request decision we keep arriving at. Speculation on an A3B model should be off by default and turned on only when the engine has a reason to believe the draft will be both accepted and wide. That is exactly the [Cascade](https://arxiv.org/abs/2506.20675) design: measure the ratio of token gains to verification cost for a request, disable speculation when that utility drops below one, and tune the draft width when it rises above. In their vLLM implementation it caps the worst-case slowdown at 5 percent instead of 1.5 times and adds 7 to 14 percent throughput where speculation helps. The lesson for us is that the gate has to be cheap and per-request, not a launch flag.

There is one honest boundary on the whole argument. This is the single-request, batch-of-one regime that a local desktop assistant lives in, and it is also the regime our work on [filling RDNA4's idle compute units](/blog/2026-05-20-the-kv-split-that-fills-rdna4-idle-compute-units-on-long-context-decode) keeps returning to. Under heavy concurrent batching, the experts are already saturated by the other requests in flight, so the union a draft adds is mostly already resident and the penalty largely disappears. The MoE speculation tax is a property of serving one user at a time, which is precisely the case a local engine on a [Radeon AI PRO R9700](https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9700.html) cares about most.

So the thing to carry forward is narrow and useful. On a sparse mixture of experts, the question is no longer whether the draft is cheap or whether the model accepts it. Both of those can be perfect and the model still slows down. The question is whether the drafted tokens route to experts the verifier was going to read anyway, and below a saturation threshold of about 94 tokens for an A3B they do not. Pick a model whose active footprint clears that threshold, or a workload that can sustain very wide drafts, or leave speculation off and ship the 135.7 tok/s baseline, which on this model is still the fastest single-user decode there is.
