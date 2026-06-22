---
title: "Why Qwen 35B cannot use ZINC's 208 tok/s batched prefill path yet"
seoTitle: "Qwen 35B Batched Prefill on RDNA4"
date: "2026-06-06"
tags:
  - zinc
  - rdna4
  - amd
  - vulkan
  - prefill
  - qwen3-6
  - qwen35
  - moe
  - ssm
  - local-llm
  - llm-inference
  - gpu-kernels
keywords:
  - Qwen 35B batched prefill
  - Qwen3.6 35B A3B RDNA4
  - ZINC RDNA4 prefill gate
  - MoE SSM batched prefill
  - canUseBatchedPrefillRdna
  - gated delta net prefill
  - block resident SSM state
  - batched MoE routing Vulkan
  - Radeon AI PRO R9700 Qwen3.6
  - local LLM time to first token
  - ZINC_BATCHED_PREFILL Qwen 35B
faqs:
  - question: "Why can't Qwen 35B use the same 208 tok/s batched prefill path as Qwen3-8B?"
    answer: "The 208 tok/s path is the dense batched-prefill path. Qwen3.5 and Qwen3.6 35B-A3B are hybrid MoE plus SSM models, so their prompt path needs batched expert routing, per-expert accumulation, and block-resident SSM state. Dense column-batched DMMV is necessary but not sufficient."
  - question: "Is Qwen 35B slow on ZINC overall?"
    answer: "No. Qwen3.6 35B-A3B is one of ZINC's strongest decode rows on RDNA4, and the June 1 dashboard shows ZINC decode ahead of llama.cpp on that model. The weak phase is prefill: prompt ingestion still trails because the hybrid model falls back to a per-token schedule."
  - question: "What has to land before the gate can open?"
    answer: "Three pieces matter most: a batched MoE route-pack path that groups tokens by expert, a batched per-expert matmul/scatter path that accumulates routed outputs back into token rows, and an SSM gated-delta kernel that walks prompt tokens with state resident in the block instead of reloading state for every token."
  - question: "Will one more DMMV shader close the Qwen 35B prefill gap?"
    answer: "No. More DMMV tuning can still help individual buckets, but the remaining Qwen 35B prefill wall is graph shape. The expensive part is moving the hybrid prompt path from thousands of decode-shaped dispatches to a small number of batched model operations."
excerpt: "Yesterday's RDNA4 post showed ZINC moving dense Qwen3-8B prefill from 42.9 to 207.9 tok/s. The tempting question is why Qwen3.6 35B-A3B cannot simply reuse that path. The answer is the hybrid wall: dense batching solved repeated weight reads for a transformer, while Qwen 35B needs batched MoE routing and block-resident SSM state before its prefill gate can come down."
seoDescription: "Why ZINC's 208 tok/s RDNA4 dense batched prefill path does not yet apply to Qwen3.6 35B-A3B, and how MoE plus SSM batching closes the gap."
---

Yesterday's RDNA4 prefill post had the kind of number that invites the wrong conclusion: Qwen3-8B moved from **42.9 tok/s** on a long per-token prompt to **207.9 tok/s** through ZINC's shipped batched-prefill path.

If you searched for **Qwen 35B batched prefill on RDNA4**, this is the practical answer: Qwen3.6 35B-A3B is already one of ZINC's best decode results, but it cannot use that dense 208 tok/s prompt path yet.

The natural question is: why not just use that path for Qwen3.6 35B-A3B?

That is today's post. The short answer is that the 208 tok/s path solved the dense transformer problem. Qwen3.5 and Qwen3.6 35B-A3B are not dense transformer prefill problems. They are hybrid MoE plus SSM prefill problems, and the difference is not a footnote. It changes the graph.

The dense win proved that the Vulkan batched-prefill machinery is real. It did not remove the hybrid gate.

## The easy misunderstanding

The dense path sounds general if you describe it too quickly:

1. Read a chunk of prompt tokens.
2. Run projection kernels over many prompt columns at once.
3. Reuse each weight row across the chunk.
4. Write a batch of KV cache entries.
5. Continue into decode with the same logits.

That is exactly what [the 42 to 208 tok/s post](/blog/2026-06-05-how-zinc-rdna4-batched-prefill-went-from-42-to-208-tok-s/) covered. It fixed dead Vulkan wiring, Q6_K dispatch, stale GPU argmax state, and the serial-over-K batched DMMV shape. The end result was a dense Qwen3-8B path that made the prompt side behave like prompt work instead of 658 tiny decode steps.

But Qwen3.6 35B-A3B is not just a bigger Qwen3-8B. It is a hybrid architecture:

| Component | Dense Qwen3-8B prefill | Qwen3.6 35B-A3B prefill |
| --- | --- | --- |
| FFN | Dense SwiGLU for every token | Routed MoE plus shared expert |
| Token routing | None | Top-k expert selection per token |
| State-space layers | None | Gated delta-net SSM layers |
| Prompt state | Attention KV cache | KV cache plus recurrent SSM state |
| Good batching primitive | Column-batched DMMV | Expert grouping plus block-resident recurrence |

The dense path answers: "How do we stop rereading the same weights once per prompt token?"

The hybrid path has to answer two more questions:

1. "How do we batch tokens that route to different experts?"
2. "How do we walk prompt tokens through a recurrent SSM state without reloading the state every token?"

Until those are answered, Qwen 35B cannot simply inherit the 208 tok/s dense result.

## The current public shape

The public benchmark story already hints at this split. In the June 1 [ZINC benchmark dashboard](/zinc/benchmarks/) data, Qwen3.6 35B-A3B is one of ZINC's best RDNA decode rows: **127.9 tok/s** for ZINC against **108.5 tok/s** for llama.cpp on the Radeon AI PRO R9700.

That is not the problem.

The same dashboard row has ZINC prefill at **154.27 tok/s** against llama.cpp at **398.82 tok/s**. The [RDNA4 tuning note](/zinc/docs/rdna4-tuning/), measured under a different profiling regime, shows the same shape with a larger prompt-side gap: **88.08 tok/s** for ZINC prefill against **181.95 tok/s** for llama.cpp.

The exact number changes with prompt shape, harness, and measurement mode. The conclusion does not:

| Phase | What the numbers say |
| --- | --- |
| Decode | ZINC's hot one-token loop can beat llama.cpp on Qwen3.6 35B-A3B. |
| Prefill | ZINC still trails because the hybrid prompt graph is not batched like llama.cpp's. |
| End-to-end | The user-visible wait inherits the prefill and harness costs. |

So the roadmap is not "make Qwen 35B fast." The roadmap is narrower and more useful: keep the decode win, then move the hybrid prompt path out of the per-token schedule.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-06-06-qwen35-decode-prefill-gap.svg" alt="Qwen 35B RDNA4 benchmark chart comparing ZINC and llama.cpp on decode and prefill. ZINC decodes Qwen3.6 35B-A3B at 127.9 tok/s against llama.cpp at 108.5 tok/s, or 117.9 percent of baseline. On prefill, ZINC reaches 154.27 tok/s against llama.cpp at 398.82 tok/s, or 38.7 percent of baseline." loading="lazy" />
  <figcaption>The educational point is phase separation: the same Qwen 35B run is a ZINC decode win and a prompt-ingestion backlog.</figcaption>
</figure>

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/qwen35-prefill-phase-budget-cycle50.svg" alt="Per-phase GPU time budget for a 154-token Qwen 3.6 35B-A3B prefill on the Radeon AI PRO R9700. The chart breaks ZINC prefill into SSM, MoE, attention, and shared-expert buckets and compares the wall time to llama.cpp." loading="lazy" />
  <figcaption>The old cycle-50 profile is still useful because it shows the shape of the wall: SSM and MoE dominate, not a missing dense DMMV variant.</figcaption>
</figure>

## The gate is doing its job

The reason Qwen 35B does not fall into the dense batched path is explicit. The gate looks conceptually like this:

```zig
fn canUseBatchedPrefillRdna(cfg: Config) bool {
    if (cfg.n_experts > 0) return false;
    if (cfg.ssm_d_inner > 0) return false;
    // dense architecture checks continue...
    return true;
}
```

That can look like a frustrating early return. It is also correct.

Removing the first guard would let a dense batched FFN path pretend it can handle sparse expert routing. It cannot. Removing the second guard would let a transformer prompt loop pretend it can handle token-recurrent SSM state. It cannot.

The dense path is allowed to batch columns because every token uses the same weight matrix in the same way. MoE breaks that. Each token chooses a small set of experts, and each selected expert sees a different subset of the prompt. SSM breaks a different assumption. Prompt token `t + 1` depends on the recurrent state after token `t`, so the prompt axis is not just a bag of independent columns.

The gate is not the performance bug. It is the place where missing architecture support is refusing to become a correctness bug.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-06-06-qwen35-prefill-gate-map.svg" alt="Architecture gate chart for Qwen 35B batched prefill on RDNA4. The dense transformer lane shows prompt chunks passing through column-batched DMMV and KV cache writes into the 208 tok/s path. The Qwen3.6 35B-A3B lane shows two closed gates: missing MoE route-pack support and missing block-resident SSM prompt state." loading="lazy" />
  <figcaption>The dense path is narrow by design. Qwen 35B needs route-packed MoE and block-resident SSM before the batched-prefill gate can safely open.</figcaption>
</figure>

## Why MoE batching is not dense batching

A dense FFN projection has a simple shape:

```text
Y[prompt_token, row] = W[row, :] dot X[prompt_token, :]
```

Batching that is mechanically clean. Put multiple `X` columns next to each other, read a row of `W`, accumulate many outputs.

MoE adds routing:

```text
experts = top_k(router(X[token]))
for expert in experts:
    Y[token] += weight[token, expert] * ExpertFFN[expert](X[token])
```

Now a batch of 154 prompt tokens is not one matrix multiply. It is a routed workload. Expert 7 might get 11 tokens, expert 12 might get 4, expert 41 might get none, and every token has several selected experts.

A useful batched MoE path needs at least four pieces:

| Piece | What it does |
| --- | --- |
| Route capture | Store `(token, expert_slot, expert_id, weight)` for the prompt chunk. |
| Expert counts | Count how many routed token rows each expert owns. |
| Packed expert rows | Build or infer a compact token list per expert so idle workgroups exit early. |
| Scatter and accumulate | Write each expert output back to the original token row with the top-k weight. |

That is why a generic column-batched DMMV is not enough. Dense batching asks "how many columns are active?" MoE batching asks "which columns belong to this expert, and where do their outputs return?"

llama.cpp's mature path handles this with matmul-id style kernels and per-expert counts. ZINC has pieces of this story in the tree and in prior experiments, but the hard part is the wire-up: route buffers, packed rows, accumulation buffers, and validation against the per-token oracle.

This is also why the previous attempt to port tiled GEMM infrastructure did not pay off. A correct shader with no caller in the hot path is not a performance feature. For MoE, wiring is the feature.

## Why SSM batching is not attention batching

Attention prefill has a large parallel shape: many query positions can be processed with a causal mask, while the KV cache receives the prompt positions.

Gated delta-net SSM is different. It carries recurrent state.

The per-token shape is easy to reason about:

```text
state_0 = previous_state
for token in prompt:
    state_next, y[token] = gated_delta_net(token, state_current)
    state_current = state_next
```

The expensive version launches that logic once per token and reloads the recurrent state every time. The useful batched version does not turn the recurrence into independent columns. It keeps the state resident inside the block, walks the prompt tokens in order, and writes the final state once.

That is the important distinction:

| Wrong mental model | Better mental model |
| --- | --- |
| "Batch SSM like dense DMMV." | "Run the token loop inside the SSM kernel while state stays resident." |
| "Parallelize every prompt token independently." | "Exploit parallelism inside the state update and avoid per-token state reloads." |
| "One more projection shader fixes it." | "The recurrence lifetime has to move from host loop to GPU block." |

The older Qwen 35B profile estimated that the SSM block was the largest prefill bucket, with `ssm_delta` and SSM projections carrying most of the cost. The exact percentages have moved over time, but the model fact has not: 30 of the 48 layers are SSM-shaped, and token-recurrent state is central to the prompt path.

That is why the hybrid gate cannot come down until SSM prefill has its own batched kernel shape.

## Why the 208 tok/s path still matters

None of this makes yesterday's dense result less important. It makes it more useful.

The dense path proved several things that the hybrid path still needs:

| Proven by dense batching | Why Qwen 35B still benefits |
| --- | --- |
| Vulkan `prefillBatched` can be a live path, not dead code. | Hybrid batching should reuse the same entry point and validation discipline. |
| Validate mode can compare batched logits against the per-token oracle. | MoE and SSM ports need the same trust boundary before the gate opens. |
| K-parallel batched Q4_K/Q6_K DMMV works on `gfx1201`. | Hybrid projections still need fast source-format row work. |
| `MAX_COLS=40` is a real RDNA4 chunk-size result. | Prompt chunking remains a scheduling knob even after MoE and SSM land. |
| Dense Gemma caught architecture-specific V handling. | Hybrid Qwen needs the same refusal to paper over model-family details. |

The right takeaway is not "dense batching failed to solve the flagship." The right takeaway is "dense batching removed one uncertainty." We now know the prompt machinery can be correct and fast on a dense model. The remaining uncertainty is the hybrid graph.

That is progress.

## The work that actually opens the gate

The next Qwen 35B prefill work should be judged by whether it removes one of the two guards.

For the MoE guard, the first useful milestone is not a giant speedup. It is a validated route-pack slice:

1. Capture router output for a prompt chunk.
2. Count selected tokens per expert.
3. Run one expert projection against a compact token list.
4. Scatter the weighted result back into the right token rows.
5. Compare against the per-token oracle.

That can start small. One layer, one projection, a limited expert subset. The point is to prove the data contract before scaling the kernel.

For the SSM guard, the first useful milestone is a block-resident recurrence slice:

1. Pick one SSM layer and one prompt chunk.
2. Load the recurrent state into GPU-local storage once.
3. Walk prompt tokens inside the kernel in order.
4. Emit the same final state and token outputs as the per-token path.
5. Validate finite deltas before allowing the live decode path to consume the result.

Again, the first version does not need to be the final fastest kernel. It needs to prove the lifetime: state enters once, token loop happens on GPU, state leaves once.

The third milestone is reporting. If a run still falls back to per-token MoE or per-token SSM, the benchmark should say so. A coherent answer is not enough evidence. A fast dense prefill number is not enough evidence. The dashboard needs to know whether the hybrid gate was actually open.

## The product implication

This matters for v0.1 because Qwen3.6 35B-A3B is exactly the kind of model people will use to judge ZINC: big enough to be interesting, sparse enough to fit on a 32 GB card, and fast enough on decode to show why the project exists.

The release story should therefore be precise:

| Claim | Status |
| --- | --- |
| ZINC can beat llama.cpp on Qwen3.6 35B-A3B decode on RDNA4. | Supported by the dashboard data. |
| ZINC has a real dense RDNA4 batched prefill path. | Supported by the Qwen3-8B 42.9 to 207.9 tok/s result. |
| Qwen3.6 35B-A3B uses that dense batched path. | Not yet. |
| The remaining Qwen 35B prefill work is one shader away. | No. It is MoE routing plus SSM state lifetime. |

That is not a weak story. It is an honest one.

Dense batching turned a raw prompt-loop idea into a measured production path. The hybrid wall tells us where the next engineering project begins. Once route-packed MoE and block-resident SSM are validated, the same `canUseBatchedPrefillRdna` gate that protects correctness today becomes the switch that turns the flagship prompt path on.

For now, Qwen 35B cannot use the 208 tok/s path because it is solving a harder problem. The useful part is that we can name the problem now.
