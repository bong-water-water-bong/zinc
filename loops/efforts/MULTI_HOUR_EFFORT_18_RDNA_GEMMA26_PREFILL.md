# Effort 18 - RDNA4 Gemma 4 26B-A4B MoE prefill parity

Date: 2026-06-01

Target model:

- RDNA node GGUF: `/root/models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf`
- Harness model key: `gemma426ba4b`
- Site artifact id: `gemma4-26b-a4b-q4k-m`
- Architecture family: Gemma 4 MoE with fused `ffn_gate_up_exps`
- Primary metric: site-aligned `decode-extended` Long Coding Draft prefill tok/s.

## Why this effort exists

The current RDNA public artifact shows that Gemma 4 26B-A4B still has a
large prefill gap on every scenario:

| scenario | prompt toks | ZINC prefill | llama.cpp prefill | ZINC pct | ZINC decode | llama.cpp decode |
|---|---:|---:|---:|---:|---:|---:|
| core | 49 | 89.10 | 497.08 | 17.9% | 89.73 | 102.00 |
| context-medium | 192 | 89.55 | 186.67 | 48.0% | 90.16 | 100.84 |
| context-long | 346 | 93.33 | 169.18 | 55.2% | 89.04 | 100.40 |
| decode-extended | 70 | 92.68 | 647.16 | 14.3% | 91.13 | 101.05 |

The primary public gap is `decode-extended` prefill: ZINC needs about
`+554.48 tok/s` to match llama.cpp. Core prefill is also far behind at
`+407.98 tok/s`.

Effort 13 covers Gemma 26B sustained decode and the CPU MoE fallback, but
decode-time matvec-ID is not enough for this prefill problem. Prefill needs
token-grouped expert work across all prompt tokens.

## Hypothesis

Effort 9 opened the Gemma batched prefill path and fixed the major dense
Gemma correctness bugs. The remaining 26B-A4B gap is likely the MoE section:
router/top-k and selected expert work are still shaped like per-token decode
instead of prompt-token grouped prefill.

The llama.cpp structural advantage is not a small shader trick:

1. Router/top-k stays on GPU.
2. Selected tokens are grouped by expert.
3. Expert gate/up/down work runs over batches of token rows.
4. Weighted expert outputs scatter back to their original token rows.

ZINC needs the same dataflow while preserving Gemma semantics:

- unit router RMS norm plus `ffn_gate_inp.scale`
- optional `pre_ffw_norm_2`
- fused `ffn_gate_up_exps`
- selected-only normalized top-k weights
- `ffn_down_exps.scale`
- `post_ffw_norm_1`, `post_ffw_norm_2`, and `post_ffw_norm`

## Measurement contract

The controller benchmark is the public Long Coding Draft chat prompt:

```text
Write an implementation plan for adding a stable benchmark preset to a
local LLM CLI. Include the command shape, warmup policy, metrics to
collect, failure handling, llama.cpp comparison, and how the site should
display prefill, decode, latency, and overall prompt+decode throughput.
```

Run shape:

- Model: `/root/models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf`
- Prompt mode: chat
- Primary metric: ZINC prefill tok/s
- Generation cap in loop: 8 tokens, because prefill is the metric
- llama.cpp target for primary scenario: `647.16 tok/s` prefill

Success is not one lucky primary sample. A useful keep should:

1. Improve long-draft prefill over the best accepted checkpoint.
2. Preserve coherent output on the five-model coherence sweep.
3. Preserve Gemma-specific MoE semantics.
4. Move in a direction that can also close core prefill.
5. Avoid regressions on context-medium/context-long, where the gap is smaller.

## First-cycle requirements

Before changing kernels, establish the exact active path:

1. Run the baseline with `ZINC_PREFILL_PROFILE=1`.
2. Record whether `cpu_moe_fallbacks` appears during prefill.
3. Record prefill sub-buckets for router/top-k/gate_up/swiglu/down/
   weighted_acc/shared.
4. Confirm whether the existing Gemma batched prefill path is active.
5. Identify whether the largest bucket is routed MoE, shared expert,
   dense attention, final LM head, or command/submit overhead.

If MoE is not a top bucket, update this file before editing code.

## Candidate implementation path

### Track 1 - Gemma MoE prefill validator

Add a validation mode that runs one Gemma MoE layer both ways:

- token-major reference path
- candidate grouped GPU path

Compare:

- router logits after Gemma scale
- top-k expert IDs and weights
- fused gate/up outputs
- SwiGLU outputs
- down outputs with `ffn_down_exps.scale`
- per-token weighted accumulation
- post-norm outputs
- final logits where practical

A validator that proves parity is a valid foundation keep even if the
production path remains off for that cycle.

### Track 2 - GPU top-k for all prompt tokens

Move router/top-k on GPU for the prompt-token batch. Do not consume it until
validation proves it matches CPU `topKSoftmax` for Gemma.

The output should be a compact selected-token table:

- token index
- expert id
- top-k slot
- normalized expert weight

Fail closed on any ID mismatch or weight mismatch above tolerance.

### Track 3 - Expert grouping and compaction

Build the token-by-expert schedule:

1. Count selected tokens per expert.
2. Prefix-sum expert offsets.
3. Compact selected token rows into expert-contiguous activation tiles.
4. Keep a scatter map back to `(token, topk_slot)`.

This is the structural prefill lever. Avoid a dispatch grid of
`token x expert x row`; that repeats the per-token decode shape and was the
dead end in earlier Qwen MoE batching.

### Track 4 - Batched fused gate/up and down

Gemma 26B uses fused `ffn_gate_up_exps`. Do not copy or split weights.

Two viable options:

1. Bind the same expert tensor twice with descriptor offsets so gate and up
   see different base offsets.
2. Write a Gemma-specific batched fused-gate-up shader with an
   `up_base_offset` push constant.

After gate/up, run SwiGLU and the expert down projection over the compacted
token rows, then scatter weighted outputs back into the original token rows.

### Track 5 - Thresholds by prompt length

The four public prompt sizes are not the same workload:

- core: about 49 ZINC prompt tokens
- long-draft: about 70
- context-medium: about 192
- context-long: about 346

Use separate thresholds if needed. A path that wins only above 192 tokens
does not close the core and long-draft gaps this effort owns.

## Known traps

- Do not optimize decode first; this effort is prefill-scored.
- Do not copy the Gemma 31B dense plan. This model's untreated gap is MoE.
- Do not stop at decode-time matvec-ID; prefill needs grouped token batches.
- Do not bypass Gemma norms or per-expert scales for speed.
- Do not split `ffn_gate_up_exps` into a copied temporary weight tensor.
- Do not keep a flag-gated optimization without same-cycle flag OFF/ON
  measurements.
- Do not run long full-matrix suites from inside the agent cycle. The
  controller owns official benchmarking.

## Full-matrix follow-up

After a material keep above 150 tok/s on the primary metric, run or prepare:

```bash
bun tools/performance_suite.mjs \
  --target rdna \
  --phase all \
  --rdna-sync \
  --rdna-build \
  --models gemma4-26b-a4b-q4k-m \
  --rdna-start-llama \
  --rdna-vk-device 1 \
  --require-rdna-device-substring GFX1201 \
  --runs 3 \
  --warmup 1
```

Do not call the effort successful until core, context-medium, context-long,
and decode-extended are all directionally sane.
