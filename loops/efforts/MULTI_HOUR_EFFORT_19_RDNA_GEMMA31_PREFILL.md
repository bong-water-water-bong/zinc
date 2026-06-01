# Effort 19 - RDNA4 Gemma 4 31B dense prefill parity

Date: 2026-06-01

Target model:

- RDNA node GGUF: `/root/models/gemma-4-31B-it-Q4_K_M.gguf`
- Harness model key: `gemma431b`
- Site artifact id: `gemma4-31b-q4k-m`
- Architecture family: Gemma 4 dense, no MoE
- Primary metric: site-aligned `decode-extended` Long Coding Draft prefill tok/s.

## Why this effort exists

Gemma 4 31B is dense, so it needs a different prefill effort from the 26B
MoE model. The current RDNA public artifact is mixed:

| scenario | prompt toks | ZINC prefill | llama.cpp prefill | ZINC pct | ZINC decode | llama.cpp decode |
|---|---:|---:|---:|---:|---:|---:|
| core | 49 | 41.64 | 201.97 | 20.6% | 24.65 | 28.55 |
| context-medium | 192 | 71.20 | 50.01 | 142.4% | 24.17 | 28.19 |
| context-long | 346 | 74.36 | 46.82 | 158.8% | 23.95 | 28.09 |
| decode-extended | 70 | 49.24 | 242.37 | 20.3% | 24.28 | 28.21 |

ZINC already beats llama.cpp on the medium and long prefill scenarios, but
the public short-prompt rows remain about 5x behind. The primary gap is
`decode-extended` prefill: ZINC needs about `+193.13 tok/s` to match
llama.cpp. Core prefill needs about `+160.33 tok/s`.

## Hypothesis

Effort 9 fixed the correctness blockers that kept Gemma 31B out of batched
prefill:

- V RMS norm
- `use_k_as_v` handling
- asymmetric full-attention Q/KV head dims
- post attention/FFN norms
- sliding-window attention behavior

That got Gemma 31B into the right architectural path, but the current public
rows show that short prompts still underperform. The remaining gap is likely
one or more of:

1. Dense projections still use DMMV-shaped kernels that reread weights across
   prompt tokens instead of a true tiled GEMM/MMQ path.
2. Fixed setup/submit/scratch overhead dominates at 49-70 prompt tokens.
3. Batched flash attention is correct but not tuned for Gemma's full-attn vs
   SWA head-dim split.
4. Chunk thresholds were tuned for larger context prompts and are wrong for
   the short public rows.

This effort owns those dense Gemma prefill gaps. It should not edit MoE
routing or Gemma 26B expert logic.

## Measurement contract

The controller benchmark is the public Long Coding Draft chat prompt:

```text
Write an implementation plan for adding a stable benchmark preset to a
local LLM CLI. Include the command shape, warmup policy, metrics to
collect, failure handling, llama.cpp comparison, and how the site should
display prefill, decode, latency, and overall prompt+decode throughput.
```

Run shape:

- Model: `/root/models/gemma-4-31B-it-Q4_K_M.gguf`
- Prompt mode: chat
- Primary metric: ZINC prefill tok/s
- Generation cap in loop: 8 tokens, because prefill is the metric
- llama.cpp target for primary scenario: `242.37 tok/s` prefill

Success is not one primary improvement. A useful keep should:

1. Improve long-draft prefill over the best accepted checkpoint.
2. Preserve coherent output on the five-model coherence sweep.
3. Improve or at least not regress core prefill.
4. Preserve the medium/long rows where ZINC already beats llama.cpp.
5. Keep Gemma-specific validation green.

## First-cycle requirements

Before changing kernels, establish the exact active path:

1. Run long-draft prefill with `ZINC_PREFILL_PROFILE=1`.
2. Run a short core-shaped prompt with the same profile if feasible.
3. Confirm `canUseBatchedPrefillRdna` takes the Gemma batched path.
4. Record dense FFN, attention, QKV/O projection, post-norm, LM-head, and
   submit/command overhead buckets.
5. Identify whether the 49-70 token rows are bottlenecked by projection
   kernels, attention, or fixed overhead.

If the profile shows a different top bucket than expected, update this file
before implementing.

## Candidate implementation path

### Track 1 - Layer-local Gemma dense validator

Add or reuse a validator that can stop after one Gemma layer and compare
token-major vs batched tensors:

- attn norm input/output
- Q/K/V projections
- full-attn vs SWA path outputs
- post_attention_norm output
- FFN gate/up/down outputs
- post_ffw_norm output
- layer residual output
- final logits where practical

This validator is required before changing production Gemma prefill
dataflow. Effort 9 proved that small Gemma ordering mistakes can produce
large logit divergence.

### Track 2 - True Q4_K GEMM/MMQ over prompt tokens

If dense projections dominate, replace the DMMV-shaped batched work with a
true tiled matrix-matrix path over prompt tokens.

Priority should follow the profile, but the expected order is:

1. FFN gate/up
2. FFN down
3. attention Q/K/V
4. attention O projection

Use `mul_mm_q4k.comp` as the in-tree reference, but do not wire it blindly.
The prompt-token batch sizes that matter here are about 49 and 70, not only
300+ tokens.

### Track 3 - Short-prompt overhead

The public gaps are short prompts. If profile shows setup overhead instead
of kernel time:

- cache/reuse per-layer command-buffer pieces where legal
- avoid rebuilding scratch layouts per chunk
- reduce unnecessary transfer/compute barriers only when a named measured
  barrier cost exists
- tune chunk thresholds specifically for 49 and 70 prompt tokens

Do not accept a change that only improves 192/346-token prompts while core
and long-draft stay flat.

### Track 4 - Batched flash attention shape tuning

If attention is hot, audit the Gemma shapes:

- full-attention layers use larger asymmetric Q/KV head dims
- SWA layers use smaller symmetric dims
- prompts under the SWA window should still use the simplest causal path

Prefer a shape-specific batched flash-attn improvement over generic shader
cleanup. Validate full-attn and SWA layers separately.

### Track 5 - Q8_1 activation quant as a follow-up

Only after a structural GEMM/MMQ path lands, test Q8_1 activation quant on
the exact Gemma 31B projection shapes. The result is not transferable from
Qwen. Keep it behind a flag and measure flag OFF/ON in the same cycle.

## Known traps

- Do not optimize Gemma 26B MoE here; this model is dense.
- Do not chase context-medium/context-long first; those rows already beat
  llama.cpp on prefill.
- Do not regress core while improving long-draft.
- Do not bypass the Gemma validator.
- Do not port Qwen shape constants or Qwen-only env flags.
- Do not change `.spv` files directly; add or edit `.comp` and build.
- Do not keep dormant shader infrastructure without a measured caller.

## Full-matrix follow-up

After a material keep above 80 tok/s on the primary metric, run or prepare:

```bash
bun tools/performance_suite.mjs \
  --target rdna \
  --phase all \
  --rdna-sync \
  --rdna-build \
  --models gemma4-31b-q4k-m \
  --rdna-start-llama \
  --rdna-vk-device 1 \
  --require-rdna-device-substring GFX1201 \
  --runs 3 \
  --warmup 1
```

Success means closing the core and long-draft prefill gaps without losing the
medium/long prefill lead.
