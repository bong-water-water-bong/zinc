# Effort 26 — CUDA gemma prefill GEMM tuning (close the 0.15–0.26× gap vs llama.cpp)

> **Status:** OPEN. The single BIGGEST CUDA-vs-llama drawback. The batched prefill
> (Effort 24) is wired + production-default, but the GEMM is "wired, not tuned":
> on the **5090** gemma prefill is **0.15–0.26× llama** — gemma-31b 57.3 vs llama
> ~380 t/s (0.15×), gemma-26b 106.3 vs ~410 (0.26×). qwen prefill BEATS llama (SSM
> bound on both sides), so this is gemma/transformer-prefill specific. See memory
> `project_cuda_perf_blog` (fresh 2026-06-13 5090 catalog) + `project_batched_prefill_design`.

## WHY the gap (diagnosed)
The batched GEMM (`gemm_*_tiled_v2`, + opt-in `gemm_q4k_tc` wmma) is MEMORY-bound:
the +9–12% TC win was small because Q4_K dequant + f32 activation reads dominate
(the fp16 multiply was never the bottleneck). llama's prefill GEMM keeps weights
+ activations in a tight fp16/quant-aware tensor-core loop. We re-read + re-dequant
each GEMM. Closing 0.15→~1× is the work.

## TARGETS (one per cycle, biggest lever first)
1. **fp16 weight cache** — dequant each gemma layer's Q4_K/Q5_K/Q6_K weight to fp16
   ONCE (per prefill, not per-GEMM call), reuse across the T-token GEMM. Kills the
   per-GEMM dequant that dominates. (Effort-24 cycle 12 hinted at this as the lever.)
2. **fp16 activations across the layer** — keep the residual/activation stream in
   fp16 through the GEMM chain (kill the f32↔fp16 recast each GEMM; norm/GeGLU
   already emit fp16, Effort-24 cycle 21).
3. **TC tile/occupancy** — full wmma utilization: bigger M/N tiles, more warps,
   double-buffered shared loads; profile the achieved % of 5090 fp16 peak.
4. **gemma-26b MoE prefill expert GEMMs** — the grouped/batched-expert GEMMs are the
   other half of the gemma-26b gap.

## GATE
`scripts/validate_catalog.sh` (ZINC_BATCHED=1, + ZINC_BATCHED_TC=1 for the TC path)
5/5 token-correct vs llama.cpp. `scripts/prefill_catalog.sh` measures the delta;
TARGET: gemma prefill 0.15–0.26× → first ≥0.5×, then llama-parity. fp16-TC path is
NOT byte-identical → token-correctness gate, not GEN_IDS identity.

## HARD RULES
isolated worktree `~/zinc-eNN`, box `~/zinc-eNN-box`, 4090-pinned
(`GPU-e59a6fce-…`) for dev / 5090 for the headline number, isolated-cache builds
(verify hash changed). Branch off LATEST origin/main, rebase often (parallel team
pushes fast), ADDITIVE (the batched path is production-default now — don't regress
it; opt-out is `ZINC_BATCHED_PREFILL=0`). NEVER touch `~/Workspace/zinc` (parallel
team's live checkout) or push to main. Gate before any commit.

## CYCLE LOG
- (none yet)
