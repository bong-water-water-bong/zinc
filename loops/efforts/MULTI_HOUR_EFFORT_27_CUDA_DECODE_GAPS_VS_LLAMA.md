# Effort 27 — CUDA decode gaps vs llama.cpp (MoE 31–42%, dense 75–82%)

> **Status:** OPEN. zinc CUDA decode averages **70% of llama** on the 5090 (avg
> 63.7 vs ~90 t/s). The gap is concentrated in MoE; dense is close. Fresh 2026-06-13
> 5090 catalog (memory `project_cuda_perf_blog`), decode tok/s zinc vs llama (% of llama):
> - **gemma-26b-a4b MoE: 47.5 vs ~153 (31%)**  ← worst
> - **qwen36-35b-a3b MoE: 52.9 vs ~126 (42%)**
> - qwen35-9b dense: 120.8 vs ~161 (75%)
> - gemma-31b dense: 46.9 vs ~57 (82%)
> - qwen36-27b dense: 50.5 vs ~55 (91%)  ← already fine, leave it
> NOTE: 5090 decode has WIDE boost variance — always interleaved A/B, medians-of-3,
> and treat <5% as noise. The MoE gap (3×, 2.4×) is real and structural, not noise.

## TARGET 1 (PRIORITY) — MoE decode (the 31–42% gap)
Per-token MoE decode = router (norm→F32 gate_inp matvec→top-k) + 8 routed-expert
matvecs (Q4_K gate/up + Q5_1 down) + Q8_0 shared expert + weighted combine, per
layer (gemma-26b: 30 layers ALL MoE; qwen-35b-a3b: most layers). Suspected levers:
- **Profile first** (clock + util + the per-op breakdown) — is it boost-starvation
  (too many tiny launches/token → GPU idles) or matvec inefficiency? (The async ring
  helped prefill; confirm decode MoE is fully on the async submit path, 1 sync/token.)
- **Fuse the per-token expert path**: router + experts + shared + combine are many
  small launches/layer; collapse where bit-exact (the Effort-24 prefill kernels are
  batched twins — the single-token decode path may still be launch-heavy).
- **Expert matvec efficiency**: dmmv_q4k/q5_1 for the 8 active experts — memory
  access / dp4a / block size vs llama's gathered-expert matvec.
GATE: validate_catalog 5/5 token-correct; perf_catalog decode A/B (interleaved,
medians-of-3) vs the pre-cycle binary AND vs llama. TARGET: 31–42% → ≥60% of llama.

## TARGET 2 — dense decode (the 75–82% gap)
qwen35-9b 75%, gemma-31b 82%. Async ring + fast matvecs already shipped → smaller,
boost-floor-sensitive. Levers: LM-head matvec (vocab×n_embd, the single biggest
read/token — Q6_K fast-dot / TC), residual matvec bandwidth %peak, remaining
fusible launches (Effort-23 playbook, but ONLY if it clears the boost floor —
needs locked clocks or ≥2-launch aggregate fusions; single-launch fusions = noise).

## HARD RULES
isolated worktree `~/zinc-eNN`, box `~/zinc-eNN-box`, 4090 for dev / 5090 for the
headline, isolated-cache builds (verify hash changed). Branch off LATEST origin/main,
rebase often, ADDITIVE, NEVER touch `~/Workspace/zinc` or push to main. Interleaved
back-to-back A/B (boost noise); never claim a win from one boosted run. Gate before commit.

## CYCLE LOG
- (none yet)
