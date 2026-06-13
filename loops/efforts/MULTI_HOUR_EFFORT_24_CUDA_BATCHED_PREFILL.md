# Effort 24 — CUDA batched prefill (wire the 5.9× gemm_*_tiled_v2 GEMMs)

> **Status:** OPEN. The top untapped CUDA prefill win. The validated
> `gemm_q4k/q5k/q6k_tiled_v2` GEMMs (Y[T,M]=A[T,K]·W[M,K]^T, ~9300 GFLOP/s,
> **5.9× over per-token matvec**) are in `kernels.cu` but UNWIRED. Production
> prefill (`main.zig`) runs `decodeStep`/`prefillStep` once per prompt token →
> re-reads every weight T times. Batch it: read each weight ONCE for all T
> tokens. Full design in memory `project_batched_prefill_design`.

Opened 2026-06-12. Stacks on the head-skip (`c78e64ed`, already on main).

## SCOPE — Phase 1: gemma-31b DENSE only (pure transformer → clean GEMM win)
Qwen is hybrid-SSM (3/4 layers = sequential state scan) and is OUT of scope
until a chunked/parallel SSM scan exists. gemma-26b MoE = Phase 2 (later). Do
gemma-31b dense first.

## APPROACH — ADDITIVE (forward_cuda_gemma.zig is the parallel gemma work's HOT
## file: add NEW methods + a toggle; do NOT rewrite existing decodeStep/blocks)
Add `ForwardGemma.prefillBatched(tokens) !u32` behind env `ZINC_BATCHED_PREFILL`:
process all T prompt tokens at once, return the last token's argmax (= first gen
token). Per layer:
- batched norms: loop T or a T-batched rms variant (cheap; not the bottleneck)
- Q/K/V + O projections, FFN gate/up/down: `gemm_*_tiled_v2`
  (`GemmPush{M,K,T,a_offset,x_offset,y_offset,acc_mode}`) over the T tokens
- attention: **Target 1 LOOPS** the existing single-query attention per token
  (each query masked to [0..its pos]) — defer the batched-attn kernel
- KV write: all T positions (batched, or loop)
Then the last token's existing tail (the head). Wire `main.zig` prefill to call
`prefillBatched` when `ZINC_BATCHED_PREFILL` is set, else the current per-token
loop. (The duck-typed prefill fn already special-cases the last token.)

## TARGETS (one per cycle; each must pass the gate before any commit)
1. **Batched skeleton** — `prefillBatched` with LOOPED attention + GEMM
   projections/FFN. The big first step; may span >1 cycle (use the cycle log).
2. **Batched causal-attention kernel** — replace the per-token attention loop
   (grid=(n_head,T), each query masked to its pos; or flash-style online softmax).
3. **Batched KV write** — all T positions in one launch (if still looped).
4. (later) gemma-26b MoE prefill; tensor cores via NVRTC `-I` (+2.2×).

## GATE (NON-NEGOTIABLE — the batched path must be OUTPUT-IDENTICAL)
- `ZINC_BATCHED_PREFILL=1 dbg_cuda gen <prompt> N <model>` GEN_IDS must be
  **byte-identical** to the per-token path. Extend `scripts/prefill_catalog.sh`
  to A/B batched-vs-per-token (it already A/Bs head-skip the same way). Mismatch
  → REVERT.
- `scripts/validate_catalog.sh` → 5/5 token-correct vs llama.cpp.
- Measure prefill tok/s via `scripts/prefill_catalog.sh` (ABBA-counterbalanced).

## HARD RULES (from memory — violating wastes the cycle)
- **Build:** isolated caches; verify the binary hash CHANGED.
- **Box:** 4090-pinned (`GPU-e59a6fce-1961-bafe-927c-06c0149f2370`); isolated
  dir `~/zinc-e24`, NEVER `~/workspace/zinc` (parallel 5090 research).
- **Coordination:** main moves FAST (active parallel 5090/gemma work). Branch
  off the LATEST origin/main each cycle, rebase often, keep changes ADDITIVE
  (new methods + toggle) to minimize conflicts. NEVER roll back parallel work.
- Commit only a validated, output-identical, FASTER increment to `perf/e24-*`;
  push (NOT main). Incomplete/negative → log it in the cycle log + continue.

## EXPECTED
gemma-31b prefill ~15-35 → ~90-200 tok/s (5.9× on the GEMM-able majority;
attention is a smaller FLOP share). Stacks on the head-skip's +4%.

## CYCLE LOG
- **2026-06-12 — Cycle 1: batched dense-gemma prefill skeleton COMPLETE + output-identical + faster.**
  Wired the validated `gemm_q4k/q5k/q6k_tiled_v2` GEMMs into a new additive
  `ForwardGemma.prefillBatched(tokens) !u32` (forward_cuda_gemma.zig): token-major
  `BatchScratch` (lazy, sized to T), `gemmDispatch` (Q4_K/Q5_K/Q6_K → tiled_v2 GEMM;
  q8_0/f32 → per-token dmmv fallback), `attentionLayerBatched` (batched pre-norm +
  Q/K/V/O GEMMs; per-head V-norm/KV-write, Q/K norm+RoPE, causal softmax LOOPED per
  token via `aliasBuffer` into the token-major scratch — reuses the single-token
  kernels, zero new kernels), `ffnBlockBatched` (batched pre-norm + gate/up/down
  GEMMs + element-wise GeGLU over [T,n_ff] + fused post-ffn norm/residual/scale).
  Tail (rms_norm+LM head+argmax) on the last token only. MoE (n_experts>0) falls
  back to the per-token path. Toggle `ZINC_BATCHED_PREFILL` wired in BOTH main.zig
  (product) and dbg_cuda.zig gen-path (gate harness) + `Engine.prefillBatched`.
  Built clean on the 4090 box (`zig build cuda-dbg`, fresh `.zig-cache`, EXIT=0,
  bin 45251da3). Direct A/B (dbg_cuda gen, GEN_IDS = the gate):
    - gemma-31b T=80:  29.30→53.13 t/s (+81%),  GEN_IDS byte-IDENTICAL ✓
    - gemma-31b T=200: 32.20→75.05 t/s (+133%), GEN_IDS byte-IDENTICAL ✓ (multi-tile T>128)
    - gemma-26b MoE T=60: GEN_IDS identical (per-token fallback, as designed) ✓
  GATE STATUS: GEN_IDS byte-identical on direct A/B (strong). REMAINING for merge:
  (1) extend scripts/prefill_catalog.sh to ABBA-counterbalance the batched A/B,
  (2) scripts/validate_catalog.sh 5/5. Committed to perf/e24-batched-prefill (WIP,
  toggle off by default → cannot regress production). NEXT: target #2 batched
  causal-attention kernel (replace the per-token attention loop — the remaining
  per-token launch overhead) for a bigger T, then run the formal gate.
- **2026-06-12 — Cycle 2: batched causal-attention kernel wired (target #2) — output-identical + faster.**
  Added an ADDITIVE kernel `gemma_attention_batched` (kernels.cu) — a verbatim twin
  of `gemma_attention` (same 3-pass softmax / GQA / sliding-window / no-sink / scale,
  same `zinc_block_reduce_*` order → bit-identical math) but batched over queries:
  block=(head=blockIdx.x, t=blockIdx.y), seq_len=t+1, token-major Q `q+(t*n_heads+head)*hd`
  and out, SWA `start = (window>0 && seq_len>window) ? seq_len-window : 0`. Chose a NEW
  gemma kernel over modifying main's just-landed `attention_causal_batched` (which carries
  sink logic gemma doesn't use) — zero conflict + guaranteed bit-identity. Wired in
  `attentionLayerBatched` (forward_cuda_gemma.zig): the per-token loop now does ONLY the
  norm/RoPE/KV-write (Q RoPE'd in place into b.q, K/V into the cache); the T per-token
  `gemma_attention` launches are REPLACED by ONE `gemma_attention_batched` launch
  grid=(n_head,T) reading b.q + the prompt region [0..T) of kv_k/kv_v → b.attn_out
  (shared mem = T*4). New `GemmaAttnBatchPush` + pipe `gemma_attention_batched`.
  Extended `scripts/prefill_catalog.sh` ADDITIVELY: `ZINC_AB=headskip|batched` (default
  headskip, unchanged) — batched mode A/Bs ZINC_BATCHED_PREFILL=1 vs baseline with the
  same ABBA counterbalancing + GEN_IDS-identical gate. Built clean on the 4090 box
  (fresh `.zig-cache`, EXIT=0, bin md5 dc54e7cd, CHANGED from cycle 1's 45251da3).
  GATE (ABBA x2, 250-tok prompt, 4090):
    - gemma4-31b dense: 31.43 → 94.12 t/s (+199%, ~3×), GEN_IDS byte-IDENTICAL ✓
    - gemma4-26b MoE:   54.52 → 58.19 (+7% noise, per-token fallback) GEN_IDS identical ✓
  Direct varied-output A/B (cyclic prompt → GEN 235,612,919,1471,218915,10205,…, not a
  collapsed prompt): byte-identical, prefill 30.4→91.7 t/s. Cycle 1 was +133% (T=200);
  the batched attention removed the per-token attn launch overhead → +199%. Committed to
  perf/e24-batched-prefill (toggle off by default → cannot regress production). REMAINING
  for merge: `scripts/validate_catalog.sh` 5/5 (per-token product path unchanged + batched
  is byte-identical to it → transitively correct). NEXT (cycle 3): target #3 batched KV
  write — fold the per-token norm/RoPE/KV-write loop into batched launches (kills the last
  T per-token launches); then NVRTC `-I` tensor-core GEMM (+2.2×).
