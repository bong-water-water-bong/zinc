# Prefill Gap Closure Plan — ZINC CUDA vs llama.cpp

**Goal:** close ZINC's prompt-prefill throughput gap vs llama.cpp on consumer NVIDIA
(RTX 4090 `sm_89`, RTX 5090 `sm_120`). Decode is already ~85-90% of llama
(short-context); **prefill is the structural gap.**

## Where we are (matched p90, RTX 4090, ZINC `main` incl. e29 cycle-3+5, vs `llama-bench` pp1041, 2026-06-22)

| model | ZINC prefill p90 (t/s) | llama pp1041 | gap | shape |
|---|--:|--:|--:|---|
| gemma4-31b      | 415 | 2542 | **6.1×**  | dense |
| qwen36-27b      | 281 | 2488 | **8.9×**  | dense (hybrid-SSM) |
| qwen35-9b       | 482 | 7814 | **16×**   | dense (hybrid-SSM) |
| gemma4-26b-a4b  | 296 | 6371 | **21.5×** | MoE |
| qwen36-35b-a3b  | 190 | 5471 | **29×**   | MoE |

Pattern: **MoE rows (21-29×) are worse than dense (6-16×)**, and **qwen dense (16×) is worse
than gemma dense (6×)** — the hybrid-SSM path adds cost on top of the GEMM gap.

## Root cause (established Effort-26 + Effort-29 + 2026-06-22 session)

1. **Dense GEMM:** ZINC prefill routes Q4_K/Q6_K dense GEMMs through **cuBLAS fp16-TC + a
   dequant→fp16 scratch round-trip** (`dequant_q4k_to_f16`/`dequant_q6k_to_f16` then
   `cublasGemmEx`). The round-trip is **only ~5% of warm, compute-bound prefill** (Effort-29
   cycle-2 fp16-weight-cache was rejected for this reason) — so "kill the round-trip" is NOT
   the lever. cuBLAS compute is near-optimal; the residual gap to llama is that **llama's MMQ
   fuses dequant *into* the Tensor-core GEMM** (one pass, no scratch) and is hand-tuned.
2. **MoE routed experts:** `moeFfnBlockBatched` (cycle-3) batches the per-token MoE FFN, and
   cycle-5 put the **shared** expert on cuBLAS — but the **routed** experts still run the
   `dmmv_*_experts_batched` **matvec** kernels, NOT Tensor cores. Effort-29 cycles 4 & 6
   proved **matvec *restructuring* is exhausted** (grouped L2-reorder and multi-token-per-block
   were both negative/dead-even on the 5090 — its large L2 + parallelism already hide the
   per-expert weight re-reads). The remaining ~15-29× on MoE is the **expert GEMM not being
   on Tensor cores**.
3. **qwen hybrid-SSM:** qwen35/36 prefill also runs the gated-delta-net **scan** (`ssm_delta_net`,
   one launch, `n_tok=T`, sequential recurrence) + conv1d + per-head norms. qwen dense (16×) >
   gemma dense (6×) suggests the SSM/attention-prep path is a second-order bottleneck — needs a
   profile to confirm its share before optimizing.

The existing hand fused-dequant TC kernel (`gemm_q4k_tc` / `_f16a` / `_lowsmem`) **already exists**
but reaches only **~22% of cuBLAS** throughput (no cp.async double-buffering; small 64×64 tile;
2 accumulator frags/warp), so cuBLAS+round-trip still wins +76% @T=512 even paying the round-trip.
**Dead ends — do NOT re-litigate** (Effort-26/29): int8 MMQ (Q4_K-asymmetric per-subblock
store-rescale epilogue tax), FP8 e4m3 (weight-traffic-bound, 2× TC buys nothing), fp16 weight
cache (~5% ceiling + 2× VRAM), m128/normf16/grouped micro-opts (in-noise), prefill CUDA graphs
(compute-bound), expert-matvec restructuring (L2 hides re-reads).

## The levers (priority order = biggest gap × tractability)

### T2 — Tensor-core grouped MoE expert GEMM  *(highest EV: MoE rows are the worst, 21-29×, and matvec is exhausted)*
Replace the routed-expert matvecs with a **grouped Tensor-core GEMM**: gather the tokens routed
to each expert into contiguous tiles, run a per-expert (or grouped/batched) fp16-TC GEMM,
scatter back. This is llama's MMQ-MoE approach. **Caveat to design around:** per-expert token
count is small (`T·top_k / n_experts` ≈ 8-64 at pp256-2048), so per-expert GEMMs are *skinny* —
naive `cublasGemmGroupedBatched` underutilizes TC at N≈8. Mitigations: (a) the win is still the
**1× weight read per expert** (vs `n_tok_e×` in the matvec) — a real traffic cut even at skinny N;
(b) consider a fused-dequant MMQ expert kernel rather than cuBLAS so skinny-N stays efficient.
Microbench-gate vs `dmmv_q4k_experts_batched` at realistic per-expert N before wiring.

### T1 — Fused dequant + Tensor-core dense GEMM (MMQ-class)  *(the dense 6-16× rows; Effort-29's T1a, the hard one)*
Bring `gemm_q4k_tc`/`_f16a` up to cuBLAS-class throughput so it beats cuBLAS+round-trip:
**cp.async double-buffering** (overlap the next tile's global load + Q4_K dequant with the current
`mma`), **larger block + register tiles** (more accumulator frags/warp), shared-mem bank-conflict-free
staging, and `mma.sync`/`wgmma` instead of `wmma` where it helps. **Gate:** an isolated microbench
must beat cuBLAS+round-trip at T≥512 BEFORE wiring it into `gemmDispatchPrefill`. Keeps Q4_K's free
per-subblock `d·sc` scaling (the thing int8 MMQ couldn't). Hard, multi-cycle.

### T3 — qwen hybrid-SSM prefill profile + scan
Profile qwen35-9b prefill (nsys per-kernel) to quantify the SSM-scan / conv1d / per-head-norm
share vs the GEMMs (explains why qwen dense 16× > gemma dense 6×). If the sequential delta-net
scan dominates, pursue a **chunked/blocked parallel scan** (recurrence-preserving) or overlap it
with the projections.

### T4 — launch/recast fusion (incremental)
Each prefill GEMM today is `dequant` + `f32_to_f16` + `cublas` = 3 launches; have norm/GeGLU
producers emit fp16 directly (`normf16`, partly explored) and share one activation recast across
same-input GEMMs. Small, but free where bit-identical.

## Methodology (per cycle)
- **Microbench-gate** each kernel vs the path it replaces BEFORE wiring (the box's ±~10% boost
  floor hides single-launch wins; isolated kernels measure cleanly).
- **Correctness gate:** `scripts/validate_catalog.sh` must stay **5/5 token-correct** (bit-identical
  for structural changes; token-tolerance for fp16/reduction-order). Build with **isolated zig
  caches** (`ZIG_LOCAL_CACHE_DIR`+`ZIG_GLOBAL_CACHE_DIR`) — the global cache serves stale binaries.
- **Perf gate:** interleaved A/B on a **clean** GPU; report p90 (use `scripts/p90_scoreboard.sh` for
  the matched prefill-p90 + decode table vs `llama-bench`, `scripts/prefill_profile.sh` for nsys
  per-kernel breakdown).
- **Test BOTH GPUs.** The 4090 is *more bandwidth-bound* than the 5090 (TC/BW ratio 0.65 vs 0.47)
  and has less L2 (72 vs 96 MB) — **a lever that's negative on the 5090's big L2 may win on the
  4090** (e.g., Effort-29 cycle-4's grouped expert reorder, negative on the 5090, is worth a
  4090-specific A/B). Don't let a 5090 negative close a 4090 lever.
- Validated win → commit to a `perf/<name>` branch, harvest to `main` (build + validate 5/5 + FF).

## Target
Beat llama.cpp prefill on ≥1 row first (gemma-31b at 6.1× is the closest dense; the MoE rows are
the biggest absolute prize via T2). Stretch: parity across the catalog.
