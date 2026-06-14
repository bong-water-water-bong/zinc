# Effort 27 — CUDA DECODE: close the llama.cpp gap on the RTX 4090 (MoE 31–42%, dense 75–91%)

> **Status:** 🔬 OPEN (spawned 2026-06-14). Runs on the **RTX 4090**
> (`GPU-e59a6fce-1961-bafe-927c-06c0149f2370`). The parallel **Effort-26 BEAT_LLAMA**
> loop owns the **5090** (`GPU-5126d018-…`) + `~/Workspace/zinc-e26` + box dir
> `~/zinc-e26` — do NOT touch them, the main checkout `~/Workspace/zinc`, or push to
> main. Work in THIS worktree + box dir `~/zinc-e27`.
> Forward paths: `src/compute/forward_cuda.zig` (qwen35/36 hybrid-SSM dense + MoE),
> `src/compute/forward_cuda_gemma.zig` (gemma4 dense + MoE), `src/shaders/cuda/kernels.cu`.

## The gap (5090 catalog 2026-06-13, zinc decode as % of llama — same shape on the 4090)
- **gemma-4-26B-A4B MoE — 31%** of llama (47.5 vs ~153) ← worst, the headline lever
- **qwen36-35B-A3B MoE — 42%** (52.9 vs ~126)
- qwen35-9B dense — 75% (120.8 vs ~161)
- gemma-4-31B dense — 82% (46.9 vs ~57)
- qwen36-27B dense — 91% — fine, leave it
4090 decode is **launch/latency-bound** (Effort-25 proved graph replay buys ~8–12% on
small dense; size-gated). llama's years-tuned gathered-expert matvecs lead on MoE.

## PRIMARY — MoE decode (the 31–42% gap)
e26/5090 is doing PREFILL (all its cycles so far), so **MoE-decode is unclaimed in
practice — own it here.** Per-token MoE decode = router norm→gate matvec→top-k, then 8
routed-expert matvecs (Q4_K gate/up + Q5_1 down) + Q8_0 shared expert + weighted combine,
per layer (gemma-26b: 30 MoE layers; qwen-35b-a3b: most). Levers:
- **PROFILE FIRST** (util/clock + per-op): launch-bound (many tiny launches/token → GPU
  idles) vs expert-matvec-bound. Let the profile pick the cycle's change.
- **Fuse the per-token expert path** (router+experts+shared+combine = many small
  launches/layer); the Effort-24 batched-prefill expert kernels are twins — the
  single-token decode path may still be launch-heavy.
- **Expert-matvec efficiency**: `dmmv_q4k`/`dmmv_q5_1` for the 8 active experts vs llama's
  gathered-expert matvec (coalescing, dp4a, block size).
- **qwen36 MoE async-pipelining** was a WIN (35b-a3b ~1.5×, branch `perf/moe-async-decode`)
  — check if it's on current main; if not, re-validate + land. **Do NOT async GEMMA decode
  (boost-saturated, proven regression); qwen MoE async is fine.**

## SECONDARY — dense decode fusion (Effort-23 playbook; proven repeatable on this box)
qwen35-9b 75%, gemma-31b 82%. Effort-23 landed 4 stacked dense-decode fusion wins
(V/KV-write, output-scale fold, Q4_K matvec-pair, qkv-norm) on branches **pushed NOT main**
(`perf/e23-*`). Lever: **aggregate ≥2 tiny per-token launches per fused kernel** to clear
this box's ±1% boost floor (single-launch fusions need locked clocks — no
`nvidia-smi -lgc` w/o sudo here). Re-validate + land the unmerged e23 wins on current main
where they still apply, or find new ones. Also: LM-head matvec (vocab×n_embd, biggest
read/token).

## HARD RULES (override the generic playbook)
- **Pin the 4090:** `export CUDA_VISIBLE_DEVICES=GPU-e59a6fce-1961-bafe-927c-06c0149f2370`;
  run validate_catalog + all measurements with `ZINC_GPU=GPU-e59a6fce-1961-bafe-927c-06c0149f2370`.
- **NEVER** touch the 5090 (`GPU-5126d018-…`), `~/Workspace/zinc-e26`, box `~/zinc-e26`,
  the main checkout `~/Workspace/zinc`, or push to main. Box build dir = `~/zinc-e27`
  (rsync source there; never `~/workspace/zinc`).
- Isolated-cache builds (`ZIG_LOCAL_CACHE_DIR`+`ZIG_GLOBAL_CACHE_DIR`; verify the binary
  md5 changed or you measure stale code).
- **Gate:** `scripts/validate_catalog.sh` (ZINC_GPU=4090) MUST stay 5/5 token-correct
  (fused kernels bit-equivalent); if correctness breaks → REVERT + document.
- **Measure interleaved back-to-back A/B** (4090 decode is boost-noisy); compare zinc decode
  tok/s vs the pre-cycle binary AND vs llama on the SAME 4090 + same gguf
  (`~/workspace/llama.cpp/build/bin/llama-bench` or the perf suite). Never trust one boosted run.
- **VALIDATED WIN** → commit ONLY that change to `perf/e27-<short-target>`, push (NOT main),
  append a dated cycle-log entry here + to memory. **NEGATIVE** → revert code, log the finding.
- Box gotchas: tok/s prints to STDERR (`2>&1`); `nohup CMD >FILE 2>&1 &` + poll FILE;
  util-gate via `--query-gpu=utilization.gpu`; gemma reloads ~18GB/call.

## CYCLE LOG
- (none yet — spawned 2026-06-14)
