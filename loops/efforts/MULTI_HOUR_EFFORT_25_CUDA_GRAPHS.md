# Effort 25 — CUDA Graphs (kill the per-launch overhead; helps decode AND prefill)

> **Status:** 🔬 OPEN. The highest-value CUDA lever found 2026-06-13. Supersedes Effort 24 (batched prefill — proven DEAD END below).

Date: 2026-06-13. Forward paths: `src/compute/forward_cuda.zig` (qwen35/36 hybrid-SSM) + `forward_cuda_gemma.zig` (gemma4). Command/stream layer: `src/cuda/command.zig`, `src/cuda/cuda_shim.c`.

## Why this effort (the finding that redirects everything)

Decode AND prefill on this hardware are **LAUNCH-BOUND, not weight/compute-bound.** Profiled (nvidia-smi DURING a batched gemma-31b prefill, T=413): **full clock 2422 MHz but only 7–12% GPU util / ~76 W of 575 W — the GPU sits ~90% idle.** qwen decode profiled the same (10–12% util). The bottleneck is the **~480-kernel-per-step launch chain** (≈60 layers × ~8 kernels: norms, Q/K/V matvecs, attention, O, FFN), whose per-kernel launch + dependency latency dwarfs each kernel's compute.

**This is why Effort 24 (batched prefill) is a DEAD END:** the batched GEMM (5.9× isolated) and `attention_causal_batched` (6–15× isolated) are fast but were *never the bottleneck* → end-to-end NEUTRAL (gemma-31b dense ~0%, gemma-26b MoE +6–8%). Isolated kernel benches don't predict end-to-end wins when the GPU is idle between kernels. **Do not pursue batched prefill.**

**The real lever = cut the launch overhead.** A 10%-util launch-bound chain is exactly what **CUDA Graphs** are for: capture the per-step kernel sequence once, then `cuGraphLaunch` to replay it as ONE launch — eliminating per-kernel CPU-side launch cost and most of the inter-kernel GPU bubbles. The async CUstream ring already serializes the commands in order, so the same stream is graph-capturable.

## Target (any catalog model; measure decode tok/s + prefill tok/s)

Decode is the headline metric (zinc loses to llama everywhere there). If graphs lift decode util from ~10% toward saturation, that's a large multi-model win. Prefill benefits identically (same chain).

## Plan (incremental, validate-before-commit)

1. **Cycle 1 — ISOLATED PROOF (`~/cuda_proto`, no repo change):** a C harness that (a) launches N≈50 chained tiny kernels on a stream the normal way, timed; (b) stream-captures the same N into a graph (`cuStreamBeginCapture` → N `cuLaunchKernel` → `cuStreamEndCapture` → `cuGraphInstantiate`), then `cuGraphLaunch` M times, timed. Report per-iteration: N-launches vs 1-graph-replay, and the per-launch overhead saved on this GPU. GATE: if graph replay isn't materially faster than N launches at small kernel sizes, CUDA graphs won't help here either → LOG it as a dead end and pivot to deeper fusion. If it IS faster (expected — launch overhead is the whole problem), proceed.
2. **Cycle 2 — wire `cuGraph` capture into `src/cuda/command.zig`:** add a graph-capture mode to the async ring — on the first decode step, stream-capture the per-step command chain into a `CUgraph` + instantiate a `CUgraphExec`; on subsequent steps with identical shape, `cuGraphLaunch` the exec instead of re-recording. Push-constants/buffer pointers that change per step (position, KV offset) must be handled (kernel-node param update via `cuGraphExecKernelNodeSetParams`, or design the kernels to read pos/offset from a small device buffer updated before each replay so the graph topology is invariant). Behind a flag (e.g. `ZINC_CUDA_GRAPH`). `-I/usr/local/cuda/include` is already enabled (a0d463af) if the graph API needs headers; the driver `cuGraph*` API is in `cuda.h`.
3. **Cycle 3+ — validate + measure:** `scripts/validate_catalog.sh` MUST stay 5/5 token-correct (graph replay must be bit-identical to the un-captured chain). Interleaved A/B decode tok/s with the flag on vs off. Extend to prefill (capture the per-layer chain). Commit each validated increment to `perf/e25-cuda-graphs-<step>`.

## Validation contract

- `scripts/validate_catalog.sh` 5/5 token-correct with the graph flag ON (replay must equal the recorded chain — same kernels, same order, same params per step). If a single token diverges, the param-update handling is wrong → fix or revert.
- Interleaved back-to-back A/B (graph on vs off) to beat boost noise; report decode tok/s, GPU util (nvidia-smi during), and ms/token.
- Isolated-cache builds (`ZIG_LOCAL_CACHE_DIR`+`ZIG_GLOBAL_CACHE_DIR`, verify binary hash changed). Pinned GPU per the runner. Isolated box dir `~/zinc-e25`, never `~/workspace/zinc`.

## HARD RULES (from the playbook)

- Profile util/power to confirm the regime before/after — the whole point is moving util UP. A win that doesn't raise util is suspect.
- Graph replay must be **bit-identical** (it's the SAME kernels) → catalog 5/5 is non-negotiable; negatives are valuable, log + revert.
- Branches not main; validate before commit; never commit host/IP/port; don't disturb the parallel work or `~/workspace/zinc`.

## Cycle log

(append dated entries per cycle: cycle | change | built+hash-changed? | catalog 5/5? | A/B result (util before→after, tok/s) | branch/sha or revert+why | next)

**2026-06-13 — Cycle 1 (ISOLATED PROOF): GATE PASSED, decisively.** Change: standalone CUDA harness `~/cuda_proto/graph_proof.cu` (box research dir, NO repo source change) — times a chain of N tiny kernels + a per-step `cudaStreamSynchronize` (mimics decode: N-kernel layer chain then argmax D2H) two ways: (A) relaunch the N kernels every step; (B) `cudaStreamBeginCapture`→N launches→`cudaStreamEndCapture`→`cudaGraphInstantiate` once, then `cudaGraphLaunch`+sync per step. Built: standalone `nvcc -O3 -arch=sm_89` (no zig build, no catalog impact). Catalog: N/A (no repo code touched). **A/B (4090, pinned `GPU-e59a6fce-…`, interleaved A/B/A/B, min-of-2, 3 runs across n_embd=4096 AND 2816, all agree):** per-step ms relaunch→graph — N=60: ~1.2→~0.34 (**~3.6–6×**); N=480 (≈60-layer real chain): **~4.5–5.0→~0.51–0.56 = ~8–9.6×**. Relaunch grows ~LINEARLY with N (the launch-bound signature); graph replay stays ~FLAT (~0.5 ms ≈ a single WSL2 sync round-trip) regardless of N → the entire per-kernel launch + inter-kernel-bubble cost collapses to one replay. This is the GPU-internal launch latency the CPU-side async ring explicitly *can't* touch (per [[zinc-cuda-backend]] gemma note), so graphs are additive to the ring. GATE: "graph replay materially faster than N launches at small kernel sizes" → YES, ~9× at the real chain length. **Branch: `perf/e25-cuda-graphs-proof` (effort-file log entry only; harness stays in `~/cuda_proto` per the research-dir convention).** NEXT cycle 2: wire `cuGraph` capture into `src/cuda/command.zig`'s async ring — capture the per-step command chain on the single CUstream on step 1, `cuGraphLaunch` on identical-shape subsequent steps; handle per-step-varying params (position, KV offset) via device-buffer reads so graph topology is invariant (else `cuGraphExecKernelNodeSetParams`). Behind `ZINC_CUDA_GRAPH`. Gate = catalog 5/5 bit-identical + interleaved decode A/B with util before→after.
