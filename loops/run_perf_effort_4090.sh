#!/usr/bin/env bash
# Effort 27 (CUDA DECODE: close the llama.cpp gap) autonomous loop. Twin of
# run_perf_effort_e26.sh but pinned to the RTX 4090 and the ~/zinc-e27 box dir, and
# it runs in its OWN git worktree — so it advances ALONGSIDE the Effort-26 BEAT_LLAMA
# prefill loop (5090, ~/Workspace/zinc-e26, ~/zinc-e26) without contending on GPU, box
# dir, or working tree. Each cycle spawns a fresh `claude -p` that reads
# MULTI_HOUR_EFFORT_27_CUDA_4090_DECODE.md, lands ONE validated DECODE increment,
# commits to a perf/e27-* branch, appends to the cycle log, then stops.
#
#   MAX_CYCLES=200 nohup bash loops/run_perf_effort_4090.sh loops/efforts/MULTI_HOUR_EFFORT_27_CUDA_4090_DECODE.md >/tmp/e27.log 2>&1 &
#   stop with:  pkill -f run_perf_effort_4090
set -u
cd "$(dirname "$0")/.." || exit 1
ROOT="$(pwd)"
EFFORT="${1:?usage: run_perf_effort_4090.sh <effort-file.md>}"
[ -f "$EFFORT" ] || { echo "no effort file: $EFFORT"; exit 1; }
MAX=${MAX_CYCLES:-200}
LOG="/tmp/perf_effort_$(basename "$EFFORT" .md).log"

PROMPT="You are autonomously advancing a scoped CUDA DECODE perf EFFORT on the ZINC inference engine (Zig), in ${ROOT} (a DEDICATED git worktree — work HERE, never in /Users/stepan/Workspace/zinc [the main checkout] or /Users/stepan/Workspace/zinc-e26 [the parallel Effort-26 5090 prefill loop]), then STOPPING after ONE validated increment.

STEP 0 — READ: your memory (MEMORY.md + project_zinc_cuda_backend.md + project_cuda_perf_blog.md + project_effort23_gemma_attn_fusion.md + project_effort25_cuda_graphs.md) AND the effort file ${EFFORT}. It holds the targets (close the llama.cpp DECODE gap: MoE 31-42%, dense 75-91% of llama), the LEVERS (PRIMARY: MoE-decode per-token expert-path fusion + expert-matvec efficiency — PROFILE launch-bound vs matvec-bound first; SECONDARY: Effort-23 dense-decode fusion playbook — aggregate >=2 tiny launches per fused kernel to clear the +-1% boost floor), the HARD RULES, the validation contract, and the cycle log. Honor all of it.

HARD CONSTRAINTS for THIS effort (they OVERRIDE the generic playbook): pin the RTX 4090 — export CUDA_VISIBLE_DEVICES=GPU-e59a6fce-1961-bafe-927c-06c0149f2370 and run validate_catalog + every measurement with ZINC_GPU=GPU-e59a6fce-1961-bafe-927c-06c0149f2370. The Effort-26 BEAT_LLAMA loop owns the 5090 (GPU-5126d018-...), ~/Workspace/zinc-e26, and the box dir ~/zinc-e26 — do NOT touch them, the main checkout /Users/stepan/Workspace/zinc, or push to main. Use the isolated box dir ~/zinc-e27 (rsync source there; never ~/workspace/zinc). Isolated-cache builds (ZIG_LOCAL_CACHE_DIR+ZIG_GLOBAL_CACHE_DIR; verify the binary md5 actually changed or you are measuring stale code). DO NOT async GEMMA decode (boost-saturated, proven regression); qwen MoE async-pipelining is fine. Box gotchas: PREFILL/DECODE tok/s prints to STDERR (2>&1); always 'nohup CMD >FILE 2>&1 &' on the box and poll FILE; util-gate A/B via --query-gpu=utilization.gpu; gemma reloads ~18GB/call.

THIS CYCLE: pick the next un-done target from the effort file (or continue an in-progress perf/e27-* branch you find), implement ONE focused DECODE change, build with isolated caches, run scripts/validate_catalog.sh (ZINC_GPU=the 4090 UUID) — it MUST stay 5/5 token-correct; if correctness breaks, REVERT and document why. Measure an interleaved A/B and compare zinc decode tok/s vs the pre-cycle binary AND vs llama.cpp on the SAME 4090 + same gguf (use ~/workspace/llama.cpp/build/bin/llama-bench or the perf suite for the baseline). If it is a VALIDATED WIN, commit ONLY this change to perf/e27-<short-target> and push it (NOT main); append a dated entry to the effort file's cycle log AND memory. If NEGATIVE, revert the code and log the finding (negatives are valuable). Clean up box scratch dirs. STOP — do not loop yourself.

NEVER: break catalog correctness, commit unvalidated code or a swept working tree (commit ONLY your scoped change), push to main, async gemma decode, disturb the 5090/Effort-26 work or /Users/stepan/Workspace/zinc or /Users/stepan/Workspace/zinc-e26, or trust a single boost-noisy measurement."

echo "=== e27 4090-decode loop: $EFFORT  (4090, root=$ROOT, max $MAX cycles)  $(date) ===" | tee -a "$LOG"
i=0
while [ "$i" -lt "$MAX" ]; do
  i=$((i + 1))
  echo "===== e27 cycle $i / $MAX  —  $(date) =====" | tee -a "$LOG"
  claude -p --permission-mode bypassPermissions --effort high "$PROMPT" 2>&1 | tee -a "$LOG"
  echo "===== e27 cycle $i done — $(date); sleeping 60s =====" | tee -a "$LOG"
  sleep 60
done
echo "=== e27 loop finished after $i cycles $(date) ===" | tee -a "$LOG"
