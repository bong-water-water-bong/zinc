#!/usr/bin/env bash
# Bench llama.cpp Qwen3-8B Q4_K_M on the local Metal backend so we have
# a comparator number for Effort 14 (M1 Max Qwen3-8B decode parity).
#
# Run AFTER the implement_metal.ts loop finishes — both processes contend
# for the GPU and concurrent runs poison each other's measurements.
#
# Background: during Phase 0 of Effort 14, llama-cli hung twice on
# warmup at 96% CPU for hours. The most likely cause is conversation
# mode waiting on stdin. This script uses -no-cnv (no-conversation) and
# explicit greedy decoding flags to avoid that.

set -euo pipefail

MODEL=/Users/stepan/Library/Caches/zinc/models/models/qwen3-8b-q4k-m/model.gguf
BIN=$HOME/Workspace/llama.cpp/build-metal/bin/llama-cli
PROMPT="The capital of France is"
N_TOKENS=128

if [[ ! -x "$BIN" ]]; then
  echo "llama-cli not found at $BIN" >&2
  exit 1
fi
if [[ ! -f "$MODEL" ]]; then
  echo "Model not found at $MODEL" >&2
  exit 1
fi

# Stop strays so the bench has a clean GPU.
pkill -f 'zig-out/bin/zinc' 2>/dev/null || true
pkill -f 'llama-cli'         2>/dev/null || true
sleep 1

echo "=== llama.cpp Qwen3-8B baseline (Metal, M1 Max) ==="
echo "Model:  $MODEL"
echo "Prompt: $PROMPT"
echo "Tokens: $N_TOKENS"
echo "Flags:  -ngl 99 -fa on --temp 0 -no-cnv --no-warmup"
echo

# Warmup (discarded) — then three timed runs. Use --simple-io to disable
# any interactive output massaging that might block on a tty in non-tty
# environments. The grep at the end extracts the perf summary.
echo "=== WARMUP (discarded) ==="
"$BIN" -m "$MODEL" -p "$PROMPT" -n 16 -ngl 99 -fa on --temp 0 -no-cnv --no-warmup --simple-io 2>&1 \
  | tail -10 || true

for run in 1 2 3; do
  echo
  echo "=== RUN $run ==="
  "$BIN" -m "$MODEL" -p "$PROMPT" -n "$N_TOKENS" -ngl 99 -fa on --temp 0 -no-cnv --no-warmup --simple-io 2>&1 \
    | grep -E 'eval time|tokens per second|prompt eval|llama_perf' \
    || echo "(no perf line found — re-run by hand and inspect output)"
done

echo
echo "Done. Compare median 'eval' tok/s against ZINC's decode tok/s for the same model."
