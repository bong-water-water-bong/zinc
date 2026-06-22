#!/usr/bin/env bash
# Overnight autonomous CUDA PREFILL perf loop. Each cycle spawns a fresh `claude -p`
# (clean context) that lands ONE validated prefill increment to a perf/* branch, then
# stops; continuity is carried by git branches + memory. Self-recovering: a cycle that
# crashes/exits nonzero does NOT stop the loop (the next cycle starts after a short
# sleep). Safe by construction: each cycle validates 5/5 token-correctness before
# committing, commits ONLY to perf/* (never main), and pins the 4090.
#
# Runs on the MAC (where the `claude` CLI lives); each cycle SSHes to the NVIDIA box
# to build + measure. Run from the repo root of a main checkout:
#   nohup bash loops/overnight_cuda_prefill.sh > /tmp/overnight_cuda_prefill.log 2>&1 &
# Watch:  tail -f /tmp/overnight_cuda_prefill.log
# Stop:   pkill -f overnight_cuda_prefill
set -u
cd "$(dirname "$0")/.." || exit 1
PROMPT="$(cat loops/overnight_cuda_prefill_prompt.txt)" || { echo "no prompt file"; exit 1; }
MAX=${MAX_CYCLES:-40}            # backstop (~8-12h of cycles)
SLEEP=${CYCLE_SLEEP:-45}
echo "=== overnight CUDA PREFILL loop START $(date -u +%FT%TZ) (max $MAX cycles) ==="
i=0
while [ "$i" -lt "$MAX" ]; do
  i=$((i + 1))
  echo "===== prefill cycle $i / $MAX — $(date -u +%FT%TZ) ====="
  # bypassPermissions so the cycle can build/ssh/git unattended; the prompt enforces
  # validate-5/5-before-commit and perf/*-only (never main).
  claude -p --permission-mode bypassPermissions --effort high "$PROMPT" \
    || echo "(cycle $i exited nonzero — self-recovering, continuing)"
  echo "===== cycle $i done — $(date -u +%FT%TZ); sleeping ${SLEEP}s ====="
  sleep "$SLEEP"
done
echo "=== overnight CUDA PREFILL loop FINISHED after $i cycles $(date -u +%FT%TZ) ==="
