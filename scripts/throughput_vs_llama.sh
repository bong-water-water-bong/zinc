#!/usr/bin/env bash
# Head-to-head SERVING throughput: zinc CUDA serve vs llama-server, same model/box/B.
# Run ON the box from the repo root. The loop's own throughput_cuda_serve.sh only
# measures zinc's INTERNAL B-scaling (vs its own slow B=1) and never compares to
# llama — this script closes that gap with the fair, identical external metric:
#   ext aggregate decode tok/s = (B*NG)/wall  (B concurrent /v1/completions clients,
#   each FORCED to exactly NG tokens: zinc budget-only EOS, llama ignore_eos).
# Engines run SEQUENTIALLY (one ~18GB model resident at a time). 5090-pinned;
# net-fence → reach servers via 127.99.0.1.  Vars: NG, ROUNDS, NP, MODEL.
#
# 2026-06-15 result (gemma-31b, NG=64): zinc B=1/2/4/8 = 1.7/2.8/4.1/4.5 tok/s vs
# llama 52/100/142/189 → zinc is ~30-42x BEHIND, gap widens with B (llama scales
# 3.6x, zinc 2.6x), and zinc DEGRADES over a sustained run (r2 collapse to ~1).
# Root cause: zinc's batched-decode kernel is compute-INEFFICIENT (100% util at B=8
# but only 4.5 tok/s) — no CUDA-graph capture + dequant-in-loop GEMM. Architecture
# scales; the per-step kernel is the bottleneck (same gap Effort 26 found in prefill).
set -u
GPU=${ZINC_GPU:-GPU-5126d018-ec86-be8b-1bf5-b5ac323d3350}
ZIG=${ZIG:-$HOME/zig-0.15.2/zig}
MODEL=${MODEL:-$HOME/workspace/models/gemma-4-31B-it-Q4_K_M.gguf}
LLAMA=${LLAMA:-$HOME/workspace/llama.cpp/build/bin/llama-server}
NG=${NG:-64}; ROUNDS=${ROUNDS:-2}; NP=${NP:-8}; BLIST=${BLIST:-"1 2 4 8"}
PROMPT="Write a detailed adventure story about a dragon who learns to write computer code."
HOST=127.99.0.1
export ZIG_GLOBAL_CACHE_DIR=${ZIG_GLOBAL_CACHE_DIR:-/tmp/tpvlgc}
cd "$(dirname "$0")/.." || exit 1
OUT=/tmp/tpvl.txt; : > "$OUT"

echo "=== build zinc serve binary ==="
$ZIG build -Dbackend=cuda -Dshaders=false >/tmp/tpvl_build.log 2>&1
ZINC=$(ls -t zig-out/bin/zinc 2>/dev/null | head -1)
[ -x "$ZINC" ] || { echo "ZINC BUILD FAIL"; grep -E 'error:' /tmp/tpvl_build.log | head; echo "=== TPVL DONE ==="; exit 1; }

gen_zinc(){  curl -N -s --max-time 900 -X POST "http://$HOST:$1/v1/completions" -H 'Content-Type: application/json' \
               -d "{\"prompt\":\"$PROMPT\",\"max_tokens\":$NG}" >/dev/null 2>&1; }
gen_llama(){ curl -N -s --max-time 900 -X POST "http://$HOST:$1/v1/completions" -H 'Content-Type: application/json' \
               -d "{\"prompt\":\"$PROMPT\",\"n_predict\":$NG,\"ignore_eos\":true}" >/dev/null 2>&1; }

sweep(){ local tag=$1 port=$2 fn=$3 B t0 t1 wall agg r i pids
  $fn "$port"
  for r in $(seq 1 "$ROUNDS"); do for B in $BLIST; do
    t0=$(date +%s.%N); pids=(); for ((i=0;i<B;i++)); do $fn "$port" & pids+=($!); done; wait "${pids[@]}"; t1=$(date +%s.%N)
    wall=$(echo "$t1-$t0"|bc -l); agg=$(echo "scale=2;$B*$NG/$wall"|bc -l)
    printf "%-5s B=%-2s ext_agg_tok/s=%-8s wall=%ss (r%s)\n" "$tag" "$B" "$agg" "$(printf '%.2f' "$wall")" "$r" | tee -a "$OUT"
  done; done; }

wait_ready(){ for _ in $(seq 1 360); do curl -s --max-time 3 "http://$HOST:$2/health" 2>/dev/null | grep -q 'ok' && return 0
    kill -0 "$1" 2>/dev/null || return 1; sleep 1; done; return 1; }

echo "=== ZINC serve (parallel $NP, budget-only EOS, NG $NG) ==="
CUDA_VISIBLE_DEVICES=$GPU ZINC_GPU=$GPU ZINC_SCHED_EOS=4294967295 "$ZINC" -m "$MODEL" -p 8190 -c 2048 --parallel $NP -n "$NG" >/tmp/tpvl_zinc.log 2>&1 &
ZPID=$!; wait_ready "$ZPID" 8190 && sweep "ZINC" 8190 gen_zinc || { echo "zinc not ready"; tail /tmp/tpvl_zinc.log; }
kill "$ZPID" 2>/dev/null; wait "$ZPID" 2>/dev/null; sleep 4

echo "=== LLAMA-server (-np $NP -ngl 99, ignore_eos) ==="
CUDA_VISIBLE_DEVICES=$GPU "$LLAMA" -m "$MODEL" --host 0.0.0.0 --port 8191 -ngl 99 -c 8192 -np $NP >/tmp/tpvl_llama.log 2>&1 &
LPID=$!; wait_ready "$LPID" 8191 && sweep "LLAMA" 8191 gen_llama || { echo "llama not ready"; tail /tmp/tpvl_llama.log; }
kill "$LPID" 2>/dev/null; wait "$LPID" 2>/dev/null

echo "=== RESULTS ==="; cat "$OUT"; echo "=== TPVL DONE ==="
