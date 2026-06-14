# Effort 28 — Multi-tenant serving + continuous (request-level) batching on CUDA

> **Status:** 🔬 OPEN (spawned 2026-06-14). Goal: ZINC+CUDA serves **many concurrent requests batched into one GPU forward** (continuous/dynamic batching, the vLLM/TGI model), not one-sequence-at-a-time. This is a **multi-cycle BUILD on ONE persistent branch `feat/e28-batching`** (NOT independent perf tweaks, NOT main). Each cycle advances the next increment, validates it, commits to `feat/e28-batching`, and updates `## CURRENT STATE` below.

Forward paths: `src/compute/forward_cuda_gemma.zig` (gemma4 dense+MoE — **the increment-1 target**), `src/compute/forward_cuda.zig` (qwen35/36 hybrid-SSM — increment 4). Serving: `src/server/routes.zig`, `src/server/http.zig`, `src/main.zig`. Scaffolding (exists, UNWIRED in CUDA): `src/scheduler/{scheduler,request,kv_cache}.zig`.

## Where we are today (the honest baseline — confirmed by code audit 2026-06-14)

- **Server is thread-per-request but GPU-SERIALIZED.** `routes.zig` spawns a thread per connection, but every generation funnels through one global `ServerState.generation_mutex` (`routes.zig:351,554,605`; `GenerationGuard` at `:346`). Concurrent clients → correct, isolated outputs, but the GPU runs **one sequence at a time**; the rest queue (head-of-line blocking). So multi-tenant *correctness* exists; multi-tenant *throughput* does not.
- **`--parallel N` (default 4, `main.zig:171,606`) is cosmetic** — parsed + logged "max N concurrent requests" (`main.zig:2198/2426`), only sizes the unused scheduler's slot array. Batches nothing.
- **CUDA forward is single-sequence.** `decodeStep(token, pos)` (`forward_cuda_gemma.zig:638`, `forward_cuda.zig:456`); monolithic per-layer KV `kv_k[L] = [max_ctx·kv_dim]` indexed `pos·kv_dim` (`forward_cuda.zig:258,657`). No batch/sequence dimension.
- **`src/scheduler/`** has a `Scheduler` (slot accounting), `Request` lifecycle, and a real paged `KvPagePool` — but the module header says *"the batched prefill/decode dispatch loop is not wired yet,"* `pendingPrefill()`/`activeDecoding()` are stubs returning `&.{}`, and it's only `_ = @import`'d (dead) in the CUDA path. `KvPagePool` IS used by the generic `forward.zig` (CPU/Metal) but NOT by `forward_cuda*.zig`.

## Why this is the single biggest decode-throughput lever (not just a serving feature)

Decode is **GPU-launch / weight-bandwidth-bound at ~7–12% util** (established across Efforts 23/25): each step re-reads the **entire model weights** to emit ONE token. Batching B sequences reads those **same weights once** and amortizes the **same kernel launches** across B tokens → throughput scales ~linearly in B until compute-bound. So the very thing that caps single-stream decode (it's bandwidth/launch-bound, not compute-bound) is exactly what makes request-batching a **multi-× aggregate-throughput win**. `llama-server` does continuous batching — so this is the *real* "beat llama" comparison for serving, and where ZINC has the most headroom.

## THE KEY INSIGHT — batched decode ≈ the batched PREFILL we already have

`prefillBatched` (`forward_cuda_gemma.zig:729`) **already runs a B-wide forward correctly**: B token-rows through `attentionLayerBatched` (`:936`) + `ffnBlockBatched` (`:1015`), with the qkv/o/ffn **GEMMs already sized for B rows** (`BatchScratch` buffers `[T, …]`). Decode-batching reuses ALL of it. The **only** single-sequence assumptions are in TWO spots:

1. **The attention kernel `gemma_attention_batched`** (`kernels.cu:3150`): `blockIdx.y = t` is the query row, and it hard-codes `seq_len = t + 1u` (causal length = the query's position *within the prompt*) over **one shared KV** base. → Generalize to: query row `b` belongs to sequence `b` at **its own** position `positions[b]`, reading **its own** KV slot. So `seq_len = positions[b] + 1`, and the K/V base gets a per-sequence slot offset. The sliding-window logic (`:3164`) carries over per-sequence unchanged.
2. **KV write + KV layout**: today one sequence writes `kv_k[L]` at `pos·kv_dim`. → slot-based: sequence in slot `s` at pos `p` writes `(s·max_ctx + p)·kv_dim`.

Everything else (norms, RoPE, GEMMs, GeGLU, sampling) is already per-row over B, or is a trivial per-row grid. **So increment 1's genuinely-new GPU work = generalize ONE attention kernel + slot-index the KV write. The rest is buffer sizing + a driver + a harness.** This is what makes a from-scratch serving engine unnecessary.

## CURRENT STATE (read this FIRST; update it LAST every cycle)

**Increment in progress:** 1 — batched-decode isolated proof (gemma-31b dense, slot KV).
**Done so far:** **1a DONE + validated** (commit on `feat/e28-batching`). Added the `dbg_cuda batch <seqs> <ngen> [model]` harness (`'|'`-separated sequences, `,`-separated ids) — it generates each sequence INDEPENDENTLY through the production path (prefillBatched/decodeStep) as the **serial reference** the 1d proof will diff against, and emits `BATCH_SEQ{j}:...`. Added the additive **slot-based KV** to `ForwardGemma`: fields `kv_k_slots/kv_v_slots/n_slots/slot_ctx` + `allocSlotKv(n_slots, slot_ctx)` / `freeSlotKv()` / `slotKvOffsetBytes(L,slot,pos)` = `(slot*slot_ctx+pos)*kv_dim(L)*4` + a `slotKvSmoke()` plumbing test. The production single-sequence `decodeStep`/`prefillBatched`/kernels are UNTOUCHED. Validated on the 5090 (fresh isolated-cache build md5 `50fcc20437…`): catalog **5/5 token-correct serial** (12/12 each); `batch` `BATCH_SEQ0/1` **token-identical to `gen`** of each sequence's prompt (369… / 240017…); `SLOTKV_SMOKE:ok`.
**Exact next step:** **1b** — write `decodeBatch(tokens[B], positions[B], slots[B])` for DENSE gemma-31b. Reuse `BatchScratch` (T=B rows) + the batched GEMM path (`attentionLayerBatched` structure: pre-norm + Q/K/V/O GEMMs + `ffnBlockBatched` are all already B-row-capable and position-independent). The ONLY new GPU work is per-sequence positions/slot KV (defer the new attention kernel to 1c): embed B rows, run the per-layer loop, final norm + logits[B,vocab] + per-row argmax → B tokens. Wire `allocSlotKv` into it (no longer just smoke). KV write/attention can stay on the prompt-style shared path for a B=1 sanity (==gen) before 1c generalizes positions.
**Open risks / notes:** (1) the batched-prefill blocks do NOT use the dense decode norm-FOLDING (`rms_norm_residual_norm`) optimization — `attentionLayerBatched`/`ffnBlockBatched` do plain per-block pre-norms — so `decodeBatch` reusing them is self-consistent but only ARGMAX-identical (not bit-identical) to the folding `decodeStep`; the proof gate is token-identical, so fine. (2) gemma-31b reloads ~18GB/call (~45s) on the box — keep B/slot_ctx/ngen modest and batch box work (each cycle's proof = ~2 model loads + the 5-model catalog). (3) slot KV VRAM = n_slots·slot_ctx·kv_dim·4·2(k+v)·n_layers; SWA layers have the larger kv_dim (4096) — fine at small n_slots/slot_ctx.

## Increments (ORDERED and DEPENDENT — do them in order; each is several cycles)

### Increment 1 — Batched-decode ISOLATED PROOF (gemma-31b dense, slot-based KV). THE ARCHITECTURAL CRUX.
Build a batched decode path for the **dense gemma-31b** (no SSM, no MoE — the cleanest forward), and PROVE it token-identical to N separate single-sequence runs. Sub-steps (commit each as it builds, all to `feat/e28-batching`; keep the production `decodeStep`/`prefillBatched` UNTOUCHED — batching is ADDITIVE behind a new entrypoint):
- **1a. Harness + slot KV.** Add a `dbg_cuda batch` subcommand (mirror the existing `dbg_cuda gen`/`gemm` harness pattern) taking B comma-sep token-id prompts; allocate slot-based KV `kv_k_slots[L] = [n_slots·max_ctx·kv_dim]` (small n_slots e.g. 4, modest max_ctx e.g. 2048 to keep VRAM/reloads cheap). No forward yet — just plumbing + a coherent single-slot smoke (B=1 must equal `gen`).
- **1b. Batched buffers + driver.** A `decodeBatch(tokens[B], positions[B], slots[B])` that reuses `BatchScratch` (B rows) for the GEMMs and runs the per-layer loop. Embed B rows, run norms/qkv/rope/o/ffn over B rows (all already B-capable), final norm + logits [B,vocab] + per-row argmax → B tokens.
- **1c. Per-sequence kv_write + the generalized attention kernel.** kv_write into each sequence's slot at its `positions[b]`. Add `gemma_attention_batched_seq` (or extend with a `positions[]`+slot-base arg): `seq_len = positions[b]+1`, K/V base offset by slot. Per-sequence causal + sliding window.
- **1d. THE PROOF.** Run B distinct prompts through `decodeBatch` for N steps (each advancing its own position), AND each prompt separately through `decodeStep`. **GATE: the B batched token streams must be TOKEN-IDENTICAL to the B serial streams.** Mixed positions (start sequences at different lengths) to exercise per-sequence `positions[]`. This proves no cross-contamination + correct per-seq KV/attention. Until 1d passes, increment 1 is NOT done.

### Increment 2 — Continuous-batching scheduler (host-side, ragged batch).
Wire `src/scheduler/scheduler.zig`: a running batch where sequences sit at **different positions** and **join/leave between steps**. Admit a new request (prefill it into a free slot via the existing single-sequence/prefillBatched path, then add it to the decode batch); evict on EOS or max_tokens and free its slot; admit a waiting request into the freed slot. Implement `pendingPrefill()`/`activeDecoding()` for real. **GATE: an interleaved arrival/exit schedule produces per-sequence outputs token-identical to isolated runs.**

### Increment 3 — Wire the server (the actual multi-tenant throughput win).
One GPU **worker thread** runs the continuous-batch loop; HTTP request threads enqueue into the scheduler and stream their tokens back (SSE) as their sequence produces them; relax/replace the global `generation_mutex`. **GATE: N concurrent HTTP clients get correct, isolated streams AND aggregate tok/s > the serialized baseline** (measure 1,2,4,8 concurrent; this is the headline number — compare to `llama-server --parallel`).

### Increment 4 — Extend to qwen (per-sequence SSM state) + batched MoE.
Per-sequence SSM recurrent state (a batch dim on the ssm-state buffers + the `ssm_delta_net` scan in `forward_cuda.zig`); batched MoE routing (reuse `build_expert_order` over the B tokens). **GATE: qwen35/36 + MoE catalog rows token-identical batched-vs-serial.**

### Increment 5 — Paged KV (memory efficiency → higher concurrency).
Replace slot-based fixed reservation with the existing `KvPagePool` block tables, so max concurrency isn't capped by `n_slots·max_ctx` VRAM. **GATE: correctness preserved + higher max concurrency at fixed VRAM.**

## Validation contract

- **The batched path is ADDITIVE.** The production single-sequence `decodeStep`/`prefillBatched`/server path must stay the default and stay correct at every commit. Add batched paths behind a new entrypoint / flag until increment 3 flips serving over — and even then, gate it.
- **Correctness gate (increments 1,2,4): batched == serial, TOKEN-IDENTICAL** per sequence (the `dbg_cuda batch` proof). Mixed positions required.
- **`scripts/validate_catalog.sh` must stay 5/5 token-correct** (`ZINC_GPU` = 5090 UUID) on the **serial** path every cycle — batching must not regress single-sequence inference.
- **Throughput gate (increment 3):** aggregate tok/s at B concurrent clients > serialized baseline, interleaved/util-gated, on the SAME 5090. Beat `llama-server --parallel B` is the bar.
- Isolated-cache builds (`ZIG_LOCAL_CACHE_DIR`+`ZIG_GLOBAL_CACHE_DIR`; verify the binary md5 changed or you are measuring stale code).

## HARD RULES

- Work in **this worktree** `/Users/stepan/Workspace/zinc-e28` on the **persistent branch `feat/e28-batching`** (accumulate increments here; push it each cycle). **Never** work in `/Users/stepan/Workspace/zinc` (main checkout) or `…/zinc-e26`/`…/zinc-e27`. **Never push to main.**
- **5090-pinned** (UUID `GPU-5126d018-ec86-be8b-1bf5-b5ac323d3350`): `export CUDA_VISIBLE_DEVICES=…` + `ZINC_GPU=…`. The box is **dedicated to this effort** now (the e26/e27 perf loops were stopped 2026-06-14). The 4090 (`GPU-e59a6fce-…`) is also free — use it for a parallel build/validate if helpful.
- Isolated box dir `~/zinc-e28`, **never** `~/workspace/zinc`. Box gotchas: `DECODE/PREFILL/GEN_IDS` print on **STDERR** (`2>&1`); `nohup … >FILE 2>&1 &` + poll the FILE (a backgrounded `ssh '… bash'` orphans the remote script); `pkill -f <pat>` self-matches the ssh argv (kill by PID); **gemma-31B reloads 18 GB/call (~45 s) and can WEDGE the WSL2 box's sshd** — keep B/max_ctx modest, batch your box work, and if SSH hangs past connect the box needs a human restart.
- **DO NOT async gemma decode** (boost-saturated, proven regression — Efforts 23/25). Orthogonal to batching; don't introduce it.
- ONE validated increment per cycle. Commit ONLY the scoped change. Update `## CURRENT STATE` (done-so-far + exact next step + risks) and append a Cycle log entry. Never commit host/IP/port. A blocked/negative cycle is still a logged finding — leave the tree clean.

## Cycle log

(append dated entries: cycle | increment+sub-step | change | built+md5-changed? | catalog 5/5 (serial)? | batched==serial proof result | branch/sha | exact next step)

- **2026-06-14 | inc 1 / sub-step 1a | `dbg_cuda batch` serial-reference harness + additive slot-based KV (`allocSlotKv`/`freeSlotKv`/`slotKvOffsetBytes`/`slotKvSmoke`) on `ForwardGemma`; production path untouched | built clean on the 5090, fresh isolated-cache (rm -rf .zig-cache + ZIG_GLOBAL_CACHE_DIR=/tmp/e28gc), bin md5 `50fcc2043743369e5077bbffba4716b4` | catalog 5/5 serial token-correct (12/12 each; qwen35-9b, qwen36-27b, qwen36-35b-a3b, gemma4-31b, gemma4-26b) | `batch` `BATCH_SEQ0`==`gen` seq0 (369…) AND `BATCH_SEQ1`==`gen` seq1 (240017…) token-identical; `SLOTKV_SMOKE:ok` | `feat/e28-batching` | NEXT: 1b — `decodeBatch(tokens[B],positions[B],slots[B])` for dense gemma-31b reusing BatchScratch + the batched GEMMs; wire allocSlotKv in for real; B=1 sanity ==gen before 1c generalizes per-seq positions/attention.**
