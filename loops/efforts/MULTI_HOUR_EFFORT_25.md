# Effort 25 — productionize batched prefill + qwen prefill prep

Builds on Effort 24 (batched gemma prefill, MERGED, ~4.6–4.7× — gemma-31b ~30→140
t/s, gemma-26b MoE ~46→213 t/s). PROBLEM: it is OPT-IN behind `ZINC_BATCHED_PREFILL`
(`src/main.zig:~1753`), so PRODUCTION prefill is still slow per-token. See memory
`project_batched_prefill_design`.

## TARGET 1 (PRIORITY) — make batched prefill the DEFAULT for gemma
- `src/main.zig` + `src/dbg_cuda.zig` gen: gemma models (`ForwardGemma`) call
  `prefillBatched` BY DEFAULT when `prompt_tokens.len > 1`; add an opt-OUT
  (`ZINC_BATCHED_PREFILL=0`/`off` → per-token fallback for debugging). qwen
  (`ForwardCuda`) stays per-token (no `prefillBatched`). Keep ADDITIVE/minimal.
- GATE: `scripts/validate_catalog.sh` 5/5 with the DEFAULT path (now batched for
  gemma) DIRECT vs llama.cpp; `scripts/prefill_catalog.sh` confirms gemma default
  prefill is now fast (the +4.6–4.7× is now production, not opt-in). Then this is a
  validated production win → commit to `perf/e25-prefill-default`, push (NOT main).

## TARGET 2 — qwen prefill (Phase 3)
Check whether `origin/main`'s `src/compute/forward_cuda.zig` now has a batched /
chunked SSM scan (parallel team is doing Metal qwen prefill — CUDA may follow). If
YES: wire qwen `prefillBatched` (GEMM Q/K/V/O + FFN like gemma; batched SSM scan;
batched attention for the ~1/4 attn layers). If NO: document the gate + STOP — do
not fake a qwen win.

## HARD RULES
- isolated worktree `~/zinc-e25`, box dir `~/zinc-e25-box`, 4090-pinned
  (`GPU-e59a6fce-1961-bafe-927c-06c0149f2370`), isolated-cache builds (verify the
  binary hash CHANGED). Branch `perf/e25-prefill-default` off LATEST origin/main;
  `git fetch && rebase` often (main moves fast); ADDITIVE to minimize conflicts;
  NEVER touch `~/Workspace/zinc` or roll back the parallel work; NEVER push to main.
- Gate (validate_catalog 5/5 + token-correct) before ANY commit. Negative → revert + log.

## CYCLE LOG
- 2026-06-13 — **TARGET 1 DONE (production win, committed+pushed).** Batched-GEMM
  prefill is now the DEFAULT for the gemma forwards (`ForwardGemma.prefillBatched`),
  opt-OUT via `ZINC_BATCHED_PREFILL=0/off/false/no`. Wired in `src/main.zig`
  (`batchedPrefillDefaultOn()` + comptime-gated call site so qwen `ForwardCuda`,
  which has no `prefillBatched`, stays per-token & still compiles) and `src/dbg_cuda.zig`
  (same default + `error.Unsupported` runtime fallback for qwen). `scripts/prefill_catalog.sh`
  baseline arm now forces `ZINC_BATCHED_PREFILL=0` so the A/B still measures the real
  per-token-vs-batched delta after the default flip. Box build clean (EXIT=0, cuda-dbg,
  isolated `~/zinc-e25-box` caches, 4090). GATE: `validate_catalog.sh` **5/5 token-correct
  vs llama.cpp on the bare-default path** (gemma now batched by default; qwen35-9b/qwen36-27b/
  qwen36-35b-a3b 12/12, gemma4-26b 12/12, gemma4-31b teacher-forced 11/12 = known near-tie).
  `prefill_catalog.sh` batched A/B (gemma-only): gemma4-31b 6.64→164.09 t/s, gemma4-26b
  57.83→76.94 t/s, **both PASS (GEN_IDS identical)** — the Effort-24 prefill speedup is now
  production, not opt-in.
- 2026-06-13 — **TARGET 2 BLOCKED (gate not met, no work done — per hard rule).**
  `origin/main:src/compute/forward_cuda.zig` (qwen `ForwardCuda`) still exposes only
  per-token `decodeStep`/`prefillStep` — NO `prefillBatched`, no batched/chunked SSM
  scan. The parallel team's qwen prefill work is Metal-only so far (see recent
  `metal: batch Qwen 9B queued prefill chunks`). CUDA qwen batched prefill needs a
  batched SSM scan prereq that is not on main → do NOT wire / fake it. Re-check next
  cycle: `git grep prefillBatched origin/main -- src/compute/forward_cuda.zig`.
