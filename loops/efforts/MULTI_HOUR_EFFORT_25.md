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
- (none yet)
