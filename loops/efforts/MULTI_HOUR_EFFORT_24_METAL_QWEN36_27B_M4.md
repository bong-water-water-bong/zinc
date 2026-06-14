# Effort 24 - Metal Qwen 3.6 27B dense-hybrid decode on M4

Created: 2026-06-12

## Objective

Close the largest current Apple Silicon M4 public-suite gap: Qwen 3.6 27B
Dense Q4_K_M on the Metal backend.

This is the 27B dense-hybrid model, not the 35B-A3B MoE model from Effort 16.
It has SSM layers and dense FFN layers, but no routed MoE. Do not carry over
MoE route-pack assumptions from the 35B work unless a fresh profile proves the
same code path is hot.

Primary target:

- Model id: `qwen36-27b-q4k-m`
- File: `Qwen3.6-27B-Q4_K_M.gguf`
- Backend: Metal
- Machine: local M4 Max / Apple9, 64 GB unified memory
- Architecture: Qwen 3.6 dense hybrid with SSM layers
- Public-suite prompt mode: raw completion

Use the managed model cache:

```bash
./zig-out/bin/zinc model pull qwen36-27b-q4k-m
```

The model is expected at:

```text
~/Library/Caches/zinc/models/models/qwen36-27b-q4k-m/model.gguf
```

## Why this is the biggest M4 gap

Latest published M4 suite data, generated `2026-06-11T23:11:26.405Z`:

| scenario | ZINC decode | llama.cpp decode | ZINC / llama | status |
|---|---:|---:|---:|---|
| core | n/a | 23.09 tok/s | n/a | ZINC timed out |
| context-medium | n/a | 22.98 tok/s | n/a | ZINC timed out |
| context-long | 10.48 tok/s | 23.02 tok/s | 45.5% | measured |
| decode-extended | 10.57 tok/s | 22.93 tok/s | 46.1% | measured |

Qwen 3.5 9B is also behind on M4, but its core row is about 62% of llama.cpp.
Qwen 3.6 27B is worse: the short rows time out and the long rows are only
about 46% of llama.cpp. This effort should target 27B first.

## Run the loop

Use the public-suite `context-long` raw prompt. It is long enough to exercise
the real steady-state path but shorter than the 256-token long-draft row, so
cycle time stays manageable.

```bash
PROMPT='Incident notes from a local inference service:
- 09:14: A developer compared two screenshots and saw 49 tok/s in one run and 37 tok/s in another.
- 09:18: The faster run generated one token after a short chat prompt. The slower run generated 160 tokens after a pasted support ticket.
- 09:22: The CLI output reports separate lines for prefill throughput and generated-token throughput.
- 09:27: The team sometimes runs with --profile, which adds per-dispatch accounting and changes CPU-side overhead.
- 09:33: The llama.cpp baseline is measured through a persistent server, while some ZINC checks were one-off CLI runs.
- 09:41: A background video export was active during one Apple Silicon run.
- 09:48: The benchmark dashboard now computes an overall prompt+decode score from prompt tokens, generated tokens, prefill speed, and decode speed.

Relevant policy:
- Publish medians from repeated warm runs, not screenshots.
- Keep the same model file, prompt mode, output cap, and backend residency for both engines.
- Treat one-token completions as coherence smoke tests, not sustained throughput measurements.
- Report prefill, decode, total latency, and prompt+decode throughput together.

Question: summarize the measurement mistake and propose the benchmark protocol the team should use next.

Engineering guidance:'

ZINC_MODEL_ID=qwen36-27b-q4k-m \
ZINC_METRIC_MODE=decode \
ZINC_PROMPT_MODE=raw \
ZINC_TEST_PROMPT="$PROMPT" \
ZINC_REFERENCE_TEXT=prompt \
ZINC_MAX_TOKENS=128 \
ZINC_MIN_DECODE_TOKENS=32 \
ZINC_TARGET_TOK_PER_SEC=23 \
ZINC_STOP_ON_TARGET=0 \
ZINC_BENCHMARK_RUNS=3 \
ZINC_BENCHMARK_WARMUPS=1 \
ZINC_BENCHMARK_CONFIRM_RUNS=4 \
ZINC_PROFILE_EVERY=1 \
ZINC_BUILD_OPTIMIZE=ReleaseFast \
ZINC_TEST_TIMEOUT_MS=300000 \
ZINC_RUN_TIMEOUT_MS=1200000 \
ZINC_CROSS_EFFORT_PROMPT="Developer question: two local LLM benchmark screenshots show different tok/s values for the same model. A useful answer explains likely causes and gives one fair measurement rule.\n\nAnswer:" \
ZINC_CROSS_EFFORT_METRIC=prefill \
ZINC_CROSS_EFFORT_PROMPT_MODE=raw \
ZINC_CROSS_EFFORT_MAX_TOKENS=32 \
ZINC_CROSS_EFFORT_EVERY=3 \
ZINC_HARD_FAMILY_COOLDOWN=1 \
ZINC_WORKLOAD_RESET_ON_CHANGE=1 \
ZINC_CODEX_REASONING_EFFORT=xhigh \
bun loops/implement_metal.ts --effort 24 --agent codex --model gpt-5.5 --cycles 100
```

For a baseline-only check:

```bash
ZINC_MODEL_ID=qwen36-27b-q4k-m \
ZINC_METRIC_MODE=decode \
ZINC_PROMPT_MODE=raw \
ZINC_TEST_PROMPT="$PROMPT" \
ZINC_REFERENCE_TEXT=prompt \
ZINC_MAX_TOKENS=128 \
ZINC_MIN_DECODE_TOKENS=32 \
ZINC_BENCHMARK_RUNS=3 \
ZINC_BENCHMARK_WARMUPS=1 \
ZINC_RUN_TIMEOUT_MS=1200000 \
bun loops/implement_metal.ts --effort 24 --dry-run
```

## Baseline interpretation

The controlling public-suite numbers are:

- current ZINC: about `10.5 decode tok/s` on the long rows
- llama.cpp: about `23.0 decode tok/s`
- first milestone: `14 tok/s`
- second milestone: `18 tok/s`
- parity target: `23 tok/s`

Do not optimize from Qwen 3.6 35B-A3B's `82 tok/s` decode result. That model
is routed MoE and has very different per-token weight traffic. The 27B dense
hybrid path streams dense FFN weights on every layer.

## Current checkpoint

Updated after local M4 run `.metal_optimize/2026-06-12T22-31-20` cycles 1-9:

- Baseline locked at `10.48 decode tok/s` on the effort prompt.
- Cycle 1 kept profile enablement for dense FFN gate/up/down byte buckets.
- Cycle 2 kept hybrid SSM+dense decode command coalescing at `10.50 tok/s`.
- Cycle 3 tried enabling the existing fused dense Q4_K gate/up+SwiGLU path for
  `.qwen35` dense/no-expert models. It reduced dense barriers (`dense
  barriers/step 256 -> 192`) but measured slower at `10.42 tok/s`; the harness
  restored the promoted-best cycle-2 tree. Do not retry this guard flip unless
  a same-cycle A/B explains why fewer barriers lost throughput and gives a
  different fix.
- Cycle 4 kept the major win: route Qwen3.6 27B exact dense Q6_K down
  projection `M=5120 K=17408` to the optimized llama-style Metal DMMV path.
  Median moved to `12.56 decode tok/s` (`+2.06 tok/s`, about `+19.6%`).
- Cycle 5 tried routing the exact dense Q4_K gate/up pair to the existing
  non-SwiGLU dual projection kernel. It regressed to `11.07 tok/s` and was
  reverted. Do not retry that route without shader-level evidence explaining
  the loss.
- Cycle 6 kept a neutral fixed-`K=17408` Q6_K pipeline specialization for the
  same dense-down shape. It measured `12.56 tok/s`, so treat it as cleanup /
  enablement around the proven cycle-4 route, not as a separate speed win.
- Cycle 7 kept a neutral fixed-`K=5120` Q4_K pipeline for dense
  `ffn_gate.weight` / `ffn_up.weight`; it measured `12.55 tok/s`.
- Cycle 8 kept a neutral fixed-`K=17408` Q4_K route for dense
  `ffn_down.weight`; it measured `12.55 tok/s`.
- Cycle 9 kept the second real win: route exact Qwen3.6 27B SSM Q6_K
  `attn_qkv.weight` `M=10240 K=5120` through the existing llama-style Metal
  Q6_K DMMV path. Median moved to `13.15 decode tok/s`.
- Cross-effort prefill improved from `11.60` to `15.00 tok/s` (`+29.3%`), so
  keep the Q6_K dense/SSM routes while watching long prompt correctness.

Cycle-2 promoted-best profile:

```text
decode buckets: dense ffn total 299.14 GiB gate 92.64 GiB up 92.64 GiB down 113.87 GiB
decode buckets: ssm proj 77.48 GiB (qkv 50.23 gate 24.52 tail 2.72) out 29.97 GiB
decode buckets: final 39.25 ms lm-head 30.11 GiB
decode q4_k hot #1: dense M=17408 K=5120 bytes=185.27 GiB calls=3968
decode q4_k hot #2: dense M=5120 K=17408 bytes=46.32 GiB calls=992
decode q6_k hot #1: dense M=5120 K=17408 bytes=67.55 GiB calls=992
decode q6_k hot #2: lm-head M=248320 K=5120 bytes=30.11 GiB calls=31
barriers/step: attn 128.0 ssm 288.0 dense 256.0 final 0.3
```

Best next directions from this checkpoint:

1. Continue from the Q6_K dense-down route, but do not repeat the generic
   route-expansion work. The accepted route is already exact to 27B
   `ffn_down.weight`.
2. Treat Q4_K dense gate/up `M=17408 K=5120` as hot, but do not re-enable the
   existing `.qwen35` fused gate/up+SwiGLU path or the non-SwiGLU dual route
   without explaining the cycle-3/cycle-5 slowdowns.
3. Continue SSM projection work from the exact Q6_K `attn_qkv.weight`
   `M=10240 K=5120` route. Do not broaden Q6_K routing to SSM out, LM head,
   or other tensors without shape-specific profile evidence.
4. The fixed-K Q4_K single-projection routes were neutral. Keep them only as
   harmless specialization cleanup; future Q4_K work needs shader-level
   evidence, not more selector variants.
5. LM head is visible but smaller than dense and SSM buckets. Do not chase it
   before the dense and SSM buckets unless a profile moves it higher.

## Post-cycle-88 checkpoint

Updated after resumed local M4 run `.metal_optimize/2026-06-12T22-31-20`
cycles 75-88:

- Best promoted cycle is cycle 80 at `15.0917 decode tok/s`, commit
  `ea522741`: exact Qwen3.6 27B Q4_K/Q4_K SSM `attn_qkv.weight` +
  `attn_gate.weight` route through the Q4 dual-row Metal dispatch.
- Cycle 81 is useful evidence but not a real speed step: dense Q4_K
  `ffn_gate.weight` + `ffn_up.weight` was routed through the same single-axis
  Q4 dual dispatch, reducing dense gate/up dispatches, but median stayed flat
  at `15.0869 decode tok/s`.
- A harness parsing bug was found after cycle 81: the prompt text contains the
  phrase "generated 160 tokens", and the old parser could record that prose as
  `tokensGenerated`. The parser now requires the engine timing form
  `Generated N tokens in X ms/s`. Post-fix samples correctly report `128 tok`.
- After the parser fix and resume, cycles 82-88 all measured a stable live band
  around `14.69-14.73 decode tok/s` and were reverted. Treat the persisted
  `15.09` best as a checkpoint that needs clean revalidation before publishing
  site metrics.
- Rejected post-best variants: Q4 dual nibble-mask vectorization, fixed-K5120
  Q4 dual route, fixed-K5120 mixed Q6/Q4 SSM pair route, tail-free Q4 dual row
  alignment, exact Q6/Q4 SSM pair specialization, dense-down tail barrier
  narrowing, and Q4 dual `float2` accumulator cleanup. Do not repeat these
  without new profiler evidence.
- Latest profile interpretation: dense FFN still dominates the slowest slot,
  and encode time is small compared with GPU execution. Cycle 81 showed that
  simply reducing dispatch count is not enough; the remaining gap is mostly
  kernel body / memory-traffic efficiency, especially dense FFN Q4 gate/up and
  Q6 down bytes.

Harness follow-ups before publishing:

1. Revalidate the saved best tree after `--resume` before using
   `state.bestTokPerSec` as the acceptance threshold.
2. Add Qwen 27B dense-decode plateau guidance equivalent to the Gemma/Qwen 35B
   plateau blocks in `loops/implement_metal.ts`: after several reverted cycles,
   require either fresh exact-shape evidence or a structural change.
3. Keep exact-shape microbench/validator foundation work even if decode is
   flat, but label it `@@@STEP_KIND: analysis` or `@@@STEP_KIND: enablement`
   and name the unlocked speed path.
4. The generic prompt still contains older Qwen 35B MoE and Gemma guidance.
   Effort 24 must continue to override that: no MoE route-pack or prefill-only
   changes for this dense-decode run unless the current 27B profile names that
   path.
5. Before updating published metrics, rerun the full public M4 suite for
   `qwen36-27b-q4k-m` on the committed tree and report the conservative median,
   not the single best cycle.

## Post-cycle-100 outcome

The 100-cycle run finished with no additional runtime keeps after cycle 81.
The worktree was clean at `metal-loop: pre-cycle-100`; cycle 100's fixed-K5120
fused Q4 gate/up+SwiGLU candidate built and tested, but measured only
`14.49 decode tok/s` and was reverted by the harness.

Committed useful artifacts from the tail of the run:

- Runtime best checkpoint: cycle 80 `ea522741` at `15.0917 decode tok/s`
  routes exact Qwen3.6 27B Q4_K/Q4_K SSM `attn_qkv.weight` +
  `attn_gate.weight` through the Q4 dual-row dispatch.
- Runtime follow-up: cycle 81 `e9e83050` routes exact dense Q4_K
  `ffn_gate.weight` + `ffn_up.weight` through the same dispatch and reduces
  dispatch count, but is effectively flat at `15.0869 decode tok/s`.
- Harness correctness: `d2b57ac1` fixes generated-token parsing so prompt
  prose such as "generated 160 tokens" is no longer mistaken for engine timing.
- Evidence enablement: `790ba581` adds `qwen27b_decode_hot` exact-shape Metal
  benchmark cases for dense Q4 gate/up, Q6 down, SSM qkv/gate/out, and LM head.
- Analysis checkpoints: `cb51dc4b` and `188f26aa` document the dense-decode
  plateau and the required revalidation gate.

Deep analysis after cycles 82-100:

1. The current runtime algorithm is bandwidth-bound in dense FFN kernels, not
   blocked primarily on CPU encode or command-buffer count. Cycle 81 reduced
   dense gate/up dispatch count without moving decode speed.
2. The reverted families cover the obvious local variants: Q4 dual
   vectorization, fixed-K5120 Q4/Q6 pair routes, SSM pair specialization,
   dense-down tail/barrier tweaks, copied dense weights, and fused dense
   gate/up+SwiGLU. Do not reopen those without exact-shape evidence.
3. The harness is too strict for evidence-only work and too trusting of a
   persisted best. It should revalidate saved best trees on resume, preserve
   labeled `analysis`/`enablement` artifacts, and run the new exact-shape
   benchmark pass automatically before asking for another shader retune.
4. The next productive step is not another blind runtime edit. First run the
   full public M4 suite and the exact-shape benchmark on the kept tree; then
   only change production code if that packet names a concrete dense Q4 gate/up
   or Q6 down kernel route that beats the current path.

## First-cycle checklist

Before editing:

1. Build with `zig build -Doptimize=ReleaseFast`.
2. Run `zig build test`.
3. Record the baseline median and sample range on the effort prompt.
4. Capture `--profile` output on cycle 1.
5. Name the largest bucket before choosing an edit.

Expected useful profile lines include:

```text
Metal profile:
dispatch/step:
barriers/step:
dmmv bytes/request:
path bytes/request:
prefill buckets:
dense barriers:
```

If the profile does not include enough detail for 27B, the first accepted
cycle may be an `@@@STEP_KIND: enablement` change that adds default-off
profiling for dense FFN, SSM projection/out, full-attention QKV/O, and LM head.

## Likely hot buckets

Use the fresh M4 profile as the authority. The expected suspects are:

1. Dense FFN decode:
   - `ffn_gate` / `ffn_up` Q4_K streams
   - `ffn_down` Q6_K stream
   - SwiGLU / residual tail dispatches
2. SSM decode:
   - QKV / gate projections
   - recurrent conv and delta update barriers
   - SSM out projection
3. Full-attention layers:
   - QKV projection and output projection
   - KV write and flash-attention barriers at longer contexts
4. LM head:
   - `output.weight` is Q6_K with large vocab
   - only chase this after the profile says it is a top bucket

## Do not repeat first

- Do not make broad threadgroup-size sweeps without exact-shape evidence.
- Do not copy Qwen 35B MoE route-pack changes into 27B; there is no MoE route.
- Do not make SSM layer-major prefill changes for a decode-scored effort unless
  the profile shows prompt ingestion is dominating the measured row.
- Do not accept a one-token or short completion as a decode win. Keep
  `ZINC_MIN_DECODE_TOKENS=32`.
- Do not quote a single sample. Use median and sample range.

## Candidate directions

1. Add or improve exact-shape Metal microbench coverage for the 27B hot DMMV
   shapes, then route only proven wins.
2. Fuse dense FFN tail work where correctness is easy to validate:
   norm -> gate/up input preparation, SwiGLU -> down input, or down -> residual.
3. Reduce decode-side command and barrier count only where the profile names
   a large bucket. Keep the dependency chain explicit.
4. Add a 27B-specific profile split if the current counters merge dense, SSM,
   and LM-head work too coarsely.
5. Once decode moves, rerun the public suite for `qwen36-27b-q4k-m` only and
   compare all four rows against llama.cpp.

## Success criteria

This effort is succeeding when:

- the public-suite `context-long` and `decode-extended` rows move above
  `14 tok/s` first, then `18 tok/s`
- core and context-medium stop timing out
- output contains prompt/protocol guidance on the effort prompt
- `zig build test` passes
- the remaining gap to llama.cpp is explained by named profile buckets
