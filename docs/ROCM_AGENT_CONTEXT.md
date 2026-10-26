# ROCm attention and coding-context fixes

The first [full-worktree OpenCode trial](OPENCODE_WORKTREE_REPORT.md) failed
before implementing its feature. It exposed two distinct limits: attention
allocated a full context of scores in shared memory, and prompt scratch plus
duplicate KV caches exhausted VRAM. A larger context setting alone did not
solve either problem.

The follow-up replaces those limits in the Qwen ROCm path. Validation used
Zig 0.17.0, Bun 1.4.4, ROCm 7.2.4, and a Radeon AI PRO R9700 with 32GB VRAM.
The model was Qwen 3.8 27B Q4_K_M, with NextN/MTP enabled and one request slot.
The [portable result summary](/zinc/validation/rocm-agent-context-2026-10-25.json)
records the source commits, checks, and scoped measurements without machine
addresses or local paths.

## What changed

- Attention above 8,192 keys uses 1,024-key tiles and an online softmax.
  Decode, causal prefill, MTP verification, and independent request slots share
  that bounded implementation. Each compiled kernel uses 4,364 bytes of static
  shared memory and no dynamic shared memory. The previous short-context
  kernels remain in use below the threshold.
- Prefill processes at most 4,096 tokens per chunk by default. Absolute RoPE
  positions, KV offsets, recurrent state, and convolution history carry across
  chunks. The NextN head is primed at the same boundaries, including the
  preceding chunk's final hidden row.
- A single request slot aliases the existing KV and recurrent allocations.
  It no longer allocates a second complete cache. Slot prefill also preserves
  buffer descriptors correctly, and partial scratch/slot allocation failures
  release allocations already made.
- GPU launch, transfer, BLAS, and synchronization failures latch an execution
  error. Generation fails instead of publishing stale results. A failed
  context returns HTTP 503 from `/health` and requires a server restart.
- Optional exact-prefix reuse restores a recurrent-state checkpoint before
  continuing prefill. It reuses only matching token IDs in a single Qwen
  request slot; it does not approximate or summarize the prompt.

The softmax recurrence follows the standard
[tiled attention formulation](https://arxiv.org/abs/2205.14135): a running
maximum rescales both the denominator and weighted values when a later tile
contains a larger score. Attention sinks participate in the final denominator.
Long-context floating-point reduction order differs from the old kernels;
numerical checks use an independent double-precision reference.

## Kernel and model checks

The GPU validator passed 25 cases. It covers tile tails, the 8K dispatch
boundary, the previously failing 16,080-token prefill, 16K/32K decode, custom
attention scales, attention sinks, GQA, head dimensions spanning multiple
output stripes, independent/permuted slots, speculative positions, and changed
parameters during graph capture/update. Legal short cases also compare with
the original GPU kernels. Output guards are checked. Invalid shared-memory
and kernel launches are rejected, and their execution errors remain latched.

The independent CPU comparison requires finite output and absolute error at
most `0.0003 * (1 + abs(reference))`. These are ROCm/R9700 GPU results; the
shared CUDA implementation needs its own hardware validation.

The 27B server then returned the exact requested key from the start of long
prompts:

| Prompt tokens | Result             |
| ------------: | ------------------ |
|         8,548 | Exact key returned |
|        17,048 | Exact key returned |
|        30,048 | Exact key returned |

A short request after those prompts also passed. A prompt above the 32,768-token
capacity returned HTTP 400, followed by another successful short request.
These retrieval probes check long-context execution; repository coding is a
separate acceptance test.

Two controlled fault-injection checks exercised recovery. An execution failure
before the first attention result produced HTTP 500 for JSON generation, HTTP
503 for health, and an SSE error without a successful finish reason. Later
generation requests also failed. No illegal GPU memory access was injected.

A separate Debug build simulated an allocation failure after a prefill chunk
had already advanced recurrent state. Clearing that partial state before the
ordinary-decode fallback produced identical response bytes and token counts
to an independent ordinary-decode control, for two requests using the same
slot. The injections exist only in private validation builds; production has
no fault-injection setting.

## Reuse repeated prompt prefixes

Set `ZINC_PROMPT_CACHE=1` when starting a single-slot Qwen server to enable
reuse. It is off by default. At a fixed prefill-chunk boundary, ZINC saves the
recurrent state, convolution history/offsets, and NextN boundary hidden state.
The attention KV prefix stays in place. A later request must have the exact
same token prefix and execution mode to restore that checkpoint. Changed or
shortened prefixes invalidate it; a failed prefill also invalidates it.
This keeps chunk boundaries and floating-point operation order unchanged.

The checkpoint consumes an additional recurrent-state allocation in VRAM.
It is not a disk cache and does not support multiple slots. The API still
reports the complete prompt length in usage; `/stats` and the prefill timing
log count only tokens actually computed.

On the R9700, six-request sequences with reuse off and on produced identical
greedy response hashes and token usage for both Qwen 3.5 9B and Qwen 3.8 27B
with NextN enabled. They cover two shared prefixes, a changed prefix, a short
request, and a long request after that reset. Only the two expected shared
prefixes reused tokens. The portable validator was also run against the
ReleaseFast 9B server: all 12 requests passed.

Run that validator on the GPU test host with an unused output directory:

```bash
bun tools/validate_rocm_prompt_cache.mjs \
  --zinc ./zig-out/bin/zinc --model /path/to/model.gguf \
  --gpu "${ZINC_GPU:?Set your visible GPU selector}" \
  --context 32768 --mtp 1 --output /tmp/zinc-prefix-validation
```

Use `--mtp 0` for a model without a NextN block. The validator owns sequential
cache-off/cache-on servers, checks exact response/usage parity and computed
prefill counters, and stops both servers. Its public result JSON omits local
paths and device selectors; raw server logs stay in the chosen directory.
These are correctness diagnostics, not a replacement for the matched
performance suite.

A captured 25,649-token coding request also produced identical content, tool
arguments, finish reason, and token usage across two uncached runs and cold/
warm cached runs. The warm cached request took 26.22 seconds versus about
71.8 seconds without reuse in this diagnostic. A llama.cpp HIP reference replay
returned a valid tool call too, but used a different rendered prompt (25,925
tokens) and chose a different next action. That is neither an exact-template
comparison nor a coding-task pass.

## Repository coding result

The backend checks above do not establish reliable autonomous coding. The
follow-up full-worktree runs still needed supervision:

| Run                                               | Outcome                                                                                                                                                                |
| ------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 27B, 32K, normal OpenCode tools                   | Stopped after 1,686 seconds of inspection; 22 tool calls, no feature edits or project tests. No GPU execution error.                                                   |
| 9B, 32K                                           | Timed out after 900 seconds; 23 tool calls produced a partial, uncompilable discovery endpoint.                                                                        |
| Resumed 9B with exact-prefix reuse                | Timed out after 1,200 seconds; 141 tool calls, including repeated calls within individual model responses. The patch still failed validation.                          |
| 27B repair with independent checks between rounds | Repaired the serializer and server wiring, then added the missing module documentation after a failed project check. The overall run still timed out at 1,200 seconds. |

In the final run, the agent executed a command that passed the protected
serializer acceptance test, 725 Zig tests, 674 Bun tests, and ROCm compilation.
However, the wrapper had not completed its next independent check when the
timeout expired: the verbose successful command log had expanded the next
model request to 28,200 prompt tokens. This run is recorded as **timed out**,
not as a successful verified coding task.

The reviewed discovery implementation is included in the source. Supervisor
review added a long-name ownership test and corrected the separate managed
startup-path handoff. These supervisor changes are not attributed to the model.
The serializer now has ten CPU tests covering JSON shape, integer fields,
escaping, Unicode, basename privacy, and allocation ownership.

The harness now gives verified rounds a different contract: edit, finish the
round, and let the wrapper run the full check outside the model conversation.
Failures return bounded diagnostic excerpts, including errors in the middle of
large logs. The complete output still streams to stderr. Unit tests cover
same-session repair, failed checks, API errors, cancellation, timeouts, and
diagnostic truncation. This latest prompt contract still needs a successful
live repository-task run.

### Harness recovery checks (2026-10-26)

Verified editing rounds now have a separate deadline, defaulting to five
minutes, and reserve time for the independent check. When editing reaches its
deadline, the wrapper stops OpenCode and checks the partial patch. A failed
check can resume the same session; a passing check can finish the task.
Cancellation and API errors still fail the run. Owned POSIX background tools
are stopped even when the main child exits first.

A CPU protocol fixture exercised actual OpenCode 1.19.4 with scripted streaming
responses. The first response wrote an incorrect implementation, then the
next response stalled. Both wrapper versions had the same 12-second overall
budget. The earlier wrapper exited 124 without running verification. The new
wrapper stopped editing after eight seconds, ran a failing acceptance check,
resumed the same session, applied the scripted correction, and passed the
independent check in 8.9 seconds. The acceptance test and portable configuration
remained unchanged. These are harness transport/lifecycle checks; no model
performed inference or demonstrated coding ability in this fixture.

That recovery change passed all 26 harness tests on Linux and macOS, covering cancellation and API
errors as well as real subprocess checks for partial edits and background
process cleanup.
The combined Zig/Bun CPU command passed; standalone Bun had 692 passing tests,
six skipped GPU/API tests, and no failures. The RDNA GPU remains unavailable,
so model-driven coding acceptance and performance measurements remain pending.

### Harness failure checks (2026-10-26)

Further failure checks found that an OpenCode JSON API error could leave the
client alive until its editing deadline. The wrapper now stops that child as
soon as it reports the error, returns failure even if the child exits zero,
and skips independent verification. A real subprocess regression checks that
the stopped client cannot continue tool work after the error.

Empty or whitespace-only tasks are rejected before reading the project config
or launching commands. Task and round timeouts are limited to positive integer
milliseconds up to 2,147,483,647: a Bun probe confirmed that a larger native
timer instead fires after one millisecond. The overall budget now uses a
monotonic clock, and verification returning zero after that deadline cannot
produce success. User interruption still takes precedence over timeout.

All 31 harness tests pass on Linux and macOS. The full CPU suites pass with
697 Bun tests, six GPU/API skips, and no failures, plus the Zig suite.
Actual OpenCode 1.19.4 also passed two CPU protocol fixtures using scripted
responses: the timeout/repair sequence still completed, and an HTTP API error
produced a failing run without starting verification. The portable config
remained unchanged in both fixtures. These checks perform no model inference;
the GPU-dependent acceptance and performance checks below remain pending.

## Remaining hardware validation

After the coding trial, a short-context regression comparison could not start:
the previously working R9700 failed to resume from PCIe sleep, and the older
control binary returned `CudaNoDevice` before loading the model. Kernel logs
reported a D3 power-state/resume failure. This is not a measured performance
regression. All owned model servers were stopped.

Active-power requests and device-only reset/rebind attempts did not recover
the GPU; a driver probe remained blocked in a firmware mailbox handshake.
Until recovery, the latest live HTTP discovery/managed-ID
checks, full-model multi-slot/graph checks, short-context regression matrix,
and another live coding acceptance run remain **pending**. No passing result
is claimed for them.

## Reproduce the GPU validator

Run this only on the GPU test host, from the repository root. It loads no model.

```bash
cc -O2 -std=c11 -D_POSIX_C_SOURCE=200809L -D__HIP_PLATFORM_AMD__=1 \
  -Isrc/cuda -I/opt/rocm/include tools/validate_rocm_attention.c \
  src/rocm/rocm_shim.c -L/opt/rocm/lib -Wl,-rpath,/opt/rocm/lib \
  -lamdhip64 -lhiprtc -lhipblas -lm -o /tmp/validate-rocm-attention
ROCR_VISIBLE_DEVICES="${ZINC_GPU:?Set your visible GPU selector}" /tmp/validate-rocm-attention
```

For repository tasks on a suitable GPU, use `--context 32768 --parallel 1`
and match that limit in `tools/setup_opencode.mjs --context 32768`.
`ZINC_PREFILL_CHUNK=<positive token count>` changes the scratch bound; increasing
it consumes more VRAM. Keep file reads targeted and inspect the worktree and
test output independently of the agent's final message.
