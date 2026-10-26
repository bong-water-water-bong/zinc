# OpenCode full-worktree trial

This is the original failed trial. The later
[ROCm context and harness follow-up](ROCM_AGENT_CONTEXT.md) records the backend
fixes and subsequent acceptance results; it does not change these outcomes.

On 2026-10-25, OpenCode 1.19.4 was assigned a real Zinc feature in a complete
Git worktree based on `09b802de`: add loaded-model discovery to the minimal
ROCm/CUDA HTTP server. The task required an OpenAI-compatible `GET /v1/models`
response, basename-only model IDs, correct JSON escaping, CPU unit tests, and a
documentation note.

**Result: failed.** Eight configurations produced 57 tool calls, zero feature
source edits, and zero project test runs. The endpoint remained unimplemented.
This was a supervised capability trial, including navigation hints, configuration
changes, and a resumed session. It was not an unattended or one-shot success.

The runtime was Zig 0.17.0, Bun 1.4.4, ROCm 7.2.4, and a Radeon AI PRO R9700
with 32GB VRAM. The models were Qwen 3.8 27B Q4_K_M and Qwen 3.5 9B Q4_K_M.
ZINC served the model directly; no proxy repaired responses or synthesized
edits. Raw prompts, machine paths, node addresses, credentials, and transcripts
are excluded from this report.

| Configuration                                        | Observed outcome                                                         |
| ---------------------------------------------------- | ------------------------------------------------------------------------ |
| 27B, 32K context                                     | GPU allocation failure during the initial repository prompt              |
| 27B, 16K context                                     | Context overflow after source inspection                                 |
| 9B, 32K, general output limits                       | Built-in file reads bypassed those limits and overflowed context         |
| 9B, 32K, Bash-only prototype                         | Attention launch exceeded shared memory; output became corrupted         |
| 9B, 16K, early compaction                            | Ended without an implementation                                          |
| 27B, shorter instructions, Bash-only                 | Repeated inspection; stopped without edits                               |
| Resumed 27B, retained recent history                 | Repeated inspection; stopped without edits                               |
| 27B, normal tools, shorter instructions, small reads | Illegal GPU memory access and corrupted output near the context boundary |

The final server log identified `attention_causal_batched_v2` with a
65,612-byte group segment at 16,080 prompt tokens, followed by illegal-memory
errors and a rocBLASLt/Tensile error. Another run rejected `naive_attention`
launches above 16K tokens. These are backend failures requiring attention-kernel
validation beyond the earlier bounded recurrent-scan checks. They are not
evidence that a larger configured context is usable.

The trial also found workflow problems. The original 31KB `AGENTS.md` consumed
much of a small context window. Moving code maps and benchmark recipes into
reference documents reduced it to about 10KB while retaining permission and
GPU placement rules. Zero-history compaction discarded source that the model
then read again. The Bash-only prototype bounded output, but did not demonstrate
a completed coding task and was discarded. The public setup script retains its
normal Read/Edit/Write tools.

The earlier nine-fixture result does not establish general repository-level
coding reliability. A real task needs a source patch, independent acceptance
checks, passing project tests, and a clean server log. A process exit code or
number of tool calls is insufficient. All model servers owned by this trial
were stopped, and the GPU was idle after cleanup.

Post-trial repository checks passed on macOS and Linux, along with 676 Bun tests
and the 263-page site build. Six model-dependent Bun tests were skipped. These
checks validate the documentation and comment cleanup; they do not turn the
failed coding trial into a pass. Five pre-existing source-comment hygiene
matches were corrected without changing engine behavior. The supervisor also
fixed Markdown links in the website renderer and mobile wrapping of long
inline code; these changes were not produced by the coding agent.

[Reproduce an isolated worktree run](OPENCODE.md#try-a-real-task-in-a-worktree).
