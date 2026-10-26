# Zig 0.17 validation samples

See [the migration report](../../ZIG_0_17_MIGRATION.md) for methodology and results.
The frozen baseline is main at `29bc350ac4cbf9f110ec628b9e177ea04ac816fb`,
compiled with Zig 0.15.2. Binary, model and shader hashes identify each GPU
comparison. Timings compare matching models and scenarios within each backend.

The final measurements use the production process allocator and preserve the
node's automatic GPU power policy. Earlier measurements and diagnostics remain
available so the unsuccessful experiments and timing swings are visible.

## Final comparisons

- `final-vulkan-auto.json.gz`: three models, three independent server rounds,
  three scenarios and three measured requests per scenario (162 samples).
  Every round passes the 2% decode regression guard and output parity checks.
- `final-rocm-auto.json.gz`: three models, two independent server rounds and
  three scenarios (108 samples), with output parity and a 2% decode guard.
- `tokenizer-final-comparison.json.gz`: 2,400 timed encodings across the three
  vocabularies and four corpora, with token IDs checked on every encoding.
  Binary, model and corpus hashes are included.

## Correctness

- `verification.json`: optimization-mode checks, Metal compilation and final
  cleanup/preservation checks.

- `moe-cols-validation.txt`: five GPU Q5_1 shapes checked against a scalar CPU reference.
- `llama-tokenizer-parity.txt`: 60 token-sequence comparisons against llama.cpp.
- `api-integration.json`: full Qwen and Gemma API/SDK suites.
- `extra-model-smoke.json.gz`: the other five GGUFs compared with 0.15.2.
- `stress-api.json`: 240 measured requests per build from six concurrent clients,
  with response parity and resident-memory samples.
- `download-https.json`: native TLS, redirects, checksum and installation checks.
- `download-fixture.json`: real HTTP transfers, cache reuse and failure cleanup.
- `tokenizer-corpus/` and `tokenizer-corpus.json`: exact benchmark inputs and hashes.
- `baseline-tokenize-benchmark.zig.txt`: benchmark source compiled against the
  frozen 0.15.2 tokenizer. The migrated benchmark is `src/bench_tokenize.zig`.

## Earlier measurements and diagnostics

- `stream-chat-screen.json.gz`, `rocm-comparison.json.gz`, `rocm-gemma.json.gz` and
  `tokenizer-comparison.json.gz`: initial comparisons with the process allocator.
- `final-gemma-stream.json.gz` and `gemma-peak.json.gz`: comparisons under `high` and
  `profile_peak`, respectively. Both builds experienced power-state variability.
- `power-profile-qwen-stream.json.gz`: the repeated comparison that exposed slow
  independent server starts. This is diagnostic evidence, not a headline result.
- `allocator-restarts.json.gz`: 38 fresh server starts across two Qwen models.
  `baseline` is 0.15.2, `candidate` is the 0.17 process allocator, and
  `candidate-smp` is the experimental concurrent allocator. Slow starts occurred
  in all three variants; medians were effectively identical.
- `power-profile-recovery.json.gz`: GPU telemetry and engine timings from a slow
  process. Returning to automatic power management recovered full speed without
  changing the process, model or executable.
- `tokenizer-smp-comparison.json.gz`: the concurrent-allocator CPU experiment.
  It preserved token IDs but encoded the Qwen corpora more slowly than the
  process allocator, so the process allocator was retained.

Measurement dumps and detailed power telemetry are losslessly compressed JSON.
Inspect a dump with `gzip -dc final-vulkan-auto.json.gz`. `archives.json` records
the SHA256 of each uncompressed file so its original bytes can be verified.
