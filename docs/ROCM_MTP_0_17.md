# Faster Qwen 3.8 27B MTP on Zig 0.17

The ROCm MTP follow-up improves Qwen 3.8 27B decode from **71.17 to
77.04 tok/s (+8.25%)** on the long coding workload, with smaller gains on the
other three workloads. These are matched reusable-server measurements on a
Radeon AI PRO R9700. Both builds use Zig 0.17.0.

The baseline is [`b81f9946`](https://github.com/zolotukhin/zinc/commit/b81f9946f021d62481e41384e731f236ff2e407a),
the published implementation before this follow-up. The measured candidate is
[`cad53271`](https://github.com/zolotukhin/zinc/commit/cad5327110a8895781bf0318e98d441548747596).
This comparison measures the MTP changes after the compiler migration.

## Results

Each cell is the median of nine measured requests across three independent
server rounds. The output caps are reached in all four cases.

| Workload | Output tokens | Baseline decode | Candidate decode | Improvement |
| --- | ---: | ---: | ---: | ---: |
| Quick Chat (`core`) | 96 | 61.08 tok/s | 61.94 tok/s | 1.41% |
| Coding Review (`context-medium`) | 160 | 80.71 tok/s | 81.50 tok/s | 0.98% |
| Incident Context (`context-long`) | 128 | 72.17 tok/s | 73.00 tok/s | 1.15% |
| Long Coding Draft (`decode-extended`) | 256 | 71.17 tok/s | **77.04 tok/s** | **8.25%** |

All twelve comparisons of per-round medians improve. The long coding request's
median total HTTP latency falls from 3,736.39 to 3,462.63 ms, a **7.33%** reduction.
Its draft acceptance rises from 58.3% to 66.9%. Median time to first output is
essentially unchanged at 139.48 versus 139.56 ms; the improvement comes during
generation.

The smaller gains are repeatable on this machine, but remain sensitive to the
GPU, model and prompt. These measurements cover one GPU and one GGUF. They do
not establish a gain on NVIDIA hardware or replace the separate
[cross-engine benchmark](https://github.com/zolotukhin/zinc/blob/main/docs/ROCM.md#fair-llamacpp-comparison).

## What changed

### One attention setup launch per layer

The speculative verifier processes the seed and up to three draft tokens
together. Its attention setup previously launched six operations: split the
packed query/gate projection, normalize queries, normalize keys, rotate queries,
rotate keys, and write keys/values to the cache.

The existing fused decode kernel now accepts a token row in `grid.y`. Each row
uses its own input offsets, absolute RoPE position and KV-cache offset. MTP
verification reuses that kernel for all rows. Six launches become one in each
of the model's sixteen attention layers: **80 fewer launches per verification
cycle**. The recurrent layers retain their existing state capture and rollback.

This path is selected only during speculative state capture when the fused
attention front end is enabled. Ordinary one-row decode uses row zero. Prefill
and catch-up retain the existing path. `ZINC_MTP_FUSED_ATTN=0` selects the six
original operations for comparison.

### Score 98,304 vocabulary rows when drafting

The ROCm draft-head default increases from 65,536 to **98,304** vocabulary
rows. The extra draft-head weight traffic buys higher acceptance, particularly
on the long coding prompt. The full model continues to verify against its
complete vocabulary; greedy output and sampling precision are unchanged.

Three drafts remain the default. CUDA retains its previous 65,536-row default
until measured on NVIDIA hardware. Use `ZINC_MTP_DRAFT_VOCAB=65536` to compare
the old draft vocabulary, or `0` to score the complete draft vocabulary.

## Measurement conditions

- Radeon AI PRO R9700, `gfx1201`, 32 GiB VRAM; Ryzen 7 9800X3D.
- Linux `7.2.2-070202-generic`, ROCm `7.2.4`, Zig `0.17.0`, ReleaseFast.
- `Qwen3.8-27B-Q4_K_M.gguf`, 17,106,775,008 bytes. SHA256:
  `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`.
- Context 4,096, one request slot, greedy sampling, `top_p=1`, thinking disabled,
  streaming `/v1/chat/completions`, identical prompts and output limits.
- Existing automatic GPU power policy, no clock changes and no profiler during
  the timed matrix. Every server is stopped before the next starts.
- Three rounds, ordered baseline/candidate, candidate/baseline,
  baseline/candidate. Each server receives two warmups and three measured
  requests per workload: **72 measured requests and 48 warmups** in total.

Decode rates come from server phase timings. Total HTTP latency and first-output
latency are recorded separately. Every measured response has the same text,
usage and finish-reason hash as its baseline reference. Streaming omits finish
metadata in this existing ROCm server, so the focused JSON tests below check
that metadata separately.

The system-default compiler was also updated to the latest stable
[official Zig 0.17.0 release](https://ziglang.org/download/index.json), with the
official archive checksum verified. Its compiler binary is byte-identical to
the pinned compiler used by both benchmark builds. The previous compiler is
preserved as a versioned installation. Compiler hashes are in the evidence.

## Validation

The independent HIP validator compares every output bit from the fused kernel
with the six original GPU operations, then checks normalization, rotation and
cache indexing against independent CPU math. All **60 cases** pass: one through
five token rows, head dimensions 128/256, half/full rotary dimensions, and
positions 0/97/4091. The inputs include zero and random values, and untouched KV
guard rows are checked. Running the same validator against the old single-row
kernel fails on the multi-row case, confirming that the test detects the
indexing defect it is intended to catch.

The local Zig suite passes all 821 cases across the main, IR, forward and
host-I/O suites; the final repeat reruns 755 main cases and reuses 66 cached
passes. The Linux ROCm ReleaseFast checks pass 517 Zig cases and the Bun suite,
check five CUDA host entry points, and compile ten standalone tools. The native
ROCm executable builds successfully.

Focused API validation checks **102 responses**: seventeen cases, JSON and SSE,
across baseline MTP, candidate MTP, and baseline ordinary greedy decode. All
text and finish-reason hashes match, and streaming text matches JSON text.
Cases cover output limits 1/2/3/4/5/7, EOS, arithmetic, Chinese, Russian,
Unicode, code, JSON, a stop-parameter request, a 4,096-token context boundary,
and subsequent request resets. One boundary request uses 4,020 prompt tokens
and 76 completion tokens; the following request resets correctly. A separate
24-response smoke run verifies the published validation helper itself.

The generic OpenAI SDK suite stops at an existing ROCm API limitation:
`/health` omits `active_requests` and the other scheduler metadata expected by
that suite. This suite does **not** pass. The minimal ROCm handler also omits
SSE finish metadata and ignores the `stop` parameter in both builds. The
focused tests establish equivalent behavior and clean resets; they do not
claim stop-string support. These API features are outside this MTP change.

## Experiments retained and rejected

Before freezing the final candidate, a one-round screen tested draft vocabulary
sizes 32,768, 49,152, 81,920, 98,304 and 131,072, two drafts, and adaptive
two/three-draft selection. A second screen isolated attention fusion and its
combination with 98,304 rows. Smaller vocabularies lost too much acceptance;
131,072 increased weight traffic without improving acceptance on these prompts.

A separate prototype supported four drafts and five verification rows, including
the additional recurrent-state history. It improved Coding Review but regressed
the other three workloads against the final three-draft candidate. It was
reverted completely.

| One-round screen | Quick Chat | Coding Review | Incident Context | Long Coding Draft |
| --- | ---: | ---: | ---: | ---: |
| Fusion screen: baseline | 61.07 | 80.83 | 72.31 | 71.18 |
| Fusion, 65,536 draft rows | 61.53 | 81.36 | 72.64 | 71.68 |
| Fusion, 98,304 draft rows | 61.87 | 81.47 | 73.02 | 77.07 |
| Four-draft screen: baseline | 61.00 | 81.04 | 72.48 | 71.33 |
| Four drafts, fusion, 98,304 rows | 59.64 | 82.70 | 71.40 | 74.41 |

Values are tok/s, two measured requests per cell after one warmup. These screens
selected the candidate; they are separate from the final three-round result.
The three screens contain 128 measured requests in total. All screen response
hashes match their references. The rejected patch and binary/kernel hashes are
archived for reproducibility.

A diagnostic `rocprofv3` run captured 43,206 dispatches. The fused batched
gate/up projection accounts for 30.68% of recorded kernel duration, batched
Q6_K projection for 19.23%, and batched Q4_K projection for 14.19%. This whole
96-token CLI trace includes initialization and prefill; it is a bottleneck
diagnostic, not the source of the server throughput table.

## Reproduce

Run model and HIP validation commands on an authorized GPU host. Set `repo` to
the repository checkout, `campaign` to a fresh directory, `model` to the GGUF,
and `ZINC_GPU` to the ROCm visibility selector. Keep the same GPU power policy
for both builds. The helpers refuse existing output files, check for active
inference processes and stop each server before returning.

```bash
git -C "$repo" worktree add --detach "$campaign/baseline" b81f9946f021d62481e41384e731f236ff2e407a
git -C "$repo" worktree add --detach "$campaign/final" cad5327110a8895781bf0318e98d441548747596
for variant in baseline final; do
  git -C "$campaign/$variant" rev-parse HEAD > "$campaign/$variant/SOURCE_REVISION"
  (cd "$campaign/$variant" && ./scripts/zig.sh build -Dbackend=rocm -Doptimize=ReleaseFast)
done

evidence="$repo/docs/validation/rocm-mtp-2026-10-25"
gzip -dc "$evidence/suite-prompts.json.gz" > "$campaign/prompts.json"
gzip -dc "$evidence/final-variants.json.gz" > "$campaign/variants.json"
gzip -dc "$evidence/api-cases.json.gz" > "$campaign/api-cases.json"
export ROCR_VISIBLE_DEVICES="${ZINC_GPU:?Set the ROCm visibility selector}"

python3 "$repo/tools/bench_mtp.py" \
  --root "$campaign" --variants "$campaign/variants.json" \
  --model "$model" --prompts "$campaign/prompts.json" --device 0 \
  --runs 3 --warmup 2 --rounds 3 --output "$campaign/measured.json"

python3 "$repo/tools/validate_mtp_api.py" \
  --baseline "$campaign/baseline" --candidate "$campaign/final" \
  --model "$model" --cases "$campaign/api-cases.json" --device 0 \
  --output "$campaign/api.json"
```

Build/run instructions for the independent GPU validator are at the top of
`tools/validate_rocm_qwen_frontend.c`. The optional API helper `--sdk` flag runs
the generic suite and reports its failure normally; omit it when reproducing
the focused parity checks on the current minimal ROCm handler.

The [evidence index](https://github.com/zolotukhin/zinc/blob/main/docs/validation/rocm-mtp-2026-10-25/README.md) describes each
archive. It includes the exact prompt fixtures, raw measured rows, response
hashes, source/binary fingerprints, rejected experiment, toolchain checksums,
profile aggregates and derived medians. `archives.json` provides checksums for
both the compressed and uncompressed bytes.
