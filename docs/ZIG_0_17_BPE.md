# Faster BPE encoding after the Zig 0.17 migration

This follow-up compares against `9bb61b12`, the already validated Zig 0.17
migration. Both builds use Zig 0.17.0, ReleaseFast and the production process
allocator. The [initial migration report](https://github.com/zolotukhin/zinc/blob/main/docs/ZIG_0_17_MIGRATION.md) separately
compares Zig 0.15.2 with the first 0.17 build.

The main three-round Vulkan matrix measures the BPE implementation at
`6c411d03`. Earlier `565d3b06` confirmation data remain separate from the final
runtime source at `386e6d47`. Binary and source fingerprints are recorded
with the measurements in [the validation directory](https://github.com/zolotukhin/zinc/tree/main/docs/validation/zig-0.17-bpe).

## Change

Ranked BPE merges reuse the initial contiguous symbol bytes. Short chunks keep
the full scan; chunks longer than 32 symbols use a priority queue and linked
neighbors. Only the two adjacencies affected by a merge are reconsidered.
Original positions preserve leftmost rank ties, and adjacency and length
checks reject stale candidates. No GPU shader, quantization, expert-count or
sampling change is involved.

The reference comparison also exposed ordinary Gemma tokens such as `<>`
being interpreted as special markers. Token types are normalized once at
vocabulary load, including declared BOS/EOS/EOT IDs and recognized end markers.
`<eos>` remains special even when the primary declared EOS differs. Conversely,
when `<|tool_response>` exists, `</s>` becomes ordinary even if the GGUF labels
it control. Message-closing `<|end|>` remains special after removal from the
end-of-generation set. Initialization tests cover these distinct cases.

The token-type filter applies to all pretokenizers, including Muse Glimmer.
GPT-OSS's `gpt-4o` metadata now selects the same splitter as `llama4`, following
the reference mapping. That splitter uses Unicode letter, number and whitespace
classes, with the reference's two overlapping letter alternatives and their
priority preserved. Its 146 reference-regex fixtures include 128 fixed-seed
combinations. `tools/generate_gpt4o_chunk_fixtures.mjs` reproduces the fixtures
and the corresponding full-input corpus.

Gemma's direct-vocabulary shortcut is restricted to newline runs, matching
the reference. Other ordinary chunks must follow their merge rules even
when the whole text exists in the vocabulary. Isolated `</s>` needs this
distinction as well as the token-type override.

The final runtime source at `386e6d47` matches complete llama.cpp arrays in all
112 valid-text model/corpus combinations: eight GGUFs and fourteen corpora.
The comparison explicitly records corrected baseline arrays in four cases
for each Gemma, four for Muse Glimmer and nine for GPT-OSS. These are
correctness changes, so their encoding speedups are not pure comparisons of
identical output arrays.

## Vulkan request latency

Hardware: Ryzen 7 9800X3D, Radeon AI PRO R9700 (GFX1201), Mesa 25.0.7.
The comparison preserves automatic power management and uses identical
GGUFs, prompts, shader trees, greedy sampling, a 4096-token context and one
inference slot. Each model has three independent server rounds, alternating
build order, two warmups per case and three measured requests per case and
build. The matrix contains 162 measured requests.

Gemma 4 26B-A4B Q4_K_M medians:

| Prompt tokens | First output, baseline | First output, candidate | Reduction |
| ---: | ---: | ---: | ---: |
| 44 | 111.1 ms | 111.3 ms | Essentially level |
| 782 | 465.1 ms | 397.4 ms | 14.6% |
| 2254 | 1811.9 ms | 1195.9 ms | 34.0% |

The long case generates 128 tokens. Total request latency falls from
2983.3 ms to 2368.8 ms, a 20.6% reduction. Long-case decode remains level:
109.15 versus 109.13 tokens per second.

| Gemma first-output reduction | Round 1 | Round 2 | Round 3 |
| --- | ---: | ---: | ---: |
| 782-token prompt | 14.79% | 14.74% | 14.44% |
| 2254-token prompt | 34.02% | 33.90% | 33.97% |

Qwen's request timings remain essentially level. Across all models and
scenarios, output hashes match. The worst per-server-round median decode
change is −0.234%, within the 2% guard. Baseline and candidate SPIR-V trees
have the same SHA256:
`8c69d185477637fa1e0ec8176937e45cd6e82575884bcba6f96cfdd5fad5b6cc`.

A separate 54-request confirmation uses the final runtime at `386e6d47`.
All output and per-round decode checks pass; the worst decode change is
−0.192%. Gemma's medium and long first-output reductions are 15.8% and 37.0%,
respectively. The chart retains the original three-round experiment; samples
from different binaries remain separate.

The final ROCm confirmation adds 54 raw-completion requests, with output
parity and every per-round decode check passing. The worst decode change is
−0.089%. Long-prompt Gemma total latency falls 26.2%. ROCm's raw-completion
results are kept separate from Vulkan's streaming chat measurements.

## Complete migration versus Zig 0.15.2

The final runtime is also compared directly with the frozen `29bc350a`
executable built with Zig 0.15.2. Vulkan uses two alternating server rounds,
two warmups per case and three measured requests per build, case and round:
108 measured requests in total. All response and usage checks match, and
every per-round decode check passes. The worst median decode change is
−0.141%.

Long-prompt first-visible-output medians, across six requests per build:

| Model | Zig 0.15.2 | Final Zig 0.17 | Reduction |
| --- | ---: | ---: | ---: |
| Qwen 3.5 9B | 1399.7 ms | 1358.6 ms | 2.9% |
| Qwen 3.6 35B-A3B | 1355.7 ms | 1285.8 ms | 5.2% |
| Gemma 4 26B-A4B | 1917.1 ms | 1194.2 ms | 37.7% |

For Gemma, the complete 128-token request is 23.3% shorter. Long-case decode
remains level at 109.09 versus 109.06 tokens per second. These measurements
compare the complete migration with 0.15.2; the earlier 34% figure isolates
the BPE follow-up between two 0.17 builds.

ROCm adds 54 matched raw-completion requests against 0.15.2. Every output
and decode check passes; the worst per-round decode change is −0.137%.
Long-request total latency falls 1.9% on Qwen 9B, 2.4% on Qwen 35B-A3B and
27.5% on Gemma. Five further Vulkan GGUF comparisons add 20 requests, with
output parity and a worst per-round decode change of −0.481% against the
frozen 0.17 build. The final performance matrices contain 290 requests.

## CPU encoding

The benchmark loads each GGUF vocabulary outside the timed region, warms
encoding three times and times 50 complete prompt encodings per corpus,
build and round. Each timed array is checked against that build's full
reference. Three alternating rounds give 3600 timed encodings for the
original four corpora.

Final-build encoding speedups over the frozen 0.17 baseline, using medians
across those three rounds:

| Model | Code | Mixed text | Prose | Multilingual |
| --- | ---: | ---: | ---: | ---: |
| Qwen 3.5 9B | 1.20× | 1.47× | 1.20× | 1.83× |
| Qwen 3.6 35B-A3B | 1.18× | 1.47× | 1.19× | 1.83× |
| Gemma 4 26B-A4B | 1.67× | 3.03× | 9.69× | 3.72× |

Seven valid edge corpora add 6,300 timed encodings, the eight-model reference
comparison adds 2,240, and invalid UTF-8 adds 160. The final CPU suites
contain 12,300 complete timed encodings. All eight invalid-UTF-8 cases
preserve the baseline arrays.
Only Gemma's mixed-text case changes among the original four corpora:
11,929 tokens become 11,905, matching the reference. Other intentional
corrections in the extended corpus are identified in the comparison record.

Two small edge cases cost more: Qwen 35B's special-marker corpus goes from
2.15 to 2.81 microseconds, and Gemma's one-byte prompt from 0.05 to 0.07
microseconds. The archived edge matrix includes these cases alongside the
long-word, long-line, whitespace and Unicode measurements.

The prose corpus places a newline after each sentence. The long request
scenario repeats the passage on one line before its final instruction.
These create different Gemma BPE chunk sizes despite similar prompt lengths.

The mixed corpus intentionally includes ordinary angle-bracket text. Its
Gemma token sequence changes to match llama.cpp after the marker fix;
the benchmark records that baseline mismatch explicitly. Supplying
`--reference-tokenizer` requires complete candidate/reference array equality.
The default mode continues to require complete equality between builds.

Valid-text edge cases add tiny prompts, long words and lines, whitespace,
Unicode boundaries and special markers. Invalid UTF-8 is checked separately
for preservation of existing behavior. Input hashes and complete-array
reference hashes are included in the artifacts.

## Correctness checks

- Unit comparisons with an independent copy of the original full scan,
  including 128 randomized inputs, overlapping rules and tied ranks.
- Large merge keys and allocation-failure coverage for queue growth, rank
  tables and complete encoding; forced resize failure keeps failure indices
  deterministic.
- Regression cases for ordinary Gemma angle brackets and declared and
  inferred special IDs whose token-type flags are ordinary.
- Exact-byte llama.cpp comparisons; `--no-escape` avoids the reference CLI
  interpreting backslash escapes before tokenization.
- All eight GGUFs checked on fourteen valid-text corpora, with 112 complete
  reference arrays matching. Live API counts and repeated generation are
  checked on all eight models; Qwen and Gemma also pass the API/SDK suite.
- Unicode GPT-4o chunking checked on 146 reference-regex fixtures, including
  128 fixed-seed combinations, and both `gpt-4o` and `llama4` GGUF selectors.
- Negative benchmark controls reject changed IDs in strict mode and reject
  the known incorrect baseline in reference mode.
- Linux runs 712 Zig cases and 652 Bun tests in each optimization mode,
  including ten standalone tools and five CUDA host entrypoints compiled.
  macOS passes 821 GPU-free Zig cases in Debug; CI compiles Metal binaries
  and tests in all four modes. ZINC_RT's CPU tests and standalone build are
  included. CUDA and Metal GPU execution are outside this RDNA campaign.
- A six-client, single-slot Gemma stress run warms 24 requests per build,
  then measures 96 per build. Responses match across both builds. After
  48 measured requests, the resident-memory range is 655,360 bytes for the
  baseline and zero for the candidate. All test servers are stopped; the
  original node binary, system compiler and automatic GPU power policy are
  preserved.

## Reproduce

Run model commands on the designated model host. Preserve the baseline in a
separate directory and build the candidate with:

```sh
./scripts/zig.sh build test check-cuda -Doptimize=ReleaseSafe
./scripts/zig.sh build -Doptimize=ReleaseFast
./scripts/zig.sh build tokenize-tool tokenize-bench -Doptimize=ReleaseFast

python3 tools/compare_builds.py \
  --baseline "$ZINC_BASELINE" --candidate "$PWD" \
  --model-root "$ZINC_MODELS" --device "$ZINC_DEVICE" \
  --models "$GGUF_MODEL_NAMES" --stream \
  --runs 3 --warmup 2 --rounds 3 --max-decode-regression-pct 2 \
  --output comparison.json

python3 tools/compare_tokenizers.py \
  --baseline "$ZINC_BASELINE" --candidate "$PWD" \
  --model-root "$ZINC_MODELS" --models "$GGUF_MODEL_NAMES" \
  --corpus-root docs/validation/zig-0.17/tokenizer-corpus \
  --reference-tokenizer "$LLAMA_TOKENIZE" \
  --runs 50 --rounds 3 --output tokenizer-comparison.json
```

For the complete-migration Vulkan comparison, set `ZINC_BASELINE_015` to
the frozen 0.15.2 build and run the same server harness with `--rounds 2`:

```sh
python3 tools/compare_builds.py \
  --baseline "$ZINC_BASELINE_015" --candidate "$PWD" \
  --model-root "$ZINC_MODELS" --device "$ZINC_DEVICE" \
  --models "$GGUF_MODEL_NAMES" --stream \
  --runs 3 --warmup 2 --rounds 2 --max-decode-regression-pct 2 \
  --output zig15-comparison.json
```

The blog's standalone SVG is generated by `tools/plot_zig17_bpe.py` from
`vulkan-final.json.gz`. Matplotlib is an optional development dependency for
regenerating that plot; it is not an engine runtime dependency.
