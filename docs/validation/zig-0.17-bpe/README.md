# BPE follow-up validation

The baseline is `9bb61b12c9e95c39a1d60ae64082f4a4127e4003`, the first
validated Zig 0.17 migration. The three-round Vulkan performance candidate is
`6c411d034e551121e0772ad4b1beeab79445de9e`. The final runtime source is
`386e6d4702d763089ede5dbd0fe2d1f562b160ef`, including reference token-type
overrides, Unicode GPT-4o splitting and Gemma's newline-only vocabulary
shortcut. Intermediate `565d3b06` and `0cb98ae3` results are retained separately.
The BPE comparisons use Zig 0.17.0, ReleaseFast and the production process
allocator on both sides. Separate final comparisons against the frozen
0.15.2 executables measure the complete migration.

Model workloads run on the designated RDNA host. Inference comparisons use
the existing automatic GPU power policy, greedy sampling, a 4096-token context,
one inference slot, reusable servers and alternating build order. Every server
is stopped before the next starts. CPU encoding times exclude GGUF vocabulary
loading and include complete prompt encoding and its allocations.

`archives.json` records the SHA256 and byte length of each uncompressed JSON
and its deterministic gzip archive. Read an archive with Python's `gzip.open`
or `gzip -dc`; generated measurements stay compressed to keep source review
readable.

The original four corpora remain in `../zig-0.17/tokenizer-corpus/`.
`corpus/` adds tiny prompts, whitespace, special markers, long words, long
lines, Unicode boundaries and intentionally invalid UTF-8. Preserve their
exact bytes, including trailing whitespace, CRLF, NUL and invalid sequences.
The invalid UTF-8 case checks preservation of existing behavior separately
from valid-text llama.cpp comparisons. `reference-corpus/` adds end-marker
overrides, isolated markers and 146 reference-regex text combinations.

`tools/compare_tokenizers.py` normally requires complete token-array equality
between builds. With `--reference-tokenizer`, the candidate must instead match
llama.cpp exactly, and any baseline mismatch is explicitly recorded. This is
needed for corrected Gemma, Muse Glimmer and GPT-OSS tokenization; the record
identifies every baseline mismatch. Timed output
hashes must remain consistent within each build; unchanged token arrays must
also have matching timed hashes across builds. Negative controls verify that
strict comparison rejects changed IDs and reference comparison rejects the
known incorrect baseline.

The reference CLI receives `--no-escape` so it tokenizes the actual file bytes.
Without this flag, llama.cpp interprets backslash escapes in the mixed corpus.

| Artifact | Coverage |
| --- | --- |
| `vulkan-final.json.gz` | Three models, three server rounds, 162 measured streaming requests; chart source |
| `final-vulkan-zig17.json.gz` | Final source against frozen 0.17, 54 streaming requests; all output and per-round decode checks pass |
| `final-rocm-zig17.json.gz` | Final source against frozen 0.17, 54 raw-completion requests; all output and per-round decode checks pass |
| `final-vulkan-zig15.json.gz` | Complete migration against frozen 0.15.2, 108 streaming requests across two server rounds; all output and decode checks pass |
| `final-rocm-zig15.json.gz` | Complete migration against frozen 0.15.2, 54 raw-completion requests; all output and decode checks pass |
| `final-extra-model-smoke.json.gz` | Final source, five additional GGUFs, 20 generation comparisons with output and decode checks |
| `final-tokenizer-main.json.gz` | Final source, four original corpora, three models, three rounds, 3,600 timed encodings |
| `final-tokenizer-edges.json.gz` | Final source, seven valid edge corpora, three models, three rounds, 6,300 timed encodings |
| `final-tokenizer-invalid.json.gz` | Final source, invalid bytes across all eight GGUFs, 160 timed encodings, baseline arrays preserved |
| `vulkan-eos-confirmation.json.gz` | Earlier `565d3b06` end-marker checkpoint, 54 measured streaming requests |
| `rocm-final.json.gz` | Earlier `565d3b06` checkpoint, 108 measured raw-completion requests |
| `extra-model-smoke.json.gz` | Earlier `565d3b06` checkpoint, five additional GGUF generation comparisons |
| `tokenizer-final.json.gz` | Earlier `565d3b06` checkpoint: four original corpora, three models, three rounds, 3,600 timed encodings |
| `tokenizer-edges.json.gz` | Earlier `565d3b06` checkpoint: seven valid edge corpora, three models, three rounds, 6,300 timed encodings |
| `tokenizer-invalid-utf8.json.gz` | Earlier `565d3b06` checkpoint: invalid bytes, three models, three rounds, 900 timed encodings |
| `tokenizer-all-models-reference.json.gz` | Final source: 112 complete-array reference checks, eight GGUFs, fourteen valid corpora, 2,240 timed encodings |
| `tokenizer-reference-before-newline-shortcut.json.gz` | Earlier `0cb98ae3` checkpoint before the isolated Gemma shortcut correction, 104 valid-text reference checks |
| `tokenizer-markers-before-universal-mask.json.gz` | Retained incomplete comparison that exposed the Muse ordinary-marker mismatch |
| `marker-diagnostic-before-universal-mask.json.gz` | Exact per-line candidate/reference IDs for the Muse and GPT-OSS mismatches |
| `gemma-whole-chunk-before-fix.json.gz` | Observed isolated `</s>` mismatch before the newline-only shortcut correction, with verified source and binary fingerprints |
| `all-models-api.json.gz` | Final source: 24 complete-array and live-API prompt-count checks, 48 repeated generation requests across eight GGUFs |
| `gemma-marker-api.json.gz` | Final source: five live-API reference cases, including explicit BOS policy and isolated `</s>` |
| `benchmark-negative-controls.json.gz` | Both strict comparison and reference comparison reject a known incorrect tokenizer |
| `api-integration.json.gz` | Final source: Qwen and Gemma API/SDK suites, thirteen checks each |
| `stress-gemma.json.gz` | Six concurrent clients, one request slot, 192 measured long-prompt requests; response parity and resident-memory checks |
| `final-summary.json.gz` | Derived medians and speedups with input artifact hashes; separate comparison baselines remain separate |
| `final-verification.json.gz` | Test counts, source/binary fingerprints, measurement gates, process cleanup and preserved host state |
| `blog-layout.json.gz` | Desktop, 390px and 320px layouts, chart visibility, overflow and report-link checks |
| `eog-reference-preflight.json.gz` | 28 exact-array reference checks before the final measured runs |
| `tokenizer-screen.json.gz` | Earlier heap-only prototype before the Gemma marker correction; all baseline arrays preserved |
| `llama-tokenizer-parity.txt` | Final exact-byte reference suite: 87 cases pass |
| `reference-before-inferred-eog-fix.txt` | Retained failing reference check that motivated the final end-marker fix |
