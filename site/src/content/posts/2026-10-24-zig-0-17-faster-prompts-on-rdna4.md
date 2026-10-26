---
title: "Zig 0.17 in ZINC: faster prompts on RDNA4"
seoTitle: "ZINC on Zig 0.17: 38% Lower Gemma First-Output Latency on RDNA4"
date: "2026-10-24"
tags:
  - zinc
  - zig
  - rdna4
  - performance
  - tokenization
  - gemma
  - qwen3-6
  - llm-inference
keywords:
  - Zig 0.17 migration
  - ZINC RDNA4 performance
  - BPE tokenizer optimization
  - Gemma 4 time to first token
  - Radeon AI PRO R9700 inference
excerpt: "ZINC on Zig 0.17 cuts Gemma's long-prompt wait by 37.7% versus 0.15.2, keeps RDNA decode steady, and checks eight GGUF tokenizers against llama.cpp."
seoDescription: "ZINC's Zig 0.17 migration cuts long-prompt first-output latency by 2.9% on Qwen 9B, 5.2% on Qwen 35B-A3B and 37.7% on Gemma, with RDNA4 decode steady."
faqs:
  - question: "Did the Zig 0.17 compiler make GPU decode faster?"
    answer: "GPU decode stayed essentially level. The improvements came from host changes: the production allocator, tokenizer storage, batched network writes and faster BPE merging. The complete migration's 37.7% Gemma gain compares with Zig 0.15.2; the chart's 34% gain isolates the BPE follow-up against the first optimized 0.17 build."
  - question: "What hardware and workload produced the 37.7% result?"
    answer: "Gemma 4 26B-A4B at Q4_K_M on a Radeon AI PRO R9700, using Vulkan with Mesa 25.0.7 and a Ryzen 7 9800X3D. The long prompt contains 2,254 tokens. Each build has six measured requests across two independent server rounds, with greedy sampling, a 4,096-token context and automatic GPU power management."
  - question: "How was tokenizer correctness checked?"
    answer: "Complete token arrays match llama.cpp in 112 model/corpus combinations across eight GGUFs. Unit tests compare BPE merges with the original full scan on randomized rules, ties, overlaps, large keys and allocation failures. Another 146 cases exercise the reference GPT-4o Unicode splitter, alongside special-token promotion and demotion tests."
draft: false
---

ZINC now builds with Zig 0.17.0. Direct comparisons with the frozen 0.15.2 build on our Radeon AI PRO R9700 node reduce long-prompt time to first visible output by **2.9% on Qwen 9B, 5.2% on Qwen 35B-A3B and 37.7% on Gemma 26B-A4B**. GPU decode stays essentially level: the worst per-server-round median change is −0.14%.

The largest gain comes from CPU tokenization. The chart below isolates the BPE follow-up against the already optimized 0.17 build. In that three-round experiment, Gemma's long-prompt first output falls from **1,811.9 ms to 1,195.9 ms**, or 34% less waiting. The complete request, generating the same 128 tokens, falls from 2,983.3 ms to 2,368.8 ms, a 20.6% reduction.

The complete migration and the BPE experiment have different baselines. A compiler upgrade, an allocator change and an algorithm change are separate experiments.

<figure class="diagram-card diagram-wide diagram-dual">
  <img class="diagram-visual" src="/blog/2026-10-24-zig-017-bpe-gemma-ttft.svg" alt="Gemma 4 first-visible-output latency on an R9700. The Zig 0.17 baseline and BPE follow-up take 111.1 and 111.3 milliseconds on the short prompt, 465.1 and 397.4 milliseconds on the 782-token prompt, and 1811.9 and 1195.9 milliseconds on the 2254-token prompt. Long-prompt latency falls 34 percent. Bars show request medians; dots show three independent server-round medians." loading="lazy" />
  <div class="diagram-mobile-alt" role="group" aria-label="Gemma first-output latency in milliseconds">GEMMA 4 · FIRST VISIBLE OUTPUT

44 prompt tokens
  Baseline    111.1 ms
  BPE         111.3 ms · level

782 prompt tokens
  Baseline    465.1 ms
  BPE         397.4 ms · −14.6%

2,254 prompt tokens
  Baseline  1,811.9 ms
  BPE       1,195.9 ms · −34.0%

3 rounds, 9 requests per build.
Both builds use Zig 0.17.0.</div>
  <figcaption>Both builds use Zig 0.17.0 and identical GPU shaders. Each bar summarizes nine measured requests across three independent server rounds. The short prompt stays level; the larger gain appears as the text chunk grows.</figcaption>
</figure>

## Two baselines, two stages

The first stage started from unmodified ZINC main at `29bc350a`, built with Zig 0.15.2. We moved entrypoints to process-provided I/O, ported synchronization and build APIs, and translated the C bindings into explicit modules. The compiler wrapper verifies the pinned release archive, and the Nix package uses the same version.

The compiler-only port was neutral on GPU decode. The host changes made the difference: using the process allocator, storing initial tokenizer symbols together, and writing JSON and SSE frames in batches. The [migration report](/zinc/docs/zig-0-17-migration/) records that first stage.

The second stage froze that optimized 0.17 build at `9bb61b12`. The BPE measurements compare against it, using the same compiler and production allocator on both sides.

After the tokenizer corrections, we also compared the complete port directly with the original 0.15.2 executable. Two alternating server rounds, three models and three scenarios give 108 measured requests. The long-prompt results are:

<div style="max-width:100%; overflow-x:auto;" role="region" aria-label="Complete migration long-prompt comparison" tabindex="0">

| Model | First output, 0.15.2 | First output, final 0.17 | Reduction |
| --- | ---: | ---: | ---: |
| Qwen 3.5 9B | 1,399.7 ms | 1,358.6 ms | 2.9% |
| Qwen 3.6 35B-A3B | 1,355.7 ms | 1,285.8 ms | 5.2% |
| Gemma 4 26B-A4B | 1,917.1 ms | 1,194.2 ms | 37.7% |

</div>

Every response and usage check matches, and every per-round decode check passes the 2% guard. These final migration results remain separate from the chart's comparison between two 0.17 builds.

## Replacing repeated BPE scans

BPE begins with a sequence of small symbols. At each step, the tokenizer finds the adjacent pair with the best merge rank, joins it, and repeats until no eligible pair remains.

Our implementation searched every adjacent pair again after every merge. It also allocated a new string for each joined result. A merge changes only its immediate neighbors, so most of those repeated lookups rediscovered information that was still valid.

This cost differs by tokenizer. Qwen generally splits text into short word-sized chunks before BPE. Gemma's splitter groups runs of non-newline text, which can make one paragraph a much larger merge problem. The same full scan becomes progressively more expensive as that chunk grows.

The new implementation keeps a priority queue for chunks longer than 32 initial symbols. It checks every initial adjacency once, then rechecks at most the two pairs affected by each merge. Original symbol positions preserve the leftmost choice when ranks tie. Stale queue entries are discarded when their symbols have grown or stopped being adjacent. Short chunks keep the simple scan.

Merged symbols also reuse the contiguous byte storage created at the start of encoding. Extending a slice replaces allocating and copying another string. The bytes remain alive until the final pieces have been converted to token IDs.

## Gemma first-output latency

For Gemma 4 26B-A4B, the Vulkan measurements are:

<div style="max-width:100%; overflow-x:auto;" role="region" aria-label="Gemma first-output latency comparison" tabindex="0">

| Prompt | First output, 0.17 baseline | First output, BPE follow-up | Change |
| --- | ---: | ---: | ---: |
| 44 tokens | 111.1 ms | 111.3 ms | Essentially level |
| 782 tokens | 465.1 ms | 397.4 ms | 14.6% less time |
| 2,254 tokens | 1,811.9 ms | 1,195.9 ms | 34.0% less time |

</div>

The long-prompt reduction held in every server round: 34.0%, 33.9% and 34.0%. The corresponding decode medians were 109.15 and 109.13 tokens per second. The improvement removes work before generation; it leaves the GPU's token loop steady.

Qwen's end-to-end timings stayed essentially level in this matrix. Its CPU encoder still improved, but encoding was a much smaller share of the request. Across the code, mixed-text, prose and multilingual corpora, the additional encoding speedups were roughly 1.2–1.8× for Qwen and 1.7–9.7× for Gemma. Those are encoding measurements, with vocabulary loading outside the timed region.

The prose encoding corpus puts a newline after each sentence. The request benchmark repeats the passage on one long line before its final instruction. For Gemma, that creates a much larger BPE chunk; prompt token count alone does not describe the CPU cost.

## Faster merging still has to produce the right IDs

We compare complete token sequences against an independent full-scan implementation in unit tests, including randomized rules, tied ranks and overlapping candidate pairs. Allocation-failure tests exercise queue growth, large keys, fallback rank tables and complete prompt encoding.

Real vocabularies exposed another issue. Gemma has ordinary vocabulary entries that look like special markers, including `<>`. Treating every matching angle-bracket string as special prematurely split the surrounding BPE text. For the input ` <>&`, ZINC emitted the pieces `▁`, `<>`, `&`; llama.cpp emitted `▁<`, `>&`.

The fix normalizes token types once when loading the vocabulary. Gemma's `<eos>` is labeled ordinary in this GGUF's token-type table, while the reference promotes it to an end-of-generation marker. The opposite exception matters too: when `<|tool_response>` exists, llama.cpp makes `</s>` ordinary even if the GGUF labels it control. Removing it from the stopping set alone leaves tokenization wrong. Regression tests cover both overrides, along with message-closing markers that remain special without ending generation.

The wider corpus found the same ordinary-marker problem in Muse Glimmer, so the token-type filter now applies to every pretokenizer. GPT-OSS also needed its `gpt-4o` metadata mapped to the shared GPT-4o/Llama4 splitter, with Unicode letter, number and whitespace handling. Isolated Gemma markers exposed one more shortcut: a whole vocabulary match skipped BPE. The reference permits that shortcut only for newline runs, so ordinary chunks now follow their merge rules.

The complete comparison matches llama.cpp arrays in 112 model/corpus combinations across eight GGUFs. Another 146 chunk cases, including fixed-seed Unicode combinations, check the splitter directly against the reference regex.

The benchmark runner records these intentional corrections explicitly. Its default mode requires matching token arrays between builds. With a reference tokenizer supplied, the candidate must match llama.cpp, and an incorrect baseline is recorded as a baseline mismatch. Negative controls check that a known incorrect tokenizer is rejected.

Even the reference command needed care. `llama-tokenize` interprets backslash escapes by default. We pass `--no-escape` for file-based comparisons so both programs tokenize the same bytes.

## Measuring the host and the GPU together

The main Vulkan comparison contains 162 measured requests across three models. Each build reuses a server for its scenarios, warms each case twice, and measures three requests per case in each of three independent server rounds. Build order alternates between rounds. Model files, prompts, shader trees, context budgets and greedy sampling settings match.

A separate 54-request run after the final tokenizer corrections passes the same output and throughput checks, with a 37.0% reduction on the long Gemma prompt. Those samples remain separate from the chart's three-round experiment.

The runner checks response content and usage, then applies a 2% decode-regression guard to each matching server round. The worst Vulkan decode delta was −0.23%. Combining every request into one median could conceal a slow restart, so the guard checks the individual rounds too.

We leave the node's automatic GPU power policy in place. Earlier locked-profile experiments produced intermittent slow starts in both the 0.15.2 baseline and the ported build; those failed measurements and recovery diagnostics remain in the migration evidence.

The CPU suite uses the production allocator, parses each vocabulary once per benchmark process, warms encoding, and times 50 complete encodings. Every timed sequence is checked against its full reference. Valid-text cases also compare complete IDs with llama.cpp; invalid UTF-8 is a separate preservation check. Native tests cover all four Zig optimization modes, and backend builds, API tests and additional GGUFs extend the checks beyond the headline model.

The [BPE validation report](/zinc/docs/zig-0-17-bpe/) contains the exact baselines, per-round results, corpus hashes and compressed measurements. To reproduce the encoding comparison on the machine holding the models:

```sh
./scripts/zig.sh build tokenize-tool tokenize-bench -Doptimize=ReleaseFast

python3 tools/compare_tokenizers.py \
  --baseline "$ZINC_BASELINE" --candidate "$PWD" \
  --model-root "$ZINC_MODELS" --models "$GGUF_MODEL_NAMES" \
  --corpus-root docs/validation/zig-0.17/tokenizer-corpus \
  --reference-tokenizer "$LLAMA_TOKENIZE" \
  --runs 50 --rounds 3 --output tokenizer-comparison.json
```
