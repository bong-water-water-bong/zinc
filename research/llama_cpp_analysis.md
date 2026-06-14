# llama.cpp Backend Analysis

Analysis of llama.cpp backends to understand current inference performance and
identify opportunities for a purpose-built engine.

## Metal Qwen3.6 27B Dense-Hybrid Decode Stall - 2026-06-13

Effort 24 is past the point where local Q4/Q6 retunes are useful. The current
Qwen3.6 27B dense-hybrid M4 profile is correct but flat around 14.7 tok/s live
and 15.09 tok/s best-kept. The slowest async decode slot reports about 1.88
GiB/token at 215.7 GiB/s effective bandwidth, with dense FFN bytes dominating:
gate 19.9%, up 19.9%, down 29.0% of the slowest slot. The remaining gap is
not a dispatch-count problem by itself; cycle 81 reduced dense gate/up dispatch
count and stayed flat, while cycles 82-93 reverted local shader, barrier, SSM,
LM-head, and copied-arena variants.

Reference pass:

- llama.cpp `ggml_metal_graph_compute` in
  `ggml/src/ggml-metal/ggml-metal-context.m` keeps graph work queued in a small
  command-buffer set, uses `commandBufferWithUnretainedReferences`, enqueues
  command buffers, and usually waits at graph/token boundaries rather than
  after every op.
- llama.cpp `ggml_metal_op_encode_impl` plus
  `ggml_metal_op_concurrency_check/reset` in
  `ggml/src/ggml-metal/ggml-metal-ops.cpp` tracks source/destination memory
  ranges and resets the encoder only when the next op conflicts with tracked
  ranges. That maps to ZINC's already-kept resource-edge barriers for Qwen27
  dense gate/up and activation joins.
- llama.cpp `ggml_metal_op_mul_mat` keeps single-token decode on mat-vec
  kernels; it does not turn this dense-hybrid decode case into a batched GEMM
  problem.
- llama.cpp `kernel_mul_mv_q4_K_f32_impl` and
  `kernel_mul_mv_q6_K_f32_impl` in `ggml-metal.metal` use row-pair/simdgroup
  mat-vec discipline. ZINC has already adapted the profitable subset via the
  exact Qwen27 Q6 dense-down route and exact Q4 gate/up QK-dual route.
- vLLM `fused_moe` aligns and packs routed expert tokens into grouped expert
  blocks after top-k. Qwen3.6 27B dense-hybrid has no routed MoE, so this does
  not apply to the Effort 24 decode hot path.

Decision for the next Effort 24 source cycle: do not spend another edit on
`dmmv_q4k_qk_dual`, `dmmv_q6k_llama`, SSM projection pairing, dense-down
materialization, final/logits tail, or command-buffer grouping unless the cycle
first brings fresh exact-shape evidence showing that path can beat the 15.09
tok/s promotion band. The useful next work is outside the runtime loop: run the
full public M4 suite on the kept tree, or collect exact Metal shape data that
separates dense Q4 gate/up, Q6 down, and the small SSM buckets before making a
default-on production change.

### Effort 24 revalidation gate

Treat the cycle-80/81 tree as a candidate checkpoint, not a publishable metric,
until one outside-loop run records the full Qwen3.6 27B M4 public suite on the
same workload contract: managed model `qwen36-27b-q4k-m`, raw prompt mode,
128-token generation cap, 322-token effort prompt fingerprint for the
context-long row, and repeated warm medians rather than single screenshots.

The revalidation packet should contain:

- public-suite `core`, `context-medium`, `context-long`, and `decode-extended`
  rows against the same llama.cpp baseline provenance;
- median and sample range for prefill, decode, total latency, and combined
  prompt+decode throughput;
- the slowest async decode slot split for dense Q4 gate/up, dense Q6 down, SSM
  qkv/gate/tail/out, and LM head;
- a note saying whether the post-parser-fix 14.69-14.73 tok/s live band or the
  15.09 tok/s promoted checkpoint is the conservative number.

Only reopen the runtime loop after that packet if it names a default-on change
that removes a measured bucket, or if an exact-shape Metal benchmark shows a
specific dense Q4 gate/up or Q6 down kernel route beating the current kept path.
Without that evidence, the llama.cpp techniques studied here
(`ggml_metal_graph_compute`, `ggml_metal_op_encode_impl`, and
`kernel_mul_mv_q{4,6}_K_f32_impl`) should be considered already represented in
the ZINC candidates, and vLLM `fused_moe` packing remains out of scope for this
dense no-expert decode target.

### Effort 24 post-cycle-100 conclusion

Cycles 82-100 did not produce another keep. The final candidate, a fixed-K5120
fused Q4 gate/up+SwiGLU path for the exact dense FFN shape, built and passed the
unit suite but verified at about 14.49 tok/s and was reverted. That closes the
local-retune phase: the obvious adaptations of llama.cpp row-pair mat-vec,
ZINC's Q4/Q6 fixed-shape selectors, barrier narrowing, dense weight
materialization, and dense activation fusion have all been tried or represented
by kept commits.

The harness gap is now part of the performance problem:

- Resume state keeps the 15.09 tok/s promoted best as a threshold even when the
  resumed live tree repeatedly measures around 14.7 tok/s. The harness should
  revalidate `bestTree` after resume before using it as a hard acceptance band.
- Analysis and enablement work can still be reverted by the throughput gate,
  which makes exact-shape evidence fragile. Labeled `analysis` and
  `enablement` steps need a non-throughput preservation path, or the loop will
  keep rediscovering the same failed kernel families.
- Effort 24 needs first-class Qwen3.6 27B dense-decode plateau guidance in
  `loops/implement_metal.ts`, parallel to the existing Gemma/Qwen35 guidance:
  after repeated reverts, require a public-suite revalidation packet or
  exact-shape Metal benchmark evidence before another default-on runtime edit.

The next credible performance attempt should start with evidence, not code:
run the public M4 suite on the kept tree, run `qwen27b_decode_hot` exact-shape
benchmarks, and compare dense Q4 gate/up and Q6 down production paths against
any proposed replacement. If those numbers do not identify a kernel that beats
the current route, publish the conservative revalidated median and move effort
to a different M4 gap.

## Architecture Detection

RDNA4 (gfx1201) is classified as `AMD_RDNA3` — no RDNA4-specific enum exists.
RDNA3 has **no entry** in `gpu_pipeline_configs`, so `get_subgroup_size()` returns 0,
meaning the pipeline creation falls through to using the driver's default subgroup size (wave64).

## Matmul Path Selection

For single-token decode (n=1):
- `ggml_vk_should_use_mmvq()` checks if MMVQ (integer dot product quantized path) should be used
- For AMD with Q4_K and k >= 2048, MMVQ returns true IF `integer_dot_product` is enabled
- Without it, the FP16 DMMV path is used with `rm_kq=2` (2 rows per workgroup)

The MMVQ path requires `GL_EXT_integer_dot_product` GLSL extension which the default
Ubuntu glslc (shaderc 2023.8) doesn't support. Enabling it requires a newer glslc, but
newer glslc versions produce SPIR-V that RADV handles badly (5x slower).

## Existing Op Fusion

The Vulkan backend already fuses many op sequences:

| Fusion | Pattern | Dispatches Saved |
|--------|---------|-----------------|
| MULTI_ADD | N consecutive ADDs → 1 | ~280 |
| RMS_NORM_MUL | RMS_NORM + MUL → 1 | ~131 |
| TOPK_MOE | SOFTMAX+ARGSORT+GET_ROWS+SUM_ROWS+CLAMP+DIV → 1 | ~360 |
| MUL_MAT_ID_MUL | MUL_MAT_ID + MUL → 1 | ~39 |
| MUL_MAT_ADD | MUL_MAT + ADD → 1 | ~9 |
| GLU | SILU/GELU + MUL (built-in op) | ~80 |

## Compute Graph (Qwen3.5-35B-A3B, single token decode)

- Total nodes: 3728
- Dispatchable ops: 2356
- After fusions: ~1500 dispatches

### Top consecutive pairs (fusion candidates)
```
ADD → ADD:                280x (handled by MULTI_ADD)
RMS_NORM → MUL:           131x (handled by RMS_NORM_MUL)
MUL → MUL_MAT:            121x
MUL_MAT → MUL_MAT:        100x
ADD → RMS_NORM:             80x
GET_ROWS → GET_ROWS:        60x
SCALE → GET_ROWS:           59x
MUL → UNARY:                58x (potential sigmoid_mul fusion)
UNARY → MUL:                51x (potential silu_mul fusion)
```

### Remaining unfused ops
```
MUL_MAT:    343 dispatches (can't fuse matmuls with each other)
UNARY:      170 dispatches (SILU, SIGMOID, SOFTPLUS)
GET_ROWS:   122 dispatches (MoE expert selection)
CPY:        121 dispatches (memory copies)
MUL:        111 dispatches (element-wise multiply)
ADD:        101 dispatches (residual connections)
GLU:         80 dispatches (already fused op)
SCALE:       60 dispatches
L2_NORM:     60 dispatches
```

## Command Buffer Submission

`nodes_per_submit = 100` — submits a command buffer every 100 nodes.

Testing showed:
- `nodes_per_submit=10000` (single submit): same performance as default
- `nodes_per_submit=10` (frequent): -8% regression

The default of 100 is already optimal.

## Why a New Engine

1. **15K+ lines of C++** in ggml-vulkan.cpp — hard to modify and extend
2. **Generic design** supporting 20+ backends — no RDNA4-specific optimization path
3. **No continuous batching** — server layer bolts it on top
4. **No paged attention** — KV cache is contiguous, limits concurrent requests
5. **SPIR-V toolchain fragility** — tightly coupled to specific glslc version
6. **Struct layout sensitivity** — adding a pipeline member can cause 20% regression due to cache effects
