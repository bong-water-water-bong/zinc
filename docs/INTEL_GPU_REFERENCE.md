# Intel Arc B-Series GPU Reference for Inference

Hardware specifications, memory-bandwidth analysis, compiler-visible opcode surface, and ZINC tuning notes for Intel Arc B-series GPUs. This page is the Intel counterpart to the AMD RDNA reference and focuses on Battlemage / Xe2 discrete cards.

Scope note: this reference was checked on 2026-05-17. It covers the currently public B-series desktop and workstation line: Arc B580, Arc B570, Arc Pro B70, Arc Pro B65, Arc Pro B60, and Arc Pro B50. ZINC's Intel Vulkan path is experimental; use this page as an engineering reference, not a claim of feature parity with the RDNA path.

## Reading The Tables

Intel product pages publish the card-level facts: Xe cores, XMX engines, clocks, VRAM, bus width, memory bandwidth, board power, PCIe link, Vulkan version, and PCI device ID. Some derived values below are marked with `*`:

- `Vector engines` = `Xe cores * 8` for Xe2-HPG B-series products.
- Desktop B580/B570 FP32 TFLOPS are derived as `vector_engines * graphics_clock_GHz * 32 FP32 ops/clock`. Intel publishes FP32 values directly for the Arc Pro cards.
- B70/B65 memory speed is derived from Intel's published `608 GB/s` over a `256 bit` bus: `608 * 8 / 256 = 19 Gbps`.
- Bandwidth-per-watt and bandwidth-per-core ratios are planning numbers. They do not include compression, cache hit rate, driver overhead, or quantization unpack cost.

## Current B-Series Line

### Product Specifications

| SKU | Segment | Launch | Xe cores | Render slices | Vector engines | XMX engines | Graphics clock | FP32 TFLOPS | INT8 TOPS | TBP | PCIe | Vulkan | Device ID |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| Arc Pro B70 | Pro AI / workstation | Q1'26 | 32 | 8 | 256 | 256 | 2280 MHz, 2800 MHz max dynamic | 22.94 | 367 | 230 W | PCIe 5.0 x16 | 1.3 | 0xE223 |
| Arc Pro B65 | Pro AI / workstation | Q1'26 | 20 | 5 | 160 | 160 | 2400 MHz | 12.28 | 197 | 200 W | PCIe 5.0 x16 | 1.3 | 0xE222 |
| Arc Pro B60 | Pro AI / workstation | Q2'25 | 20 | 5 | 160 | 160 | 2400 MHz, 2000 MHz LP | 12.28 | 197 | 200 W, 120-200 W LP | PCIe 5.0 x8 | 1.3 | 0xE211 |
| Arc Pro B50 | Pro SFF workstation | Q3'25 | 16 | 4 | 128 | 128 | 1700 MHz, 2600 MHz max dynamic | 10.65 | 170 | 70 W | PCIe 5.0 x8 | 1.4 | 0xE212 |
| Arc B580 | Desktop gaming / creator | Q4'24 | 20 | 5 | 160 | 160 | 2670 MHz | 13.67* | 233 | 190 W | PCIe 4.0 x8 | 1.3 | 0xE20B |
| Arc B570 | Desktop gaming / creator | Q4'24 | 18 | 5 | 144 | 144 | 2500 MHz | 11.52* | 203 | 150 W | PCIe 4.0 x8 | 1.3 | 0xE20C |

### Memory System

| SKU | VRAM | Bus | Memory speed | Bandwidth | GB/Xe core | GB/s/Xe core | GB/s/W |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Arc Pro B70 | 32 GB GDDR6, ECC | 256 bit | 19 Gbps* | 608 GB/s | 1.00 | 19.0 | 2.64 |
| Arc Pro B65 | 32 GB GDDR6 | 256 bit | 19 Gbps* | 608 GB/s | 1.60 | 30.4 | 3.04 |
| Arc Pro B60 | 24 GB GDDR6 | 192 bit | 19 Gbps | 456 GB/s | 1.20 | 22.8 | 2.28 |
| Arc Pro B50 | 16 GB GDDR6, ECC | 128 bit | 14 Gbps | 224 GB/s | 1.00 | 14.0 | 3.20 |
| Arc B580 | 12 GB GDDR6 | 192 bit | 19 Gbps | 456 GB/s | 0.60 | 22.8 | 2.40 |
| Arc B570 | 10 GB GDDR6 | 160 bit | 19 Gbps | 380 GB/s | 0.56 | 21.1 | 2.53 |

The important inference split is not "gaming card versus pro card"; it is memory capacity and sustained bandwidth:

- B70 and B65 are the only public B-series cards with both 32 GB VRAM and 608 GB/s. They are the natural targets for 27B dense and 35B MoE GGUFs.
- B60 has the same high-level compute shape as B580 but doubles the memory capacity to 24 GB. It is useful when the model fits only on the pro card, but it does not increase bandwidth over B580.
- B50 is a low-power 16 GB card. Its 224 GB/s bandwidth is the limiting factor for single-stream decode, not the 16 GB capacity.
- B580 is the strongest consumer B-series small-model card: 12 GB, 456 GB/s, high clock, and 233 INT8 TOPS.
- B570 is capacity-limited first. The 10 GB VRAM pool is tight for 8B-class models once KV cache and temporary buffers are included.

## LLM Inference Analysis

### Decode Roofline

Single-token LLM decode is usually memory-bandwidth bound. The rough upper bound is:

```text
decode_tokens_per_second <= sustained_bandwidth_bytes_per_second / active_weight_bytes_per_token
```

That makes B65 unusual: it has the same 608 GB/s memory bandwidth as B70 but only 20 Xe cores. For single-stream decode on quantized weights, B65 can be much closer to B70 than its FP32 or INT8 TOPS suggest, assuming the kernels keep memory coalesced and maintain enough resident hardware threads.

The ranking for raw decode bandwidth is:

```text
B70 = B65 > B580 = B60 > B570 > B50
608      608     456    456     380    224 GB/s
```

For ZINC's current DMMV-heavy decode path, memory bandwidth matters more than peak matrix TOPS until the model or kernel becomes arithmetic-heavy. Q4_K/Q5_K/Q6_K unpacking can move part of the cost back to ALU, but the large weight stream is still the dominant term.

### Prefill And Batched Work

Prompt prefill is different. Prefill exposes matrix-matrix work and larger attention tiles, so XMX/DPAS can matter once the Vulkan driver exposes cooperative matrix properties that match the shader's data types and tile shapes.

Expected prefill ordering:

| SKU | Prefill expectation | Why |
| --- | --- | --- |
| B70 | Best B-series target | 32 Xe cores, 256 XMX engines, 608 GB/s |
| B580 | Strong for 8B-class prompts | 20 Xe cores, 160 XMX, high clock, 456 GB/s |
| B65 | Good but compute-capped versus B70 | Same 608 GB/s as B70, but 20 Xe cores |
| B60 | Similar compute to B580, more VRAM | 20 Xe cores, 456 GB/s, 24 GB capacity |
| B570 | Lower memory and compute than B580 | 18 Xe cores, 380 GB/s |
| B50 | Capacity useful, bandwidth low | 16 GB, but only 224 GB/s |

For ZINC, the right sequence is:

1. Get DMMV and attention coherent with subgroup-specialized scalar/vector kernels.
2. Benchmark B-series decode against the same GGUFs on the same node.
3. Only then wire XMX/DPAS into batched prefill through `VK_KHR_cooperative_matrix` if the driver reports usable matrix properties.

### Model Fit Guide

Use `./zig-out/bin/zinc --check --model-id <id>` for exact fit. The table below is a planning guide for Q4_K-ish GGUFs plus ZINC temporary buffers:

| VRAM class | B-series cards | Practical target |
| --- | --- | --- |
| 10 GB | B570 | 7B/8B only, short to moderate context |
| 12 GB | B580 | 8B comfortably, 12B only if buffers and context are small |
| 16 GB | B50 | 8B with more context; 12B possible depending on architecture |
| 24 GB | B60 | 20B class and some 27B/35B MoE experiments with tight KV budgeting |
| 32 GB | B65, B70 | 27B dense and 35B MoE targets; best fit for ZINC's larger catalog models |

KV cache can dominate long-context serving. A backend-independent estimate is:

```text
kv_bytes = layers * kv_heads * head_dim * 2 * bytes_per_scalar * context_tokens
```

The `2` is for K and V. GQA/MLA/MoE details change `kv_heads` and `head_dim`; KV quantization changes `bytes_per_scalar`. For B570/B580, the KV budget usually decides the maximum useful context before raw compute does.

## Xe2-HPG Architecture Notes

Intel's oneAPI guide lists Arc B580 as Xe2-HPG / Battlemage and gives the useful programming-level shape:

| Property | Xe2-HPG B580 reference value | Inference consequence |
| --- | ---: | --- |
| Xe cores | 20 | Main scaling unit for B-series cards |
| Vector engines per Xe core | 8 | One Xe core exposes 8 SIMD vector engines |
| Hardware threads per vector engine | 8 | Occupancy hides send/memory latency |
| Supported subgroup sizes | 16, 32 | Do not use RDNA-style wave64 assumptions |
| GRF per thread | 128 / 256 regular / large mode | Register pressure can force spills and reduce occupancy |
| Register width | 512 bits | Wider than Alchemist's 256-bit entry in the oneAPI table |
| L1 cache per Xe core | 256 KB | Shared by vector engines in the Xe core |
| SLM per Xe core | 128 KB | Workgroup-managed local memory |
| Max SLM per workgroup | 128 KB | Useful for tiled attention/prefill, but can reduce residency |
| L3 cache | 18 MB on B580 | Shared last-level GPU cache before GDDR6 |
| Max workgroup size | 1024 | API ceiling, not automatically the fastest local size |

The important mental model shift from RDNA is that an Intel subgroup is a compiler-vectorized SIMD thread, commonly 16 or 32 work-items wide. A 64-thread workgroup on Intel is usually two or four subgroups, not one wave64. Ported shaders should treat subgroup size as a specialization input, not as a compile-time constant inherited from RDNA.

### Memory Hierarchy

For compute kernels, the useful hierarchy is:

```text
GRF registers
  -> SLM / shared local memory, per Xe core, explicit workgroup-managed cache
  -> L1 cache, per Xe core
  -> L3 cache, shared across the GPU
  -> GDDR6 VRAM
```

SLM is not automatically coherent with global memory. Treat it as a scratchpad: load from global, synchronize within the workgroup, compute, write back. The oneAPI guide describes SLM as higher-bandwidth/lower-latency memory local to a Xe core and scoped to the workgroup scheduled there.

For DMMV and attention:

- Use contiguous subgroup-lane memory access. Intel's docs emphasize that memory access shape across the subgroup controls SIMD lane and memory efficiency.
- Prefer block-like loads/stores for tiled work. The compiler may lower structured access into more efficient send messages.
- Avoid atomics, fences, and cross-workgroup coordination in the token hot path.
- Watch register spills. IGC reports ahead-of-time spill warnings, and `unitrace` can report spill/private-memory bytes for JIT paths.

## Opcode And ISA Surface

Intel does not publish a current Xe2 native ISA manual in the same direct style as AMD's RDNA ISA PDFs. The public surface we can rely on is:

- Intel oneAPI documentation for Xe architecture, subgroups, SLM, and XMX.
- Intel Graphics Compiler source, especially the G4/vISA opcode tables and send-op tables.
- Older public Intel processor graphics architecture material and Alchemist PRM files for the general EU instruction model.

This means the table below is best read as the compiler-visible opcode/mnemonic surface relevant to B-series shader work, not a guaranteed complete Xe2 binary encoding manual.

### Execution Model Primitives

| Primitive | What it means for kernels |
| --- | --- |
| SIMD execution size | Instructions operate over SIMD lanes, typically 16 or 32 work-items for Xe2-HPG compute. |
| Predication | Per-lane flag predicates disable lanes without changing the instruction stream. Useful but branch divergence still increases dynamic work. |
| Execution mask | Implicit lane mask tracks active lanes through control flow. |
| Flag/condition modifiers | `cmp` and arithmetic can set flags such as equal, greater, less, overflow, unordered. |
| Source modifiers | Negate, absolute value, and related modifiers can fold simple transforms into operand fetch. |
| Saturation | Clamp arithmetic to destination range when the instruction supports it. |
| Regioning | GRF operands can be addressed with strides and subregister regions. This is powerful but easy to turn into scattered access. |

### Opcode Families

| Family | Representative mnemonics | Inference relevance |
| --- | --- | --- |
| Move/select | `mov`, `sel`, `csel`, `movi`, `smov`, `fcvt` | Copies, predicated select, conversions, dequant plumbing |
| Integer/bit logic | `and`, `or`, `xor`, `not`, `bfe`, `bfi1`, `bfi2`, `bfrev`, `fbh`, `fbl`, `cbit`, `bfn` | Quantized unpack, masks, bitfield extraction, packed GGUF block decode |
| Scalar/vector arithmetic | `add`, `mul`, `avg`, `frc`, `rndu`, `rndd`, `rnde`, `rndz`, `mac`, `mach`, `mad`, `madm`, `add3`, `addc`, `subb`, `shr`, `shl`, `asr`, `ror`, `rol`, `lzd` | DMMV inner loops, address math, reductions, quant scale application |
| Dot/matrix | `dp4a`, `dpas`, `dpasw` | INT8 dot products and XMX systolic matrix operations |
| Compare | `cmp`, `cmpn` | Masks, stop conditions, bounds checks, reductions |
| Control flow | `if`, `else`, `endif`, `while`, `brd`, `brc`, `break`, `cont`, `goto`, `jmpi`, `call`, `return`, `halt`, `join` | Branching and loops; divergence should be minimized in SIMD kernels |
| Math unit | `math` functions such as reciprocal, log, exp, sqrt, rsqrt, pow, sin, cos, integer divide | Softmax, normalization, activation approximations, though compilers may lower or approximate |
| Send/message | `send`, `sendc`, `sends`, `sendsc` | Loads, stores, scatter/gather, atomics, sampler, fences, barriers |
| Sync/misc | `wait`, `nop`, `sync_nop`, `sync_allrd`, `sync_allwr`, `sync_fence` | Ordering and scoreboard control; keep out of hot paths unless required |

### Data Types

IGC's type table includes:

| Syntax | Meaning | Notes |
| --- | --- | --- |
| `ub`, `b` | 8-bit unsigned/signed integer | Quant blocks, packed metadata |
| `uw`, `w` | 16-bit unsigned/signed integer | Halfword unpack, offsets |
| `ud`, `d` | 32-bit unsigned/signed integer | Most address and loop math |
| `uq`, `q` | 64-bit unsigned/signed integer | Pointers and large offsets |
| `hf` | IEEE FP16 | Common XMX/prefill input type |
| `bf` | BF16 | Useful if driver/compiler exposes BF16 matrix support |
| `f` | FP32 | Accumulators, normalization, logits |
| `df` | FP64 | Present in architecture tables, not useful for LLM hot paths |
| `tf32` | TensorFloat-32 container | Matrix path only when exposed by the compiler/device |
| `hf8`, `bf8`, `e2m1` | FP8 / low-bit compiler IR types | Do not assume usable B-series Vulkan exposure without probing features |

For Vulkan, always query device features and cooperative-matrix properties. Do not infer FP8, BF16, or TF32 shader support from a compiler enum alone.

### DPAS And XMX

`dpas` stands for Dot Product Accumulate Systolic. Intel's XMX documentation describes XMX as systolic hardware executing DPAS-style operations for low-precision matrix work. In SYCL, the lower-level route is `joint_matrix_mad`; in Vulkan, the portable route is `VK_KHR_cooperative_matrix` / `SPV_KHR_cooperative_matrix` when the driver advertises matching properties.

For ZINC:

- DPAS/XMX is a prefill and batched-matmul opportunity first.
- Single-token DMMV should not be forced onto XMX until profiling proves the packing and tile overhead are worth it.
- Q4_K and related GGUF formats need unpacking. If the driver exposes only FP16/BF16/INT8 cooperative matrices, a native 4-bit path still needs a conversion strategy.
- The B70 is the obvious XMX target. The B65/B60/B580 all expose 160 XMX engines; the real difference is memory bandwidth and clock.

### Send Messages

Intel memory operations are message based. The public compiler send-op table includes:

| Send group | Representative operations | ZINC interpretation |
| --- | --- | --- |
| Loads | `load`, `load_strided`, `load_quad`, `load_block2d`, status variants | Use for weight/KV reads; contiguous/block forms are the goal |
| Stores | `store`, `store_strided`, `store_quad`, `store_block2d`, uncompressed variants | Use for hidden buffers, KV writes, logits, staging |
| Atomics | integer, floating, and BF16 add/sub/min/max/CAS variants | Avoid in decode unless a reduction cannot be expressed within a subgroup/workgroup |
| Fences/barriers | `fence`, `signal_barrier`, named/system barriers, `wait` | Correctness tools; latency hazards in the token loop |
| Sampler/render | sample, gather4, render read/write | Mostly irrelevant for ZINC compute kernels |

The practical lesson is that memory latency is not a simple load instruction latency. A load is a message with routing, coalescing, cache, and response behavior. Occupancy and access shape are the performance levers.

## ZINC Bring-Up Notes

### Device Detection

Use both PCI device ID and device name. The public B-series IDs are:

| SKU | Device ID |
| --- | --- |
| Arc Pro B70 | 0xE223 |
| Arc Pro B65 | 0xE222 |
| Arc Pro B60 | 0xE211 |
| Arc Pro B50 | 0xE212 |
| Arc B580 | 0xE20B |
| Arc B570 | 0xE20C |

Recommended ZINC defaults by SKU:

| SKU | Bandwidth default | Xe cores | Subgroup starting point | Notes |
| --- | ---: | ---: | ---: | --- |
| B70 | 608 GB/s | 32 | 32 | Best 32 GB target |
| B65 | 608 GB/s | 20 | 32 | Decode-friendly memory/core ratio |
| B60 | 456 GB/s | 20 | 32 | 24 GB capacity, B580-class bandwidth |
| B50 | 224 GB/s | 16 | 32 | Low-power 16 GB, bandwidth limited |
| B580 | 456 GB/s | 20 | 32 | Best consumer 8B target |
| B570 | 380 GB/s | 18 | 32 | Capacity-limited |

Avoid a single "B-series = 640 GB/s" heuristic. The actual public line spans 224 to 608 GB/s.

### Vulkan Capability Probe

Before judging performance, log these fields from `vulkaninfo` or ZINC diagnostics:

- `subgroupSize`, `minSubgroupSize`, `maxSubgroupSize`, and `VK_EXT_subgroup_size_control`
- `VK_KHR_shader_float16_int8`
- `VK_KHR_8bit_storage`
- `VK_KHR_shader_integer_dot_product`
- `VK_KHR_cooperative_matrix`
- `vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR` output if cooperative matrix is present
- maximum workgroup size and maximum shared memory / workgroup
- PCIe generation/link width and whether Resizable BAR is enabled in firmware

Resizable BAR is worth treating as mandatory for benchmark nodes. Intel's support guidance describes it as required for optimal Arc performance; without it, host-visible VRAM access and upload behavior can be misleading.

### Shader Tuning Starting Points

| Kernel family | Initial Intel strategy |
| --- | --- |
| DMMV Q4_K/Q5_K/Q6_K/Q8_0 | Specialize for subgroup 32 first. Compare local sizes 32, 64, and 128. Keep one output row per subgroup until profiling says otherwise. |
| Dequant/unpack | Use vectorized packed loads and bitfield ops. Keep scale/min metadata contiguous. Avoid scattered per-lane byte reads. |
| RMS norm / reductions | Use subgroup reductions for the first reduction stage; spill to SLM only when reducing across multiple subgroups. |
| Flash attention | Use 16/32 subgroup reductions and SLM tiles. Tune tile width against SLM residency and L3 hit rate. |
| MoE routing | Router logits are small; avoid global atomics. CPU top-k may be acceptable until GPU routing is measured. |
| Batched prefill | Prototype cooperative matrix only after scalar/subgroup path is coherent. Query matrix tile shapes instead of assuming RDNA's 16x16x16 path. |
| KV cache | Align pages and rows to cache-friendly boundaries. For B570/B580, cap context before temp buffers push the card into memory pressure. |

### Benchmark Interpretation

Use the same clean-node discipline as RDNA:

- stop stale `zinc`, `llama-server`, and other GPU users
- warm once before measuring
- measure CLI decode separately from HTTP latency
- collect at least three runs
- record driver, kernel, Mesa/intel driver package, Vulkan ICD, and firmware/BIOS ReBAR state

For B65/B70 specifically, compare:

1. 8B decode, to validate DMMV against B580/B60-class cards.
2. 27B/35B fit and decode, to prove the 32 GB cards are buying usable capacity.
3. long-context attention, to see whether L3/SLM tuning or KV bandwidth dominates.
4. batched prefill with and without cooperative matrix, if the driver exposes it.

## Card-By-Card Engineering Summary

### Arc Pro B70

B70 is the flagship B-series inference card: 32 GB, 608 GB/s, 32 Xe cores, 256 XMX engines, 367 INT8 TOPS, and PCIe 5.0 x16. It is the first target for ZINC's large-model Intel work because it has both enough memory and enough compute to make prefill tuning meaningful. Expect decode to scale primarily with 608 GB/s bandwidth, while prefill should benefit from the extra XMX engines if cooperative matrix is usable.

### Arc Pro B65

B65 keeps the 32 GB / 608 GB/s memory system but drops to 20 Xe cores. That makes it a strong decode candidate: for bandwidth-bound single-token inference, it may be near B70 while using less silicon. It is weaker for large prompt prefill and batched serving because matrix compute and scheduler occupancy have less headroom.

### Arc Pro B60

B60 is best understood as a 24 GB, pro-oriented version of the 20 Xe-core Battlemage shape. It has the same 456 GB/s bandwidth class as B580 but twice the memory capacity. It is useful for models that do not fit on B580, but it will not fix a bandwidth-bound decode bottleneck by itself.

### Arc Pro B50

B50 is the power-efficient and small-form-factor option: 16 GB, 70 W, no auxiliary power connector, ECC support on Intel's product page, and 224 GB/s memory bandwidth. It is attractive for compact 8B inference boxes, but the bandwidth is roughly half of B580/B60 and about 37% of B65/B70. Do not expect high single-stream decode throughput.

### Arc B580

B580 is the consumer card to bring up first for small models. It has 20 Xe cores, 160 XMX engines, a high 2670 MHz graphics clock, 12 GB VRAM, and 456 GB/s bandwidth. Its practical ceiling is memory capacity, not compute. It should be a good 8B ZINC target once subgroup and memory-message behavior are tuned.

### Arc B570

B570 is a trimmed 18 Xe-core, 10 GB, 380 GB/s card. The bandwidth is still respectable, but the 10 GB memory pool leaves little room for larger GGUFs, long context, and temporary buffers. It is a useful correctness and lower-end coverage target, not the main optimization target.

## Open Questions For ZINC

- Do B-series Vulkan drivers expose cooperative matrix shapes that map cleanly to GGUF prefill data types?
- Is subgroup 32 always the best decode width, or do some B-series drivers default to subgroup 16 for specific shaders?
- Does B65 match B70 on single-stream decode once kernels are memory-bound?
- How much practical bandwidth does each card sustain on ZINC's quantized DMMV, not synthetic copy tests?
- Can Q4_K unpack be restructured to use INT8 dot-product or XMX paths without losing the bandwidth advantage of compact weights?
- Does Arc Pro ECC materially reduce bandwidth on B70/B50, and can it be toggled or queried reliably on Linux?
- What Linux kernel, Mesa/ANV, and firmware versions are required for stable B70/B65 operation on the benchmark node?

## References

### Intel Product Pages

- [Intel Arc B580 Graphics specifications](https://www.intel.com/content/www/us/en/products/sku/241598/intel-arc-b580-graphics/specifications.html)
- [Intel Arc B570 Graphics specifications](https://www.intel.com/content/www/us/en/products/sku/241676/intel-arc-b570-graphics/specifications.html)
- [Intel Arc Pro B70 Graphics specifications](https://www.intel.com/content/www/us/en/products/sku/245797/intel-arc-pro-b70-graphics/specifications.html)
- [Intel Arc Pro B65 Graphics specifications](https://www.intel.com/content/www/us/en/products/sku/245796/intel-arc-pro-b65-graphics/specifications.html)
- [Intel Arc Pro B60 Graphics specifications](https://www.intel.com/content/www/us/en/products/sku/243916/intel-arc-pro-b60-graphics/specifications.html)
- [Intel Arc Pro B50 Graphics specifications](https://www.intel.com/content/www/us/en/products/sku/242615/intel-arc-pro-b50-graphics/specifications.html)
- [Intel Arc B-Series desktop overview](https://www.intel.com/content/www/us/en/products/docs/discrete-gpus/arc/desktop/b-series/overview.html)
- [Intel Arc Pro B-Series workstation overview](https://www.intel.com/content/www/us/en/products/docs/discrete-gpus/arc/workstations/b-series/overview.html)
- [Intel Arc B-Series Graphics Quick Reference Guide](https://cdrdv2-public.intel.com/839907/Intel%20Arc%20B-Series%20Graphics%20Quick%20Reference%20Guide%20V1.1.pdf)

### Architecture And Programming

- [Intel oneAPI GPU Optimization Guide: Xe GPU Architecture](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/intel-xe-gpu-architecture.html)
- [Intel oneAPI GPU Optimization Guide: Sub-Groups and SIMD Vectorization](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/sub-groups-and-simd-vectorization.html)
- [Intel oneAPI GPU Optimization Guide: Shared Local Memory](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/shared-local-memory.html)
- [Intel oneAPI GPU Optimization Guide: Programming Intel XMX Using SYCL Joint Matrix](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/programming-intel-xmx-using-sycl-joint-matrix.html)
- [Intel oneAPI GPU Optimization Guide: Boost Matrix Multiplication Performance with Intel Xe Matrix Extensions](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/boost-matrix-multiplication-performance-with-intel.html)
- [Intel Processor Graphics: Architecture, ISA and Microarchitecture](https://www.intel.cn/content/dam/develop/external/us/en/documents/intel-graphics-architecture-isa-and-microarchitecture-698638.pdf)
- [Vulkan VK_KHR_cooperative_matrix reference](https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_cooperative_matrix.html)
- [Intel support: What Is Resizable BAR and How Do I Enable It?](https://www.intel.com/content/www/us/en/support/articles/000090831/graphics.html)

### Compiler Source

- [Intel Graphics Compiler](https://github.com/intel/intel-graphics-compiler)
- [IGC G4 instruction list](https://github.com/intel/intel-graphics-compiler/blob/master/visa/G4_Instruction.h)
- [IGC G4 opcode and type definitions](https://github.com/intel/intel-graphics-compiler/blob/master/visa/G4_Opcode.h)
- [IGC send operation table](https://github.com/intel/intel-graphics-compiler/blob/master/visa/iga/IGALibrary/IR/EnumSendOpInfo.hpp)
