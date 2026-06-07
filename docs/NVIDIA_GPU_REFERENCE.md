# NVIDIA Ampere/Ada/Blackwell GPU Reference for Inference

Hardware specifications, SM microarchitecture, PTX/SASS instruction surface, and compute-architecture reference for LLM inference on NVIDIA consumer GPUs across three generations: Ampere (GeForce RTX 30, `sm_86`), Ada Lovelace (RTX 40, `sm_89`), and Blackwell (RTX 50, `sm_120`), plus the workstation siblings that share the same dies. Consolidated from the NVIDIA Ampere GA102 / Ada / RTX Blackwell architecture whitepapers, the CUDA C++ Programming Guide, the PTX ISA spec, official NVIDIA product pages, and the TechPowerUp GPU database. Framed for ZINC's CUDA backend (`docs/cuda-backend.md`), where **decode is memory-bandwidth-bound matrix-vector (DMMV)** — pick the SKU on VRAM capacity + bandwidth — and **prefill is compute-bound matmul** that lives on the Tensor cores. ZINC's int8 GEMMs are **not** Tensor-core based: they use `__dp4a`, the 1:1 CUDA analog of AMD `v_dot4_i32_i8` (`dotPacked4x8AccSatEXT`), so the matmul port is mechanical.

> Note: Cross-checked on 2026-06-06 against the NVIDIA Ampere GA102 whitepaper (v2.1), Ada GPU Architecture whitepaper (v2.02), RTX Blackwell GPU Architecture whitepaper (v1.1), the CUDA C++ Programming Guide "Technical Specifications per Compute Capability" table, the PTX ISA spec (Release 8.8 / 9.x), official nvidia.com product pages, and the TechPowerUp GPU database. Memory bandwidth equals `bus_bits / 8 × effective_Gbps` and matches the cited per-SKU values. Register-file capacity equals `SM_count × 256 KB` (the per-SM file is fixed at 256 KB across all three consumer generations). Die-level facts (transistors, die size, process, SM composition, register file, L1/Shared, L2-per-controller) are from the NVIDIA whitepapers. Per-SKU CUDA-core count, boost clock, VRAM, bus width, and TGP are from nvidia.com; SM/Tensor-core count, L2 size, and memory speed are cross-checked against TechPowerUp (whose `/gpu-specs/` pages return HTTP 403 to automated fetchers, so those values are corroborated via search snippets, NVIDIA pages, and arithmetic). Cycle-cost and cache-latency tables are **engineering estimates from third-party microbenchmarks** (Citadel/Jia, "Demystifying Ampere" arXiv 2208.11174 — measured on the **A100** `sm_80` data-center die, reused here as the Ampere reference; consumer GA10x `sm_86` latencies may differ; Chips and Cheese RTX 4090) — NVIDIA does not publish per-instruction cycle counts for these consumer parts. SASS mnemonics (HMMA/IMMA/IDP4A) are observed via `cuobjdump`/`nvdisasm`, not documented by NVIDIA. The RTX PRO 6000 Blackwell boost clock (2617 MHz) is from the NVIDIA RTX PRO 6000 Workstation Edition datasheet. Remaining genuinely-provisional numbers are called out inline and in the caveat list at the end.

## Hardware Specifications

### Blackwell — GB20x (RTX 50, `sm_120`)

Full dies (whitepaper Appendices A–C): GB202 = 192 SM / 24576 CUDA / 768 Tensor / 128 MB L2 / 512-bit; GB203 = 84 SM / 10752 CUDA / 64 MB L2 / 256-bit; GB205 = 50 SM / 6400 CUDA / 48 MB L2 / 192-bit; GB206 = 36 SM / 4608 CUDA / 32 MB L2 / 128-bit.

| | RTX 5090 | RTX 5080 | RTX 5070 Ti | RTX 5070 | RTX 5060 Ti | RTX 5060 |
|---|---|---|---|---|---|---|
| **Die** | GB202 | GB203 | GB203 | GB205 | GB206 | GB206 |
| **SM count** | 170 | 84 | 70 | 48 | 36 | 30 |
| **CUDA cores** | 21760 | 10752 | 8960 | 6144 | 4608 | 3840 |
| **Tensor cores (5th gen)** | 680 | 336 | 280 | 192 | 144 | 120 |
| **RT cores (4th gen)** | 170 | 84 | 70 | 48 | 36 | 30 |
| **VRAM** | 32 GB GDDR7 | 16 GB GDDR7 | 16 GB GDDR7 | 12 GB GDDR7 | 16/8 GB GDDR7 | 8 GB GDDR7 |
| **Memory bus** | 512-bit | 256-bit | 256-bit | 192-bit | 128-bit | 128-bit |
| **Memory speed** | 28 Gbps | 30 Gbps | 28 Gbps | 28 Gbps | 28 Gbps | 28 Gbps |
| **Memory bandwidth** | 1792 GB/s | 960 GB/s | 896 GB/s | 672 GB/s | 448 GB/s | 448 GB/s |
| **L2 cache** | 96 MB | 64 MB | 48 MB | 48 MB | 32 MB | 24 MB |
| **Boost clock** | 2407 MHz | 2617 MHz | 2452 MHz | 2512 MHz | 2572 MHz | 2497 MHz |
| **TGP** | 575 W | 360 W | 300 W | 250 W | 180 W | 145 W |
| **PCIe** | Gen 5 | Gen 5 | Gen 5 | Gen 5 | Gen 5 | Gen 5 |
| **Transistors** | 92.2 B | 45.6 B | 45.6 B | 31.1 B | 21.9 B | 21.9 B |
| **Die size** | 750 mm² | 378 mm² | 378 mm² | 263 mm² | 181 mm² | 181 mm² |
| **Process** | TSMC 4N | TSMC 4N | TSMC 4N | TSMC 4N | TSMC 4N | TSMC 4N |

The RTX 5090 is a cut GB202 (170 of 192 SMs, 96 MB of the die's full 128 MB L2). The RTX 5080 is the full GB203. The RTX 5060 is a cut GB206: 30 of 36 SMs and **24 MB L2** — a 25% trim of the full die's 32 MB (the 5060 Ti keeps all 32 MB). The 5060 Ti ships in 16 GB and 8 GB variants — same GPU, same 448 GB/s. Process for every consumer Blackwell die is **TSMC 4N** (a 5 nm-class node, the same one Ada used), not the data-center "4NP". The boost clocks are correct as shown: the **5090's 2407 MHz is genuinely below the 5080's 2617 MHz** — the 575 W 512-bit flagship clocks lower than the 360 W 256-bit part by design, not a transcription error.

### Ada Lovelace — AD10x (RTX 40, `sm_89`)

Full dies (TSMC 4N): AD102 = 144 SM / 18432 CUDA / 576 Tensor / 96 MB L2; AD103 = 80 SM / 10240 / 64 MB; AD104 = 60 SM / 7680 / 48 MB; AD106 = 36 SM / 4608 / 32 MB; AD107 = 24 SM / 3072 / 24 MB.

| | RTX 4090 | RTX 4080 Super | RTX 4080 | RTX 4070 Ti Super | RTX 4070 Ti | RTX 4070 Super | RTX 4070 | RTX 4060 Ti | RTX 4060 |
|---|---|---|---|---|---|---|---|---|---|
| **Die** | AD102-300 | AD103-400 | AD103-300 | AD103-275 | AD104-400 | AD104-350 | AD104-250 | AD106 | AD107 |
| **SM count** | 128 | 80 | 76 | 66 | 60 | 56 | 46 | 34 | 24 |
| **CUDA cores** | 16384 | 10240 | 9728 | 8448 | 7680 | 7168 | 5888 | 4352 | 3072 |
| **Tensor cores (4th gen)** | 512 | 320 | 304 | 264 | 240 | 224 | 184 | 136 | 96 |
| **VRAM** | 24 GB G6X | 16 GB G6X | 16 GB G6X | 16 GB G6X | 12 GB G6X | 12 GB G6X | 12 GB G6X | 8/16 GB G6 | 8 GB G6 |
| **Memory bus** | 384-bit | 256-bit | 256-bit | 256-bit | 192-bit | 192-bit | 192-bit | 128-bit | 128-bit |
| **Memory speed** | 21 Gbps | 23 Gbps | 22.4 Gbps | 21 Gbps | 21 Gbps | 21 Gbps | 21 Gbps | 18 Gbps | 17 Gbps |
| **Memory bandwidth** | 1008 GB/s | 736 GB/s | 716.8 GB/s | 672 GB/s | 504 GB/s | 504 GB/s | 504 GB/s | 288 GB/s | 272 GB/s |
| **L2 cache** | 72 MB | 64 MB | 64 MB | 48 MB | 48 MB | 48 MB | 36 MB | 32 MB | 24 MB |
| **Boost clock** | 2520 MHz | 2550 MHz | 2505 MHz | 2610 MHz | 2610 MHz | 2475 MHz | 2475 MHz | 2535 MHz | 2460 MHz |
| **TGP** | 450 W | 320 W | 320 W | 285 W | 285 W | 220 W | 200 W | 160/165 W | 115 W |

The RTX 4090 ships a cut AD102-300 with **72 MB L2** enabled out of the full die's **96 MB** — "72 MB" is the 4090's enabled cache, not the full die. The non-Super RTX 4070 has 36 MB L2 (cut AD104); the 4070 Super/Ti use the full 48 MB. PCIe Gen 4 across the line. FP8 is the headline 4th-gen feature (table below).

### Ampere — GA10x (RTX 30, `sm_86`)

GA102 (Samsung 8N, 28.3 B, 628.4 mm²) full die: 84 SM / 10752 CUDA / 336 Tensor / 6 MB L2 / 384-bit. GA104 (17.4 B, 392.5 mm²) full die: 48 SM / 6144 / 4 MB / 256-bit. GA106 (~12 B, 276 mm²) full die: 30 SM / 3840 / 192-bit.

| | RTX 3090 Ti | RTX 3090 | RTX 3080 Ti | RTX 3080 12G | RTX 3080 10G | RTX 3070 Ti | RTX 3070 | RTX 3060 Ti | RTX 3060 |
|---|---|---|---|---|---|---|---|---|---|
| **Die** | GA102 | GA102 | GA102 | GA102 | GA102 | GA104 | GA104 | GA104 | GA106 |
| **SM count** | 84 | 82 | 80 | 70 | 68 | 48 | 46 | 38 | 28 |
| **CUDA cores** | 10752 | 10496 | 10240 | 8960 | 8704 | 6144 | 5888 | 4864 | 3584 |
| **Tensor cores (3rd gen)** | 336 | 328 | 320 | 280 | 272 | 192 | 184 | 152 | 112 |
| **VRAM** | 24 GB G6X | 24 GB G6X | 12 GB G6X | 12 GB G6X | 10 GB G6X | 8 GB G6X | 8 GB G6 | 8 GB G6/G6X | 12 GB G6 |
| **Memory bus** | 384-bit | 384-bit | 384-bit | 384-bit | 320-bit | 256-bit | 256-bit | 256-bit | 192-bit |
| **Memory speed** | 21 Gbps | 19.5 Gbps | 19 Gbps | 19 Gbps | 19 Gbps | 19 Gbps | 14 Gbps | 14/19 Gbps | 15 Gbps |
| **Memory bandwidth** | 1008 GB/s | 936 GB/s | 912 GB/s | 912 GB/s | 760 GB/s | 608 GB/s | 448 GB/s | 448/608 GB/s | 360 GB/s |
| **L2 cache** | 6 MB | 6 MB | 6 MB | 6 MB | 5 MB | 4 MB | 4 MB | 4 MB | 3 MB |
| **Boost clock** | 1860 MHz | 1695 MHz | 1665 MHz | 1710 MHz | 1710 MHz | 1770 MHz | 1725 MHz | 1665 MHz | 1777 MHz |
| **TGP** | 450 W | 350 W | 350 W | 350 W | 320 W | 290 W | 220 W | 200 W | 170 W |

**Ampere L2 is small** — 6 MB on a full 384-bit GA102 (512 KB × twelve 32-bit controllers), trimmed to 5 MB (320-bit RTX 3080 10G) and 3 MB (192-bit GA106) where narrower buses disable memory controllers and their attached L2 slices. A 12 MB figure in some secondary tables is the Ada AD102, not Ampere. The 3060 Ti ships GDDR6 (448 GB/s) and GDDR6X (608 GB/s) variants; both rows are shown. PCIe Gen 4. The GA100 datacenter part (A100) is `sm_80`, a different die, not covered here.

### Workstation siblings (same dies, 2–3× VRAM, ECC)

Pro SKUs binned to the full (or near-full) die — the opposite of the cut-down consumer flagships. Relevant because the extra VRAM lets one card hold a model that needs two consumer cards, and the compute capability matches the consumer twin exactly (dev-on-consumer, deploy-on-pro). The VRAM multiplier vs the consumer twin is **exactly 2× for the Ampere/Ada pros** (A6000 48 GB vs 3090 Ti 24 GB; RTX 6000 Ada 48 GB vs 4090 24 GB) but **3× for the Blackwell pro** (RTX PRO 6000 96 GB vs RTX 5090 32 GB) — the 5090 already carries 32 GB, so the pro card jumps to 96 GB rather than 64.

| | RTX A6000 | RTX 6000 Ada | RTX PRO 6000 Blackwell |
|---|---|---|---|
| **Die / arch** | GA102 / Ampere | AD102 / Ada | GB202 / Blackwell |
| **Consumer die-mate** | RTX 3090 Ti | RTX 4090 | RTX 5090 |
| **SM count** | 84 (full) | 142 | 188 |
| **CUDA cores** | 10752 | 18176 | 24064 |
| **Tensor cores** | 336 (3rd) | 568 (4th) | 752 (5th) |
| **VRAM (ECC)** | 48 GB GDDR6 | 48 GB GDDR6 | 96 GB GDDR7 |
| **VRAM vs consumer twin** | 2× (24 → 48) | 2× (24 → 48) | 3× (32 → 96) |
| **Memory bus** | 384-bit | 384-bit | 512-bit |
| **Memory bandwidth** | 768 GB/s | 960 GB/s | 1792 GB/s |
| **L2 cache** | 6 MB | 96 MB | 128 MB |
| **Boost clock** | 1800 MHz | 2505 MHz | 2617 MHz |
| **TGP** | 300 W | 300 W | 600 W |
| **PCIe / NVLink** | Gen 4 / 2-way | Gen 4 / no | Gen 5 / no |
| **Compute capability** | `sm_86` | `sm_89` | `sm_120` |

The RTX PRO 6000 row is the 600 W Workstation Edition (boost 2617 MHz, base 1590 MHz, per the NVIDIA Workstation Edition datasheet); the Max-Q and Server variants share the die and 96 GB but clock lower (~300 W). RTX 6000 Ada memory clock (20 Gbps) is derived from 960 GB/s over 384-bit. ECC is on by default on all three workstation cards; the bandwidth cost is the standard inline-ECC overhead (see "GDDR ECC" under Driver / Toolchain).

## SM (Streaming Multiprocessor) Architecture

![NVIDIA SM Architecture](/nvidia-sm.svg)

The SM is NVIDIA's analog of an AMD Compute Unit. At the top level it is structurally constant across Ampere GA10x, Ada AD10x, and Blackwell GB20x **consumer** GPUs: 128 CUDA cores, 4 Tensor cores, 256 KB register file, 128 KB unified L1/shared.

| Per SM | Ampere GA10x (8.6) | Ada AD10x (8.9) | Blackwell GB20x (12.0) |
|---|---|---|---|
| CUDA cores (FP32) | 128 | 128 | 128 |
| INT32 lanes | 64 (shared w/ FP32) | 64 (shared w/ FP32) | 64 (shared w/ FP32) |
| Tensor cores | 4 (3rd gen) | 4 (4th gen) | 4 (5th gen) |
| RT cores | 1 (2nd gen) | 1 (3rd gen) | 1 (4th gen) |
| Processing blocks (sub-partitions) | 4 | 4 | 4 |
| Register file | 256 KB (65,536 × 32-bit) | 256 KB | 256 KB |
| Register file / partition | 64 KB | 64 KB | 64 KB |
| Max registers / thread | 255 | 255 | 255 |
| Unified L1 + shared (physical) | 128 KB | 128 KB | 128 KB |
| Max shared memory / SM | 100 KB | 100 KB | 100 KB |
| Max shared memory / block | 99 KB | 99 KB | 99 KB |
| Shared carveout options (KB) | 0/8/16/32/64/100 | 0/8/16/32/64/100 | 0/8/16/32/64/100 |

```
                         One SM (Ampere / Ada / Blackwell consumer)
  ┌───────────────────────────────────────────────────────────────────────┐
  │ Partition 0      Partition 1      Partition 2      Partition 3         │
  │ ┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐     │
  │ │ warp sched │   │ warp sched │   │ warp sched │   │ warp sched │     │
  │ │ 16 FP32    │   │ 16 FP32    │   │ 16 FP32    │   │ 16 FP32    │     │
  │ │ 16 FP32/I32│   │ 16 FP32/I32│   │ 16 FP32/I32│   │ 16 FP32/I32│     │
  │ │ 1 Tensor   │   │ 1 Tensor   │   │ 1 Tensor   │   │ 1 Tensor   │     │
  │ │ 1 SFU·4LSU │   │ 1 SFU·4LSU │   │ 1 SFU·4LSU │   │ 1 SFU·4LSU │     │
  │ │ 64 KB regs │   │ 64 KB regs │   │ 64 KB regs │   │ 64 KB regs │     │
  │ └────────────┘   └────────────┘   └────────────┘   └────────────┘     │
  │ ── 128 KB unified L1 data cache / shared memory (shared by 4 part.) ── │
  │ ── 1 RT core ────────────────────────────────────────────────────── │
  └───────────────────────────────────────────────────────────────────────┘
```

Each SM is split into **four processing blocks (sub-partitions)**, each holding: a **64 KB register file** (16,384 × 32-bit), **one warp scheduler + one dispatch unit**, an L0 instruction cache, **one Tensor core**, 4 load/store units, 1 SFU, and 32 FP32-capable CUDA lanes. Of those 32 lanes per partition, 16 are FP32-only and 16 are FP32-or-INT32, so a partition issues 32 FP32/clk; the four partitions together do **128 FP32 ops/clk**, or 64 FP32 + 64 INT32/clk. This dual-FP32 datapath is the Ampere change vs Turing (which had one FP32-only + one INT32-only path), doubling FP32 rate; Ada and Blackwell consumer parts keep the same split-issue model (the whitepaper notes "many common INT operations run at up to 2× throughput, but not all"). There is no fully-unified 128-wide INT path on consumer Blackwell — INT-heavy index/dequant math (quantized-weight unpacking) still competes with half the FP32 issue.

Representative SKUs and the resulting memory subsystems (whitepaper spec tables). Note the SM/CUDA/L2/register-file rows below are the **shipping** values for the named card, not the full die:

| | RTX 3090 (GA102) | RTX 4090 (AD102) | RTX 5090 (GB202) |
|---|---|---|---|
| SMs (shipping) | 82 | 128 | 170 |
| CUDA cores | 10,496 | 16,384 | 21,760 |
| Tensor cores | 328 (3rd) | 512 (4th) | 680 (5th) |
| Peak FP32 (non-Tensor) | ~35.6 TFLOPS | 82.6 TFLOPS | 104.8 TFLOPS |
| L2 (shipping / full die) | 6 / 6 MB | 72 / 96 MB | 96 / 128 MB |
| Register file (shipping / full die) | 20.5 / 21 MB | 32 / 36 MB | 42.5 / 48 MB |
| Boost clock | 1695 MHz | 2520 MHz | 2407 MHz |

CUDA cores = SMs × 128; Tensor cores = SMs × 4. Peak non-Tensor FP32 = 2 × CUDA_cores × boost_clock (the 2× FP32/clk from the dual datapath). **Register file = SM_count × 256 KB** — so the shipping cards carry 82×256 KB = 20.5 MiB (3090), 128×256 KB = 32 MiB (4090), 170×256 KB = 42.5 MiB (5090), while the full dies (84/144/192 SM) carry 21 / 36 / 48 MiB. The shipping figures are the ones a kernel actually allocates against for occupancy; the full-die figures only apply to the un-cut SKU (3090 Ti / RTX 6000 Ada is 142-SM / RTX PRO 6000 is 188-SM, none of which is the consumer flagship).

**L2 / memory:** 512 KB L2 per 32-bit memory controller → 6 MB on full GA102, 96 MB on full AD102, 128 MB on full GB202. **L2, not L1, is the inference lever across these generations:** Ada's L2 jumped 16× over Ampere (6 → 96 MB) and Blackwell carries 128 MB, while per-SM L1/shared is flat at 128 KB. A large L2 amplifies effective bandwidth for the KV cache and repeatedly-touched activations — NVIDIA's analog of AMD Infinity Cache, except it sits at the L2 level rather than as a separate victim cache.

**Contrast with an AMD RDNA4 CU:** an SM's 4 partitions ≈ a WGP's pair of CUs each split into 2 SIMDs — both fan a wide vector engine into independently-scheduled sub-units.

| Property | NVIDIA SM | AMD RDNA4 CU (gfx1201) |
|---|---|---|
| FP32 ALUs | 128 CUDA cores | 64 stream processors (2× SIMD32) |
| Sub-units | 4 partitions, 1 scheduler each | 2 SIMD32 + 1 scalar unit |
| SIMT width | warp = 32 (fixed) | wave32 or wave64 (selectable) |
| Register file | 256 KB/SM, **static**, 256-reg/warp granularity | 192 KB VGPR/SIMD (384 KB/CU), **dynamic** 16/32-reg blocks |
| Scratchpad | up to 100 KB shared of 128 KB unified L1+shared | 64 KB LDS/CU (128 KB/WGP) |
| Matrix unit | 4 Tensor cores/SM | 2 AI accelerators (WMMA)/CU |
| L1 | 128 KB unified L1+shared/SM | 32 KB L0 vector cache/CU |

The decisive differences for inference: NVIDIA's warp is a fixed 32 lanes (RDNA4 can run measured-optimal wave64 for DMMV); AMD's per-SIMD VGPR file is larger and dynamically allocated, so high-register NVIDIA kernels lose occupancy more abruptly; and NVIDIA's bandwidth amplification lives in a very large L2 vs AMD's Infinity Cache.

## Warp Execution Model (32-lane SIMT)

The execution unit is the **warp = 32 threads** (fixed across all NVIDIA GPUs; there is no wave64 equivalent). All 32 lanes execute in lockstep (SIMT), one PC per warp. Each partition's warp scheduler picks one ready warp per cycle, so an SM issues up to **4 warp-instructions/clk**. ZINC's wave64 GLSL workgroups map to a **warp32 block** — use the existing Vulkan wave32 fallback (cross-subgroup shared-memory merge) as the porting template (`docs/cuda-backend.md` §4).

| Property | Value |
|---|---|
| Threads per warp | 32 (fixed) |
| Register width per lane | 32 bits |
| Max registers / thread | 255 |
| Max threads / block | 1024 |
| Max warps / SM | 48 (8.6 / 8.9 / 12.0) · 64 (8.0 A100 / 9.0 H100) |
| Max threads / SM | 1536 (8.6 / 8.9 / 12.0) · 2048 (A100) |
| Max thread blocks / SM | 16 (8.6 / 8.9) · 24 (12.0) · 32 (A100 / H100) |

> Note: consumer Blackwell (`sm_120`) matches Ada (`sm_89`) at **48 warps / 1536 threads / SM** — it does *not* inherit the data-center 64-warp/2048-thread limits of A100 (`sm_80`) or Hopper (`sm_90`). The block limit rises to 24 on `sm_120`.

**Occupancy vs registers/thread.** Registers are statically allocated per warp at 256-register granularity (8 regs/thread × 32 lanes) out of the SM's whole **65,536-register file** (256 KB), so high register pressure caps resident warps abruptly. On consumer Ampere/Ada/Blackwell the warp cap is 48, so register pressure only bites above ~32 regs/thread; below that the warp/block limit dominates.

| Registers / thread | Warps/SM (cap 48: 8.6/8.9/12.0) | Warps/SM (cap 64: A100/H100) |
|---|---|---|
| ≤32 | 48 (max) | 64 (max) |
| 40 | 48 | 51 |
| 48 | 42 | 42 |
| 64 | 32 | 32 |
| 96 | 21 | 21 |
| 128 | 16 | 16 |
| 168 | 12 | 12 |
| 255 | 8 | 8 |

Both columns derive from the same 65,536-register file at 256-reg/warp granularity: `warps = floor(65536 / round_up(regs × 32, 256))`, then clamped to the architecture's warp cap (48 or 64). The cap-48 column is the consumer-GPU column to read; the rows at and below 48 regs/thread are identical between the two caps because the register file, not the warp cap, is the binding limit there. Shared memory caps occupancy independently: a block using the 100 KB consumer maximum leaves room for ~1 block/SM out of the 128 KB L1+shared pool. **For decode (DMMV)** the limiter is VRAM bandwidth, not occupancy — keep ≤32 regs/thread and modest shared use so dozens of warps each stream a different weight row and hide GDDR latency. **For prefill** feed full tiles to the 4 Tensor cores/SM.

## Memory Hierarchy

```
Thread → Registers (256 KB/SM, 64 KB/partition, ≤255 regs/thread, 0 cyc)
  ↓
Warp → Shared memory (up to 100 KB/SM; carved from 128 KB unified L1+shared) — 32 banks × 4 B
  ↓
SM → L1 data cache (remainder of the 128 KB unified block, 128 B line)
  ↓
GPU → L2 cache (6 MB GA102 · 72/96 MB AD102 · 96/128 MB GB202, 128 B line)
  ↓
GPU → GDDR6 / GDDR6X / GDDR7 VRAM (360 GB/s – 1792 GB/s)
  ↓
Host ↔ GPU → PCIe 4.0 / 5.0 x16 (model load, KV/weight offload)
```

**Cache capacity / latency per generation** (microbenchmark estimates; cycle ↔ ns depends on clock). The "Capacity" column is the size of the level, not a bandwidth — per-level bandwidth is in the separate list below:

| Access | Ampere (A100, measured) | Ada (RTX 4090, measured) | Capacity | Cache line |
|---|---|---|---|---|
| Register | 0 | 0 | 256 KB/SM | — |
| Shared (no conflict) | ~23 cyc | ~30 cyc | ≤100 KB/SM | — |
| L1 hit | ~33 cyc | ~32 cyc | ≤128 KB/SM | 128 B (4 × 32 B sectors) |
| L2 hit | ~200 cyc | ~285 cyc | 6 / 72–96 / 96–128 MB | 128 B (4 × 32 B sectors) |
| Global / DRAM (miss) | ~290 cyc | ~571 cyc | 8–32 GB | 32 B sector |
| PCIe 4.0 / 5.0 x16 | ~µs | ~µs | host RAM | — |

> The Ampere latency column is from "Demystifying the Nvidia Ampere Architecture" (arXiv 2208.11174), measured on an **A100 (`sm_80`, data-center die)** — not a consumer GA10x (`sm_86`) part. It is reused here as the Ampere reference; consumer GA10x latencies may differ (the consumer L2 is far smaller, 6 MB vs 40 MB). The Ada column is from Chips and Cheese's RTX 4090 microbenchmark.

**Key bandwidths for inference** (order-of-magnitude; the per-level on-chip rates are estimates, the VRAM/PCIe rates are spec):

- **Shared memory:** 32 banks × 4 B = **128 B/clk/SM** at full speed (one 32-lane access with no bank conflict). At ~2.5 GHz that is ~320 GB/s per SM, aggregating to tens of TB/s across all SMs — the reason tiled prefill stages operands in shared.
- **L1 data cache:** ~**128 B/clk/SM** read on a hit (one 128 B line/clk), same order as shared since they are the same physical SRAM.
- **L2:** aggregate **multi-TB/s** — order-of-magnitude `L2_bytes_per_clk × boost_clock`. Ada/Blackwell roughly doubled L2 bandwidth-per-slice vs Ampere alongside the 16× capacity jump; the exact bytes/clk is not published, so treat this as "much higher than VRAM, lower than shared/L1 aggregate."
- **VRAM:** **360 GB/s (3060) – 1792 GB/s (5090)** theoretical (`bus_bits/8 × Gbps`); large DMMV typically sustains ~60–85% of spec after overhead and KV-cache reads.
- **PCIe 4.0 x16:** ~**32 GB/s per direction** (64 GB/s bidir); **PCIe 5.0 x16:** ~**64 GB/s per direction** (128 GB/s bidir) — model load and CPU/GPU offload only.

**GDDR6X vs GDDR7.** GDDR6X (Ampere/Ada) uses **PAM4** signaling, 19–22.4 Gbps/pin (NVIDIA shipped 19 Gbps on the RTX 3090, up to 22.4 Gbps on the RTX 4080). GDDR7 (Blackwell) uses **PAM3**, 28 Gbps/pin on the RTX 50 series with a standard roadmap to 32–48 Gbps. The 5090's 512-bit × 28 Gbps GDDR7 is the single biggest decode lever in the lineup: 1792 GB/s, +78% over the 4090's 1008 GB/s. GDDR7 adds on-die error detection/replay; software ECC on consumer boards costs bandwidth — leave it off for inference unless correctness demands it (see "GDDR ECC" under Driver / Toolchain).

**Cache line & sectoring.** L1 and L2 use a **128-byte line split into 4 × 32-byte sectors**, each filled independently from DRAM. On compute capability 6.0+ the global-load access/transaction unit is **32 bytes** whether or not the load is L1-cached (Best Practices Guide; the 128 B-line/sector detail is from Nsight Compute docs).

**Coalescing.** The 32 threads of a warp issuing aligned, contiguous accesses coalesce into the minimum number of 32-byte sector transactions — 32 consecutive 4-byte words (128 B) → 4 sectors = 1 line. Scattered/strided access inflates the sector count (worst case 32 separate transactions), the dominant decode-perf failure mode. For DMMV weights stream sequentially so coalescing is natural; for paged-KV attention, align pages to 128-byte lines.

**Shared-memory banking.** Shared memory is **32 banks, 4 bytes (32 bits) wide**. Threads hitting distinct banks (or all reading one address = broadcast) run at full speed; an N-way bank conflict serializes into N transactions. Pad shared arrays by one element per row to avoid conflicts on columnar access — the same strategy as AMD's LDS. Used for cross-lane reductions, KV/activation tiles, and dequant scale/min staging.

**PCIe.** PCIe 4.0 x16 = 32 GB/s per direction (64 GB/s bidir, 16 GT/s/lane); PCIe 5.0 x16 = 64 GB/s per direction (128 GB/s bidir, 32 GT/s/lane). Relevant only for model upload and CPU/GPU offload — Blackwell's Gen 5 doubles host↔device over Ada's Gen 4.

### Decode Roofline and Model Fit Guide

Single-token LLM decode is memory-bandwidth-bound, so the rough upper bound is:

```text
decode_tok/s  <=  sustained_bandwidth_bytes/s  /  active_weight_bytes_per_token
```

Decode tok/s tracks the bandwidth column almost linearly. Ranking across the consumer flagships (and the inference takeaway: pick on **VRAM capacity first, bandwidth second** — CUDA/Tensor-core counts only move prefill):

```text
5090 (1792) > 4090 = 3090 Ti (1008) > 5080 (960) > 3090 (936) > 5070 Ti (896)
  > 4080S (736) > 5070 (672) = 4070-Ti-S (672) > 4070-family (504) > 5060 (448) > 4060 (272)
```

VRAM is the model-size gate. Use `./zig-out/bin/zinc --check --model-id <id>` for exact fit; the table is a planning guide for Q4_K-ish GGUFs plus ZINC temporary buffers and a modest KV budget:

| VRAM | NVIDIA cards | Practical target |
|---|---|---|
| 8 GB | 5060, 5060 Ti-8G, 4060, 3070 | 7B–8B at 4-bit, short/moderate context |
| 10–12 GB | 5070, 4070-family, 3080, 3060 | 8B comfortably; 13B only with tight buffers/context |
| 16 GB | 5080, 5070 Ti, 4080/Super, 5060 Ti-16G | 13B–14B at 4-bit with usable context |
| 24 GB | 4090, 3090/Ti | 30B at 4–8-bit, or 70B at 3-bit with care |
| 32 GB | 5090 | 70B at 3–4-bit, or 30B at 8-bit with long context |
| 48–96 GB | RTX A6000 / 6000 Ada / PRO 6000 | 70B at Q4 (one card); 96 GB → Q8 or ~120B MoE |

KV cache can dominate long-context serving; a backend-independent estimate is `kv_bytes = layers × kv_heads × head_dim × 2 × bytes_per_scalar × context_tokens` (the `2` is K and V; GQA/MLA and KV quantization change the per-token term). On the 8–12 GB cards the KV budget usually decides max useful context before compute does. The large L2 on Ada/Blackwell (72–96 MB) holds the KV cache of several short-context streams resident, pushing effective decode bandwidth above the GDDR spec.

## Tensor Cores

A Tensor core performs a warp-cooperative matrix multiply-accumulate `D = A·B + C`. Each SM has 4 (one per partition) across all three generations; peak-throughput growth comes from clocks plus narrower datatypes (FP8 on Ada, FP4 on Blackwell), not more cores per SM. NVIDIA's published per-core rate: Ampere GA10x = **128 dense / 256 sparse** FP16 FMA ops/clk/core (512/1024 per SM), vs Turing's 64/core; Ada and Blackwell keep 128 dense FP16 FMA/clk/core. That 128-FMA/clk/core dense rate is what makes the peak-TFLOPS table below consistent: e.g. 5090 = 680 cores × 128 × 2 (FMA = 2 FLOP) × 2.407 GHz ≈ 419 dense FP16-acc TFLOPS.

### Supported types per generation (`out` = accumulator; sparse = 2:4 structured, 2× rate)

| Input type | Out | Ampere 3rd | Ada 4th | Blackwell 5th | Min target ISA |
|---|---|---|---|---|---|
| FP16 | FP16 / FP32 | Yes | Yes | Yes | `sm_70` |
| BF16 | FP32 | Yes | Yes | Yes | `sm_80` |
| TF32 | FP32 | Yes | Yes | Yes | `sm_80` |
| INT8 (u8/s8) | INT32 | Yes | Yes | Yes | `sm_75` |
| INT4 (u4/s4) | INT32 | Yes | Yes | **No** (dropped) | `sm_75` |
| INT1 / binary (b1) | INT32 | Yes | Yes | **No** (dropped) | `sm_75` |
| FP8 e4m3 / e5m2 | FP16 / FP32 | **No** | Yes | Yes | `sm_89` |
| FP6 e3m2 / e2m3 | FP32 | No | No | Yes (block-scaled) | `sm_120a` |
| FP4 e2m1 | FP32 | No | No | Yes (block-scaled, NVFP4) | `sm_120a` |

Key boundaries for inference: **FP8 starts at Ada (`sm_89`)** — Ampere (`sm_86`) cannot run FP8 `mma`, which is why Ada is the entry point for FP8-quantized LLM inference. **Blackwell consumer drops the INT4 and binary Tensor paths** — the RTX 5090 spec table has no INT4 TOPS row; low-bit on Blackwell is FP4 (NVFP4), not INT4. **FP4/FP6 are block-scaled (microscaling)** — the 5th-gen core handles NVFP4 element grouping, per-block scaling, and the 4-bit matmul in hardware via `.kind::mxf4nvf4` / `.block_scale` PTX qualifiers. Per NVIDIA's framing, FP4 doubles dense Tensor throughput vs **FP8** (whitepaper Fig. 8) and halves memory vs **FP16**; FP6 element grouping is supported but the consumer SKU publishes only FP4/FP8 TFLOPS (FP6 runs at roughly the FP8 rate). None of the consumer dies has meaningful FP64 Tensor throughput (Ampere GA10x has none; Ada/Blackwell carry a minimal number purely for FP64 program correctness).

Data-format bit layouts (PTX ISA), the storage formats the dequant path targets:

| Format | Bits | Exp | Mant | Notes |
|---|---|---|---|---|
| FP16 (IEEE half) | 16 | 5 | 10 | — |
| BF16 | 16 | 8 | 7 | FP32 range, FP16 storage |
| TF32 | 19 used (in FP32 reg) | 8 | 10 | Tensor-only input; 8-bit exp like FP32, 10-bit mantissa like FP16 |
| FP8 e4m3 | 8 | 4 | 3 | no inf; NaN only `0x7f`/`0xff` |
| FP8 e5m2 | 8 | 5 | 2 | wider range, less precision |
| FP6 e3m2 | 6 | 3 | 2 | no inf/NaN; packed |
| FP6 e2m3 | 6 | 2 | 3 | no inf/NaN; packed |
| FP4 e2m1 | 4 | 2 | 1 | no inf/NaN; packed; NVFP4 |

### Peak Tensor throughput per flagship (`dense / sparse`, whitepaper Appendix A)

| Metric | RTX 3090 | RTX 4090 | RTX 5090 |
|---|---|---|---|
| FP16 (FP16 acc) TFLOPS | 142 / 284 | 330.3 / 660.6 | 419 / 838 |
| FP16 (FP32 acc) TFLOPS | 71 / 142 | 165.2 / 330.4 | 209.5 / 419 |
| BF16 (FP32 acc) TFLOPS | 71 / 142 | 165.2 / 330.4 | 209.5 / 419 |
| TF32 TFLOPS | 35.6 / 71 | 82.6 / 165.2 | 104.8 / 209.5 |
| FP8 (FP16 acc) TFLOPS | — | 660.6 / 1321.2 | 838 / 1676 |
| FP6 (FP32 acc) TFLOPS | — | — | — (no published consumer TFLOPS; rate ≈ FP8) |
| FP4 (FP32 acc) TFLOPS | — | — | 1676 / 3352 |
| INT8 TOPS | 284 / 568 | 660.6 / 1321.2 | 838 / 1676 |
| INT4 TOPS | 568 / 1136 | 1321.2 / 2642.4 | — (no Tensor path) |

Internal consistency of this table: FP16/FP16-acc = 2× FP16/FP32-acc; INT8 = FP8; INT4 = 2× INT8; FP4 = 2× FP8; sparse = 2× dense throughout. The RTX 5090's headline "3,352 AI TOPS" is exactly the FP4 sparse figure. Each narrower datatype roughly doubles throughput (FP16 → FP8 → FP4), and 2:4 sparsity adds another 2× on top. FP6 has no published consumer TFLOPS row — NVIDIA does not quote one for the 5090 — but it runs at approximately the FP8 rate (838 dense / 1676 sparse); the row is shown with a dash so the table stays exhaustive across every type in the support matrix above.

### mma.sync PTX tile shapes (warp-level `mma.sync.aligned.mMnNkK...`)

Output tile is M×N = 16×8 for the modern `m16n8kK` family; K (contraction depth) grows as the datatype narrows because more elements fit per 32-bit register lane.

| Multiplicand type | Supported shapes | Min target ISA |
|---|---|---|
| FP16 `.f16` | m8n8k4 (sm_70), m16n8k8 (sm_75), m16n8k16 (sm_80) | `sm_70` |
| BF16 `.bf16` | m16n8k8, m16n8k16 | `sm_80` |
| TF32 `.tf32` | m16n8k4, m16n8k8 | `sm_80` |
| INT8 `.u8`/`.s8` | m8n8k16 (sm_75), m16n8k16, m16n8k32 (sm_80) | `sm_75` |
| INT4 `.u4`/`.s4` | m8n8k32 (sm_75), m16n8k32, m16n8k64 (sm_80) | `sm_75` |
| FP8 `.e4m3`/`.e5m2` | m16n8k32 (m16n8k16 + f16 ctype, PTX 8.7) | `sm_89` |
| FP6/FP4 `.e3m2`/`.e2m3`/`.e2m1` | m16n8k32 with `.kind::mxf*` / `.block_scale` | `sm_120a` |

The canonical inference shapes are **m16n8k16 for FP16/BF16**, **m16n8k8 for TF32**, **m16n8k16/32 for INT8**, and **m16n8k32 for FP8/FP4**. The C++ `nvcuda::wmma` API (CUDA Programming Guide) exposes coarser fixed tiles — 16×16×16 / 32×8×16 / 8×32×16 for FP16/BF16/INT8, 8×8×32 for INT4, 8×8×128 for b1, 16×16×8 for TF32. Sub-byte `wmma` (`u4`/`s4`/`b1`) is **deprecated and removed at `sm_90`**; FP8/FP6/FP4 are not in `wmma` — reach them through `mma.sync` PTX, CUTLASS, or cuBLASLt. `ldmatrix.sync.aligned` stages operands (global → shared via `cp.async`/TMA → `ldmatrix` → `mma`), with `.b6x16_p32`/`.b4x16_p64` packed source formats for FP6/FP4.

### When to Use Tensor Cores for Inference

| Operation | Use Tensor Cores? | Why |
|---|---|---|
| Prefill matmul (QKV/O/MLP, large M) | Yes | Compute-bound, M = seq_len ≫ 1, fills tiles, FLOPS dominates |
| Prefill attention (QKᵀ, S·V) | Yes | Large M×N×K GEMM; FlashAttention is built on `mma`/`wgmma` |
| Decode DMMV (single token, M=1) | No | Memory-bandwidth-bound; M=1 fills 1/16 of a 16×8 tile, no BW saved |
| Batched decode (M = 4–16 tokens) | Maybe | Skinny GEMM; pays off once M ≈ tile M (8/16) and intensity > ridge |
| KV-cache append / RoPE / dequant | No | Element-wise / gather, not matrix multiply |
| RMSNorm / softmax / sampling | No | Reductions and transcendentals |

Rule of thumb: cross over to Tensor Cores when the matmul's free dimension M ≥ ~8–16 **and** arithmetic intensity exceeds the roofline ridge point. Pure single-stream decode never does; all of prefill and large-batch serving do.

## Instruction Set Highlights (PTX / SASS)

### Integer dot product — the ZINC int8 GEMM primitive

| Intrinsic | PTX | Computes | Min arch |
|---|---|---|---|
| `__dp4a(int,int,int)` | `dp4a.s32.s32 d,a,b,c` | 4× int8 pairwise mul, summed + accumulated into int32 | `sm_61` |
| `__dp4a(unsigned,…)` | `dp4a.u32.u32` | unsigned 4× uint8 dot + accumulate | `sm_61` |
| `__dp2a_lo / __dp2a_hi` | `dp2a{.lo\|.hi}.atype.btype` | 2-way dot: one operand as two 16-bit values, the other's low/high 2 bytes as 8-bit, + accumulate int32 | `sm_61` |

PTX semantics, verbatim: dp4a — *"Four pairs of 8-bit values are extracted from the operands, multiplied together pairwise, and the results are accumulated."* **This is the load-bearing mapping for ZINC.** `__dp4a` is the exact CUDA analog of AMD `v_dot4_i32_i8` / `V_DOT4_I32_IU8` and the Vulkan `dotPacked4x8AccSatEXT` path. ZINC's quantized matmul (Q4_K/Q5_K/Q6_K/Q8_0) dequantizes to int8 and reduces with `__dp4a`, identical in structure to the GLSL shaders; the M0 smoke test confirms `1*5+2*6+3*7+4*8 = 70` via `__dp4a` on the RTX 4090. `__dp2a` is the int4×int8 / mixed path analog of AMD's `V_DOT2`. (SASS: `__dp4a` lowers to an `IDP4A`-family instruction on recent arches per community `cuobjdump`, but the exact mnemonic is not contractual — verify with `cuobjdump -sass`.)

### Packed FP16 / BF16, conversion, warp, async, vector

- **`__hfma2` (`HFMA2`)** — two FP16 FMAs in one instruction (`a*b+c` lane-wise), 2× FP16 throughput. The FP16 analog of AMD `V_DOT2_F32_F16` + `V_PACK_B32_F16`; use for FP16 DMMV accumulate. `__half2` is 2× FP16 packed in 32 bits. **BF16 (`__nv_bfloat16`) is native only on `sm_80+`** (Ampere+) — emulated and slow below; keep BF16 weights on Ampere+.
- **`cvt` (`F2F`/`I2F`/`F2I`)** — the dequant hot path: unpack 4-bit/8-bit weight → `cvt` to FP16/FP32, or keep int8 for `__dp4a`. `cvt.rna.tf32.f32` is the Tensor-core input cast (prefill only). Analog of AMD `V_CVT_F32_F16`.
- **Warp intrinsics** (32-bit lane `mask`, full warp `0xffffffff`; `_sync` family introduced CUDA 9): `__shfl_down_sync` (tree reductions for dot/softmax/RMS sum), `__shfl_sync` (broadcast scale/min), `__ballot_sync`/`__activemask` (top-k/sampling masks), `__any_sync`/`__all_sync`. **`__reduce_add_sync` / `__reduce_max_sync` are hardware-accelerated only on `sm_80+`** — Ampere+ get a one-instruction warp reduce; keep a `__shfl_down_sync` fallback for portability. ZINC's kernels already use `__shfl_down_sync` (`src/shaders/cuda/kernels.cu`).
- **`cp.async` (`cp.async.{ca,cg}.shared.global`)** — DMA global→shared **without** staging through registers, overlapping compute; **`sm_80+`** (Ampere). The natural prologue for tiled `mul_mm_*` / `*_full_dp4a` prefill GEMMs (overlap weight-tile load with the prior tile's `__dp4a` accumulate) — analog of AMD's split `S_BARRIER_SIGNAL/WAIT` + `GLOBAL_LOAD` overlap. **Decode DMMV does not benefit** (memory-bound, M=1, no tile reuse).
- **TMA (`cp.async.bulk.tensor`)** — dedicated Tensor Memory Accelerator: one elected thread hands a tensor descriptor to a HW engine that does the whole multi-dim transfer + bounds handling. **`sm_90` (Hopper), carried to `sm_120` (Blackwell)** — *not* on the RTX 4090 (`sm_89`), where only plain `cp.async` is available. A prefill-only optimization (ZINC M3).
- **128-bit vectorized loads** — `float4`/`int4`/`uint4` → one 16-byte coalesced `LDG.E.128` per thread (analog of AMD `GLOBAL_LOAD_DWORDX4`). For DMMV, vectorize the packed-weight load to `int4` so a warp issues 512 B/load and saturates the GDDR path.

### Quantized Weight Unpacking (Q4_K, decode DMMV)

Q4_K stores 256 weights per super-block: 4-bit quants + per-sub-block 6-bit scales/mins, with FP16 `d`/`dmin`. The decode (matrix-vector, M=1) inner loop, mirroring the AMD doc but on the `__dp4a` integer path:

```
// One warp lane processing a Q4_K super-block (256 weights), decode M=1.
 1. LDG.E.128 (int4)     load 128 B of packed 4-bit quants (256 vals)   [~400 clk, pipelined]
 2. LDG.E (half2)        load d, dmin (FP16 super-block scales)          [pipelined]
 3. get_scale_min_k4()   unpack 6-bit per-sub-block sc[j], m[j]          [bit ops]
 4. AND / SHR / BFE      isolate low/high nibble -> int8 quant in [0,15] [BFE/AND/SHR]

 --- ZINC int8 path (preferred; maps to AMD v_dot4_i32_i8) ---
 5. pack 4× int8 quants into one 32-bit word         (q0..q3)
 6. pack 4× int8 activations (pre-quantized x)        (a0..a3)
 7. acc_i32 = __dp4a(q_word, a_word, acc_i32)         // 4 MACs / instruction, 1 IDP4A
 8. repeat 5-7 for the 64 dp4a calls covering 256 weights
 9. y += d*sc[j]*(float)acc_i32 - dmin*m[j]*sum_a     // apply scales once per sub-block

 --- FP fallback (activations in FP16) ---
 5'. cvt int8 q -> half ; __hfma2: acc2 += (d*sc)*q_half2 * x_half2     // 2 FMAs/instr

10. __reduce_add_sync(0xffffffff, acc)               // sm_80+, else __shfl_down_sync loop
11. lane 0:  STG.E  y[row]                            // one output element
```

The whole quant-DMMV family (Q4_K/Q5_K/Q6_K/Q8_0/F32) is variations of this: change the unpack (steps 3–4) and the accumulate width; the `__dp4a` reduce and the warp-reduce tail are shared. GGML block layouts (Q4_K/Q5_K/Q6_K 256-elem super-blocks; Q8_0/Q5_1/Q5_0 32-elem; MXFP4 32-elem) are bit-identical to the Vulkan path, so the in-shader unpack ports verbatim.

### ZINC porting cheat-sheet (Vulkan/GLSL → AMD → CUDA)

The int8 GEMM core is a literal one-to-one swap; the rest of the port is variations on this table (`docs/cuda-backend.md` §4):

| Vulkan / GLSL (ZINC) | AMD RDNA (`V_*`) | NVIDIA CUDA |
|---|---|---|
| `dotPacked4x8AccSatEXT` | `V_DOT4_I32_IU8` | `__dp4a` (`IDP4A`) |
| `subgroupAdd` / clustered | DPP `row_shr` + `V_ADD` | `__shfl_down_sync` / `__reduce_add_sync` |
| `subgroupMax` | DPP + `V_MAX_F32` | `__reduce_max_sync` / `__shfl_*` |
| `subgroupShuffle` / `BroadcastFirst` | `V_READLANE` / `DS_BPERMUTE` | `__shfl_sync` |
| `subgroupBallot` / `Elect` | `V_CMP` → exec mask | `__ballot_sync` / `__activemask` |
| `vec4`/`uvec4` SSBO (128-bit) | `GLOBAL_LOAD_DWORDX4` | `float4` / `int4` (`LDG.128`) |
| packed FP16 dot | `V_DOT2_F32_F16` / `V_PACK_B32_F16` | `__hfma2` (`HFMA2`) / `__half2` |
| LDS tile + `S_BARRIER` | `DS_WRITE` + `S_BARRIER_SIGNAL/WAIT` | `__shared__` + `cp.async` + `__syncthreads` |
| wave64 workgroup | wave64 SIMD | warp32 block (wave32 fallback path) |
| push constants | inline SGPR consts | kernel params / `__constant__` block |

## Cycle Costs

> NVIDIA does not publish per-instruction cycle counts or cache latencies for these consumer parts. The numbers below are **order-of-magnitude engineering estimates** from community microbenchmarks and the whitepapers' issue-rate descriptions — use them to reason about bottlenecks, not for cycle-accurate modeling. Verify on-device with Nsight Compute (`ncu`); see the Driver / Toolchain section for the exact commands.

### Instruction throughput (per SM, per clock)

| Instruction | Throughput / SM / clk | Latency (cyc) | Notes |
|---|---|---|---|
| `FFMA` (fma.f32) | 128 | ~4 | Core of all matmul; 128 FP32 lanes |
| `FADD` / `FMUL` (f32) | 128 | ~4 | Simple FP ALU |
| `HFMA2` (packed FP16) | 128 (= 256 FP16 FMAs) | ~4 | 2-wide packed — use for FP16 DMMV |
| `__dp4a` (`IDP4A`) | 64 (= 256 int8 MACs) | ~4 | INT32 datapath: 64 lanes/clk, 4 MACs each |
| `IADD3` / `IMAD` (s32) | 64 | ~4 | Shares the FP32-or-INT32 datapath |
| `cvt` (F2F/I2F) | ~64 | ~4 | Dequant conversions |
| `MUFU` (`__expf`/rcp/rsqrt/sqrt) | ~16–32 | ~20+ | SFU, shared per partition — softmax/RMS bottleneck |
| `SHFL` (warp shuffle) | 32 | ~20+ | Cross-lane reductions |
| `__reduce_*_sync` | 1 warp-reduce / few clk | — | HW reduce, `sm_80+` |
| Tensor `mma` FP16→FP32 m16n8k16 | (matrix op) | ~33 | Prefill GEMM only; not used by ZINC's `__dp4a` path |

> FP32 FFMA latency is ~4 cyc on consumer Ampere/Ada microbenchmarks; the A100 paper measured ~2 cyc on the GA100 datapath, so latency is datapath-dependent (treat as ~2–4, an estimate not a spec). SFU and shared latencies are approximate.

### Memory latency (microbenchmark, see Memory Hierarchy table)

`shared ~20–30 cyc · L1 ~32 cyc · L2 ~200–285 cyc · global ~290–570 cyc · PCIe ~µs.` Consumer Ada/Blackwell DRAM lands at the high cycle end because of the higher boost clock; absolute ns latency is closer to Ampere than the cycle counts imply. The Ampere column is an A100 (`sm_80`) measurement reused as the Ampere reference — consumer GA10x (`sm_86`) may differ.

### Sync / launch costs

| Operation | Cost | Notes |
|---|---|---|
| `__syncthreads()` (block barrier) | ~tens of cyc | Full block sync |
| `__syncwarp(mask)` | ~few cyc | Warp-level reconverge (Volta+ independent thread scheduling) |
| Single kernel launch (CPU-side) | ~µs-scale | Replaced by one CUDA Graph launch in the decode loop |
| CUDA Graph launch (whole graph) | one submit | Amortizes N per-token launches into one |
| Stream-ordered work | 0 (implicit order) | Same stream is ordered; cross-stream → `CUevent` |

The standard decode-loop optimization is to record the static per-token kernel sequence once and replay it with one `cudaGraphLaunch`, collapsing N per-launch CPU overheads into a single submit — the direct analog of replaying a pre-recorded Vulkan command buffer.

## Worked Examples

### Single-token DMMV (Q4_K) — RTX 4090

One output row of a decode projection: weight matrix W is [N×K], activation x is [K×1]. Take K=4096, FP16-equivalent reads, RTX 4090 (1008 GB/s).

```
Per output element: K multiply-adds (dot product), reading K weights once (M=1, no reuse).
  Bytes read (weights): N×K×2          FLOPs: 2×N×K
  Arithmetic intensity: 2·N·K / (2·N·K) = 1 FLOP/byte (FP16)

Roofline ridge (RTX 4090): 330e12 FLOP/s ÷ 1008e9 B/s ≈ 328 FLOP/byte.
  1 FLOP/byte ≪ 328  ⇒ deeply memory-bound. The 330 TFLOPS Tensor core is irrelevant.

Time for one [4096×4096] FP16 projection at decode:
  Bytes = 4096×4096×2 = 33.6 MB ;  t = 33.6e6 / 1008e9 ≈ 33 µs   (bandwidth floor)
  A Tensor-core GEMM would still read those 33.6 MB, then leave the cores ~99.7% idle.

Whole 7B model at Q4 (~4 GB resident weights streamed per token):
  RTX 5090: 1792 / 4 ≈ 448 tok/s ceiling      RTX 4090: 1008 / 4 ≈ 252 tok/s
  RTX 5070:  672 / 4 ≈ 168 tok/s              RTX 5060:  448 / 4 ≈ 112 tok/s
```

Feeding M=1 into a 16×8×16 `mma` tile uses 1 of 16 M-rows (6.25% tile utilization, zero bandwidth saved), so decode runs on a hand-tuned DMMV CUDA-core kernel (per-thread dot products, coalesced `int4` weight loads, dequant fused via `__dp4a`), Tensor cores idle. Quantization helps decode only by **shrinking weight bytes** (FP8 halves vs FP16, FP4 quarters), not by using Tensor math. Real decode lands at ~60–85% of ceiling after kernel/scheduling overhead and KV-cache reads; the large L2 (72/96 MB) amplifies effective bandwidth by holding the KV cache resident. Contrast: prefill with M=4096 has arithmetic intensity = M ≫ 328 ridge → compute-bound, and a [4096³] FP16 GEMM runs in ~0.42 ms on the 4090 Tensor cores (330 dense TFLOPS) vs ~1.66 ms on CUDA cores — 4× slower.

### Softmax (1024 elements, one warp-block) — CUDA terms

```
Phase 1 — max:   each lane loads 32 floats (coalesced 128 B lines); local FMAX;
                 warp max via __reduce_max_sync(0xffffffff, m)   // 1 HW reduce on sm_80+,
                                                                 // else 5× __shfl_down_sync
Phase 2 — exp:   (x - max) FADD; __expf -> MUFU.EX2 (~16-32/clk, the bottleneck, shared SFU);
                 warp sum via __reduce_add_sync(0xffffffff, s)
Phase 3 — norm:  1× MUFU.RCP (1/sum) + FMUL; coalesced store
```

Bottleneck is `MUFU.EX2` on the shared SFU. On `sm_61`/`sm_75` the two reductions become `log2(32)=5`-step `__shfl_down_sync` loops (~5× the reduction cost) — exactly the wave64→warp32 width change to validate numerically when porting (`docs/cuda-backend.md` §7). Fuse softmax into the preceding attention matmul to avoid a global round-trip.

### RMS norm (hidden_dim=4096) — CUDA terms

```
Phase 1 — sum of squares: each thread FFMA-accumulates x²; warp + block reduce via
                          __shfl_down_sync + __shared__
Phase 2 — normalize:      1× MUFU.RSQ (1/sqrt(mean_sq + eps)); 4096× FMUL by that scale;
                          4096× FMUL by the learned weight (the "MUL" in fused RMS_NORM_MUL);
                          coalesced store
```

Fusing RMS_NORM + weight-mul into one kernel saves a full global read+write pass over the hidden state — the same win as on AMD. ZINC's `rms_norm_mul` is kernel #2 in the M1 decode path (`docs/cuda-backend.md` §5).

## CUDA Graphs and Launch

Decode is a **static graph**: the same kernel sequence runs every token, only data pointers change — the exact case CUDA Graphs and Vulkan pre-recorded command buffers both target.

| | CUDA Graphs | Vulkan command buffer |
|---|---|---|
| Capture | `cudaStreamBeginCapture`/`EndCapture`, or explicit graph API | `vkBeginCommandBuffer` / record / `vkEndCommandBuffer` |
| Instantiate | `cudaGraphInstantiate` → executable graph | (recorded buffer is directly submittable) |
| Replay | `cudaGraphLaunch` (one CPU call for the whole graph) | `vkQueueSubmit` |
| Update | `cudaGraphExecUpdate` / node params for changed pointers | descriptor-set updates / push constants |
| Win | One submit for N kernels; whole-graph view enables driver scheduling | One submit for N dispatches, no re-recording |

Per-individual-launch CPU overhead is ~µs; replacing N launches/token with one graph launch is the standard decode-loop optimization. Use graph-update to swap KV pointers per token without re-capturing.

**Streams and events.** Streams are ordered queues; work in different non-default streams overlaps (compute/copy overlap, multi-stream batching). Use a non-default stream + pinned host memory for async H2D/D2H (`cudaMemcpyAsync`) to overlap weight upload with compute; use `CUevent` (`cudaEventRecord`/`cudaStreamWaitEvent`) for cross-stream sync and GPU timing rather than `cudaDeviceSynchronize`. ZINC mirrors Metal's `commitAsync`/`wait`/`releaseCompleted` onto a `CUstream` + `CUevent` pending-command ring (`docs/cuda-backend.md` §3).

**NVRTC vs offline nvcc.** `nvcc` drives AOT compile; `ptxas` assembles PTX → SASS (cubin) for one specific compute capability. A **cubin** loads instantly but runs only on that arch; **PTX** is a virtual ISA the driver JIT-compiles to SASS at load when no matching cubin is present (forward-compatible, one-time first-launch stall, cached in `~/.nv/ComputeCache`). **NVRTC** compiles a CUDA C++ *string* → PTX (or cubin) in-memory at runtime, loaded via `cuModuleLoadData(Ex)`. ZINC starts with **NVRTC runtime compilation** (mirrors Metal's `createPipeline(msl_source)`; kernels live as `.cu`/string sources, compiled to PTX for the running GPU's arch on load — handles `sm_120` vs `sm_89` transparently); an offline `nvcc → cubin` step is a later optimization (`docs/cuda-backend.md` §4, §6).

## Ampere vs Ada vs Blackwell (inference diff)

| Feature | Ampere (RTX 30, `sm_86`) | Ada (RTX 40, `sm_89`) | Blackwell (RTX 50, `sm_120`) | Inference impact |
|---|---|---|---|---|
| Tensor core gen | 3rd | 4th | 5th | new low-precision each gen |
| New low-precision | — (TF32/BF16) | **FP8 (E4M3/E5M2)** | **FP4, FP6 (NVFP4)** | ~2× dense Tensor per gen for quantized prefill |
| INT4/INT1 Tensor | Yes | Yes | **Dropped** | low-bit on Blackwell is FP4, not INT4 |
| Top-SKU L2 | 6 MB (3090) | 72 MB (4090) | 96 MB (5090) | KV-cache / weight reuse on-chip; effective-BW amplification |
| Memory | GDDR6X (PAM4) | GDDR6X (PAM4) | **GDDR7 (PAM3)** | 5090 1792 GB/s vs 4090 1008 (+78%) — lifts decode directly |
| Top-SKU bandwidth | 1008 GB/s (3090 Ti) | 1008 GB/s (4090) | **1792 GB/s (5090)** | single-stream decode ceiling |
| `cp.async` / warp HW-reduce | **Yes** (Ampere added) | Yes | Yes | tiled-GEMM prologue; 1-instr softmax/RMS reduce |
| TMA (`cp.async.bulk`) | No | No | **Yes (`sm_90+`/`sm_120`)** | prefill bulk transfer (M3) — *not* on Ada |
| PCIe | Gen 4 | Gen 4 | **Gen 5** | 2× host↔device — faster model load/offload |
| Process | Samsung 8N | TSMC 4N | TSMC 4N | higher clocks (>2.5 GHz), better perf/W |
| Compute capability | 8.6 | 8.9 | 12.0 | recompile to `sm_120`; FP4 needs CUTLASS 4.2+ |

For ZINC's two deployment cards: the **RTX 4090 (`sm_89`)** has FP8 Tensor and `cp.async` but no TMA and no FP4; the **RTX 5090 (`sm_120`)** adds FP4/NVFP4 Tensor, TMA, GDDR7, and the optional `mma.sync` Tensor-core GEMM win (M3). Both decode on the `__dp4a` DMMV path regardless of Tensor generation.

## Driver / Toolchain Config for Inference

### Build targets

`nvcc -arch=sm_86` (Ampere) / `sm_89` (Ada) / `sm_120` (Blackwell), or `-gencode arch=compute_XX,code=sm_XX`. A cubin built for one arch does not run on another; `sm_120` cubins are not compatible with data-center Blackwell `sm_100` or `sm_121`. FP8 (`__nv_fp8_*`) needs CUDA 12.x+; FP4/NVFP4 needs the `sm_120a` PTX target and CUTLASS 4.2+. NVRTC compiles per the running GPU's arch on load, so ZINC's `.cu` kernels handle `sm_120`/`sm_89` transparently without a per-arch build matrix.

### Inspecting register pressure, occupancy, and spills

This is the direct NVIDIA analog of AMD's `RADV_DEBUG=shaderstats` (VGPR count / occupancy / spill dump). Three tools, increasing fidelity:

**1. `ptxas -v` (compile-time, the fastest loop).** Pass `-Xptxas -v` to `nvcc` (or `--ptxas-options=-v`); NVRTC takes the same option via `nvrtcCompileProgram` flags. Per kernel, per target arch it prints registers, spills, stack frame, and constant-memory use:

```bash
nvcc -arch=sm_89 -Xptxas -v -cubin kernels.cu
# ptxas info : Compiling entry function 'dmmv_q4k' for 'sm_89'
# ptxas info : Function properties for dmmv_q4k
#     0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
# ptxas info : Used 38 registers, 96 bytes cmem[0]
```

Read it like `RADV_DEBUG=shaderstats`: **"Used N registers"** is registers/thread (cross-reference the occupancy table — 38 here caps at 42 warps on `sm_89`); **any non-zero "spill stores/loads"** means the kernel overflowed the register file to local memory (an L1/L2/DRAM round-trip per spill — the thing to eliminate on a hot decode kernel). Note the PTX register count bears little relation to the final SASS count, so always read the `ptxas` (SASS-stage) number, not the PTX.

**2. `cuobjdump --dump-resource-usage` (`-res-usage`, post-link).** Reads registers + shared-memory use back out of an already-built cubin or executable — useful when ZINC ships a cubin and you want to confirm what actually got linked:

```bash
cuobjdump -res-usage ./zig-out/bin/zinc        # per-kernel REG:/SHARED: per arch
cuobjdump -sass ./zig-out/bin/zinc | grep IDP4A # confirm __dp4a lowered to the int-dot SASS
```

**3. Nsight Compute (`ncu`, on-device, ground truth).** The only tool that reports **achieved** (not theoretical) occupancy and measured spill traffic on the real GPU — this is what "verify on-device with Nsight Compute" throughout this doc means:

```bash
ncu --set full -k dmmv_q4k ./zig-out/bin/zinc --model-id <id>     # full report, one kernel
ncu --section Occupancy --section LaunchStats ./zig-out/bin/zinc   # registers/thread + occupancy only
# Achieved Occupancy, Registers Per Thread, and the spill metric
# l1tex__data_pipe_lsu_wavefronts_mem_lg_cmd_read (local-memory reads = spill loads)
```

For a decode (DMMV) kernel the target is high achieved occupancy at ≤32 regs/thread with zero spills; for a prefill Tensor GEMM, occupancy matters less than Tensor-pipe utilization (`sm__pipe_tensor_op_*_cycles_active`).

### Device selection, persistence mode, MPS, clocks

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID    # device indices match nvidia-smi (default is FASTEST_FIRST)
export CUDA_VISIBLE_DEVICES=0,1        # restrict / reorder visible GPUs
sudo nvidia-smi -pm 1                  # persistence mode: driver stays loaded, cuts cold-context init
nvidia-cuda-mps-control -d             # MPS: concurrent multi-process sharing of one card
sudo nvidia-smi -c EXCLUSIVE_PROCESS   # recommended compute mode under MPS
sudo nvidia-smi -lgc <min>,<max>       # lock SM clocks ;  -pl <watts> sets the power limit
```

Set `CUDA_DEVICE_ORDER=PCI_BUS_ID` so indices match `nvidia-smi` and stay stable across runs (essential for pinning ranks; on ZINC's box `cuda:0`=5090, `cuda:1`=4090, but a non-login SSH does not inherit the export — select by cc/SM count at runtime if needed).

**Persistence mode (`nvidia-smi -pm 1`)** removes the cold-CUDA-context driver-init cost — the exact "cold/warm GPU" regime ZINC tracks on Metal, where the first process pays a one-time init each launch. *Measured impact: not yet benchmarked on ZINC's CUDA box.* The expected shape is a one-time per-process latency reduction (the driver context is already resident), with no steady-state tok/s change once warmed — so it matters for short-lived CLI invocations, not for a long-running server. Either way: **always warm up before measuring** (cuda-backend.md notes the CUDA cold path mirrors Metal's).

**MPS (`nvidia-cuda-mps-control -d`)** enables genuinely concurrent (not time-sliced) kernel execution from co-located processes sharing one card — relevant only when running several inference servers on one GPU. *Measured impact: not yet benchmarked on ZINC.* It does not speed up a single inference stream; the win is aggregate throughput when many small-batch clients would otherwise serialize on the default context.

**Clock/power locking (`-lgc`, `-pl`)** pins SM clocks / power for reproducible benchmarking — the NVIDIA analog of AMD's `RADV_PROFILE_PSTATE`, but note the AMD lesson: on memory-bound RDNA4 work, *forcing peak clocks regressed throughput ~23%* because the power budget got pulled from the memory subsystem to the (idle) shader cores. The same trap can exist on NVIDIA — pinning SM clocks high on a bandwidth-bound decode kernel can shift the power cap away from the memory controllers. *Not yet measured on ZINC's NVIDIA box; benchmark both ways before locking clocks for a decode workload.*

### GDDR ECC

GDDR6X/GDDR7 carry on-die error detection; **software/inline ECC** (selectable on the workstation cards, on by default there) reserves a fraction of memory and bandwidth for check bits. NVIDIA does not publish a single consumer figure; the standard inline-ECC bandwidth/capacity overhead is in the ~5–12% range depending on the scheme, comparable to the ~10% AMD measured for RDNA4 GECC. *Not yet measured on ZINC's NVIDIA box.* For inference where occasional bit flips are tolerable, leave ECC **off** (the consumer GeForce cards have no user-facing ECC toggle anyway — ECC is a workstation/data-center feature; on the RTX A6000 / 6000 Ada / PRO 6000 toggle it with `sudo nvidia-smi -e 0` then reboot, and re-measure tok/s the way AMD did for GECC: +9% on Qwen3 Q4_K when disabled).

### WSL2 specifics (the deployment constraint)

NVIDIA on WSL2 is a **paravirtualized GPU, CUDA-only**:

- The driver exposes CUDA + DirectX/DirectML but ships **no NVIDIA Vulkan ICD** — `nvidia_icd.json` is absent, so `vulkaninfo` falls back to `llvmpipe` (CPU). Vulkan compute backends do **not** see the NVIDIA GPU under WSL2 — the CUDA backend is the only path that touches the hardware (proven empirically in `docs/cuda-backend.md` §1).
- `/dev/dxg` is the only GPU node (proxies work to the Windows kernel driver via `dxgkrnl`/WDDM); there are **no `/dev/nvidia*`** nodes.
- `libcuda.so.1` is a **stub** in `/usr/lib/wsl/lib/` (auto-mounted; `nvidia-smi` lives at `/usr/lib/wsl/lib/nvidia-smi`) that forwards to the real Windows driver. **Do not install a Linux NVIDIA display driver inside WSL** — install only the Windows driver + the CUDA toolkit (without its driver) in WSL.
- Profiling caveat: `ncu`/Nsight Compute works under WSL2 but needs the matching Windows driver + the WSL CUDA toolkit; `nvidia-smi -pm`/`-lgc`/`-e` (persistence, clock-lock, ECC) are managed by the Windows driver and may be no-ops or unavailable from inside the WSL guest — set them on the Windows host.

Contrast with AMD/RDNA on native Linux, where RADV Vulkan is the primary path; on the WSL2 NVIDIA box a Vulkan inference path is a non-starter.

## Official Documentation Links

### Architecture whitepapers (primary spec source)
- [NVIDIA Ampere GA102 GPU Architecture Whitepaper (v2.1, PDF)](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.1.pdf)
- [NVIDIA Ada GPU Architecture Whitepaper (v2.02, PDF)](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf)
- [NVIDIA RTX Blackwell GPU Architecture Whitepaper (v1.1, PDF)](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)
- [NVIDIA RTX PRO 6000 Blackwell Workstation Edition Datasheet (PDF)](https://www.nvidia.com/content/dam/en-zz/Solutions/data-center/rtx-pro-6000-blackwell-workstation-edition/workstation-blackwell-rtx-pro-6000-workstation-edition-nvidia-us-3519208-web.pdf)

### CUDA / PTX programming references
- [CUDA C++ Programming Guide — Compute Capabilities](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities)
- [PTX ISA — `mma`, `wgmma`, `ldmatrix`, `dp4a`, `cp.async`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
- [CUDA Math API — SIMD Intrinsics (`__dp4a`, `__dp2a`)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SIMD.html)
- [Ampere Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html) · [Ada Tuning Guide](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html) · [Blackwell Compatibility Guide](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/)
- [Using CUDA Warp-Level Primitives (NVIDIA blog)](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/)
- [Controlling Data Movement on Ampere — `cp.async` (NVIDIA blog)](https://developer.nvidia.com/blog/controlling-data-movement-to-boost-performance-on-ampere-architecture/)

### Toolchain & deployment
- [NVRTC (Runtime Compilation)](https://docs.nvidia.com/cuda/nvrtc/index.html) · [NVCC](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/nvcc.html)
- [CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html) · [CUDA Environment Variables](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html)
- [CUDA Binary Utilities — `cuobjdump` / `nvdisasm` (`-res-usage`, `-sass`)](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html)
- [Nsight Compute CLI (`ncu`) — sections, sets, metrics](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)
- [nvidia-smi](https://docs.nvidia.com/deploy/nvidia-smi/index.html) · [Multi-Process Service (MPS)](https://docs.nvidia.com/deploy/mps/) · [CUDA on WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

### Product pages, per-SKU databases, deep dives
- [GeForce RTX 50 series](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/) · [RTX 40 series](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/) · [RTX 30 series](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/)
- [CUDA GPUs — Compute Capability list](https://developer.nvidia.com/cuda-gpus) · [TechPowerUp GPU Database](https://www.techpowerup.com/gpu-specs/)
- [Demystifying the Nvidia Ampere Architecture (arXiv 2208.11174)](https://arxiv.org/abs/2208.11174) · [Dissecting the Nvidia Hopper GPU (arXiv 2402.13499)](https://arxiv.org/abs/2402.13499) · [Microbenchmarking the RTX 4090 — Chips and Cheese](https://chipsandcheese.com/p/microbenchmarking-nvidias-rtx-4090)
