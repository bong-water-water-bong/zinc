# ZINC_RT recurring CS model-slice substitution - measured dead

Date: 2026-06-14
Node: RDNA4 R9700, Linux 6.17.0-35-generic, `amdgpu` CS path
Model: `/root/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`
Prompt: `The capital of France is`
Max tokens: 96

## Scope

This measured-dead note covers the current in-tree direct model-slice path:
`src/zinc_rt/ring/cs.zig` submits gfx1201 row-range kernels through
`DRM_IOCTL_AMDGPU_CS`, stages activations and source-format weight windows into
a GTT input BO, waits through `DRM_IOCTL_AMDGPU_WAIT_CS`, then reads a GTT
output BO.

The path is correct enough for M1 evidence. It is not the final design-doc T1
or T2 path: every row-range batch still pays host staging plus a kernel-managed
CS submit/wait. The question tested here was whether increasing recurring
consumed slices can improve the default 96-token decode throughput by replacing
enough CPU matvec rows.

## Remote evidence

Current harness dry-run confirmation (`bun loops/zinc_rt_autopilot.ts --dry-run`,
2026-06-14):

```text
AB: vk=110.1✓ rt=35.1✓ ratio=0.319
```

The current default shortcut-free run after syncing the local tree to
`/root/zinc`:

```bash
RADV_PERFTEST=coop_matrix \
./zig-out/bin/zinc \
  -m /root/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  --prompt 'The capital of France is' \
  --max-tokens 96
```

Observed result:

```text
Generated 96 tokens in 2733.0 ms - 35.13 tok/s
direct_compute_ops=136
direct_decode_model_slices=65
direct_compute_kind=dmmv_row_range
real_model_slice=1
shortcut_free=1
benchmark_shortcuts=none
```

Disabling the direct tier entirely is effectively identical for decode:

```bash
RADV_PERFTEST=coop_matrix \
ZINC_RT_TIER=t_cpu \
./zig-out/bin/zinc \
  -m /root/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  --prompt 'The capital of France is' \
  --max-tokens 96
```

Observed result:

```text
Generated 96 tokens in 2733.2 ms - 35.12 tok/s
direct_compute_ops=0
direct_decode_model_slices=0
model_execution=cpu_fallback
shortcut_free=1
benchmark_shortcuts=none
```

That establishes that the current first-token LM-head proof is not a material
throughput tax or win.

The every-token broad direct-slice substitution is much slower:

```bash
RADV_PERFTEST=coop_matrix \
ZINC_RT_DIRECT_DECODE_FULL_SLICE=1 \
ZINC_RT_DIRECT_DECODE_SLICE_CADENCE=1 \
./zig-out/bin/zinc \
  -m /root/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  --prompt 'The capital of France is' \
  --max-tokens 96
```

Observed result:

```text
Generated 96 tokens in 5665.9 ms - 16.94 tok/s
direct_compute_ops=10521
direct_decode_model_slices=10355
real_model_slice=1
shortcut_free=1
benchmark_shortcuts=none
```

The narrower recurring router substitution also regresses:

```bash
RADV_PERFTEST=coop_matrix \
ZINC_RT_DIRECT_ROUTER_DECODE=1 \
ZINC_RT_DIRECT_ROUTER_DECODE_CADENCE=0 \
./zig-out/bin/zinc \
  -m /root/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  --prompt 'The capital of France is' \
  --max-tokens 96
```

Observed result:

```text
Generated 96 tokens in 2918.1 ms - 32.90 tok/s
direct_compute_ops=516
direct_decode_model_slices=445
op=router_q8_0_row_range_parallel64_trusted_primary
real_model_slice=1
shortcut_free=1
benchmark_shortcuts=none
```

All three runs produced the same coherent text prefix and stayed
shortcut-free. The regressions therefore come from execution cost, not output
quality or benchmark shortcuts.

## Conclusion

Recurring model-slice substitution through the current kernel-managed CS path
is measured dead as a throughput strategy on this node. The current slices are
useful correctness evidence, but they do not amortize:

- each batch copies activations and a source-format weight window through a GTT
  staging BO;
- each batch still enters the kernel through `DRM_IOCTL_AMDGPU_CS`;
- each batch waits through `DRM_IOCTL_AMDGPU_WAIT_CS`;
- the replaced CPU work is too small for router and partial MoE/SSM prefixes,
  while broad recurring slices multiply the submit/staging cost.

This does not prove direct execution is impossible on R9700. It proves that
more recurring coverage on the existing GTT-staged CS verifier path cannot move
the default benchmark toward Vulkan parity.

## Do not repeat

- Do not enable `ZINC_RT_DIRECT_DECODE_FULL_SLICE=1` at per-token cadence as a
  performance change.
- Do not enable `ZINC_RT_DIRECT_ROUTER_DECODE=1` with cadence `0` as a
  performance change.
- Do not count higher `direct_decode_model_slices` as progress unless the
  96-token shortcut-free A/B improves by the post-coverage throughput gate.

## Next useful work

1. Move one high-value row range to persistent device-local or BAR-resident
   weights so the direct path no longer memcpy-stages the weight window through
   GTT on every token.
2. Revisit true T1/T2 doorbell submission only when the queue path can retire a
   shader-written signal, not just admission or CP packet smokes.
3. If staying on `DRM_IOCTL_AMDGPU_CS`, batch multiple consumed model ranges
   from the same token into one IB only when they share the same staged input
   and replace enough CPU rows to beat the extra staging bytes.
