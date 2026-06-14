# ZINC_RT T1/T2 direct queue advance - measured blocked

Date: 2026-06-14
Node class: RDNA4 R9700, Linux 6.17.0-35-generic
Model: `/root/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`
Prompt: `The capital of France is`
Max tokens: 96

## Scope

This note covers the direct-execution substrate, not another row-range coverage
tweak. The current default benchmark already consumes a small real model slice:
a 64-row LM-head Q4_0 prefix on the first decode token and one full router
Q8_0 row range every 64 decode tokens. Those slices are useful M1 evidence, but
they retire through the kernel-managed `DRM_IOCTL_AMDGPU_CS` verifier path.

The question for this cycle was whether the next M1 step can move to the true
direct queues described by `docs/ZINC_RT_DESIGN.md`, or whether more work on
the current CS verifier path is known-dead without changing substrate or
amortization.

## Fresh cycle verification

`bun loops/zinc_rt_autopilot.ts --dry-run` was run from the repo root after the
design and effort notes were re-read. It rsynced the current tree to the RDNA4
node, built both backends, and reproduced the same execution gap:

```text
vulkan median decode: 109.7 tok/s
zinc_rt median decode: 35.3 tok/s
ratio: 0.322
```

The synced `zinc_rt` run remained coherent and shortcut-free, but still reported
`model_execution=host_assisted_model_slice`, `driver=amdgpu_cs`,
`direct_compute_kind=dmmv_row_range`, and `direct_decode_model_slices=5`.
That means this cycle did not uncover a hidden M1 direct execution path already
available behind the current flags.

## Remote substrate evidence

Host capability check:

```text
kernel=6.17.0-35-generic
user_queue=-1
render_nodes=/dev/dri/renderD128 /dev/dri/renderD129
```

Forced T2 UMQ probe:

```bash
ZINC_RT_TIER=t2_umq ./zig-out/bin/zinc --probe-tier
```

Result:

```text
ZINC_RT M1 runtime initialized (tier=t2_umq driver=amdgpu_umq vulkan=0)
T2 UMQ admission failed: status=compute_userq_unavailable query=compute_userq_slots_missing
```

Conclusion: kernel 6.17 is not enough on this node. The firmware/kernel query
still reports no usable compute user-queue slots, so T2 cannot lower decode
packets until `USERQ_CREATE` for compute actually succeeds.

Forced T1 PM4 probe:

```bash
ZINC_RT_TIER=t1_pm4 ./zig-out/bin/zinc --probe-tier
```

Result, summarized to avoid machine-local addresses:

```text
ZINC_RT M1 runtime initialized (tier=t1_pm4 driver=amdgpu_cs vulkan=0)
T1 KFD compute queue admission passed: AMDKFD_IOC_CREATE_QUEUE ... PM4 NOP staged in ring ... AMDKFD_IOC_DESTROY_QUEUE OK
AMDGPU CS compute-ring PM4 WRITE_DATA retired with persistent BO list ... wait_status=0
```

Conclusion: the KFD part proves create/destroy admission and BO sizing only.
In-tree `src/zinc_rt/ring/kfd.zig` intentionally stops before ringing the
doorbell and retiring a fence; the current retiring compute path is still
`src/zinc_rt/ring/cs.zig`, which submits IBs through `DRM_IOCTL_AMDGPU_CS` and
waits with `DRM_IOCTL_AMDGPU_WAIT_CS`.

## Benchmark-visible evidence

Default ZINC_RT run after syncing this tree:

```text
Generated 96 tokens in 2711.0 ms - 35.41 tok/s
model_execution=host_assisted_model_slice
execution_tier=t1_pm4
driver=amdgpu_cs
direct_compute_ops=5
direct_compute_kind=dmmv_row_range
direct_decode_model_slices=5
consumed_gpu_model_value=1
real_model_slice=1
shortcut_free=1
benchmark_shortcuts=none
```

Forced CPU tier on the same prompt:

```text
Generated 96 tokens in 2711.2 ms - 35.41 tok/s
model_execution=cpu_fallback
execution_tier=t_cpu
driver=cpu
direct_compute_ops=0
direct_decode_model_slices=0
consumed_gpu_model_value=0
real_model_slice=0
shortcut_free=1
benchmark_shortcuts=none
```

The default consumed CS slices do not move 96-token decode throughput versus
the CPU fallback path. That is acceptable for correctness evidence, but it is
not an execution strategy that can close the 35 tok/s versus 110 tok/s gap by
adding more probes.

Every-token LM-head prefix substitution was also tested without changing code:

```bash
RADV_PERFTEST=coop_matrix \
ZINC_RT_DIRECT_LM_HEAD_DECODE_CADENCE=1 \
./zig-out/bin/zinc-zinc_rt \
  -m /root/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  --prompt 'The capital of France is' \
  --max-tokens 96
```

Result:

```text
Generated 96 tokens in 2752.1 ms - 34.88 tok/s
direct_compute_ops=178
direct_decode_model_slices=178
direct_compute_kind=dmmv_row_range
last_fence=180
consumed_gpu_compute_value=1
consumed_gpu_model_value=1
real_model_slice=1
shortcut_free=1
benchmark_shortcuts=none
```

The output text and first 20 tokens matched the default coherent run, but the
extra retired CS row-range work was slower than the 35.3-35.4 tok/s default.
That rules out simply increasing LM-head prefix cadence on the current
GTT-staged CS path as a keepable M1 execution step.

This agrees with `research/ZINC_RT_CS_RECURRING_SLICE_DEAD_2026-06-14.md`:
recurring per-token direct slices through the current GTT-staged
`DRM_IOCTL_AMDGPU_CS` path were measured at 16.94 tok/s for broad decode slices
and 32.90 tok/s for every-token router replacement. The overhead is submission,
staging, and fence retirement, not model quality.

## Decision

Direct execution cannot advance on this node by increasing current verifier
coverage:

- T2 UMQ is blocked before `USERQ_CREATE` by missing compute user-queue slots.
- T1 KFD is currently an admission smoke, not a retired shader-dispatch path.
- The only retired GPU model slices run through kernel-managed CS, and that
  path has already been measured throughput-dead for recurring substitution.

The next keepable M1 change should therefore change one of these facts:

1. make KFD ring submission retire a shader-written signal from a real queue;
2. make T2 `USERQ_CREATE` pass on a node/kernel that exposes compute slots;
3. move CS slices to persistent resident inputs/weights and batch enough work
   per fence to beat the CPU rows they replace.

Do not spend another cycle only increasing `direct_decode_model_slices`,
changing CS row cadence, or adding PM4 markers. Those can improve confidence in
the verifier, but they do not create competitive M1 execution on this node.
