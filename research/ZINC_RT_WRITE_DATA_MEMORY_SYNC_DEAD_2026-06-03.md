# ZINC_RT WRITE_DATA memory-sync signal attempt - measured dead

Date: 2026-06-03
Node: RDNA4 R9700, Linux 6.17.0-29-generic, `amdgpu` CS path
Model: `/root/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`

## Scope

The consumed direct DMMV row-range path currently records:

1. `DISPATCH_DIRECT`
2. `WRITE_DATA` 64-bit signal to the GTT signal page, with `dst_sel=5`
   (`memory async/direct`) and `WR_CONFIRM=1`
3. `DRM_IOCTL_AMDGPU_WAIT_CS`
4. CPU reads the signal page and direct output page

The row-parallel DMMV dead reports left open whether a stronger post-dispatch
signal could make shader stores visible before the CPU readback. This attempt
tested the least invasive packet change: keep the same 64-bit signal ABI, but
write the post-dispatch sentinel through the PM4 `WRITE_DATA` memory-sync path
(`dst_sel=1`, memory sync via GRBM) instead of the async direct memory path.

## What was changed for the test

`src/zinc_rt/ring/packet.zig` temporarily gained:

```zig
pub fn writeData64MemorySync(...)
```

with packet control bits:

```text
dst_sel = 1 << 8
WR_CONFIRM = 1 << 20
engine_sel = 0 << 30
```

All shader-backed direct compute signals in `src/zinc_rt/ring/cs.zig` were
temporarily switched to this helper:

- `argmaxTop2`
- `argmaxF32Range`
- `rmsNormElement0`
- `dmmvF32RowRange`
- `dmmvQ4_0RowRange`
- `dmmvQ4_0ArgmaxRowRange`
- `dmmvQ8_0RowRange`
- `dmmvQ8_0TwoRowRanges`

The simple CP-only `COPY_DATA` token-boundary path was intentionally left on
the original async `writeData64`.

## Remote evidence

The required build gate passed before runtime smoke:

```bash
cd /root/zinc
zig build test -Dbackend=zinc_rt -Dshaders=false --summary all
zig build -Doptimize=ReleaseFast -Dbackend=zinc_rt -Dshaders=false
```

Runtime smoke command:

```bash
RADV_PERFTEST=coop_matrix \
ZINC_RT_MAX_DECODE_TOKENS=2 \
./zig-out/bin/zinc \
  -m /root/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  --prompt 'The capital of France is' \
  --max-tokens 2
```

Observed failure lines:

```text
warning(zinc_rt_forward): M1 AMDGPU CS direct router row-range unavailable (SignalMismatch); router logits remain host-computed
warning(zinc_rt_forward): M1 AMDGPU CS direct rms_norm elem0 unavailable (SignalMismatch); final norm element remains host-computed
warning(zinc_rt_forward): M1 AMDGPU CS direct LM-head Q4_0 argmax-prefix unavailable (SignalMismatch); selected token remains host-computed
warning(zinc_rt_forward): M1 AMDGPU CS direct LM-head Q4_0 best-row unavailable (SignalMismatch); selected logit remains host-computed
warning(zinc_rt_forward): M1 AMDGPU CS direct argmax compute unavailable (SignalMismatch); first token remains host-selected
```

Final summary from the same run:

```text
model_execution=cpu_fallback
direct_compute_ops=0
direct_compute_kind=none
direct_decode_model_slices=0
real_model_slice=0
```

The CP-only token boundary still worked in that run:

```text
direct_token_boundary=amdgpu_cs_copy_data
copies=1
```

## Conclusion

The 64-bit `WRITE_DATA` memory-sync packet shape is not a safe replacement for
the current async direct signal ABI on the bench node. It prevents the signal
sentinel from matching and removes all shader-backed direct model-slice
evidence from the benchmark.

The working serial direct DMMV path was restored to `writeData64` with
`dst_sel=5`. Do not retry this exact `writeData64MemorySync` substitution as a
row-parallel visibility fix.

## Next useful probes

1. Test a 32-bit memory-sync availability write in isolation before using it as
   a model-slice fence; Mesa examples commonly use a 32-bit availability word.
2. Add a shader-written signal pointer to a tiny TGID/SGPR dump kernel, so the
   completion sentinel is produced by the shader rather than by a following CP
   packet.
3. If packet-side polling is needed, use an opt-in `WAIT_REG_MEM` probe with a
   bounded diagnostic run; do not put an unbounded memory wait on the default
   direct model-slice path.

## References

- AMDGPU PM4 header field definitions:
  https://codebrowser.dev/linux/linux/drivers/gpu/drm/amd/amdgpu/sid.h.html
- GFX12 `WAIT_REG_MEM` packet emission reference:
  https://codebrowser.dev/linux/linux/drivers/gpu/drm/amd/amdgpu/gfx_v12_0.c.html
