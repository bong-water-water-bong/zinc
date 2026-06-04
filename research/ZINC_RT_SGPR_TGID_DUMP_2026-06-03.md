# ZINC_RT SGPR/TGID dump - measured blocker

Date: 2026-06-03
Node: RDNA4 R9700, Linux 6.17.0-29-generic, `amdgpu` CS path
Model: `/root/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`

## Scope

Row-parallel direct DMMV variants left output rows at NaN sentinels. This probe
added an opt-in shader-written SGPR/TGID dump behind:

```bash
ZINC_RT_DIRECT_TGID_DUMP=1
```

The diagnostic kernel stores candidate SGPRs `s8..s19` into rows keyed by any
candidate value `< groups`, and writes the completion signal from shader code
instead of relying on a following CP `WRITE_DATA`.

## Remote Evidence

Build gate passed first:

```bash
cd /root/zinc
zig build test -Dbackend=zinc_rt -Dshaders=false --summary all
zig build -Doptimize=ReleaseFast -Dbackend=zinc_rt -Dshaders=false
```

Runtime smoke:

```bash
RADV_PERFTEST=coop_matrix \
ZINC_RT_DIRECT_TGID_DUMP=1 \
ZINC_RT_MAX_DECODE_TOKENS=1 \
./zig-out/bin/zinc \
  -m /root/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  --prompt 'The capital of France is' \
  --max-tokens 1
```

Observed diagnostic lines:

```text
info(zinc_rt_forward): M1 AMDGPU CS SGPR/TGID dump: groups=4 shader_signal_ok=1 signal=0x5a494e435254b001 expected=0x5a494e435254b001 marker=0x53475052
info(zinc_rt_forward): M1 AMDGPU CS SGPR/TGID dump row=0: s8=0xffffffff s9=0xffffffff s10=0xffffffff s11=0xffffffff s12=0xffffffff s13=0xffffffff s14=0xffffffff s15=0xffffffff s16=0xffffffff s17=0xffffffff s18=0xffffffff s19=0xffffffff
info(zinc_rt_forward): M1 AMDGPU CS SGPR/TGID dump row=1: s8=0xffffffff s9=0xffffffff s10=0xffffffff s11=0xffffffff s12=0xffffffff s13=0xffffffff s14=0xffffffff s15=0xffffffff s16=0xffffffff s17=0xffffffff s18=0xffffffff s19=0xffffffff
info(zinc_rt_forward): M1 AMDGPU CS SGPR/TGID dump row=2: s8=0xffffffff s9=0xffffffff s10=0xffffffff s11=0xffffffff s12=0xffffffff s13=0xffffffff s14=0xffffffff s15=0xffffffff s16=0xffffffff s17=0xffffffff s18=0xffffffff s19=0xffffffff
info(zinc_rt_forward): M1 AMDGPU CS SGPR/TGID dump row=3: s8=0xffffffff s9=0xffffffff s10=0xffffffff s11=0xffffffff s12=0xffffffff s13=0xffffffff s14=0xffffffff s15=0xffffffff s16=0xffffffff s17=0xffffffff s18=0xffffffff s19=0xffffffff
```

The same run still consumed a real model slice and emitted coherent text:

```text
info(zinc_rt): Output text:  Paris
info(zinc_rt): ZINC_RT M1 model_execution=host_assisted_model_slice execution_tier=t1_pm4 driver=amdgpu_cs vulkan=0 ... direct_compute_kind=dmmv_row_range ... real_model_slice=1 shortcut_free=1 benchmark_shortcuts=none
```

## Conclusion

The shader dispatch retired and shader-written memory was visible: both the
signal and fixed marker matched. The row-parallel DMMV failure is therefore not
explained by CP-side completion signaling alone.

None of the candidate SGPRs `s8..s19` held workgroup ids `0..3` under the
current `COMPUTE_PGM_RSRC2=0x90` packet shape. The previous row-parallel DMMV
kernels that used `s8` as `workgroup_id_x` cannot advance on this ABI.

## Next Useful Work

1. Find the correct gfx1201 PM4/system-SGPR enable bits for workgroup ids, using
   Mesa/RADV register programming as the reference.
2. Repeat the SGPR dump with corrected `COMPUTE_PGM_RSRC2` before touching DMMV
   math again.
3. Only re-enable row-parallel DMMV after the dump shows a candidate SGPR
   containing `0,1,2,3` for a four-workgroup dispatch.
