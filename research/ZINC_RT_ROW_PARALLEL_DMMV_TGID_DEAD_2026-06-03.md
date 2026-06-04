# ZINC_RT row-parallel DMMV TGID attempt - measured dead

Date: 2026-06-03
Node: RDNA4 R9700, Linux 6.17.0-29-generic, `amdgpu` CS path
Model: `/root/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`

## Scope

The existing consumed direct DMMV row-range kernels in
`src/zinc_rt/ring/cs.zig` are correctness probes: one workitem serially walks a
compact F32/Q4_0/Q8_0 row range. This cycle tested whether that path could be
promoted to one dispatched workgroup per row while keeping the same consumed
router, SSM, and LM-head validators.

## Variants Tested

All variants passed the remote build gate before runtime smoke:

```bash
cd /root/zinc
zig build test -Dbackend=zinc_rt -Dshaders=false --summary all
zig build -Doptimize=ReleaseFast -Dbackend=zinc_rt -Dshaders=false
```

Runtime smoke used:

```bash
RADV_PERFTEST=coop_matrix ./zig-out/bin/zinc \
  -m /root/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  --prompt 'The capital of France is' \
  --max-tokens 8
```

Tested row-id layouts:

- `PGM_RSRC2=0x90`, row id read from `s8`
- `PGM_RSRC2=0x88`, row id read from `s8`
- `PGM_RSRC2=0x90`, row id read from `s16`

## Remote Evidence

Each row-parallel variant left DMMV outputs at the NaN sentinel. Representative
failure lines from the final `s16` run:

```text
warning(zinc_rt_forward): M1 AMDGPU CS direct router row-range produced non-finite row 0; router logits remain host-computed
warning(zinc_rt_forward): M1 AMDGPU CS direct LM-head Q4_0 argmax-prefix produced non-finite row 0; selected token remains host-computed
warning(zinc_rt_forward): M1 AMDGPU CS direct LM-head Q4_0 best-row produced non-finite row 11751; selected logit remains host-computed
```

The run completed coherently, but only scalar probes were consumed:

```text
direct_compute_ops=3 direct_compute_kind=argmax_rms_norm_elem0
direct_decode_model_slices=0 real_model_slice=0
```

This means the PM4 admission, scalar `rms_norm_elem0`, and `argmax_top2` probes
still retire, but the row-parallel DMMV shader does not produce visible row 0
stores under the current ABI/packet stream.

## Decision

Do not ship row-parallel DMMV as the default path. It removes the benchmark's
existing consumed `dmmv_row_range` evidence and downgrades execution back to
`host_assisted_direct_probe`. The working serial DMMV row-range code was
restored after this measurement.

## Next Probes

1. Add a tiny SGPR/TGID dump kernel that stores candidate SGPRs for several
   workgroups before attempting DMMV math again.
2. Add a real `ACQUIRE_MEM`/GL2 writeback packet after multi-workgroup dispatch
   if the SGPR dump proves TGID delivery is correct.
3. Keep any row-parallel DMMV retry behind an opt-in env flag until row 0 is
   finite and the LM-head Q4_0 prefix validator reports `dmmv_row_range`.
