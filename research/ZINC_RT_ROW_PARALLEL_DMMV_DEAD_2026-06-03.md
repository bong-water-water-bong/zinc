# ZINC_RT row-parallel DMMV CS attempt - measured dead

Date: 2026-06-03
Node: RDNA4 R9700, Linux 6.17.0-29-generic, `amdgpu` CS path
Model: `/root/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`

## What was tried

The existing consumed direct DMMV row-range kernels in `src/zinc_rt/ring/cs.zig`
compute a compact Q4_0/Q8_0 row range with one workitem serially looping over
rows. This attempt added gfx1201 GAS kernels that consume `workgroup_id_x` as
the row id and dispatch `rows` workgroups:

- `src/zinc_rt/isa/gfx1201/dmmv_q4_0_row_range_parallel.s`
- `src/zinc_rt/isa/gfx1201/dmmv_q8_0_row_range_parallel.s`

The PM4 stream was also tested with a Mesa-shaped
`EVENT_WRITE(CS_PARTIAL_FLUSH)` packet after `DISPATCH_DIRECT(rows, 1, 1)`.
Mesa's register database gives `CS_PARTIAL_FLUSH = 7`, and its event-write
macro uses event index 4 for `*_PARTIAL_FLUSH`.

The row-parallel path is retained only behind:

```bash
ZINC_RT_DIRECT_DMMV_ROW_PARALLEL=1
```

Default runs stay on the known-good serial direct DMMV row-range path.

## Remote evidence

Build gate passed after the diagnostic path was added:

```bash
cd /root/zinc
zig build test -Dbackend=zinc_rt -Dshaders=false --summary all
zig build -Doptimize=ReleaseFast -Dbackend=zinc_rt -Dshaders=false
```

Runtime smoke with the row-parallel path enabled:

```bash
RADV_PERFTEST=coop_matrix \
ZINC_RT_MAX_DECODE_TOKENS=2 \
ZINC_RT_DIRECT_SSM_Q8_ROW_RANGE_MAX_SUCCESSES=2 \
ZINC_RT_DIRECT_DMMV_ROW_PARALLEL=1 \
./zig-out/bin/zinc \
  -m /root/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  --prompt 'The capital of France is' \
  --max-tokens 2
```

Observed failure lines:

```text
warning(zinc_rt_forward): M1 AMDGPU CS direct LM-head Q4_0 argmax-prefix produced non-finite row 0; selected token remains host-computed
warning(zinc_rt_forward): M1 AMDGPU CS direct LM-head Q4_0 best-row produced non-finite row 11751; selected logit remains host-computed
warning(zinc_rt_forward): M1 AMDGPU CS direct LM-head Q4_0 argmax-prefix produced non-finite row 0; selected token remains host-computed
warning(zinc_rt_forward): M1 AMDGPU CS direct LM-head Q4_0 best-row produced non-finite row 11; selected logit remains host-computed
```

The output page was initialized with NaN sentinels before submission, so
`non-finite row 0` means the row-parallel Q4_0 shader did not produce a visible
store for the first row. The following did still retire correctly in the same
run:

- T1 KFD admission smoke
- AMDGPU CS `WRITE_DATA` smoke
- prefill direct F32 router row-range
- `rms_norm_elem0`
- `argmax_top2`

## Conclusion

The blocker is not model math accuracy. The row-parallel Q4_0 dispatch fails
before validation can compare numeric deltas: row outputs remain at the NaN
sentinel even after inserting `EVENT_WRITE(CS_PARTIAL_FLUSH)` before the signal
write. The active default path must not use this variant yet because it would
remove decode-phase consumed LM-head/SSM direct row-range evidence from the
benchmark.

## Next useful probes

1. Add a tiny opt-in SGPR dump kernel that stores `s8..s11` for several
   workgroups, to prove whether `workgroup_id_x` is actually delivered where
   the ABI assumes it is.
2. If TGID delivery is correct, add a real `ACQUIRE_MEM`/GL2 writeback packet
   after dispatch; `CS_PARTIAL_FLUSH` alone was insufficient.
3. Keep row-parallel DMMV behind `ZINC_RT_DIRECT_DMMV_ROW_PARALLEL=1` until a
   smoke run shows finite row 0 and a passing LM-head Q4_0 prefix validation.
