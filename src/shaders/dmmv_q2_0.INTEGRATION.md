# Q2_0 (1.58-bit ternary) DMMV — integration recipe

The kernel is done: `src/shaders/dmmv_q2_0.comp` (SPIR-V-validated with glslc).
It implements the prism-ml Q2_0 format used by `Ternary-Bonsai-*-gguf`, verified
bit-exact vs F16 (cos = 1.000000; see `q2_0_decode.py` at repo root / the
1bit-systems `tools/q2_0_decode.py`).

Measured target (reference llama.cpp Vulkan, same GPU): **278.8 tok/s** decode,
coherent — vs 22 tok/s for the same model in F16. This kernel is the path to that
number natively in ZINC.

## Format (verified)
`block_q2_0`: 128 elements / 34 bytes. fp16 `d` at bytes 0-1, then `qs[32]`
(2-bit codes, LSB-first, 4/byte). `value(j) = (int(code)-1) * d`, code∈{0,1,2,3}→{-1,0,+1,+2}.

## ⚠️ Type-id collision (the one real gotcha)
GGML type id **42** is already `stq1_0` in `gguf.zig` (AngelSlim "Sherry",
256 elems / 42 bytes — a DIFFERENT format). Do NOT reuse the stq1_0 path.
Disambiguate by on-disk block geometry: a type-42 tensor is Q2_0 iff its byte
span == (nelem/128)*34; it is stq1_0 iff == (nelem/256)*42.

## Wiring (mirror the existing stq1_0 spots)
1. `build.zig` (~line 208): add `"dmmv_q2_0",` to the shader list.
2. `gguf.zig`: add enum `q2_0` (internal marker, not a raw GGUF id) with
   `blockSize()=>128`, `bytesPerBlock()=>34`.
3. `gguf.zig` loader (~line 407, `@enumFromInt`): when raw id==42, pick `.q2_0`
   vs `.stq1_0` by matching the tensor's actual data size to the two geometries.
4. `dmmv.zig`: add `pipeline_q2_0` field (~333), load `dmmv_q2_0.spv` (~898),
   assign (~2115), dispatch `.q2_0 => pipeline_q2_0` (~2281), rows-per-wg
   `(M+1)/2` (~2345, ~2776), deinit (~5471). Push constants + 3 bindings match
   stq1_0 exactly (M,K,a_offset,x_offset,y_offset,acc_mode).
5. `forward.zig`: ensure the embedding/`to_float` path also handles `.q2_0`
   (the "Unsupported embedding quant type 42, using zeros" warning is that gap).

## Test
`zinc -m Ternary-Bonsai-1.7B-Q2_0.gguf --prompt "The capital of France is" --chat -n 40`
Expect coherent text (~"...is Paris...") and ~150-280 tok/s.
