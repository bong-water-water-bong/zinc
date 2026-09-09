# Merged fork decode on strixhalo gfx1151 (Radeon 8060S) — 2026-09-09

Build: fork main synced with upstream (merge 79cbae6e + c-abi 4614bd13 +
q2_0-fix 565cc609), zig 0.15.2 ReleaseFast, Vulkan backend.
Method: raw prompt "The capital of France is", 96 generated tokens, r3,
box load ~0.8. Oracle-identical tokens (12095/13/576/6722/315/9625/374/1083...).

| model | run1 | run2 | run3 | median | stock-Vulkan bar |
|---|---|---|---|---|---|
| Qwen3-0.6B Q4_K_M | 305.5 | 319.3 | 298.8 | 305.5 | 350 |
| Qwen3-1.7B Q4_K_M | 148.4 | 151.8 | 151.7 | 151.7 | 168.46 |
| Qwen3-4B Q4_K_M | 77.4 | 75.4 | 75.5 | 75.5 | 76.47 |

Notes: 20-token smoke run measured 344.5 tok/s at 0.6B (warm-start effect);
4B sits at the stock-Vulkan bar. Untuned fresh-upstream build; zinc's published
llama.cpp-beating results are RDNA4-dGPU (R9700), this part is UMA gfx1151.
Reference points on the same part: strictly-loom HRX-ALONE 227.5/122.7/58.4;
fork-Vulkan (dual-engine) 359.5/168.4/77.2.
