// mmq_v2_kernel.cu — Fused Q4_K dequant + tensor-core GEMM kernel (M1 baseline).
//
// This is the STARTING POINT for the MMQ v2 project. It's a correct wmma-based
// Q4_K prefill GEMM with a 64×64 block tile. M2 will add cp.async 3-stage
// pipeline + mma.sync + 128×128 tile. See docs/MMQ_V2_DESIGN.md.
//
// Computes: Y[M, T] = W[M, K] × X[T, K]^T
//   W: Q4_K row-major [M, K]  (K must be multiple of 256)
//   X: fp32 row-major [T, K]
//   Y: fp32 row-major [M, T]

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdint>

#define QK4_K 256
#define Q4_K_BLOCK_BYTES 176

// ---- Q4_K dequant helpers (match GGUF spec) ----
static __device__ __forceinline__ float half_to_float_u16(uint16_t h) {
    __half_raw r; r.x = h;
    return __half2float(*reinterpret_cast<__half*>(&r));
}

// Extract 6-bit scale+min for sub-block j (0..7) from the 12-byte scales array
static __device__ __forceinline__ void extract_scale_min(
    int j, const unsigned char* scales, uint8_t& sc, uint8_t& mn)
{
    if (j < 4) {
        sc = scales[j] & 63u;
        mn = ((scales[j] >> 6) | ((scales[j + 4] << 2) & 0xC0));
    } else {
        sc = ((scales[j - 4] >> 4) | ((scales[j - 0] << 2) & 0x3C)) & 63u;
        mn = (scales[j - 0] >> 6) & 0x3;
    }
}

// ---- Configuration ----
#define BM 64    // block M tile (weight rows)
#define BN 64    // block N tile (tokens)
#define BK 32    // block K tile (inner dim, = 1 Q4_K sub-block chunk)
#define NWARPS 4 // 4 warps = 128 threads
#define WM 32    // warp M tile
#define WN 32    // warp N tile
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// ---- The kernel ----
// Block tile [BM, BN] = [64, 64]. 4 warps arranged 2×2, each computes [WM, WN] = [32, 32].
// K-loops in chunks of BK=32 (one Q4_K sub-block at a time).
extern __global__ void mmq_v2_kernel_q4k(
    const unsigned char* __restrict__ W_q4k,  // [M, K/256 * 176] row-major
    const float* __restrict__ X_f32,          // [T, K] row-major
    float* __restrict__ Y_f32,                // [M, T] row-major
    int M, int K, int T)
{
    // Q4_K: 256 elements per superblock, K must be multiple of 256
    // Each superblock = 176 bytes. Sub-blocks of 32 elements within.
    // BK=32 = 1 sub-block. 8 sub-blocks per superblock.
    const int blocks_per_row = K / 256;      // Q4_K superblocks per weight row
    const int k_chunks = K / BK;             // total K-chunks of size BK
    const int sub_per_block = 256 / BK;      // sub-blocks per superblock (= 8 for BK=32)

    // Shared memory: weight fp16 [BM, BK] + activation fp16 [BK, BN]
    // Plus a small cache for the Q4_K superblock headers
    __shared__ __half Ws[BM * BK];           // 64 × 32 × 2 = 4 KB
    __shared__ __half Xs[BK * BN];           // 32 × 64 × 2 = 4 KB

    // Thread layout
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int warp_m = warp_id / 2;  // 0..1
    const int warp_n = warp_id % 2;  // 0..1

    // Block tile position
    const int bm0 = blockIdx.x * BM;
    const int bn0 = blockIdx.y * BN;

    // Accumulators: 2×2 wmma fragments per warp (each [16,16])
    using namespace nvcuda::wmma;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[2][2];
    for (int mi = 0; mi < 2; mi++)
        for (int ni = 0; ni < 2; ni++)
            fill_fragment(acc[mi][ni], 0.0f);

    // K-loop
    for (int k_chunk = 0; k_chunk < k_chunks; k_chunk++) {
        const int k0 = k_chunk * BK;                    // K offset for this chunk
        const int superblock = k_chunk / sub_per_block; // which 256-element superblock
        const int sub_in_block = k_chunk % sub_per_block; // which sub-block (0..7)
        const int chunk_offset = sub_in_block * BK;     // element offset within superblock

        // ---- Load + dequant Q4_K weight tile [BM, BK] into Ws ----
        // Each Q4_K superblock has 256 elements. We extract sub_in_block's 32 elements.
        // All 128 threads cooperatively load BM=64 rows × BK=32 elements.
        // = 64 × 32 = 2048 fp16 values to produce. 128 threads → 16 values/thread.
        {
            const int n_vals = BM * BK;  // 2048
            const int vals_per_thread = (n_vals + NWARPS * 32 - 1) / (NWARPS * 32);
            for (int v = 0; v < vals_per_thread; v++) {
                int idx = tid + v * (NWARPS * 32);
                if (idx >= n_vals) break;
                int row = idx / BK;   // 0..BM-1
                int col = idx % BK;   // 0..BK-1
                int global_row = bm0 + row;
                if (global_row >= M) {
                    Ws[row * BK + col] = __float2half(0.0f);
                    continue;
                }
                // Locate the Q4_K superblock for this row
                const unsigned char* blk = W_q4k +
                    (size_t)global_row * blocks_per_row * Q4_K_BLOCK_BYTES +
                    superblock * Q4_K_BLOCK_BYTES;
                // Parse header
                float d = half_to_float_u16((uint16_t)(blk[0] | (blk[1] << 8)));
                float dmin = half_to_float_u16((uint16_t)(blk[2] | (blk[3] << 8)));
                const unsigned char* scales = blk + 4;   // 12 bytes
                const unsigned char* qh = blk + 16;      // 32 bytes
                const unsigned char* qs = blk + 48;      // 128 bytes
                // Extract scale/min for this sub-block
                int j = sub_in_block;  // 0..7 (sub-block index = chunk index within superblock)
                // chunk = col >> 6 (0..3 for BK=32; col is 0..31, but within the sub-block
                // the "chunk" concept maps to the 4 quarters of 64 elements within 256).
                // Actually for sub_in_block: each 32-element sub-block spans one "chunk"
                // of 32 consecutive elements. The Q4_K layout has 4 chunks of 64 elements
                // each, further split into upper/lower nibble halves. Map our sub-block
                // to the right (chunk, half_) pair.
                // Superblock = 256 elements = 4 chunks × 64 elements.
                // Each chunk = 64 elements = 2 halves × 32 elements.
                // sub_in_block (0..7 for BK=32, 8 sub-blocks) maps to (chunk, half_).
                int chunk = sub_in_block / 2;    // 0..3
                int half_ = sub_in_block % 2;    // 0..1
                uint8_t sc, mn;
                // Extract 6-bit sc and mn for sub-block index j = chunk*2+half_
                int sixbit_j = chunk * 2 + half_;
                if (sixbit_j < 4) {
                    sc = scales[sixbit_j] & 63u;
                    mn = ((scales[sixbit_j] >> 6) | ((scales[sixbit_j + 4] << 2) & 0xC0));
                } else {
                    sc = ((scales[sixbit_j - 4] >> 4) | ((scales[sixbit_j] << 2) & 0x3C)) & 63u;
                    mn = (scales[sixbit_j] >> 6) & 0x3;
                }
                // Dequant element col (0..31) within this sub-block
                int l = col;  // 0..31
                uint8_t ql = qs[chunk * 32u + l];
                uint32_t nib = (half_ == 0u) ? (ql & 0xFu) : (uint32_t)(ql >> 4);
                uint32_t bit = (qh[l] >> (2u * chunk + half_)) & 1u;
                uint32_t q5 = nib + (bit ? 16u : 0u);
                float val = d * (float)sc * (float)q5 - dmin * (float)mn;
                Ws[row * BK + col] = __float2half(val);
            }
        }

        // ---- Load + convert fp32 activation [BK, BN] into Xs ----
        // X is [T, K] row-major. We want X^T[k, t] = X[t, k] for k in [k0, k0+BK), t in [bn0, bn0+BN).
        // Store as [BK, BN] row-major in Xs (col_major for wmma matrix_b).
        {
            const int n_vals = BK * BN;  // 32 × 64 = 2048
            for (int v = 0; v < (n_vals + NWARPS * 32 - 1) / (NWARPS * 32); v++) {
                int idx = tid + v * (NWARPS * 32);
                if (idx >= n_vals) break;
                int kk = idx / BN;  // 0..BK-1 (K index)
                int nn = idx % BN;  // 0..BN-1 (N/token index)
                int global_t = bn0 + nn;
                int global_k = k0 + kk;
                float val = 0.0f;
                if (global_t < T && global_k < K) {
                    val = X_f32[global_t * K + global_k];  // X is [T, K] row-major
                }
                // Store transposed: Xs[kk * BN + nn] = X^T[kk, nn] = X[nn, kk]
                // For wmma col_major matrix_b [BK, BN]: data is [BN, BK] row-major
                // So store as Xs[nn * BK + kk] (row-major [BN, BK])
                // But our shared mem is [BK, BN]... let me use the layout wmma expects.
                // wmma col_major B: stride = BK (leading dim), data = B[k, n] at k*ldim + n
                // Actually for col_major, B[k, n] is at offset k + n * ldim where ldim = BK
                // Simplest: store as Xs[kk + nn * BK] (col_major [BK, BN])
                Xs[kk + nn * BK] = __float2half(val);
            }
        }

        __syncthreads();

        // ---- Compute: wmma on [BM, BK] × [BK, BN] ----
        // Each warp owns [WM=32, WN=32] = 2×2 wmma tiles.
        for (int mi = 0; mi < 2; mi++) {
            for (int ni = 0; ni < 2; ni++) {
                // For BK=32, we do 2 K-steps of WMMA_K=16
                for (int ki = 0; ki < BK / WMMA_K; ki++) {
                    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, row_major> a_frag;
                    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, col_major> b_frag;

                    // Load A: weight [WMMA_M, WMMA_K] from Ws
                    int a_row = warp_m * WM + mi * WMMA_M;
                    int a_col = ki * WMMA_K;
                    load_matrix_sync(a_frag, Ws + a_row * BK + a_col, BK);

                    // Load B: activation [WMMA_K, WMMA_N] from Xs (col_major)
                    int b_row = ki * WMMA_K;
                    int b_col = warp_n * WN + ni * WMMA_N;
                    load_matrix_sync(b_frag, Xs + b_row + b_col * BK, BK);

                    mma_sync(acc[mi][ni], a_frag, b_frag, acc[mi][ni]);
                }
            }
        }

        __syncthreads();
    }

    // ---- Store output ----
    for (int mi = 0; mi < 2; mi++) {
        for (int ni = 0; ni < 2; ni++) {
            int out_row = bm0 + warp_m * WM + mi * WMMA_M;
            int out_col = bn0 + warp_n * WN + ni * WMMA_N;
            // Store as fp32 row-major [M, T]
            store_matrix_sync(
                Y_f32 + out_row * T + out_col,
                acc[mi][ni], T, mem_row_major);
        }
    }
}
