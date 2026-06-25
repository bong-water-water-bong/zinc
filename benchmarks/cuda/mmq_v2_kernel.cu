// mmq_v2_kernel.cu — M6: cp.async double-buffered mma.sync kernel.
//
// THE bottleneck (proven in M0-M5): global→shared tile load takes 54× longer
// than TC compute (1.4 µs load vs 0.026 µs compute per K-iter). All previous
// optimizations were irrelevant because the load dominates.
//
// M6 fix: cp.async double-buffer. Issue next tile's load asynchronously while
// computing on current tile. Hides the load latency behind TC compute.
//
// Pipeline:
//   Prologue: cp.async load tile 0 → buf[0], commit, wait
//   Steady:   cp.async load tile kc+1 → buf[next], commit
//             mma on buf[cur] (overlaps with async load)
//             wait_group 1, sync
//   Epilogue: mma on last tile

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#define QK4_K 256
#define Q4_K_BLOCK_BYTES 176

#define BM 64
#define BN 64
#define BK 32
#define PAD 4
#define WSTRIDE (BK + PAD)  // 36
#define NWARPS 4
#define WM 32
#define WN 32
#define MMA_M 16
#define MMA_N 8
#define MMA_K 16

// ---- cp.async PTX helpers ----

// Copy 16 bytes from global to shared (async, bypasses registers)
static __device__ __forceinline__ void cp_async_16(void* smem, const void* gmem) {
    uint32_t smem_addr = __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_addr), "l"(gmem));
}

static __device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}

// wait_group 1: wait until ≤1 group is pending (the one just issued)
static __device__ __forceinline__ void cp_async_wait_prev() {
    asm volatile("cp.async.wait_group 1;\n");
}

static __device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;\n");
}

// ---- ldmatrix + mma.sync PTX ----

static __device__ __forceinline__ void ldmatrix_x4(
    uint32_t (&r)[4], const void* smem_ptr)
{
    uint32_t addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(addr));
}

static __device__ __forceinline__ void ldmatrix_x2_trans(
    uint32_t (&r)[2], const void* smem_ptr)
{
    uint32_t addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(r[0]), "=r"(r[1])
        : "r"(addr));
}

// mma.sync m16n8k16 with FP16 accumulate (2× throughput vs FP32 accumulate)
static __device__ __forceinline__ void mma_m16n8k16_f16acc(
    uint32_t (&d)[2], const uint32_t (&a)[4], const uint32_t (&b)[2], const uint32_t (&c)[2])
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};\n"
        : "=r"(d[0]), "=r"(d[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "r"(c[0]), "r"(c[1]));
}

// mma.sync m16n8k16 with FP32 accumulate (for Q4_K kernel)
static __device__ __forceinline__ void mma_m16n8k16(
    float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2], const float (&c)[4])
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

// ---- Issue cp.async for a [BM, BK] fp16 tile from global to shared ----
// Each row: BK fp16 values = BK*2 bytes. cp.async copies 16 bytes at a time.
// BM rows × BK*2/BYTES_PER_CP = BM × (BK*2/16) = 64 × 4 = 256 copies.
// 128 threads → 2 copies per thread.
static __device__ __forceinline__ void issue_tile_load(
    __half* smem_tile,          // [BM * WSTRIDE] destination
    const __half* global_base,  // row-major [?, K], starting at (row_start, k0)
    int row_start, int K, int k0)
{
    constexpr int CP_SIZE = 16;  // bytes per cp.async
    constexpr int BYTES_PER_ROW = BK * 2;
    constexpr int CPS_PER_ROW = BYTES_PER_ROW / CP_SIZE;  // 32*2/16 = 4
    constexpr int TOTAL_CPS = BM * CPS_PER_ROW;           // 64 * 4 = 256
    constexpr int CPS_PER_THREAD = (TOTAL_CPS + 128 - 1) / 128;  // 2

    const int tid = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < CPS_PER_THREAD; i++) {
        int cp_idx = tid + i * 128;
        if (cp_idx >= TOTAL_CPS) break;
        int row = cp_idx / CPS_PER_ROW;        // 0..BM-1
        int chunk = cp_idx % CPS_PER_ROW;      // 0..3
        // Global address: base + row * K + k0 + chunk * (CP_SIZE/2)
        const __half* gptr = global_base + (size_t)(row_start + row) * K + k0 + chunk * (CP_SIZE / 2);
        // Shared address: smem + row * WSTRIDE + chunk * (CP_SIZE/2)
        __half* sptr = smem_tile + row * WSTRIDE + chunk * (CP_SIZE / 2);
        cp_async_16(sptr, gptr);
    }
}

// ============================================================
// M6: cp.async double-buffered f16 GEMM (diagnostic)
// ============================================================
extern __global__ void mmq_v2_kernel_f16_only(
    const __half* __restrict__ W_f16,
    const __half* __restrict__ X_f16,
    float* __restrict__ Y_f32,
    int M, int K, int T)
{
    const int k_chunks = K / BK;

    // Double-buffered shared memory
    // 2 × (64×36×2 + 36×64×2) = 2 × 9216 = 18432 bytes → 2 CTAs/SM
    __shared__ __half Ws[2][BM * WSTRIDE];
    __shared__ __half Xs[2][BN * WSTRIDE];  // row-major [BN, WSTRIDE] for cp.async

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane = tid % 32;
    const int warp_m = warp_id / 2;
    const int warp_n = warp_id % 2;
    const int bm0 = blockIdx.x * BM;
    const int bn0 = blockIdx.y * BN;

    // FP16 accumulators: 2 M-groups × 4 N-groups × 2 half2 = 16 half2/thread
    uint32_t acc[2][4][2];
    #pragma unroll
    for (int mi = 0; mi < 2; mi++)
        #pragma unroll
        for (int ni = 0; ni < 4; ni++) {
            acc[mi][ni][0] = 0;
            acc[mi][ni][1] = 0;
        }

    // ---- Prologue: load tile 0 via cp.async ----
    {
        int k0 = 0;
        // Weight: W_f16 is [M, K] row-major. Copy [BM, BK] tile.
        issue_tile_load(Ws[0], W_f16, bm0, K, k0);
        // Activation: X_f16 is [T, K] row-major. Copy [BN, BK] tile.
        issue_tile_load(Xs[0], X_f16, bn0, K, k0);
        cp_async_commit();
        cp_async_wait_all();
        __syncthreads();
    }

    // ---- Steady state: double-buffered pipeline ----
    #pragma unroll 1
    for (int kc = 0; kc < k_chunks - 1; kc++) {
        int cur = kc % 2;
        int nxt = (kc + 1) % 2;
        int k0_next = (kc + 1) * BK;

        // Issue async load for next tile → buf[nxt]
        issue_tile_load(Ws[nxt], W_f16, bm0, K, k0_next);
        issue_tile_load(Xs[nxt], X_f16, bn0, K, k0_next);
        cp_async_commit();

        // Compute mma on current tile buf[cur] (overlaps with async load)
        int k0 = kc * BK;
        #pragma unroll
        for (int ki = 0; ki < BK / MMA_K; ki++) {
            // Load A fragments
            uint32_t a_frag[2][4];
            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                int group = lane / 8, row_in_grp = lane % 8;
                int tile_row = row_in_grp + (group >= 2 ? 8 : 0);
                int tile_col = (group % 2) * 8;
                ldmatrix_x4(a_frag[mi],
                    &Ws[cur][(warp_m*WM + mi*MMA_M + tile_row) * WSTRIDE + ki*MMA_K + tile_col]);
            }
            // Load B fragments
            uint32_t b_frag[4][2];
            #pragma unroll
            for (int ni = 0; ni < 4; ni++) {
                int group = lane / 8, col_in_grp = lane % 8;
                int k_local = group * 8;
                // Xs is row-major [BN, WSTRIDE]: element (nn, kk) at nn*WSTRIDE+kk
                // For col_major B: B[k, n] = Xs[n*WSTRIDE + k]
                // ldmatrix.x2.trans: thread provides address of a "row" (actually column)
                // Thread t in group g: n = warp_n*WN + ni*MMA_N + col_in_grp
                //                      k_start = ki*MMA_K + g*8
                // Address: &Xs[cur][(warp_n*WN + ni*MMA_N + col_in_grp) * WSTRIDE + ki*MMA_K + k_local]
                int abs_n = warp_n*WN + ni*MMA_N + col_in_grp;
                int abs_k = ki*MMA_K + k_local;
                ldmatrix_x2_trans(b_frag[ni],
                    &Xs[cur][abs_n * WSTRIDE + abs_k]);
            }
            // Compute
            #pragma unroll
            for (int mi = 0; mi < 2; mi++)
                #pragma unroll
                for (int ni = 0; ni < 4; ni++)
                    mma_m16n8k16_f16acc(acc[mi][ni], a_frag[mi], b_frag[ni], acc[mi][ni]);
        }

        // Wait for next tile to finish loading
        cp_async_wait_prev();
        __syncthreads();
    }

    // ---- Epilogue: compute last tile ----
    {
        int cur = (k_chunks - 1) % 2;
        #pragma unroll
        for (int ki = 0; ki < BK / MMA_K; ki++) {
            uint32_t a_frag[2][4];
            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                int group = lane / 8, row_in_grp = lane % 8;
                int tile_row = row_in_grp + (group >= 2 ? 8 : 0);
                int tile_col = (group % 2) * 8;
                ldmatrix_x4(a_frag[mi],
                    &Ws[cur][(warp_m*WM + mi*MMA_M + tile_row) * WSTRIDE + ki*MMA_K + tile_col]);
            }
            uint32_t b_frag[4][2];
            #pragma unroll
            for (int ni = 0; ni < 4; ni++) {
                int group = lane / 8, col_in_grp = lane % 8;
                int k_local = group * 8;
                int abs_n = warp_n*WN + ni*MMA_N + col_in_grp;
                int abs_k = ki*MMA_K + k_local;
                ldmatrix_x2_trans(b_frag[ni],
                    &Xs[cur][abs_n * WSTRIDE + abs_k]);
            }
            #pragma unroll
            for (int mi = 0; mi < 2; mi++)
                #pragma unroll
                for (int ni = 0; ni < 4; ni++)
                    mma_m16n8k16_f16acc(acc[mi][ni], a_frag[mi], b_frag[ni], acc[mi][ni]);
        }
    }

    // ---- Store output (convert fp16 accumulators → fp32) ----
    // FP16 accumulator mma D fragment: 2 uint32_t = 2 half2 per thread
    // Mapping: row_group = lane/4 (0..7), col_pair = lane%4 (0..3)
    //   d[0] = {result[rg*2, cp*2], result[rg*2+1, cp*2]}
    //   d[1] = {result[rg*2, cp*2+1], result[rg*2+1, cp*2+1]}
    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        #pragma unroll
        for (int ni = 0; ni < 4; ni++) {
            int base_row = bm0 + warp_m * WM + mi * MMA_M;
            int base_col = bn0 + warp_n * WN + ni * MMA_N;
            int rg = lane / 4, cp = lane % 4;
            int r = rg * 2, c = cp * 2;
            __half2 d0 = *(__half2*)&acc[mi][ni][0];
            __half2 d1 = *(__half2*)&acc[mi][ni][1];
            if (base_row+r<M && base_col+c<T) Y_f32[(base_row+r)*T+base_col+c] = __low2float(d0);
            if (base_row+r+1<M && base_col+c<T) Y_f32[(base_row+r+1)*T+base_col+c] = __high2float(d0);
            if (base_row+r<M && base_col+c+1<T) Y_f32[(base_row+r)*T+base_col+c+1] = __low2float(d1);
            if (base_row+r+1<M && base_col+c+1<T) Y_f32[(base_row+r+1)*T+base_col+c+1] = __high2float(d1);
        }
    }
}

// ============================================================
// Q4_K dequant + mma.sync (serial load — for comparison with cp.async)
// ============================================================
static __device__ __forceinline__ float h2f_u16(uint16_t h) {
    __half_raw r; r.x = h; return __half2float(*(__half*)&r);
}

extern __global__ void mmq_v2_kernel_q4k(
    const unsigned char* __restrict__ W_q4k,
    const float* __restrict__ X_f32,
    float* __restrict__ Y_f32,
    int M, int K, int T)
{
    const int blocks_per_row = K / 256;
    const int k_chunks = K / BK;
    const int n_threads = NWARPS * 32;

    __shared__ __half Ws[BM * WSTRIDE];
    __shared__ __half Xs[BN * WSTRIDE];

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane = tid % 32;
    const int warp_m = warp_id / 2;
    const int warp_n = warp_id % 2;
    const int bm0 = blockIdx.x * BM;
    const int bn0 = blockIdx.y * BN;

    float acc[2][4][4];
    #pragma unroll
    for (int mi = 0; mi < 2; mi++)
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            #pragma unroll
            for (int i = 0; i < 4; i++)
                acc[mi][ni][i] = 0.0f;

    for (int kc = 0; kc < k_chunks; kc++) {
        const int k0 = kc * BK;
        const int superblock = k0 / 256;
        const int sub_in_block = (k0 % 256) / 32;

        // Dequant Q4_K → Ws
        {
            const int n_vals = BM * BK;
            for (int v = 0; v < (n_vals + n_threads - 1) / n_threads; v++) {
                int idx = tid + v * n_threads;
                if (idx >= n_vals) break;
                int row = idx / BK, col = idx % BK;
                int gr = bm0 + row;
                if (gr >= M) { Ws[row * WSTRIDE + col] = __float2half(0.0f); continue; }
                const unsigned char* blk = W_q4k +
                    (size_t)gr * blocks_per_row * Q4_K_BLOCK_BYTES + superblock * Q4_K_BLOCK_BYTES;
                float d = h2f_u16((uint16_t)(blk[0] | (blk[1] << 8)));
                float dmin = h2f_u16((uint16_t)(blk[2] | (blk[3] << 8)));
                const unsigned char* scales = blk + 4, *qh = blk + 16, *qs = blk + 48;
                int chunk = sub_in_block/2, half_ = sub_in_block%2;
                int j = chunk*2+half_;
                uint8_t sc, mn;
                if (j<4) { sc=scales[j]&63u; mn=(scales[j]>>6)|((scales[j+4]<<2)&0xC0); }
                else { sc=((scales[j-4]>>4)|((scales[j]<<2)&0x3C))&63u; mn=scales[j]>>6; }
                int l = col;
                uint8_t ql = qs[chunk*32u+l];
                uint32_t nib=(half_==0u)?(ql&0xFu):(uint32_t)(ql>>4);
                uint32_t bit=(qh[l]>>(2u*chunk+half_))&1u;
                uint32_t q5=nib+(bit?16u:0u);
                Ws[row*WSTRIDE+col] = __float2half(d*(float)sc*(float)q5-dmin*(float)mn);
            }
        }

        // Load activation → Xs (row-major [BN, WSTRIDE])
        {
            const int n_vals = BK * BN;
            for (int v = 0; v < (n_vals + n_threads - 1) / n_threads; v++) {
                int idx = tid + v * n_threads;
                if (idx >= n_vals) break;
                int kk = idx % BK, nn = idx / BK;
                int gt = bn0 + nn, gk = k0 + kk;
                float val = (gt < T && gk < K) ? X_f32[gt * K + gk] : 0.0f;
                Xs[nn * WSTRIDE + kk] = __float2half(val);
            }
        }

        __syncthreads();

        #pragma unroll
        for (int ki = 0; ki < BK / MMA_K; ki++) {
            uint32_t a_frag[2][4];
            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                int group = lane / 8, row_in_grp = lane % 8;
                int tile_row = row_in_grp + (group >= 2 ? 8 : 0);
                int tile_col = (group % 2) * 8;
                ldmatrix_x4(a_frag[mi],
                    &Ws[(warp_m*WM + mi*MMA_M + tile_row) * WSTRIDE + ki*MMA_K + tile_col]);
            }
            uint32_t b_frag[4][2];
            #pragma unroll
            for (int ni = 0; ni < 4; ni++) {
                int group = lane / 8, col_in_grp = lane % 8;
                int k_local = group * 8;
                int abs_n = warp_n*WN + ni*MMA_N + col_in_grp;
                int abs_k = ki*MMA_K + k_local;
                ldmatrix_x2_trans(b_frag[ni], &Xs[abs_n * WSTRIDE + abs_k]);
            }
            #pragma unroll
            for (int mi = 0; mi < 2; mi++)
                #pragma unroll
                for (int ni = 0; ni < 4; ni++)
                    mma_m16n8k16(acc[mi][ni], a_frag[mi], b_frag[ni], acc[mi][ni]);
        }
        __syncthreads();
    }

    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        #pragma unroll
        for (int ni = 0; ni < 4; ni++) {
            int base_row = bm0 + warp_m * WM + mi * MMA_M;
            int base_col = bn0 + warp_n * WN + ni * MMA_N;
            int r = (lane / 4) * 2, c = (lane % 4) * 2;
            if (base_row+r < M && base_col+c < T) Y_f32[(base_row+r)*T+base_col+c] = acc[mi][ni][0];
            if (base_row+r < M && base_col+c+1 < T) Y_f32[(base_row+r)*T+base_col+c+1] = acc[mi][ni][1];
            if (base_row+r+8 < M && base_col+c < T) Y_f32[(base_row+r+8)*T+base_col+c] = acc[mi][ni][2];
            if (base_row+r+8 < M && base_col+c+1 < T) Y_f32[(base_row+r+8)*T+base_col+c+1] = acc[mi][ni][3];
        }
    }
}
